"""Framework runtime for the SM120 NVFP4 x NVFP4 split MegaMoE kernels."""

from __future__ import annotations

import dataclasses
import os
import time
import warnings
from dataclasses import dataclass, field
from typing import Any

import torch

from .comm import (
    PeerMemorySession,
    _no_dist,
    compute_peer_offsets,
    ensure_not_capturing,
    free_sym_tensor,
    sym_byte_view,
    sym_zeros,
)
from .staging import ACTIVATION_DTYPE
from .weights import SCALE_DTYPE, TransformedWeights, ceil_div, round_up

DECODE_GRAPH_COMPILE_BUCKETS = (7, 16, 32, 64, 128, 168, 256, 320, 384)
_WARNED_DECODE_CAPACITY_FALLBACKS: set[int] = set()
# Largest compile bucket that may run the p2p_split EP8 backend. Its xnuma
# return buffers need about 2*n_cross*2*T*topk*(2*hidden+16) bytes per rank
# (DSV4.1-Flash: ~0.38 GB at T=384, ~4 GB at T=4096), so only the decode
# buckets qualify by default.
P2P_SPLIT_MAX_TOKENS_ENV = "FLASHINFER_MEGAMOE_P2P_SPLIT_MAX_TOKENS"


def plan_p2p_split_buckets(config, topology, buckets) -> set[int]:
    """Compile buckets that run p2p_split on an EP spanning NUMA nodes.

    Same-NUMA peers keep direct P2P; every cross-NUMA peer is served through
    a local shadow segment kept in sync over GIN (``transport.xnuma``). With
    no explicit ``comm_backend`` knob a bucket follows the measured anchor
    table (or its rule for untabulated shapes); ``"p2p_split"`` selects it for
    every eligible bucket and ``"gin"`` disables it. Buckets above
    ``FLASHINFER_MEGAMOE_P2P_SPLIT_MAX_TOKENS`` (default: the largest decode
    bucket) always use gin.
    """
    if _no_dist() or not getattr(topology, "cross_numa_peer_count", 0):
        return set()
    requested = (config.knobs or {}).get("comm_backend")
    if requested == "gin":
        return set()
    from moe_sm120_nvfp4_split.heuristic import MegaMoEHeuristicInput
    from moe_sm120_nvfp4_split.runtime.green_context import query_sm_resource_info
    from moe_sm120_nvfp4_split.transport.shadow import ShadowLayout
    from moe_sm120_nvfp4_split.tuned_table import (
        SPLIT_MAX_EXPERTS_PER_RANK,
        preferred_ep8_backend,
    )

    if config.local_experts > SPLIT_MAX_EXPERTS_PER_RANK:
        return set()
    try:
        ShadowLayout.from_topology(topology, config.rank)
    except (AttributeError, ValueError):
        return set()
    limit = int(
        os.environ.get(P2P_SPLIT_MAX_TOKENS_ENV, str(DECODE_GRAPH_COMPILE_BUCKETS[-1]))
    )
    num_sms, minimum, alignment = query_sm_resource_info()
    chosen = set()
    for bucket in buckets:
        if bucket > limit:
            continue
        if requested == "p2p_split":
            chosen.add(bucket)
            continue
        shape = MegaMoEHeuristicInput(
            tokens_per_rank=bucket,
            hidden=config.hidden,
            intermediate=2 * config.intermediate,
            num_topk=config.num_topk,
            num_total_experts=config.num_total_experts,
            data_parallel_size=config.data_parallel_size,
            tensor_parallel_size=config.tensor_parallel_size,
            expert_parallel_size=config.world_size,
            ep_same_numa_peer_count=topology.same_numa_peer_count,
            ep_cross_numa_peer_count=topology.cross_numa_peer_count,
            num_sms=num_sms,
            sm_min_partition=minimum,
            sm_partition_alignment=alignment,
        )
        if preferred_ep8_backend(shape) == "p2p_split":
            chosen.add(bucket)
    return chosen


def _p2p_split_plan_supported(spec, bundle) -> bool:
    from moe_sm120_nvfp4_split.kernel_fc2_combine import Sm120Fc2CombineKernel

    return (
        spec.kernel.comm_backend == "p2p_direct"
        and not spec.kernel.rank_local_combine
        and type(bundle.k2) is Sm120Fc2CombineKernel
    )


def select_graph_compile_bucket(
    requested_tokens_per_rank: int | None,
    workspace_capacity: int,
) -> int:
    """Map a collective graph row count to a kernel specialization.

    Buckets above the decode range intentionally fall back to the full
    workspace capacity. This keeps the prefill policy unchanged while decode
    graphs use the measured low-latency specializations.
    """
    if workspace_capacity <= 0:
        raise ValueError("workspace_capacity must be positive")
    if requested_tokens_per_rank is None:
        return workspace_capacity
    if not 0 < requested_tokens_per_rank <= workspace_capacity:
        raise ValueError(
            f"compile token count {requested_tokens_per_rank} is outside "
            f"workspace capacity {workspace_capacity}"
        )
    selected = next(
        (
            bucket
            for bucket in DECODE_GRAPH_COMPILE_BUCKETS
            if requested_tokens_per_rank <= bucket <= workspace_capacity
        ),
        workspace_capacity,
    )
    if (
        selected == workspace_capacity
        and workspace_capacity > DECODE_GRAPH_COMPILE_BUCKETS[-1]
        and requested_tokens_per_rank > DECODE_GRAPH_COMPILE_BUCKETS[-1]
        and requested_tokens_per_rank < workspace_capacity
        and requested_tokens_per_rank <= 2 * DECODE_GRAPH_COMPILE_BUCKETS[-1]
        and workspace_capacity not in _WARNED_DECODE_CAPACITY_FALLBACKS
    ):
        _WARNED_DECODE_CAPACITY_FALLBACKS.add(workspace_capacity)
        warnings.warn(
            "SM120 NVFP4 MegaMoE decode compile request "
            f"({requested_tokens_per_rank} rows) exceeds the largest dedicated "
            f"decode graph bucket ({DECODE_GRAPH_COMPILE_BUCKETS[-1]}); "
            f"falling back to workspace capacity ({workspace_capacity}). "
            "Decode performance may regress because this selects the prefill "
            "kernel heuristic.",
            RuntimeWarning,
            stacklevel=2,
        )
    return selected


@dataclass(frozen=True)
class MegaMoESm120Nvfp4Config:
    rank: int
    world_size: int
    max_tokens_per_rank: int
    num_topk: int
    num_total_experts: int
    hidden: int
    intermediate: int
    gate_up_clamp: float | None = None
    input_norm_const: float = 1.0
    data_parallel_size: int = 1
    tensor_parallel_size: int = 1
    knobs: dict[str, Any] | None = None
    numerical_contract: str = "ordinary_topk"

    @property
    def local_experts(self) -> int:
        return self.num_total_experts // self.world_size

    def validate(self) -> None:
        if self.numerical_contract not in ("ordinary_topk", "owner_bf16"):
            raise ValueError("unknown NVFP4 numerical contract")
        if self.numerical_contract == "ordinary_topk" and (self.knobs or {}).get(
            "rank_local_combine"
        ):
            raise ValueError(
                "rank_local_combine requires owner_bf16 numerical permission"
            )
        if self.world_size <= 0 or not 0 <= self.rank < self.world_size:
            raise ValueError("invalid EP rank/world size")
        if self.num_total_experts % self.world_size:
            raise ValueError("num_total_experts must be divisible by EP world size")
        if self.hidden % 32 or self.intermediate % 32:
            raise ValueError(
                "hidden and post-SwiGLU intermediate must be multiples of 32"
            )
        if self.num_topk <= 0 or self.num_topk > self.num_total_experts:
            raise ValueError("invalid top-k")
        if self.max_tokens_per_rank <= 0:
            raise ValueError("max_tokens_per_rank must be positive")
        if self.input_norm_const <= 0:
            raise ValueError("input_norm_const must be positive")


@dataclass
class MegaMoESm120Nvfp4Inputs:
    activation: torch.Tensor
    activation_scale: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    fc1_weight: torch.Tensor
    fc1_weight_scale: torch.Tensor
    fc2_weight: torch.Tensor
    fc2_weight_scale: torch.Tensor
    output: torch.Tensor


@dataclass
class _ExecutionStorage:
    local_workspace: torch.Tensor
    shared_workspace: torch.Tensor
    combine_output: torch.Tensor
    epilogue_args: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    rank_combine_ready: torch.Tensor | None = None
    green_trace: torch.Tensor | None = None
    network: Any = None
    xnuma: Any = None


@dataclass
class _ExecutionBuffers:
    bundle: Any
    spec: Any
    local_workspace: torch.Tensor
    shared_workspace: torch.Tensor
    combine_output: torch.Tensor
    epilogue_args: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    rank_combine_ready: torch.Tensor | None = None
    green_trace: torch.Tensor | None = None
    network: Any = None
    xnuma: Any = None


@dataclass
class _CompiledSplit:
    execution: _ExecutionBuffers
    k1: Any
    k2: Any
    k2_drain: Any
    k2_finalizer: Any
    k3: Any
    streams: tuple[torch.cuda.Stream, torch.cuda.Stream, torch.cuda.Stream]
    graph: Any
    runtime_k1: dict[str, Any]
    runtime_k2: dict[str, Any]
    runtime_k2_drain: dict[str, Any] | None
    runtime_k2_finalizer: dict[str, Any] | None


@dataclass
class MegaMoESm120Nvfp4Workspace:
    config: MegaMoESm120Nvfp4Config
    x: torch.Tensor
    x_scale: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    output: torch.Tensor
    fc1_alpha: torch.Tensor
    fc2_alpha: torch.Tensor
    fc1_norm_const: torch.Tensor
    _sym_roots: list[torch.Tensor] = field(default_factory=list)
    _comm_session: Any = None
    _control_group: Any = None
    _topology: Any = None
    _storages: dict[int, _ExecutionStorage] = field(default_factory=dict)
    _execution_plans: dict[int, tuple[Any, Any]] = field(default_factory=dict)
    _compiled_k12: dict[int, tuple[Any, Any]] = field(default_factory=dict)
    _compiled_aux: dict[int, tuple[Any, Any]] = field(default_factory=dict)
    _executions: dict[int, _ExecutionBuffers] = field(default_factory=dict)
    _compile_bucket: int = 0
    _frontends: dict[tuple[int, ...], "MegaMoESm120Nvfp4Frontend"] = field(
        default_factory=dict
    )
    _green_resources: dict[tuple[int, int], Any] = field(default_factory=dict)
    _capture_streams: Any = None
    _transport_streams: dict[str, Any] = field(default_factory=dict)
    _staged_tokens: int = 0
    # p2p_split: bucket -> "p2p_split", the preallocated (combine_output,
    # shared_workspace) inside the shadowed prefix, and the shared xnuma
    # control-region layout and device epoch.
    _bucket_backends: dict[int, str] = field(default_factory=dict)
    _split_buffers: dict[int, tuple[torch.Tensor, torch.Tensor]] = field(
        default_factory=dict
    )
    _xnuma_region: Any = None
    _xnuma_epoch: Any = None
    # Forwarder CTAs per cross source, one value for every p2p_split bucket.
    _xnuma_fsplit: int = 2
    _destroyed: bool = False

    def _has_prepared_frontend(self, transformed_weights, output: torch.Tensor) -> bool:
        key = _frontend_graph_cache_key(
            transformed_weights, output, self._compile_bucket
        )
        frontend = self._frontends.get(key)
        return frontend is not None and frontend._compiled is not None

    def destroy(self) -> None:
        if self._destroyed:
            return
        for frontend in self._frontends.values():
            frontend.release()
        self._frontends.clear()
        for resources in self._green_resources.values():
            resources.close()
        self._green_resources.clear()
        self._executions.clear()
        self._compiled_k12.clear()
        self._compiled_aux.clear()
        self._execution_plans.clear()
        self._storages.clear()
        from .quant_stage import (
            forget_staged_tokens,
        )

        forget_staged_tokens(self.topk_ids)
        for root in reversed(self._sym_roots):
            free_sym_tensor(root)
        self._sym_roots.clear()
        if self._comm_session is not None:
            self._comm_session.close()
        self._destroyed = True


class MegaMoESm120Nvfp4Frontend:
    """Lazy JIT + native Green Context graph session for direct-P2P EP."""

    _LOCAL_RESET_REGIONS = (
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
    )

    # ``nvlink_barrier_counter`` is deliberately not reset by graph replay.
    # It carries the phase/sign paired with the persistent two-slot
    # ``nvlink_barrier_signal`` allocation. Resetting only the counter can make
    # a later replay observe a stale target value and leave the rank barrier
    # before every peer has entered the current invocation.

    def __init__(
        self,
        workspace: MegaMoESm120Nvfp4Workspace,
        *,
        compile_bucket: int,
        control_group: Any = None,
    ) -> None:
        self.workspace = workspace
        self.config = workspace.config
        self.compile_bucket = compile_bucket
        self.control_group = control_group
        self._compiled: _CompiledSplit | None = None
        if compile_bucket not in (
            *DECODE_GRAPH_COMPILE_BUCKETS,
            self.config.max_tokens_per_rank,
        ):
            raise ValueError(f"unsupported NVFP4 compile bucket {compile_bucket}")

    @staticmethod
    def _to_cute(
        tensor: torch.Tensor,
        assumed_align: int = 16,
        *,
        static_layout: bool = False,
    ):
        import cutlass.torch as cutlass_torch

        from moe_sm120_nvfp4_split.runtime.low_precision_dlpack import wrap

        result = cutlass_torch.from_dlpack(wrap(tensor), assumed_align=assumed_align)
        if static_layout:
            return result
        return result.mark_layout_dynamic(
            leading_dim=cutlass_torch.get_leading_dim(tensor)
        )

    def _barrier(self) -> None:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.barrier(group=self.control_group)

    def _is_p2p_split(self, bucket: int) -> bool:
        backends = getattr(self.workspace, "_bucket_backends", None) or {}
        return backends.get(bucket) == "p2p_split"

    def _heuristic_overrides(self, *, p2p_split: bool = False):
        from moe_sm120_nvfp4_split.heuristic import MegaMoEHeuristicOverrides

        fields = {field.name for field in dataclasses.fields(MegaMoEHeuristicOverrides)}
        values = dict(self.config.knobs or {})
        unknown = sorted(set(values) - fields)
        if unknown:
            raise ValueError(f"unknown SM120 NVFP4 tuning knobs: {unknown}")
        cross = self.workspace._topology.cross_numa_peer_count
        backend = "gin" if cross else "p2p_direct"
        allowed = {backend, "p2p_split"} if cross else {backend}
        if values.get("comm_backend", backend) not in allowed:
            raise ValueError(
                "Communication override conflicts with discovered EP topology"
            )
        if p2p_split:
            # Kernels see one P2P group; cross-NUMA peers are shadowed.
            if values.get("rank_local_combine"):
                raise ValueError("p2p_split requires rank_local_combine off")
            values["rank_local_combine"] = False
            backend = "p2p_direct"
        values["comm_backend"] = backend
        return MegaMoEHeuristicOverrides(**values)

    def _select_spec(self, tokens_per_rank: int):
        from moe_sm120_nvfp4_split.api import (
            MegaMoEProblemSpec,
            SplitKernelBuildOptions,
            select_compile_spec,
        )
        from moe_sm120_nvfp4_split.runtime.green_context import (
            query_green_context_sm_counts,
            query_sm_resource_info,
        )

        cfg = self.config
        jit = None
        if (
            os.environ.get("MEGA_SPLIT_GLOBALTIMER") == "1"
            or os.environ.get("MEGA_SPLIT_K2_TILE_TRACE") == "1"
            or "MEGA_FC1_UTILITY_POSTPROCESS" in os.environ
        ):
            from moe_sm120_nvfp4_split.jit_config import Sm120JitConfig

            jit = Sm120JitConfig.from_environment()
        num_sms, minimum, alignment = query_sm_resource_info()
        problem = MegaMoEProblemSpec(
            tokens_per_rank=tokens_per_rank,
            num_topk=cfg.num_topk,
            num_total_experts=cfg.num_total_experts,
            hidden=cfg.hidden,
            intermediate=2 * cfg.intermediate,
            expert_parallel_size=cfg.world_size,
            expert_parallel_rank=cfg.rank,
            data_parallel_size=cfg.data_parallel_size,
            tensor_parallel_size=cfg.tensor_parallel_size,
            gate_up_clamp=cfg.gate_up_clamp,
        )
        # The native policy calls local P2P + cross-NUMA GIN gin_hybrid;
        # FlashInfer's topology-owned transport calls the same mode gin.
        p2p_split = self._is_p2p_split(tokens_per_rank)
        overrides = self._heuristic_overrides(p2p_split=p2p_split)
        if overrides.comm_backend == "gin":
            overrides = dataclasses.replace(overrides, comm_backend="gin_hybrid")
        same_numa = self.workspace._topology.same_numa_peer_count
        cross_numa = self.workspace._topology.cross_numa_peer_count
        if p2p_split:
            # Same folded view as the native runner: one P2P group.
            same_numa, cross_numa = same_numa + cross_numa, 0
        spec = select_compile_spec(
            problem=problem,
            ep_same_numa_peer_count=same_numa,
            ep_cross_numa_peer_count=cross_numa,
            num_sms=num_sms,
            sm_min_partition=minimum,
            sm_partition_alignment=alignment,
            overrides=overrides,
            numerical_contract=cfg.numerical_contract,
            jit=jit,
        )
        if spec.kernel.comm_backend == "gin_hybrid":
            spec = dataclasses.replace(
                spec, kernel=dataclasses.replace(spec.kernel, comm_backend="gin")
            )
        if spec.kernel.comm_backend == "gin":
            resources = self._green_resources_for(spec)
            actual_k1, actual_k2 = resources.k1_sm_count, resources.k2_sm_count
        else:
            actual_k1, actual_k2 = query_green_context_sm_counts(
                k1_sm_count=spec.kernel.k1_sms
            )
        if (actual_k1, actual_k2) != (spec.kernel.k1_sms, spec.kernel.k2_sms):
            raise RuntimeError(
                "Green Context partition differs from the heuristic: "
                f"actual={(actual_k1, actual_k2)}, "
                f"expected={(spec.kernel.k1_sms, spec.kernel.k2_sms)}"
            )
        return dataclasses.replace(
            spec,
            build=SplitKernelBuildOptions(
                concurrent_k1_k2=True,
                k1_active_clusters=actual_k1,
                k2_active_clusters=actual_k2,
            ),
        )

    def _green_resources_for(self, spec):
        from moe_sm120_nvfp4_split.runtime.green_context import (
            NativeGreenContextPair,
            NativeGreenContextResources,
        )

        k = spec.kernel
        network = k.comm_backend == "gin"
        key = (
            torch.cuda.current_device(),
            k.k1_sms,
            k.k2_sms,
            k.tx_sms if network else 0,
            k.rx_sms if network else 0,
        )
        resources = self.workspace._green_resources.get(key)
        if resources is None:
            if network:
                resources = NativeGreenContextResources.create(
                    device=key[0],
                    k1_sm_count=k.k1_sms,
                    k2_sm_count=k.k2_sms,
                    comm_tx_sm_count=k.tx_sms,
                    comm_rx_sm_count=k.rx_sms,
                )
            else:
                resources = NativeGreenContextPair.create(
                    device=key[0], k1_sm_count=k.k1_sms
                )
            self.workspace._green_resources[key] = resources
        return resources

    def _runtime_tensors(
        self,
        inputs: MegaMoESm120Nvfp4Inputs,
        local_workspace: torch.Tensor,
        shared_workspace: torch.Tensor,
        combine_output: torch.Tensor,
        epilogue_args: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        rank_combine_ready: torch.Tensor | None,
        green_trace: torch.Tensor | None,
        stream,
    ) -> dict[str, Any]:
        from moe_sm120_nvfp4_split.support.src.sym_buffer import SymBufferHost

        network = self.workspace._executions[self.compile_bucket].network
        if network is None:
            base, offsets = compute_peer_offsets(
                shared_workspace, self.config.world_size
            )
            activation, activation_sf, topk_weights = (
                inputs.activation,
                inputs.activation_scale,
                inputs.topk_weights,
            )
        else:
            base, offsets = network.symmetric_base, network.peer_offsets_list
            activation, activation_sf, topk_weights = (
                network._transport.activation_rx,
                network._transport.sf_rx,
                network._transport.topk_weight_rx,
            )
        fc1_alpha, fc2_alpha, fc1_norm_const = epilogue_args
        return dict(
            # Dispatch addresses packed bytes; the GEMM builds its logical
            # Float4 view from the dispatched local buffer.
            activation=self._to_cute(activation.view(torch.uint8)),
            activation_sf=self._to_cute(activation_sf),
            topk_idx=self._to_cute(inputs.topk_ids),
            topk_weights=self._to_cute(topk_weights),
            fc1_weight=self._to_cute(inputs.fc1_weight),
            fc1_weight_sf=self._to_cute(inputs.fc1_weight_scale),
            fc2_weight=self._to_cute(inputs.fc2_weight),
            fc2_weight_sf=self._to_cute(inputs.fc2_weight_scale),
            fc1_alpha=self._to_cute(fc1_alpha, assumed_align=4),
            fc2_alpha=self._to_cute(fc2_alpha, assumed_align=4),
            fc1_norm_const=self._to_cute(fc1_norm_const, assumed_align=4),
            combine_output=self._to_cute(combine_output),
            combine_ready_flags=(
                self._to_cute(rank_combine_ready, assumed_align=4)
                if rank_combine_ready is not None
                else None
            ),
            fc2_block_done_counter=None,
            local_workspace=self._to_cute(local_workspace, static_layout=True),
            shared_workspace=self._to_cute(shared_workspace),
            green_trace=(
                self._to_cute(green_trace, static_layout=True)
                if green_trace is not None
                else None
            ),
            peer_rank_ptr_mapper_host=SymBufferHost(
                base_addr=base,
                offsets=offsets,
                rank_idx=self.config.rank,
                num_max_ranks=self.config.world_size,
            ),
            stream=stream,
        )

    def _ensure_plan(self, bucket: int) -> tuple[Any, Any]:
        plan = self.workspace._execution_plans.get(bucket)
        if plan is not None:
            return plan
        from moe_sm120_nvfp4_split.api import build_split_kernels

        spec = self._select_spec(bucket)
        if os.environ.get("FLASHINFER_MEGAMOE_LOG_COMPILE_SPEC") == "1":
            kernel = spec.kernel
            print(
                "[FlashInfer MegaMoE compile] "
                f"rank={self.config.rank} "
                f"capacity={self.config.max_tokens_per_rank} "
                f"bucket={bucket} K1={kernel.k1_tile} "
                f"K2={kernel.k2_tile}/stage{kernel.k2_stages} "
                f"bundle={kernel.ready_queue_bundle} "
                f"tail_reclaim={getattr(kernel, 'k2_tail_reclaim', False)} "
                f"rank_cache={kernel.dispatch_rank_cache} "
                f"rank_combine={kernel.rank_local_combine} "
                f"dispatch_warps={kernel.dispatch_warps} "
                f"k1_ctas_per_sm={kernel.k1_ctas_per_sm} "
                f"k2_launch_factor={kernel.k2_launch_factor} "
                f"prefetch_ahead={kernel.weight_prefetch_ahead} "
                f"backend={'p2p_split' if self._is_p2p_split(bucket) else kernel.comm_backend} "
                f"green={kernel.k1_sms}/{kernel.k2_sms}",
                flush=True,
            )
        bundle = build_split_kernels(spec)
        if self._is_p2p_split(bucket):
            # Peer-store barriers wait only for same-NUMA ranks; cross-NUMA
            # sources are delivered by the xnuma transport.
            same_numa_mask = ((1 << self.config.world_size) - 1) & ~int(
                self.workspace._topology.force_ibgda_peer_mask
            )
            for kernel in (
                bundle.k1,
                bundle.k2,
                getattr(bundle, "k2_drain", None),
                getattr(bundle, "k2_finalizer", None),
            ):
                if (
                    kernel is not None
                    and getattr(kernel, "token_comm", None) is not None
                ):
                    kernel.token_comm.barrier_peer_mask = same_numa_mask
        plan = (spec, bundle)
        self.workspace._execution_plans[bucket] = plan
        return plan

    def _create_xnuma(self, bundle, combine_output):
        from types import SimpleNamespace

        from moe_sm120_nvfp4_split.transport.xnuma import XnumaTransport
        from moe_sm120_nvfp4_split.transport.xnuma_layout import XnumaLayout

        workspace, cfg, bucket = self.workspace, self.config, self.compile_bucket
        region = workspace._xnuma_region
        spec = workspace._execution_plans[bucket][0]
        layout = XnumaLayout(
            fsplit=workspace._xnuma_fsplit,
            world=cfg.world_size,
            n_cross=region.n_cross,
            experts_per_rank=cfg.local_experts,
            tokens_per_rank=bucket,
            topk=cfg.num_topk,
            hidden=cfg.hidden,
            pack=_xnuma_pack(spec.kernel) and region.pack,
        )
        inputs = (
            workspace.x[:bucket],
            workspace.x_scale[:bucket],
            workspace.topk_ids[:bucket],
            workspace.topk_weights[:bucket],
        )
        view = SimpleNamespace(
            rank=cfg.rank,
            _kernel_k2=bundle.k2,
            my_activation=inputs[0],
            my_activation_sf=inputs[1],
            my_topk_idx=inputs[2],
            my_topk_weights=inputs[3],
            combine_output=combine_output,
        )
        transport = XnumaTransport(
            view,
            workspace._comm_session,
            layout,
            region=region,
            epoch=workspace._xnuma_epoch,
            input_regions=inputs,
            zero_reserved=False,
        )
        # Compile-time hooks; set before K1/K2 are compiled for this bucket.
        bundle.k1.token_comm.xnuma = transport.k1_config()
        for kernel in (bundle.k2, getattr(bundle, "k2_drain", None)):
            if kernel is not None:
                kernel.xnuma_progress_addr = transport.k2_progress_addr
        return transport

    def _ensure_execution(self) -> _ExecutionBuffers:
        execution = self.workspace._executions.get(self.compile_bucket)
        if execution is not None:
            return execution
        ensure_not_capturing("NVFP4 shared execution-buffer allocation")

        spec, bundle = self._ensure_plan(self.compile_bucket)
        storage = self.workspace._storages.get(self.compile_bucket)
        if storage is None:
            # NCCL workspace allocation is not ordered on every peer caller
            # stream. Drain in-flight peer-dependent graphs before entering it.
            torch.cuda.synchronize()
            p2p_split = self._is_p2p_split(self.compile_bucket)
            if p2p_split:
                # Preallocated inside the shadowed arena prefix.
                combine_output, split_shared = self.workspace._split_buffers[
                    self.compile_bucket
                ]
            else:
                combine_output, root = sym_byte_view(
                    (self.compile_bucket, self.config.num_topk, self.config.hidden),
                    torch.bfloat16,
                    session=self.workspace._comm_session,
                )
                self.workspace._sym_roots.append(root)
            network = None
            xnuma = None
            if p2p_split:
                local_workspace = torch.zeros(
                    (bundle.local_workspace_bytes,), dtype=torch.uint8, device="cuda"
                )
                shared_workspace = split_shared
                rank_combine_ready = None
                xnuma = self._create_xnuma(bundle, combine_output)
            elif spec.kernel.comm_backend == "gin":
                from .network_runtime import create_network_runtime

                network = create_network_runtime(
                    self.workspace, self.compile_bucket, spec, bundle, combine_output
                )
                local_workspace, shared_workspace = (
                    network.local_workspace,
                    network.shared_workspace,
                )
                rank_combine_ready = network.rank_combine_ready
                self.workspace._sym_roots.append(shared_workspace)
                if rank_combine_ready is not None:
                    self.workspace._sym_roots.append(rank_combine_ready)
            else:
                local_workspace = torch.zeros(
                    (bundle.local_workspace_bytes,), dtype=torch.uint8, device="cuda"
                )
                shared_workspace = sym_zeros(
                    (bundle.shared_workspace_bytes,),
                    torch.uint8,
                    session=self.workspace._comm_session,
                )
                self.workspace._sym_roots.append(shared_workspace)
                rank_combine_ready = None
                if spec.kernel.rank_local_combine:
                    rank_combine_ready = sym_zeros(
                        (self.compile_bucket + 1, self.config.world_size),
                        torch.int32,
                        session=self.workspace._comm_session,
                    )
                    self.workspace._sym_roots.append(rank_combine_ready)
            epilogue_args = (
                self.workspace.fc1_alpha,
                self.workspace.fc2_alpha,
                self.workspace.fc1_norm_const,
            )
            green_trace = None
            if (
                os.environ.get("MEGA_SPLIT_GLOBALTIMER") == "1"
                or os.environ.get("MEGA_SPLIT_K2_TILE_TRACE") == "1"
            ):
                from moe_sm120_nvfp4_split.split_timestamp import (
                    K2_TILE_TRACE_WORDS,
                    TRACE_BASE_WORDS,
                )

                trace_words = TRACE_BASE_WORDS
                if os.environ.get("MEGA_SPLIT_K2_TILE_TRACE") == "1":
                    trace_words += K2_TILE_TRACE_WORDS
                green_trace = torch.zeros(
                    (trace_words,), dtype=torch.int64, device="cuda"
                )
            storage = _ExecutionStorage(
                local_workspace=local_workspace,
                shared_workspace=shared_workspace,
                combine_output=combine_output,
                epilogue_args=epilogue_args,
                rank_combine_ready=rank_combine_ready,
                green_trace=green_trace,
                network=network,
                xnuma=xnuma,
            )
            self.workspace._storages[self.compile_bucket] = storage
        execution = _ExecutionBuffers(
            bundle=bundle,
            spec=spec,
            local_workspace=storage.local_workspace,
            shared_workspace=storage.shared_workspace,
            combine_output=storage.combine_output,
            epilogue_args=storage.epilogue_args,
            rank_combine_ready=storage.rank_combine_ready,
            green_trace=storage.green_trace,
            network=storage.network,
            xnuma=storage.xnuma,
        )
        self.workspace._executions[self.compile_bucket] = execution
        return execution

    def _ensure_compiled(self, inputs: MegaMoESm120Nvfp4Inputs) -> _CompiledSplit:
        if self._compiled is not None:
            return self._compiled
        ensure_not_capturing("CuTeDSL compile and Green Context graph creation")

        import cuda.bindings.driver as cuda
        import cutlass.cute as cute
        from moe_sm120_nvfp4_split.api import (
            compile_combine_reduce,
            compile_rank_local_combine,
        )
        from moe_sm120_nvfp4_split.runtime.green_context import (
            NativeGreenContextGraph,
        )

        profile_start = time.monotonic()
        execution = self._ensure_execution()
        bundle = execution.bundle
        spec = execution.spec
        local_workspace = execution.local_workspace
        shared_workspace = execution.shared_workspace
        combine_output = execution.combine_output
        epilogue_args = execution.epilogue_args
        green_trace = execution.green_trace

        # Layers sharing this workspace execute serially. Reuse capture
        # streams instead of allocating three persistent streams per layer.
        # Excess streams can alias hardware queues used by persistent waiters.
        if self.workspace._capture_streams is None:
            self.workspace._capture_streams = (
                torch.cuda.Stream(priority=0),
                torch.cuda.Stream(priority=-1),
                torch.cuda.Stream(priority=0),
            )
        root_stream, k1_stream, k2_stream = self.workspace._capture_streams
        root_cuda = cuda.CUstream(root_stream.cuda_stream)
        k1_cuda = cuda.CUstream(k1_stream.cuda_stream)
        k2_cuda = cuda.CUstream(k2_stream.cuda_stream)
        base_runtime = self._runtime_tensors(
            inputs,
            local_workspace,
            shared_workspace,
            combine_output,
            epilogue_args,
            execution.rank_combine_ready,
            green_trace,
            root_cuda,
        )
        runtime_k1 = dict(base_runtime, stream=k1_cuda)
        runtime_k2 = dict(base_runtime, stream=k2_cuda)
        # Residency knobs multiply the launch grids on unchanged physical
        # Green partitions (k1_ctas_per_sm: two K1 CTAs per K1 SM;
        # k2_launch_factor: two 8-warp K2 CTAs per K2 SM).
        k1_launch_clusters = spec.kernel.k1_sms * spec.kernel.k1_ctas_per_sm
        compile_k1 = dict(runtime_k1, max_active_clusters=k1_launch_clusters)
        k2_ctas_per_sm = (
            2 if (spec.kernel.k2_warps == 12 and spec.kernel.k2_tile[1] == 16) else 1
        )
        k2_launch_clusters = (
            spec.kernel.k2_sms * k2_ctas_per_sm * spec.kernel.k2_launch_factor
        )
        compile_k2 = dict(
            runtime_k2,
            max_active_clusters=k2_launch_clusters,
        )
        compiled_pair = self.workspace._compiled_k12.get(self.compile_bucket)
        if compiled_pair is None:
            compiled_pair = (
                cute.compile(bundle.k1, **compile_k1),
                cute.compile(bundle.k2, **compile_k2),
            )
            self.workspace._compiled_k12[self.compile_bucket] = compiled_pair
        compiled_k1, compiled_k2 = compiled_pair
        kernels_compiled = time.monotonic()

        runtime_drain = None
        runtime_finalizer = None
        # Like K1/K2, these kernels take runtime tensor/stream arguments.
        # Reuse code per workspace bucket, while each graph binds its own
        # arguments. Recompiling the persistent drain per output grows the
        # CuTe compiler state for no specialization benefit.
        compiled_aux = self.workspace._compiled_aux.get(self.compile_bucket)
        compiled_drain, compiled_finalizer = compiled_aux or (None, None)
        k2_drain_launch_clusters = 0
        k2_drain = getattr(bundle, "k2_drain", None)
        k2_finalizer = getattr(bundle, "k2_finalizer", None)
        if k2_drain is not None:
            runtime_drain = dict(base_runtime, stream=k1_cuda)
            # The dual-N8 specialization is verified at two resident CTAs/SM.
            # Use both slots on the SMs released by K1 during the drain phase.
            drain_ctas_per_sm = (
                2
                if (spec.kernel.k2_warps == 12 and spec.kernel.k2_tile[1] == 16)
                else 1
            )
            k2_drain_launch_clusters = spec.kernel.k1_sms * drain_ctas_per_sm
            if compiled_aux is None:
                compiled_drain = cute.compile(
                    k2_drain,
                    **dict(
                        runtime_drain,
                        max_active_clusters=k2_drain_launch_clusters,
                    ),
                )

        if k2_finalizer is not None:
            runtime_finalizer = dict(base_runtime, stream=root_cuda)
            if compiled_aux is None:
                compiled_finalizer = cute.compile(
                    k2_finalizer,
                    **dict(runtime_finalizer, max_active_clusters=1),
                )

        if compiled_aux is None:
            self.workspace._compiled_aux[self.compile_bucket] = (
                compiled_drain,
                compiled_finalizer,
            )

        combine_bucket = combine_output[: self.compile_bucket]
        output_bucket = inputs.output[: self.compile_bucket]
        if spec.kernel.rank_local_combine:
            if execution.rank_combine_ready is None:
                raise RuntimeError("NVFP4 rank-local combine ready storage is missing")
            compiled_k3, runtime_k3 = compile_rank_local_combine(
                combine_bucket,
                execution.rank_combine_ready,
                output_bucket,
                inputs.topk_ids,
                world_size=self.config.world_size,
                num_experts_per_rank=self.config.local_experts,
                stream=root_cuda,
            )
        else:
            # Unrouted (negative) slots of a valid token have no K2 row in
            # combine_output; K3 must skip them rather than read stale rows.
            k3_plan = compile_combine_reduce(
                combine_bucket,
                output_bucket,
                None,
                topk_idx=inputs.topk_ids,
                stream=root_cuda,
            )
            compiled_k3, combine_cute, output_cute, score_cute, k3_stream, idx_cute = (
                k3_plan
            )
            runtime_k3 = dict(
                combine_cute=combine_cute,
                reduced_cute=output_cute,
                topk_score_cute=score_cute,
                stream=k3_stream,
                topk_idx_cute=idx_cute,
            )

        k1_executor = compiled_k1.to(None)
        k2_executor = compiled_k2.to(None)
        drain_executor = compiled_drain.to(None) if compiled_drain else None
        finalizer_executor = compiled_finalizer.to(None) if compiled_finalizer else None
        k3_executor = compiled_k3.to(None)
        all_kernels_ready = time.monotonic()
        # Preserve per-layer graph/weight bindings; only the device/SM partition
        # is shared by sequential calls on this pooled workspace.
        green_resources = self._green_resources_for(spec)
        network = execution.network
        transport_roles = network.compiled_roles() if network is not None else {}
        transport_kernels = tuple(
            k for group in transport_roles.values() for k in group
        )
        if network is not None:
            transport_kernels += (network._transport.compiled_dispatch_route_builder,)
        graph = NativeGreenContextGraph.capture(
            xnuma=execution.xnuma,
            green_resources=green_resources,
            root_stream=root_stream,
            k1_stream=k1_stream,
            k2_stream=k2_stream,
            k1_executor=k1_executor,
            k2_executor=k2_executor,
            k2_drain_executor=drain_executor,
            k2_finalizer_executor=finalizer_executor,
            k3_executor=k3_executor,
            compiled_kernels=transport_kernels
            + tuple(
                k
                for k in (
                    compiled_k1,
                    compiled_k2,
                    compiled_drain,
                    compiled_finalizer,
                    compiled_k3,
                )
                if k is not None
            ),
            compiled_role_owners={
                0: (compiled_k1, compiled_drain),
                1: (compiled_k2,),
                **transport_roles,
            },
            launch_reset=network.begin if network is not None else None,
            launch_transport_end=network.end if network is not None else None,
            launch_k1=lambda: k1_executor(**runtime_k1),
            launch_k2=lambda: k2_executor(**runtime_k2),
            launch_k2_drain=(
                (lambda: drain_executor(**runtime_drain))
                if drain_executor is not None
                else None
            ),
            launch_k2_finalizer=(
                (
                    lambda: finalizer_executor(
                        **dict(
                            runtime_finalizer,
                            stream=cuda.CUstream(
                                torch.cuda.current_stream().cuda_stream
                            ),
                        )
                    )
                )
                if finalizer_executor is not None
                else None
            ),
            # Used both for normal replay and insertion into an outer graph.
            launch_k3=lambda: k3_executor(
                **dict(
                    runtime_k3,
                    stream=cuda.CUstream(torch.cuda.current_stream().cuda_stream),
                )
            ),
            k1_sm_count=spec.kernel.k1_sms,
            k1_grid_clusters=k1_launch_clusters,
            k2_grid_clusters=k2_launch_clusters,
            k2_drain_grid_clusters=(
                k2_drain_launch_clusters if k2_drain_launch_clusters > 0 else None
            ),
        )
        graph_captured = time.monotonic()
        if os.environ.get("FLASHINFER_MEGAMOE_LOG_COMPILE_SPEC") == "1":
            print(
                "[FlashInfer MegaMoE frontend] "
                f"rank={self.config.rank} "
                f"bucket={self.compile_bucket} "
                f"k2_launch_ctas={k2_launch_clusters} "
                f"k2_drain_launch_ctas={k2_drain_launch_clusters} "
                f"layer={len(self.workspace._frontends)} "
                f"k12_compile={kernels_compiled - profile_start:.3f}s "
                f"k3_compile={all_kernels_ready - kernels_compiled:.3f}s "
                f"graph={graph_captured - all_kernels_ready:.3f}s "
                f"total={graph_captured - profile_start:.3f}s",
                flush=True,
            )
        self._barrier()
        self._compiled = _CompiledSplit(
            execution=execution,
            k1=compiled_k1,
            k2=compiled_k2,
            k2_drain=compiled_drain,
            k2_finalizer=compiled_finalizer,
            k3=compiled_k3,
            streams=(root_stream, k1_stream, k2_stream),
            graph=graph,
            runtime_k1=runtime_k1,
            runtime_k2=runtime_k2,
            runtime_k2_drain=runtime_drain,
            runtime_k2_finalizer=runtime_finalizer,
        )
        return self._compiled

    def _reset_execution(self, execution: _ExecutionBuffers) -> None:
        from moe_sm120_nvfp4_split.runtime import fused_fill

        kernel = execution.bundle.k1
        # MEGA_SPLIT_FUSED_RESET (default on): one fill kernel instead of one
        # memset node per region in front of K1.
        ops = []
        for name in self._LOCAL_RESET_REGIONS:
            if name not in kernel._local_offsets:
                continue
            offset = int(kernel._local_offsets[name])
            size = int(kernel._local_region_by_name[name].nbytes)
            ops.append((execution.local_workspace[offset : offset + size], 0))
        if "rank_combine_route_map" in kernel._local_offsets:
            name = "rank_combine_route_map"
            offset = int(kernel._local_offsets[name])
            size = int(kernel._local_region_by_name[name].nbytes)
            ops.append(
                (
                    execution.local_workspace[offset : offset + size].view(torch.int32),
                    -1,
                )
            )
        if ops and ops[0][0].is_cuda and fused_fill.enabled():
            fused_fill.launch_fused_fill(ops, torch.cuda.current_stream())
        else:
            for tensor, value in ops:
                tensor.fill_(value)
        if execution.rank_combine_ready is not None:
            # Shared by all layers using this bucket. A device increment also
            # advances on every outer-graph replay, unlike a captured fill_(n).
            execution.rank_combine_ready[-1, 0].add_(1)
        # The graph finalizer owns peer-written expert_recv_count and its
        # published sum. It clears both between two rank barriers; clearing
        # either here can erase an early peer's next-epoch dispatch write.
        if execution.green_trace is not None:
            execution.green_trace.zero_()

    def run(self, inputs: MegaMoESm120Nvfp4Inputs) -> torch.Tensor:
        compiled = self._ensure_compiled(inputs)
        # Caller reset -> launch_ready -> Green K1/K2/K3 -> launch_done. Keep
        # cache admission outside the Green graph's kernel-node rebinding.
        if compiled.execution.network is None:
            self._reset_execution(compiled.execution)
        compiled.graph.launch(torch.cuda.current_stream())
        return inputs.output

    def release(self) -> None:
        compiled = self._compiled
        if compiled is None:
            return
        ensure_not_capturing("NVFP4 workspace release")
        compiled.graph.close()
        self._compiled = None


def set_compile_tokens_per_rank(
    workspace: MegaMoESm120Nvfp4Workspace,
    compile_tokens_per_rank: int | None,
) -> None:
    workspace._compile_bucket = select_graph_compile_bucket(
        compile_tokens_per_rank,
        workspace.config.max_tokens_per_rank,
    )


def allocate_workspace(
    config: MegaMoESm120Nvfp4Config,
    *,
    control_group: Any = None,
) -> MegaMoESm120Nvfp4Workspace:
    config.validate()
    ensure_not_capturing("NVFP4 workspace allocation")
    # Plan every supported bucket before allocating the symmetric arena.
    # This preserves one offset table across tensors and avoids registration
    # or allocation during graph capture. Layers share this workspace.
    from types import SimpleNamespace

    from moe_sm120_nvfp4_split.runtime.gpu_topology import (
        discover_gpu_topology,
        gather_ep_transport_topology,
    )

    if _no_dist():
        topology = SimpleNamespace(
            same_numa_peer_count=0, cross_numa_peer_count=0, force_ibgda_peer_mask=0
        )
    else:
        topology = gather_ep_transport_topology(
            discover_gpu_topology(torch.cuda.current_device(), config.rank),
            ep_rank=config.rank,
            ep_control_group=control_group,
        )
    buckets = sorted(
        {
            config.max_tokens_per_rank,
            *(
                n
                for n in DECODE_GRAPH_COMPILE_BUCKETS
                if n <= config.max_tokens_per_rank
            ),
        }
    )
    split_buckets = plan_p2p_split_buckets(config, topology, buckets)
    planned = SimpleNamespace(
        config=config,
        _execution_plans={},
        _topology=topology,
        _green_resources={},
        _bucket_backends={bucket: "p2p_split" for bucket in split_buckets},
    )
    size = config.max_tokens_per_rank
    scale_columns = round_up(ceil_div(config.hidden, 16), 4)
    sizes = [
        size * config.hidden // 2,
        size * scale_columns,
        size * config.num_topk * 8,
        size * config.num_topk * 4,
        size * config.hidden * 2,
    ]
    split_shared_bytes = {}
    split_fsplits: list[int] = []
    split_packs: list[bool] = []
    push_buckets: list[int] = []
    for bucket in buckets:
        frontend = MegaMoESm120Nvfp4Frontend(
            planned, compile_bucket=bucket, control_group=control_group
        )
        spec, bundle = frontend._ensure_plan(bucket)
        if bucket in split_buckets and not _p2p_split_plan_supported(spec, bundle):
            # The xnuma forwarder needs the 8-warp K2 and per-slot combine.
            split_buckets.discard(bucket)
            del planned._bucket_backends[bucket]
            del planned._execution_plans[bucket]
            spec, bundle = frontend._ensure_plan(bucket)
        if bucket in split_buckets:
            split_shared_bytes[bucket] = bundle.shared_workspace_bytes
            split_fsplits.append(spec.kernel.xnuma_fsplit)
            split_packs.append(_xnuma_pack(spec.kernel))
        if (
            getattr(spec.kernel, "push_dispatch", False)
            and spec.kernel.kernel_comm_backend == "p2p_direct"
            and not _no_dist()
        ):
            push_buckets.append(bucket)
        sizes += [
            bundle.shared_workspace_bytes,
            bucket * config.num_topk * config.hidden * 2,
        ]
        if spec.kernel.rank_local_combine:
            sizes.append((bucket + 1) * config.world_size * 4)

    def page(n):
        return (n + 4095) // 4096 * 4096

    # Push dispatch: world shadows of the input span (x | x_scale | topk ids |
    # topk weights) right after it, each with the span's layout; source s's
    # rows land at x + (s + 1) * stride on the owner.
    push_stride = sum(page(n) for n in sizes[:4]) if push_buckets else 0
    if push_buckets:
        sizes.insert(4, config.world_size * push_stride)
    n_prefix = 6 if push_buckets else 5
    capacity = sum(page(n) for n in sizes)
    shadow = region = None
    shadow_bytes = reserved_bytes = 0
    if split_buckets:
        from moe_sm120_nvfp4_split.transport.shadow import ShadowLayout
        from moe_sm120_nvfp4_split.transport.xnuma_layout import XnumaLayout

        shadow = ShadowLayout.from_topology(topology, config.rank)
        xnuma_fsplit = _uniform_xnuma_fsplit(split_fsplits)
        # One control-region placement for every p2p_split bucket: commit
        # words are cumulative signal-adds keyed by address.
        region = XnumaLayout(
            fsplit=xnuma_fsplit,
            world=config.world_size,
            n_cross=shadow.num_cross,
            experts_per_rank=config.local_experts,
            tokens_per_rank=max(split_buckets),
            topk=config.num_topk,
            hidden=config.hidden,
            pack=any(split_packs),
        )
        reserved_bytes = region.reserved_bytes
        # Shadows cover the inputs and output, then each p2p_split bucket's
        # combine output and shared workspace, allocated first below.
        shadow_bytes = sum(page(n) for n in sizes[:n_prefix]) + sum(
            page(bucket * config.num_topk * config.hidden * 2)
            + page(split_shared_bytes[bucket])
            for bucket in sorted(split_buckets)
        )
    session = (
        None
        if _no_dist()
        else PeerMemorySession(
            config.rank,
            config.world_size,
            capacity,
            control_group,
            network_peer_mask=topology.force_ibgda_peer_mask,
            shadow=shadow,
            shadow_bytes=shadow_bytes,
            reserved_bytes=reserved_bytes,
        )
    )
    roots: list[torch.Tensor] = []
    x, root = sym_byte_view(
        (config.max_tokens_per_rank, config.hidden // 2),
        ACTIVATION_DTYPE,
        session=session,
    )
    roots.append(root)
    scale_columns = round_up(ceil_div(config.hidden, 16), 4)
    x_scale, root = sym_byte_view(
        (config.max_tokens_per_rank, scale_columns), SCALE_DTYPE, session=session
    )
    roots.append(root)
    topk_ids = sym_zeros(
        (config.max_tokens_per_rank, config.num_topk), torch.int64, session=session
    )
    topk_ids.fill_(-1)
    roots.append(topk_ids)
    topk_weights = sym_zeros(
        (config.max_tokens_per_rank, config.num_topk), torch.float32, session=session
    )
    roots.append(topk_weights)
    push_shadow = None
    if push_buckets:
        push_shadow = sym_zeros(
            (config.world_size * push_stride,), torch.uint8, session=session
        )
        roots.append(push_shadow)
        stride = int(push_shadow.data_ptr()) - int(x.data_ptr())
        span_end = int(topk_weights.data_ptr()) + topk_weights.numel() * 4
        if stride != push_stride or span_end - int(x.data_ptr()) > stride:
            raise RuntimeError("push dispatch shadow must follow the input span")
        from moe_sm120_nvfp4_split.token_comm import PushDispatchConfig

        for bucket in push_buckets:
            tc = planned._execution_plans[bucket][1].k1.token_comm
            mask = tc.barrier_peer_mask & ~(1 << config.rank)
            peers = tuple(r for r in range(config.world_size) if (mask >> r) & 1)
            if peers:
                tc.push = PushDispatchConfig(stride=stride, peers=peers, peer_mask=mask)
    output, root = sym_byte_view(
        (config.max_tokens_per_rank, config.hidden), torch.bfloat16, session=session
    )
    roots.append(root)
    split_buffers = {}
    for bucket in sorted(split_buckets):
        combine_output, root = sym_byte_view(
            (bucket, config.num_topk, config.hidden), torch.bfloat16, session=session
        )
        roots.append(root)
        shared_workspace = sym_zeros(
            (split_shared_bytes[bucket],), torch.uint8, session=session
        )
        roots.append(shared_workspace)
        split_buffers[bucket] = (combine_output, shared_workspace)
    if split_buckets and session.cursor > session.shadow_limit:
        raise RuntimeError("p2p_split buffers exceed the shadowed arena prefix")
    epilogue_args = tuple(
        torch.ones((config.local_experts,), dtype=torch.float32, device="cuda")
        for _ in range(3)
    )
    workspace = MegaMoESm120Nvfp4Workspace(
        config=config,
        x=x,
        x_scale=x_scale,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        output=output,
        fc1_alpha=epilogue_args[0],
        fc2_alpha=epilogue_args[1],
        fc1_norm_const=epilogue_args[2],
        _sym_roots=roots,
        _comm_session=session,
        _execution_plans=planned._execution_plans,
        _green_resources=planned._green_resources,
        _topology=topology,
        _control_group=control_group,
        _compile_bucket=config.max_tokens_per_rank,
        _bucket_backends=planned._bucket_backends,
        _split_buffers=split_buffers,
        _xnuma_region=region,
        _xnuma_epoch=(
            torch.zeros((1,), dtype=torch.int64, device="cuda")
            if split_buckets
            else None
        ),
        _xnuma_fsplit=xnuma_fsplit if split_buckets else 2,
    )
    return workspace


def _xnuma_pack(kernel_config) -> bool:
    """p2p_split packed dispatch (MEGAMOE_XNUMA_PACK=0|1 overrides the config)."""
    env = os.environ.get("MEGAMOE_XNUMA_PACK", "")
    if env in ("0", "1"):
        return env == "1"
    return bool(getattr(kernel_config, "xnuma_pack", False))


def _uniform_xnuma_fsplit(selected: list[int]) -> int:
    """One forwarder split for all p2p_split buckets of a workspace.

    The buckets share one epoch and cumulative commit words, and the scatter
    waits for ``fsplit * epoch`` commits, so every call must commit the same
    number of forwarder CTAs. Take the value most buckets selected (ties: the
    larger); ``MEGAMOE_XNUMA_FSPLIT`` overrides it (diagnostics).
    """
    override = os.environ.get("MEGAMOE_XNUMA_FSPLIT")
    if override is not None:
        return int(override)
    if not selected:
        return 2
    return max(set(selected), key=lambda value: (selected.count(value), value))


def _frontend_graph_cache_key(transformed_weights, output, bucket):
    return tuple(
        tensor.data_ptr() for pair in transformed_weights for tensor in pair
    ) + (output.data_ptr(), bucket)


def run_split_mega_moe(
    workspace: MegaMoESm120Nvfp4Workspace,
    transformed_weights: TransformedWeights,
    *,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    if workspace._destroyed:
        raise RuntimeError("NVFP4 workspace has been destroyed")
    (fc1_weight, fc1_scale), (fc2_weight, fc2_scale) = transformed_weights
    graph_output = workspace.output if output is None else output
    weight_key = _frontend_graph_cache_key(
        transformed_weights, graph_output, workspace._compile_bucket
    )
    frontend = workspace._frontends.get(weight_key)
    if frontend is None:
        frontend = MegaMoESm120Nvfp4Frontend(
            workspace,
            compile_bucket=workspace._compile_bucket,
            control_group=workspace._control_group,
        )
        workspace._frontends[weight_key] = frontend
    inputs = MegaMoESm120Nvfp4Inputs(
        activation=workspace.x[: workspace._compile_bucket],
        activation_scale=workspace.x_scale[: workspace._compile_bucket],
        topk_ids=workspace.topk_ids[: workspace._compile_bucket],
        topk_weights=workspace.topk_weights[: workspace._compile_bucket],
        fc1_weight=fc1_weight,
        fc1_weight_scale=fc1_scale,
        fc2_weight=fc2_weight,
        fc2_weight_scale=fc2_scale,
        output=graph_output[: workspace._compile_bucket],
    )
    return frontend.run(inputs)


__all__ = [
    "DECODE_GRAPH_COMPILE_BUCKETS",
    "MegaMoESm120Nvfp4Config",
    "MegaMoESm120Nvfp4Inputs",
    "MegaMoESm120Nvfp4Workspace",
    "allocate_workspace",
    "run_split_mega_moe",
    "select_graph_compile_bucket",
    "set_compile_tokens_per_rank",
]
