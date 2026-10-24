# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Assembly boundary for the native persistent MXFP8 x MXFP4 MegaMoE."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Callable

import torch

if TYPE_CHECKING:
    from .frontend import FrontendConfig, KernelLaunchSpec


_logger = logging.getLogger(__name__)


class KernelAssemblyPendingError(RuntimeError):
    """Raised when a native assembly dependency is unavailable."""


@dataclass(frozen=True)
class DeviceAssemblyProtocol:
    required_launch_regions: tuple[str, ...] = ("local", "shared", "route_terms")
    required_local_views: tuple[str, ...] = (
        "l1_arrival_count",
        "expert_send_count",
        "grid_sync_counter",
        "fc1_done_counter",
        "l1_token_buffer",
        "nvlink_barrier_counter",
        "l1_sf_buffer",
        "l1_topk_weights_buffer",
        "token_src_metadata",
        "fc1_output",
        "fc1_output_sf",
    )
    dispatch_products: tuple[str, ...] = (
        "expert_recv_count_sum",
        "expert_pool_offsets",
        "token_src_metadata",
        "fc1_ready_counters",
    )
    weight_scale_products: tuple[str, ...] = (
        "fc1_weight_sfa_logical",
        "fc2_weight_sfa_logical",
    )


DEVICE_ASSEMBLY_PROTOCOL = DeviceAssemblyProtocol()
# The M128/N128 single-rank assembly passed the B200 compile and numerical gate.
ASSEMBLY_READY = True


def _missing_protocols() -> tuple[str, ...]:
    probes = (
        ("device_workspace", "DeviceWorkspacePartition"),
        ("dispatch", "SingleRankMxfp8DispatchPhase"),
        ("weight_scale_views", "SwizzledWeightScaleViews"),
        ("packed_weight_views", "PackedWeightViews"),
    )
    missing = []
    for module, symbol in probes:
        try:
            imported = __import__(f"{__package__}.{module}", fromlist=[symbol])
            getattr(imported, symbol)
        except (ImportError, AttributeError):
            missing.append(f"{module}.{symbol}")
    return tuple(missing)


@dataclass(frozen=True)
class _CompiledAssembly:
    persistent: Any
    reducer: Any
    num_sms: int


@dataclass(frozen=True)
class _BoundMegaLaunch:
    """Warmed runtime views; stream rebinding never revisits DLPack adapters."""

    compiled: Any
    runtime: dict[str, Any]
    reducer_runtime: dict[str, Any]
    partition: Any
    route_terms: Any
    world_size: int
    launch_stream: Any

    def __call__(self) -> None:
        with torch.cuda.stream(self.launch_stream):
            # Shared prefixes are collectively reset in the preceding
            # kernel tail. A host-side shared clear can erase peer writes.
            if self.world_size == 1:
                self.partition.zero_prefixes_()
            # No route_terms clear: the reducer skips -1 routes and FC2
            # rewrites every valid route row each launch, including routes
            # that were emptied and restored between launches.
            self.compiled.persistent(**self.runtime)
            if self.compiled.reducer is not None:
                self.compiled.reducer(**self.reducer_runtime)

    def rebind_stream(self, stream: int):
        import cuda.bindings.driver as cuda

        runtime = dict(self.runtime)
        reducer_runtime = dict(self.reducer_runtime)
        runtime["stream"] = cuda.CUstream(stream)
        reducer_runtime["stream"] = runtime["stream"]
        return replace(
            self,
            runtime=runtime,
            reducer_runtime=reducer_runtime,
            launch_stream=torch.cuda.ExternalStream(
                stream, device=self.route_terms.device
            ),
        )


def _to_cute(tensor: torch.Tensor, align: int) -> Any:
    import cutlass.torch as cutlass_torch

    result = cutlass_torch.from_dlpack(tensor, assumed_align=align)
    return result.mark_layout_dynamic(leading_dim=cutlass_torch.get_leading_dim(tensor))


def _layout_adapter(cls, *, hidden: int, intermediate: int, experts: int):
    obj = cls.__new__(cls)
    obj.hidden_size = hidden
    obj.intermediate_size = intermediate
    obj.num_experts = experts
    return obj


class _MegaAdapterBindings:
    """Frontend adapter for one native persistent kernel plus reducer."""

    def __init__(self, *, workspace, tactic, gate_up_clamp, fast_math) -> None:
        self.workspace = workspace
        self.tactic = dict(tactic)
        self.gate_up_clamp = gate_up_clamp
        self.fast_math = bool(fast_math)

    @staticmethod
    def assembly_protocol() -> DeviceAssemblyProtocol:
        return DEVICE_ASSEMBLY_PROTOCOL

    def _validate_static_contract(self, config, spec) -> None:
        expected = {
            "hidden": config.hidden,
            "intermediate": config.intermediate,
            "num_topk": config.topk,
            "num_local_experts": config.num_experts,
            "max_tokens_per_rank": config.max_tokens,
        }
        mismatch = {
            key: (getattr(self.workspace, key, None), value)
            for key, value in expected.items()
            if getattr(self.workspace, key, None) != value
        }
        if mismatch:
            raise ValueError(f"frontend/workspace static contract mismatch: {mismatch}")
        missing = tuple(
            key
            for key in DEVICE_ASSEMBLY_PROTOCOL.required_launch_regions
            if key not in spec.workspace_regions
        )
        if missing:
            raise ValueError(
                "persistent kernel launch is missing workspace regions: "
                + ", ".join(missing)
            )
        if self.tactic != dict(self.workspace.tactic):
            raise ValueError("adapter tactic does not match workspace tactic")
        mma_m = self.tactic["mma_tiler_mnk"][0]
        if mma_m not in (128, 256):
            raise ValueError("persistent assembly supports M128 or M256 (2-CTA)")
        if mma_m == 256:
            if tuple(self.tactic["cluster_shape_mnk"]) != (2, 1, 1):
                raise ValueError("M256 (2-CTA) requires cluster_shape_mnk=(2, 1, 1)")
            if config.hidden % 256 or (2 * config.intermediate) % 256:
                raise ValueError(
                    "M256 (2-CTA) requires hidden and 2*intermediate to be "
                    "multiples of 256"
                )
        # N32 is not supported yet: it faults on a misaligned shared-memory
        # access and needs 32-token dispatch padding.
        if self.tactic["mma_tiler_mnk"][1] not in (64, 128, 192):
            raise ValueError("persistent assembly supports N64, N128, or N192")

    def _runtime(self, spec, num_sms: int) -> dict[str, Any]:
        import cuda.bindings.driver as cuda
        from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe.src.src.sym_buffer import (
            SymBufferHost,
        )
        from .device_workspace import DeviceWorkspacePartition
        from .packed_weight_views import PackedWeightViews
        from .weight_scale_views import SwizzledWeightScaleViews

        partition = DeviceWorkspacePartition.from_launch_regions(
            self.workspace.plan, spec.workspace_regions
        )
        v = partition.to_cute()
        weights = PackedWeightViews(
            spec.fc1_weight,
            spec.fc2_weight,
            hidden_size=self.workspace.hidden,
            intermediate_size=self.workspace.intermediate,
        ).to_cute()
        scales = SwizzledWeightScaleViews(
            spec.fc1_weight_scales,
            spec.fc2_weight_scales,
            hidden_size=self.workspace.hidden,
            intermediate_size=self.workspace.intermediate,
        ).to_cute()
        local_zero = partition.local_workspace[
            : partition.plan.local.zero_prefix_bytes
        ].view(torch.int32)
        shared_zero = partition.shared_workspace[
            : partition.plan.shared.zero_prefix_bytes
        ].view(torch.int32)
        input_sf = spec.activation_scales.view(torch.uint8).view(torch.int32)
        return dict(
            input_token_buffer=_to_cute(spec.activation, 16),
            input_sf_buffer=_to_cute(input_sf, 4),
            input_topk_idx=_to_cute(spec.route_ids, 8),
            input_topk_weights=_to_cute(spec.route_scores, 4),
            fc1_weight=weights.fc1_weight,
            fc2_weight=weights.fc2_weight,
            fc1_weight_scales=scales.fc1_weight_scales,
            fc2_weight_scales=scales.fc2_weight_scales,
            expert_send_count=v["expert_send_count"],
            expert_recv_count=v["shared.expert_recv_count"],
            expert_recv_count_sum=v["shared.expert_recv_count_sum"],
            src_token_topk_idx=v["shared.src_token_topk_idx"],
            token_src_metadata=v["token_src_metadata"],
            l1_arrival_count=v["l1_arrival_count"],
            l1_token_buffer=v["l1_token_buffer"],
            l1_sf_buffer=v["l1_sf_buffer"],
            l1_topk_weights_buffer=v["l1_topk_weights_buffer"],
            nvlink_barrier_signal=v["shared.nvlink_barrier_signal"],
            nvlink_barrier_counter=v["nvlink_barrier_counter"],
            grid_sync_counter=v["grid_sync_counter"],
            fc1_done_counter=v["fc1_done_counter"],
            fc1_output=v["fc1_output"],
            fc1_output_sf=v["fc1_output_sf"],
            route_terms=v["route_terms"],
            output=_to_cute(spec.output, 16),
            local_zero_prefix=_to_cute(local_zero, 4),
            shared_zero_prefix=_to_cute(shared_zero, 4),
            load_balance_counter=v.get("load_balance_counter"),
            peer_rank_ptr_mapper_host=SymBufferHost(
                base_addr=self.workspace.symmetric_base or spec.activation.data_ptr(),
                offsets=self.workspace.peer_offsets_list,
                rank_idx=self.workspace.rank,
                num_max_ranks=self.workspace.world_size,
            ),
            stream=cuda.CUstream(spec.stream),
            expert_data_row_offsets=None,
            expert_scale_row_offsets=None,
        )


class PersistentFc12KernelBase:
    """Shared compute assembly; communication is a compile-time specialization."""

    # Concrete launchers supply these before assembling the shared kernel.
    tactic: dict[str, Any]
    workspace: Any
    gate_up_clamp: float | None
    fast_math: bool

    mega = False
    resolved_num_stages = None
    resolved_persistent_clusters = None

    def _fit_operand_stages(
        self,
        config,
        *,
        wc,
        local_rank,
        warps,
        cluster_shape_mn,
        load_balance_mode,
        flag_batch,
    ) -> int:
        """Deepest operand pipeline that fits shared memory for this geometry.

        Builds the same scheduler and dispatch storage the device region
        allocates, so the fit tracks hidden size and routing geometry.
        """
        from types import SimpleNamespace

        import cutlass

        from .device_scheduler import DeviceSchedulerParams
        from .dispatch import Mxfp8DispatchPhase
        from .kernel_pipelines import fit_operand_stages

        dispatch = None
        if self.mega:
            dispatch = Mxfp8DispatchPhase(
                wc,
                rank=local_rank,
                cluster_shape_mn=cluster_shape_mn,
                dispatch_warp_start=warps.dispatch[0],
                num_other_warps=len(warps.epilogue) + 4,
                flag_batch=flag_batch,
                cta_local_counts=self.tactic.get("cta_local_counts", False),
            )
        fit = fit_operand_stages(
            mma_tiler_mnk=tuple(self.tactic["mma_tiler_mnk"]),
            scheduler_params=SimpleNamespace(
                num_stages=DeviceSchedulerParams.DEFAULT_NUM_STAGES,
                expert_count=config.num_experts,
                load_balance_mode=load_balance_mode,
            ),
            dispatch_phase=dispatch,
            epilogue_warps=len(warps.epilogue),
            smem_capacity_bytes=cutlass.utils.get_smem_capacity_in_bytes(),
            # 2-CTA stages are smaller per CTA; allow the deeper pipelines
            # they fit (the 1-CTA cap stays at the measured default).
            maximum_stages=12 if self.tactic["mma_tiler_mnk"][0] == 256 else 8,
        )
        _logger.info(
            "MXFP8 x MXFP4 mega: num_stages=auto -> %d (%d of %d bytes, H=%d I=%d)",
            fit.stages,
            fit.used_bytes,
            fit.available_bytes,
            config.hidden,
            config.intermediate,
        )
        return fit.stages

    def _make_launcher(self, config, max_active_clusters: int):
        import cutlass
        import cutlass.cute as cute
        import cutlass.utils.blockscaled_layout as blockscaled_utils
        from cutlass.cute.nvgpu import cpasync
        from .device_epilogue import (
            EXCHANGE_LANE_STRIDE,
            EXCHANGE_ROWS,
            DeviceEpilogueConfig,
            PersistentMxfp8Mxfp4Epilogue,
            EPILOGUE_WARPS,
        )
        from .device_scheduler import DeviceSchedulerParams
        from .dispatch import Mxfp8DispatchPhase
        from .fc2_epilogue import Fc2RouteStore
        from .kernel_driver import (
            DriverViews,
            PersistentM128DeviceDriver,
            PhaseTmaViews,
        )
        from .kernel_pipelines import KernelPipelines, KernelWarpIds, OperandSmem
        from .packed_weight_views import PackedWeightViews
        from .scale_tma_staging import Mxfp8Mxfp4TmaStaging
        from .weight_scale_views import SwizzledWeightScaleViews

        # CuTe's staged-function frontend resolves module symbols by global
        # name rather than retaining ordinary Python closure cells.  Publish
        # these lazy imports only once compilation has explicitly been asked
        # for, preserving CPU-only import safety.
        globals().update(
            cutlass=cutlass,
            cute=cute,
            blockscaled_utils=blockscaled_utils,
        )

        requested_clusters = self.tactic.get("persistent_clusters")
        clusters = (
            max_active_clusters
            if requested_clusters is None
            else min(requested_clusters, max_active_clusters)
        )
        if clusters < 1:
            raise ValueError("at least one persistent cluster is required")
        self.resolved_persistent_clusters = clusters
        m, n, _ = self.tactic["mma_tiler_mnk"]
        cm, cn, _ = self.tactic["cluster_shape_mnk"]
        mega = self.mega
        cta_local_counts = self.tactic.get("cta_local_counts", False)
        in_kernel_reduce = mega and self.tactic.get("in_kernel_reduce", False)
        # The tactic selects the epilogue width (Mega always resolves one);
        # the lean proxy's fixed tactic has none and follows the environment.
        epilogue_warps = self.tactic.get("epilogue_warps", EPILOGUE_WARPS)
        warps = KernelWarpIds.for_epilogue(epilogue_warps, mega=mega)
        # wait accounting (debug knob): the mega stats region offset.
        from .wait_stats_config import mega_stats_word_offset

        workspace = getattr(self, "workspace", None)
        stats_word_offset = (
            mega_stats_word_offset(workspace.plan)
            if mega and workspace is not None
            else None
        )
        weight_layouts = _layout_adapter(
            PackedWeightViews,
            hidden=config.hidden,
            intermediate=config.intermediate,
            experts=config.num_experts,
        )
        sf_layouts = _layout_adapter(
            SwizzledWeightScaleViews,
            hidden=config.hidden,
            intermediate=config.intermediate,
            experts=config.num_experts,
        )
        wc = self.workspace.plan.config
        local_rank = self.workspace.rank
        pool_tokens = wc.pool_token_capacity
        input_sf_capacity = wc.pool_sf_capacity
        output_sf_capacity = (
            wc.pool_token_capacity + wc.num_experts_per_rank * wc.sf_padding_block
            if mega
            else wc.pool_sf_capacity
        )
        cluster_size = cm * cn
        active_sms = clusters * cluster_size
        group_hint = self.tactic.get("group_hint") or clusters
        # Per-CTA feature tile: a 2-CTA (M256) MMA spans two CTAs of 128 rows.
        cta_m = m // 2 if m == 256 else m
        if m == 256 and (config.hidden % 256 or (2 * config.intermediate) % 256):
            raise ValueError(
                "M256 (2-CTA) requires hidden and 2*intermediate to be multiples of 256"
            )
        # Every CTA publishes FC1 completion once per 128-row tile, so FC2
        # waits for all ceil(2I / cta_m) tiles of the token block.
        fc2_threshold = (2 * config.intermediate + cta_m - 1) // cta_m
        claim_order = (
            self.tactic.get("claim_order", "expert") if self.mega else "expert"
        )
        load_balance_mode = self.tactic["load_balance_mode"]
        if not self.mega and load_balance_mode != "static":
            raise ValueError("lean FC12 requires static load balancing")
        flag_batch = self.tactic["flag_batch"]
        weight_l2_evict_first = self.tactic.get("weight_l2_evict_first", False)
        num_stages = self.tactic["num_stages"]
        if num_stages == "auto":
            num_stages = self._fit_operand_stages(
                config,
                wc=wc,
                local_rank=local_rank,
                warps=warps,
                cluster_shape_mn=(cm, cn),
                load_balance_mode=load_balance_mode,
                flag_batch=flag_batch,
            )
        self.resolved_num_stages = num_stages
        _logger.info(
            "MXFP8 x MXFP4 mega: persistent_clusters=%d, num_stages=%d",
            clusters,
            num_stages,
        )
        gate_up_clamp = self.gate_up_clamp
        fast_math = self.fast_math

        @cute.kernel
        def kernel(
            p1a,
            p1at,
            p1b,
            p1bt,
            p1sa,
            p1sat,
            p1sb,
            p1sbt,
            p2a,
            p2at,
            p2b,
            p2bt,
            p2sa,
            p2sat,
            p2sb,
            p2sbt,
            dispatch_args,
            scheduler_params,
            fc1_output,
            fc1_output_sf,
            fc1_done_counter,
            route_terms,
            reduce_output,
        ):
            # Recreate static layout owners in the isolated device region.
            # TMA descriptors/tensors themselves were formed in the enclosing
            # JIT trace and are the only phase objects crossing this boundary.
            staging = Mxfp8Mxfp4TmaStaging(
                mma_tiler_mn=(m, n),
                cluster_shape_mn=(cm, cn),
                num_stages=num_stages,
            )
            dispatch = None
            if cutlass.const_expr(mega):
                dispatch = Mxfp8DispatchPhase(
                    wc,
                    rank=local_rank,
                    cluster_shape_mn=(cm, cn),
                    dispatch_warp_start=warps.dispatch[0],
                    num_other_warps=len(warps.epilogue) + 4,
                    flag_batch=flag_batch,
                    cta_local_counts=cta_local_counts,
                )
            epilogue = PersistentMxfp8Mxfp4Epilogue(
                DeviceEpilogueConfig(
                    cta_tile_features=cta_m,
                    cta_tile_tokens=n,
                    epilogue_warps=len(warps.epilogue),
                    gate_up_clamp=gate_up_clamp,
                    fast_math=fast_math,
                )
            )
            # FC1 publishes scale bytes in the same 32x4x4 atom layout that
            # FC2's SFB TMA descriptor consumes.  The runtime tensor only
            # describes the backing row-major allocation; reconstruct the
            # logical scale-factor view in this device trace before stores.
            fc1_output_sf_logical = cute.make_tensor(
                fc1_output_sf.iterator,
                blockscaled_utils.tile_atom_to_shape_SF(
                    (
                        output_sf_capacity,
                        config.intermediate,
                        1,
                    ),
                    32,
                ),
            )
            # The workspace is contiguous with a compile-time intermediate
            # width. Expose that stride to FC1's native byte stores so token
            # offsets fold into immediates instead of dynamic row multiplies.
            fc1_output_logical = cute.make_tensor(
                fc1_output.iterator,
                cute.make_layout(
                    (pool_tokens, config.intermediate), stride=(config.intermediate, 1)
                ),
            )
            phase1 = (p1a, p1at, p1b, p1bt, p1sa, p1sat, p1sb, p1sbt)
            phase2 = (p2a, p2at, p2b, p2bt, p2sa, p2sat, p2sb, p2sbt)
            bidx, _, _ = cute.arch.block_idx()
            mma_v = bidx % cute.size(staging.tiled_mma.thr_id.shape)
            rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
            coord = staging.cluster_layout_vmnk.get_flat_coord(rank)
            sfb_coord = staging.cluster_layout_sfb_vmnk.get_flat_coord(rank)
            smem = cutlass.utils.SmemAllocator()
            pipes = KernelPipelines(
                staging=staging,
                scheduler_params=scheduler_params,
                dispatch_phase=dispatch,
                epilogue=epilogue,
                warp_ids=warps,
            )
            storage = smem.allocate(pipes.shared_storage_type())
            if cutlass.const_expr(mega and cta_local_counts):
                scheduler_params.expert_token_sizes = dispatch.bind_local_counts(
                    dispatch_args, storage.dispatch
                )
            epilogue_exchange = cute.make_tensor(
                storage.epilogue_exchange.data_ptr(),
                cute.make_layout(
                    (len(warps.epilogue) // 2, EXCHANGE_ROWS, 32),
                    stride=(
                        EXCHANGE_ROWS * EXCHANGE_LANE_STRIDE,
                        EXCHANGE_LANE_STRIDE,
                        1,
                    ),
                ),
            )
            operands = pipes.allocate_operand_smem(smem)
            if cutlass.const_expr(in_kernel_reduce):
                dispatch.bind_in_kernel_reduce(route_terms, reduce_output)
            mainloop = pipes.create_mainloop_pipelines(storage)
            sched = pipes.create_scheduler(
                storage, block_idx=cute.arch.block_idx(), grid_dim=cute.arch.grid_dim()
            )
            tmem = pipes.make_tmem_allocator(storage)

            def partition(parts):
                return staging.partition_tma_loads(
                    tma_atom_a=parts[0],
                    tma_tensor_a=parts[1],
                    tma_atom_b=parts[2],
                    tma_tensor_b=parts[3],
                    tma_atom_sfa=parts[4],
                    tma_tensor_sfa=parts[5],
                    tma_atom_sfb=parts[6],
                    tma_tensor_sfb=parts[7],
                    smem_a=operands.weight,
                    smem_b=operands.activation,
                    smem_sfa=operands.weight_scale,
                    smem_sfb=operands.activation_scale,
                    mma_tile_coord_v=mma_v,
                    block_in_cluster_coord_vmnk=coord,
                    block_in_cluster_coord_sfb_vmnk=sfb_coord,
                )

            p1, p2 = partition(phase1), partition(phase2)
            masks = staging.make_multicast_masks(coord, sfb_coord)
            pv1 = PhaseTmaViews(
                atoms=(phase1[0], phase1[2], phase1[4], phase1[6]),
                global_partitions=(p1[1], p1[3], p1[5], p1[7]),
                tensors=(phase1[1], phase1[3], phase1[5], phase1[7]),
                multicast_masks=masks,
            )
            pv2 = PhaseTmaViews(
                atoms=(phase2[0], phase2[2], phase2[4], phase2[6]),
                global_partitions=(p2[1], p2[3], p2[5], p2[7]),
                tensors=(phase2[1], phase2[3], phase2[5], phase2[7]),
                multicast_masks=masks,
            )
            warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            if warp == pipes.warp_ids.tma_a or warp == pipes.warp_ids.tma_b:
                for atom in phase1[::2]:
                    cpasync.prefetch_descriptor(atom)
                for atom in phase2[::2]:
                    cpasync.prefetch_descriptor(atom)
            fc2_store = Fc2RouteStore(
                route_terms,
                dispatch_args.peer_rank_ptr_mapper if mega else None,
            )
            PersistentM128DeviceDriver(
                kernel_pipelines=pipes,
                stats_word_offset=stats_word_offset,
                weight_l2_evict_first=weight_l2_evict_first,
            )(
                mainloop_pipelines=mainloop,
                scheduler_bundle=sched,
                operand_smem=operands,
                tma_operand_smem=OperandSmem(p1[0], p1[2], p1[4], p1[6]),
                tmem_allocator=tmem,
                views=DriverViews(
                    fc1=pv1,
                    fc2=pv2,
                    fc1_output=fc1_output_logical,
                    fc1_output_scales=fc1_output_sf_logical,
                    route_store=fc2_store,
                    fc1_done_counter=fc1_done_counter,
                    token_metadata=dispatch_args.token_src_metadata if mega else None,
                ),
                dispatch_args=dispatch_args,
                dispatch_storage=storage.dispatch if mega else None,
                epilogue_exchange=epilogue_exchange,
                fc1_k_tiles=(config.hidden + staging.mma_tiler[2] - 1)
                // staging.mma_tiler[2],
                fc2_k_tiles=(config.intermediate + staging.mma_tiler[2] - 1)
                // staging.mma_tiler[2],
            )

        class Launcher:
            @cute.jit
            def __call__(
                self,
                input_token_buffer,
                input_sf_buffer,
                input_topk_idx,
                input_topk_weights,
                fc1_weight,
                fc2_weight,
                fc1_weight_scales,
                fc2_weight_scales,
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
                fc1_done_counter,
                fc1_output,
                fc1_output_sf,
                route_terms,
                output,
                local_zero_prefix,
                shared_zero_prefix,
                load_balance_counter,
                peer_rank_ptr_mapper_host,
                stream,
                expert_data_row_offsets,
                expert_scale_row_offsets,
            ):
                # Descriptor construction belongs to this enclosing JIT trace.
                staging = Mxfp8Mxfp4TmaStaging(
                    mma_tiler_mn=(m, n),
                    cluster_shape_mn=(cm, cn),
                    num_stages=num_stages,
                )
                args = None
                counts = expert_recv_count_sum
                if cutlass.const_expr(mega):
                    args = self.make_dispatch_args(
                        input_token_buffer,
                        input_sf_buffer,
                        input_topk_idx,
                        input_topk_weights,
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
                        route_terms,
                    )
                    counts_i32 = cute.recast_tensor(
                        expert_recv_count_sum, cutlass.Int32
                    )
                    counts = cute.make_tensor(
                        counts_i32.iterator,
                        cute.make_layout((config.num_experts,), stride=(2,)),
                    )
                params = DeviceSchedulerParams(
                    expert_token_sizes=counts,
                    expert_count=config.num_experts,
                    expert_data_row_offsets=expert_data_row_offsets,
                    expert_scale_row_offsets=expert_scale_row_offsets,
                    gate_up_size=2 * config.intermediate,
                    hidden_size=config.hidden,
                    cta_tile_tokens=n,
                    cta_tile_features=cta_m,
                    cluster_shape_token_feature=(cn, cm),
                    group_hint=group_hint,
                    token_padding_block=wc.token_padding_block,
                    sf_padding_block=wc.sf_padding_block,
                    fc1_ready_counter_ptr=l1_arrival_count.iterator if mega else None,
                    fc1_done_counter_ptr=fc1_done_counter.iterator,
                    fc2_ready_threshold=fc2_threshold,
                    load_balance_mode=load_balance_mode,
                    claim_order=claim_order,
                    load_balance_counter_ptr=(
                        load_balance_counter.iterator
                        if load_balance_mode != "static"
                        else None
                    ),
                )
                # wait accounting (debug knob, mega only).
                params.wait_stats_offset = stats_word_offset
                w = weight_layouts.make_logical_views(fc1_weight, fc2_weight)
                ws = sf_layouts.make_logical_views(fc1_weight_scales, fc2_weight_scales)
                x1 = cute.make_tensor(
                    cute.recast_ptr(
                        l1_token_buffer.iterator, dtype=cutlass.Float8E4M3FN
                    ),
                    cute.make_layout(
                        (pool_tokens, config.hidden, 1), stride=(config.hidden, 1, 0)
                    ),
                )
                x2 = cute.make_tensor(
                    fc1_output.iterator,
                    cute.make_layout(
                        (pool_tokens, config.intermediate, 1),
                        stride=(config.intermediate, 1, 0),
                    ),
                )
                xs1 = cute.make_tensor(
                    cute.recast_ptr(l1_sf_buffer.iterator, dtype=cutlass.Float8E8M0FNU),
                    blockscaled_utils.tile_atom_to_shape_SF(
                        (input_sf_capacity, config.hidden, 1), 32
                    ),
                )
                xs2 = cute.make_tensor(
                    fc1_output_sf.iterator,
                    blockscaled_utils.tile_atom_to_shape_SF(
                        (output_sf_capacity, config.intermediate, 1),
                        32,
                    ),
                )
                phase1 = staging.make_tma_atoms(
                    w.fc1_weight_a_logical,
                    x1,
                    ws.fc1_weight_sfa_logical,
                    xs1,
                )
                phase2 = staging.make_tma_atoms(
                    w.fc2_weight_a_logical,
                    x2,
                    ws.fc2_weight_sfa_logical,
                    xs2,
                )
                kernel(
                    phase1[0],
                    phase1[1],
                    phase1[2],
                    phase1[3],
                    phase1[4],
                    phase1[5],
                    phase1[6],
                    phase1[7],
                    phase2[0],
                    phase2[1],
                    phase2[2],
                    phase2[3],
                    phase2[4],
                    phase2[5],
                    phase2[6],
                    phase2[7],
                    args,
                    params,
                    fc1_output,
                    fc1_output_sf,
                    fc1_done_counter,
                    route_terms,
                    output if in_kernel_reduce else None,
                ).launch(
                    grid=(cm, cn, clusters),
                    block=(warps.threads_per_cta, 1, 1),
                    cluster=(cm, cn, 1),
                    stream=stream,
                    min_blocks_per_mp=1,
                )

            @cute.jit
            def make_dispatch_args(
                self,
                input_token_buffer,
                input_sf_buffer,
                input_topk_idx,
                input_topk_weights,
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
                route_terms,
            ):
                dispatch = Mxfp8DispatchPhase(
                    wc,
                    cluster_shape_mn=(cm, cn),
                    rank=local_rank,
                    dispatch_warp_start=warps.dispatch[0],
                    num_other_warps=len(warps.epilogue) + 4,
                    flag_batch=flag_batch,
                )
                mapper = peer_rank_ptr_mapper_host.make_device_obj()
                args = dispatch.make_args(
                    input_token_buffer=input_token_buffer,
                    input_sf_buffer=input_sf_buffer,
                    topk_idx=input_topk_idx,
                    input_topk_weights_buffer=input_topk_weights,
                    expert_send_count=expert_send_count,
                    expert_recv_count=expert_recv_count,
                    expert_recv_count_sum=expert_recv_count_sum,
                    src_token_topk_idx=src_token_topk_idx,
                    fc1_input_token_buffer=l1_token_buffer,
                    fc1_input_sf_buffer=l1_sf_buffer,
                    fc1_input_topk_weights_buffer=l1_topk_weights_buffer,
                    fc1_ready_counter=l1_arrival_count,
                    token_src_metadata=token_src_metadata,
                    # The helper offsets combine_output in bytes.
                    combine_output=cute.recast_tensor(route_terms, cutlass.Uint8),
                    nvlink_barrier_signal=nvlink_barrier_signal,
                    nvlink_barrier_counter=nvlink_barrier_counter,
                    grid_sync_counter=grid_sync_counter,
                    local_zero_prefix=local_zero_prefix,
                    shared_zero_prefix=shared_zero_prefix,
                    peer_rank_ptr_mapper=mapper,
                    sm_count=active_sms,
                )
                return args

        return Launcher()


class PersistentMxfp8Mxfp4KernelAdapter(_MegaAdapterBindings, PersistentFc12KernelBase):
    """Mega specialization of the shared FC12 kernel, with NVLink dispatch."""

    mega = True

    def compile(self, config: "FrontendConfig", spec: "KernelLaunchSpec") -> Any:
        self._validate_static_contract(config, spec)
        missing = _missing_protocols()
        if missing:
            raise KernelAssemblyPendingError(
                "native persistent MXFP8 x MXFP4 assembly is waiting for: "
                + ", ".join(missing)
            )
        import cutlass.cute as cute
        import cutlass.utils as cutlass_utils
        from .fc2_epilogue import DeterministicTopKReducer

        num_sms = torch.cuda.get_device_properties(
            spec.activation.device
        ).multi_processor_count
        cm, cn, _ = self.tactic["cluster_shape_mnk"]
        max_active_clusters = cutlass_utils.HardwareInfo(
            device_id=spec.activation.device.index
        ).get_max_active_clusters(cm * cn)
        runtime = self._runtime(spec, num_sms)
        persistent = cute.compile(
            self._make_launcher(config, max_active_clusters), **runtime
        )
        reducer_runtime = dict(
            route_terms=_to_cute(spec.workspace_regions["route_terms"], 16),
            route_scores=_to_cute(spec.route_scores, 4),
            route_ids=_to_cute(spec.route_ids, 8),
            output=_to_cute(spec.output, 16),
            stream=runtime["stream"],
        )
        reducer = None
        if not self.tactic.get("in_kernel_reduce", False):
            reducer = cute.compile(
                DeterministicTopKReducer(hidden=config.hidden, num_topk=config.topk),
                **reducer_runtime,
            )
        return _CompiledAssembly(persistent, reducer, num_sms)

    def bind(self, compiled, config, spec) -> Callable[[], None]:
        self._validate_static_contract(config, spec)
        if not isinstance(compiled, _CompiledAssembly):
            raise TypeError("compiled object was not produced by this adapter")
        from .device_workspace import DeviceWorkspacePartition

        runtime = self._runtime(spec, compiled.num_sms)
        reducer_runtime = dict(
            route_terms=_to_cute(spec.workspace_regions["route_terms"], 16),
            route_scores=_to_cute(spec.route_scores, 4),
            route_ids=_to_cute(spec.route_ids, 8),
            output=_to_cute(spec.output, 16),
            stream=runtime["stream"],
        )
        partition = DeviceWorkspacePartition.from_launch_regions(
            self.workspace.plan, spec.workspace_regions
        )
        launch_stream = torch.cuda.ExternalStream(
            spec.stream, device=spec.activation.device
        )

        return _BoundMegaLaunch(
            compiled,
            runtime,
            reducer_runtime,
            partition,
            spec.workspace_regions["route_terms"],
            self.workspace.world_size,
            launch_stream,
        )

    @staticmethod
    def rebind_stream(launch: Callable[[], None], stream: int) -> Callable[[], None]:
        if not isinstance(launch, _BoundMegaLaunch):
            raise TypeError("stream rebinding requires a warmed mega launch")
        return launch.rebind_stream(stream)


__all__ = [
    "ASSEMBLY_READY",
    "DEVICE_ASSEMBLY_PROTOCOL",
    "DeviceAssemblyProtocol",
    "KernelAssemblyPendingError",
    "PersistentFc12KernelBase",
    "PersistentMxfp8Mxfp4KernelAdapter",
]
