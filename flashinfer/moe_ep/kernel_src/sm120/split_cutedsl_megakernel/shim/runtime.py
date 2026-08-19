"""Framework runtime for the SM120 MXFP4 x MXFP8 split MegaMoE kernels."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

import torch

from .comm import (
    compute_peer_offsets,
    ensure_not_capturing,
    free_sym_tensor,
    sym_byte_view,
    sym_zeros,
)
from .staging import ACTIVATION_DTYPE
from .weights import SCALE_DTYPE, TransformedWeights, ceil_div, round_up


@dataclass(frozen=True)
class MegaMoESm120W4A8Config:
    rank: int
    world_size: int
    max_tokens_per_rank: int
    num_topk: int
    num_total_experts: int
    hidden: int
    intermediate: int
    gate_up_clamp: float | None = None
    data_parallel_size: int = 1
    tensor_parallel_size: int = 1
    knobs: dict[str, Any] | None = None

    @property
    def local_experts(self) -> int:
        return self.num_total_experts // self.world_size

    def validate(self) -> None:
        if self.world_size <= 0 or not 0 <= self.rank < self.world_size:
            raise ValueError("invalid EP rank/world size")
        if self.num_total_experts % self.world_size:
            raise ValueError("num_total_experts must be divisible by EP world size")
        if self.hidden % 32 or self.intermediate % 32:
            raise ValueError("hidden and post-SwiGLU intermediate must be multiples of 32")
        if self.num_topk <= 0 or self.num_topk > self.num_total_experts:
            raise ValueError("invalid top-k")
        if self.max_tokens_per_rank <= 0:
            raise ValueError("max_tokens_per_rank must be positive")


@dataclass
class MegaMoESm120W4A8Inputs:
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
class _ExecutionBuffers:
    bundle: Any
    spec: Any
    local_workspace: torch.Tensor
    shared_workspace: torch.Tensor
    combine_output: torch.Tensor
    epilogue_args: tuple[torch.Tensor, torch.Tensor, torch.Tensor]


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
class MegaMoESm120W4A8Workspace:
    config: MegaMoESm120W4A8Config
    x: torch.Tensor
    x_scale: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    output: torch.Tensor
    _sym_roots: list[torch.Tensor] = field(default_factory=list)
    _control_group: Any = None
    _execution: _ExecutionBuffers | None = None
    _frontends: dict[tuple[int, ...], "MegaMoESm120W4A8Frontend"] = field(
        default_factory=dict
    )
    _staged_tokens: int = 0
    _destroyed: bool = False

    def destroy(self) -> None:
        if self._destroyed:
            return
        for frontend in self._frontends.values():
            frontend.release()
        self._frontends.clear()
        self._execution = None
        for root in reversed(self._sym_roots):
            free_sym_tensor(root)
        self._sym_roots.clear()
        self._destroyed = True


class MegaMoESm120W4A8Frontend:
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
    )

    # ``nvlink_barrier_counter`` is deliberately not reset by graph replay.
    # It carries the phase/sign paired with the persistent two-slot
    # ``nvlink_barrier_signal`` allocation. Resetting only the counter can make
    # a later replay observe a stale target value and leave the rank barrier
    # before every peer has entered the current invocation.

    def __init__(
        self,
        workspace: MegaMoESm120W4A8Workspace,
        *,
        control_group: Any = None,
    ) -> None:
        self.workspace = workspace
        self.config = workspace.config
        self.control_group = control_group
        self._compiled: _CompiledSplit | None = None

    @staticmethod
    def _to_cute(
        tensor: torch.Tensor,
        assumed_align: int = 16,
        *,
        static_layout: bool = False,
    ):
        import cutlass.torch as cutlass_torch

        result = cutlass_torch.from_dlpack(tensor, assumed_align=assumed_align)
        if static_layout:
            return result
        return result.mark_layout_dynamic(
            leading_dim=cutlass_torch.get_leading_dim(tensor)
        )

    def _barrier(self) -> None:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.barrier(group=self.control_group)

    def _heuristic_overrides(self):
        from moe_sm120_mxfp8_split.heuristic import MegaMoEHeuristicOverrides

        fields = {field.name for field in dataclasses.fields(MegaMoEHeuristicOverrides)}
        values = dict(self.config.knobs or {})
        unknown = sorted(set(values) - fields)
        if unknown:
            raise ValueError(f"unknown SM120 W4A8 tuning knobs: {unknown}")
        if values.get("comm_backend", "p2p_direct") != "p2p_direct":
            raise NotImplementedError(
                "the initial FlashInfer W4A8 backend supports p2p_direct only"
            )
        values["comm_backend"] = "p2p_direct"
        return MegaMoEHeuristicOverrides(**values)

    def _select_spec(self):
        from moe_sm120_mxfp8_split.api import (
            MegaMoEProblemSpec,
            SplitKernelBuildOptions,
            select_compile_spec,
        )
        from moe_sm120_mxfp8_split.runtime.green_context import (
            query_green_context_sm_counts,
            query_sm_resource_info,
        )

        cfg = self.config
        num_sms, minimum, alignment = query_sm_resource_info()
        problem = MegaMoEProblemSpec(
            tokens_per_rank=cfg.max_tokens_per_rank,
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
        spec = select_compile_spec(
            problem=problem,
            ep_same_numa_peer_count=cfg.world_size - 1,
            ep_cross_numa_peer_count=0,
            num_sms=num_sms,
            sm_min_partition=minimum,
            sm_partition_alignment=alignment,
            overrides=self._heuristic_overrides(),
        )
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

    def _runtime_tensors(
        self,
        inputs: MegaMoESm120W4A8Inputs,
        local_workspace: torch.Tensor,
        shared_workspace: torch.Tensor,
        combine_output: torch.Tensor,
        epilogue_args: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        stream,
    ) -> dict[str, Any]:
        from src.sym_buffer import SymBufferHost

        base, offsets = compute_peer_offsets(shared_workspace, self.config.world_size)
        fc1_alpha, fc2_alpha, fc1_norm_const = epilogue_args
        return dict(
            activation=self._to_cute(inputs.activation),
            activation_sf=self._to_cute(inputs.activation_scale),
            topk_idx=self._to_cute(inputs.topk_ids),
            topk_weights=self._to_cute(inputs.topk_weights),
            fc1_weight=self._to_cute(inputs.fc1_weight),
            fc1_weight_sf=self._to_cute(inputs.fc1_weight_scale),
            fc2_weight=self._to_cute(inputs.fc2_weight),
            fc2_weight_sf=self._to_cute(inputs.fc2_weight_scale),
            fc1_alpha=self._to_cute(fc1_alpha, assumed_align=4),
            fc2_alpha=self._to_cute(fc2_alpha, assumed_align=4),
            fc1_norm_const=self._to_cute(fc1_norm_const, assumed_align=4),
            combine_output=self._to_cute(combine_output),
            combine_ready_flags=None,
            fc2_block_done_counter=None,
            local_workspace=self._to_cute(local_workspace, static_layout=True),
            shared_workspace=self._to_cute(shared_workspace),
            green_trace=None,
            peer_rank_ptr_mapper_host=SymBufferHost(
                base_addr=base,
                offsets=offsets,
                rank_idx=self.config.rank,
                num_max_ranks=self.config.world_size,
            ),
            stream=stream,
        )

    def _ensure_execution(self) -> _ExecutionBuffers:
        execution = self.workspace._execution
        if execution is not None:
            return execution
        ensure_not_capturing("W4A8 shared execution-buffer allocation")

        from moe_sm120_mxfp8_split.api import build_split_kernels

        spec = self._select_spec()
        bundle = build_split_kernels(spec)
        local_workspace = torch.zeros(
            (bundle.local_workspace_bytes,), dtype=torch.uint8, device="cuda"
        )
        shared_workspace = sym_zeros(
            (bundle.shared_workspace_bytes,), torch.uint8
        )
        self.workspace._sym_roots.append(shared_workspace)
        combine_output, root = sym_byte_view(
            (
                self.config.max_tokens_per_rank,
                self.config.num_topk,
                self.config.hidden,
            ),
            torch.bfloat16,
        )
        self.workspace._sym_roots.append(root)
        epilogue_args = tuple(
            torch.ones(
                (self.config.local_experts,),
                dtype=torch.float32,
                device="cuda",
            )
            for _ in range(3)
        )
        execution = _ExecutionBuffers(
            bundle=bundle,
            spec=spec,
            local_workspace=local_workspace,
            shared_workspace=shared_workspace,
            combine_output=combine_output,
            epilogue_args=epilogue_args,
        )
        self.workspace._execution = execution
        return execution

    def _ensure_compiled(self, inputs: MegaMoESm120W4A8Inputs) -> _CompiledSplit:
        if self._compiled is not None:
            return self._compiled
        ensure_not_capturing("CuTeDSL compile and Green Context graph creation")

        import cuda.bindings.driver as cuda
        import cutlass.cute as cute
        from moe_sm120_mxfp8_split.api import compile_combine_reduce
        from moe_sm120_mxfp8_split.runtime.green_context import (
            NativeGreenContextGraph,
        )

        execution = self._ensure_execution()
        bundle = execution.bundle
        spec = execution.spec
        local_workspace = execution.local_workspace
        shared_workspace = execution.shared_workspace
        combine_output = execution.combine_output
        epilogue_args = execution.epilogue_args

        root_stream = torch.cuda.Stream(priority=0)
        k1_stream = torch.cuda.Stream(priority=-1)
        k2_stream = torch.cuda.Stream(priority=0)
        root_cuda = cuda.CUstream(root_stream.cuda_stream)
        k1_cuda = cuda.CUstream(k1_stream.cuda_stream)
        k2_cuda = cuda.CUstream(k2_stream.cuda_stream)
        base_runtime = self._runtime_tensors(
            inputs,
            local_workspace,
            shared_workspace,
            combine_output,
            epilogue_args,
            root_cuda,
        )
        runtime_k1 = dict(base_runtime, stream=k1_cuda)
        runtime_k2 = dict(base_runtime, stream=k2_cuda)
        compile_k1 = dict(runtime_k1, max_active_clusters=spec.kernel.k1_sms)
        compile_k2 = dict(runtime_k2, max_active_clusters=spec.kernel.k2_sms)
        compiled_k1 = cute.compile(bundle.k1, **compile_k1)
        compiled_k2 = cute.compile(bundle.k2, **compile_k2)

        runtime_drain = None
        runtime_finalizer = None
        compiled_drain = None
        compiled_finalizer = None
        if bundle.k2_drain is not None:
            runtime_drain = dict(base_runtime, stream=k1_cuda)
            runtime_finalizer = dict(base_runtime, stream=root_cuda)
            compiled_drain = cute.compile(
                bundle.k2_drain,
                **dict(runtime_drain, max_active_clusters=spec.kernel.k1_sms),
            )
            compiled_finalizer = cute.compile(
                bundle.k2_finalizer,
                **dict(runtime_finalizer, max_active_clusters=1),
            )

        k3_plan = compile_combine_reduce(
            combine_output,
            inputs.output,
            None,
            stream=root_cuda,
        )
        compiled_k3, combine_cute, output_cute, score_cute, k3_stream = k3_plan
        runtime_k3 = dict(
            combine_cute=combine_cute,
            reduced_cute=output_cute,
            topk_score_cute=score_cute,
            stream=k3_stream,
        )

        k1_executor = compiled_k1.to(None)
        k2_executor = compiled_k2.to(None)
        drain_executor = compiled_drain.to(None) if compiled_drain else None
        finalizer_executor = (
            compiled_finalizer.to(None) if compiled_finalizer else None
        )
        k3_executor = compiled_k3.to(None)
        graph = NativeGreenContextGraph.capture(
            root_stream=root_stream,
            k1_stream=k1_stream,
            k2_stream=k2_stream,
            k1_executor=k1_executor,
            k2_executor=k2_executor,
            k2_drain_executor=drain_executor,
            k3_executor=k3_executor,
            launch_k1=lambda: k1_executor(**runtime_k1),
            launch_k2=lambda: k2_executor(**runtime_k2),
            launch_k2_drain=(
                (lambda: drain_executor(**runtime_drain))
                if drain_executor is not None
                else None
            ),
            launch_k2_finalizer=(
                (lambda: finalizer_executor(**runtime_finalizer))
                if finalizer_executor is not None
                else None
            ),
            launch_k3=lambda: k3_executor(**runtime_k3),
            launch_reset=lambda: self._reset_execution(execution),
            k1_sm_count=spec.kernel.k1_sms,
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
        kernel = execution.bundle.k1
        for name in self._LOCAL_RESET_REGIONS:
            if name not in kernel._local_offsets:
                continue
            offset = int(kernel._local_offsets[name])
            size = int(kernel._local_region_by_name[name].nbytes)
            execution.local_workspace[offset : offset + size].zero_()
        if "expert_recv_count_sum" in kernel._shared_offsets:
            offset = int(kernel._shared_offsets["expert_recv_count_sum"])
            size = int(kernel._shared_region_by_name["expert_recv_count_sum"].nbytes)
            execution.shared_workspace[offset : offset + size].zero_()

    def run(self, inputs: MegaMoESm120W4A8Inputs) -> torch.Tensor:
        compiled = self._ensure_compiled(inputs)
        compiled.graph.launch(torch.cuda.current_stream())
        return inputs.output

    def release(self) -> None:
        compiled = self._compiled
        if compiled is None:
            return
        ensure_not_capturing("W4A8 workspace release")
        compiled.graph.close()
        self._compiled = None


def allocate_workspace(
    config: MegaMoESm120W4A8Config,
    *,
    control_group: Any = None,
) -> MegaMoESm120W4A8Workspace:
    config.validate()
    roots: list[torch.Tensor] = []
    x, root = sym_byte_view(
        (config.max_tokens_per_rank, config.hidden), ACTIVATION_DTYPE
    )
    roots.append(root)
    scale_columns = round_up(ceil_div(config.hidden, 32), 4)
    x_scale, root = sym_byte_view(
        (config.max_tokens_per_rank, scale_columns), SCALE_DTYPE
    )
    roots.append(root)
    topk_ids = sym_zeros(
        (config.max_tokens_per_rank, config.num_topk), torch.int64
    )
    topk_ids.fill_(-1)
    roots.append(topk_ids)
    topk_weights = sym_zeros(
        (config.max_tokens_per_rank, config.num_topk), torch.float32
    )
    roots.append(topk_weights)
    output, root = sym_byte_view(
        (config.max_tokens_per_rank, config.hidden), torch.bfloat16
    )
    roots.append(root)
    workspace = MegaMoESm120W4A8Workspace(
        config=config,
        x=x,
        x_scale=x_scale,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        output=output,
        _sym_roots=roots,
        _control_group=control_group,
    )
    return workspace


def run_split_mega_moe(
    workspace: MegaMoESm120W4A8Workspace,
    transformed_weights: TransformedWeights,
    *,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    if workspace._destroyed:
        raise RuntimeError("W4A8 workspace has been destroyed")
    (fc1_weight, fc1_scale), (fc2_weight, fc2_scale) = transformed_weights
    graph_output = workspace.output if output is None else output
    weight_key = tuple(
        tensor.data_ptr()
        for tensor in (fc1_weight, fc1_scale, fc2_weight, fc2_scale)
    ) + (graph_output.data_ptr(),)
    frontend = workspace._frontends.get(weight_key)
    if frontend is None:
        frontend = MegaMoESm120W4A8Frontend(
            workspace,
            control_group=workspace._control_group,
        )
        workspace._frontends[weight_key] = frontend
    inputs = MegaMoESm120W4A8Inputs(
        activation=workspace.x,
        activation_scale=workspace.x_scale,
        topk_ids=workspace.topk_ids,
        topk_weights=workspace.topk_weights,
        fc1_weight=fc1_weight,
        fc1_weight_scale=fc1_scale,
        fc2_weight=fc2_weight,
        fc2_weight_scale=fc2_scale,
        output=graph_output,
    )
    return frontend.run(inputs)


__all__ = [
    "MegaMoESm120W4A8Config",
    "MegaMoESm120W4A8Inputs",
    "MegaMoESm120W4A8Workspace",
    "allocate_workspace",
    "run_split_mega_moe",
]
