"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# The Cake W4A16 MegaMoE backend (``sm103_bf16_nvfp4_bf16_cake``): the
# ``MegaKernelBackend`` over two Cake programs, the fused dispatch + FC1 +
# SwiGLU + FC2 + return program and the PDL-chained combine program.
#
# One forward is one launch per program of the route on the current torch
# stream (the fused program, then the combine program unless the combine is
# fused into the tail); the host never synchronizes, never memsets and never
# allocates in the forward path.
# Every counter the programs use is device-resident and zeroed once at
# allocation: the fused program resets its intra-rank barrier words at its
# tail and increments the launch epoch, the completion counters are
# monotonic, so a captured graph replays without host state. The caller's
# tensors are bound directly when they satisfy the programs' pointer contract
# (contiguous bf16 rows, int32 ids, float32 weights, exactly the token bound);
# otherwise they are staged into workspace copies.

from __future__ import annotations

import contextlib
import dataclasses
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import torch

from flashinfer.moe_ep.backends.mega.kernel.sm100.common.bf16_staging import (
    validate_bf16_forward_inputs,
)
from flashinfer.moe_ep.config import BootstrapConfig, FinalizeMode, FleetParams
from flashinfer.moe_ep.core.kernel.base import MegaKernelBackend
from flashinfer.moe_ep.core.kernel.registry import register_mega_kernel
from flashinfer.moe_ep.core.kernel.workspace_pool import (
    epilogue_pool_key,
    knobs_pool_key,
)
from flashinfer.moe_ep.core.runtime import NVSHMEM, TORCH_DIST
from flashinfer.moe_ep.core.validation.common import (
    MoEEpArchError,
    MoEEpConfigError,
    validate_mega_fleet_params,
)
from flashinfer.moe_ep.weights import MoEWeightPack

from .cake_config import Sm103_Bf16_Nvfp4_Bf16_Cake_MegaMoeConfig
from .cake_plan import (
    COMBINE,
    COMBINE_PER_FORWARD_ARGS,
    FUSED,
    FUSED_COMBINE,
    FUSED_PER_FORWARD_ARGS,
    GRID_ARGS,
    HIDDEN_ALIGN,
    INTERMEDIATE_ALIGN,
    KERNEL_NAME,
    PROCESS_GROUP,
    SELECTABLE_CELLS,
    Cell,
    Geometry,
    cell_from_knobs,
    combine_grid,
    fused_grid,
    local_buffers,
    select_cell,
    symmetric_buffers,
    validate_geometry,
)
from .cake_programs import (
    TARGETS,
    ProgramHandle,
    admitted_capabilities,
    combine_handle,
    fused_handle,
)
from .cake_symmetric import (
    SymmetricBuffer,
    allocate_flag_pad,
    allocate_symmetric,
    check_self_entry,
)
from .cake_weights import (
    TransformedCakeW4a16MegaWeights,
    preprocess_mega_weights,
    validate_transformed_mega_weights,
)

if TYPE_CHECKING:
    from flashinfer.moe_ep.tensors import MoEEpTensors

MASKED_ROUTE = -1
POINTER_ALIGNMENT = 16


# ---------------------------------------------------------------------------
# Workspace.
# ---------------------------------------------------------------------------
@dataclass
class StagedInputs:
    """The tensors of one forward as the programs read them."""

    hidden: torch.Tensor  # bf16 [bound, H]
    ids: torch.Tensor  # int32 [bound, 8]
    weights: torch.Tensor  # float32 [bound, 8]
    tokens: int  # this rank's live tokens
    bound: int  # the token count the programs see (>= tokens)


@dataclass
class LaunchPlan:
    """The positional calls of one (workspace, weights, token bound, cell);
    ``combine`` is ``None`` when the combine is fused into the tail."""

    cell: Cell
    bound: int
    fused: ProgramHandle
    fused_values: List[Any]
    fused_dynamic: Dict[str, int]
    combine: Optional[ProgramHandle]
    combine_values: List[Any]
    combine_dynamic: Dict[str, int]
    weights: Any  # the TransformedCakeW4a16MegaWeights the plan binds


@dataclass
class CakeWorkspace:
    """Per-EP-group workspace of the route (pooled across layers)."""

    device: torch.device
    group: Any
    rank: int
    world_size: int
    geometry: Geometry
    swiglu_limit: float
    use_torch_stream: Callable[[], Any]
    symmetric: Dict[str, SymmetricBuffer]
    pg_flags: SymmetricBuffer
    local: Dict[str, torch.Tensor]
    compile_tokens_per_rank: Optional[int] = None
    staged: Optional[StagedInputs] = None
    launches: int = 0
    plans: Dict[Tuple[int, int, str], LaunchPlan] = field(default_factory=dict)
    fused_grids: Dict[str, int] = field(default_factory=dict)  # by cell suffix
    destroyed: bool = False

    @property
    def capacity(self) -> int:
        return self.geometry.capacity

    @property
    def output(self) -> torch.Tensor:
        return self.local["output"]

    def destroy(self) -> None:
        """Release the buffers (collective when the symmetric heap is shared)."""

        if self.destroyed:
            return
        self.destroyed = True
        torch.cuda.synchronize(self.device)
        if self.world_size > 1 and self.group is not None:
            import torch.distributed as dist

            dist.barrier(group=self.group)
        self.plans.clear()
        self.fused_grids.clear()
        self.staged = None
        for buffer in (*self.symmetric.values(), self.pg_flags):
            buffer.free()
        self.symmetric = {}
        self.local = {}


def resolve_bound(
    num_tokens: int, compile_tokens_per_rank: Optional[int], capacity: int
) -> int:
    """The token count the programs see: the collective graph bucket when one
    is set, else this rank's live token count."""

    if compile_tokens_per_rank is None:
        bound = num_tokens
    else:
        if isinstance(compile_tokens_per_rank, bool) or not isinstance(
            compile_tokens_per_rank, int
        ):
            raise MoEEpConfigError(
                f"{KERNEL_NAME}: compile_tokens_per_rank must be an int or None"
            )
        bound = compile_tokens_per_rank
    if not 0 <= bound <= capacity:
        raise MoEEpConfigError(
            f"{KERNEL_NAME}: token bound {bound} must be in [0, {capacity}] "
            "(the workspace capacity)"
        )
    if num_tokens > bound:
        raise MoEEpConfigError(
            f"{KERNEL_NAME}: token bound {bound} is below this rank's {num_tokens} tokens"
        )
    return bound


def _device_scope(device: torch.device) -> Any:
    return (
        torch.cuda.device(device) if device.type == "cuda" else contextlib.nullcontext()
    )


def _direct(tensor: torch.Tensor, dtype: torch.dtype, rows: int) -> bool:
    """Whether the programs can read ``tensor`` in place."""

    return (
        tensor.dtype == dtype
        and tensor.is_contiguous()
        and int(tensor.shape[0]) == rows
        and tensor.data_ptr() % POINTER_ALIGNMENT == 0
    )


def _check_pointer_contract(t: MoEEpTensors) -> None:
    """The programs read the caller's activations and routing in place:
    contiguous rows at a 16-byte aligned base. Strided or offset views are
    rejected rather than staged (the staging copies exist for the graph
    bucket and for int64 routing ids only)."""

    for name, tensor in (
        ("hidden_states", t.hidden_states),
        ("topk_ids", t.topk_ids),
        ("topk_weights", t.topk_weights),
    ):
        if not tensor.is_contiguous() or tensor.data_ptr() % POINTER_ALIGNMENT:
            raise MoEEpConfigError(
                f"{KERNEL_NAME}: {name} must be contiguous and {POINTER_ALIGNMENT}-byte "
                "aligned (strided or offset views are not staged)"
            )


def _adopt_config(config: Any) -> Sm103_Bf16_Nvfp4_Bf16_Cake_MegaMoeConfig:
    """The backend's config: an instance is used as is; any other object whose
    ``kernel_name`` is this kernel is rebuilt field by field through the same
    fail-closed validation (the config class is duck-typed)."""

    if isinstance(config, Sm103_Bf16_Nvfp4_Bf16_Cake_MegaMoeConfig):
        return config
    kernel_name = getattr(config, "kernel_name", None)
    if kernel_name != KERNEL_NAME:
        raise MoEEpConfigError(
            f"{KERNEL_NAME} takes a config with kernel_name={KERNEL_NAME!r}; got "
            f"{type(config).__name__} with kernel_name={kernel_name!r}"
        )
    fields = {
        f.name: getattr(config, f.name)
        for f in dataclasses.fields(Sm103_Bf16_Nvfp4_Bf16_Cake_MegaMoeConfig)
        if f.init and hasattr(config, f.name)
    }
    try:
        return Sm103_Bf16_Nvfp4_Bf16_Cake_MegaMoeConfig(**fields)
    except (TypeError, ValueError) as error:
        raise MoEEpConfigError(f"{KERNEL_NAME}: {error}") from error


def _config_proposal(config: Sm103_Bf16_Nvfp4_Bf16_Cake_MegaMoeConfig) -> tuple:
    """A comparable image of the config (tensors by shape and dtype)."""

    image = []
    for f in dataclasses.fields(config):
        value = getattr(config, f.name)
        if isinstance(value, torch.Tensor):
            value = ("tensor", tuple(value.shape), str(value.dtype))
        elif isinstance(value, dict):
            value = tuple(sorted(value.items()))
        elif isinstance(value, FinalizeMode):
            value = value.name
        image.append((f.name, value))
    return tuple(image)


# ---------------------------------------------------------------------------
# Backend.
# ---------------------------------------------------------------------------
@register_mega_kernel(KERNEL_NAME)
class Sm103Bf16Nvfp4Bf16CakeMegaKernelBackend(MegaKernelBackend):
    """BF16 x NVFP4 -> BF16 MegaMoE over the fused Cake program and its combine."""

    supports_output_view = True
    supports_live_token_tuning = False
    supports_auto_workspaces = False

    @classmethod
    def kernel_name(cls) -> str:
        return KERNEL_NAME

    _autotune_winner: dict[str, Any] | None

    def __init__(self, config: Any) -> None:
        config = _adopt_config(config)
        super().__init__(config)
        self._kernel_config = config
        self._pinned_cell = cell_from_knobs(config.knobs)
        self._plan_keys: Dict[int, set] = {}

    # ------------------------------------------------------------------ init

    def runtime_requirements(self, bootstrap: BootstrapConfig) -> frozenset[str]:
        """``torch.distributed`` and NVSHMEM at every world size: a world of
        one initializes the same single transport (its route buffers are
        plain device tensors)."""

        del bootstrap
        return frozenset({TORCH_DIST, NVSHMEM})

    def validate_init(
        self, bootstrap: BootstrapConfig, fleet_params: FleetParams
    ) -> None:
        if torch.cuda.is_available():
            capability = tuple(torch.cuda.get_device_capability())
            if capability not in TARGETS:
                raise MoEEpArchError(
                    f"{KERNEL_NAME} requires an exact sm_100a or sm_103a device; the "
                    f"host has sm_{capability[0]}{capability[1]}"
                )
            if capability != (10, 3) and capability not in admitted_capabilities():
                raise MoEEpArchError(
                    f"{KERNEL_NAME}: no {TARGETS[capability]} programs ship yet for "
                    f"sm_{capability[0]}{capability[1]}"
                )
        config = self._kernel_config
        validate_mega_fleet_params(
            fleet_params,
            bootstrap.world_size,
            intermediate_size=config.intermediate_size,
            top_k=config.top_k,
            alignment=INTERMEDIATE_ALIGN,
            hidden_alignment=HIDDEN_ALIGN,
        )
        try:
            validate_geometry(
                world_size=bootstrap.world_size,
                num_experts=fleet_params.num_experts,
                hidden=fleet_params.token_hidden_size,
                intermediate=config.intermediate_size,
                top_k=config.top_k,
                capacity=fleet_params.max_tokens_per_rank,
            )
        except ValueError as error:
            raise MoEEpConfigError(str(error)) from error
        if bootstrap.stream != 0:
            raise MoEEpConfigError(
                f"{KERNEL_NAME} launches on the current torch stream only; "
                f"BootstrapConfig.stream must be 0, got {bootstrap.stream}"
            )
        local_experts = fleet_params.num_experts // bootstrap.world_size
        for name in ("fc1_alpha", "fc2_alpha"):
            alpha = getattr(config, name)
            if alpha is not None and (
                not isinstance(alpha, torch.Tensor)
                or alpha.dtype != torch.float32
                or tuple(alpha.shape) != (local_experts,)
            ):
                raise MoEEpConfigError(
                    f"{KERNEL_NAME}: config.{name} must be an FP32 tensor of shape "
                    f"[{local_experts}] (this rank's experts)"
                )

    # --------------------------------------------------------------- weights

    def preprocess_weights(
        self, weights: MoEWeightPack, fleet_params: FleetParams
    ) -> TransformedCakeW4a16MegaWeights:
        try:
            return preprocess_mega_weights(
                weights,
                intermediate_size=self._kernel_config.intermediate_size,
                hidden_size=fleet_params.token_hidden_size,
            )
        except ValueError as error:
            raise MoEEpConfigError(str(error)) from error

    def validate_transformed_weights(
        self,
        transformed_weights: Any,
        bootstrap: BootstrapConfig,
        fleet_params: FleetParams,
    ) -> None:
        try:
            validate_transformed_mega_weights(
                transformed_weights,
                intermediate_size=self._kernel_config.intermediate_size,
                hidden_size=fleet_params.token_hidden_size,
                world_size=bootstrap.world_size,
                num_experts=fleet_params.num_experts,
            )
        except ValueError as error:
            raise MoEEpConfigError(str(error)) from error

    # ------------------------------------------------------------- workspace

    def _workspace_pool_key(self, fleet_params: FleetParams) -> Any:
        config = self._kernel_config
        return (
            KERNEL_NAME,
            torch.cuda.current_device(),
            self.ep_rank,
            self.ep_world_size,
            id(self._ep_comm_group),
            fleet_params.num_experts,
            fleet_params.max_tokens_per_rank,
            fleet_params.token_hidden_size,
            config.intermediate_size,
            config.top_k,
            config.resolved_gate_up_clamp,
            config.finalize_mode,
            epilogue_pool_key(config.fc1_alpha),
            epilogue_pool_key(config.fc2_alpha),
            knobs_pool_key(config.knobs),
        )

    def _check_collective_agreement(self, fleet_params: FleetParams) -> None:
        """Every EP rank must construct the same backend (config + geometry)."""

        if self.ep_world_size <= 1:
            return
        import torch.distributed as dist

        proposal = (
            KERNEL_NAME,
            _config_proposal(self._kernel_config),
            (
                fleet_params.num_experts,
                fleet_params.max_tokens_per_rank,
                fleet_params.token_hidden_size,
                fleet_params.dtype_bytes,
            ),
        )
        gathered: List[Any] = [None] * self.ep_world_size
        dist.all_gather_object(gathered, proposal, group=self.ep_comm_group)
        if any(candidate != proposal for candidate in gathered):
            raise MoEEpConfigError(
                f"all {self.ep_world_size} EP ranks must construct the same "
                f"{KERNEL_NAME} backend (kernel config and fleet geometry differ "
                "across ranks)"
            )

    def _allocate_workspace(self, fleet_params: FleetParams) -> CakeWorkspace:
        """Collective: symmetric buffers, flag pad, local buffers.

        ``allocate_symmetric`` zeroes every symmetric buffer (the completion
        counters and the published headers among them), the local counters
        are allocated zeroed, and the trailing device synchronize plus
        process-group barrier is the rendezvous the route requires: the
        one-time zeroing is complete and visible on every rank before any
        rank's first launch. The host never writes a counter afterwards.
        """

        import tvm_ffi

        if torch.cuda.is_current_stream_capturing():
            raise MoEEpConfigError(
                f"{KERNEL_NAME}: workspace allocation cannot run during CUDA graph capture"
            )
        config = self._kernel_config
        world, rank = self.ep_world_size, self.ep_rank
        group = self.ep_comm_group if world > 1 else self._ep_comm_group
        try:
            validate_geometry(
                world_size=world,
                num_experts=fleet_params.num_experts,
                hidden=fleet_params.token_hidden_size,
                intermediate=config.intermediate_size,
                top_k=config.top_k,
                capacity=fleet_params.max_tokens_per_rank,
            )
        except ValueError as error:
            raise MoEEpConfigError(str(error)) from error
        self._check_collective_agreement(fleet_params)
        device = torch.device("cuda", torch.cuda.current_device())
        geometry = Geometry(
            world_size=world,
            rank=rank,
            num_experts=fleet_params.num_experts,
            hidden=fleet_params.token_hidden_size,
            intermediate=config.intermediate_size,
            top_k=config.top_k,
            capacity=fleet_params.max_tokens_per_rank,
            sm_count=int(
                torch.cuda.get_device_properties(device).multi_processor_count
            ),
        )
        # The programs load here (not at the first forward) and the fused grid
        # is agreed across the ranks before any buffer exists.
        fused_grids = self._collective_fused_grids(geometry, device, group)
        if not FUSED_COMBINE:
            combine_handle(geometry.hidden, device)
        symmetric = {
            name: allocate_symmetric(
                shape, dtype, device=device, world_size=world, rank=rank
            )
            for name, (shape, dtype) in symmetric_buffers(geometry).items()
        }
        for name, buffer in symmetric.items():
            check_self_entry(buffer, rank=rank, name=name)
        pg_flags = allocate_flag_pad(
            device=device, group=group, world_size=world, rank=rank
        )
        local = {
            name: torch.zeros(shape, dtype=dtype, device=device)
            for name, (shape, dtype) in local_buffers(geometry).items()
        }
        local["staged_ids"].fill_(MASKED_ROUTE)
        for name in ("fc1_alpha", "fc2_alpha"):
            alpha = getattr(config, name)
            if alpha is None:
                local[name].fill_(1.0)
            else:
                local[name].copy_(alpha)
        clamp = config.resolved_gate_up_clamp
        workspace = CakeWorkspace(
            device=device,
            group=group,
            rank=rank,
            world_size=world,
            geometry=geometry,
            swiglu_limit=float("inf") if clamp is None else float(clamp),
            use_torch_stream=tvm_ffi.use_torch_stream,
            symmetric=symmetric,
            pg_flags=pg_flags,
            local=local,
            fused_grids=fused_grids,
        )
        torch.cuda.synchronize(device)
        if world > 1:
            import torch.distributed as dist

            dist.barrier(group=group)
        return workspace

    # ---------------------------------------------------------------- forward

    def validate_forward(
        self, t: MoEEpTensors, fleet_params: FleetParams, *, quantize_input: bool
    ) -> None:
        del quantize_input
        if t.scales is not None:
            raise MoEEpConfigError(
                f"{KERNEL_NAME} does not accept activation quantization fields"
            )
        if t.fc1_norm_const is not None:
            raise MoEEpConfigError(f"{KERNEL_NAME} does not support fc1_norm_const")
        if t.fc1_activation_per_token_scale is not None:
            raise MoEEpConfigError(
                f"{KERNEL_NAME} does not support fc1_activation_per_token_scale"
            )
        if t.hidden_states.ndim != 2 or t.topk_ids.ndim != 2:
            raise MoEEpConfigError(f"{KERNEL_NAME}: activations and routing must be 2D")
        _check_pointer_contract(t)
        validate_bf16_forward_inputs(
            t.hidden_states,
            t.topk_ids,
            t.topk_weights,
            fleet_params,
            top_k=self._kernel_config.top_k,
            scales=t.scales,
        )
        if t.topk_ids.dtype not in (torch.int32, torch.int64):
            raise MoEEpConfigError(f"{KERNEL_NAME}: topk_ids must be int32 or int64")
        if t.topk_weights.dtype != torch.float32:
            raise MoEEpConfigError(f"{KERNEL_NAME}: topk_weights must be FP32")
        if (
            not t.hidden_states.is_cuda
            or t.topk_ids.device != t.hidden_states.device
            or t.topk_weights.device != t.hidden_states.device
        ):
            raise MoEEpConfigError(
                f"{KERNEL_NAME}: activations and routing must share a CUDA device"
            )
        local_experts = fleet_params.num_experts // self.ep_world_size
        for name in ("fc1_alpha", "fc2_alpha"):
            alpha = getattr(t, name)
            if alpha is not None and (
                alpha.dtype != torch.float32
                or tuple(alpha.shape) != (local_experts,)
                or alpha.device != t.hidden_states.device
            ):
                raise MoEEpConfigError(
                    f"{KERNEL_NAME}: {name} must be FP32 [local_experts] on the "
                    "activation device"
                )

    def validate_capture_ready(self, workspace: Any, transformed_weights: Any) -> None:
        ws = self._live(workspace)
        if ws.launches == 0:
            raise MoEEpConfigError(
                f"{KERNEL_NAME}: workspace is not warmed for CUDA graph capture; run "
                "one eager forward on all EP ranks first (layer.warmup(...))"
            )
        if not isinstance(transformed_weights, TransformedCakeW4a16MegaWeights):
            raise MoEEpConfigError(
                f"{KERNEL_NAME}: transformed weights must be "
                "TransformedCakeW4a16MegaWeights"
            )

    def set_compile_tokens_per_rank(
        self, workspace: Any, compile_tokens_per_rank: Optional[int]
    ) -> None:
        """Rank-uniform token bound of the next invocation (``None`` = the
        live token count of each rank)."""

        ws = self._live(workspace)
        if compile_tokens_per_rank is not None:
            resolve_bound(0, compile_tokens_per_rank, ws.capacity)
        ws.compile_tokens_per_rank = (
            None if compile_tokens_per_rank is None else int(compile_tokens_per_rank)
        )

    def stage_inputs(
        self, t: MoEEpTensors, workspace: Any, *, quantize_input: bool
    ) -> None:
        del quantize_input
        ws = self._live(workspace)
        tokens = int(t.num_tokens)
        bound = resolve_bound(tokens, ws.compile_tokens_per_rank, ws.capacity)
        local = ws.local
        sources = (t.hidden_states, t.topk_ids, t.topk_weights)
        targets = {
            tensor.untyped_storage().data_ptr()
            for tensor in (
                local["staged_hidden"],
                local["staged_ids"],
                local["staged_weights"],
                local["output"],
            )
        }
        if any(
            tensor.numel() != 0 and tensor.untyped_storage().data_ptr() in targets
            for tensor in sources
        ):
            raise MoEEpConfigError(
                f"{KERNEL_NAME}: inputs must not alias the staging workspace"
            )
        if t.hidden_states.device != ws.device:
            raise MoEEpConfigError(
                f"{KERNEL_NAME}: inputs are on {t.hidden_states.device}, the workspace "
                f"on {ws.device}"
            )
        _check_pointer_contract(t)
        hidden = t.hidden_states
        if not _direct(hidden, torch.bfloat16, bound):
            staged = local["staged_hidden"]
            if tokens:
                staged[:tokens].copy_(hidden)
            hidden = staged[:bound]
        ids = t.topk_ids
        if not _direct(ids, torch.int32, bound):
            staged = local["staged_ids"]
            if tokens:
                staged[:tokens].copy_(ids)
            if bound > tokens:
                staged[tokens:bound].fill_(MASKED_ROUTE)
            ids = staged[:bound]
        weights = t.topk_weights
        if not _direct(weights, torch.float32, bound):
            staged = local["staged_weights"]
            if tokens:
                staged[:tokens].copy_(weights)
            weights = staged[:bound]
        destinations, epilogue = [], []
        for source, destination in (
            (t.fc1_alpha, local["fc1_alpha"]),
            (t.fc2_alpha, local["fc2_alpha"]),
        ):
            if source is not None:
                destinations.append(destination)
                epilogue.append(source)
        if epilogue:
            torch._foreach_copy_(destinations, epilogue)
        ws.staged = StagedInputs(
            hidden=hidden, ids=ids, weights=weights, tokens=tokens, bound=bound
        )

    def workspace_output(self, workspace: Any) -> torch.Tensor:
        return self._live(workspace).output

    def compute(
        self,
        workspace: Any,
        transformed_weights: Any,
        *,
        output: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """The fused program, then (unless the combine is fused into its
        tail) the combine program with programmatic stream serialization, on
        the current stream; no host sync, no host memset, no allocation.

        ``output=None`` returns a view of the workspace output (valid under
        stream ordering until the next launch on this workspace). A caller
        output of exactly the token bound is written in place; otherwise the
        combine writes the workspace output and the live rows are copied.
        """

        ws = self._live(workspace)
        staged = ws.staged
        if staged is None:
            raise MoEEpConfigError(
                f"{KERNEL_NAME}: compute() requires stage_inputs() on this workspace first"
            )
        tokens, bound = staged.tokens, staged.bound
        plan = self._plan(ws, transformed_weights, bound, self._cell(ws))
        direct_output = (
            output is not None
            and bound == tokens
            and output.dtype == torch.bfloat16
            and output.device == ws.device
            and output.ndim == 2
            and int(output.shape[1]) == ws.geometry.hidden
            and output.is_contiguous()
            and output.data_ptr() % POINTER_ALIGNMENT == 0
        )
        target = output if direct_output else ws.output
        fused_values = list(plan.fused_values)
        for name, index in plan.fused_dynamic.items():
            fused_values[index] = self._dynamic(name, staged, target)
        combine_values = list(plan.combine_values)
        for name, index in plan.combine_dynamic.items():
            combine_values[index] = self._dynamic(name, staged, target)
        with _device_scope(ws.device), ws.use_torch_stream():
            plan.fused.launch(*fused_values)
            if plan.combine is not None:
                plan.combine.launch(*combine_values)
        ws.launches += 1
        if self._autotune_winner is None:
            self._autotune_winner = plan.cell.as_knobs()
        if output is None:
            return ws.output[:tokens]
        if not direct_output:
            output[:tokens].copy_(ws.output[:tokens])
        return output[:tokens]

    @staticmethod
    def _dynamic(name: str, staged: StagedInputs, output: torch.Tensor) -> Any:
        if name == "hidden":
            return staged.hidden
        if name == "topk_ids":
            return staged.ids
        if name == "topk_weights":
            return staged.weights
        if name == "output":
            return output
        raise AssertionError(name)

    # ------------------------------------------------------------------ plan

    def _cell(self, ws: CakeWorkspace) -> Cell:
        if self._pinned_cell is not None:
            return self._pinned_cell
        g = ws.geometry
        bound = (
            ws.compile_tokens_per_rank
            if ws.compile_tokens_per_rank is not None
            else g.capacity
        )
        return select_cell(bound, g.world_size, g.top_k, g.num_experts)

    def _candidate_cells(self) -> Tuple[Cell, ...]:
        """The cells one workspace of this backend can launch: the pinned
        cell, else every cell the token rule can select."""

        if self._pinned_cell is not None:
            return (self._pinned_cell,)
        return SELECTABLE_CELLS

    def _local_fused_grids(
        self, geometry: Geometry, device: torch.device, cells: Tuple[Cell, ...]
    ) -> Dict[str, int]:
        """CTAs of the fused program per cell on this rank: ``SM count //
        cta_group`` clusters, capped by the program's residency probe when
        the binding exports one, times the cluster size."""

        grids: Dict[str, int] = {}
        index = int(device.index or 0)
        for cell in cells:
            handle = fused_handle(cell, geometry.hidden, device)
            cluster_ctas = int(math.prod(handle.cluster))
            probe = (
                int(handle.max_active_clusters(index))
                if handle.max_active_clusters is not None
                else None
            )
            grids[cell.suffix] = fused_grid(geometry.sm_count, cluster_ctas, probe)[0]
        return grids

    def _collective_fused_grids(
        self, geometry: Geometry, device: torch.device, group: Any
    ) -> Dict[str, int]:
        """The rank-uniform fused grids: the minimum over the EP ranks of the
        local rule, so every rank launches the same persistent grid."""

        grids = self._local_fused_grids(geometry, device, self._candidate_cells())
        if geometry.world_size > 1:
            import torch.distributed as dist

            gathered: List[Any] = [None] * geometry.world_size
            dist.all_gather_object(gathered, grids, group=group)
            grids = {
                suffix: min(int(candidate[suffix]) for candidate in gathered)
                for suffix in grids
            }
        for suffix, grid in grids.items():
            if grid > geometry.max_grid_ctas:
                raise MoEEpConfigError(
                    f"{KERNEL_NAME}: FUSED {suffix} would launch {grid} CTAs; the "
                    f"workspace scratch holds {geometry.max_grid_ctas}"
                )
        return grids

    def _plan(self, ws: CakeWorkspace, tw: Any, bound: int, cell: Cell) -> LaunchPlan:
        key = (id(tw), bound, cell.suffix)
        cached = ws.plans.get(key)
        if cached is not None and cached.weights is tw:
            return cached
        # Building a plan is host-only work (views of existing buffers and
        # scalars), so a new token bound under CUDA graph capture is fine.
        if not isinstance(tw, TransformedCakeW4a16MegaWeights):
            raise MoEEpConfigError(
                f"{KERNEL_NAME}: transformed weights must be "
                f"TransformedCakeW4a16MegaWeights, got {type(tw).__name__}"
            )
        g = ws.geometry
        try:
            validate_transformed_mega_weights(
                tw,
                intermediate_size=g.intermediate,
                hidden_size=g.hidden,
                world_size=g.world_size,
                num_experts=g.num_experts,
                device=ws.device,
            )
        except ValueError as error:
            raise MoEEpConfigError(str(error)) from error
        fused = fused_handle(cell, g.hidden, ws.device)
        combine = None if FUSED_COMBINE else combine_handle(g.hidden, ws.device)
        values = self._values(ws, tw, bound, cell)
        grid = ws.fused_grids.get(cell.suffix)
        if grid is None:
            raise MoEEpConfigError(
                f"{KERNEL_NAME}: the workspace was allocated without cell "
                f"{cell.suffix}; knobs that pin a cell must be set before the "
                "workspace is created"
            )
        values[GRID_ARGS[0]] = grid
        fused_values, fused_dynamic = self._positional(
            fused, values, FUSED_PER_FORWARD_ARGS
        )
        # The combine program: its ``hidden`` is the i32 hidden size and its grid
        # covers the capacity (CTAs at or past the live count exit), so the
        # launch is identical for every token bound.
        combine_values: List[Any] = []
        combine_dynamic: Dict[str, int] = {}
        if combine is not None:
            combine_table = dict(values)
            combine_table["hidden"] = g.hidden
            combine_table.update(
                zip(GRID_ARGS, combine_grid(g.capacity, g.hidden), strict=True)
            )
            combine_values, combine_dynamic = self._positional(
                combine, combine_table, COMBINE_PER_FORWARD_ARGS
            )
        plan = LaunchPlan(
            cell=cell,
            bound=bound,
            fused=fused,
            fused_values=fused_values,
            fused_dynamic=fused_dynamic,
            combine=combine,
            combine_values=combine_values,
            combine_dynamic=combine_dynamic,
            weights=tw,
        )
        ws.plans[key] = plan
        self._plan_keys.setdefault(id(ws), set()).add(key)
        return plan

    @staticmethod
    def _positional(
        handle: ProgramHandle, values: Dict[str, Any], per_forward: frozenset
    ) -> Tuple[List[Any], Dict[str, int]]:
        """The positional call of ``handle`` from the value table, and the
        indices of the per-forward arguments (``None`` placeholders)."""

        positional: List[Any] = []
        dynamic: Dict[str, int] = {}
        for index, name in enumerate(handle.args):
            if name in per_forward:
                dynamic[name] = index
                positional.append(None)
                continue
            if name not in values:
                raise MoEEpConfigError(
                    f"{KERNEL_NAME}: {handle.identifier} takes {name!r}, which this host "
                    "does not supply"
                )
            positional.append(values[name])
        return positional, dynamic

    @staticmethod
    def _values(
        ws: CakeWorkspace, tw: TransformedCakeW4a16MegaWeights, bound: int, cell: Cell
    ) -> Dict[str, Any]:
        """Every argument this host supplies, by name (the per-forward ones
        excluded), for one (workspace, weights, token bound, cell)."""

        g = ws.geometry
        rows, tiles = g.pool_rows(cell), g.max_tiles(cell)
        local = ws.local
        values: Dict[str, Any] = {
            "W13": tw.w13,
            "S13": tw.w13_scale.view(torch.uint8),
            "W2": tw.w2,
            "S2": tw.w2_scale.view(torch.uint8),
            "pool": local["pool"][:rows],
            "pool_ptr": local["pool"][:rows],
            "fc1_out": local["fc1_out"][:rows],
            "fc1_out_st": local["fc1_out"][:rows],
            "fc1_out_ld": local["fc1_out"][:rows],
            "route_src": local["route_src"][:rows],
            "tile_expert": local["tile_expert"][:tiles],
            "tile_mn_limit": local["tile_mn_limit"][:tiles],
            "tile_count": local["tile_count"],
            "expert_seg": local["expert_seg"],
            "cta_hist": local["cta_hist"],
            "sync_words": local["sync_words"],
            "epoch": local["epoch"],
            "fc1_ready": local["fc1_ready"][:tiles],
            "fc1_done": local["fc1_done"][:tiles],
            "fc2_done": local["fc2_done"][:tiles],
            "fc1_alpha": local["fc1_alpha"],
            "fc2_alpha": local["fc2_alpha"],
            f"{PROCESS_GROUP}_world": g.world_size,
            f"{PROCESS_GROUP}_rank": g.rank,
            f"{PROCESS_GROUP}_flags": ws.pg_flags.peers,
            "swiglu_limit": ws.swiglu_limit,
            "T_r": bound,
            "cap": g.capacity,
            "H": g.hidden,
            "I": g.intermediate,
            "E_local": g.local_experts,
            "fc1_m_tiles": g.fc1_m_tiles,
            "fc1_m_div": g.fc1_m_tiles,
            "fc2_m_tiles": g.fc2_m_tiles,
            "fc2_m_div": g.fc2_m_tiles,
            GRID_ARGS[1]: 1,
            GRID_ARGS[2]: 1,
        }
        for name, buffer in ws.symmetric.items():
            values[name] = buffer.local
            values[f"{name}_peers"] = buffer.peers
        # The combine program reads the live token count from the device word the
        # fused program stored (``published_meta[0]``).
        values["tokens_per_rank"] = ws.symmetric["published_meta"].local
        return values

    # --------------------------------------------------------------- release

    def _forget_local_workspace_state(self, workspace: Any) -> None:
        """Drop the launch plans (and their weight references) this instance
        cached on ``workspace``; other layers' entries in the pooled workspace
        stay."""

        keys = self._plan_keys.pop(id(workspace), set())
        if isinstance(workspace, CakeWorkspace):
            for key in keys:
                workspace.plans.pop(key, None)

    @staticmethod
    def _live(workspace: Any) -> CakeWorkspace:
        if not isinstance(workspace, CakeWorkspace):
            raise TypeError(
                f"{KERNEL_NAME} workspace must be created by this backend, "
                f"got {type(workspace).__name__}"
            )
        if workspace.destroyed:
            raise RuntimeError(f"{KERNEL_NAME} workspace has been destroyed")
        return workspace


__all__ = [
    "COMBINE",
    "FUSED",
    "KERNEL_NAME",
    "CakeWorkspace",
    "LaunchPlan",
    "Sm103Bf16Nvfp4Bf16CakeMegaKernelBackend",
    "StagedInputs",
    "resolve_bound",
]
