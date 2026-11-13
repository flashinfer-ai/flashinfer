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

# MOSCA MegaMoE backend for MoEEpMegaLayer (experimental, SM100, FP8 x FP4).
#
# Registered as ``kernel_name="sm100_fp8_fp4_bf16_mosca"`` when the package is
# imported (core never imports it: importing the config is the opt-in), so this
# module stays lightweight at import time: the JIT loaders, tvm-ffi and
# symmetric memory are imported inside the functions that use them.
#
# One forward on every EP rank:
#   stage_inputs  quantize this rank's rows (BF16 -> per-32 UE8M0 FP8) and copy
#                 the routing into the symmetric buffer (``staging.py``);
#   compute       plan the call for the per-rank token count (host planner),
#                 load the kernel module of that configuration, launch it on the
#                 current stream; in the replicated token layout, copy the
#                 gathered rows into the caller's output.
# The kernel synchronizes the ranks itself (NVLink barriers in symmetric
# memory): every EP rank must launch every forward, zero-token ranks included,
# with the same configuration (same per-rank token bound, same features).

from __future__ import annotations

import dataclasses
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from ....api_logging import ExperimentalWarning, warn_experimental_backend_once
from ....moe_ep.config import BootstrapConfig, FleetParams
from ....moe_ep.core.kernel.base import MegaKernelBackend
from ....moe_ep.core.kernel.registry import register_mega_kernel
from ....moe_ep.core.runtime import TORCH_DIST
from ....moe_ep.core.validation.common import (
    MoEEpArchError,
    MoEEpConfigError,
    validate_mega_arch,
    validate_mega_fleet_params,
    validate_mega_forward_inputs,
)
from ....moe_ep.weights import MoEWeightPack
from ..catalog import SM100_FP8_FP4_MEGAMOE
from .config import KERNEL_NAME, Sm100_Fp8_Fp4_Bf16_Mosca_MegaMoeConfig
from .weights import (
    TransformedMoscaMegaWeights,
    preprocess_mega_weights,
    validate_transformed_mega_weights,
)

if TYPE_CHECKING:
    from ....moe_ep.tensors import MoEEpTensors

STABLE_API_NAME = "MoEEpMegaLayer"
# The kernel packs the token count into 14 bits when it gathers (DGX_BCAST).
MAX_GATHER_TOKENS_PER_RANK = (1 << 14) - 1


_WARNED_OUTSIDE_COVERAGE: set = set()


def enabled_features(config: Sm100_Fp8_Fp4_Bf16_Mosca_MegaMoeConfig) -> tuple[str, ...]:
    """Names (as in ../catalog.py) of the MOSCA features ``config`` turns on."""
    names = []
    if config.small_t:
        names.append("small_t")
    if config.sparse_count:
        names.append("sparse_count")
    if config.gathers_output:
        names.append("in_kernel_all_gather")
    if config.tile_rule:
        names.append("tile_rule")
    return tuple(names)


def warn_if_outside_validated_coverage(
    config: Sm100_Fp8_Fp4_Bf16_Mosca_MegaMoeConfig,
    fleet_params: FleetParams,
    world_size: int,
    compute_capability: tuple[int, int],
) -> bool:
    """Warn once per configuration that no validated cell (../catalog.py) covers.

    The kernel still runs: every MOSCA switch is result-preserving by design and
    the kernel with every switch off is upstream DeepGEMM's. What is missing
    outside the cells is evidence, so the warning says so. Returns whether the
    configuration is covered.
    """
    features = enabled_features(config)
    cell = SM100_FP8_FP4_MEGAMOE.find_cell(
        num_experts=fleet_params.num_experts,
        hidden=fleet_params.token_hidden_size,
        intermediate=config.intermediate_size,
        top_k=config.top_k,
        ep_size=world_size,
        token_layout=config.token_layout,
        compute_capability=(int(compute_capability[0]), int(compute_capability[1])),
        features=features,
    )
    if cell is not None:
        return True
    key = (
        fleet_params.num_experts,
        fleet_params.token_hidden_size,
        config.intermediate_size,
        config.top_k,
        world_size,
        config.token_layout,
        tuple(compute_capability),
        features,
    )
    if key not in _WARNED_OUTSIDE_COVERAGE:
        _WARNED_OUTSIDE_COVERAGE.add(key)
        major, minor = compute_capability
        warnings.warn(
            f"{KERNEL_NAME}: {fleet_params.num_experts} experts, hidden "
            f"{fleet_params.token_hidden_size}, intermediate "
            f"{config.intermediate_size}, top-{config.top_k}, EP{world_size}, "
            f"token_layout={config.token_layout!r}, features {list(features)} on "
            f"sm_{major}{minor} is outside the validated coverage "
            "(flashinfer/experimental/mosca/README.md, catalog.py). The kernel "
            "runs, but no validated cell vouches for this configuration's "
            "outputs and speed as a whole.",
            ExperimentalWarning,
            stacklevel=3,
        )
    return False


def split_rows(num_tokens: int, world_size: int, rank: int) -> tuple[int, int]:
    """``(start, count)`` of ``rank``'s slice under ``torch.tensor_split`` (first ranks take one more)."""
    base, rem = divmod(num_tokens, world_size)
    return rank * base + min(rank, rem), base + (1 if rank < rem else 0)


@dataclass
class MoscaWorkspace:
    """One EP group's symmetric buffer and its views (pooled across layers)."""

    device: torch.device
    group: Any
    rank: int
    world_size: int
    capacity: int  # per-rank kernel token capacity (requested)
    num_experts: int
    hidden: int
    num_max_tokens_per_rank: int  # aligned capacity (a template argument)
    num_sms: int
    gather: bool
    layout: tuple[int, ...]
    buffer: torch.Tensor  # uint8 [total bytes], symmetric
    handle: Any
    peers: Any  # tvm_ffi.Shape of every rank's buffer base
    x: torch.Tensor
    x_sf: torch.Tensor
    topk_idx: torch.Tensor
    topk_weights: torch.Tensor
    gathered: torch.Tensor | None
    own_rows: torch.Tensor | None  # y of a gathering launch (this rank's rows)
    staged_tokens: int = 0
    staged_row_offset: int = -1
    staged_total_tokens: int = 0
    plans: dict[tuple, Any] = field(default_factory=dict)  # see _params_for
    destroyed: bool = False

    def destroy(self) -> None:
        if self.destroyed:
            return
        self.destroyed = True
        torch.cuda.synchronize(self.device)
        if self.world_size > 1:
            import torch.distributed as dist

            dist.barrier(group=self.group)
        self.plans.clear()
        self.x = self.x_sf = self.topk_idx = self.topk_weights = None  # type: ignore[assignment]
        self.gathered = self.own_rows = None
        self.handle = None
        self.buffer = None  # type: ignore[assignment]


def _view(
    buffer: torch.Tensor, offset: int, shape: tuple[int, ...], dtype: torch.dtype
) -> torch.Tensor:
    numel = 1
    for s in shape:
        numel *= s
    nbytes = numel * torch.empty(0, dtype=dtype).element_size()
    return buffer[offset : offset + nbytes].view(dtype).view(shape)


@register_mega_kernel(KERNEL_NAME)
class Sm100MoscaMegaKernelBackend(MegaKernelBackend):
    supports_output_view = False

    def __init__(self, config: Sm100_Fp8_Fp4_Bf16_Mosca_MegaMoeConfig) -> None:
        if not isinstance(config, Sm100_Fp8_Fp4_Bf16_Mosca_MegaMoeConfig):
            raise TypeError(
                f"{KERNEL_NAME} config must be Sm100_Fp8_Fp4_Bf16_Mosca_MegaMoeConfig, "
                f"got {type(config).__name__}"
            )
        super().__init__(config)
        self._kernel_config = config
        self._agreed_geometries: set[tuple] = set()
        warn_experimental_backend_once(STABLE_API_NAME, KERNEL_NAME)

    @classmethod
    def kernel_name(cls) -> str:
        return KERNEL_NAME

    def runtime_requirements(self, bootstrap: BootstrapConfig) -> frozenset[str]:
        del bootstrap
        return frozenset({TORCH_DIST})

    def prepare_workspace(
        self, bootstrap: BootstrapConfig, fleet_params: FleetParams
    ) -> Any:
        """Allocate (or reuse) the workspace, then, outside CUDA graph capture:
        check that every EP rank built the same backend for this geometry, and
        build every kernel module this config can launch (``precompile``)."""
        self._ensure_ep_bootstrap(bootstrap)
        capturing = (
            torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
        )
        geometry = self._geometry(fleet_params)
        if not capturing and geometry not in self._agreed_geometries:
            self._check_collective_agreement(fleet_params)
            self._agreed_geometries.add(geometry)
        workspace = super().prepare_workspace(bootstrap, fleet_params)
        if not capturing and self._kernel_config.precompile:
            try:
                self._precompile(workspace)
            except MoEEpConfigError:
                # Every rank fails together (_agree), and the layer never stores
                # a workspace whose preparation raised: release it here, or the
                # pooled symmetric buffer leaks.
                self.destroy(workspace)
                raise
        return workspace

    @staticmethod
    def _geometry(fleet_params: FleetParams) -> tuple:
        return (
            fleet_params.num_experts,
            fleet_params.max_tokens_per_rank,
            fleet_params.token_hidden_size,
            fleet_params.dtype_bytes,
        )

    def _check_collective_agreement(self, fleet_params: FleetParams) -> None:
        """Every EP rank must construct the same backend (config and geometry)."""
        if self.ep_world_size <= 1:
            return
        import torch.distributed as dist

        proposal = (
            KERNEL_NAME,
            dataclasses.asdict(self._kernel_config),
            self._geometry(fleet_params),
        )
        gathered: list[Any] = [None] * self.ep_world_size
        dist.all_gather_object(gathered, proposal, group=self.ep_comm_group)
        if any(candidate != proposal for candidate in gathered):
            raise MoEEpConfigError(
                f"every EP rank must construct the same {KERNEL_NAME} backend "
                "(kernel config or fleet geometry differs across ranks)"
            )

    def launch_configs(self, workspace: Any) -> list[Any]:
        """Every distinct launch configuration (``jit.KernelParams``) this config
        can select on ``workspace``: one per planner decision over the per-rank
        token counts a forward can stage, 0 to the requested capacity (not the
        aligned one: counts above the capacity are refused in stage_inputs)."""
        ws: MoscaWorkspace = workspace
        return list(
            dict.fromkeys(self._params_for(ws, t) for t in range(ws.capacity + 1))
        )

    def _precompile(self, ws: "MoscaWorkspace") -> None:
        """Build every launch configuration now, then load it; all EP ranks
        agree on each step. The builds are sharded by rank (each module under
        its own lock), so they run in parallel and no rank relinks a module
        another rank is loading."""
        from . import jit

        configs = self.launch_configs(ws)
        self._agree(
            ws,
            "building",
            lambda: jit.build_kernel_modules(
                configs, shard=ws.rank, num_shards=ws.world_size
            ),
        )
        self._agree(ws, "loading", lambda: [jit.kernel_module(p) for p in configs])

    @staticmethod
    def _on_every_rank(world: int, device: torch.device, group: Any, step: Any) -> Any:
        """Run the rank-local ``step``; if it raised on any EP rank, raise on every rank.

        The rank where ``step`` failed re-raises its own exception; the other
        ranks raise ``MoEEpConfigError``. Use it before a collective that would
        otherwise wait for a rank that failed.
        """
        error: Exception | None = None
        result = None
        try:
            result = step()
        except Exception as exc:  # noqa: BLE001 - re-raised on every rank below
            error = exc
        if world > 1:
            import torch.distributed as dist

            ok = torch.tensor([0 if error else 1], dtype=torch.int32, device=device)
            dist.all_reduce(ok, op=dist.ReduceOp.MIN, group=group)
            if error is None and not bool(ok.item()):
                raise MoEEpConfigError(
                    f"{KERNEL_NAME}: workspace preparation failed on another EP rank"
                )
        if error is not None:
            raise error
        return result

    @staticmethod
    def _agree(ws: "MoscaWorkspace", what: str, step: Any) -> None:
        """Run ``step`` on every rank; raise on every rank if it failed on any."""
        error: Exception | None = None
        try:
            step()
        except Exception as exc:  # noqa: BLE001 - re-raised on every rank below
            error = exc
        failed_elsewhere = False
        if ws.world_size > 1:
            import torch.distributed as dist

            ok = torch.tensor([0 if error else 1], dtype=torch.int32, device=ws.device)
            dist.all_reduce(ok, op=dist.ReduceOp.MIN, group=ws.group)
            failed_elsewhere = not bool(ok.item())
        if error is not None:
            raise MoEEpConfigError(
                f"{KERNEL_NAME}: {what} the kernel modules failed: {error}"
            ) from error
        if failed_elsewhere:
            raise MoEEpConfigError(
                f"{KERNEL_NAME}: {what} the kernel modules failed on another EP rank"
            )

    # ------------------------------------------------------------------ init

    def _kernel_capacity(self, fleet_params: FleetParams, world_size: int) -> int:
        """Per-rank kernel token capacity: the layer bound, split over the ranks when replicated."""
        cap = fleet_params.max_tokens_per_rank
        if self._kernel_config.gathers_output:
            cap = -(-cap // world_size)
        return cap

    def validate_init(
        self, bootstrap: BootstrapConfig, fleet_params: FleetParams
    ) -> None:
        validate_mega_arch()
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            if major != 10:
                raise MoEEpArchError(
                    f"{KERNEL_NAME} (experimental) requires an SM100-family GPU "
                    f"(compute capability 10.x); host has sm_{major}{minor}"
                )
        k = self._kernel_config
        validate_mega_fleet_params(
            fleet_params,
            bootstrap.world_size,
            intermediate_size=k.intermediate_size,
            top_k=k.top_k,
            hidden_alignment=256,
        )
        if fleet_params.dtype_bytes not in (None, 2):
            raise MoEEpConfigError(
                f"{KERNEL_NAME} stages BF16 activations (dtype_bytes=2)"
            )
        if k.top_k > fleet_params.num_experts:
            raise MoEEpConfigError("top_k must not exceed num_experts")
        if not k.precompile and bootstrap.world_size > 1:
            raise MoEEpConfigError(
                f"{KERNEL_NAME}: precompile=False needs a single EP rank; with "
                "more, a rank that compiles inside forward (or fails to) leaves "
                "its peers waiting in the kernel's NVLink barrier"
            )
        if k.gathers_output:
            cap = self._kernel_capacity(fleet_params, bootstrap.world_size)
            if cap > MAX_GATHER_TOKENS_PER_RANK:
                raise MoEEpConfigError(
                    f"{KERNEL_NAME} replicated layout: per-rank capacity {cap} exceeds "
                    f"{MAX_GATHER_TOKENS_PER_RANK} (the gathered launch packs the token "
                    "count into 14 bits)"
                )
        if torch.cuda.is_available():
            warn_if_outside_validated_coverage(
                k,
                fleet_params,
                bootstrap.world_size,
                torch.cuda.get_device_capability(),
            )

    def preprocess_weights(
        self, weights: MoEWeightPack, fleet_params: FleetParams
    ) -> TransformedMoscaMegaWeights:
        return preprocess_mega_weights(
            weights,
            intermediate_size=self._kernel_config.intermediate_size,
            hidden_size=fleet_params.token_hidden_size,
        )

    def validate_transformed_weights(
        self,
        transformed_weights: TransformedMoscaMegaWeights,
        bootstrap: BootstrapConfig,
        fleet_params: FleetParams,
    ) -> None:
        validate_transformed_mega_weights(
            transformed_weights,
            intermediate_size=self._kernel_config.intermediate_size,
            hidden_size=fleet_params.token_hidden_size,
            num_local_experts=fleet_params.num_experts // self.ep_world_size,
        )

    # ------------------------------------------------------------- workspace

    def _workspace_pool_key(self, fleet_params: FleetParams) -> Any:
        k = self._kernel_config
        return (
            KERNEL_NAME,
            torch.cuda.current_device(),
            id(self.ep_comm_group) if self.ep_world_size > 1 else None,
            fleet_params.num_experts,
            self._kernel_capacity(fleet_params, self.ep_world_size),
            k.top_k,
            fleet_params.token_hidden_size,
            k.intermediate_size,
            k.gathers_output,
            k.num_sms,
        )

    def _allocate_workspace(self, fleet_params: FleetParams) -> MoscaWorkspace:
        import tvm_ffi

        from . import jit

        if torch.cuda.is_current_stream_capturing():
            raise MoEEpConfigError(
                "MOSCA MegaMoE workspace allocation cannot run during CUDA graph capture"
            )
        k = self._kernel_config
        device = torch.device("cuda", torch.cuda.current_device())
        world, rank = self.ep_world_size, self.ep_rank
        group = self.ep_comm_group if world > 1 else None

        def prepare() -> tuple[int, int, int, tuple[int, ...]]:
            jit.require_supported_device(device)
            host = jit.host_module()
            alignment = int(host.token_alignment())
            capacity = self._kernel_capacity(fleet_params, world)
            aligned = -(-capacity // alignment) * alignment
            if k.gathers_output and (
                aligned > MAX_GATHER_TOKENS_PER_RANK or world * aligned >= 1 << 18
            ):
                raise MoEEpConfigError(
                    f"{KERNEL_NAME} replicated layout: the aligned per-rank capacity "
                    f"{aligned} (x{world} ranks) does not fit the gathered launch's "
                    "packed arguments (capacity < 2**14, ranks * capacity < 2**18)"
                )
            device_sms = torch.cuda.get_device_properties(device).multi_processor_count
            num_sms = device_sms - device_sms % 2 if k.num_sms is None else k.num_sms
            if num_sms > device_sms:
                raise MoEEpConfigError(
                    f"{KERNEL_NAME}: num_sms={num_sms} exceeds the device's {device_sms} SMs"
                )
            layout = tuple(
                int(v)
                for v in host.layout(
                    world,
                    fleet_params.num_experts,
                    aligned,
                    k.top_k,
                    fleet_params.token_hidden_size,
                    k.intermediate_size,
                    num_sms,
                    k.gathers_output,
                )
            )
            return capacity, aligned, num_sms, layout

        # Every rank finishes its local preparation (planner build, buffer
        # layout) before any rank enters the symmetric-memory rendezvous below:
        # a rank that failed here would leave its peers waiting there.
        capacity, aligned, num_sms, layout = self._on_every_rank(
            world, device, group, prepare
        )
        total_bytes = layout[0]
        if world == 1:
            buffer = torch.empty(total_bytes, dtype=torch.uint8, device=device)
            handle = None
            ptrs = [buffer.data_ptr()]
        else:
            from ....comm.torch_symmetric_memory import _alloc_symm_buffer_bytes

            ptrs, buffer, handle = _alloc_symm_buffer_bytes(
                total_bytes, world, torch.uint8, device, group.group_name
            )
        # The kernel's counters and signals start from zero (as upstream's SymmBuffer).
        buffer.zero_()
        torch.cuda.synchronize(device)
        if world > 1:
            import torch.distributed as dist

            dist.barrier(group=group)

        hidden = fleet_params.token_hidden_size
        top_k = k.top_k
        (_, _, gathered_offset, _, _, off_x, off_x_sf, off_idx, off_w, *_rest) = layout
        gathered = None
        own_rows = None
        if k.gathers_output:
            gathered = _view(
                buffer, gathered_offset, (world * aligned, hidden), torch.bfloat16
            )
            own_rows = torch.empty(
                (aligned, hidden), dtype=torch.bfloat16, device=device
            )
        return MoscaWorkspace(
            device=device,
            group=group,
            rank=rank,
            world_size=world,
            capacity=capacity,
            num_experts=fleet_params.num_experts,
            hidden=hidden,
            num_max_tokens_per_rank=aligned,
            num_sms=num_sms,
            gather=k.gathers_output,
            layout=layout,
            buffer=buffer,
            handle=handle,
            peers=tvm_ffi.Shape(tuple(int(p) for p in ptrs)),
            x=_view(buffer, off_x, (aligned, hidden), torch.float8_e4m3fn),
            x_sf=_view(buffer, off_x_sf, (aligned, hidden // 128), torch.int32),
            topk_idx=_view(buffer, off_idx, (aligned, top_k), torch.int64),
            topk_weights=_view(buffer, off_w, (aligned, top_k), torch.float32),
            gathered=gathered,
            own_rows=own_rows,
        )

    # --------------------------------------------------------------- forward

    def validate_forward(
        self, t: "MoEEpTensors", fleet_params: FleetParams, *, quantize_input: bool
    ) -> None:
        validate_mega_forward_inputs(
            t.hidden_states,
            t.topk_ids,
            t.topk_weights,
            fleet_params,
            top_k=self._kernel_config.top_k,
            quantize_input=quantize_input,
            scales=t.scales,
        )

    def stage_inputs(
        self, t: "MoEEpTensors", workspace: Any, *, quantize_input: bool
    ) -> None:
        from .staging import stage_inputs

        ws: MoscaWorkspace = workspace
        total = t.hidden_states.shape[0]
        if ws.gather:
            start, count = split_rows(total, ws.world_size, ws.rank)
        else:
            start, count = 0, total
        if count > ws.capacity:
            raise MoEEpConfigError(
                f"{count} rows exceed the per-rank capacity {ws.capacity}"
            )
        if torch.cuda.is_current_stream_capturing():
            from . import jit

            if not jit.is_loaded(self._params_for(ws, count)):
                raise MoEEpConfigError(
                    f"{KERNEL_NAME}: no kernel module is built for {count} tokens per "
                    "rank; run that token count before CUDA graph capture, or keep "
                    "precompile=True"
                )
        rows = slice(start, start + count)
        if quantize_input:
            stage_inputs(
                t.hidden_states[rows],
                t.topk_weights[rows],
                t.topk_ids[rows],
                ws.x[:count],
                ws.x_sf[:count],
                ws.topk_idx[:count],
                ws.topk_weights[:count],
            )
        else:
            if t.scales is None:
                raise MoEEpConfigError(
                    "quantize_input=False requires MoEEpTensors.scales"
                )
            ws.x[:count].view(torch.uint8).copy_(
                t.hidden_states[rows].view(torch.uint8)
            )
            ws.x_sf[:count].copy_(t.scales[rows].view(torch.int32))
            ws.topk_idx[:count].copy_(t.topk_ids[rows])
            ws.topk_weights[:count].copy_(t.topk_weights[rows])
        ws.staged_tokens = count
        ws.staged_row_offset = start if ws.gather else -1
        ws.staged_total_tokens = total

    def _params_for(self, ws: MoscaWorkspace, num_tokens: int) -> Any:
        # The workspace is pooled across layers and may be shared by backends whose
        # configs differ only in compile-time switches, so the cache key carries them.
        k = self._kernel_config
        key = (
            num_tokens,
            k.small_t,
            k.sparse_count,
            k.tile_rule,
            k.activation_clamp,
            k.fast_math,
        )
        params = ws.plans.get(key)
        if params is not None:
            return params
        from . import jit

        host = jit.host_module()
        try:
            plan = tuple(
                int(v)
                for v in host.plan(
                    ws.world_size,
                    ws.num_experts,
                    ws.num_max_tokens_per_rank,
                    num_tokens,
                    k.top_k,
                    ws.hidden,
                    k.intermediate_size,
                    ws.num_sms,
                    k.tile_rule or "",
                )
            )
        except (
            Exception
        ) as exc:  # the planner raises tvm-ffi errors (e.g. a bad tile_rule)
            raise MoEEpConfigError(
                f"{KERNEL_NAME}: no launch configuration for {num_tokens} tokens per "
                f"rank: {exc}"
            ) from exc
        params = jit.KernelParams(
            num_max_tokens_per_rank=ws.num_max_tokens_per_rank,
            hidden=ws.hidden,
            intermediate_hidden=k.intermediate_size,
            num_experts=ws.num_experts,
            num_topk=k.top_k,
            num_ranks=ws.world_size,
            num_sms=ws.num_sms,
            block_m=plan[0],
            block_n=plan[1],
            block_k=plan[2],
            store_block_m=plan[5],
            sf_block_m=plan[6],
            sf_block_n=plan[7],
            gran_k=plan[8],
            num_ring_tokens=plan[9],
            num_sf_ring_tokens=plan[10],
            swizzle_acts_mode=plan[11],
            swizzle_weights_mode=plan[12],
            num_stages=plan[13],
            smem_size=plan[14],
            num_dispatch_threads=plan[15],
            num_non_epilogue_threads=plan[16],
            num_epilogue_threads=plan[17],
            num_bytes_per_pull=plan[18],
            activation_clamp=float(k.activation_clamp),
            fast_math=k.fast_math,
            small_t=k.small_t,
            sparse_count=k.sparse_count,
            gather=k.gathers_output,
        )
        ws.plans[key] = params
        return params

    def kernel_params(self, workspace: Any, num_tokens: int) -> Any:
        """Launch configuration (``jit.KernelParams``) of a call with ``num_tokens`` rows per rank."""
        return self._params_for(workspace, num_tokens)

    def compute(
        self,
        workspace: Any,
        transformed_weights: TransformedMoscaMegaWeights,
        *,
        output: torch.Tensor | None,
    ) -> torch.Tensor:
        from . import jit

        ws: MoscaWorkspace = workspace
        if ws.destroyed:
            raise MoEEpConfigError("MOSCA MegaMoE workspace has been destroyed")
        if output is None:
            raise MoEEpConfigError(f"{KERNEL_NAME} returns an owned output only")
        (l1_w, l1_sf), (l2_w, l2_sf) = transformed_weights
        count = ws.staged_tokens
        module = jit.kernel_module(self._params_for(ws, count))
        if ws.gather:
            y = ws.own_rows[:count]
        else:
            if output.shape[0] < count:
                raise MoEEpConfigError("output holds fewer rows than the staged tokens")
            y = output[:count]
        stream = torch.cuda.current_stream(ws.device).cuda_stream
        module.launch(
            y,
            l1_w,
            l1_sf,
            l2_w,
            l2_sf,
            ws.buffer,
            ws.peers,
            ws.rank,
            ws.staged_row_offset,
            stream,
        )
        if ws.gather:
            output[: ws.staged_total_tokens].copy_(
                ws.gathered[: ws.staged_total_tokens]
            )
        return output

    def _forget_workspace_state(self, workspace: Any) -> None:
        if isinstance(workspace, MoscaWorkspace):
            workspace.plans.clear()


__all__ = [
    "MoscaWorkspace",
    "Sm100MoscaMegaKernelBackend",
    "enabled_features",
    "split_rows",
    "warn_if_outside_validated_coverage",
]
