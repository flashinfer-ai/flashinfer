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

# Cake W8A8 MXFP8 MegaMoE EP16 backend for SM103 (experimental).
#
# Registered under ``kernel_name="sm103_mxfp8_mxfp8_bf16_cake"`` and reached
# through the stable :class:`~flashinfer.moe_ep.modes.mega_layer.MoEEpMegaLayer`
# API. One forward is a fixed sequence of five Cake-generated programs on the
# current stream (publish-push: in-kernel MXFP8 quantization, combine-window
# and routing-state reset, one pushed slot per token into the owner ranks'
# inboxes plus a flag; local pack of the owned rows per expert after a local
# flag wait; SwapAB FC1 with SwiGLU and requant; SwapAB FC2 with token-back
# push and a completion signal into every rank's counter; counted source-order
# combine that waits on its local counter) with no host synchronization, no
# host memset and no allocation. There is no input barrier and no partial
# barrier. The fc1, fc2 and combine bindings launch with the
# programmatic-stream-serialization attribute (the route's programmatic
# dependent launch edges local_pack -> fc1, fc1 -> fc2 and fc2 -> combine,
# ``cake_plan.PDL_EDGES``); the other two launches carry no attributes.
#
# Contract: EP16 x 32 local experts (512 total), hidden 3072, intermediate
# 5120, top-8, at most 64 tokens per rank, at most 32 routes to one expert per
# forward. Routes at or beyond row 32 of an expert are not packed or computed;
# the local_pack program counts them in a device-side flag that
# :func:`routing_overflow` exposes (no implicit synchronization). Every rank
# publishes the same rank-uniform token bound: ``compile_tokens_per_rank`` when
# given, else the workspace capacity; rows up to the bound are padded (``-1``
# routes). A ``-1`` (masked) slot inside a live row contributes zero: the
# publish-push program zeroes this rank's combine read window of the term
# buffer every forward, so a slot no FC2 push writes stays zero instead of
# keeping a previous forward's term. The inbox flags and the fc2 completion
# counter are never reset, so every rank must execute the same number of
# forwards (the layer protocol: one eager warmup on all ranks, then eager
# forwards or graph replays in lockstep). CUDA graph capture needs an explicit
# bound and that prior eager forward.
#
# Import-time surface: this module registers the backend when imported by
# ``flashinfer.moe_ep`` (through ``backends/mega/kernel/sm103``), so it stays
# lightweight: tvm-ffi and the JIT loader (``cake_jit``) are imported inside
# the functions that use them, never at module level.

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

import torch
import torch.distributed as dist

from ...api_logging import warn_experimental_backend_once
from ...moe_ep.config import BootstrapConfig, FleetParams
from ...moe_ep.core.kernel.base import MegaKernelBackend
from ...moe_ep.core.kernel.registry import register_mega_kernel
from ...moe_ep.core.runtime import NVSHMEM, TORCH_DIST
from ...moe_ep.core.validation.common import MoEEpArchError, MoEEpConfigError
from ...moe_ep.weights import MoEWeightPack
from .cake_config import KERNEL_NAME, Sm103_Mxfp8_Mxfp8_Bf16_Cake_MegaMoeConfig
from .cake_plan import (
    ACTIVE_ROWS_PER_EXPERT,
    FC1_K128_TILES,
    FC1_ROWS,
    FC2_COMPLETION_WORDS,
    FC2_K128_TILES,
    HIDDEN,
    INTERMEDIATE,
    LOCAL_EXPERTS,
    MAX_RECV_TOKENS,
    MAX_ROUTES,
    MAX_ROWS_PER_EXPERT,
    MAX_TOKENS_PER_RANK,
    NUM_EXPERTS,
    PUSH_FLAGS_SHAPE,
    PUSH_GENERATION_WORDS,
    PUSH_INBOX_SHAPE,
    SCALE_TILE_K128,
    STAGES,
    TOP_K,
    TOTAL_EXPERT_ROWS,
    WORLD_SIZE,
    StageGrids,
    StaticPlan,
    scale_bytes,
    stage_grids,
    static_plan,
)
from .cake_staging import (
    resolve_bound,
    stage_inputs as _stage_inputs,
    validate_forward_inputs,
)
from .cake_symmetric import (
    SymmetricBuffer,
    allocate_flag_pad,
    allocate_symmetric,
    check_self_entry,
)
from .cake_weights import (
    TransformedCakeW8a8MegaWeights,
    preprocess_mega_weights,
    validate_transformed_mega_weights,
)

if TYPE_CHECKING:
    from ...moe_ep.modes.mega_layer import MoEEpMegaLayer
    from ...moe_ep.tensors import MoEEpTensors

# tvm-ffi entry name of every stage binding. The contract is the binding file
# name and its export: ``cake_w8a8_megamoe_ep16_<stage>_binding.cu`` exports
# exactly one function named ``<stage>``, so the module attribute of a stage is
# the stage name itself.
STAGE_ENTRIES: dict[str, str] = {stage: stage for stage in STAGES}

STABLE_API_NAME = "MoEEpMegaLayer"

StageCall = tuple[Callable[..., Any], tuple[Any, ...]]


@dataclass
class CakeWorkspace:
    """Per-EP-group workspace of the route (pooled across layers)."""

    device: torch.device
    group: Any
    rank: int
    world_size: int
    capacity: int  # FleetParams.max_tokens_per_rank (<= 64)
    module: Any
    entries: dict[str, Callable[..., Any]]
    # ``tvm_ffi.use_torch_stream`` resolved once at allocation (the forward
    # path imports nothing).
    use_torch_stream: Callable[[], Any]
    plan: StaticPlan
    # Symmetric (peer-mapped) buffers (``SYMMETRIC_BUFFERS``).
    push_inbox: SymmetricBuffer  # uint8 [16, 64, 3232]: slot per (source rank, token)
    push_flags: (
        SymmetricBuffer  # uint32 [16, 64], zeroed once at allocation, never reset
    )
    local_terms: SymmetricBuffer  # bf16 [1024, 8, 3072]
    # uint32 [2], zeroed once at allocation, never written by the host: [0] done
    # counter (+1 from every rank per forward, monotonic), [1] fc2 CTA arrivals.
    fc2_completion: SymmetricBuffer
    pg_flags: (
        SymmetricBuffer  # uint32 [PG_FLAG_WORDS = 66], zeroed once, never re-zeroed
    )
    # Rank-local buffers.
    push_generation: (
        torch.Tensor
    )  # uint32 [2], zeroed once: [0] forwards completed, [1] local_pack CTA arrivals
    recv_weights: torch.Tensor  # float32 [8192]
    expert_counts: torch.Tensor  # int32 [32]
    route_map: torch.Tensor  # int32 [4096]
    inverse_rows: torch.Tensor  # int32 [8192]
    overflow: (
        torch.Tensor
    )  # int32 [1]; local_pack counts routes at or beyond row 32 of an expert
    fc1_input_fp8: torch.Tensor  # e4m3 [32, 128, 3072]
    fc1_input_scales: torch.Tensor  # uint8 [393216]
    fc2_input_fp8: torch.Tensor  # e4m3 [32, 128, 5120]
    fc2_input_scales: torch.Tensor  # uint8 [655360]
    # Staging (publish sources) and output.
    hidden: torch.Tensor  # bf16 [64, 3072]
    ids: torch.Tensor  # int64 [64, 8]
    weights: torch.Tensor  # float32 [64, 8]
    output: torch.Tensor  # bf16 [64, 3072]
    compile_tokens_per_rank: int | None = None
    staged_tokens: int = 0
    staged_bound: int | None = None
    launches: int = 0
    grids: dict[int, StageGrids] = field(default_factory=dict)
    launch_plans: dict[tuple[int, int], tuple[list[StageCall], Any]] = field(
        default_factory=dict
    )
    destroyed: bool = False

    def destroy(self) -> None:
        """Release the symmetric buffers (collective; all ranks together)."""

        if self.destroyed:
            return
        self.destroyed = True
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.group)
        self.launch_plans.clear()
        self.grids.clear()
        self.entries = {}
        for buffer in (
            self.pg_flags,
            self.fc2_completion,
            self.local_terms,
            self.push_flags,
            self.push_inbox,
        ):
            buffer.free()


# Symmetric (peer-mapped) buffers of one workspace, allocated collectively in
# this order: name -> (shape, dtype). ``allocate_symmetric`` zeroes each once;
# ``push_flags`` and ``fc2_completion`` are never reset afterwards (manifest
# ``route_contract.push_dispatch.generation_rule`` and
# ``route_contract.fc2_completion.initialization_rule``).
SYMMETRIC_BUFFERS: dict[str, tuple[tuple[int, ...], torch.dtype]] = {
    "push_inbox": (PUSH_INBOX_SHAPE, torch.uint8),
    "push_flags": (PUSH_FLAGS_SHAPE, torch.uint32),
    "local_terms": ((MAX_RECV_TOKENS, TOP_K, HIDDEN), torch.bfloat16),
    "fc2_completion": ((FC2_COMPLETION_WORDS,), torch.uint32),
}


# ---------------------------------------------------------------------------
# Stage call interface (Python -> hand-written bindings).
#
# Each function returns the positional argument tuple of one binding call: the
# program's device arguments in the manifest's physical order, then ``grid_x``
# (all grids are one-dimensional; block size and dynamic shared memory are
# fixed constants of the binding). The device-argument kinds map as follows:
# buffer and TMA operand -> the tensor (``TensorView``; the binding encodes the
# tensor map), parameter and process-group scalar -> ``int`` (``int64_t``,
# range-checked to i32 in the binding), peer table -> a tuple of exactly 16
# signed int64 images (``tvm::ffi::Shape``; slot r = rank r, all non-null,
# ``SymmetricBuffer.peer_images``). A symmetric buffer travels as its local
# tensor followed by its peer table. Every program has a tensor argument, so
# every binding selects the device and stream from its first tensor.
# ---------------------------------------------------------------------------
def publish_push_args(ws: CakeWorkspace, grids: StageGrids) -> tuple[Any, ...]:
    """source_hidden bf16[64,3072], source_ids i64[64,8], source_weights f32[64,8],
    local_terms (sym local) bf16 [1024,8,3072] (this rank's combine read window,
    rows [rank * tokens, (rank + 1) * tokens), is zeroed), expert_counts i32[32],
    route_map i32[4096], overflow i32[1] (all three reset), push_generation u32[2],
    tokens, pg_world, pg_rank, pg_flags (16-entry peer table), push_inbox (sym
    local) u8 [16,64,3232], push_inbox_peers, push_flags (sym local) u32 [16,64],
    push_flags_peers, grid_x = tokens (one CTA per staged token)."""

    return (
        ws.hidden,
        ws.ids,
        ws.weights,
        ws.local_terms.local,
        ws.expert_counts,
        ws.route_map,
        ws.overflow,
        ws.push_generation,
        grids.tokens_per_rank,
        ws.world_size,
        ws.rank,
        ws.pg_flags.peer_images,
        ws.push_inbox.local,
        ws.push_inbox.peer_images,
        ws.push_flags.local,
        ws.push_flags.peer_images,
        grids.publish_push,
    )


def local_pack_args(ws: CakeWorkspace, grids: StageGrids) -> tuple[Any, ...]:
    """recv_weights f32[8192], expert_counts, route_map, inverse_rows, overflow,
    quantized u8 view of fc1_input_fp8 [32,128,3072], scales_cp u8[393216],
    push_generation u32[2], tokens_per_rank, active_rows_per_expert = 32 (routes
    at or beyond this row of an expert are counted in overflow, not packed),
    scale_tile_k128 = 1, pg_world, pg_rank, pg_flags (16-entry peer table),
    push_inbox (sym local), push_inbox_peers, push_flags (sym local),
    push_flags_peers, grid_x = 16 * tokens (one CTA per (source rank, token))."""

    return (
        ws.recv_weights,
        ws.expert_counts,
        ws.route_map,
        ws.inverse_rows,
        ws.overflow,
        ws.fc1_input_fp8.view(torch.uint8),
        ws.fc1_input_scales,
        ws.push_generation,
        grids.tokens_per_rank,
        ACTIVE_ROWS_PER_EXPERT,
        SCALE_TILE_K128,
        ws.world_size,
        ws.rank,
        ws.pg_flags.peer_images,
        ws.push_inbox.local,
        ws.push_inbox.peer_images,
        ws.push_flags.local,
        ws.push_flags.peer_images,
        grids.local_pack,
    )


def fc1_args(
    ws: CakeWorkspace, tw: TransformedCakeW8a8MegaWeights, grids: StageGrids
) -> tuple[Any, ...]:
    """SwapAB FC1 (routed rows on the MMA-N axis):
    A = w13 u8 view [32,10240,3072] (k-tile-major payload, 256-row blocks),
    B = fc1_input_fp8 u8 view [32,128,3072],
    SFA = w13_scales.view(32,-1,32,32), SFB = fc1_input_scales.view(32,-1,16,32),
    C_fp8 = fc2_input_fp8 e4m3 [32,128,5120], SF_out = fc2_input_scales u8[655360],
    route_map, route_weights = recv_weights, tile_expert i32[1280], tile_m_local i32[1280],
    expert_row_offsets i32[32], scale_gate = ones, clamp_limit = infinities,
    act_alpha = ones, act_beta = zeros, N = 10240, K_tiles = 24, grid_m = 1280,
    grid_n = 1, rows_per_expert = 128, grid_x = persistent grid."""

    plan = ws.plan
    return (
        tw.w13.view(torch.uint8),
        ws.fc1_input_fp8.view(torch.uint8),
        tw.w13_scales.view(LOCAL_EXPERTS, -1, 32, 32),
        ws.fc1_input_scales.view(LOCAL_EXPERTS, -1, 16, 32),
        ws.fc2_input_fp8,
        ws.fc2_input_scales,
        ws.route_map,
        ws.recv_weights,
        plan.fc1_tile_expert,
        plan.fc1_tile_m_local,
        plan.expert_row_offsets,
        plan.ones,
        plan.infinities,
        plan.ones,
        plan.zeros,
        FC1_ROWS,
        FC1_K128_TILES,
        plan.fc1_grid_m,
        1,
        MAX_ROWS_PER_EXPERT,
        grids.fc1,
    )


def fc2_args(
    ws: CakeWorkspace, tw: TransformedCakeW8a8MegaWeights, grids: StageGrids
) -> tuple[Any, ...]:
    """SwapAB FC2 with token-back push:
    A = w2 u8 view [32,3072,5120] (k-tile-major payload, 128-row blocks),
    B = fc2_input_fp8 u8 view [32,128,5120],
    SFA = w2_scales_swap.view(32,-1,16,32), SFB = fc2_input_scales.view(32,-1,16,32),
    C = local_terms (sym local) bf16 [1024,8,3072], route_map, tile_expert i32[768],
    tile_m_local i32[768], expert_row_offsets, routes_per_rank = tokens * 8,
    N = 3072, K_tiles = 40, grid_m = 768,
    remote_pg_world, remote_pg_rank, remote_pg_flags (the same 16-entry flag-pad table),
    remote_terms = local_terms (sym local), remote_terms_peers,
    fc2_completion (sym local) u32 [2] (the last CTA adds +1 into word 0 of every
    rank; word 1 counts this grid's CTA arrivals), fc2_completion_peers,
    grid_x = persistent grid."""

    plan = ws.plan
    return (
        tw.w2.view(torch.uint8),
        ws.fc2_input_fp8.view(torch.uint8),
        tw.w2_scales_swap.view(LOCAL_EXPERTS, -1, 16, 32),
        ws.fc2_input_scales.view(LOCAL_EXPERTS, -1, 16, 32),
        ws.local_terms.local,
        ws.route_map,
        plan.fc2_tile_expert,
        plan.fc2_tile_m_local,
        plan.expert_row_offsets,
        grids.tokens_per_rank * TOP_K,
        HIDDEN,
        FC2_K128_TILES,
        plan.fc2_grid_m,
        ws.world_size,
        ws.rank,
        ws.pg_flags.peer_images,
        ws.local_terms.local,
        ws.local_terms.peer_images,
        ws.fc2_completion.local,
        ws.fc2_completion.peer_images,
        grids.fc2,
    )


def combine_args(ws: CakeWorkspace, grids: StageGrids) -> tuple[Any, ...]:
    """local_terms (sym local) bf16 [1024,8,3072], output bf16 [64,3072],
    push_generation u32 [2] (the word local_pack advances; the wait target is
    16 * push_generation[0]), tokens_per_rank, source_rank, pg_world, pg_rank,
    pg_flags (16-entry peer table), fc2_completion (sym local) u32 [2] (thread 0
    of each CTA polls word 0 until it reached the target), fc2_completion_peers,
    grid_x = 3 * tokens (one CTA per 1024-column slice of a token)."""

    return (
        ws.local_terms.local,
        ws.output,
        ws.push_generation,
        grids.tokens_per_rank,
        ws.rank,
        ws.world_size,
        ws.rank,
        ws.pg_flags.peer_images,
        ws.fc2_completion.local,
        ws.fc2_completion.peer_images,
        grids.combine,
    )


StageArgs = Callable[
    [CakeWorkspace, TransformedCakeW8a8MegaWeights, StageGrids], tuple[Any, ...]
]

# Argument builder of every forward stage; ``STAGES`` is the launch order.
STAGE_ARGS: dict[str, StageArgs] = {
    "publish_push": lambda ws, tw, grids: publish_push_args(ws, grids),
    "local_pack": lambda ws, tw, grids: local_pack_args(ws, grids),
    "fc1": fc1_args,
    "fc2": fc2_args,
    "combine": lambda ws, tw, grids: combine_args(ws, grids),
}


def build_stage_calls(
    ws: CakeWorkspace, tw: TransformedCakeW8a8MegaWeights, grids: StageGrids
) -> list[StageCall]:
    """The (binding, arguments) pairs of one forward, in ``STAGES`` launch order."""

    entries = ws.entries
    return [(entries[stage], STAGE_ARGS[stage](ws, tw, grids)) for stage in STAGES]


# ---------------------------------------------------------------------------
# Backend.
# ---------------------------------------------------------------------------
@register_mega_kernel(KERNEL_NAME)
class Sm103Mxfp8Mxfp8CakeMegaKernelBackend(MegaKernelBackend):
    supports_output_view = True

    def __init__(self, config: Sm103_Mxfp8_Mxfp8_Bf16_Cake_MegaMoeConfig) -> None:
        if not isinstance(config, Sm103_Mxfp8_Mxfp8_Bf16_Cake_MegaMoeConfig):
            raise TypeError(
                f"{KERNEL_NAME} config must be Sm103_Mxfp8_Mxfp8_Bf16_Cake_MegaMoeConfig, "
                f"got {type(config).__name__}"
            )
        if config.fast_math is not True:
            raise NotImplementedError("fast_math=False is not supported")
        super().__init__(config)
        self._kernel_config = config
        # Launch-plan cache keys this instance inserted, per workspace id, so a
        # destroyed layer releases its weights from the pooled workspace.
        self._plan_keys: dict[int, set[tuple[int, int]]] = {}
        warn_experimental_backend_once(STABLE_API_NAME, KERNEL_NAME)

    # ------------------------------------------------------------------ setup

    @classmethod
    def kernel_name(cls) -> str:
        return KERNEL_NAME

    def runtime_requirements(self, bootstrap: BootstrapConfig) -> frozenset[str]:
        del bootstrap
        return frozenset({TORCH_DIST, NVSHMEM})

    def _geometry_context(self, world_size: int, fleet_params: FleetParams) -> str:
        return (
            f"world_size={world_size}, num_experts={fleet_params.num_experts}, "
            f"hidden={fleet_params.token_hidden_size}, "
            f"intermediate={self._kernel_config.intermediate_size}, "
            f"top_k={self._kernel_config.top_k}, "
            f"max_tokens_per_rank={fleet_params.max_tokens_per_rank}, "
            f"dtype_bytes={fleet_params.dtype_bytes}"
        )

    def _validate_geometry(self, world_size: int, fleet_params: FleetParams) -> None:
        context = self._geometry_context(world_size, fleet_params)
        checks = (
            (world_size == WORLD_SIZE, f"requires EP world size {WORLD_SIZE}"),
            (
                fleet_params.num_experts == NUM_EXPERTS,
                f"requires {NUM_EXPERTS} experts ({LOCAL_EXPERTS} per rank)",
            ),
            (
                fleet_params.token_hidden_size == HIDDEN,
                f"requires token_hidden_size {HIDDEN}",
            ),
            (
                self._kernel_config.intermediate_size == INTERMEDIATE,
                f"requires intermediate_size {INTERMEDIATE}",
            ),
            (self._kernel_config.top_k == TOP_K, f"requires top_k {TOP_K}"),
            (
                1 <= fleet_params.max_tokens_per_rank <= MAX_TOKENS_PER_RANK,
                f"requires max_tokens_per_rank in [1, {MAX_TOKENS_PER_RANK}]",
            ),
            (fleet_params.dtype_bytes == 2, "requires bf16 tokens (dtype_bytes=2)"),
        )
        for ok, reason in checks:
            if not ok:
                raise MoEEpConfigError(
                    f"{KERNEL_NAME}: unsupported configuration ({context}): {reason}"
                )

    def _check_collective_agreement(self, fleet_params: FleetParams) -> None:
        """Every EP rank must construct the same backend (config + geometry)."""

        if self.ep_world_size <= 1:
            return
        proposal = (
            KERNEL_NAME,
            dataclasses.asdict(self._kernel_config),
            (
                fleet_params.num_experts,
                fleet_params.max_tokens_per_rank,
                fleet_params.token_hidden_size,
                fleet_params.dtype_bytes,
            ),
        )
        gathered: list[Any] = [None] * self.ep_world_size
        dist.all_gather_object(gathered, proposal, group=self.ep_comm_group)
        if any(candidate != proposal for candidate in gathered):
            raise MoEEpConfigError(
                f"all {WORLD_SIZE} EP ranks must construct the same sm103 Cake backend "
                "(kernel config and fleet geometry differ across ranks)"
            )

    def validate_init(
        self, bootstrap: BootstrapConfig, fleet_params: FleetParams
    ) -> None:
        if torch.cuda.is_available():
            capability = tuple(torch.cuda.get_device_capability())
            if capability != (10, 3):
                raise MoEEpArchError(
                    f"{KERNEL_NAME} requires sm_103 (compute capability 10.3); host has "
                    f"sm_{capability[0]}{capability[1]}"
                )
            from .cake_jit import require_supported_capability

            try:
                require_supported_capability(capability)
            except RuntimeError as error:
                raise MoEEpArchError(str(error)) from error
        self._validate_geometry(bootstrap.world_size, fleet_params)
        if bootstrap.stream != 0:
            raise MoEEpConfigError(
                f"{KERNEL_NAME} launches on the current torch stream only; "
                f"BootstrapConfig.stream must be 0, got {bootstrap.stream}"
            )
        if dist.is_available() and dist.is_initialized():
            self._check_collective_agreement(fleet_params)

    def preprocess_weights(
        self, weights: MoEWeightPack, fleet_params: FleetParams
    ) -> TransformedCakeW8a8MegaWeights:
        from .cake_jit import load_module

        del fleet_params
        device = torch.device("cuda", torch.cuda.current_device())
        return preprocess_mega_weights(weights, module=load_module(device))

    def validate_transformed_weights(
        self,
        transformed_weights: Any,
        bootstrap: BootstrapConfig,
        fleet_params: FleetParams,
    ) -> None:
        del bootstrap, fleet_params
        validate_transformed_mega_weights(transformed_weights)

    def _workspace_pool_key(self, fleet_params: FleetParams) -> Any:
        return (
            KERNEL_NAME,
            torch.cuda.current_device(),
            self.ep_rank,
            self.ep_world_size,
            id(self._ep_comm_group),
            fleet_params.max_tokens_per_rank,
        )

    # -------------------------------------------------------------- workspace

    def _allocate_workspace(self, fleet_params: FleetParams) -> CakeWorkspace:
        """Collective: symmetric buffers, flag pad, local buffers, static plan.

        ``allocate_symmetric`` zeroes every symmetric buffer (the inbox flags
        and the fc2 completion counter among them) and ``push_generation`` is
        allocated zeroed; the trailing device synchronize plus process-group
        barrier is the rendezvous the route requires: that one-time zeroing is
        complete and visible on every rank before any rank's first
        ``publish_push`` and first ``fc2``. The host never writes the flags,
        the counter or the generation words afterwards.
        """

        import tvm_ffi

        from .cake_jit import load_module

        if torch.cuda.is_current_stream_capturing():
            raise MoEEpConfigError(
                f"{KERNEL_NAME}: workspace allocation cannot run during CUDA graph capture"
            )
        group = self.ep_comm_group
        rank, world = self.ep_rank, self.ep_world_size
        self._validate_geometry(world, fleet_params)
        self._check_collective_agreement(fleet_params)
        device = torch.device("cuda", torch.cuda.current_device())
        module = load_module(device)
        entries = {stage: getattr(module, STAGE_ENTRIES[stage]) for stage in STAGES}

        def local(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
            return torch.zeros(shape, dtype=dtype, device=device)

        symmetric = {
            name: allocate_symmetric(shape, dtype, device=device)
            for name, (shape, dtype) in SYMMETRIC_BUFFERS.items()
        }
        # Symmetric self entry, asserted once per workspace (manifest
        # route_contract.push_dispatch.peer_table_rule and
        # route_contract.fc2_completion.counter_layout): slot ``rank`` of
        # every peer table is the local pointer the bindings receive beside
        # it. The flag pad is bound as a table only (no local pointer), so it
        # has no self-entry check.
        for name, buffer in symmetric.items():
            check_self_entry(buffer, rank=rank, name=name)
        pg_flags = allocate_flag_pad(device=device, group=group)
        workspace = CakeWorkspace(
            device=device,
            group=group,
            rank=rank,
            world_size=world,
            capacity=fleet_params.max_tokens_per_rank,
            module=module,
            entries=entries,
            use_torch_stream=tvm_ffi.use_torch_stream,
            plan=static_plan(device),
            push_inbox=symmetric["push_inbox"],
            push_flags=symmetric["push_flags"],
            local_terms=symmetric["local_terms"],
            fc2_completion=symmetric["fc2_completion"],
            pg_flags=pg_flags,
            push_generation=local((PUSH_GENERATION_WORDS,), torch.uint32),
            recv_weights=local((MAX_ROUTES,), torch.float32),
            expert_counts=local((LOCAL_EXPERTS,), torch.int32),
            route_map=local((TOTAL_EXPERT_ROWS,), torch.int32),
            inverse_rows=local((MAX_ROUTES,), torch.int32),
            overflow=local((1,), torch.int32),
            fc1_input_fp8=local(
                (LOCAL_EXPERTS, MAX_ROWS_PER_EXPERT, HIDDEN), torch.float8_e4m3fn
            ),
            fc1_input_scales=local(
                (scale_bytes(MAX_ROWS_PER_EXPERT, HIDDEN),), torch.uint8
            ),
            fc2_input_fp8=local(
                (LOCAL_EXPERTS, MAX_ROWS_PER_EXPERT, INTERMEDIATE),
                torch.float8_e4m3fn,
            ),
            fc2_input_scales=local(
                (scale_bytes(MAX_ROWS_PER_EXPERT, INTERMEDIATE),), torch.uint8
            ),
            hidden=local((MAX_TOKENS_PER_RANK, HIDDEN), torch.bfloat16),
            ids=torch.full(
                (MAX_TOKENS_PER_RANK, TOP_K), -1, dtype=torch.int64, device=device
            ),
            weights=local((MAX_TOKENS_PER_RANK, TOP_K), torch.float32),
            output=local((MAX_TOKENS_PER_RANK, HIDDEN), torch.bfloat16),
        )
        torch.cuda.synchronize(device)
        dist.barrier(group=group)
        return workspace

    # ---------------------------------------------------------------- forward

    def validate_forward(
        self, t: MoEEpTensors, fleet_params: FleetParams, *, quantize_input: bool
    ) -> None:
        validate_forward_inputs(
            t.hidden_states,
            t.topk_ids,
            t.topk_weights,
            fleet_params,
            quantize_input=quantize_input,
            scales=t.scales,
        )

    def validate_capture_ready(self, workspace: Any, transformed_weights: Any) -> None:
        ws = self._live(workspace)
        if ws.launches == 0:
            raise MoEEpConfigError(
                f"{KERNEL_NAME}: workspace is not warmed for CUDA graph capture; run "
                "one eager forward on all EP ranks first (layer.warmup(...))"
            )
        if not isinstance(transformed_weights, TransformedCakeW8a8MegaWeights):
            raise MoEEpConfigError(
                f"{KERNEL_NAME}: transformed weights must be TransformedCakeW8a8MegaWeights"
            )

    def set_compile_tokens_per_rank(
        self, workspace: Any, compile_tokens_per_rank: int | None
    ) -> None:
        """Rank-uniform token bound of the next invocation (``None`` = capacity)."""

        ws = self._live(workspace)
        if compile_tokens_per_rank is not None:
            # Validates type and range against the workspace capacity.
            resolve_bound(0, compile_tokens_per_rank, ws.capacity)
        ws.compile_tokens_per_rank = (
            None if compile_tokens_per_rank is None else int(compile_tokens_per_rank)
        )

    def stage_inputs(
        self, t: MoEEpTensors, workspace: Any, *, quantize_input: bool
    ) -> None:
        ws = self._live(workspace)
        if not quantize_input:
            raise MoEEpConfigError(
                f"{KERNEL_NAME} quantizes activations in-kernel; prequantized inputs are not supported"
            )
        bound = resolve_bound(t.num_tokens, ws.compile_tokens_per_rank, ws.capacity)
        if (
            torch.cuda.is_current_stream_capturing()
            and ws.compile_tokens_per_rank is None
        ):
            raise MoEEpConfigError(
                f"{KERNEL_NAME}: CUDA graph capture needs an explicit rank-uniform bound "
                "(MoEEpMegaLayer.stage_inputs(t, compile_tokens_per_rank=...))"
            )
        # Staging launches nothing: the publish-push program zeroes this
        # rank's combine read window and resets the routing state on the device.
        ws.staged_tokens = _stage_inputs(
            t.hidden_states,
            t.topk_ids,
            t.topk_weights,
            hidden_out=ws.hidden,
            ids_out=ws.ids,
            weights_out=ws.weights,
            bound=bound,
        )
        ws.staged_bound = bound

    def workspace_output(self, workspace: Any) -> torch.Tensor:
        return self._live(workspace).output

    def _launch_plan(self, ws: CakeWorkspace, tw: Any, bound: int) -> list[StageCall]:
        key = (id(tw), bound)
        cached = ws.launch_plans.get(key)
        if cached is not None and cached[1] is tw:
            return cached[0]
        if not isinstance(tw, TransformedCakeW8a8MegaWeights):
            raise MoEEpConfigError(
                f"{KERNEL_NAME}: transformed weights must be TransformedCakeW8a8MegaWeights, "
                f"got {type(tw).__name__}"
            )
        validate_transformed_mega_weights(tw, device=ws.device)
        grids = ws.grids.get(bound)
        if grids is None:
            grids = ws.grids[bound] = stage_grids(bound, ws.device.index)
        calls = build_stage_calls(ws, tw, grids)
        ws.launch_plans[key] = (calls, tw)
        self._plan_keys.setdefault(id(ws), set()).add(key)
        return calls

    def _forget_local_workspace_state(self, workspace: Any) -> None:
        """Drop the launch plans (and their weight references) this instance
        cached on ``workspace``; other layers' entries in the pooled workspace
        stay."""

        keys = self._plan_keys.pop(id(workspace), set())
        if isinstance(workspace, CakeWorkspace):
            for key in keys:
                workspace.launch_plans.pop(key, None)

    def compute(
        self,
        workspace: Any,
        transformed_weights: Any,
        *,
        output: torch.Tensor | None,
    ) -> torch.Tensor:
        """Five launches on the current stream (``publish_push``, ``local_pack``,
        ``fc1``, ``fc2``, ``combine``); no host sync, no host memset, no
        allocation.

        ``publish_push`` zeroes this rank's combine read window of the term
        buffer and resets the routing state before it releases its inbox
        flags. The zero is ordered: the previous forward's remote FC2 pushes
        all landed before this rank's previous ``combine`` passed (it waited
        until every rank's ``fc2`` signalled completion), and a remote FC2 of
        this forward writes a token's window only after that rank's
        ``local_pack`` consumed the token's flag, which this rank stores after
        the zero. Slots ``local_pack`` does not pack (masked ``-1`` routes,
        overflowed routes) then contribute zero. There is no input barrier:
        ``local_pack`` waits on its local flags with an untimed acquire poll
        (a late peer stalls this rank, it never traps). There is no partial
        barrier: the last CTA of every rank's ``fc2`` adds +1 into word 0 of
        every rank's ``fc2_completion`` counter, and each ``combine`` CTA
        waits (untimed, acquire) until its LOCAL counter reached
        ``16 * push_generation[0]``, the generation ``local_pack`` of this
        forward stored; a rank therefore overwrites a peer's inbox slot in
        forward g + 1 only after every rank's ``fc2(g)``, hence every
        ``local_pack(g)``, finished. The flags, the counter and the generation
        words were zeroed once at allocation, followed by the collective
        rendezvous, and the host never writes them (the counter is monotonic;
        its compare is wrap-safe, so no reset is ever needed).

        The ``fc1``, ``fc2`` and ``combine`` bindings launch with
        ``cudaLaunchAttributeProgrammaticStreamSerialization`` (the device
        programs ``griddepcontrol.wait`` before reading the previous stage's
        output), so the edges local_pack -> fc1, fc1 -> fc2 and
        fc2 -> combine are programmatic in eager streams and become
        programmatic graph edges under CUDA-graph capture; the launch count
        and order are unchanged. The ``local_pack`` binding sets its preferred
        shared-memory carveout once per process (a function attribute) before
        its first launch.
        """

        ws = self._live(workspace)
        bound = ws.staged_bound
        if bound is None:
            raise MoEEpConfigError(
                f"{KERNEL_NAME}: compute() requires stage_inputs() on this workspace first"
            )
        n = ws.staged_tokens
        calls = self._launch_plan(ws, transformed_weights, bound)
        with torch.cuda.device(ws.device), ws.use_torch_stream():
            for entry, args in calls:
                entry(*args)
        ws.launches += 1
        if output is None:
            return ws.output[:n]
        output[:n].copy_(ws.output[:n])
        return output[:n]

    def routing_overflow(self, workspace: Any) -> torch.Tensor:
        """Device int32 ``[1]`` counter: nonzero after a forward in which an
        expert received more than 32 routes (that forward's output is not
        valid). ``local_pack`` counts every route at or beyond row 32 of an
        expert here instead of packing it; the next forward's ``publish_push``
        zeroes it. Reading it on the host (``.item()``) is the caller's
        explicit synchronization.
        """

        return self._live(workspace).overflow

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


def routing_overflow(layer: MoEEpMegaLayer) -> torch.Tensor:
    """Device int32 ``[1]`` overflow counter of ``layer``'s most recent forward.

    Nonzero means an expert received more than 32 routes in that forward and
    the forward's output is not valid; the next forward's ``publish_push``
    zeroes it. The tensor is returned without synchronization: reading it on
    the host (``.item()``) is the caller's explicit synchronization. ``layer``
    is the :class:`~flashinfer.moe_ep.MoEEpMegaLayer` that ran the forward
    (through this backend); the counter belongs to the workspace that forward
    used.
    """

    kernel = getattr(layer, "_staged_kernel", None)
    workspace = getattr(layer, "_staged_workspace", None)
    if kernel is None or workspace is None:
        raise RuntimeError(
            f"{KERNEL_NAME}: routing_overflow needs a staged forward on the layer "
            "(run warmup() or forward() first)"
        )
    if not isinstance(kernel, Sm103Mxfp8Mxfp8CakeMegaKernelBackend):
        raise TypeError(
            f"routing_overflow needs a MoEEpMegaLayer running {KERNEL_NAME}, got a "
            f"{type(kernel).__name__} backend"
        )
    return kernel.routing_overflow(workspace)


__all__ = [
    "KERNEL_NAME",
    "STAGE_ARGS",
    "STAGE_ENTRIES",
    "SYMMETRIC_BUFFERS",
    "CakeWorkspace",
    "Sm103Mxfp8Mxfp8CakeMegaKernelBackend",
    "Sm103_Mxfp8_Mxfp8_Bf16_Cake_MegaMoeConfig",
    "TransformedCakeW8a8MegaWeights",
    "build_stage_calls",
    "combine_args",
    "fc1_args",
    "fc2_args",
    "local_pack_args",
    "preprocess_mega_weights",
    "publish_push_args",
    "routing_overflow",
]
