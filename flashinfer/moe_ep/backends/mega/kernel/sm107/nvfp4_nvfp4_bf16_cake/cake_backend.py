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

# The ``sm107_nvfp4_nvfp4_bf16_cake`` mega backend: the Cake NVFP4 MegaMoE
# route on sm_107a (see cake_plan.py for the four programs of a forward).
#
# Weights take the layout of ``sm107_nvfp4_nvfp4_bf16_cutedsl`` (its
# ``preprocess_mega_weights``: K-major E2M1 codes with FC1's gate and up
# interleaved in 16-row stripes, E4M3 scales in 32x4x4 atoms), which the
# programs read as ``[E_local, rows, K / 2]`` codes and
# ``[E_local, rows / 128, K / 64, 512]`` scale atoms. The workspace holds the
# symmetric buffers the router and the mega program share (peer-mapped
# through NVSHMEM), the process group's flag pad and the rank-local tables,
# pools and counters, all zeroed once: the programs leave every counter as
# they found it. A forward stages the activations (the STAGE program, or a
# copy of NVFP4 codes) and the routes (int32 ids, -1 past the live tokens,
# FP32 weights), then launches the tier's router, mega program and reduce on
# the current stream.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import torch

from ......config import BootstrapConfig, FleetParams
from ......core.kernel.base import MegaKernelBackend
from ......core.kernel.registry import register_mega_kernel
from ......core.runtime import sm107_block_scaled_runtime_requirements
from ......core.validation.common import (
    MoEEpConfigError,
    validate_mega_arch_sm107,
)
from ......weights import MoEWeightPack
from ...sm103.bf16_nvfp4_bf16_cake.cake_symmetric import (
    SymmetricBuffer,
    allocate_flag_pad,
    allocate_symmetric,
    check_self_entry,
)
from ..nvfp4_nvfp4_bf16_cutedsl.staging import validate_sm107_nvfp4_forward_inputs
from ..nvfp4_nvfp4_bf16_cutedsl.weights import (
    TransformedMegaWeights,
    preprocess_mega_weights,
    validate_transformed_mega_weights,
)
from ..validation import validate_input_norm_const, validate_nvfp4_scalars
from . import cake_programs
from .cake_config import Sm107_Nvfp4_Nvfp4_Bf16_Cake_MegaMoeConfig
from .cake_plan import (
    E_PAD,
    EXPERTS_LOCAL,
    GRID_ARGS,
    HIDDEN,
    INTERMEDIATE,
    LAMPORT_SENTINEL,
    MIN_CUDA_DEDUP,
    STAGE,
    TOP_K,
    Tier,
    local_buffers,
    mega_grid,
    scale_multiplier,
    select_tier,
    symmetric_buffers,
    symmetric_fills,
    validate_geometry,
)

if TYPE_CHECKING:
    from ......tensors import MoEEpTensors

_SCALARS = ("fc1_alpha", "fc2_alpha", "fc1_norm_const")


@dataclass
class CakeNvfp4Workspace:
    """The buffers of one tier on one rank."""

    tier: Tier
    device: torch.device
    rank: int
    world_size: int
    symmetric: Dict[str, SymmetricBuffer]
    flags: SymmetricBuffer
    local: Dict[str, torch.Tensor]
    output: (
        torch.Tensor
    )  # [tokens, H] bf16: the reduce's output when the caller passes none
    mega_ctas: int
    ones: torch.Tensor  # the scalars' default, FP32 ones [E_local]
    staged_tokens: Optional[int] = None
    scalars: Dict[str, torch.Tensor] = field(default_factory=dict)

    def destroy(self) -> None:
        for buffer in self.symmetric.values():
            buffer.free()
        self.flags.free()
        self.symmetric = {}
        self.local = {}


def _views(workspace: CakeNvfp4Workspace) -> Dict[str, Any]:
    """Every argument the programs take by name, but the weights, the scalars
    and the per-call tensors: the local tables and counters, the pool and FC1
    output under each name the mega program binds them (one TMA view each),
    the symmetric buffers with their peer tables and the process group."""
    tier, local = workspace.tier, workspace.local
    values: Dict[str, Any] = dict(local)
    pool, fc1_out = local["pool"], local["fc1_out"]
    pool_sf, fc1_out_sf = local["pool_sf"], local["fc1_out_sf"]
    values.update(
        pool_b=pool,
        pool_b16=pool,
        pool_bh=pool,
        pool_data=pool,
        fc1_out_st=fc1_out,
        fc1_out_ld=fc1_out,
        fc1_out_ld16=fc1_out,
        fc1_out_ldh=fc1_out,
        pool_sfb=pool_sf.view(torch.uint16),
        pool_sfb1=pool_sf.view(torch.uint16),
        fc1_out_sfb=fc1_out_sf.view(torch.uint16),
        fc1_out_sfb1=fc1_out_sf.view(torch.uint16),
        pool_capacity=tier.pool_rows,
        num_ctas=workspace.mega_ctas,
        e_pad=E_PAD,
        block_words=tier.block_words,
        size_counter_words=(E_PAD + 1) if tier.multi else 0,
        push_ctas=tier.push_ctas,
        dedup_cap=tier.tokens,
        mega_pg_world=workspace.world_size,
        mega_pg_rank=workspace.rank,
        mega_pg_flags=workspace.flags.peers,
    )
    for name, buffer in workspace.symmetric.items():
        values[name] = buffer.local
        values[f"{name}_peers"] = buffer.peers
    return values


def _weight_views(transformed: TransformedMegaWeights) -> Dict[str, torch.Tensor]:
    """The programs' weight arguments: the K-major code storage the
    transformed views stand on, and the scale atoms as 16-bit words."""
    (w1, sf1), (w2, sf2) = transformed
    experts = w1.shape[0]
    return {
        "W1": w1.permute(0, 2, 1).view(torch.uint8),
        "S1": sf1.view(torch.uint8)
        .view(experts, 2 * INTERMEDIATE // 128, HIDDEN // 64, 512)
        .view(torch.uint16),
        "W2": w2.permute(0, 2, 1).view(torch.uint8),
        "S2": sf2.view(torch.uint8)
        .view(experts, HIDDEN // 128, INTERMEDIATE // 64, 512)
        .view(torch.uint16),
    }


def _call(handle: cake_programs.ProgramHandle, values: Dict[str, Any], grid) -> None:
    missing = [name for name in handle.kernel_args if name not in values]
    if missing:
        raise RuntimeError(
            f"{handle.identifier}: the host supplies no value for {missing}"
        )
    handle.launch(*(values[name] for name in handle.kernel_args), *grid)


@register_mega_kernel("sm107_nvfp4_nvfp4_bf16_cake")
class Sm107Nvfp4CakeMegaKernelBackend(MegaKernelBackend):
    """Rubin NVFP4 x NVFP4 MegaMoE from generated Cake programs."""

    supports_output_view = True

    def __init__(self, config: Sm107_Nvfp4_Nvfp4_Bf16_Cake_MegaMoeConfig) -> None:
        super().__init__(config)
        self._kernel_config = config

    @classmethod
    def create(cls, config):
        return cls(config)

    @classmethod
    def kernel_name(cls) -> str:
        return "sm107_nvfp4_nvfp4_bf16_cake"

    def runtime_requirements(self, bootstrap: BootstrapConfig):
        return sm107_block_scaled_runtime_requirements(bootstrap)

    def validate_init(
        self, bootstrap: BootstrapConfig, fleet_params: FleetParams
    ) -> None:
        validate_mega_arch_sm107()
        self._ensure_ep_bootstrap(bootstrap)
        k = self._kernel_config
        validate_input_norm_const(k.input_norm_const)
        try:
            validate_geometry(
                hidden=fleet_params.token_hidden_size,
                intermediate=k.intermediate_size,
                num_experts=fleet_params.num_experts,
                top_k=k.top_k,
                world_size=self.ep_world_size,
            )
            tier = select_tier(fleet_params.max_tokens_per_rank)
        except ValueError as error:
            raise MoEEpConfigError(str(error)) from None
        if tier.dedup:
            from flashinfer.jit.cpp_ext import (
                get_cuda_version,
                is_cuda_version_at_least,
            )

            if not is_cuda_version_at_least(MIN_CUDA_DEDUP):
                raise MoEEpConfigError(
                    f"the Cake NVFP4 MegaMoE programs of {tier.tokens} tokens per rank "
                    f"read %locality_domain_id, which needs CUDA {MIN_CUDA_DEDUP} or "
                    f"newer to compile; nvcc is {get_cuda_version()}"
                )
        if torch.cuda.is_available() and not cake_programs.programs_available():
            raise MoEEpConfigError(
                f"{cake_programs.LOADER_MODULE} does not publish every Cake NVFP4 "
                "MegaMoE program for sm_107a"
            )
        validate_nvfp4_scalars(
            self._scalar_values(),
            num_local_experts=EXPERTS_LOCAL,
            device=torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cuda"),
        )

    def preprocess_weights(
        self, weights: MoEWeightPack, fleet_params: FleetParams
    ) -> TransformedMegaWeights:
        return preprocess_mega_weights(
            weights,
            intermediate_size=self._kernel_config.intermediate_size,
            hidden_size=fleet_params.token_hidden_size,
        )

    def validate_transformed_weights(
        self,
        transformed_weights: TransformedMegaWeights,
        bootstrap: BootstrapConfig,
        fleet_params: FleetParams,
    ) -> None:
        validate_transformed_mega_weights(
            transformed_weights,
            intermediate_size=self._kernel_config.intermediate_size,
            hidden_size=fleet_params.token_hidden_size,
            world_size=self.ep_world_size,
            num_experts=fleet_params.num_experts,
        )

    def _allocate_workspace(self, fleet_params: FleetParams) -> CakeNvfp4Workspace:
        tier = select_tier(fleet_params.max_tokens_per_rank)
        device = torch.device("cuda", torch.cuda.current_device())
        rank, world = self.ep_rank, self.ep_world_size
        group = self.ep_comm_group if world > 1 else None
        symmetric = {}
        for name, (shape, dtype) in symmetric_buffers(tier).items():
            buffer = allocate_symmetric(
                shape, dtype, device=device, world_size=world, rank=rank
            )
            check_self_entry(buffer, rank=rank, name=name)
            symmetric[name] = buffer
        for name in symmetric_fills(tier):
            symmetric[name].local.view(torch.int32).fill_(LAMPORT_SENTINEL)
        flags = allocate_flag_pad(
            device=device, group=group, world_size=world, rank=rank
        )
        check_self_entry(flags, rank=rank, name="mega_pg_flags")
        local = {
            name: torch.zeros(shape, dtype=dtype, device=device)
            for name, (shape, dtype) in local_buffers(tier).items()
        }
        local["topk_indices"].fill_(-1)
        # The mega grid is the same on every rank: the smallest device's.
        sm_count = torch.cuda.get_device_properties(device).multi_processor_count
        if world > 1:
            import torch.distributed as dist

            # Every rank's buffers are armed before any rank's first router
            # writes into them: the gather below is the first collective.
            torch.cuda.synchronize(device)
            counts = [None] * world
            dist.all_gather_object(counts, sm_count, group=group)
            sm_count = min(int(c) for c in counts)
        torch.cuda.synchronize(device)
        return CakeNvfp4Workspace(
            tier=tier,
            device=device,
            rank=rank,
            world_size=world,
            symmetric=symmetric,
            flags=flags,
            local=local,
            output=torch.empty(
                (tier.tokens, HIDDEN), dtype=torch.bfloat16, device=device
            ),
            mega_ctas=mega_grid(sm_count, tier),
            ones=torch.ones(EXPERTS_LOCAL, dtype=torch.float32, device=device),
        )

    def workspace_output(self, workspace: CakeNvfp4Workspace) -> torch.Tensor:
        return workspace.output

    def validate_forward(
        self,
        t: "MoEEpTensors",
        fleet_params: FleetParams,
        *,
        quantize_input: bool,
    ) -> None:
        validate_nvfp4_scalars(
            self._scalar_values(t),
            num_local_experts=EXPERTS_LOCAL,
            device=t.hidden_states.device,
        )
        validate_sm107_nvfp4_forward_inputs(
            t.hidden_states,
            t.topk_ids,
            t.topk_weights,
            fleet_params,
            top_k=TOP_K,
            quantize_input=quantize_input,
            scales=t.scales,
        )

    def _scalar_values(self, t=None) -> Dict[str, Optional[torch.Tensor]]:
        """Per-call scalar tensors ahead of the configuration's."""
        values = {}
        for name in _SCALARS:
            override = getattr(t, name, None)
            values[name] = (
                override if override is not None else getattr(self._kernel_config, name)
            )
        return values

    def stage_inputs(
        self, t: "MoEEpTensors", workspace: CakeNvfp4Workspace, *, quantize_input: bool
    ) -> None:
        """Stage the activations into this rank's symmetric source rows and the
        routes into the router's int32 / FP32 tables (-1 past the live tokens)."""
        tokens = int(t.hidden_states.shape[0])
        cap = workspace.tier.tokens
        if tokens > cap:
            raise MoEEpConfigError(
                f"{tokens} tokens exceed the workspace's tier of {cap} tokens per rank"
            )
        activation = workspace.symmetric["activation"].local
        activation_sf = workspace.symmetric["activation_sf"].local
        if tokens > 0:
            if quantize_input:
                # The STAGE program also copies each token's routes into the
                # router's tables (one launch instead of three); it reads them
                # as int32 / FP32 words.
                handle = cake_programs.programs().handle(STAGE, workspace.device)
                norm = float(self._kernel_config.input_norm_const)
                x = t.hidden_states.contiguous()
                if x.data_ptr() % 16:
                    # STAGE reads each row in 16-byte vectors: a view that
                    # starts off a 16-byte boundary is copied into storage
                    # that starts on one.
                    x = x.clone()
                values = {
                    "x": x.view(torch.uint32),
                    "data": activation.view(torch.uint32),
                    "sf": activation_sf,
                    "route_ids": t.topk_ids.to(torch.int32)
                    .contiguous()
                    .view(torch.uint32),
                    "route_scores": t.topk_weights.to(torch.float32)
                    .contiguous()
                    .view(torch.uint32),
                    "table_ids": workspace.local["topk_indices"].view(torch.uint32),
                    "table_scores": workspace.local["topk_scores"].view(torch.uint32),
                    "norm": norm,
                    "mult": scale_multiplier(norm),
                }
                import tvm_ffi

                with tvm_ffi.use_torch_stream():
                    _call(handle, values, (tokens, 1, 1))
            else:
                activation[:tokens].copy_(t.hidden_states.view(torch.uint8))
                columns = t.scales.shape[1]
                activation_sf[:tokens].zero_()
                activation_sf[:tokens, :columns].copy_(t.scales.view(torch.uint8))
                workspace.local["topk_indices"][:tokens].copy_(t.topk_ids)
                workspace.local["topk_scores"][:tokens].copy_(t.topk_weights)
        if workspace.staged_tokens is None or workspace.staged_tokens > tokens:
            workspace.local["topk_indices"][tokens:].fill_(-1)
        workspace.staged_tokens = tokens
        workspace.scalars = {
            name: (value if value is not None else workspace.ones)
            for name, value in self._scalar_values(t).items()
        }

    def compute(
        self,
        workspace: CakeNvfp4Workspace,
        transformed_weights: TransformedMegaWeights,
        *,
        output: torch.Tensor | None,
    ) -> torch.Tensor:
        tokens = workspace.staged_tokens
        if tokens is None:
            raise RuntimeError(
                "compute() called before stage_inputs(); no token count is staged."
            )
        tier, device = workspace.tier, workspace.device
        # The reduce stores whole rows in 16-byte vectors. A caller output it
        # cannot store into (strided, or starting off a 16-byte boundary) takes
        # a copy of the workspace output, decided before any program launches.
        direct = output is None or (
            output.is_contiguous() and output.data_ptr() % 16 == 0
        )
        out = output if direct and output is not None else workspace.output[:tokens]
        values = _views(workspace)
        values.update(_weight_views(transformed_weights))
        values.update(
            fc1_alpha=workspace.scalars["fc1_alpha"].contiguous().float(),
            fc2_alpha=workspace.scalars["fc2_alpha"].contiguous().float(),
            norm_const=workspace.scalars["fc1_norm_const"].contiguous().float(),
        )
        table = cake_programs.programs()
        router = table.handle(tier.router, device)
        mega = table.handle(tier.mega, device)
        reduce = table.handle(tier.reduce, device)
        import tvm_ffi

        with tvm_ffi.use_torch_stream():
            _call(router, values, (tier.router_ctas, 1, 1))
            _call(mega, values, (workspace.mega_ctas, 1, 1))
            if tokens > 0:
                _call(
                    reduce,
                    {
                        "slot_rows": workspace.symmetric["pre_reduced"]
                        .local.view(-1)
                        .view(torch.uint32),
                        "topk_idx": workspace.local["topk_indices"],
                        "out": out.view(torch.uint32),
                        "world": workspace.world_size,
                    },
                    (tokens, 1, 1),
                )
        if not direct:
            output[:tokens].copy_(out)
            return output
        return out

    def _workspace_pool_key(self, fleet_params: FleetParams) -> Tuple[Any, ...]:
        return (
            "sm107_nvfp4_nvfp4_bf16_cake",
            torch.cuda.current_device(),
            self.ep_rank,
            self.ep_world_size,
            id(self._ep_comm_group),
            select_tier(fleet_params.max_tokens_per_rank).tokens,
        )

    def _forget_workspace_state(self, workspace: CakeNvfp4Workspace) -> None:
        workspace.staged_tokens = None


__all__ = ["CakeNvfp4Workspace", "Sm107Nvfp4CakeMegaKernelBackend", "GRID_ARGS"]
