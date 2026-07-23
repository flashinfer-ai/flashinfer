"""SM90 (Hopper) pull-style FP8 mega-MoE kernel backend.

The fused kernel consumes FP8 expert weights in kernel-ready layout (K-major
fp8 weights + per-mode scale planes + the fp8 dequant-scale vectors).
``MoEWeightPack`` supplies canonical bf16 ``w13``/``w2``;
``preprocess_weights()`` quantizes them per ``fp8_scale_mode`` (per-tensor
scalars or DeepGEMM-style 128x128 block scales).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from .....config import BootstrapConfig, FleetParams
from .....core.kernel.base import MegaKernelBackend
from .....core.kernel.registry import register_mega_kernel
from .....core.runtime import sm90_pull_fp8_runtime_requirements
from .....core.validation.common import (
    validate_mega_arch_sm90,
    validate_mega_fleet_params,
)
from .....weights import MoEWeightPack
from .config import Sm90PullFp8MegaMoeConfig
from .staging import (
    stage_mega_moe_inputs,
    staged_tokens,
    validate_sm90_fp8_forward_inputs,
)
from .weights import (
    TransformedMegaWeights,
    preprocess_mega_weights,
    validate_transformed_mega_weights,
)

if TYPE_CHECKING:
    from .....tensors import MoEEpTensors


def _resolve_gate_up_clamp(config: Sm90PullFp8MegaMoeConfig) -> float | None:
    if config.gate_up_clamp is not None:
        return config.gate_up_clamp
    return config.activation_clamp


@register_mega_kernel("sm90_pull_fp8")
class Sm90PullFp8MegaKernelBackend(MegaKernelBackend):
    def __init__(self, config: Sm90PullFp8MegaMoeConfig) -> None:
        super().__init__(config)
        self._kernel_config: Sm90PullFp8MegaMoeConfig = config

    @classmethod
    def kernel_name(cls) -> str:
        return "sm90_pull_fp8"

    def runtime_requirements(self, bootstrap: BootstrapConfig) -> frozenset[str]:
        return sm90_pull_fp8_runtime_requirements(bootstrap)

    def validate_init(
        self,
        bootstrap: BootstrapConfig,
        fleet_params: FleetParams,
    ) -> None:
        validate_mega_arch_sm90()
        validate_mega_fleet_params(
            fleet_params,
            bootstrap.world_size,
            intermediate_size=self._kernel_config.intermediate_size,
            top_k=self._kernel_config.top_k,
            # cutedsl tiles are tail-safe (ceil-div + predicated epilogue);
            # the per-tensor bound is TMA row alignment + SF-word packing (64).
            # Blockwise scales are 128x128 (Fp8WeightScaleBlock*), so blockwise
            # sessions need the full 128.
            alignment=(
                128 if self._kernel_config.fp8_scale_mode == "blockwise" else 64
            ),
        )

    def preprocess_weights(
        self,
        weights: MoEWeightPack,
        fleet_params: FleetParams,
    ) -> TransformedMegaWeights:
        k = self._kernel_config
        return preprocess_mega_weights(
            weights,
            intermediate_size=k.intermediate_size,
            hidden_size=fleet_params.token_hidden_size,
            kind=k.kind,
            fp8_scale_mode=k.fp8_scale_mode,
            fc1_activation_dequant_scale=k.fc1_activation_dequant_scale,
            fc2_activation_dequant_scale=k.fc2_activation_dequant_scale,
        )

    def validate_transformed_weights(
        self,
        transformed_weights: TransformedMegaWeights,
        bootstrap: BootstrapConfig,
        fleet_params: FleetParams,
    ) -> None:
        k = self._kernel_config
        validate_transformed_mega_weights(
            transformed_weights,
            intermediate_size=k.intermediate_size,
            hidden_size=fleet_params.token_hidden_size,
            kind=k.kind,
            fp8_scale_mode=k.fp8_scale_mode,
            world_size=self.ep_world_size,
            num_experts=fleet_params.num_experts,
        )

    def _allocate_workspace(self, fleet_params: FleetParams) -> Any:
        # Backend talks only to the pull_style_cutedsl_megakernel shim (never
        # src/ directly).
        from .....kernel_src.sm90.pull_style_cutedsl_megakernel import (
            get_symm_buffer_for_hopper_fp8_mega_moe,
        )

        k = self._kernel_config
        fp = fleet_params
        return get_symm_buffer_for_hopper_fp8_mega_moe(
            fp.num_experts,
            fp.max_tokens_per_rank,
            k.top_k,
            fp.token_hidden_size,
            k.intermediate_size,
            self.ep_rank,
            self.ep_world_size,
            kind=k.kind,
            fp8_scale_mode=k.fp8_scale_mode,
            fp8_accum_mode=k.fp8_accum_mode,
            swap_ab=k.swap_ab,
            mma_tiler_mnk=k.mma_tiler_mnk,
            load_balance_mode=k.load_balance_mode,
            gate_up_clamp=_resolve_gate_up_clamp(k),
            activation_clamp=k.activation_clamp,
            in_kernel_fc2_reduce=k.in_kernel_fc2_reduce,
            token_back_by_dispatch=k.token_back_by_dispatch,
        )

    def validate_forward(
        self,
        t: "MoEEpTensors",
        fleet_params: FleetParams,
        *,
        quantize_input: bool,
    ) -> None:
        validate_sm90_fp8_forward_inputs(
            t.hidden_states,
            t.topk_ids,
            t.topk_weights,
            fleet_params,
            top_k=self._kernel_config.top_k,
            quantize_input=quantize_input,
            kind=self._kernel_config.kind,
            fp8_scale_mode=self._kernel_config.fp8_scale_mode,
            scales=t.scales,
        )

    def stage_inputs(
        self,
        t: "MoEEpTensors",
        workspace: Any,
        *,
        quantize_input: bool,
    ) -> None:
        num_tokens = t.hidden_states.shape[0]
        k = self._kernel_config
        if quantize_input:
            stage_mega_moe_inputs(
                t.hidden_states,
                t.topk_weights,
                t.topk_ids,
                workspace.x,
                workspace.x_sf,
                workspace.topk_idx,
                workspace.topk_weights,
                kind=k.kind,
                fp8_scale_mode=k.fp8_scale_mode,
                fc1_activation_dequant_scale=k.fc1_activation_dequant_scale,
            )
        else:
            # Pre-staged fp8 + scales: byte-copy into the symmetric buffers.
            # Backend talks only to the pull_style_cutedsl_megakernel shim.
            if k.fp8_scale_mode == "blockwise":
                from .....kernel_src.sm90.pull_style_cutedsl_megakernel import (
                    Fp8BlockScaleK,
                )

                sf_cols = workspace.hidden // Fp8BlockScaleK
                assert t.scales is not None
                workspace.x_sf[:num_tokens].zero_()
                workspace.x_sf[:num_tokens, :sf_cols].copy_(
                    t.scales[:num_tokens, :sf_cols].to(torch.float32)
                )
            else:
                from .....kernel_src.sm90.pull_style_cutedsl_megakernel import (
                    Fp8E8M0SfVecSize,
                    ceil_div,
                )

                sf_cols = ceil_div(workspace.hidden, Fp8E8M0SfVecSize)
                assert t.scales is not None
                # Full-row zero first: the mult-of-4 pad SF bytes ride the
                # dispatch LDG.32 words and must be defined.
                workspace.x_sf[:num_tokens].view(torch.uint8).zero_()
                workspace.x_sf[:num_tokens, :sf_cols].view(torch.uint8).copy_(
                    t.scales[:num_tokens, :sf_cols].view(torch.uint8)
                )

            workspace.x[:num_tokens].view(torch.uint8).copy_(
                t.hidden_states[:num_tokens].view(torch.uint8)
            )
            workspace.topk_idx[:num_tokens].copy_(t.topk_ids)
            workspace.topk_weights[:num_tokens].copy_(t.topk_weights)
            capacity = workspace.x.shape[0]
            if num_tokens < capacity:
                workspace.topk_idx[num_tokens:capacity].fill_(-1)
            from .staging import _note_staged_tokens

            _note_staged_tokens(workspace.topk_idx, num_tokens)

    def compute(
        self,
        workspace: Any,
        transformed_weights: TransformedMegaWeights,
        *,
        output: torch.Tensor | None,
    ) -> torch.Tensor:
        # Backend talks only to the pull_style_cutedsl_megakernel shim.
        from .....kernel_src.sm90.pull_style_cutedsl_megakernel import (
            hopper_fp8_mega_moe,
        )

        if output is not None:
            num_tokens = output.shape[0]
        else:
            staged = staged_tokens(workspace.topk_idx)
            if staged is None:
                raise ValueError(
                    "compute(output=None) requires stage_inputs() to have "
                    "staged this workspace first"
                )
            num_tokens = staged

        kcfg = self._kernel_config
        view = hopper_fp8_mega_moe(
            output,
            transformed_weights[0],
            transformed_weights[1],
            workspace,
            num_tokens=num_tokens,
            gate_up_clamp=_resolve_gate_up_clamp(kcfg),
            activation_clamp=kcfg.activation_clamp,
            fast_math=kcfg.fast_math,
        )
        # output=None -> zero-copy: the kernel's reduced result stays in the
        # workspace and the caller consumes the [:n] view under stream
        # ordering (valid until the next launch on this session's buffers).
        return output if output is not None else view

    def _workspace_pool_key(self, fleet_params: FleetParams) -> Any:
        import torch

        k = self._kernel_config
        fp = fleet_params
        return (
            "sm90_pull_fp8",
            torch.cuda.current_device(),
            self.ep_rank,
            self.ep_world_size,
            id(self._ep_comm_group),
            fp.num_experts,
            fp.max_tokens_per_rank,
            k.top_k,
            fp.token_hidden_size,
            k.intermediate_size,
            k.kind,
            k.fp8_scale_mode,
            k.fp8_accum_mode,
            k.swap_ab,
            k.mma_tiler_mnk,
            k.load_balance_mode,
            _resolve_gate_up_clamp(k),
            k.in_kernel_fc2_reduce,
            k.token_back_by_dispatch,
        )
