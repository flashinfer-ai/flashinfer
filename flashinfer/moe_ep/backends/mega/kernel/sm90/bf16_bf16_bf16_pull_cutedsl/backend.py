"""SM90 (Hopper) pull-style BF16 mega-MoE kernel backend.

Native BF16 twin of ``fp8_fp8_bf16_pull_cutedsl``: the fused CuTeDSL kernel
pulls bf16 tokens over NVSHMEM, runs BF16 WGMMA FC1/FC2 with fp32
accumulation and returns bf16 rows.  ``MoEWeightPack`` supplies canonical
bf16 ``w13``/``w2``; ``preprocess_weights()`` only re-lays them out (gate/up
interleave + K-major), never quantizes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from ......config import BootstrapConfig, FleetParams
from ......core.kernel.base import MegaKernelBackend
from ......core.kernel.registry import register_mega_kernel
from ......core.runtime import sm90_pull_bf16_runtime_requirements
from ......core.validation.common import (
    validate_mega_arch_sm90,
    validate_mega_fleet_params,
)
from ......weights import MoEWeightPack
from .config import Sm90_Bf16_Bf16_Bf16_PullCutedsl_MegaMoeConfig
from .staging import (
    stage_mega_moe_inputs,
    staged_tokens,
    validate_sm90_bf16_forward_inputs,
)
from .weights import (
    TransformedMegaWeights,
    preprocess_mega_weights,
    validate_transformed_mega_weights,
)

if TYPE_CHECKING:
    from ......tensors import MoEEpTensors

_NAME = "sm90_bf16_bf16_bf16_pull_cutedsl"

# BF16 shape contract, mirroring the shim config's check: hidden a multiple of
# the fc2 N tile (drop-harness contract, moe_hopper_bf16/run_mega_tests.sh);
# intermediate a multiple of the bf16 TMA K atom (64 two-byte elements per
# 128 B swizzle row).
_HIDDEN_ALIGN = 256
_INTERMEDIATE_ALIGN = 64


def _resolve_gate_up_clamp(
    config: Sm90_Bf16_Bf16_Bf16_PullCutedsl_MegaMoeConfig,
) -> float | None:
    if config.gate_up_clamp is not None:
        return config.gate_up_clamp
    return config.activation_clamp


@register_mega_kernel(_NAME)
class Sm90PullBf16MegaKernelBackend(MegaKernelBackend):
    def __init__(self, config: Sm90_Bf16_Bf16_Bf16_PullCutedsl_MegaMoeConfig) -> None:
        super().__init__(config)
        self._kernel_config: Sm90_Bf16_Bf16_Bf16_PullCutedsl_MegaMoeConfig = config
        if config.knobs is not None:
            if not (isinstance(config.knobs, dict) or config.knobs == "auto"):
                raise ValueError(
                    f"knobs must be None, a knob dict, or 'auto'; got {config.knobs!r}"
                )
            if any(
                v is not None
                for v in (
                    config.swap_ab,
                    config.pingpong,
                    config.mma_tiler_mnk,
                    config.cluster_shape_mnk,
                )
            ):
                raise ValueError(
                    "knobs= is mutually exclusive with the explicit geometry "
                    "fields (swap_ab / pingpong / mma_tiler_mnk / "
                    "cluster_shape_mnk)"
                )
        # knobs="auto": tune at the first compute() (weights + staged inputs
        # exist there), then keep the winner for the session.
        self._autotune_pending = config.knobs == "auto"
        if self._autotune_pending:
            import warnings

            warnings.warn(
                "knobs='auto' runs a COLLECTIVE compile+timing sweep at the "
                "first forward — never use it inside a serving engine. Tune "
                "offline instead (python -m flashinfer.moe_ep.tune); winners "
                "persist in the knob cache and knobs=None then resolves them "
                "with a pure lookup.",
                UserWarning,
                stacklevel=3,
            )

    @classmethod
    def kernel_name(cls) -> str:
        return _NAME

    def runtime_requirements(self, bootstrap: BootstrapConfig) -> frozenset[str]:
        return sm90_pull_bf16_runtime_requirements(bootstrap)

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
            # Two different bounds (same as the shim): intermediate only
            # needs the 64-element TMA K atom, hidden the 256-wide fc2 N tile.
            alignment=_INTERMEDIATE_ALIGN,
            hidden_alignment=_HIDDEN_ALIGN,
        )

    def preprocess_weights(
        self,
        weights: MoEWeightPack,
        fleet_params: FleetParams,
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

    def _allocate_workspace(self, fleet_params: FleetParams) -> Any:
        # Backend talks only to the pull_style_cutedsl_megakernel shim (never
        # src/ directly).
        from ......kernel_src.sm90.pull_style_cutedsl_megakernel import (
            get_symm_buffer_for_hopper_bf16_mega_moe,
        )

        k = self._kernel_config
        fp = fleet_params
        return get_symm_buffer_for_hopper_bf16_mega_moe(
            fp.num_experts,
            fp.max_tokens_per_rank,
            k.top_k,
            fp.token_hidden_size,
            k.intermediate_size,
            self.ep_rank,
            self.ep_world_size,
            knobs=k.knobs if isinstance(k.knobs, dict) else None,
            swap_ab=k.swap_ab,
            pingpong=k.pingpong,
            mma_tiler_mnk=k.mma_tiler_mnk,
            cluster_shape_mnk=k.cluster_shape_mnk,
            load_balance_mode=k.load_balance_mode,
            gate_up_clamp=_resolve_gate_up_clamp(k),
            activation_clamp=k.activation_clamp,
            in_kernel_fc2_reduce=k.enable_in_kernel_fc2_reduce,
            token_back_by_dispatch=k.token_back_by_dispatch,
            token_back_mode=k.token_back_mode,
            active_dispatch_warps=k.active_dispatch_warps,
            fc1_store_offload=k.fc1_store_offload,
            fc1_early_done_publish=k.fc1_early_done_publish,
            fold_producer_warps=k.fold_producer_warps,
            generate_c=k.generate_c,
            tail_split_pairs=k.tail_split_pairs,
        )

    def validate_forward(
        self,
        t: "MoEEpTensors",
        fleet_params: FleetParams,
        *,
        quantize_input: bool,
    ) -> None:
        validate_sm90_bf16_forward_inputs(
            t.hidden_states,
            t.topk_ids,
            t.topk_weights,
            fleet_params,
            top_k=self._kernel_config.top_k,
            quantize_input=quantize_input,
            scales=t.scales,
        )

    def stage_inputs(
        self,
        t: "MoEEpTensors",
        workspace: Any,
        *,
        quantize_input: bool,
    ) -> None:
        # The activation is bf16 on both paths (nothing to quantize), so
        # quantize_input only changes the validation above.
        stage_mega_moe_inputs(
            t.hidden_states,
            t.topk_weights,
            t.topk_ids,
            workspace.x,
            workspace.topk_idx,
            workspace.topk_weights,
        )

    def compute(
        self,
        workspace: Any,
        transformed_weights: TransformedMegaWeights,
        *,
        output: torch.Tensor | None,
    ) -> torch.Tensor:
        # Backend talks only to the pull_style_cutedsl_megakernel shim.
        from ......kernel_src.sm90.pull_style_cutedsl_megakernel import (
            hopper_bf16_mega_moe,
        )

        if output is not None:
            num_tokens = output.shape[0]
        else:
            if self._autotune_pending:
                raise ValueError(
                    "compute(output=None) is incompatible with knobs='auto' "
                    "(the autotune sweep needs a caller output buffer)"
                )
            staged = staged_tokens(workspace.topk_idx)
            if staged is None:
                raise ValueError(
                    "compute(output=None) requires stage_inputs() to have "
                    "staged this workspace first"
                )
            num_tokens = staged

        kcfg = self._kernel_config
        if self._autotune_pending:
            # COLLECTIVE: every EP rank reaches this first compute() together,
            # so the candidate sweep stays in lockstep (see shim/autotune_bf16.py).
            from ......kernel_src.sm90.pull_style_cutedsl_megakernel import (
                autotune_hopper_bf16_mega_moe,
            )

            autotune_hopper_bf16_mega_moe(
                output,
                transformed_weights[0],
                transformed_weights[1],
                workspace,
                num_tokens=num_tokens,
                gate_up_clamp=_resolve_gate_up_clamp(kcfg),
                activation_clamp=kcfg.activation_clamp,
            )
            # Cleared only on success: if the collective tune raises, a retried
            # compute() re-attempts it (all ranks fail together, so lockstep
            # holds).
            self._autotune_pending = False
        view = hopper_bf16_mega_moe(
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
        k = self._kernel_config
        if k.knobs == "auto":
            # Autotune retunes (and recompiles) the workspace's shared
            # frontend at first compute; give each session its own buffer.
            return None
        import torch

        from ......core.kernel.workspace_pool import knobs_pool_key

        fp = fleet_params
        return (
            _NAME,
            torch.cuda.current_device(),
            self.ep_rank,
            self.ep_world_size,
            id(self._ep_comm_group),
            fp.num_experts,
            fp.max_tokens_per_rank,
            k.top_k,
            fp.token_hidden_size,
            k.intermediate_size,
            k.swap_ab,
            k.pingpong,
            k.mma_tiler_mnk,
            k.cluster_shape_mnk,
            k.load_balance_mode,
            _resolve_gate_up_clamp(k),
            k.enable_in_kernel_fc2_reduce,
            k.token_back_by_dispatch,
            k.token_back_mode,
            k.active_dispatch_warps,
            k.fc1_store_offload,
            k.fc1_early_done_publish,
            k.fold_producer_warps,
            k.generate_c,
            k.tail_split_pairs,
            knobs_pool_key(k.knobs),
        )
