"""SM120 swap-AB CuTeDSL MXFP8 mega-MoE kernel backend.

The fused kernel consumes MXFP8 expert weights in kernel-ready layout (K-major
fp8 weight views + atom-swizzled E8M0 scale factors). ``MoEWeightPack``
supplies canonical bf16 ``w13``/``w2`` by default; ``preprocess_weights()``
quantizes, interleaves (groups of 8), and swizzles them. Pass pre-quantized
MXFP8 weights via ``w13``/``w2`` plus ``w13_scale``/``w2_scale`` to skip
re-quantization.

Requires sm_120/sm_121 (Blackwell-consumer); the kernel tree is
process-exclusive with the sm_100 and sm_90 trees.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from ......config import BootstrapConfig, FleetParams
from ......core.kernel.base import MegaKernelBackend
from ......core.kernel.registry import register_mega_kernel
from ......core.runtime import sm120_mxfp8_cutedsl_runtime_requirements
from ......core.validation.common import (
    validate_mega_arch_sm120,
    validate_mega_fleet_params,
)
from ......weights import MoEWeightPack
from .config import Sm120_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig
from .staging import stage_mega_moe_inputs, validate_sm120_mxfp8_forward_inputs
from .weights import (
    TransformedMegaWeights,
    preprocess_mega_weights,
    validate_transformed_mega_weights,
)

if TYPE_CHECKING:
    from ......tensors import MoEEpTensors


def _resolve_gate_up_clamp(
    config: Sm120_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig,
) -> float | None:
    if config.gate_up_clamp is not None:
        return config.gate_up_clamp
    return config.activation_clamp


@register_mega_kernel("sm120_mxfp8_mxfp8_bf16_cutedsl")
class Sm120Mxfp8CutedslMegaKernelBackend(MegaKernelBackend):
    def __init__(self, config: Sm120_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig) -> None:
        super().__init__(config)
        self._kernel_config: Sm120_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig = config
        if config.in_kernel_fc2_reduce:
            # Verified 2026-08-06 on RTX PRO 6000 (sm_120): the vendored
            # drop's REDG path is broken upstream — the kernel team's own
            # mega_runner crashes with an illegal memory access under
            # --in_kernel_fc2_reduce, and the path is absent from their test
            # scripts. Re-enable after a drop that validates it (VENDOR.md).
            raise NotImplementedError(
                "in_kernel_fc2_reduce is not functional in the current SM120 "
                "kernel drop (upstream REDG path crashes); use the default "
                "explicit topk reduce."
            )
        if _resolve_gate_up_clamp(config) is not None:
            # Verified 2026-08-06 at kernel-source level: the drop's
            # kernel_fc12 stores gate_up_clamp and never reads it (dead
            # plumbing; kernel output is bit-identical with and without the
            # clamp). Silently dropping a DeepSeek-V4 swiglu_limit would be a
            # correctness hazard, so reject until a drop wires it (VENDOR.md).
            raise NotImplementedError(
                "gate_up_clamp/activation_clamp is not functional in the "
                "current SM120 kernel drop (the kernel ignores it); leave it "
                "unset."
            )
        if config.knobs is not None and not isinstance(config.knobs, dict):
            raise ValueError(
                "sm120_mxfp8_mxfp8_bf16_cutedsl supports only knobs=None or an "
                f"explicit knob dict (no autotune/knob-cache yet); got "
                f"{config.knobs!r}"
            )

    @classmethod
    def kernel_name(cls) -> str:
        return "sm120_mxfp8_mxfp8_bf16_cutedsl"

    def runtime_requirements(self, bootstrap: BootstrapConfig) -> frozenset[str]:
        return sm120_mxfp8_cutedsl_runtime_requirements(bootstrap)

    def validate_init(
        self,
        bootstrap: BootstrapConfig,
        fleet_params: FleetParams,
    ) -> None:
        validate_mega_arch_sm120()
        validate_mega_fleet_params(
            fleet_params,
            bootstrap.world_size,
            intermediate_size=self._kernel_config.intermediate_size,
            top_k=self._kernel_config.top_k,
            # The SM120 swap-AB tiles are tail-safe like the sm100 cutedsl
            # kernels; the binding bound is the MXFP8 SF block (32), which is
            # what the runner itself validates (hidden % 32, intermediate
            # halves % 32).
            alignment=32,
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
            kind=self._kernel_config.kind,
            gate_up_clamp=_resolve_gate_up_clamp(self._kernel_config),
            activation_clamp=self._kernel_config.activation_clamp,
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
            kind=self._kernel_config.kind,
            world_size=self.ep_world_size,
            num_experts=fleet_params.num_experts,
        )

    def _allocate_workspace(self, fleet_params: FleetParams) -> Any:
        from ......kernel_src.sm120.swapab_cutedsl_megakernel import (
            get_symm_buffer_for_sm120_mxfp8_mega_moe,
        )

        k = self._kernel_config
        fp = fleet_params
        knobs = k.knobs
        if knobs is None or "mma_tiler_mnk" not in knobs:
            # The drop's shim-default tiler N=128 produces silently wrong
            # output cells at EVERY world size on dense data (verified
            # 2026-08-06 at ws1, 2026-08-07 at ws2 and ws4 on RTX PRO 6000 /
            # RTX 6000D): rel-L2 vs the bf16 dense reference degrades from
            # the ~6.35% MXFP8 band to 8-28% once tokens fill past an N=64
            # tile, with run-to-run magnitude variation (race-like). The
            # drop's own ws4 "bit-exact" check used 1%-sparse test data,
            # which cannot see it. N=64 stays in band everywhere (~23%
            # slower at large batch). See VENDOR.md; an explicit
            # mma_tiler_mnk knob overrides this pin.
            knobs = {**(knobs or {}), "mma_tiler_mnk": (64, 64, 128)}
        return get_symm_buffer_for_sm120_mxfp8_mega_moe(
            fp.num_experts,
            fp.max_tokens_per_rank,
            k.top_k,
            fp.token_hidden_size,
            k.intermediate_size,
            self.ep_rank,
            self.ep_world_size,
            kind=k.kind,
            gate_up_clamp=_resolve_gate_up_clamp(k),
            activation_clamp=k.activation_clamp,
            in_kernel_fc2_reduce=k.in_kernel_fc2_reduce,
            token_back_mode=k.token_back_mode,
            knobs=knobs,
        )

    def validate_forward(
        self,
        t: "MoEEpTensors",
        fleet_params: FleetParams,
        *,
        quantize_input: bool,
    ) -> None:
        validate_sm120_mxfp8_forward_inputs(
            t.hidden_states,
            t.topk_ids,
            t.topk_weights,
            fleet_params,
            top_k=self._kernel_config.top_k,
            quantize_input=quantize_input,
            kind=self._kernel_config.kind,
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
        if quantize_input:
            stage_mega_moe_inputs(
                t.hidden_states,
                t.topk_weights,
                t.topk_ids,
                workspace.x,
                workspace.x_sf,
                workspace.topk_idx,
                workspace.topk_weights,
                kind=self._kernel_config.kind,
            )
        else:
            # Backend talks only to the swapab_cutedsl_megakernel shim
            # (never src/ directly).
            from ......kernel_src.sm120.swapab_cutedsl_megakernel import (
                Mxfp8BlockSize,
                ceil_div,
                round_up,
            )

            hidden = workspace.hidden
            hidden_sf_cols = ceil_div(hidden, Mxfp8BlockSize)
            hidden_sf_cols_padded = round_up(hidden_sf_cols, 4)

            workspace.x[:num_tokens].view(torch.uint8).copy_(
                t.hidden_states[:num_tokens].view(torch.uint8)
            )
            assert t.scales is not None
            workspace.x_sf[:num_tokens].zero_()
            workspace.x_sf[:num_tokens, :hidden_sf_cols].view(torch.uint8).copy_(
                t.scales[:num_tokens, :hidden_sf_cols].view(torch.uint8)
            )
            if t.scales.shape[1] >= hidden_sf_cols_padded:
                workspace.x_sf[
                    :num_tokens, hidden_sf_cols:hidden_sf_cols_padded
                ].zero_()
            workspace.topk_idx[:num_tokens].copy_(t.topk_ids)
            workspace.topk_weights[:num_tokens].copy_(t.topk_weights)
            capacity = workspace.x.shape[0]
            if num_tokens < capacity:
                workspace.topk_idx[num_tokens:capacity].fill_(-1)
        # Live-token memo for compute(output=None); the SM120 tree has no
        # staged-token registry shim, so the count rides on the workspace.
        workspace._staged_tokens = num_tokens

    def compute(
        self,
        workspace: Any,
        transformed_weights: TransformedMegaWeights,
        *,
        output: torch.Tensor | None,
    ) -> torch.Tensor:
        from ......kernel_src.sm120.swapab_cutedsl_megakernel import (
            sm120_mxfp8_mega_moe,
        )

        if output is not None:
            num_tokens = output.shape[0]
        else:
            staged = getattr(workspace, "_staged_tokens", None)
            if staged is None:
                raise ValueError(
                    "compute(output=None) requires stage_inputs() to have "
                    "staged this workspace first"
                )
            num_tokens = staged

        kcfg = self._kernel_config
        view = sm120_mxfp8_mega_moe(
            output,
            transformed_weights[0],
            transformed_weights[1],
            workspace,
            num_tokens=num_tokens,
            gate_up_clamp=_resolve_gate_up_clamp(kcfg),
            activation_clamp=kcfg.activation_clamp,
            fast_math=kcfg.fast_math,
        )
        # output=None -> zero-copy: the reduced result stays in the workspace
        # and the caller consumes the [:n] view under stream ordering (valid
        # until the next launch on this session's buffers).
        return output if output is not None else view

    def _workspace_pool_key(self, fleet_params: FleetParams) -> Any:
        k = self._kernel_config
        import torch

        from ......core.kernel.workspace_pool import knobs_pool_key

        fp = fleet_params
        return (
            "sm120_mxfp8_mxfp8_bf16_cutedsl",
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
            _resolve_gate_up_clamp(k),
            k.in_kernel_fc2_reduce,
            k.token_back_mode,
            knobs_pool_key(k.knobs),
        )
