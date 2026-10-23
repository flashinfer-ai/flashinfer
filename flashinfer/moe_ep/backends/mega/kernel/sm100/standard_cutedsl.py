"""Opt-in standard Blackwell inference from the current provider export."""

from __future__ import annotations

import torch

from .....core.kernel.base import MegaKernelBackend
from .....core.kernel.registry import register_mega_kernel
from .....core.runtime import (
    mxfp8_cutedsl_runtime_requirements,
    nvfp4_cutedsl_runtime_requirements,
)
from .....core.validation.common import (
    MoEEpConfigError,
    validate_mega_arch,
    validate_mega_fleet_params,
)
from .....weights import PrequantizedMoEWeights


class _StandardBlockScaledBackend(MegaKernelBackend):
    supports_output_view = True
    quant_kind: str | None = None

    def __init__(self, config):
        super().__init__(config)
        if getattr(config, "kernel_variant", "inference") != "inference":
            raise MoEEpConfigError(
                "The standard backend requires kernel_variant='inference'."
            )
        if config.knobs == "auto":
            raise MoEEpConfigError(
                "The standard backend requires fixed or default knobs."
            )
        if config.knobs is not None and not isinstance(config.knobs, dict):
            raise MoEEpConfigError("knobs must be a dictionary or None.")
        if getattr(config, "swiglu_alpha", None) is not None:
            raise MoEEpConfigError(
                "Parameterized SwiGLU is not in this provider snapshot."
            )
        if not getattr(config, "do_finalize", True):
            raise MoEEpConfigError(
                "Deferred finalization is not in this provider snapshot."
            )
        if (
            getattr(config, "activation", "swiglu") == "situ"
            and config.situ_linear_beta is None
        ):
            raise MoEEpConfigError("This provider snapshot requires both SiTU bounds.")
        if any(
            getattr(config, name, None) is not None
            for name in ("mma_tiler_mnk", "cluster_shape_mn", "schedule_policy")
        ):
            raise MoEEpConfigError("Use knobs for standard inference geometry.")

    def _kind(self):
        return self.quant_kind or self._config.kind

    def runtime_requirements(self, bootstrap):
        require = (
            nvfp4_cutedsl_runtime_requirements
            if self._kind() == "nvfp4"
            else mxfp8_cutedsl_runtime_requirements
        )
        return require(bootstrap)

    def validate_init(self, bootstrap, fleet_params):
        validate_mega_arch()
        self._ensure_ep_bootstrap(bootstrap)
        validate_mega_fleet_params(
            fleet_params,
            bootstrap.world_size,
            intermediate_size=self._config.intermediate_size,
            top_k=self._config.top_k,
            alignment=64,
        )
        self._workspace_config(fleet_params)

    def _workspace_config(self, fleet):
        from .....kernel_src.sm100.cutedsl_megamoe import default_knobs
        from .....kernel_src.sm100.next_cutedsl_megamoe import Sm100BlockScaledMoeConfig

        k = self._config
        kind = self._kind()
        wire = getattr(k, "combine_dtype", "bf16")
        knobs = default_knobs(
            fleet.max_tokens_per_rank,
            dtype="nvfp4" if kind == "nvfp4" else "mxfp8",
            combine_dtype=wire,
            enable_in_kernel_fc2_reduce=k.enable_in_kernel_fc2_reduce,
        )
        knobs.update(k.knobs or {})
        if not knobs.pop("force_static_sched", True):
            raise MoEEpConfigError(
                "force_static_sched=False has no standard-kernel mapping."
            )
        if knobs.pop("enable_iket", False):
            raise MoEEpConfigError("The standard adapter does not expose IKET capture.")
        if knobs.pop("num_sched_stages", None) not in (None, 2):
            raise MoEEpConfigError("The standard scheduler has two stages.")
        if knobs.pop("clc_bundle_size", None) is not None:
            raise MoEEpConfigError("clc_bundle_size has no standard-kernel mapping.")
        allowed = {
            "mma_tiler_mnk",
            "cluster_shape_mnk",
            "use_2cta_instrs",
            "group_hint",
            "flag_batch",
            "epi_flag_batch",
            "token_back_mode",
            "token_back_by_dispatch",
            "load_balance_mode",
            "non_ubulk_fc2_store",
            "in_kernel_fc2_reduce",
        }
        if unknown := set(knobs) - allowed:
            raise MoEEpConfigError(
                f"Unsupported standard-kernel knobs: {sorted(unknown)}"
            )
        tile = tuple(knobs.get("mma_tiler_mnk", (256, 128, 128)))
        cluster = tuple(knobs.get("cluster_shape_mnk", (2, 1, 1)))
        if len(cluster) != 3 or cluster[2] != 1:
            raise MoEEpConfigError("Standard inference requires cluster K=1.")
        if knobs.get("use_2cta_instrs", tile[0] == 256) != (tile[0] == 256):
            raise MoEEpConfigError("use_2cta_instrs must match tile M.")
        token_back = knobs.get("token_back_mode", "epi_warps")
        if knobs.get("token_back_by_dispatch", False):
            token_back = "reuse_dispatch_warps"
        balance = knobs.get("load_balance_mode", "static")
        if balance not in ("static", "atomic_counter"):
            raise MoEEpConfigError(f"Unsupported load_balance_mode {balance!r}.")
        reduce = bool(knobs.get("in_kernel_fc2_reduce", False))
        if reduce and not k.enable_in_kernel_fc2_reduce:
            raise MoEEpConfigError("In-kernel reduction must be enabled in the config.")
        return Sm100BlockScaledMoeConfig(
            num_total_experts=fleet.num_experts,
            max_tokens_per_rank=fleet.max_tokens_per_rank,
            num_topk=k.top_k,
            hidden=fleet.token_hidden_size,
            intermediate=k.intermediate_size,
            rank=self.ep_rank,
            world_size=self.ep_world_size,
            quant_kind=kind,
            mma_tiler_mnk=tile,
            cluster_shape_mn=cluster[:2],
            schedule_policy=("grouped", knobs.get("group_hint")),
            work_id_mode="grid_stride" if balance == "static" else "atomic_counter",
            token_in_flag_batch=knobs.get("flag_batch", 1),
            epi_flag_batches=tuple(knobs.get("epi_flag_batch", (1, 1))),
            token_back_mode=token_back,
            fc2_use_bulk=not knobs.get("non_ubulk_fc2_store", True),
            gate_up_clamp=k.gate_up_clamp
            if k.gate_up_clamp is not None
            else k.activation_clamp,
            activation=getattr(k, "activation", "swiglu"),
            situ_beta=getattr(k, "situ_beta", None),
            situ_linear_beta=getattr(k, "situ_linear_beta", None),
            apply_topk_at_fc1=getattr(k, "apply_topk_in_fc1", True),
            combine_dtype=wire,
            reduce_topk_in_kernel=reduce,
            enable_fc1_activation_per_token_scale=getattr(
                k, "enable_fc1_activation_per_token_scale", False
            ),
        )

    def preprocess_weights(self, weights, fleet_params):
        from .....kernel_src.sm100.next_cutedsl_megamoe import (
            preprocess_block_scaled_weights,
            preprocess_prequantized_block_scaled_weights,
        )

        kwargs = dict(
            quant_kind=self._kind(), intermediate_size=self._config.intermediate_size
        )
        if isinstance(weights, PrequantizedMoEWeights):
            scale_dtype = (
                torch.float8_e4m3fn if self._kind() == "nvfp4" else torch.float8_e8m0fnu
            )

            def typed_scale(scale):
                return scale.view(scale_dtype) if scale.dtype == torch.uint8 else scale

            return preprocess_prequantized_block_scaled_weights(
                weights.w13,
                weights.w2,
                typed_scale(weights.w13_scale),
                typed_scale(weights.w2_scale),
                hidden_size=fleet_params.token_hidden_size,
                **kwargs,
            )
        return preprocess_block_scaled_weights(weights.w13, weights.w2, **kwargs)

    def validate_transformed_weights(
        self, transformed_weights, bootstrap, fleet_params
    ):
        from .....kernel_src.sm100.next_cutedsl_megamoe import validate_standard_weights

        validate_standard_weights(
            transformed_weights,
            self._workspace_config(fleet_params),
            torch.device("cuda", torch.cuda.current_device()),
        )

    def _allocate_workspace(self, fleet_params):
        from .....kernel_src.sm100.next_cutedsl_megamoe import (
            Sm100BlockScaledSymmBuffer,
        )

        workspace = Sm100BlockScaledSymmBuffer(self._workspace_config(fleet_params))
        if self._kind() == "nvfp4":
            for name in ("fc1_alpha", "fc2_alpha", "fc1_norm_const"):
                value = getattr(self._config, name)
                if isinstance(value, (int, float)):
                    getattr(workspace, name).fill_(value)
                elif value is not None:
                    getattr(workspace, name).copy_(value)
        return workspace

    def _workspace_pool_key(self, fleet_params):
        from .....core.kernel.workspace_pool import epilogue_pool_key

        return (
            self.kernel_name(),
            torch.cuda.current_device(),
            self.ep_rank,
            id(self._ep_comm_group),
            self._workspace_config(fleet_params),
            tuple(
                epilogue_pool_key(getattr(self._config, name, None))
                for name in ("fc1_alpha", "fc2_alpha", "fc1_norm_const")
            ),
        )

    def _legacy_backend(self):
        from . import mxfp8_mxfp8_bf16_cutedsl, nvfp4_nvfp4_bf16_cutedsl

        return (
            nvfp4_nvfp4_bf16_cutedsl
            if self._kind() == "nvfp4"
            else mxfp8_mxfp8_bf16_cutedsl
        )

    def validate_forward(self, t, fleet_params, *, quantize_input):
        staging = self._legacy_backend().staging
        if self._kind() == "nvfp4":
            staging.validate_nvfp4_forward_inputs(
                t.hidden_states,
                t.topk_ids,
                t.topk_weights,
                fleet_params,
                top_k=self._config.top_k,
                quantize_input=quantize_input,
                scales=t.scales,
            )
            staging.validate_fc1_activation_per_token_scale(
                t.fc1_activation_per_token_scale,
                num_tokens=t.hidden_states.shape[0],
                enabled=self._config.enable_fc1_activation_per_token_scale,
            )
        else:
            staging.validate_mxfp8_forward_inputs(
                t.hidden_states,
                t.topk_ids,
                t.topk_weights,
                fleet_params,
                top_k=self._config.top_k,
                quantize_input=quantize_input,
                scales=t.scales,
                kind=self._kind(),
            )
        for name in ("fc1_alpha", "fc2_alpha", "fc1_norm_const"):
            value = getattr(t, name)
            if value is not None and (
                self._kind() != "nvfp4"
                or value.dtype != torch.float32
                or tuple(value.shape)
                != (fleet_params.num_experts // self.ep_world_size,)
                or value.device != t.hidden_states.device
            ):
                raise MoEEpConfigError(
                    f"{name} requires one FP32 value per local NVFP4 expert."
                )

    def stage_inputs(self, t, workspace, *, quantize_input):
        n = t.hidden_states.shape[0]
        cfg = workspace.config
        if self._kind() == "nvfp4":
            self._legacy_backend().staging.validate_fc1_activation_per_token_scale(
                t.fc1_activation_per_token_scale,
                num_tokens=n,
                enabled=cfg.enable_fc1_activation_per_token_scale,
            )
        if quantize_input:
            kwargs = (
                dict(norm_const=self._config.input_norm_const)
                if self._kind() == "nvfp4"
                else dict(kind=self._kind())
            )
            self._legacy_backend().staging.stage_mega_moe_inputs(
                t.hidden_states,
                t.topk_weights,
                t.topk_ids,
                workspace.x,
                workspace.x_sf[:, : cfg.hidden // cfg.sf_vec_size],
                workspace.topk_idx,
                workspace.topk_weights,
                **kwargs,
            )
        else:
            workspace.x[:n].view(torch.uint8).copy_(t.hidden_states.view(torch.uint8))
            width = cfg.hidden // cfg.sf_vec_size
            workspace.x_sf[:n].view(torch.uint8).zero_()
            workspace.x_sf[:n, :width].view(torch.uint8).copy_(
                t.scales[:, :width].view(torch.uint8)
            )
            workspace.topk_idx[:n].copy_(t.topk_ids)
            workspace.topk_weights[:n].copy_(t.topk_weights)
            workspace.topk_idx[n:].fill_(-1)
            from .....kernel_src.sm100.cutedsl_megamoe import note_staged_tokens

            note_staged_tokens(workspace.topk_idx, n)
        workspace.note_staged_tokens(n)
        if cfg.enable_fc1_activation_per_token_scale:
            workspace.fc1_activation_per_token_scale[:n].copy_(
                t.fc1_activation_per_token_scale
            )
        for name in ("fc1_alpha", "fc2_alpha", "fc1_norm_const"):
            value = getattr(t, name)
            target = getattr(workspace, name)
            if target is None:
                continue
            if value is None:
                value = getattr(self._config, name, None)
            if value is None or isinstance(value, (int, float)):
                target.fill_(1.0 if value is None else value)
            else:
                target.copy_(value)

    def validate_capture_ready(self, workspace, transformed_weights):
        if workspace._compiled is None:
            raise RuntimeError("Warm up the standard workspace before graph capture.")

    def compute(self, workspace, transformed_weights, *, output):
        from .....kernel_src.sm100.next_cutedsl_megamoe import (
            sm100_block_scaled_mega_moe,
        )

        return sm100_block_scaled_mega_moe(
            output,
            *transformed_weights,
            workspace,
            fast_math=self._config.fast_math,
        )


@register_mega_kernel("sm100_nvfp4_nvfp4_bf16_cutedsl_next")
class StandardNvfp4Backend(_StandardBlockScaledBackend):
    quant_kind = "nvfp4"

    @classmethod
    def kernel_name(cls):
        return "sm100_nvfp4_nvfp4_bf16_cutedsl_next"


@register_mega_kernel("sm100_mxfp8_mxfp8_bf16_cutedsl_next")
class StandardMxfp8Backend(_StandardBlockScaledBackend):
    @classmethod
    def kernel_name(cls):
        return "sm100_mxfp8_mxfp8_bf16_cutedsl_next"
