# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Blackwell backend hooks for the collective complete-EP selector."""

from __future__ import annotations

import dataclasses


def _legacy_knobs(config):
    names = (
        "mma_tiler_mnk",
        "cluster_shape_mnk",
        "use_2cta_instrs",
        "load_balance_mode",
        "group_hint",
        "force_static_sched",
        "clc_bundle_size",
        "num_sched_stages",
        "flag_batch",
        "epi_flag_batch",
        "non_ubulk_fc2_store",
        "in_kernel_fc2_reduce",
        "enable_iket",
        "token_back_mode",
        "transform_buffer",
        "accumulator_overlap",
        "transform_k_tile",
    )
    knobs = {name: getattr(config, name) for name in names if hasattr(config, name)}
    if "token_back_mode" not in knobs:
        knobs["token_back_mode"] = (
            "reuse_dispatch_warps" if config.token_back_by_dispatch else "epi_warps"
        )
    return knobs


class BlackwellEpTuningMixin:
    """Freeze a prepared plan while leaving arithmetic in its original backend."""

    def ep_weight_pack(self, weights):
        # The outer selector accepts [up, gate]; Mega kernels accept [gate, up].
        import torch
        from .....weights import MoEWeightPack

        intermediate = self._config.intermediate_size
        if weights.w13.shape[1] != 2 * intermediate:
            raise ValueError(
                "Gated EP weights must have 2 * intermediate_size FC1 rows"
            )
        return MoEWeightPack(
            torch.cat(
                (weights.w13[:, intermediate:], weights.w13[:, :intermediate]), dim=1
            ),
            weights.w2,
        )

    def ep_tuning_contract(self, fleet_params):
        from .common.activation_batch_invariant import batch_invariant_output_mode

        self.validate_init(self._ep_bootstrap, fleet_params)
        cfg = self._config
        if batch_invariant_output_mode(cfg) != "finalized":
            raise NotImplementedError("EP selection requires finalized output")
        if not getattr(cfg, "batch_invariant", False) and (
            getattr(cfg, "swiglu_alpha", None) not in (None, 1.0)
            or getattr(cfg, "swiglu_beta", None) not in (None, 0.0)
        ):
            raise NotImplementedError(
                "EP selection requires standard SwiGLU constants unless "
                "batch_invariant=True"
            )
        kind = (
            self._kind()
            if callable(getattr(self, "_kind", None))
            else getattr(cfg, "kind", None)
        )
        if kind is None:
            kind = (
                "bf16_nvfp4"
                if "bf16_nvfp4" in cfg.kernel_name
                else "nvfp4"
                if "nvfp4" in cfg.kernel_name
                else "bf16"
            )
        result = {
            "revision": "blackwell-complete-ep-v1",
            "quantization": kind,
            "intermediate": cfg.intermediate_size,
            "top_k": cfg.top_k,
            "activation": getattr(cfg, "activation", "swiglu"),
            "gate_up_clamp": cfg.gate_up_clamp
            if cfg.gate_up_clamp is not None
            else getattr(cfg, "activation_clamp", None),
            "situ_beta": getattr(cfg, "situ_beta", None),
            "situ_linear_beta": getattr(cfg, "situ_linear_beta", None),
            "per_token_activation": getattr(
                cfg, "enable_fc1_activation_per_token_scale", False
            ),
            "nvfp4_4over6": None,
            "apply_routing_weights_before_fc2": getattr(
                cfg, "apply_routing_weights_before_fc2", True
            ),
            "combine_dtype": getattr(cfg, "combine_dtype", "bf16"),
        }
        if getattr(cfg, "batch_invariant", False):
            result["batch_invariant_fingerprint"] = self.batch_invariant_fingerprint(
                fleet_params
            )
        return result

    def ep_selected_config(self, workspace):
        if hasattr(workspace, "_frontend"):
            knobs = _legacy_knobs(workspace._frontend.config)
        else:
            from .....kernel_src.sm100.next_cutedsl_megamoe.shim.tuning import knob_keys

            knobs = {
                name: getattr(workspace.config, name)
                for name in knob_keys(workspace.config)
            }
        return dataclasses.replace(self._config, knobs=knobs)

    def ep_prepare_fixed(self, workspace, weights):
        """Compile and bind locally; the outer collective agrees before launch."""
        if not hasattr(workspace, "_frontend"):
            workspace.prepare(*weights)
            return
        frontend = workspace._frontend
        name = type(frontend.config).__name__
        if name == "MegaMoEBf16Nvfp4Config":
            import torch
            from .....cute_dsl.megamoe.bf16_nvfp4.frontend import MegaMoEBf16Nvfp4Inputs

            staging = workspace._staging_inputs
            n = workspace.num_max_tokens if staging is None else staging[0].shape[0]
            workspace._ep_tuning_output = torch.empty(
                (n, workspace.hidden), dtype=torch.bfloat16, device=workspace.x.device
            )
            frontend.make_launch_thunk(
                MegaMoEBf16Nvfp4Inputs(
                    workspace.x,
                    workspace.topk_idx,
                    workspace.topk_weights,
                    weights[0][0],
                    weights[0][1],
                    workspace.fc1_alpha,
                    weights[1][0],
                    weights[1][1],
                    workspace.fc2_alpha,
                    workspace.kernel_combine_output,
                    workspace._ep_tuning_output,
                    staging,
                    workspace.fc1_norm_const if workspace._use_fc1_norm_const else None,
                )
            )
            return
        from .....kernel_src.sm100 import cutedsl_megamoe as api

        launchers = {
            "MegaMoEBf16Config": api.bf16_mega_launch_thunk,
            "MegaMoENvfp4Config": api.nvfp4_mega_launch_thunk,
            "MegaMoEMxfp8Config": api.mxfp8_mega_launch_thunk,
            "MegaMoEBf16Mxfp8Config": api.bf16_mxfp8_mega_launch_thunk,
        }
        launchers[name](*weights, workspace)

    def ep_force_tune(self, workspace, tensors, weights, output):
        if hasattr(workspace, "_frontend"):
            # Legacy's native tuner already uses staged, collective preparation.
            self._autotune_pending = True
            self.stage_inputs(tensors, workspace, quantize_input=True)
            self.compute(workspace, weights, output=output)
            return
        from flashinfer.autotuner import AutoTuner
        from .forward_tuning import ForwardExecution
        from .....kernel_src.sm100.next_cutedsl_megamoe.shim.tuning_protocol import (
            NAMESPACE,
            BlackwellTuningProtocol,
        )

        self.stage_inputs(tensors, workspace, quantize_input=True)
        execution = ForwardExecution(self, tensors, quantize_input=True, output=output)
        AutoTuner.get().choose_with_protocol(
            NAMESPACE,
            BlackwellTuningProtocol(
                output,
                *weights,
                workspace,
                num_tokens=tensors.num_tokens,
                install_winner=True,
                process_group=self.ep_comm_group if self.ep_world_size > 1 else None,
                allow_nondeterministic=getattr(
                    self._config, "enable_in_kernel_fc2_reduce", False
                ),
                constraints=self._genphase_constraints()
                if getattr(self._config, "kernel_variant", None) == "genphase"
                else None,
                make_trial=execution.make_trial,
                execution_contract=execution.contract(),
            ),
            force_tune=True,
        )
