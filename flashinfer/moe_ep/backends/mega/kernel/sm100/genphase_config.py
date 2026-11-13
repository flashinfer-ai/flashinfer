"""Shared GenPhase configuration selection for the Blackwell backend adapters."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

import torch

from .....config import FinalizeMode, FleetParams
from .....core.validation.common import MoEEpConfigError

if TYPE_CHECKING:
    from .mxfp8_mxfp8_bf16_cutedsl.config import (
        Sm100_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig,
    )
    from .nvfp4_nvfp4_bf16_cutedsl.config import (
        Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig,
    )


class GenphaseConfigMixin:
    """Resolve dtype-aware geometry and agree on the selection across EP ranks."""

    _config: (
        Sm100_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig
        | Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig
    )

    def _init_genphase_config(self):
        """Validate GenPhase features before selecting or allocating a workspace."""
        k = self._config
        if k.knobs not in (None, "auto") and not isinstance(k.knobs, dict):
            raise MoEEpConfigError(
                "GenPhase knobs must be None, 'auto', or a knob dictionary"
            )
        if (
            k.enable_in_kernel_fc2_reduce
            or getattr(k, "combine_dtype", "bf16") != "bf16"
        ):
            raise MoEEpConfigError(
                "Blackwell GenPhase requires separate BF16 reduction."
            )
        if k.finalize_mode not in (FinalizeMode.AUTO, FinalizeMode.DEFAULT):
            raise MoEEpConfigError(
                "Blackwell GenPhase supports only the default finalize kernel."
            )
        if (
            k.finalize_mode is FinalizeMode.AUTO
            and k.num_valid_tokens_tensor is not None
        ):
            raise MoEEpConfigError(
                "Blackwell GenPhase does not support AUTO finalize with "
                "num_valid_tokens_tensor; persistent finalization is unsupported."
            )
        if k.activation != "swiglu" or k.swiglu_alpha is not None:
            raise MoEEpConfigError("Blackwell GenPhase supports standard SwiGLU.")
        if getattr(k, "enable_fc1_activation_per_token_scale", False):
            raise MoEEpConfigError(
                "Blackwell GenPhase does not support per-token FC1 scales."
            )
        self._genphase_configs: dict[tuple, Any] = {}
        self._genphase_constraints()

    def _genphase_constraints(self):
        """Merge explicit fields and knob overrides, rejecting conflicting values."""
        k = self._config
        names = ("mma_tiler_mnk", "cluster_shape_mn", "schedule_policy")
        knobs = dict(k.knobs) if isinstance(k.knobs, dict) else {}
        unknown = set(knobs) - set(names)
        if unknown:
            raise MoEEpConfigError(
                f"unknown Blackwell GenPhase knobs: {sorted(unknown)}"
            )
        for name in names:
            explicit = getattr(k, name)
            if explicit is not None:
                if name in knobs and knobs[name] != explicit:
                    raise MoEEpConfigError(f"conflicting explicit GenPhase {name}")
                knobs[name] = explicit
        return knobs

    def _genphase_base_config(self, fleet_params):
        """Describe the EP workload before selecting performance knobs."""
        from .....kernel_src.sm100.next_cutedsl_megamoe import Sm100GenphaseMoeConfig

        k = self._config
        return Sm100GenphaseMoeConfig(
            num_total_experts=fleet_params.num_experts,
            max_tokens_per_rank=fleet_params.max_tokens_per_rank,
            num_topk=k.top_k,
            hidden=fleet_params.token_hidden_size,
            intermediate=k.intermediate_size,
            rank=self.ep_rank,
            world_size=self.ep_world_size,
            quant_kind=getattr(k, "kind", "nvfp4"),
            gate_up_clamp=(
                k.gate_up_clamp if k.gate_up_clamp is not None else k.activation_clamp
            ),
            apply_routing_weights_before_fc2=k.apply_routing_weights_before_fc2,
        )

    def _genphase_config(self, fleet_params: FleetParams):
        """Memoize cache or heuristic selection for one workload and device.

        Automatic workspaces start from the capacity heuristic. Their v2
        selection uses the actual staged batch size during warmup or forward.
        """
        from .....kernel_src.sm100.next_cutedsl_megamoe import resolve_genphase_knobs

        try:
            base = self._genphase_base_config(fleet_params)
            constraints = self._genphase_constraints()
            gpu = torch.cuda.is_available()
            device = torch.cuda.current_device() if gpu else None
            props = torch.cuda.get_device_properties(device) if gpu else None
            sms = props.multi_processor_count if props is not None else 4
            hardware = (
                (device, props.major, props.minor, props.name, sms)
                if props is not None
                else None
            )
            automatic = self._config.knobs == "auto"
            live_tokens = None if automatic else fleet_params.tuning_live_tokens
            key = (base, repr(constraints), live_tokens, hardware)
            if key not in self._genphase_configs:
                knobs, _ = resolve_genphase_knobs(
                    base,
                    sm_count=sms,
                    live_tokens=live_tokens,
                    constraints=constraints,
                    use_cache=gpu and not automatic,
                )
                self._genphase_configs[key] = dataclasses.replace(base, **knobs)
            return self._genphase_configs[key]
        except (TypeError, ValueError) as exc:
            raise MoEEpConfigError(str(exc)) from exc

    def _check_genphase_selection(self, config):
        """Require identical configuration selection across EP ranks at allocation."""
        # Selection is construction-time work; never inspect live routing on forward.
        if self.ep_world_size > 1:
            import torch.distributed as dist

            proposal = dataclasses.asdict(config)
            proposal.pop("rank")
            proposals = [None] * self.ep_world_size
            dist.all_gather_object(proposals, proposal, group=self.ep_comm_group)
            if any(p != proposal for p in proposals):
                raise MoEEpConfigError(
                    "GenPhase configuration/cache selection differs across EP ranks"
                )
