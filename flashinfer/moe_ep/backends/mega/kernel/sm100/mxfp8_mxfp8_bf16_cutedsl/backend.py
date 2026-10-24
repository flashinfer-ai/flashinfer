"""CuTeDSL MXFP8 mega-MoE kernel backend.

The fused kernel consumes MXFP8 expert weights in kernel-ready layout (fp8
weights + atom-swizzled E8M0 scale factors). ``MoEWeightPack`` supplies
canonical bf16 ``w13``/``w2`` by default; ``preprocess_weights()`` quantizes
and swizzles them. Pass pre-quantized MXFP8 weights via ``w13``/``w2`` plus
``w13_scale``/``w2_scale`` to skip re-quantization.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

import torch

from ......config import BootstrapConfig, FinalizeMode, FleetParams
from ......core.kernel.base import MegaKernelBackend
from ......core.kernel.registry import register_mega_kernel
from ......core.runtime import mxfp8_cutedsl_runtime_requirements
from ......core.validation.common import (
    MoEEpArchError,
    MoEEpConfigError,
    validate_mega_arch,
    validate_mega_fleet_params,
)
from ......weights import MoEWeightPack
from .config import Sm100_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig
from .staging import (
    stage_genphase_inputs,
    stage_mega_moe_inputs,
    validate_genphase_forward_inputs,
    validate_mxfp8_forward_inputs,
)
from .weights import (
    TransformedMegaWeights,
    preprocess_mega_weights,
    validate_transformed_mega_weights,
)

if TYPE_CHECKING:
    from ......tensors import MoEEpTensors


def _resolve_gate_up_clamp(
    config: Sm100_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig,
) -> float | None:
    if config.gate_up_clamp is not None:
        return config.gate_up_clamp
    return config.activation_clamp


@register_mega_kernel(
    "sm100_mxfp8_mxfp8_bf16_cutedsl", deprecated_aliases=("mxfp8_cutedsl",)
)
class Mxfp8CutedslMegaKernelBackend(MegaKernelBackend):
    supports_output_view = True

    def __init__(self, config: Sm100_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig) -> None:
        """Validate pipeline options and initialize deferred tuning state."""
        super().__init__(config)
        self._kernel_config: Sm100_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig = config
        if config.kernel_variant not in ("inference", "genphase"):
            raise MoEEpConfigError(
                f"unsupported kernel_variant {config.kernel_variant!r}"
            )
        if config.kernel_variant == "genphase":
            if (
                config.knobs is not None
                and config.knobs != "auto"
                and not isinstance(config.knobs, dict)
            ):
                raise MoEEpConfigError(
                    "GenPhase knobs must be None, 'auto', or a knob dictionary"
                )
            self._genphase_configs: dict[tuple, Any] = {}
            self._genphase_constraints()
            if config.enable_in_kernel_fc2_reduce:
                raise MoEEpConfigError(
                    "Blackwell GenPhase requires separate reduction."
                )
            if config.resolved_finalize_mode is not FinalizeMode.DEFAULT:
                raise MoEEpConfigError(
                    "Blackwell GenPhase supports only the default finalize "
                    "kernel; persistent or unfinalized output requires "
                    "kernel_variant='inference'."
                )
            policy = config.schedule_policy
            if policy is not None and (
                not isinstance(policy, tuple)
                or len(policy) != 2
                or policy[0] != "grouped"
                or (
                    policy[1] is not None
                    and (type(policy[1]) is not int or policy[1] <= 0)
                )
            ):
                raise MoEEpConfigError(
                    "GenPhase schedule_policy must be ('grouped', None) "
                    "or ('grouped', a positive integer)."
                )
        elif (
            config.mma_tiler_mnk is not None
            or config.cluster_shape_mn is not None
            or config.schedule_policy is not None
        ):
            raise MoEEpConfigError(
                "mma_tiler_mnk/cluster_shape_mn/schedule_policy require "
                "kernel_variant='genphase'; "
                "use knobs for the inference pipeline."
            )
        self.supports_unfinalized_output = config.finalize_mode is FinalizeMode.NONE
        # knobs="auto": tune at the first compute() (weights + staged inputs
        # exist there), then keep the winner for the session.
        self._autotune_pending = config.knobs == "auto"
        if self._autotune_pending:
            import warnings

            warnings.warn(
                "knobs='auto' runs a COLLECTIVE multi-minute compile+timing "
                "sweep at the first forward — never use it inside a serving "
                "engine. Tune offline instead (python -m flashinfer.moe_ep.tune"
                "); winners persist in the knob cache and knobs=None then "
                "resolves them with a pure lookup.",
                UserWarning,
                stacklevel=3,
            )

    @classmethod
    def kernel_name(cls) -> str:
        return "sm100_mxfp8_mxfp8_bf16_cutedsl"

    def runtime_requirements(self, bootstrap: BootstrapConfig) -> frozenset[str]:
        return mxfp8_cutedsl_runtime_requirements(bootstrap)

    def _genphase_constraints(self):
        """Merge explicit fields and knob overrides, rejecting conflicting values."""
        k = self._kernel_config
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
        from ......kernel_src.sm100.next_cutedsl_megamoe import Sm100GenphaseMoeConfig

        k = self._kernel_config
        return Sm100GenphaseMoeConfig(
            num_total_experts=fleet_params.num_experts,
            max_tokens_per_rank=fleet_params.max_tokens_per_rank,
            num_topk=k.top_k,
            hidden=fleet_params.token_hidden_size,
            intermediate=k.intermediate_size,
            rank=self.ep_rank,
            world_size=self.ep_world_size,
            quant_kind=k.kind,
            gate_up_clamp=_resolve_gate_up_clamp(k),
        )

    def _genphase_config(self, fleet_params: FleetParams):
        """Memoize cache or heuristic selection for one workload and device.

        Auto tuning bypasses the disk cache here and retunes its default
        workspace during warmup. Public auto layers cannot create additional
        capacity profiles through ``create_workspace()``.
        """
        from ......kernel_src.sm100.next_cutedsl_megamoe import resolve_genphase_knobs

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
            key = (base, repr(constraints), fleet_params.tuning_live_tokens, hardware)
            if key not in self._genphase_configs:
                knobs, _ = resolve_genphase_knobs(
                    base,
                    sm_count=sms,
                    live_tokens=fleet_params.tuning_live_tokens,
                    constraints=constraints,
                    use_cache=gpu and self._kernel_config.knobs != "auto",
                )
                self._genphase_configs[key] = dataclasses.replace(base, **knobs)
            return self._genphase_configs[key]
        except (TypeError, ValueError) as exc:
            raise MoEEpConfigError(str(exc)) from exc

    def live_token_buckets(self, fleet_params: FleetParams) -> list[int]:
        """List cached GenPhase profiles when no explicit knob constraints apply."""
        if self._kernel_config.kernel_variant != "genphase":
            return super().live_token_buckets(fleet_params)
        if (
            self._kernel_config.knobs is not None
            or self._genphase_constraints()
            or not torch.cuda.is_available()
        ):
            return []
        from ......kernel_src.sm100.next_cutedsl_megamoe import (
            genphase_live_token_buckets,
        )

        return genphase_live_token_buckets(self._genphase_base_config(fleet_params))

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

    def _workspace_pool_request(self, fleet_params):
        """Return a pool key and allocator; mutable auto workspaces stay unpooled."""
        if self._kernel_config.kernel_variant != "genphase":
            return super()._workspace_pool_request(fleet_params)
        if self._kernel_config.knobs == "auto":
            return None
        from ......kernel_src.sm100.next_cutedsl_megamoe import Sm100GenphaseSymmBuffer

        config = self._genphase_config(fleet_params)
        self._check_genphase_selection(config)
        key = (
            self.kernel_name(),
            torch.cuda.current_device(),
            id(self._ep_comm_group),
            "genphase",
            config,
        )
        return key, lambda: Sm100GenphaseSymmBuffer(config)

    def validate_init(
        self,
        bootstrap: BootstrapConfig,
        fleet_params: FleetParams,
    ) -> None:
        validate_mega_arch()
        validate_mega_fleet_params(
            fleet_params,
            bootstrap.world_size,
            intermediate_size=self._kernel_config.intermediate_size,
            top_k=self._kernel_config.top_k,
            # cutedsl tiles are tail-safe (ceil-div + predicated epilogue);
            # the binding bound is TMA row alignment, not the dg SF word.
            alignment=64,
        )
        if self._kernel_config.kernel_variant == "genphase":
            if torch.cuda.is_available():
                cc = torch.cuda.get_device_capability()
                if cc not in ((10, 0), (10, 3)):
                    raise MoEEpArchError(
                        f"Blackwell GenPhase requires SM100/103; got {cc}."
                    )
            self._genphase_config(fleet_params)

    def preprocess_weights(
        self,
        weights: MoEWeightPack,
        fleet_params: FleetParams,
    ) -> TransformedMegaWeights:
        if self._kernel_config.kernel_variant == "genphase":
            return preprocess_mega_weights(
                weights,
                intermediate_size=self._kernel_config.intermediate_size,
                hidden_size=fleet_params.token_hidden_size,
                kind=self._kernel_config.kind,
                kernel_variant="genphase",
            )
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
        if self._kernel_config.kernel_variant == "genphase":
            from ......kernel_src.sm100.next_cutedsl_megamoe import (
                validate_genphase_weights,
            )

            try:
                validate_genphase_weights(
                    transformed_weights,
                    self._genphase_config(fleet_params),
                    torch.device("cuda", torch.cuda.current_device()),
                )
            except ValueError as exc:
                raise MoEEpConfigError(str(exc)) from exc
            return
        validate_transformed_mega_weights(
            transformed_weights,
            intermediate_size=self._kernel_config.intermediate_size,
            hidden_size=fleet_params.token_hidden_size,
            kind=self._kernel_config.kind,
            world_size=self.ep_world_size,
            num_experts=fleet_params.num_experts,
        )

    supports_live_token_tuning = True
    _knob_cache_module = (
        "flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe.shim.knob_cache"
    )

    def _tuning_key(self, fleet_params: FleetParams) -> dict[str, Any]:
        k = self._kernel_config
        return dict(
            dtype=k.kind,
            world_size=self.ep_world_size,
            hidden=fleet_params.token_hidden_size,
            intermediate=k.intermediate_size,
            num_experts=fleet_params.num_experts,
            topk=k.top_k,
            max_tokens=fleet_params.max_tokens_per_rank,
            enable_in_kernel_fc2_reduce=k.enable_in_kernel_fc2_reduce,
            use_custom_finalize=k.finalize_mode is FinalizeMode.NONE,
        )

    def _allocate_workspace(self, fleet_params: FleetParams) -> Any:
        """Allocate the selected pipeline's workspace collectively on the EP group."""
        if self._kernel_config.kernel_variant == "genphase":
            from ......kernel_src.sm100.next_cutedsl_megamoe import (
                Sm100GenphaseSymmBuffer,
            )

            config = self._genphase_config(fleet_params)
            self._check_genphase_selection(config)
            return Sm100GenphaseSymmBuffer(config)
        from ......kernel_src.sm100.cutedsl_megamoe import (
            get_symm_buffer_for_mxfp8_mega_moe,
        )

        k = self._kernel_config
        fp = fleet_params
        workspace = get_symm_buffer_for_mxfp8_mega_moe(
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
            enable_in_kernel_fc2_reduce=k.enable_in_kernel_fc2_reduce,
            use_persistent_finalize_kernel=k.resolved_finalize_mode
            is FinalizeMode.PERSISTENT,
            use_custom_finalize=k.finalize_mode is FinalizeMode.NONE,
            num_valid_tokens_tensor=k.num_valid_tokens_tensor,
            knobs=self._workspace_knobs(fleet_params),
        )
        if k.finalize_mode is FinalizeMode.NONE:
            workspace._unfinalized_route_map = torch.arange(
                fp.max_tokens_per_rank * k.top_k,
                dtype=torch.int32,
                device=workspace.topk_idx.device,
            ).view(fp.max_tokens_per_rank, k.top_k)
            # The kernel folds routing weights into FC1.
            workspace._unfinalized_weights = torch.ones_like(workspace.topk_weights)
        return workspace

    def validate_capture_ready(self, workspace, transformed_weights):
        """Reject GenPhase capture until collective auto warmup has completed."""
        if self._kernel_config.kernel_variant == "genphase" and self._autotune_pending:
            raise MoEEpConfigError(
                "GenPhase knobs='auto' requires collective warmup before CUDA graph capture"
            )

    def validate_forward(
        self,
        t: "MoEEpTensors",
        fleet_params: FleetParams,
        *,
        quantize_input: bool,
    ) -> None:
        if self._kernel_config.kernel_variant == "genphase":
            validate_genphase_forward_inputs(
                t,
                fleet_params,
                top_k=self._kernel_config.top_k,
                quantize_input=quantize_input,
            )
            return
        validate_mxfp8_forward_inputs(
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
        if self._kernel_config.kernel_variant == "genphase":
            stage_genphase_inputs(t, workspace, quantize_input=quantize_input)
            return
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
            # Backend talks only to the cutedsl_megamoe shim (never src/ directly).
            from ......kernel_src.sm100.cutedsl_megamoe import (
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
            from ......kernel_src.sm100.cutedsl_megamoe import note_staged_tokens

            note_staged_tokens(workspace.topk_idx, num_tokens)

    def compute(
        self,
        workspace: Any,
        transformed_weights: TransformedMegaWeights,
        *,
        output: torch.Tensor | None,
    ) -> torch.Tensor:
        """Launch staged inputs, tuning once first when explicitly requested.

        All EP ranks must enter the initial auto sweep together outside graph
        capture. With ``output=None``, the result borrows workspace storage.
        """
        if self._kernel_config.kernel_variant == "genphase":
            from ......kernel_src.sm100.next_cutedsl_megamoe import (
                sm100_genphase_mega_moe,
            )

            num_tokens = (
                output.shape[0] if output is not None else workspace.staged_tokens()
            )
            if self._autotune_pending:
                from ......kernel_src.sm100.next_cutedsl_megamoe import (
                    autotune_sm100_genphase_mega_moe,
                )

                winner = autotune_sm100_genphase_mega_moe(
                    output,
                    transformed_weights[0],
                    transformed_weights[1],
                    workspace,
                    num_tokens=num_tokens,
                    constraints=self._genphase_constraints(),
                    process_group=self._ep_comm_group,
                )
                workspace.retune(winner)
                self._autotune_winner = winner
                self._autotune_pending = False
            view = sm100_genphase_mega_moe(
                output,
                transformed_weights[0],
                transformed_weights[1],
                workspace,
                num_tokens=num_tokens,
                fast_math=self._kernel_config.fast_math,
            )
            return output if output is not None else view
        from ......kernel_src.sm100.cutedsl_megamoe import mxfp8_mega_moe, staged_tokens

        if self.supports_unfinalized_output:
            raise ValueError("finalize_mode=NONE requires compute_unfinalized()")
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
            # so the candidate sweep stays in lockstep (see shim/autotune.py).
            from ......kernel_src.sm100.cutedsl_megamoe import autotune_mxfp8_mega_moe

            self._autotune_winner = autotune_mxfp8_mega_moe(
                output,
                transformed_weights[0],
                transformed_weights[1],
                workspace,
                num_tokens=num_tokens,
                gate_up_clamp=_resolve_gate_up_clamp(kcfg),
                activation_clamp=kcfg.activation_clamp,
            )
            # Cleared only on success: if the collective tune raises, a retried
            # compute() re-attempts it (all ranks fail together, so lockstep holds).
            self._autotune_pending = False
        view = mxfp8_mega_moe(
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

    def compute_unfinalized(
        self,
        workspace: Any,
        transformed_weights: TransformedMegaWeights,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from ......kernel_src.sm100.cutedsl_megamoe import (
            MegaMoEMxfp8Inputs,
            staged_tokens,
        )

        if not self.supports_unfinalized_output:
            raise ValueError("compute_unfinalized() requires finalize_mode=NONE")
        num_tokens = staged_tokens(workspace.topk_idx)
        if num_tokens is None:
            raise ValueError("compute_unfinalized() requires stage_inputs() first")
        clamp = _resolve_gate_up_clamp(self._kernel_config)
        if clamp is not None:
            workspace._frontend.set_gate_up_clamp(clamp)
        # Launch the full padded buffer even when this rank has no live tokens.
        workspace._frontend.run(
            MegaMoEMxfp8Inputs(
                activation=workspace.x,
                activation_sf=workspace.x_sf,
                topk_idx=workspace.topk_idx,
                topk_weights=workspace.topk_weights,
                fc1_weight=transformed_weights[0][0],
                fc1_weight_sf=transformed_weights[0][1],
                fc2_weight=transformed_weights[1][0],
                fc2_weight_sf=transformed_weights[1][1],
                output_activation=workspace.output_activation,
                num_valid_tokens=workspace.num_valid_tokens,
            ),
            num_tokens=None,
            sync=False,
        )
        partials, _root, _region = workspace._frontend.custom_finalize_workspace()
        return (
            partials[:num_tokens].view(
                num_tokens * self._kernel_config.top_k, workspace.hidden
            ),
            workspace._unfinalized_weights[:num_tokens],
            workspace._unfinalized_route_map[:num_tokens],
        )

    def _workspace_pool_key(self, fleet_params: FleetParams) -> Any:
        """Key immutable workspace configurations; return None for auto tuning."""
        k = self._kernel_config
        if k.kernel_variant == "genphase":
            if k.knobs == "auto":
                return None
            return (
                self.kernel_name(),
                torch.cuda.current_device(),
                id(self._ep_comm_group),
                "genphase",
                self._genphase_config(fleet_params),
            )
        if k.knobs == "auto":
            # Autotune retunes (and recompiles) the workspace's shared
            # frontend at first compute; give each session its own buffer.
            return None
        from ......core.kernel.workspace_pool import epilogue_pool_key, knobs_pool_key

        fp = fleet_params
        return (
            "sm100_mxfp8_mxfp8_bf16_cutedsl",
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
            k.enable_in_kernel_fc2_reduce,
            k.resolved_finalize_mode,
            epilogue_pool_key(k.num_valid_tokens_tensor),
            knobs_pool_key(self._workspace_knobs(fleet_params)),
        )

    def _forget_workspace_state(self, workspace) -> None:
        # The fused-stage memos key on topk_idx.data_ptr(); the symmetric
        # heap reuses freed addresses, so evict before the buffer dies.
        # sys.modules lookup (not an import): if the shim was never loaded,
        # no memo exists and the heavy import must not happen.
        import sys

        quant_stage = sys.modules.get(
            "flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe.shim.quant_stage"
        )
        topk_idx = getattr(workspace, "topk_idx", None)
        if quant_stage is not None and topk_idx is not None:
            quant_stage.forget_staged_tokens(topk_idx)
