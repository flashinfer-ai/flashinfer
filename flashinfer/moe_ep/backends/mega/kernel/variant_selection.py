"""Workspace lifecycle shared by MegaMoE variant selectors."""

from __future__ import annotations

import dataclasses
from abc import abstractmethod
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import torch

from flashinfer.autotuner import AutoTuner, TuningContext

from ....core.kernel.base import MegaKernelBackend
from ....core.tuning_measurement import measurement_settings
from ....core.tuning_state import AutoTuningState, prepared_for_serving, record_prepared
from ....core.validation.common import MoEEpConfigError

if TYPE_CHECKING:
    from ....config import FleetParams
    from ....tensors import MoEEpTensors


@dataclasses.dataclass
class VariantWorkspace:
    fleet: FleetParams
    kernel: MegaKernelBackend
    workspace: Any
    _tuning_state: AutoTuningState = dataclasses.field(default_factory=AutoTuningState)
    tensors: MoEEpTensors | None = None
    quantize_input: bool = True


class VariantBackend(MegaKernelBackend):
    """Select a fixed variant and its knobs before graph capture."""

    supports_output_view = True
    supports_auto_workspaces = True
    variant_namespace: str
    native_namespace: str

    @staticmethod
    @abstractmethod
    def _validate_context(context: TuningContext) -> None:
        """Validate the autotuner options supported by the native backend."""

    _measurement_settings = staticmethod(measurement_settings)

    @abstractmethod
    def _protocol(self, workspace, weights, output):
        """Build the architecture's complete-forward selection protocol."""

    @classmethod
    def _context(cls) -> TuningContext:
        context = AutoTuner.get().protocol_context(cls.variant_namespace)
        cls._validate_context(context)
        return context

    @classmethod
    def _cache_identity(cls, context: TuningContext) -> tuple:
        native_context = context.tuner.protocol_context(cls.native_namespace)
        return context.cache_identity, native_context.skipped

    @classmethod
    def _identity(cls, context: TuningContext) -> tuple:
        return cls._cache_identity(context), cls._measurement_settings(context)

    @classmethod
    def _pending(cls, state: AutoTuningState, context: TuningContext) -> bool:
        cache_identity = cls._cache_identity(context)
        if state.cache_identity != cache_identity:
            return True
        if context.tuning_enabled:
            return state.tuned_identity != cls._identity(context)
        return state.winner is None and state.lookup_scope != (
            context.scope,
            cache_identity,
        )

    def __init__(self, config, backend_type):
        super().__init__(config)
        if config.knobs != "auto":
            raise MoEEpConfigError("kernel_variant='auto' requires knobs='auto'")
        self._backend_type = backend_type
        self._fallback = backend_type(
            dataclasses.replace(config, kernel_variant="inference")
        )

    def ep_weight_pack(self, weights):
        return self._fallback.ep_weight_pack(weights)

    def ep_selected_config(self, workspace):
        return workspace.kernel.ep_selected_config(workspace.workspace)

    def ep_tuning_fallback(self):
        return self._fallback.ep_tuning_fallback()

    def ep_force_tune(self, workspace, tensors, weights, output):
        self.stage_inputs(tensors, workspace, quantize_input=True)
        AutoTuner.get().choose_with_protocol(
            self.variant_namespace,
            self._protocol(workspace, weights, output),
            force_tune=True,
        )

    def ep_tuning_contract(self, fleet_params):
        return self._fallback.ep_tuning_contract(fleet_params)

    @classmethod
    def kernel_name(cls):
        return cls.variant_namespace

    def bind_ep_bootstrap(self, bootstrap):
        super().bind_ep_bootstrap(bootstrap)
        self._fallback.bind_ep_bootstrap(bootstrap)

    def runtime_requirements(self, bootstrap):
        return self._fallback.runtime_requirements(bootstrap)

    def validate_init(self, bootstrap, fleet_params):
        self._ensure_ep_bootstrap(bootstrap)
        self._fallback.validate_init(bootstrap, fleet_params)

    def preprocess_weights(self, weights, fleet_params):
        # Both variants consume the same format-specific weight layout.
        return self._fallback.preprocess_weights(weights, fleet_params)

    def prepare_transformed_weights(self, weights, bootstrap, fleet_params):
        return self._fallback.prepare_transformed_weights(
            weights, bootstrap, fleet_params
        )

    def validate_transformed_weights(self, weights, bootstrap, fleet_params):
        self._fallback.validate_transformed_weights(weights, bootstrap, fleet_params)

    def validate_forward(self, t, fleet_params, *, quantize_input):
        self._fallback.validate_forward(t, fleet_params, quantize_input=quantize_input)

    def _gather(self, value):
        if self.ep_world_size == 1:
            return [value]
        import torch.distributed as dist

        values = [None] * self.ep_world_size
        dist.all_gather_object(values, value, group=self.ep_comm_group)
        return values

    def _backend(self, variant, knobs="auto"):
        backend = self._backend_type(
            dataclasses.replace(
                self._config, kernel_variant=variant, knobs=deepcopy(knobs)
            )
        )
        backend.bind_ep_bootstrap(self._ep_bootstrap)
        return backend

    def _fixed_backend(self, workspace):
        config = self._fallback.ep_selected_config(workspace)
        return self._backend(config.kernel_variant, config.knobs)

    def _allocate_workspace(self, fleet_params):
        workspace = self._fallback.prepare_workspace(self._ep_bootstrap, fleet_params)
        return VariantWorkspace(fleet_params, self._fixed_backend(workspace), workspace)

    def coordinate_workspace_replacement(self, workspace):
        context = self._context()
        state = workspace._tuning_state
        if torch.cuda.is_current_stream_capturing():
            if state.cache_identity != self._cache_identity(context):
                raise RuntimeError(
                    "Warm up variant selection after changing the autotune context"
                )
            return False
        if prepared_for_serving(state, context):
            return False
        replace = (
            state.graph_captured
            and self._pending(state, context)
            and (
                context.tuning_enabled
                or state.cache_identity != self._cache_identity(context)
            )
        )
        decisions = self._gather((context.tuning_enabled, replace))
        if any(enabled != context.tuning_enabled for enabled, _ in decisions):
            raise ValueError("EP ranks disagree on whether variant tuning is enabled")
        return any(replace for _, replace in decisions)

    def validate_capture_ready(self, workspace, weights):
        context = self._context()
        if self._pending(workspace._tuning_state, context) and context.tuning_enabled:
            raise RuntimeError(
                "Warm up pending variant tuning before CUDA graph capture"
            )
        self._prepare_capture(workspace, weights)

    def _prepare_capture(self, workspace, weights):
        workspace.kernel.validate_capture_ready(workspace.workspace, weights)

    def stage_inputs(self, t, workspace, *, quantize_input):
        workspace.tensors = t
        workspace.quantize_input = quantize_input
        workspace.kernel.stage_inputs(
            t, workspace.workspace, quantize_input=quantize_input
        )

    def workspace_output(self, workspace):
        return workspace.workspace.output_activation

    def get_tuned_knobs(self, workspace=None):
        if workspace is None or workspace._tuning_state.winner is None:
            raise RuntimeError(
                "MegaMoE has no completed variant selection for this workspace"
            )
        return deepcopy(workspace._tuning_state.winner["knobs"])

    def get_tuned_config(self, workspace=None):
        knobs = self.get_tuned_knobs(workspace)
        return dataclasses.replace(
            self._config,
            kernel_variant=workspace._tuning_state.winner["variant"],
            knobs=knobs,
        )

    def compute(self, workspace, transformed_weights, *, output):
        context = self._context()
        state = workspace._tuning_state
        if torch.cuda.is_current_stream_capturing():
            state.graph_captured = True
        elif not prepared_for_serving(state, context):
            decisions = self._gather(
                (
                    context.tuning_enabled,
                    self._pending(state, context),
                    state.graph_captured,
                    context.skipped and state.winner is not None,
                )
            )
            if any(enabled != context.tuning_enabled for enabled, *_ in decisions):
                raise ValueError(
                    "EP ranks disagree on whether variant tuning is enabled"
                )
            select = any(pending for _, pending, _, _ in decisions)
            if select and any(captured for _, _, captured, _ in decisions):
                if context.tuning_enabled or any(
                    fallback for *_, fallback in decisions
                ):
                    raise RuntimeError(
                        "Create another explicit workspace before later variant selection"
                    )
                # Another replay context may retry an earlier miss. Captured
                # storage keeps its prepared fallback without another lookup.
                select = False
            if select:
                AutoTuner.get().choose_with_protocol(
                    self.variant_namespace,
                    self._protocol(workspace, transformed_weights, output),
                )
                # Selection can replace the staged storage.
                workspace.kernel.stage_inputs(
                    workspace.tensors,
                    workspace.workspace,
                    quantize_input=workspace.quantize_input,
                )
            record_prepared(state, context)
        workspace.tensors = None
        return workspace.kernel.compute(
            workspace.workspace, transformed_weights, output=output
        )

    def destroy(self, workspace):
        if workspace is not None:
            workspace.kernel.destroy(workspace.workspace)
