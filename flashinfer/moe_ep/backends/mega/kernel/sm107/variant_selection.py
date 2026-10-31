"""Complete-forward selection between Rubin MegaMoE variants."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from copy import deepcopy
from typing import TYPE_CHECKING

import torch

from flashinfer.autotuner import AutoTuner, TuningContext, TuningResult

from .....core.kernel.base import MegaKernelBackend
from .....core.tuning_state import AutoTuningState
from .....core.validation.common import MoEEpConfigError
from .forward_tuning import ForwardExecution
from .automatic import prepared_for_serving, record_prepared

if TYPE_CHECKING:
    from .....config import FleetParams
    from .....tensors import MoEEpTensors
    from .....kernel_src.sm107.next_cutedsl_megamoe.shim.block_scaled import (
        Sm107BlockScaledSymmBuffer,
    )
    from .automatic import RubinBackend
    from .mxfp8_mxfp8_bf16_cutedsl.config import (
        Sm107_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig,
    )
    from .mxfp8_mxfp4_bf16_cutedsl.config import (
        Sm107_Mxfp8_Mxfp4_Bf16_Cutedsl_MegaMoeConfig,
    )
    from .nvfp4_nvfp4_bf16_cutedsl.config import (
        Sm107_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig,
    )
    from .nvfp4_nvfp4_bf16_cutedsl.weights import TransformedMegaWeights

NAMESPACE = "moe_ep.sm107.cutedsl.variants.v1"


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _context() -> TuningContext:
    from .....kernel_src.sm107.next_cutedsl_megamoe.shim.tuning_protocol import (
        validate_context,
    )

    context = AutoTuner.get().protocol_context(NAMESPACE)
    validate_context(context)
    return context


def _cache_identity(context: TuningContext) -> tuple:
    from .....kernel_src.sm107.next_cutedsl_megamoe.shim.tuning_protocol import (
        NAMESPACE as NATIVE_NAMESPACE,
    )

    return context.cache_identity, context.tuner.protocol_context(
        NATIVE_NAMESPACE
    ).skipped


def _identity(context: TuningContext) -> tuple:
    from .....core.tuning_measurement import measurement_settings

    return _cache_identity(context), measurement_settings(context)


def _pending(state: AutoTuningState, context: TuningContext) -> bool:
    if state.cache_identity != _cache_identity(context):
        return True
    if context.tuning_enabled:
        return state.tuned_identity != _identity(context)
    return state.winner is None and state.lookup_scope != (
        context.scope,
        _cache_identity(context),
    )


@dataclasses.dataclass
class VariantWorkspace:
    fleet: FleetParams
    kernel: RubinBackend
    workspace: Sm107BlockScaledSymmBuffer
    _tuning_state: AutoTuningState = dataclasses.field(default_factory=AutoTuningState)
    tensors: MoEEpTensors | None = None
    quantize_input: bool = True


class RubinVariantBackend(MegaKernelBackend):
    """Select a variant and its knobs by complete-forward latency."""

    _config: (
        Sm107_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig
        | Sm107_Mxfp8_Mxfp4_Bf16_Cutedsl_MegaMoeConfig
        | Sm107_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig
    )
    supports_output_view = True
    supports_auto_workspaces = True

    def __init__(self, config, backend_type):
        super().__init__(config)
        if config.knobs != "auto":
            raise MoEEpConfigError("kernel_variant='auto' requires knobs='auto'")
        self._backend_type = backend_type
        self._fallback = backend_type(
            dataclasses.replace(config, kernel_variant="inference")
        )

    @classmethod
    def kernel_name(cls):
        return NAMESPACE

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
        from .....kernel_src.sm107.next_cutedsl_megamoe import KNOB_KEYS

        knobs = {key: getattr(workspace.config, key) for key in KNOB_KEYS}
        return self._backend(workspace.config.kernel_variant, knobs)

    def _allocate_workspace(self, fleet_params):
        workspace = self._fallback.prepare_workspace(self._ep_bootstrap, fleet_params)
        return VariantWorkspace(fleet_params, self._fixed_backend(workspace), workspace)

    def _eligible(self, fleet):
        from .....kernel_src.sm107.next_cutedsl_megamoe import (
            Sm107BlockScaledSymmBuffer,
            sm107_candidates,
        )
        from .....kernel_src.sm107.next_cutedsl_megamoe.shim.autotune import (
            is_valid_sm107,
        )

        eligible = []
        for variant in ("inference", "genphase"):
            backend = self._backend(variant)
            error = None
            try:
                cfg = backend._resolved_config(fleet)
                valid = False
                for knobs in sm107_candidates(
                    cfg.quant_kind,
                    kernel_variant=variant,
                    allow_in_kernel_fc2_reduce=self._config.in_kernel_fc2_reduce
                    and cfg.combine_dtype == "bf16",
                ):
                    if not is_valid_sm107(knobs, cfg):
                        continue
                    try:
                        Sm107BlockScaledSymmBuffer._build_kernel(
                            dataclasses.replace(cfg, **knobs)
                        )
                    except (ValueError, NotImplementedError):
                        continue
                    valid = True
                    break
                if not valid:
                    error = "no native candidate supports this configuration"
            except (ValueError, MoEEpConfigError, NotImplementedError) as exc:
                error = str(exc)
            if not any(self._gather(error)):
                eligible.append(variant)
        if not eligible:
            raise MoEEpConfigError(
                "no Rubin MegaMoE variant supports this configuration"
            )
        return eligible

    def coordinate_workspace_replacement(self, workspace):
        context = _context()
        state = workspace._tuning_state
        if torch.cuda.is_current_stream_capturing():
            if state.cache_identity != _cache_identity(context):
                raise RuntimeError(
                    "Warm up variant selection after changing the autotune context"
                )
            return False
        if prepared_for_serving(state, context):
            return False
        replace = (
            state.graph_captured
            and _pending(state, context)
            and (
                context.tuning_enabled
                or state.cache_identity != _cache_identity(context)
            )
        )
        decisions = self._gather((context.tuning_enabled, replace))
        if any(enabled != context.tuning_enabled for enabled, _ in decisions):
            raise ValueError("EP ranks disagree on whether variant tuning is enabled")
        return any(replace for _, replace in decisions)

    def validate_capture_ready(self, workspace, weights):
        context = _context()
        if _pending(workspace._tuning_state, context) and context.tuning_enabled:
            raise RuntimeError(
                "Warm up pending variant tuning before CUDA graph capture"
            )
        workspace.workspace.prepare(*weights)

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
        context = _context()
        state = workspace._tuning_state
        if torch.cuda.is_current_stream_capturing():
            state.graph_captured = True
        elif not prepared_for_serving(state, context):
            decisions = self._gather(
                (
                    context.tuning_enabled,
                    _pending(state, context),
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
                    NAMESPACE,
                    _VariantProtocol(self, workspace, transformed_weights, output),
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


class _VariantProtocol:
    def __init__(
        self,
        backend: RubinVariantBackend,
        owner: VariantWorkspace,
        weights: TransformedMegaWeights,
        output: torch.Tensor | None,
    ):
        if owner.tensors is None:
            raise RuntimeError("Stage inputs before selecting a variant")
        self.tensors = owner.tensors
        self.backend, self.owner = backend, owner
        self.weights, self.output = weights, output

    def _contract(self, variants, context):
        from .....kernel_src.sm107.next_cutedsl_megamoe.shim.tuning_protocol import (
            RubinTuningProtocol,
        )

        owner = self.owner
        execution = ForwardExecution(
            owner.kernel,
            self.tensors,
            quantize_input=owner.quantize_input,
            output=self.output,
        )
        # Native compatibility covers numerical settings, SM budget and environment.
        native = RubinTuningProtocol(
            self.output,
            self.weights[0],
            self.weights[1],
            owner.workspace,
            num_tokens=self.tensors.num_tokens,
            process_group=self.backend.ep_comm_group
            if self.backend.ep_world_size > 1
            else None,
            make_trial=execution.make_trial,
            execution_contract=execution.contract(),
        )._contract()
        native["config"].pop("kernel_variant")
        return {
            "native": native,
            "timing_boundary": "complete_forward_candidates",
            "variants": variants,
            "allow_nondeterministic": self.backend._config.in_kernel_fc2_reduce,
            "native_skipped": _cache_identity(context)[1] and not context.force_tune,
            "capacity": owner.fleet.max_tokens_per_rank,
            "inputs": self.backend._gather(self.tensors.num_tokens),
            "measurement_policy": context.measurement_policy.manifest_fields()
            if context.measurement_policy
            else {},
        }

    def _install(self, record):
        from .....kernel_src.sm107.next_cutedsl_megamoe.shim.autotune import (
            _prepare_collectively,
        )

        backend, owner = self.backend, self.owner
        kernel = backend._backend(record["variant"], record["knobs"])
        workspace = kernel._allocate_workspace(owner.fleet)
        _prepare_collectively(
            lambda: workspace.prepare(*self.weights),
            backend.ep_world_size,
            backend.ep_comm_group if backend.ep_world_size > 1 else None,
        )
        owner.kernel.destroy(owner.workspace)
        owner.kernel, owner.workspace = kernel, workspace

    def _compatible(self, record, contract, measurement, context):
        if not isinstance(record, dict) or record.get("contract") != contract:
            return False
        if context.tuning_enabled and record.get("measurement") != measurement:
            return False
        if record.get("variant") not in contract["variants"]:
            return False
        try:
            from .....kernel_src.sm107.next_cutedsl_megamoe import (
                Sm107BlockScaledSymmBuffer,
            )
            from .....kernel_src.sm107.next_cutedsl_megamoe.shim.knob_cache import (
                _knobs_from_json,
            )

            knobs = _knobs_from_json(record["knobs"])
            knob_record = record.get("knob_record")
            if knob_record is not None and (
                not isinstance(knob_record, dict)
                or _knobs_from_json(knob_record["knobs"]) != knobs
            ):
                return False
            kernel = self.backend._backend(record["variant"], knobs)
            cfg = kernel._resolved_config(self.owner.fleet)
            if (
                cfg.reduce_topk_in_kernel
                and not self.backend._config.in_kernel_fc2_reduce
            ):
                return False
            Sm107BlockScaledSymmBuffer._build_kernel(cfg)
            return True
        except Exception as exc:
            from flashinfer.jit.core import logger

            # Cached resource checks do not allocate or launch collective kernels.
            logger.warning(f"[Autotuner]: Ignoring invalid MegaMoE variant: {exc}")
            return False

    def run(self, context: TuningContext):
        from .....core.tuning_measurement import measurement_settings
        from .....kernel_src.sm107.next_cutedsl_megamoe.shim.tuning_protocol import (
            NAMESPACE as NATIVE_NAMESPACE,
            RubinTuningProtocol,
            _publish_record,
        )

        backend, owner = self.backend, self.owner
        state = owner._tuning_state
        measurement = measurement_settings(context)
        native_context = context.tuner.protocol_context(
            NATIVE_NAMESPACE, force_tune=context.force_tune
        )
        controls = (
            context.tuning_enabled,
            context.force_tune,
            context.skipped,
            context.disk_cache_enabled,
            _cache_identity(context)[1],
            measurement,
        )
        if any(value != controls for value in backend._gather(controls)):
            raise ValueError("EP ranks disagree on variant tuning controls")
        if context.skipped:
            if state.winner is not None:
                replacement = backend._allocate_workspace(owner.fleet)
                owner.kernel.destroy(owner.workspace)
                owner.kernel, owner.workspace = (
                    replacement.kernel,
                    replacement.workspace,
                )
            state.lookup_scope = (context.scope, _cache_identity(context))
            state.record_selection(None, _cache_identity(context), None)
            return TuningResult(None)
        variants = backend._eligible(owner.fleet)
        contract = self._contract(variants, context)
        if any(value != contract for value in backend._gather(contract)):
            raise ValueError(
                "EP ranks disagree on variant geometry or numerical policy"
            )
        key = hashlib.sha256(_json(contract).encode()).hexdigest()
        record = None
        if not context.force_tune:
            records = backend._gather(context.lookup(key))
            valid = [
                self._compatible(value, contract, measurement, context)
                for value in records
            ]
            validity = backend._gather(valid)
            record = next(
                (
                    value
                    for i, value in enumerate(records)
                    if all(rank[i] for rank in validity)
                ),
                None,
            )
        cache_hit = record is not None
        inner_records = []
        if record is None and not context.tuning_enabled:
            if state.winner is not None:
                replacement = backend._allocate_workspace(owner.fleet)
                owner.kernel.destroy(owner.workspace)
                owner.kernel, owner.workspace = (
                    replacement.kernel,
                    replacement.workspace,
                )
            state.lookup_scope = (context.scope, _cache_identity(context))
            state.record_selection(None, _cache_identity(context), None)
            return TuningResult(None)
        if record is None:
            from .....kernel_src.sm107.next_cutedsl_megamoe import KNOB_KEYS
            from .....kernel_src.sm107.next_cutedsl_megamoe.shim import knob_cache
            from .....kernel_src.sm107.next_cutedsl_megamoe.shim.autotune import (
                NoPassingCandidate,
                _sweep_sm107_block_scaled_mega_moe,
                sm107_candidates,
            )

            candidates = []
            group = backend.ep_comm_group if backend.ep_world_size > 1 else None
            live_tokens = max(backend._gather(self.tensors.num_tokens))

            for variant in variants:
                kernel = backend._backend(variant)
                workspace = kernel._allocate_workspace(owner.fleet)
                execution = ForwardExecution(
                    kernel,
                    self.tensors,
                    quantize_input=owner.quantize_input,
                    output=self.output,
                )
                kernel.stage_inputs(
                    self.tensors, workspace, quantize_input=owner.quantize_input
                )
                # Skipping inner tuning still permits comparing the variants'
                # heuristic configurations, with the same full-forward objective.
                knobs_to_try = (
                    [{name: getattr(workspace.config, name) for name in KNOB_KEYS}]
                    if contract["native_skipped"]
                    else sm107_candidates(
                        workspace.config.quant_kind,
                        kernel_variant=variant,
                        allow_in_kernel_fc2_reduce=backend._config.in_kernel_fc2_reduce
                        and workspace.config.combine_dtype == "bf16",
                    )
                )
                protocol = RubinTuningProtocol(
                    self.output,
                    self.weights[0],
                    self.weights[1],
                    workspace,
                    num_tokens=self.tensors.num_tokens,
                    candidates=knobs_to_try,
                    allow_nondeterministic=backend._config.in_kernel_fc2_reduce,
                    process_group=group,
                    make_trial=execution.make_trial,
                    execution_contract=execution.contract(),
                )
                query = protocol._query(live_tokens)
                native_contract = protocol._contract()
                knob_record = protocol._lookup_cached(
                    query, native_contract, native_context, measurement
                )
                candidate_sets = [knobs_to_try]
                if knob_record is not None:
                    # Recheck and time cached knobs at this workload. A bucket
                    # fallback's old score cannot select the overall winner.
                    candidate_sets.insert(
                        0, [knob_cache._knobs_from_json(knob_record["knobs"])]
                    )
                for choices in candidate_sets:
                    try:
                        knobs, score, _ = _sweep_sm107_block_scaled_mega_moe(
                            self.output,
                            self.weights[0],
                            self.weights[1],
                            workspace,
                            num_tokens=self.tensors.num_tokens,
                            candidates=choices,
                            process_group=group,
                            measurement=measurement,
                            make_trial=execution.make_trial,
                        )
                        break
                    except NoPassingCandidate:
                        knob_record = None
                else:
                    kernel.destroy(workspace)
                    continue
                if knob_record is None:
                    knob_record = knob_cache.make_record(
                        knobs, **query, p50_us=score, source="autotune-complete-forward"
                    )
                    knob_record["timing_boundary"] = "complete_forward"
                    knob_record["protocol_contract"] = native_contract
                    knob_record["measurement_policy"] = (
                        context.measurement_policy.manifest_fields()
                        if context.measurement_policy
                        else {}
                    )
                    knob_record["profiling_policy"] = measurement
                # Preserve a reused record's original capacity, live count and
                # score; measuring one cached choice is not a new knob search.
                inner_records.append(knob_record)
                candidates.append(
                    {
                        "variant": variant,
                        "knobs": knobs,
                        "p50_us": score,
                        "knob_record": knob_record,
                    }
                )
                kernel.destroy(workspace)
            if not candidates:
                raise MoEEpConfigError(
                    "no Rubin MegaMoE variant has a passing tuning candidate"
                )
            record = {
                **min(candidates, key=lambda candidate: candidate["p50_us"]),
                "contract": contract,
                "measurement": measurement,
            }
        # Managed JSON decoding preserves payload lists; native knobs use tuples.
        from .....kernel_src.sm107.next_cutedsl_megamoe.shim import knob_cache

        record = deepcopy(record)
        record["knobs"] = knob_cache._knobs_from_json(record["knobs"])
        self._install(record)
        error = None
        if backend.ep_rank == 0:
            try:
                if not native_context.skipped:
                    for inner_record in inner_records:
                        _publish_record(native_context, inner_record)
                context.publish(key, record)
                if (
                    (cache_hit or native_context.skipped)
                    and context.disk_cache_enabled
                    and record.get("knob_record") is not None
                ):
                    from flashinfer.jit.core import logger

                    try:
                        knob_cache.write_record(record["knob_record"])
                    except Exception as exc:
                        logger.warning(
                            f"[Autotuner]: Cannot write MegaMoE cache file: {exc}"
                        )
            except Exception as exc:
                error = str(exc)
        errors = backend._gather(error)
        if any(errors):
            raise RuntimeError(f"Variant result publication failed: {errors}")
        state.record_selection(
            record,
            _cache_identity(context),
            _identity(context) if context.tuning_enabled else None,
        )
        return TuningResult(
            record,
            cache_hit=cache_hit,
            candidates=0 if cache_hit else len(variants),
            successful_candidates=0 if cache_hit else len(candidates),
        )
