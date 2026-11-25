"""Complete-forward selection between Rubin MegaMoE variants."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from copy import deepcopy
from typing import TYPE_CHECKING

from flashinfer.autotuner import TuningContext, TuningResult
from flashinfer.moe_ep.shims.sm107.cutedsl_megamoe.tuning_protocol import (
    NAMESPACE as NATIVE_NAMESPACE,
)
from flashinfer.moe_ep.shims.sm107.cutedsl_megamoe.tuning_protocol import (
    validate_context,
)

from .....core.tuning_state import prepared_for_serving as prepared_for_serving
from .....core.validation.common import MoEEpConfigError
from ..variant_selection import VariantBackend, VariantWorkspace
from .forward_tuning import ForwardExecution
from .shared_expert import Sm107SharedExpertWeights

if TYPE_CHECKING:
    import torch

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


class RubinVariantBackend(VariantBackend):
    """Select a variant and its knobs by complete-forward latency."""

    _config: (
        Sm107_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig
        | Sm107_Mxfp8_Mxfp4_Bf16_Cutedsl_MegaMoeConfig
        | Sm107_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig
    )
    supports_shared_expert = True
    variant_namespace = NAMESPACE
    native_namespace = NATIVE_NAMESPACE
    _validate_context = staticmethod(validate_context)

    def _protocol(self, workspace, weights, output):
        return _VariantProtocol(self, workspace, weights, output)

    def _prepare_capture(self, workspace, weights):
        workspace.kernel.ep_prepare_fixed(workspace.workspace, weights)

    def _eligible(self, fleet):
        from flashinfer.moe_ep.shims.sm107.cutedsl_megamoe import (
            Sm107BlockScaledSymmBuffer,
            sm107_candidates,
        )
        from flashinfer.moe_ep.shims.sm107.cutedsl_megamoe.autotune import (
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
                    combine_dedup=cfg.combine_dedup,
                    token_back_mode=cfg.token_back_mode,
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


_context = RubinVariantBackend._context
_cache_identity = RubinVariantBackend._cache_identity
_identity = RubinVariantBackend._identity
_pending = RubinVariantBackend._pending


class _VariantProtocol:
    def __init__(
        self,
        backend: RubinVariantBackend,
        owner: VariantWorkspace,
        weights: TransformedMegaWeights | Sm107SharedExpertWeights,
        output: torch.Tensor | None,
        *,
        candidates: dict[str, list[dict]] | None = None,
        warmup_iters: int = 3,
        timed_iters: int = 10,
        install_winner: bool = True,
    ):
        if owner.tensors is None:
            raise RuntimeError("Stage inputs before selecting a variant")
        self.tensors = owner.tensors
        self.backend, self.owner = backend, owner
        self.weights, self.output = weights, output
        self.candidates = candidates
        self.warmup_iters, self.timed_iters = warmup_iters, timed_iters
        self.install_winner = install_winner

    def _contract(self, variants, context):
        from flashinfer.moe_ep.shims.sm107.cutedsl_megamoe.tuning_protocol import (
            RubinTuningProtocol,
        )

        owner = self.owner
        execution = ForwardExecution(
            owner.kernel,
            self.tensors,
            quantize_input=owner.quantize_input,
            output=self.output,
            shared_weights=getattr(self.weights, "shared", None),
        )
        # Native compatibility covers numerical settings, SM budget and environment.
        if isinstance(self.weights, Sm107SharedExpertWeights):
            owner.workspace.bind_shared_weights(self.weights.shared)
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
            # Earlier outer winners could come from another workload's knobs.
            "search_version": 2,
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
        from flashinfer.moe_ep.shims.sm107.cutedsl_megamoe.autotune import (
            _prepare_collectively,
        )

        backend, owner = self.backend, self.owner
        kernel = backend._backend(record["variant"], record["knobs"])
        workspace = kernel._allocate_workspace(owner.fleet)
        _prepare_collectively(
            lambda: kernel.ep_prepare_fixed(workspace, self.weights),
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
            from flashinfer.moe_ep.shims.sm107.cutedsl_megamoe import (
                Sm107BlockScaledSymmBuffer,
            )
            from flashinfer.moe_ep.shims.sm107.cutedsl_megamoe.knob_cache import (
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
        from flashinfer.moe_ep.shims.sm107.cutedsl_megamoe.tuning_protocol import (
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
            self.candidates,
            self.warmup_iters,
            self.timed_iters,
            self.install_winner,
        )
        if any(value != controls for value in backend._gather(controls)):
            raise ValueError("EP ranks disagree on variant tuning controls")
        if self.warmup_iters < 1 or self.timed_iters < 1:
            raise ValueError("variant tuning requires positive iteration counts")
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
            from flashinfer.moe_ep.shims.sm107.cutedsl_megamoe import KNOB_KEYS
            from flashinfer.moe_ep.shims.sm107.cutedsl_megamoe import knob_cache
            from flashinfer.moe_ep.shims.sm107.cutedsl_megamoe.autotune import (
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
                    shared_weights=getattr(self.weights, "shared", None),
                )
                if isinstance(self.weights, Sm107SharedExpertWeights):
                    workspace.bind_shared_weights(self.weights.shared)
                kernel.stage_inputs(
                    self.tensors, workspace, quantize_input=owner.quantize_input
                )
                # Skipping inner tuning still permits comparing the variants'
                # heuristic configurations, with the same full-forward objective.
                knobs_to_try = (
                    [{name: getattr(workspace.config, name) for name in KNOB_KEYS}]
                    if contract["native_skipped"]
                    else self.candidates[variant]
                    if self.candidates is not None
                    else sm107_candidates(
                        workspace.config.quant_kind,
                        kernel_variant=variant,
                        allow_in_kernel_fc2_reduce=backend._config.in_kernel_fc2_reduce
                        and workspace.config.combine_dtype == "bf16",
                        **(
                            dict(
                                combine_dedup=True,
                                token_back_mode=workspace.config.token_back_mode,
                            )
                            if workspace.config.combine_dedup
                            else {}
                        ),
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
                    # Remeasure each variant's cached winner before comparing them.
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
                            warmup_iters=self.warmup_iters,
                            timed_iters=self.timed_iters,
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
                # Remeasuring one cached choice does not replace its search score.
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
        from flashinfer.moe_ep.shims.sm107.cutedsl_megamoe import knob_cache

        record = deepcopy(record)
        record["knobs"] = knob_cache._knobs_from_json(record["knobs"])
        if self.install_winner:
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
        if self.install_winner:
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
