# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Complete-forward selection between Blackwell MegaMoE variants."""

from __future__ import annotations

import dataclasses
import math

import torch

from flashinfer.autotuner import TuningResult
from flashinfer.moe_ep.shims.sm100.cutedsl_megamoe.sources import knob_cache
from flashinfer.moe_ep.shims.sm100.cutedsl_megamoe.sources.autotune import (
    NoPassingCandidate,
    _prepare_collectively,
    _sweep_sm100_mega_moe,
)
from flashinfer.moe_ep.shims.sm100.cutedsl_megamoe.sources.measurement import (
    measurement_settings,
)
from flashinfer.moe_ep.shims.sm100.cutedsl_megamoe.sources.tuning import (
    knob_keys,
    sm100_genphase_candidates,
    sm100_standard_candidates,
)
from flashinfer.moe_ep.shims.sm100.cutedsl_megamoe.sources.tuning_protocol import (
    NAMESPACE as NATIVE_NAMESPACE,
    BlackwellTuningProtocol,
    validate_context,
)

from .....config import FinalizeMode
from .....core.selection import Collective, record_key
from .....core.validation.common import MoEEpConfigError
from ..variant_selection import VariantBackend
from .forward_tuning import ForwardExecution

NAMESPACE = "moe_ep.sm100.cutedsl.variants.v1"


def _workspace_type(variant):
    from flashinfer.moe_ep.shims.sm100.cutedsl_megamoe.sources import (
        Sm100BlockScaledSymmBuffer,
        Sm100GenphaseSymmBuffer,
    )

    return (
        Sm100GenphaseSymmBuffer if variant == "genphase" else Sm100BlockScaledSymmBuffer
    )


def _candidates(cfg, live_tokens, *, allow_nondeterministic=False):
    if cfg.kernel_variant == "genphase":
        return sm100_genphase_candidates(
            cfg,
            sm_count=torch.cuda.get_device_properties(
                torch.cuda.current_device()
            ).multi_processor_count,
            live_tokens=live_tokens,
        )
    return sm100_standard_candidates(cfg, allow_nondeterministic=allow_nondeterministic)


class BlackwellVariantBackend(VariantBackend):
    variant_namespace = NAMESPACE
    native_namespace = NATIVE_NAMESPACE
    _validate_context = staticmethod(validate_context)
    _measurement_settings = staticmethod(measurement_settings)

    def __init__(self, config, backend_type):
        if config.batch_invariant or config.finalize_mode is FinalizeMode.NONE:
            raise MoEEpConfigError(
                "Batch-invariant and unfinalized execution require an explicit kernel_variant"
            )
        if any(
            getattr(config, name) is not None
            for name in ("mma_tiler_mnk", "cluster_shape_mn", "schedule_policy")
        ):
            raise MoEEpConfigError(
                "GenPhase geometry overrides require kernel_variant='genphase'"
            )
        super().__init__(config, backend_type)

    def _protocol(self, workspace, weights, output):
        return _VariantProtocol(self, workspace, weights, output)

    def _eligible(self, fleet):
        configs = {}
        for variant in ("inference", "genphase"):
            cfg, error = None, None
            try:
                kernel = self._backend(variant)
                # GenPhase's collective configuration check belongs to
                # workspace preparation, after ranks agree on eligibility.
                cfg = (
                    kernel._genphase_config(fleet)
                    if variant == "genphase"
                    else kernel._workspace_config(fleet)
                )
                _workspace_type(variant)._build_kernel(cfg)
            except Exception as exc:
                error = (
                    f"{type(exc).__name__}: {exc}",
                    isinstance(exc, (TypeError, ValueError, NotImplementedError)),
                )
            errors = [error for error in self._gather(error) if error is not None]
            if not errors:
                configs[variant] = cfg
            elif any(not unsupported for _, unsupported in errors):
                raise RuntimeError(f"Blackwell {variant} preparation failed: {errors}")
            elif variant == "inference":
                raise MoEEpConfigError(f"Blackwell inference is unsupported: {errors}")
        return configs


class _VariantProtocol:
    def __init__(
        self,
        backend,
        owner,
        weights,
        output,
        *,
        candidates=None,
        warmup_iters=8,
        timed_iters=32,
        calls_per_sample=8,
        install_winner=True,
    ):
        if owner.tensors is None:
            raise RuntimeError("Stage inputs before selecting a variant")
        self.backend = backend
        self.owner = owner
        self.weights = weights
        self.output = output
        self.tensors = owner.tensors
        self.candidates = candidates
        self.warmup_iters = warmup_iters
        self.timed_iters = timed_iters
        self.calls_per_sample = calls_per_sample
        self.install_winner = install_winner
        self.collective = Collective(backend._ep_bootstrap)

    def _execution(self, kernel):
        return ForwardExecution(
            kernel,
            self.tensors,
            quantize_input=self.owner.quantize_input,
            output=self.output,
        )

    def _compatible(self, record, contract, configs, measurement, context):
        if not isinstance(record, dict) or record.get("contract") != contract:
            return False
        if context.tuning_enabled and record.get("measurement") != measurement:
            return False
        variant = record.get("variant")
        if not isinstance(variant, str) or variant not in configs:
            return False
        try:
            knobs = knob_cache._decode(record, configs[variant])
            allow_ikr = (
                variant == "inference"
                and contract["execution"]["allow_nondeterministic"]
            )
            if knobs is None or (
                knobs.get("reduce_topk_in_kernel", False) and not allow_ikr
            ):
                return False
            inner = record.get("knob_record")
            if inner is not None:
                key = contract["kernels"][variant]
                if (
                    not isinstance(inner, dict)
                    or contract["native_skipped"]
                    or any(inner.get(name) != value for name, value in key.items())
                    or inner.get("timing_boundary") != "complete_forward"
                    or inner.get("protocol_contract")
                    != {
                        "kernel": key,
                        "execution": contract["execution"]["layouts"],
                    }
                    or inner.get("live_tokens") != max(contract["inputs"])
                    or knob_cache.allows_nondeterministic(inner) != allow_ikr
                    or knob_cache._decode(inner, configs[variant]) != knobs
                    or not math.isfinite(inner["p50_us"])
                    or inner["p50_us"] <= 0
                ):
                    return False
            _workspace_type(variant)._build_kernel(
                dataclasses.replace(configs[variant], **knobs)
            )
        except Exception:
            # Every rank must report whether it can use this saved configuration.
            return False
        return True

    def _install(self, record, configs):
        variant = record["variant"]
        kernel = self.backend._backend(
            variant, knob_cache._decode(record, configs[variant])
        )
        workspace = kernel._allocate_workspace(self.owner.fleet)
        _prepare_collectively(
            lambda: kernel.ep_prepare_fixed(workspace, self.weights),
            self.collective.world,
            self.collective.group,
        )
        self.owner.kernel.destroy(self.owner.workspace)
        self.owner.kernel, self.owner.workspace = kernel, workspace

    def _restore_fallback(self, context):
        owner, backend = self.owner, self.backend
        state = owner._tuning_state
        if state.winner is not None:
            replacement = backend._allocate_workspace(owner.fleet)
            owner.kernel.destroy(owner.workspace)
            owner.kernel, owner.workspace = replacement.kernel, replacement.workspace
        identity = backend._cache_identity(context)
        state.record_selection(None, identity, None)
        state.lookup_scope = context.scope, identity
        return TuningResult(None)

    def _measure_variant(self, variant, native, live_tokens, measurement):
        backend, owner = self.backend, self.owner
        kernel = backend._backend(variant)
        workspace = kernel._allocate_workspace(owner.fleet)
        cfg = workspace.config
        execution = self._execution(kernel)

        def prepare_inputs():
            execution.prepare_trial_output()
            kernel.stage_inputs(
                self.tensors, workspace, quantize_input=owner.quantize_input
            )

        _prepare_collectively(
            prepare_inputs, self.collective.world, self.collective.group
        )
        counts = dict(
            num_tokens=self.tensors.num_tokens,
            warmup_iters=self.warmup_iters,
            timed_iters=self.timed_iters,
            calls_per_sample=self.calls_per_sample,
            process_group=self.collective.group,
            make_trial=execution.make_trial,
        )
        allow_nondeterministic = (
            variant == "inference" and backend._config.enable_in_kernel_fc2_reduce
        )
        knob_record = None
        try:
            if native.skipped:
                # Still compare the two heuristic variants when inner tuning is skipped.
                default = {name: getattr(cfg, name) for name in knob_keys(cfg)}
                knobs, score, _ = _sweep_sm100_mega_moe(
                    execution.output,
                    *self.weights,
                    workspace,
                    candidates=[default],
                    measurement=measurement,
                    **counts,
                )
            else:
                choices = (
                    self.candidates[variant]
                    if self.candidates is not None
                    else _candidates(
                        cfg, live_tokens, allow_nondeterministic=allow_nondeterministic
                    )
                )
                protocol = BlackwellTuningProtocol(
                    execution.output,
                    *self.weights,
                    workspace,
                    candidates=choices,
                    execution_contract=execution.contract(),
                    allow_nondeterministic=allow_nondeterministic,
                    remeasure_cached=True,
                    **counts,
                )
                knobs = protocol.run(native).value
                score = protocol.latency_us
                knob_record = protocol.selected_record
        except NoPassingCandidate:
            workspace.destroy()
            return None
        workspace.destroy()
        if knobs is None or score is None:
            raise RuntimeError("Blackwell variant measurement did not return a winner")
        result = {"variant": variant, "knobs": knobs, "p50_us": score}
        if knob_record is not None:
            result["knob_record"] = knob_record
        return result

    def run(self, context):
        backend, owner = self.backend, self.owner
        native = context.tuner.protocol_context(
            NATIVE_NAMESPACE, force_tune=context.force_tune
        )
        validate_context(context)
        measurement = measurement_settings(
            context, self.calls_per_sample, self.warmup_iters
        )
        self.collective.agree(
            (
                context.tuning_enabled,
                context.force_tune,
                context.skipped,
                context.disk_cache_enabled,
                native.skipped,
                measurement,
                self.candidates,
                self.timed_iters,
                self.install_winner,
            ),
            "Blackwell variant tuning controls",
        )
        if context.skipped:
            return self._restore_fallback(context)
        validate_context(native)
        if min(self.warmup_iters, self.timed_iters, self.calls_per_sample) < 1:
            raise ValueError("Variant tuning requires positive iteration counts")

        configs = backend._eligible(owner.fleet)
        policies = {variant: knob_cache._key(cfg) for variant, cfg in configs.items()}
        self.collective.agree(policies, "Blackwell variant workload or device")
        live_tokens = self.collective.gather(self.tensors.num_tokens)
        contract = {
            "kernels": policies,
            "inputs": live_tokens,
            "execution": {
                "allow_nondeterministic": backend._config.enable_in_kernel_fc2_reduce,
                "layouts": self.collective.gather(
                    self._execution(owner.kernel).contract()
                ),
            },
            "native_skipped": native.skipped,
            "measurement_policy": context.measurement_policy.manifest_fields()
            if context.measurement_policy is not None
            else {},
        }
        if backend._config.enable_in_kernel_fc2_reduce:
            # Earlier winners could reuse a narrower deterministic search.
            contract["ikr_search_version"] = 1
        key = record_key(contract)
        record = None
        if not context.force_tune:
            record = self.collective.lookup(
                context,
                key,
                lambda r: self._compatible(r, contract, configs, measurement, context),
            )
        cache_hit = record is not None
        successful = 0
        if record is None:
            if not context.tuning_enabled:
                return self._restore_fallback(context)
            options = []
            for variant in configs:
                result = self._measure_variant(
                    variant, native, max(live_tokens), measurement
                )
                if result is not None:
                    options.append(result)
            if not options:
                raise MoEEpConfigError(
                    "No Blackwell MegaMoE variant has a passing tuning candidate"
                )
            successful = len(options)
            record = {
                **min(options, key=lambda result: result["p50_us"]),
                "contract": contract,
                "measurement": measurement,
            }
        record = {
            **record,
            "knobs": knob_cache._decode(record, configs[record["variant"]]),
        }
        if self.install_winner:
            self._install(record, configs)
        self.collective.publish(context, key, record)
        if cache_hit and context.disk_cache_enabled and "knob_record" in record:

            def restore_knob_record():
                if self.collective.rank == 0:
                    knob_cache.write_record(record["knob_record"], overwrite=False)

            self.collective.prepare(restore_knob_record)
        if self.install_winner:
            state = owner._tuning_state
            state.record_selection(
                record,
                backend._cache_identity(context),
                backend._identity(context) if context.tuning_enabled else None,
            )
        return TuningResult(
            record,
            cache_hit=cache_hit,
            candidates=0 if cache_hit else len(configs),
            successful_candidates=successful,
        )
