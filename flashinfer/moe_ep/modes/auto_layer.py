# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Collective selection of complete expert-parallel forwards."""

from __future__ import annotations

import copy
import dataclasses
import os
import weakref
from contextlib import ExitStack
from typing import Sequence

import torch
from torch import nn

from flashinfer.autotuner import AutoTuner, TuningResult
from flashinfer.autotuner.autotuner import _collect_metadata
from flashinfer.fused_moe.layer import NoEligibleBackend

from ..algo_knobs import AlgoKnob
from ..config import BootstrapConfig, FleetParams
from ..tensors import MoEEpTensors
from ..weights import MoEWeightPack
from ..backends.split.kernel.fused_moe.backend import FusedMoeSplitKernelBackend
from ..core.kernel.registry import create_mega_kernel
from ..core.runtime import (
    TORCH_DIST,
    bootstrap_moe_ep_runtime,
    finalize_moe_ep_runtime,
    ensure_moe_ep_cuda_device,
)
from ..core.selection import (
    BACKEND_NAMESPACE,
    NAMESPACE,
    Collective,
    NoEligibleCandidate,
    plain,
    record_key,
    tensor_layout,
)
from ..core.tuning_measurement import CaptureUnsupported, measurement_settings
from ..core.tuning_state import AutoTuningState, prepared_for_serving, record_prepared
from ..core.validation import validate_fleet_weights, validate_split_forward_inputs
from ..core.validation.common import validate_ll_hidden_size
from ..weights import UnquantizedMoEWeights
from .config import (
    AutoConfig,
    MegaConfig,
    SplitConfig,
    _validate_batch_invariant_candidates,
)
from .mega_layer import MoEEpMegaLayer
from .selection import (
    MegaPlan,
    SplitPlan,
    SplitProtocol,
    numerical_contract,
    valid_score,
)
from .split_layer import MoEEpSplitLayer


def _aliases_input(t):
    return (
        t.output is not None
        and t.output.numel() > 0
        and any(
            t.output.untyped_storage().data_ptr() == value.untyped_storage().data_ptr()
            for value in (t.hidden_states, t.topk_ids, t.topk_weights)
        )
    )


def _signature(t, row_bound=None):
    def layout(value):
        result = tensor_layout(value)
        if result is not None and row_bound is not None:
            result[1][0] = row_bound
        return result

    return [
        layout(t.hidden_states),
        layout(t.topk_ids),
        layout(t.topk_weights),
        layout(t.output),
        t.runtime_max_tokens_per_rank,
        _aliases_input(t),
        [
            name
            for name in (
                "scales",
                "fc1_alpha",
                "fc2_alpha",
                "fc1_norm_const",
                "fc1_activation_per_token_scale",
                "recv_count",
                "num_tokens_per_expert",
            )
            if getattr(t, name) is not None
        ],
    ]


def _common_candidate(config):
    if isinstance(config, SplitConfig) and hasattr(config.kernel, "moe_config"):
        moe = config.kernel.moe_config
        config = dataclasses.replace(
            config,
            kernel=dataclasses.replace(
                config.kernel,
                moe_config=dataclasses.replace(
                    moe, experts=dataclasses.replace(moe.experts, local_expert_offset=0)
                ),
            ),
        )
    return plain(config)


def _identity(context):
    # Child skip policies affect which implementations the outer score represents.
    return context.cache_identity, tuple(sorted(context.tuner._effective_skip_ops))


class MoEEpAutoWorkspace:
    """A prepared input profile and its selected EP plan.

    Later selection retains storage referenced by earlier captures. Destroy this
    handle collectively only after retiring every graph that used it. Live row
    counts may vary within the prepared bound; other input layouts stay fixed.
    """

    def __init__(self, layer, fleet, signature, *, row_bound=None):
        self._owner = weakref.ref(layer)
        self.fleet = fleet
        self.signature = signature
        self.row_bound = row_bound
        self._tuned_rows = None
        self._state = AutoTuningState()
        self._plan = None
        self._retained = []
        self._selection = None
        self._destroyed = False

    def _install(self, plan, selection):
        previous = self._plan
        if previous is not None:
            if self._state.graph_captured:
                self._retained.append(previous)
            else:
                previous.destroy()
        self._plan, self._selection = plan, selection
        self._state.graph_captured = False

    def destroy(self):
        """Release all plan storage after its graphs have been retired."""
        if self._destroyed:
            return
        try:
            with ExitStack() as cleanup:
                for plan in reversed([self._plan, *self._retained]):
                    if plan is not None:
                        cleanup.callback(plan.destroy)
        finally:
            self._plan = None
            self._retained.clear()
            self._destroyed = True


class MoEEpAutoLayer(nn.Module):
    """Tune explicit Mega and split candidates over the complete EP forward.

    Construct with canonical BF16 weights and candidates implementing the same
    quantization, activation and expert geometry. Inputs and routing tensors
    must be contiguous; expert IDs use int32/int64 and routing weights float32.
    FC1 rows use ``[up, gate]``
    order; candidate preparation converts the backend layout. Each split
    candidate pins its transport. Eligible compute tactics use dispatch,
    compute and combine, without a preliminary compute-only search.

    Call ``warmup`` collectively in ``autotune_v2()`` before capture. A disabled
    context loads a compatible plan or prepares the first eligible candidate's
    fallback. A workspace accepts live row counts up to its agreed
    ``runtime_max_tokens_per_rank``, or capacity when that hint is omitted.
    Ordinary serving reuses its plan without selector collectives. An enabled
    tuning context selects again for a changed live workload. Explicit
    workspaces let the caller release old captures and their storage independently.
    """

    def __init__(
        self,
        bootstrap: BootstrapConfig,
        fleet_params: FleetParams,
        weights: MoEWeightPack,
        config: AutoConfig,
        fleet_knobs: Sequence[AlgoKnob] = (),
    ) -> None:
        super().__init__()
        ensure_moe_ep_cuda_device(bootstrap)
        self._device = torch.device("cuda", torch.cuda.current_device())
        self._bootstrap, self._fleet, self._config = bootstrap, fleet_params, config
        self._runtime = (
            bootstrap_moe_ep_runtime(bootstrap, frozenset({TORCH_DIST}))
            if bootstrap.auto_bootstrap
            else None
        )
        self._collective = Collective(bootstrap)
        self._prototypes = {}
        self._workspaces: list[MoEEpAutoWorkspace] = []
        self._failed_plans: list[MegaPlan | SplitPlan] = []
        self._default: MoEEpAutoWorkspace | None = None
        self._destroyed = False
        c = self._collective
        try:
            c.prepare(lambda: _validate_batch_invariant_candidates(config.candidates))
            c.agree((len(config.candidates), config.candidate), "candidate selection")

            def validate_weights():
                if not isinstance(weights, UnquantizedMoEWeights) or any(
                    w.dtype != torch.bfloat16 or w.device != self._device or w.ndim != 3
                    for w in (weights.w13, weights.w2)
                ):
                    raise ValueError(
                        "EP selection requires canonical BF16 weights on the layer's CUDA device"
                    )
                validate_fleet_weights(weights, fleet_params, bootstrap.world_size)
                hidden = fleet_params.token_hidden_size
                if (
                    weights.w13.shape[1:] != (2 * weights.w2.shape[2], hidden)
                    or weights.w2.shape[1] != hidden
                ):
                    raise ValueError(
                        "EP weights must have FC1 [local_experts, 2 * intermediate, hidden] "
                        "and FC2 [local_experts, hidden, intermediate] shapes"
                    )

            c.prepare(validate_weights)
            c.agree(
                c.prepare(lambda: [_common_candidate(v) for v in config.candidates]),
                "candidate configurations",
            )
            c.agree(plain(fleet_params), "fleet parameters")
            c.agree(c.prepare(lambda: plain(fleet_knobs)), "fleet knobs")
            configs = c.prepare(lambda: plain(config.candidates))
            config = copy.deepcopy(config)
            self._config = config
            self._config_contract = c.gather(configs)
            self._fleet_knobs = c.gather(c.prepare(lambda: plain(fleet_knobs)))
            contracts, rejected = [], {}
            indices = (
                range(len(config.candidates))
                if config.candidate is None
                else [config.candidate]
            )
            for index in indices:
                candidate = config.candidates[index]
                prototype, error, fatal = None, None, False
                try:
                    if isinstance(candidate, MegaConfig):
                        backend = create_mega_kernel(candidate.megakernel)
                        backend.bind_ep_bootstrap(bootstrap)
                        backend.ep_tuning_contract(fleet_params)
                        native_weights = backend.ep_weight_pack(weights)
                        try:
                            prototype = MoEEpMegaLayer(
                                bootstrap, fleet_params, native_weights, candidate
                            )
                        finally:
                            del native_weights
                    else:
                        from .. import have_nccl_ep, have_nixl_ep

                        name = getattr(candidate.comm, "backend_name", candidate.comm)
                        if isinstance(name, str):
                            validate_ll_hidden_size(fleet_params, name)
                        available = {"nccl_ep": have_nccl_ep, "nixl_ep": have_nixl_ep}
                        if (
                            isinstance(name, str)
                            and name in available
                            and not available[name]()
                        ):
                            raise NotImplementedError(
                                f"Split transport {name} is not available"
                            )
                        prototype = MoEEpSplitLayer(
                            bootstrap,
                            fleet_params,
                            weights,
                            fleet_knobs,
                            backend=candidate,
                        )
                    contract = numerical_contract(candidate, prototype, fleet_params)
                    if isinstance(prototype, MoEEpSplitLayer):
                        assert isinstance(prototype._kernel, FusedMoeSplitKernelBackend)
                        prototype._kernel._ensure_compute(fleet_params)
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    fatal = not isinstance(
                        exc,
                        (ValueError, TypeError, NotImplementedError, NoEligibleBackend),
                    )
                errors = c.gather((error, fatal))
                if any(e[0] for e in errors):
                    if prototype is not None:
                        prototype.destroy()
                    if any(e[1] for e in errors):
                        raise RuntimeError(f"EP candidate preparation failed: {errors}")
                    rejected[index] = [e[0] for e in errors]
                    continue
                c.agree(contract, "candidate numerical contract")
                contracts.append(contract)
                self._prototypes[index] = prototype
            if not contracts:
                raise ValueError(f"No eligible EP candidate: {rejected}")
            if any(contract != contracts[0] for contract in contracts):
                raise ValueError("EP candidates must use the same numerical contract")
            self._numerical_contract = contracts[0]
            self._environment = c.gather(
                {
                    **_collect_metadata(),
                    "device": torch.cuda.get_device_name(),
                    "capability": list(torch.cuda.get_device_capability()),
                    "sm_count": torch.cuda.get_device_properties(
                        torch.cuda.current_device()
                    ).multi_processor_count,
                }
            )
            self._mega_revisions = c.gather(
                {
                    str(i): p._kernel.ep_tuning_contract(fleet_params)
                    for i, p in self._prototypes.items()
                    if isinstance(p, MoEEpMegaLayer)
                }
            )
            self.rejected_candidates = rejected
        except Exception:
            self.destroy()
            raise

    def _validate_device(self, tensors):
        if any(
            value is not None and value.device != self._device
            for value in (
                tensors.hidden_states,
                tensors.topk_ids,
                tensors.topk_weights,
                tensors.output,
            )
        ):
            raise ValueError("EP inputs and output must use the layer's CUDA device")

    def _validate(self, t, fleet):
        self._validate_device(t)
        validate_split_forward_inputs(
            t.hidden_states, t.topk_ids, t.topk_weights, fleet
        )
        if t.hidden_states.dtype != torch.bfloat16 or not t.hidden_states.is_cuda:
            raise ValueError("EP selection requires BF16 CUDA inputs")
        if t.topk_ids.ndim != 2 or t.topk_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("EP routing IDs must be a 2D int32 or int64 tensor")
        if t.topk_weights.dtype != torch.float32:
            raise ValueError("EP routing weights must use float32")
        if any(
            not value.is_contiguous()
            for value in (t.hidden_states, t.topk_ids, t.topk_weights)
        ):
            raise ValueError("EP inputs and routing tensors must be contiguous")
        if t.topk_ids.shape[1] != self._numerical_contract["top_k"]:
            raise ValueError(
                "Input routing top-k does not match the configured candidates"
            )
        for name in (
            "scales",
            "fc1_alpha",
            "fc2_alpha",
            "fc1_norm_const",
            "fc1_activation_per_token_scale",
            "recv_count",
            "num_tokens_per_expert",
        ):
            if getattr(t, name) is not None:
                raise ValueError(f"EP selection does not accept {name}")
        if t.output is not None and (
            t.output.shape != t.hidden_states.shape
            or t.output.dtype != torch.bfloat16
            or t.output.device != t.hidden_states.device
        ):
            raise ValueError(
                "output must match hidden_states shape, device and BF16 dtype"
            )
        if _aliases_input(t):
            raise ValueError("EP tuning output must not alias an input")
        if t.runtime_max_tokens_per_rank is not None and not (
            max(t.num_tokens, 1)
            <= t.runtime_max_tokens_per_rank
            <= fleet.max_tokens_per_rank
        ):
            raise ValueError(
                "runtime_max_tokens_per_rank must be positive and cover this batch within capacity"
            )

    def create_workspace(
        self, tensors: MoEEpTensors, *, max_tokens_per_rank: int | None = None
    ) -> MoEEpAutoWorkspace:
        """Collectively prepare a row bound using the current v2 tuning controls."""
        if self._destroyed or torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Create EP workspaces on a live layer before capture")
        fleet = dataclasses.replace(
            self._fleet,
            max_tokens_per_rank=(
                self._fleet.max_tokens_per_rank
                if max_tokens_per_rank is None
                else max_tokens_per_rank
            ),
        )
        self._collective.agree(plain(fleet), "workspace capacity and layout")
        self._collective.prepare(lambda: self._validate(tensors, fleet))
        row_bound = (
            fleet.max_tokens_per_rank
            if tensors.runtime_max_tokens_per_rank is None
            else tensors.runtime_max_tokens_per_rank
        )
        self._collective.agree(row_bound, "workspace row bound")
        workspace = MoEEpAutoWorkspace(
            self, fleet, _signature(tensors, row_bound), row_bound=row_bound
        )
        try:
            self._select(tensors, workspace)
        except Exception:
            workspace.destroy()
            raise
        self._workspaces.append(workspace)
        return workspace

    def _check(self, tensors, workspace):
        if not isinstance(workspace, MoEEpAutoWorkspace):
            raise TypeError("workspace must be created by this EP layer")
        if self._destroyed or workspace._destroyed or workspace._owner() is not self:
            raise RuntimeError("EP workspace does not belong to this live layer")
        self._validate_device(tensors)
        if (
            tensors.hidden_states.ndim != 2
            or tensors.num_tokens > workspace.row_bound
            or any(
                value is not None and value.shape[:1] != tensors.hidden_states.shape[:1]
                for value in (tensors.topk_ids, tensors.topk_weights, tensors.output)
            )
            or _signature(tensors, workspace.row_bound) != workspace.signature
        ):
            raise ValueError(
                "Input profile changed; collectively prepare another EP workspace"
            )

    def _select(self, tensors, workspace):
        tuner = AutoTuner.get()
        context = tuner.protocol_context(NAMESPACE)
        state = workspace._state
        if prepared_for_serving(state, context) and workspace._plan is not None:
            return
        identity = _identity(context)
        if torch.cuda.is_current_stream_capturing():
            if (
                workspace._plan is None
                or state.cache_identity != identity
                or (
                    context.tuning_enabled
                    and (
                        state.tuned_identity != identity
                        or workspace._tuned_rows != tensors.num_tokens
                    )
                )
            ):
                raise RuntimeError(
                    "Warm up EP selection under these controls before capture"
                )
            state.graph_captured = True
            return
        pending = (
            workspace._plan is None
            or state.cache_identity != identity
            or (
                context.tuning_enabled
                and (
                    state.tuned_identity != identity
                    or workspace._tuned_rows != tensors.num_tokens
                )
            )
            or (
                state.winner is None and state.lookup_scope != (context.scope, identity)
            )
        )
        requests = self._collective.gather(
            (context.tuning_enabled, context.skipped, pending)
        )
        if any(r[:2] != requests[0][:2] for r in requests):
            raise ValueError("EP ranks disagree on tuning controls")
        if any(r[2] for r in requests):
            tuner.choose_with_protocol(
                NAMESPACE, _OuterProtocol(self, workspace, tensors)
            )
        record_prepared(state, context)

    def warmup(
        self, tensors: MoEEpTensors, *, workspace: MoEEpAutoWorkspace | None = None
    ) -> None:
        """Prepare and execute a profile collectively, outside graph capture."""
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("EP warmup must happen before capture")
        if workspace is not None or self._default is not None:
            self._collective.prepare(
                lambda: self._check(tensors, workspace or self._default)
            )
        self.forward(tensors, workspace=workspace)
        torch.cuda.synchronize()

    def forward(
        self, tensors: MoEEpTensors, *, workspace: MoEEpAutoWorkspace | None = None
    ) -> torch.Tensor:
        """Execute the prepared plan; the workspace owns the returned output.

        Pass ``tensors.output`` for caller-owned output. Otherwise the next call
        on this workspace may overwrite the returned tensor.
        """
        if workspace is None:
            if self._default is None:
                self._default = self.create_workspace(tensors)
            workspace = self._default
        self._check(tensors, workspace)
        self._select(tensors, workspace)
        if torch.cuda.is_current_stream_capturing():
            workspace._state.graph_captured = True
        return workspace._plan.forward(tensors)

    def get_selected_plan(self, *, workspace: MoEEpAutoWorkspace | None = None) -> dict:
        """Return the candidate index and its fixed variant/knobs or compute tactic."""
        workspace = workspace or self._default
        if (
            self._destroyed
            or workspace is None
            or workspace._owner() is not self
            or workspace._destroyed
            or workspace._selection is None
        ):
            raise RuntimeError("Prepare an EP workspace before reading its plan")
        return copy.deepcopy(workspace._selection)

    def destroy(self):
        """Release this layer after retiring every graph that references it."""
        if self._destroyed:
            return
        try:
            with ExitStack() as cleanup:
                if self._runtime is not None:
                    cleanup.callback(finalize_moe_ep_runtime, self._runtime)
                for item in reversed(
                    [
                        *self._workspaces,
                        *self._failed_plans,
                        *self._prototypes.values(),
                    ]
                ):
                    cleanup.callback(item.destroy)
        finally:
            self._workspaces.clear()
            self._failed_plans.clear()
            self._prototypes.clear()
            self._runtime = None
            self._destroyed = True


class _OuterProtocol:
    def __init__(self, layer, workspace, tensors):
        self.layer, self.workspace, self.tensors = layer, workspace, tensors
        self.c = layer._collective

    def _new_plan(self, index, *, selected=None, fallback=False, force_tune=False):
        try:
            plan = self._prepare_plan(
                index, selected=selected, fallback=fallback, force_tune=force_tune
            )
        except NoEligibleCandidate as exc:
            self.layer.rejected_candidates[index] = str(exc)
            return None
        self.layer.rejected_candidates.pop(index, None)
        return plan

    def _prepare_plan(self, index, *, selected=None, fallback=False, force_tune=False):
        prototype, t = self.layer._prototypes[index], self.tensors
        output = self.c.prepare(
            lambda: torch.empty(
                self.workspace.row_bound,
                t.hidden_states.shape[1],
                dtype=t.hidden_states.dtype,
                device=t.hidden_states.device,
            )
        )
        if isinstance(prototype, MoEEpMegaLayer):
            plan = self.c.prepare(
                lambda: MegaPlan(
                    prototype,
                    self.workspace.fleet,
                    output,
                    selected=selected,
                    fallback=fallback,
                ),
                optional=True,
            )
            try:
                plan.prepare(t, force_tune=force_tune)
                if selected is None:
                    fixed_selection = plan.selection()
                    plan.destroy()
                    plan = self.c.prepare(
                        lambda: MegaPlan(
                            prototype,
                            self.workspace.fleet,
                            output,
                            selected=fixed_selection,
                        )
                    )
                    plan.prepare(t)
            except Exception:
                # A peer may still be inside the collective kernel. Do not
                # free its storage while propagating the launch failure.
                self.layer._failed_plans.append(plan)
                raise
        else:
            plan = self.c.prepare(
                lambda: SplitPlan(prototype, self.workspace.fleet, output, t),
                optional=True,
            )
            try:
                plan.discover(t)
                if selected is not None:
                    choices = self.c.prepare(plan.enumerate, optional=True)
                    tuner = AutoTuner.get()
                    valid = any(
                        v["runner"] == selected[0]
                        and (
                            selected[1] in plain(v["tactics"])
                            or (
                                selected[1] == -1
                                and (
                                    tuner.protocol_context(BACKEND_NAMESPACE).skipped
                                    or tuner.protocol_context(
                                        f"moe_{v['backend']}"
                                    ).skipped
                                )
                            )
                        )
                        for v in choices
                    )
                    if not all(self.c.gather(valid)):
                        raise NoEligibleCandidate(
                            "The cached split runner or tactic is not eligible for this profile"
                        )
                    SplitProtocol(
                        plan, t, self.c, self._split_contract, self._settings
                    ).check_replay(
                        tuner.protocol_context(BACKEND_NAMESPACE), selected, choices
                    )
                    self.c.prepare(lambda: plan.prepare(selected))
                    plan.warmup(t)
            except NoEligibleCandidate:
                plan.destroy()
                raise
            except Exception:
                self.layer._failed_plans.append(plan)
                raise
        return plan

    def run(self, context):
        if not context.skipped and (
            context.tuning_buckets is not None or context.round_up
        ):
            raise NotImplementedError(
                "EP selection uses explicit workspace profiles, not tuning_buckets or round_up"
            )
        layer, ws, c = self.layer, self.workspace, self.c
        layer.rejected_candidates = {
            index: reason
            for index, reason in layer.rejected_candidates.items()
            if index not in layer._prototypes
        }
        settings = measurement_settings(context)
        c.agree(settings, "measurement policy")
        c.agree(sorted(context.tuner._effective_skip_ops), "skip_ops")
        contract = plain(
            {
                "weights": "up_gate",
                "candidates": layer._config_contract,
                "forced": layer._config.candidate,
                "eligible": list(layer._prototypes),
                "fleet": ws.fleet,
                "fleet_knobs": layer._fleet_knobs,
                "inputs": c.gather(_signature(self.tensors)),
                "workspace": c.gather(ws.signature),
                "numerical": layer._numerical_contract,
                "environment": layer._environment,
                "implementations": layer._mega_revisions,
                "measurement": settings,
                "skip_ops": sorted(context.tuner._effective_skip_ops),
                "padding_exclusion": os.getenv(
                    "FLASHINFER_MOE_EP_EXCLUDE_PADDING_ROWS", "0"
                ),
                "quantization_environment": {
                    k: v
                    for k, v in os.environ.items()
                    if k.startswith("FLASHINFER_NVFP4_4OVER6")
                },
            }
        )
        self._settings = settings
        self._split_contract = {
            k: v
            for k, v in contract.items()
            if k not in ("candidates", "forced", "eligible", "implementations")
        }
        key = record_key(contract)

        def compatible(record):
            if not valid_score(record) or record.get("contract") != contract:
                return False
            index = record.get("candidate")
            if type(index) is not int or index not in layer._prototypes:
                return False
            selected = record.get("selected")
            if isinstance(layer._prototypes[index], MoEEpMegaLayer):
                if not (
                    isinstance(selected, dict)
                    and set(selected) == {"kernel_variant", "knobs"}
                    and isinstance(selected["kernel_variant"], str)
                    and isinstance(selected["knobs"], dict)
                ):
                    return False
                try:
                    layer._prototypes[index]._kernel.ep_validate_selection(
                        ws.fleet, selected
                    )
                except (ValueError, TypeError, NotImplementedError):
                    return False
                return True
            return (
                isinstance(selected, list)
                and len(selected) == 2
                and type(selected[0]) is int
            )

        cached = None if context.force_tune else c.lookup(context, key, compatible)
        plan, winner, measured = None, None, 0
        if cached is not None:
            plan = self._new_plan(cached["candidate"], selected=cached["selected"])
            if plan is not None:
                winner = cached
            else:
                cached = None
        if plan is None:
            try:
                for index in layer._prototypes:
                    trial = self._new_plan(
                        index, fallback=context.skipped, force_tune=context.force_tune
                    )
                    if trial is None:
                        continue
                    keep_trial, failed = False, False
                    try:
                        if isinstance(trial, SplitPlan):
                            try:
                                result = context.tuner.choose_with_protocol(
                                    BACKEND_NAMESPACE,
                                    SplitProtocol(
                                        trial,
                                        self.tensors,
                                        c,
                                        self._split_contract,
                                        settings,
                                        fallback=context.skipped,
                                    ),
                                    force_tune=context.force_tune,
                                )
                            except (NoEligibleCandidate, CaptureUnsupported) as exc:
                                layer.rejected_candidates[index] = str(exc)
                                continue
                            selected = result["selection"]
                        else:
                            selected = trial.selection()
                        entry = {
                            "candidate": index,
                            "selected": selected,
                            "contract": contract,
                        }
                        if not context.tuning_enabled or context.skipped:
                            winner, plan, keep_trial = entry, trial, True
                            break
                        try:
                            entry["p50_us"] = c.measure(
                                lambda: trial.forward(self.tensors),
                                self.tensors.hidden_states.device,
                                settings,
                                warmup=lambda: trial.warmup(self.tensors),
                            )
                        except CaptureUnsupported as exc:
                            layer.rejected_candidates[index] = str(exc)
                            continue
                        measured += 1
                        if winner is None or entry["p50_us"] < winner["p50_us"]:
                            winner = entry
                    except Exception:
                        failed = True
                        layer._failed_plans.append(trial)
                        raise
                    finally:
                        if not keep_trial and not failed:
                            trial.destroy()
                if winner is None:
                    raise RuntimeError(
                        f"No EP candidate completed measurement: {layer.rejected_candidates}"
                    )
                if plan is None:
                    plan = self._new_plan(
                        winner["candidate"], selected=winner["selected"]
                    )
                    if plan is None:
                        raise RuntimeError(
                            "The selected EP plan became ineligible during preparation"
                        )
                if context.tuning_enabled:
                    c.publish(context, key, winner)
            except Exception:
                if plan is not None:
                    plan.destroy()
                raise
        ws._install(
            plan, {"candidate": winner["candidate"], "selected": winner["selected"]}
        )
        identity = _identity(context)
        ws._state.record_selection(
            winner if valid_score(winner) else None,
            identity,
            identity if context.tuning_enabled else None,
        )
        ws._state.lookup_scope = (context.scope, identity)
        ws._tuned_rows = self.tensors.num_tokens if context.tuning_enabled else None
        return TuningResult(
            ws._selection,
            cache_hit=cached is not None,
            candidates=measured,
            successful_candidates=measured,
        )
