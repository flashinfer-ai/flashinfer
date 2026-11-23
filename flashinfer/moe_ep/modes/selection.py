"""Prepared Mega and split candidates for the complete EP selector."""

from __future__ import annotations

import dataclasses
import math
from contextlib import contextmanager

import torch

from flashinfer.autotuner import TuningResult

from ..core.selection import Collective, NoEligibleCandidate, plain, record_key, tuples
from ..core.tuning_measurement import CaptureUnsupported
from .config import MegaConfig, SplitConfig, _batch_invariant_numerical_contract
from .mega_layer import MoEEpMegaLayer
from .split_layer import MoEEpSplitLayer


def numerical_contract(config, prototype, fleet):
    if isinstance(config, MegaConfig):
        if not config.quantize_input or not config.preprocess_weights:
            raise ValueError(
                "EP selection currently requires canonical BF16 inputs and weights"
            )
        k = config.megakernel
        if not getattr(k, "batch_invariant", False) and (
            getattr(k, "input_norm_const", 1.0) != 1.0
            or any(
                getattr(k, name, None) is not None
                for name in ("fc1_alpha", "fc2_alpha", "fc1_norm_const")
            )
        ):
            raise ValueError(
                "EP selection does not support Mega-only scaling overrides"
            )
        contract = prototype._kernel.ep_tuning_contract(fleet).copy()
        contract.pop("revision")
        batch_invariant = _batch_invariant_numerical_contract(config)
        if batch_invariant is not None:
            contract["batch_invariant"] = batch_invariant
        return contract

    from flashinfer.fused_moe.api import QuantFormat, SiTU, SwiGLU
    from flashinfer.quantization.nvfp4_quantization_utils import resolve_nvfp4_4over6
    from flashinfer.tllm_enums import DEFAULT_SWIGLU_LIMIT

    if not hasattr(config.kernel, "moe_config"):
        raise NotImplementedError(
            "EP selection requires a FusedMoeKernelConfig for split candidates"
        )
    cfg = config.kernel.moe_config
    if (
        cfg.experts.num_fused_shared_experts
        or cfg.routing.routed_scaling_factor not in (None, 1.0)
    ):
        raise ValueError(
            "EP selection requires routed experts with unit output scaling"
        )
    if cfg.quant.output is not QuantFormat.BF16 or not cfg.finalize.do_finalize:
        raise ValueError("EP selection requires finalized BF16 output")
    kinds = {
        (QuantFormat.NVFP4, QuantFormat.NVFP4): "nvfp4",
        (QuantFormat.MXFP4, QuantFormat.MXFP8): "mxfp4_mxfp8",
        (QuantFormat.MXFP4, QuantFormat.MXFP4): "mxfp4",
        (QuantFormat.MXFP8, QuantFormat.MXFP8): "mxfp8_e4m3",
        (QuantFormat.BF16, QuantFormat.BF16): "bf16",
        (QuantFormat.DeepSeekFp8, QuantFormat.DeepSeekFp8): "deepseek_fp8",
    }
    act = cfg.activation
    if isinstance(act, SwiGLU):
        if act.alpha != 1.0 or act.beta != 0.0:
            raise ValueError("EP selection requires standard SwiGLU alpha and beta")
        activation, clamp, beta, linear = (
            "swiglu",
            (None if act.limit == DEFAULT_SWIGLU_LIMIT else act.limit),
            None,
            None,
        )
    elif isinstance(act, SiTU):
        activation, clamp, beta, linear = (
            "situ",
            act.clamp_limit,
            act.gate_scale,
            act.linear_scale,
        )
    else:
        raise ValueError("EP selection supports SwiGLU and SiTU")
    return {
        "quantization": kinds.get(cfg.quant.pair, str(cfg.quant.pair)),
        "intermediate": cfg.experts.intermediate_size,
        "top_k": cfg.routing.top_k,
        "activation": activation,
        "gate_up_clamp": clamp,
        "situ_beta": beta,
        "situ_linear_beta": linear,
        "per_token_activation": bool(cfg.quant.per_token_scale),
        "nvfp4_4over6": (
            plain(resolve_nvfp4_4over6(cfg.quant.nvfp4_4over6))
            if cfg.quant.activation is QuantFormat.NVFP4
            else None
        ),
        "apply_routing_weights_before_fc2": False,
        "combine_dtype": "bf16",
    }


class MegaPlan:
    def __init__(self, prototype, fleet, output, *, selected=None, fallback=False):
        cfg = prototype._mega_config
        kernel = cfg.megakernel
        if selected is not None:
            kernel = dataclasses.replace(kernel, **tuples(selected))
        elif fallback and kernel.knobs == "auto":
            kernel = prototype._kernel.ep_tuning_fallback()
        elif kernel.knobs == "auto" and not prototype._kernel.supports_auto_workspaces:
            from flashinfer.autotuner import AutoTuner
            from ..core.selection import NAMESPACE

            if not AutoTuner.get().protocol_context(NAMESPACE).tuning_enabled:
                kernel = prototype._kernel.ep_tuning_fallback()
        self.layer = MoEEpMegaLayer(
            dataclasses.replace(prototype._bootstrap, auto_bootstrap=False),
            fleet,
            None,
            dataclasses.replace(
                cfg,
                megakernel=kernel,
                preprocess_weights=False,
                transformed_weights=prototype._get_transformed_weights(),
            ),
        )
        self.output = output

    def forward(self, tensors):
        return self.layer.forward(
            dataclasses.replace(
                tensors,
                output=(
                    tensors.output
                    if tensors.output is not None
                    else self.output[: tensors.num_tokens]
                ),
            )
        )

    def prepare(self, tensors, *, force_tune=False):
        if self.layer._megakernel_config.knobs != "auto":

            def prepare_fixed():
                if self.layer._workspace is None:
                    # Plans own their scratch independently of the process pool.
                    self.layer._workspace = self.layer._kernel._allocate_workspace(
                        self.layer._fleet_params
                    )
                self.layer.stage_inputs(tensors)
                self.layer._kernel.ep_prepare_fixed(
                    self.layer._workspace, self.layer._get_transformed_weights()
                )

            Collective(self.layer._bootstrap).prepare(prepare_fixed)
            self.layer.compute_staged(
                output=(
                    tensors.output
                    if tensors.output is not None
                    else self.output[: tensors.num_tokens]
                )
            )
        elif force_tune:
            self.layer._kernel.ep_force_tune(
                self.layer._ensure_workspace(),
                tensors,
                self.layer._get_transformed_weights(),
                (
                    tensors.output
                    if tensors.output is not None
                    else self.output[: tensors.num_tokens]
                ),
            )
        else:
            self.forward(tensors)

    def selection(self):
        layer = self.layer
        cfg = layer._kernel.ep_selected_config(layer._ensure_workspace())
        result = {"knobs": plain(cfg.knobs)}
        if hasattr(cfg, "kernel_variant"):
            result["kernel_variant"] = cfg.kernel_variant
        return result

    def warmup(self, tensors):
        layer = self.layer
        Collective(layer._bootstrap).prepare(lambda: layer.stage_inputs(tensors))
        return layer.compute_staged(
            output=(
                tensors.output
                if tensors.output is not None
                else self.output[: tensors.num_tokens]
            )
        )

    def destroy(self):
        self.layer.destroy()


class SplitPlan:
    def __init__(self, prototype, fleet, output, tensors):
        self.layer = MoEEpSplitLayer(
            dataclasses.replace(prototype._bootstrap, auto_bootstrap=False),
            fleet,
            None,
            prototype._fleet_knobs,
            backend=SplitConfig(prototype._comm_backend, prototype._kernel_config),
            _prepared_weights=prototype._kernel._transformed_weights,
        )
        self.output = output
        self.state = None
        self.context = None
        self.choices = None
        self._staging = None

    def _stage(self, tensors, *, force=False):
        if self.layer._uses_communication():
            return tensors
        bound = self.output.shape[0]
        if not force and tensors.num_tokens == bound:
            return tensors
        if self._staging is None:
            from flashinfer.cute_dsl import is_cute_dsl_available

            self._fused_stage = None
            if is_cute_dsl_available():
                from ..core.bf16_stage import fused_bf16_stage

                self._fused_stage = fused_bf16_stage
            self._staging = dataclasses.replace(
                tensors,
                hidden_states=torch.empty_like(self.output),
                topk_ids=torch.empty(
                    bound,
                    tensors.topk_ids.shape[1],
                    dtype=tensors.topk_ids.dtype,
                    device=tensors.topk_ids.device,
                ),
                topk_weights=torch.empty(
                    bound,
                    tensors.topk_weights.shape[1],
                    dtype=tensors.topk_weights.dtype,
                    device=tensors.topk_weights.device,
                ),
                output=None,
            )
        # Fleet handles require fixed routing rows. Mask the entire tail on
        # each call, including after a graph replay with a larger live batch.
        staged = self._staging
        if self._fused_stage is None or not self._fused_stage(
            tensors.hidden_states,
            tensors.topk_ids,
            tensors.topk_weights,
            staged.hidden_states,
            staged.topk_ids,
            staged.topk_weights,
        ):
            n = tensors.num_tokens
            staged.hidden_states[:n].copy_(tensors.hidden_states)
            staged.topk_ids[:n].copy_(tensors.topk_ids)
            staged.topk_weights[:n].copy_(tensors.topk_weights)
            staged.topk_ids[n:].fill_(-1)
        return staged

    def discover(self, tensors):
        c = Collective(self.layer._bootstrap)
        bound = c.prepare(lambda: self._stage(tensors, force=True))
        self.state = c.prepare(
            lambda: self.layer.create_graph_state(
                bound, out=self.output[: bound.num_tokens]
            ),
            optional=True,
        )

        # Complete the transport round trip even though setup only needs its
        # received tensors. Some transports reset their state during combine.
        def receive(ctx):
            self.context = ctx
            return torch.zeros(
                *ctx.expert_tensors.shape[:2],
                ctx.fleet_params.token_hidden_size,
                device=ctx.expert_tensors.device,
                dtype=torch.bfloat16,
            )

        self._warmup(tensors, compute=receive)
        return self.context

    @contextmanager
    def _prepare_stage(self, stage):
        if stage not in ("quantize", "compute"):
            yield
            return
        failure = None
        try:
            yield
        except Exception as exc:
            failure = exc

        def check():
            if failure is not None:
                raise failure

        Collective(self.layer._bootstrap).prepare(check)

    def _warmup(self, tensors, *, compute=None):
        # Coordinate local packing and compute before dispatch and combine.
        # Transport launch failures remain fatal; their peers may be blocked.
        previous = self.layer.stage_hook
        self.layer.stage_hook = self._prepare_stage
        try:
            staged = Collective(self.layer._bootstrap).prepare(
                lambda: self._stage(tensors)
            )
            return self.layer._forward_prepared(
                staged,
                self.state,
                compute=compute,
                output=self.output[: staged.num_tokens],
            )
        finally:
            self.layer.stage_hook = previous

    def warmup(self, tensors):
        return self._warmup(tensors)

    def enumerate(self):
        self.choices = self.layer._kernel.tuning_candidates(self.context)
        return self.choices

    def backend_config(self, index):
        from flashinfer.fused_moe.api import BackendOptions

        cfg = self.layer._kernel_config
        moe = cfg.moe_config
        return SplitConfig(
            self.layer._comm_backend,
            dataclasses.replace(
                cfg,
                moe_config=dataclasses.replace(
                    moe, backend=BackendOptions(candidates=(tuple(moe.backend)[index],))
                ),
            ),
        )

    def prepare(self, selection):
        self.layer._kernel.prepare_tactic(self.context, tuples(selection))

    def forward(self, tensors):
        staged = self._stage(tensors)
        out = self.layer._forward_prepared(
            staged, self.state, output=self.output[: staged.num_tokens]
        )[: tensors.num_tokens]
        if tensors.output is not None and tensors.output.data_ptr() != out.data_ptr():
            tensors.output.copy_(out)
            return tensors.output
        return out

    def destroy(self):
        # Received views must be released before their transport allocation.
        self.context = None
        self.choices = None
        self.layer.destroy()
        self.state = None
        self._staging = None


class SplitProtocol:
    def __init__(
        self, plan, tensors, collective, contract, settings, *, fallback=False
    ):
        self.plan, self.tensors, self.collective = plan, tensors, collective
        self.contract, self.settings = contract, settings
        self.fallback = fallback

    def _runner_contract(self, context, choice, peers):
        return {
            **self.contract,
            "candidate": self.collective.gather(
                plain(self.plan.backend_config(choice["runner"]))
            ),
            "runners": [
                {k: v for k, v in peer.items() if k not in ("runner", "tactics")}
                for peer in peers
            ],
            "measurement": self.settings,
            "fallback": context.tuner.protocol_context(
                f"moe_{choice['backend']}", force_tune=context.force_tune
            ).skipped,
        }

    def _records(self, context, contract):
        records = self.collective.gather(context.lookup(record_key(contract)))
        records = [
            r
            for r in records
            if isinstance(r, dict)
            and r.get("contract") == plain(contract)
            and isinstance(r.get("capture_rejected", []), list)
        ]
        rejected = []
        for record in records:
            for tactic in record.get("capture_rejected", []):
                if tactic not in rejected:
                    rejected.append(tactic)
        return records, rejected

    def check_replay(self, context, selection, choices):
        choice = next(v for v in choices if v["runner"] == selection[0])
        peers = self.collective.gather(plain(choice))
        contract = self._runner_contract(context, choice, peers)
        _, rejected = self._records(context, contract)
        if plain(selection[1]) in rejected:
            raise NoEligibleCandidate("The cached split tactic failed capture")

    def run(self, context):
        c, plan = self.collective, self.plan
        choices = c.prepare(plan.enumerate, optional=True)
        all_choices = c.gather(plain(choices))
        # A tactic must be legal on every rank. Runner positions identify
        # distinct configurations of the same backend.
        agreed = []
        for choice in choices:
            index = choice["runner"]
            peers = [
                next(
                    (
                        v
                        for v in rank
                        if v["runner"] == index
                        and v["implementation"] == choice["implementation"]
                    ),
                    None,
                )
                for rank in all_choices
            ]
            if any(peer is None for peer in peers):
                continue
            tactics = [
                tuples(t)
                for t in peers[0]["tactics"]
                if all(t in peer["tactics"] for peer in peers)
            ]
            if tactics:
                agreed.append((choice, tactics, peers))
        c.agree([(v[0]["runner"], plain(v[1])) for v in agreed], "split tactics")
        if not agreed:
            raise NoEligibleCandidate("No split tactic is supported on every EP rank")
        if self.fallback or context.skipped:
            selection = (agreed[0][0]["runner"], -1)
            c.prepare(lambda: plan.prepare(selection))
            return TuningResult({"selection": plain(selection)})
        winners, measured = [], 0
        fallback_selection = None
        for choice, tactics, peers in agreed:
            contract = self._runner_contract(context, choice, peers)
            if contract["fallback"]:
                tactics = [-1]
            key = record_key(contract)
            records, rejected = (
                ([], []) if context.force_tune else self._records(context, contract)
            )
            cached = next(
                (
                    record
                    for record in records
                    if valid_score(record)
                    and isinstance(record.get("selection"), list)
                    and len(record["selection"]) == 2
                    and type(record["selection"][0]) is int
                    and record["selection"][1] in [plain(t) for t in tactics]
                    and record["selection"][1] not in rejected
                ),
                None,
            )
            if cached is not None:
                # The pinned backend is stable when its candidate position changes.
                cached = {
                    **cached,
                    "selection": [choice["runner"], cached["selection"][1]],
                }
            # A default tactic can resolve to a rejected configuration. Only
            # use that fallback for runners without a recorded capture failure.
            if not rejected and fallback_selection is None:
                fallback_selection = (choice["runner"], -1)
            if not context.tuning_enabled:
                if cached is not None:
                    winners.append(cached)
                continue
            cached_tactic = tuples(cached["selection"])[1] if cached else None
            trials = (
                [cached_tactic] + [t for t in tactics if t != cached_tactic]
                if cached
                else tactics
            )
            passing = []

            def publish():
                record = {"contract": plain(contract), "capture_rejected": rejected}
                if passing:
                    record.update(min(passing, key=lambda v: v["p50_us"]))
                # Each rank may have its own persistent store. Update all of
                # them so reload cannot restore a rejection after recovery.
                c.prepare(lambda: context.publish(key, record))
                return record

            for tactic in trials:
                selection = (choice["runner"], tactic)
                c.prepare(lambda: plan.prepare(selection))
                try:
                    score = c.measure(
                        lambda: plan.forward(self.tensors),
                        self.tensors.hidden_states.device,
                        self.settings,
                        warmup=lambda: plan.warmup(self.tensors),
                    )
                except CaptureUnsupported:
                    if plain(tactic) not in rejected:
                        rejected.append(plain(tactic))
                    # A later preparation or launch error must not lose this failure.
                    publish()
                    continue
                if plain(tactic) in rejected:
                    rejected.remove(plain(tactic))
                measured += 1
                passing.append(
                    {
                        "selection": plain(selection),
                        "p50_us": score,
                        "contract": plain(contract),
                    }
                )
                # A usable cached winner avoids a new sweep. If capture fails,
                # the remaining agreed tactics still need to be tried.
                if cached and tactic == cached_tactic:
                    break
            record = publish()
            if passing:
                winners.append(record)
        if (
            not winners
            and not context.tuning_enabled
            and fallback_selection is not None
        ):
            c.prepare(lambda: plan.prepare(fallback_selection))
            return TuningResult({"selection": plain(fallback_selection)})
        if not winners:
            raise CaptureUnsupported("No split tactic completed full-EP measurement")
        winner = min(winners, key=lambda v: v["p50_us"])
        c.prepare(lambda: plan.prepare(winner["selection"]))
        return TuningResult(
            winner,
            cache_hit=not context.tuning_enabled,
            candidates=measured,
            successful_candidates=measured,
        )


def valid_score(record):
    return (
        isinstance(record, dict)
        and isinstance(record.get("p50_us"), (int, float))
        and math.isfinite(record["p50_us"])
        and record["p50_us"] > 0
    )
