# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Shared autotuner controls around Rubin's native collective search."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import TYPE_CHECKING, Any

from flashinfer.autotuner import TuningContext, TuningResult
from flashinfer.jit.core import logger

from . import knob_cache
from .autotune import (
    KNOB_KEYS,
    _prepare_collectively,
    is_valid_sm107,
    sm107_candidates,
)
from .tuning_measurement import measurement_settings

if TYPE_CHECKING:
    import torch
    from torch.distributed import ProcessGroup

    from .block_scaled import Sm107BlockScaledSymmBuffer, TransformedBlockScaledWeights

NAMESPACE = "moe_ep.sm107.cutedsl.native.v1"


def validate_context(context: TuningContext) -> None:
    if not context.skipped and (context.tuning_buckets is not None or context.round_up):
        raise NotImplementedError(
            "Rubin MegaMoE uses its workspace capacity and live-token cache lookup; "
            "tuning_buckets and round_up=True are not supported."
        )


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class RubinTuningProtocol:
    """Retain native candidates, inputs, numerical checks and event timing."""

    def __init__(
        self,
        y: torch.Tensor | None,
        l1: TransformedBlockScaledWeights,
        l2: TransformedBlockScaledWeights,
        workspace: Sm107BlockScaledSymmBuffer,
        *,
        num_tokens: int | None = None,
        candidates: list[dict[str, Any]] | None = None,
        warmup_iters: int = 3,
        timed_iters: int = 10,
        process_group: ProcessGroup | None = None,
        install_winner: bool = False,
        allow_nondeterministic: bool | None = None,
    ) -> None:
        self.y, self.l1, self.l2 = y, l1, l2
        self.workspace = workspace
        self.cfg = workspace.config
        self.num_tokens = (
            workspace.staged_tokens() if num_tokens is None else num_tokens
        )
        self.candidates = (
            sm107_candidates(
                self.cfg.quant_kind, kernel_variant=self.cfg.kernel_variant
            )
            if candidates is None
            else candidates
        )
        self.allow_nondeterministic = (
            any(
                k.get("reduce_topk_in_kernel", self.cfg.reduce_topk_in_kernel)
                for k in self.candidates
            )
            if allow_nondeterministic is None
            else allow_nondeterministic
        )
        self.warmup_iters, self.timed_iters = warmup_iters, timed_iters
        self.group = process_group
        self.install_winner = install_winner

    def _gather(self, value):
        import torch.distributed as dist

        if self.cfg.world_size == 1:
            return [value]
        values = [None] * self.cfg.world_size
        dist.all_gather_object(values, value, group=self.group)
        return values

    def _query(self, live_tokens):
        cfg = self.cfg
        return dict(
            dtype=cfg.quant_kind,
            kernel_variant=cfg.kernel_variant,
            combine_dtype=cfg.combine_dtype,
            world_size=cfg.world_size,
            hidden=cfg.hidden,
            intermediate=cfg.intermediate,
            num_experts=cfg.num_total_experts,
            topk=cfg.num_topk,
            max_tokens=cfg.max_tokens_per_rank,
            live_tokens=live_tokens,
            allow_nondeterministic=self.allow_nondeterministic,
            apply_routing_weights_before_fc2=cfg.apply_routing_weights_before_fc2,
            activation=cfg.activation,
            situ_beta=cfg.situ_beta,
            situ_linear_beta=cfg.situ_linear_beta,
            gate_up_clamp=cfg.gate_up_clamp,
        )

    def _contract(self):
        import torch
        from flashinfer.autotuner.autotuner import _collect_metadata

        config = dataclasses.asdict(self.cfg)
        for key in (*KNOB_KEYS, "rank", "max_tokens_per_rank"):
            config.pop(key, None)
        device = None
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            device = [props.name, props.major, props.minor, props.multi_processor_count]
        return {
            "backend_revision": knob_cache.BACKEND_REVISION,
            "environment": _collect_metadata(),
            "config": config,
            "device": device,
        }

    def _compatible(self, record, contract, context, measurement):
        if not isinstance(record, dict):
            return False
        if record.get("protocol_contract") not in (None, contract):
            return False
        policy = context.measurement_policy
        policy_fields = policy.manifest_fields() if policy is not None else {}
        if record.get("measurement_policy", {}) != policy_fields:
            return False
        if context.tuning_enabled and record.get("profiling_policy") != measurement:
            return False
        try:
            from .block_scaled import Sm107BlockScaledSymmBuffer

            capacity = record["max_tokens"]
            live = record.get("live_tokens", capacity)
            if live is None:
                live = capacity
            if type(capacity) is not int or capacity < 1:
                return False
            if type(live) is not int or not 0 <= live <= capacity:
                return False
            knobs = knob_cache._knobs_from_json(record["knobs"])
            valid = is_valid_sm107(knobs, self.cfg) and (
                self.allow_nondeterministic
                or not knobs.get("reduce_topk_in_kernel", False)
            )
            if valid:
                Sm107BlockScaledSymmBuffer._build_kernel(
                    dataclasses.replace(self.cfg, **knobs)
                )
            return valid
        except Exception as exc:
            # Cached configuration/resource validation must not launch or allocate.
            logger.warning(
                f"[Autotuner]: Ignoring invalid MegaMoE configuration: {exc}"
            )
            return False

    def _lookup(self, entries, query, contract, context, measurement):
        records = {}
        for rank_entries in self._gather(entries):
            for entry in rank_entries:
                try:
                    records.setdefault(_json(entry), entry)
                except (TypeError, ValueError):
                    continue
        entries = knob_cache._matching_entries(
            entries=[e for e in records.values() if isinstance(e, dict)],
            **{k: v for k, v in query.items() if k != "live_tokens"},
        )
        valid = [self._compatible(e, contract, context, measurement) for e in entries]
        ranks = self._gather(valid)
        return knob_cache.lookup_record(
            entries=[
                e for i, e in enumerate(entries) if all(rank[i] for rank in ranks)
            ],
            **query,
        )

    def run(self, context: TuningContext) -> TuningResult:
        import torch
        import torch.distributed as dist

        validate_context(context)
        workspace, cfg = self.workspace, self.cfg
        state = workspace._tuning_state
        policy = context.measurement_policy
        policy_fields = policy.manifest_fields() if policy is not None else {}
        measurement = measurement_settings(context)
        capturing = (
            torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
        )
        if (
            context.skipped
            and (capturing or state.graph_captured)
            and state.winner is not None
        ):
            raise RuntimeError(
                "skip_ops requires fallback preparation before capture; "
                "use a new workspace for a captured explicit handle."
            )
        if capturing:
            state.graph_captured = True
            if context.tuning_enabled:
                raise RuntimeError(
                    "Warm up pending Rubin autotuning before CUDA graph capture."
                )
            return TuningResult(None)
        if self.install_winner and state.graph_captured:
            if context.tuning_enabled:
                raise RuntimeError(
                    "Rubin autotuning cannot replace a captured workspace; "
                    "create a new workspace and tune it before capture."
                )
            return TuningResult(None)
        if cfg.world_size > 1 and (
            not dist.is_initialized()
            or dist.get_world_size(self.group) != cfg.world_size
            or dist.get_rank(self.group) != cfg.rank
        ):
            raise ValueError(
                "autotune process_group must match the workspace EP rank and world size"
            )

        lookup_done = not context.tuning_enabled and state.lookup_scope == (
            context.scope,
            context.cache_identity,
        )
        # Misses are remembered locally, but every rank must join a peer's retry.
        if all(self._gather(lookup_done)):
            return TuningResult(None)

        spec = dataclasses.asdict(cfg)
        spec.pop("rank")
        proposal = (
            spec,
            self.candidates,
            self.warmup_iters,
            self.timed_iters,
            self.allow_nondeterministic,
            self.install_winner,
            context.tuning_enabled,
            context.force_tune,
            context.disk_cache_enabled,
            context.skipped,
            policy_fields,
            measurement,
        )
        if any(_json(p) != _json(proposal) for p in self._gather(proposal)):
            raise ValueError(
                "EP ranks disagree on Rubin tuning geometry, candidates or controls"
            )
        if context.skipped:
            state.lookup_scope = (context.scope, context.cache_identity)
            return TuningResult(None)
        counts = self._gather(self.num_tokens)
        if any(
            type(n) is not int or not 0 <= n <= cfg.max_tokens_per_rank for n in counts
        ):
            raise ValueError("stage a valid num_tokens on every rank before autotuning")
        if not self.candidates or self.warmup_iters < 1 or self.timed_iters < 1:
            raise ValueError(
                "autotune requires candidates and positive iteration counts"
            )
        if not self.allow_nondeterministic and any(
            k.get("reduce_topk_in_kernel", cfg.reduce_topk_in_kernel)
            for k in self.candidates
        ):
            raise ValueError(
                "in-kernel reduction candidates require nondeterminism permission"
            )

        query, contract = self._query(max(counts)), self._contract()
        if any(_json(c) != _json(contract) for c in self._gather(contract)):
            raise ValueError(
                "EP ranks disagree on the Rubin tuning implementation or device"
            )
        record = None
        if not context.force_tune:
            record = self._lookup(
                list(context.records().values()), query, contract, context, measurement
            )
            if record is None and context.disk_cache_enabled:
                path = knob_cache.knob_cache_path()
                try:
                    entries = [] if path is None else knob_cache._load_entries(path)
                except Exception as exc:
                    logger.warning(
                        f"[Autotuner]: Cannot read MegaMoE cache file: {exc}"
                    )
                    entries = []
                record = self._lookup(entries, query, contract, context, measurement)
        cache_hit = record is not None
        if record is None:
            if not context.tuning_enabled:
                state.lookup_scope = (
                    context.scope,
                    context.cache_identity,
                )
                return TuningResult(None)
            from .autotune import _sweep_sm107_block_scaled_mega_moe

            knobs, score, successful = _sweep_sm107_block_scaled_mega_moe(
                self.y,
                self.l1,
                self.l2,
                workspace,
                num_tokens=self.num_tokens,
                candidates=self.candidates,
                warmup_iters=self.warmup_iters,
                timed_iters=self.timed_iters,
                process_group=self.group,
                measurement=measurement,
            )
            record = knob_cache.make_record(
                knobs,
                **query,
                p50_us=score,
                source="autotune-sampled-oracle",
            )
            record["measurement_policy"] = policy_fields
            record["profiling_policy"] = measurement
        else:
            successful = 0
            knobs = knob_cache._knobs_from_json(record["knobs"])

        if self.install_winner and not cache_hit:
            # Resource validation is rank-local; agree before replacing any storage.
            reason = None
            try:
                from .block_scaled import Sm107BlockScaledSymmBuffer

                Sm107BlockScaledSymmBuffer._build_kernel(
                    dataclasses.replace(cfg, **knobs)
                )
            except (ValueError, NotImplementedError) as exc:
                reason = str(exc)
            reasons = self._gather(reason)
            if any(reasons):
                raise RuntimeError(
                    f"Rubin winner is not valid on every EP rank: {reasons}"
                )
        if self.install_winner:
            _prepare_collectively(
                lambda: workspace.install_tuned_knobs(knobs, self.l1, self.l2),
                cfg.world_size,
                self.group,
            )

        error = None
        if cfg.rank == 0:
            try:
                record = {**record, "protocol_contract": contract}
                key = hashlib.sha256(
                    _json(
                        {
                            k: v
                            for k, v in record.items()
                            if k
                            not in (
                                "knobs",
                                "p50_us",
                                "source",
                                "tuned_at",
                                "measurement_policy",
                                "profiling_policy",
                            )
                        }
                    ).encode()
                ).hexdigest()
                context.publish(key, record)
                # File consumers must see winners selected from either cache.
                if context.disk_cache_enabled:
                    try:
                        knob_cache.write_record(record)
                    except Exception as exc:
                        logger.warning(
                            f"[Autotuner]: Cannot write MegaMoE cache file: {exc}"
                        )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        errors = self._gather(error)
        if any(errors):
            raise RuntimeError(f"Rubin tuning result publication failed: {errors}")
        return TuningResult(
            knobs,
            cache_hit=cache_hit,
            candidates=0 if cache_hit else len(self.candidates),
            successful_candidates=successful,
        )
