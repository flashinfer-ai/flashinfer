# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Shared autotuner controls around Blackwell's native collective search."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import TYPE_CHECKING, Any

from flashinfer.autotuner import TuningContext, TuningResult
from flashinfer.jit.core import logger
from flashinfer.moe_ep.core.tuning import token_bucket_sort_key

from flashinfer.moe_ep.shims.sm100.cutedsl_megamoe.sources import knob_cache
from flashinfer.moe_ep.shims.sm100.cutedsl_megamoe.sources.autotune import (
    NoPassingCandidate,
    _prepare_collectively,
)
from flashinfer.moe_ep.shims.sm100.cutedsl_megamoe.sources.tuning import (
    knob_keys,
    sm100_genphase_candidates,
    sm100_standard_candidates,
)
from flashinfer.moe_ep.shims.sm100.cutedsl_megamoe.sources.measurement import (
    measurement_settings,
)

if TYPE_CHECKING:
    import torch
    from torch.distributed import ProcessGroup

    from flashinfer.moe_ep.shims.sm100.cutedsl_megamoe.sources.block_scaled import (
        BlockScaledSymmBuffer,
        TransformedBlockScaledWeights,
    )

NAMESPACE = "moe_ep.sm100.next_cutedsl.native.v1"


def validate_context(context: TuningContext) -> None:
    if not context.skipped and (context.tuning_buckets is not None or context.round_up):
        raise NotImplementedError(
            "Blackwell MegaMoE uses its workspace capacity and live-token cache lookup; "
            "tuning_buckets and round_up=True are not supported."
        )


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class BlackwellTuningProtocol:
    """Collectively select standard or GenPhase knobs under v2 controls."""

    def __init__(
        self,
        y: torch.Tensor | None,
        l1: TransformedBlockScaledWeights,
        l2: TransformedBlockScaledWeights,
        workspace: BlockScaledSymmBuffer,
        *,
        num_tokens: int | None = None,
        candidates: list[dict[str, Any]] | None = None,
        warmup_iters: int = 8,
        timed_iters: int = 32,
        calls_per_sample: int = 8,
        constraints: dict[str, Any] | None = None,
        process_group: ProcessGroup | None = None,
        install_winner: bool = False,
        allow_nondeterministic: bool | None = None,
        make_trial=None,
        execution_contract: dict | None = None,
        remeasure_cached: bool = False,
    ) -> None:
        if (make_trial is None) != (execution_contract is None):
            raise ValueError(
                "complete-forward tuning requires both a trial and its execution contract"
            )
        self.make_trial, self.execution_contract = make_trial, execution_contract
        self.y, self.l1, self.l2 = y, l1, l2
        self.workspace = workspace
        self.cfg = workspace.config
        self.num_tokens = (
            workspace.staged_tokens() if num_tokens is None else num_tokens
        )
        self.candidates = candidates
        self.constraints = constraints or {}
        self.calls_per_sample = calls_per_sample
        self.allow_nondeterministic = (
            any(
                k.get("reduce_topk_in_kernel", self.cfg.reduce_topk_in_kernel)
                for k in candidates
            )
            if allow_nondeterministic is None and candidates is not None
            else bool(allow_nondeterministic)
        )
        self.warmup_iters, self.timed_iters = warmup_iters, timed_iters
        self.group = process_group
        self.install_winner = install_winner
        self.remeasure_cached = remeasure_cached
        self.latency_us: float | None = None
        self.selected_record: dict[str, Any] | None = None

    def _gather(self, value):
        import torch.distributed as dist

        if self.cfg.world_size == 1:
            return [value]
        values = [None] * self.cfg.world_size
        dist.all_gather_object(values, value, group=self.group)
        return values

    def _query(self, live_tokens):
        return {**knob_cache._key(self.cfg), "live_tokens": live_tokens}

    def _contract(self):
        base = knob_cache._key(self.cfg)
        return (
            {"kernel": base, "execution": self._gather(self.execution_contract)}
            if self.make_trial is not None
            else base
        )

    def _compatible(
        self, record, contract, context, measurement, *, kernel_replay=False
    ):
        if not isinstance(record, dict):
            return False
        if knob_cache.allows_nondeterministic(record) != self.allow_nondeterministic:
            return False
        complete_forward = self.make_trial is not None and not kernel_replay
        boundary = "complete_forward" if complete_forward else "kernel"
        if record.get("timing_boundary", "kernel") != boundary:
            return False
        if complete_forward and record.get("protocol_contract") != contract:
            return False
        if record.get("protocol_contract") not in (None, contract):
            return False
        policy = context.measurement_policy
        policy_fields = policy.manifest_fields() if policy is not None else {}
        if record.get("measurement_policy", {}) != policy_fields:
            return False
        if context.tuning_enabled and record.get("profiling_policy") != {
            k: v for k, v in measurement.items() if k != "warmup_iters"
        }:
            return False
        try:
            capacity = record["problem"]["max_tokens_per_rank"]
            live = record["live_tokens"]
            if type(capacity) is not int or capacity < 1:
                return False
            if type(live) is not int or not 0 <= live <= capacity:
                return False
            knobs = knob_cache._decode(record, self.cfg)
            valid = (
                knobs is not None
                and all(knobs.get(k) == v for k, v in self.constraints.items())
                and (
                    self.allow_nondeterministic
                    or not knobs.get("reduce_topk_in_kernel", False)
                )
            )
            if valid:
                type(self.workspace)._build_kernel(
                    dataclasses.replace(self.cfg, **knobs)
                )
            return valid
        except Exception as exc:
            # Cached configuration/resource validation must not launch or allocate.
            logger.warning(
                f"[Autotuner]: Ignoring invalid MegaMoE configuration: {exc}"
            )
            return False

    def _lookup(
        self,
        entries,
        query,
        contract,
        context,
        measurement,
        *,
        kernel_replay=False,
        allow_live_token_fallback=False,
    ):
        records = {}
        for rank_entries in self._gather(entries):
            for entry in rank_entries:
                try:
                    records.setdefault(_json(entry), entry)
                except (TypeError, ValueError):
                    continue
        workload = (
            {key: value for key, value in query.items() if key != "live_tokens"}
            if allow_live_token_fallback
            else query
        )
        entries = [
            e
            for e in records.values()
            if isinstance(e, dict) and all(e.get(k) == v for k, v in workload.items())
        ]
        if allow_live_token_fallback:
            entries = knob_cache._replay_entries(
                reversed(entries), query["live_tokens"], self.cfg.max_tokens_per_rank
            )
        valid = [
            self._compatible(
                e, contract, context, measurement, kernel_replay=kernel_replay
            )
            for e in entries
        ]
        ranks = self._gather(valid)
        matches = [e for i, e in enumerate(entries) if all(rank[i] for rank in ranks)]
        # Agree even if ranks loaded different winners from independent stores.
        if not matches:
            return None
        return matches[0] if allow_live_token_fallback else matches[-1]

    def _sweep(self, candidates, measurement):
        from flashinfer.moe_ep.shims.sm100.cutedsl_megamoe.sources.autotune import (
            _sweep_sm100_mega_moe,
        )

        return _sweep_sm100_mega_moe(
            self.y,
            self.l1,
            self.l2,
            self.workspace,
            num_tokens=self.num_tokens,
            candidates=candidates,
            warmup_iters=self.warmup_iters,
            timed_iters=self.timed_iters,
            calls_per_sample=self.calls_per_sample,
            process_group=self.group,
            measurement=measurement,
            make_trial=self.make_trial,
        )

    def run(self, context: TuningContext) -> TuningResult:
        import torch
        import torch.distributed as dist

        validate_context(context)
        workspace, cfg = self.workspace, self.cfg
        state = workspace._tuning_state
        policy = context.measurement_policy
        policy_fields = policy.manifest_fields() if policy is not None else {}
        measurement = measurement_settings(
            context, self.calls_per_sample, self.warmup_iters
        )
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
                    "Warm up pending Blackwell autotuning before CUDA graph capture."
                )
            return TuningResult(None)
        if self.install_winner and state.graph_captured:
            if context.tuning_enabled:
                raise RuntimeError(
                    "Blackwell autotuning cannot replace a captured workspace; "
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
            self.calls_per_sample,
            self.constraints,
            self.allow_nondeterministic,
            self.install_winner,
            self.remeasure_cached,
            context.tuning_enabled,
            context.force_tune,
            context.disk_cache_enabled,
            context.skipped,
            policy_fields,
            measurement,
            self.make_trial is not None,
        )
        if any(_json(p) != _json(proposal) for p in self._gather(proposal)):
            raise ValueError(
                "EP ranks disagree on Blackwell tuning geometry, candidates or controls"
            )
        if context.skipped:
            state.lookup_scope = (context.scope, context.cache_identity)
            return TuningResult(None)
        if self.constraints and cfg.kernel_variant != "genphase":
            raise ValueError(
                "constraints apply to GenPhase; pass complete standard candidates"
            )
        counts = self._gather(self.num_tokens)
        if any(
            type(n) is not int or not 0 <= n <= cfg.max_tokens_per_rank for n in counts
        ):
            raise ValueError("stage a valid num_tokens on every rank before autotuning")
        if self.candidates is None:
            self.candidates = (
                sm100_genphase_candidates(
                    cfg,
                    sm_count=torch.cuda.get_device_properties(
                        torch.cuda.current_device()
                    ).multi_processor_count,
                    live_tokens=max(counts),
                    constraints=self.constraints,
                )
                if cfg.kernel_variant == "genphase"
                else sm100_standard_candidates(
                    cfg, allow_nondeterministic=self.allow_nondeterministic
                )
            )
        if not self.candidates or any(
            type(n) is not int or n < 1
            for n in (self.warmup_iters, self.timed_iters, self.calls_per_sample)
        ):
            raise ValueError(
                "autotune requires candidates and positive iteration counts"
            )
        if any(set(k) != set(knob_keys(cfg)) for k in self.candidates):
            raise ValueError("autotune candidates must set every Blackwell knob")
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
                "EP ranks disagree on the Blackwell tuning implementation or device"
            )
        record = None
        record_contract = contract
        if not context.force_tune:
            memory_entries = list(context.records().values())
            disk_entries: list[dict[str, Any]] = []
            record = self._lookup(memory_entries, query, contract, context, measurement)
            if record is None and context.disk_cache_enabled:
                path = knob_cache.knob_cache_path()
                try:
                    disk_entries = (
                        [] if path is None else knob_cache._read(path)["entries"]
                    )
                except Exception as exc:
                    logger.warning(
                        f"[Autotuner]: Cannot read MegaMoE cache file: {exc}"
                    )
                record = self._lookup(
                    disk_entries, query, contract, context, measurement
                )
            kernel_replay_allowed = (
                self.make_trial is not None
                and self.install_winner
                and not context.tuning_enabled
            )
            if record is None and kernel_replay_allowed:
                # Honor older fixed-variant offline records on replay misses.
                # Keep them in the kernel timing boundary.
                for entries in (memory_entries, disk_entries):
                    record = self._lookup(
                        entries,
                        query,
                        contract["kernel"],
                        context,
                        measurement,
                        kernel_replay=True,
                    )
                    if record is not None:
                        record_contract = contract["kernel"]
                        break
            if record is None and not context.tuning_enabled:
                # Exact records in either store take precedence over bucket replay.
                entries = [*disk_entries, *memory_entries]
                live_tokens = query["live_tokens"]
                record = self._lookup(
                    entries,
                    query,
                    contract,
                    context,
                    measurement,
                    allow_live_token_fallback=True,
                )
                if kernel_replay_allowed:
                    kernel_record = self._lookup(
                        entries,
                        query,
                        contract["kernel"],
                        context,
                        measurement,
                        kernel_replay=True,
                        allow_live_token_fallback=True,
                    )
                    if kernel_record is not None and (
                        record is None
                        or token_bucket_sort_key(
                            kernel_record["live_tokens"], live_tokens
                        )
                        < token_bucket_sort_key(record["live_tokens"], live_tokens)
                    ):
                        record = kernel_record
                        record_contract = contract["kernel"]
        if record is not None and self.remeasure_cached and context.tuning_enabled:
            try:
                _, self.latency_us, _ = self._sweep(
                    [knob_cache._decode(record, cfg)], measurement
                )
            except NoPassingCandidate:
                record = None
        cache_hit = record is not None
        if record is None:
            if not context.tuning_enabled:
                state.lookup_scope = (
                    context.scope,
                    context.cache_identity,
                )
                return TuningResult(None)
            knobs, score, successful = self._sweep(self.candidates, measurement)
            self.latency_us = score
            record = knob_cache.make_record(
                knobs,
                cfg,
                live_tokens=max(counts),
                p50_us=score,
                allow_nondeterministic=self.allow_nondeterministic,
            )
            record["measurement_policy"] = policy_fields
            record["profiling_policy"] = {
                k: v for k, v in measurement.items() if k != "warmup_iters"
            }
            if self.make_trial is not None:
                record["timing_boundary"] = "complete_forward"
        else:
            successful = 0
            knobs = knob_cache._decode(record, cfg)

        if self.install_winner and not cache_hit:
            # Resource validation is rank-local; agree before replacing any storage.
            reason = None
            try:
                type(self.workspace)._build_kernel(dataclasses.replace(cfg, **knobs))
            except (ValueError, NotImplementedError) as exc:
                reason = str(exc)
            reasons = self._gather(reason)
            if any(reasons):
                raise RuntimeError(
                    f"Blackwell winner is not valid on every EP rank: {reasons}"
                )
        if self.install_winner:
            _prepare_collectively(
                lambda: workspace.install_tuned_knobs(knobs, self.l1, self.l2),
                cfg.world_size,
                self.group,
            )

        record = {**record, "protocol_contract": record_contract}
        if self.allow_nondeterministic:
            record["allow_nondeterministic"] = True
        self.selected_record = record
        error = None
        if cfg.rank == 0:
            try:
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
                        knob_cache.write_record(
                            record,
                            overwrite=record["live_tokens"] == query["live_tokens"],
                        )
                    except Exception as exc:
                        logger.warning(
                            f"[Autotuner]: Cannot write MegaMoE cache file: {exc}"
                        )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        errors = self._gather(error)
        if any(errors):
            raise RuntimeError(f"Blackwell tuning result publication failed: {errors}")
        return TuningResult(
            knobs,
            cache_hit=cache_hit,
            candidates=0 if cache_hit else len(self.candidates),
            successful_candidates=successful,
        )
