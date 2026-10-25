# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Delegation for operations with their own tuning loop."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar

if TYPE_CHECKING:
    from .autotuner import AutoTuner

T = TypeVar("T")

_PROTOCOL_KEY = "__protocol__"


def _protocol_cache_key(namespace: str, key: str) -> tuple[str, str, str]:
    if not isinstance(namespace, str) or not namespace:
        raise ValueError("protocol namespace must be a nonempty string")
    if not isinstance(key, str):
        raise TypeError("protocol record key must be a string")
    return (_PROTOCOL_KEY, namespace, key)


# Scope identity is local and never enters a cache key or collective payload.
_tuning_scope: ContextVar[object | None] = ContextVar("tuning_scope", default=None)


@dataclass(frozen=True)
class TuningContext:
    """Shared controls and record storage for one protocol invocation.

    The protocol decides when a cache record is usable and when to publish a
    result. For collective operations, it also coordinates those decisions
    across participants. The context does not perform synchronization.
    """

    tuner: AutoTuner
    namespace: str
    tuning_enabled: bool
    force_tune: bool
    scope: object | None = None

    @property
    def skipped(self) -> bool:
        """Whether this invocation must use the operation's built-in fallback."""
        return not self.force_tune and self.namespace in self.tuner._effective_skip_ops

    @property
    def disk_cache_enabled(self) -> bool:
        """Whether the caller permits persistence, including compatibility files."""
        return (
            not self.tuner._in_v2_context
            or self.tuner._active_managed_store is not None
        )

    @property
    def measurement_policy(self):
        """Active policy, or the attached store's policy during bare serving."""
        policy = self.tuner._effective_measure_policy
        store = self.tuner._active_managed_store
        if policy is None and not self.tuner._in_v2_context and store is not None:
            from flashinfer.autotune_cache import MeasurementPolicy

            fields = store.manifest
            cold = fields.get("measure_cold_l2")
            policy = MeasurementPolicy(
                execution_mode=fields.get("measure_execution_mode", "auto"),
                cold_l2=None if cold is None else cold == "True",
                _timer=fields.get("measure_timer", "auto"),
                graph_timing=fields.get("measure_graph_timing"),
            )
        return policy

    @property
    def cuda_graph_profile_replays(self) -> int | None:
        """Inherited replay-count override, or None for the protocol's default."""
        return self.tuner._override_cuda_graph_profile_replays

    @property
    def tuning_buckets(self) -> tuple[int, ...] | None:
        """Inherited bucket override, or None for the protocol's own capacities."""
        return self.tuner._override_tuning_buckets

    @property
    def round_up(self) -> bool:
        """Whether the caller requested upward bucket rounding."""
        return self.tuner._override_round_up

    @property
    def cache_identity(self) -> tuple:
        """Identify the store/policy, cache generation and skip state."""
        return (
            self.tuner._winner_cache_identity(),
            self.tuner._protocol_cache_generation,
            self.skipped,
        )

    def records(self) -> dict[str, Any]:
        """Return a copy of this protocol's records for custom lookup rules."""
        if self.skipped:
            return {}
        return self.tuner.get_protocol_records(self.namespace)

    def lookup(self, key: str) -> Any:
        """Return a copy of one record, or None if absent or skipped."""
        if self.skipped:
            return None
        return self.tuner.get_protocol_record(self.namespace, key)

    def publish(self, key: str, value: Any) -> None:
        """Cache a result in memory; disk writes follow ambient tuning controls."""
        if self.skipped:
            return
        persist = (
            self.tuner.is_tuning_mode
            and self.namespace not in self.tuner._effective_skip_ops
        )
        self.tuner.publish_protocol_record(self.namespace, key, value, persist=persist)


@dataclass(frozen=True)
class TuningResult(Generic[T]):
    value: T
    cache_hit: bool = False
    candidates: int = 0
    successful_candidates: int = 0


class TuningProtocol(Protocol[T]):
    """Own selection, preparation, measurement and adoption of a result.

    AutoTuner supplies its controls, lock and record storage for single-device
    or distributed tuning. The protocol owns input bucketing, preparation,
    measurement and any distributed coordination. It must implement or reject
    the measurement policy, inherited replay counts and bucket overrides.

    A skipped operation must use its built-in fallback without cache selection.
    Publish selected records through the context; returning a TuningResult
    only supplies the caller's value and tuning statistics.
    """

    def run(self, context: TuningContext) -> TuningResult[T]: ...
