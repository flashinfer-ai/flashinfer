# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Selection state for a prepared collective operation."""

from dataclasses import dataclass
from typing import Any


@dataclass
class AutoTuningState:
    initial_knobs: dict[str, Any] | None = None
    winner: dict[str, Any] | None = None
    cache_identity: tuple | None = None
    tuned_identity: tuple | None = None
    lookup_scope: tuple | None = None
    graph_captured: bool = False
    serving_generation: int | None = None

    def record_selection(
        self,
        winner: dict[str, Any] | None,
        cache_identity: tuple,
        tuned_identity: tuple | None,
    ) -> None:
        """Keep identities paired with the configuration currently installed."""
        self.winner = winner
        self.cache_identity = cache_identity
        self.tuned_identity = tuned_identity


def prepared_for_serving(state: AutoTuningState, context) -> bool:
    # An unrelated thread's tuning context must not trigger collectives here.
    return (
        context.scope is None
        and not context.force_tune
        and state.serving_generation == context.execution_generation
    )


def record_prepared(state: AutoTuningState, context) -> None:
    import torch

    state.serving_generation = (
        context.execution_generation
        if context.scope is None
        and not context.force_tune
        and not (torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())
        else None
    )
