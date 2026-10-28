# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Selection state owned by one MegaMoE workspace."""

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
