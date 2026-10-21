# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Family heuristics for ``frost_gemm``: the facts callable and the proposal
callable the engine manifest names (``engines/manifest.py``).

Same shape as the SDPA family (``cudnn.sdpa.fwd.heuristics``): the analyzer
turns the graph into facts once, ``recommend`` proposes complete
``PlanConfig(engine_id, GemmKnobs)`` entries from those facts alone, and every
listed plan carries the exact tile config the engine will build -- so a recorded
``(engine_id, knobs)`` replays the same kernel after the automatic pick changes.

The shared planner proposes up to eight entries in calibrated paths, including
geometry, split-K and swapAB choices, with the single-plan recommendation first.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from flashinfer.experimental.qwen_3_5_block._cudnn_frost.engines.base import PlanConfig

from .knobs import GemmKnobs

_LOG = logging.getLogger("cudnn.gemm.frost.heuristics")

ENGINE = "frost_gemm"


@dataclass(frozen=True)
class GemmFacts:
    """What the ranking and the engine share about one GEMM graph."""

    chain: Any  # fusion_ir.FusionChain
    dynamic_shapes: bool  # the graph declared dynamic shapes (split-K is skipped)


def analyze_facts(graph) -> Optional[GemmFacts]:
    """The manifest's ``analyzer`` hook: facts for a GEMM graph, or None when
    the FROST analyzer cannot express it (the backend then serves it alone)."""
    from .compiler import _graph_dynamic_shapes
    from .graph_analyzer import _graph_has_gemm, analyze

    if not _graph_has_gemm(graph):
        return None
    try:
        chain = analyze(graph)
    except (NotImplementedError, ValueError, KeyError) as exc:
        _LOG.debug("frost_gemm analyzer declined the graph: %s", exc)
        return None
    return GemmFacts(chain=chain, dynamic_shapes=_graph_dynamic_shapes(graph))


def recommend(kind: str, facts: GemmFacts, offered: Dict[str, int]) -> List[PlanConfig]:
    """Ordered, supported plans carrying exact public-replay knobs."""
    engine_id = offered.get(ENGINE)
    if engine_id is None:
        return []
    from .compiler import plan_configs, probe_chain, probe_cutedsl
    from .planning import MAX_PLAN_CONFIGS

    try:
        probe_cutedsl()
        configs = plan_configs(facts.chain, dynamic_shapes=facts.dynamic_shapes)
    except (NotImplementedError, ValueError, KeyError) as exc:
        _LOG.debug("frost_gemm proposes nothing (%s): %s", kind, exc)
        return []
    plans = {}
    for config in configs:
        try:
            probe_chain(facts.chain, config)
            knobs = GemmKnobs.from_config(config)
        except (NotImplementedError, ValueError, KeyError) as exc:
            _LOG.debug("frost_gemm skips candidate %s (%s): %s", config.name, kind, exc)
            continue
        plans.setdefault(knobs, PlanConfig(engine_id, knobs))
        if len(plans) == MAX_PLAN_CONFIGS:
            break
    return list(plans.values())
