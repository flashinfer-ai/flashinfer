# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frontend heuristics: produce the ranked plan list for a graph.

``create_execution_plans([heur_mode.A, ...])`` collects the inputs — the parsed
facts, the engine ids on offer, and the backend's own entries tagged with the
mode that produced each — and assembles ``graph.plans``.

Two layers, deliberately separated:

- The FAMILY hook is :func:`recommend`-shaped: ``(kind, facts, offered) ->
  [PlanConfig]`` — pure, backend-blind, import-light. It answers one question:
  which of MY engines serve these facts, with which complete knob assignments,
  best first. It never sees the backend, modes, or another family.

- PLACEMENT lives HERE (:func:`_assemble`), once for every family. Inside each
  mode block the backend's own entries form one BLOCK, and the family says
  where that block goes by putting the :data:`BACKEND` marker in its list:
  ``[ours..., BACKEND]`` where the family is measured ahead of the backend on
  that shard, ``[BACKEND, ours...]`` where it is not (the SDPA-forward family's
  ``placement.py`` is the worked example, every threshold a benchmark cell). A
  list without the marker keeps the historical order, ours first. Placement
  only decides the default winner: an autotune (build ALL) pass measures every
  entry regardless of order, and ``build_plans`` walks past a declined entry.
  The delegating entry, dedup, and the mode strip are placement bookkeeping
  and stay out of the families.

An engine answers two questions only: can I serve this graph
(``check_support``) and compile me this config (``build_plan``).

A family that declares no hook falls back to one default plan per accepting
engine, ahead of the backend's entries (the historical order). Every python engine belongs to a
family — the manifest is the only way one exists — so a graph has a family's
proposals or only the backend's.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from .base import BaseEngine, PlanConfig, decline_types
from .engine_ids import BACKEND_BLOCK_ENGINE_ID

_LOG = logging.getLogger("cudnn.engines.heuristics")

BACKEND = PlanConfig(BACKEND_BLOCK_ENGINE_ID)
"""Placement marker for a family hook's list: the backend's own ranked block for
the mode goes where this entry sits. :func:`_assemble` expands it; it never
reaches ``graph.plans``. Compare by ``engine_id`` (``is_backend_block``)."""


def is_backend_block(cfg: PlanConfig) -> bool:
    return cfg.engine_id == BACKEND_BLOCK_ENGINE_ID


def _place(proposals: List[PlanConfig], block: List[PlanConfig]) -> List[PlanConfig]:
    """Splice the backend's ``block`` where the family put :data:`BACKEND`.
    No marker: the historical order, ours first. Only the first marker places
    the block; a repeat is dropped, so one block per mode."""
    proposals = list(proposals)
    if not any(is_backend_block(p) for p in proposals):
        return proposals + block
    out: List[PlanConfig] = []
    placed = False
    for p in proposals:
        if not is_backend_block(p):
            out.append(p)
        elif not placed:
            out += block
            placed = True
    return out


def default_modes() -> List[Any]:
    """The modes assumed when the caller named none — the backend's own default."""
    import flashinfer.experimental.qwen_3_5_block._cudnn_frost as cudnn

    return [cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK]


def accepts(engine: BaseEngine, graph) -> bool:
    """Whether ``engine`` will serve ``graph``, declines being routing not error."""
    try:
        engine.check_support(graph)
    except decline_types() as exc:
        _LOG.debug("engine %s declined the graph: %s", engine.name, exc)
        return False
    return True


def decline_reasons(engines: List[BaseEngine], graph) -> Tuple[Dict[str, str], List[str]]:
    """Every candidate's verdict on ``graph``: ``{name: message}`` for the engines that decline it and the
    names of those that accept.  The failure-path twin of :func:`accepts` -- when the ranked list comes out
    empty, "which side had nothing, and why" is the first question and the backend's text answers half."""
    declined, accepted = {}, []
    for engine in engines:
        try:
            engine.check_support(graph)
        except decline_types() as exc:
            declined[engine.name] = str(exc)  # "<engine>: <reason>" already on the FROST engines
        else:
            accepted.append(engine.name)
    return declined, accepted


def _unranked(graph, engines: List[BaseEngine], backend_plans: List[PlanConfig]) -> List[PlanConfig]:
    """The list for a family with no hook: accepting engines, then the backend."""
    return [PlanConfig(e.engine_id, None) for e in engines if accepts(e, graph)] + [_strip(c) for c in backend_plans]


def _strip(cfg: PlanConfig) -> PlanConfig:
    """A final-list entry: (engine_id, knobs[, cpp_index]) — the mode tag is
    assembly bookkeeping and never reaches ``graph.plans``."""
    if cfg.mode is None and cfg.cpp_index is None:
        return cfg
    return PlanConfig(cfg.engine_id, cfg.knobs, cpp_index=cfg.cpp_index)


def _assemble(modes: List[Any], recommend: Callable[[str], List[PlanConfig]], backend_plans: List[PlanConfig]) -> List[PlanConfig]:
    """The final ranked list: mode block by mode block in the caller's order,
    the backend's entries placed where the family's :data:`BACKEND` marker
    says (ours first when the family gives no marker).

    ``recommend(kind)`` is the family's hook already bound to (facts, offered):
    ``kind`` is ``"A"`` (candidates worth timing, best first — also the answer
    to B, which asks for a wider search the families have none to give) or
    ``"FALLBACK"`` (the config expected to build where A's choice may not).

    An untagged backend entry is the delegating one: OSS candidates C++ holds
    but never exposes as plans, so it cannot be enumerated. It belongs to no
    mode, and it is NOT a pure OSS entry — Graph::build_plans tries the OSS
    engine and, if that one declines, falls through to the native
    engine_configs already enqueued. So it leads the BACKEND's entries but not
    ours: ahead of our OPENSOURCE block it would answer an OSS-coverage
    question with a native kernel.

    Asking for ``[A, FALLBACK]`` puts every tuned candidate — both sides' —
    ahead of every fallback. A plan repeated across blocks keeps its first
    position. Identity is (engine, knobs): cpp_index is only WHERE one backend
    query put a plan, so keying on it would let one config both modes return
    through as two entries — and an autotuner would build and time it twice.
    """
    import flashinfer.experimental.qwen_3_5_block._cudnn_frost as cudnn

    delegating = [c for c in backend_plans if c.mode is None]
    out: List[PlanConfig] = []
    for mode in modes:
        if mode == cudnn.heur_mode.OPENSOURCE:
            # python-only + delegating: the marker must not pull native entries
            # into an OPENSOURCE block, and the delegating entry never leads
            # ours (below), so the marker is dropped here.
            out += [p for p in recommend("A") if not is_backend_block(p)] + delegating
        elif mode in (cudnn.heur_mode.A, cudnn.heur_mode.B):
            out += _place(recommend("A"), delegating + [c for c in backend_plans if c.mode == mode])
        elif mode == cudnn.heur_mode.FALLBACK:
            out += _place(recommend("FALLBACK"), delegating + [c for c in backend_plans if c.mode == mode])
    # A delegate with no mode asked for it (the backend has engines but exposed
    # no plans) would otherwise be dropped.
    out += delegating

    seen, ranked = set(), []
    for cfg in out:
        key = (cfg.engine_id, repr(cfg.knobs))
        if key not in seen:
            seen.add(key)
            ranked.append(_strip(cfg))
    return ranked


def rank(graph, engines: List[BaseEngine], backend_plans: List[PlanConfig], modes: Optional[List[Any]] = None) -> List[PlanConfig]:
    """The ranked plan list for ``graph`` — what ``create_execution_plans`` stores.

    ``engines`` are this graph's python candidates and ``backend_plans`` the
    backend's own entries, each already tagged with its ``mode``.
    """
    from . import manifest

    modes = list(modes) if modes else default_modes()
    family = manifest.family_for(graph)
    recommend = manifest.resolve_heuristics(family) if family is not None else None
    if recommend is None:
        return _unranked(graph, engines, backend_plans)

    analyzer = manifest.resolve_analyzer(family)
    facts = graph._facts_for(analyzer) if analyzer is not None else None
    if facts is None:
        # The family claims the graph by node type but its analyzer cannot
        # express it. Nothing to recommend on; the backend serves it.
        return [_strip(c) for c in backend_plans]

    # Engines that can serve THIS graph, which is what a family hook means by
    # "offered" -- the hookless path (_unranked) has always filtered this way.
    # It has to happen here and not in the families: _build_plan_at() builds a
    # ranked entry WITHOUT re-affirming check_support, so whatever reaches the
    # list is taken as eligible. A family that ranks an engine it cannot filter
    # would otherwise launch a kernel on a graph that engine rejects -- caught
    # as an illegal memory access when the KDA hook first ranked its 128-only
    # sm90 engines ahead of a head-dim-64 graph.
    offered = {e.name: e.engine_id for e in engines if accepts(e, graph)}
    plans = _assemble(modes, lambda kind: list(recommend(kind, facts, offered)), backend_plans)
    own = set(offered.values())
    for cfg in plans:
        from .engine_ids import is_python_engine

        if is_python_engine(cfg.engine_id) and cfg.engine_id not in own:
            raise ValueError(f"heuristics for {family.name} returned python engine_id {cfg.engine_id}, which the family does not own or offer")
        assert not is_backend_block(cfg), "the BACKEND marker must be expanded by _assemble, never ranked"
    return plans
