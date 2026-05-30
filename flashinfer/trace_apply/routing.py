# Copyright (c) 2025 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Routing: map a call's definition identity to its single registered solution.

Routing key = ``(fi_api, const-axes, input-dtypes)`` — the definition identity:

* ``const-axes`` are the compile-time shape the definition is specialized for
  (``hidden_size=4096``, ``head_dim=128`` …). *Var* axes (``batch_size``,
  ``num_kv_indices`` …) are NOT in the key — one solution handles all of them.
* ``input-dtypes`` distinguishes dtype-specialized solutions (fp16 vs bf16) for
  the same shape; without it they would collide on one key.

This is the "one solution per definition, no winner-picking" model.
"""

from __future__ import annotations

import logging

from flashinfer.trace.solution import Solution
from flashinfer.trace_apply.config import ApplyPolicy, Definition, TraceConfig

_log = logging.getLogger("flashinfer.trace_apply")


# Routing key components are kept as sorted tuples so the key is hashable and
# order-independent.
_ConstAxes = tuple  # tuple[tuple[str, int], ...]
_Dtypes = frozenset  # frozenset[tuple[str, str]]


class IndexKey(tuple):
    """``(fi_api, const-axes, input-dtypes)`` as a hashable tuple."""

    __slots__ = ()

    def __new__(
        cls,
        fi_api: str,
        const_axes: frozenset[tuple[str, int]],
        input_dtypes: frozenset[tuple[str, str]],
    ) -> "IndexKey":
        return super().__new__(cls, (fi_api, const_axes, input_dtypes))

    @property
    def fi_api(self) -> str:
        return self[0]


class Candidate:
    """The single solution routed for a definition (plus its definition)."""

    __slots__ = ("solution", "definition")

    def __init__(self, solution: Solution, definition: Definition) -> None:
        self.solution = solution
        self.definition = definition


class Index:
    """The routing table built from definitions + solutions."""

    __slots__ = ("by_key", "const_axis_names")

    def __init__(self) -> None:
        self.by_key: dict[IndexKey, Candidate] = {}
        # fi_api -> set of const-axis names (used to slice the runtime axis
        # vector down to the routing key).
        self.const_axis_names: dict[str, set[str]] = {}

    def get(self, key: IndexKey) -> Candidate | None:
        return self.by_key.get(key)

    def has_candidates_for(self, fi_api: str) -> bool:
        return any(k.fi_api == fi_api for k in self.by_key)

    def const_names(self, fi_api: str) -> set[str]:
        return self.const_axis_names.get(fi_api, set())


def build_index(config: TraceConfig) -> Index:
    """Build the routing table from definitions + solutions.

    Each Solution declares its ``definition``; we map that definition's identity
    ``(fi_api, const-axes, input-dtypes)`` to the solution. Exactly one solution
    per definition — duplicates keep the first and warn.
    """
    idx = Index()
    for sol in config.solutions.values():
        defn = config.definitions.get(sol.definition)
        if defn is None:
            _log.debug(
                "Trace Apply: solution %s references unknown definition %s; skipping.",
                sol.name,
                sol.definition,
            )
            continue
        fi_api = defn.fi_api()
        if fi_api is None:
            _log.debug(
                "Trace Apply: definition %s has no fi_api tag; skipping.", defn.name
            )
            continue
        key = IndexKey(
            fi_api,
            frozenset(defn.const_axes().items()),
            defn.input_dtypes(),
        )
        if key in idx.by_key:
            _log.warning(
                "Trace Apply: multiple solutions for definition identity of %s (%s); "
                "keeping %s, ignoring %s. (One solution per definition is expected.)",
                defn.name,
                fi_api,
                idx.by_key[key].solution.name,
                sol.name,
            )
            continue
        idx.by_key[key] = Candidate(solution=sol, definition=defn)
        idx.const_axis_names.setdefault(fi_api, set()).update(defn.const_axes().keys())
    return idx


# ---------------------------------------------------------------------------
# Lookup + filters
# ---------------------------------------------------------------------------


def _arch_ok(target_hardware: tuple[str, ...], sm_arch: str | None) -> bool:
    """Soft arch safety by *compute capability* (no GPU-SKU name table).

    If the solution lists explicit ``sm<NN>`` targets, the running SM must be
    among them. If it lists only non-``sm`` tokens (e.g. ``"cuda"``) we cannot
    determine an arch constraint, so we don't block.
    """
    if sm_arch is None:
        return True
    sm_tokens = {t.lower() for t in target_hardware if t.lower().startswith("sm")}
    if not sm_tokens:
        return True
    return sm_arch.lower() in sm_tokens


def _passes_filters(c: Candidate, sm_arch: str | None, policy: ApplyPolicy) -> bool:
    if not _arch_ok(c.solution.spec.target_hardware, sm_arch):
        return False
    if policy.allowed_authors is not None and c.solution.author not in policy.allowed_authors:
        return False
    return True


def lookup(
    index: Index,
    fi_api: str,
    const_axes: dict[str, int],
    input_dtypes: frozenset[tuple[str, str]],
    sm_arch: str | None,
    policy: ApplyPolicy | None = None,
) -> Candidate | None:
    """Return the routed solution for the call's identity, or None on a miss."""
    pol = policy or ApplyPolicy()
    key = IndexKey(fi_api, frozenset(const_axes.items()), input_dtypes)
    cand = index.get(key)
    if cand is None:
        return None
    return cand if _passes_filters(cand, sm_arch, pol) else None


def explain(
    index: Index,
    fi_api: str,
    const_axes: dict[str, int],
    input_dtypes: frozenset[tuple[str, str]],
    sm_arch: str | None,
    policy: ApplyPolicy | None = None,
) -> dict:
    """Structured account of how a lookup resolves (introspection/debugging)."""
    pol = policy or ApplyPolicy()
    key = IndexKey(fi_api, frozenset(const_axes.items()), input_dtypes)
    cand = index.get(key)
    selected = cand if (cand and _passes_filters(cand, sm_arch, pol)) else None
    return {
        "fi_api": fi_api,
        "const_axes": dict(const_axes),
        "input_dtypes": dict(input_dtypes),
        "sm_arch": sm_arch,
        "routed": cand.solution.name if cand else None,
        "selected": (
            {
                "solution": selected.solution.name,
                "definition": selected.definition.name,
                "author": selected.solution.author,
                "language": selected.solution.spec.language.value,
            }
            if selected
            else None
        ),
        "rejected_reason": (
            None
            if selected
            else ("no candidate" if cand is None else "failed arch/author filter")
        ),
    }


__all__ = ["IndexKey", "Candidate", "Index", "build_index", "lookup", "explain"]
