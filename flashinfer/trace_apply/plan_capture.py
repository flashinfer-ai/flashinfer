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

"""Plan/run state recovery for stateful Wrapper-class APIs.

Wrapper classes (``BatchDecodeWithPagedKVCacheWrapper`` et al.) split a call
across ``plan()`` and ``run()``: ``run()`` gets the query + paged KV cache,
while the indexing tensors (``kv_indptr``/``kv_indices``/…) and ``sm_scale`` are
passed to ``plan()`` and stored on the instance. A solution is written against
the *Definition's inputs*, so to extract axes and invoke the solution we recover
the plan-time tensors and merge them into the run() namespace.

Two recovery sources, in priority order:
1. **Instance attributes** (``self_attrs``) — robust to fast-path planners that
   bypass the public ``plan()`` (e.g. SGLang's ``fast_decode_plan``).
2. **Stashed plan() kwargs** — when the public ``plan()`` was wrapped and ran.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass, field
from threading import Lock
from typing import Any


@dataclass(slots=True, frozen=True)
class StatefulAdapter:
    plan_attr: str  # name of the plan method on the same class
    # template_input_json_key -> plan() parameter name
    plan_inputs: dict[str, str] = field(default_factory=dict)
    # template_input_json_key -> wrapper-instance attribute name (preferred path)
    self_attrs: dict[str, str] = field(default_factory=dict)


# Keyed by fi_api (module.qualname of the wrapper's run method).
STATEFUL_ADAPTERS: dict[str, StatefulAdapter] = {
    "flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper.run": StatefulAdapter(
        plan_attr="plan",
        plan_inputs={
            "kv_indptr": "indptr",
            "kv_indices": "indices",
            "sm_scale": "sm_scale",
        },
        self_attrs={
            "kv_indptr": "_paged_kv_indptr_buf",
            "kv_indices": "_paged_kv_indices_buf",
            "sm_scale": "_sm_scale",
        },
    ),
    "flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper.run": StatefulAdapter(
        plan_attr="plan",
        plan_inputs={
            "qo_indptr": "qo_indptr",
            "kv_indptr": "paged_kv_indptr",
            "kv_indices": "paged_kv_indices",
            "sm_scale": "sm_scale",
        },
    ),
    "flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper.run": StatefulAdapter(
        plan_attr="plan",
        plan_inputs={
            "qo_indptr": "qo_indptr",
            "kv_indptr": "kv_indptr",
            "sm_scale": "sm_scale",
        },
    ),
    "flashinfer.mla._core.BatchMLAPagedAttentionWrapper.run": StatefulAdapter(
        plan_attr="plan",
        plan_inputs={
            "kv_indptr": "kv_indptr",
            "kv_indices": "kv_indices",
            "sm_scale": "sm_scale",
        },
    ),
}


def adapter_for(fi_api: str) -> StatefulAdapter | None:
    return STATEFUL_ADAPTERS.get(fi_api)


def is_stateful(fi_api: str) -> bool:
    return fi_api in STATEFUL_ADAPTERS


# ---------------------------------------------------------------------------
# Plan-kwargs stash (keyed by the wrapper instance; freed when it is GC'd)
# ---------------------------------------------------------------------------

_plan_lock = Lock()
_plan_by_instance: "weakref.WeakKeyDictionary[Any, dict[str, Any]]" = (
    weakref.WeakKeyDictionary()
)


def stash_plan_kwargs(instance: Any, kwargs: dict[str, Any]) -> None:
    """Record the bound kwargs seen at ``plan()`` time for this instance."""
    try:
        with _plan_lock:
            _plan_by_instance[instance] = dict(kwargs)
    except TypeError:
        # Instance not weak-referenceable (unusual); skip rather than leak/crash
        # — run() will fall back to reading instance attributes.
        pass


def fetch_plan_kwargs(instance: Any) -> dict[str, Any]:
    with _plan_lock:
        return dict(_plan_by_instance.get(instance, {}))


def _ns_key(template: Any, json_key: str) -> str:
    """The namespace (param) name a template input is read under."""
    desc = template.inputs.get(json_key) if template is not None else None
    return (getattr(desc, "param", None) or json_key) if desc is not None else json_key


def augment_namespace(
    adapter: StatefulAdapter,
    template: Any,
    namespace: dict[str, Any],
    self_obj: Any,
) -> dict[str, Any]:
    """Merge plan-derived inputs into the run() namespace, in place.

    Prefers reading the buffers off the wrapper instance (robust to fast-path
    planners); falls back to the kwargs stashed by a wrapped public ``plan()``.
    """
    if self_obj is None:
        return namespace
    for json_key, attr in adapter.self_attrs.items():
        val = getattr(self_obj, attr, None)
        if val is not None:
            namespace[_ns_key(template, json_key)] = val
    plan_bound = fetch_plan_kwargs(self_obj)
    for json_key, plan_param in adapter.plan_inputs.items():
        key = _ns_key(template, json_key)
        if namespace.get(key) is None:
            val = plan_bound.get(plan_param)
            if val is not None:
                namespace[key] = val
    return namespace


__all__ = [
    "STATEFUL_ADAPTERS",
    "StatefulAdapter",
    "adapter_for",
    "augment_namespace",
    "fetch_plan_kwargs",
    "is_stateful",
    "stash_plan_kwargs",
]
