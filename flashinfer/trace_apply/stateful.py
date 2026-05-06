"""Plan/run adapters for stateful flashinfer wrapper APIs.

Wrapper classes (``BatchDecodeWithPagedKVCacheWrapper`` et al.) split a call
across ``plan()`` and ``run()``: ``run()`` receives the query + paged KV cache,
while the indexing tensors (``kv_indptr``/``kv_indices``/...) and ``sm_scale``
are passed to ``plan()`` and stored on the instance. A Trace Apply candidate is
written against the *Definition's inputs* (q, k_cache, v_cache, kv_indptr,
kv_indices, sm_scale), so to extract axes and to invoke the candidate we must
recover the plan-time tensors.

Each adapter maps ``template_input_json_key -> plan() parameter name`` for the
inputs that originate at ``plan()``. Inputs not listed come from ``run()`` and
are resolved straight from the run signature (via the template descriptor).

Plan parameter names verified against the live signatures:
  decode.plan(indptr, indices, last_page_len, ...)
  prefill_paged.plan(qo_indptr, paged_kv_indptr, paged_kv_indices, ...)
  prefill_ragged.plan(qo_indptr, kv_indptr, ...)
  mla.plan(qo_indptr, kv_indptr, kv_indices, kv_len_arr, ...)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True, frozen=True)
class StatefulAdapter:
    plan_attr: str  # attribute name of the plan method on the same class
    # template_input_json_key -> plan() parameter name (used when the engine
    # calls the public plan(); we wrap it and stash the kwargs).
    plan_inputs: dict[str, str] = field(default_factory=dict)
    # template_input_json_key -> wrapper-instance attribute name. Read off
    # ``self`` at run() time. This is the robust path: it works whether the
    # engine called plan() OR a fast-path planner (e.g. SGLang's
    # fast_decode_plan) that writes the buffers directly. Preferred over the
    # plan-stash when both are available.
    self_attrs: dict[str, str] = field(default_factory=dict)


# Keyed by fi_api (module.qualname of the wrapper's run method).
STATEFUL_ADAPTERS: dict[str, StatefulAdapter] = {
    "flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper.run": StatefulAdapter(
        plan_attr="plan",
        plan_inputs={"kv_indptr": "indptr", "kv_indices": "indices", "sm_scale": "sm_scale"},
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
        plan_inputs={"qo_indptr": "qo_indptr", "kv_indptr": "kv_indptr", "sm_scale": "sm_scale"},
    ),
    "flashinfer.mla._core.BatchMLAPagedAttentionWrapper.run": StatefulAdapter(
        plan_attr="plan",
        plan_inputs={"kv_indptr": "kv_indptr", "kv_indices": "kv_indices", "sm_scale": "sm_scale"},
    ),
}


def adapter_for(fi_api: str) -> StatefulAdapter | None:
    return STATEFUL_ADAPTERS.get(fi_api)


def is_stateful(fi_api: str) -> bool:
    return fi_api in STATEFUL_ADAPTERS


__all__ = ["StatefulAdapter", "STATEFUL_ADAPTERS", "adapter_for", "is_stateful"]
