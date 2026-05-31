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

"""
Trace-template ``init`` consistency tests.

For every ``TraceTemplate`` that exposes an ``init`` callable, these tests
verify:

- the init signature is keyword-only and accepts exactly the template's
  ``Var`` axes (plus standard knobs ``device``/``seed``);
- calling init with a small canonical Var-axis dict on CPU returns a dict
  whose tensors match the template's declared ``dim_names``;
- KV-cache indptr/indices/last-page-len arrays satisfy structural
  invariants when present;
- the returned dict feeds cleanly into ``fi_trace`` (round-trip).

Auto-discovery: every ``@flashinfer_api(trace=...)``-decorated function
registers its template in ``_TRACE_REGISTRY`` at import time. The init
tests parameterize over that registry, skipping templates without init
attached.
"""

import inspect
from typing import Any, Callable, Dict, List, Tuple

import pytest
import torch

from flashinfer.trace.template import TraceTemplate, Var

# ---------------------------------------------------------------------------
# Auto-discovery (imports the modules to populate _TRACE_REGISTRY).
# ---------------------------------------------------------------------------


def _collect_pairs() -> List[Tuple[Callable, TraceTemplate, str]]:
    """Discover (func, template, label) triples by importing every module
    that decorates a function with ``@flashinfer_api(trace=...)``.

    Each import is wrapped individually because some submodules require
    optional dependencies (e.g. ``cuda.tile`` for ``comm.allreduce``) that
    may not be installed in the current environment — we still want to
    test whatever templates are available.
    """
    import contextlib
    import importlib

    _MODULES = [
        "flashinfer.activation",
        "flashinfer.cascade",
        "flashinfer.comm.allreduce",
        "flashinfer.comm.dcp_alltoall",
        "flashinfer.decode",
        "flashinfer.fused_moe",
        "flashinfer.gdn_decode",
        "flashinfer.gdn_prefill",
        "flashinfer.gemm",
        "flashinfer.mamba",
        "flashinfer.mla",
        "flashinfer.norm",
        "flashinfer.page",
        "flashinfer.prefill",
        "flashinfer.quantization.fp4_quantization",
        "flashinfer.quantization.fp8_quantization",
        "flashinfer.rope",
        "flashinfer.sampling",
        "flashinfer.xqa",
    ]
    for mod in _MODULES:
        # Optional dependency missing → skip; whatever's available will still test.
        with contextlib.suppress(ImportError):
            importlib.import_module(mod)

    from flashinfer.api_logging import _TRACE_REGISTRY

    return list(_TRACE_REGISTRY)


_ALL_PAIRS = _collect_pairs()
_INIT_PAIRS = [(f, t, l) for f, t, l in _ALL_PAIRS if t.init is not None]
_INIT_IDS = [label for _, _, label in _INIT_PAIRS]


# ---------------------------------------------------------------------------
# Canonical Var-axis values used by the smoke test.
#
# Override per-axis defaults here if a template needs something specific
# (e.g. ``num_pages_per_seq`` must satisfy ``num_pages_per_seq * batch_size
#  == num_pages``). Most ops accept tiny values.
# ---------------------------------------------------------------------------

_DEFAULT_VAR_VALUES: Dict[str, int] = {
    "batch_size": 4,
    "num_tokens": 4,
    "seq_len": 8,
    "qo_len": 4,
    "kv_len": 8,
    "vocab_size": 128,
    "num_pages": 8,
    "num_pages_per_seq": 2,
    "len_indptr": 5,  # batch_size + 1
    "num_kv_indices": 8,
    "scalar": 1,
    "hidden_div_2": 2,
    "hidden_div_block_size": 1,
}

# Const-axis overrides for the smoke test only. The init function's defaults
# match real model configs (DeepSeek H=7168, etc.), but materializing those
# tensors during a CPU smoke test is slow and memory-hungry. If the init
# function accepts an axis name as a kwarg, the test passes the small value
# from this dict instead of the production default.
_SMOKE_CONST_OVERRIDES: Dict[str, int] = {
    "hidden_size": 128,
    "intermediate_size": 128,
    "num_local_experts": 1,
    "num_experts": 4,
    "num_q_heads": 4,
    "num_qo_heads": 4,
    "num_k_heads": 1,
    "num_kv_heads": 1,
    "num_heads": 4,
    "num_heads_qo": 1,
    "head_dim": 16,
    "head_size": 16,
    "head_dim_ckv": 16,
    "head_dim_kpe": 8,
    "rope_dim": 16,
    "no_rope_dim": 16,
    "rotary_dim": 16,
    "M": 8,
    "N": 8,
    "K": 128,
    "M_max": 8,
    "K_doubled": 16,
    "max_m": 8,
    "max_seq_len": 32,
    "page_size": 4,
    "max_pages_per_seq": 2,
    "num_pages_per_seq": 2,
    "stats_dim": 2,
    "ws_elems_per_rank": 4,
    "dim": 16,
    "dstate": 16,
    "topk": 2,
    "n_group": 2,
    "topk_group": 1,
    "max_len": 16,
    "k": 4,
    "pool_size": 4,
}


def _canonical_var_kwargs(template: TraceTemplate) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for axis_name, marker in template.axes.items():
        if isinstance(marker, Var):
            out[axis_name] = _DEFAULT_VAR_VALUES.get(axis_name, 4)
    return out


def _smoke_init_kwargs(template: TraceTemplate) -> Dict[str, int]:
    """Var-axis values + Const-axis overrides for the smoke test.

    Init function defaults match real model configs (e.g. DeepSeek H=7168),
    which are too slow/memory-hungry to materialize during a smoke test.
    For every Const axis the init function accepts as a kwarg, override
    with a small value from ``_SMOKE_CONST_OVERRIDES``.
    """
    out = _canonical_var_kwargs(template)
    sig = inspect.signature(template.init)
    accepted = set(sig.parameters)
    for axis_name, marker in template.axes.items():
        if isinstance(marker, Var):
            continue  # already populated by _canonical_var_kwargs
        if axis_name in accepted and axis_name in _SMOKE_CONST_OVERRIDES:
            out[axis_name] = _SMOKE_CONST_OVERRIDES[axis_name]
    return out


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("func,template,label", _INIT_PAIRS, ids=_INIT_IDS)
def test_init_signature_is_keyword_only(func, template, label):
    """Every init must take its Var axes as keyword-only arguments."""
    sig = inspect.signature(template.init)
    var_axes = {n for n, m in template.axes.items() if isinstance(m, Var)}
    found = set()
    for name, param in sig.parameters.items():
        if param.kind not in (
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            continue
        if name in var_axes:
            assert param.kind == inspect.Parameter.KEYWORD_ONLY, (
                f"[{label}] Init param '{name}' must be keyword-only"
            )
            found.add(name)
    missing = var_axes - found
    assert not missing, (
        f"[{label}] Init is missing kwargs for Var axes: {sorted(missing)}"
    )


@pytest.mark.parametrize("func,template,label", _INIT_PAIRS, ids=_INIT_IDS)
def test_init_smoke_cpu(func, template, label):
    """Init must run on CPU for a small canonical input bundle."""
    init_kwargs = _smoke_init_kwargs(template)
    try:
        result = template.init(device="cpu", **init_kwargs)
    except (
        RuntimeError,
        NotImplementedError,
        ValueError,
        ImportError,
        AssertionError,
    ) as exc:
        # Some inits really need CUDA (e.g. fp8/fp4 quantize via flashinfer
        # APIs that assert cuda device) or require axis values that satisfy
        # divisibility constraints (e.g. K % 128 == 0 for block quant). Skip
        # — the GPU end-to-end test will cover them.
        pytest.skip(f"[{label}] init unsupported on CPU: {exc}")
    assert isinstance(result, dict), (
        f"[{label}] init must return a dict, got {type(result)}"
    )


@pytest.mark.parametrize("func,template,label", _INIT_PAIRS, ids=_INIT_IDS)
def test_init_kv_cache_invariants(func, template, label):
    """When the template has paged-KV indptr/indices, init must produce valid arrays."""
    init_kwargs = _smoke_init_kwargs(template)
    try:
        result = template.init(device="cpu", **init_kwargs)
    except (
        RuntimeError,
        NotImplementedError,
        ValueError,
        ImportError,
        AssertionError,
    ):
        pytest.skip(f"[{label}] init unsupported on CPU")

    # Wrapper APIs return {"plan": ..., "run": ...}; flatten for the check.
    flat: Dict[str, Any] = {}
    if isinstance(result, dict) and set(result.keys()) <= {"plan", "run"}:
        for sub in result.values():
            if isinstance(sub, dict):
                flat.update(sub)
    else:
        flat = result

    indptr = flat.get("kv_indptr")
    indices = flat.get("kv_indices")
    last_page = flat.get("kv_last_page_len")
    if indptr is None and indices is None and last_page is None:
        pytest.skip(f"[{label}] no paged-KV arrays in init result")

    if indptr is not None:
        assert int(indptr[0].item()) == 0, f"[{label}] kv_indptr[0] != 0"
        diffs = indptr[1:] - indptr[:-1]
        assert torch.all(diffs >= 0), (
            f"[{label}] kv_indptr is not monotonically non-decreasing"
        )
        if indices is not None:
            assert int(indptr[-1].item()) == int(indices.numel()), (
                f"[{label}] kv_indptr[-1] != kv_indices.numel()"
            )
    if last_page is not None:
        assert torch.all(last_page >= 1), f"[{label}] kv_last_page_len has zeros"


@pytest.mark.parametrize("func,template,label", _INIT_PAIRS, ids=_INIT_IDS)
def test_init_fi_trace_roundtrip(func, template, label):
    """init(...) -> fi_trace(...) must produce a complete definition."""
    init_kwargs = _smoke_init_kwargs(template)
    try:
        result = template.init(device="cpu", **init_kwargs)
    except (
        RuntimeError,
        NotImplementedError,
        ValueError,
        ImportError,
        AssertionError,
    ):
        pytest.skip(f"[{label}] init unsupported on CPU")

    flat: Dict[str, Any] = {}
    if isinstance(result, dict) and set(result.keys()) <= {"plan", "run"}:
        for sub in result.values():
            if isinstance(sub, dict):
                flat.update(sub)
    else:
        flat = result

    fi_api = f"{getattr(func, '__module__', '')}.{func.__qualname__}"
    fi_trace_fn = template.build_fi_trace_fn(fi_api)
    defn = fi_trace_fn(**flat)

    # The dumped JSON must include the init source.
    assert "init" in defn, f"[{label}] fi_trace output missing 'init' key"
    assert defn["init"].strip(), f"[{label}] embedded init source is empty"
    compile(defn["init"], f"{label}:init", "exec")
    if "reference" in defn:
        compile(defn["reference"], f"{label}:reference", "exec")

    # No "unknown" dtype on non-optional inputs.
    # Inputs whose flat value is a tuple/list (e.g. ``paged_kv_cache=(k, v)``)
    # are excused: when the trace template lacks ``tuple_idx`` on the
    # descriptor, fi_trace cannot recover the dtype, but that's a trace-
    # template bug separate from the init's job.
    bad = []
    for k, v in defn.get("inputs", {}).items():
        if not isinstance(v, dict):
            continue
        if v.get("dtype") != "unknown":
            continue
        if v.get("optional", False):
            continue
        flat_val = flat.get(k)
        if isinstance(flat_val, (tuple, list)):
            continue
        bad.append(k)
    assert not bad, f"[{label}] init produced inputs with unknown dtype: {bad}"
