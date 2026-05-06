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
TraceTemplate consistency tests.

These tests act as "linters" for trace templates. They catch mistakes like:
  - Wrong parameter names in the template (param= mismatch with the API)
  - Const axes that can never get a value (not in any tensor's dim_names)
  - fi_trace() returning "unknown" dtypes or missing Const-axis values

Two levels of checking
----------------------
1. **Structural** (no GPU, no real tensors): verify that every ``param=``
   reference in the template exists in the decorated function's signature,
   and that every ``Const`` axis has at least one tensor source.

2. **End-to-end** (CPU tensors, no GPU): call ``fi_trace`` with minimal
   auto-generated tensors and assert the returned dict is complete.

How to add a new template
--------------------------
When you add ``@flashinfer_api(trace=my_trace)`` to a function, add an
entry to ``_TEMPLATE_FUNC_PAIRS`` and optionally a targeted end-to-end test.
See the docstring in ``flashinfer/trace/templates/__init__.py`` for the full
how-to guide.
"""

import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple

import pytest
import torch

from flashinfer.trace.template import Const, Scalar, Tensor, TraceTemplate, Var

# ---------------------------------------------------------------------------
# Structural checker utilities
# ---------------------------------------------------------------------------


def _resolved_param(json_key: str, descriptor) -> str:
    """Return the function-parameter name that descriptor maps to."""
    p = getattr(descriptor, "param", None)
    return p if p is not None else json_key


def _get_sig_params(func: Callable) -> Optional[set]:
    """
    Return the set of parameter names for *func*, stripping ``self``/``cls``.
    Returns None if the signature cannot be inspected.
    """
    # Unwrap decorators to reach the original signature
    original = func
    for attr in ("__wrapped__", "__func__"):
        if hasattr(original, attr):
            original = getattr(original, attr)
    try:
        sig = inspect.signature(original)
    except (ValueError, TypeError):
        return None
    return {name for name, p in sig.parameters.items() if name not in ("self", "cls")}


def assert_template_signature_consistency(
    func: Callable,
    template: TraceTemplate,
    *,
    label: str = "",
) -> None:
    """
    Assert that every non-optional ``param=`` reference in *template* resolves
    to a valid parameter name of *func*.

    Optional inputs are skipped: they may reference plan-phase metadata (e.g.
    ``kv_indptr``) that lives in the wrapper's ``plan()`` method rather than
    ``run()``, and is intentionally absent from the run-time signature.

    This catches mistakes like renaming a function parameter without
    updating the corresponding ``param=`` in the template.
    """
    param_names = _get_sig_params(func)
    if param_names is None:
        return  # Cannot inspect — skip

    errors: List[str] = []
    for json_key, descriptor in template.inputs.items():
        if not isinstance(descriptor, (Tensor, Scalar)):
            continue
        if getattr(descriptor, "optional", False):
            continue  # Plan-phase or truly optional inputs may not be in run() sig
        p = _resolved_param(json_key, descriptor)
        if p not in param_names:
            errors.append(
                f"  Input '{json_key}' → param='{p}' not found in "
                f"{func.__qualname__}({sorted(param_names)})"
            )

    pfx = f"[{label}] " if label else ""
    assert not errors, (
        f"{pfx}Template '{template.name_prefix or template.op_type}' "
        f"has param mismatches:\n" + "\n".join(errors)
    )


def assert_template_axes_covered(
    template: TraceTemplate,
    *,
    label: str = "",
    func: Optional[Callable] = None,
) -> None:
    """
    Assert that every ``Const`` axis in *template* has at least one source:

    1. A tensor input whose ``dim_names`` contain the axis name, OR
    2. A scalar input whose key matches the axis name (scalar-kwarg fallback), OR
    3. A parameter of *func* matching the axis name (scalar-kwarg fallback for
       integer function arguments like ``top_k``, ``n_group``, ``block_size``).
    """
    tensor_dim_names: set = set()
    scalar_keys: set = set()
    for json_key, descriptor in template.inputs.items():
        if isinstance(descriptor, Tensor):
            tensor_dim_names.update(descriptor.dim_names)
        elif isinstance(descriptor, Scalar):
            scalar_keys.add(json_key)

    func_param_names: set = set()
    if func is not None:
        sig_params = _get_sig_params(func)
        if sig_params is not None:
            func_param_names = sig_params

    uncovered = [
        name
        for name, marker in template.axes.items()
        if isinstance(marker, Const)
        and name not in tensor_dim_names
        and name not in scalar_keys
        and name not in func_param_names
    ]

    pfx = f"[{label}] " if label else ""
    assert not uncovered, (
        f"{pfx}Template '{template.name_prefix or template.op_type}' "
        f"has Const axes with no tensor/scalar source: {uncovered}"
    )


# ---------------------------------------------------------------------------
# Auto-tensor generation for end-to-end checks
# ---------------------------------------------------------------------------

_DTYPE_MAP: Dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int32": torch.int32,
    "int64": torch.int64,
    "float8_e4m3fn": torch.float8_e4m3fn,
    "uint8": torch.uint8,
}


# Per-key sample values for integer scalars. A plain 0 is a valid int32 value
# but makes no semantic sense for block_size/top_k/etc. — using small positive
# defaults produces definitions that could actually be run.
_INT_SAMPLE_DEFAULTS: Dict[str, int] = {
    "block_size": 16,
    "top_k": 1,
    "n_group": 1,
    "topk_group": 1,
    "num_experts": 1,
    "intermediate_size": 1,
    "hidden_size": 1,
}


def _make_sample_kwargs(template: TraceTemplate, axis_size: int = 4) -> Dict[str, Any]:
    """
    Build minimal CPU tensors/scalars for every non-optional input in *template*.

    Each axis defaults to *axis_size*. Tuple inputs (``tuple_idx`` set) are
    collected into a tuple and stored under the shared ``param`` key.
    """
    sizes = {name: axis_size for name in template.axes}

    # Accumulate tuple parts: param → list indexed by tuple_idx
    tuple_parts: Dict[str, list] = {}
    kwargs: Dict[str, Any] = {}

    for json_key, descriptor in template.inputs.items():
        if isinstance(descriptor, Scalar):
            if descriptor.optional:
                continue
            p = _resolved_param(json_key, descriptor)
            if descriptor.dtype == "int32":
                kwargs[p] = _INT_SAMPLE_DEFAULTS.get(p, 1)
            else:
                kwargs[p] = 1.0

        elif isinstance(descriptor, Tensor):
            if descriptor.optional:
                continue
            p = _resolved_param(json_key, descriptor)
            shape = [sizes.get(d, axis_size) for d in descriptor.dim_names]
            if not shape:
                continue
            # Prefer the descriptor's own dtype hint; fall back to bfloat16
            dtype = _DTYPE_MAP.get(descriptor.dtype or "", torch.bfloat16)
            t = torch.zeros(shape, dtype=dtype)

            if descriptor.tuple_idx is not None:
                parts = tuple_parts.setdefault(p, [None, None])
                # Grow the list if needed
                while len(parts) <= descriptor.tuple_idx:
                    parts.append(None)
                parts[descriptor.tuple_idx] = t
            else:
                kwargs[p] = t

    # Finalise tuple inputs
    for p, parts in tuple_parts.items():
        kwargs[p] = tuple(parts)

    return kwargs


def assert_fi_trace_complete(
    func: Callable,
    template: TraceTemplate,
    *,
    label: str = "",
    axis_size: int = 4,
) -> Dict[str, Any]:
    """
    Call ``fi_trace`` with auto-generated sample tensors and verify:
    - No exception is raised
    - All ``Const`` axes have a ``value`` in the returned dict
    - No input or output has ``dtype == "unknown"``
    """
    sample_kwargs = _make_sample_kwargs(template, axis_size=axis_size)
    fi_api = f"{getattr(func, '__module__', '')}.{func.__qualname__}"
    fi_trace_fn = template.build_fi_trace_fn(fi_api)

    try:
        defn = fi_trace_fn(**sample_kwargs)
    except Exception as exc:  # noqa: BLE001
        pfx = f"[{label}] " if label else ""
        pytest.fail(
            f"{pfx}fi_trace raised an exception for template "
            f"'{template.name_prefix or template.op_type}': {exc}"
        )

    pfx = f"[{label}] " if label else ""
    name_tag = f"'{template.name_prefix or template.op_type}'"

    # Const axes must have resolved values
    missing_values = [
        name
        for name, entry in defn.get("axes", {}).items()
        if entry["type"] == "const" and "value" not in entry
    ]
    assert not missing_values, (
        f"{pfx}Template {name_tag}: Const axes missing values: {missing_values}"
    )

    # No "unknown" dtypes in non-optional inputs (optional inputs may be absent at run time)
    unknown_inputs = [
        k
        for k, v in defn.get("inputs", {}).items()
        if isinstance(v, dict)
        and v.get("dtype") == "unknown"
        and not v.get("optional", False)
    ]
    assert not unknown_inputs, (
        f"{pfx}Template {name_tag}: inputs with unknown dtype: {unknown_inputs}"
    )

    # No "unknown" dtypes in outputs
    unknown_outputs = [
        k
        for k, v in defn.get("outputs", {}).items()
        if isinstance(v, dict) and v.get("dtype") == "unknown"
    ]
    assert not unknown_outputs, (
        f"{pfx}Template {name_tag}: outputs with unknown dtype: {unknown_outputs}"
    )

    return defn


# ---------------------------------------------------------------------------
# Auto-discovery via _TRACE_REGISTRY
#
# @flashinfer_api(trace=...) automatically registers every (func, template)
# pair in flashinfer.api_logging._TRACE_REGISTRY at decoration time.
# We just need to import the modules that contain the decorated functions to
# trigger those decorators, then read the registry.
#
# To add a new kernel: no changes needed here — simply add
# @flashinfer_api(trace=my_template) to your function and the tests will
# pick it up automatically.
# ---------------------------------------------------------------------------


def _collect_template_func_pairs() -> List[Tuple[Callable, TraceTemplate, str]]:
    """
    Return all (func, template, label) pairs by reading _TRACE_REGISTRY.

    Imports are done lazily here so that missing GPU drivers don't prevent
    the structural tests from running.
    """
    # Trigger @flashinfer_api decorators by importing all modules that use them.
    import flashinfer.decode  # BatchDecodeWithPagedKVCacheWrapper
    import flashinfer.fused_moe  # trtllm_fp8_block_scale_moe
    import flashinfer.gdn_decode  # gated_delta_rule_decode, gated_delta_rule_mtp
    import flashinfer.gdn_prefill  # chunk_gated_delta_rule
    import flashinfer.gemm  # mm_bf16, mm_fp8, mm_mxfp8, mm_fp4
    import flashinfer.mla  # BatchMLAPagedAttentionWrapper
    import flashinfer.norm  # rmsnorm, fused_add_rmsnorm
    import flashinfer.prefill  # BatchPrefillWithPagedKVCacheWrapper, Ragged
    import flashinfer.sampling  # noqa: F401  # top_k_sampling_from_probs, etc.

    from flashinfer.api_logging import _TRACE_REGISTRY

    return list(_TRACE_REGISTRY)


_ALL_PAIRS = _collect_template_func_pairs()
_PAIR_IDS = [label for _, _, label in _ALL_PAIRS]


# ---------------------------------------------------------------------------
# Parameterized structural tests (no GPU required)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("func,template,label", _ALL_PAIRS, ids=_PAIR_IDS)
def test_template_signature_consistency(func, template, label):
    """Every param= reference in the template must exist in the function's signature."""
    assert_template_signature_consistency(func, template, label=label)


@pytest.mark.parametrize("func,template,label", _ALL_PAIRS, ids=_PAIR_IDS)
def test_template_axes_covered(func, template, label):
    """Every Const axis must be reachable from at least one input tensor, scalar, or function param."""
    assert_template_axes_covered(template, label=label, func=func)


# ---------------------------------------------------------------------------
# End-to-end checks: fi_trace with auto-generated CPU tensors
#
# The simpler ops (no tuple inputs, standard dtypes) are checked
# automatically. Wrappers with complex inputs (tuple paged_kv_cache, fp8
# scale tensors) are skipped here — their correctness is covered by the
# targeted tests in tests/test_fi_trace.py.
# ---------------------------------------------------------------------------

_E2E_SKIP = {
    # Tuple inputs (paged_kv_cache) need manual construction:
    "gqa_paged_decode",
    "gqa_paged_prefill",
    # MoE fp8: top_k / intermediate_size are scalar kwargs (not tensor dims) and
    # hidden_states_scale is optional — covered by test_fi_trace_complete_moe_routing.
    "moe_fp8_block_scale_ds_routing",
    "moe_fp8_block_scale_default_routing",
    "moe_fp8_block_scale_renormalize_routing",
    "moe_fp8_block_scale_llama4_routing",
    "moe_fp8_block_scale_renormalize_naive_routing",
    "moe_fp8_block_scale_topk_routing",
    # MoE fp4: same reason — covered by test_fi_trace_complete_moe_fp4_routing.
    "moe_fp4_block_scale_ds_routing",
    "moe_fp4_block_scale_default_routing",
    "moe_fp4_block_scale_renormalize_routing",
    "moe_fp4_block_scale_llama4_routing",
    "moe_fp4_block_scale_renormalize_naive_routing",
    "moe_fp4_block_scale_topk_routing",
}

_E2E_PAIRS = [(f, t, l) for f, t, l in _ALL_PAIRS if l not in _E2E_SKIP]
_E2E_IDS = [label for _, _, label in _E2E_PAIRS]


@pytest.mark.parametrize("func,template,label", _E2E_PAIRS, ids=_E2E_IDS)
def test_fi_trace_complete(func, template, label):
    """fi_trace with auto-generated CPU tensors must return a complete definition."""
    assert_fi_trace_complete(func, template, label=label)


# ---------------------------------------------------------------------------
# Targeted end-to-end checks for templates skipped above
# ---------------------------------------------------------------------------


def test_fi_trace_complete_gqa_paged_decode():
    """GQA paged decode: tuple paged_kv_cache input handled correctly."""
    from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper
    from flashinfer.trace.templates.attention import gqa_paged_decode_trace  # noqa: F401

    B, H, KV, D, P, NP = 4, 8, 4, 64, 16, 8
    q = torch.zeros(B, H, D, dtype=torch.bfloat16)
    k = torch.zeros(NP, P, KV, D, dtype=torch.bfloat16)
    v = torch.zeros(NP, P, KV, D, dtype=torch.bfloat16)

    defn = BatchDecodeWithPagedKVCacheWrapper.run.fi_trace(q=q, paged_kv_cache=(k, v))
    assert defn["axes"]["num_qo_heads"]["value"] == H
    assert defn["axes"]["page_size"]["value"] == P
    # Optional plan-phase inputs (kv_indptr, kv_indices, sm_scale) may have "unknown" dtype
    # when not passed to run(); only check non-optional inputs.
    non_optional_unknown = [
        k
        for k, v in defn["inputs"].items()
        if isinstance(v, dict)
        and v.get("dtype") == "unknown"
        and not v.get("optional", False)
    ]
    assert not non_optional_unknown, (
        f"Non-optional inputs with unknown dtype: {non_optional_unknown}"
    )
    assert "unknown" not in str(defn["outputs"])


@pytest.mark.parametrize(
    "routing_method_type,top_k,extra_kwargs,expected_name_prefix",
    [
        # routing_method_type 0 — Default (softmax top-k)
        (0, 4, {}, "moe_fp8_block_scale_default_routing"),
        # routing_method_type 1 — Renormalize (top-k then softmax)
        (1, 4, {}, "moe_fp8_block_scale_renormalize_routing"),
        # routing_method_type 2 — DeepSeekV3 (group routing; needs n_group / topk_group)
        (2, 4, {"n_group": 4, "topk_group": 2}, "moe_fp8_block_scale_ds_routing"),
        # routing_method_type 3 — Llama4 (top-1 sigmoid)
        (3, 1, {}, "moe_fp8_block_scale_llama4_routing"),
        # routing_method_type 4 — RenormalizeNaive (softmax → top-k → renorm)
        (4, 4, {}, "moe_fp8_block_scale_renormalize_naive_routing"),
        # routing_method_type 5 — TopK (uniform weights, no score normalisation)
        (5, 4, {}, "moe_fp8_block_scale_topk_routing"),
    ],
    ids=["default", "renormalize", "ds", "llama4", "renormalize_naive", "topk"],
)
def test_fi_trace_complete_moe_routing(
    routing_method_type, top_k, extra_kwargs, expected_name_prefix
):
    """MoE routing variants: fp8 + scale tensor shapes handled correctly for each routing type."""
    from flashinfer.fused_moe import trtllm_fp8_block_scale_moe

    T, E, EL, H, I, BS = 4, 16, 2, 256, 64, 128
    defn = trtllm_fp8_block_scale_moe.fi_trace(
        routing_logits=torch.zeros(T, E, dtype=torch.float32),
        routing_bias=torch.zeros(E, dtype=torch.bfloat16),
        hidden_states=torch.zeros(T, H, dtype=torch.float8_e4m3fn),
        hidden_states_scale=torch.ones(H // BS, T, dtype=torch.float32),
        gemm1_weights=torch.zeros(EL, 2 * I, H, dtype=torch.float8_e4m3fn),
        gemm1_weights_scale=torch.ones(EL, (2 * I) // BS, H // BS, dtype=torch.float32),
        gemm2_weights=torch.zeros(EL, H, I, dtype=torch.float8_e4m3fn),
        gemm2_weights_scale=torch.ones(EL, H // BS, I // BS, dtype=torch.float32),
        num_experts=E,
        top_k=top_k,
        intermediate_size=I,
        local_expert_offset=0,
        local_num_experts=EL,
        routed_scaling_factor=1.0,
        routing_method_type=routing_method_type,
        **extra_kwargs,
    )
    assert defn["op_type"] == "moe"
    assert defn["axes"]["num_local_experts"]["value"] == EL
    assert defn["axes"]["hidden_size"]["value"] == H
    assert defn["axes"]["top_k"]["value"] == top_k
    assert defn["name"].startswith(expected_name_prefix)
    assert "unknown" not in str(defn["inputs"])


@pytest.mark.parametrize(
    "routing_method_type,top_k,extra_kwargs,expected_name_prefix",
    [
        (0, 4, {}, "moe_fp4_block_scale_default_routing"),
        (1, 4, {}, "moe_fp4_block_scale_renormalize_routing"),
        (2, 4, {"n_group": 4, "topk_group": 2}, "moe_fp4_block_scale_ds_routing"),
        (3, 1, {}, "moe_fp4_block_scale_llama4_routing"),
        (4, 4, {}, "moe_fp4_block_scale_renormalize_naive_routing"),
        (5, 4, {}, "moe_fp4_block_scale_topk_routing"),
    ],
    ids=["default", "renormalize", "ds", "llama4", "renormalize_naive", "topk"],
)
def test_fi_trace_complete_moe_fp4_routing(
    routing_method_type, top_k, extra_kwargs, expected_name_prefix
):
    """MoE routing variants: fp4 + scale tensor shapes handled correctly for each routing type."""
    from flashinfer.fused_moe import trtllm_fp4_block_scale_moe

    # NvFP4: block_size=16, packed hidden → [T, H//2], scale → [T, H//16]
    T, E, EL, H, I, BS = 4, 16, 2, 256, 64, 16
    defn = trtllm_fp4_block_scale_moe.fi_trace(
        routing_logits=torch.zeros(T, E, dtype=torch.float32),
        routing_bias=None,
        hidden_states=torch.zeros(T, H // 2, dtype=torch.uint8),
        hidden_states_scale=torch.zeros(T, H // BS, dtype=torch.float8_e4m3fn),
        gemm1_weights=torch.zeros(EL, 2 * I, H // 2, dtype=torch.uint8),
        gemm1_weights_scale=torch.zeros(EL, 2 * I, H // BS, dtype=torch.float8_e4m3fn),
        gemm1_bias=None,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        gemm2_weights=torch.zeros(EL, H, I // 2, dtype=torch.uint8),
        gemm2_weights_scale=torch.zeros(EL, H, I // BS, dtype=torch.float8_e4m3fn),
        gemm2_bias=None,
        output1_scale_scalar=torch.ones(EL, dtype=torch.float32),
        output1_scale_gate_scalar=torch.ones(EL, dtype=torch.float32),
        output2_scale_scalar=torch.ones(EL, dtype=torch.float32),
        num_experts=E,
        top_k=top_k,
        intermediate_size=I,
        local_expert_offset=0,
        local_num_experts=EL,
        routed_scaling_factor=None,
        routing_method_type=routing_method_type,
        **extra_kwargs,
    )
    assert defn["op_type"] == "moe"
    assert defn["axes"]["num_local_experts"]["value"] == EL
    assert defn["axes"]["hidden_size"]["value"] == H
    assert defn["axes"]["top_k"]["value"] == top_k
    assert defn["name"].startswith(expected_name_prefix)
    non_optional_unknown = [
        k
        for k, v in defn["inputs"].items()
        if isinstance(v, dict) and v.get("dtype") == "unknown" and not v.get("optional")
    ]
    assert not non_optional_unknown, (
        f"Non-optional inputs with unknown dtype: {non_optional_unknown}"
    )


# ---------------------------------------------------------------------------
# Meta-tests: verify the checkers themselves catch broken templates
#
# These create intentionally wrong templates inline and assert that the
# checker utilities raise AssertionError.  If a checker ever silently
# ignores a bug, these tests will fail.
# ---------------------------------------------------------------------------


def _make_gdn_decode_func():
    """Return the real gated_delta_rule_decode for use in meta-tests."""
    import flashinfer.gdn_decode

    return flashinfer.gdn_decode.gated_delta_rule_decode


def test_checker_rejects_wrong_param():
    """Signature checker must catch a param= that doesn't exist in the function."""
    # 'state' in gated_delta_rule_decode is a required positional arg.
    # Deliberately map it to a non-existent param name 'hidden_state'.
    broken = TraceTemplate(
        op_type="gdn",
        name_prefix="gdn_decode_broken_param",
        axes={"batch_size": Var(), "head_size": Const(abbrev="d")},
        inputs={
            "q": Tensor(["batch_size", "head_size"]),
            # 'state' exists in the real function; 'hidden_state' does not.
            "state": Tensor(["batch_size", "head_size"], param="hidden_state"),
        },
        outputs={"output": Tensor(["batch_size", "head_size"], dtype_from="q")},
    )
    func = _make_gdn_decode_func()
    with pytest.raises(AssertionError, match="param=.*hidden_state.*not found"):
        assert_template_signature_consistency(func, broken, label="meta-test")


def test_checker_rejects_uncovered_const_axis():
    """Axes checker must catch a Const axis that has no tensor or function-param source."""
    broken = TraceTemplate(
        op_type="gdn",
        name_prefix="gdn_decode_broken_axis",
        axes={
            "batch_size": Var(),
            "head_size": Const(abbrev="d"),
            # 'mystery_dim' is a Const axis but appears in no tensor dim_names,
            # no Scalar input key, and no parameter of gated_delta_rule_decode.
            "mystery_dim": Const(abbrev="m"),
        },
        inputs={"q": Tensor(["batch_size", "head_size"])},
        outputs={"output": Tensor(["batch_size", "head_size"], dtype_from="q")},
    )
    func = _make_gdn_decode_func()
    with pytest.raises(AssertionError, match="mystery_dim"):
        assert_template_axes_covered(broken, label="meta-test", func=func)


def test_checker_rejects_unknown_dtype_in_e2e():
    """End-to-end checker must catch a template whose output dtype resolves to 'unknown'."""
    # dtype_from="nonexistent_input" refers to an input key that doesn't exist,
    # so the output dtype will be "unknown" at fi_trace time.
    broken = TraceTemplate(
        op_type="gdn",
        name_prefix="gdn_decode_broken_dtype",
        axes={"batch_size": Var(), "head_size": Const(abbrev="d")},
        inputs={"q": Tensor(["batch_size", "head_size"])},
        outputs={
            "output": Tensor(
                ["batch_size", "head_size"], dtype_from="nonexistent_input"
            )
        },
    )
    func = _make_gdn_decode_func()
    with pytest.raises(AssertionError, match="unknown dtype"):
        assert_fi_trace_complete(func, broken, label="meta-test")
