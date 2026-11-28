"""Verify dumped ``init``/``reference`` source strings are runnable standalone.

Dump-time verification calls the live module functions, whose helpers resolve
via module globals — so a rendered string that misses an inlined dependency
(e.g. ``_trtllm_kv_from_cache``) still "works" at dump time and only breaks
for downstream consumers that ``exec()`` the JSON. These tests exec the
rendered strings in a fresh namespace and invoke them once on CPU.
"""

import json
from pathlib import Path

import pytest
import torch

FI_TRACE_OUT = Path(__file__).resolve().parents[2] / "trace/fi_trace_out"
BLOCK_SPARSE_JSON = (
    FI_TRACE_OUT / "trtllm_batch_decode_block_sparse_h16_kv2_d128_ps16.json"
)

_INIT_KWARGS = dict(
    num_tokens=4,
    num_heads=16,
    num_kv_heads=2,
    head_dim=128,
    page_size=16,
    batch_size=4,
    max_pages_per_seq=8,
    device="cpu",
)


def _exec_in_fresh_namespace(source: str) -> dict:
    namespace: dict = {}
    exec(source, namespace)  # noqa: S102
    return namespace


@pytest.mark.cpu_only
def test_block_sparse_json_init_and_reference_standalone():
    """The committed block-sparse JSON must exec and run without flashinfer."""
    doc = json.loads(BLOCK_SPARSE_JSON.read_text())

    init_ns = _exec_in_fresh_namespace(doc["init"])
    init_fn = init_ns["_trtllm_batch_decode_block_sparse_init"]
    inputs = init_fn(**_INIT_KWARGS)
    assert inputs["enable_block_sparse_attention"] is True
    assert inputs["block_tables"].shape == (2, 4, 8)
    assert inputs["seq_lens"].shape == (2, 4)

    ref_ns = _exec_in_fresh_namespace(doc["reference"])
    ref_fn = ref_ns["_trtllm_batch_decode_block_sparse_reference"]
    output = ref_fn(**inputs)
    assert output.shape == inputs["query"].shape
    assert output.dtype == inputs["query"].dtype
    assert torch.isfinite(output.float()).all()


@pytest.mark.cpu_only
def test_dcp_token_json_init_uses_schema_head_dim():
    """The saved DCP trace must initialize its constant width without an override."""
    doc = json.loads(
        (FI_TRACE_OUT / "decode_cp_a2a_lse_reduce_tokens_d512.json").read_text()
    )
    init_ns = _exec_in_fresh_namespace(doc["init"])
    init_fn = init_ns["_decode_cp_a2a_lse_reduce_token_init"]
    inputs = init_fn(batch=4, heads=2, cp_size=8, workspace_bytes=16, device="cpu")

    head_dim = doc["axes"]["head_dim"]["value"]
    assert inputs["partial_o"].shape == (4, 8, 2, head_dim)
    assert inputs["partial_lse"].shape == (4, 8, 2)
    assert inputs["partial_lse"].dtype == torch.float32
    assert inputs["mask"].shape == (4,)
    assert inputs["mask"].dtype == torch.bool

    ref_ns = _exec_in_fresh_namespace(doc["reference"])
    output = ref_ns["_decode_cp_a2a_lse_reduce_reference"](**inputs)
    assert output.shape == (4, 2, head_dim)
    assert output.dtype == inputs["partial_o"].dtype
    assert torch.isfinite(output.float()).all()


@pytest.mark.cpu_only
def test_dcp_trace_dispatch_selects_template_for_fi_init():
    """fi_init passes axis values, so dispatch must not require partial_o."""
    from flashinfer.trace.templates.comm import (
        decode_cp_a2a_lse_reduce_token_trace,
        decode_cp_a2a_lse_reduce_trace,
        decode_cp_a2a_lse_reduce_trace_dispatch,
    )

    tpl = decode_cp_a2a_lse_reduce_trace_dispatch(batch=4, heads=2, cp_size=8)
    assert tpl is decode_cp_a2a_lse_reduce_token_trace
    inputs = tpl.init(batch=4, heads=2, cp_size=8, device="cpu")
    assert inputs["partial_o"].shape == (4, 8, 2, 512)
    assert inputs["mask"].shape == (4,)

    tpl = decode_cp_a2a_lse_reduce_trace_dispatch(heads=2, cp_size=8)
    assert tpl is decode_cp_a2a_lse_reduce_trace
    inputs = tpl.init(heads=2, cp_size=8, device="cpu")
    assert inputs["partial_o"].shape == (8, 2, 128)


@pytest.mark.cpu_only
def test_dense_trtllm_decode_reference_renders_standalone():
    """The dense decode reference shares helpers; its rendered source must too."""
    from flashinfer.trace.template import _render_reference_source
    from flashinfer.trace.templates.attention import (
        _trtllm_batch_decode_init,
        _trtllm_batch_decode_reference,
    )

    source = _render_reference_source(_trtllm_batch_decode_reference)
    namespace = _exec_in_fresh_namespace(source)
    assert "_trtllm_kv_from_cache" in namespace
    assert "_trtllm_paged_attention_reference" in namespace

    inputs = _trtllm_batch_decode_init(**_INIT_KWARGS)
    output = namespace["_trtllm_batch_decode_reference"](**inputs)
    assert output.shape == inputs["query"].shape
    assert torch.isfinite(output.float()).all()


@pytest.mark.cpu_only
def test_trtllm_gen_routing_init_renders_standalone():
    """The routing init must not leak flashinfer names into the dumped source.

    ``routing_method`` is the trap here: writing ``RoutingMethodType.Renormalize``
    dumps fine (the live module resolves it) but leaves a NameError for anyone
    exec'ing the JSON, so the init emits the plain enum value instead.
    """
    from flashinfer.trace.template import _render_init_source
    from flashinfer.trace.templates.moe import _trtllm_gen_routing_init

    namespace = _exec_in_fresh_namespace(_render_init_source(_trtllm_gen_routing_init))
    inputs = namespace["_trtllm_gen_routing_init"](
        num_tokens=4, num_experts=8, top_k=2, device="cpu"
    )
    assert inputs["routing_logits"].shape == (4, 8)
    assert inputs["routing_method"] == 1  # RoutingMethodType.Renormalize
    assert inputs["top_k"] == 2


PAGED_MQA_FP4_JSON = FI_TRACE_OUT / "fp4_paged_mqa_logits_nn2_H64_Dp64_bs64.json"
PAGED_MQA_FP8_JSON = FI_TRACE_OUT / "fp8_paged_mqa_logits_nn2_H64_D128_bs64.json"

# num_heads / head_dim are fixed by the FP4 kernel; keep the rest small so the
# pure-torch reference stays quick on CPU.
_PMQA_INIT_KWARGS = dict(
    batch_size=1,
    next_n=1,
    num_heads=64,
    head_dim=128,
    block_size=64,
    max_seq_len=256,
    device="cpu",
)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "json_path,prefix",
    [
        (PAGED_MQA_FP4_JSON, "_fp4_paged_mqa_logits"),
        (PAGED_MQA_FP8_JSON, "_fp8_paged_mqa_logits"),
    ],
    ids=["fp4", "fp8"],
)
def test_paged_mqa_logits_json_init_and_reference_standalone(json_path, prefix):
    """The committed paged-MQA JSONs must exec and run without flashinfer.

    A rendered string that omits an inlined dependency still "works" at dump
    time, because the live module globals resolve it -- so this only shows up
    for a consumer that exec()s the JSON. The committed FP4 artifact previously
    called _pack_ue8m0_to_int() without defining it (and emitted
    _quantize_to_fp4_e2m1 twice), which this test now catches.
    Regression for PR #4365 review r3825110225.
    """
    doc = json.loads(json_path.read_text())

    init_ns = _exec_in_fresh_namespace(doc["init"])
    init_fn = init_ns[f"{prefix}_init"]
    inputs = init_fn(**_PMQA_INIT_KWARGS)

    ref_ns = _exec_in_fresh_namespace(doc["reference"])
    ref_fn = ref_ns[f"{prefix}_reference"]
    out = ref_fn(**inputs)

    rows = _PMQA_INIT_KWARGS["batch_size"] * _PMQA_INIT_KWARGS["next_n"]
    assert out.shape == (rows, _PMQA_INIT_KWARGS["max_seq_len"])
    # dtype comes from the reference's own default; it must match what the
    # artifact's output schema advertises.
    assert str(out.dtype).rsplit(".", 1)[-1] == doc["outputs"]["logits"]["dtype"]
    assert torch.isfinite(out.float()[out.float() > float("-inf")]).all()


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "json_path,prefix,weight_bias",
    [
        (
            FI_TRACE_OUT / "fused_add_rmsnorm_h5120.json",
            "_fused_add_rmsnorm",
            0.0,
        ),
        (
            FI_TRACE_OUT / "gemma_fused_add_rmsnorm_h4608.json",
            "_gemma_fused_add_rmsnorm",
            1.0,
        ),
    ],
    ids=["fused_add_rmsnorm", "gemma_fused_add_rmsnorm"],
)
def test_fused_add_rmsnorm_json_reference_returns_updated_residual(
    json_path, prefix, weight_bias
):
    """The committed fused-add-RMSNorm JSONs must return every declared output.

    The kernel writes two buffers (``input`` <- norm output, ``residual`` <-
    ``residual + input``), so the schema declares ``output`` and ``residual``.
    The reference used to compute the sum in a local and return only the norm
    output, so consumers that exec() the JSON never saw the updated residual.
    With input = 1 and residual = 2, the updated residual must be 3.
    Regression for issue #9829.
    """
    doc = json.loads(json_path.read_text())
    hidden_size = doc["axes"]["hidden_size"]["value"]

    init_ns = _exec_in_fresh_namespace(doc["init"])
    inputs = init_ns[f"{prefix}_init"](batch_size=2, device="cpu")
    ref_ns = _exec_in_fresh_namespace(doc["reference"])
    ref_fn = ref_ns[f"{prefix}_reference"]

    hidden = torch.ones(2, hidden_size, dtype=inputs["input"].dtype)
    residual = torch.full_like(hidden, 2.0)
    weight = torch.ones(hidden_size, dtype=inputs["weight"].dtype)
    result = ref_fn(hidden, residual, weight)

    assert isinstance(result, tuple)
    assert len(result) == len(doc["outputs"])
    output, new_residual = result
    for value, key in ((output, "output"), (new_residual, "residual")):
        assert list(value.shape) == [2, hidden_size]
        assert str(value.dtype).rsplit(".", 1)[-1] == doc["outputs"][key]["dtype"]
    assert torch.equal(new_residual, torch.full_like(hidden, 3.0))
    # rmsnorm of a constant row of 3s is 1 (times weight + weight_bias).
    torch.testing.assert_close(
        output.float(), torch.full((2, hidden_size), 1.0 + weight_bias)
    )
    # The reference is functional: it must not touch the caller's buffers.
    assert torch.equal(residual, torch.full_like(hidden, 2.0))
