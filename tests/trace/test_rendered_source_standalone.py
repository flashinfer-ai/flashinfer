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

FI_TRACE_OUT = Path(__file__).parent / "fi_trace_out"
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
    max_context_len=256,
    device="cpu",
)


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
    assert out.shape == (rows, _PMQA_INIT_KWARGS["max_context_len"])
    # dtype comes from the reference's own default; it must match what the
    # artifact's output schema advertises.
    assert str(out.dtype).rsplit(".", 1)[-1] == doc["outputs"]["logits"]["dtype"]
    assert torch.isfinite(out.float()[out.float() > float("-inf")]).all()
