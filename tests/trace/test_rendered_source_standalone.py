"""Verify dumped ``init``/``reference`` source strings are runnable standalone.

Dump-time verification calls the live module functions, whose helpers resolve
via module globals — so a rendered string that misses an inlined dependency
(e.g. ``_trtllm_kv_from_cache``) still "works" at dump time and only breaks
for downstream consumers that ``exec()`` the JSON. These tests exec the
rendered strings in a fresh namespace and invoke them once on CPU.
"""

import json
from pathlib import Path

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
