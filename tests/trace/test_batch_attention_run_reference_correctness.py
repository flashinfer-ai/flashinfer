"""Reference correctness test for the batch_attention_run trace API."""

import json

import torch

import flashinfer
from flashinfer.trace.templates.attention import batch_attention_run_trace
from flashinfer.utils import get_compute_capability
import pytest

from tests.test_helpers.paged_kv import make_paged_kv_cache_pair
from tests.trace.reference_utils import (
    _check,
)


@pytest.mark.parametrize(
    "query_tokens,heads,kv_heads,dim,pages,page_size,stride_mode",
    [
        (1, 8, 2, 64, 1, 16, None),
        (1, 4, 1, 128, 2, 8, None),
        (129, 8, 2, 64, 10, 16, "head"),
    ],
)
def test_batch_attention_run_reference_correctness(
    query_tokens,
    heads,
    kv_heads,
    dim,
    pages,
    page_size,
    stride_mode,
    tmp_path,
    monkeypatch,
):
    """Check the actual persistent API, including one strided trace-emission case."""
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    inputs = batch_attention_run_trace.init(
        device="cuda",
        num_qo_tokens=query_tokens,
        num_qo_heads=heads,
        num_kv_heads=kv_heads,
        head_dim=dim,
        num_pages=pages,
        page_size=page_size,
        seed=20260917,
    )
    plan, run = inputs["plan"], inputs["run"]
    q, kv = run["q"], run["kv_cache"]
    if stride_mode is not None:
        kv = make_paged_kv_cache_pair(*kv, "NHD", stride_mode, 8)
        definition = flashinfer.BatchAttention.run.fi_trace(
            q=q, kv_cache=kv, save_dir=tmp_path
        )
        assert any(
            tag.startswith("fi_api:") and tag.endswith("BatchAttention.run")
            for tag in definition["tags"]
        )
        assert definition["axes"]["head_dim"]["value"] == dim
        assert definition["axes"]["num_qo_heads"]["value"] == heads
        assert definition["outputs"]["lse"]["dtype"] == "float32"
        assert "reference" in definition
        assert (
            json.loads((tmp_path / (definition["name"] + ".json")).read_text())
            == definition
        )
    expected = batch_attention_run_trace.reference(q, kv)
    assert all(torch.isfinite(t).all() for t in expected)
    qo_indptr = torch.tensor([0, query_tokens], dtype=torch.int32, device=q.device)
    kv_lengths = (
        plan["kv_indptr"][1:] - plan["kv_indptr"][:-1] - 1
    ) * page_size + plan["kv_last_page_len"]
    wrapper = flashinfer.BatchAttention(kv_layout="NHD")
    wrapper.plan(
        qo_indptr,
        plan["kv_indptr"],
        plan["kv_indices"],
        kv_lengths,
        heads,
        kv_heads,
        dim,
        dim,
        page_size,
        causal=False,
        q_data_type=plan["q_data_type"],
        kv_data_type=plan["kv_data_type"],
    )
    out = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    lse = torch.empty(q.shape[:2], device=q.device, dtype=torch.float32)
    # This fixture exceeds the SM120/121 cooperative-grid limit.
    # Avoid leaving a CUDA launch error for the next test.
    if dim == 128 and get_compute_capability(q.device)[0] == 12:
        pytest.xfail("SM120/121 persistent BatchAttention cooperative-launch limit")
    actual = wrapper.run(q, kv, out=out, lse=lse)
    _check(batch_attention_run_trace, expected, actual, atol=1e-2, rtol=1e-2)
    for result, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(
            result.float(), reference.float(), atol=1e-2, rtol=1e-2
        )
    torch.cuda.synchronize()
