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
"""Tests for paged FP8/FP4 MQA scoring and selective TopK.

Reference implementations are adapted from TRT-LLM test scripts.
"""

import inspect
from types import SimpleNamespace
import pytest
import torch
import flashinfer
from flashinfer.fi_trace import fi_trace
from flashinfer.experimental.attn_scores import selective_logits as _backend
from tests.trace.test_fi_trace import _check_defn

pytestmark = pytest.mark.filterwarnings(
    "ignore:.*FP8PagedMQATopKWrapper.*:flashinfer.api_logging.ExperimentalWarning"
)


class _CpuCapacityPlan:
    """CPU stand-in for the CUDA capacity plan, holding only what tracing reads."""

    def __init__(self, *, top_k, math_mode, **_capacities):
        self.top_k, self.math_mode = top_k, math_mode
        self._step_prepared = False


def _cpu_selective_wrapper(monkeypatch, **kwargs):
    """Construct a selective wrapper whose capacity plan needs no CUDA device."""
    from flashinfer.experimental.attn_scores import selective_capacity

    monkeypatch.setattr(
        selective_capacity, "_CapacitySelectiveTopKPlan", _CpuCapacityPlan
    )
    return flashinfer.FP8PagedMQATopKWrapper(
        strategy="selective",
        max_context_length=64,
        max_chunk_size=4,
        max_batch_size=1,
        num_pages=1,
        device="cuda",
        **kwargs,
    )


@pytest.mark.parametrize("math_mode", ["relu_fast", "relu_strict"])
def test_fp8_paged_mqa_topk_bound_trace(math_mode, monkeypatch):
    """Selective traces combine bound capacities with per-step device prefixes."""
    from flashinfer.fi_trace import fi_trace

    wrapper = _cpu_selective_wrapper(
        monkeypatch, math_mode=math_mode, top_k=32, page_size=64
    )
    q = torch.empty(4, 8, 128, dtype=torch.float8_e4m3fn)
    kv_fused = torch.empty(1, 64, 1, 132, dtype=torch.uint8)
    weights = torch.empty(4, 8, dtype=torch.float32)
    block_table = torch.zeros(1, 1, dtype=torch.int32)
    wrapper._capacity_plan._step_prepared = True
    wrapper._capacity_plan._supplied_batch = 1
    wrapper._capacity_plan._state = SimpleNamespace(
        block_table=block_table,
        qo_indptr=torch.tensor([0, 4], dtype=torch.int32),
        kv_indptr=torch.tensor([0, 64], dtype=torch.int32),
        capacity=SimpleNamespace(
            max_context_length=64, max_chunk_size=4, max_batch_size=1
        ),
    )

    definition = fi_trace(
        wrapper.run,
        q=q,
        weights=weights,
        kv_fused=kv_fused,
    )
    _check_defn(
        definition,
        "paged_mqa_topk",
        "FP8PagedMQATopKWrapper.run",
    )
    assert definition["tags"][0] == (
        "fi_api:flashinfer.attn_scores.selective_logits.FP8PagedMQATopKWrapper.run"
    )
    assert definition["axes"]["top_k"]["value"] == 32
    assert definition["outputs"]["indices"] == {
        "shape": ["total_q", "top_k"],
        "dtype": "int32",
        "description": "Request-local exact causal TopK indices; inactive rows and short-row suffixes are -1.",
    }
    assert definition["inputs"]["qo_indptr"]["dtype"] == "int32"
    assert definition["inputs"]["kv_indptr"]["dtype"] == "int32"
    assert definition["inputs"]["math_mode"]["dtype"] == "string"


def test_fp8_paged_mqa_full_positional_trace():
    from types import SimpleNamespace

    wrapper = object.__new__(_backend.FP8PagedMQATopKWrapper)
    wrapper._strategy = "full"
    wrapper._capacity_plan = None
    wrapper._math_mode = wrapper._math_mode_at_construction = "relu_strict"
    wrapper._compression_ratio = wrapper._compression_ratio_at_construction = 1
    wrapper._full_top_k = 512
    wrapper._full_batches = 1
    wrapper._full_step_is_prepared = True
    q = torch.empty(2, 8, 128, dtype=torch.float8_e4m3fn)
    kv = torch.empty(1, 64, 1, 132, dtype=torch.uint8)
    weights = torch.empty(2, 8)
    table = torch.zeros(1, 1, dtype=torch.int32)
    wrapper._full_state = SimpleNamespace(
        qo_indptr=torch.tensor([0, 2], dtype=torch.int32),
        kv_indptr=torch.tensor([0, 64], dtype=torch.int32),
        block_table=table,
    )
    bound = inspect.signature(wrapper.run).bind(q, weights, kv)
    positional = fi_trace(wrapper.run, **bound.arguments)
    named = fi_trace(wrapper.run, q=q, weights=weights, kv_fused=kv)
    assert positional == named
    assert positional["axes"]["num_heads"]["value"] == 8
    assert positional["axes"]["page_size"]["value"] == 64


def test_fp8_paged_mqa_topk_trace_requires_plan(monkeypatch):
    from flashinfer.fi_trace import fi_trace

    wrapper = _cpu_selective_wrapper(monkeypatch, page_size=64)
    with pytest.raises(RuntimeError, match=r"plan\(\) must be called"):
        fi_trace(
            wrapper.run,
            q=torch.empty(1, 8, 128, dtype=torch.float8_e4m3fn),
            weights=torch.empty(1, 8, dtype=torch.float32),
            kv_fused=torch.empty(1, 64, 1, 132, dtype=torch.uint8),
        )


@pytest.mark.parametrize("ratio", [1, 4])
def test_capacity_topk_trace_fail_closed_steps(ratio):
    from flashinfer.experimental.attn_scores.trace import (
        fp8_paged_mqa_selective_topk_trace,
        fp8_paged_mqa_compressed_topk_trace,
    )

    template = (
        fp8_paged_mqa_selective_topk_trace
        if ratio == 1
        else fp8_paged_mqa_compressed_topk_trace
    )
    inputs = template.init(
        total_q=4, batch_size=2, max_kv_len=64, top_k=4, device="cpu"
    )
    arguments = {
        **inputs["constructor"],
        **inputs["plan"],
        **inputs["run"],
    }
    arguments["qo_indptr"] = torch.tensor([0, 1, 2], dtype=torch.int32)
    output = template.reference(**arguments)
    assert template.check(output, output, **arguments)
    assert (output[2:] == -1).all()
    damaged = output.clone()
    damaged[2:] = 0
    assert not template.check(output, damaged, **arguments)
    for queries, tokens in (
        ([1, 1, 2], [0, 64, 128]),
        ([0, 2, 1], [0, 64, 128]),
        ([0, 1, 5], [0, 64, 128]),
        ([0, 1, 2], [0, 65, 129]),
    ):
        invalid = dict(
            arguments,
            qo_indptr=torch.tensor(queries, dtype=torch.int32),
            kv_indptr=torch.tensor(tokens, dtype=torch.int32),
        )
        output = template.reference(**invalid)
        assert (output == -1).all()
        assert template.check(output, output, **invalid)
        assert not template.check(output, torch.zeros_like(output), **invalid)
    arguments["block_table"] = torch.full_like(arguments["block_table"], -1)
    output = template.reference(**arguments)
    assert (output == -1).all()
    assert template.check(output, output, **arguments)


@pytest.mark.parametrize(
    ("top_k", "supported"),
    [(0, False), (1, True), (513, True), (1536, True), (2048, True), (2049, False)],
)
def test_fp8_paged_mqa_topk_trace_top_k_domain(top_k, supported):
    from flashinfer.experimental.attn_scores.selective_logits import (
        _validate_selective_top_k,
    )
    from flashinfer.experimental.attn_scores.trace import (
        fp8_paged_mqa_selective_topk_trace,
    )

    if supported:
        _validate_selective_top_k(top_k)
    else:
        with pytest.raises(ValueError):
            _validate_selective_top_k(top_k)
    top_k_constraints = [
        constraint
        for constraint in fp8_paged_mqa_selective_topk_trace.constraints
        if "top_k" in constraint
    ]
    assert (
        all(eval(constraint, {}, {"top_k": top_k}) for constraint in top_k_constraints)
        == supported
    )


@pytest.mark.parametrize("top_k", [512, 2048])
def test_fp8_paged_mqa_topk_compressed_trace_contract(top_k, monkeypatch):
    wrapper = _cpu_selective_wrapper(
        monkeypatch, compression_ratio=4, top_k=top_k, page_size=64
    )
    kwargs = dict(
        q=torch.empty(4, 8, 128, dtype=torch.float8_e4m3fn),
        weights=torch.empty(4, 8),
        kv_fused=torch.empty(1, 64, 1, 132, dtype=torch.uint8),
    )
    with pytest.raises(RuntimeError, match=r"plan\(\) must be called"):
        fi_trace(wrapper.run, **kwargs)
    wrapper._capacity_plan._step_prepared = True
    wrapper._capacity_plan._supplied_batch = 1
    wrapper._capacity_plan._state = SimpleNamespace(
        qo_indptr=torch.tensor([0, 4], dtype=torch.int32),
        kv_indptr=torch.tensor([0, 4], dtype=torch.int32),
        block_table=torch.zeros(1, 1, dtype=torch.int32),
        capacity=SimpleNamespace(
            max_context_length=4, max_chunk_size=4, max_batch_size=1
        ),
    )
    definition = fi_trace(wrapper.run, **kwargs)
    _check_defn(definition, "paged_mqa_topk", "FP8PagedMQATopKWrapper.run")
    assert definition["axes"]["compression_ratio"]["value"] == 4
    assert definition["inputs"]["compression_ratio"]["dtype"] == "int32"
    assert "compression_ratio == 4" in definition["constraints"]
    assert "top_k <= 2048" in definition["constraints"]
    assert "init" in definition
    wrapper._capacity_plan._step_prepared = False
    with pytest.raises(RuntimeError, match=r"plan\(\) must be called"):
        fi_trace(wrapper.run, **kwargs)


@pytest.mark.parametrize("ratio", [1, 4])
@pytest.mark.parametrize("math_mode", ["relu_strict", "relu_fast"])
def test_selective_topk_nan_reference_and_check(ratio, math_mode):
    """The selective trace ranks NaN below numeric scores in both schemas."""
    from flashinfer.experimental.attn_scores.trace import (
        fp8_paged_mqa_compressed_topk_trace,
        fp8_paged_mqa_selective_topk_trace,
    )

    template = (
        fp8_paged_mqa_compressed_topk_trace
        if ratio == 4
        else fp8_paged_mqa_selective_topk_trace
    )
    initial = template.init(
        total_q=3,
        batch_size=1,
        page_size=128,
        max_kv_len=64,
        top_k=4,
        device="cpu",
    )
    inputs = {**initial["plan"], **initial["run"]}
    inputs.update(top_k=4, math_mode=math_mode, compression_ratio=ratio)
    q, weights, fused = (inputs[key] for key in ("q", "weights", "kv_fused"))
    q.copy_(torch.ones(q.shape).to(q.dtype))
    weights.fill_(1)
    weights[-1].fill_(float("nan"))
    flat = fused.flatten(1)
    keys = flat[:, : 128 * 128].view(torch.float8_e4m3fn)
    keys.copy_(torch.ones(keys.shape).to(keys.dtype))
    scales = flat[:, 128 * 128 :].view(torch.float32)
    scales.fill_(float("nan"))
    scales[:, :3] = torch.tensor([1.0, 2.0, float("inf")])

    reference = template.reference(**inputs)
    assert template.check(reference, reference, **inputs)
    assert {0, 1, 2}.issubset(reference[0].tolist())
    corrupted = reference.clone()
    outside = next(index for index in range(3, 15) if index not in reference[0])
    corrupted[0, (reference[0] == 2).nonzero().item()] = outside
    assert not template.check(reference, corrupted, **inputs)
