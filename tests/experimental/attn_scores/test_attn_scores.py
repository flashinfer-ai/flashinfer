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

from __future__ import annotations

import gc
import importlib.util
import itertools
import math
import weakref
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import contextlib
import os

import torch

from flashinfer import FP8PagedMQATopKWrapper
from flashinfer.experimental.attn_scores import selective_logits as sl
from flashinfer.experimental.attn_scores.selective_logits import (
    _fp8_mqa_selective_logits,
    _SelectiveLogitsMode,
)
from flashinfer.utils import is_sm100a_supported

from tests.experimental.attn_scores._selective_engine_case import (
    SelectiveEngineCase,
    _capture_cuda_graph,
)


pytestmark = pytest.mark.filterwarnings(
    "ignore:.*FP8PagedMQATopKWrapper.*:flashinfer.api_logging.ExperimentalWarning"
)

# Selective FP8 MQA: inputs, independent oracles, kernels and wrapper lifecycle


@pytest.fixture
def ieee_matmul():
    # This backend proxy exposes a setter, not a deletable instance attribute.
    # Restore through the setter so pytest cleanup also works on PyTorch 2.13.
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


@pytest.fixture
def isolated_capture_gc():
    """Keep forced-collection oracles independent of predecessor garbage."""
    enabled = gc.isenabled()
    gc.collect()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()


_requires_selective_sm100 = pytest.mark.skipif(
    not torch.cuda.is_available()
    or importlib.util.find_spec("cutlass") is None
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="selective FP8 logits requires SM100 or SM103",
)


@contextmanager
def _reject_library_graph_calls(monkeypatch, *, allow_repair_body=False):
    """Only a repair conditional body may capture inside the caller's graph."""

    def reject(*args, **kwargs):
        raise AssertionError(
            "the wrapper must not capture, instantiate or launch graphs"
        )

    with monkeypatch.context() as patch:
        graph_type = torch.cuda.CUDAGraph
        for name in (
            "__init__",
            "capture_begin",
            "capture_end",
            "instantiate",
            "replay",
        ):
            if hasattr(graph_type, name):
                patch.setattr(graph_type, name, reject)
        patch.setattr(torch.cuda, "graph", reject)
        runtime = sl._capture_runtime()
        for name in (
            "cudaStreamBeginCapture",
            "cudaStreamBeginCaptureToGraph",
            "cudaStreamEndCapture",
            "cudaGraphInstantiate",
            "cudaGraphInstantiateWithFlags",
            "cudaGraphLaunch",
        ):
            if hasattr(runtime, name):
                patch.setattr(runtime, name, reject)
        from flashinfer.experimental.attn_scores.jit import selective_repair_if

        get_module = selective_repair_if.get_selective_repair_if_module
        body_calls = []

        class GuardedRepairModule:
            def __init__(self, module):
                self.module = module

            def __getattr__(self, name):
                operation = getattr(self.module, name)
                if name not in ("begin_if", "begin_body", "end_body", "finish_if"):
                    return operation

                def guarded(*args):
                    assert allow_repair_body
                    assert torch.cuda.is_current_stream_capturing()
                    body_calls.append(name)
                    return operation(*args)

                return guarded

        patch.setattr(
            selective_repair_if,
            "get_selective_repair_if_module",
            lambda: GuardedRepairModule(get_module()),
        )
        yield body_calls


@_requires_selective_sm100
@pytest.mark.parametrize("strategy", ["full", "selective"])
@pytest.mark.parametrize("budget", [0, True, 1.5, "512"])
def test_capacity_workspace_budget_validation(strategy, budget):
    with pytest.raises(ValueError, match="max_workspace_size"):
        FP8PagedMQATopKWrapper(
            strategy=strategy,
            max_context_length=4096,
            max_chunk_size=8,
            max_batch_size=1,
            max_workspace_size=budget,
            num_pages=32,
            device="cuda",
        )


@_requires_selective_sm100
def test_full_workspace_budget_exact_and_default():
    torch.backends.cuda.matmul.allow_tf32 = False
    inputs = _request_inputs([5], [1031], heads=64)
    q, _, _, weights, qo, kv = inputs
    pool, original = _paged_inputs(inputs)
    table = torch.zeros((1, 32), dtype=torch.int32, device=q.device)
    table[:, : original.shape[1]] = original
    options = dict(
        strategy="full",
        num_heads=64,
        top_k=512,
        max_context_length=4096,
        max_chunk_size=8,
        max_batch_size=1,
        kv_layout=pool,
    )
    with pytest.raises(ValueError, match="at least 65536 bytes"):
        FP8PagedMQATopKWrapper(**options, max_workspace_size=65535)
    expected = _oracle_rows(inputs)
    arena_sizes = []
    for budget in (None, 512 * 1024 * 1024, 64 * 1024, 128 * 1024):
        wrapper = FP8PagedMQATopKWrapper(**options, max_workspace_size=budget)
        wrapper.plan(qo, kv, table, num_query_rows=q.shape[0])
        wrapper.prepare_pool(pool)
        _check_selection(wrapper.run(q, weights, pool), expected, 512)
        scores = wrapper._full_capacity_plan._scores
        arena_bytes = scores.numel() * scores.element_size()
        assert arena_bytes <= (512 * 1024 * 1024 if budget is None else budget)
        arena_sizes.append(arena_bytes)
        del scores
        del wrapper
    assert arena_sizes[0] == arena_sizes[1]


@_requires_selective_sm100
@pytest.mark.parametrize("strategy", ["full", "selective"])
@pytest.mark.parametrize("q_lengths", [[4, 4], [3, 5], [0, 3, 2]])
def test_capacity_host_length_hints_contract(strategy, q_lengths, monkeypatch):
    torch.backends.cuda.matmul.allow_tf32 = False
    lengths = [1025 + 128 * i for i in range(len(q_lengths))]
    inputs = _request_inputs(q_lengths, lengths, heads=64)
    q, _, _, weights, qo, kv = inputs
    pool, original = _paged_inputs(inputs)
    table = torch.zeros((len(q_lengths), 32), dtype=torch.int32, device=q.device)
    table[:, : original.shape[1]] = original
    wrapper = FP8PagedMQATopKWrapper(
        strategy=strategy,
        num_heads=64,
        top_k=512,
        max_context_length=4096,
        max_chunk_size=16,
        max_batch_size=len(q_lengths),
        max_workspace_size=128 * 1024,
        kv_layout=pool,
    )
    kwargs = dict(num_query_rows=q.shape[0])
    wrapper.plan(qo, kv, table, **kwargs)
    wrapper.prepare_pool(pool)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = wrapper.run(q, weights, pool)
    # Publish different lengths and physical page maps after the graph was
    # captured. Hinted eager planning must refresh its capacity metadata too.
    lengths = [length - 257 for length in lengths]
    kv.copy_(
        torch.tensor(
            [0, *itertools.accumulate(lengths)], device=q.device, dtype=kv.dtype
        )
    )
    for request in range(len(lengths)):
        pages = (lengths[request] + 127) // 128
        table[request, :pages].copy_(original[request, :pages].flip(0))
    hints = (qo.cpu(), torch.tensor(lengths, dtype=torch.int32))
    expected = _paged_step_oracle(q, weights, pool, table, hints[0], lengths)
    pointers = tuple(t.data_ptr() for t in wrapper.input_buffers().values())
    for bounds in (hints[1], hints[1] + 100):
        wrapper.plan(qo, kv, table, host_length_hints=(hints[0], bounds), **kwargs)
        _check_selection(wrapper.run(q, weights, pool), expected, 512)
        captured.fill_(-2)
        graph.replay()
        _check_selection(captured, expected, 512)
        assert wrapper.error.item() == 0
    assert pointers == tuple(t.data_ptr() for t in wrapper.input_buffers().values())
    if strategy == "full":
        wrapper.plan(qo, kv, table, host_length_hints=hints, **kwargs)
        wrapper.run(q, weights, pool)
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated()
        wrapper.plan(qo, kv, table, host_length_hints=hints, **kwargs)
        wrapper.run(q, weights, pool)
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == allocated
    for bad in ((hints[0], hints[1] - 1), (hints[0] + 1, hints[1])):
        # The second prefix must stay a valid host prefix, but differ on device.
        if bad[0][0] != 0:
            changed = hints[0].clone()
            changed[-1] -= 1
            bad = (changed, hints[1])
        wrapper.plan(qo, kv, table, host_length_hints=bad, **kwargs)
        assert (wrapper.run(q, weights, pool) == -1).all()
        assert wrapper.error.item() & 2
    wrapper.plan(qo, kv, table, **kwargs)
    _check_selection(wrapper.run(q, weights, pool), expected, 512)
    # Simulate capture for the rejection check, keeping the real stream valid.
    with monkeypatch.context() as patch:
        patch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
        # Patch the class: restoring an instance's bound method would leave
        # state.stage_step -> state, a test-owned cycle that delays private
        # library retirement until an unrelated capture's cyclic collection.
        patch.setattr(
            type(
                wrapper._full_state
                if strategy == "full"
                else wrapper._capacity_plan._state
            ),
            "stage_step",
            lambda *a, **kw: pytest.fail("CUDA staging before rejection"),
        )
        with pytest.raises(RuntimeError, match="host_length_hints"):
            wrapper.plan(qo, kv, table, host_length_hints=hints, **kwargs)
    state = wrapper._full_state if strategy == "full" else wrapper._capacity_plan._state
    owner, state_owner = weakref.ref(wrapper), weakref.ref(state)
    del graph, state, wrapper
    assert owner() is None
    assert state_owner() is None


def _paged_step_oracle(q, weights, pool, table, prefix, lengths):
    """Independent FP32 scores using the supplied physical page map."""
    expected = []
    page_size, head_dim = pool.shape[1], q.shape[-1]
    for request, length in enumerate(lengths):
        pages = pool.index_select(
            0, table[request, : (length + page_size - 1) // page_size].long()
        ).flatten(1)
        keys = (
            pages[:, : page_size * head_dim]
            .contiguous()
            .view(q.dtype)
            .reshape(-1, head_dim)
        )
        scales = (
            pages[:, page_size * head_dim :].contiguous().view(torch.float32).flatten()
        )
        start, stop = int(prefix[request]), int(prefix[request + 1])
        for row in range(start, stop):
            end = length - (stop - start) + row - start + 1
            dots = q[row].float() @ keys[:end].float().T
            expected.append((dots.relu() * weights[row, :, None]).sum(0) * scales[:end])
    return expected


@_requires_selective_sm100
def test_full_host_hint_pitch_bucket_edge():
    torch.backends.cuda.matmul.allow_tf32 = False
    inputs = _request_inputs([3], [9001], heads=64)
    q, _, _, weights, qo, kv = inputs
    pool, original = _paged_inputs(inputs)
    table = torch.zeros((1, 256), dtype=torch.int32, device=q.device)
    table[:, : original.shape[1]] = original
    wrapper = FP8PagedMQATopKWrapper(
        strategy="full",
        num_heads=64,
        top_k=512,
        max_context_length=32768,
        max_chunk_size=8,
        max_batch_size=1,
        max_workspace_size=512 * 1024,
        kv_layout=pool,
    )
    wrapper.plan(qo, kv, table, num_query_rows=3)
    wrapper.prepare_pool(pool)
    for length, bound, pitch in ((8000, 8192, 8192), (8500, 9000, 16384)):
        kv[1] = length
        wrapper.plan(
            qo,
            kv,
            table,
            num_query_rows=3,
            host_length_hints=(qo.cpu(), torch.tensor([bound], dtype=torch.int32)),
        )
        assert wrapper._full_capacity_plan._host_hints.pitch == pitch
        selected = wrapper.run(q, weights, pool)
        expected = _paged_step_oracle(q, weights, pool, table, qo.cpu(), [length])
        _check_selection(selected, expected, 512)
        assert wrapper.error.item() == 0


@_requires_selective_sm100
@pytest.mark.parametrize("strategy", ["full", "selective"])
@pytest.mark.parametrize("next_n", [1, 2, 4])
def test_capacity_uniform_plan_capture_and_invalid_step(strategy, next_n):
    torch.backends.cuda.matmul.allow_tf32 = False
    inputs = _request_inputs([next_n, next_n], [1025, 2049], heads=64)
    q, keys, scales, weights, qo, kv = inputs
    pool, original = _paged_inputs(inputs)
    table = torch.zeros((2, 32), device=q.device, dtype=torch.int32)
    table[:, : original.shape[1]] = original
    wrapper = FP8PagedMQATopKWrapper(
        strategy=strategy,
        num_heads=64,
        top_k=512,
        max_context_length=4096,
        max_chunk_size=32,
        max_batch_size=4,
        kv_layout=pool,
    )
    wrapper.plan(qo, kv, table, num_query_rows=q.shape[0], next_n=next_n)
    wrapper.prepare_pool(pool)
    expected = []
    offset = 0
    for request, length in enumerate((1025, 2049)):
        for local in range(next_n):
            row = request * next_n + local
            end = length - next_n + local + 1
            dots = q[row].float() @ keys[offset : offset + end].float().T
            expected.append(
                (dots.relu() * weights[row, :, None]).sum(0)
                * scales[offset : offset + end]
            )
        offset += length
    _check_selection(wrapper.run(q, weights, pool), expected, 512)
    pointers = tuple(t.data_ptr() for t in wrapper.input_buffers().values())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.plan(qo, kv, table, num_query_rows=q.shape[0], next_n=next_n)
        selected = wrapper.run(q, weights, pool)
    graph.replay()
    _check_selection(selected, expected, 512)
    assert wrapper.error.item() == 0
    # Lengths and physical page maps change while all captured addresses stay fixed.
    kv.copy_(torch.tensor([0, 800, 2300], device=q.device, dtype=kv.dtype))
    table.copy_(table.flip(0))
    graph.replay()
    changed = []
    for request, length in enumerate((800, 1500)):
        pages = pool.index_select(
            0, table[request, : (length + 127) // 128].long()
        ).flatten(1)
        changed_keys = pages[:, : 128 * 128].contiguous().view(q.dtype).reshape(-1, 128)
        changed_scales = (
            pages[:, 128 * 128 :].contiguous().view(torch.float32).reshape(-1)
        )
        for local in range(next_n):
            row = request * next_n + local
            end = length - next_n + local + 1
            dots = q[row].float() @ changed_keys[:end].float().T
            changed.append(
                (dots.relu() * weights[row, :, None]).sum(0) * changed_scales[:end]
            )
    _check_selection(selected, changed, 512)
    assert wrapper.error.item() == 0
    # Transfer one query between requests; storage extent stays unchanged.
    qo[1] += 1
    graph.replay()
    assert wrapper.error.item() & 2
    assert (selected == -1).all()
    qo[1] -= 1
    graph.replay()
    assert wrapper.error.item() == 0
    _check_selection(selected, changed, 512)
    assert pointers == tuple(t.data_ptr() for t in wrapper.input_buffers().values())
    for invalid in (0, 3, True, 1.0):
        with pytest.raises(ValueError, match="next_n"):
            wrapper.plan(qo, kv, table, next_n=invalid)
    del graph
    owner = weakref.ref(wrapper)
    del wrapper
    assert owner() is None


@_requires_selective_sm100
@pytest.mark.parametrize("next_n", [1, 2, 4])
def test_capacity_uniform_full_odd_and_empty_storage(next_n):
    inputs = _request_inputs([next_n, next_n], [1025, 2049], heads=64)
    q, _, _, weights, qo, kv = inputs
    pool, original = _paged_inputs(inputs)
    table = torch.zeros((2, 32), dtype=torch.int32, device=q.device)
    table[:, : original.shape[1]] = original
    q = torch.cat((q, q[:1]))
    weights = torch.cat((weights, weights[:1]))
    wrapper = FP8PagedMQATopKWrapper(
        strategy="full",
        num_heads=64,
        top_k=512,
        max_context_length=4096,
        max_chunk_size=32,
        max_batch_size=4,
        kv_layout=pool,
    )
    wrapper.plan(qo, kv, table, num_query_rows=q.shape[0], next_n=next_n)
    wrapper.prepare_pool(pool)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.plan(qo, kv, table, num_query_rows=q.shape[0], next_n=next_n)
        selected = wrapper.run(q, weights, pool)
    graph.replay()
    assert wrapper.error.item() == 0
    _check_selection(selected[: 2 * next_n], _oracle_rows(inputs), 512)
    assert (selected[-1] == -1).all()
    qo[1] += 1
    graph.replay()
    assert wrapper.error.item() & 2
    assert (selected == -1).all()
    prefix = torch.zeros(1, device=q.device, dtype=torch.int32)
    for mode in (None, next_n):
        wrapper.plan(prefix, prefix, table[:0], num_query_rows=0, next_n=mode)
        wrapper.prepare_pool(pool)
        empty_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(empty_graph):
            wrapper.plan(prefix, prefix, table[:0], num_query_rows=0, next_n=mode)
            empty = wrapper.run(q[:0], weights[:0], pool)
        empty_graph.replay()
        assert wrapper.error.item() == 0
        assert empty.shape == (0, 512)


@_requires_selective_sm100
def test_full_capacity_stripes_captured_plan_refresh(monkeypatch, ieee_matmul):
    inputs = _request_inputs([3, 0, 2], [1031, 0, 2051], heads=64)
    q, _, _, weights, qo, kv = inputs
    pool, original_table = _paged_inputs(inputs)
    table = torch.zeros((3, 32), dtype=torch.int32, device=q.device)
    table[:, : original_table.shape[1]] = original_table
    wrapper = FP8PagedMQATopKWrapper(
        strategy="full",
        num_heads=64,
        top_k=512,
        max_context_length=4096,
        max_chunk_size=7,
        max_batch_size=3,
        max_workspace_size=64 * 1024,
        kv_layout=pool,
    )
    wrapper.prepare_pool(pool)
    wrapper.plan(qo, kv, table, num_query_rows=5)
    wrapper.prepare_pool(pool)
    plan = wrapper._full_capacity_plan
    import importlib
    from flashinfer.attn_scores import attn_scores as scorer

    def reject_layer_dispatch(*args, **kwargs):
        raise AssertionError("prepared full run must use bound launches")

    monkeypatch.setattr(scorer, "fp8_paged_mqa_logits", reject_layer_dispatch)
    monkeypatch.setattr(
        importlib.import_module("flashinfer.topk_varlen.topk_varlen"),
        "top_k_varlen",
        reject_layer_dispatch,
    )
    pointers = [
        t.data_ptr()
        for t in (
            plan._scores,
            plan._q,
            plan._weights,
            plan._indices,
            plan._tables,
            plan._lengths,
            plan._hints,
            plan._gvr_workspace,
        )
    ]

    def check(selected):
        assert wrapper.error.item() == 0
        qs, ks = qo.tolist(), kv.tolist()
        expected = []
        for request in range(3):
            count, length = qs[request + 1] - qs[request], ks[request + 1] - ks[request]
            pages = pool.index_select(
                0, table[request, : (length + 127) // 128].long()
            ).flatten(1)
            keys = (
                pages[:, : 128 * 128]
                .contiguous()
                .view(q.dtype)
                .reshape(-1, 128)[:length]
            )
            scales = (
                pages[:, 128 * 128 :]
                .contiguous()
                .view(torch.float32)
                .reshape(-1)[:length]
            )
            for local in range(count):
                row = qs[request] + local
                end = max(0, length - count + local + 1)
                dots = q[row].float() @ keys[:end].float().T
                expected.append(
                    (dots.relu() * weights[row, :, None]).sum(0) * scales[:end]
                )
        expected.extend([torch.empty(0, device=q.device)] * (5 - len(expected)))
        _check_selection(selected, expected, 512)
        assert wrapper.error.item() == 0
        assert pointers == [
            t.data_ptr()
            for t in (
                plan._scores,
                plan._q,
                plan._weights,
                plan._indices,
                plan._tables,
                plan._lengths,
                plan._hints,
                plan._gvr_workspace,
            )
        ]

    check(wrapper.run(q, weights, pool))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.plan(qo, kv, table, num_query_rows=5)
        selected = wrapper.run(q, weights, pool)
    for queries, lengths in (
        ([1, 2, 2], [511, 2, 2]),
        ([1, 0, 2], [17, 0, 2]),
        ([0, 0, 0], [0, 0, 0]),
    ):
        qo.copy_(
            torch.tensor(
                [0, *torch.tensor(queries).cumsum(0).tolist()],
                device=q.device,
                dtype=qo.dtype,
            )
        )
        kv.copy_(
            torch.tensor(
                [0, *torch.tensor(lengths).cumsum(0).tolist()],
                device=q.device,
                dtype=kv.dtype,
            )
        )
        table.copy_(table.roll(1, dims=1))
        graph.replay()
        torch.cuda.synchronize()
        check(selected)


@_requires_selective_sm100
@pytest.mark.parametrize("strategy", ["full", "selective"])
@pytest.mark.parametrize("tied", [False, True])
def test_wrapper_leaves_all_graph_ownership_to_caller(
    monkeypatch, strategy, tied, isolated_capture_gc
):
    """Preparation/eager/captured execution uses no library-created graphs."""
    inputs = _request_inputs([2, 1], [9001, 257])
    q, _, _, weights, qo, kv = inputs
    if tied:
        weights.zero_()
    pool, table = _paged_inputs(inputs)
    expected = _oracle_rows(inputs)
    with _reject_library_graph_calls(monkeypatch):
        wrapper = FP8PagedMQATopKWrapper(
            strategy=strategy,
            max_context_length=9001,
            max_chunk_size=3,
            max_batch_size=2,
            kv_layout=pool,
        )
        wrapper.plan(qo, kv, table)
        wrapper.prepare_pool(pool)
        eager = wrapper.run(q, weights, pool).clone()
    torch.cuda.synchronize()
    _check_selection(eager, expected, 512)
    graph = torch.cuda.CUDAGraph()
    # Rebind the prepared native module through the same guard for capture.
    with (
        torch.cuda.graph(graph),
        _reject_library_graph_calls(
            monkeypatch, allow_repair_body=strategy == "selective"
        ) as body_calls,
    ):
        if strategy == "selective":
            from flashinfer.experimental.attn_scores.jit.selective_repair_if import (
                get_selective_repair_if_module,
            )

            wrapper._capacity_plan._engine._repair_if_module = (
                get_selective_repair_if_module()
            )
            wrapper.plan(qo, kv, table)
        captured = wrapper.run(q, weights, pool)
        assert body_calls == (
            ["begin_if", "begin_body", "end_body", "finish_if"]
            if strategy == "selective"
            else []
        )
    graph.replay()
    torch.cuda.synchronize()
    _check_selection(captured, expected, 512)
    assert wrapper.error.item() == wrapper.repair_error.item() == 0
    if strategy == "full":
        plan = wrapper._full_capacity_plan
        callables = [
            plan._metadata,
            *plan._uniform_metadata.values(),
            plan._pack,
            plan._scatter,
        ]
    else:
        plan = wrapper._capacity_plan
        callables = [plan._state._compiled, plan._state._sample_compiled]
    modules = [
        weakref.ref(compiled.jit_module)
        for compiled in callables
        if compiled is not None
    ]
    del callables, plan
    # Caller-owned graphs must be retired outside subsequent captures.
    del graph
    owner = weakref.ref(wrapper)
    del wrapper
    assert owner() is None
    assert all(
        reference() is None
        or (reference().is_unloaded() and not reference().cuda_library)
        for reference in modules
    )
    from cutlass.cutlass_dsl.cuda_jit_executor import CudaDialectJitModule

    unload = CudaDialectJitModule.unload
    calls = []

    def observe_unload(module):
        if module.cuda_library:
            calls.append(torch.cuda.is_current_stream_capturing())
        return unload(module)

    monkeypatch.setattr(CudaDialectJitModule, "unload", observe_unload)
    output = torch.zeros(1, device=q.device)
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    # Unlike torch.cuda.graph's context manager, capture_begin does not collect
    # before capture: CuTe's already-unloaded self-cycles are collected inside it.
    with torch.cuda.stream(stream):
        graph.capture_begin()
        gc.collect()
        output.fill_(1)
        graph.capture_end()
    graph.replay()
    torch.cuda.synchronize()
    assert output.item() == 1
    assert calls == []


@pytest.mark.parametrize(
    "capability", [(9, 0), (10, 0), (10, 3), (10, 7), (12, 0), (12, 1)]
)
def test_selective_constructor_device_support(monkeypatch, capability):
    """Unsupported devices reject before device-scope entry or workspace allocation."""
    from flashinfer.experimental.attn_scores import selective_capacity

    monkeypatch.setattr(sl, "get_compute_capability", lambda device: capability)
    device = torch.device("cuda:0")
    if capability in ((10, 0), (10, 3)):
        sl._validate_selective_device(device)
        return

    def unexpected_cuda_work(*args, **kwargs):
        raise AssertionError(
            "unsupported construction must not allocate or enter device scope"
        )

    monkeypatch.setattr(torch.cuda, "device", unexpected_cuda_work)
    monkeypatch.setattr(selective_capacity, "_CapacityStepInputs", unexpected_cuda_work)
    with pytest.raises(ValueError, match="requires SM100 or SM103"):
        FP8PagedMQATopKWrapper(
            strategy="selective",
            max_context_length=1024,
            max_chunk_size=1,
            max_batch_size=1,
            num_pages=8,
            device=device,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("layout_source", ["explicit", "exemplar"])
def test_selective_constructor_rejects_device_before_native_jit(
    monkeypatch, layout_source
):
    """Unsupported CUDA devices reject before native JIT or workspace allocation."""
    if torch.cuda.get_device_capability() in ((10, 0), (10, 3)):
        pytest.skip("requires an unsupported selective device")

    from flashinfer.experimental.attn_scores import selective_capacity
    from flashinfer.experimental.attn_scores.jit import (
        selective_repair_if,
        selective_sample_ladder,
    )

    device = torch.device("cuda", torch.cuda.current_device())
    layout = (
        {"num_pages": 8, "device": device}
        if layout_source == "explicit"
        else {
            "kv_layout": torch.empty((8, 128, 1, 132), dtype=torch.uint8, device=device)
        }
    )

    def unexpected_preparation(*args, **kwargs):
        raise AssertionError("unsupported device reached native JIT or allocation")

    monkeypatch.setattr(
        selective_sample_ladder,
        "gen_selective_sample_ladder_module",
        unexpected_preparation,
    )
    monkeypatch.setattr(
        selective_repair_if, "gen_selective_repair_if_module", unexpected_preparation
    )
    monkeypatch.setattr(
        selective_capacity, "_CapacityStepInputs", unexpected_preparation
    )
    with pytest.raises(ValueError, match="requires SM100 or SM103"):
        FP8PagedMQATopKWrapper(
            strategy="selective",
            max_context_length=1024,
            max_chunk_size=1,
            max_batch_size=1,
            **layout,
        )


@pytest.mark.parametrize(
    "device, expected", [("cuda:0", 0), ("cuda:1", 1), ("cuda", 2)]
)
def test_selective_compile_device_ordinal(monkeypatch, device, expected):
    from flashinfer.experimental.attn_scores.selective_logits import _device_ordinal

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    assert _device_ordinal(torch.device(device)) == expected


def _compressed_wrapper_fixture(q_lengths, token_lengths, heads, page_size):
    physical_lengths = [length // 4 for length in token_lengths]
    q, kv, scale, weights, cu_q, cu_physical = _request_inputs(
        q_lengths, physical_lengths, heads
    )
    token_prefix = [0]
    for tokens in token_lengths:
        token_prefix.append(token_prefix[-1] + tokens)
    cu_tokens = torch.tensor(token_prefix, device=q.device, dtype=torch.int32)
    if kv.shape[0]:
        fused, table = _paged_inputs(
            (q, kv, scale, weights, cu_q, cu_physical), page_size
        )
        for request, length in enumerate(physical_lengths):
            live_pages = math.ceil(length / page_size)
            table[request, live_pages:].fill_(2**31 - 1)
            if length % page_size:
                page = int(table[request, live_pages - 1].item())
                fused.flatten(1)[page, : page_size * 128].view(page_size, 128)[
                    length % page_size :
                ].fill_(0x7F)
    else:
        fused = torch.empty((0, page_size, 1, 132), device=q.device, dtype=torch.uint8)
        table = torch.empty((len(q_lengths), 0), device=q.device, dtype=torch.int32)
    return q, fused, weights, cu_q, cu_tokens, table


@_requires_selective_sm100
@pytest.mark.parametrize("strategy", ["selective"])
def test_compression_ratio_one_keeps_original_plan_and_schema(strategy):
    q, kv, scales, weights, _, _ = _inputs(17, 8192, 8)
    cu_q = torch.tensor([0, 17], device=q.device, dtype=torch.int32)
    cu_kv = torch.tensor([0, 8192], device=q.device, dtype=torch.int32)
    fused, table = _paged_inputs((q, kv, scales, weights, cu_q, cu_kv))
    capacities = dict(
        max_context_length=8192, max_chunk_size=17, max_batch_size=1, kv_layout=fused
    )
    wrappers = [
        FP8PagedMQATopKWrapper(strategy=strategy, **capacities),
        FP8PagedMQATopKWrapper(strategy=strategy, compression_ratio=1, **capacities),
    ]
    outputs = []
    for wrapper in wrappers:
        wrapper.plan(cu_q, cu_kv, table)
        outputs.append(wrapper.run(q, weights, fused).clone())
        assert wrapper.compression_ratio == 1
    torch.cuda.synchronize()
    for output in outputs:
        _check_selection(
            output, _oracle_rows((q, kv, scales, weights, cu_q, cu_kv)), 512
        )
    from flashinfer.experimental.attn_scores.trace import (
        fp8_paged_mqa_selective_topk_trace,
        fp8_paged_mqa_topk_trace,
        fp8_paged_mqa_topk_trace_dispatch,
    )

    assert all(
        fp8_paged_mqa_topk_trace_dispatch(self=wrapper)
        is (
            fp8_paged_mqa_selective_topk_trace
            if strategy == "selective"
            else fp8_paged_mqa_topk_trace
        )
        for wrapper in wrappers
    )
    assert "compression_ratio" not in fp8_paged_mqa_selective_topk_trace.inputs
    for ratio in (0, 2, True, 4.0):
        with pytest.raises(ValueError, match="compression_ratio must be"):
            FP8PagedMQATopKWrapper(
                strategy=strategy, compression_ratio=ratio, **capacities
            )
    with pytest.raises(ValueError, match="only supported by selective"):
        FP8PagedMQATopKWrapper(strategy="full", compression_ratio=4, **capacities)


def _check_compressed_wrapper(output, payload, top_k):
    q, fused, weights, cu_q, cu_tokens, table = payload
    page_size = fused.shape[1]
    packed = fused.flatten(1)
    q_prefix, token_prefix = cu_q.tolist(), cu_tokens.tolist()
    for request in range(len(q_prefix) - 1):
        qb, qe = q_prefix[request : request + 2]
        tokens = token_prefix[request + 1] - token_prefix[request]
        physical = tokens // 4
        if physical:
            raw = packed[table[request, : math.ceil(physical / page_size)].long()]
            keys = (
                raw[:, : page_size * 128]
                .contiguous()
                .view(q.dtype)
                .reshape(-1, 128)[:physical]
                .float()
            )
            scales = (
                raw[:, page_size * 128 :]
                .contiguous()
                .view(torch.float32)
                .reshape(-1)[:physical]
            )
        for local, row in enumerate(range(qb, qe)):
            end = (tokens - (qe - qb) + local + 1) // 4
            count = min(top_k, end)
            assert (output[row, count:] == -1).all()
            ids = output[row, :count].long()
            assert ((ids >= 0) & (ids < end)).all()
            assert ids.unique().numel() == count
            if count:
                scores = (
                    (q[row].float() @ keys[:end].T).relu_() * weights[row, :, None]
                ).sum(0) * scales[:end]
                torch.testing.assert_close(
                    scores[ids].sort().values,
                    scores.topk(count).values.sort().values,
                    atol=5e-5,
                    rtol=1e-5,
                )


@_requires_selective_sm100
@pytest.mark.parametrize(
    ("heads", "page_size", "top_k", "q_lengths", "token_lengths"),
    [
        (8, 32, 512, [1, 2, 3], [1, 9, 11]),
        (64, 64, 2048, [37], [37]),
        (8, 128, 512, [1, 17], [1, 40001]),
        (8, 128, 512, [1, 2, 3], [1, 2, 3]),
        (64, 128, 512, [1, 17], [1, 40001]),
        (64, 128, 512, [9], [40001]),
        (64, 128, 2048, [9], [40001]),
        (64, 128, 2048, [1, 37], [1, 37]),
    ],
)
def test_engine_compressed_original_token_bounds_and_graph(
    heads, page_size, top_k, q_lengths, token_lengths, monkeypatch, ieee_matmul
):
    """Nonzero-Q empty requests, per-request floors, poisoned pages and repair."""
    monkeypatch.setenv("FLASHINFER_VALIDATE_INPUTS", "1")
    payload = _compressed_wrapper_fixture(q_lengths, token_lengths, heads, page_size)
    q, fused, weights, cu_q, cu_tokens, table = payload
    wrapper = SelectiveEngineCase()
    wrapper.plan(
        q,
        fused,
        weights,
        cu_q,
        cu_tokens,
        table,
        top_k=top_k,
        num_heads=heads,
        num_sms=4,
        compression_ratio=4,
    )
    output = wrapper.run(q=q, kv_fused=fused, weights=weights, block_table=table)
    torch.cuda.synchronize()
    _check_compressed_wrapper(output, payload, top_k)
    if page_size == 32:
        from flashinfer.experimental.attn_scores.trace import (
            _compressed_topk_reference,
            _compressed_topk_check,
        )

        trace_args = dict(
            q=q,
            kv_fused=fused,
            weights=weights,
            cu_q=cu_q,
            cu_kv=cu_tokens,
            block_table=table,
            top_k=top_k,
            compression_ratio=4,
        )
        reference = _compressed_topk_reference(**trace_args)
        assert _compressed_topk_check(reference, output, **trace_args)
        bad = output.clone()
        bad[0, 0] = 0  # The first nonzero-Q request has no physical key0.
        assert not _compressed_topk_check(reference, bad, **trace_args)
    expected_prefix = [0]
    for length in token_lengths:
        expected_prefix.append(expected_prefix[-1] + length // 4)
    assert wrapper._physical_cu_kv.tolist() == expected_prefix
    if expected_prefix[-1] == 0:
        assert wrapper._candidate_prepared is None
        assert wrapper._empty_physical
    if heads == 64 and q_lengths == [9]:
        engine = wrapper
        # R4 single-pool prefill is published by the H64 candidate variant.
        assert engine._candidate_schedule.h64_candidate
        assert engine._candidate_schedule.publication == "warp_striped"
        assert engine._candidate_prepared.query_tile == 4
        assert engine._candidate_prepared.num_umma_stages == 1
        assert engine._candidate_prepared.split_kv == 1
        assert engine._candidate_counts.shape[1] == 5
        assert engine._candidate_schedule.sample_count_segments == 32
    force_repair = max(token_lengths) == 40001
    if force_repair:
        engine = wrapper
        threshold, threshold_args = engine._sample_threshold_launch

        def forced_threshold(*args):
            threshold(*args)
            engine._thresholds.fill_(float("inf"))

        engine._sample_threshold_launch = forced_threshold, threshold_args
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = wrapper.run(q=q, kv_fused=fused, weights=weights, block_table=table)
    for _ in range(2):
        q.copy_((q.float() * 0.5).to(q.dtype))
        weights.mul_(0.75)
        for request, length in enumerate(token_lengths):
            if length // 4 >= 2 * page_size:
                table[request, :2].copy_(table[request, :2].flip(0))
        graph.replay()
        torch.cuda.synchronize()
        _check_compressed_wrapper(captured, payload, top_k)
        assert captured.data_ptr() == output.data_ptr()
    if force_repair:
        repair_engine = wrapper
        assert repair_engine._repair_flags.count_nonzero().item() > 0
        assert repair_engine._exact_error.item() == 0


@_requires_selective_sm100
@pytest.mark.parametrize("strategy", ["selective"])
@pytest.mark.parametrize("heads", [8, 64])
def test_compressed_wrapper_empty_live_empty_replan(strategy, monkeypatch, heads):
    """One bound pool supports empty/live steps; a failed pool binding retries."""
    empty = _compressed_wrapper_fixture([1], [1], heads, 128)
    live = _compressed_wrapper_fixture([1], [9], heads, 128)
    q, fused, weights, cu_q, cu_tokens, table = live
    specialization = dict(strategy=strategy, num_heads=heads, compression_ratio=4)
    capacities = dict(max_context_length=9, max_chunk_size=1, max_batch_size=1)
    wrapper = FP8PagedMQATopKWrapper(**specialization, **capacities, kv_layout=fused)
    wrapper.prepare_pool(fused)
    pool_binding = wrapper._capacity_plan._pool_bindings.copy()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.plan(cu_q, cu_tokens, table)
        output = wrapper.run(q, weights, fused)
    for tokens in (1, 9, 1):
        cu_tokens[-1] = tokens
        graph.replay()
        torch.cuda.synchronize()
        _check_compressed_wrapper(
            output, (q, fused, weights, cu_q, cu_tokens, table), 512
        )
        assert wrapper.error.item() == wrapper.repair_error.item() == 0
        assert wrapper._capacity_plan._pool_bindings == pool_binding
    del graph
    # Capacity changes construct a new wrapper rather than re-reserving one.
    empty_wrapper = FP8PagedMQATopKWrapper(
        **specialization,
        max_context_length=1,
        max_chunk_size=1,
        max_batch_size=1,
        kv_layout=empty[1],
    )
    empty_wrapper.plan(empty[3], empty[4], empty[5])
    result = empty_wrapper.run(empty[0], empty[2], empty[1])
    torch.cuda.synchronize()
    assert (result == -1).all()
    wrapper = FP8PagedMQATopKWrapper(**specialization, **capacities, kv_layout=fused)
    with monkeypatch.context() as patch:
        patch.setattr(
            type(wrapper._capacity_plan),
            "_build_engine",
            lambda self, _: (_ for _ in ()).throw(
                RuntimeError("injected compressed preparation failure")
            ),
        )
        with pytest.raises(
            RuntimeError, match="injected compressed preparation failure"
        ):
            wrapper.prepare_pool(fused)
    assert not wrapper._capacity_plan._pool_bindings
    with pytest.raises(RuntimeError, match="plan must be called"):
        wrapper.run(q, weights, fused)
    wrapper.prepare_pool(fused)
    cu_tokens[-1] = 9
    wrapper.plan(cu_q, cu_tokens, table, num_query_rows=q.shape[0])
    output = wrapper.run(q, weights, fused)
    torch.cuda.synchronize()
    _check_compressed_wrapper(output, (q, fused, weights, cu_q, cu_tokens, table), 512)
    # Restoring an instance's bound method would keep the plan and its private
    # CUDA modules alive until an unrelated cyclic collection.
    owner = weakref.ref(wrapper)
    plan_owner = weakref.ref(wrapper._capacity_plan)
    state_owner = weakref.ref(wrapper._capacity_plan._state)
    del wrapper
    assert owner() is None
    assert plan_owner() is None
    assert state_owner() is None


_requires_selective_cuda = pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("cutlass") is None,
    reason="CUDA and CuTe DSL are required",
)

_PAGED_INPUT_CACHE: dict[
    tuple[torch.Tensor, torch.Tensor, int], tuple[torch.Tensor, torch.Tensor]
] = {}


@_requires_selective_sm100
def test_candidate_publication_rejects_layout_from_another_split():
    inputs = _inputs(q_rows=1, kv_rows=128)
    prepared = _prepare(inputs, split_kv=1)
    values = torch.empty((16, 6144), device="cuda", dtype=torch.int64)
    split_two_lane_counts = torch.empty((16, 65), device="cuda", dtype=torch.int32)

    with pytest.raises(ValueError, match="split_kv=1 candidate publication requires"):
        _launch(
            _SelectiveLogitsMode.CANDIDATE,
            6144,
            values,
            split_two_lane_counts,
            prepared,
            inputs,
            torch.full((1,), float("-inf"), device="cuda"),
        )


@_requires_selective_sm100
def test_unified_wrapper_requires_explicit_supported_strategy():
    from flashinfer import FP8PagedMQATopKWrapper

    with pytest.raises(ValueError, match="both strategies require"):
        FP8PagedMQATopKWrapper(strategy="full")
    with pytest.raises(ValueError, match="strategy must be 'full' or 'selective'"):
        FP8PagedMQATopKWrapper(strategy="auto")


def _inputs(
    q_rows: int = 17,
    kv_rows: int = 387,
    num_heads: int = 8,
    head_dim: int = 128,
):
    torch.manual_seed(1)
    device = "cuda"
    q = (torch.randn(q_rows, num_heads, head_dim, device=device) * 0.1).to(
        torch.float8_e4m3fn
    )
    kv_f32 = torch.randn(kv_rows, head_dim, device=device) * 0.1
    amax = kv_f32.abs().amax(dim=1).clamp_min(1e-6)
    scale = torch.pow(2.0, torch.ceil(torch.log2(amax / 448.0)))
    kv = (kv_f32 / scale[:, None]).to(torch.float8_e4m3fn)
    weights = torch.randn(q_rows, num_heads, device=device, dtype=torch.float32)
    if q_rows == 1:
        cu_q = torch.tensor([0, 1], device=device, dtype=torch.int32)
        cu_kv = torch.tensor([0, kv_rows], device=device, dtype=torch.int32)
    else:
        first_kv = min(132, kv_rows // 2)
        cu_q = torch.tensor([0, 1, q_rows], device=device, dtype=torch.int32)
        cu_kv = torch.tensor([0, first_kv, kv_rows], device=device, dtype=torch.int32)
    return q, kv, scale, weights, cu_q, cu_kv


def _expected_sample_count(end, top_k):
    if end > 32 * top_k:
        return min(end, 3840)
    count = 256
    while count * top_k < 32 * end:
        count *= 2
    return min(end, 3840, count)


def _expected_sample_rank(end, count, top_k, grouped=True):
    if end == 0:
        return 0
    rank = math.ceil(min(4096, max(512, 4 * top_k)) * count / end)
    if grouped and count < 3840:
        mean = top_k * count / end
        rank = max(rank, int(mean + 4.5 * math.sqrt(mean)) + 1)
    return max(1, min(count, rank))


def _expected_sample_positions(end, count, group, device):
    position = torch.arange(count, device=device, dtype=torch.int64)
    if count == 3840:
        return position * (end - 1) // (count - 1)
    hashed = (position + group * 0x9E3779B9 + 0x243F6A88) & 0xFFFFFFFF
    hashed = ((hashed ^ (hashed >> 16)) * 0x85EBCA6B) & 0xFFFFFFFF
    hashed = ((hashed ^ (hashed >> 13)) * 0xC2B2AE35) & 0xFFFFFFFF
    hashed ^= hashed >> 16
    begin = position * end // max(count, 1)
    finish = (position + 1) * end // max(count, 1)
    return begin + ((hashed * (finish - begin).clamp_min(1)) >> 32)


def _request_inputs(q_lengths, kv_lengths, heads=8, head_dim=128):
    """Fresh raw owners and explicit request prefixes; no wrapper or graph state."""
    q, kv, scales, weights, _, _ = _inputs(
        sum(q_lengths), sum(kv_lengths), heads, head_dim
    )
    q_prefix, kv_prefix = [0], [0]
    for q_length, kv_length in zip(q_lengths, kv_lengths, strict=True):
        q_prefix.append(q_prefix[-1] + q_length)
        kv_prefix.append(kv_prefix[-1] + kv_length)
    return (
        q,
        kv,
        scales,
        weights,
        torch.tensor(q_prefix, device=q.device, dtype=torch.int32),
        torch.tensor(kv_prefix, device=q.device, dtype=torch.int32),
    )


def _valid_lengths(cu_q: torch.Tensor, cu_kv: torch.Tensor) -> list[int]:
    q_prefix = cu_q.cpu().tolist()
    kv_prefix = cu_kv.cpu().tolist()
    result = []
    for batch in range(len(q_prefix) - 1):
        q_len = q_prefix[batch + 1] - q_prefix[batch]
        kv_len = kv_prefix[batch + 1] - kv_prefix[batch]
        result.extend(kv_len - q_len + row + 1 for row in range(q_len))
    return result


def _fuse_paged_kv(
    kv: torch.Tensor, scale: torch.Tensor, block_size: int = 128
) -> torch.Tensor:
    """Pack contiguous KV and scales for the public paged-logits API."""
    head_dim = kv.shape[1]
    num_blocks = kv.shape[0] // block_size
    fused = torch.zeros(
        num_blocks,
        block_size * (head_dim + 4),
        device=kv.device,
        dtype=torch.uint8,
    )
    kv_blocks = kv.view(num_blocks, block_size, head_dim)
    scale_blocks = scale.view(num_blocks, block_size)
    scale_offset = block_size * head_dim
    for block in range(num_blocks):
        fused[block, :scale_offset] = kv_blocks[block].view(torch.uint8).reshape(-1)
        fused[block, scale_offset:] = (
            scale_blocks[block].contiguous().view(torch.uint8).reshape(-1)
        )
    return fused.view(num_blocks, block_size, 1, head_dim + 4)


def _paged_inputs(inputs, block_size: int = 128):
    """Build reusable request-local pages with a reversed physical mapping."""
    q, kv, scale, weights, cu_q, cu_kv = inputs
    key = (kv, scale, block_size)
    cached = _PAGED_INPUT_CACHE.get(key)
    if cached is not None:
        return cached
    q_prefix = cu_q.cpu().tolist()
    kv_prefix = cu_kv.cpu().tolist()
    pages_per_compute_tile = 128 // block_size
    max_pages = max(
        ((kv_prefix[i + 1] - kv_prefix[i] + 127) // 128 * pages_per_compute_tile)
        for i in range(len(kv_prefix) - 1)
    )
    table = torch.zeros(
        (len(q_prefix) - 1, max_pages), device=kv.device, dtype=torch.int32
    )
    page_values = []
    page_scales = []
    for batch in range(len(q_prefix) - 1):
        begin, end = kv_prefix[batch : batch + 2]
        for local_page, offset in enumerate(range(begin, end, block_size)):
            valid = min(block_size, end - offset)
            page = torch.zeros(
                (block_size, kv.shape[1]), device=kv.device, dtype=kv.dtype
            )
            page_scale = torch.ones(block_size, device=scale.device, dtype=scale.dtype)
            page[:valid] = kv[offset : offset + valid]
            page_scale[:valid] = scale[offset : offset + valid]
            table[batch, local_page] = len(page_values)
            page_values.append(page)
            page_scales.append(page_scale)
    paged_kv = torch.stack(page_values)
    paged_scale = torch.stack(page_scales)
    fused = _fuse_paged_kv(
        paged_kv.reshape(-1, kv.shape[1]), paged_scale.reshape(-1), block_size
    )
    # Exercise a non-identity physical layout in every selective test while
    # preserving each request's logical page order through the block table.
    fused = fused.flip(0).contiguous()
    table = fused.shape[0] - 1 - table
    _PAGED_INPUT_CACHE[key] = (fused, table)
    return fused, table


@pytest.mark.parametrize("heads,query_tile", [(8, 16), (64, 4)])
@pytest.mark.parametrize("ratio", [1, 4])
def test_fp8_grouped_public_scorer_live_maps(
    heads, query_tile, ratio, monkeypatch, ieee_matmul
):
    """Private grouped scoring preserves request bounds and live graph inputs."""
    from flashinfer.attn_scores.attn_scores import _fp8_paged_mqa_logits
    from flashinfer.attn_scores.fp8_kernel_config import _FP8MQAKernelConfig
    from flashinfer.attn_scores.query_group_metadata import _FP8QueryGroups

    if not torch.cuda.is_available() or not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("grouped FP8 scorer requires SM100 or SM103")
    q_lengths = [query_tile + 1, 0, 1]
    token_lengths = [ratio * 385 + ratio - 1, ratio * 129, 1]
    physical_lengths = [length // ratio for length in token_lengths]
    inputs = _request_inputs(q_lengths, physical_lengths, heads)
    q, kv, scales, weights, cu_q, cu_kv = inputs
    token_prefix = [0]
    for length in token_lengths:
        token_prefix.append(token_prefix[-1] + length)
    groups = _FP8QueryGroups.build(cu_q.cpu().tolist(), token_prefix, ratio, query_tile)
    fused, table = _paged_inputs(inputs)
    # Neither an empty-query request nor a zero-key row may dereference its map.
    table[1].fill_(2**30)
    if physical_lengths[2] == 0:
        table[2].fill_(2**30)
    gather = torch.tensor(groups.gather_rows, device=q.device, dtype=torch.int64)
    requests = torch.tensor(groups.request_ids, device=q.device, dtype=torch.int64)
    invalid = torch.tensor(groups.invalid_queries, device=q.device)
    grouped_q = torch.empty(
        (len(groups.rows), query_tile, heads, 128), device=q.device, dtype=q.dtype
    )
    grouped_weights = torch.empty(
        (len(gather), heads), device=q.device, dtype=weights.dtype
    )
    grouped_table = torch.empty(
        (len(groups.rows), table.shape[1]), device=q.device, dtype=table.dtype
    )
    metadata = torch.tensor(groups.rows, device=q.device, dtype=torch.int32)
    ends = torch.tensor(groups.group_ends, device=q.device, dtype=torch.int32)
    config = _FP8MQAKernelConfig(128, 1, 160, query_tile, 3, 3)
    output = torch.full(
        (q.shape[0], config.padded_columns(max(physical_lengths))),
        float("nan"),
        device=q.device,
    )

    def run():
        torch.index_select(
            q.view(torch.uint8),
            0,
            gather,
            out=grouped_q.view(-1, heads, 128).view(torch.uint8),
        )
        torch.index_select(weights, 0, gather, out=grouped_weights)
        grouped_q.view(-1, heads, 128).view(torch.uint8).masked_fill_(
            invalid[:, None, None], 0
        )
        grouped_weights.masked_fill_(invalid[:, None], 0)
        torch.index_select(table, 0, requests, out=grouped_table)
        return _fp8_paged_mqa_logits(
            grouped_q,
            fused,
            grouped_weights,
            grouped_table,
            ends,
            max(physical_lengths),
            kernel_config=config,
            query_metadata=metadata,
            output_rows=q.shape[0],
            out=output,
        )

    def check():
        row = 0
        physical_prefix = cu_kv.cpu().tolist()
        for req, (queries, tokens) in enumerate(
            zip(q_lengths, token_lengths, strict=True)
        ):
            base = physical_prefix[req]
            for local in range(queries):
                end = (tokens - queries + local + 1) // ratio
                dots = q[row].float() @ kv[base : base + end].float().T
                expected = (dots.relu() * weights[row, :, None]).sum(0) * scales[
                    base : base + end
                ]
                torch.testing.assert_close(
                    output[row, :end], expected, atol=1e-4, rtol=1e-4
                )
                row += 1

    run()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    check()
    old_output = output.clone()
    q.copy_((-q.float()).to(q.dtype))
    weights.mul_(0.5)
    # Keep logical keys unchanged while changing the physical map and payload.
    fused.copy_(fused.flip(0))
    valid_pages = table < fused.shape[0]
    table[valid_pages] = fused.shape[0] - 1 - table[valid_pages]
    graph.replay()
    check()
    assert not torch.allclose(old_output[0, :128], output[0, :128])
    run()
    check()
    if heads == 8 and ratio == 4:
        config = _FP8MQAKernelConfig(128, 1, 160, query_tile, 2, 2)
        run()
        check()


def _prepare(inputs, **kwargs):
    from flashinfer.experimental.attn_scores.selective_logits import (
        _prepare_fp8_mqa_selective_logits,
    )

    q, _kv, _scale, weights, cu_q, cu_kv = inputs
    kv_fused, block_table = _paged_inputs(inputs)
    return _prepare_fp8_mqa_selective_logits(
        q, kv_fused, weights, cu_q, cu_kv, block_table, **kwargs
    )


def _oracle_rows(inputs) -> list[torch.Tensor]:
    """Evaluate the reference per-head-ReLU FP32 score definition."""
    q, kv, scale, weights, cu_q, cu_kv = inputs
    q_prefix = cu_q.cpu().tolist()
    kv_prefix = cu_kv.cpu().tolist()
    result = []
    for batch in range(len(q_prefix) - 1):
        q_begin, q_end = q_prefix[batch : batch + 2]
        kv_begin, kv_end = kv_prefix[batch : batch + 2]
        q_len = q_end - q_begin
        kv_len = kv_end - kv_begin
        for local_row in range(q_len):
            valid = kv_len - q_len + local_row + 1
            dots = torch.matmul(
                q[q_begin + local_row].float(),
                kv[kv_begin : kv_begin + valid].float().T,
            )
            weighted = (torch.relu(dots) * weights[q_begin + local_row, :, None]).sum(
                dim=0
            )
            result.append(weighted * scale[kv_begin : kv_begin + valid])
    return result


def _launch(
    mode,
    capacity,
    values,
    index_counts,
    prepared,
    inputs,
    policy_values=None,
    radix_shift=0,
):
    from flashinfer.experimental.attn_scores.selective_logits import (
        _fp8_mqa_selective_logits,
    )

    q, _kv, _scale, weights, _cu_q, _cu_kv = inputs
    kv_fused, block_table = _paged_inputs(inputs)
    _fp8_mqa_selective_logits(
        q,
        kv_fused,
        weights,
        block_table,
        prepared,
        mode=mode,
        capacity=capacity,
        values=values,
        index_counts=index_counts,
        policy_values=policy_values,
        radix_shift=radix_shift,
    )


@_requires_selective_sm100
@pytest.mark.parametrize("num_heads", [8, 64])
@pytest.mark.parametrize("capacity", [7, 512])
@pytest.mark.parametrize("ties", [False, True])
def test_fused_exact_emission_banks(num_heads, capacity, ties):
    inputs = _inputs(q_rows=33, num_heads=num_heads)
    if ties:
        inputs[3].zero_()
    prepared = _prepare(inputs, num_sms=4, num_heads=num_heads)
    padded = (
        (33 + prepared.query_tile - 1) // prepared.query_tile
    ) * prepared.query_tile
    bank_rows = ((padded + 3) // 4) * 4
    sampled = torch.zeros((padded, 512), device="cuda", dtype=torch.float32)
    unused = torch.empty((padded, 1), device="cuda", dtype=torch.int32)
    _launch(_SelectiveLogitsMode.SAMPLE, 512, sampled, unused, prepared, inputs)
    lengths = _valid_lengths(inputs[4], inputs[5])
    thresholds = torch.stack(
        [sampled[row, valid // 2] for row, valid in enumerate(lengths)]
    )
    values = torch.full((2 * bank_rows, capacity), -7, device="cuda", dtype=torch.int32)
    counts = torch.empty((2 * bank_rows, 1), device="cuda", dtype=torch.int32)
    _launch(
        _SelectiveLogitsMode.EXACT,
        capacity,
        values,
        counts,
        prepared,
        inputs,
        thresholds,
    )
    for row, valid in enumerate(lengths):
        for bank in (0, 1):
            expected = (
                torch.nonzero(
                    (sampled[row, :valid] > thresholds[row])
                    if bank == 0
                    else (sampled[row, :valid] == thresholds[row])
                )
                .flatten()
                .to(torch.int32)
            )
            count = int(counts[row + bank * bank_rows, 0])
            assert count == expected.numel()
            actual = values[row + bank * bank_rows, : min(count, capacity)]
            assert actual.unique().numel() == actual.numel()
            assert torch.isin(actual, expected).all()
            if count <= capacity:
                torch.testing.assert_close(actual.sort().values, expected.sort().values)
    assert (counts[33:bank_rows] == 0).all()
    assert (counts[bank_rows + 33 :] == 0).all()


@_requires_selective_sm100
def test_selective_sample_matches_public_paged_logits_gemm():
    """Prove both logits implementations agree on identical causal inputs."""
    from flashinfer import fp8_paged_mqa_logits

    torch.manual_seed(7)
    q_rows, kv_rows = 16, 256
    q = (torch.randn(q_rows, 8, 128, device="cuda") * 0.1).to(torch.float8_e4m3fn)
    kv_f32 = torch.randn(kv_rows, 128, device="cuda") * 0.1
    amax = kv_f32.abs().amax(dim=1).clamp_min(1e-6)
    scale = torch.pow(2.0, torch.ceil(torch.log2(amax / 448.0)))
    kv = (kv_f32 / scale[:, None]).to(torch.float8_e4m3fn)
    weights = torch.randn(q_rows, 8, device="cuda", dtype=torch.float32)
    cu_q = torch.tensor([0, q_rows], device="cuda", dtype=torch.int32)
    cu_kv = torch.tensor([0, kv_rows], device="cuda", dtype=torch.int32)
    inputs = (q, kv, scale, weights, cu_q, cu_kv)

    prepared = _prepare(inputs)
    selective = torch.full((q_rows, kv_rows), float("nan"), device="cuda")
    unused_counts = torch.empty((q_rows, 1), device="cuda", dtype=torch.int32)
    _launch(
        _SelectiveLogitsMode.SAMPLE,
        kv_rows,
        selective,
        unused_counts,
        prepared,
        inputs,
    )

    paged = fp8_paged_mqa_logits(
        q.unsqueeze(0),
        _fuse_paged_kv(kv, scale),
        weights,
        torch.arange(2, device="cuda", dtype=torch.int32).unsqueeze(0),
        torch.tensor([kv_rows], device="cuda", dtype=torch.int32),
        kv_rows,
    )
    torch.cuda.synchronize()

    for row in range(q_rows):
        causal_end = kv_rows - q_rows + row + 1
        torch.testing.assert_close(
            selective[row, :causal_end],
            paged[row, :causal_end],
            atol=5e-5,
            rtol=1e-5,
        )
        assert torch.isnan(selective[row, causal_end:]).all()


@_requires_selective_sm100
def test_selective_h64_reads_arbitrary_physical_page_mapping():
    """Exercise the selective route directly with H64 and reordered pages."""
    from flashinfer.experimental.attn_scores.selective_logits import (
        _fp8_mqa_selective_logits,
        _prepare_fp8_mqa_selective_logits,
    )

    inputs = _inputs(q_rows=4, kv_rows=256, num_heads=64, head_dim=128)
    q, _kv, _scale, weights, cu_q, cu_kv = inputs
    kv_fused, block_table = _paged_inputs(inputs, block_size=64)
    prepared = _prepare_fp8_mqa_selective_logits(
        q,
        kv_fused,
        weights,
        cu_q,
        cu_kv,
        block_table,
        num_heads=64,
        head_dim=128,
    )
    assert prepared.query_tile == 2
    values = torch.full((4, 256), float("nan"), device="cuda")
    counts = torch.empty((4, 1), device="cuda", dtype=torch.int32)
    _fp8_mqa_selective_logits(
        q,
        kv_fused,
        weights,
        block_table,
        prepared,
        mode=_SelectiveLogitsMode.SAMPLE,
        capacity=256,
        values=values,
        index_counts=counts,
    )
    torch.cuda.synchronize()
    for row, expected in enumerate(_oracle_rows(inputs)):
        torch.testing.assert_close(
            values[row, : expected.numel()], expected, atol=5e-5, rtol=1e-5
        )
        assert torch.isnan(values[row, expected.numel() :]).all()

    candidates = torch.empty((4, 256), device="cuda", dtype=torch.int64)
    candidate_counts = torch.empty((4, 1), device="cuda", dtype=torch.int32)
    thresholds = torch.full((4,), float("-inf"), device="cuda")
    _fp8_mqa_selective_logits(
        q,
        kv_fused,
        weights,
        block_table,
        prepared,
        mode=_SelectiveLogitsMode.CANDIDATE,
        capacity=256,
        values=candidates,
        index_counts=candidate_counts,
        policy_values=thresholds,
    )
    torch.cuda.synchronize()
    for row, valid in enumerate(_valid_lengths(cu_q, cu_kv)):
        assert int(candidate_counts[row, 0]) == valid
        indices = (candidates[row, :valid] & 0xFFFFFFFF).sort().values
        torch.testing.assert_close(
            indices, torch.arange(valid, device="cuda", dtype=torch.int64)
        )


@_requires_selective_sm100
def test_unified_wrapper_matches_ordinary_logits_topk():
    """Public capacity replay agrees with the ordinary full scorer."""
    from flashinfer import fp8_paged_mqa_logits

    q, kv, scale, weights, _, _ = _inputs(q_rows=1024, kv_rows=2048)
    cu_q = torch.tensor([0, q.shape[0]], device=q.device, dtype=torch.int32)
    cu_kv = torch.tensor([0, kv.shape[0]], device=q.device, dtype=torch.int32)
    fused = _fuse_paged_kv(kv, scale)
    table = torch.arange(
        kv.shape[0] // 128, device=q.device, dtype=torch.int32
    ).unsqueeze(0)
    batches = q.shape[0] // 16
    ordinary = fp8_paged_mqa_logits(
        q.reshape(batches, 16, 8, 128),
        fused,
        weights,
        table.expand(batches, -1).contiguous(),
        torch.full((batches,), kv.shape[0], device=q.device, dtype=torch.int32),
        kv.shape[0],
    ).reshape(q.shape[0], -1)
    wrapper = FP8PagedMQATopKWrapper(
        strategy="selective",
        max_context_length=2048,
        max_chunk_size=q.shape[0],
        max_batch_size=1,
        kv_layout=fused,
    )
    wrapper.plan(cu_q, cu_kv, table)
    for _ in range(2):
        selected = wrapper.run(q, weights, fused)
        torch.cuda.synchronize()
        assert selected.shape == (q.shape[0], 512) and selected.dtype == torch.int32
        assert wrapper.error.item() == wrapper.repair_error.item() == 0
        _check_selection(
            selected,
            [ordinary[r, :end] for r, end in enumerate(_valid_lengths(cu_q, cu_kv))],
            512,
        )


@_requires_selective_sm100
@pytest.mark.parametrize("strategy", ["full", "selective"])
def test_topk_common_serving_contract(strategy):
    """Both strategies stage metadata once, accept layer out, and fail closed."""
    inputs = _request_inputs([2, 0, 1], [1024, 0, 2048])
    q, _, _, weights, qo, kv = inputs
    pool, table = _paged_inputs(inputs)
    q = torch.cat((q, torch.zeros_like(q[:2])))
    weights = torch.cat((weights, torch.zeros_like(weights[:2])))
    wrapper = FP8PagedMQATopKWrapper(
        strategy=strategy,
        max_context_length=2048,
        max_chunk_size=5,
        max_batch_size=3,
        kv_layout=pool,
    )
    buffers = wrapper.input_buffers()
    buffers["q"].copy_(q)
    buffers["weights"].copy_(weights)
    wrapper.plan(qo, kv, table)
    wrapper.prepare_pool(pool)
    other_pool = pool.clone()
    wrapper.prepare_pool(other_pool)
    out = torch.empty((5, 512), device=q.device, dtype=torch.int32)
    expected = _oracle_rows(inputs)
    for layer_pool in (pool, other_pool):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = wrapper.run(buffers["q"], buffers["weights"], layer_pool, out=out)
        graph.replay()
        torch.cuda.synchronize()
        assert result.data_ptr() == out.data_ptr()
        _check_selection(out[:3], expected, 512)
        assert (out[3:] == -1).all()
        assert wrapper.error.item() == wrapper.repair_error.item() == 0
        del graph
    wrapper.clear_pools()
    bad_table = torch.full_like(table, -1)
    wrapper.plan(qo, kv, bad_table)
    wrapper.prepare_pool(pool)
    wrapper.run(q, weights, pool, out=out)
    torch.cuda.synchronize()
    assert wrapper.error.item() == 8
    assert (out == -1).all()


@_requires_selective_sm100
def test_full_capacity_bucket_accepts_smaller_step(monkeypatch):
    """Full plans and trace axes follow the live batch, not reserved capacity."""
    from flashinfer.fi_trace import fi_trace
    from flashinfer.trace.template import TraceTemplate

    inputs = _request_inputs([2, 1], [1024, 2048])
    q, _, _, weights, qo, kv = inputs
    pool, table = _paged_inputs(inputs)
    wrapper = FP8PagedMQATopKWrapper(
        strategy="full",
        max_context_length=2048,
        max_chunk_size=5,
        max_batch_size=4,
        kv_layout=pool,
    )
    wrapper.plan(qo, kv, table, num_query_rows=q.shape[0])
    wrapper.prepare_pool(pool)
    expected = _oracle_rows(inputs)
    result = wrapper.run(q, weights, pool)
    torch.cuda.synchronize()
    _check_selection(result, expected, 512)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = wrapper.run(q, weights, pool)
    graph.replay()
    torch.cuda.synchronize()
    _check_selection(captured, expected, 512)
    observed = []
    build = TraceTemplate.build_fi_trace_fn

    def observe_build(template, api):
        trace = build(template, api)

        def observe_inputs(*args, **kwargs):
            observed.append(
                (kwargs["block_table"].shape[0], kwargs["qo_indptr"].numel())
            )
            return trace(*args, **kwargs)

        return observe_inputs

    monkeypatch.setattr(TraceTemplate, "build_fi_trace_fn", observe_build)
    definition = fi_trace(wrapper.run, q=q, weights=weights, kv_fused=pool)
    assert observed == [(2, 3)]
    assert definition["inputs"]["block_table"]["shape"][0] == "batch_size"


@_requires_selective_sm100
def test_capacity_retired_metadata_modules_do_not_unload_during_capture(
    monkeypatch, isolated_capture_gc
):
    """A predecessor's cyclic CuTe modules must be retired before another capture."""
    import weakref

    from flashinfer.experimental.attn_scores.selective_capacity import (
        _CapacityStepInputs,
        _SelectiveCapacity,
    )
    from cutlass.cutlass_dsl.cuda_jit_executor import CudaDialectJitModule

    capacity = _SelectiveCapacity(8, 128, 128, 1, 8192, 1, 1)
    state = _CapacityStepInputs(capacity, device="cuda", num_pages=64)
    state.prepare_sampling(512)
    module_type = CudaDialectJitModule
    original_unload = module_type.unload
    unloads = []

    def observe_unload(module):
        if module.cuda_library:
            unloads.append(torch.cuda.is_current_stream_capturing())
        return original_unload(module)

    monkeypatch.setattr(module_type, "unload", observe_unload)
    state.prepare()
    state_ref = weakref.ref(state)
    compiled_refs = [weakref.ref(state._compiled), weakref.ref(state._sample_compiled)]
    module_refs = [
        weakref.ref(state._compiled.jit_module),
        weakref.ref(state._sample_compiled.jit_module),
    ]
    output = torch.empty(1, device="cuda")
    graph = torch.cuda.CUDAGraph()
    enabled = gc.isenabled()
    gc.disable()
    try:
        del state
        # Refcount release remains strict: collecting while the owner is alive
        # would hide a product-owned cycle and its deferred CUDA destruction.
        assert state_ref() is None
        assert all(reference() is None for reference in compiled_refs)
        assert unloads == [False, False]
        assert all(
            reference() is None
            or (reference().is_unloaded() and not reference().cuda_library)
            for reference in module_refs
        )
        with torch.cuda.graph(graph):
            gc.collect()
            output.fill_(1)
        graph.replay()
        torch.cuda.synchronize()
        assert output.item() == 1
        assert unloads == [False, False]
    finally:
        if enabled:
            gc.enable()


@_requires_selective_sm100
@pytest.mark.parametrize("top_k", [512, 1024, 2048])
def test_unified_wrapper_full_strategy_ragged_causal_topk(monkeypatch, top_k):
    """Full and selective strategies share the exact public result contract."""
    from flashinfer import FP8PagedMQATopKWrapper

    inputs = _request_inputs((2, 3), (4096, 4352))
    q, kv, scale, weights, cu_q, cu_kv = inputs
    kv_fused, block_table = _paged_inputs(inputs)
    oracle = _oracle_rows(inputs)

    torch_topk = torch.topk

    def reject_dense_topk(*args, **kwargs):
        raise AssertionError(
            "full strategy must use the native variable-length selector"
        )

    monkeypatch.setattr(torch, "topk", reject_dense_topk)

    wrapper = FP8PagedMQATopKWrapper(
        strategy="full",
        num_heads=q.shape[1],
        head_dim=q.shape[2],
        page_size=kv_fused.shape[1],
        top_k=top_k,
        max_context_length=int((cu_kv[1:] - cu_kv[:-1]).max().item()),
        max_chunk_size=q.shape[0],
        max_batch_size=cu_q.numel() - 1,
        kv_layout=kv_fused,
    )
    wrapper.plan(cu_q, cu_kv, block_table, num_query_rows=q.shape[0])
    wrapper.prepare_pool(kv_fused)
    wrapper.run(q, weights, kv_fused)
    first = wrapper.run(q=q, kv_fused=kv_fused, weights=weights)
    # Unwritten causal tails are not part of the selector's input domain.
    # Poison them to catch accidental dense selection after future refactors.
    invalid = (
        torch.arange(wrapper._full_scores.shape[1], device=q.device)[None, :]
        >= wrapper._full_capacity_plan._lengths[: wrapper._full_scores.shape[0], None]
    )
    wrapper._full_scores.masked_fill_(invalid, float("inf"))
    second = wrapper.run(q=q, kv_fused=kv_fused, weights=weights)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = wrapper.run(q=q, kv_fused=kv_fused, weights=weights)
    graph.replay()
    torch.cuda.synchronize()
    monkeypatch.setattr(torch, "topk", torch_topk)

    assert first.data_ptr() == second.data_ptr()
    assert first.data_ptr() == captured.data_ptr()
    assert first.shape == (q.shape[0], top_k)
    assert first.dtype == torch.int32
    _check_selection(first, oracle, top_k)

    # Retire owners by refcount: delayed metadata unload during another
    # capture is unsafe even though the wrapper owns no graph.
    owner = weakref.ref(wrapper)
    gc_enabled = gc.isenabled()
    gc.disable()
    try:
        del wrapper
        assert owner() is None
    finally:
        if gc_enabled:
            gc.enable()


@_requires_selective_sm100
def test_full_wrapper_plan_capture_preserves_prepared_pool():
    """Prepared full planning records device refreshes in the caller graph."""
    inputs = _request_inputs((2,), (1024,))
    q, _, _, weights, qo_indptr, kv_indptr = inputs
    fused, table = _paged_inputs(inputs)
    wrapper = FP8PagedMQATopKWrapper(
        strategy="full",
        num_heads=q.shape[1],
        head_dim=q.shape[2],
        page_size=fused.shape[1],
        top_k=512,
        max_context_length=int((kv_indptr[1:] - kv_indptr[:-1]).max().item()),
        max_chunk_size=q.shape[0],
        max_batch_size=qo_indptr.numel() - 1,
        kv_layout=fused,
    )
    wrapper.plan(qo_indptr, kv_indptr, table, num_query_rows=q.shape[0])
    wrapper.prepare_pool(fused)
    wrapper.run(q, weights, fused)
    oracle = _oracle_rows(inputs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.plan(qo_indptr, kv_indptr, table, num_query_rows=q.shape[0])
        captured = wrapper.run(q, weights, fused)
    graph.replay()
    torch.cuda.synchronize()
    _check_selection(captured, oracle, 512)
    selected = wrapper.run(q, weights, fused)
    torch.cuda.synchronize()
    _check_selection(selected, oracle, 512)


@_requires_selective_sm100
def test_full_wrapper_gvr2_workspace_capture_and_replan(monkeypatch, ieee_matmul):
    """The full prefill route owns scratch across eager and graph streams."""
    from flashinfer.topk_varlen.kernels import gvr2_topk_host

    q_rows, kv_rows = 8192, 65536
    q, kv, scale, weights, _, _ = _inputs(q_rows=q_rows, kv_rows=kv_rows)
    cu_q = torch.tensor([0, q_rows], device="cuda", dtype=torch.int32)
    cu_kv = torch.tensor([0, kv_rows], device="cuda", dtype=torch.int32)
    fused, table = _paged_inputs((q, kv, scale, weights, cu_q, cu_kv))
    wrapper = FP8PagedMQATopKWrapper(
        strategy="full",
        num_heads=q.shape[1],
        head_dim=q.shape[2],
        page_size=fused.shape[1],
        top_k=512,
        max_context_length=int((cu_kv[1:] - cu_kv[:-1]).max().item()),
        max_chunk_size=q.shape[0],
        max_batch_size=cu_q.numel() - 1,
        kv_layout=fused,
    )
    wrapper.plan(cu_q, cu_kv, table, num_query_rows=q.shape[0])
    wrapper.prepare_pool(fused)
    wrapper.run(q, weights, fused)
    # Keep a separate eager result because every layer shares output storage.
    planned_output = wrapper._selected.clone()
    slab = wrapper._full_gvr2_workspace
    assert slab is not None and slab.dtype == torch.int32
    assert slab.numel() * slab.element_size() == gvr2_topk_host.workspace_bytes()
    assert slab.device == q.device and slab.data_ptr() % 16 == 0

    def check_rows(output, logical_kv, logical_scale):
        ends = kv_rows - q_rows + torch.arange(q_rows, device=q.device) + 1
        assert ((output >= 0) & (output < ends[:, None])).all()
        # The faithful capture geometry is large. Independent score/TopK
        # checks cover representative boundary/interior rows; small ragged
        # tests separately cover every row and all supported output widths.
        for row in (0, 15, 16, q_rows // 2, q_rows - 1):
            end = kv_rows - q_rows + row + 1
            dots = q[row].float() @ logical_kv[:end].float().T
            scores = (torch.relu(dots) * weights[row, :, None]).sum(0)
            scores *= logical_scale[:end]
            indices = output[row].long()
            assert indices.unique().numel() == 512
            torch.testing.assert_close(
                scores[indices].sort().values,
                scores.topk(512).values.sort().values,
                atol=5e-5,
                rtol=1e-5,
            )

    check_rows(planned_output, kv, scale)
    wrapper.run(q=q.view_as(q), kv_fused=fused, weights=weights)
    stream = torch.cuda.Stream(device=q.device)
    stream.wait_stream(torch.cuda.current_stream(q.device))
    with torch.cuda.stream(stream):
        first = wrapper.run(q=q, kv_fused=fused, weights=weights)
    stream.synchronize()
    check_rows(first, kv, scale)
    outer = torch.cuda.CUDAGraph()
    with torch.cuda.graph(outer, stream=stream):
        captured = wrapper.run(q=q, kv_fused=fused, weights=weights)
    outer.replay()
    torch.cuda.synchronize()
    assert captured.data_ptr() == first.data_ptr()
    check_rows(captured, kv, scale)

    q.copy_(q.float().roll(1, dims=1).to(q.dtype))
    weights.copy_(weights.flip(1))
    kv.copy_(kv.float().roll(1, dims=0).to(kv.dtype))
    scale.copy_(scale.roll(1))
    fused.copy_(_fuse_paged_kv(kv, scale).flip(0))
    table.copy_(table.flip(1))
    # Refresh maps without replacing scratch or the prepared caller graph.
    wrapper.plan(cu_q, cu_kv, table)
    outer.replay()
    torch.cuda.synchronize()
    reversed_kv = kv.float().view(-1, 128, kv.shape[1]).flip(0).reshape_as(kv)
    reversed_scale = scale.view(-1, 128).flip(0).reshape_as(scale)
    check_rows(captured, reversed_kv, reversed_scale)
    assert wrapper._full_gvr2_workspace is slab

    one_q = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    one_query = q[:1]
    one_weights = weights[:1]
    other = FP8PagedMQATopKWrapper(
        strategy="full",
        num_heads=one_query.shape[1],
        head_dim=one_query.shape[2],
        page_size=fused.shape[1],
        top_k=512,
        max_context_length=int((cu_kv[1:] - cu_kv[:-1]).max().item()),
        max_chunk_size=one_query.shape[0],
        max_batch_size=one_q.numel() - 1,
        kv_layout=fused,
    )
    other.plan(one_q, cu_kv, table, num_query_rows=one_query.shape[0])
    other.prepare_pool(fused)
    other.run(one_query, one_weights, fused)
    assert other._full_gvr2_workspace.data_ptr() != slab.data_ptr()
    other_slab = other._full_gvr2_workspace
    with pytest.raises(ValueError, match="requires top_k in"):
        FP8PagedMQATopKWrapper(
            strategy="full",
            top_k=8,
            max_context_length=1,
            max_chunk_size=1,
            max_batch_size=1,
            kv_layout=fused,
        )
    assert other._full_gvr2_workspace is other_slab
    assert other.run(q=one_query, kv_fused=fused, weights=one_weights).shape == (1, 512)

    def fail_prepare(*args, **kwargs):
        raise RuntimeError("injected full GVR preparation failure")

    with monkeypatch.context() as patch:
        patch.setattr(type(wrapper._full_capacity_plan), "_bind_launches", fail_prepare)
        with pytest.raises(RuntimeError, match="injected full GVR preparation failure"):
            wrapper.plan(one_q, cu_kv, table, num_query_rows=one_query.shape[0])
            wrapper.prepare_pool(fused)
            wrapper.run(one_query, one_weights, fused)
        with pytest.raises(RuntimeError, match="injected full GVR preparation failure"):
            wrapper.run(one_query, one_weights, fused)
    assert wrapper._full_gvr2_workspace is slab
    # An eager retry prepares the missing binding after the failure is removed.
    wrapper.run(one_query, one_weights, fused)
    assert wrapper._full_gvr2_workspace is slab
    recovered = wrapper.run(q=one_query, kv_fused=fused, weights=one_weights)
    torch.cuda.synchronize()
    assert recovered.shape == (1, 512)
    _check_selection(
        recovered,
        _oracle_rows(
            (one_query, reversed_kv, reversed_scale, one_weights, one_q, cu_kv)
        ),
        512,
    )
    owner = weakref.ref(wrapper)
    plan_owner = weakref.ref(wrapper._full_capacity_plan)
    del wrapper
    assert owner() is None
    assert plan_owner() is None


@_requires_selective_sm100
@pytest.mark.parametrize("top_k", [1, 8, 17, 32, 128, 511, 512, 2048])
def test_unified_wrapper_ragged_batch_and_top_k(top_k):
    """Public TopK supports ragged requests across the selected K domain."""
    inputs = _request_inputs((2, 3), (7168, 9216))
    q, _, _, weights, cu_q, cu_kv = inputs
    fused, table = _paged_inputs(inputs)
    wrapper = FP8PagedMQATopKWrapper(
        strategy="selective",
        top_k=top_k,
        max_context_length=9216,
        max_chunk_size=5,
        max_batch_size=2,
        kv_layout=fused,
    )
    wrapper.plan(cu_q, cu_kv, table)
    selected = wrapper.run(q, weights, fused)
    torch.cuda.synchronize()
    assert wrapper.error.item() == wrapper.repair_error.item() == 0
    _check_selection(selected, _oracle_rows(inputs), top_k)


@_requires_selective_sm100
@pytest.mark.parametrize("strategy", ["full", "selective"])
@pytest.mark.parametrize("top_k", [512, 1024, 2048])
def test_unified_wrapper_underfilled_causal_rows(strategy, top_k):
    """Every short row keeps all keys and pads its suffix with -1."""
    from flashinfer.experimental.attn_scores.trace import (
        fp8_paged_mqa_topk_trace,
        fp8_paged_mqa_selective_topk_trace,
    )

    inputs = _request_inputs((2, 3), (2, 4))
    q, _, _, weights, cu_q, cu_kv = inputs
    fused, table = _paged_inputs(inputs)
    if strategy == "selective":
        wrapper = FP8PagedMQATopKWrapper(
            strategy=strategy,
            top_k=top_k,
            max_context_length=4,
            max_chunk_size=5,
            max_batch_size=2,
            kv_layout=fused,
        )
        wrapper.plan(cu_q, cu_kv, table)
        selected = wrapper.run(q, weights, fused)
        template = fp8_paged_mqa_selective_topk_trace
        trace_args = dict(
            q=q,
            weights=weights,
            qo_indptr=cu_q,
            kv_indptr=cu_kv,
            kv_fused=fused,
            block_table=table,
            top_k=top_k,
        )
    else:
        wrapper = FP8PagedMQATopKWrapper(
            strategy=strategy,
            top_k=top_k,
            max_context_length=4,
            max_chunk_size=5,
            max_batch_size=2,
            kv_layout=fused,
        )
        wrapper.plan(cu_q, cu_kv, table, num_query_rows=q.shape[0])
        wrapper.prepare_pool(fused)
        wrapper.run(q, weights, fused)
        selected = wrapper.run(q=q, kv_fused=fused, weights=weights)
        template = fp8_paged_mqa_topk_trace
        trace_args = dict(
            q=q,
            kv_fused=fused,
            weights=weights,
            block_table=table,
            qo_indptr=cu_q,
            kv_indptr=cu_kv,
            top_k=top_k,
        )
    torch.cuda.synchronize()
    _check_selection(selected, _oracle_rows(inputs), top_k)
    reference = template.reference(**trace_args)
    assert template.check(reference, selected, **trace_args)
    bad = selected.clone()
    bad[0, _valid_lengths(cu_q, cu_kv)[0]] = 0
    assert not template.check(reference, bad, **trace_args)


@_requires_selective_sm100
def test_engine_compact_repair_queue_changes_cuda_graph():
    """Replaying a growing then empty repair queue cannot consume stale TopK rows."""
    from flashinfer import FP8PagedMQATopKWrapper

    rows, kv_rows = 33, 32768
    q, kv, scale, weights, _, _ = _inputs(q_rows=rows, kv_rows=kv_rows)
    cu_q = torch.tensor([0, rows], device="cuda", dtype=torch.int32)
    cu_kv = torch.tensor([0, kv_rows], device="cuda", dtype=torch.int32)
    fused, table = _paged_inputs((q, kv, scale, weights, cu_q, cu_kv))
    payload = q, fused, weights, table
    full = FP8PagedMQATopKWrapper(
        strategy="full",
        num_heads=q.shape[1],
        head_dim=q.shape[2],
        page_size=fused.shape[1],
        top_k=512,
        max_context_length=int((cu_kv[1:] - cu_kv[:-1]).max().item()),
        max_chunk_size=q.shape[0],
        max_batch_size=cu_q.numel() - 1,
        kv_layout=fused,
    )
    full.plan(cu_q, cu_kv, table, num_query_rows=q.shape[0])
    full.prepare_pool(fused)
    full.run(q, weights, fused)
    reference = full.run(payload[0], payload[2], payload[1]).long()
    scores = torch.nn.utils.rnn.pad_sequence(
        _oracle_rows((q, kv, scale, weights, cu_q, cu_kv)),
        batch_first=True,
        padding_value=float("-inf"),
    )
    expected = scores.gather(1, reference).sort(1).values
    wrapper = SelectiveEngineCase()
    wrapper.plan(q, fused, weights, cu_q, cu_kv, table)
    assert wrapper._compact_repair

    force_rows = torch.zeros(1, device="cuda", dtype=torch.int32)
    forced_thresholds = torch.full_like(wrapper._thresholds, float("inf"))
    row_ids = torch.arange(rows, device="cuda")
    threshold, threshold_args = wrapper._sample_threshold_launch

    def controlled_threshold(*args):
        threshold(*args)
        torch.where(
            row_ids < force_rows,
            forced_thresholds,
            wrapper._thresholds,
            out=wrapper._thresholds,
        )

    wrapper._sample_threshold_launch = controlled_threshold, threshold_args
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        selected = wrapper.run(*payload)
    counts = []
    for forced in (0, 1, rows, 0):
        force_rows.fill_(forced)
        graph.replay()
        torch.cuda.synchronize()
        counts.append(int(wrapper._repair_active_count.item()))
        assert counts[-1] >= forced
        assert int(wrapper._exact_error.item()) == 0
        indices = selected.long()
        assert ((indices >= 0) & (indices < wrapper._row_ends[:, None])).all()
        ordered = indices.sort(1).values
        assert (ordered[:, 1:] != ordered[:, :-1]).all()
        torch.testing.assert_close(
            scores.gather(1, indices).sort(1).values, expected, atol=1e-4, rtol=1e-4
        )
    assert counts[0] == counts[-1] == 0


@_requires_selective_sm100
@pytest.mark.parametrize("top_k", [512, 2048])
@pytest.mark.parametrize("heads", [8, 64])
def test_engine_underfilled_causal_rows_skip_repair(top_k, heads):
    """Complete short rows skip both selected repair layouts."""

    q, kv, scale, weights, _cu_q, _cu_kv = _inputs(
        q_rows=16, kv_rows=128, num_heads=heads
    )
    cu_q = torch.tensor([0, 16], device="cuda", dtype=torch.int32)
    cu_kv = torch.tensor([0, 128], device="cuda", dtype=torch.int32)
    inputs = (q, kv, scale, weights, cu_q, cu_kv)
    kv_fused, block_table = _paged_inputs(inputs)

    wrapper = SelectiveEngineCase()
    wrapper.plan(
        q,
        kv_fused,
        weights,
        cu_q,
        cu_kv,
        block_table,
        top_k=top_k,
        num_heads=heads,
        num_sms=4,
    )
    _assert_repair_position_storage(wrapper)
    for _ in range(3):
        # Every row publishes all its keys. Repair launches must predicate
        # away score reads/writes even with stale state from another layer.
        wrapper._ranks.fill_(top_k)
        selected = wrapper.run(q, kv_fused, weights, block_table).long()
        torch.cuda.synchronize()
        assert not wrapper._repair_flags.any().item()

    assert wrapper._compact_repair == (heads == 8)
    assert int(wrapper._exact_error.item()) == 0
    assert torch.count_nonzero(wrapper._repair_flags).item() == 0
    if heads == 8:
        assert int(wrapper._second_active_count.item()) == 0
    for row, valid in enumerate(_valid_lengths(cu_q, cu_kv)):
        indices = selected[row, :valid]
        assert ((indices >= 0) & (indices < valid)).all()
        assert (selected[row, valid:] == -1).all()
        assert torch.equal(
            torch.unique(indices).cpu(), torch.arange(valid, dtype=torch.int64)
        )


@_requires_selective_sm100
@pytest.mark.parametrize(
    ("num_heads", "query_tile"),
    [(128, 1), (64, 2), (32, 4), (8, 16)],
)
def test_engine_segmented_publication_query_tiles(num_heads, query_tile):
    """Q1/Q2/Q4/Q16 use the same segmented public candidate path."""
    from flashinfer import fp8_paged_mqa_logits

    q_rows = query_tile + 1
    kv_rows = 131072
    q, kv, scale, weights, _cu_q, _cu_kv = _inputs(
        q_rows=q_rows, kv_rows=kv_rows, num_heads=num_heads, head_dim=128
    )
    cu_q = torch.tensor([0, q_rows], device="cuda", dtype=torch.int32)
    cu_kv = torch.tensor([0, kv_rows], device="cuda", dtype=torch.int32)
    kv_fused = _fuse_paged_kv(kv, scale)
    block_table = torch.arange(
        kv_rows // 128 - 1, -1, -1, device="cuda", dtype=torch.int32
    ).unsqueeze(0)
    ordinary = fp8_paged_mqa_logits(
        q.unsqueeze(0),
        kv_fused,
        weights,
        block_table,
        torch.tensor([kv_rows], device="cuda", dtype=torch.int32),
        kv_rows,
    )

    wrapper = SelectiveEngineCase()
    wrapper.plan(
        q,
        kv_fused,
        weights,
        cu_q,
        cu_kv,
        block_table,
        num_heads=num_heads,
        head_dim=128,
        num_sms=2,
    )
    assert wrapper._candidate_prepared.query_tile == query_tile
    warp_ranked = num_heads == 64
    assert wrapper._candidate_schedule.publication == (
        "warp_striped" if warp_ranked else "lane_local"
    )
    assert wrapper._candidate_counts.shape[1] == (5 if warp_ranked else 33)
    assert wrapper._candidate_schedule.sample_count_segments == 32
    selected = wrapper.run(q, kv_fused, weights, block_table).long()
    torch.cuda.synchronize()

    assert int(wrapper._exact_error.item()) == 0
    if wrapper._padded_rows > q_rows:
        assert (wrapper._candidate_counts[q_rows:] == 0).all()
    for row in range(q_rows):
        row_end = kv_rows - q_rows + row + 1
        assert ((selected[row] >= 0) & (selected[row] < row_end)).all()
        actual_scores = ordinary[row, selected[row]]
        expected_scores = torch.topk(ordinary[row, :row_end], 512).values
        torch.testing.assert_close(
            actual_scores.sort().values,
            expected_scores.sort().values,
            atol=5e-5,
            rtol=1e-5,
        )


@_requires_selective_sm100
@pytest.mark.parametrize(
    ("num_heads", "q_rows", "expected_split"),
    [(8, 32, 1), (8, 64, 2), (64, 6, 1), (64, 12, 1)],
)
def test_engine_selects_split_kv_from_sm_waves(num_heads, q_rows, expected_split):
    """The public plan selects the smallest layout with best SM-wave use."""

    kv_rows = 4096
    inputs = _request_inputs((q_rows,), (kv_rows,), num_heads)
    q, kv, scale, weights, cu_q, cu_kv = inputs
    kv_fused, block_table = _paged_inputs(inputs)
    oracle = _oracle_rows(inputs)

    wrapper = SelectiveEngineCase()
    wrapper.plan(
        q,
        kv_fused,
        weights,
        cu_q,
        cu_kv,
        block_table,
        num_heads=num_heads,
        head_dim=128,
        num_sms=3,
    )
    assert wrapper._candidate_prepared.split_kv == expected_split
    assert (
        wrapper._candidate_schedule.query_tile == wrapper._candidate_prepared.query_tile
    )
    assert wrapper._candidate_schedule.split_kv == expected_split
    warp_ranked = num_heads == 64
    # Split-1 H64 prefill uses the H64 candidate variant's four warp segments.
    assert wrapper._candidate_schedule.h64_candidate == (
        warp_ranked and expected_split == 1
    )
    assert wrapper._candidate_schedule.publication == (
        "warp_striped" if warp_ranked else "lane_local"
    )
    assert wrapper._candidate_counts.shape[1] == (
        1 + (4 if warp_ranked else 32) * expected_split
    )
    assert wrapper._candidate_schedule.sample_count_segments == 32 * expected_split
    selected = wrapper.run(q, kv_fused, weights, block_table).long()
    torch.cuda.synchronize()

    assert int(wrapper._exact_error.item()) == 0
    _check_selection(selected, oracle, 512)


@_requires_selective_sm100
@pytest.mark.parametrize(
    ("num_heads", "kv_rows", "top_k", "expected_split", "force_repair"),
    (
        (8, 4096, 8, 16, False),
        (64, 8192, 17, 32, True),
        (8, 16384, 512, 64, False),
        (8, 16384, 512, 64, True),
    ),
)
def test_engine_automatic_high_split_cuda_graph(
    num_heads, kv_rows, top_k, expected_split, force_repair
):
    """Public decode planning exercises every pooled split and exact repair."""

    inputs = _inputs(q_rows=1, kv_rows=kv_rows, num_heads=num_heads, head_dim=128)
    q, kv, scale, weights, cu_q, cu_kv = inputs
    if force_repair:
        q.zero_()
        kv.zero_()
        scale.fill_(1.0)
    kv_fused, block_table = _paged_inputs(inputs)
    oracle = _oracle_rows(inputs)

    wrapper = SelectiveEngineCase()
    wrapper.plan(
        q,
        kv_fused,
        weights,
        cu_q,
        cu_kv,
        block_table,
        top_k=top_k,
        num_heads=num_heads,
        head_dim=128,
        num_sms=148,
    )
    assert wrapper._candidate_schedule.split_kv == expected_split
    assert wrapper._candidate_schedule.publication == "warp_pooled"
    assert wrapper._candidate_counts.shape[1] == expected_split + 1
    if num_heads == 8:
        assert wrapper._candidate_schedule.logical_n == 8
        for prepared in (
            wrapper._candidate_prepared,
            wrapper._sample_prepared,
            wrapper._repair_prepared,
            wrapper._prepared,
        ):
            assert prepared.query_tile == 1
            assert prepared.num_epi_subtiles == 2

    eager = wrapper.run(q, kv_fused, weights, block_table)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = wrapper.run(q, kv_fused, weights, block_table)
    graph.replay()
    torch.cuda.synchronize()

    assert eager.data_ptr() == captured.data_ptr()
    assert int(wrapper._exact_error.item()) == 0
    if force_repair:
        assert (wrapper._repair_flags != 0).all()
    indices = captured[0].long()
    scores = oracle[0]
    assert ((indices >= 0) & (indices < scores.numel())).all()
    assert indices.unique().numel() == top_k
    torch.testing.assert_close(
        scores[indices].sort().values,
        torch.topk(scores, top_k).values.sort().values,
        atol=5e-5,
        rtol=1e-5,
    )


@_requires_selective_sm100
@pytest.mark.parametrize("page_size", (32, 64, 128))
@pytest.mark.parametrize("math_mode", ("relu_strict", "relu_fast"))
def test_engine_decode_uses_n8_request_tile(page_size, math_mode):
    """Packed H8 Q1 requests use unpadded N8 in eager and graph execution."""

    batches = 4
    kv_per_request = 2048
    inputs = _request_inputs((1,) * batches, (kv_per_request,) * batches)
    q, kv, scale, weights, cu_q, cu_kv = inputs
    kv_fused, block_table = _paged_inputs(inputs, block_size=page_size)
    oracle = _oracle_rows(inputs)

    wrapper = SelectiveEngineCase(math_mode=math_mode)
    wrapper.plan(
        q,
        kv_fused,
        weights,
        cu_q,
        cu_kv,
        block_table,
        num_sms=4,
    )
    assert wrapper._candidate_schedule.logical_n == 8
    for prepared in (
        wrapper._candidate_prepared,
        wrapper._sample_prepared,
        wrapper._repair_prepared,
        wrapper._prepared,
    ):
        assert prepared.query_tile == 1
        assert prepared.num_epi_subtiles == 2
        assert prepared.num_umma_stages == 2
    eager = wrapper.run(q, kv_fused, weights, block_table).clone().long()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = wrapper.run(q, kv_fused, weights, block_table)
    graph.replay()
    torch.cuda.synchronize()

    assert int(wrapper._exact_error.item()) == 0
    for selected in (eager, captured.long()):
        _check_selection(selected, oracle, 512)


@_requires_selective_sm100
def test_engine_split_kv_handles_heterogeneous_batch():
    """Ragged request costs share one automatic candidate publication plan."""

    inputs = _request_inputs((16, 48), (2048, 4096))
    q, kv, scale, weights, cu_q, cu_kv = inputs
    kv_fused, block_table = _paged_inputs(inputs)
    oracle = _oracle_rows(inputs)

    wrapper = SelectiveEngineCase()
    wrapper.plan(
        q,
        kv_fused,
        weights,
        cu_q,
        cu_kv,
        block_table,
        num_sms=3,
    )
    assert wrapper._candidate_prepared.split_kv == 2
    assert wrapper._candidate_schedule.publication == "lane_local"
    assert wrapper._candidate_counts.shape[1] == 65
    selected = wrapper.run(q, kv_fused, weights, block_table).long()
    torch.cuda.synchronize()

    assert int(wrapper._exact_error.item()) == 0
    _check_selection(selected, oracle, 512)


@_requires_selective_sm100
def test_engine_q2_segmented_overflow_repair_cuda_graph():
    """Inactive store warps retain lifecycle while Q2 repairs exact overflow."""

    q_rows = 3
    kv_rows = 131072
    num_heads = 64
    q, kv, scale, weights, _cu_q, _cu_kv = _inputs(
        q_rows=q_rows, kv_rows=kv_rows, num_heads=num_heads, head_dim=128
    )
    q.zero_()
    kv.zero_()
    scale.fill_(1.0)
    cu_q = torch.tensor([0, q_rows], device="cuda", dtype=torch.int32)
    cu_kv = torch.tensor([0, kv_rows], device="cuda", dtype=torch.int32)
    kv_fused = _fuse_paged_kv(kv, scale)
    block_table = torch.arange(
        kv_rows // 128 - 1, -1, -1, device="cuda", dtype=torch.int32
    ).unsqueeze(0)

    wrapper = SelectiveEngineCase()
    wrapper.plan(
        q,
        kv_fused,
        weights,
        cu_q,
        cu_kv,
        block_table,
        num_heads=num_heads,
        head_dim=128,
        num_sms=4,
    )
    eager = wrapper.run(q, kv_fused, weights, block_table)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = wrapper.run(q, kv_fused, weights, block_table)
    graph.replay()
    torch.cuda.synchronize()

    assert eager.data_ptr() == captured.data_ptr()
    assert (wrapper._repair_flags != 0).all()
    assert int(wrapper._exact_error.item()) == 0
    assert (wrapper._candidate_counts[q_rows:] == 0).all()
    row_ends = torch.arange(
        kv_rows - q_rows + 1, kv_rows + 1, device="cuda", dtype=torch.int32
    )
    selected = captured
    assert ((selected >= 0) & (selected < row_ends[:, None])).all()
    assert all(row.unique().numel() == 512 for row in selected)


@_requires_selective_sm100
@pytest.mark.parametrize("force_ties", [False, True])
@pytest.mark.parametrize("math_mode", ["relu_fast", "relu_strict", None])
def test_engine_sampled_common_and_exact_repair(force_ties, math_mode):
    """The public run owns sampling and fail-closed repair on arbitrary pages."""
    from flashinfer import fp8_paged_mqa_logits

    q, kv, scale, weights, _cu_q, _cu_kv = _inputs(q_rows=16, kv_rows=8192)
    if force_ties:
        q.zero_()
        kv.zero_()
        scale.fill_(1.0)
    cu_q = torch.tensor([0, q.shape[0]], device="cuda", dtype=torch.int32)
    cu_kv = torch.tensor([0, kv.shape[0]], device="cuda", dtype=torch.int32)
    inputs = (q, kv, scale, weights, cu_q, cu_kv)
    kv_fused, block_table = _paged_inputs(inputs)
    ordinary = fp8_paged_mqa_logits(
        q.unsqueeze(0),
        kv_fused,
        weights,
        block_table,
        torch.tensor([kv.shape[0]], device="cuda", dtype=torch.int32),
        kv.shape[0],
        **({} if math_mode is None else {"math_mode": math_mode}),
    )

    wrapper = SelectiveEngineCase(
        **{} if math_mode is None else {"math_mode": math_mode}
    )
    assert wrapper.math_mode == ("relu_strict" if math_mode is None else math_mode)
    wrapper.plan(q, kv_fused, weights, cu_q, cu_kv, block_table, num_sms=4)
    selected = wrapper.run(q, kv_fused, weights, block_table)
    torch.cuda.synchronize()

    assert int(wrapper._exact_error.item()) == 0
    if force_ties:
        assert (wrapper._repair_flags != 0).all()
    for row, valid in enumerate(_valid_lengths(cu_q, cu_kv)):
        actual_indices = selected[row].long()
        assert ((actual_indices >= 0) & (actual_indices < valid)).all()
        assert actual_indices.unique().numel() == 512
        actual_scores = ordinary[row, actual_indices]
        expected_scores = torch.topk(ordinary[row, :valid], 512).values
        torch.testing.assert_close(
            actual_scores.sort().values,
            expected_scores.sort().values,
            atol=5e-5,
            rtol=1e-5,
        )


@_requires_selective_sm100
@pytest.mark.parametrize("ties", [True, False], ids=["tied", "distinct"])
@pytest.mark.parametrize("q_lengths", [[2], [2, 2]], ids=["compact", "standard"])
def test_unified_wrapper_exact_repair_negative_threshold(q_lengths, ties):
    """Exact repair keys a negative TopK threshold in both repair layouts."""

    # Positive dots with negative weights give negative scores. Tied rows
    # overflow both bounded passes; varying scales also tests strict ordering.
    inputs = _request_inputs(q_lengths, [9001] * len(q_lengths))
    q, kv, scale, weights, cu_q, cu_kv = inputs
    q.copy_(torch.ones(q.shape, device=q.device).to(q.dtype))
    kv.copy_(torch.ones(kv.shape, device=kv.device).to(kv.dtype))
    scale.fill_(1.0)
    if not ties:
        # Distinct negative scores exercise ordering, not just sign mapping.
        scale.copy_(torch.linspace(1.0, 2.0, scale.numel(), device=scale.device))
    weights.fill_(-1.0)
    kv_fused, block_table = _paged_inputs(inputs)

    wrapper = FP8PagedMQATopKWrapper(
        strategy="selective",
        max_context_length=9001,
        max_chunk_size=q.shape[0],
        max_batch_size=len(q_lengths),
        kv_layout=kv_fused,
    )
    wrapper.prepare_pool(kv_fused)
    engine = wrapper._capacity_plan._engine
    assert engine._compact_repair == (len(q_lengths) == 1)
    wrapper.plan(cu_q, cu_kv, block_table, num_query_rows=q.shape[0])
    selected = wrapper.run(q, weights, kv_fused)
    torch.cuda.synchronize()

    assert wrapper.error.item() == wrapper.repair_error.item() == 0
    if ties:
        assert int(engine._second_repair_flags.count_nonzero()) == q.shape[0]
    expected = _oracle_rows(inputs)
    assert all(bool((row < 0).all()) for row in expected)
    _check_selection(selected, expected, 512)
    if not ties:
        outside = torch.topk(expected[0], 513).indices[-1]
        corrupted = selected.clone()
        corrupted[0, 0] = outside
        with pytest.raises(AssertionError):
            _check_selection(corrupted, expected, 512)


def _nan_score_inputs(q_lengths, kv_lengths, heads=8):
    inputs = _request_inputs(q_lengths, kv_lengths, heads=heads)
    q, kv, scale, weights, cu_q, cu_kv = inputs
    q.copy_(torch.ones(q.shape, device=q.device).to(q.dtype))
    kv.copy_(torch.ones(kv.shape, device=kv.device).to(kv.dtype))
    scale.fill_(float("nan"))
    weights.fill_(-1.0)
    weights[0].fill_(1.0)
    for start in cu_kv[:-1].tolist():
        scale[start : start + 257] = torch.linspace(1.0, 2.0, 257, device=q.device)
        scale[start] = float("inf")
    for end in cu_q[1:].tolist():
        weights[end - 1].fill_(float("nan"))
    return inputs


def _nan_as_negative_infinity_rows(inputs):
    # Preserve genuine +/-inf; nan_to_num's defaults would also clamp them.
    return [
        torch.where(torch.isnan(scores), float("-inf"), scores)
        for scores in _oracle_rows(inputs)
    ]


@_requires_selective_sm100
@pytest.mark.parametrize("math_mode", ["relu_fast", "relu_strict"])
@pytest.mark.parametrize("q_lengths", [[3], [3, 3]], ids=["compact", "standard"])
def test_unified_wrapper_nan_scores_rank_as_negative_infinity(q_lengths, math_mode):
    """Sampling, publication and exact repair agree on mixed/all-NaN rows."""
    inputs = _nan_score_inputs(q_lengths, [9001] * len(q_lengths))
    q, _, _, weights, cu_q, cu_kv = inputs
    fused, table = _paged_inputs(inputs)
    wrapper = FP8PagedMQATopKWrapper(
        strategy="selective",
        math_mode=math_mode,
        max_context_length=9001,
        max_chunk_size=q.shape[0],
        max_batch_size=len(q_lengths),
        kv_layout=fused,
    )
    wrapper.prepare_pool(fused)
    engine = wrapper._capacity_plan._engine
    assert engine._compact_repair == (len(q_lengths) == 1)
    assert engine._sample_prepared is not None
    expected = _nan_as_negative_infinity_rows(inputs)
    assert torch.isposinf(expected[0]).any()
    assert torch.isneginf(expected[-1]).all()

    def check(selected):
        torch.cuda.synchronize()
        assert wrapper.error.item() == wrapper.repair_error.item() == 0
        # Fewer than K finite keys force -inf ties into both bounded arenas.
        assert engine._second_repair_flags.count_nonzero().item() == q.shape[0]
        assert not torch.isnan(engine._sample_scores).any()
        _check_selection(selected, expected, 512)

    wrapper.plan(cu_q, cu_kv, table, num_query_rows=q.shape[0])
    check(wrapper.run(q, weights, fused))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.plan(cu_q, cu_kv, table, num_query_rows=q.shape[0])
        captured = wrapper.run(q, weights, fused)
    graph.replay()
    check(captured)


@_requires_selective_sm100
@pytest.mark.parametrize("math_mode", ["relu_fast", "relu_strict"])
def test_h64_candidate_nan_scores_rank_as_negative_infinity(math_mode):
    """The row-per-warpgroup publisher shares the generic NaN ordering."""
    inputs = _nan_score_inputs([5], [16384], heads=64)
    q, _, _, weights, cu_q, cu_kv = inputs
    fused, table = _paged_inputs(inputs)
    wrapper = SelectiveEngineCase(math_mode=math_mode)
    wrapper.plan(q, fused, weights, cu_q, cu_kv, table, num_heads=64, num_sms=2)
    assert wrapper._candidate_schedule.h64_candidate
    expected = _nan_as_negative_infinity_rows(inputs)

    def check(selected):
        torch.cuda.synchronize()
        assert wrapper._exact_error.item() == 0
        assert wrapper._second_repair_flags.count_nonzero().item() == q.shape[0]
        assert not torch.isnan(wrapper._sample_scores).any()
        _check_selection(selected, expected, 512)

    check(wrapper.run(q, fused, weights, table))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = wrapper.run(q, fused, weights, table)
    graph.replay()
    check(captured)


@_requires_selective_sm100
@pytest.mark.parametrize("math_mode", ["relu_fast", "relu_strict"])
def test_sample_candidate_overflow_and_exact_modes(math_mode):
    from flashinfer.experimental.attn_scores.selective_logits import (
        _finalize_fp8_mqa_packed_candidates,
    )

    inputs = _inputs()
    q, kv, scale, weights, cu_q, cu_kv = inputs
    immutable = tuple(tensor.clone() for tensor in (q, kv, scale, weights))
    prepared = _prepare(inputs, math_mode=math_mode)
    rows = q.shape[0]
    width = 257
    sampled = torch.full((32, width), float("nan"), device="cuda")
    unused = torch.empty((32, width + 1), device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError, match="_SelectiveLogitsMode"):
        _launch("sample", width, sampled, unused, prepared, inputs)
    _launch(_SelectiveLogitsMode.SAMPLE, width, sampled, unused, prepared, inputs)
    assert prepared._launches
    assert len(prepared._launches) == 1
    cache_key = next(iter(prepared._launches))
    cached_sample_launch = prepared._launches[cache_key]
    unused.zero_()
    _launch(_SelectiveLogitsMode.SAMPLE, width, sampled, unused, prepared, inputs)
    assert prepared._launches[cache_key] is cached_sample_launch
    torch.cuda.synchronize()

    valid_lengths = _valid_lengths(cu_q, cu_kv)
    assert valid_lengths[0] == 132
    assert valid_lengths[-1] == 255
    assert torch.isfinite(sampled[0, :132]).all()
    assert torch.isfinite(sampled[rows - 1, :255]).all()
    assert torch.isnan(sampled[0, 132:]).all()
    for row, expected in enumerate(_oracle_rows(inputs)):
        torch.testing.assert_close(
            sampled[row, : expected.numel()], expected, atol=5e-5, rtol=1e-5
        )

    thresholds = torch.stack(
        [sampled[row, valid_lengths[row] // 2] for row in range(rows)]
    )
    capacity = width
    values = torch.empty((32, capacity), device="cuda", dtype=torch.int64)
    index_counts = torch.empty((32, 1), device="cuda", dtype=torch.int32)
    _launch(
        _SelectiveLogitsMode.CANDIDATE,
        capacity,
        values,
        index_counts,
        prepared,
        inputs,
        thresholds,
    )
    torch.cuda.synchronize()
    candidate_cache_key = next(
        key for key in prepared._launches if key[0] == _SelectiveLogitsMode.CANDIDATE
    )
    cached_candidate_launch = prepared._launches[candidate_cache_key]
    thresholds.add_(0.0)
    _launch(
        _SelectiveLogitsMode.CANDIDATE,
        capacity,
        values,
        index_counts,
        prepared,
        inputs,
        thresholds,
    )
    assert prepared._launches[candidate_cache_key] is cached_candidate_launch
    torch.cuda.synchronize()
    for row, valid in enumerate(valid_lengths):
        expected = torch.nonzero(sampled[row, :valid] >= thresholds[row]).flatten()
        count = int(index_counts[row, 0])
        packed = values[row, :count]
        actual_indices = (packed & 0xFFFFFFFF).to(torch.int32)
        order = actual_indices.argsort()
        actual_indices = actual_indices[order]
        assert count == expected.numel(), (
            f"row {row}: published {count}, expected {expected.numel()}, "
            f"unique={actual_indices.unique().numel()}"
        )
        torch.testing.assert_close(
            actual_indices, expected.to(torch.int32).sort().values
        )
        actual_score_bits = ((packed[order] >> 32) & 0xFFFFFFFF).to(torch.int32)
        expected_score_bits = sampled[row, actual_indices.long()].view(torch.int32)
        torch.testing.assert_close(actual_score_bits, expected_score_bits)

    tiny_values = torch.full((32, 5), -7, device="cuda", dtype=torch.int64)
    tiny_indices = torch.full((32, 1), -11, device="cuda", dtype=torch.int32)
    _launch(
        _SelectiveLogitsMode.REPAIR,
        3,
        tiny_values,
        tiny_indices,
        prepared,
        inputs,
        thresholds,
    )
    torch.cuda.synchronize()
    assert torch.equal(tiny_indices[:rows, 0], index_counts[:rows, 0])
    assert (tiny_indices[:rows, 0] > 3).any()
    assert (tiny_values[:, 3:] == -7).all()
    lengths = torch.full((rows + 2,), -13, device="cuda", dtype=torch.int32)
    repair_flags = torch.full((rows + 2,), -17, device="cuda", dtype=torch.int32)
    _finalize_fp8_mqa_packed_candidates(
        tiny_indices,
        tiny_values,
        lengths,
        repair_flags,
        rows=rows,
        capacity=3,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(lengths[:rows], tiny_indices[:rows, 0].clamp(max=3))
    assert (repair_flags[:rows] == 1).all()
    assert (lengths[rows:] == -13).all()
    assert (repair_flags[rows:] == -17).all()

    # Bank 0 holds > T and bank 1 holds == T. Banks taller than the 32
    # padded rows check the epilogue's `out_row + bank_rows` addressing for
    # every row and leave both padded tails untouched.
    bank_rows = 40
    exact_values = torch.full(
        (2 * bank_rows, width), -7, device="cuda", dtype=torch.int32
    )
    exact_counts = torch.empty((2 * bank_rows, 1), device="cuda", dtype=torch.int32)
    _launch(
        _SelectiveLogitsMode.EXACT,
        width,
        exact_values,
        exact_counts,
        prepared,
        inputs,
        thresholds,
    )
    torch.cuda.synchronize()
    for bank, compare in ((0, torch.gt), (1, torch.eq)):
        for row, valid in enumerate(valid_lengths):
            expected = torch.nonzero(
                compare(sampled[row, :valid], thresholds[row])
            ).flatten()
            count = int(exact_counts[bank * bank_rows + row, 0])
            actual = exact_values[bank * bank_rows + row, :count].sort().values
            torch.testing.assert_close(actual, expected.to(torch.int32).sort().values)
        tail = slice(bank * bank_rows + rows, (bank + 1) * bank_rows)
        assert (exact_counts[tail] == 0).all()
        assert (exact_values[tail] == -7).all()

    with pytest.raises(ValueError, match="too narrow"):
        _launch(
            _SelectiveLogitsMode.SAMPLE,
            width,
            sampled,
            torch.empty((32, 0), device="cuda", dtype=torch.int32),
            prepared,
            inputs,
        )
    for before, after in zip(immutable, (q, kv, scale, weights), strict=True):
        torch.testing.assert_close(after, before)


@_requires_selective_sm100
@pytest.mark.parametrize("shift", [24, 16, 8, 0])
@pytest.mark.parametrize("math_mode", ["relu_fast", "relu_strict"])
def test_radix_histogram_matches_sampled_score_bits(shift, math_mode):
    inputs = _inputs(q_rows=1, kv_rows=130)
    q, kv, scale, weights, cu_q, cu_kv = inputs
    prepared = _prepare(inputs, math_mode=math_mode)
    sampled = torch.empty((16, 130), device="cuda", dtype=torch.float32)
    unused = torch.empty((16, 256), device="cuda", dtype=torch.int32)
    _launch(_SelectiveLogitsMode.SAMPLE, 130, sampled, unused, prepared, inputs)
    torch.cuda.synchronize()

    bits = sampled[0].view(torch.int32)
    ordered_i32 = bits ^ torch.where(
        bits < 0,
        torch.full_like(bits, -1),
        torch.full_like(bits, -2147483648),
    )
    ordered = ordered_i32.to(torch.int64) & 0xFFFFFFFF
    if shift == 24:
        prefix = torch.zeros(1, device="cuda", dtype=torch.int32)
        selected = ordered
    else:
        prefix = (ordered[0] >> (shift + 8)).to(torch.int32).reshape(1)
        selected = ordered[(ordered >> (shift + 8)) == prefix[0]]
    expected = torch.bincount(((selected >> shift) & 0xFF).long(), minlength=256)

    hist = torch.empty((16, 256), device="cuda", dtype=torch.int32)
    dummy = torch.empty((16, 1), device="cuda", dtype=torch.int32)
    _launch(
        _SelectiveLogitsMode.RADIX_HISTOGRAM,
        1,
        dummy,
        hist,
        prepared,
        inputs,
        prefix,
        radix_shift=shift,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(hist[0], expected.to(torch.int32))


@_requires_selective_sm100
@pytest.mark.parametrize("num_heads", [8, 64])
@pytest.mark.parametrize(
    "mode", [_SelectiveLogitsMode.SAMPLE, _SelectiveLogitsMode.RADIX_HISTOGRAM]
)
def test_selective_scale_stage_survives_persistent_epilogue(num_heads, mode):
    """Non-publisher epilogues must finish reading scales before stage reuse."""
    q_rows, kv_rows = 33, 32768
    q, kv, scale, weights, _, _ = _inputs(q_rows, kv_rows, num_heads)
    q.copy_(torch.ones(q.shape, device="cuda").to(q.dtype))
    kv.copy_(torch.ones(kv.shape, device="cuda").to(kv.dtype))
    weights.fill_(1)
    # Every K128 tile has a different exact power-of-two scale. A reused
    # stage must not silently substitute a later tile's scale in the epilogue.
    scale.copy_(torch.pow(2.0, (torch.arange(kv_rows, device="cuda") // 128) % 7 - 3))
    cu_q = torch.tensor([0, q_rows], device="cuda", dtype=torch.int32)
    cu_kv = torch.tensor([0, kv_rows], device="cuda", dtype=torch.int32)
    inputs = q, kv, scale, weights, cu_q, cu_kv
    prepared = _prepare(inputs, num_heads=num_heads, num_sms=4)
    reference = torch.matmul(q[0].float(), kv.float().T).relu().sum(0) * scale
    ends = _valid_lengths(cu_q, cu_kv)
    is_sample = mode == _SelectiveLogitsMode.SAMPLE
    capacity = kv_rows if is_sample else 1
    values = torch.empty(
        (48, capacity), device="cuda", dtype=torch.float32 if is_sample else torch.int32
    )
    counts = torch.empty((48, 256), device="cuda", dtype=torch.int32)
    prefixes = torch.zeros(q_rows, device="cuda", dtype=torch.int32)

    def run():
        _launch(
            mode, capacity, values, counts, prepared, inputs, prefixes, radix_shift=24
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(3):
        graph.replay()
        torch.cuda.synchronize()
        for row, end in enumerate(ends):
            if mode == _SelectiveLogitsMode.SAMPLE:
                torch.testing.assert_close(
                    values[row, :end], reference[:end], rtol=0, atol=0
                )
            else:
                keys = reference[:end].view(torch.int32).to(torch.int64) ^ 0x80000000
                expected = torch.bincount((keys >> 24) & 255, minlength=256).int()
                torch.testing.assert_close(counts[row], expected, rtol=0, atol=0)


@_requires_selective_sm100
def test_explicit_nonzero_row_bounds_predicate_every_store():
    inputs = _inputs(q_rows=1, kv_rows=130)
    q, kv, scale, weights, cu_q, cu_kv = inputs
    starts = torch.tensor([17], device="cuda", dtype=torch.int32)
    ends = torch.tensor([119], device="cuda", dtype=torch.int32)
    prepared = _prepare(
        inputs,
        row_starts=starts,
        row_ends=ends,
    )
    sampled = torch.full((16, 130), float("nan"), device="cuda")
    unused = torch.empty((16, 1), device="cuda", dtype=torch.int32)
    _launch(_SelectiveLogitsMode.SAMPLE, 130, sampled, unused, prepared, inputs)
    torch.cuda.synchronize()
    assert torch.isnan(sampled[0, :17]).all()
    assert torch.isfinite(sampled[0, 17:119]).all()
    assert torch.isnan(sampled[0, 119:]).all()


@_requires_selective_sm100
def test_explicit_atomic_modes_split_one_q16_tile_across_ctas():
    from flashinfer.experimental.attn_scores.selective_logits import (
        _finalize_fp8_mqa_packed_candidates,
    )

    inputs = _inputs(q_rows=1, kv_rows=1024)
    q, kv, scale, weights, cu_q, cu_kv = inputs
    starts = torch.zeros(1, device="cuda", dtype=torch.int32)
    ends = torch.full((1,), 1024, device="cuda", dtype=torch.int32)
    prepared = _prepare(
        inputs,
        row_starts=starts,
        row_ends=ends,
        num_sms=4,
        kv_chunk_size=256,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(
        prepared.schedule_meta.cpu(),
        torch.tensor([[0, 0], [0, 1], [0, 2], [0, 3], [1, 0]], dtype=torch.int32),
    )

    capacity = 1024
    values = torch.empty((16, capacity), device="cuda", dtype=torch.int64)
    counts = torch.empty((16, 1), device="cuda", dtype=torch.int32)
    lengths = torch.empty(1, device="cuda", dtype=torch.int32)
    flags = torch.empty(1, device="cuda", dtype=torch.int32)
    threshold = torch.full((1,), float("-inf"), device="cuda")
    _launch(
        _SelectiveLogitsMode.REPAIR,
        capacity,
        values,
        counts,
        prepared,
        inputs,
        threshold,
    )
    _finalize_fp8_mqa_packed_candidates(
        counts,
        values,
        lengths,
        flags,
        rows=1,
        capacity=capacity,
    )
    torch.cuda.synchronize()
    assert int(lengths[0]) == capacity
    actual_indices = (values[0, :capacity] & 0xFFFFFFFF).sort().values
    torch.testing.assert_close(
        actual_indices, torch.arange(capacity, device="cuda", dtype=torch.int64)
    )

    segmented_counts = torch.empty((16, 5), device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError, match="requires exclusive query tiles"):
        _launch(
            _SelectiveLogitsMode.CANDIDATE,
            capacity,
            values,
            segmented_counts,
            prepared,
            inputs,
            threshold,
        )


def _append_candidate_segments(inputs, prepared, threshold, *, capacity, count_columns):
    """Launch the H8 component fixture with fresh, 16-row-padded storage."""
    padded_rows = math.ceil(inputs[0].shape[0] / 16) * 16
    values = torch.empty(
        (padded_rows, capacity), device=inputs[0].device, dtype=torch.int64
    )
    counts = torch.empty(
        (padded_rows, count_columns), device=inputs[0].device, dtype=torch.int32
    )
    _launch(
        _SelectiveLogitsMode.CANDIDATE,
        capacity,
        values,
        counts,
        prepared,
        inputs,
        threshold,
    )
    return values, counts


def _finalize_candidate_segments(values, counts, *, rows):
    from flashinfer.experimental.attn_scores.selective_logits import (
        _finalize_fp8_mqa_packed_candidates,
    )

    lengths = torch.empty(rows, device=values.device, dtype=torch.int32)
    flags = torch.empty(rows, device=values.device, dtype=torch.int32)
    _finalize_fp8_mqa_packed_candidates(
        counts, values, lengths, flags, rows=rows, capacity=values.shape[1]
    )
    torch.cuda.synchronize()
    return lengths, flags


@_requires_selective_sm100
@pytest.mark.parametrize(
    "split_kv,q_rows,kv_rows,capacity,overflow_capacity",
    [(2, 17, 1025, 6144, 256), (4, 17, 1025, 8192, 512), (4, 1, 641, 8192, None)],
)
def test_split_kv_uses_disjoint_segments(
    split_kv, q_rows, kv_rows, capacity, overflow_capacity
):
    inputs = _inputs(q_rows=q_rows, kv_rows=kv_rows)
    prepared = _prepare(inputs, num_sms=4, split_kv=split_kv)
    padded_rows = math.ceil(q_rows / 16) * 16
    columns = 4 * split_kv + 1
    threshold = torch.full((q_rows,), float("-inf"), device="cuda")
    expected = torch.tensor(
        _valid_lengths(inputs[4], inputs[5]), device="cuda", dtype=torch.int32
    )

    def append_and_finalize(capacity):
        values, counts = _append_candidate_segments(
            inputs, prepared, threshold, capacity=capacity, count_columns=columns
        )
        raw_counts = counts[:q_rows, 1:].clone()
        lengths, flags = _finalize_candidate_segments(values, counts, rows=q_rows)
        torch.testing.assert_close(raw_counts.sum(dim=1, dtype=torch.int32), expected)
        return values, counts, lengths, flags, raw_counts

    values, counts, lengths, flags, raw_counts = append_and_finalize(capacity)
    torch.testing.assert_close(lengths, expected)
    # The first request from _inputs has only 132 keys, below the default K512.
    # Complete short rows still require the finalizer's underfill flag.
    torch.testing.assert_close(flags, (expected < 512).to(torch.int32))
    owner_counts = raw_counts.reshape(q_rows, split_kv, 4).sum(dim=2)
    for row, length in enumerate(expected.tolist()):
        active_owners = min(math.ceil(length / 256), split_kv)
        assert (owner_counts[row, :active_owners] > 0).all()
        assert (owner_counts[row, active_owners:] == 0).all()
        actual = (values[row, :length] & 0xFFFFFFFF).sort().values
        torch.testing.assert_close(
            actual, torch.arange(length, device="cuda", dtype=torch.int64)
        )

    wrong_counts = torch.empty(
        (padded_rows, 2 * split_kv + 1), device="cuda", dtype=torch.int32
    )
    with pytest.raises(ValueError, match=f"requires Qx{columns}"):
        _launch(
            _SelectiveLogitsMode.CANDIDATE,
            capacity,
            values,
            wrong_counts,
            prepared,
            inputs,
            threshold,
        )

    if overflow_capacity is None:
        return  # The 641-key case specifically checks three unequal active owners.
    _, _, lengths, flags, _ = append_and_finalize(overflow_capacity)
    per_owner = overflow_capacity // split_kv
    retained = []
    for length in expected.tolist():
        pairs = math.ceil(length / 256)
        owners = min(pairs, split_kv)
        total = 0
        for owner in range(owners):
            first = math.ceil(owner * pairs / owners) * 256
            last = min(math.ceil((owner + 1) * pairs / owners) * 256, length)
            total += min(max(last - first, 0), per_owner)
        retained.append(total)
    torch.testing.assert_close(
        lengths, torch.tensor(retained, device="cuda", dtype=torch.int32)
    )
    assert (flags == 1).all()


@_requires_selective_sm100
def test_split_kv_eight_uses_disjoint_qx33_warp_segments():
    inputs = _inputs(q_rows=1, kv_rows=2049)
    _q, _kv, _scale, _weights, cu_q, cu_kv = inputs
    prepared = _prepare(inputs, num_sms=8, split_kv=8)
    capacity = 6144
    threshold = torch.full((1,), float("-inf"), device="cuda")
    values, counts = _append_candidate_segments(
        inputs, prepared, threshold, capacity=capacity, count_columns=33
    )
    raw_counts = counts[0, 1:].clone()
    lengths, _flags = _finalize_candidate_segments(values, counts, rows=1)

    expected_length = _valid_lengths(cu_q, cu_kv)[0]
    assert int(raw_counts.sum()) == expected_length
    assert all(
        int(raw_counts[split_idx * 4 : (split_idx + 1) * 4].sum()) > 0
        for split_idx in range(8)
    )
    assert int(lengths[0]) == expected_length
    actual_indices = (values[0, :expected_length] & 0xFFFFFFFF).sort().values
    torch.testing.assert_close(
        actual_indices,
        torch.arange(expected_length, device="cuda", dtype=torch.int64),
    )


@_requires_selective_sm100
@pytest.mark.parametrize("split_kv", (16, 32, 64))
def test_high_split_kv_pools_one_ballot_segment_per_owner(split_kv: int):
    inputs = _inputs(q_rows=1, kv_rows=16385)
    prepared = _prepare(inputs, num_sms=64, split_kv=split_kv)
    capacity = 32768
    threshold = torch.full((1,), float("-inf"), device="cuda")
    values, counts = _append_candidate_segments(
        inputs, prepared, threshold, capacity=capacity, count_columns=split_kv + 1
    )
    torch.cuda.synchronize()

    raw_counts = counts[0, 1:].to(torch.int64)
    assert (raw_counts > 0).all()
    assert int(raw_counts.sum()) == 16385
    segment_capacity = capacity // split_kv
    published = torch.cat(
        [
            values[0, segment * segment_capacity :][: int(raw_counts[segment].item())]
            for segment in range(split_kv)
        ]
    )
    actual_indices = (published & 0xFFFFFFFF).sort().values
    torch.testing.assert_close(
        actual_indices,
        torch.arange(16385, device="cuda", dtype=torch.int64),
    )


@_requires_selective_sm100
def test_split_kv_eight_resets_warp_cursors_between_persistent_work_items():
    batches = 32
    kv_per_request = 2049
    q, kv, scale, weights, _cu_q, _cu_kv = _inputs(
        q_rows=batches, kv_rows=batches * kv_per_request
    )
    cu_q = torch.arange(batches + 1, device="cuda", dtype=torch.int32)
    cu_kv = cu_q * kv_per_request
    inputs = (q, kv, scale, weights, cu_q, cu_kv)
    prepared = _prepare(inputs, num_sms=16, split_kv=8)
    capacity = 6144
    threshold = torch.full((batches,), float("-inf"), device="cuda")
    values, counts = _append_candidate_segments(
        inputs, prepared, threshold, capacity=capacity, count_columns=33
    )
    lengths, _flags = _finalize_candidate_segments(values, counts, rows=batches)

    assert (lengths == kv_per_request).all()
    expected = torch.arange(kv_per_request, device="cuda", dtype=torch.int64)
    for row in range(batches):
        actual = (values[row, :kv_per_request] & 0xFFFFFFFF).sort().values
        torch.testing.assert_close(actual, expected)


@_requires_selective_sm100
def test_all_partial_q16_request_tails_and_packed_origins():
    torch.manual_seed(2)
    q_lengths = torch.arange(1, 16, device="cuda", dtype=torch.int32)
    kv_lengths = torch.full_like(q_lengths, 260)
    kv_lengths[-1] = 257
    cu_q = torch.cat((torch.zeros(1, device="cuda", dtype=torch.int32), q_lengths))
    cu_kv = torch.cat((torch.zeros(1, device="cuda", dtype=torch.int32), kv_lengths))
    cu_q.cumsum_(0)
    cu_kv.cumsum_(0)
    rows = int(q_lengths.sum())
    kv_rows = int(kv_lengths.sum())
    q = (torch.randn(rows, 8, 128, device="cuda") * 0.1).to(torch.float8_e4m3fn)
    kv_f32 = torch.randn(kv_rows, 128, device="cuda") * 0.1
    amax = kv_f32.abs().amax(dim=1).clamp_min(1e-6)
    scale = torch.pow(2.0, torch.ceil(torch.log2(amax / 448.0)))
    kv = (kv_f32 / scale[:, None]).to(torch.float8_e4m3fn)
    weights = torch.randn(rows, 8, device="cuda", dtype=torch.float32)
    inputs = (q, kv, scale, weights, cu_q, cu_kv)
    prepared = _prepare(inputs)
    sampled = torch.full((128, 260), float("nan"), device="cuda")
    counts = torch.empty((128, 1), device="cuda", dtype=torch.int32)
    _launch(_SelectiveLogitsMode.SAMPLE, 260, sampled, counts, prepared, inputs)
    torch.cuda.synchronize()

    for row, expected in enumerate(_oracle_rows(inputs)):
        torch.testing.assert_close(
            sampled[row, : expected.numel()], expected, atol=5e-5, rtol=1e-5
        )
        assert torch.isnan(sampled[row, expected.numel() :]).all()


def _capture_refreshed_sample(
    prepared, inputs, sampled, counts, *, row_starts=None, row_ends=None
):
    from flashinfer.experimental.attn_scores.selective_logits import (
        _refresh_fp8_mqa_selective_logits,
    )

    cu_q, cu_kv = inputs[4:]
    capacity = sampled.shape[1]
    _launch(_SelectiveLogitsMode.SAMPLE, capacity, sampled, counts, prepared, inputs)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        sampled.fill_(float("nan"))
        _refresh_fp8_mqa_selective_logits(
            prepared, cu_q, cu_kv, row_starts=row_starts, row_ends=row_ends
        )
        _launch(
            _SelectiveLogitsMode.SAMPLE, capacity, sampled, counts, prepared, inputs
        )
    return graph


@_requires_selective_sm100
def test_cuda_graph_refreshes_explicit_bounds_without_reallocation():
    inputs = _inputs(q_rows=1, kv_rows=130)
    starts = torch.tensor([17], device="cuda", dtype=torch.int32)
    ends = torch.tensor([119], device="cuda", dtype=torch.int32)
    prepared = _prepare(
        inputs,
        row_starts=starts,
        row_ends=ends,
    )
    sampled = torch.empty((16, 130), device="cuda", dtype=torch.float32)
    unused = torch.empty((16, 1), device="cuda", dtype=torch.int32)
    graph = _capture_refreshed_sample(
        prepared, inputs, sampled, unused, row_starts=starts, row_ends=ends
    )

    starts.fill_(23)
    ends.fill_(77)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.isnan(sampled[0, :23]).all()
    assert torch.isfinite(sampled[0, 23:77]).all()
    assert torch.isnan(sampled[0, 77:]).all()


@_requires_selective_sm100
def test_cuda_graph_refreshes_row_end_base_and_inactive_tail():
    inputs = _inputs(q_rows=2, kv_rows=130)
    starts = torch.zeros(2, device="cuda", dtype=torch.int32)
    ends = torch.tensor([1, 0], device="cuda", dtype=torch.int32)
    prepared = _prepare(
        inputs,
        row_starts=starts,
        row_ends=ends,
        row_end_base=64,
    )
    sampled = torch.empty((16, 130), device="cuda", dtype=torch.float32)
    unused = torch.empty((16, 1), device="cuda", dtype=torch.int32)
    graph = _capture_refreshed_sample(
        prepared, inputs, sampled, unused, row_starts=starts, row_ends=ends
    )

    ends[0] = 0
    ends[1] = 3
    graph.replay()
    torch.cuda.synchronize()
    assert torch.isnan(sampled[0]).all()
    assert torch.isfinite(sampled[1, :67]).all()
    assert torch.isnan(sampled[1, 67:]).all()


@_requires_selective_sm100
def test_cuda_graph_refreshes_request_prefixes_without_reallocation():
    inputs = _inputs()
    cu_q = inputs[4]
    prepared = _prepare(inputs)
    sampled = torch.empty((32, 257), device="cuda", dtype=torch.float32)
    counts = torch.empty((32, 1), device="cuda", dtype=torch.int32)
    graph = _capture_refreshed_sample(prepared, inputs, sampled, counts)

    cu_q[1] = 2
    graph.replay()
    torch.cuda.synchronize()
    for row, expected in enumerate(_oracle_rows(inputs)):
        torch.testing.assert_close(
            sampled[row, : expected.numel()], expected, atol=5e-5, rtol=1e-5
        )
        assert torch.isnan(sampled[row, expected.numel() :]).all()


@_requires_selective_sm100
def test_zero_ties_use_bit_exact_equal_policy():
    inputs = list(_inputs(q_rows=1, kv_rows=130))
    inputs[3].zero_()
    inputs = tuple(inputs)
    q, kv, scale, weights, cu_q, cu_kv = inputs
    prepared = _prepare(inputs)
    bank_rows = 16
    values = torch.empty((2 * bank_rows, 130), device="cuda", dtype=torch.int32)
    counts = torch.empty((2 * bank_rows, 1), device="cuda", dtype=torch.int32)
    positive_zero = torch.zeros(1, device="cuda", dtype=torch.float32)
    _launch(
        _SelectiveLogitsMode.EXACT,
        130,
        values,
        counts,
        prepared,
        inputs,
        positive_zero,
    )
    torch.cuda.synchronize()
    assert int(counts[bank_rows, 0]) == 130
    assert int(counts[0, 0]) == 0

    # -0.0 keys strictly below +0.0, so no score ties it; the greater bank
    # holds every score whose ordered key exceeds the -0.0 key.
    negative_zero = torch.full((1,), -0.0, device="cuda", dtype=torch.float32)
    _launch(
        _SelectiveLogitsMode.EXACT,
        130,
        values,
        counts,
        prepared,
        inputs,
        negative_zero,
    )
    torch.cuda.synchronize()
    assert int(counts[bank_rows, 0]) == 0

    def ordered_keys(scores):
        bits = scores.view(torch.int32).long() & 0xFFFFFFFF
        return torch.where(bits >= 1 << 31, bits ^ 0xFFFFFFFF, bits ^ (1 << 31))

    scores = _oracle_rows(inputs)[0]
    expected = torch.nonzero(
        ordered_keys(scores) > ordered_keys(negative_zero.to(scores.device))
    ).flatten()
    count = int(counts[0, 0])
    actual = values[0, :count].sort().values
    torch.testing.assert_close(actual, expected.to(torch.int32).sort().values)


@_requires_selective_sm100
def test_cuda_graph_replays_into_caller_owned_buffers():
    inputs = _inputs(q_rows=1, kv_rows=130)
    q, kv, scale, weights, cu_q, cu_kv = inputs
    prepared = _prepare(inputs)
    threshold = torch.zeros(1, device="cuda", dtype=torch.float32)
    values = torch.empty((16, 130), device="cuda", dtype=torch.int64)
    index_counts = torch.empty((16, 1), device="cuda", dtype=torch.int32)
    _launch(
        _SelectiveLogitsMode.CANDIDATE,
        130,
        values,
        index_counts,
        prepared,
        inputs,
        threshold,
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _launch(
            _SelectiveLogitsMode.CANDIDATE,
            130,
            values,
            index_counts,
            prepared,
            inputs,
            threshold,
        )
    first = None
    for _ in range(3):
        graph.replay()
        torch.cuda.synchronize()
        count = int(index_counts[0, 0])
        current = values[0, :count].clone()
        if first is None:
            first = current
        else:
            torch.testing.assert_close(current.sort().values, first.sort().values)
    threshold.fill_(float("inf"))
    graph.replay()
    torch.cuda.synchronize()
    assert int(index_counts[0, 0]) == 0
    threshold.zero_()
    graph.replay()
    torch.cuda.synchronize()
    restored_count = int(index_counts[0, 0])
    torch.testing.assert_close(
        values[0, :restored_count].sort().values,
        first.sort().values,
    )


@_requires_selective_sm100
@pytest.mark.parametrize(
    "metadata_name", ["context_lens", "tile_meta", "schedule_meta"]
)
def test_cached_launch_rebinds_prepared_metadata_storage(metadata_name):
    from flashinfer.experimental.attn_scores.selective_logits import (
        _refresh_fp8_mqa_selective_logits,
    )

    inputs = _inputs(q_rows=1, kv_rows=130)
    q, kv, scale, weights, cu_q, cu_kv = inputs
    prepared = _prepare(inputs)
    values = torch.empty((16, 130), device="cuda", dtype=torch.float32)
    counts = torch.empty((16, 1), device="cuda", dtype=torch.int32)

    _launch(_SelectiveLogitsMode.SAMPLE, 130, values, counts, prepared, inputs)
    torch.cuda.synchronize()
    assert len(prepared._launches) == 1
    cache_key = next(iter(prepared._launches))
    cached = prepared._launches[cache_key]

    metadata = getattr(prepared, metadata_name)
    replacement = torch.empty_like(metadata)
    assert replacement.data_ptr() != metadata.data_ptr()
    metadata.set_(replacement)
    _refresh_fp8_mqa_selective_logits(prepared, cu_q, cu_kv)
    _launch(_SelectiveLogitsMode.SAMPLE, 130, values, counts, prepared, inputs)
    torch.cuda.synchronize()

    assert prepared._launches[cache_key] is not cached
    expected = _oracle_rows(inputs)[0]
    torch.testing.assert_close(
        values[0, : expected.numel()], expected, atol=5e-5, rtol=1e-5
    )


@_requires_selective_sm100
def test_segmented_candidate_store_skips_empty_request_with_sparse_scores(monkeypatch):
    """One CTA crosses an empty request through the score-plane store path."""
    q, kv, scale, weights, _, _ = _inputs(q_rows=3, kv_rows=388)
    cu_q = torch.tensor([0, 1, 2, 3], device=q.device, dtype=torch.int32)
    cu_kv = torch.tensor([0, 129, 129, 388], device=q.device, dtype=torch.int32)
    inputs = q, kv, scale, weights, cu_q, cu_kv
    # The empty request is legal only with explicit inactive-row bounds;
    # causal Q-length-one/KV-length-zero would violate the caller contract.
    monkeypatch.setenv("FLASHINFER_VALIDATE_INPUTS", "1")
    starts = torch.zeros(3, device=q.device, dtype=torch.int32)
    ends = torch.tensor([129, 0, 259], device=q.device, dtype=torch.int32)
    prepared = _prepare(inputs, num_sms=1, row_starts=starts, row_ends=ends)
    torch.cuda.synchronize()
    assert prepared.schedule_meta[:, 0].tolist() == [0, prepared.tiles]
    assert prepared.schedule_meta[:, 1].count_nonzero().item() == 0
    oracle = _oracle_rows(inputs)
    assert [row.numel() for row in oracle] == [129, 0, 259]
    padded = math.ceil(3 / prepared.query_tile) * prepared.query_tile
    capacity = 2048
    values = torch.empty((padded, capacity), device=q.device, dtype=torch.int64)
    counts = torch.empty((padded, 33), device=q.device, dtype=torch.int32)
    lengths = torch.empty(3, device=q.device, dtype=torch.int32)
    flags = torch.empty_like(lengths)
    for threshold in (float("-inf"), 0.0, float("inf")):
        thresholds = torch.full((3,), threshold, device=q.device)
        _launch(
            _SelectiveLogitsMode.CANDIDATE,
            capacity,
            values,
            counts,
            prepared,
            inputs,
            thresholds,
        )
        sl._finalize_fp8_mqa_packed_candidates(
            counts, values, lengths, flags, rows=3, capacity=capacity
        )
        torch.cuda.synchronize()
        for row, expected_scores in enumerate(oracle):
            expected = torch.nonzero(expected_scores >= threshold).flatten()
            count = int(lengths[row])
            assert count == expected.numel()
            packed = values[row, :count]
            indices = (packed & 0xFFFFFFFF).long()
            order = indices.argsort()
            torch.testing.assert_close(indices[order], expected)
            actual_scores = (
                ((packed[order] >> 32) & 0xFFFFFFFF).int().view(torch.float32)
            )
            torch.testing.assert_close(
                actual_scores, expected_scores[expected], atol=5e-5, rtol=1e-5
            )
        assert counts[1].count_nonzero().item() == 0
        assert counts[3:].count_nonzero().item() == 0
        if threshold == 0.0:
            assert 0 < int(lengths[0] + lengths[2]) < 388

    # A full Q tile has distinct bounds inside an interior KV pair. Another
    # row keeps the tile context longer, so tail-only masking cannot pass.
    q, kv, scale, weights, _, _ = _inputs(q_rows=18, kv_rows=2050)
    cu_q = torch.tensor([0, 16, 17, 18], device=q.device, dtype=torch.int32)
    cu_kv = torch.tensor([0, 1025, 1025, 2050], device=q.device, dtype=torch.int32)
    inputs = q, kv, scale, weights, cu_q, cu_kv
    starts = torch.tensor(
        [31 + row for row in range(16)] + [0, 129], device=q.device, dtype=torch.int32
    )
    ends = torch.tensor(
        [129 + 32 * row for row in range(16)] + [0, 1025],
        device=q.device,
        dtype=torch.int32,
    )
    oracle = _oracle_rows(inputs)
    thresholds = torch.full((18,), float("-inf"), device=q.device)
    capacity = 6144
    # Lane-local, striped ballot, and pooled ballot share the changed bounded
    # append sites; no algorithm or tuning choice varies within a layout.
    for segments, split_kv in ((32, 1), (4, 1), (16, 16)):
        prepared = _prepare(
            inputs, num_sms=1, split_kv=split_kv, row_starts=starts, row_ends=ends
        )
        assert prepared.query_tile == 16
        padded = math.ceil(18 / prepared.query_tile) * prepared.query_tile
        values = torch.empty((padded, capacity), device=q.device, dtype=torch.int64)
        counts = torch.empty((padded, segments + 1), device=q.device, dtype=torch.int32)
        lengths = torch.empty(18, device=q.device, dtype=torch.int32)
        flags = torch.empty_like(lengths)
        _launch(
            _SelectiveLogitsMode.CANDIDATE,
            capacity,
            values,
            counts,
            prepared,
            inputs,
            thresholds,
        )
        raw_counts = counts.clone()
        sl._finalize_fp8_mqa_packed_candidates(
            counts, values, lengths, flags, rows=18, capacity=capacity
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(lengths, ends - starts)
        torch.testing.assert_close(
            raw_counts[:18, 1:].sum(dim=1, dtype=torch.int32), ends - starts
        )
        assert counts[16].count_nonzero().item() == 0
        assert counts[18:].count_nonzero().item() == 0
        for row, expected_scores in enumerate(oracle):
            expected = torch.arange(int(starts[row]), int(ends[row]), device=q.device)
            packed = values[row, : int(lengths[row])]
            indices = (packed & 0xFFFFFFFF).long()
            order = indices.argsort()
            torch.testing.assert_close(indices[order], expected)
            actual_scores = (
                ((packed[order] >> 32) & 0xFFFFFFFF).int().view(torch.float32)
            )
            torch.testing.assert_close(
                actual_scores, expected_scores[expected], atol=5e-5, rtol=1e-5
            )


@_requires_selective_sm100
@pytest.mark.parametrize(("num_segments", "split_kv"), [(32, 1), (64, 2), (128, 4)])
def test_lane_segmented_candidates_preserve_partial_q_and_tail_rows(
    num_segments, split_kv
):
    from flashinfer.experimental.attn_scores.selective_logits import (
        _finalize_fp8_mqa_packed_candidates,
    )

    inputs = _inputs(q_rows=17, kv_rows=1025)
    q, kv, scale, weights, cu_q, cu_kv = inputs
    prepared = _prepare(inputs, split_kv=split_kv)
    threshold = torch.full((17,), float("-inf"), device="cuda", dtype=torch.float32)
    expected_lengths = torch.tensor(
        _valid_lengths(cu_q, cu_kv), device="cuda", dtype=torch.int32
    )
    lengths = torch.empty(17, device="cuda", dtype=torch.int32)
    flags = torch.empty(17, device="cuda", dtype=torch.int32)

    capacity = 6144
    values = torch.empty((32, capacity), device="cuda", dtype=torch.int64)
    counts = torch.empty((32, num_segments + 1), device="cuda", dtype=torch.int32)
    _launch(
        _SelectiveLogitsMode.CANDIDATE,
        capacity,
        values,
        counts,
        prepared,
        inputs,
        threshold,
    )
    _finalize_fp8_mqa_packed_candidates(
        counts, values, lengths, flags, rows=17, capacity=capacity
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(lengths, expected_lengths)
    for row in range(17):
        count = int(lengths[row])
        actual_indices = (values[row, :count] & 0xFFFFFFFF).sort().values
        torch.testing.assert_close(
            actual_indices, torch.arange(count, device="cuda", dtype=torch.int64)
        )

    overflow_capacity = 256
    overflow_values = torch.empty(
        (32, overflow_capacity), device="cuda", dtype=torch.int64
    )
    overflow_counts = torch.empty(
        (32, num_segments + 1), device="cuda", dtype=torch.int32
    )
    _launch(
        _SelectiveLogitsMode.CANDIDATE,
        overflow_capacity,
        overflow_values,
        overflow_counts,
        prepared,
        inputs,
        threshold,
    )
    raw_overflow_counts = overflow_counts[:17].clone()
    _finalize_fp8_mqa_packed_candidates(
        overflow_counts,
        overflow_values,
        lengths,
        flags,
        rows=17,
        capacity=overflow_capacity,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(
        raw_overflow_counts[:, 1:].sum(dim=1, dtype=torch.int32), expected_lengths
    )
    segment_capacity = overflow_capacity // num_segments
    segment_overflow = (raw_overflow_counts[:, 1:] > segment_capacity).any(dim=1)
    assert segment_overflow.any()
    expected_published = torch.where(
        segment_overflow,
        torch.full_like(expected_lengths, overflow_capacity + 1),
        expected_lengths,
    )
    torch.testing.assert_close(overflow_counts[:17, 0], expected_published)
    torch.testing.assert_close(
        lengths,
        raw_overflow_counts[:, 1:]
        .clamp(max=segment_capacity)
        .sum(dim=1, dtype=torch.int32),
    )
    assert (flags == 1).all()


@_requires_selective_sm100
def test_adapter_rejects_unsafe_layouts_before_launch():
    from flashinfer.experimental.attn_scores.selective_logits import (
        _prepare_fp8_mqa_selective_logits,
    )

    q, kv, scale, weights, cu_q, cu_kv = _inputs(q_rows=1, kv_rows=130)
    inputs = (q, kv, scale, weights, cu_q, cu_kv)
    kv_fused, block_table = _paged_inputs(inputs)
    raw = torch.empty(kv_fused.numel() + 1, device="cuda", dtype=torch.uint8)[1:]
    raw.copy_(kv_fused.reshape(-1))
    misaligned_kv = raw.view_as(kv_fused)
    with pytest.raises(ValueError, match="16-byte aligned"):
        _prepare_fp8_mqa_selective_logits(
            q, misaligned_kv, weights, cu_q, cu_kv, block_table
        )

    strided_weights = torch.empty((1, 16), device="cuda", dtype=torch.float32)[:, ::2]
    strided_weights.copy_(weights)
    with pytest.raises(ValueError, match="contiguous"):
        _prepare_fp8_mqa_selective_logits(
            q, kv_fused, strided_weights, cu_q, cu_kv, block_table
        )

    with pytest.raises(ValueError, match="split_kv must be 1, 2, 4, 8, 16, 32, or 64"):
        _prepare_fp8_mqa_selective_logits(
            q, kv_fused, weights, cu_q, cu_kv, block_table, split_kv=3
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        _prepare_fp8_mqa_selective_logits(
            q,
            kv_fused,
            weights,
            cu_q,
            cu_kv,
            block_table,
            kv_chunk_size=256,
            split_kv=2,
        )

    q, kv, scale, weights, cu_q, _cu_kv = _inputs(q_rows=17, kv_rows=387)
    cu_kv = torch.tensor([0, 130, 387], device="cuda", dtype=torch.int32)
    inputs = (q, kv, scale, weights, cu_q, cu_kv)
    prepared = _prepare(inputs)
    assert prepared.batches == 2


@_requires_selective_sm100
def test_prepared_problem_shape_configuration_reaches_compiled_kernel():
    from flashinfer.experimental.attn_scores.selective_logits import (
        _finalize_fp8_mqa_packed_candidates,
    )

    inputs = _inputs(q_rows=17, kv_rows=387, num_heads=4, head_dim=64)
    q, kv, scale, weights, cu_q, cu_kv = inputs
    prepared = _prepare(
        inputs,
        num_heads=4,
        head_dim=64,
    )
    assert prepared.num_heads == 4
    assert prepared.head_dim == 64

    values = torch.empty((32, 387), device="cuda", dtype=torch.float32)
    counts = torch.empty((32, 1), device="cuda", dtype=torch.int32)
    _launch(
        _SelectiveLogitsMode.SAMPLE,
        387,
        values,
        counts,
        prepared,
        inputs,
    )
    torch.cuda.synchronize()
    expected_rows = _oracle_rows(inputs)
    for row, expected in enumerate(expected_rows):
        torch.testing.assert_close(values[row, : expected.numel()], expected)

    capacity = 1024
    packed = torch.empty((32, capacity), device="cuda", dtype=torch.int64)
    segmented_counts = torch.empty((32, 33), device="cuda", dtype=torch.int32)
    thresholds = torch.full((q.shape[0],), float("-inf"), device="cuda")
    _launch(
        _SelectiveLogitsMode.CANDIDATE,
        capacity,
        packed,
        segmented_counts,
        prepared,
        inputs,
        thresholds,
    )
    raw_counts = segmented_counts.clone()
    lengths = torch.empty(q.shape[0], device="cuda", dtype=torch.int32)
    flags = torch.empty_like(lengths)
    _finalize_fp8_mqa_packed_candidates(
        segmented_counts,
        packed,
        lengths,
        flags,
        rows=q.shape[0],
        capacity=capacity,
    )
    torch.cuda.synchronize()
    valid_lengths = torch.tensor(
        _valid_lengths(cu_q, cu_kv), device="cuda", dtype=torch.int32
    )
    torch.testing.assert_close(
        raw_counts[: q.shape[0], 1:].sum(dim=1), valid_lengths.to(torch.int64)
    )
    torch.testing.assert_close(lengths, valid_lengths)
    for row, valid in enumerate(valid_lengths.tolist()):
        row_packed = packed[row, :valid]
        actual_indices = (row_packed & 0xFFFFFFFF).to(torch.int32)
        order = actual_indices.argsort()
        torch.testing.assert_close(
            actual_indices[order],
            torch.arange(valid, device="cuda", dtype=torch.int32),
        )
        actual_score_bits = ((row_packed[order] >> 32) & 0xFFFFFFFF).to(torch.int32)
        torch.testing.assert_close(
            actual_score_bits,
            values[row, :valid].view(torch.int32),
        )


@_requires_selective_sm100
def test_prepared_problem_shape_supports_logical_n_256():
    inputs = _inputs(q_rows=1, kv_rows=128, num_heads=256, head_dim=32)
    q, _kv, _scale, _weights, _cu_q, _cu_kv = inputs
    prepared = _prepare(inputs, num_heads=256, head_dim=32)
    assert prepared.query_tile == 1
    assert prepared.num_umma_stages == 1

    values = torch.empty((1, 128), device="cuda", dtype=torch.float32)
    counts = torch.empty((1, 1), device="cuda", dtype=torch.int32)
    _launch(
        _SelectiveLogitsMode.SAMPLE,
        128,
        values,
        counts,
        prepared,
        inputs,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(values[0], _oracle_rows(inputs)[0])


@_requires_selective_sm100
def test_metadata_schedule_has_no_fixed_tile_limit():
    """Runtime global scratch supports schedules beyond the old 8192-tile cap."""
    q_rows = 8193
    inputs = _inputs(q_rows=q_rows, kv_rows=8320, num_heads=256, head_dim=32)
    q, kv, scale, weights, _cu_q, _cu_kv = inputs
    cu_q = torch.tensor([0, q_rows], device="cuda", dtype=torch.int32)
    cu_kv = torch.tensor([0, kv.shape[0]], device="cuda", dtype=torch.int32)
    inputs = (q, kv, scale, weights, cu_q, cu_kv)

    prepared = _prepare(inputs, num_heads=256, head_dim=32, num_sms=4)
    torch.cuda.synchronize()

    assert prepared.query_tile == 1
    assert prepared.tiles == q_rows
    assert prepared.schedule_prefix.shape == (q_rows,)
    assert int(prepared.schedule_prefix[-1]) == q_rows
    assert torch.equal(
        prepared.schedule_meta[-1].cpu(), torch.tensor([q_rows, 0], dtype=torch.int32)
    )


@_requires_selective_sm100
def test_prepared_problem_shape_rejects_unsupported_specializations():
    inputs = _inputs(q_rows=1, kv_rows=128, num_heads=4, head_dim=96)
    q, kv, scale, weights, cu_q, cu_kv = inputs
    with pytest.raises(ValueError, match="head_dim must be a multiple"):
        _prepare(
            (
                q[:, :, :65].contiguous(),
                kv[:, :65].contiguous(),
                scale,
                weights,
                cu_q,
                cu_kv,
            ),
            num_heads=4,
            head_dim=65,
        )

    inputs = _inputs(q_rows=1, kv_rows=128, num_heads=257, head_dim=32)
    with pytest.raises(ValueError, match="no legal selective FP8 query tile"):
        _prepare(inputs, num_heads=257, head_dim=32)


@_requires_selective_sm100
def test_opt_in_validation_rejects_invalid_device_metadata(monkeypatch):
    from flashinfer.experimental.attn_scores.selective_logits import (
        _prepare_fp8_mqa_selective_logits,
    )

    monkeypatch.setenv("FLASHINFER_VALIDATE_INPUTS", "1")
    inputs = _inputs(q_rows=17, kv_rows=387)
    q, _kv, _scale, weights, cu_q, cu_kv = inputs
    kv_fused, block_table = _paged_inputs(inputs, block_size=32)

    bad_cu_q = cu_q.clone()
    bad_cu_q[-1] = q.shape[0] - 1
    with pytest.raises(ValueError, match=r"cu_q\[-1\]"):
        _prepare_fp8_mqa_selective_logits(
            q, kv_fused, weights, bad_cu_q, cu_kv, block_table
        )

    bad_cu_kv = cu_kv.clone()
    bad_cu_kv[1] = -1
    with pytest.raises(ValueError, match="monotonically"):
        _prepare_fp8_mqa_selective_logits(
            q, kv_fused, weights, cu_q, bad_cu_kv, block_table
        )

    short_causal_kv = cu_kv.clone()
    short_causal_kv[1] = 0
    with pytest.raises(ValueError, match="greater than"):
        _prepare_fp8_mqa_selective_logits(
            q, kv_fused, weights, cu_q, short_causal_kv, block_table
        )

    too_short = block_table[:, :-1].contiguous()
    with pytest.raises(ValueError, match="requires at least"):
        _prepare_fp8_mqa_selective_logits(q, kv_fused, weights, cu_q, cu_kv, too_short)

    bad_table = block_table.clone()
    bad_table[0, 0] = kv_fused.shape[0]
    with pytest.raises(ValueError, match="physical block index"):
        _prepare_fp8_mqa_selective_logits(q, kv_fused, weights, cu_q, cu_kv, bad_table)

    single = _inputs(q_rows=1, kv_rows=128)
    q1, _kv1, _scale1, weights1, cu_q1, cu_kv1 = single
    fused1, table1 = _paged_inputs(single)
    starts = torch.zeros(1, device="cuda", dtype=torch.int32)
    ends = torch.full((1,), 129, device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError, match="explicit row"):
        _prepare_fp8_mqa_selective_logits(
            q1,
            fused1,
            weights1,
            cu_q1,
            cu_kv1,
            table1,
            row_starts=starts,
            row_ends=ends,
        )


@_requires_selective_sm100
@pytest.mark.parametrize("q_rows", [1, 33])
@pytest.mark.parametrize("strategy", ["full", "selective"])
def test_wrapper_page_map_updates_in_eager_and_graph(strategy, q_rows):
    """Both execution modes consume the current physical mapping."""
    from flashinfer import FP8PagedMQATopKWrapper

    q, kv, scale, weights, _, _ = _inputs(q_rows=q_rows, kv_rows=8192)
    cu_q = torch.tensor([0, q_rows], device="cuda", dtype=torch.int32)
    cu_kv = torch.tensor([0, kv.shape[0]], device="cuda", dtype=torch.int32)
    fused = _fuse_paged_kv(kv, scale)
    table = torch.arange(fused.shape[0], device="cuda", dtype=torch.int32)[None, :]
    top_k = 512 if strategy == "full" else 8
    selective = dict(
        top_k=top_k,
        max_context_length=8192,
        max_chunk_size=q_rows,
        max_batch_size=1,
        kv_layout=fused,
    )
    wrapper = FP8PagedMQATopKWrapper(strategy=strategy, **selective)
    if strategy == "full":
        wrapper.plan(cu_q, cu_kv, table, num_query_rows=q.shape[0])
        wrapper.prepare_pool(fused)
        wrapper.run(q, weights, fused)
    else:
        wrapper.prepare_pool(fused)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.plan(cu_q, cu_kv, table, num_query_rows=q.shape[0])
        output = (
            wrapper.run(q=q, kv_fused=fused, weights=weights)
            if strategy == "full"
            else wrapper.run(q, weights, fused)
        )
    for shift in (1, 3):
        # Replay must consume new payload values as well as the new page map.
        q.copy_((-q.float()).to(q.dtype))
        weights.copy_(weights.roll(1, dims=1))
        kv.copy_(kv.float().roll(shift, dims=0).to(kv.dtype))
        scale.copy_(scale.roll(shift))
        fused.copy_(_fuse_paged_kv(kv, scale))
        order = torch.arange(fused.shape[0], device="cuda").roll(shift)
        table.copy_(order[None, :])
        mapped_kv = kv.reshape(-1, 128, 128)[order].reshape_as(kv)
        mapped_scale = scale.reshape(-1, 128)[order].reshape_as(scale)
        oracle = _oracle_rows((q, mapped_kv, mapped_scale, weights, cu_q, cu_kv))
        wrapper.plan(cu_q, cu_kv, table, num_query_rows=q.shape[0])
        eager = (
            wrapper.run(q=q, kv_fused=fused, weights=weights)
            if strategy == "full"
            else wrapper.run(q, weights, fused)
        ).clone()
        graph.replay()
        torch.cuda.synchronize()
        for actual in (eager, output):
            _check_selection(actual, oracle, top_k)


@_requires_selective_sm100
@pytest.mark.parametrize("q_rows", [1, 33])
def test_full_wrapper_failed_preparation_rejects_run_and_retries(monkeypatch, q_rows):
    from flashinfer import FP8PagedMQATopKWrapper

    q, kv, scale, weights, _, _ = _inputs(q_rows=q_rows, kv_rows=2048)
    cu_q = torch.tensor([0, q_rows], device="cuda", dtype=torch.int32)
    cu_kv = torch.tensor([0, kv.shape[0]], device="cuda", dtype=torch.int32)
    fused = _fuse_paged_kv(kv, scale)
    table = torch.arange(fused.shape[0], device="cuda", dtype=torch.int32)[None, :]
    wrapper = FP8PagedMQATopKWrapper(
        strategy="full",
        num_heads=q.shape[1],
        head_dim=q.shape[2],
        page_size=fused.shape[1],
        top_k=512,
        max_context_length=int((cu_kv[1:] - cu_kv[:-1]).max().item()),
        max_chunk_size=q.shape[0],
        max_batch_size=cu_q.numel() - 1,
        kv_layout=fused,
    )
    wrapper.plan(cu_q, cu_kv, table, num_query_rows=q.shape[0])
    wrapper.prepare_pool(fused)
    wrapper.run(q, weights, fused)
    unprepared_pool = fused.clone()

    def fail_prepare(*args, **kwargs):
        raise RuntimeError("injected preparation failure")

    with monkeypatch.context() as patch:
        patch.setattr(type(wrapper._full_capacity_plan), "_bind_launches", fail_prepare)
        with pytest.raises(RuntimeError, match="injected preparation failure"):
            wrapper.prepare_pool(unprepared_pool)
        with pytest.raises(RuntimeError, match="injected preparation failure"):
            wrapper.run(q=q, kv_fused=unprepared_pool, weights=weights)
    _check_selection(
        wrapper.run(q, weights, unprepared_pool),
        _oracle_rows((q, kv, scale, weights, cu_q, cu_kv)),
        512,
    )
    _check_selection(
        wrapper.run(q, weights, fused),
        _oracle_rows((q, kv, scale, weights, cu_q, cu_kv)),
        512,
    )

    wrapper.plan(cu_q, cu_kv, table, num_query_rows=q.shape[0])
    wrapper.prepare_pool(unprepared_pool)
    wrapper.run(q, weights, unprepared_pool)
    output = wrapper.run(q=q, kv_fused=unprepared_pool, weights=weights)
    torch.cuda.synchronize()
    assert output.shape == (q_rows, 512)
    for row, scores in enumerate(_oracle_rows((q, kv, scale, weights, cu_q, cu_kv))):
        torch.testing.assert_close(
            scores[output[row].long()].sort().values,
            scores.topk(512).values.sort().values,
            atol=5e-5,
            rtol=1e-5,
        )
    owner = weakref.ref(wrapper)
    plan_owner = weakref.ref(wrapper._full_capacity_plan)
    del wrapper
    assert owner() is None
    assert plan_owner() is None


@_requires_selective_sm100
@pytest.mark.parametrize("heads", [8, 64])
def test_selective_math_mode_cache_alternation(heads):
    """Prepared and process caches cannot reuse a different score recipe."""
    from dataclasses import FrozenInstanceError

    inputs = _inputs(3, 259, heads)
    rows = _oracle_rows(inputs)
    prepared = [
        _prepare(inputs, num_heads=heads, num_sms=4, math_mode=mode)
        for mode in ("relu_fast", "relu_strict")
    ]
    prepared.append(_prepare(inputs, num_heads=heads, num_sms=4))
    assert prepared[2].math_mode == "relu_strict"
    cached = []
    # Hold runtime storage fixed: rebinding an output correctly rebuilds the
    # prepared launch object independently of the numerical compile policy.
    values = torch.empty((16, 259), device="cuda", dtype=torch.float32)
    counts = torch.empty((16, 1), device="cuda", dtype=torch.int32)
    for policy in (*prepared, prepared[0]):
        _launch(_SelectiveLogitsMode.SAMPLE, 259, values, counts, policy, inputs)
        torch.cuda.synchronize()
        for row, expected in enumerate(rows):
            torch.testing.assert_close(
                values[row, : expected.numel()], expected, atol=5e-5, rtol=1e-5
            )
        assert all(key[-2] == policy.math_mode for key in policy._launches)
        cached.append(next(iter(policy._launches.values())))
    assert cached[0] is cached[3]
    assert cached[0] is not cached[1]
    assert cached[0].compiled is cached[3].compiled
    assert cached[0].compiled is not cached[1].compiled
    assert cached[1].compiled is cached[2].compiled
    with pytest.raises(FrozenInstanceError):
        prepared[0].math_mode = "relu_strict"


@_requires_selective_sm100
@pytest.mark.parametrize("heads", [8, 64])
@pytest.mark.parametrize("last_head", [False, True])
@pytest.mark.parametrize("sign", [-1, 1])
def test_selective_sample_and_repair_range_safe_relu(heads, last_head, sign):
    inputs = _inputs(17, 387, heads)
    q, kv, scales, weights, _, _ = inputs
    active = heads - 1 if last_head else 0
    q.zero_()
    kv.zero_()
    weights.zero_()
    q[:, active, 0] = 1
    kv[:, 0] = 1
    scales.fill_(0.5)
    weights[:, active] = sign * torch.finfo(torch.float32).max
    prepared = _prepare(inputs, num_heads=heads, num_sms=4, math_mode="relu_strict")
    fused, table = _paged_inputs(inputs)
    tile = prepared.query_tile
    padded = (q.shape[0] + tile - 1) // tile * tile
    samples = torch.empty((padded, 512), device=q.device, dtype=torch.float32)
    records = torch.empty((padded, 6144), device=q.device, dtype=torch.int64)
    counts = torch.empty((padded, 1), device=q.device, dtype=torch.int32)
    unused = torch.empty_like(counts)
    thresholds = torch.full((q.shape[0],), -torch.inf, device=q.device)

    def sample():
        _fp8_mqa_selective_logits(
            q,
            fused,
            weights,
            table,
            prepared,
            mode=_SelectiveLogitsMode.SAMPLE,
            capacity=512,
            values=samples,
            index_counts=unused,
        )

    sample()
    _fp8_mqa_selective_logits(
        q,
        fused,
        weights,
        table,
        prepared,
        mode=_SelectiveLogitsMode.REPAIR,
        capacity=6144,
        values=records,
        index_counts=counts,
        policy_values=thresholds,
    )
    expected = sign * torch.finfo(torch.float32).max / 2
    for row, length in enumerate(_valid_lengths(inputs[-2], inputs[-1])):
        assert bool(torch.isfinite(samples[row, :length]).all())
        torch.testing.assert_close(
            samples[row, :length],
            torch.full_like(samples[row, :length], expected),
            atol=0,
            rtol=0,
        )
        count = int(counts[row, 0])
        assert count == length
        packed = records[row, :count]
        indices = packed.bitwise_and(0xFFFFFFFF)
        assert indices.unique().numel() == length
        assert bool(((indices >= 0) & (indices < length)).all())
        values = (packed >> 32).to(torch.int32).view(torch.float32)
        assert bool(torch.isfinite(values).all())
        torch.testing.assert_close(
            values, torch.full_like(values, expected), atol=0, rtol=0
        )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        sample()
    q[:, active, 0] = -1
    graph.replay()
    for row, length in enumerate(_valid_lengths(inputs[-2], inputs[-1])):
        assert bool((samples[row, :length] == 0).all())


@contextmanager
def _preserve_cuda_device_streams(devices):
    original = torch.cuda.current_device()
    saved_streams = [torch.cuda.current_stream(device) for device in devices]

    def stream_handles():
        return [torch.cuda.current_stream(device).cuda_stream for device in devices]

    try:
        yield stream_handles
    finally:
        for stream in saved_streams:
            with torch.cuda.device(stream.device):
                torch.cuda.set_stream(stream)
        torch.cuda.set_device(original)


@_requires_selective_sm100
@pytest.mark.parametrize("route", ["full", "selective"])
def test_public_wrapper_on_non_current_device_after_warm_cache(route):
    if torch.cuda.device_count() < 2:
        pytest.skip("Requires two CUDA devices")
    wrappers = []
    with _preserve_cuda_device_streams((0, 1)) as stream_handles:
        torch.cuda.set_device(0)
        caller = torch.cuda.Stream(device=0)
        torch.cuda.set_stream(caller)
        expected_streams = stream_handles()
        # Compile on device0 first, then reuse same-CC caches on device1.
        for target in (0, 1):
            with torch.cuda.device(target):
                inputs = _inputs(5, 8192)
                q, _, _, weights, cu_q, cu_kv = inputs
                cu_q.copy_(torch.tensor([0, 1, 5], device=q.device, dtype=torch.int32))
                fused, table = _paged_inputs(inputs, 32)
            selective = dict(
                page_size=32,
                max_context_length=8192,
                max_chunk_size=5,
                max_batch_size=2,
                kv_layout=fused,
            )
            wrapper = FP8PagedMQATopKWrapper(
                strategy="full" if route == "full" else "selective",
                **selective,
            )
            if route == "full":
                wrapper.plan(cu_q, cu_kv, table, num_query_rows=q.shape[0])
                wrapper.prepare_pool(fused)
                wrapper.run(q, weights, fused)
            else:
                wrapper.prepare_pool(fused)
            assert torch.cuda.current_device() == 0
            assert stream_handles() == expected_streams
            wrappers.append(wrapper)
            for tied in (False, True):
                if tied:
                    weights.zero_()
                if route == "selective":
                    wrapper.plan(cu_q, cu_kv, table, num_query_rows=q.shape[0])
                output = (
                    wrapper.run(q=q, kv_fused=fused, weights=weights)
                    if route == "full"
                    else wrapper.run(q, weights, fused)
                )
                assert torch.cuda.current_device() == 0
                assert stream_handles() == expected_streams
                with torch.cuda.device(target):
                    graph = torch.cuda.CUDAGraph()
                    capture_stream = torch.cuda.Stream(device=q.device)
                    with torch.cuda.graph(graph, stream=capture_stream):
                        if route == "selective":
                            wrapper.plan(cu_q, cu_kv, table, num_query_rows=q.shape[0])
                        captured = (
                            wrapper.run(q=q, kv_fused=fused, weights=weights)
                            if route == "full"
                            else wrapper.run(q, weights, fused)
                        )
                    assert stream_handles() == expected_streams
                    graph.replay()
                    for indices in (output, captured):
                        for row, scores in enumerate(_oracle_rows(inputs)):
                            count = min(512, scores.numel())
                            selected = indices[row, :count].long()
                            assert bool(
                                ((selected >= 0) & (selected < scores.numel())).all()
                            )
                            assert selected.unique().numel() == count
                            torch.testing.assert_close(
                                scores[selected].sort().values,
                                scores.topk(count).values.sort().values,
                                atol=5e-5,
                                rtol=1e-5,
                            )
                assert torch.cuda.current_device() == 0
                assert stream_handles() == expected_streams


@_requires_selective_sm100
@pytest.mark.parametrize("page_size", [32, 64, 128])
@pytest.mark.parametrize("heads", [8, 64])
@pytest.mark.parametrize("route", ["full", "selective"])
def test_natural_page_table_replay_and_replan(page_size, heads, route, monkeypatch):
    monkeypatch.setenv("FLASHINFER_VALIDATE_INPUTS", "1")
    q, kv, scales, weights, _, _ = _inputs(5, 257 + 9001, heads)
    cu_q = torch.tensor([0, 3, 5], device="cuda", dtype=torch.int32)
    cu_kv = torch.tensor([0, 257, 257 + 9001], device="cuda", dtype=torch.int32)
    fused, padded_table = _paged_inputs(
        (q, kv, scales, weights, cu_q, cu_kv), page_size
    )
    table = padded_table[:, : math.ceil(9001 / page_size)].contiguous()
    table[0, math.ceil(257 / page_size) :] = 2**31 - 1
    packed = fused.flatten(1)
    page_keys = packed[:, : page_size * 128].contiguous().view(kv.dtype)
    page_keys = page_keys.reshape(-1, page_size, 128).float()
    page_scales = packed[:, page_size * 128 :].contiguous().view(torch.float32)
    strategy = "full" if route == "full" else "selective"
    wrapper = FP8PagedMQATopKWrapper(
        strategy=strategy,
        num_heads=heads,
        page_size=page_size,
        max_context_length=9001,
        max_chunk_size=5,
        max_batch_size=2,
        kv_layout=fused,
    )

    def check(indices, lengths):
        for request, length in enumerate(lengths):
            q_begin, q_end = (0, 3) if request == 0 else (3, 5)
            physical = table[request, : math.ceil(length / page_size)].long()
            keys = page_keys[physical].reshape(-1, 128)[:length]
            scale = page_scales[physical].reshape(-1)[:length]
            scores = torch.einsum("qhd,kd->qhk", q[q_begin:q_end].float(), keys)
            scores.relu_()
            scores = (scores * weights[q_begin:q_end, :, None]).sum(1) * scale
            for local_row, row in enumerate(range(q_begin, q_end)):
                end = length - (q_end - q_begin) + local_row + 1
                count = min(end, 512)
                selected = indices[row, :count].long()
                assert bool(((selected >= 0) & (selected < end)).all())
                assert selected.unique().numel() == count
                torch.testing.assert_close(
                    scores[local_row, selected].sort().values,
                    scores[local_row, :end].topk(count).values.sort().values,
                    atol=2e-5,
                    rtol=2e-5,
                )
                assert bool((indices[row, count:] == -1).all())

    for generation, length in enumerate((9001, 8999)):
        if generation:
            weights.zero_()
        cu_kv[-1] = 257 + length
        wrapper.plan(cu_q, cu_kv, table, num_query_rows=q.shape[0])
        wrapper.prepare_pool(fused)
        check(wrapper.run(q, weights, fused), (257, length))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            wrapper.plan(cu_q, cu_kv, table, num_query_rows=q.shape[0])
            output = wrapper.run(q, weights, fused)
        for _ in range(2):
            for request, request_length in enumerate((257, length)):
                pages = math.ceil(request_length / page_size)
                table[request, :pages].copy_(table[request, :pages].flip(0))
            graph.replay()
            check(output, (257, length))
        del graph
        if generation and strategy == "selective":
            assert (
                wrapper._capacity_plan._engine._second_repair_flags[3:5]
                .count_nonzero()
                .item()
                == 2
            )
            assert wrapper.repair_error.item() == 0


@_requires_selective_sm100
@pytest.mark.parametrize(
    "page_size,heads,internal,lengths",
    [
        pytest.param(
            page, heads, internal, (0, 129, 259), id=f"{internal}-{heads}-{page}"
        )
        for internal in (False, True)
        for heads in (8, 64)
        for page in (32, 64, 128)
    ]
    + [
        pytest.param(
            128, heads, True, (129, 0, 9001), id=f"interspersed-True-{heads}-128"
        )
        for heads in (8, 64)
    ],
)
def test_inactive_sample_rows_and_logical_page_clamp(
    page_size, heads, internal, lengths, monkeypatch
):
    monkeypatch.setenv("FLASHINFER_VALIDATE_INPUTS", "1")
    active_requests = [request for request, length in enumerate(lengths) if length]
    inactive_request = next(
        request for request, length in enumerate(lengths) if not length
    )
    interspersed = inactive_request == 1
    pages = [math.ceil(length / page_size) for length in lengths]
    physical_lengths = [2 * count * page_size for count in pages]
    q, kv, scales, weights, _, _ = _inputs(3, sum(physical_lengths), heads)
    cu_q = torch.tensor([0, 1, 2, 3], device=q.device, dtype=torch.int32)
    original_prefix = torch.tensor(
        [0, physical_lengths[0], sum(physical_lengths[:2]), sum(physical_lengths)],
        device=q.device,
        dtype=torch.int32,
    )
    fused, table = _paged_inputs(
        (q, kv, scales, weights, cu_q, original_prefix), page_size
    )
    table[inactive_request].fill_(2**31 - 1)
    sample_pages = torch.full(
        (3, max(pages)), 2**31 - 1, device=q.device, dtype=torch.int64
    )
    for request in active_requests:
        sample_pages[request, : pages[request]] = 2 * torch.arange(
            pages[request], device=q.device
        )
        table[request, 2 * pages[request] :].fill_(2**31 - 1)
    sample_prefix = torch.tensor(
        [0, lengths[0], sum(lengths[:2]), sum(lengths)],
        device=q.device,
        dtype=torch.int32,
    )
    starts = torch.zeros(3, device=q.device, dtype=torch.int32)
    ends = torch.tensor(lengths, device=q.device, dtype=torch.int32)
    prepared = sl._prepare_fp8_mqa_selective_logits(
        q,
        fused,
        weights,
        cu_q,
        sample_prefix,
        table,
        row_starts=starts,
        row_ends=ends,
        num_heads=heads,
        # One CTA forces Q lookahead across the empty middle request rather
        # than assigning each live request to a separate producer role.
        num_sms=1 if interspersed else 4,
    )
    if interspersed:
        assert prepared.context_lens[:3].tolist() == list(lengths)
        # The final request owns tile 2; padded tiles have no work.
        assert prepared.schedule_meta[:, 0].tolist() == [0, 3]
        assert prepared.schedule_meta[:, 1].count_nonzero().item() == 0
    padded = math.ceil(3 / prepared.query_tile) * prepared.query_tile
    capacity = math.ceil(max(lengths) / 128) * 128
    values = torch.full((padded, capacity), float("nan"), device=q.device)
    counts = torch.empty((padded, 1), device=q.device, dtype=torch.int32)

    def run():
        sl._fp8_mqa_selective_logits(
            q,
            fused,
            weights,
            table,
            prepared,
            mode=sl._SelectiveLogitsMode.SAMPLE,
            capacity=capacity,
            values=values,
            index_counts=counts,
            sample_page_indices=sample_pages if internal else None,
        )

    packed = fused.flatten(1)
    page_keys = (
        packed[:, : page_size * 128]
        .contiguous()
        .view(kv.dtype)
        .reshape(-1, page_size, 128)
        .float()
    )
    page_scales = packed[:, page_size * 128 :].contiguous().view(torch.float32)

    def check():
        assert torch.isnan(values[inactive_request]).all()
        for request in active_requests:
            positions = torch.arange(lengths[request], device=q.device)
            logical_pages = positions // page_size
            if internal:
                logical_pages = sample_pages[request, logical_pages]
            physical_pages = table[request, logical_pages].long()
            offsets = positions % page_size
            keys = page_keys[physical_pages, offsets]
            scale = page_scales[physical_pages, offsets]
            scores = torch.einsum("hd,kd->hk", q[request].float(), keys).relu_()
            expected = (scores * weights[request, :, None]).sum(0) * scale
            torch.testing.assert_close(
                values[request, : lengths[request]], expected, atol=2e-5, rtol=2e-5
            )
            assert torch.isnan(values[request, lengths[request] :]).all()

    run()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(2):
        for request in active_requests:
            count = 2 * pages[request]
            table[request, :count].copy_(table[request, :count].flip(0))
            sample_pages[request, : pages[request]].copy_(
                sample_pages[request, : pages[request]].flip(0)
            )
        graph.replay()
        check()


@pytest.fixture(autouse=True)
def _clear_paged_input_cache():
    # Reuse pages within one case without keeping its GPU tensors across cases.
    _PAGED_INPUT_CACHE.clear()
    try:
        yield
    finally:
        _PAGED_INPUT_CACHE.clear()


# Sample threshold and parallel ladder


@_requires_selective_sm100
@pytest.mark.parametrize("page_size", [32, 64])
@pytest.mark.parametrize("load_warps", [4, 8])
def test_sparse_gather4_sample_small_pages(page_size, load_warps):
    """Gather key rows and scales across shuffled sub-tile physical pages."""
    import flashinfer.experimental.attn_scores.selective_logits as sl

    rows, length, samples = 16, 4096, 256
    q, kv, scales, weights, qo, kv_prefix = _inputs(rows, length, 64)
    qo = torch.tensor([0, rows], dtype=torch.int32, device="cuda")
    kv_prefix = torch.tensor([0, length], dtype=torch.int32, device="cuda")
    weights.abs_()
    fused, table = _paged_inputs((q, kv, scales, weights, qo, kv_prefix), page_size)
    positions = torch.div(
        torch.arange(samples, device="cuda") * (length - 1),
        samples - 1,
        rounding_mode="floor",
    )
    starts = torch.zeros(rows, dtype=torch.int32, device="cuda")
    ends = torch.full_like(starts, samples)
    prepared = sl._prepare_fp8_mqa_selective_logits(
        q,
        fused,
        weights,
        qo,
        kv_prefix,
        table,
        row_starts=starts,
        row_ends=ends,
        query_tile=2,
        num_heads=64,
        head_dim=128,
        num_sms=1,
    )
    sample_prefix = torch.tensor([0, samples], dtype=torch.int32, device="cuda")
    packed, packed_table = _paged_inputs(
        (q, kv[positions], scales[positions], weights, qo, sample_prefix), page_size
    )
    native_prepared = sl._prepare_fp8_mqa_selective_logits(
        q,
        packed,
        weights,
        qo,
        sample_prefix,
        packed_table,
        row_starts=starts,
        row_ends=ends,
        query_tile=2,
        num_heads=64,
        head_dim=128,
        num_sms=1,
    )
    actual = torch.empty((rows, samples), device="cuda")
    on_the_fly = torch.empty_like(actual)
    native = torch.empty_like(actual)
    counts = torch.zeros((rows, 1), dtype=torch.int32, device="cuda")
    indices = positions.view(1, -1).contiguous()
    physical_ends = torch.full_like(ends, length)

    def run():
        sl._fp8_mqa_selective_logits(
            q,
            fused,
            weights,
            table,
            prepared,
            mode=sl._SelectiveLogitsMode.SAMPLE,
            capacity=samples,
            values=actual,
            index_counts=counts,
            sample_page_indices=indices,
            _sparse_sample_load_warps=load_warps,
            _sparse_sample_group_rows=rows,
        )
        sl._fp8_mqa_selective_logits(
            q,
            fused,
            weights,
            table,
            prepared,
            mode=sl._SelectiveLogitsMode.SAMPLE,
            capacity=samples,
            values=on_the_fly,
            index_counts=counts,
            _sparse_sample_load_warps=load_warps,
            _sparse_sample_group_rows=rows,
            _sparse_sample_metadata=(physical_ends, ends, qo, kv_prefix),
        )
        sl._fp8_mqa_selective_logits(
            q,
            packed,
            weights,
            packed_table,
            native_prepared,
            mode=sl._SelectiveLogitsMode.SAMPLE,
            capacity=samples,
            values=native,
            index_counts=counts,
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for changed in (False, True):
        if changed:
            fused.copy_(fused.flip(0))
            table.copy_(fused.shape[0] - 1 - table)
        graph.replay()
        torch.testing.assert_close(actual, native, rtol=0, atol=0)
        torch.testing.assert_close(on_the_fly, actual, rtol=0, atol=0)
        reference = (
            (q.float() @ kv[positions].float().T).relu() * weights[:, :, None]
        ).sum(1) * scales[positions][None, :]
        torch.testing.assert_close(actual, reference, rtol=1e-4, atol=1e-4)


def test_sparse_sample_groups_contain_whole_query_tiles():
    for tile in range(1, 17):
        group = sl._select_sparse_sample_group_rows(tile)
        assert group == (128 if 128 % tile == 0 else tile)
        assert group % tile == 0


def test_sparse_sample_gather4_coordinate_bounds():
    max_pages = (2**31 - 1) // 132
    assert sl._sparse_sample_gather4_serves(max_pages, 128 * 132, 128, 128)
    assert not sl._sparse_sample_gather4_serves(max_pages + 1, 128 * 132, 128, 128)
    assert not sl._sparse_sample_gather4_serves(8, 128 * 132 + 16, 128, 128)
    assert not sl._sparse_sample_gather4_serves(8, 128 * 68, 128, 64)


@_requires_selective_sm100
def test_capacity_sparse_sample_unsupported_layout_uses_pages(monkeypatch):
    """Exercise the wide-pool selection without allocating a 256 GiB pool."""
    supports = sl._sparse_sample_gather4_serves
    monkeypatch.setattr(
        sl,
        "_sparse_sample_gather4_serves",
        lambda pages, stride, page_size, dim: supports(
            (2**31 - 1) // 132 + 1, stride, page_size, dim
        ),
    )
    inputs = _request_inputs([4], [8192], heads=64)
    q, _, _, weights, qo, kv = inputs
    fused, table = _paged_inputs(inputs)
    wrapper = FP8PagedMQATopKWrapper(
        strategy="selective",
        num_heads=64,
        top_k=512,
        max_context_length=8192,
        max_chunk_size=4,
        max_batch_size=1,
        kv_layout=fused,
    )
    wrapper.plan(qo, kv, table)
    selected = wrapper.run(q, weights, fused)
    engine = wrapper._capacity_plan._engine
    assert engine._sample_sparse_metadata is None
    assert wrapper._capacity_plan._state._sample_kernel.group_rows == 0
    _check_selection(selected, _oracle_rows(inputs), 512)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.plan(qo, kv, table)
        captured = wrapper.run(q, weights, fused)
    graph.replay()
    _check_selection(captured, _oracle_rows(inputs), 512)
    assert wrapper.error.item() == wrapper.repair_error.item() == 0


@_requires_selective_sm100
@pytest.mark.parametrize("heads,tile,rows", [(8, 5, 15), (64, 2, 256), (4, 16, 256)])
def test_sparse_sample_tile_groups_match_indexed(heads, tile, rows):
    """Q5 shares whole tiles; H64/H4 share 128 request-local rows."""
    q, keys, scales, weights, _, _ = _inputs(rows, 8192, heads)
    qo = torch.tensor([0, rows], device="cuda", dtype=torch.int32)
    kv = torch.tensor([0, 8192], device="cuda", dtype=torch.int32)
    fused, table = _paged_inputs((q, keys, scales, weights, qo, kv))
    samples = 3840
    group = sl._select_sparse_sample_group_rows(tile)
    assert group == (5 if heads == 8 else 128)
    assert group == (128 if rows % 128 == 0 else tile)
    ends = torch.arange(rows, device="cuda", dtype=torch.int32) + 4096
    valid = torch.full_like(ends, samples)

    def reference_positions():
        return (
            torch.div(
                torch.arange(samples, device="cuda")[None, :]
                * (ends.long().view(-1, group)[:, :1] - 1).clamp_min(0),
                (valid.long().view(-1, group)[:, :1] - 1).clamp_min(1),
                rounding_mode="floor",
            )
            .clamp(min=0, max=8191)
            .contiguous()
        )

    positions = reference_positions()
    prepared = sl._prepare_fp8_mqa_selective_logits(
        q,
        fused,
        weights,
        qo,
        kv,
        table,
        row_starts=torch.zeros_like(ends),
        row_ends=valid,
        query_tile=tile,
        num_heads=heads,
        head_dim=128,
        num_sms=1,
    )
    padded = (rows + 15) // 16 * 16
    indexed = torch.empty((padded, samples), device="cuda")
    onfly = torch.empty_like(indexed)
    counts = torch.zeros((padded, 1), device="cuda", dtype=torch.int32)

    def run():
        for output, extra in (
            (indexed, dict(sample_page_indices=positions)),
            (onfly, dict(_sparse_sample_metadata=(ends, valid, qo, kv))),
        ):
            sl._fp8_mqa_selective_logits(
                q,
                fused,
                weights,
                table,
                prepared,
                mode=sl._SelectiveLogitsMode.SAMPLE,
                capacity=samples,
                values=output,
                index_counts=counts,
                _sparse_sample_load_warps=4,
                _sparse_sample_group_rows=group,
                **extra,
            )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for change in (0, 17):
        ends.add_(change)
        positions.copy_(reference_positions())
        graph.replay()
        torch.testing.assert_close(onfly[:rows], indexed[:rows], rtol=0, atol=0)
        for row in range(rows):
            sample_keys = positions[row // group]
            dots = q[row].float() @ keys[sample_keys].float().T
            expected = (dots.relu() * weights[row, :, None]).sum(0) * scales[
                sample_keys
            ]
            torch.testing.assert_close(onfly[row], expected, atol=1e-4, rtol=1e-4)


@_requires_selective_sm100
def test_sparse_sample_positions_match_int64_reference():
    """Production int32 metadata produces wide, exact sample coordinates."""
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    from flashinfer.experimental.attn_scores.kernels._fp8_mqa_selective import (
        sparse_sample_position,
    )
    from flashinfer.experimental.attn_scores.selective_logits import _current_stream

    class Positions:
        @cute.kernel
        def kernel(self, metadata, output):
            tid, _, _ = cute.arch.thread_idx()
            row, _, _ = cute.arch.block_idx()
            for offset in cutlass.range_constexpr(15):
                i = tid + offset * 256
                output[row, i] = sparse_sample_position(
                    i, metadata[row, 0], metadata[row, 1], metadata[row, 2]
                )

        @cute.jit
        def __call__(self, metadata, output, stream):
            self.kernel(metadata, output).launch(
                grid=(output.shape[0], 1, 1), block=(256, 1, 1), stream=stream
            )

    cases = [(0, 1, 0), (0, 1, 16), (31, 31, 31)] + [
        (end - 1, count - 1, end + 127)
        for end in (6143, 6144, 6145, 32768, 560001, 1048576)
        for count in (128, 256, 3840)
    ]
    metadata = torch.tensor(cases, device="cuda", dtype=torch.int32)
    output = torch.empty((len(cases), 3840), device="cuda", dtype=torch.int64)
    args = (from_dlpack(metadata), from_dlpack(output))
    stream = _current_stream(output.device)
    compiled = cute.compile(Positions(), *args, stream)
    compiled(*args, stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled(*args, _current_stream(output.device))
    for changed in (False, True):
        if changed:
            metadata[:, 0].add_(17)
        graph.replay()
        # Form the reference product in int64 before dividing and clamping.
        expected = torch.div(
            torch.arange(3840, device="cuda")[None, :] * metadata[:, :1],
            metadata[:, 1:2],
            rounding_mode="floor",
        ).minimum(metadata[:, 2:3])
        torch.testing.assert_close(output, expected, rtol=0, atol=0)


@_requires_selective_sm100
@pytest.mark.parametrize(
    "heads,page_size,ratio,prefix",
    [
        (64, 128, 1, (0, 129, 129, 260)),
        (64, 64, 4, (0, 129, 129, 260)),
        (8, 32, 1, (0, 5, 5, 15)),
    ],
)
def test_capacity_sparse_sample_ragged_score_oracle(
    monkeypatch, heads, page_size, ratio, prefix
):
    """Each causal group shares its count/rank and enumerates short ranges."""
    rows = prefix[-1]
    q, keys, scales, weights, _, _ = _inputs(rows, 8192, heads)
    weights.abs_()
    fused = _fuse_paged_kv(keys, scales, page_size)
    qo = torch.tensor(prefix, device=q.device, dtype=torch.int32)
    kv = torch.tensor(
        [0, 257 * ratio, 257 * ratio, 769 * ratio], device=q.device, dtype=torch.int32
    )
    pages = fused.shape[0]
    table = torch.stack(
        [
            torch.arange(pages, device=q.device, dtype=torch.int32).roll(request * 7)
            for request in range(3)
        ]
    )
    compiled_groups = []
    compile_for_device = sl._compile_for_device

    def record_compile(device, kernel, *args, **kwargs):
        if type(kernel).__name__ == "SelectiveCapacitySampleMetadataKernel":
            compiled_groups.append(kernel.group_rows)
        return compile_for_device(device, kernel, *args, **kwargs)

    monkeypatch.setattr(sl, "_compile_for_device", record_compile)
    wrapper = FP8PagedMQATopKWrapper(
        strategy="selective",
        num_heads=heads,
        page_size=page_size,
        compression_ratio=ratio,
        top_k=512,
        max_context_length=8192 * ratio,
        max_chunk_size=rows,
        max_batch_size=3,
        kv_layout=fused,
    )
    wrapper.plan(qo, kv, table)
    wrapper.run(q, weights, fused)
    engine = wrapper._capacity_plan._engine
    group = sl._select_sparse_sample_group_rows(engine._sample_prepared.query_tile)
    assert compiled_groups == [group]

    def check(lengths):
        counts = engine._sample_ends.tolist()
        ranks = engine._sample_ranks.tolist()
        for request, (begin, end) in enumerate(
            zip(prefix[:-1], prefix[1:], strict=True)
        ):
            for first in range(begin, end, group):
                first_end = (
                    lengths[request] * ratio - (end - begin) + first - begin + 1
                ) // ratio
                count = _expected_sample_count(first_end, 512)
                rank = _expected_sample_rank(first_end, count, 512)
                last = min(first + group, end)
                assert counts[first:last] == [count] * (last - first)
                assert ranks[first:last] == [rank] * (last - first)
                positions = _expected_sample_positions(
                    first_end, count, (first - begin) // group, q.device
                )
                assert positions.unique().numel() == count
                assert int(positions[-1]) <= first_end - 1
                if first_end == count:
                    torch.testing.assert_close(
                        positions, torch.arange(first_end, device=q.device)
                    )
                physical = (
                    table[request, positions // page_size].long() * page_size
                    + positions % page_size
                )
                for row in range(first, last):
                    dots = q[row].float() @ keys[physical].float().T
                    expected = (dots.relu() * weights[row, :, None]).sum(0) * scales[
                        physical
                    ]
                    torch.testing.assert_close(
                        engine._sample_scores[row, :count],
                        expected,
                        atol=1e-4,
                        rtol=1e-4,
                    )
        assert wrapper.error.item() == wrapper.repair_error.item() == 0

    check((257, 0, 512))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.plan(qo, kv, table)
        wrapper.run(q, weights, fused)
    for lengths in ((4096, 0, 8192), (257, 0, 512)):
        kv.copy_(
            torch.tensor(
                [0, lengths[0] * ratio, lengths[0] * ratio, sum(lengths) * ratio],
                device=q.device,
                dtype=torch.int32,
            )
        )
        table.copy_(table.roll(1, dims=1))
        graph.replay()
        check(lengths)


@_requires_selective_sm100
def test_context_scaled_sample_eager_hints_and_device_replay(monkeypatch):
    inputs = list(_request_inputs([256], [65536], heads=64))
    q, _, _, weights, qo, kv = inputs
    fused, original_table = _paged_inputs(tuple(inputs), block_size=64)
    table = torch.zeros((1, 2560), device=q.device, dtype=torch.int32)
    table[:, : original_table.shape[1]].copy_(original_table)
    wrapper = FP8PagedMQATopKWrapper(
        strategy="selective",
        num_heads=64,
        page_size=64,
        top_k=2048,
        max_context_length=163840,
        max_chunk_size=256,
        max_batch_size=1,
        kv_layout=fused,
    )
    wrapper.plan(qo, kv, table)
    wrapper.prepare_pool(fused)
    sample_calls = []
    score = sl._fp8_mqa_selective_logits

    def record_score(*args, **kwargs):
        if kwargs.get("mode") == sl._SelectiveLogitsMode.SAMPLE:
            sample_calls.append(True)
        return score(*args, **kwargs)

    monkeypatch.setattr(sl, "_fp8_mqa_selective_logits", record_score)
    for length, count in ((2304, 256), (4096, 256), (8192, 256), (65536, 1024)):
        kv[1].fill_(length)
        for hints in (
            None,
            (
                torch.tensor([0, 256], dtype=torch.int32),
                torch.tensor([length], dtype=torch.int32),
            ),
        ):
            wrapper.plan(qo, kv, table, host_length_hints=hints)
            sample_calls.clear()
            selected = wrapper.run(q, weights, fused)
            _check_selection(selected, _oracle_rows(inputs), 2048)
            assert bool(sample_calls) == (hints is None or length > 6144)
            if length == 2304:
                assert wrapper._capacity_plan._engine._repair_flags.count_nonzero() == 0
            assert wrapper._capacity_plan._engine._sample_ends.unique().tolist() == [
                count
            ]
    output = torch.empty((256, 2048), device=q.device, dtype=torch.int32)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.plan(qo, kv, table)
        wrapper.run(q, weights, fused, out=output)
    for length, count in ((8192, 256), (65536, 1024)):
        # Device writes only: no new H2D metadata or host-derived sample count.
        kv[1].fill_(length)
        output.fill_(-2)
        graph.replay()
        _check_selection(output, _oracle_rows(inputs), 2048)
        assert wrapper._capacity_plan._engine._sample_ends.unique().tolist() == [count]
        assert wrapper.error.item() == wrapper.repair_error.item() == 0

    # A small eager bound must invalidate, not turn a larger real step into
    # an unchecked all-key bypass.
    kv[1].fill_(8192)
    wrapper.plan(
        qo,
        kv,
        table,
        host_length_hints=(
            torch.tensor([0, 256], dtype=torch.int32),
            torch.tensor([4096], dtype=torch.int32),
        ),
    )
    assert (wrapper.run(q, weights, fused) == -1).all()
    assert wrapper.error.item() != 0
    wrapper.plan(qo, kv, table)
    _check_selection(wrapper.run(q, weights, fused), _oracle_rows(inputs), 2048)


@_requires_selective_sm100
@pytest.mark.parametrize("q_rows", [1, 129])
@pytest.mark.parametrize("top_k,heads,ratio", [(2048, 64, 1), (512, 8, 4)])
def test_context_scaled_sample_expected_hit_floor(
    q_rows, top_k, heads, ratio, ieee_matmul
):
    physical_keys = 32 * top_k + 1 + (q_rows - 1) // ratio
    inputs = _request_inputs([q_rows], [physical_keys], heads=heads)
    q, keys, scales, weights, qo, _ = inputs
    kv = torch.tensor(
        [0, ratio * (32 * top_k + 1) + q_rows - 1],
        device=q.device,
        dtype=torch.int32,
    )
    fused, original_table = _paged_inputs(inputs)
    table = torch.zeros((1, 64 * top_k // 128), device=q.device, dtype=torch.int32)
    table[:, : original_table.shape[1]].copy_(original_table)
    wrapper = FP8PagedMQATopKWrapper(
        strategy="selective",
        num_heads=heads,
        page_size=128,
        top_k=top_k,
        compression_ratio=ratio,
        max_context_length=64 * top_k * ratio,
        max_chunk_size=q_rows,
        max_batch_size=1,
        kv_layout=fused,
    )
    wrapper.plan(qo, kv, table)
    wrapper.prepare_pool(fused)
    output = torch.empty((q_rows, top_k), device=q.device, dtype=torch.int32)

    def run_eager():
        wrapper.plan(qo, kv, table)
        wrapper.run(q, weights, fused, out=output)

    run_eager()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_eager()
    group = sl._select_sparse_sample_group_rows(
        wrapper._capacity_plan._engine._sample_prepared.query_tile
    )
    for first_end, expected in (
        (4 * top_k, 256),
        (8 * top_k, 256),
        (16 * top_k, 512),
        (16 * top_k + 1, 1024),
        (32 * top_k, 1024),
        (32 * top_k + 1, 3840),
    ):
        # Move only device metadata; the captured plan must recompute counts.
        kv[1].fill_(ratio * first_end + q_rows - 1)
        for refresh in (run_eager, graph.replay):
            output.fill_(-2)
            refresh()
            counts = wrapper._capacity_plan._engine._sample_ends[:q_rows].cpu()
            assert counts[0].item() == expected
            for first in range(0, q_rows, group):
                count = counts[first].item()
                end = first_end + first // ratio
                assert (counts[first : first + group] == count).all()
                assert count * top_k >= 32 * end
                assert count <= min(end, 3840)
            if ratio == 4:
                _check_compressed_wrapper(
                    output, (q, fused, weights, qo, kv, table), top_k
                )
            else:
                reference = _oracle_rows((q, keys, scales, weights, qo, kv))
                _check_selection(output, reference, top_k)
            assert wrapper.error.item() == wrapper.repair_error.item() == 0


def _sample_ladder_reference(samples, valid, ranks, ends, top_k, segments):
    """Independent histogram crossings and bank estimates; no production helper."""
    rows, device = valid.numel(), samples.device
    expected_ladder = torch.empty((rows, 3), device=device)
    expected_populations = torch.zeros((rows, 3), dtype=torch.int32, device=device)
    expected_bins = torch.full((rows,), 2047, dtype=torch.int32, device=device)
    expected_thresholds = torch.empty(rows, device=device)
    for row in range(rows):
        count = int(valid[row])
        bits = samples[row, :count].half().view(torch.uint16).long()
        keys = torch.where(bits & 0x8000 != 0, bits, (bits ^ 0xFFFF) & 0x7FFF)
        prefix = torch.bincount(keys, minlength=65536).cumsum(0)
        bins = []
        coarse_values, coarse_hits = [], []
        for rung in range(3):
            target = max(1, min(count, int(ranks[row])) // (4 >> rung))
            index = int(torch.searchsorted(prefix, target)) if count else 65535
            bins.append(index >> 5)
            coarse_hits.append(int(prefix[min((index | 31), 65535)]) if count else 0)
            floor = min(0xFC00, index | 31)
            encoding = floor if floor & 0x8000 else (floor ^ 0xFFFF) & 0x7FFF
            coarse_values.append(
                torch.tensor(encoding, dtype=torch.int32, device=device)
                .short()
                .view(torch.float16)
                .float()
                if count
                else torch.tensor(float("-inf"), device=device)
            )
            expected_ladder[row, rung] = float("-inf")
            if count:
                expected_ladder[row, rung] = samples[row, :count][keys == index].min()
                expected_populations[row, rung] = prefix[index]

        def fits(hits):
            sigma = math.sqrt(max(hits, 1))
            scale = int(ends[row]) / max(count, 1)
            lower = max(0, hits - 3 * sigma) * scale
            mean = (hits + 3 * sigma) * scale / segments
            return (
                lower >= top_k and mean + 3 * math.sqrt(max(mean, 1)) < 6144 // segments
            )

        needs_threshold = int(ends[row]) > min(6144, top_k + 256)
        refine = count > 0 and needs_threshold and not any(map(fits, coarse_hits))
        if not refine:
            expected_ladder[row] = torch.stack(coarse_values)
            expected_populations[row] = torch.tensor(coarse_hits, device=device)
        chosen = 2
        for rung in (2, 1, 0):
            hits = int(expected_populations[row, rung])
            sigma = math.sqrt(max(hits, 1))
            scale = int(ends[row]) / max(count, 1)
            lower = max(0, hits - 3 * sigma) * scale
            bank_mean = (hits + 3 * sigma) * scale / segments
            bank_upper = bank_mean + 3 * math.sqrt(max(bank_mean, 1))
            if lower >= top_k and bank_upper < 6144 // segments:
                chosen = rung
        expected_thresholds[row] = float("-inf")
        if count and needs_threshold:
            expected_thresholds[row] = expected_ladder[row, chosen]
            expected_bins[row] = bins[chosen]

    return expected_thresholds, expected_bins, expected_ladder, expected_populations


@_requires_selective_sm100
@pytest.mark.parametrize("segments", [4, 8, 16, 32, 64, 128])
def test_native_ladder_matches_signed_histogram(segments):
    """Crossings, inclusive populations, and bank policy match a host oracle."""
    from flashinfer.experimental.attn_scores.jit.selective_sample_ladder import (
        get_selective_sample_ladder_module,
    )

    gen = torch.Generator(device="cuda").manual_seed(1)
    samples = torch.randn((8, 1024), generator=gen, device="cuda") * 16
    samples[1].zero_()
    samples[2].fill_(-1e6)
    samples[3].fill_(1e6)
    samples[4].abs_().neg_()
    valid = torch.tensor(
        [0, 1024, 1024, 1024, 1, 511, 768, 1024], dtype=torch.int32, device="cuda"
    )
    ranks = torch.tensor(
        [1, 128, 256, 512, 1, 64, 192, 256], dtype=torch.int32, device="cuda"
    )
    ends = torch.tensor(
        [16, 8192, 16384, 16384, 8, 32768, 8192, 65536],
        dtype=torch.int32,
        device="cuda",
    )
    thresholds = torch.empty(8, device="cuda")
    bins = torch.empty(8, dtype=torch.int32, device="cuda")
    ladder = torch.empty((8, 3), device="cuda")
    populations = torch.empty((8, 3), dtype=torch.int32, device="cuda")
    actual = thresholds, bins, ladder, populations
    get_selective_sample_ladder_module().run(
        samples, valid, ranks, ends, *actual, 512, 6144, segments
    )
    expected = _sample_ladder_reference(samples, valid, ranks, ends, 512, segments)
    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, want, atol=0, rtol=0)


# Native sample ladder


@_requires_selective_sm100
def test_native_ladder_refines_boundary_bin_without_rounding_floor_up():
    """A dense coarse bin must not force the whole sampled population through."""
    from flashinfer.experimental.attn_scores.jit.selective_sample_ladder import (
        get_selective_sample_ladder_module,
    )

    positive = torch.linspace(8.001, 8.24, 3840, device="cuda")
    samples = torch.stack((positive, -positive))
    valid = torch.full((2,), 3840, dtype=torch.int32, device="cuda")
    ranks = torch.full_like(valid, 1024)
    ends = torch.full_like(valid, 32768)
    actual = (
        torch.empty(2, device="cuda"),
        torch.empty_like(valid),
        torch.empty((2, 3), device="cuda"),
        torch.empty((2, 3), dtype=torch.int32, device="cuda"),
    )
    module = get_selective_sample_ladder_module()

    def run():
        module.run(samples, valid, ranks, ends, *actual, 512, 6144, 4)

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(2):
        expected = _sample_ladder_reference(samples, valid, ranks, ends, 512, 4)
        graph.replay()
        for got, want in zip(actual, expected, strict=True):
            torch.testing.assert_close(got, want, rtol=0, atol=0)
        assert bool((actual[3] < valid[:, None]).all())
        # Populations describe the actual FP32 floor, not the rounded FP16
        # center (which can lie above some scores assigned to the same bin).
        observed = (samples[:, None, :] >= actual[2][:, :, None]).sum(-1)
        torch.testing.assert_close(observed.int(), actual[3], rtol=0, atol=0)
        samples.copy_(samples.flip(0).contiguous())


@_requires_selective_sm100
@pytest.mark.parametrize("top_k", [1, 31, 128, 512])
@pytest.mark.parametrize("segments", [4, 8, 12, 16, 32, 64, 128])
def test_cuda_sample_ladder_policy_and_graph(top_k, segments):
    from flashinfer.experimental.attn_scores.jit.selective_sample_ladder import (
        get_selective_sample_ladder_module,
    )

    torch.manual_seed(1)
    rows, width = 33, 3840
    # The scorer pads its row storage independently of logical policy rows.
    samples = torch.randn((rows + 3, width + 16), device="cuda")[:, :width]
    valid = torch.empty(rows, device="cuda", dtype=torch.int32)
    ranks = torch.empty_like(valid)
    ends = torch.empty_like(valid)

    def outputs():
        return (
            torch.empty(rows, device="cuda"),
            torch.empty_like(valid),
            torch.empty((rows, 3), device="cuda"),
            torch.empty((rows, 3), dtype=torch.int32, device="cuda"),
        )

    actual = outputs()
    module = get_selective_sample_ladder_module()

    def run():
        module.run(samples, valid, ranks, ends, *actual, top_k, 6144, segments)

    valid.fill_(width)
    ranks.fill_(128)
    ends.fill_(32768)
    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for case in range(5):
        if case == 0:
            samples.zero_()
        elif case == 1:
            samples.normal_()
        elif case == 2:
            samples.fill_(7.5)
            samples[:, ::7] = -3.0
        elif case == 3:
            samples.uniform_(-1e-7, 1e-7)
        else:
            # Values around half-precision bin boundaries, including infinities.
            values = torch.tensor(
                [
                    -float("inf"),
                    -65504,
                    -1.001,
                    -1.0,
                    -0.0,
                    0.0,
                    1.0,
                    1.001,
                    65504,
                    float("inf"),
                ],
                device="cuda",
            )
            samples.copy_(values[torch.arange(width, device="cuda") % len(values)])
        valid.copy_(torch.linspace(0, width, rows, device="cuda").int())
        ranks.copy_(torch.clamp(valid // 7, min=1))
        ends.copy_(torch.arange(rows, device="cuda", dtype=torch.int32) * 32768)
        ends[1] = top_k - 1
        ends[2] = top_k
        expected = _sample_ladder_reference(
            samples, valid, ranks, ends, top_k, segments
        )
        graph.replay()
        for want, got in zip(expected, actual, strict=True):
            torch.testing.assert_close(got, want, rtol=0, atol=0)


# Segmented candidate selection


@pytest.mark.parametrize("capacity", [6144, 8192])
@pytest.mark.parametrize("segments", [1, 4, 8, 12, 16, 32, 64, 128])
@pytest.mark.parametrize("top_k", [1, 8, 32, 128, 512, 2048])
def test_segmented_packed_topk(segments, top_k, capacity):
    if not torch.cuda.is_available() or not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("Requires SM100a")
    # One pool is H64-only at 8192; lane-local banks use 8192 split 32..128 ways.
    if capacity % segments or (segments == 1 and capacity != 8192):
        pytest.skip("not a production capacity/segment layout")
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    from flashinfer.experimental.attn_scores.kernels.selective_logits_metadata import (
        SelectivePackedTopKKernel,
    )

    torch.manual_seed(1)
    rows = 9
    bank = capacity // segments
    scores = torch.randn(rows, capacity, device="cuda")
    scores[5].zero_()
    indices = torch.arange(capacity, device="cuda", dtype=torch.int64).expand(rows, -1)
    values = (scores.view(torch.int32).long() << 32) | indices
    counts = torch.zeros((rows, segments + 1), dtype=torch.int32, device="cuda")
    counts[1, 1] = min(top_k - 1, bank)
    counts[2, 1:] = bank
    counts[3, 1] = bank + 1  # Segment overflow even below global capacity.
    counts[4, 1:] = bank // 2
    counts[5, 1:] = bank
    counts[6, 1:] = 1  # Inactive slots must never participate.
    short = min(top_k - 1, bank)
    counts[7, 1] = short  # Short causal row that published all of its keys.
    counts[8, 1] = short  # Short causal row missing its last key.
    row_ends = torch.full((rows,), 1 << 30, dtype=torch.int32, device="cuda")
    row_ends[7] = short
    row_ends[8] = short + 1
    lengths = torch.empty(rows, dtype=torch.int32, device="cuda")
    flags = torch.empty_like(lengths)
    selected = torch.empty((rows, top_k), dtype=torch.int32, device="cuda")
    kernel = SelectivePackedTopKKernel(top_k, capacity, segments)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    args = (
        from_dlpack(values, assumed_align=16),
        from_dlpack(lengths, assumed_align=16),
        from_dlpack(selected, assumed_align=16),
        cutlass.Int32(rows),
        from_dlpack(counts, assumed_align=16),
        from_dlpack(flags, assumed_align=16),
        from_dlpack(row_ends, assumed_align=16),
    )
    compiled = cute.compile(kernel.segmented, *args, stream)
    compiled(*args, stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled(*args, cuda.CUstream(torch.cuda.current_stream().cuda_stream))
    for replay in range(2):
        if replay:
            counts[:, 1:].zero_()
        graph.replay()
        keep = counts[:, 1:].clamp(max=bank)
        total = counts[:, 1:].sum(1)
        overflow = (counts[:, 1:] > bank).any(1)
        torch.testing.assert_close(lengths.long(), keep.sum(1))
        torch.testing.assert_close(
            counts[:, 0].long(), torch.where(overflow, capacity + 1, total)
        )
        # Only rows short of min(top_k, row_end) need repair.
        required = row_ends.long().clamp(max=top_k)
        expected_flags = (total < required) | (total > capacity) | overflow
        torch.testing.assert_close(flags.bool(), expected_flags)
        if not replay:
            # total < top_k alone would send the complete short row to repair.
            assert int(flags[7]) == 0 and int(flags[8]) == 1
        for row in range(rows):
            valid = (torch.arange(capacity, device="cuda") % bank) < keep[
                row
            ].repeat_interleave(bank)
            valid_indices = valid.nonzero().flatten()
            n = min(top_k, valid_indices.numel())
            got = selected[row, :n].long()
            assert got.unique().numel() == n
            assert valid[got].all()
            torch.testing.assert_close(
                scores[row, got].sort().values,
                scores[row, valid].topk(n).values.sort().values,
                atol=0,
                rtol=0,
            )
            assert (selected[row, n:] == -1).all()


@_requires_selective_sm100
def test_h64_candidate_variant_rejects_active_rows_after_cached_launch():
    """Dynamic metadata cannot reuse the H64 variant's static row schedule."""
    from flashinfer.experimental.attn_scores.selective_logits import (
        _fp8_mqa_selective_logits,
        _prepare_fp8_mqa_selective_logits,
        _refresh_fp8_mqa_selective_logits,
        _SelectiveLogitsMode,
    )

    inputs = _request_inputs((8,), (4096,), heads=64)
    q, _, _, weights, qo, kv = inputs
    fused, table = _paged_inputs(inputs)
    starts = torch.zeros(8, dtype=torch.int32, device=q.device)
    ends = torch.full_like(starts, 4096)
    prepared = _prepare_fp8_mqa_selective_logits(
        q,
        fused,
        weights,
        qo,
        kv,
        table,
        row_starts=starts,
        row_ends=ends,
        num_heads=64,
        query_tile=4,
        num_umma_stages=1,
        num_sms=2,
    )
    values = torch.empty((8, 8192), dtype=torch.int64, device=q.device)
    counts = torch.empty((8, 5), dtype=torch.int32, device=q.device)
    thresholds = torch.full((8,), float("-inf"), device=q.device)
    kwargs = dict(
        mode=_SelectiveLogitsMode.CANDIDATE,
        capacity=8192,
        values=values,
        index_counts=counts,
        policy_values=thresholds,
        _h64_candidate=True,
    )
    _fp8_mqa_selective_logits(q, fused, weights, table, prepared, **kwargs)
    torch.cuda.synchronize()
    assert bool((counts[:, 1:].sum(1) == 4096).all())
    active = torch.tensor([3], dtype=torch.int32, device=q.device)
    _refresh_fp8_mqa_selective_logits(
        prepared,
        qo,
        kv,
        row_starts=starts,
        row_ends=ends,
        active_rows=active,
    )
    with pytest.raises(ValueError, match="does not support active_rows metadata"):
        _fp8_mqa_selective_logits(q, fused, weights, table, prepared, **kwargs)


@_requires_selective_sm100
@pytest.mark.parametrize("page_size", [32, 64, 128])
@pytest.mark.parametrize("math_mode", ["relu_strict", "relu_fast"])
def test_h64_candidate_variant_page_sizes_and_partial_tail(
    page_size, math_mode, ieee_matmul
):
    """Subpage assembly, scale addressing and map replay agree with FP32."""
    inputs = list(_inputs(q_rows=37, kv_rows=16387, num_heads=64))
    q, kv, _scale, weights, _, _ = inputs
    inputs[4] = torch.tensor([0, 37], device="cuda", dtype=torch.int32)
    inputs[5] = torch.tensor([0, kv.shape[0]], device="cuda", dtype=torch.int32)
    fused, natural_table = _paged_inputs(tuple(inputs), block_size=page_size)
    table = torch.full(
        (1, natural_table.shape[1] + 4),
        fused.shape[0] + 17,
        device=q.device,
        dtype=torch.int32,
    )
    live_pages = (kv.shape[0] + page_size - 1) // page_size
    table[:, :live_pages].copy_(natural_table[:, :live_pages])
    raw = fused.flatten(1)
    last_page = int(natural_table[0, live_pages - 1])
    raw[last_page, : page_size * 128].view(page_size, 128)[
        kv.shape[0] % page_size :
    ].fill_(0x7F)
    engine = SelectiveEngineCase(math_mode=math_mode)
    engine.plan(
        q,
        fused,
        weights,
        inputs[4],
        inputs[5],
        table,
        top_k=512,
        num_heads=64,
        head_dim=128,
        num_sms=2,
    )
    assert engine._candidate_schedule.h64_candidate
    selected = engine.run(q, fused, weights, table)
    torch.cuda.synchronize()
    _check_selection(selected, _oracle_rows(inputs), 512)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        selected = engine.run(q, fused, weights, table)
    for ties in (False, True):
        if ties:
            weights.zero_()
        else:
            table[:, :2].copy_(table[:, :2].flip(1))
            weights.mul_(0.75)
        graph.replay()
        torch.cuda.synchronize()
        expected = _paged_step_oracle(q, weights, fused, table, [0, 37], [kv.shape[0]])
        _check_selection(selected, expected, 512)
        if ties:
            assert engine._repair_flags.all()
            assert engine._second_repair_flags.all()
        assert engine._exact_error.item() == 0


@_requires_selective_sm100
@pytest.mark.parametrize("top_k", [512, 2048])
@pytest.mark.parametrize("ratio", [1, 4])
@pytest.mark.parametrize("h64_candidate", [False, True])
def test_single_pool8192_changed_map_tail_graph(
    top_k, ratio, h64_candidate, monkeypatch, ieee_matmul
):
    """Split-1 H64 candidates, partial Q/page and mutable map replay.

    The pooled layout publishes one repair-wide pool; the specialized H64
    candidate variant splits the same backing into four warp segments.
    """
    physical = 16387
    q, kv, scales, weights, _, _ = _inputs(9, physical, 64)
    cu_q = torch.tensor([0, 9], device="cuda", dtype=torch.int32)
    cu_physical = torch.tensor([0, physical], device="cuda", dtype=torch.int32)
    fused, table = _paged_inputs((q, kv, scales, weights, cu_q, cu_physical))
    tokens = physical * ratio + ratio - 1
    cu_tokens = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)
    raw = fused.flatten(1)
    last = int(table[0, -1].item())
    raw[last, : 128 * 128].view(128, 128)[physical % 128 :].fill_(0x7F)
    wrapper = SelectiveEngineCase()
    _force_h64_candidate_layout(monkeypatch, h64_candidate)
    wrapper.plan(
        q,
        fused,
        weights,
        cu_q,
        cu_tokens,
        table,
        top_k=top_k,
        num_heads=64,
        num_sms=4,
        compression_ratio=ratio,
    )
    assert wrapper._candidate_schedule.candidate_capacity == 8192
    assert wrapper._candidate_schedule.sample_count_segments == 32
    assert wrapper._candidate_schedule.h64_candidate is h64_candidate
    assert wrapper._candidate_counts.shape[1] == (5 if h64_candidate else 2)
    assert wrapper._candidate_values.shape[1] == 8192
    assert wrapper._sample_threshold_launch[1][-2:] == (
        wrapper._candidate_schedule.candidate_capacity,
        wrapper._candidate_schedule.count_segments,
    )
    assert not wrapper._compact_repair
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = wrapper.run(q, fused, weights, table)
    for changed in (False, True):
        if changed:
            table[:, :2].copy_(table[:, :2].flip(1))
            weights.mul_(0.75)
        graph.replay()
        torch.cuda.synchronize()
        mapped = raw[table[0].long()]
        keys = mapped[:, : 128 * 128].contiguous().view(q.dtype).reshape(-1, 128)
        scale = mapped[:, 128 * 128 :].contiguous().view(torch.float32).reshape(-1)
        for row in range(9):
            end = (tokens - 9 + row + 1) // ratio
            scores = (
                (q[row].float() @ keys[:end].float().T).relu_() * weights[row, :, None]
            ).sum(0) * scale[:end]
            ids = output[row].long()
            assert ((ids >= 0) & (ids < end)).all() and ids.unique().numel() == top_k
            torch.testing.assert_close(
                scores[ids].sort().values,
                scores.topk(top_k).values.sort().values,
                atol=5e-5,
                rtol=1e-5,
            )


@_requires_selective_sm100
@pytest.mark.parametrize("top_k", [512, 1024, 2048])
@pytest.mark.parametrize("ratio", [1, 4])
def test_lane_local8192_changed_map_tail_graph(top_k, ratio, monkeypatch, ieee_matmul):
    """K>=1024 H8 prefill lane banks use the repair-wide backing; K512 keeps 6144."""
    physical, rows = 16387, 33
    q, kv, scales, weights, _, _ = _inputs(rows, physical, 8)
    cu_q = torch.tensor([0, rows], device="cuda", dtype=torch.int32)
    cu_physical = torch.tensor([0, physical], device="cuda", dtype=torch.int32)
    fused, table = _paged_inputs((q, kv, scales, weights, cu_q, cu_physical))
    tokens = physical * ratio + ratio - 1
    cu_tokens = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)
    raw = fused.flatten(1)
    last = int(table[0, -1].item())
    raw[last, : 128 * 128].view(128, 128)[physical % 128 :].fill_(0x7F)
    wrapper = SelectiveEngineCase()
    wrapper.plan(
        q,
        fused,
        weights,
        cu_q,
        cu_tokens,
        table,
        top_k=top_k,
        num_heads=8,
        num_sms=4,
        compression_ratio=ratio,
    )
    schedule = wrapper._candidate_schedule
    assert schedule.publication == "lane_local"
    assert schedule.candidate_capacity == (8192 if top_k >= 1024 else 6144)
    assert wrapper._candidate_counts.shape[1] == 1 + 32 * schedule.split_kv
    assert wrapper._candidate_values.shape[1] == 8192
    assert wrapper._sample_threshold_launch[1][-2:] == (
        schedule.candidate_capacity,
        schedule.count_segments,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = wrapper.run(q, fused, weights, table)
    for changed in (False, True):
        if changed:
            table[:, :2].copy_(table[:, :2].flip(1))
            weights.mul_(0.75)
        graph.replay()
        torch.cuda.synchronize()
        mapped = raw[table[0].long()]
        keys = mapped[:, : 128 * 128].contiguous().view(q.dtype).reshape(-1, 128)
        scale = mapped[:, 128 * 128 :].contiguous().view(torch.float32).reshape(-1)
        for row in range(rows):
            end = (tokens - rows + row + 1) // ratio
            scores = (
                (q[row].float() @ keys[:end].float().T).relu_() * weights[row, :, None]
            ).sum(0) * scale[:end]
            ids = output[row].long()
            assert ((ids >= 0) & (ids < end)).all() and ids.unique().numel() == top_k
            torch.testing.assert_close(
                scores[ids].sort().values,
                scores.topk(top_k).values.sort().values,
                atol=5e-5,
                rtol=1e-5,
            )


@pytest.mark.parametrize(
    ("heads", "dimension", "tile", "page", "top_k", "long_query", "expected"),
    [
        (4, 128, 16, 128, 1024, True, True),
        (4, 128, 16, 128, 2048, True, True),
        (4, 128, 16, 128, 512, True, False),
        (4, 128, 16, 128, 2048, False, False),
        (4, 128, 8, 128, 2048, True, False),
        (4, 128, 16, 64, 2048, True, False),
        (4, 256, 16, 128, 2048, True, False),
        (8, 128, 16, 128, 2048, True, False),
        (32, 128, 16, 128, 2048, True, False),
        (64, 128, 16, 128, 2048, True, False),
    ],
)
def test_h4_pooled_candidate_gate(
    heads, dimension, tile, page, top_k, long_query, expected
):
    from flashinfer.experimental.attn_scores.selective_logits import (
        _h4_pooled_candidate_serves,
    )

    assert (
        _h4_pooled_candidate_serves(heads, dimension, tile, page, top_k, long_query)
        is expected
    )


@_requires_selective_sm100
def test_capacity_h4_pooled_candidate_removes_lane_bank_overflow(monkeypatch):
    """Four striped warp pools absorb a skew that overflows lane quotas."""
    from flashinfer import FP8PagedMQATopKWrapper
    import flashinfer.experimental.attn_scores.selective_logits as sl

    physical, rows, ratio = 131072, 32, 4
    q, kv, scales, weights, _, _ = _inputs(rows, physical, 4)
    q.fill_(1)
    weights.fill_(1)
    scales.fill_(1)
    signs = torch.where(torch.arange(physical, device="cuda") % 128 < 4, 1.0, -1.0)
    kv.copy_(signs[:, None].expand(-1, 128).to(kv.dtype))
    qo = torch.tensor([0, rows], dtype=torch.int32, device="cuda")
    physical_prefix = torch.tensor([0, physical], dtype=torch.int32, device="cuda")
    fused, table = _paged_inputs((q, kv, scales, weights, qo, physical_prefix))
    tokens = physical * ratio
    kv_prefix = torch.tensor([0, tokens], dtype=torch.int32, device="cuda")
    with monkeypatch.context() as control:
        control.setattr(sl, "_h4_pooled_candidate_serves", lambda *args: False)
        previous = sl._select_candidate_schedule(
            q.device,
            4,
            128,
            [rows],
            [physical],
            num_sms=torch.cuda.get_device_properties(q.device).multi_processor_count,
            top_k=2048,
        )
    wrapper = FP8PagedMQATopKWrapper(
        strategy="selective",
        num_heads=4,
        head_dim=128,
        page_size=128,
        top_k=2048,
        compression_ratio=ratio,
        max_context_length=tokens,
        max_chunk_size=rows,
        max_batch_size=1,
        kv_layout=fused,
    )
    wrapper.plan(qo, kv_prefix, table)
    wrapper.prepare_pool(fused)
    engine = wrapper._capacity_plan._engine
    schedule = engine._candidate_schedule
    assert (schedule.publication, schedule.count_segments, schedule.split_kv) == (
        "warp_striped",
        4,
        1,
    )
    assert schedule.candidate_capacity == 8192
    assert schedule.sample_count_segments == previous.sample_count_segments
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        selected = wrapper.run(q, weights, fused)
    for factor in (1.0, 0.75):
        weights.fill_(factor)
        graph.replay()
        scores = (q[0].float() @ kv.float().T).relu().mul(weights[0, :, None]).sum(0)
        for row in range(rows):
            end = (tokens - rows + row + 1) // ratio
            ids = selected[row].long()
            assert ((ids >= 0) & (ids < end)).all()
            assert ids.unique().numel() == 2048
            torch.testing.assert_close(
                scores[ids].sort().values,
                scores[:end].topk(2048).values.sort().values,
                rtol=0,
                atol=0,
            )
        assert not bool(engine._repair_flags[:rows].any())
        assert int(wrapper.error) == int(wrapper.repair_error) == 0


@_requires_selective_sm100
@pytest.mark.parametrize("segments", [0, 4])
@pytest.mark.parametrize("top_k", [7, 512, 2048])
def test_finalizer_flags_only_incomplete_rows(segments, top_k):
    """Single-row plans finalize candidates; short complete rows stay unflagged."""
    from flashinfer.experimental.attn_scores.selective_logits import (
        _finalize_fp8_mqa_packed_candidates,
    )

    from flashinfer.experimental.attn_scores._selective_contract import _REPAIR_CAPACITY

    capacity, rows = _REPAIR_CAPACITY, 6
    span = capacity // max(segments, 1)
    totals = [0, top_k - 1, top_k - 1, top_k, top_k + 3, 0]
    ends = [0, top_k - 1, top_k, 1 << 30, 1 << 30, 1 << 30]
    counts = torch.zeros((rows, segments + 1), dtype=torch.int32, device="cuda")
    for row, total in enumerate(totals):
        if segments:
            counts[row, 1] = min(total, span)
            counts[row, 2] = total - min(total, span)
        else:
            counts[row, 0] = total
    row_ends = torch.tensor(ends, dtype=torch.int32, device="cuda")
    expected_new = [t < min(top_k, e) for t, e in zip(totals, ends, strict=True)]
    expected_old = [t < top_k for t in totals]
    # The short complete row distinguishes the two conditions.
    assert expected_new[1] != expected_old[1]
    for ends_arg, expected in ((row_ends, expected_new), (None, expected_old)):
        values = torch.zeros((rows, capacity), dtype=torch.int64, device="cuda")
        lengths = torch.empty(rows, dtype=torch.int32, device="cuda")
        flags = torch.empty(rows, dtype=torch.int32, device="cuda")
        _finalize_fp8_mqa_packed_candidates(
            counts.clone(),
            values,
            lengths,
            flags,
            rows=rows,
            capacity=capacity,
            top_k=top_k,
            row_ends=ends_arg,
        )
        torch.cuda.synchronize()
        assert flags.bool().tolist() == expected


# Exact repair


@_requires_selective_cuda
def test_repair_threshold_negative_overflow_is_not_nan():
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    from flashinfer.experimental.attn_scores.kernels.selective_logits_metadata import (
        SelectiveRepairThresholdKernel,
    )
    from flashinfer.experimental.attn_scores.selective_logits import _current_stream

    counts = torch.zeros((3, 1), dtype=torch.int32, device="cuda")
    flags = torch.ones(3, dtype=torch.int32, device="cuda")
    thresholds = torch.full((3,), -60000.0, device="cuda")
    bins = torch.tensor([2008, 2016, 2047], dtype=torch.int32, device="cuda")
    ends = torch.full((3,), 8192, dtype=torch.int32, device="cuda")
    repaired_thresholds = torch.empty_like(thresholds)
    repaired_ends = torch.empty_like(ends)
    tensors = (
        counts,
        flags,
        thresholds,
        bins,
        ends,
        repaired_thresholds,
        repaired_ends,
    )
    args = tuple(
        from_dlpack(t).mark_layout_dynamic(leading_dim=1)
        if t.ndim == 2
        else from_dlpack(t).mark_compact_shape_dynamic(mode=0, divisibility=1)
        for t in tensors
    ) + (cutlass.Int32(3),)
    stream = _current_stream(counts.device)
    kernel = cute.compile(SelectiveRepairThresholdKernel(512), *args, stream)
    kernel(*args, stream)
    assert torch.isneginf(repaired_thresholds).all()
    torch.testing.assert_close(repaired_ends, ends)


def _prepare_radix_pass(
    histogram, prefixes, ranks, thresholds, *, clear_histogram, shift
):
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    from flashinfer.experimental.attn_scores.kernels.selective_logits_metadata import (
        SelectiveTopKRadixSelectKernel,
    )
    from flashinfer.experimental.attn_scores.selective_logits import _current_stream

    args = (
        from_dlpack(histogram).mark_layout_dynamic(leading_dim=1),
        *(
            from_dlpack(t).mark_compact_shape_dynamic(mode=0, divisibility=1)
            for t in (prefixes, ranks, thresholds)
        ),
        cutlass.Int32(histogram.shape[0]),
    )
    stream = _current_stream(histogram.device)
    compiled = cute.compile(
        SelectiveTopKRadixSelectKernel(clear_histogram=clear_histogram),
        *args,
        cutlass.Int32(shift),
        stream,
    )
    return compiled, args, stream


@_requires_selective_cuda
@pytest.mark.parametrize("clear_histogram", [False, True])
def test_exact_radix_starts_a_new_prefix_on_each_first_pass(clear_histogram):
    """Compact repair can reuse a slot whose previous score key was nonzero."""
    import cutlass

    histogram = torch.zeros((1, 256), dtype=torch.int32, device="cuda")
    prefixes = torch.full((1,), -559038737, dtype=torch.int32, device="cuda")
    ranks = torch.full((1,), 8, dtype=torch.int32, device="cuda")
    thresholds = torch.empty_like(prefixes)
    compiled, args, stream = _prepare_radix_pass(
        histogram,
        prefixes,
        ranks,
        thresholds,
        clear_histogram=clear_histogram,
        shift=24,
    )
    for key in (0xC4800000, 0x3B7FFFFF, 0xBF800000, 0x407FFFFF, 0xC4800000):
        ranks.fill_(8)
        for shift in (24, 16, 8, 0):
            if not clear_histogram:
                histogram.zero_()
            histogram[0, (key >> shift) & 255] = 512
            compiled(*args, cutlass.Int32(shift), stream)
            observed = int(prefixes.item()) & 0xFFFFFFFF
            assert observed == key >> shift
            if clear_histogram:
                assert torch.count_nonzero(histogram).item() == 0
        expected_bits = key ^ (0x80000000 if key & 0x80000000 else 0xFFFFFFFF)
        assert (int(thresholds.item()) & 0xFFFFFFFF) == expected_bits


@_requires_selective_cuda
@pytest.mark.parametrize("rows", [33, 257])
def test_exact_radix_consume_reset_matches_histogram_reference(rows):
    import cutlass

    torch.manual_seed(1)
    histogram = torch.zeros((rows, 256), dtype=torch.int32, device="cuda")
    prefixes = torch.zeros(rows, dtype=torch.int32, device="cuda")
    ranks = torch.zeros_like(prefixes)
    thresholds = torch.full_like(prefixes, -1)
    compiled, args, stream = _prepare_radix_pass(
        histogram, prefixes, ranks, thresholds, clear_histogram=True, shift=0
    )
    for active in (rows, 0, 1, rows // 2, rows):
        histogram.random_(0, 17)
        histogram[active:].zero_()
        ranks.copy_(histogram.sum(1).clamp(min=1).div(2, rounding_mode="floor"))
        initial_ranks = ranks.clone()
        descending = histogram.flip(1).cumsum(1)
        offset = (descending >= ranks[:, None]).to(torch.int32).argmax(1)
        expected_bins = 255 - offset
        before = torch.where(
            offset > 0,
            descending.gather(1, (offset - 1).clamp(min=0)[:, None]).flatten(),
            0,
        )
        prefixes.fill_(0x123456)
        compiled(*args, cutlass.Int32(0), stream)
        torch.testing.assert_close(
            prefixes[:active].long(), (0x123456 << 8) | expected_bins[:active]
        )
        torch.testing.assert_close(
            ranks[:active].long(), initial_ranks[:active] - before[:active]
        )
        torch.testing.assert_close(thresholds[:active], ~prefixes[:active])
        assert torch.count_nonzero(histogram).item() == 0
        assert (prefixes[active:] == 0x123456).all()


# Repair metadata and queue reset


@_requires_selective_sm100
@pytest.mark.parametrize("participants,threads", [(288, 384), (544, 640)])
def test_selective_tmem_warp_role_barrier(participants, threads):
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    from cutlass.utils import SmemAllocator

    from flashinfer.experimental.attn_scores.kernels._fp8_mqa_selective import (
        sync_warp_roles,
    )
    from flashinfer.experimental.attn_scores.selective_logits import _current_stream

    class RoleBarrier:
        @cute.kernel
        def kernel(self, output):
            tid, _, _ = cute.arch.thread_idx()
            shared = SmemAllocator().allocate_tensor(
                element_type=cutlass.Int32,
                layout=cute.make_layout((1,)),
                byte_alignment=4,
            )
            if tid == 0:
                shared[0] = cutlass.Int32(17)
            if tid < participants - 32:
                sync_warp_roles(1, participants)
                output[tid] = shared[0]
                sync_warp_roles(2, participants - 32)
            elif (tid >= participants + 32) & (tid < participants + 64):
                sync_warp_roles(1, participants)
                output[tid] = shared[0]

        @cute.jit
        def __call__(self, output, stream):
            self.kernel(output).launch(
                grid=(1, 1, 1), block=(threads, 1, 1), stream=stream
            )

    output = torch.zeros(threads, dtype=torch.int32, device="cuda")
    args = (from_dlpack(output),)
    compiled = cute.compile(RoleBarrier(), *args, _current_stream(output.device))
    compiled.to()
    module = compiled.jit_module
    owner = weakref.ref(compiled)
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            compiled(*args, _current_stream(output.device))
        expected = torch.zeros_like(output)
        expected[: participants - 32] = 17
        expected[participants + 32 : participants + 64] = 17
        for _ in range(2):
            output.zero_()
            graph.replay()
            torch.testing.assert_close(output, expected)
    finally:
        torch.cuda.synchronize()
        graph.reset()
        # CuTe's executor can keep its module in a bound-method cycle. Retire
        # this test-owned library before another caller starts graph capture.
        module.unload()
    del compiled
    assert owner() is None
    assert module.is_unloaded() and not module.cuda_library


@_requires_selective_sm100
@pytest.mark.parametrize("query_tile", [4, 16])
def test_compact_metadata_partial_tiles_bound_flag_reads(query_tile):
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    from flashinfer.experimental.attn_scores.kernels.selective_logits_metadata import (
        SelectiveLogitsCompactMetadataScheduleKernel,
    )
    from flashinfer.experimental.attn_scores.selective_logits import _current_stream

    qo = torch.tensor([0, 1, 3], dtype=torch.int32, device="cuda")
    starts = torch.zeros(3, dtype=torch.int32, device="cuda")
    ends = torch.tensor([11, 22, 33], dtype=torch.int32, device="cuda")
    # Leave the backing tail uninitialized so initcheck also detects a flag
    # load evaluated outside a partial request tile, even if its result is masked.
    flag_storage = torch.empty(3 + query_tile, dtype=torch.int32, device="cuda")
    flags = flag_storage[:3]
    flags.zero_()
    contexts = torch.empty(2, dtype=torch.int32, device="cuda")
    metadata = torch.empty((2, 4 + 2 * query_tile), dtype=torch.int32, device="cuda")
    schedule = torch.empty((5, 2), dtype=torch.int32, device="cuda")
    prefix = torch.empty(2, dtype=torch.int32, device="cuda")
    args = (
        *(
            from_dlpack(t)
            for t in (qo, starts, ends, flags, contexts, metadata, schedule, prefix)
        ),
        cutlass.Int32(2),
        cutlass.Int32(3),
    )
    stream = _current_stream(qo.device)
    compiled = cute.compile(
        SelectiveLogitsCompactMetadataScheduleKernel(4, query_tile, 256, paged=True),
        *args,
        stream,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled(*args, _current_stream(qo.device))
    for pattern in ([0, 0, 0], [1, 0, 1], [1, 1, 1], [0, 1, 0]):
        flags.copy_(torch.tensor(pattern, device=qo.device, dtype=torch.int32))
        graph.replay()
        expected = []
        for request, (begin, end) in enumerate(((0, 1), (1, 3))):
            if any(pattern[begin:end]):
                bounds = [
                    ends[i].item() if pattern[i] else 0 for i in range(begin, end)
                ]
                expected.append(
                    [begin, request, end - begin, 0]
                    + [0] * query_tile
                    + bounds
                    + [0] * (query_tile - len(bounds))
                )
        count = len(expected)
        assert (contexts[count:] == 0).all()
        if count:
            actual = metadata[:count]
            actual = actual[actual[:, 0].argsort()]
            torch.testing.assert_close(
                actual, torch.tensor(expected, device=qo.device, dtype=torch.int32)
            )


@_requires_selective_sm100
@pytest.mark.parametrize("query_tile", [1, 2, 4, 16])
@pytest.mark.parametrize("use_active_count", [False, True])
def test_active_metadata_graph_replay_and_prefix_bounds(query_tile, use_active_count):
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    from flashinfer.experimental.attn_scores.kernels.selective_logits_metadata import (
        SelectiveLogitsExplicitMetadataScheduleKernel,
    )
    from flashinfer.experimental.attn_scores.selective_logits import _current_stream

    rows, sms = 33, 4
    tiles = (rows + query_tile - 1) // query_tile
    cu_q = torch.tensor([0, rows], dtype=torch.int32, device="cuda")
    cu_kv = torch.tensor([0, 8192], dtype=torch.int32, device="cuda")
    starts = torch.zeros(rows, dtype=torch.int32, device="cuda")
    ends = torch.full_like(starts, 8192)
    count = torch.zeros(1, dtype=torch.int32, device="cuda")
    contexts = torch.full((tiles,), -1, dtype=torch.int32, device="cuda")
    metadata = torch.full(
        (tiles, 4 + 2 * query_tile), -1, dtype=torch.int32, device="cuda"
    )
    schedule = torch.empty((sms + 1, 2), dtype=torch.int32, device="cuda")
    prefix_storage = torch.full((tiles + 32,), -777, dtype=torch.int32, device="cuda")
    prefix = prefix_storage[:tiles]
    args = (
        from_dlpack(cu_q),
        from_dlpack(cu_kv),
        from_dlpack(starts),
        from_dlpack(ends),
        cutlass.Int32(0),
        from_dlpack(count) if use_active_count else None,
        from_dlpack(contexts),
        from_dlpack(metadata),
        from_dlpack(schedule),
        from_dlpack(prefix),
        cutlass.Int32(1),
    )
    stream = _current_stream(cu_q.device)
    compiled = cute.compile(
        SelectiveLogitsExplicitMetadataScheduleKernel(sms, query_tile, 256, paged=True),
        *args,
        stream,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled(*args, _current_stream(cu_q.device))
    for requested in (-1, 0, 1, 17, 33, 40, 2, 0):
        count.fill_(requested)
        graph.replay()
        active = min(max(requested, 0), rows) if use_active_count else rows
        live_tiles = (active + query_tile - 1) // query_tile
        assert bool((prefix_storage[tiles:] == -777).all())
        assert bool((contexts[:live_tiles] == 8192).all())
        expected = []
        total = live_tiles * 32
        for sm in range(sms + 1):
            work = sm * (total // sms) + min(sm, total % sms)
            expected.append((work // 32, work % 32))
        torch.testing.assert_close(
            schedule, torch.tensor(expected, device="cuda", dtype=torch.int32)
        )
        if live_tiles:
            assert (
                int(metadata[live_tiles - 1, 2])
                == active - (live_tiles - 1) * query_tile
            )
            assert int(prefix[live_tiles - 1]) == total


# Conditional repair


@_requires_selective_sm100
@pytest.mark.parametrize("compact", [False, True])
def test_repair_if_skips_body_and_resets_visible_state(compact):
    from flashinfer.experimental.attn_scores.jit.selective_repair_if import (
        get_selective_repair_if_module,
    )

    module = get_selective_repair_if_module()
    flags = torch.zeros(33, dtype=torch.int32, device="cuda")
    second = torch.empty_like(flags)
    error = torch.empty(1, dtype=torch.int32, device="cuda")
    first_count = torch.empty_like(error) if compact else None
    second_count = torch.empty_like(error) if compact else None
    counter = torch.zeros_like(error)
    observed = torch.empty_like(counter)
    body_owner = module.create_body_stream(flags.device.index)
    body_stream = torch.cuda.ExternalStream(body_owner(), device=flags.device)
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        module.preflight_body(flags, body_stream.cuda_stream)
        state = module.begin_if(flags, second, error, first_count, second_count)
        module.begin_body(flags, body_stream.cuda_stream, state[2])
        with torch.cuda.stream(body_stream):
            counter.add_(1)
        module.end_body(flags, body_stream.cuda_stream, state[2])
        module.finish_if(flags, state)
        # This consumer must depend on the conditional body, not its predicate.
        observed.copy_(counter)
    for active in (0, 1, 33, 0):
        flags.zero_()
        flags[:active].fill_(1)
        second.fill_(17)
        error.fill_(19)
        counter.zero_()
        if compact:
            first_count.fill_(23)
            second_count.fill_(29)
        graph.replay()
        assert observed.item() == (active > 0)
        assert error.item() == 0
        assert not second.any().item()
        if compact:
            assert first_count.item() == second_count.item() == 0


# Sampling skew and fused repair


def _assert_repair_position_storage(engine):
    """Check exact views without inspecting expired scratch payloads."""
    positions = engine._repair_topk_positions_i32
    whole, greater, equal = (
        engine._exact_indices,
        engine._greater_indices,
        engine._equal_indices,
    )
    rows, top_k = engine._rows, engine._top_k
    bank_rows = greater.shape[0]
    assert positions.shape == (rows, top_k)
    assert positions.dtype == greater.dtype == equal.dtype == torch.int32
    assert positions.is_contiguous() and positions.stride() == (top_k, 1)
    assert greater.is_contiguous() and equal.is_contiguous()
    assert greater.stride() == equal.stride() == (top_k, 1)
    assert greater.shape == equal.shape == (bank_rows, top_k)
    assert whole.shape == (2 * bank_rows, top_k)
    assert bank_rows >= rows and bank_rows % 4 == 0
    assert greater.storage_offset() == whole.storage_offset() == 0
    assert equal.storage_offset() == bank_rows * top_k
    assert equal.data_ptr() - greater.data_ptr() == bank_rows * top_k * 4
    assert greater.untyped_storage().data_ptr() == equal.untyped_storage().data_ptr()
    h64 = (
        engine._candidate_prepared.num_heads == 64
        and engine._candidate_prepared.head_dim == 128
    )
    if h64:
        assert (
            positions.untyped_storage().data_ptr() == whole.untyped_storage().data_ptr()
        )
        assert positions.data_ptr() == greater.data_ptr()
        assert positions.storage_offset() == greater.storage_offset()
        assert positions.numel() * positions.element_size() <= bank_rows * top_k * 4
        assert positions.untyped_storage().nbytes() == whole.numel() * 4
    else:
        assert (
            positions.untyped_storage().data_ptr() != whole.untyped_storage().data_ptr()
        )


@_requires_selective_sm100
@pytest.mark.parametrize("page_size", [32, 64, 128])
@pytest.mark.parametrize("heads", [4, 8, 64])
def test_internal_sample_ragged_page_lookup(page_size, heads):
    import flashinfer.experimental.attn_scores.selective_logits as sl

    head_dim = 64 if heads == 4 else 128
    data = _inputs(17, 9001, heads, head_dim)
    q, kv, scales, weights, cu_q, cu_kv = data
    fused, table = _paged_inputs(data, page_size)
    wrapper = SelectiveEngineCase()
    wrapper.plan(
        q,
        fused,
        weights,
        cu_q,
        cu_kv,
        table,
        num_heads=heads,
        head_dim=head_dim,
        num_sms=4,
    )
    for _ in range(2):
        wrapper.run(q, fused, weights, table)
        assert wrapper._exact_error.item() == 0
    # Independent eager sample oracle for a short request and a long request.
    for q_begin, q_end, kv_begin, kv_end in ((0, 1, 0, 132), (1, 17, 132, 9001)):
        length = kv_end - kv_begin
        pages = math.ceil(length / page_size)
        count = min(pages, sl._SAMPLE_CAPACITY // page_size)
        sampled = [
            page * page_size + offset
            for page in (i * (pages - 1) // max(count - 1, 1) for i in range(count))
            for offset in range(min(page_size, length - page * page_size))
        ]
        offsets = torch.tensor(sampled, device=q.device, dtype=torch.long)
        scores = torch.einsum(
            "qhd,kd->qhk", q[q_begin:q_end].float(), kv[kv_begin + offsets].float()
        ).relu_()
        scores = (scores * weights[q_begin:q_end, :, None]).sum(1)
        scores *= scales[kv_begin + offsets][None]
        ends = wrapper._sample_ends[q_begin:q_end]
        valid = torch.arange(len(sampled), device=q.device)[None] < ends[:, None]
        torch.testing.assert_close(
            wrapper._sample_scores[q_begin:q_end, : len(sampled)][valid],
            scores[valid],
            atol=2e-5,
            rtol=2e-5,
        )


@_requires_selective_sm100
@pytest.mark.parametrize(
    "page_size,top_k,high_on_sampled",
    [
        (32, 512, False),
        (32, 8, True),
        (64, 128, False),
        (64, 32, True),
        (128, 8, False),
        (128, 512, True),
    ],
)
def test_engine_page_correlated_sampling(page_size, top_k, high_on_sampled):
    rows, length, heads, dim = 17, 9001, 8, 128
    pages = math.ceil(length / page_size)
    sample_pages = min(pages, 3840 // page_size)
    selected_pages = {
        i * (pages - 1) // max(sample_pages - 1, 1) for i in range(sample_pages)
    }
    values = torch.tensor(
        [
            float((i // page_size in selected_pages) == high_on_sampled)
            for i in range(length)
        ],
        device="cuda",
    )
    q = torch.ones((rows, heads, dim), device="cuda").to(torch.float8_e4m3fn)
    kv = values[:, None].expand(length, dim).contiguous().to(torch.float8_e4m3fn)
    scales = torch.ones(length, device="cuda")
    weights = torch.ones((rows, heads), device="cuda")
    cu_q = torch.tensor([0, rows], device="cuda", dtype=torch.int32)
    cu_kv = torch.tensor([0, length], device="cuda", dtype=torch.int32)
    fused, table = _paged_inputs((q, kv, scales, weights, cu_q, cu_kv), page_size)
    wrapper = SelectiveEngineCase()
    wrapper.plan(q, fused, weights, cu_q, cu_kv, table, top_k=top_k, num_sms=4)
    # Independent eager FP32 score oracle, not the sampler or either selector.
    scores = torch.einsum("qhd,kd->qhk", q.float(), kv.float()).relu_()
    scores = (scores * weights[:, :, None]).sum(1) * scales[None]
    for _ in range(3):
        indices = wrapper.run(q, fused, weights, table).long()
        assert wrapper._exact_error.item() == 0
        for row in range(rows):
            end = length - rows + row + 1
            assert ((indices[row] >= 0) & (indices[row] < end)).all()
            assert indices[row].unique().numel() == top_k
            torch.testing.assert_close(
                scores[row, indices[row]].sort().values,
                scores[row, :end].topk(top_k).values.sort().values,
                atol=0,
                rtol=0,
            )


# Chain replanning


# Selective payload lifecycle


# Capture and stream ownership


@_requires_selective_sm100
@pytest.mark.parametrize("ratio", [1, 4])
@pytest.mark.parametrize("query_tile", [1, 16])
def test_benchmark_full_step_refreshes_lengths_and_page_maps(ratio, query_tile):
    """The separated layer consumes the last device-planned lengths and map."""
    from flashinfer.attn_scores.fp8_kernel_config import _FP8MQAKernelConfig
    from flashinfer.attn_scores.query_group_metadata import _FP8QueryGroups

    spec = importlib.util.spec_from_file_location(
        "bench_fp8_mqa_selective_logits",
        os.path.join(
            os.path.dirname(__file__),
            "../../../benchmarks/bench_fp8_mqa_selective_logits.py",
        ),
    )
    assert spec is not None and spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)

    inputs = _request_inputs([query_tile, query_tile], [1024, 1024])
    q, keys, scales, weights, qo_indptr, physical_kv = inputs
    pool, table = _paged_inputs(inputs)
    kv_indptr = physical_kv * ratio
    groups = _FP8QueryGroups.build(
        qo_indptr.cpu().tolist(), kv_indptr.cpu().tolist(), ratio, query_tile
    )
    full = benchmark._FullGVR(
        q,
        pool,
        weights,
        table,
        groups,
        _FP8MQAKernelConfig(128, 1, 160, query_tile, num_q_stages=3, num_kv_stages=3),
        512,
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
    )
    assert full.scorer_mode == (
        "ordinary-causal-envelope"
        if ratio == 4 and query_tile > 1
        else "ordinary-causal"
    )
    assert full.scorer_width == 1024 + (
        query_tile - 1 if ratio == 4 and query_tile > 1 else 0
    )
    full.step()
    full.layer()
    torch.cuda.synchronize()
    step_graph, layer_graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with torch.cuda.graph(step_graph):
        full.step()
    with torch.cuda.graph(layer_graph):
        full.layer()
    for lengths in ([1024, 1024], [768, 512]):
        if lengths[0] == 768:
            # Keep the pool address but replace its data, so a private stale
            # pool copy cannot satisfy the captured layer's reference check.
            key_bytes = keys.view(torch.uint8)
            key_bytes.copy_(key_bytes.flip(0))
            pool.copy_(_fuse_paged_kv(keys, scales).flip(0))
        kv_indptr.copy_(
            torch.tensor(
                [0, lengths[0] * ratio, sum(lengths) * ratio],
                device=q.device,
                dtype=torch.int32,
            )
        )
        table.copy_(table.roll(1, dims=1))
        step_graph.replay()
        layer_graph.replay()
        torch.cuda.synchronize()
        oracle = []
        for request, length in enumerate(lengths):
            for local in range(query_tile):
                end = (length * ratio - query_tile + local + 1) // ratio
                slots = torch.arange(end, device=q.device)
                page = pool.shape[0] - 1 - table[request, slots // 128].long()
                physical = page * 128 + slots % 128
                row = request * query_tile + local
                dots = q[row].float() @ keys[physical].float().T
                oracle.append(
                    (dots.relu() * weights[row, :, None]).sum(0) * scales[physical]
                )
        _check_selection(full.output, oracle, 512)


@_requires_selective_sm100
@pytest.mark.parametrize("ratio", [1, 4])
def test_capacity_plan_capture_requires_prepared_pool(ratio):
    """A new wrapper must not capture a step before its first pool binding."""
    inputs = _request_inputs([2], [1024])
    q, _, _, weights, qo_indptr, physical_kv = inputs
    pool, table = _paged_inputs(inputs)
    kv_indptr = physical_kv * ratio
    for capacity_rows, layout in (
        (2, dict(kv_layout=pool)),
        (4, dict(num_pages=pool.shape[0], device=pool.device)),
    ):
        kv_indptr.copy_(physical_kv * ratio)
        wrapper = FP8PagedMQATopKWrapper(
            strategy="selective",
            compression_ratio=ratio,
            max_context_length=1024 * ratio,
            max_chunk_size=capacity_rows,
            max_batch_size=1,
            **layout,
        )
        rejected_graph = torch.cuda.CUDAGraph()
        with (
            torch.cuda.graph(rejected_graph),
            pytest.raises(RuntimeError, match="prepare_pool.*capturing plan"),
        ):
            wrapper.plan(qo_indptr, kv_indptr, table, num_query_rows=2)

        wrapper.prepare_pool(pool)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            wrapper.plan(qo_indptr, kv_indptr, table, num_query_rows=2)
            selected = wrapper.run(q, weights, pool)
        for length in (1024, 768):
            kv_indptr[1] = length * ratio
            graph.replay()
            torch.cuda.synchronize()
            expected_inputs = (*inputs[:5], kv_indptr // ratio)
            _check_selection(selected, _oracle_rows(expected_inputs), 512)
            assert wrapper.error.item() == wrapper.repair_error.item() == 0
        del graph, rejected_graph


@_requires_selective_sm100
def test_capacity_step_shape_writes_skip_repeats_until_outer_capture():
    """Repeated steps skip shape writes only until a capture can rewrite them."""
    from flashinfer.experimental.attn_scores.selective_capacity import (
        _CapacityStepInputs,
        _SelectiveCapacity,
    )

    capacity = _SelectiveCapacity(8, 128, 32, 1, 65536, 8, 4)
    state = _CapacityStepInputs(capacity, device="cuda", num_pages=8)
    table = torch.zeros_like(state.block_table)
    shape_a = torch.tensor([0, 2, 5], dtype=torch.int32, device="cuda")
    shape_b = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device="cuda")

    def stage(rows, prefix):
        state.stage_step(prefix, prefix, table, num_query_rows=rows)

    stage(5, shape_a)
    torch.cuda.synchronize()
    assert state.step_shape.tolist() == [5, 2]
    state.step_shape.fill_(-7)
    stage(5, shape_a)
    torch.cuda.synchronize()
    assert state.step_shape.tolist() == [-7, -7]
    stage(3, shape_b)
    torch.cuda.synchronize()
    assert state.step_shape.tolist() == [3, 3]

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        stage(3, shape_b)
    stage(5, shape_a)
    graph.replay()
    torch.cuda.synchronize()
    assert state.step_shape.tolist() == [3, 3]
    stage(5, shape_a)
    torch.cuda.synchronize()
    assert state.step_shape.tolist() == [5, 2]
    state.step_shape.fill_(-7)
    stage(5, shape_a)
    torch.cuda.synchronize()
    assert state.step_shape.tolist() == [5, 2]


@_requires_selective_sm100
def test_capacity_constructor_contract(monkeypatch):
    """Both strategies reserve capacities and inspect an unretained pool layout."""
    inputs = _request_inputs([2], [1024])
    q, _, _, weights, qo, kv = inputs
    pool, table = _paged_inputs(inputs)
    capacities = dict(max_context_length=1024, max_chunk_size=2, max_batch_size=1)
    full = FP8PagedMQATopKWrapper(strategy="full", **capacities, kv_layout=pool)
    assert set(full.input_buffers()) == {
        "q",
        "weights",
        "qo_indptr",
        "kv_indptr",
        "block_table",
    }
    with pytest.raises(ValueError, match="both strategies require"):
        FP8PagedMQATopKWrapper(strategy="full", kv_layout=pool)
    with pytest.raises(ValueError, match="both strategies require"):
        FP8PagedMQATopKWrapper(strategy="selective", kv_layout=pool)
    with pytest.raises(ValueError, match="kv_layout or num_pages and device"):
        FP8PagedMQATopKWrapper(strategy="selective", **capacities)
    with pytest.raises(ValueError, match="either kv_layout or num_pages"):
        FP8PagedMQATopKWrapper(
            strategy="selective", **capacities, kv_layout=pool, num_pages=1
        )
    with pytest.raises(ValueError, match="num_pages must be"):
        FP8PagedMQATopKWrapper(
            strategy="selective", **capacities, num_pages=-1, device=pool.device
        )
    with pytest.raises(ValueError, match="device must be CUDA"):
        FP8PagedMQATopKWrapper(
            strategy="selective", **capacities, num_pages=1, device="cpu"
        )
    with pytest.raises(ValueError, match="kv_layout must be"):
        FP8PagedMQATopKWrapper(
            strategy="selective", **capacities, kv_layout=pool.view(torch.int8)
        )
    rejected_graph = torch.cuda.CUDAGraph()
    with (
        torch.cuda.graph(rejected_graph),
        pytest.raises(RuntimeError, match="outside capture"),
    ):
        FP8PagedMQATopKWrapper(strategy="selective", **capacities, kv_layout=pool)
    del rejected_graph

    from flashinfer.experimental.attn_scores import selective_logits

    def reject_compile(*args, **kwargs):
        raise AssertionError("construction and initial staging must not compile")

    with monkeypatch.context() as cold:
        cold.setattr(selective_logits, "_compile_for_device", reject_compile)
        exemplar = FP8PagedMQATopKWrapper(
            strategy="selective", **capacities, kv_layout=pool
        )
        paged = FP8PagedMQATopKWrapper(
            strategy="selective", **capacities, num_pages=pool.shape[0], device="cuda"
        )
        for wrapper in (exemplar, paged):
            wrapper.plan(qo, kv, table)
    for wrapper in (exemplar, paged):
        plan = wrapper._capacity_plan
        assert wrapper.strategy == "selective"
        assert wrapper.math_mode == "relu_strict"
        assert plan._pool_layout == exemplar._capacity_plan._pool_layout
        assert plan._engine is None  # The scoring chain compiles lazily.
        # A bad initial step must be validated when the first pool is bound,
        # through either explicit preparation or eager run's automatic binding.
        wrapper.plan(qo, torch.zeros_like(kv), table)
        if wrapper is exemplar:
            wrapper.prepare_pool(pool)
        invalid = wrapper.run(q, weights, pool)
        torch.cuda.synchronize()
        assert wrapper.error.item() & 2
        assert bool((invalid == -1).all())
        wrapper.plan(qo, kv, table)
        selected = wrapper.run(q, weights, pool)
        torch.cuda.synchronize()
        assert plan._engine is not None
        assert wrapper.error.item() == wrapper.repair_error.item() == 0
        _check_selection(selected, _oracle_rows(inputs), 512)


@_requires_selective_sm100
def test_capacity_wrappers_isolate_caller_graphs():
    """Live wrappers keep separate workspaces across alternating caller graphs."""
    inputs = _request_inputs([2, 1], [9001, 7000])
    q, _, _, weights, cu_q, cu_kv = inputs
    pool, table = _paged_inputs(inputs)
    wrappers, graphs, outputs, expected = [], [], [], []
    for top_k in (512, 2048, 512):
        # Each wrapper stays referenced: a caller graph addresses its workspace.
        w = FP8PagedMQATopKWrapper(
            strategy="selective",
            top_k=top_k,
            max_context_length=9001,
            max_chunk_size=3,
            max_batch_size=2,
            kv_layout=pool,
        )
        w.plan(cu_q, cu_kv, table)
        expected.append(w.run(q, weights, pool).clone())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            w.plan(cu_q, cu_kv, table)
            outputs.append(w.run(q, weights, pool))
        wrappers.append(w)
        graphs.append(graph)
    torch.cuda.synchronize()
    for _ in range(3):
        for w, graph, output, want in zip(
            wrappers, graphs, outputs, expected, strict=True
        ):
            graph.replay()
            torch.cuda.synchronize()
            assert w.error.item() == w.repair_error.item() == 0
            # Tie-aware: each wrapper's own width and oracle, not bitwise sets.
            assert output.shape == want.shape
            _check_selection(output, _oracle_rows(inputs), want.shape[1])


@_requires_selective_sm100
@pytest.mark.parametrize("ratio", [1, 4])
def test_capacity_capture_binds_unprepared_pool(ratio):
    """After one preparation, caller capture binds an unseen equal-layout pool."""
    inputs = _request_inputs([1, 1], [9001, 9001])
    q, keys, scales, weights, cu_q, physical_kv = inputs
    pool_a, table = _paged_inputs(inputs)
    inputs_b = (q, keys, scales.flip(0).contiguous(), weights, cu_q, physical_kv)
    pool_b, table_b = _paged_inputs(inputs_b)
    assert torch.equal(table, table_b)
    original_kv = physical_kv * ratio

    def wrapper():
        return FP8PagedMQATopKWrapper(
            strategy="selective",
            compression_ratio=ratio,
            max_context_length=9001 * ratio,
            max_chunk_size=2,
            max_batch_size=2,
            kv_layout=pool_a,
        )

    # Nothing compiled yet: capture cannot bind any pool.
    cold = wrapper()
    cold.plan(cu_q, original_kv, table)
    rejected = torch.cuda.CUDAGraph()
    with (
        torch.cuda.graph(rejected),
        pytest.raises(RuntimeError, match="must precede caller capture"),
    ):
        cold.run(q, weights, pool_a)
    del rejected

    w = wrapper()
    w.prepare_pool(pool_a)
    out_a = torch.empty((2, 512), device=q.device, dtype=torch.int32)
    out_b = torch.empty_like(out_a)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        w.plan(cu_q, original_kv, table)
        w.run(q, weights, pool_a, out=out_a)
        w.run(q, weights, pool_b, out=out_b)
    # The oracle is tie-aware; index sets may differ only among equal scores.
    for factor in (1.0, 0.5):
        weights.mul_(factor)
        graph.replay()
        torch.cuda.synchronize()
        assert w.error.item() == w.repair_error.item() == 0
        _check_selection(out_a, _oracle_rows(inputs), 512)
        _check_selection(out_b, _oracle_rows(inputs_b), 512)
    w.plan(cu_q, original_kv, table)
    _check_selection(w.run(q, weights, pool_b), _oracle_rows(inputs_b), 512)
    torch.cuda.synchronize()


def test_selective_compile_materializes_before_publication(monkeypatch):
    cute_module = pytest.importorskip("cutlass.cute")
    events = []
    compiled = SimpleNamespace(to=lambda device: events.append(("load", device)))

    def compile_kernel(*args, **kwargs):
        events.append(("compile", kwargs["options"]))
        return compiled

    monkeypatch.setattr(torch.cuda, "device", lambda _: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (10, 0))
    monkeypatch.setattr(cute_module, "compile", compile_kernel)
    assert sl._compile_for_device(torch.device("cuda:2"), object()) is compiled
    assert events == [("compile", "--gpu-arch sm_100a"), ("load", 2)]


@_requires_selective_sm100
@pytest.mark.parametrize("heads,ratio", [(64, 4), (64, 1), (8, 4)])
def test_capacity_materializes_kernels_before_capture(monkeypatch, heads, ratio):
    """First-use CuTe libraries must load before caller capture."""
    from cutlass.cutlass_dsl.cuda_jit_executor import CudaDialectJitCompiledFunction

    original = CudaDialectJitCompiledFunction._load_cuda_library

    def load(compiled):
        assert not torch.cuda.is_current_stream_capturing(), compiled.function_name
        return original(compiled)

    monkeypatch.setattr(CudaDialectJitCompiledFunction, "_load_cuda_library", load)
    test_capacity_shared_layer_workspace(heads, ratio)


@_requires_selective_sm100
@pytest.mark.parametrize("heads", [8, 64])
@pytest.mark.parametrize("ratio", [1, 4])
def test_capacity_shared_layer_workspace(heads, ratio):
    """Distinct pools share scratch without stale repair bindings or lost output."""
    import weakref

    inputs = _request_inputs([1, 1], [9001, 9001], heads=heads)
    q, keys, scales, weights, cu_q, physical_kv = inputs
    q.copy_(torch.ones(q.shape, device=q.device).to(q.dtype))
    keys.copy_(torch.ones(keys.shape, device=keys.device).to(keys.dtype))
    weights.fill_(-1)
    scales.copy_(torch.linspace(1.00001, 1.0001, scales.numel(), device=q.device))
    pool_a, table = _paged_inputs(inputs)
    reversed_scales = scales.flip(0).contiguous()
    inputs_b = (q, keys, reversed_scales, weights, cu_q, physical_kv)
    pool_b, table_b = _paged_inputs(inputs_b)
    assert torch.equal(table, table_b)
    exemplar = pool_a.clone()
    exemplar_ref = weakref.ref(exemplar)
    # Layout inspection must not rely on collection of import/decorator frames
    # to release the exemplar, even on the first construction in a process.
    gc_enabled = gc.isenabled()
    gc.disable()
    try:
        wrapper = FP8PagedMQATopKWrapper(
            strategy="selective",
            num_heads=heads,
            compression_ratio=ratio,
            max_context_length=9001 * ratio,
            max_chunk_size=2,
            max_batch_size=2,
            kv_layout=exemplar,
        )
        del exemplar
        assert exemplar_ref() is None
    finally:
        if gc_enabled:
            gc.enable()
    wrapper.prepare_pool(pool_a)
    engine = wrapper._capacity_plan._engine
    workspace = engine._candidate_values
    wrapper.prepare_pool(pool_b)
    assert wrapper._capacity_plan._engine is engine
    assert engine._candidate_values is workspace
    original_kv = physical_kv * ratio
    wrapper.plan(cu_q, original_kv, table)
    out_a = torch.empty((2, 512), device=q.device, dtype=torch.int32)
    out_b = torch.empty_like(out_a)
    wrapper.run(q, weights, pool_a, out=out_a)
    torch.cuda.synchronize()
    assert engine._second_repair_flags.count_nonzero().item() == 2
    saved_a = out_a.clone()
    wrapper.run(q, weights, pool_b, out=out_b)
    torch.cuda.synchronize()
    _check_selection(out_a, _oracle_rows(inputs), 512)
    _check_selection(out_b, _oracle_rows(inputs_b), 512)
    assert torch.equal(out_a, saved_a)
    with pytest.raises(AssertionError):
        _check_selection(out_a, _oracle_rows(inputs_b), 512)
    assert wrapper.repair_error.item() == 0
    wrapper.repair_error.fill_(8)
    wrapper.run(q, weights, pool_b)
    torch.cuda.synchronize()
    assert wrapper.repair_error.item() == 8

    # A later layer must not re-read a caller map changed after plan.
    invalid = torch.full_like(table, -1)
    wrapper.plan(cu_q, original_kv, invalid)
    invalid.copy_(table)
    for pool in (pool_a, pool_b):
        result = wrapper.run(q, weights, pool)
        torch.cuda.synchronize()
        assert wrapper.error.item() == 8
        assert (result == -1).all()
    with pytest.raises(ValueError, match="layout"):
        wrapper.run(q, weights, pool_a[:-1])

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.plan(cu_q, original_kv, table)
        with pytest.raises(RuntimeError, match="clear_pools must run outside"):
            wrapper.clear_pools()
        wrapper.run(q, weights, pool_a, out=out_a)
        wrapper.run(q, weights, pool_b, out=out_b)
    for factor in (0.5, 1.5):
        weights.mul_(factor)
        graph.replay()
        torch.cuda.synchronize()
        _check_selection(out_a, _oracle_rows(inputs), 512)
        _check_selection(out_b, _oracle_rows(inputs_b), 512)
        assert wrapper.error.item() == wrapper.repair_error.item() == 0

    # Retiring pool bindings releases tensor owners without rebuilding scratch.
    del graph
    retired_pool = pool_a.clone()
    retired_ref = weakref.ref(retired_pool)
    wrapper.prepare_pool(retired_pool)
    wrapper.run(q, weights, retired_pool, out=out_a)
    wrapper.run(q, weights, pool_b, out=out_b)
    del retired_pool
    assert retired_ref() is not None
    wrapper.clear_pools()
    assert retired_ref() is None
    assert wrapper._capacity_plan._engine is engine
    assert engine._candidate_values is workspace
    assert not wrapper._capacity_plan._pool_bindings
    wrapper.run(q, weights, pool_b, out=out_b)
    torch.cuda.synchronize()
    _check_selection(out_b, _oracle_rows(inputs_b), 512)


@_requires_selective_sm100
def test_capacity_first_compact_pool_precompiles_metadata(monkeypatch, record_property):
    """A cold compact-repair pool never compiles metadata during capture."""
    import time

    from flashinfer.experimental.attn_scores import selective_logits

    # Preserve existing compiled owners while giving this plan a cold metadata
    # cache, so the assertion does not depend on earlier tests' specializations.
    monkeypatch.setattr(selective_logits, "_METADATA_COMPILED", {})
    real_compile = selective_logits._compile_for_device
    metadata_compiles = []

    def compile_outside_capture(device, kernel, *args):
        if type(kernel).__name__ in (
            "SelectiveLogitsMetadataScheduleKernel",
            "SelectiveLogitsExplicitMetadataScheduleKernel",
        ):
            assert not torch.cuda.is_current_stream_capturing()
            metadata_compiles.append(type(kernel).__name__)
        return real_compile(device, kernel, *args)

    monkeypatch.setattr(
        selective_logits, "_compile_for_device", compile_outside_capture
    )
    inputs = _request_inputs([2], [9001])
    q, _, _, weights, qo, kv = inputs
    fused, table = _paged_inputs(inputs)
    wrapper = FP8PagedMQATopKWrapper(
        strategy="selective",
        max_context_length=9001,
        max_chunk_size=2,
        max_batch_size=1,
        kv_layout=fused,
    )
    started = time.perf_counter()
    wrapper.prepare_pool(fused)
    record_property("first_prepare_pool_seconds", time.perf_counter() - started)
    assert wrapper._capacity_plan._engine._compact_repair
    assert metadata_compiles
    assert any(key[-1] for key in selective_logits._METADATA_COMPILED)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.plan(qo, kv, table)
        selected = wrapper.run(q, weights, fused)
    graph.replay()
    torch.cuda.synchronize()
    assert wrapper.error.item() == wrapper.repair_error.item() == 0
    _check_selection(selected, _oracle_rows(inputs), 512)


@_requires_selective_sm100
@pytest.mark.parametrize("ratio", [1, 4])
@pytest.mark.parametrize("physical_capacity", [6143, 6144, 6145])
def test_capacity_sampling_physical_boundary(
    ratio, physical_capacity, monkeypatch, ieee_matmul
):
    inputs = _request_inputs([3], [physical_capacity])
    q, keys, scales, weights, qo, _ = inputs
    fused, table = _paged_inputs(inputs)
    original_capacity = physical_capacity * ratio + ratio - 1
    kv = torch.tensor([0, original_capacity], dtype=torch.int32, device=q.device)
    wrapper = FP8PagedMQATopKWrapper(
        strategy="selective",
        compression_ratio=ratio,
        max_context_length=original_capacity,
        max_chunk_size=3,
        max_batch_size=1,
        kv_layout=fused,
    )
    wrapper.prepare_pool(fused)
    engine = wrapper._capacity_plan._engine
    assert engine._needs_sampling == (physical_capacity > 6144)
    if not engine._needs_sampling:
        assert bool((engine._selected_bins == 2047).all())
    wrapper.plan(qo, kv, table)
    wrapper.run(q, weights, fused)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.plan(qo, kv, table)
        selected = wrapper.run(q, weights, fused)
    for original_length in (
        original_capacity,
        512 * ratio + ratio - 1,
        original_capacity,
    ):
        kv[1] = original_length
        q.copy_((-q.float()).to(q.dtype))
        weights.mul_(-0.75)
        fused.copy_(fused.flip(0))
        table.copy_(fused.shape[0] - 1 - table)
        engine._candidate_counts.fill_(2**30)
        graph.replay()
        torch.cuda.synchronize()
        assert wrapper.error.item() == wrapper.repair_error.item() == 0
        assert engine._needs_sampling == (physical_capacity > 6144)
        if ratio == 4:
            _check_compressed_wrapper(selected, (q, fused, weights, qo, kv, table), 512)
        else:
            _check_selection(
                selected, _oracle_rows((q, keys, scales, weights, qo, kv)), 512
            )


@_requires_selective_sm100
@pytest.mark.parametrize("ratio", [1, 4])
def test_capacity_sampling_on_off_same_bucket(ratio, monkeypatch, ieee_matmul):
    from flashinfer.experimental.attn_scores.selective_capacity import (
        _CapacitySelectiveTopKPlan,
    )

    inputs = _request_inputs([2], [6144])
    q, keys, scales, weights, qo, _ = inputs
    fused, table = _paged_inputs(inputs)
    kv = torch.tensor([0, 6144 * ratio], dtype=torch.int32, device=q.device)
    graphs, outputs, wrappers = [], [], []
    for force_sampling in (False, True):
        # Compare sampling and all-key publication within identical capacities.
        # The forced control is private to this test, not a public routing knob.
        with monkeypatch.context() as patch:
            if force_sampling:
                patch.setattr(
                    _CapacitySelectiveTopKPlan,
                    "_needs_sampling",
                    property(lambda _: True),
                )
            wrapper = FP8PagedMQATopKWrapper(
                strategy="selective",
                compression_ratio=ratio,
                max_context_length=6144 * ratio,
                max_chunk_size=2,
                max_batch_size=1,
                kv_layout=fused,
            )
            wrapper.prepare_pool(fused)
            assert wrapper._capacity_plan._engine._needs_sampling == force_sampling
            wrapper.plan(qo, kv, table)
            wrapper.run(q, weights, fused)
            graph = torch.cuda.CUDAGraph()
            with _capture_cuda_graph(graph):
                wrapper.plan(qo, kv, table)
                selected = wrapper.run(q, weights, fused)
            graphs.append(graph)
            outputs.append(selected)
            wrappers.append(wrapper)
    for tied in (False, True, False):
        q.copy_(torch.randn(q.shape, device=q.device).mul_(0.1).to(q.dtype))
        if tied:
            q.zero_()
        for wrapper, graph, selected in zip(wrappers, graphs, outputs, strict=True):
            graph.replay()
            torch.cuda.synchronize()
            assert wrapper.error.item() == wrapper.repair_error.item() == 0
            if ratio == 4:
                _check_compressed_wrapper(
                    selected, (q, fused, weights, qo, kv, table), 512
                )
            else:
                _check_selection(
                    selected, _oracle_rows((q, keys, scales, weights, qo, kv)), 512
                )


@_requires_selective_sm100
@pytest.mark.parametrize("heads,ratio", [(8, 1), (8, 4), (64, 1), (64, 4)])
def test_capacity_selective_chain_live_steps_and_repair(heads, ratio):
    inputs = _inputs(8, 8192, num_heads=heads)
    q, keys, scales, weights, _, _ = inputs
    fused = _fuse_paged_kv(keys, scales)
    table = (
        torch.arange(3 * (65536 // ratio // 128), device=q.device, dtype=torch.int32)
        .remainder(fused.shape[0])
        .view(3, -1)
    )
    top_k = 512 if heads == 8 else 2048
    wrapper = FP8PagedMQATopKWrapper(
        strategy="selective",
        num_heads=heads,
        top_k=top_k,
        compression_ratio=ratio,
        max_context_length=65536,
        max_chunk_size=8,
        max_batch_size=3,
        kv_layout=fused,
    )
    wrapper.prepare_pool(fused)
    plan = wrapper._capacity_plan
    device_sms = torch.cuda.get_device_properties(q.device).multi_processor_count
    assert plan._engine._candidate_prepared.num_sms == device_sms
    assert plan._engine._sample_prepared.num_sms == device_sms
    launches = plan._pool_bindings[plan._pool_key(fused)][1]
    wrapper.prepare_pool(fused.view_as(fused))

    def prefixes(lengths):
        return torch.tensor(
            [0, *torch.tensor(lengths).cumsum(0).tolist()],
            dtype=torch.int32,
            device=q.device,
        )

    def check(q_lengths, kv_lengths, *, tied=False, observed=None):
        rows = sum(q_lengths)
        step_weights = torch.zeros_like(weights[:rows]) if tied else weights[:rows]
        qo, kv = prefixes(q_lengths), prefixes(kv_lengths)
        if observed is None:
            wrapper.plan(qo, kv, table, num_query_rows=q[:rows].shape[0])
        selected = (
            wrapper.run(q[:rows], step_weights, fused) if observed is None else observed
        )
        torch.cuda.synchronize()
        assert wrapper.error.item() == 0
        assert wrapper.repair_error.item() == 0
        expected, offset = [], 0
        for request, (q_length, kv_length) in enumerate(
            zip(q_lengths, kv_lengths, strict=True)
        ):
            for local in range(q_length):
                end = (kv_length - q_length + local + 1) // ratio
                slots = torch.arange(end, device=q.device)
                physical = table[request, slots // 128].long() * 128 + slots % 128
                dots = q[offset + local].float() @ keys[physical].float().T
                expected.append(
                    (dots.relu() * step_weights[offset + local, :, None]).sum(0)
                    * scales[physical]
                )
            offset += q_length
        _check_selection(selected, expected, top_k)
        assert plan._pool_bindings[plan._pool_key(fused)][1] is launches
        return selected

    check([1, 0, 3], [65535, 7, 8195])
    check([1, 0, 3], [65535, 7, 8195], tied=True)
    assert plan._engine._repair_flags.count_nonzero().item() > 0
    assert plan._engine._second_repair_flags.count_nonzero().item() > 0
    table.copy_(table.roll(1, dims=1))
    check([2, 0, 3], [9, 0, 17])
    check([1, 1], [1000, 1001])
    check([8], [777])
    check([], [])

    # Invalid device metadata suppresses scoring, then a valid step recovers.
    table.fill_(-1)
    wrapper.plan(prefixes([1]), prefixes([128]), table, num_query_rows=q[:1].shape[0])
    selected = wrapper.run(q[:1], weights[:1], fused)
    torch.cuda.synchronize()
    assert wrapper.error.item() == 8
    assert (selected == -1).all()
    table.copy_(
        torch.arange(table.numel(), device=q.device, dtype=torch.int32)
        .remainder(fused.shape[0])
        .view_as(table)
    )
    check([2], [2049])

    # Equal-layout pools share the workspace but keep separate launch bindings.
    # Each physical row must exceed K, or every index remains a valid selection
    # after rebinding and cannot detect execution against the retired map.
    rebind_lengths = [8191 * ratio, 4095 * ratio + 1]
    previous = check([1, 2], rebind_lengths).clone()
    old_owners = fused, table
    fused, table = fused.clone(), table.roll(1, dims=1)
    wrapper.prepare_pool(fused)
    launches = plan._pool_bindings[plan._pool_key(fused)][1]
    with pytest.raises(AssertionError):
        check([1, 2], rebind_lengths, observed=previous)
    check([1, 2], rebind_lengths)
    assert old_owners[0].data_ptr() != fused.data_ptr()
    buffers = wrapper.input_buffers()
    buffers["q"][:3].copy_(q[:3])
    buffers["weights"][:3].copy_(weights[:3])
    buffers["qo_indptr"][:3].copy_(prefixes([1, 2]))
    buffers["kv_indptr"][:3].copy_(prefixes(rebind_lengths))
    destination = torch.empty((3, top_k), device=q.device, dtype=torch.int32)
    wrapper.plan(
        buffers["qo_indptr"][:3],
        buffers["kv_indptr"][:3],
        table,
        num_query_rows=buffers["q"][:3].shape[0],
    )
    assert (
        wrapper.run(buffers["q"][:3], buffers["weights"][:3], fused, out=destination)
        is destination
    )
    check([1, 2], rebind_lengths, observed=destination)
    with pytest.raises(ValueError, match="cross-alias"):
        wrapper.plan(
            buffers["kv_indptr"][:3],
            buffers["qo_indptr"][:3],
            table,
            num_query_rows=q[:3].shape[0],
        )
        wrapper.run(q[:3], weights[:3], fused)
    protected_output = (
        plan._engine._candidate_values.view(torch.int32)
        .flatten()[:top_k]
        .view(1, top_k)
    )
    with pytest.raises(ValueError, match="overlap"):
        wrapper.plan(
            prefixes([1]), prefixes([8192]), table, num_query_rows=q[:1].shape[0]
        )
        wrapper.run(q[:1], weights[:1], fused, out=protected_output)
    qo, kv = prefixes([2, 1]), prefixes([3073, 1031])
    outer = torch.cuda.CUDAGraph()
    with torch.cuda.graph(outer):
        wrapper.plan(qo, kv, table, num_query_rows=q[:3].shape[0])
        output = wrapper.run(q[:3], weights[:3], fused)
    for q_values, kv_values in (([2, 1], [3073, 1031]), ([1, 2], [999, 1003])):
        qo.copy_(prefixes(q_values))
        kv.copy_(prefixes(kv_values))
        outer.replay()
        torch.cuda.synchronize()
        check(q_values, kv_values, observed=output)


@_requires_selective_sm100
@pytest.mark.parametrize("ratio", [1, 4])
@pytest.mark.parametrize("heads", [8, 64])
@pytest.mark.parametrize("bucket", [2, 8])
def test_capacity_decode_bucket_multirow_steps(heads, ratio, bucket):
    q, keys, scales, weights, _, _ = _inputs(bucket, 8192, num_heads=heads)
    fused = _fuse_paged_kv(keys, scales)
    table = torch.arange(8192 // 128, device=q.device, dtype=torch.int32).repeat(
        bucket, 1
    )
    top_k = 512 if heads == 8 else 2048
    wrapper = FP8PagedMQATopKWrapper(
        strategy="selective",
        num_heads=heads,
        top_k=top_k,
        compression_ratio=ratio,
        max_context_length=8192 * ratio,
        max_chunk_size=bucket,
        max_batch_size=bucket,
        kv_layout=fused,
    )
    wrapper.prepare_pool(fused)

    def prefixes(lengths):
        return torch.tensor(
            [0, *torch.tensor(lengths).cumsum(0).tolist()],
            dtype=torch.int32,
            device=q.device,
        )

    q_lengths = [2, 0] if bucket == 2 else [1, 2, 0, 3]
    rows = sum(q_lengths)
    lengths = [8192 * ratio - request for request in range(len(q_lengths))]
    qo, kv = prefixes(q_lengths), prefixes(lengths)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.plan(qo, kv, table, num_query_rows=q[:rows].shape[0])
        selected = wrapper.run(q[:rows], weights[:rows], fused)
    for step_lengths in (q_lengths, list(reversed(q_lengths))):
        qo.copy_(prefixes(step_lengths))
        table.copy_(table.roll(1, dims=1))
        graph.replay()
        torch.cuda.synchronize()
        assert wrapper.error.item() == 0
        assert wrapper.repair_error.item() == 0
        oracle, offset = [], 0
        for request, (q_length, kv_length) in enumerate(
            zip(step_lengths, lengths, strict=True)
        ):
            for local in range(q_length):
                end = (kv_length - q_length + local + 1) // ratio
                slots = torch.arange(end, device=q.device)
                physical = table[request, slots // 128].long() * 128 + slots % 128
                dots = q[offset + local].float() @ keys[physical].float().T
                oracle.append(
                    (dots.relu() * weights[offset + local, :, None]).sum(0)
                    * scales[physical]
                )
            offset += q_length
        _check_selection(selected, oracle, top_k)


@_requires_selective_sm100
@pytest.mark.parametrize("ratio", [1, 4])
def test_capacity_reused_output_clears_inactive_rows(ratio):
    """Repeated layers rewrite caller-modified output, including unscored rows."""
    rows = ratio + 2
    q, keys, scales, weights, _, _ = _inputs(rows, 128, num_heads=8)
    fused = _fuse_paged_kv(keys, scales)
    table = torch.zeros((1, 1), device=q.device, dtype=torch.int32)

    def prefix(values):
        return torch.tensor(values, device=q.device, dtype=torch.int32)

    for empty in (False, True):
        pool = fused[:0] if empty else fused
        wrapper = FP8PagedMQATopKWrapper(
            strategy="selective",
            top_k=512,
            compression_ratio=ratio,
            max_context_length=ratio,
            max_chunk_size=rows,
            max_batch_size=1,
            kv_layout=pool,
        )
        wrapper.prepare_pool(pool)
        wrapper.plan(
            prefix([0, 0 if empty else 1]),
            prefix([0, 0 if empty else ratio]),
            table,
            num_query_rows=rows,
        )
        selected = wrapper.run(q, weights, pool)
        expected = torch.full_like(selected, -1)
        if not empty:
            expected[0, 0] = 0
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            wrapper.run(q, weights, pool, out=selected)
        selected.fill_(17)
        wrapper.run(q, weights, pool, out=selected)
        torch.testing.assert_close(selected, expected)
        selected.fill_(17)
        graph.replay()
        torch.testing.assert_close(selected, expected)
        assert wrapper.error.item() == wrapper.repair_error.item() == 0

        wrapper.plan(prefix([1, 1]), prefix([0, 0]), table, num_query_rows=rows)
        selected.fill_(17)
        graph.replay()
        assert wrapper.error.item() == 2
        assert (selected == -1).all()


@_requires_selective_sm100
@pytest.mark.parametrize("ratio", [1, 4])
def test_capacity_selective_public_validation(ratio):
    q, keys, scales, weights, _, _ = _inputs(2, 1024, num_heads=8)
    fused = _fuse_paged_kv(keys, scales)
    table = (
        torch.arange(2 * (8192 // ratio // 128), device=q.device, dtype=torch.int32)
        .remainder(fused.shape[0])
        .view(2, -1)
    )
    specialization = dict(strategy="selective", top_k=512, compression_ratio=ratio)
    capacities = dict(max_context_length=8192, max_chunk_size=2, max_batch_size=2)
    wrapper = FP8PagedMQATopKWrapper(**specialization, **capacities, kv_layout=fused)
    wrapper.prepare_pool(fused)

    def prefix(values):
        return torch.tensor(values, device=q.device, dtype=torch.int32)

    qo, kv = prefix([0, 1, 2]), prefix([0, 1024, 2048])
    for queries, tokens, code in (
        ([1, 1, 2], [0, 1024, 2048], 2),
        ([0, 2, 1], [0, 1024, 2048], 2),
        ([0, 1, 3], [0, 1024, 2048], 4),
        ([0, 1, 2], [0, 8193, 9217], 4),
    ):
        wrapper.plan(prefix(queries), prefix(tokens), table, num_query_rows=q.shape[0])
        selected = wrapper.run(q, weights, fused)
        torch.cuda.synchronize()
        assert wrapper.error.item() == code
        assert (selected == -1).all()
    wrapper.plan(qo, kv, table, num_query_rows=q.shape[0])
    selected = wrapper.run(q, weights, fused)
    torch.cuda.synchronize()
    assert wrapper.error.item() == wrapper.repair_error.item() == 0
    count = min(512, 1024 // ratio)
    assert (selected[:, :count] >= 0).all()
    assert (selected[:, count:] == -1).all()

    with pytest.raises(ValueError, match="fit the planned"):
        wrapper.run(
            torch.empty((3, 8, 128), device=q.device, dtype=q.dtype),
            torch.empty((3, 8), device=q.device),
            fused,
        )
    with pytest.raises(ValueError, match="indptrs"):
        wrapper.plan(
            prefix([0, 1, 1, 2]),
            prefix([0, 1024, 1024, 2048]),
            table,
            num_query_rows=q.shape[0],
        )
        wrapper.run(q, weights, fused)
    with pytest.raises(ValueError, match="out must be"):
        wrapper.plan(qo, kv, table, num_query_rows=q.shape[0])
        wrapper.run(
            q,
            weights,
            fused,
            out=torch.empty((2, 513), device=q.device, dtype=torch.int32),
        )

    q_owner = torch.empty(2 * 512 * 4, device=q.device, dtype=torch.uint8)
    aliased_q = q_owner[: q.numel()].view(q.dtype).view_as(q)
    aliased_q.copy_(q)
    with pytest.raises(ValueError, match="out must not overlap"):
        wrapper.plan(qo, kv, table, num_query_rows=aliased_q.shape[0])
        wrapper.run(
            aliased_q, weights, fused, out=q_owner.view(torch.int32).view(2, 512)
        )
    weight_owner = torch.empty((2, 512), device=q.device)
    aliased_weights = weight_owner.flatten()[: weights.numel()].view_as(weights)
    aliased_weights.copy_(weights)
    with pytest.raises(ValueError, match="out must not overlap"):
        wrapper.plan(qo, kv, table, num_query_rows=q.shape[0])
        wrapper.run(q, aliased_weights, fused, out=weight_owner.view(torch.int32))

    with pytest.raises(ValueError, match="sample virtual prefixes"):
        FP8PagedMQATopKWrapper(
            **specialization,
            max_context_length=8192,
            max_chunk_size=2,
            max_batch_size=(2**31 - 1) // 3840 + 1,
            kv_layout=fused,
        )

    # Contiguity does not fix singleton-axis strides. Such a layout is a
    # different pool identity: the constructed wrapper rejects it at run, and
    # only an exemplar (not num_pages/device) can describe it for a new wrapper.
    strides = list(fused.stride())
    strides[2] = 0
    variant = fused.as_strided(fused.shape, strides)
    assert variant.is_contiguous() and variant.stride() != fused.stride()
    with pytest.raises(ValueError, match="layout"):
        wrapper.run(q, weights, variant)
    paged = FP8PagedMQATopKWrapper(
        **specialization, **capacities, num_pages=fused.shape[0], device=fused.device
    )
    paged.prepare_pool(fused)
    with pytest.raises(ValueError, match="layout"):
        paged.prepare_pool(variant)
    wrapper = FP8PagedMQATopKWrapper(**specialization, **capacities, kv_layout=variant)
    wrapper.plan(qo, kv, table)
    selected = wrapper.run(q, weights, variant)
    torch.cuda.synchronize()
    oracle = []
    for row in range(2):
        slots = torch.arange(1024 // ratio, device=q.device)
        physical = table[row, slots // 128].long() * 128 + slots % 128
        dots = q[row].float() @ keys[physical].float().T
        oracle.append((dots.relu() * weights[row, :, None]).sum(0) * scales[physical])
    _check_selection(selected, oracle, 512)
    assert wrapper.error.item() == wrapper.repair_error.item() == 0


@_requires_selective_sm100
@pytest.mark.parametrize("ratio", [1, 4])
def test_capacity_step_inputs_refresh_and_graph_replay(ratio):
    from flashinfer.experimental.attn_scores.selective_capacity import (
        _CapacityStepInputs,
        _SelectiveCapacity,
    )

    capacity = _SelectiveCapacity(8, 128, 32, ratio, 65536, 8, 4)
    fused = torch.empty((8, 32, 1, 132), dtype=torch.uint8, device="cuda")
    columns = 2048 // ratio
    table = (torch.arange(4 * columns, dtype=torch.int32, device="cuda") % 8).view(
        4, columns
    )
    state = _CapacityStepInputs(capacity, device=fused.device, num_pages=fused.shape[0])
    state.prepare_sampling(512)
    q = torch.zeros((8, 8, 128), dtype=torch.float8_e4m3fn, device="cuda")
    weights = torch.ones((8, 8), dtype=torch.float32, device="cuda")

    def step(q_prefix, kv_prefix):
        # New prefix objects must never replace the graph's input addresses.
        qo = torch.tensor(q_prefix, dtype=torch.int32, device="cuda")
        kv = torch.tensor(kv_prefix, dtype=torch.int32, device="cuda")
        state.stage_payload(q, weights)
        state.stage_step(qo, kv, table, num_query_rows=q.shape[0])
        graph.replay()
        torch.cuda.synchronize()

    def valid(q_prefix, kv_prefix):
        assert state.error.item() == 0
        lengths = [b - a for a, b in zip(kv_prefix[:-1], kv_prefix[1:], strict=True)]
        physical = [0]
        for length in lengths:
            physical.append(physical[-1] + length // ratio)
        expected_q = q_prefix + [q_prefix[-1]] * (5 - len(q_prefix))
        expected_kv = physical + [physical[-1]] * (5 - len(physical))
        assert state.safe_qo_indptr.tolist() == expected_q
        assert state.physical_kv_indptr.tolist() == expected_kv
        assert state.active_shape.tolist() == [q_prefix[-1], len(lengths)]
        ends, owners = [], []
        for request, (qb, qe, length) in enumerate(
            zip(q_prefix[:-1], q_prefix[1:], lengths, strict=True)
        ):
            ends.extend(
                (length - (qe - qb) + row + 1) // ratio for row in range(qe - qb)
            )
            owners.extend([request] * (qe - qb))
        assert state.row_ends.tolist() == ends + [0] * (8 - len(ends))
        assert state.row_requests.tolist() == owners + [-1] * (8 - len(owners))
        expected_counts, expected_ranks = [], []
        pages_per_request = []
        for length in lengths:
            physical_length = length // ratio
            pages = math.ceil(physical_length / 32)
            chosen = min(pages, 3840 // 32)
            indices = [i * (pages - 1) // max(chosen - 1, 1) for i in range(chosen)]
            pages_per_request.append(indices)
        for row_end, owner in zip(ends, owners, strict=True):
            coordinates = [
                page * 32 + offset
                for page in pages_per_request[owner]
                for offset in range(min(32, lengths[owner] // ratio - page * 32))
            ]
            count = sum(index < row_end for index in coordinates)
            expected_counts.append(count)
            expected_ranks.append(
                0
                if row_end == 0
                else max(1, min(count, math.ceil(2048 * count / row_end)))
            )
        assert state.sample_ends.tolist() == expected_counts + [0] * (8 - len(ends))
        assert state.sample_ranks.tolist() == expected_ranks + [0] * (8 - len(ends))
        padded_pages = [pages + [0] * (120 - len(pages)) for pages in pages_per_request]
        padded_pages.extend([[0] * 120 for _ in range(4 - len(padded_pages))])
        assert state.sample_page_indices.tolist() == padded_pages

    state.stage_payload(q, weights)
    state.stage_step(
        torch.zeros(1, dtype=torch.int32, device="cuda"),
        torch.zeros(1, dtype=torch.int32, device="cuda"),
        table,
        num_query_rows=q.shape[0],
    )
    state.refresh()
    state.refresh_sampling()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        state.refresh()
        state.refresh_sampling()
    for qo, kv in (
        ([0, 2, 2, 5], [0, 9, 9, 26]),
        ([0, 2, 2, 5], [0, 9, 14, 31]),
        ([0, 2, 2, 5], [0, 3, 3, 10]),
        ([0, 2, 2, 5], [0, 32767, 32768, 65535]),
        ([0, 1], [0, 64]),
        ([0], [0]),
    ):
        step(qo, kv)
        valid(qo, kv)

    table.fill_(-1)
    step([0, 3, 1, 5], [0, 9, 9, 26])
    assert state.error.item() == 2  # Invalid prefixes suppress page-map reads.
    step([0, 2, 2, 5], [0, 9, 9, 26])
    assert state.error.item() == 8
    assert state.safe_qo_indptr.count_nonzero().item() == 0
    assert state.physical_kv_indptr.count_nonzero().item() == 0
    assert state.active_shape.count_nonzero().item() == 0
    assert state.row_ends.count_nonzero().item() == 0
    assert (state.row_requests == -1).all()
    table.copy_(
        (torch.arange(table.numel(), dtype=torch.int32, device="cuda") % 8).view_as(
            table
        )
    )
    step([0, 1], [0, 65537])
    assert state.error.item() == 4
    step([0, 2, 2, 5], [0, 3, 3, 10])
    valid([0, 2, 2, 5], [0, 3, 3, 10])

    # A host shape failure must precede even the first staging copy.
    old_q = state.q.clone()
    with pytest.raises(ValueError, match="fit the planned"):
        state.stage_payload(
            torch.empty((9, 8, 128), dtype=q.dtype, device=q.device),
            torch.empty((9, 8), dtype=weights.dtype, device=weights.device),
        )
    assert torch.equal(state.q.view(torch.uint8), old_q.view(torch.uint8))


@_requires_selective_sm100
@pytest.mark.parametrize(
    "topology,heads,rows,length,top_k",
    [
        ("high_split", 8, 1, 131072, 512),
        ("owner_pool", 8, 1, 1024, 512),
        ("segmented", 8, 2, 32768, 512),
        ("h64_standard", 64, 9, 9001, 2048),
    ],
)
def test_capacity_chain_topologies_replay(
    topology, heads, rows, length, top_k, monkeypatch
):
    if topology == "h64_standard":
        # Fill one unsplit Q4 wave with the actual device geometry. A tiny
        # request chooses split-KV instead of the H64 candidate variant.
        rows = 4 * torch.cuda.get_device_properties("cuda").multi_processor_count
    inputs = _request_inputs([rows], [length], heads)
    q, _, _, weights, qo, kv = inputs
    fused, table = _paged_inputs(inputs)
    if topology == "h64_standard":
        q.zero_()  # Overflow publication and exercise standard H64 recovery.
    wrapper = FP8PagedMQATopKWrapper(
        strategy="selective",
        num_heads=heads,
        top_k=top_k,
        max_context_length=length,
        max_chunk_size=rows,
        max_batch_size=1,
        kv_layout=fused,
    )
    wrapper.prepare_pool(fused)
    engine = wrapper._capacity_plan._engine
    schedule = engine._candidate_schedule
    if topology == "high_split":
        assert schedule.split_kv > 8
    elif topology == "owner_pool":
        assert schedule.split_kv <= 8
        assert schedule.count_segments == schedule.split_kv
    elif topology == "segmented":
        assert schedule.count_segments > 1
    else:
        assert not engine._compact_repair
    wrapper.plan(qo, kv, table, num_query_rows=rows)
    _check_selection(wrapper.run(q, weights, fused), _oracle_rows(inputs), top_k)
    if topology == "h64_standard":
        assert engine._repair_flags.count_nonzero().item() > 0
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.plan(qo, kv, table, num_query_rows=rows)
        selected = wrapper.run(q, weights, fused)
    for tied in (False, True):
        q.copy_(torch.randn(q.shape, device=q.device).mul_(0.1).to(q.dtype))
        if tied:
            q.zero_()
        weights.mul_(-0.75)
        fused.copy_(fused.flip(0))
        table.copy_(fused.shape[0] - 1 - table)
        graph.replay()
        torch.cuda.synchronize()
        assert wrapper.error.item() == wrapper.repair_error.item() == 0
        _check_selection(selected, _oracle_rows(inputs), top_k)


@_requires_selective_sm100
@pytest.mark.parametrize("ratio", [1, 4])
def test_capacity_page_checks_stride_past_grid(ratio):
    from flashinfer.experimental.attn_scores.selective_capacity import (
        _CapacityStepInputs,
        _SelectiveCapacity,
    )

    check_ctas = 2 * torch.cuda.get_device_properties("cuda").multi_processor_count
    batches = check_ctas // 2 + 1
    columns, page_size = 2049, 32
    physical_length = columns * page_size
    capacity = _SelectiveCapacity(
        8, 128, page_size, ratio, physical_length * ratio, 2, batches
    )
    state = _CapacityStepInputs(capacity, device="cuda", num_pages=2)
    qo = torch.zeros(batches + 1, dtype=torch.int32, device="cuda")
    kv = torch.zeros_like(qo)
    qo[-1] = 1
    kv[-1] = physical_length * ratio
    table = torch.full((batches, columns), -1, dtype=torch.int32, device="cuda")
    table[-1].zero_()  # Invalid padding in empty requests must never be read.
    state.stage_step(qo, kv, table, num_query_rows=2)
    state.refresh()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        state.refresh()
    graph.replay()
    torch.cuda.synchronize()
    assert state.error.item() == 0
    assert state.active_shape.tolist() == [1, batches]
    assert state.row_ends.tolist() == [physical_length, 0]
    assert state.row_requests.tolist() == [batches - 1, -1]

    # Unit check_ctas+1 is CTA1's second iteration, in the second page chunk.
    # CTA0 and a single-pass check both miss this live invalid page.
    state.block_table[-1, 2048] = 2
    graph.replay()
    torch.cuda.synchronize()
    assert state.page_flags[0].item() == 0
    assert state.page_flags[1].item() == 8
    assert state.page_flags.count_nonzero().item() == 1
    assert state.error.item() == 8
    assert state.safe_qo_indptr.count_nonzero().item() == 0
    assert state.physical_kv_indptr.count_nonzero().item() == 0
    assert state.active_shape.count_nonzero().item() == 0
    assert state.row_ends.count_nonzero().item() == 0
    assert (state.row_requests == -1).all()
    state.block_table[-1, 2048] = 0
    graph.replay()
    torch.cuda.synchronize()
    assert state.error.item() == 0
    assert state.page_flags.count_nonzero().item() == 0
    assert state.row_requests.tolist() == [batches - 1, -1]


@_requires_selective_sm100
def test_capacity_schedule_large_batch_owner_walk():
    batches = 4097
    kv_length = 1024
    owners = [0, 2048, 4096]
    q_lengths = [int(request in owners) for request in range(batches)]
    kv_lengths = [kv_length * length for length in q_lengths]
    inputs = _request_inputs(q_lengths, kv_lengths)
    q, _, _, weights, qo, kv = inputs
    fused, table = _paged_inputs(inputs)
    pages = table[owners].clone()
    wrapper = FP8PagedMQATopKWrapper(
        strategy="selective",
        max_context_length=kv_length,
        max_chunk_size=3,
        max_batch_size=batches,
        kv_layout=fused,
    )
    wrapper.prepare_pool(fused)
    wrapper.plan(qo, kv, table, num_query_rows=3)
    _check_selection(wrapper.run(q, weights, fused), _oracle_rows(inputs), 512)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.plan(qo, kv, table, num_query_rows=3)
        selected = wrapper.run(q, weights, fused)

    # More than 4096 requests select the walk fallback; empty requests change
    # the owners without changing query storage, pool layout or graph bindings.
    for owners in ([1, 2047, 4095], [0, 2048, 4096]):
        q_prefix, kv_prefix = [0], [0]
        for request in range(batches):
            length = int(request in owners)
            q_prefix.append(q_prefix[-1] + length)
            kv_prefix.append(kv_prefix[-1] + kv_length * length)
        qo.copy_(torch.tensor(q_prefix, dtype=qo.dtype, device=qo.device))
        kv.copy_(torch.tensor(kv_prefix, dtype=kv.dtype, device=kv.device))
        table.zero_()
        table[owners] = pages
        graph.replay()
        torch.cuda.synchronize()
        assert wrapper.error.item() == wrapper.repair_error.item() == 0
        assert wrapper._capacity_plan._state.row_requests.tolist() == owners
        _check_selection(selected, _oracle_rows(inputs), 512)


@pytest.mark.parametrize("error,status", [(0, 0), (0, 1), (0, 2), (901, 2)])
def test_capture_status_names_invalid_boundary(monkeypatch, error, status):
    runtime = SimpleNamespace(
        cudaError_t=SimpleNamespace(cudaSuccess=0),
        cudaStreamCaptureStatus=SimpleNamespace(cudaStreamCaptureStatusInvalidated=2),
        cudaStreamGetCaptureInfo=lambda stream: (error, status, 123),
    )
    monkeypatch.setattr(sl, "_capture_runtime", lambda: runtime)
    stream = SimpleNamespace(cuda_stream=17)
    if error or status == 2:
        with pytest.raises(RuntimeError, match="after predicate launch"):
            sl._check_capture_status(stream, "predicate launch")
    else:
        sl._check_capture_status(stream, "predicate launch")


@_requires_selective_sm100
def test_caller_capture_status_failure_keeps_wrapper_usable(monkeypatch):
    """Discarding a failed caller graph does not retire the prepared bindings."""
    inputs = _request_inputs([2], [9001])
    q, _, _, weights, qo, kv = inputs
    pool, table = _paged_inputs(inputs)
    wrapper = FP8PagedMQATopKWrapper(
        strategy="selective",
        max_context_length=9001,
        max_chunk_size=2,
        max_batch_size=1,
        kv_layout=pool,
    )
    wrapper.prepare_pool(pool)
    wrapper.plan(qo, kv, table)
    invalid_runtime = SimpleNamespace(
        cudaError_t=SimpleNamespace(cudaSuccess=0),
        cudaStreamCaptureStatus=SimpleNamespace(cudaStreamCaptureStatusInvalidated=2),
        cudaStreamGetCaptureInfo=lambda stream: (0, 2, 0),
    )
    rejected = torch.cuda.CUDAGraph()
    with torch.cuda.graph(rejected), monkeypatch.context() as patch:
        patch.setattr(sl, "_capture_runtime", lambda: invalid_runtime)
        with pytest.raises(sl._CaptureStatusError, match="layer payload copies"):
            wrapper.run(q, weights, pool)
    # Only the status query is injected; the real stream remains valid so the
    # test can close/discard the caller's partial graph without a device reset.
    del rejected
    eager = wrapper.run(q, weights, pool).clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = wrapper.run(q, weights, pool)
    graph.replay()
    torch.cuda.synchronize()
    expected = _oracle_rows(inputs)
    for output in (eager, captured):
        _check_selection(output, expected, 512)
    assert wrapper.error.item() == wrapper.repair_error.item() == 0


@pytest.mark.parametrize(
    "fail_at",
    [
        None,
        "plan input copies",
        "plan repair-status reset",
        "plan metadata refresh",
        "plan sampling refresh",
        "plan schedule refresh",
    ],
)
def test_capacity_caller_capture_names_plan_boundary(monkeypatch, fail_at):
    from flashinfer.experimental.attn_scores.selective_capacity import (
        _CapacitySelectiveTopKPlan,
    )

    checks = []
    stream = SimpleNamespace(cuda_stream=17)

    def check(actual_stream, step):
        assert actual_stream is stream
        checks.append(step)
        if step == fail_at:
            raise sl._CaptureStatusError(f"invalid after {step}")

    monkeypatch.setattr(sl, "_check_capture_status", check)
    monkeypatch.setattr(torch.cuda, "device", lambda device: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: stream)
    plan = SimpleNamespace(
        _capacity=SimpleNamespace(max_chunk_size=4),
        _state=SimpleNamespace(
            device="cuda:0",
            is_prepared=True,
            _host_hint_validation=SimpleNamespace(stage=lambda hints: None),
            stage_step=lambda *args, **kwargs: None,
            refresh=lambda: None,
            refresh_sampling=lambda: None,
        ),
        _layout_needs_engine=lambda: False,
        _repair_error=SimpleNamespace(zero_=lambda: None),
        _needs_sampling=True,
        _refresh_schedules=lambda: None,
        _step_prepared=True,
    )
    plan._check_caller_capture = lambda capturing, step: (
        _CapacitySelectiveTopKPlan._check_caller_capture(plan, capturing, step)
    )
    prefix = SimpleNamespace(numel=lambda: 2)
    if fail_at is not None:
        with pytest.raises(sl._CaptureStatusError, match=fail_at):
            _CapacitySelectiveTopKPlan.plan(plan, prefix, prefix, object())
        assert not plan._step_prepared
        assert checks[-1] == fail_at
    else:
        _CapacitySelectiveTopKPlan.plan(plan, prefix, prefix, object())
        assert plan._step_prepared
        assert len(checks) == 5


def _case():
    q, kv, scales, weights, _, _ = _inputs(16, 8192)
    cu_q = torch.tensor([0, 16], dtype=torch.int32, device=q.device)
    cu_kv = torch.tensor([0, 8192], dtype=torch.int32, device=q.device)
    inputs = q, kv, scales, weights, cu_q, cu_kv
    fused, table = _paged_inputs(inputs)
    return inputs, fused, table


@_requires_selective_sm100
def test_owned_body_stream_survives_pool_wrap_replan_and_multiple_wrappers(monkeypatch):
    inputs, fused, table = _case()
    q, _, _, weights, cu_q, cu_kv = inputs
    pooled = [torch.cuda.Stream(device=q.device) for _ in range(96)]
    wrappers = [SelectiveEngineCase() for _ in range(2)]
    repair_calls = []
    original_repair = sl._FP8PagedMQASelectiveTopKEngine._run_repair

    def record_repair(engine, *args):
        repair_calls.append(torch.cuda.is_current_stream_capturing())
        return original_repair(engine, *args)

    monkeypatch.setattr(
        sl._FP8PagedMQASelectiveTopKEngine, "_run_repair", record_repair
    )
    handles = []
    for length in (1024, 8192, 1024):
        cu_kv[-1] = length
        # Bounded ties fit the candidate buffer; long ties overflow it.
        weights.zero_()
        for wrapper_index, wrapper in enumerate(wrappers):
            wrapper.plan(q, fused, weights, cu_q, cu_kv, table, num_sms=4)
            body = wrapper._repair_body_stream
            assert body.cuda_stream not in {stream.cuda_stream for stream in pooled}
            if length == 1024 and len(handles) < 2:
                handles.append(body.cuda_stream)
            assert body.cuda_stream == handles[wrapper_index]
            before = len(repair_calls)
            _check_selection(
                wrapper.run(q, fused, weights, table), _oracle_rows(inputs), 512
            )
            # No eager repair kernel is enqueued for the zero-repair steps,
            # including the short step following a forced exact repair.
            assert repair_calls[before:] == ([False] if length == 8192 else [])
            assert bool(wrapper._repair_flags.any()) == (length == 8192)
            for parent in (pooled[0], pooled[-1]):
                graph = torch.cuda.CUDAGraph()
                copied = torch.empty_like(wrapper._selected)
                with torch.cuda.graph(graph, stream=parent):
                    copied.copy_(wrapper.run(q, fused, weights, table))
                graph.replay()
                _check_selection(copied, _oracle_rows(inputs), 512)
                with torch.cuda.stream(body):
                    assert not torch.cuda.is_current_stream_capturing()
    assert len(set(handles)) == 2


@_requires_selective_sm100
@pytest.mark.parametrize("failure_site", ["begin", "body", "end"])
def test_body_capture_failure_preserves_exception_and_recovers(
    failure_site, monkeypatch
):
    inputs, fused, table = _case()
    q, _, _, weights, cu_q, cu_kv = inputs
    wrapper = SelectiveEngineCase()
    wrapper.plan(q, fused, weights, cu_q, cu_kv, table, num_sms=4)
    module = wrapper._repair_if_module
    body = wrapper._repair_body_stream
    original_repair = wrapper._run_repair
    finish_calls = []

    def finish(*args):
        finish_calls.append(True)
        return module.finish_if(*args)

    def fail_body(*args):
        wrapper._exact_error.zero_()
        raise RuntimeError("injected repair body failure")

    def fail_begin(*args):
        module.begin_body(*args)
        raise RuntimeError("injected repair begin failure")

    def fail_end(*args):
        module.end_body(*args)
        raise RuntimeError("injected repair end failure")

    proxy = SimpleNamespace(
        begin_if=module.begin_if,
        begin_body=fail_begin if failure_site == "begin" else module.begin_body,
        end_body=fail_end if failure_site == "end" else module.end_body,
        finish_if=finish,
        preflight_body=module.preflight_body,
        body_capture_status=module.body_capture_status,
        abort_body=module.abort_body,
    )
    monkeypatch.setattr(wrapper, "_repair_if_module", proxy)
    if failure_site == "body":
        monkeypatch.setattr(wrapper, "_run_repair", fail_body)
    graph = torch.cuda.CUDAGraph()
    with (
        pytest.raises(RuntimeError, match=f"injected repair {failure_site} failure"),
        torch.cuda.graph(graph),
    ):
        wrapper.run(q, fused, weights, table)
    # Only the outer context ends its caller-owned capture; the body is idle.
    assert not torch.cuda.is_current_stream_capturing()
    assert module.body_capture_status(q, body.cuda_stream) == 0
    assert not finish_calls
    monkeypatch.setattr(wrapper, "_repair_if_module", module)
    monkeypatch.setattr(wrapper, "_run_repair", original_repair)
    _check_selection(wrapper.run(q, fused, weights, table), _oracle_rows(inputs), 512)
    recovered = torch.cuda.CUDAGraph()
    with torch.cuda.graph(recovered):
        output = wrapper.run(q, fused, weights, table)
    recovered.replay()
    _check_selection(output, _oracle_rows(inputs), 512)


# Selective row tails and storage replanning


def _force_h64_candidate_layout(monkeypatch, variant: bool) -> None:
    """Pin single-pool H64 publication to the variant or the pooled layout."""
    import flashinfer.experimental.attn_scores.selective_logits as selective_logits

    monkeypatch.setattr(
        selective_logits,
        "_h64_candidate_variant_serves",
        lambda page_size: variant and page_size == 128,
    )


@_requires_selective_sm100
@pytest.mark.parametrize("heads,ratio", [(8, 4), (64, 1)])
@pytest.mark.parametrize("decode", [False, True])
def test_capacity_sparse_sample_dynamic_groups(heads, ratio, decode):
    """Request-local groups and heterogeneous B148 use refreshed device metadata."""
    rows, batches = (148, 148) if decode else (260, 3)
    q, keys, scales, weights, _, _ = _inputs(rows, 8192, num_heads=heads)
    fused = _fuse_paged_kv(keys, scales)
    table = (
        torch.arange(batches * 64, device=q.device, dtype=torch.int32)
        .remainder(fused.shape[0])
        .view(batches, 64)
    )
    top_k = 512 if heads == 8 else 2048
    wrapper = FP8PagedMQATopKWrapper(
        strategy="selective",
        num_heads=heads,
        top_k=top_k,
        compression_ratio=ratio,
        max_context_length=8192 * ratio,
        max_chunk_size=rows,
        max_batch_size=batches,
        kv_layout=fused,
    )
    wrapper.prepare_pool(fused)
    assert wrapper._capacity_plan._engine._sample_sparse_metadata is not None
    qo = torch.zeros(batches + 1, dtype=torch.int32, device=q.device)
    kv = torch.zeros_like(qo)

    def update(step):
        if decode:
            q_lengths = [int(step != 1 or r % 3 != 0) for r in range(batches)]
            kv_lengths = [
                max(q_lengths[r], ((r * 193 + step * 61) % 8193) * ratio)
                for r in range(batches)
            ]
        else:
            q_lengths = ([129, 0, 131], [64, 129, 67], [1, 0, 0])[step]
            kv_lengths = (
                [8192 * ratio, 0, 7011 * ratio],
                [6145 * ratio, 8003 * ratio, 4099 * ratio],
                [1, 0, 0],
            )[step]
        qo.copy_(
            torch.tensor(
                [0, *torch.tensor(q_lengths).cumsum(0).tolist()], device=q.device
            )
        )
        kv.copy_(
            torch.tensor(
                [0, *torch.tensor(kv_lengths).cumsum(0).tolist()], device=q.device
            )
        )
        table.copy_(table.roll(1, dims=1))
        return q_lengths, kv_lengths

    def check(selected, lengths):
        torch.cuda.synchronize()
        assert wrapper.error.item() == wrapper.repair_error.item() == 0
        expected, offset = [], 0
        for request, (ql, kl) in enumerate(zip(*lengths, strict=True)):
            for local in range(ql):
                end = (kl - ql + local + 1) // ratio
                slots = torch.arange(end, device=q.device)
                physical = table[request, slots // 128].long() * 128 + slots % 128
                dots = q[offset + local].float() @ keys[physical].float().T
                expected.append(
                    (dots.relu() * weights[offset + local, :, None]).sum(0)
                    * scales[physical]
                )
            offset += ql
        _check_selection(selected, expected, top_k)
        assert (selected[offset:] == -1).all()

    lengths = update(0)
    wrapper.plan(qo, kv, table, num_query_rows=rows)
    check(wrapper.run(q, weights, fused), lengths)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.plan(qo, kv, table, num_query_rows=rows)
        selected = wrapper.run(q, weights, fused)
    for step in (0, 1, 2, 0):
        lengths = update(step)
        graph.replay()
        check(selected, lengths)


def _check_selection(selected, expected_rows, top_k):
    for row, scores in enumerate(expected_rows):
        count = min(top_k, scores.numel())
        indices = selected[row, :count].long()
        assert ((indices >= 0) & (indices < scores.numel())).all()
        assert indices.unique().numel() == count
        torch.testing.assert_close(
            scores[indices].sort().values,
            scores.topk(count).values.sort().values,
            atol=5e-5,
            rtol=1e-5,
        )
        assert (selected[row, count:] == -1).all()


@_requires_selective_sm100
@pytest.mark.parametrize("heads", [8, 64])
@pytest.mark.parametrize("split_kv", [1, 4])
def test_q1_owner_pool_raw_counts_and_capacity_boundaries(
    heads, split_kv, record_property
):
    from flashinfer.experimental.attn_scores.selective_logits import (
        _finalize_fp8_mqa_packed_candidates,
    )

    capacity = 6144
    top_k = 512 if heads == 8 else 2048
    for kv_rows in (31, capacity, capacity + 1, 8193):
        inputs = _inputs(q_rows=1, kv_rows=kv_rows, num_heads=heads)
        prepared = _prepare(
            inputs,
            num_sms=8,
            num_heads=heads,
            query_tile=1,
            num_umma_stages=2,
            split_kv=split_kv,
        )
        values = torch.empty((16, capacity), device="cuda", dtype=torch.int64)
        counts = torch.empty((16, split_kv + 1), device="cuda", dtype=torch.int32)
        lengths = torch.empty(1, device="cuda", dtype=torch.int32)
        flags = torch.empty_like(lengths)
        threshold = torch.full((1,), float("-inf"), device="cuda")
        _launch(
            _SelectiveLogitsMode.CANDIDATE,
            capacity,
            values,
            counts,
            prepared,
            inputs,
            threshold,
        )
        raw = counts[0, 1:].clone()
        _finalize_fp8_mqa_packed_candidates(
            counts,
            values,
            lengths,
            flags,
            rows=1,
            capacity=capacity,
            top_k=top_k,
            _candidate_prepared=prepared,
        )
        torch.cuda.synchronize()
        assert int(raw.sum()) == kv_rows
        overflow = bool((raw > capacity // split_kv).any())
        assert int(counts[0, 0]) == (capacity + 1 if overflow else kv_rows)
        assert bool(flags[0]) == (overflow or kv_rows < top_k)
        record_property(
            f"raw_counts_{kv_rows}",
            dict(
                logical_row=0,
                owners=raw.cpu().tolist(),
                published_count=int(counts[0, 0]),
                overflow=overflow,
            ),
        )
        if kv_rows > capacity + 1:
            assert int(counts[0, 0]) != int(raw.sum())
        if not overflow:
            ids = (values[0, :kv_rows] & 0xFFFFFFFF).sort().values
            torch.testing.assert_close(
                ids, torch.arange(kv_rows, device="cuda", dtype=torch.int64)
            )
        # Raw owner counts are copied before the overflow sentinel is used.
        # Reusing the same normal launch must also reset every owner cursor.
        threshold.fill_(float("inf"))
        _launch(
            _SelectiveLogitsMode.CANDIDATE,
            capacity,
            values,
            counts,
            prepared,
            inputs,
            threshold,
        )
        _finalize_fp8_mqa_packed_candidates(
            counts,
            values,
            lengths,
            flags,
            rows=1,
            capacity=capacity,
            top_k=top_k,
            _candidate_prepared=prepared,
        )
        torch.cuda.synchronize()
        assert not bool(counts[0].any()) and int(lengths[0]) == 0
        assert int(flags[0]) == 1


@_requires_selective_sm100
@pytest.mark.parametrize("heads", [8, 64])
def test_q1_owner_pool_removes_only_local_vector_overflow(heads, record_property):
    from flashinfer.experimental.attn_scores.selective_logits import (
        _finalize_fp8_mqa_packed_candidates,
    )

    inputs = _inputs(q_rows=1, kv_rows=8192, num_heads=heads)
    q, kv, scales, weights, _, _ = inputs
    q.fill_(1)
    scales.fill_(1)
    weights.fill_(1)
    signs = torch.where(torch.arange(8192, device="cuda") % 4 == 0, 1.0, -1.0)
    kv.copy_(signs[:, None].expand(-1, 128).to(kv.dtype))
    prepared = _prepare(
        inputs, num_sms=4, num_heads=heads, query_tile=1, num_umma_stages=2, split_kv=1
    )
    threshold = torch.full((1,), 0.5, device="cuda")
    for segments in (4, 1):
        values = torch.empty((16, 6144), device="cuda", dtype=torch.int64)
        counts = torch.empty((16, segments + 1), device="cuda", dtype=torch.int32)
        lengths = torch.empty(1, device="cuda", dtype=torch.int32)
        flags = torch.empty_like(lengths)
        _launch(
            _SelectiveLogitsMode.CANDIDATE,
            6144,
            values,
            counts,
            prepared,
            inputs,
            threshold,
        )
        raw = counts[0, 1:].clone()
        _finalize_fp8_mqa_packed_candidates(
            counts,
            values,
            lengths,
            flags,
            rows=1,
            capacity=6144,
            top_k=2048,
            _candidate_prepared=prepared,
        )
        torch.cuda.synchronize()
        assert int(raw.sum()) == 2048
        assert int(flags[0]) == (1 if segments == 4 else 0)
        assert int(counts[0, 0]) == (6145 if segments == 4 else 2048)
        record_property(
            f"segments_{segments}",
            dict(
                logical_row=0,
                raw_counts=raw.cpu().tolist(),
                published_count=int(counts[0, 0]),
            ),
        )
        if segments == 1:
            ids = (values[0, :2048] & 0xFFFFFFFF).sort().values
            torch.testing.assert_close(
                ids, torch.arange(0, 8192, 4, device="cuda", dtype=torch.int64)
            )


@_requires_selective_sm100
@pytest.mark.parametrize(
    ("heads", "q_len", "kv_len"),
    [
        (8, 4, 257),
        (8, 16, 257),
        (8, 16, 8193),
        (64, 8, 513),
        (64, 8, 514),
        (64, 8, 8193),
    ],
)
@pytest.mark.parametrize("top_k", [64, 512])
def test_full_query_tile_causal_end_inside_penultimate_kv_pair(
    heads, q_len, kv_len, top_k
):
    """Full Q tiles must bound every KV pair that crosses a row's causal end.

    With an odd tile KV-block count and a context just past an even block
    boundary, early rows end inside the second-to-last 256-token KV pair.
    """
    inputs = _request_inputs([q_len], [kv_len], heads=heads)
    q, _, _, weights, cu_q, cu_kv = inputs
    fused, table = _paged_inputs(inputs)
    wrapper = SelectiveEngineCase()
    wrapper.plan(
        q,
        fused,
        weights,
        cu_q,
        cu_kv,
        table,
        top_k=top_k,
        num_heads=heads,
        head_dim=128,
    )
    selected = wrapper.run(q, fused, weights, table)
    torch.cuda.synchronize()
    _check_selection(selected, _oracle_rows(inputs), top_k)


@_requires_selective_sm100
@pytest.mark.parametrize(
    ("heads", "q_lengths", "kv_lengths"),
    [
        (8, [1, 1, 1], [600, 9000, 20000]),
        (8, [37], [20037]),
        (64, [9, 1], [12009, 3001]),
    ],
)
@pytest.mark.parametrize("ratio", [1, 4])
@pytest.mark.parametrize("top_k", [513, 1024, 2047])
def test_selective_top_k_between_512_and_2048(
    heads, q_lengths, kv_lengths, ratio, top_k
):
    """Every TopK width up to 2048 returns exact request-local selections."""
    if ratio == 1:
        inputs = _request_inputs(q_lengths, kv_lengths, heads=heads)
        q, _, _, weights, cu_q, cu_kv = inputs
        fused, table = _paged_inputs(inputs)
        payload = (q, fused, weights, cu_q, cu_kv, table)
    else:
        payload = _compressed_wrapper_fixture(
            q_lengths, [4 * length for length in kv_lengths], heads, 128
        )
        q, fused, weights, _, _, table = payload
    wrapper = FP8PagedMQATopKWrapper(
        strategy="selective",
        num_heads=heads,
        top_k=top_k,
        compression_ratio=ratio,
        max_context_length=max(kv_lengths) * ratio,
        max_chunk_size=q.shape[0],
        max_batch_size=len(q_lengths),
        kv_layout=fused,
    )
    wrapper.plan(payload[3], payload[4], table)
    selected = wrapper.run(q, weights, fused)
    torch.cuda.synchronize()
    assert selected.shape == (q.shape[0], top_k)
    assert wrapper.error.item() == wrapper.repair_error.item() == 0
    if ratio == 1:
        _check_selection(selected, _oracle_rows(inputs), top_k)
    else:
        _check_compressed_wrapper(selected, payload, top_k)


@_requires_selective_sm100
@pytest.mark.parametrize("ratio", [1, 4])
def test_policy_row_tails_forced_repair(ratio):
    """Q9/H64 leaves a padded Q2 lane in repair, radix and exact emission."""
    rows, length, top_k = 9, 9001, 512
    if ratio == 4:
        payload = _compressed_wrapper_fixture([rows], [length * 4 + 1], 64, 128)
        expected = None
    else:
        inputs = _inputs(rows, length, 64)
        q, kv, scales, weights, cu_q, cu_kv = inputs
        cu_q = torch.tensor([0, rows], device=q.device, dtype=torch.int32)
        cu_kv = torch.tensor([0, length], device=q.device, dtype=torch.int32)
        fused, table = _paged_inputs((q, kv, scales, weights, cu_q, cu_kv))
        payload = q, fused, weights, cu_q, cu_kv, table
    q, fused, weights, cu_q, cu_kv, table = payload
    # Ties overflow both bounded arenas, forcing all terminal radix passes and
    # EXACT rather than merely compiling the affected helpers.
    weights.zero_()
    if ratio == 1:
        expected = _oracle_rows((q, kv, scales, weights, cu_q, cu_kv))
    wrapper = SelectiveEngineCase()
    wrapper.plan(
        *payload, top_k=top_k, num_heads=64, num_sms=4, compression_ratio=ratio
    )
    assert wrapper._repair_prepared.query_tile == wrapper._prepared.query_tile == 2
    for policy in (
        wrapper._repair_thresholds,
        wrapper._prefixes,
        wrapper._threshold_bits,
    ):
        # Logical-size buffers cannot mask an out-of-bounds padded-row read.
        assert policy.numel() == rows

    def check(output):
        torch.cuda.synchronize()
        if ratio == 4:
            _check_compressed_wrapper(output, payload, top_k)
        else:
            _check_selection(output, expected, top_k)
        assert (wrapper._repair_flags != 0).all()
        assert (wrapper._second_repair_flags != 0).all()
        assert wrapper._exact_error.item() == 0

    check(wrapper._run_impl(q, fused, weights, table))
    check(wrapper.run(q, fused, weights, table))
    graph = torch.cuda.CUDAGraph()
    with _capture_cuda_graph(graph):
        captured = wrapper.run(q, fused, weights, table)
    for _ in range(2):
        graph.replay()
        check(captured)


# H64 query-tile scheduling and graph replay


@_requires_selective_sm100
@pytest.mark.parametrize(
    "q_rows,top_k,ties,num_sms",
    [
        # Fallback/query4 boundaries, publication layouts and persistent tails.
        (2, 512, False, 2),
        (2, 2048, True, 4),
        (4, 2048, False, 4),
        (4, 512, True, 2),
        (5, 512, False, 2),
        (5, 2048, True, 2),
        (5, 512, True, 4),
        (5, 2048, False, 4),
        (7, 512, False, 4),
        (7, 2048, True, 2),
        (33, 512, True, 2),
        (33, 2048, False, 4),
    ],
)
def test_h64_query4_eager_graph_and_tails(q_rows, top_k, ties, num_sms):
    inputs = list(_inputs(q_rows=q_rows, kv_rows=16384, num_heads=64))
    q, kv, scale, weights, _, _ = inputs
    inputs[4] = torch.tensor([0, q_rows], device="cuda", dtype=torch.int32)
    inputs[5] = torch.tensor([0, kv.shape[0]], device="cuda", dtype=torch.int32)
    if ties:
        q.zero_()
    fused, table = _paged_inputs(tuple(inputs))
    wrapper = SelectiveEngineCase(math_mode="relu_strict")
    wrapper.plan(
        q,
        fused,
        weights,
        inputs[4],
        inputs[5],
        table,
        top_k=top_k,
        num_heads=64,
        head_dim=128,
        num_sms=num_sms,
    )
    candidate = wrapper._candidate_prepared
    assert candidate.query_tile == (4 if q_rows > 4 else 2)
    assert candidate.num_umma_stages == (1 if q_rows > 4 else 2)
    if q_rows == 5:
        assert candidate.split_kv == (1 if num_sms == 2 else 2)
        # Split-1 H64 prefill uses the H64 candidate variant's 4 segments.
        assert wrapper._candidate_schedule.h64_candidate == (num_sms == 2)
        assert wrapper._candidate_schedule.publication == "warp_striped"
        assert wrapper._candidate_schedule.count_segments == (4 if num_sms == 2 else 8)
        assert wrapper._candidate_schedule.sample_count_segments == (
            32 if num_sms == 2 else 64
        )
    for prepared in (
        wrapper._sample_prepared,
        wrapper._repair_prepared,
        wrapper._prepared,
    ):
        assert prepared.query_tile == 2
        assert prepared.num_umma_stages == 2
    selected = wrapper.run(q, fused, weights, table)
    torch.cuda.synchronize()
    _check_selection(selected, _oracle_rows(inputs), top_k)
    if ties:
        assert (wrapper._repair_flags != 0).all()
    assert int(wrapper._exact_error.item()) == 0
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = wrapper.run(q, fused, weights, table)
    graph.replay()
    torch.cuda.synchronize()
    _check_selection(captured, _oracle_rows(inputs), top_k)
    assert selected.data_ptr() == captured.data_ptr()
    # Change physical placement and logical Q/scale/weight values in place.
    fused.copy_(fused.flip(0))
    table.copy_(fused.shape[0] - 1 - table)
    q.copy_(q.float().neg().to(q.dtype))
    weights[:, ::2].mul_(-1)
    scale[::2].mul_(2)
    fused.flatten(1)[:, 128 * 128 :].view(torch.float32)[:, ::2].mul_(2)
    graph.replay()
    torch.cuda.synchronize()
    _check_selection(captured, _oracle_rows(inputs), top_k)
    assert int(wrapper._exact_error.item()) == 0


@_requires_selective_sm100
@pytest.mark.parametrize("top_k", [512, 2048])
@pytest.mark.parametrize(
    ("q_lengths", "kv_lengths"),
    [([1, 36], [132, 16252]), ([40, 7, 64], [5000, 300, 9000])],
)
def test_h64_candidate_variant_matches_pooled_layout(
    q_lengths, kv_lengths, top_k, monkeypatch
):
    """The H64 candidate variant publishes the pooled layout's candidates.

    Covers ragged multi-request prefill, partial query tiles and short rows.
    """
    inputs = _request_inputs(q_lengths, kv_lengths, heads=64)
    q, _kv, _scale, weights, cu_q, cu_kv = inputs
    fused, table = _paged_inputs(inputs)
    oracle = _oracle_rows(inputs)
    rows = sum(q_lengths)
    selected, lengths = {}, {}
    for h64_candidate in (False, True):
        _force_h64_candidate_layout(monkeypatch, h64_candidate)
        engine = SelectiveEngineCase(math_mode="relu_strict")
        engine.plan(
            q,
            fused,
            weights,
            cu_q,
            cu_kv,
            table,
            top_k=top_k,
            num_heads=64,
            head_dim=128,
            num_sms=2,
        )
        assert engine._candidate_schedule.h64_candidate is h64_candidate
        selected[h64_candidate] = engine.run(q, fused, weights, table).clone()
        torch.cuda.synchronize()
        lengths[h64_candidate] = engine._candidate_lengths[:rows].clone()
        assert int(engine._exact_error.item()) == 0
        _check_selection(selected[h64_candidate], oracle, top_k)
    # Identical scores and thresholds publish identical candidate populations.
    torch.testing.assert_close(lengths[True], lengths[False], atol=0, rtol=0)
    torch.testing.assert_close(
        selected[True].sort(1).values, selected[False].sort(1).values, atol=0, rtol=0
    )


@_requires_selective_sm100
@pytest.mark.parametrize("top_k", [512, 2048])
def test_query4_mixed_empty_short_and_tail_request(top_k):
    inputs = list(_inputs(q_rows=6, kv_rows=16384, num_heads=64))
    q, _, _, weights, _, _ = inputs
    inputs[4] = torch.tensor([0, 0, 1, 6], device="cuda", dtype=torch.int32)
    inputs[5] = torch.tensor([0, 0, 64, 16384], device="cuda", dtype=torch.int32)
    fused, table = _paged_inputs(tuple(inputs))
    table[0].fill_(fused.shape[0] + 17)
    wrapper = SelectiveEngineCase(math_mode="relu_strict")
    wrapper.plan(
        q,
        fused,
        weights,
        inputs[4],
        inputs[5],
        table,
        top_k=top_k,
        num_heads=64,
        head_dim=128,
        num_sms=4,
    )
    assert wrapper._candidate_prepared.query_tile == 4
    selected = wrapper.run(q, fused, weights, table)
    torch.cuda.synchronize()
    _check_selection(selected, _oracle_rows(inputs), top_k)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = wrapper.run(q, fused, weights, table)
    graph.replay()
    torch.cuda.synchronize()
    _check_selection(captured, _oracle_rows(inputs), top_k)
    assert int(wrapper._exact_error.item()) == 0


# K2048 population targets and rank refresh


def _check_rank_bounds(wrapper, top_k):
    valid = wrapper._sample_ends.cpu().tolist()
    ends = wrapper._row_ends.cpu().tolist()
    if wrapper._sample_sparse_metadata is not None:
        qo = wrapper._cu_q.cpu().tolist()
        group = sl._select_sparse_sample_group_rows(wrapper._sample_prepared.query_tile)
        ends = [
            ends[begin + (row - begin) // group * group]
            for begin, last in zip(qo[:-1], qo[1:], strict=True)
            for row in range(begin, last)
        ]
        assert valid == [_expected_sample_count(end, top_k) for end in ends]
    # Independent expected population targets for K512 and K2048.
    expected = [
        _expected_sample_rank(
            end, count, top_k, wrapper._sample_sparse_metadata is not None
        )
        for count, end in zip(valid, ends, strict=True)
    ]
    assert wrapper._sample_ranks.cpu().tolist() == expected
    assert max(expected) <= wrapper._sample_rank_capacity
    assert max(expected) <= sl._SAMPLE_CAPACITY


@_requires_selective_sm100
@pytest.mark.parametrize("heads", [8, 64])
def test_k2048_changed_lengths_refresh_ranks_and_repair(heads):
    inputs = list(_inputs(q_rows=5, kv_rows=16384, num_heads=heads))
    q, _, _, weights, cu_q, _ = inputs
    cu_q = torch.tensor([0, 5], device="cuda", dtype=torch.int32)
    inputs[4] = cu_q
    inputs[5] = torch.tensor([0, 16384], device="cuda", dtype=torch.int32)
    fused, table = _paged_inputs(tuple(inputs))
    wrapper = FP8PagedMQATopKWrapper(
        strategy="selective",
        num_heads=heads,
        top_k=2048,
        max_context_length=16384,
        max_chunk_size=5,
        max_batch_size=1,
        kv_layout=fused,
    )
    wrapper.prepare_pool(fused)
    engine = wrapper._capacity_plan._engine
    rank_pointer = None
    prior_ranks = None
    retained = None
    for generation, length in enumerate((12288, 12289, 16384, 12288)):
        inputs[5] = torch.tensor([0, length], device="cuda", dtype=torch.int32)
        if generation == 1:
            weights[0].zero_()  # One row needs the exact overflow path.
        wrapper.plan(cu_q, inputs[5], table)
        selected = wrapper.run(q, weights, fused)
        _check_rank_bounds(engine, 2048)
        assert engine._row_ends[-1].item() == length
        ranks = engine._sample_ranks.cpu().tolist()
        if rank_pointer is None:
            rank_pointer = engine._sample_ranks.data_ptr()
            retained = (
                engine._sample_scores,
                engine._candidate_values,
                engine._selected,
            )
        else:
            assert engine._sample_ranks.data_ptr() == rank_pointer
            # A one-token change can retain the group's rounded target rank.
            if generation != 1:
                assert ranks != prior_ranks
            assert all(
                current is original
                for current, original in zip(
                    (
                        engine._sample_scores,
                        engine._candidate_values,
                        engine._selected,
                    ),
                    retained,
                    strict=True,
                )
            )
        prior_ranks = ranks
        torch.cuda.synchronize()
        _check_selection(selected, _oracle_rows(inputs), 2048)
        assert wrapper.error.item() == wrapper.repair_error.item() == 0
        if generation:
            assert int(engine._repair_flags[0]) != 0


def test_wrapper_math_mode_rejects_before_cuda_work():
    for invalid in ("auto", "", None, 1):
        with pytest.raises(ValueError, match="math_mode"):
            FP8PagedMQATopKWrapper(strategy="selective", math_mode=invalid)
    wrapper = object.__new__(sl.FP8PagedMQATopKWrapper)
    wrapper._math_mode = wrapper._math_mode_at_construction = "relu_strict"
    assert wrapper.math_mode == "relu_strict"
    with pytest.raises(AttributeError):
        wrapper.math_mode = "relu_fast"
    wrapper._math_mode = "relu_fast"
    with pytest.raises(RuntimeError, match="math_mode"):
        wrapper.run(q=None, kv_fused=None, weights=None)


def _full_plan_regression_inputs(strategy="full"):
    torch.backends.cuda.matmul.allow_tf32 = False
    inputs = _request_inputs([2], [1031], heads=64)
    q, _, _, weights, qo, kv = inputs
    pool, original = _paged_inputs(inputs)
    table = torch.zeros((1, 32), device=q.device, dtype=torch.int32)
    table[:, : original.shape[1]] = original
    wrapper = FP8PagedMQATopKWrapper(
        strategy=strategy,
        num_heads=64,
        top_k=512,
        max_context_length=4096,
        max_chunk_size=8,
        max_batch_size=1,
        kv_layout=pool,
    )
    return wrapper, inputs, pool, table


@_requires_selective_sm100
@pytest.mark.parametrize("hinted", [False, True])
def test_full_first_eager_run_prepares_lazily(monkeypatch, hinted, ieee_matmul):
    wrapper, inputs, pool, table = _full_plan_regression_inputs()
    q, _, _, weights, qo, kv = inputs
    options = dict(num_query_rows=2, next_n=2)
    if hinted:
        options["host_length_hints"] = (
            qo.cpu(),
            torch.tensor([1031], dtype=torch.int32),
        )
    wrapper.plan(qo, kv, table, **options)
    expected = _oracle_rows(inputs)
    _check_selection(wrapper.run(q, weights, pool), expected, 512)
    torch.cuda.synchronize()

    def reject_setup(*args, **kwargs):
        pytest.fail("warmed full run must not prepare or synchronize with the host")

    plan = wrapper._full_capacity_plan
    allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
    with monkeypatch.context() as patch:
        patch.setattr(type(plan), "prepare_pool", reject_setup)
        patch.setattr(type(plan._host_hints), "prepare_pool", reject_setup)
        patch.setattr(torch.cuda, "synchronize", reject_setup)
        for name in ("item", "cpu", "tolist"):
            patch.setattr(torch.Tensor, name, reject_setup)
        for _ in range(2):
            selected = wrapper.run(q, weights, pool)
    assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
    _check_selection(selected, expected, 512)
    assert wrapper.error.item() == wrapper.repair_error.item() == 0
    owner, plan_owner = weakref.ref(wrapper), weakref.ref(plan)
    del plan, wrapper
    assert owner() is None
    assert plan_owner() is None


@_requires_selective_sm100
def test_full_unwarmed_capture_rejection_keeps_capture_valid(monkeypatch, ieee_matmul):
    wrapper, inputs, pool, table = _full_plan_regression_inputs()
    q, _, _, weights, qo, kv = inputs
    wrapper.plan(qo, kv, table, num_query_rows=2, next_n=2)
    sentinel = torch.zeros(1, device=q.device)

    def reject_setup(*args, **kwargs):
        pytest.fail("capture miss must fail before preparation or CUDA launches")

    with monkeypatch.context() as patch:
        patch.setattr(type(wrapper._full_capacity_plan), "prepare_pool", reject_setup)
        patch.setattr(type(wrapper._full_capacity_plan), "_execute", reject_setup)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            with pytest.raises(
                RuntimeError, match="one eager run must precede caller capture"
            ):
                wrapper.run(q, weights, pool)
            sentinel.fill_(1)
        graph.replay()
        assert sentinel.item() == 1
        layout_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(layout_graph):
            with pytest.raises(RuntimeError, match="capturing a new full layout"):
                wrapper.plan(qo, kv, table, num_query_rows=2, next_n=1)
            sentinel.fill_(2)
        layout_graph.replay()
        assert sentinel.item() == 2
    wrapper.plan(qo, kv, table, num_query_rows=2, next_n=2)
    _check_selection(wrapper.run(q, weights, pool), _oracle_rows(inputs), 512)
    warmed = torch.cuda.CUDAGraph()
    with torch.cuda.graph(warmed):
        selected = wrapper.run(q, weights, pool)
    warmed.replay()
    _check_selection(selected, _oracle_rows(inputs), 512)


@_requires_selective_sm100
def test_full_hinted_lazy_preparation_failure_retries(monkeypatch, ieee_matmul):
    wrapper, inputs, pool, table = _full_plan_regression_inputs()
    q, _, _, weights, qo, kv = inputs
    wrapper.plan(
        qo,
        kv,
        table,
        num_query_rows=2,
        host_length_hints=(qo.cpu(), torch.tensor([1031], dtype=torch.int32)),
    )
    hinted_type = type(wrapper._full_capacity_plan._host_hints)
    bind = hinted_type._bind
    failed = False

    def fail_once(self, *args):
        nonlocal failed
        if not failed:
            failed = True
            raise RuntimeError("injected hinted binding failure")
        return bind(self, *args)

    with monkeypatch.context() as patch:
        patch.setattr(hinted_type, "_bind", fail_once)
        with pytest.raises(RuntimeError, match="injected hinted binding failure"):
            wrapper.run(q, weights, pool)
        selected = wrapper.run(q, weights, pool)
    _check_selection(selected, _oracle_rows(inputs), 512)
    assert wrapper.error.item() == wrapper.repair_error.item() == 0
    owner = weakref.ref(wrapper)
    del wrapper
    assert owner() is None


@_requires_selective_sm100
def test_full_hinted_run_rejects_capture_before_payload(monkeypatch):
    wrapper, inputs, pool, table = _full_plan_regression_inputs()
    q, _, _, weights, qo, kv = inputs
    hints = (qo.cpu(), torch.tensor([1031], dtype=torch.int32))
    wrapper.plan(qo, kv, table, num_query_rows=2, host_length_hints=hints)
    wrapper.prepare_pool(pool)
    _check_selection(wrapper.run(q, weights, pool), _oracle_rows(inputs), 512)
    graph = torch.cuda.CUDAGraph()
    sentinel = torch.zeros(1, device=q.device)

    def reject_payload(*args, **kwargs):
        pytest.fail("hinted capture reached payload staging")

    with monkeypatch.context() as patch:
        patch.setattr(type(wrapper._full_state), "_payload_copies", reject_payload)
        with torch.cuda.graph(graph):
            with pytest.raises(RuntimeError, match="host_length_hints"):
                wrapper.run(q, weights, pool)
            sentinel.fill_(1)
    graph.replay()
    assert sentinel.item() == 1
    wrapper.plan(qo, kv, table, num_query_rows=2, next_n=2)
    wrapper.prepare_pool(pool)
    _check_selection(wrapper.run(q, weights, pool), _oracle_rows(inputs), 512)


@_requires_selective_sm100
@pytest.mark.parametrize("strategy", ["full", "selective"])
@pytest.mark.parametrize("invalid", ["next_n", "table", "hints"])
def test_capacity_failed_plan_requires_replan(strategy, invalid):
    wrapper, inputs, pool, table = _full_plan_regression_inputs(strategy)
    q, _, _, weights, qo, kv = inputs
    valid = dict(num_query_rows=2, next_n=2)
    wrapper.plan(qo, kv, table, **valid)
    wrapper.prepare_pool(pool)
    expected = _oracle_rows(inputs)
    _check_selection(wrapper.run(q, weights, pool), expected, 512)
    bad = dict(valid)
    bad_table = table
    if invalid == "next_n":
        bad["next_n"] = 3
    elif invalid == "table":
        bad.update(num_query_rows=3, next_n=1)
        bad_table = table.to(torch.int64)
    else:
        bad["host_length_hints"] = (qo, kv)
    with pytest.raises(ValueError):
        wrapper.plan(qo, kv, bad_table, **bad)
    with pytest.raises(RuntimeError, match="plan must be called"):
        wrapper.run(q, weights, pool)
    wrapper.plan(qo, kv, table, **valid)
    wrapper.prepare_pool(pool)
    _check_selection(wrapper.run(q, weights, pool), expected, 512)


@_requires_selective_sm100
@pytest.mark.parametrize("hinted", [False, True])
def test_full_pool_warmup_preserves_inactive_output(hinted):
    wrapper, inputs, pool, table = _full_plan_regression_inputs()
    q, keys, scales, weights, qo, kv = inputs
    # Prime the eager hint buffers with more live rows than the next uniform
    # step. An unhinted plan intentionally does not refresh those buffers.
    prior_q = torch.cat((q, q))
    prior_weights = torch.cat((weights, weights))
    prior_qo = torch.tensor([0, 4], device=q.device, dtype=torch.int32)
    wrapper.plan(
        prior_qo,
        kv,
        table,
        num_query_rows=4,
        host_length_hints=(prior_qo.cpu(), torch.tensor([1031], dtype=torch.int32)),
    )
    wrapper.prepare_pool(pool)
    _check_selection(
        wrapper.run(prior_q, prior_weights, pool),
        _oracle_rows((prior_q, keys, scales, prior_weights, prior_qo, kv)),
        512,
    )
    padded_q = torch.cat((q, q[:1]))
    padded_weights = torch.cat((weights, weights[:1]))
    kwargs = dict(num_query_rows=3, next_n=2)
    if hinted:
        kwargs["host_length_hints"] = (
            qo.cpu(),
            torch.tensor([1031], dtype=torch.int32),
        )
    wrapper.plan(qo, kv, table, **kwargs)
    expected = [*_oracle_rows(inputs), torch.empty(0, device=q.device)]
    for layer_pool in (pool, pool.clone()):
        wrapper.prepare_pool(layer_pool)
        result = wrapper.run(padded_q, padded_weights, layer_pool)
        _check_selection(result, expected, 512)
        assert (result[2] == -1).all()
        assert wrapper.error.item() == 0
