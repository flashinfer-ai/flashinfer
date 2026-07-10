"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import pytest
import torch

import flashinfer

NUM_QO_HEADS = 8
NUM_KV_HEADS = 8
HEAD_DIM = 128
PAGE_SIZE = 16


def _make_decode_wrapper(use_cuda_graph=False, batch_size=1, max_num_pages=8):
    """Build a decode wrapper; in graph mode, allocate the fixed-size index
    buffers the CUDA-graph contract requires."""
    workspace = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
    if use_cuda_graph:
        return flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            workspace,
            "NHD",
            use_cuda_graph=True,
            paged_kv_indptr_buffer=torch.empty(
                batch_size + 1, dtype=torch.int32, device="cuda:0"
            ),
            paged_kv_indices_buffer=torch.empty(
                max_num_pages, dtype=torch.int32, device="cuda:0"
            ),
            paged_kv_last_page_len_buffer=torch.empty(
                batch_size, dtype=torch.int32, device="cuda:0"
            ),
        )
    return flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD")


def _plan_decode(wrapper, kv_len=37):
    """Plan a single-request batch of length ``kv_len`` and return matching
    (q, kv_cache) inputs for run()."""
    num_pages = (kv_len + PAGE_SIZE - 1) // PAGE_SIZE
    kv_indptr = torch.tensor([0, num_pages], dtype=torch.int32, device="cuda:0")
    kv_indices = torch.arange(num_pages, dtype=torch.int32, device="cuda:0")
    kv_last_page_len = torch.tensor(
        [(kv_len - 1) % PAGE_SIZE + 1], dtype=torch.int32, device="cuda:0"
    )
    wrapper.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        NUM_QO_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        PAGE_SIZE,
        q_data_type=torch.float16,
    )
    kv_cache = torch.randn(
        num_pages,
        2,
        PAGE_SIZE,
        NUM_KV_HEADS,
        HEAD_DIM,
        dtype=torch.float16,
        device="cuda:0",
    )
    q = torch.randn(1, NUM_QO_HEADS, HEAD_DIM, dtype=torch.float16, device="cuda:0")
    return q, kv_cache


def _warmup_on_side_stream(fn, iters=3):
    """Run ``fn`` a few times on a side stream before graph capture (the
    torch-documented warmup idiom, as used by the existing CG decode test)."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(iters):
            fn()
    torch.cuda.current_stream().wait_stream(s)


def test_decode_default_mode_capture_raises():
    """Capturing run() of a default-mode decode wrapper must fail loudly,
    with a message that names the problem, the fix, and the escape hatch —
    and must leave the wrapper usable in eager mode afterwards."""
    wrapper = _make_decode_wrapper(use_cuda_graph=False)
    q, kv_cache = _plan_decode(wrapper)
    wrapper.run(q, kv_cache)  # eager run must not raise

    g = torch.cuda.CUDAGraph()
    with pytest.raises(RuntimeError) as excinfo, torch.cuda.graph(g):
        wrapper.run(q, kv_cache)
    msg = str(excinfo.value)
    assert "BatchDecodeWithPagedKVCacheWrapper" in msg
    assert "use_cuda_graph=False" in msg  # names the problem
    assert "use_cuda_graph=True" in msg  # names the fix
    assert "FLASHINFER_ALLOW_UNSAFE_GRAPH_CAPTURE" in msg  # names the escape hatch

    # the aborted capture must not poison subsequent eager use
    wrapper.run(q, kv_cache)
    torch.cuda.synchronize()


def test_prefill_paged_default_mode_capture_raises():
    """Same guard on the paged prefill wrapper (the case reported in #3904)."""
    workspace = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD")
    qo_indptr = torch.tensor([0, 8], dtype=torch.int32, device="cuda:0")
    kv_indptr = torch.tensor([0, 2], dtype=torch.int32, device="cuda:0")
    kv_indices = torch.arange(2, dtype=torch.int32, device="cuda:0")
    kv_last_page_len = torch.tensor([PAGE_SIZE], dtype=torch.int32, device="cuda:0")
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        NUM_QO_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        PAGE_SIZE,
        causal=True,
        q_data_type=torch.float16,
    )
    q = torch.randn(8, NUM_QO_HEADS, HEAD_DIM, dtype=torch.float16, device="cuda:0")
    kv_cache = torch.randn(
        2, 2, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=torch.float16, device="cuda:0"
    )
    wrapper.run(q, kv_cache)

    g = torch.cuda.CUDAGraph()
    with (
        pytest.raises(RuntimeError, match="BatchPrefillWithPagedKVCacheWrapper"),
        torch.cuda.graph(g),
    ):
        wrapper.run(q, kv_cache)


def test_prefill_ragged_default_mode_capture_raises():
    """Same guard on the ragged prefill wrapper."""
    workspace = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(workspace, "NHD")
    qo_indptr = torch.tensor([0, 8], dtype=torch.int32, device="cuda:0")
    kv_indptr = torch.tensor([0, 32], dtype=torch.int32, device="cuda:0")
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        NUM_QO_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        causal=True,
        q_data_type=torch.float16,
    )
    q = torch.randn(8, NUM_QO_HEADS, HEAD_DIM, dtype=torch.float16, device="cuda:0")
    k = torch.randn(32, NUM_KV_HEADS, HEAD_DIM, dtype=torch.float16, device="cuda:0")
    v = torch.randn(32, NUM_KV_HEADS, HEAD_DIM, dtype=torch.float16, device="cuda:0")
    wrapper.run(q, k, v)

    g = torch.cuda.CUDAGraph()
    with (
        pytest.raises(RuntimeError, match="BatchPrefillWithRaggedKVCacheWrapper"),
        torch.cuda.graph(g),
    ):
        wrapper.run(q, k, v)


@pytest.mark.parametrize("env_value,should_bypass", [("1", True), ("0", False)])
def test_escape_hatch_env_var(monkeypatch, env_value, should_bypass):
    """FLASHINFER_ALLOW_UNSAFE_GRAPH_CAPTURE=1 restores the old (unsafe)
    behavior — this is exactly what the reporter's setup did by accident —
    while "0" must NOT bypass the check."""
    monkeypatch.setenv("FLASHINFER_ALLOW_UNSAFE_GRAPH_CAPTURE", env_value)
    wrapper = _make_decode_wrapper(use_cuda_graph=False)
    q, kv_cache = _plan_decode(wrapper)

    g = torch.cuda.CUDAGraph()
    if should_bypass:
        _warmup_on_side_stream(lambda: wrapper.run(q, kv_cache))
        with torch.cuda.graph(g):
            wrapper.run(q, kv_cache)
        g.replay()
        torch.cuda.synchronize()
    else:
        wrapper.run(q, kv_cache)
        with (
            pytest.raises(RuntimeError, match="use_cuda_graph=False"),
            torch.cuda.graph(g),
        ):
            wrapper.run(q, kv_cache)


def test_graph_mode_wrapper_capture_not_blocked():
    """The guard must not false-positive on a wrapper constructed with
    use_cuda_graph=True. (Full graph-mode replay correctness is covered by
    test_batch_decode_kernels.py::test_cuda_graph_batch_decode_with_paged_kv_cache.)
    """
    wrapper = _make_decode_wrapper(use_cuda_graph=True, batch_size=1)
    q, kv_cache = _plan_decode(wrapper)
    _warmup_on_side_stream(lambda: wrapper.run(q, kv_cache))

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        wrapper.run(q, kv_cache)
    g.replay()
    torch.cuda.synchronize()
