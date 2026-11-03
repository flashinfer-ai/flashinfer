# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Resolve the capture-status bindings on their first use inside capture."""

import os
import subprocess
import sys

import torch

from tests.experimental.attn_scores.test_attn_scores import (
    FP8PagedMQATopKWrapper,
    _check_selection,
    _oracle_rows,
    _paged_inputs,
    _request_inputs,
    _requires_selective_sm100,
    sl,
)


@_requires_selective_sm100
def test_selective_first_capture_resolves_runtime_in_fresh_process(monkeypatch):
    if os.environ.get("FLASHINFER_TEST_FRESH_CAPTURE_CHILD") != "1":
        child = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                f"{__file__}::test_selective_first_capture_resolves_runtime_in_fresh_process",
                "-s",
            ],
            env={**os.environ, "FLASHINFER_TEST_FRESH_CAPTURE_CHILD": "1"},
            capture_output=True,
            text=True,
            timeout=300,
        )
        print(child.stdout)
        assert child.returncode == 0, child.stdout + child.stderr
        assert "FIRST_RUNTIME_IMPORT_CAPTURE_PASS" in child.stdout
        return

    torch.backends.cuda.matmul.allow_tf32 = False
    inputs = _request_inputs([4], [4096], heads=64)
    q, _, _, weights, qo, kv = inputs
    pool, original = _paged_inputs(inputs)
    table = torch.zeros((1, 64), dtype=torch.int32, device=q.device)
    table[:, : original.shape[1]] = original
    wrapper = FP8PagedMQATopKWrapper(
        strategy="selective",
        num_heads=64,
        top_k=512,
        max_context_length=8192,
        max_chunk_size=4,
        max_batch_size=1,
        kv_layout=pool,
    )
    wrapper.plan(qo, kv, table)
    wrapper.prepare_pool(pool)
    runtime = sl._capture_runtime
    assert runtime.cache_info().misses == 0  # No pre-resolution or cache reset.
    imported = []

    def resolve():
        if runtime.cache_info().misses == 0:
            assert torch.cuda.is_current_stream_capturing()
            imported.append("cuda.bindings.runtime" in sys.modules)
        return runtime()

    monkeypatch.setattr(sl, "_capture_runtime", resolve)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.plan(qo, kv, table)
        captured = wrapper.run(q, weights, pool)
    captured.fill_(-2)
    graph.replay()
    _check_selection(captured, _oracle_rows(inputs), 512)
    assert wrapper.error.item() == 0
    assert len(imported) == 1 and runtime.cache_info().misses == 1
    print("FIRST_RUNTIME_IMPORT_CAPTURE_PASS", {"runtime_preimported": imported[0]})
