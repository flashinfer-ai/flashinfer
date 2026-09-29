# Copyright (c) 2026 by FlashInfer team.
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

"""Real SM120 regressions for launch-owned tensor-map descriptors.

Each case runs in a child process: stale invalid descriptors in an unfixed
binding can fault the CUDA context, which must not poison subsequent tests.
"""

import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from flashinfer.cute_dsl.sparse.bsa_attn_sm120 import (
    bsa_attn_sm120_blk64_sage_fwd,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or tuple(torch.cuda.get_device_capability()) != (12, 0),
    reason="generated Sage block-sparse attention requires compute capability 12.0",
)


def _inputs(value):
    # Uniform attention to a constant V has an exact BF16 result, independent
    # of Q/K quantization and V token permutation. Distinct live allocations
    # expose a valid stale descriptor silently reading/writing the wrong tensor.
    return dict(
        Q=torch.zeros((1, 1, 64, 128), dtype=torch.int8, device="cuda"),
        K=torch.zeros((1, 1, 128, 128), dtype=torch.int8, device="cuda"),
        V=torch.full((1, 1, 128, 128), value, dtype=torch.float32, device="cuda").to(
            torch.float8_e4m3fn
        ),
        Q_scale=torch.ones((1, 1, 4), dtype=torch.float32, device="cuda"),
        K_scale=torch.ones((1, 1, 2), dtype=torch.float32, device="cuda"),
        V_scale=torch.ones((1, 1, 128), dtype=torch.float32, device="cuda"),
        q2k_block_index=torch.tensor([[[[0, 1]]]], dtype=torch.int32, device="cuda"),
        q2k_block_nums=torch.full((1, 1, 1), 2, dtype=torch.int32, device="cuda"),
        block_sparse_num=2,
        out=torch.empty((1, 1, 64, 128), dtype=torch.bfloat16, device="cuda"),
    )


def _workspace():
    workspace = torch.full((512,), 0xA5, dtype=torch.uint8, device="cuda")
    assert workspace.data_ptr() % 128 == 0
    return workspace


def _launch(inputs, workspace):
    result = bsa_attn_sm120_blk64_sage_fwd(
        inputs["Q"],
        inputs["K"],
        inputs["V"],
        inputs["Q_scale"],
        inputs["K_scale"],
        inputs["V_scale"],
        inputs["q2k_block_index"],
        inputs["block_sparse_num"],
        q2k_block_nums=inputs["q2k_block_nums"],
        out=inputs["out"],
        tma_descriptor_workspace=workspace,
        uniform_block_count=True,
        contiguous_block_indices=True,
        backend="cake",
    )
    assert result is inputs["out"]


def _check(inputs, value):
    torch.testing.assert_close(
        inputs["out"], torch.full_like(inputs["out"], value), rtol=1e-2, atol=1e-2
    )


def _check_workspace(workspace, value):
    torch.testing.assert_close(
        workspace, torch.full_like(workspace, value), rtol=0, atol=0
    )


def _zeroed_workspace():
    inputs, workspace = _inputs(1.0), _workspace()
    _launch(inputs, workspace)
    _check(inputs, 1.0)
    _check_workspace(workspace, 0xA5)
    # No host synchronization between the overwrite and the consuming launch:
    # the descriptor parameters must be independent of workspace contents.
    workspace.zero_()
    inputs["out"].fill_(17.0)
    _launch(inputs, workspace)
    _check(inputs, 1.0)
    _check_workspace(workspace, 0)


def _stale_valid_descriptor():
    first, second = _inputs(1.0), _inputs(-1.0)
    first_workspace, second_workspace = _workspace(), _workspace()
    _launch(first, first_workspace)
    _launch(second, second_workspace)
    _check(first, 1.0)
    _check(second, -1.0)

    # Both tensor sets stay live. With the old pointer ABI, this copies valid
    # maps into a slot whose host key still describes the other binding: it
    # returns successfully but overwrites first.out instead of second.out.
    # With launch-owned descriptors, both workspaces remain untouched.
    second_workspace.copy_(first_workspace)
    first["out"].fill_(19.0)
    second["out"].fill_(17.0)
    _launch(second, second_workspace)
    _check(second, -1.0)
    _check(first, 19.0)
    _check_workspace(first_workspace, 0xA5)
    _check_workspace(second_workspace, 0xA5)


def _recycled_workspace_view():
    inputs, storage = _inputs(1.0), _workspace()
    workspace = storage[:]
    _launch(inputs, workspace)
    _check(inputs, 1.0)
    _check_workspace(workspace, 0xA5)
    address = workspace.data_ptr()
    del workspace

    # A caller-owned arena deterministically recycles the exact slot for a new
    # Tensor object. Its underlying CUDA allocation ID cannot change. Reusing
    # the storage this way is also how a caching allocator can evade that key.
    storage.fill_(0)
    workspace = storage[:]
    assert workspace.data_ptr() == address
    inputs["out"].fill_(17.0)
    _launch(inputs, workspace)
    _check(inputs, 1.0)
    _check_workspace(workspace, 0)


def _graph_replay():
    first, second, eager = _inputs(1.0), _inputs(-1.0), _inputs(0.5)
    workspace = _workspace()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _launch(first, workspace)
    stream.synchronize()

    # Capture both a warmed binding and a new binding. Every graph must retain
    # its own by-value maps after the host's temporary descriptor storage ends,
    # even when other graphs and eager launches bind different tensors.
    first_graph, second_graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with torch.cuda.graph(first_graph, stream=stream):
        _launch(first, workspace)
    with torch.cuda.graph(second_graph, stream=stream):
        _launch(second, workspace)
    for _ in range(3):
        _launch(eager, workspace)
        _check(eager, 0.5)
        eager["out"].fill_(23.0)
        first["out"].fill_(19.0)
        second["out"].fill_(17.0)
        workspace.zero_()
        second_graph.replay()
        _check(second, -1.0)
        _check(first, 19.0)
        _check(eager, 23.0)
        workspace.fill_(0xA5)
        first_graph.replay()
        _check(first, 1.0)
        _check(second, -1.0)
        _check(eager, 23.0)
        _check_workspace(workspace, 0xA5)


def _ordered_stream_reuse():
    first, second = _inputs(1.0), _inputs(-1.0)
    workspace = _workspace()
    producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
    producer.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(producer):
        _launch(first, workspace)
        workspace.zero_()
    consumer.wait_stream(producer)
    with torch.cuda.stream(consumer):
        _launch(second, workspace)
    torch.cuda.current_stream().wait_stream(consumer)
    _check(first, 1.0)
    _check(second, -1.0)
    _check_workspace(workspace, 0)


def _empty_workspace():
    inputs = _inputs(1.0)
    workspace = torch.empty(0, dtype=torch.uint8, device="cuda")
    _launch(inputs, workspace)
    _check(inputs, 1.0)


_SCENARIOS = {
    "zeroed": _zeroed_workspace,
    "stale_valid": _stale_valid_descriptor,
    "recycled_view": _recycled_workspace_view,
    "graph_replay": _graph_replay,
    "ordered_streams": _ordered_stream_reuse,
    "empty": _empty_workspace,
}


@pytest.mark.parametrize("scenario", _SCENARIOS)
def test_cake_sage_tma_workspace(scenario):
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), scenario],
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout
    assert f"TMA_WORKSPACE_PASS:{scenario}" in result.stdout


if __name__ == "__main__":
    selected = sys.argv[1]
    _SCENARIOS[selected]()
    torch.cuda.synchronize()
    print(f"TMA_WORKSPACE_PASS:{selected}", flush=True)
