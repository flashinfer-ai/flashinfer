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

"""Scratch ownership across eager streams and independently captured graphs."""

from types import SimpleNamespace

import pytest
import torch

from flashinfer.cudnn import cudnn_recurrent_kda, linear_attention
from tests.test_helpers.cudnn_linear_attention import requires_cudnn_linear_attention


def test_workspace_is_owned_by_stream_or_capture(monkeypatch):
    stream = SimpleNamespace(cuda_stream=101)
    capturing = False
    observed = []

    def execute(*args, workspace, **kwargs):
        observed.append(workspace)

    graph = SimpleNamespace(
        execute=execute,
        get_workspace_size=lambda: 256,
        _fi_la_workspace_size=256,
        _fi_la_ordered=True,
        _fi_la_uids=(1, 2, 3, 10, 11, 100, 1000),
    )
    monkeypatch.setattr(
        linear_attention,
        "_build_la_graph",
        lambda *args, **kwargs: (graph, ()),
        raising=False,
    )
    monkeypatch.setattr(linear_attention, "_create_cudnn_handle", lambda value: value)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *args: stream)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: None)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    q, gate = torch.ones(2, 1, 4), torch.ones(2, 1)

    def run():
        linear_attention._run_la_graph(
            "gdn",
            q,
            q,
            q,
            gate,
            gate,
            torch.tensor([0, 2], dtype=torch.int32),
            torch.empty_like(q),
            scale=0.5,
            use_qk_l2norm=False,
            use_beta_sigmoid=False,
            safe_gate=False,
            gate_lower_bound=None,
            batch_invariant=False,
        )

    run()
    run()
    assert observed[0].data_ptr() == observed[1].data_ptr()
    stream = SimpleNamespace(cuda_stream=202)
    run()
    assert observed[2].data_ptr() != observed[0].data_ptr()
    stream = SimpleNamespace(cuda_stream=101)
    capturing = True
    run()
    run()
    assert len({value.data_ptr() for value in observed[2:]}) == 3
    assert all(value.data_ptr() != observed[0].data_ptr() for value in observed[2:])


def _make_call(length, heads, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def random(*shape):
        return torch.randn(
            shape, generator=generator, device="cuda", dtype=torch.float32
        )

    q, k = (
        torch.nn.functional.normalize(random(length, heads, 128), dim=-1).bfloat16()
        for _ in range(2)
    )
    v = random(length, heads, 128).bfloat16()
    g = (-0.01 - random(length, heads, 128).abs() * 0.03).bfloat16()
    beta = (random(length, heads).sigmoid() * 0.5 + 0.2).bfloat16()
    state = (random(1, heads, 128, 128) * 0.05).bfloat16()
    final, output = torch.empty_like(state), torch.empty_like(v)
    offsets = torch.tensor([0, length], device="cuda", dtype=torch.int32)

    def run():
        return cudnn_recurrent_kda(
            q,
            k,
            v,
            g,
            beta,
            initial_state=state,
            output_state=final,
            output=output,
            output_final_state=True,
            cu_seqlens=offsets,
            use_qk_l2norm_in_kernel=False,
        )

    run()
    torch.cuda.synchronize()
    expected = output.clone(), final.clone()
    return SimpleNamespace(run=run, output=output, final=final, expected=expected)


@requires_cudnn_linear_attention
@pytest.mark.parametrize("mode", ["eager", "graphs"])
def test_independent_calls_do_not_share_concurrent_scratch(mode):
    # Different inputs, shapes and state buffers make accidental scratch reuse
    # observable; comparing with isolated executions avoids kernel-tuning knobs.
    calls = [_make_call(4096, 8, 41), _make_call(6144, 4, 43)]
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    current = torch.cuda.current_stream()
    for stream in streams:
        stream.wait_stream(current)
    graphs = []
    if mode == "graphs":
        # Capturing on the same stream must still give each graph private scratch.
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(current)
        for call in calls:
            with torch.cuda.stream(capture_stream):
                call.run()
                capture_stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=capture_stream):
                    call.run()
            graphs.append(graph)
        current.wait_stream(capture_stream)
    else:
        for stream, call in zip(streams, calls, strict=True):
            with torch.cuda.stream(stream):
                call.run()
    torch.cuda.synchronize()

    for _ in range(12):
        ready = torch.cuda.Event()
        # Delay both streams behind one event so host submission finishes before
        # either workload starts, then run repeated kernels to expose overlap.
        torch.cuda._sleep(20_000_000)
        ready.record()
        for index, (stream, call) in enumerate(zip(streams, calls, strict=True)):
            stream.wait_event(ready)
            with torch.cuda.stream(stream):
                for _ in range(4):
                    if mode == "graphs":
                        graphs[index].replay()
                    else:
                        call.run()
        torch.cuda.synchronize()
        for call in calls:
            torch.testing.assert_close(
                call.output, call.expected[0], rtol=2e-3, atol=2e-3
            )
            torch.testing.assert_close(
                call.final, call.expected[1], rtol=2e-3, atol=2e-3
            )
