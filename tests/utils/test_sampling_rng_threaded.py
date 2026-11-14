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

# Exercise generator reservation through real sampling kernels and graph replay.

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, local

import pytest
import torch

from flashinfer import sampling

pytestmark = pytest.mark.gpu_exclusive

SAMPLERS = (
    "sampling_from_probs",
    "sampling_from_logits",
    "top_k_sampling_from_probs",
    "top_p_sampling_from_probs",
    "min_p_sampling_from_probs",
    "top_k_top_p_sampling_from_probs",
)


def _sample(name, values, **kwargs):
    if name == "top_k_sampling_from_probs":
        kwargs["top_k"] = 16
    elif name == "top_p_sampling_from_probs":
        kwargs["top_p"] = 0.7
    elif name == "min_p_sampling_from_probs":
        kwargs["min_p"] = 0.1
    elif name == "top_k_top_p_sampling_from_probs":
        kwargs.update(top_k=16, top_p=0.7, filter_apply_order="joint")
    return getattr(sampling, name)(values, **kwargs)


@pytest.mark.parametrize("name", SAMPLERS)
@pytest.mark.parametrize("use_default_generator", [False, True])
def test_threaded_sampling_matches_reserved_rng_ranges(
    monkeypatch, name, use_default_generator
):
    device = torch.device("cuda", torch.cuda.current_device())
    values = torch.linspace(-2, 2, 4 * 128, device=device).reshape(4, 128).softmax(-1)
    _sample(name, values, seed=1234, offset=0)  # Compile before starting threads.
    streams = [torch.cuda.Stream(device=device) for _ in range(4)]
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream(device))
    start = Barrier(len(streams))
    trace = local()
    reserve = sampling.get_seed_and_offset

    def record_reservation(increment, generator=None, device=None):
        seed, offset = reserve(increment, generator, device)
        trace.reservation = (seed, offset, (increment + 3) // 4 * 4)
        return seed, offset

    monkeypatch.setattr(sampling, "get_seed_and_offset", record_reservation)
    with torch.random.fork_rng(devices=[device.index]):
        generator = (
            torch.cuda.default_generators[device.index]
            if use_default_generator
            else torch.Generator(device=device)
        )
        generator.manual_seed(1234)
        generator.set_offset(128)

        def draw(stream):
            torch.cuda.set_device(device)
            start.wait(timeout=30)
            results = []
            with torch.cuda.stream(stream):
                for _ in range(32):
                    output = _sample(
                        name,
                        values,
                        generator=None if use_default_generator else generator,
                    )
                    results.append((*trace.reservation, output))
            stream.synchronize()
            return results

        with ThreadPoolExecutor(max_workers=len(streams)) as executor:
            results = [
                item
                for group in executor.map(draw, streams, timeout=120)
                for item in group
            ]

        next_offset = 128
        for seed, offset, length, output in sorted(results, key=lambda item: item[1]):
            assert offset == next_offset, "concurrent calls reused a Philox range"
            next_offset += length
            expected = _sample(name, values, seed=seed, offset=offset)
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
        assert generator.get_offset() == next_offset


@pytest.mark.parametrize("name", SAMPLERS)
def test_sampling_explicit_rng_cuda_graph_replay(name):
    device = torch.device("cuda", torch.cuda.current_device())
    values = torch.linspace(-2, 2, 4 * 128, device=device).reshape(4, 128).softmax(-1)
    seed = torch.tensor([1234], dtype=torch.int64, device=device)
    offset = torch.tensor([0], dtype=torch.int64, device=device)
    _sample(name, values, seed=seed, offset=offset)
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            output = _sample(name, values, seed=seed, offset=offset)
        for value in (0, 128, 4096):
            offset.fill_(value)
            graph.replay()
            expected = _sample(name, values, seed=seed, offset=offset)
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
    finally:
        graph.reset()
