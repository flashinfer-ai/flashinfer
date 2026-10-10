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

"""ATen tracing retains sampling launches and per-request RNG state."""

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from flashinfer import sampling


@pytest.mark.parametrize("indexed", [False, True])
@pytest.mark.parametrize("rng", ["tensor", "scalar", "signed_scalar"])
@torch.no_grad()
def test_sampling_pipeline_compiles_and_replays(indexed, rng):
    logits = torch.randn(4, 257, device="cuda", dtype=torch.bfloat16)
    temperature = torch.ones(4, device="cuda")
    top_k = torch.full((4,), 80, dtype=torch.int32, device="cuda")
    top_p = torch.full((4,), 0.8, device="cuda")
    indices = (
        torch.tensor([2, 0, 2, 1], device="cuda", dtype=torch.int32)
        if indexed
        else None
    )
    seed = {"scalar": 17, "signed_scalar": -97, "tensor": (1 << 64) - 97}[rng]
    offset = (1 << 32) + 3
    if rng == "tensor":
        seed = torch.full((4,), seed, device="cuda", dtype=torch.uint64)
        offset = torch.full((4,), offset, device="cuda", dtype=torch.uint64)

    def forward(logits, temperature, top_k, top_p):
        masked = sampling.top_k_mask_logits(logits, top_k)
        probs = sampling.softmax(masked, temperature=temperature, enable_pdl=False)
        probs = sampling.top_p_renorm_probs(probs, top_p, is_deterministic=True)
        tokens, valid = sampling.sampling_from_probs(
            probs,
            indices=indices,
            deterministic=True,
            seed=seed,
            offset=offset,
            return_valid=True,
        )
        return probs, tokens, valid

    forward(logits, temperature, top_k, top_p)
    traced = make_fx(forward)(logits, temperature, top_k, top_p)
    targets = {node.target for node in traced.graph.nodes if node.op == "call_function"}
    for name in (
        "softmax_out",
        "top_k_mask_logits_out",
        "top_p_renorm_probs_out",
        "sampling_from_probs_out",
    ):
        assert getattr(torch.ops.flashinfer, name).default in targets
    compiled = torch.compile(traced, fullgraph=True, dynamic=False)
    compiled(logits, temperature, top_k, top_p)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = compiled(logits, temperature, top_k, top_p)
    for step in range(3):
        logits.normal_()
        temperature.fill_(0.7 + step * 0.1)
        top_k.fill_(40 + 10 * step)
        top_p.fill_(0.5 + 0.1 * step)
        if rng == "tensor":
            seed.fill_((1 << 64) - 97 - step)
            offset.fill_((1 << 32) + step)
        expected = forward(logits, temperature, top_k, top_p)
        allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
        for result, reference in zip(actual, expected, strict=True):
            torch.testing.assert_close(result, reference, rtol=0, atol=0)
        assert actual[2].all()
