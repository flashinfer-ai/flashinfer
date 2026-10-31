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

import functools

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from flashinfer import sampling


@pytest.mark.parametrize("indexed", [False, True])
@pytest.mark.parametrize("rng", ["tensor", "scalar", "signed_scalar"])
@pytest.mark.parametrize("min_p", [None, 0.1])
@torch.no_grad()
def test_sampling_pipeline_compiles_and_replays(indexed, rng, min_p):
    logits = torch.randn(4, 257, device="cuda", dtype=torch.bfloat16)
    temperature = torch.ones(4, device="cuda")
    top_k = torch.full((4,), 80, dtype=torch.int32, device="cuda")
    top_p = torch.full((4,), 0.8, device="cuda")
    min_p_tensor = torch.full((4,), min_p or 0.0, device="cuda")
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
        sample = (
            sampling.sampling_from_probs
            if min_p is None
            else functools.partial(
                sampling.min_p_sampling_from_probs, min_p=min_p_tensor
            )
        )
        tokens, valid = sample(
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
        "sampling_from_probs_out" if min_p is None else "min_p_sampling_from_probs_out",
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
        min_p_tensor.fill_(step * 0.2)
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


@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("drafts", [0, 1, 7])
@pytest.mark.parametrize("device_rng", [False, True])
@torch.no_grad()
def test_chain_sampling_compiles_and_replays(batch, drafts, device_rng):
    draft = torch.full((batch, drafts, 257), 1 / 257, device="cuda")
    ids = torch.zeros((batch, drafts), dtype=torch.int64, device="cuda")
    target = torch.full((batch, drafts + 1, 257), 1 / 257, device="cuda")
    accepted = torch.zeros(batch, dtype=torch.int32, device="cuda")
    emitted = torch.zeros_like(accepted)
    seed, offset = -97, (1 << 32) + 3
    if device_rng:
        seed = torch.full((batch,), (1 << 64) - 97, dtype=torch.uint64, device="cuda")
        offset = torch.full((batch,), offset, dtype=torch.uint64, device="cuda")

    def forward(draft, ids, target, accepted, emitted):
        before = accepted.clone()
        result = sampling.chain_speculative_sampling(
            draft,
            ids,
            target,
            accepted,
            emitted,
            deterministic=True,
            seed=seed,
            offset=offset,
        )
        return (*result, before)

    inputs = draft, ids, target, accepted, emitted
    forward(*inputs)
    traced = make_fx(forward)(*inputs)
    targets = {node.target for node in traced.graph.nodes if node.op == "call_function"}
    assert torch.ops.flashinfer.chain_speculative_sampling_out.default in targets
    compiled = torch.compile(traced, fullgraph=True, dynamic=False)
    compiled(*inputs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = compiled(*inputs)
    for step, prefix in enumerate((drafts, 0, drafts // 2)):
        draft.copy_(torch.softmax(torch.randn_like(draft), -1))
        target.copy_(torch.softmax(torch.randn_like(target), -1))
        target[:, :prefix].copy_(draft[:, :prefix])
        target[:, prefix:drafts, 0] = 0
        target[:, prefix:drafts] /= target[:, prefix:drafts].sum(-1, keepdim=True)
        if device_rng:
            seed.fill_((1 << 64) - 97 - step)
            offset.fill_((1 << 32) + step)
        accepted.fill_(11 + step)
        emitted.fill_(19 + step)
        expected = tuple(value.clone() for value in forward(*inputs))
        accepted.fill_(11 + step)
        emitted.fill_(19 + step)
        allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
        for result, reference in zip(actual, expected, strict=True):
            torch.testing.assert_close(result, reference, rtol=0, atol=0)
        torch.testing.assert_close(accepted, expected[1], rtol=0, atol=0)
        torch.testing.assert_close(emitted, expected[2], rtol=0, atol=0)
        assert (emitted == 19 + step + prefix).all()
        assert (actual[3] == 11 + step).all()
