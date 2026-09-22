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

import math

import numpy as np
import pytest
import torch

from flashinfer.cake_sampling import (
    cake_sampling_route,
    choose_stage1,
    choose_stage23,
    top_k_top_p_sampling_from_probs,
)
from flashinfer.jit.cake_sampling import arch_dir_for_capability


def _require_blackwell():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    arch = arch_dir_for_capability(torch.cuda.get_device_capability())
    if arch is None:
        pytest.skip("frozen radix sampling kernels need SM100 or SM103")
    return arch


def _probs(batch, vocab, seed=536, scale=1.0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.softmax(
        torch.randn(batch, vocab, device="cuda", generator=g) * scale, dim=-1
    )


def _exact_support(row: np.ndarray, k: int, p: float):
    vocab = row.shape[0]
    order = np.lexsort((np.arange(vocab), -row.astype(np.float64)))[: min(k, vocab)]
    v = row[order].astype(np.float32)
    incl = np.cumsum(v, dtype=np.float32)
    total = np.float32(v.sum(dtype=np.float32))
    kept = (incl - v) < np.float32(p) * total
    last = int(np.nonzero(kept)[0].max())
    return order[: last + 1], v[: last + 1] / incl[last]


@pytest.mark.parametrize(
    ("batch", "vocab", "k", "p"),
    [
        (1, 32768, 50, 0.9),
        (8, 128256, 50, 0.9),
        (2, 151936, 10, 0.5),
        (4, 262144, 1000, 0.95),
        (3, 32003, 64, 0.9),
        (1, 128256, 1, 0.9),
        (2, 128256, 50, 1e-6),
        (16, 128256, 50, 0.9),
    ],
)
def test_support_matches_top_k_first_semantics(batch, vocab, k, p):
    _require_blackwell()
    probs = _probs(batch, vocab)
    renorm = torch.empty(batch, 1024, device="cuda")
    # The slab order is allocation order inside stage 1 and differs between runs, so read the
    # indices from the same call that produced ``renorm`` (a second stage-1 run would permute).
    ws = (
        torch.empty(batch, 1024, device="cuda", dtype=torch.float32),
        torch.empty(batch, 1024, device="cuda", dtype=torch.int32),
        torch.empty(batch, device="cuda", dtype=torch.int32),
    )
    out = top_k_top_p_sampling_from_probs(
        probs,
        k,
        p,
        philox_seed=0xC0FFEE,
        philox_offset=3,
        renorm_out=renorm,
        workspace=ws,
    )
    vals, idxs, cnt = ws
    torch.cuda.synchronize()
    assert cake_sampling_route(probs, k) == "pipeline"
    pn = probs.cpu().numpy()
    for r in range(batch):
        kept_idx, kept_p = _exact_support(pn[r], k, p)
        n = int(cnt[r])
        assert n == k
        slab_idx = idxs[r, :n].cpu().numpy()
        assert set(slab_idx.tolist()) == set(
            np.lexsort((np.arange(vocab), -pn[r].astype(np.float64)))[:k].tolist()
        )
        dev_kept = slab_idx[renorm[r, :n].cpu().numpy() > 0]
        assert set(dev_kept.tolist()) == set(kept_idx.tolist())
        assert int(out[r]) in kept_idx.tolist()
        got = dict(
            zip(slab_idx.tolist(), renorm[r, :n].cpu().numpy().tolist(), strict=False)
        )
        for i, pi in zip(kept_idx.tolist(), kept_p.tolist(), strict=False):
            assert math.isclose(got[i], pi, rel_tol=1e-4, abs_tol=1e-6)


def test_deterministic_replay_and_offset_sensitivity():
    _require_blackwell()
    probs = _probs(16, 128256)
    a = top_k_top_p_sampling_from_probs(
        probs, 50, 0.9, philox_seed=11, philox_offset=5
    ).clone()
    for _ in range(5):
        b = top_k_top_p_sampling_from_probs(
            probs, 50, 0.9, philox_seed=11, philox_offset=5
        )
        torch.cuda.synchronize()
        assert torch.equal(a, b)
    c = top_k_top_p_sampling_from_probs(probs, 50, 0.9, philox_seed=11, philox_offset=6)
    torch.cuda.synchronize()
    assert not torch.equal(a, c)


def test_generator_advances_like_flashinfer_sampling():
    _require_blackwell()
    from flashinfer.sampling import get_seed_and_offset

    probs = _probs(6, 32768)
    g1 = torch.Generator(device="cuda").manual_seed(4242)
    g2 = torch.Generator(device="cuda").manual_seed(4242)
    top_k_top_p_sampling_from_probs(probs, 50, 0.9, generator=g1)
    get_seed_and_offset(6, g2, torch.device("cuda"))
    assert torch.equal(g1.get_state(), g2.get_state())


def test_statistical_total_variation():
    _require_blackwell()
    batch, vocab, k, p = 4, 32768, 50, 0.9
    probs = _probs(batch, vocab, scale=3.0)
    pn = probs.cpu().numpy()
    draws = 20000
    counts = [dict() for _ in range(batch)]
    for t in range(draws):
        out = top_k_top_p_sampling_from_probs(
            probs, k, p, philox_seed=2024, philox_offset=t
        )
        for r, tok in enumerate(out.cpu().tolist()):
            counts[r][tok] = counts[r].get(tok, 0) + 1
    for r in range(batch):
        kept_idx, kept_p = _exact_support(pn[r], k, p)
        ref = dict(zip(kept_idx.tolist(), kept_p.tolist(), strict=False))
        assert set(counts[r]) <= set(ref)
        tv = 0.5 * sum(abs(counts[r].get(i, 0) / draws - ref[i]) for i in ref)
        assert tv < 3.0 * math.sqrt(len(ref) / (2 * math.pi * draws))


def test_cuda_graph_capture_and_replay():
    _require_blackwell()
    probs = _probs(8, 128256)
    out = torch.empty(8, device="cuda", dtype=torch.int32)
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(2):
            top_k_top_p_sampling_from_probs(
                probs, 50, 0.9, philox_seed=7, philox_offset=1, out=out
            )
    torch.cuda.synchronize()
    ref = out.clone()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        top_k_top_p_sampling_from_probs(
            probs, 50, 0.9, philox_seed=7, philox_offset=1, out=out
        )
    out.zero_()
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, ref)


def test_per_request_tensors_and_routes():
    arch = _require_blackwell()
    batch, vocab = 6, 128256
    probs = _probs(batch, vocab)
    ks = torch.tensor([1, 10, 50, 100, 500, 1000], device="cuda", dtype=torch.int32)
    ps = torch.tensor(
        [0.1, 0.5, 0.9, 0.95, 1.0, 0.99], device="cuda", dtype=torch.float32
    )
    out = top_k_top_p_sampling_from_probs(
        probs, ks, ps, top_k_max=1000, philox_seed=1, philox_offset=0
    )
    torch.cuda.synchronize()
    pn = probs.cpu().numpy()
    for r in range(batch):
        kept_idx, _ = _exact_support(pn[r], int(ks[r]), float(ps[r]))
        assert int(out[r]) in kept_idx.tolist()
    assert cake_sampling_route(probs, 50) == "pipeline"
    assert cake_sampling_route(probs, None) == "fallback:no_top_k"
    assert cake_sampling_route(probs, vocab) == "fallback:top_k_disabled"
    assert cake_sampling_route(probs, 1025) == "fallback:top_k_gt_slab"
    assert cake_sampling_route(probs.half(), 50) == "fallback:dtype"
    assert (
        cake_sampling_route(torch.empty(256, 262144, device="cuda"), 50)
        == "fallback:large_batch"
    )
    assert choose_stage1(arch, 1, 128256) == (8, 32)
    assert choose_stage1(arch, 64, 128256) == (4, 64)
    assert choose_stage23(arch, 50) == (32, 2)
    # Fallback path still returns int32 samples of the right shape.
    res = top_k_top_p_sampling_from_probs(probs, vocab, 0.9)
    assert res.dtype == torch.int32 and res.shape == (batch,)
