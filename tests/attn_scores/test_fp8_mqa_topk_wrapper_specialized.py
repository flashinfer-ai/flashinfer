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
"""FP8MQATopKWrapper backend='specialized': selection correctness vs an
fp32 reference, multi-request support (which cutedsl refuses), determinism,
and the validated contract's rejection semantics."""

import pytest
import torch

from flashinfer.attn_scores import fp8_paged_mqa_topk_indexer as _functional  # noqa: F401
from flashinfer.attn_scores.mqa_topk import FP8MQATopKWrapper
from flashinfer.attn_scores.prefill_indexer import (
    VALIDATED_CUDA_VERSIONS,  # noqa: F401 - re-exported for debugging
    _toolkit_validation_reasons,
)

requires_sm103 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="specialized backend is SM103-only",
)
requires_validated_toolkit = pytest.mark.skipif(
    bool(_toolkit_validation_reasons()),
    reason="specialized backend refuses unvalidated CUDA toolkits "
    "(same gate as dispatch, torch build + JIT nvcc)",
)

H, D, TOPK, RATIO = 4, 128, 512, 4


def _mk(kv_lens, q_lens, seed):
    """Continuous fp8 K pool, packed prefixes (every non-final kv prefix a
    multiple of 256 tokens), storage padded to a multiple of 64 keys."""
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(seed)
    kv_ind, starts = [0], []
    for kv in kv_lens:
        starts.append(kv_ind[-1])
        kv_ind.append(kv_ind[-1] + kv)
    assert all(v % 256 == 0 for v in kv_ind[:-1]), "test case must be aligned"
    live_keys = kv_ind[-1] // RATIO
    n_keys = -(-live_keys // 64) * 64  # pad storage to a page multiple
    k = (torch.randn(n_keys, D, generator=g, device=dev) * 0.5).to(torch.float8_e4m3fn)
    t = sum(q_lens)
    q = (torch.randn(t, H, D, generator=g, device=dev) * 0.5).to(torch.float8_e4m3fn)
    qo = [0]
    for ql in q_lens:
        qo.append(qo[-1] + ql)
    return dict(
        k=k,
        q=q,
        qo=torch.tensor(qo, dtype=torch.int32, device=dev),
        kv=torch.tensor(kv_ind, dtype=torch.int32, device=dev),
        starts=starts,
        kv_lens=kv_lens,
        q_lens=q_lens,
        k_scale=torch.ones(n_keys, dtype=torch.float32, device=dev),
        weights=torch.full((t, H), 0.0883883, device=dev),
    )


def _reference_rows(inp):
    """fp32 sum_h relu(q_h . k), top-512 complete visible keys, ascending."""
    kf = inp["k"].to(torch.float32)
    rows = []
    t0 = 0
    for r, (q_len, kv_len) in enumerate(
        zip(inp["q_lens"], inp["kv_lens"], strict=True)
    ):
        key0 = inp["starts"][r] // RATIO
        for i in range(q_len):
            pos = kv_len - q_len + i
            vis = (pos + 1) // RATIO
            row = torch.full((TOPK,), -1, dtype=torch.int32, device="cuda")
            if vis > 0:
                qf = inp["q"][t0 + i].to(torch.float32)
                sc = (qf @ kf[key0 : key0 + vis].T).clamp_min(0).sum(0)
                n = min(vis, TOPK)
                # lower-index tie-breaking: stable sort on (-score, index)
                order = torch.argsort(sc, descending=True, stable=True)[:n]
                row[:n] = order.sort().values.to(torch.int32)
            rows.append(row)
        t0 += q_len
    return torch.stack(rows)


def _wrapper(inp, max_bs):
    w = FP8MQATopKWrapper(
        max_context_length=1 << 15,
        max_chunk_size=1 << 14,
        max_batch_size=max_bs,
        device="cuda",
        backend="specialized",
        num_heads=H,
        compression_ratio=RATIO,
        num_keys=inp["k"].shape[0],
    )
    w.prepare_pool(inp["k"], k_scale=inp["k_scale"])
    w.plan(inp["qo"], inp["kv"])
    return w


@requires_sm103
@requires_validated_toolkit
@pytest.mark.parametrize(
    "kv_lens,q_lens",
    [
        ([1024], [1024]),  # direct regime, B=1 (full prefill)
        ([2930], [2930]),  # scored regime (visible > 512), B=1
        ([512, 2816, 300], [512, 2816, 300]),  # multi-request, aligned
    ],
)
def test_selection_matches_reference(kv_lens, q_lens):
    inp = _mk(kv_lens, q_lens, seed=23)
    w = _wrapper(inp, max_bs=len(kv_lens))
    out = w.run(inp["q"], inp["weights"], inp["k"], k_scale=inp["k_scale"])
    gold = _reference_rows(inp)
    assert out.dtype == torch.int32 and out.shape == gold.shape
    assert torch.equal(out, gold)
    # deterministic across replays; out= honored
    buf = torch.empty_like(gold)
    out2 = w.run(inp["q"], inp["weights"], inp["k"], k_scale=inp["k_scale"], out=buf)
    assert out2 is buf and torch.equal(out2, gold)


@requires_sm103
@requires_validated_toolkit
def test_multi_request_vs_cutedsl_refusal():
    """The specialized backend serves the multi-request batch the cutedsl
    backend's constructor refuses."""
    inp = _mk([512, 2816], [512, 2816], seed=5)
    w = _wrapper(inp, max_bs=2)
    out = w.run(inp["q"], inp["weights"], inp["k"], k_scale=inp["k_scale"])
    assert torch.equal(out, _reference_rows(inp))


@requires_sm103
@requires_validated_toolkit
def test_rejection_semantics():
    inp = _mk([1024], [1024], seed=9)
    # constructor rejects unservable configurations loudly
    with pytest.raises(ValueError, match="num_heads=4"):
        FP8MQATopKWrapper(
            max_context_length=4096,
            max_chunk_size=64,
            max_batch_size=1,
            device="cuda",
            backend="specialized",
            num_heads=32,
            compression_ratio=4,
        )
    w = _wrapper(inp, max_bs=1)
    # non-unit k_scale rejected
    bad_scale = inp["k_scale"].clone()
    bad_scale[0] = 2.0
    with pytest.raises(ValueError, match="unit k_scale"):
        w.run(inp["q"], inp["weights"], inp["k"], k_scale=bad_scale)
    # non-uniform weights rejected (selection would not be weight-invariant)
    bad_w = inp["weights"].clone()
    bad_w[0, 0] *= 2
    with pytest.raises(ValueError, match="per-row-constant"):
        w.run(inp["q"], bad_w, inp["k"], k_scale=inp["k_scale"])
    # misaligned multi-request segment rejected at plan
    w2 = FP8MQATopKWrapper(
        max_context_length=4096,
        max_chunk_size=4096,
        max_batch_size=2,
        device="cuda",
        backend="specialized",
        num_heads=H,
        compression_ratio=RATIO,
        num_keys=inp["k"].shape[0],
    )
    w2.prepare_pool(inp["k"], k_scale=inp["k_scale"])
    qo = torch.tensor([0, 100, 1024], dtype=torch.int32, device="cuda")
    kv = torch.tensor([0, 100, 1024], dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="multiple of 256"):
        w2.plan(qo, kv)
    # chunked prefill (q_len < kv_len) is outside the validated contract
    qo3 = torch.tensor([0, 8], dtype=torch.int32, device="cuda")
    kv3 = torch.tensor([0, 1024], dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="full-prefill"):
        w2.plan(qo3, kv3)


@requires_sm103
def test_unvalidated_toolkit_refusal():
    if not _toolkit_validation_reasons():
        pytest.skip("validated toolkit: constructor serves")
    with pytest.raises(ValueError, match="validated CUDA toolkit"):
        FP8MQATopKWrapper(
            max_context_length=4096,
            max_chunk_size=64,
            max_batch_size=1,
            device="cuda",
            backend="specialized",
            num_heads=H,
            compression_ratio=4,
        )


@requires_sm103
@requires_validated_toolkit
def test_auto_routes_to_specialized_for_multi_request_full_prefill():
    """backend='auto' is a router: it selects the specialized backend for
    the multi-request full-prefill class it serves (default-visible, no
    opt-in), and reports the resolution."""
    inp = _mk([512, 2816], [512, 2816], seed=41)
    w = FP8MQATopKWrapper(
        max_context_length=1 << 15,
        max_chunk_size=1 << 14,
        max_batch_size=2,
        device="cuda",
        backend="auto",
        num_heads=H,
        compression_ratio=RATIO,
        num_keys=inp["k"].shape[0],
    )
    w.prepare_pool(inp["k"], k_scale=inp["k_scale"])
    w.plan(inp["qo"], inp["kv"])
    assert w.resolved_backend == "specialized"
    out = w.run(inp["q"], inp["weights"], inp["k"], k_scale=inp["k_scale"])
    assert torch.equal(out, _reference_rows(inp))


@requires_sm103
def test_auto_exhausted_configuration_raises():
    """auto with a configuration neither backend can serve raises, naming
    both reasons (specialized: wrong head count; cutedsl: multi-request)."""
    if not _toolkit_validation_reasons():
        match = "single-request|auto: no backend"
    else:
        match = "auto: no backend|single-request"
    with pytest.raises(ValueError, match=match):
        FP8MQATopKWrapper(
            max_context_length=4096,
            max_chunk_size=64,
            max_batch_size=6,
            device="cuda",
            backend="auto",
            num_heads=32,
            compression_ratio=4,
        )
