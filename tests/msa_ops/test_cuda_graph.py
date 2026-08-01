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

CUDA-graph capturability of the MSA decode pipeline: with the proxy's
``max_seqlen_q`` / ``max_k_tiles`` passed, no ``.cpu()``/``.item()`` sync may
run during capture (vLLM captures the indexer inside the decode graph).
"""

import math

import pytest
import torch

from flashinfer.utils import is_sm12x_supported

BLK = 128


def _skip():
    if not is_sm12x_supported(torch.device("cuda")):
        pytest.skip("MSA decode pipeline requires SM120/SM121")


def _cu(lens, dev):
    c = torch.zeros(len(lens) + 1, dtype=torch.int32, device=dev)
    c[1:] = torch.tensor(lens, device=dev).cumsum(0)
    return c


def _build_decode_pipeline(B, ctx):
    """Callable running proxy_fp4 -> topk -> sparse_decode at the M3 decode
    config, with the proxy's plan-time dims passed so the call is capturable."""
    from flashinfer.msa_ops import (
        msa_proxy_score_fp4,
        msa_sparse_decode_attention,
        msa_topk_select,
    )
    from flashinfer.msa_ops.proxy_score import _quantize_qk_to_nvfp4

    dev = "cuda"
    topk, nb = 16, ctx // BLK
    Hq, Hkv = 64, 4
    scale = 1.0 / math.sqrt(128)
    torch.manual_seed(0)
    cuq, cuk = _cu([1] * B, dev), _cu([ctx] * B, dev)
    qi = torch.randn(B, 4, 128, dtype=torch.bfloat16, device=dev) / 3
    ki = torch.randn(B * ctx, 1, 128, dtype=torch.bfloat16, device=dev) / 3
    qf, qs, iq = _quantize_qk_to_nvfp4(qi)
    kf, ks, ik = _quantize_qk_to_nvfp4(ki)
    qa = torch.randn(B, Hq, 128, dtype=torch.bfloat16, device=dev) / 3
    ka = torch.randn(B * ctx, Hkv, 128, dtype=torch.bfloat16, device=dev) / 3
    va = torch.randn(B * ctx, Hkv, 128, dtype=torch.bfloat16, device=dev) / 3

    def run():
        # max_seqlen_q / max_k_tiles known at plan time (decode -> 1, ctx/128).
        sc = msa_proxy_score_fp4(
            qf,
            kf,
            qs,
            ks,
            iq,
            ik,
            cuq,
            cuk,
            causal=True,
            reduce_heads=True,
            max_seqlen_q=1,
            max_k_tiles=nb,
        )  # (1, nb, B)
        sel = msa_topk_select(sc, topk)  # (B, 1, topk)
        q2k = sel.permute(1, 0, 2).expand(Hkv, B, topk).contiguous()  # (Hkv, B, topk)
        return msa_sparse_decode_attention(
            qa,
            ka,
            va,
            q2k,
            cu_seqlens_k=cuk,
            seqlen_q=1,
            causal=True,
            softmax_scale=scale,
        )

    return run


def test_msa_decode_pipeline_cuda_graph():
    """Graph replay of the captured pipeline must be bit-identical to eager."""
    _skip()
    run = _build_decode_pipeline(B=64, ctx=4096)

    for _ in range(3):  # JIT compile / warm caches before capture
        eager = run()
    torch.cuda.synchronize()

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(side)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out_g = run()
    g.replay()
    torch.cuda.synchronize()

    assert torch.equal(out_g, eager)


def test_msa_prefill_cuda_graph():
    """The prefill is capturable as a single call: work metadata is built
    on device, so no host copy/sync runs during capture."""
    _skip()
    from flashinfer.msa_ops import msa_sparse_attention

    dev = "cuda"
    B, S, Hq, Hkv, topk, hd = 1, 4096, 64, 4, 16, 128
    scale = 1.0 / math.sqrt(hd)
    cu = _cu([S] * B, dev)
    torch.manual_seed(0)
    q = torch.randn(B * S, Hq, hd, dtype=torch.bfloat16, device=dev) / 3
    k = torch.randn(B * S, Hkv, hd, dtype=torch.bfloat16, device=dev) / 3
    v = torch.randn(B * S, Hkv, hd, dtype=torch.bfloat16, device=dev) / 3
    q2k = torch.full((Hkv, B * S, topk), -1, dtype=torch.int32, device=dev)
    for t in range(S):
        c = min(topk, t // BLK + 1)
        q2k[:, t, :c] = torch.arange(c, dtype=torch.int32, device=dev)

    run = lambda: msa_sparse_attention(  # noqa: E731
        q, k, v, q2k, cu, cu, causal=True, softmax_scale=scale
    )
    for _ in range(3):  # JIT compile / warm caches before capture
        eager = run()
    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(side)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out_g = run()
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(out_g, eager)


def test_proxy_capture_requires_plan_dims():
    """Without the plan-time dims the proxy must refuse to sync during capture
    (clear error), instead of emitting the cryptic CPU<->CUDA copy failure."""
    _skip()
    from flashinfer.msa_ops import msa_proxy_score_fp4
    from flashinfer.msa_ops.proxy_score import _quantize_qk_to_nvfp4

    dev = "cuda"
    B, ctx = 8, 2048
    cuq, cuk = _cu([1] * B, dev), _cu([ctx] * B, dev)
    qi = torch.randn(B, 4, 128, dtype=torch.bfloat16, device=dev) / 3
    ki = torch.randn(B * ctx, 1, 128, dtype=torch.bfloat16, device=dev) / 3
    qf, qs, iq = _quantize_qk_to_nvfp4(qi)
    kf, ks, ik = _quantize_qk_to_nvfp4(ki)

    # Warm up (compile) outside capture.
    msa_proxy_score_fp4(
        qf, kf, qs, ks, iq, ik, cuq, cuk, causal=True, reduce_heads=True
    )
    torch.cuda.synchronize()

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph()
    with (
        torch.cuda.stream(side),
        pytest.raises(ValueError, match="max_seqlen_q and max_k_tiles"),
        torch.cuda.graph(g),
    ):
        # No max_seqlen_q / max_k_tiles -> would sync -> must raise.
        msa_proxy_score_fp4(
            qf,
            kf,
            qs,
            ks,
            iq,
            ik,
            cuq,
            cuk,
            causal=True,
            reduce_heads=True,
        )
