"""CUDA-graph capturability of the MSA decode pipeline (proxy -> topk -> sparse
decode). Production serving (vLLM) runs the indexer inside a captured decode
graph, so the pipeline must be sync-free: the proxy's ``max_seqlen_q`` /
``max_k_tiles`` are the only host-derived dims and must be passed (or come from a
pre-allocated score buffer) so no ``.cpu()``/``.item()`` runs during capture."""

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
    """Return a callable running proxy_fp4 -> topk -> sparse_decode at the M3
    decode config (index Hq4/Hkv1 reduced to 1; attention Hq64/Hkv4), with the
    proxy's plan-time dims passed so the call is capturable."""
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
    """Capture proxy_fp4 -> topk -> sparse_decode and confirm graph replay is
    bit-identical to eager (the pipeline is sync-free with plan-time dims)."""
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

    # graph replay reuses the captured kernels on identical inputs -> exact match
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

    # warm up (compile) outside capture
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
        # no max_seqlen_q / max_k_tiles -> would sync -> must raise
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
