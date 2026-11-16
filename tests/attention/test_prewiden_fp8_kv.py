"""Tests for the opt-in FP8-KV pre-widen path of the FA2 paged prefill wrapper."""

import pytest
import torch

import flashinfer
from flashinfer.jit.fp8_kv_widen import get_fp8_kv_widen_module

needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"
)


def _make_problem(
    layout, q_len, kv_len, hq, hkv, d, page, dtype=torch.float8_e4m3fn, seed=0
):
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(seed)
    q = (torch.randn(q_len, hq, d, generator=g, device=dev) * 0.5).to(torch.bfloat16)
    n_pages = (kv_len + page - 1) // page
    pool = n_pages + 7
    perm = torch.randperm(pool, generator=g, device=dev).to(torch.int32)
    idx = perm[:n_pages].contiguous()
    if layout == "HND":
        shape = (pool, hkv, page, d)
    else:
        shape = (pool, page, hkv, d)
    k = torch.randn(shape, generator=g, device=dev).to(dtype)
    v = torch.randn(shape, generator=g, device=dev).to(dtype)
    return q, k, v, idx


def _wrapper(layout, kv_dtype, prewiden, q_len, kv_len, hq, hkv, d, page, idx):
    dev = torch.device("cuda")
    ws = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=dev)
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        ws, layout, backend="fa2", prewiden_fp8_kv=prewiden
    )
    qo_indptr = torch.tensor([0, q_len], dtype=torch.int32, device=dev)
    kv_indptr = torch.tensor([0, idx.numel()], dtype=torch.int32, device=dev)
    last = torch.tensor(
        [kv_len - (idx.numel() - 1) * page], dtype=torch.int32, device=dev
    )
    w.plan(
        qo_indptr,
        kv_indptr,
        idx,
        last,
        hq,
        hkv,
        d,
        page,
        causal=True,
        sm_scale=1.0 / d**0.5,
        q_data_type=torch.bfloat16,
        kv_data_type=kv_dtype,
        o_data_type=torch.bfloat16,
        fixed_split_size=-1,
        disable_split_kv=False,
    )
    return w


@needs_cuda
def test_gather_widen_matches_torch():
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(3)
    pool, page, hkv, d = 19, 16, 4, 256
    idx = torch.randperm(pool, generator=g, device=dev)[:11].to(torch.int32)
    k = torch.randn(pool, hkv, page, d, generator=g, device=dev).to(torch.float8_e4m3fn)
    v = torch.randn(pool, hkv, page, d, generator=g, device=dev).to(torch.float8_e4m3fn)
    ko = torch.empty(idx.numel(), hkv, page, d, dtype=torch.bfloat16, device=dev)
    vo = torch.empty_like(ko)
    get_fp8_kv_widen_module().gather_widen_fp8_kv(k, v, ko, vo, idx)
    torch.cuda.synchronize()
    torch.testing.assert_close(ko, k.index_select(0, idx.long()).to(torch.bfloat16))
    torch.testing.assert_close(vo, v.index_select(0, idx.long()).to(torch.bfloat16))


@needs_cuda
@pytest.mark.parametrize("layout", ["HND", "NHD"])
@pytest.mark.parametrize(
    "dtype", [torch.float8_e4m3fn, torch.float8_e5m2], ids=["e4m3", "e5m2"]
)
def test_prewiden_matches_bf16_kernel(layout, dtype):
    q_len, kv_len, hq, hkv, d, page = 256, 2048, 8, 2, 256, 16
    q, k, v, idx = _make_problem(layout, q_len, kv_len, hq, hkv, d, page, dtype)
    # Exact widening and the same FA2 kernel on both sides: the outputs must be
    # bitwise identical, which also checks that k/v scales are applied alike.
    k16 = k.index_select(0, idx.long()).to(torch.bfloat16)
    v16 = v.index_select(0, idx.long()).to(torch.bfloat16)
    compact = torch.arange(idx.numel(), dtype=torch.int32, device="cuda")

    w_fp8 = _wrapper(layout, dtype, False, q_len, kv_len, hq, hkv, d, page, idx)
    w_pw = _wrapper(layout, dtype, True, q_len, kv_len, hq, hkv, d, page, idx)
    w_bf16 = _wrapper(
        layout, torch.bfloat16, False, q_len, kv_len, hq, hkv, d, page, compact
    )

    out_fp8 = torch.empty(q_len, hq, d, dtype=torch.bfloat16, device="cuda")
    out_pw = torch.empty_like(out_fp8)
    out_bf16 = torch.empty_like(out_fp8)
    w_fp8.run(q, (k, v), k_scale=1.125, v_scale=0.875, out=out_fp8)
    w_pw.run(q, (k, v), k_scale=1.125, v_scale=0.875, out=out_pw)
    w_bf16.run(q, (k16, v16), k_scale=1.125, v_scale=0.875, out=out_bf16)
    torch.cuda.synchronize()

    assert torch.equal(out_pw, out_bf16)
    # The FP8 in-kernel repack differs only in summation order.
    torch.testing.assert_close(out_pw, out_fp8, rtol=2e-2, atol=2e-2)


@needs_cuda
def test_prewiden_cuda_graph_replay():
    q_len, kv_len, hq, hkv, d, page = 128, 1024, 8, 2, 256, 16
    q, k, v, idx = _make_problem(
        "HND", q_len, kv_len, hq, hkv, d, page, torch.float8_e4m3fn
    )
    w = _wrapper("HND", torch.float8_e4m3fn, True, q_len, kv_len, hq, hkv, d, page, idx)
    out = torch.empty(q_len, hq, d, dtype=torch.bfloat16, device="cuda")
    w.run(q, (k, v), k_scale=1.1, v_scale=0.9, out=out)
    torch.cuda.synchronize()
    ref = out.clone()

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        w.run(q, (k, v), k_scale=1.1, v_scale=0.9, out=out)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        w.run(q, (k, v), k_scale=1.1, v_scale=0.9, out=out)
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, ref)


@needs_cuda
def test_prewiden_rejects_unsupported_inputs():
    q_len, kv_len, hq, hkv, d, page = 64, 512, 8, 2, 256, 16
    q, k, v, idx = _make_problem(
        "HND", q_len, kv_len, hq, hkv, d, page, torch.float8_e4m3fn
    )
    with pytest.raises(ValueError, match="only supported with backend"):
        flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            torch.zeros(16 << 20, dtype=torch.uint8, device="cuda"),
            "HND",
            backend="cudnn",
            prewiden_fp8_kv=True,
        )
    with pytest.raises(ValueError, match="requires an fp8"):
        _wrapper("HND", torch.bfloat16, True, q_len, kv_len, hq, hkv, d, page, idx)

    w = _wrapper("HND", torch.float8_e4m3fn, True, q_len, kv_len, hq, hkv, d, page, idx)
    out = torch.empty(q_len, hq, d, dtype=torch.bfloat16, device="cuda")
    # A non-contiguous page cache is outside the opt-in contract: same shape,
    # but every other head-dim element skipped.
    k_pad = torch.empty(
        k.shape[:-1] + (2 * k.shape[-1],), dtype=k.dtype, device=k.device
    )
    k_pad[..., ::2] = k
    k_bad = k_pad[..., ::2]
    assert not k_bad.is_contiguous()
    with pytest.raises(ValueError, match="contiguous"):
        w.run(q, (k_bad, v), out=out)


@needs_cuda
def test_prewiden_rejects_fp8_query():
    q_len, kv_len, hq, hkv, d, page = 64, 512, 8, 2, 256, 16
    _, k, v, idx = _make_problem(
        "HND", q_len, kv_len, hq, hkv, d, page, torch.float8_e4m3fn
    )
    dev = torch.device("cuda")
    ws = torch.zeros(16 << 20, dtype=torch.uint8, device=dev)
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        ws, "HND", backend="fa2", prewiden_fp8_kv=True
    )
    with pytest.raises(ValueError, match="fp16/bf16 query"):
        w.plan(
            torch.tensor([0, q_len], dtype=torch.int32, device=dev),
            torch.tensor([0, idx.numel()], dtype=torch.int32, device=dev),
            idx,
            torch.tensor(
                [kv_len - (idx.numel() - 1) * page], dtype=torch.int32, device=dev
            ),
            hq,
            hkv,
            d,
            page,
            causal=True,
            q_data_type=torch.float8_e4m3fn,
            kv_data_type=torch.float8_e4m3fn,
            o_data_type=torch.bfloat16,
        )
