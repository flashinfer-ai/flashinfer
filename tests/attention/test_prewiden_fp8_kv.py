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


def _plan_args(q_len, kv_len, n_pages, hq, hkv, d, page):
    dev = torch.device("cuda")
    return dict(
        qo_indptr=torch.tensor([0, q_len], dtype=torch.int32, device=dev),
        paged_kv_indptr=torch.tensor([0, n_pages], dtype=torch.int32, device=dev),
        paged_kv_last_page_len=torch.tensor(
            [kv_len - (n_pages - 1) * page], dtype=torch.int32, device=dev
        ),
        num_qo_heads=hq,
        num_kv_heads=hkv,
        head_dim_qk=d,
        page_size=page,
        causal=True,
        sm_scale=1.0 / d**0.5,
        q_data_type=torch.bfloat16,
        o_data_type=torch.bfloat16,
        fixed_split_size=-1,
        disable_split_kv=False,
    )


def _wrapper(layout, kv_dtype, prewiden, q_len, kv_len, hq, hkv, d, page, idx):
    dev = torch.device("cuda")
    ws = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=dev)
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        ws, layout, backend="fa2", prewiden_fp8_kv=prewiden
    )
    n_pages = idx.numel()
    w.plan(
        paged_kv_indices=idx,
        kv_data_type=kv_dtype,
        **_plan_args(q_len, kv_len, n_pages, hq, hkv, d, page),
    )
    return w


def _graph_wrapper(n_pages, hq, hkv, d, page, custom_mask_buf=None):
    dev = torch.device("cuda")
    return flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=dev),
        "HND",
        use_cuda_graph=True,
        qo_indptr_buf=torch.zeros(2, dtype=torch.int32, device=dev),
        paged_kv_indptr_buf=torch.zeros(2, dtype=torch.int32, device=dev),
        paged_kv_indices_buf=torch.zeros(n_pages, dtype=torch.int32, device=dev),
        paged_kv_last_page_len_buf=torch.zeros(1, dtype=torch.int32, device=dev),
        custom_mask_buf=custom_mask_buf,
        mask_indptr_buf=(
            torch.zeros(2, dtype=torch.int32, device=dev)
            if custom_mask_buf is not None
            else None
        ),
        backend="fa2",
        prewiden_fp8_kv=True,
    )


def _ref_run(q, k, v, idx, q_len, kv_len, hq, hkv, d, page):
    """BF16 kernel on the torch-widened compact cache: the design-contract oracle."""
    k16 = k.index_select(0, idx.long()).to(torch.bfloat16)
    v16 = v.index_select(0, idx.long()).to(torch.bfloat16)
    compact = torch.arange(idx.numel(), dtype=torch.int32, device="cuda")
    w = _wrapper("HND", torch.bfloat16, False, q_len, kv_len, hq, hkv, d, page, compact)
    out = torch.empty(q_len, hq, d, dtype=torch.bfloat16, device="cuda")
    w.run(q, (k16, v16), k_scale=1.125, v_scale=0.875, out=out)
    torch.cuda.synchronize()
    return out


def _run(w, q, k, v, out):
    w.run(q, (k, v), k_scale=1.125, v_scale=0.875, out=out)
    torch.cuda.synchronize()


def _capture(w, q, k, v, out):
    _run(w, q, k, v, out)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        w.run(q, (k, v), k_scale=1.125, v_scale=0.875, out=out)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        w.run(q, (k, v), k_scale=1.125, v_scale=0.875, out=out)
    return graph


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
    # Finite FP8 values widen exactly, so the gate is bitwise.
    assert torch.equal(ko, k.index_select(0, idx.long()).to(torch.bfloat16))
    assert torch.equal(vo, v.index_select(0, idx.long()).to(torch.bfloat16))


@needs_cuda
@pytest.mark.parametrize("layout", ["HND", "NHD"])
@pytest.mark.parametrize(
    "dtype", [torch.float8_e4m3fn, torch.float8_e5m2], ids=["e4m3", "e5m2"]
)
@pytest.mark.parametrize("d", [128, 256])
def test_prewiden_matches_bf16_kernel(layout, dtype, d):
    q_len, kv_len, hq, hkv, page = 256, 2048, 8, 2, 16
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
    _run(w, q, k, v, out)
    ref = out.clone()
    graph = _capture(w, q, k, v, out)
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, ref)


@needs_cuda
def test_prewiden_graph_buf_keeps_physical_ids_and_replay_updates():
    """The caller's indices buffer is both plan argument and graph buffer."""
    q_len, kv_len, hq, hkv, d, page = 128, 1024, 8, 2, 256, 16
    q, k, v, idx = _make_problem("HND", q_len, kv_len, hq, hkv, d, page, seed=0)
    _, _, _, idx2 = _make_problem("HND", q_len, kv_len, hq, hkv, d, page, seed=1)
    n_pages = idx.numel()
    ref = _ref_run(q, k, v, idx, q_len, kv_len, hq, hkv, d, page)
    ref2 = _ref_run(q, k, v, idx2, q_len, kv_len, hq, hkv, d, page)

    w = _graph_wrapper(n_pages, hq, hkv, d, page)
    buf = w._paged_kv_indices_buf
    buf.copy_(idx)
    w.plan(
        paged_kv_indices=buf,
        kv_data_type=torch.float8_e4m3fn,
        **_plan_args(q_len, kv_len, n_pages, hq, hkv, d, page),
    )
    # plan must not overwrite the physical mapping with the compact table
    assert torch.equal(buf, idx)
    out = torch.empty(q_len, hq, d, dtype=torch.bfloat16, device="cuda")
    _run(w, q, k, v, out)
    assert torch.equal(out, ref)

    graph = _capture(w, q, k, v, out)
    buf.copy_(idx2)  # in-place page-id update between replays
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, ref2)


@needs_cuda
def test_prewiden_graph_replan_reuses_capture():
    q_len, kv_len, hq, hkv, d, page = 128, 1024, 8, 2, 256, 16
    q, k, v, idx = _make_problem("HND", q_len, kv_len, hq, hkv, d, page, seed=0)
    _, _, _, idx2 = _make_problem("HND", q_len, kv_len, hq, hkv, d, page, seed=1)
    n_pages = idx.numel()
    ref2 = _ref_run(q, k, v, idx2, q_len, kv_len, hq, hkv, d, page)

    w = _graph_wrapper(n_pages, hq, hkv, d, page)
    args = _plan_args(q_len, kv_len, n_pages, hq, hkv, d, page)
    w.plan(paged_kv_indices=idx, kv_data_type=torch.float8_e4m3fn, **args)
    out = torch.empty(q_len, hq, d, dtype=torch.bfloat16, device="cuda")
    graph = _capture(w, q, k, v, out)
    k_ptr = w._pw_k_shadow.data_ptr()
    compact_ptr = w._paged_kv_indices_buf.data_ptr()

    # compatible replan: captured storage must be reused, replay sees new ids
    w.plan(paged_kv_indices=idx2, kv_data_type=torch.float8_e4m3fn, **args)
    assert w._pw_k_shadow.data_ptr() == k_ptr
    assert w._paged_kv_indices_buf.data_ptr() == compact_ptr
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, ref2)

    # incompatible replan (different referenced page count) must raise
    short_args = _plan_args(q_len, 512, 32, hq, hkv, d, page)
    with pytest.raises(ValueError, match="replan"):
        w.plan(
            paged_kv_indices=idx[:32].contiguous(),
            kv_data_type=torch.float8_e4m3fn,
            **short_args,
        )


@needs_cuda
def test_prewiden_eager_inplace_index_update():
    q_len, kv_len, hq, hkv, d, page = 128, 1024, 8, 2, 256, 16
    q, k, v, idx = _make_problem("HND", q_len, kv_len, hq, hkv, d, page, seed=0)
    _, _, _, idx2 = _make_problem("HND", q_len, kv_len, hq, hkv, d, page, seed=1)
    ref2 = _ref_run(q, k, v, idx2, q_len, kv_len, hq, hkv, d, page)
    w = _wrapper("HND", torch.float8_e4m3fn, True, q_len, kv_len, hq, hkv, d, page, idx)
    idx.copy_(idx2)  # int32 CUDA argument is borrowed, not snapshotted
    out = torch.empty(q_len, hq, d, dtype=torch.bfloat16, device="cuda")
    _run(w, q, k, v, out)
    assert torch.equal(out, ref2)


@needs_cuda
def test_prewiden_capacity_slack_indices():
    """An indices tensor longer than indptr[-1] is sliced to the referenced pages."""
    q_len, kv_len, hq, hkv, d, page = 128, 1024, 8, 2, 256, 16
    q, k, v, idx = _make_problem("HND", q_len, kv_len, hq, hkv, d, page)
    ref = _ref_run(q, k, v, idx, q_len, kv_len, hq, hkv, d, page)
    idx_long = torch.cat([idx, torch.zeros(3, dtype=torch.int32, device="cuda")])
    ws = torch.zeros(128 << 20, dtype=torch.uint8, device="cuda")
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        ws, "HND", backend="fa2", prewiden_fp8_kv=True
    )
    # indptr references only the first n_pages entries of the longer tensor
    w.plan(
        paged_kv_indices=idx_long,
        kv_data_type=torch.float8_e4m3fn,
        **_plan_args(q_len, kv_len, idx.numel(), hq, hkv, d, page),
    )
    out = torch.empty(q_len, hq, d, dtype=torch.bfloat16, device="cuda")
    _run(w, q, k, v, out)
    assert torch.equal(out, ref)


@needs_cuda
def test_prewiden_misaligned_source_rejected():
    q_len, kv_len, hq, hkv, d, page = 128, 1024, 8, 2, 256, 16
    q, k, v, idx = _make_problem("HND", q_len, kv_len, hq, hkv, d, page)
    pool = k.shape[0]
    pe = hkv * page * d
    kb = torch.empty(pool * pe + 1, dtype=k.dtype, device="cuda")
    kb[1:] = k.reshape(-1)
    k_bad = kb[1:].view(pool, hkv, page, d)
    assert k_bad.is_contiguous() and k_bad.data_ptr() % 16 != 0
    w = _wrapper("HND", torch.float8_e4m3fn, True, q_len, kv_len, hq, hkv, d, page, idx)
    out = torch.empty(q_len, hq, d, dtype=torch.bfloat16, device="cuda")
    with pytest.raises((RuntimeError, ValueError), match="aligned"):
        w.run(q, (k_bad, v), out=out)


@needs_cuda
def test_prewiden_rejects_retained_custom_mask_buf():
    q_len, kv_len, hq, hkv, d, page = 128, 1024, 8, 2, 256, 16
    q, k, v, idx = _make_problem("HND", q_len, kv_len, hq, hkv, d, page)
    mask_buf = torch.zeros(q_len * kv_len, dtype=torch.uint8, device="cuda")
    w = _graph_wrapper(idx.numel(), hq, hkv, d, page, custom_mask_buf=mask_buf)
    with pytest.raises(ValueError, match="custom"):
        w.plan(
            paged_kv_indices=idx,
            kv_data_type=torch.float8_e4m3fn,
            **_plan_args(q_len, kv_len, idx.numel(), hq, hkv, d, page),
        )


@needs_cuda
def test_prewiden_multi_request_partial_last_page_duplicate_pages():
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(5)
    hq, hkv, d, page = 8, 2, 256, 16
    q0, q1, kv0, kv1 = 64, 64, 100, 256
    n0, n1 = (kv0 + page - 1) // page, kv1 // page
    pool = n0 + n1 + 5
    perm = torch.randperm(pool, generator=g, device=dev).to(torch.int32)
    idx0 = perm[:n0]
    idx1 = torch.cat([idx0[:3], perm[n0 : n0 + n1 - 3]])  # shared prefix
    idx = torch.cat([idx0, idx1]).contiguous()
    k = torch.randn(pool, hkv, page, d, generator=g, device=dev).to(torch.float8_e4m3fn)
    v = torch.randn(pool, hkv, page, d, generator=g, device=dev).to(torch.float8_e4m3fn)
    q = (torch.randn(q0 + q1, hq, d, generator=g, device=dev) * 0.5).to(torch.bfloat16)
    k16 = k.index_select(0, idx.long()).to(torch.bfloat16)
    v16 = v.index_select(0, idx.long()).to(torch.bfloat16)
    compact = torch.arange(idx.numel(), dtype=torch.int32, device=dev)

    def mk(prewiden, kv_dt, table):
        w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            torch.zeros(128 << 20, dtype=torch.uint8, device=dev),
            "HND",
            backend="fa2",
            prewiden_fp8_kv=prewiden,
        )
        w.plan(
            torch.tensor([0, q0, q0 + q1], dtype=torch.int32, device=dev),
            torch.tensor([0, n0, n0 + n1], dtype=torch.int32, device=dev),
            table,
            torch.tensor([kv0 - (n0 - 1) * page, page], dtype=torch.int32, device=dev),
            hq,
            hkv,
            d,
            page,
            causal=True,
            sm_scale=1.0 / d**0.5,
            q_data_type=torch.bfloat16,
            kv_data_type=kv_dt,
            o_data_type=torch.bfloat16,
            fixed_split_size=-1,
            disable_split_kv=False,
        )
        return w

    w_pw = mk(True, torch.float8_e4m3fn, idx)
    w_bf = mk(False, torch.bfloat16, compact)
    o1 = torch.empty(q0 + q1, hq, d, dtype=torch.bfloat16, device=dev)
    o2 = torch.empty_like(o1)
    w_pw.run(q, (k, v), k_scale=1.125, v_scale=0.875, out=o1)
    w_bf.run(q, (k16, v16), k_scale=1.125, v_scale=0.875, out=o2)
    torch.cuda.synchronize()
    assert torch.equal(o1, o2)


@needs_cuda
def test_prewiden_fp16_e5m2():
    q_len, kv_len, hq, hkv, d, page = 128, 1024, 8, 2, 256, 16
    q, k, v, idx = _make_problem(
        "HND", q_len, kv_len, hq, hkv, d, page, torch.float8_e5m2
    )
    q = q.to(torch.float16)
    k16 = k.index_select(0, idx.long()).to(torch.float16)
    v16 = v.index_select(0, idx.long()).to(torch.float16)
    compact = torch.arange(idx.numel(), dtype=torch.int32, device="cuda")

    def mk(prewiden, kv_dt, table):
        w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            torch.zeros(128 << 20, dtype=torch.uint8, device="cuda"),
            "HND",
            backend="fa2",
            prewiden_fp8_kv=prewiden,
        )
        args = _plan_args(q_len, kv_len, idx.numel(), hq, hkv, d, page)
        args["q_data_type"] = torch.float16
        args["o_data_type"] = torch.float16
        w.plan(paged_kv_indices=table, kv_data_type=kv_dt, **args)
        return w

    w_pw = mk(True, torch.float8_e5m2, idx)
    w_bf = mk(False, torch.float16, compact)
    o1 = torch.empty(q_len, hq, d, dtype=torch.float16, device="cuda")
    o2 = torch.empty_like(o1)
    w_pw.run(q, (k, v), k_scale=1.125, v_scale=0.875, out=o1)
    w_bf.run(q, (k16, v16), k_scale=1.125, v_scale=0.875, out=o2)
    torch.cuda.synchronize()
    assert torch.equal(o1, o2)


@needs_cuda
@pytest.mark.parametrize(
    "src_dtype", [torch.float8_e4m3fn, torch.float8_e5m2], ids=["e4m3", "e5m2"]
)
def test_gather_widen_special_value_convention(src_dtype):
    """All 256 codes: finite values are bitwise exact; NaN/Inf follow the
    in-kernel repack's vec_cast convention (BF16 output maps them to large
    finite values on every architecture)."""
    dev = torch.device("cuda")
    pool, page, hkv, d = 4, 16, 2, 256
    pe = hkv * page * d
    codes = torch.arange(256, dtype=torch.uint8, device=dev).repeat(
        (pool * pe + 255) // 256
    )[: pool * pe]
    k = codes.view(src_dtype).view(pool, hkv, page, d)
    v = k.clone()
    idx = torch.arange(pool, dtype=torch.int32, device=dev)
    ko = torch.empty(pool, hkv, page, d, dtype=torch.bfloat16, device=dev)
    vo = torch.empty_like(ko)
    get_fp8_kv_widen_module().gather_widen_fp8_kv(k, v, ko, vo, idx)
    torch.cuda.synchronize()
    ref = k.to(torch.bfloat16)
    special = ~torch.isfinite(ref.reshape(-1))
    a = ko.view(torch.int16).reshape(-1).to(torch.int32)
    b = ref.view(torch.int16).reshape(-1).to(torch.int32)
    assert torch.equal(a[~special], b[~special])
    got = ko.reshape(-1)[special]
    assert torch.isfinite(got).all()
    if src_dtype == torch.float8_e4m3fn:
        # the two e4m3 NaN codes widen to +/-480
        assert torch.equal(got.abs(), torch.full_like(got, 480.0))
    else:
        # e5m2 Inf codes widen to +/-65536; NaN codes to finite values above
        # the largest finite e5m2 magnitude (57344)
        want = ref.reshape(-1)[special]
        inf_mask = torch.isinf(want)
        assert torch.equal(got[inf_mask].abs(), torch.full_like(got[inf_mask], 65536.0))
        assert (got[~inf_mask].abs() > 57344.0).all()

    # End to end, pre-widen must equal the BF16 kernel fed with the kernel's
    # own widened values (identical plans), special values included.
    q_len, kv_len, hq = 64, pool * page, 8
    g = torch.Generator(device=dev).manual_seed(7)
    q = (torch.randn(q_len, hq, d, generator=g, device=dev) * 0.5).to(torch.bfloat16)
    w_pw = _wrapper("HND", src_dtype, True, q_len, kv_len, hq, hkv, d, page, idx)
    w_bf = _wrapper("HND", torch.bfloat16, False, q_len, kv_len, hq, hkv, d, page, idx)
    o1 = torch.empty(q_len, hq, d, dtype=torch.bfloat16, device=dev)
    o2 = torch.empty_like(o1)
    w_pw.run(q, (k, v), out=o1)
    w_bf.run(q, (ko, vo), out=o2)
    torch.cuda.synchronize()
    assert torch.equal(o1, o2)


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
def test_prewiden_rejects_more_combinations():
    dev = torch.device("cuda")
    q_len, kv_len, hq, hkv, d, page = 64, 512, 8, 2, 256, 16
    q, k, v, idx = _make_problem(
        "HND", q_len, kv_len, hq, hkv, d, page, torch.float8_e4m3fn
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            torch.zeros(16 << 20, dtype=torch.uint8, device=dev),
            "HND",
            backend="fa2",
            batch_invariant=True,
            prewiden_fp8_kv=True,
        )
    ws = torch.zeros(16 << 20, dtype=torch.uint8, device=dev)
    args = _plan_args(q_len, kv_len, idx.numel(), hq, hkv, d, page)
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        ws, "HND", backend="fa2", prewiden_fp8_kv=True
    )
    with pytest.raises(ValueError, match="output dtype"):
        w.plan(
            paged_kv_indices=idx,
            kv_data_type=torch.float8_e4m3fn,
            **{**args, "o_data_type": torch.float16},
        )
    with pytest.raises(ValueError, match="plain paged"):
        w.plan(
            paged_kv_indices=idx,
            kv_data_type=torch.float8_e4m3fn,
            logits_soft_cap=10.0,
            **args,
        )
    with pytest.raises(ValueError, match="multiples"):
        w.plan(
            paged_kv_indices=idx,
            kv_data_type=torch.float8_e4m3fn,
            **{**args, "head_dim_qk": 250},
        )


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
