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

Tests for the SM103 "specialized" fp8 token-interleaved GQA-context backend
of trtllm_batch_context_with_kv_cache: correctness vs an fp32 reference,
backend-dispatch semantics (auto / explicit / kill switch / allowlist
boundary), and CUDA-graph replay == eager.
"""

import functools
import gc
import os
import traceback
import weakref

import pytest
import torch

import flashinfer.gqa_ctx_fp8_ikv_cuda as spec_mod
from flashinfer.prefill import (
    batch_context_with_kv_cache,
    trtllm_batch_context_with_kv_cache,
)

requires_sm103 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="the specialized fp8-iKV GQA context kernel is SM103-only",
)

SCALE = 0.08838834764831845  # 128 ** -0.5
PAGE, HQ, D = 128, 16, 128


def _make_inputs(q_lens, kv_lens, seed=0, btw_pad=0, shuffle_pages=False):
    """btw_pad widens the block table past the pages any request needs, with
    arbitrary (valid-pool) ids in the unused columns, and shuffle_pages
    scatters the real page ids -- the serving call form, where tables are as
    wide as the engine's maximum pages per sequence."""
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(seed)
    tq = sum(q_lens)
    pprs = [(k + PAGE - 1) // PAGE for k in kv_lens]
    np_, btw = sum(pprs), max(pprs) + btw_pad
    q = (torch.randn(tq, HQ, D, generator=g, device=dev) * 0.5).to(torch.float8_e4m3fn)
    kv_buf = (torch.randn(np_, PAGE, 2, D, generator=g, device=dev) * 0.5).to(
        torch.float8_e4m3fn
    )
    if btw_pad:
        # garbage ids (any pool slot) where serving engines leave stale columns
        bt = torch.randint(
            0, np_, (len(q_lens), btw), generator=g, dtype=torch.int32, device=dev
        )
    else:
        bt = torch.zeros(len(q_lens), btw, dtype=torch.int32, device=dev)
    if shuffle_pages:
        page_ids = torch.randperm(np_, generator=g, device=dev).to(torch.int32)
    else:
        page_ids = torch.arange(np_, dtype=torch.int32, device=dev)
    s = 0
    for i, ppr in enumerate(pprs):
        bt[i, :ppr] = page_ids[s : s + ppr]
        s += ppr
    cu_q = torch.tensor(
        [0] + list(torch.tensor(q_lens).cumsum(0)), dtype=torch.int32, device=dev
    )
    cu_kv = torch.tensor(
        [0] + list(torch.tensor(kv_lens).cumsum(0)), dtype=torch.int32, device=dev
    )
    seq_lens = torch.tensor(kv_lens, dtype=torch.int32, device=dev)
    return dict(
        q=q,
        kv_buf=kv_buf,
        bt=bt,
        cu_q=cu_q,
        cu_kv=cu_kv,
        seq_lens=seq_lens,
        max_q=max(q_lens),
        max_kv=max(kv_lens),
    )


def _reference(inp):
    """fp32 chunked-prefill causal GQA over the token-interleaved cache."""
    dev = inp["q"].device
    cu = inp["cu_q"].to(torch.int64).tolist()
    outs = []
    for r in range(inp["seq_lens"].shape[0]):
        q_len = cu[r + 1] - cu[r]
        kv_len = int(inp["seq_lens"][r])
        n_pages = (kv_len + PAGE - 1) // PAGE
        flat = inp["kv_buf"][inp["bt"][r, :n_pages].to(torch.int64)]
        k = flat[:, :, 0, :].reshape(-1, D)[:kv_len].to(torch.float32)
        v = flat[:, :, 1, :].reshape(-1, D)[:kv_len].to(torch.float32)
        qf = inp["q"][cu[r] : cu[r + 1]].to(torch.float32)
        sc = torch.einsum("qhd,ld->qhl", qf, k) * SCALE
        pos = torch.arange(q_len, device=dev)
        vis = kv_len - q_len + pos + 1
        mask = torch.arange(kv_len, device=dev).view(1, 1, -1) >= vis.view(-1, 1, 1)
        pr = torch.softmax(sc.masked_fill(mask, float("-inf")), dim=-1)
        outs.append(torch.einsum("qhl,ld->qhd", pr, v))
    return torch.cat(outs, 0).to(torch.bfloat16)


def _kwargs(inp):
    kv = inp["kv_buf"]
    return dict(
        query=inp["q"],
        kv_cache=(kv[:, :, 0, :].unsqueeze(1), kv[:, :, 1, :].unsqueeze(1)),
        workspace_buffer=_ws(),
        block_tables=inp["bt"],
        seq_lens=inp["seq_lens"],
        max_q_len=inp["max_q"],
        max_kv_len=inp["max_kv"],
        bmm1_scale=SCALE,
        bmm2_scale=1.0,
        batch_size=int(inp["seq_lens"].shape[0]),
        cum_seq_lens_q=inp["cu_q"],
        cum_seq_lens_kv=inp["cu_kv"],
        window_left=-1,
    )


def _api(inp, out, backend="auto"):
    return batch_context_with_kv_cache(out=out, backend=backend, **_kwargs(inp))


_WS = {}


def _ws():
    if "b" not in _WS:
        _WS["b"] = torch.zeros(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    return _WS["b"]


def _out_like(inp):
    return torch.empty(inp["q"].shape[0], HQ, D, dtype=torch.bfloat16, device="cuda")


@requires_sm103
@pytest.mark.parametrize(
    "q_lens,kv_lens",
    [
        ([2] * 8, [2] * 8),  # tiny-KV batch (census class B8 q2 kv2)
        ([2] * 64, [2] * 64),  # tiny-KV batch, large B
        ([128], [128]),  # single-page single request
        ([128, 128], [128, 128]),  # uniform narrow-table attn_coop_b7
        ([2, 1, 1, 1], [100, 1, 1, 1]),  # ragged, deep-context causal row
        ([1], [3]),  # tiny-cap route
        ([2], [4]),
        ([128] * 10, [128] * 10),  # uniform full-page batch routes
    ],
)
def test_specialized_correct_and_routed(q_lens, kv_lens):
    inp = _make_inputs(q_lens, kv_lens, seed=7)
    gold = _reference(inp).float()
    before = dict(spec_mod.DISPATCH_STATS)
    out = _out_like(inp)
    _api(inp, out)
    assert spec_mod.DISPATCH_STATS["specialized"] == before["specialized"] + 1, (
        spec_mod.LAST_FALLBACK_REASONS
    )
    diff = (out.float() - gold).abs()
    tol = 0.005 + 0.05 * gold.abs()
    assert float((diff <= tol).float().mean()) == 1.0
    # explicit specialized produces the same bits as auto
    out2 = _out_like(inp)
    _api(inp, out2, backend="specialized")
    assert torch.equal(out, out2)


@requires_sm103
@pytest.mark.parametrize(
    "q_lens,kv_lens",
    [
        ([2] * 8, [2] * 8),  # tiny q2/kv2, B<=16: attn_tiny_async<BTW1=false>
        ([2] * 64, [2] * 64),  # tiny q2/kv2, large B, BTW1=false
        ([3] * 4, [3] * 4),  # q3 route
        ([4] * 4, [4] * 4),  # q4 route
        ([6] * 4, [6] * 4),  # q6 route
        ([8] * 4, [8] * 4),  # q8 route
        ([2, 1, 1, 1], [100, 1, 1, 1]),  # ragged deep-context causal row
        ([1], [3]),  # tiny-cap route
        ([128], [128]),  # B=1 q128 (attn_split; table width immaterial but covered)
        ([128] * 7, [128] * 7),  # B>1 uniform q128, wide table: attn_fast U128
        ([128] * 10, [128] * 10),  # B>1 uniform q128, wide table: generic U128
        ([128] * 15, [128] * 15),  # census-row B15 q128 in the serving form
    ],
)
def test_specialized_wide_block_table(q_lens, kv_lens):
    """Serving engines pass block tables as wide as the engine's maximum pages
    per sequence, with arbitrary ids in the unused columns and non-contiguous
    page placement.  That call form selects the BTW1=false tiny variants and
    the wide-table U128 kernels (attn_fast / the generic U128 launch) that a
    minimum-width contiguous table never reaches."""
    inp = _make_inputs(q_lens, kv_lens, seed=29, btw_pad=7, shuffle_pages=True)
    assert inp["bt"].shape[1] >= 8
    gold = _reference(inp).float()
    before = dict(spec_mod.DISPATCH_STATS)
    out = _out_like(inp)
    _api(inp, out)
    assert spec_mod.DISPATCH_STATS["specialized"] == before["specialized"] + 1, (
        spec_mod.LAST_FALLBACK_REASONS
    )
    diff = (out.float() - gold).abs()
    tol = 0.005 + 0.05 * gold.abs()
    assert float((diff <= tol).float().mean()) == 1.0
    # the serving form must be bitwise repeatable too
    out2 = _out_like(inp)
    _api(inp, out2)
    assert torch.equal(out, out2)


def test_failed_load_does_not_retain_dispatch_arguments(monkeypatch):
    attempts = 0

    def fail():
        nonlocal attempts
        attempts += 1
        raise FileNotFoundError("missing artifact")

    monkeypatch.setattr(spec_mod, "_MODULE_LOAD_FAILURE", None)
    monkeypatch.setattr(spec_mod, "get_gqa_ctx_fp8_ikv_module", fail)
    monkeypatch.setattr(spec_mod, "supported_with_reasons", lambda *args: [])

    def retry():
        q = torch.empty(1)
        with pytest.raises(
            RuntimeError, match="FileNotFoundError: missing artifact"
        ) as err:
            spec_mod.maybe_dispatch_specialized(
                "specialized",
                q,
                None,
                None,
                None,
                1,
                1,
                SCALE,
                1.0,
                1,
                None,
                None,
                None,
                {},
            )
        return weakref.ref(q), len(traceback.extract_tb(err.value.__traceback__))

    results = [retry() for _ in range(8)]
    gc.collect()
    assert attempts == 1
    assert all(ref() is None for ref, _ in results)
    assert len({depth for _, depth in results}) == 1


@requires_sm103
def test_missing_specialized_artifact_uses_cached_stock(monkeypatch):
    inp = _make_inputs([2, 2], [2, 2])
    expected = trtllm_batch_context_with_kv_cache(
        backend="trtllm-gen", out=_out_like(inp), **_kwargs(inp)
    )
    attempts = 0

    def missing():
        nonlocal attempts
        attempts += 1
        raise FileNotFoundError("no specialized artifact")

    monkeypatch.setenv("FLASHINFER_DISABLE_JIT", "1")
    monkeypatch.setattr(spec_mod, "_MODULE_LOAD_FAILURE", None)
    monkeypatch.setattr(spec_mod, "get_gqa_ctx_fp8_ikv_module", missing)
    before = dict(spec_mod.DISPATCH_STATS)
    for _ in range(2):
        assert torch.equal(_api(inp, _out_like(inp)), expected)
        assert spec_mod.LAST_FALLBACK_REASONS == [
            "specialized_module_unavailable:FileNotFoundError"
        ]
    with pytest.raises(RuntimeError, match="no specialized artifact"):
        _api(inp, _out_like(inp), backend="specialized")
    assert attempts == 1
    assert {
        "specialized": before["specialized"],
        "fallback": before["fallback"] + 2,
    } == spec_mod.DISPATCH_STATS


def _offset_copy(t):
    backing = torch.empty(t.numel() + 1, dtype=t.dtype, device=t.device)
    view = backing[1:].view_as(t)
    view.copy_(t)
    assert view.data_ptr() % 16 != 0
    return view


def _direct(inp, out):
    k = inp["kv_buf"][:, :, 0, :].unsqueeze(1)
    spec_mod.get_gqa_ctx_fp8_ikv_module().gqa_ctx_fp8_ikv_run(
        inp["q"],
        k,
        inp["bt"],
        inp["cu_q"],
        inp["cu_kv"],
        inp["seq_lens"],
        inp["max_q"],
        inp["max_kv"],
        out,
    )


@requires_sm103
@pytest.mark.parametrize("offset", [None, "q", "kv_buf", "out"])
def test_direct_ffi_alignment(offset):
    inp = _make_inputs([2, 2], [2, 2])
    out = _out_like(inp)
    if offset is None:
        _direct(inp, out)
        torch.testing.assert_close(
            out.float(), _reference(inp).float(), atol=0.005, rtol=0.05
        )
        return
    if offset == "out":
        out = _offset_copy(out)
    else:
        inp[offset] = _offset_copy(inp[offset])
    with pytest.raises(ValueError, match="16-byte aligned"):
        _direct(inp, out)


@requires_sm103
@pytest.mark.parametrize("pad", [0, 7])
def test_direct_table_width(pad):
    inp = _make_inputs([2, 2], [2, 2], btw_pad=pad, shuffle_pages=True)
    out = _out_like(inp).fill_(float("nan"))
    _direct(inp, out)
    torch.testing.assert_close(
        out.float(), _reference(inp).float(), atol=0.005, rtol=0.05
    )


@requires_sm103
@pytest.mark.parametrize(
    "case,message",
    [
        ("short_out", "out must"),
        ("strided_out", "out must"),
        ("fp16_out", "out must"),
        ("strided_q", "q must"),
        ("table_rank", "tensor rank"),
        ("seq_rank", "tensor rank"),
        ("table_rows", "shapes"),
        ("table_stride", "contiguous int32"),
        ("cu_q_short", "shapes"),
        ("cu_kv_short", "shapes"),
        ("cu_q_stride", "contiguous int32"),
        ("seq_dtype", "contiguous int32"),
        ("page_stride", "token-interleaved"),
        ("cpu_offsets", "same CUDA device"),
    ],
)
def test_direct_rejects_invalid_views(case, message):
    inp = _make_inputs([2, 2], [2, 2], btw_pad=1)
    backing = _out_like(inp).fill_(-7)
    out = backing
    if case == "short_out":
        out = backing[:-1]  # aligned; full backing still allocated
    elif case == "strided_out":
        out = backing.transpose(0, 1).contiguous().transpose(0, 1)
    elif case == "fp16_out":
        out = backing.view(torch.float16)
    elif case == "strided_q":
        inp["q"] = inp["q"].transpose(0, 1).contiguous().transpose(0, 1)
    elif case == "table_rank":
        inp["bt"] = inp["bt"].flatten()
    elif case == "seq_rank":
        inp["seq_lens"] = inp["seq_lens"].unsqueeze(1)
    elif case == "table_rows":
        inp["bt"] = inp["bt"][:1]
    elif case == "table_stride":
        inp["bt"] = inp["bt"][:, :1]
    elif case == "cu_q_short":
        inp["cu_q"] = inp["cu_q"][:-1]
    elif case == "cu_kv_short":
        inp["cu_kv"] = inp["cu_kv"][:-1]
    elif case == "cu_q_stride":
        inp["cu_q"] = inp["cu_q"].repeat_interleave(2)[::2]
    elif case == "seq_dtype":
        inp["seq_lens"] = inp["seq_lens"].long()
    elif case == "page_stride":
        inp["kv_buf"] = inp["kv_buf"].repeat_interleave(2, dim=0)[::2]
    else:
        inp["cu_kv"] = inp["cu_kv"].cpu()
    with pytest.raises(ValueError, match=message):
        _direct(inp, out)
    assert (backing == -7).all().item()


@requires_sm103
@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="requires a second CUDA device"
)
def test_direct_rejects_other_cuda_device():
    inp = _make_inputs([2], [2])
    inp["cu_kv"] = inp["cu_kv"].to(
        f"cuda:{(inp['q'].device.index + 1) % torch.cuda.device_count()}"
    )
    with pytest.raises(ValueError, match="same CUDA device"):
        _direct(inp, _out_like(inp))


@requires_sm103
@pytest.mark.parametrize("empty", ["batch", "rows", "bound"])
def test_direct_no_work(empty):
    inp = _make_inputs([1], [1])
    capacity = _out_like(inp).fill_(-7)
    out = capacity
    inp["cu_q"].zero_()
    if empty == "batch":
        for key in ("bt", "seq_lens"):
            inp[key] = inp[key][:0]
        for key in ("cu_q", "cu_kv"):
            inp[key] = inp[key][:1]
    elif empty == "rows":
        inp["q"], out = inp["q"][:0], capacity[:0]
    else:
        inp["max_q"] = 0
    _direct(inp, out)
    # A checked torch launch immediately after Run also surfaces a pending
    # synchronous launch-configuration error (not just an async device fault).
    assert (capacity == -7).all().item()
    torch.cuda.synchronize()


@requires_sm103
@pytest.mark.parametrize(
    "case,reason",
    [
        ("q", "query_not_16B_aligned"),
        ("kv_buf", "kv_not_16B_aligned"),
        ("out", "out_not_16B_aligned"),
        ("out_shape", "out_not_bf16_contiguous"),
        ("out_stride", "out_not_bf16_contiguous"),
        ("v_shape", "kv_layout_not_token_interleaved"),
        ("v_stride", "kv_layout_not_token_interleaved"),
        ("page_stride", "kv_layout_not_token_interleaved"),
        ("table_stride", "block_tables_geometry"),
        ("table_rows", "block_tables_geometry"),
        ("batch_size", "seq_lens_shape"),
        ("seq_stride", "seq_lens_not_contiguous"),
        ("cu_q_short", "cum_seq_lens_q_shape"),
        ("cu_kv_short", "cum_seq_lens_kv_shape"),
        ("cu_stride", "cum_seq_lens_q_not_int32_cuda_contiguous"),
    ],
)
def test_invalid_layout_routing(monkeypatch, case, reason):
    import flashinfer.prefill as prefill

    inp = _make_inputs([2, 2], [2, 2], btw_pad=1)
    out = _out_like(inp)
    if case in ("q", "kv_buf"):
        inp[case] = _offset_copy(inp[case])
    elif case == "out":
        out = _offset_copy(out)
    elif case == "out_shape":
        out = out[:-1]
    elif case == "out_stride":
        out = torch.empty_like(out.transpose(0, 1).contiguous()).transpose(0, 1)
    elif case == "page_stride":
        inp["kv_buf"] = inp["kv_buf"].repeat_interleave(2, dim=0)[::2]
    kwargs = _kwargs(inp)
    k, v = kwargs["kv_cache"]
    if case == "v_shape":
        kwargs["kv_cache"] = k, v[..., :-1]
    elif case == "v_stride":
        kwargs["kv_cache"] = k, v[..., :1].expand_as(v)
    elif case == "table_stride":
        kwargs["block_tables"] = inp["bt"][:, :1]
    elif case == "table_rows":
        kwargs["block_tables"] = inp["bt"][:1]
    elif case == "batch_size":
        kwargs["batch_size"] += 1
    elif case == "seq_stride":
        kwargs["seq_lens"] = inp["seq_lens"].repeat_interleave(2)[::2]
    elif case == "cu_q_short":
        kwargs["cum_seq_lens_q"] = inp["cu_q"][:-1]
    elif case == "cu_kv_short":
        kwargs["cum_seq_lens_kv"] = inp["cu_kv"][:-1]
    elif case == "cu_stride":
        kwargs["cum_seq_lens_q"] = inp["cu_q"].repeat_interleave(2)[::2]

    calls = []

    @functools.wraps(trtllm_batch_context_with_kv_cache)
    def stock(**kw):
        calls.append(kw["backend"])
        return kw["out"]

    monkeypatch.setattr(prefill, "trtllm_batch_context_with_kv_cache", stock)
    monkeypatch.setattr(
        spec_mod, "_load_module_or_failure", lambda: pytest.fail("eligible")
    )
    assert batch_context_with_kv_cache(out=out, **kwargs) is out
    assert reason in spec_mod.LAST_FALLBACK_REASONS
    with pytest.raises(ValueError, match=reason):
        batch_context_with_kv_cache(out=out, backend="specialized", **kwargs)
    assert calls == ["trtllm-gen"]


@requires_sm103
@pytest.mark.parametrize(
    "option,value,error,reason",
    [
        ("window_left", -1.0, TypeError, "dense_window"),
        ("o_sf_vec_size", 16, AssertionError, "no_o_sf_vec_size"),
    ],
)
def test_stock_invalid_options(monkeypatch, option, value, error, reason):
    inp = _make_inputs([2], [2])
    kwargs = _kwargs(inp)
    kwargs[option] = value
    monkeypatch.setattr(
        spec_mod, "_load_module_or_failure", lambda: pytest.fail("eligible")
    )
    with pytest.raises(error) as stock:
        trtllm_batch_context_with_kv_cache(
            out=_out_like(inp), backend="trtllm-gen", **kwargs
        )
    with pytest.raises(error) as auto:
        batch_context_with_kv_cache(out=_out_like(inp), **kwargs)
    assert str(auto.value) == str(stock.value)
    assert reason in spec_mod.LAST_FALLBACK_REASONS
    with pytest.raises(ValueError, match=reason):
        batch_context_with_kv_cache(out=_out_like(inp), backend="specialized", **kwargs)


@requires_sm103
def test_u2_inactive_v_padding():
    inp = _make_inputs([2, 2], [2, 2])
    # Keep the U2 physical capacity and launch bounds, shorten request 0.
    inp["cu_q"].sub_(inp["cu_q"].new_tensor([0, 1, 1]))
    inp["cu_kv"].copy_(inp["cu_q"])
    inp["seq_lens"][0] = 1
    inactive = inp["kv_buf"][0, 1:, 1, :].view(torch.uint8)
    gold = _reference(inp).float()
    outputs = []
    for byte in (0, 0x7F):  # E4M3 zero versus NaN
        inactive.fill_(byte)
        out = _out_like(inp).fill_(-7)
        _api(inp, out, backend="specialized")
        torch.testing.assert_close(out[:3].float(), gold, atol=0.005, rtol=0.05)
        assert (out[3] == -7).all().item()
        outputs.append(out)
    assert torch.equal(*outputs)


def _make_padded_q128_inputs(B, q_len, seed=0):
    """Padded-capacity packing: physical q/out rows say uniform q128
    (q.shape[0] == B*128, max_q_len == 128) while cu_q packs the actual
    shorter requests tightly.  This is the shape that must defeat the
    uniform-q128 speculation per request, not just per batch."""
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(seed)
    q = (torch.randn(B * PAGE, HQ, D, generator=g, device=dev) * 0.5).to(
        torch.float8_e4m3fn
    )
    kv_buf = (torch.randn(B, PAGE, 2, D, generator=g, device=dev) * 0.5).to(
        torch.float8_e4m3fn
    )
    bt = torch.arange(B, dtype=torch.int32, device=dev).view(B, 1)
    cu_q = q_len * torch.arange(B + 1, dtype=torch.int32, device=dev)
    cu_kv = PAGE * torch.arange(B + 1, dtype=torch.int32, device=dev)
    seq_lens = torch.full((B,), PAGE, dtype=torch.int32, device=dev)
    return dict(
        q=q,
        kv_buf=kv_buf,
        bt=bt,
        cu_q=cu_q,
        cu_kv=cu_kv,
        seq_lens=seq_lens,
        max_q=PAGE,
        max_kv=PAGE,
    )


@requires_sm103
@pytest.mark.parametrize(
    "B,q_len", [(2, 127), (7, 125), (10, 125), (10, 127), (15, 125)]
)
def test_u128_padded_capacity_short_requests(B, q_len):
    # Regression: with padded physical capacity and shorter actual requests,
    # the uniform-q128 tail warps used to write a full garbage result to each
    # request's FIRST row (tok fell back to q0), racing the CTA that owns it.
    inp = _make_padded_q128_inputs(B, q_len, seed=17)
    gold = _reference(inp).float()
    nvalid = B * q_len
    before = dict(spec_mod.DISPATCH_STATS)
    out = _out_like(inp).fill_(-7)
    _api(inp, out)
    assert spec_mod.DISPATCH_STATS["specialized"] == before["specialized"] + 1, (
        spec_mod.LAST_FALLBACK_REASONS
    )
    valid = out[:nvalid].float()
    diff = (valid - gold).abs()
    tol = 0.005 + 0.05 * gold.abs()
    assert float((diff <= tol).float().mean()) == 1.0
    # the race was also nondeterministic: identical inputs must be bitwise
    # repeatable on the valid rows
    out2 = _out_like(inp)
    _api(inp, out2)
    assert torch.equal(out[:nvalid], out2[:nvalid])
    assert (out[nvalid:] == -7).all().item()


@requires_sm103
def test_u128_replay_changed_query_packing():
    # Capture the U128 general route at full occupancy, then change only the
    # packed query offsets. Both the graph bindings and the launch bounds stay fixed.
    B = 10
    inp = _make_padded_q128_inputs(B, PAGE, seed=17)
    out = _out_like(inp)
    _api(inp, out, backend="specialized")
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.graph(graph, stream=stream):
        _api(inp, out, backend="specialized")
    torch.cuda.current_stream().wait_stream(stream)
    for lengths in ([PAGE] * B, [125, 127] * (B // 2)):
        inp["cu_q"][1:].copy_(inp["cu_q"].new_tensor(lengths).cumsum(0))
        out.fill_(-7)
        graph.replay()
        live = sum(lengths)
        torch.testing.assert_close(
            out[:live].float(), _reference(inp).float(), atol=0.005, rtol=0.05
        )
        assert (out[live:] == -7).all().item()


@requires_sm103
@pytest.mark.parametrize("B", [65535, 65536])
def test_u2_large_batch_beyond_chunk_boundary(B):
    # Regression: uniform q2/kv2 batches larger than 65535 used to be launched
    # in request chunks with q/out pointers advanced by the chunk origin while
    # the kernel's speculation fallback consumed cu_q's ABSOLUTE offsets --
    # applying the origin twice: rows past the boundary were written outside
    # the passed output view and the final logical rows were left untouched.
    dev = torch.device("cuda")
    # Keep the historical double-rebase destination (and its Q read) inside
    # test allocations, but outside the views passed to the binding.
    rows = B * 2
    q = torch.zeros(rows * 2, HQ, D, dtype=torch.uint8, device=dev).view(
        torch.float8_e4m3fn
    )[:rows]
    kv_buf = torch.empty(1, PAGE, 2, D, device=dev)
    kv_buf[:, :, 0, :] = 0.0  # K = 0 -> uniform softmax
    kv_buf[:, :, 1, :] = 1.0  # V = 1 -> every output element is exactly 1
    kv_buf = kv_buf.to(torch.float8_e4m3fn)
    inp = dict(
        q=q,
        kv_buf=kv_buf,
        bt=torch.zeros(B, 1, dtype=torch.int32, device=dev),
        cu_q=2 * torch.arange(B + 1, dtype=torch.int32, device=dev),
        cu_kv=2 * torch.arange(B + 1, dtype=torch.int32, device=dev),
        seq_lens=torch.full((B,), 2, dtype=torch.int32, device=dev),
        max_q=2,
        max_kv=2,
    )
    before = dict(spec_mod.DISPATCH_STATS)
    backing = torch.full((rows * 2, HQ, D), -8.0, dtype=torch.bfloat16, device=dev)
    out = backing[:rows]
    _api(inp, out)
    assert spec_mod.DISPATCH_STATS["specialized"] == before["specialized"] + 1, (
        spec_mod.LAST_FALLBACK_REASONS
    )
    # every logical row (poisoned beforehand) must have been written, exactly
    # within the passed view
    assert (out == 1).all().item()
    assert (backing[rows:] == -8).all().item()


@requires_sm103
def test_allowlist_boundary_multi_page_stays_stock():
    # kv > 128 (multi-page) is outside the allowlist: auto must fall back.
    inp = _make_inputs([4], [512], seed=3)
    before = dict(spec_mod.DISPATCH_STATS)
    out = _out_like(inp)
    _api(inp, out)
    assert spec_mod.DISPATCH_STATS["specialized"] == before["specialized"]
    assert "multi_page_regime" in spec_mod.LAST_FALLBACK_REASONS
    gold = _reference(inp).float()
    diff = (out.float() - gold).abs()
    # Sanity bound only: the stock path's own numeric floor at this contract
    # is not this PR's subject.
    assert float((diff <= 0.005 + 0.05 * gold.abs()).float().mean()) >= 0.95
    with pytest.raises(ValueError, match="fallback_reasons"):
        _api(inp, _out_like(inp), backend="specialized")


@requires_sm103
def test_kill_switch_restores_stock():
    inp = _make_inputs([2] * 8, [2] * 8, seed=11)
    os.environ[spec_mod.KILL_SWITCH_ENV] = "1"
    try:
        before = dict(spec_mod.DISPATCH_STATS)
        out = _out_like(inp)
        _api(inp, out)
        assert spec_mod.DISPATCH_STATS["specialized"] == before["specialized"]
        assert spec_mod.LAST_FALLBACK_REASONS[0].startswith("kill_switch")
        gold = _reference(inp).float()
        diff = (out.float() - gold).abs()
        # Sanity bound only (see test_allowlist_boundary note).
        assert float((diff <= 0.005 + 0.05 * gold.abs()).float().mean()) >= 0.95
        with pytest.raises(ValueError, match="fallback_reasons"):
            _api(inp, _out_like(inp), backend="specialized")
    finally:
        os.environ.pop(spec_mod.KILL_SWITCH_ENV, None)


@requires_sm103
def test_cuda_graph_replay_matches_eager():
    inp = _make_inputs([2, 2], [2, 4], seed=5)
    out = _out_like(inp)
    _api(inp, out, backend="specialized")  # build before capture
    eager = out.clone()
    torch.testing.assert_close(
        eager.float(), _reference(inp).float(), atol=0.005, rtol=0.05
    )
    g = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _api(inp, out)
    torch.cuda.current_stream().wait_stream(stream)
    with torch.cuda.graph(g):
        _api(inp, out, backend="specialized")
    for _ in range(3):
        out.zero_()
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, eager)
    # Change only the per-request KV lengths, not the launch bounds or buffers.
    inp["seq_lens"].copy_(inp["seq_lens"].flip(0))
    inp["cu_kv"][1:].copy_(inp["seq_lens"].cumsum(0))
    out.fill_(float("nan"))
    g.replay()
    torch.testing.assert_close(
        out.float(), _reference(inp).float(), atol=0.005, rtol=0.05
    )
    # replay with mutated inputs: the graph must read the live buffers
    kv = inp["kv_buf"]
    kv.copy_((torch.randn_like(kv.to(torch.float32)) * 0.5).to(torch.float8_e4m3fn))
    inp["q"].copy_(
        (torch.randn_like(inp["q"].to(torch.float32)) * 0.5).to(torch.float8_e4m3fn)
    )
    g.replay()
    torch.cuda.synchronize()
    gold2 = _reference(inp).float()
    diff2 = (out.float() - gold2).abs()
    assert float((diff2 <= 0.005 + 0.05 * gold2.abs()).float().mean()) == 1.0


@requires_sm103
def test_direct_reports_immediate_launch_error(tmp_path):
    # No production fault-injection switch: compile a private copy in which
    # just the default tiny launch has an invalid grid. No device code from
    # that launch can execute. Build and tensor setup are outside raises().
    from flashinfer.jit import env as jit_env
    from flashinfer.jit.core import gen_jit_spec, sm103a_nvcc_flags

    name = "gqa_ctx_fp8_ikv_sm103.cu"
    source = (jit_env.FLASHINFER_CSRC_DIR / name).read_text()
    launch = "LAUNCH(attn_tiny, 2);"
    assert source.count(launch) == 1
    faulty = tmp_path / name
    faulty.write_text(
        source.replace(
            launch, "attn_tiny<2><<<0, 64, 0, st>>>(qp, kvp, btp, cqp, slp, op, btw);"
        )
    )
    run = (
        gen_jit_spec(
            "gqa_ctx_fp8_ikv_sm103_invalid_grid_test",
            [faulty],
            extra_cuda_cflags=sm103a_nvcc_flags
            + ["-DQ2_CAUSAL_PRUNE=1", "-DTINY_KVCAP=1"],
        )
        .build_and_load()
        .gqa_ctx_fp8_ikv_run
    )
    inp = _make_inputs([1], [1])
    k = inp["kv_buf"][:, :, 0, :].unsqueeze(1)
    out = _out_like(inp).fill_(-7)
    torch.cuda.synchronize()
    with pytest.raises(
        RuntimeError,
        match="gqa_ctx_fp8_ikv_run launch at request 0: invalid configuration",
    ):
        run(
            inp["q"],
            k,
            inp["bt"],
            inp["cu_q"],
            inp["cu_kv"],
            inp["seq_lens"],
            1,
            1,
            out,
        )
    assert (out == -7).all().item()


@requires_sm103
def test_generalized_entry_parity_and_dispatch():
    """batch_context_with_kv_cache is a front over NAMED real backends:
    "specialized" links directly to the agent-generated kernel, "trtllm"/
    "cake" are the trtllm-named entry's paths, and "auto" only selects
    among the names (no behavior of its own)."""
    from flashinfer.prefill import (
        BATCH_CONTEXT_BACKENDS,
        batch_context_with_kv_cache,
    )

    assert BATCH_CONTEXT_BACKENDS == ("auto", "specialized", "trtllm", "cake")
    inp = _make_inputs([2] * 8, [2] * 8, seed=11)
    kwargs = _kwargs(inp)
    # "auto" selects "specialized" on its allowlist (dispatch stats move)
    before = dict(spec_mod.DISPATCH_STATS)
    gen_out = batch_context_with_kv_cache(out=_out_like(inp), **kwargs)
    assert spec_mod.DISPATCH_STATS["specialized"] == before["specialized"] + 1
    # the direct specialized name produces the same bits, without touching
    # the trtllm-named entry's dispatch
    gen_spec = batch_context_with_kv_cache(
        out=_out_like(inp), backend="specialized", **kwargs
    )
    assert torch.equal(gen_spec, gen_out)
    # the vendor-named entry stays STOCK: it has no specialized backend
    with pytest.raises(ValueError, match="trtllm-gen|cake"):
        trtllm_batch_context_with_kv_cache(
            out=_out_like(inp), backend="specialized", **kwargs
        )
    # the named "trtllm" backend equals the entry's trtllm-gen path
    gen_stock = batch_context_with_kv_cache(
        out=_out_like(inp), backend="trtllm", **kwargs
    )
    direct_stock = trtllm_batch_context_with_kv_cache(
        out=_out_like(inp), backend="trtllm-gen", **kwargs
    )
    assert torch.equal(gen_stock, direct_stock)
    # off-allowlist: auto selects "trtllm"; explicit "specialized" raises
    big = _make_inputs([2] * 2, [600] * 2, seed=13)  # multi-page KV
    bkwargs = _kwargs(big)
    before = dict(spec_mod.DISPATCH_STATS)
    auto_out = batch_context_with_kv_cache(out=_out_like(big), **bkwargs)
    assert spec_mod.DISPATCH_STATS["specialized"] == before["specialized"]
    assert spec_mod.DISPATCH_STATS["fallback"] == before["fallback"] + 1
    assert "multi_page_regime" in spec_mod.LAST_FALLBACK_REASONS
    trt_out = trtllm_batch_context_with_kv_cache(
        out=_out_like(big), backend="trtllm-gen", **bkwargs
    )
    assert torch.equal(auto_out, trt_out)
    with pytest.raises(ValueError, match="fallback_reasons"):
        batch_context_with_kv_cache(
            out=_out_like(big), backend="specialized", **bkwargs
        )
    # invalid backend raises at the front
    with pytest.raises(ValueError, match="backend must be one of"):
        batch_context_with_kv_cache(out=_out_like(inp), backend="bogus", **kwargs)
