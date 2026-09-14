import pytest
import torch

from flashinfer.mla._sparse_mla_sm120 import _get_sparse_mla_sm120_decode_module
from flashinfer.mla._sparse_mla_sm120._dsv4_nvfp4 import (
    get_sparse_mla_nvfp4_sm120_module,
)
from flashinfer.mla import nvfp4_quantize_pack_sparse_mla_cache
from tests.attention.sparse_mla_test_utils import (
    quantize_kv_dsv4,
    quantize_kv_glm53_nope,
    require_sm12x,
)


@pytest.fixture
def sm12x():
    require_sm12x()


@pytest.mark.usefixtures("sm12x")
@pytest.mark.parametrize(
    "route,tokens,heads,dual",
    [
        ("decode", 2, 16, False),
        ("decode", 8, 64, True),
        ("prefill", 65, 32, False),
        ("prefill", 65, 64, True),
    ],
)
def test_dsv4_nvfp4_masked_rows_ignore_poisoned_slot_zero(route, tokens, heads, dual):
    torch.manual_seed(5075)
    cache = nvfp4_quantize_pack_sparse_mla_cache(
        torch.randn(4, 64, 512, device="cuda", dtype=torch.bfloat16) * 0.1
    )
    extra = cache.clone() if dual else None
    q = torch.randn(tokens, heads, 512, device="cuda", dtype=torch.bfloat16) * 0.1
    idx = torch.randint(1, 256, (tokens, 512), device="cuda", dtype=torch.int32)
    idx[:, 1::2] = -1
    exidx = idx.clone() if dual else None
    splits = 16 if dual else 8
    mid = torch.empty(tokens, heads, splits, 512, device="cuda", dtype=torch.bfloat16)
    mlse = torch.empty(tokens, heads, splits, device="cuda")
    out, lse = torch.empty_like(q), torch.empty(tokens, heads, device="cuda")
    module = get_sparse_mla_nvfp4_sm120_module()

    def call():
        if route == "decode":
            module.sparse_mla_sm120_nvfp4_decode(
                q,
                cache,
                idx,
                mid,
                mlse,
                out,
                lse,
                splits,
                512**-0.5,
                None,
                None,
                extra,
                exidx,
                None,
                1,
                False,
            )
        else:
            module.sparse_mla_sm120_nvfp4_prefill(
                q,
                cache,
                idx,
                out,
                lse,
                512**-0.5,
                None,
                None,
                extra,
                exidx,
                None,
            )

    call()
    expected = out.clone(), lse.clone()
    for pool in (cache, extra):
        if pool is not None:
            page = pool[0].view(-1)
            page[:352] = 255
            page[64 * 352 : 64 * 352 + 32] = 255
    call()
    assert torch.isfinite(out).all()
    assert torch.equal(out, expected[0]) and torch.equal(lse, expected[1])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for _ in range(3):
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, expected[0]) and torch.equal(lse, expected[1])


def _fp8_call(cache, prefill, model=1, extra=None):
    q = torch.zeros(1, 16, 512, device="cuda", dtype=torch.bfloat16)
    idx = torch.full((1, 128), -1, device="cuda", dtype=torch.int32)
    out, lse = torch.empty_like(q), torch.empty(1, 16, device="cuda")
    mid = torch.empty(1, 16, 4, 512, device="cuda", dtype=torch.bfloat16)
    mlse = torch.empty(1, 16, 4, device="cuda")
    m = _get_sparse_mla_sm120_decode_module()
    if prefill:
        m.sparse_mla_sm120_paged_attention(
            q,
            cache,
            idx,
            out,
            lse,
            512**-0.5,
            model,
            1 if model == 5 else 2,
            None,
            None,
            extra,
            idx if extra is not None else None,
            None,
            False,
        )
    else:
        m.sparse_mla_sm120_decode_dsv4(
            q,
            cache,
            idx,
            mid,
            mlse,
            out,
            lse,
            4,
            512**-0.5,
            None,
            None,
            extra,
            idx if extra is not None else None,
            None,
            model,
            1,
            False,
        )
    torch.cuda.synchronize()


@pytest.mark.parametrize(
    "prefill,issue,model",
    [
        (True, "origin", 1),
        (False, "page", 1),
        (False, "origin", 5),
        (True, "page", 5),
        (True, "capacity", 1),
        (False, "capacity", 1),
    ],
)
def test_cache_alignment_and_flat_capacity_rejected(prefill, issue, model, sm12x):
    bpt = 528 if model == 5 else 584
    width = 64 * bpt
    stride = width + 8 if issue == "page" else width
    if issue == "capacity":
        stride = width - 16
    offset = 1 if issue == "origin" else 0
    storage = torch.zeros(2 * width + 32, device="cuda", dtype=torch.uint8)
    cache = storage.as_strided((2, width), (stride, 1), offset)
    message = "data pointer" if issue == "origin" else "block stride"
    with pytest.raises(RuntimeError, match=message):
        _fp8_call(cache, prefill, model)


@pytest.mark.parametrize(
    "prefill,layout,extra,model",
    [
        (True, "3d", False, 1),
        (False, "hnd", False, 1),
        (False, "nhd", True, 5),
        (True, "nhd", True, 5),
    ],
)
def test_footer_row_gap_rejected(prefill, layout, extra, model, sm12x):
    bpt = 528 if model == 5 else 584
    cache = torch.zeros(2, 64, bpt + 16, device="cuda", dtype=torch.uint8)[..., :bpt]
    if layout == "hnd":
        cache = cache.unsqueeze(1)
    elif layout == "nhd":
        cache = cache.unsqueeze(2)
    packed = torch.zeros(2, 64 * bpt, device="cuda", dtype=torch.uint8)
    with pytest.raises(RuntimeError, match="packed"):
        _fp8_call(packed if extra else cache, prefill, model, cache if extra else None)


@pytest.mark.parametrize("tokens,heads", [(1, 8), (4, 8), (4, 16), (4, 24)])
def test_glm_decode_preserves_scratch_guards(tokens, heads, sm12x):
    splits = 34
    scratch_heads = heads if heads == 8 else (heads + 15) // 16 * 16

    def guarded(shape, dtype):
        size = 1
        for dim in shape:
            size *= dim
        storage = torch.full((2 * size,), 37, device="cuda", dtype=dtype)
        return storage[:size].view(shape), storage[size:]

    mid, guard = guarded((tokens, scratch_heads, splits, 512), torch.bfloat16)
    mlse, lguard = guarded((tokens, scratch_heads, splits), torch.float32)
    cache = quantize_kv_glm53_nope(
        torch.full((1, 64, 1, 512), 0.5, device="cuda", dtype=torch.bfloat16)
    )[..., :528].contiguous()
    q = torch.zeros(tokens, heads, 512, device="cuda", dtype=torch.bfloat16)
    idx = torch.full((tokens, 2176), -1, device="cuda", dtype=torch.int32)
    idx[:, 0] = 1
    out, lse = torch.empty_like(q), torch.empty(tokens, heads, device="cuda")
    m = _get_sparse_mla_sm120_decode_module()

    def call():
        m.sparse_mla_sm120_decode_dsv3_2(
            q,
            cache,
            idx,
            mid,
            mlse,
            out,
            lse,
            splits,
            512**-0.5,
            None,
            None,
            3,
            1,
        )

    call()
    torch.cuda.synchronize()
    assert torch.all(guard == 37) and torch.all(lguard == 37)
    torch.testing.assert_close(out, torch.full_like(out, 0.5), atol=1e-3, rtol=1e-3)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.all(guard == 37) and torch.all(lguard == 37)


@pytest.mark.parametrize("prefill,glm", [(False, False), (True, False), (False, True)])
def test_flat_page_pitch_preserved(prefill, glm, sm12x):
    torch.manual_seed(53)
    model, bpt = (3, 528) if glm else (1, 584)
    quantize = quantize_kv_glm53_nope if glm else quantize_kv_dsv4
    packed = quantize(
        torch.randn(4, 64, 1, 512, device="cuda", dtype=torch.bfloat16) * 0.1
    )
    packed = packed[..., :bpt].contiguous().view(4, 64 * bpt)
    storage = torch.full((4, 64 * bpt + 16), 255, device="cuda", dtype=torch.uint8)
    pitched = storage[:, : 64 * bpt]
    pitched.copy_(packed)
    q = torch.randn(4, 32, 512, device="cuda", dtype=torch.bfloat16) * 0.1
    idx = torch.randint(
        64, 256, (4, 2176 if glm else 128), device="cuda", dtype=torch.int32
    )
    splits = (idx.shape[1] + 63) // 64
    mid = torch.empty(4, 32, splits, 512, device="cuda", dtype=torch.bfloat16)
    mlse = torch.empty(4, 32, splits, device="cuda")
    m = _get_sparse_mla_sm120_decode_module()

    def run(cache):
        out, lse = torch.empty_like(q), torch.empty(4, 32, device="cuda")
        if prefill:
            m.sparse_mla_sm120_paged_attention(
                q,
                cache,
                idx,
                out,
                lse,
                512**-0.5,
                model,
                2,
                None,
                None,
                None,
                None,
                None,
                False,
            )
        elif glm:
            m.sparse_mla_sm120_decode_dsv3_2(
                q,
                cache,
                idx,
                mid,
                mlse,
                out,
                lse,
                splits,
                512**-0.5,
                None,
                None,
                model,
                1,
            )
        else:
            m.sparse_mla_sm120_decode_dsv4(
                q,
                cache,
                idx,
                mid,
                mlse,
                out,
                lse,
                splits,
                512**-0.5,
                None,
                None,
                None,
                None,
                None,
                model,
                1,
                False,
            )
        return out, lse

    expected = run(packed)
    actual = run(pitched)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))


@pytest.mark.parametrize("append,issue", [(False, "row_gap"), (True, "page_overlap")])
def test_dsv41_writer_rejects_unpacked_footer(append, issue, sm12x):
    page_size, bpt = 3, 288
    row_stride = bpt + 16 if issue == "row_gap" else bpt
    page_stride = page_size * row_stride if issue == "row_gap" else page_size * bpt - 16
    storage = torch.full(
        (2 * page_size * row_stride + 32,), 91, device="cuda", dtype=torch.uint8
    )
    cache = storage.as_strided((2, page_size, bpt), (page_stride, row_stride, 1))
    if append:
        cache = cache.unsqueeze(1)
    latent = torch.zeros(6, 512, device="cuda", dtype=torch.bfloat16)
    module = _get_sparse_mla_sm120_decode_module()
    with pytest.raises(RuntimeError, match="packed|payload"):
        if append:
            module.sparse_mla_sm120_dsv41_fp4_quantize_append(
                latent, torch.arange(6, device="cuda", dtype=torch.int32), cache
            )
        else:
            module.sparse_mla_sm120_dsv41_fp4_quantize_pack(latent, cache)
    assert torch.all(storage == 91)


def test_dsv41_writer_pitched_footer_pack_append_graph(sm12x):
    torch.manual_seed(41)
    module = _get_sparse_mla_sm120_decode_module()
    for page_size in (3, 5):
        latent = (
            torch.randn(2 * page_size, 512, device="cuda", dtype=torch.bfloat16) * 0.1
        )
        expected = torch.empty(2, page_size, 288, device="cuda", dtype=torch.uint8)
        module.sparse_mla_sm120_dsv41_fp4_quantize_pack(latent, expected)
        backing = torch.full(
            (2, page_size * 288 + 16), 91, device="cuda", dtype=torch.uint8
        )
        cache = backing[:, : page_size * 288].view(2, page_size, 1, 288)
        slots = torch.arange(2 * page_size, device="cuda", dtype=torch.int32)
        module.sparse_mla_sm120_dsv41_fp4_quantize_pack(latent, cache)
        assert torch.equal(cache.squeeze(2), expected)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            module.sparse_mla_sm120_dsv41_fp4_quantize_append(latent, slots, cache)
        latent.mul_(0.5)
        graph.replay()
        module.sparse_mla_sm120_dsv41_fp4_quantize_pack(latent, expected)
        assert torch.equal(cache.squeeze(2), expected)
        assert torch.all(backing[:, page_size * 288 :] == 91)


def test_dsv4_nvfp4_final_helper_rejects_stage1_only(monkeypatch, sm12x):
    from flashinfer.mla._sparse_mla_sm120 import _dsv4_nvfp4 as dsv4_nvfp4

    q = torch.zeros(1, 16, 512, device="cuda", dtype=torch.bfloat16)
    cache = nvfp4_quantize_pack_sparse_mla_cache(
        torch.zeros(2, 64, 512, device="cuda", dtype=torch.bfloat16)
    )
    idx = torch.zeros(1, 128, device="cuda", dtype=torch.int32)
    original_empty, original_like = torch.empty, torch.empty_like

    def poisoned_empty(*args, **kwargs):
        return original_empty(*args, **kwargs).fill_(float("nan"))

    def poisoned_like(*args, **kwargs):
        return original_like(*args, **kwargs).fill_(float("nan"))

    with monkeypatch.context() as patcher:
        patcher.setattr(torch, "empty", poisoned_empty)
        patcher.setattr(torch, "empty_like", poisoned_like)
        try:
            out, lse = dsv4_nvfp4._nvfp4_sparse_mla_decode(
                q, cache, idx, 512**-0.5, stage1_only=True
            )
        except ValueError as exc:
            assert "stage1_only" in str(exc)
        else:
            assert torch.isnan(out).all() and torch.isnan(lse).all()
            pytest.fail("stage1_only returned unwritten final output and LSE")
    out, lse = dsv4_nvfp4._nvfp4_sparse_mla_decode(q, cache, idx, 512**-0.5)
    assert torch.isfinite(out).all() and torch.isfinite(lse).all()
    mid = torch.full((1, 16, 2, 512), float("nan"), device="cuda", dtype=torch.bfloat16)
    mlse = torch.full((1, 16, 2), float("nan"), device="cuda")
    out.fill_(float("nan"))
    lse.fill_(float("nan"))
    dsv4_nvfp4.get_sparse_mla_nvfp4_sm120_module().sparse_mla_sm120_nvfp4_decode(
        q,
        cache,
        idx,
        mid,
        mlse,
        out,
        lse,
        2,
        512**-0.5,
        None,
        None,
        None,
        None,
        None,
        1,
        True,
    )
    assert torch.isfinite(mid).all() and torch.isfinite(mlse).all()
    assert torch.isnan(out).all() and torch.isnan(lse).all()


@pytest.mark.parametrize("prefill,rank", [(False, 3), (True, 4)])
def test_attention_page_overlap_rejected(prefill, rank, sm12x):
    storage = torch.zeros(2 * 64 * 584, device="cuda", dtype=torch.uint8)
    cache = storage.as_strided((2, 64, 584), (64 * 584 - 16, 584, 1))
    if rank == 4:
        cache = cache.unsqueeze(1)
    with pytest.raises(RuntimeError, match="block stride|page.*capacity"):
        _fp8_call(cache, prefill)
