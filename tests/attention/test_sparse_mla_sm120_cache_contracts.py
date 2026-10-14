import pytest
import torch

import flashinfer
from flashinfer.mla import (
    dsv41_fp8_quantize_append_sparse_mla_cache,
    dsv41_fp8_quantize_pack_sparse_mla_cache,
    nvfp4_quantize_pack_sparse_mla_cache,
)
from flashinfer.mla._sparse_mla_sm120 import _get_sparse_mla_sm120_decode_module
from flashinfer.mla._sparse_mla_sm120._dsv4_nvfp4 import (
    get_sparse_mla_nvfp4_sm120_module,
)
from tests.attention.sparse_mla_test_utils import (
    dequantize_kv_dsv4_1,
    quantize_kv_dsv4,
    quantize_kv_dsv4_1,
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
                1.0,
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
                1.0,
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
            1.0,
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
            1.0,
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
                1.0,
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
                1.0,
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
        1.0,
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


# DSV4_1 FP8 (528 B) main-cache writer: bit-equality against the torch reference
# quantizer ``quantize_kv_dsv4_1`` (the trajectory the SM120 DSv4.1 kernels and
# their tests assume), slot semantics and page geometry mirroring the V41_FP4
# writer, and ABI agreement with the hand-written DSv4.1 reader.

_FP8_BPT = 528


def _dsv41_fp8_special_latent(
    nb: int, bs: int, dtype: torch.dtype, device: str = "cuda"
) -> torch.Tensor:
    """``[nb, bs, 1, 512]`` latent rows that exercise every branch of the quantizer."""
    latent = (
        torch.randn(nb, bs, 1, 512, device=device, dtype=torch.float32) / 10.0
    ).clamp(-1, 1)
    latent[0, 0, 0, :32] *= 4000.0  # huge group: scale well above 1
    latent[0, 1, 0, 32:64] = 1e-5  # tiny group: amax floor -> 2^-13 scale
    latent[0, 2, 0, 64:96] = 0.0  # all-zero group (amax floor, zero codes)
    latent[0, 3, 0, 96:128] = -0.0  # negative zero keeps its sign (0x80)
    latent[0, 4, 0, 128] = (
        448.0 * 2.0**-3
    )  # amax exactly 448 * 2^k -> exact power-of-two ratio
    latent[0, 5, 0, 160] = (
        448.0 * 2.0**-3 * (1 + 2.0**-7)
    )  # one ulp above -> next scale
    latent[1, 0, 0, 0] = float("nan")  # NaN poisons its 32-wide group only
    latent[1, 1, 0, 40] = float(
        "inf"
    )  # infinite group: 0x7F for inf, signed zeros elsewhere
    latent[1, 1, 0, 41] = -float("inf")
    latent[1, 2, 0, 64:96] = torch.linspace(
        -448.0, 448.0, 32, device=device
    )  # saturation edges
    latent[1, 3, 0, 96:128] = 0.0
    latent[1, 3, 0, 97] = 2.0**-20  # sub-E4M3 magnitudes round to zero
    if dtype == torch.float16:
        latent = latent.clamp(-60000.0, 60000.0)
    return latent.to(dtype)


def _pack_reference(latent: torch.Tensor, kv_layout: str) -> torch.Tensor:
    ref = quantize_kv_dsv4_1(latent)  # [nb, bs, 1, 528] (NHD form)
    return ref if kv_layout == "NHD" else ref.permute(0, 2, 1, 3)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("kv_layout", ["HND", "NHD"])
@pytest.mark.parametrize("input_layout", ["3d", "hnd", "nhd"])
def test_dsv41_fp8_quantize_pack_matches_reference(
    dtype, kv_layout, input_layout, sm12x
):
    """Every page / layout form of the full-page pack is bit-identical to
    ``quantize_kv_dsv4_1`` (scale floor, exact power-of-two ratios, saturation,
    signed zero, NaN / inf groups, FP16 and BF16 inputs)."""
    torch.manual_seed(528)
    nb, bs = 3, 8
    latent = _dsv41_fp8_special_latent(nb, bs, dtype)
    inputs = {
        "3d": latent.squeeze(2),
        "hnd": latent.permute(0, 2, 1, 3).contiguous(),
        "nhd": latent,
    }[input_layout]
    cache = dsv41_fp8_quantize_pack_sparse_mla_cache(inputs, kv_layout=kv_layout)
    ref = _pack_reference(latent, kv_layout)
    assert cache.shape == ref.shape and cache.dtype == torch.uint8
    assert torch.equal(cache, ref)
    # The written pages decode to the same BF16 rows as the reference pages
    # (compared bitwise: the NaN / inf groups decode to NaN).
    decoded = dequantize_kv_dsv4_1(
        cache.permute(0, 2, 1, 3) if kv_layout == "HND" else cache
    )
    expected = dequantize_kv_dsv4_1(quantize_kv_dsv4_1(latent))
    assert torch.equal(decoded.view(torch.int16), expected.view(torch.int16))


def test_dsv41_fp8_quantize_pack_rounding_edges(sm12x):
    """Scale selection is the exact power-of-two ceiling of ``amax / 448`` (the
    torch ``log2().ceil()`` trajectory agrees for every BF16 / FP16 amax) and
    the E4M3 codes round to nearest-even through the whole subnormal range."""
    torch.manual_seed(1)
    device = torch.device("cuda")
    # 512 groups x 32 values whose amax sweeps every BF16 mantissa / exponent
    # combination between 2^-20 and 2^15, plus random fill.
    groups = 512 * 16
    fill = torch.randn(groups, 32, device=device) / 7.0
    exponents = torch.randint(-20, 16, (groups,), device=device).float()
    mantissas = 1.0 + torch.randint(0, 128, (groups,), device=device).float() / 128.0
    fill[:, 0] = (mantissas * torch.pow(2.0, exponents)).to(torch.bfloat16).float()
    fill[:, 1:] *= fill[:, :1].abs() * 0.75
    latent = fill.view(32, 16, 1, 512).to(torch.bfloat16)
    cache = dsv41_fp8_quantize_pack_sparse_mla_cache(latent.squeeze(2), kv_layout="NHD")
    assert torch.equal(cache, quantize_kv_dsv4_1(latent))


@pytest.mark.parametrize("slot_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("cache_form", ["2d", "3d", "hnd", "nhd"])
def test_dsv41_fp8_quantize_append_matches_pack(slot_dtype, cache_form, sm12x):
    """Append-by-slot lands the pack bytes at the mapped slots, ignores negative
    and out-of-range slots, resolves duplicates to the lowest-index row and
    preserves every unaddressed data row and scale slot."""
    torch.manual_seed(2)
    device = torch.device("cuda")
    nb, bs = 4, 8
    latent = _dsv41_fp8_special_latent(nb, bs, torch.bfloat16)
    full = dsv41_fp8_quantize_pack_sparse_mla_cache(latent.squeeze(2), kv_layout="NHD")
    flat_full = full.reshape(nb, bs * _FP8_BPT)
    backing = torch.full((nb, bs * _FP8_BPT), 0x91, device=device, dtype=torch.uint8)
    cache = {
        "2d": backing,
        "3d": backing.view(nb, bs, _FP8_BPT),
        "hnd": backing.view(nb, 1, bs, _FP8_BPT),
        "nhd": backing.view(nb, bs, 1, _FP8_BPT),
    }[cache_form]
    rows = latent.reshape(-1, 512)
    slot_mapping = torch.randperm(nb * bs, device=device).to(slot_dtype)
    slot_mapping[9] = slot_mapping[5]  # duplicate: token 5 wins
    slot_mapping[12] = -1  # padding
    slot_mapping[13] = nb * bs + 3  # out of range: ignored
    slot_mapping[14] = -7
    assert dsv41_fp8_quantize_append_sparse_mla_cache(rows, slot_mapping, cache) is None
    torch.cuda.synchronize()
    slots = slot_mapping.tolist()
    written = set()
    for tok, slot in enumerate(slots):
        if slot < 0 or slot >= nb * bs or slot in slots[:tok]:
            continue
        written.add(slot)
        page_s, entry_s = divmod(slot, bs)
        page_t, entry_t = divmod(tok, bs)
        data_s, data_t = entry_s * 512, entry_t * 512
        sc_s, sc_t = bs * 512 + entry_s * 16, bs * 512 + entry_t * 16
        assert torch.equal(
            backing[page_s, data_s : data_s + 512],
            flat_full[page_t, data_t : data_t + 512],
        )
        assert torch.equal(
            backing[page_s, sc_s : sc_s + 16], flat_full[page_t, sc_t : sc_t + 16]
        )
    for slot in range(nb * bs):
        if slot in written:
            continue
        page, entry = divmod(slot, bs)
        assert torch.all(backing[page, entry * 512 : (entry + 1) * 512] == 0x91)
        sc = bs * 512 + entry * 16
        assert torch.all(backing[page, sc : sc + 16] == 0x91)


@pytest.mark.parametrize("append,issue", [(False, "row_gap"), (True, "page_overlap")])
def test_dsv41_fp8_writer_rejects_unpacked_footer(append, issue, sm12x):
    page_size, bpt = 3, _FP8_BPT
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
            module.sparse_mla_sm120_dsv41_fp8_quantize_append(
                latent, torch.arange(6, device="cuda", dtype=torch.int32), cache
            )
        else:
            module.sparse_mla_sm120_dsv41_fp8_quantize_pack(latent, cache)
    assert torch.all(storage == 91)


def test_dsv41_fp8_writer_rejects_bad_arguments(sm12x):
    latent = torch.zeros(2, 3, 512, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="kv_layout"):
        dsv41_fp8_quantize_pack_sparse_mla_cache(latent, kv_layout="HDN")
    with pytest.raises(ValueError, match="page dimension"):
        dsv41_fp8_quantize_pack_sparse_mla_cache(latent.reshape(6, 512))
    with pytest.raises(ValueError, match="singleton latent-head axis"):
        dsv41_fp8_quantize_pack_sparse_mla_cache(latent.reshape(2, 3, 2, 256))
    cache = dsv41_fp8_quantize_pack_sparse_mla_cache(latent)
    assert cache.shape == (2, 1, 3, _FP8_BPT)
    with pytest.raises(ValueError, match="slot_mapping"):
        dsv41_fp8_quantize_append_sparse_mla_cache(
            latent.reshape(6, 512),
            torch.zeros(6, device="cuda", dtype=torch.int16),
            cache,
        )
    with pytest.raises(ValueError, match="contiguous 1D"):
        dsv41_fp8_quantize_append_sparse_mla_cache(
            latent.reshape(6, 512),
            torch.zeros(2, 3, device="cuda", dtype=torch.int32),
            cache,
        )
    with pytest.raises(RuntimeError, match="last dimension"):
        dsv41_fp8_quantize_append_sparse_mla_cache(
            latent.reshape(6, 512),
            torch.zeros(6, device="cuda", dtype=torch.int32),
            torch.zeros(2, 1, 3, 288, device="cuda", dtype=torch.uint8),
        )
    with pytest.raises(RuntimeError, match="rows"):
        dsv41_fp8_quantize_append_sparse_mla_cache(
            latent.reshape(6, 512),
            torch.zeros(5, device="cuda", dtype=torch.int32),
            cache,
        )


def test_dsv41_fp8_writer_pitched_footer_pack_append_graph(sm12x):
    """Non-power-of-two pages inside a pitched (padded) pool: pack into the
    pitched view equals the packed pack, append replays bitwise under CUDA
    graph capture and never touches the padding."""
    torch.manual_seed(41)
    for page_size in (3, 5):
        latent = (
            torch.randn(2 * page_size, 512, device="cuda", dtype=torch.bfloat16) * 0.1
        )
        expected = dsv41_fp8_quantize_pack_sparse_mla_cache(
            latent.view(2, page_size, 512), kv_layout="NHD"
        ).squeeze(2)
        assert torch.equal(
            expected.unsqueeze(2), quantize_kv_dsv4_1(latent.view(2, page_size, 1, 512))
        )
        backing = torch.full(
            (2, page_size * _FP8_BPT + 16), 91, device="cuda", dtype=torch.uint8
        )
        cache = backing[:, : page_size * _FP8_BPT].view(2, page_size, 1, _FP8_BPT)
        slots = torch.arange(2 * page_size, device="cuda", dtype=torch.int32)
        _get_sparse_mla_sm120_decode_module().sparse_mla_sm120_dsv41_fp8_quantize_pack(
            latent, cache
        )
        assert torch.equal(cache.squeeze(2), expected)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            dsv41_fp8_quantize_append_sparse_mla_cache(latent, slots, cache)
        latent.mul_(0.5)
        graph.replay()
        torch.cuda.synchronize()
        expected = dsv41_fp8_quantize_pack_sparse_mla_cache(
            latent.view(2, page_size, 512), kv_layout="NHD"
        ).squeeze(2)
        assert torch.equal(cache.squeeze(2), expected)
        for _ in range(2):
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(cache.squeeze(2), expected)
        assert torch.all(backing[:, page_size * _FP8_BPT :] == 91)


def test_dsv41_fp8_writer_cache_feeds_the_dsv41_reader(sm12x):
    """The written pages are consumed unchanged by the hand-written DSv4.1
    BF16 decode: attention over a device-written pool equals attention over
    the torch-quantized pool bit for bit, and matches the dequantized reference."""
    from tests.attention.sparse_mla_test_utils import _ref_sparse_attn

    torch.manual_seed(7)
    device = torch.device("cuda")
    nb, bs, heads, topk = 4, 61, 16, 128
    latent = (
        torch.randn(nb, bs, 1, 512, device=device, dtype=torch.bfloat16) / 10.0
    ).clamp(-1, 1)
    written = dsv41_fp8_quantize_pack_sparse_mla_cache(
        latent.squeeze(2), kv_layout="NHD"
    )
    reference_cache = quantize_kv_dsv4_1(latent)
    assert torch.equal(written, reference_cache)
    q = (torch.randn(2, heads, 512, device=device, dtype=torch.bfloat16) / 10.0).clamp(
        -1, 1
    )
    indices = torch.randint(0, nb * bs, (2, topk), device=device, dtype=torch.int32)
    outputs = []
    for cache in (written, reference_cache):
        wrapper = flashinfer.mla.SparseMLASm120Wrapper(
            kv_scale_format="ue8m0_g32", compute_precision="bf16", device=device
        )
        out = torch.empty_like(q)
        wrapper.run(q, cache, indices, out, 512**-0.5)
        outputs.append(out)
    torch.cuda.synchronize()
    assert torch.equal(outputs[0], outputs[1])
    ref, _ = _ref_sparse_attn(q, dequantize_kv_dsv4_1(written), indices, 512**-0.5, 512)
    torch.testing.assert_close(outputs[0], ref, atol=5e-2, rtol=5e-2)
