"""DeepSeek-V4.1 inline-scale KV records on SM120.

FlashMLA's V4.1 ABI (2026.09.30 release, adopted by vLLM) stores each token's
data bytes followed by its own scale bytes: 528 B for the FP8 cache (512 E4M3
+ 16 UE8M0 over 32-wide groups), 288 B for the V41_FP4 cache (256 E2M1 + 32
E4M3 over 16-wide groups). The footer-scale form keeps ``page_size * data``
bytes ahead of ``page_size * scales`` bytes per page. The inline formats run
the DSV4_1_INLINE kernels, which bulk-copy each record whole and address the
records as rows of the whole cache; the writers use that a one-token footer
page is an inline record. These tests pin the byte layout, vLLM's writer, and
footer-equal attention on every DSV4_1 route.
"""

import pytest
import torch

import flashinfer
from flashinfer.mla import (
    dsv41_fp4_quantize_append_sparse_mla_cache,
    dsv41_fp4_quantize_pack_sparse_mla_cache,
    dsv41_fp8_quantize_append_sparse_mla_cache,
    dsv41_fp8_quantize_pack_sparse_mla_cache,
)
from flashinfer.mla._sparse_mla_sm120._execution import check_inline_scale_cache
from tests.attention.sparse_mla_test_utils import require_sm12x

_FP8 = (512, 16)  # data bytes, scale bytes per token
_FP4 = (256, 32)


@pytest.fixture
def sm12x():
    require_sm12x()


def _footer_to_inline(footer: torch.Tensor, data: int, scales: int) -> torch.Tensor:
    """Rearrange contiguous footer pages ``[nb, bs, 1, bpt]`` into inline rows."""
    nb, bs, _, bpt = footer.shape
    flat = footer.reshape(nb, bs * bpt)
    rows = torch.cat(
        [
            flat[:, : bs * data].reshape(nb, bs, data),
            flat[:, bs * data :].reshape(nb, bs, scales),
        ],
        dim=-1,
    )
    return rows.reshape(nb, bs, 1, bpt)


def _packed_block_view(
    rows: torch.Tensor, span: int, fill: int = 0x5A, pitch: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Place inline rows ``[nb, bs, 1, bpt]`` in blocks of ``span >= bs`` rows.

    Mirrors vLLM's packed KV blocks, where one physical block holds several
    layers' pages and a layer's cache is a page-strided view. Rows sit
    ``pitch`` bytes apart (default: packed). Returns the ``[nb, bs, 1, bpt]``
    view and its backing buffer (the rows past ``bs`` and the bytes past each
    record keep ``fill``).
    """
    nb, bs, _, bpt = rows.shape
    pitch = pitch or bpt
    backing = torch.full(
        (nb, span * pitch), fill, dtype=torch.uint8, device=rows.device
    )
    view = backing.as_strided((nb, bs, 1, bpt), (span * pitch, pitch, pitch, 1))
    view.copy_(rows)
    return view, backing


def _row_indices(indices: torch.Tensor, page_size: int, span: int) -> torch.Tensor:
    """Map paged slot IDs to the row IDs of a cache whose pages are ``span`` rows apart."""
    page, slot = indices // page_size, indices % page_size
    return torch.where(indices < 0, indices, page * span + slot)


# --- Cache geometry checks (host only) -------------------------------------


@pytest.mark.parametrize("form", ["2d", "3d", "hnd", "nhd"])
@pytest.mark.parametrize("pitch", [528, 544, 1056])
def test_inline_cache_check_accepts_row_layouts(form, pitch):
    """Rows at a uniform 16-byte multiple pitch with a whole-row page stride
    are rows of the whole cache, in multi-row and one-row pages."""
    backing = torch.zeros(3, 7 * pitch, dtype=torch.uint8)
    cache3 = backing.as_strided((3, 5, 528), (7 * pitch, pitch, 1))
    cache = {
        "2d": torch.zeros(4, pitch, dtype=torch.uint8)[:, :528],
        "3d": cache3,
        "hnd": cache3.unsqueeze(1),
        "nhd": cache3.unsqueeze(2),
    }[form]
    check_inline_scale_cache(cache, 528, "kv_cache")


@pytest.mark.parametrize(
    "shape,strides,match",
    [
        ((2, 4, 528), (4 * 536, 536, 1), "16-byte multiple"),  # unaligned pitch
        ((4, 1, 288), (280, 280, 1), "16-byte multiple"),  # pitch below the record
        ((2, 4, 528), (4 * 528 + 16, 528, 1), "rows of the whole cache"),
        ((2, 4, 528), (4 * 544 + 528, 544, 1), "rows of the whole cache"),
    ],
)
def test_inline_cache_check_rejects_non_row_layouts(shape, strides, match):
    backing = torch.zeros(16 * 544, dtype=torch.uint8)
    cache = backing.as_strided(shape, strides)
    with pytest.raises(ValueError, match=match):
        check_inline_scale_cache(cache, shape[-1], "kv_cache")


def test_inline_cache_check_rejects_wrong_width_and_dtype():
    with pytest.raises(ValueError, match="528-byte record"):
        check_inline_scale_cache(torch.zeros(2, 4, 512, dtype=torch.uint8), 528, "kv")
    with pytest.raises(ValueError, match="uint8"):
        check_inline_scale_cache(torch.zeros(2, 4, 528, dtype=torch.int8), 528, "kv")


# --- Writers ----------------------------------------------------------------


def _latent(nb: int, bs: int, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    return (torch.randn(nb, bs, 512, device="cuda", dtype=torch.bfloat16) / 10.0).clamp(
        -1, 1
    )


@pytest.mark.parametrize("kv_layout", ["HND", "NHD"])
@pytest.mark.parametrize(
    "pack,fmt",
    [
        (dsv41_fp8_quantize_pack_sparse_mla_cache, _FP8),
        (dsv41_fp4_quantize_pack_sparse_mla_cache, _FP4),
    ],
)
def test_inline_pack_is_footer_pack_per_token(pack, fmt, kv_layout, sm12x):
    """The inline pack stores the footer pack's bytes, each token's scales
    moved right after its data."""
    latent = _latent(3, 61, seed=41)
    footer = pack(latent, kv_layout="NHD")
    inline = pack(latent, kv_layout=kv_layout, inline_scale=True)
    if kv_layout == "HND":
        assert inline.shape == (3, 1, 61, footer.shape[-1])
        inline = inline.transpose(1, 2)
    assert inline.shape == footer.shape
    assert torch.equal(inline, _footer_to_inline(footer, *fmt))


def _vllm_insert_v41_mxfp8(
    k: torch.Tensor, slots: torch.Tensor, blocks: torch.Tensor, block_size: int
) -> None:
    """Torch port of vLLM's ``_quantize_and_insert_k_mxfp8_kernel`` (vLLM
    f8460ef236, vllm/models/deepseek_v41/common/ops/cache_utils.py): token
    ``slot`` lands at ``block * block_stride + pos * 528``, its 16 UE8M0
    scales at ``+512``."""
    x = k.float().view(-1, 16, 32)
    amax = x.abs().amax(-1).clamp_min(1e-4)
    exponent = torch.ceil(torch.log2(amax * (1.0 / 448.0)))
    fp8 = (x * torch.exp2(-exponent).unsqueeze(-1)).clamp(-448.0, 448.0)
    data = fp8.to(torch.float8_e4m3fn).view(torch.uint8).view(-1, 512)
    encoded = (exponent + 127.0).clamp(0.0, 255.0).to(torch.uint8)
    for i, slot in enumerate(slots.tolist()):
        if slot == -1:
            continue
        block, pos = divmod(slot, block_size)
        row = blocks[block, pos * 528 : (pos + 1) * 528]
        row[:512] = data[i]
        row[512:] = encoded[i]


def test_inline_fp8_append_matches_vllm_writer_in_packed_blocks(sm12x):
    """FlashInfer's inline FP8 append writes the same bytes as vLLM's V4.1
    cache writer into a packed-block cache (row IDs on the FlashInfer side,
    vLLM slot IDs on the vLLM side)."""
    nb, bs, span = 4, 61, 67
    latent = _latent(1, nb * bs, seed=528).view(nb * bs, 512)
    slots = torch.randperm(nb * bs, device="cuda")[: nb * bs - 7]
    slots[3] = -1  # padding is ignored
    latent = latent[: slots.numel()]
    expected = torch.full((nb, span * 528), 0x5A, dtype=torch.uint8, device="cuda")
    _vllm_insert_v41_mxfp8(latent, slots, expected, bs)

    cache, backing = _packed_block_view(
        torch.full((nb, bs, 1, 528), 0x5A, dtype=torch.uint8, device="cuda"), span
    )
    dsv41_fp8_quantize_append_sparse_mla_cache(
        latent,
        _row_indices(slots, bs, span),
        cache.squeeze(2),
        inline_scale=True,
    )
    torch.cuda.synchronize()
    assert torch.equal(backing, expected)


def test_inline_fp4_append_matches_inline_pack_in_packed_blocks(sm12x):
    nb, bs, span = 3, 32, 41
    latent = _latent(nb, bs, seed=288)
    rows = dsv41_fp4_quantize_pack_sparse_mla_cache(
        latent, kv_layout="NHD", inline_scale=True
    )
    cache, backing = _packed_block_view(torch.full_like(rows, 0x5A), span)
    slots = torch.arange(nb * bs, device="cuda")
    dsv41_fp4_quantize_append_sparse_mla_cache(
        latent.reshape(-1, 512),
        _row_indices(slots, bs, span),
        cache,
        inline_scale=True,
    )
    expected, _ = _packed_block_view(rows, span)
    torch.cuda.synchronize()
    assert torch.equal(cache, expected)
    pad = backing.view(nb, span, 288)[:, bs:]
    assert torch.all(pad == 0x5A)


@pytest.mark.parametrize(
    "append,width",
    [
        (dsv41_fp8_quantize_append_sparse_mla_cache, 528),
        (dsv41_fp4_quantize_append_sparse_mla_cache, 288),
    ],
)
def test_inline_append_one_row_pages_use_the_pitch(append, width, sm12x):
    """A ``[num_rows, record]`` cache with a padded row pitch takes row IDs."""
    pack = (
        dsv41_fp8_quantize_pack_sparse_mla_cache
        if width == 528
        else dsv41_fp4_quantize_pack_sparse_mla_cache
    )
    latent = _latent(1, 40, seed=width)
    rows = pack(latent, kv_layout="NHD", inline_scale=True).view(40, width)
    backing = torch.full((40, width + 32), 0x5A, dtype=torch.uint8, device="cuda")
    slots = torch.randperm(40, device="cuda")
    append(latent.view(40, 512), slots, backing[:, :width], inline_scale=True)
    torch.cuda.synchronize()
    assert torch.equal(backing[slots, :width], rows)
    assert torch.all(backing[:, width:] == 0x5A)


@pytest.mark.parametrize(
    "width,pitch", [(528, 544), (528, 1056), (288, 304), (288, 576)]
)
def test_inline_append_padded_row_pitch_in_packed_blocks(width, pitch, sm12x):
    """Rows at a padded pitch in packed blocks: each row lands at
    ``row * pitch`` and the row and block padding stay untouched."""
    pack, append = (
        (
            dsv41_fp8_quantize_pack_sparse_mla_cache,
            dsv41_fp8_quantize_append_sparse_mla_cache,
        )
        if width == 528
        else (
            dsv41_fp4_quantize_pack_sparse_mla_cache,
            dsv41_fp4_quantize_append_sparse_mla_cache,
        )
    )
    nb, bs, span = 3, 32, 37
    latent = _latent(nb, bs, seed=pitch)
    rows = pack(latent, kv_layout="NHD", inline_scale=True)
    cache, backing = _packed_block_view(torch.full_like(rows, 0x5A), span, pitch=pitch)
    expected, expected_backing = _packed_block_view(rows, span, pitch=pitch)
    slots = torch.randperm(nb * bs, device="cuda")
    append(
        latent.reshape(-1, 512)[slots],
        _row_indices(slots, bs, span),
        cache.squeeze(2),
        inline_scale=True,
    )
    torch.cuda.synchronize()
    assert torch.equal(backing, expected_backing)


# --- Attention ----------------------------------------------------------------


def _caches(extra: str, main_bs: int, extra_bs: int, seed: int):
    main_latent = _latent(6, main_bs, seed)
    footer = dsv41_fp8_quantize_pack_sparse_mla_cache(main_latent, kv_layout="NHD")
    extra_footer = None
    if extra != "none":
        extra_latent = _latent(5, extra_bs, seed + 1)
        if extra == "fp4":
            extra_footer = dsv41_fp4_quantize_pack_sparse_mla_cache(
                extra_latent, kv_layout="NHD"
            )
        else:
            extra_footer = dsv41_fp8_quantize_pack_sparse_mla_cache(
                extra_latent, kv_layout="NHD"
            )
    return footer, extra_footer


def _inline_view(footer: torch.Tensor, fmt, span: int, pitch: int = 0) -> torch.Tensor:
    view, _ = _packed_block_view(_footer_to_inline(footer, *fmt), span, pitch=pitch)
    return view


@pytest.mark.parametrize("tokens", [4, 128])  # decode / prefill
@pytest.mark.parametrize("precision", ["default", "fp8", "bf16"])
@pytest.mark.parametrize("extra", ["none", "fp8", "fp4"])
@pytest.mark.parametrize("main_span_pad,extra_span_pad", [(0, 0), (5, 3)])
def test_inline_attention_matches_footer(
    tokens, precision, extra, main_span_pad, extra_span_pad, sm12x
):
    """The same records stored inline (optionally in packed blocks) give the
    footer-form output and LSE bit for bit on every DSV4_1 route."""
    heads, topk, extra_topk = 16, 128, 64
    main_bs, extra_bs = 32, 64
    footer, extra_footer = _caches(extra, main_bs, extra_bs, seed=tokens)
    torch.manual_seed(7 + tokens)
    q = (
        torch.randn(tokens, heads, 512, device="cuda", dtype=torch.bfloat16) / 10.0
    ).clamp(-1, 1)
    indices = torch.randint(
        0, 6 * main_bs, (tokens, topk), device="cuda", dtype=torch.int32
    )
    indices[0, -3:] = -1
    extra_indices = None
    if extra != "none":
        extra_indices = torch.randint(
            0, 5 * extra_bs, (tokens, extra_topk), device="cuda", dtype=torch.int32
        )
    fp4 = extra == "fp4"
    extra_fmt = _FP4 if fp4 else _FP8

    def run(scale_format, cache, extra_cache, idx, extra_idx):
        wrapper = flashinfer.mla.SparseMLASm120Wrapper(
            kv_scale_format=scale_format,
            extra_kv_fp4=fp4,
            compute_precision=precision,
            device=torch.device("cuda"),
        )
        out = torch.empty(tokens, heads, 512, device="cuda", dtype=torch.bfloat16)
        lse = wrapper.run(
            q,
            cache,
            idx,
            out,
            512**-0.5,
            extra_kv_cache=extra_cache,
            extra_indices=extra_idx,
            return_lse=True,
        )
        return out, lse

    main_span, extra_span = main_bs + main_span_pad, extra_bs + extra_span_pad
    try:
        ref_out, ref_lse = run(
            "ue8m0_g32", footer, extra_footer, indices, extra_indices
        )
    except ValueError as error:
        pytest.skip(f"no footer-form route for this configuration: {error}")
    out, lse = run(
        "ue8m0_g32_inline",
        _inline_view(footer, _FP8, main_span),
        None
        if extra_footer is None
        else _inline_view(extra_footer, extra_fmt, extra_span),
        _row_indices(indices, main_bs, main_span),
        None
        if extra_indices is None
        else _row_indices(extra_indices, extra_bs, extra_span),
    )
    torch.cuda.synchronize()
    assert torch.isfinite(ref_out).all()
    assert torch.equal(out, ref_out)
    assert torch.equal(lse, ref_lse)


@pytest.mark.parametrize("tokens", [4, 128])  # decode / prefill
@pytest.mark.parametrize("precision", ["default", "bf16"])
@pytest.mark.parametrize(
    "extra,main_pitch,extra_pitch",
    [
        ("fp8", 528, 544),
        ("fp8", 544, 1056),
        ("fp4", 528, 304),
        ("fp4", 1056, 576),
    ],
)
def test_inline_padded_row_pitches_match_footer(
    tokens, precision, extra, main_pitch, extra_pitch, sm12x
):
    """Both caches read rows at their own padded pitch (in packed blocks)."""
    heads, topk, extra_topk = 16, 128, 64
    main_bs, extra_bs = 32, 64
    footer, extra_footer = _caches(extra, main_bs, extra_bs, seed=31)
    torch.manual_seed(37 + tokens)
    q = (
        torch.randn(tokens, heads, 512, device="cuda", dtype=torch.bfloat16) / 10.0
    ).clamp(-1, 1)
    indices = torch.randint(
        0, 6 * main_bs, (tokens, topk), device="cuda", dtype=torch.int32
    )
    extra_indices = torch.randint(
        0, 5 * extra_bs, (tokens, extra_topk), device="cuda", dtype=torch.int32
    )
    fp4 = extra == "fp4"

    def run(scale_format, cache, extra_cache, idx, extra_idx):
        wrapper = flashinfer.mla.SparseMLASm120Wrapper(
            kv_scale_format=scale_format,
            extra_kv_fp4=fp4,
            compute_precision=precision,
            device=torch.device("cuda"),
        )
        out = torch.empty(tokens, heads, 512, device="cuda", dtype=torch.bfloat16)
        lse = wrapper.run(
            q,
            cache,
            idx,
            out,
            512**-0.5,
            extra_kv_cache=extra_cache,
            extra_indices=extra_idx,
            return_lse=True,
        )
        return out, lse

    main_span, extra_span = main_bs + 3, extra_bs + 2
    ref_out, ref_lse = run("ue8m0_g32", footer, extra_footer, indices, extra_indices)
    out, lse = run(
        "ue8m0_g32_inline",
        _inline_view(footer, _FP8, main_span, main_pitch),
        _inline_view(extra_footer, _FP4 if fp4 else _FP8, extra_span, extra_pitch),
        _row_indices(indices, main_bs, main_span),
        _row_indices(extra_indices, extra_bs, extra_span),
    )
    torch.cuda.synchronize()
    assert torch.equal(out, ref_out)
    assert torch.equal(lse, ref_lse)


def _metadata(scale_format, cache, extra_cache, indices, extra_indices, fp4):
    """Inspect a wrapper call's cache metadata: ``(model, AttentionMetadata)``."""
    from flashinfer.mla._sparse_mla_sm120._execution import resolve_model_type
    from flashinfer.mla._sparse_mla_sm120._prepared import validate_metadata

    q = torch.zeros(indices.shape[0], 16, 512, device="cuda", dtype=torch.bfloat16)
    model = resolve_model_type(512, scale_format)
    tensors = (q, cache, indices, torch.empty_like(q), None, None, extra_cache)
    tensors += (extra_indices, None, None, None, None)
    meta = validate_metadata(*tensors, model=model, is_dsv4_nvfp4=False, extra_fp4=fp4)
    return model, meta


def test_inline_formats_select_the_native_inline_model(sm12x):
    """The inline formats resolve to DSV4_1_INLINE on the cache's own pages,
    with the record width as the row stride."""
    from flashinfer.mla._sparse_mla_sm120._policy import (
        _MODEL_TYPE_DSV4_1,
        _MODEL_TYPE_DSV4_1_INLINE,
    )

    footer, extra_footer = _caches("fp4", 64, 64, seed=11)
    main = _inline_view(footer, _FP8, 67)
    extra = _inline_view(extra_footer, _FP4, 64)
    indices = torch.zeros(4, 128, device="cuda", dtype=torch.int32)
    extra_indices = torch.zeros(4, 64, device="cuda", dtype=torch.int32)
    model, meta = _metadata(
        "ue8m0_g32_inline", main, extra, indices, extra_indices, True
    )
    assert model == _MODEL_TYPE_DSV4_1_INLINE
    assert (meta.page_size, meta.extra_page_size) == (64, 64)
    assert meta.page_stride_bytes == 67 * 528 and meta.row_stride_bytes == 528
    assert meta.extra_row_stride_bytes == 288
    padded = _inline_view(extra_footer, _FP4, 64, 304)
    _, meta = _metadata("ue8m0_g32_inline", main, padded, indices, extra_indices, True)
    assert (meta.extra_row_stride_bytes, meta.extra_page_stride_bytes) == (
        304,
        64 * 304,
    )
    pitched = torch.zeros(64, 544, dtype=torch.uint8, device="cuda")[:, :528]
    _, meta = _metadata("ue8m0_g32_inline", pitched, None, indices, None, False)
    assert (meta.page_size, meta.row_stride_bytes) == (1, 544)
    model, _ = _metadata(
        "ue8m0_g32", footer, extra_footer, indices, extra_indices, True
    )
    assert model == _MODEL_TYPE_DSV4_1


@pytest.mark.parametrize("tokens", [4, 128])  # decode / prefill
@pytest.mark.parametrize("precision", ["default", "bf16"])
@pytest.mark.parametrize("extra", ["none", "fp4"])
def test_inline_one_row_pages_with_row_pitch_match_footer(
    tokens, precision, extra, sm12x
):
    """A ``[num_rows, 528]`` main cache with a padded row pitch is read with the
    pitch as its row stride."""
    heads, topk, extra_topk, bs = 16, 128, 64, 32
    footer, extra_footer = _caches(extra, bs, 64, seed=17)
    rows = _footer_to_inline(footer, *_FP8).reshape(-1, 528)
    pitched = torch.full((rows.shape[0], 544), 0x5A, dtype=torch.uint8, device="cuda")
    pitched[:, :528] = rows
    torch.manual_seed(tokens)
    q = (
        torch.randn(tokens, heads, 512, device="cuda", dtype=torch.bfloat16) / 10.0
    ).clamp(-1, 1)
    indices = torch.randint(
        0, rows.shape[0], (tokens, topk), device="cuda", dtype=torch.int32
    )
    extra_indices = (
        None
        if extra_footer is None
        else torch.randint(
            0, 5 * 64, (tokens, extra_topk), device="cuda", dtype=torch.int32
        )
    )

    def run(scale_format, cache, extra_cache):
        wrapper = flashinfer.mla.SparseMLASm120Wrapper(
            kv_scale_format=scale_format,
            extra_kv_fp4=extra == "fp4",
            compute_precision=precision,
            device=torch.device("cuda"),
        )
        out = torch.empty(tokens, heads, 512, device="cuda", dtype=torch.bfloat16)
        wrapper.run(
            q,
            cache,
            indices,
            out,
            512**-0.5,
            extra_kv_cache=extra_cache,
            extra_indices=extra_indices,
        )
        return out

    ref = run("ue8m0_g32", footer, extra_footer)
    out = run(
        "ue8m0_g32_inline",
        pitched[:, :528],
        None if extra_footer is None else _footer_to_inline(extra_footer, *_FP4),
    )
    torch.cuda.synchronize()
    assert torch.equal(out, ref)


@pytest.mark.parametrize("tokens", [4, 128])  # decode / prefill
@pytest.mark.parametrize("precision", ["default", "bf16"])
@pytest.mark.parametrize("extra", ["none", "fp8", "fp4"])
@pytest.mark.parametrize("extra_pitch_pad", [0, 16])
def test_inline_attention_is_cuda_graph_capturable(
    tokens, precision, extra, extra_pitch_pad, sm12x
):
    heads, topk, extra_topk, bs = 16, 128, 64, 64
    footer, extra_footer = _caches(extra, bs, 64, seed=23)
    cache = _footer_to_inline(footer, *_FP8)
    extra_fmt = _FP4 if extra == "fp4" else _FP8
    extra_cache = (
        None
        if extra_footer is None
        else _inline_view(extra_footer, extra_fmt, 64, sum(extra_fmt) + extra_pitch_pad)
    )
    torch.manual_seed(29)
    q = (
        torch.randn(tokens, heads, 512, device="cuda", dtype=torch.bfloat16) / 10.0
    ).clamp(-1, 1)
    indices = torch.randint(0, 6 * bs, (tokens, topk), device="cuda", dtype=torch.int32)
    extra_indices = (
        None
        if extra_cache is None
        else torch.randint(
            0, 5 * 64, (tokens, extra_topk), device="cuda", dtype=torch.int32
        )
    )
    wrapper = flashinfer.mla.SparseMLASm120Wrapper(
        kv_scale_format="ue8m0_g32_inline",
        extra_kv_fp4=extra == "fp4",
        compute_precision=precision,
        device=torch.device("cuda"),
    )
    out = torch.empty(tokens, heads, 512, device="cuda", dtype=torch.bfloat16)

    def call():
        wrapper.run(
            q,
            cache,
            indices,
            out,
            512**-0.5,
            extra_kv_cache=extra_cache,
            extra_indices=extra_indices,
        )

    call()  # warm up: JIT, plan and workspace outside capture
    torch.cuda.synchronize()
    eager = out.clone()
    out.zero_()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager)


@pytest.mark.parametrize("fp4", [False, True])
def test_dsv4_entry_inline_formats_match_footer(fp4, sm12x):
    """``trtllm_batch_decode_sparse_mla_dsv4`` reads the inline forms
    (``fp8_dsv41_inline`` / ``fp8_dsv41_inline_fp4_ca``) like their footer forms."""
    tokens, heads, topk, extra_topk = 8, 16, 128, 64
    main_bs, extra_bs = 64, 64
    footer, extra_footer = _caches("fp4" if fp4 else "fp8", main_bs, extra_bs, seed=3)
    torch.manual_seed(5)
    q = (
        torch.randn(tokens, 1, heads, 512, device="cuda", dtype=torch.bfloat16) / 10.0
    ).clamp(-1, 1)
    idx = torch.randint(
        0, 6 * main_bs, (tokens, topk), device="cuda", dtype=torch.int32
    )
    extra_idx = torch.randint(
        0, 5 * extra_bs, (tokens, extra_topk), device="cuda", dtype=torch.int32
    )
    lens = torch.full((tokens,), topk, dtype=torch.int32, device="cuda")
    extra_lens = torch.full((tokens,), extra_topk, dtype=torch.int32, device="cuda")

    def run(fmt, cache, extra_cache):
        return flashinfer.mla.trtllm_batch_decode_sparse_mla_dsv4(
            query=q,
            swa_kv_cache=cache,
            workspace_buffer=torch.empty(1, dtype=torch.int8, device="cuda"),
            sparse_indices=idx,
            compressed_kv_cache=extra_cache,
            swa_topk_lens=lens,
            extra_sparse_indices=extra_idx,
            extra_sparse_topk_lens=extra_lens,
            bmm1_scale=512**-0.5,
            kv_layout="NHD",
            kv_cache_format=fmt,
        )

    footer_fmt = "fp8_dsv41_fp4_ca" if fp4 else "fp8_dsv41"
    inline_fmt = "fp8_dsv41_inline_fp4_ca" if fp4 else "fp8_dsv41_inline"
    ref = run(footer_fmt, footer, extra_footer)
    out = run(
        inline_fmt,
        _footer_to_inline(footer, *_FP8),
        _footer_to_inline(extra_footer, *(_FP4 if fp4 else _FP8)),
    )
    torch.cuda.synchronize()
    assert torch.equal(out, ref)


def test_cake_rejects_inline_formats(sm12x):
    with pytest.raises(ValueError, match="inline"):
        flashinfer.mla.SparseMLASm120Wrapper(
            kv_scale_format="ue8m0_g32_inline",
            extra_kv_fp4=True,
            backend="cake",
            device=torch.device("cuda"),
        )
    q = torch.zeros(1, 1, 16, 512, device="cuda", dtype=torch.bfloat16)
    cache = torch.zeros(2, 64, 1, 528, device="cuda", dtype=torch.uint8)
    extra = torch.zeros(2, 64, 1, 288, device="cuda", dtype=torch.uint8)
    idx = torch.zeros(1, 64, device="cuda", dtype=torch.int32)
    lens = torch.full((1,), 64, device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError, match="inline"):
        flashinfer.mla.trtllm_batch_decode_sparse_mla_dsv4(
            query=q,
            swa_kv_cache=cache,
            workspace_buffer=torch.empty(1, dtype=torch.int8, device="cuda"),
            sparse_indices=idx,
            compressed_kv_cache=extra,
            swa_topk_lens=lens,
            extra_sparse_indices=idx,
            extra_sparse_topk_lens=lens,
            bmm1_scale=512**-0.5,
            kv_layout="NHD",
            kv_cache_format="fp8_dsv41_inline_fp4_ca",
            backend="cake",
        )
