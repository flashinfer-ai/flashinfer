"""Independent mega-MLA math and efficient unfused GPU baseline.

Cache fixtures use upstream packers plus independent decoders. These transforms
are deliberately outside the attention kernel and are also the performance
baseline: no eager Torch trigonometry/quantization in the measured region.
"""

import math
import torch
import pytest

triton = pytest.importorskip("triton")
import triton.language as tl  # noqa: E402


@triton.jit
def _rope(X, Y, Positions, CS, H: tl.constexpr):
    row = tl.program_id(0)
    d = tl.arange(0, 512)
    x = tl.load(X + row * 512 + d).to(tl.float32)
    other = tl.load(X + row * 512 + (d ^ 1)).to(tl.float32)
    pos = tl.load(Positions + row // H)
    r = (d - 448) // 2
    c = tl.load(CS + pos * 64 + r, d >= 448, other=1)
    s = tl.load(CS + pos * 64 + 32 + r, d >= 448, other=0)
    y = x * c + tl.where(d % 2 == 0, -other, other) * s
    tl.store(Y + row * 512 + d, y)


@triton.jit
def _epilogue(X, Y, SF, Positions, CS, H: tl.constexpr):
    row = tl.program_id(0)
    d = tl.arange(0, 512)
    x = tl.load(X + row * 512 + d).to(tl.float32)
    other = tl.load(X + row * 512 + (d ^ 1)).to(tl.float32)
    pos = tl.load(Positions + row // H)
    r = (d - 448) // 2
    c = tl.load(CS + pos * 64 + r, d >= 448, other=1)
    s = tl.load(CS + pos * 64 + 32 + r, d >= 448, other=0)
    y = x * c + tl.where(d % 2 == 0, other, -other) * s
    blocks = tl.reshape(y, (16, 32))
    amax = tl.maximum(tl.max(tl.abs(blocks), 1), 1.0e-4)
    # IEEE exponent rounding, also handles exact powers of two.
    scale = amax / 448.0
    bits = scale.to(tl.int32, bitcast=True)
    rounded = (bits + 0x7FFFFF) & 0x7F800000
    scale = rounded.to(tl.float32, bitcast=True)
    quant = blocks / scale[:, None]
    tl.store(Y + row * 512 + d, tl.reshape(quant, (512,)))
    tl.store(SF + row * 16 + tl.arange(0, 16), rounded >> 23)


def rope(q, positions, cs, out):
    _rope[(q.shape[0] * q.shape[1],)](
        q, out, positions, cs, q.shape[1], enable_fp_fusion=False
    )


def epilogue(context, positions, cs, out, scales):
    _epilogue[(context.shape[0] * context.shape[1],)](
        context, out, scales, positions, cs, context.shape[1], enable_fp_fusion=False
    )


def rope_reference(x, positions, cs, inverse=False):
    out = x.double().clone()
    pair = x[..., 448:].double().reshape(*x.shape[:-1], 32, 2)
    c = cs[positions.long(), :32].double()[:, None, :]
    s = cs[positions.long(), 32:].double()[:, None, :] * (-1 if inverse else 1)
    out[..., 448::2] = pair[..., 0] * c - pair[..., 1] * s
    out[..., 449::2] = pair[..., 0] * s + pair[..., 1] * c
    return out


def unpack_output(q, sf):
    scales = (sf.to(torch.int32) << 23).view(torch.float32)
    return (q.float().reshape(*q.shape[:-1], 16, 32) * scales[..., None]).reshape(
        q.shape
    )


def fixture(
    tokens=1,
    heads=64,
    cache_format="fp8",
    pool_tokens=4096,
    seed=42,
    ragged=False,
    *,
    topk=128,
    extra_topk=512,
    extra_page_size=64,
    extra_pool_tokens=None,
    input_scale=1.0,
    with_sink=True,
    keep_decoded=True,
):
    from flashinfer.mla import (
        SparseMLASm120Wrapper,
        nvfp4_quantize_pack_sparse_mla_cache,
        dsv41_fp8_quantize_pack_sparse_mla_cache,
        dsv41_fp4_quantize_pack_sparse_mla_cache,
    )
    from tests.attention.sparse_mla_test_utils import (
        quantize_kv_dsv4,
        dequantize_kv_dsv4,
        _dequantize_nvfp4_cache,
        dequantize_kv_dsv4_1,
        dequantize_kv_dsv4_1_fp4,
    )
    from tests.attention.sparse_mla_test_utils import require_sm12x

    require_sm12x()
    torch.manual_seed(seed)
    q = torch.randn(tokens, heads, 512, device="cuda", dtype=torch.bfloat16)
    if input_scale != 1.0:
        q = (q * input_scale).clamp(-1, 1)
    positions = torch.randint(0, 100001, (tokens,), device="cuda", dtype=torch.int32)
    freq = torch.arange(32, device="cuda").float() / 32
    angles = torch.arange(100001, device="cuda").float()[:, None] * (10000.0**-freq)
    cs = torch.cat((angles.cos(), angles.sin()), -1)
    caches, decoded = [], []
    pool_sizes = (pool_tokens, extra_pool_tokens or pool_tokens)
    for cache_index, page_size in enumerate((64, extra_page_size)):
        if cache_index == 1 and extra_topk == 0:
            caches.append(None)
            decoded.append(None)
            continue
        if pool_sizes[cache_index] % page_size:
            raise ValueError("pool size must be a multiple of page size")
        kv = torch.randn(
            pool_sizes[cache_index] // page_size,
            page_size,
            1,
            512,
            device="cuda",
            dtype=torch.bfloat16,
        )
        if input_scale != 1.0:
            kv = (kv * input_scale).clamp(-1, 1)
        if cache_format in ("v41_fp8", "v41_mixed"):
            fp4 = cache_format == "v41_mixed" and cache_index == 1
            pack = (
                dsv41_fp4_quantize_pack_sparse_mla_cache
                if fp4
                else dsv41_fp8_quantize_pack_sparse_mla_cache
            )
            if keep_decoded:
                footer = pack(kv.squeeze(2), kv_layout="NHD")
                dec = (dequantize_kv_dsv4_1_fp4 if fp4 else dequantize_kv_dsv4_1)(
                    footer
                )
            # The same packer emits FlashMLA/vLLM's inline record ABI.
            cache = pack(kv.squeeze(2), kv_layout="NHD", inline_scale=True)
        elif cache_format == "nvfp4":
            cache = nvfp4_quantize_pack_sparse_mla_cache(kv.squeeze(2), kv_layout="NHD")
            if keep_decoded:
                dec = _dequantize_nvfp4_cache(cache)
        else:
            cache = quantize_kv_dsv4(kv)
            if keep_decoded:
                dec = dequantize_kv_dsv4(cache)
        caches.append(cache)
        decoded.append(dec.reshape(-1, 512) if keep_decoded else None)
    indices = [
        torch.randint(n, (tokens, k), device="cuda", dtype=torch.int32) if k else None
        for n, k in zip(pool_sizes, (topk, extra_topk), strict=True)
    ]
    lengths = [
        torch.full((tokens,), k, device="cuda", dtype=torch.int32) if k else None
        for k in (topk, extra_topk)
    ]
    if ragged:
        for idx, lens in zip(indices, lengths, strict=True):
            if idx is None:
                continue
            lens.copy_(torch.arange(tokens, device="cuda") % (idx.shape[1] + 1))
            idx[:, ::7] = -1
    sink = torch.randn(heads, device="cuda") if with_sink else None
    rotated, context = torch.empty_like(q), torch.empty_like(q)
    out = torch.empty_like(q, dtype=torch.float8_e4m3fn)
    scales = torch.empty(tokens, heads, 16, device="cuda", dtype=torch.uint8)
    if cache_format in ("v41_fp8", "v41_mixed"):
        wrapper = SparseMLASm120Wrapper(
            backend="sparse",
            kv_scale_format="ue8m0_g32_inline",
            compute_precision="fp8",
            extra_kv_fp4=cache_format == "v41_mixed",
        )
    else:
        wrapper = SparseMLASm120Wrapper(kv_cache_format=cache_format, backend="sparse")

    def core():
        return wrapper.run(
            rotated,
            caches[0],
            indices[0],
            context,
            512**-0.5,
            topk_length=lengths[0],
            attn_sink=sink,
            extra_kv_cache=caches[1],
            extra_indices=indices[1],
            extra_topk_length=lengths[1],
            return_lse=True,
            lse_scale=math.log(2),
        )

    def prologue():
        rope(q, positions, cs, rotated)

    def post():
        epilogue(context, positions, cs, out, scales)

    def baseline():
        prologue()
        lse = core()
        post()
        return out, scales, lse

    return locals()


def reference(f):
    q = rope_reference(f["q"], f["positions"], f["cs"]).bfloat16().double()
    rows, valid = [], []
    for dec, idx, lens in zip(f["decoded"], f["indices"], f["lengths"], strict=True):
        if idx is None:
            continue
        rows.append(dec[idx.clamp_min(0).long()].double())
        valid.append(
            (idx >= 0)
            & (torch.arange(idx.shape[1], device="cuda")[None] < lens[:, None])
        )
    kv = torch.cat(rows, 1)
    valid = torch.cat(valid, 1)
    scores = (q @ kv.transpose(1, 2)) * 512**-0.5
    scores.masked_fill_(~valid[:, None, :], -torch.inf)
    lse = torch.logsumexp(scores, -1)
    if f["sink"] is not None:
        lse = torch.logaddexp(lse, f["sink"].double()[None])
    # An empty selection without a sink has LSE=-inf and zero context.
    # Avoid -inf - -inf while preserving every finite-row probability.
    normalization = torch.where(torch.isneginf(lse), torch.zeros_like(lse), lse)
    context = torch.exp(scores - normalization[..., None]) @ kv
    return rope_reference(context, f["positions"], f["cs"], inverse=True), lse


def fused_call(f):
    """Native fused kernels with the same prepared dispatch as the baseline."""
    return f["wrapper"].run_mega(
        f["q"],
        f["caches"][0],
        f["indices"][0],
        f["out"],
        512**-0.5,
        positions=f["positions"],
        cos_sin=f["cs"],
        output_scales=f["scales"],
        topk_length=f["lengths"][0],
        attn_sink=f["sink"],
        extra_kv_cache=f["caches"][1],
        extra_indices=f["indices"][1],
        extra_topk_length=f["lengths"][1],
        return_lse=True,
        lse_scale=math.log(2),
    )
