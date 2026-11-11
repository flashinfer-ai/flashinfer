"""Sparse prefill fixtures, independent math and unfused GPU transforms.

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
    # WARNING: match mega_rotate's explicit multiply-then-FMA order. CUDA 13.0
    # can contract ordinary mul/add despite enable_fp_fusion=False; choosing
    # a different product for FMA can change the subsequent BF16/FP8 rounding.
    y = tl.fma(x, c, tl.where(d % 2 == 0, -other, other) * s)
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
    # Preserve the same multiply-then-FMA order for inverse RoPE.
    y = tl.fma(x, c, tl.where(d % 2 == 0, other, -other) * s)
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
    inline=True,
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
            cache = pack(kv.squeeze(2), kv_layout="NHD", inline_scale=inline)
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
            kv_scale_format="ue8m0_g32_inline" if inline else "ue8m0_g32",
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


def prefill_fixture(
    requests=1,
    cache_format="v41_fp8",
    *,
    heads=64,
    prefill_tokens=65,
    prefill_cp_size=4,
    prefill_cp_rank=0,
    history_tokens=0,
    with_sink=True,
    keep_decoded=True,
    inline=True,
):
    """Interleaved query CP with the full gathered KV pool on each device.

    Each query shares its sparse candidate list across all heads. Requests own
    disjoint pools; the secondary pool has one compressed entry per two tokens.
    """
    if prefill_tokens < 1 or not 0 <= prefill_cp_rank < prefill_cp_size:
        raise ValueError("invalid local query size or CP rank")
    global_tokens = prefill_tokens * prefill_cp_size
    primary_origin = max(0, history_tokens - 127)
    primary_slots = (history_tokens + global_tokens - primary_origin + 63) // 64 * 64
    extra_slots = ((history_tokens + global_tokens) // 2 + 63) // 64 * 64
    f = fixture(
        requests * prefill_tokens,
        heads,
        cache_format,
        pool_tokens=requests * primary_slots,
        extra_pool_tokens=requests * extra_slots,
        keep_decoded=keep_decoded,
        inline=inline,
        with_sink=with_sink,
    )
    generator = torch.Generator().manual_seed(43)
    positions, primary, secondary, plen, slen = [], [], [], [], []
    for request in range(requests):
        for row in range(prefill_tokens):
            pos = history_tokens + prefill_cp_rank + row * prefill_cp_size
            positions.append(pos)
            ids = torch.arange(max(0, pos - 127), pos + 1, dtype=torch.int32)
            selected = torch.randperm((pos + 1) // 2, generator=generator)[:512]
            a = torch.full((128,), -1, dtype=torch.int32)
            b = torch.full((512,), -1, dtype=torch.int32)
            a[: ids.numel()] = request * primary_slots + ids - primary_origin
            b[: selected.numel()] = request * extra_slots + selected.to(torch.int32)
            primary.append(a)
            secondary.append(b)
            plen.append(ids.numel())
            slen.append(selected.numel())
    f["positions"].copy_(torch.tensor(positions, dtype=torch.int32))
    for i, (ids, lengths) in enumerate(((primary, plen), (secondary, slen))):
        f["indices"][i].copy_(torch.stack(ids))
        f["lengths"][i].copy_(torch.tensor(lengths, dtype=torch.int32))
    return f


def independent_reference(f, rows=None):
    """FP64 attention over independently decoded cache; robust to empty selections."""
    if rows is None:
        rows = slice(None)
    q = rope_reference(f["q"][rows], f["positions"][rows], f["cs"]).bfloat16().double()
    selected, valid = [], []
    for decoded, indices, lengths in zip(
        f["decoded"], f["indices"], f["lengths"], strict=True
    ):
        if indices is None:
            continue
        if decoded is None:
            raise ValueError("independent reference requires keep_decoded=True")
        idx = indices[rows]
        lens = (
            lengths[rows]
            if lengths is not None
            else torch.full((idx.shape[0],), idx.shape[1], device=idx.device)
        )
        selected.append(decoded[idx.clamp_min(0).long()].double())
        valid.append(
            (idx >= 0)
            & (torch.arange(idx.shape[1], device=idx.device)[None] < lens[:, None])
        )
    kv, mask = torch.cat(selected, dim=1), torch.cat(valid, dim=1)
    scores = (q @ kv.transpose(1, 2)) * 512**-0.5
    scores.masked_fill_(~mask[:, None, :], -torch.inf)
    lse = torch.logsumexp(scores, dim=-1)
    if f["sink"] is not None:
        lse = torch.logaddexp(lse, f["sink"].double()[None])
    probabilities = torch.exp(scores - lse[..., None])
    probabilities = torch.where(torch.isfinite(lse[..., None]), probabilities, 0)
    context = probabilities @ kv
    return rope_reference(context, f["positions"][rows], f["cs"], inverse=True), lse


def numerical_metrics(f, lse, rows=None):
    expected, expected_lse = independent_reference(f, rows)
    actual = unpack_output(f["out"], f["scales"])
    if rows is not None:
        actual, lse = actual[rows], lse[rows]
    relative_l2 = (
        (
            (actual.double() - expected).square().sum()
            / expected.square().sum().clamp_min(1e-30)
        )
        .sqrt()
        .item()
    )
    finite = torch.isfinite(expected_lse)
    assert torch.equal(torch.isfinite(lse), finite)
    return dict(
        relative_l2=relative_l2,
        lse_max_abs=(lse.double()[finite] - expected_lse[finite]).abs().max().item()
        if finite.any()
        else 0.0,
        finite_output=bool(torch.isfinite(actual).all()),
    )


def fused_call(f, *, out_lse=None):
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
        out_lse=out_lse,
    )


def poison_mega_destinations(out, scales, lse, expected_scales):
    """Make every missing output write observable, including after graph capture."""
    out.fill_(float("nan"))
    lse.fill_(float("nan"))
    scales.copy_(expected_scales.bitwise_xor(0xFF))


def assert_mega_outputs(actual, expected):
    """Compare separate-transform and fused results without hiding missing writes."""
    out, scales, lse = actual
    expected_out, expected_scales, expected_lse = expected
    assert torch.isfinite(out.float()).all()
    assert not torch.isnan(lse).any()
    torch.testing.assert_close(lse, expected_lse, atol=2e-6, rtol=1e-6)
    assert (out.float() != expected_out.float()).float().mean().item() < 0.001
    # E4M3 spacing is at most 1/8 of a normal value; subnormal spacing is
    # 2**-9. Bound the size as well as the count of rounding disagreements.
    # Exact scale equality below makes this a bound in common output units.
    torch.testing.assert_close(
        out.float(), expected_out.float(), atol=2**-9, rtol=0.125
    )
    torch.testing.assert_close(scales, expected_scales, atol=0, rtol=0)


def decode_fixture(requests=1, *, heads=64, history_tokens=32768, seed=42, inline=True):
    """Seven verification rows/request, shifted SWA and independent causal top-512.

    Requests own disjoint cache pools. The compressed history uses ratio two.
    Cache packing and index generation are outside the measured region.
    """
    primary_stride = 192  # 128-entry window plus six successive verification rows.
    extra_stride = ((history_tokens + 7) // 2 + 63) // 64 * 64
    f = fixture(
        requests * 7,
        heads,
        "v41_fp8",
        pool_tokens=requests * primary_stride,
        extra_pool_tokens=requests * extra_stride,
        seed=seed,
        inline=inline,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    primary = torch.empty(requests * 7, 128, dtype=torch.int32)
    extra = torch.empty(requests * 7, 512, dtype=torch.int32)
    positions = torch.empty(requests * 7, dtype=torch.int32)
    for request in range(requests):
        for row in range(7):
            t = request * 7 + row
            positions[t] = history_tokens + row
            primary[t] = request * primary_stride + torch.arange(row, row + 128)
            available = (history_tokens + row + 1) // 2
            extra[t] = (
                request * extra_stride
                + torch.randperm(available, generator=generator)[:512]
            )
    f["positions"].copy_(positions)
    f["indices"][0].copy_(primary)
    f["indices"][1].copy_(extra)
    return f


def decode_call(f, impl, mega=True):
    options = dict(
        topk_length=f["lengths"][0],
        attn_sink=f["sink"],
        extra_kv_cache=f["caches"][1],
        extra_indices=f["indices"][1],
        extra_topk_length=f["lengths"][1],
        return_lse=True,
        lse_scale=math.log(2),
        impl=impl,
    )
    if mega:
        return f["wrapper"].run_mega(
            f["q"],
            f["caches"][0],
            f["indices"][0],
            f["out"],
            512**-0.5,
            positions=f["positions"],
            cos_sin=f["cs"],
            output_scales=f["scales"],
            **options,
        )
    f["prologue"]()
    lse = f["wrapper"].run(
        f["rotated"],
        f["caches"][0],
        f["indices"][0],
        f["context"],
        512**-0.5,
        **options,
    )
    epilogue(f["context"], f["positions"], f["cs"], f["out"], f["scales"])
    return lse


def native_prefill_calls(f, module=None):
    from flashinfer.jit.mla import gen_sparse_mla_sm120_module

    module = module or gen_sparse_mla_sm120_module().build_and_load()
    q, caches, indices, lengths = f["q"], f["caches"], f["indices"], f["lengths"]
    device = torch.cuda.get_device_properties(q.device)
    plan = module.resolve_dsv4_nvfp4(
        q.shape[0],
        q.shape[1],
        indices[0].shape[1],
        indices[1].shape[1],
        caches[0].shape[1],
        caches[1].shape[1],
        caches[0].stride(0),
        caches[1].stride(0),
        0,
        device.multi_processor_count,
        device.shared_memory_per_block_optin,
        True,
        False,
        lengths[0] is not None,
        lengths[1] is not None,
        f["sink"] is not None,
    )
    lse = torch.empty(q.shape[:2], device=q.device, dtype=torch.float32)
    mid, mlse = f["context"], f["context"].view(torch.float32)

    def arguments(query):
        return (
            plan,
            query,
            caches[0],
            indices[0],
            mid,
            mlse,
            f["context"],
            lse,
            512**-0.5,
            lengths[0],
            f["sink"],
            caches[1],
            indices[1],
            lengths[1],
            math.log(2),
        )

    def mega():
        module.dsv4_nvfp4_execute_attention_rope_fp8(
            *arguments(q),
            f["positions"],
            f["cs"],
            f["out"].view(torch.uint8),
            f["scales"],
        )
        return lse

    def baseline():
        f["prologue"]()
        module.dsv4_nvfp4_execute_attention(*arguments(f["rotated"]))
        f["post"]()
        return lse

    return mega, baseline, lse
