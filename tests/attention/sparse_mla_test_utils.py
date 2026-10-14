"""Sparse-MLA reference and input helpers without import-time device probing."""

import math

import pytest
import torch
import flashinfer


def require_sm12x():
    from flashinfer.utils import is_sm12x_supported

    if not torch.cuda.is_available() or not is_sm12x_supported(torch.device("cuda")):
        pytest.skip("Sparse-MLA SM120 requires SM12x.")


@pytest.fixture
def sm120_module():
    from flashinfer.jit.core import current_compilation_context
    from flashinfer.jit.cpp_ext import is_cuda_version_at_least

    if not any(
        major == 12 for major, _ in current_compilation_context.TARGET_CUDA_ARCHS
    ):
        pytest.skip("Sparse-MLA compiled queries require an SM12x compilation target.")
    if not is_cuda_version_at_least("12.9"):
        pytest.skip("Sparse-MLA compiled queries require CUDA >= 12.9.")


@pytest.fixture
def ordinary_format_facts(monkeypatch):
    from flashinfer.mla._sparse_mla_sm120 import _calibration

    facts = {
        model: dict(
            max_heads=128,
            min_topk=513 if model == 4 else 1,
            bytes_per_token=width,
            fp4_bytes_per_token=288,
        )
        for model, width in enumerate((656, 584, 656, 528, 1160, 528))
    }
    monkeypatch.setattr(_calibration, "format_info", facts.__getitem__)
    return facts


# Quantization helpers.


def _cast_scale_inv_to_ue8m0(scales_inv: torch.Tensor) -> torch.Tensor:
    """Round inverse scale to the nearest power-of-2 (FlashMLA convention)."""
    return torch.pow(2, torch.clamp_min(scales_inv, 1e-4).log2().ceil())


def _fp32_to_ue8m0_bytes(scale_fp32: torch.Tensor) -> torch.Tensor:
    """Extract the IEEE-754 exponent byte of an FP32 power-of-2 scale."""
    bits = scale_fp32.to(torch.float32).view(torch.int32)
    return ((bits >> 23) & 0xFF).to(torch.uint8)


def _quantize_kv_footer(
    kv_bf16: torch.Tensor,
    d_nope: int,
    d_rope: int,
    tile_size: int,
    scale_bytes: int,
) -> torch.Tensor:
    """Pack bf16 KV into an FP8 FOOTER-scale layout.

    Shared by DSv4 (448/64, tile 64, 7 scales + 1 pad byte) and DOTS3_SWA
    (1024/64, tile 128, 8 scales, no pad). Layout per block of ``bs`` tokens:
    ``[bs * (d_nope + d_rope*2) data | bs * scale_bytes footer]``.
    """
    num_tiles = d_nope // tile_size
    assert num_tiles * tile_size == d_nope
    assert scale_bytes >= num_tiles
    data_stride = d_nope + d_rope * 2
    bpt = data_stride + scale_bytes
    nb, bs, hk, d = kv_bf16.shape
    assert d == d_nope + d_rope and hk == 1
    kv = kv_bf16.squeeze(2)

    block_bytes = bs * bpt
    result_flat = torch.zeros(nb, block_bytes, dtype=torch.uint8, device=kv.device)

    for ti in range(num_tiles):
        tile = kv[..., ti * tile_size : (ti + 1) * tile_size].float()
        amax = tile.abs().amax(dim=-1).clamp(min=1e-4)
        scale = _cast_scale_inv_to_ue8m0(amax / 448.0)
        fp8 = (tile / scale.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
        ue8m0 = _fp32_to_ue8m0_bytes(scale)

        for tok in range(bs):
            data_off = tok * data_stride + ti * tile_size
            result_flat[:, data_off : data_off + tile_size] = fp8[:, tok].view(
                torch.uint8
            )
            scale_off = bs * data_stride + tok * scale_bytes + ti
            result_flat[:, scale_off] = ue8m0[:, tok]

    rope = kv[..., d_nope:].to(torch.bfloat16).contiguous().view(torch.uint8)
    rope = rope.reshape(nb, bs, d_rope * 2)
    for tok in range(bs):
        rope_off = tok * data_stride + d_nope
        result_flat[:, rope_off : rope_off + d_rope * 2] = rope[:, tok]

    return result_flat.view(nb, bs, 1, bpt)


def quantize_kv_dsv4(kv_bf16: torch.Tensor) -> torch.Tensor:
    """Pack bf16 KV into DSv4 FP8 FOOTER format."""
    return _quantize_kv_footer(kv_bf16, 448, 64, 64, 8)


def quantize_kv_dots3_swa(kv_bf16: torch.Tensor) -> torch.Tensor:
    """Pack bf16 KV into DOTS3_SWA FP8 FOOTER format (1160 B/token)."""
    return _quantize_kv_footer(kv_bf16, 1024, 64, 128, 8)


def _dequantize_kv_footer(
    packed: torch.Tensor,
    d_nope: int,
    d_rope: int,
    tile_size: int,
    scale_bytes: int,
) -> torch.Tensor:
    """Unpack an FP8 FOOTER layout → bf16. Inverse of :func:`_quantize_kv_footer`."""
    num_tiles = d_nope // tile_size
    data_stride = d_nope + d_rope * 2
    bpt = data_stride + scale_bytes
    d_qk = d_nope + d_rope
    nb, bs, _, _ = packed.shape
    result = torch.zeros(nb, bs, d_qk, dtype=torch.bfloat16, device=packed.device)
    p = packed.view(nb, bs * bpt)

    for tok in range(bs):
        data_off = tok * data_stride
        scale_off = bs * data_stride + tok * scale_bytes
        for ti in range(num_tiles):
            fp8_off = data_off + ti * tile_size
            fp8 = p[:, fp8_off : fp8_off + tile_size].view(torch.float8_e4m3fn).float()
            ue8m0 = p[:, scale_off + ti]
            scale = torch.pow(2.0, ue8m0.float() - 127.0)
            result[:, tok, ti * tile_size : (ti + 1) * tile_size] = (
                fp8 * scale.unsqueeze(-1)
            ).to(torch.bfloat16)
        rope_off = data_off + d_nope
        rope_bytes = p[:, rope_off : rope_off + d_rope * 2].contiguous()
        result[:, tok, d_nope:] = rope_bytes.view(torch.bfloat16).reshape(nb, d_rope)

    return result.view(nb, bs, 1, d_qk)


def dequantize_kv_dsv4(packed: torch.Tensor) -> torch.Tensor:
    """Unpack DSV4 FP8 FOOTER → bf16. Inverse of :func:`quantize_kv_dsv4`."""
    return _dequantize_kv_footer(packed, 448, 64, 64, 8)


def dequantize_kv_dots3_swa(packed: torch.Tensor) -> torch.Tensor:
    """Unpack DOTS3_SWA FP8 FOOTER → bf16. Inverse of :func:`quantize_kv_dots3_swa`."""
    return _dequantize_kv_footer(packed, 1024, 64, 128, 8)


def quantize_kv_dsv4_1(kv_bf16: torch.Tensor) -> torch.Tensor:
    """Pack bf16 KV into DeepSeek-V4.1 FP8 FOOTER format (528 B/token):
    512 B all-FP8 data (rope lanes included, no BF16 segment) + 16 B footer
    of 16 UE8M0 scales over 32-wide groups."""
    return _quantize_kv_footer(kv_bf16, 512, 0, 32, 16)


def dequantize_kv_dsv4_1(packed: torch.Tensor) -> torch.Tensor:
    """Unpack DSV4_1 FP8 FOOTER → bf16. Inverse of :func:`quantize_kv_dsv4_1`."""
    return _dequantize_kv_footer(packed, 512, 0, 32, 16)


# DeepSeek-V4.1 FP4 (V41_FP4) pack — FlashMLA tests/quant.py trajectory.

_E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _quantize_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Round fp32 to e2m1 codes (uint8) with cvt.rn.satfinite.e2m1x2.f32
    semantics: ties to even, saturate to +-6, NaN -> code 0."""
    mags = torch.tensor(_E2M1_MAGNITUDES, device=x.device)
    sign = torch.signbit(x).to(torch.uint8) << 3
    a = torch.nan_to_num(x.float().abs(), nan=0.0, posinf=6.0).clamp_max(6.0)
    mids = (mags[:-1] + mags[1:]) / 2
    code = torch.bucketize(a, mids, right=True)
    on_tie = (a.unsqueeze(-1) == mids).any(dim=-1)
    tie_code = torch.bucketize(a, mids, right=False)
    code = torch.where(on_tie, tie_code + (tie_code & 1), code)
    return sign | code.to(torch.uint8)


def quantize_kv_dsv4_1_fp4(kv_bf16: torch.Tensor) -> torch.Tensor:
    """Pack bf16 KV into DeepSeek-V4.1 FP4 format (288 B/token): 256 B packed
    E2M1 covering all 512 dims (rope included; even index in the low nibble)
    plus a 32 B E4M3 scale row (scale = amax/6 over 16-wide groups). A page
    block stores pbs data rows followed by pbs scale rows."""
    nb, bs, hk, d = kv_bf16.shape
    assert hk == 1 and d == 512
    xt = kv_bf16.reshape(nb, bs, d).float().view(nb, bs, 32, 16)
    amax = torch.nan_to_num(xt.abs(), nan=float("inf")).amax(dim=-1)
    scale = torch.clamp(amax / 6.0, 2.0**-9, 448.0).to(torch.float8_e4m3fn)
    scale = torch.where(torch.isinf(amax), torch.full_like(scale, float("nan")), scale)
    codes = _quantize_e2m1(xt / scale.float().unsqueeze(-1)).view(nb, bs, d)
    packed = codes[..., 0::2] | (codes[..., 1::2] << 4)  # [nb, bs, 256]

    bpt = 288
    result = torch.empty(nb, bs * bpt, dtype=torch.uint8, device=kv_bf16.device)
    result[:, : bs * 256] = packed.reshape(nb, bs * 256)
    result[:, bs * 256 :] = scale.view(torch.uint8).reshape(nb, bs * 32)
    return result.view(nb, bs, 1, bpt)


def dequantize_kv_dsv4_1_fp4(packed: torch.Tensor) -> torch.Tensor:
    """Unpack V41_FP4 → bf16. Exact: e2m1 x e4m3 has at most 6 significant
    bits, so the product is exact in bf16."""
    nb, bs, _, bpt = packed.shape
    assert bpt == 288
    p = packed.view(nb, bs * bpt)
    data = p[:, : bs * 256].view(nb, bs, 256)
    scales = p[:, bs * 256 :].view(nb, bs, 32).view(torch.float8_e4m3fn).float()
    codes = torch.empty(nb, bs, 512, dtype=torch.uint8, device=packed.device)
    codes[..., 0::2] = data & 0xF
    codes[..., 1::2] = data >> 4
    mags = torch.tensor(_E2M1_MAGNITUDES, device=packed.device)
    vals = mags[(codes & 7).long()]
    vals = torch.where((codes & 8) != 0, -vals, vals)
    out = vals.view(nb, bs, 32, 16) * scales.unsqueeze(-1)
    return out.reshape(nb, bs, 1, 512).to(torch.bfloat16)


# DSv3.2 INLINE pack.


def quantize_kv_dsv3_2(kv_bf16: torch.Tensor) -> torch.Tensor:
    """Pack bf16 KV into DSv3.2 FP8 INLINE format."""
    d_nope, d_rope, tile_size, num_tiles = 512, 64, 128, 4
    scale_bytes = num_tiles * 4  # 16
    bpt = d_nope + scale_bytes + d_rope * 2  # 656
    nb, bs, hk, d = kv_bf16.shape
    assert d == d_nope + d_rope and hk == 1
    nt = nb * bs  # total token count across all blocks
    kv = kv_bf16.reshape(nt, d)

    result = torch.zeros(nt, bpt, dtype=torch.uint8, device=kv.device)

    for ti in range(num_tiles):
        tile = kv[:, ti * tile_size : (ti + 1) * tile_size].float()
        amax = tile.abs().amax(dim=-1).clamp(min=1e-4)
        scale = _cast_scale_inv_to_ue8m0(amax / 448.0)  # power-of-2 FP32
        fp8 = (tile / scale.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
        result[:, ti * tile_size : (ti + 1) * tile_size] = fp8.view(torch.uint8)
        result[:, d_nope + ti * 4 : d_nope + (ti + 1) * 4] = (
            scale.view(torch.float32).view(torch.uint8).view(nt, 4)
        )

    rope = kv[:, d_nope:].to(torch.bfloat16).contiguous().view(torch.uint8)
    result[:, d_nope + scale_bytes :] = rope.view(nt, d_rope * 2)
    return result.view(nb, bs, 1, bpt)


def quantize_kv_glm_nsa(kv_bf16: torch.Tensor) -> torch.Tensor:
    """Pack bf16 KV into the 656B inline layout with arbitrary FP32 scales."""
    d_nope, d_rope, tile_size, num_tiles = 512, 64, 128, 4
    scale_bytes = num_tiles * 4
    bpt = d_nope + scale_bytes + d_rope * 2
    nb, bs, hk, d = kv_bf16.shape
    assert d == d_nope + d_rope and hk == 1
    nt = nb * bs
    kv = kv_bf16.reshape(nt, d)
    result = torch.zeros(nt, bpt, dtype=torch.uint8, device=kv.device)

    for ti in range(num_tiles):
        tile = kv[:, ti * tile_size : (ti + 1) * tile_size].float()
        scale = (tile.abs().amax(dim=-1).clamp(min=1e-4) / 448.0).to(torch.float32)
        fp8 = (tile / scale.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
        result[:, ti * tile_size : (ti + 1) * tile_size] = fp8.view(torch.uint8)
        result[:, d_nope + ti * 4 : d_nope + (ti + 1) * 4] = (
            scale.view(torch.float32).view(torch.uint8).view(nt, 4)
        )

    rope = kv[:, d_nope:].to(torch.bfloat16).contiguous().view(torch.uint8)
    result[:, d_nope + scale_bytes :] = rope.view(nt, d_rope * 2)
    return result.view(nb, bs, 1, bpt)


def quantize_kv_glm53_nope(kv_bf16: torch.Tensor) -> torch.Tensor:
    """Pack native NoPE KV into the 656B ABI with arbitrary FP32 scales."""
    d_nope, tile_size, num_tiles = 512, 128, 4
    bpt = 656
    nb, bs, hk, d = kv_bf16.shape
    assert d == d_nope and hk == 1
    nt = nb * bs
    kv = kv_bf16.reshape(nt, d)
    result = torch.zeros(nt, bpt, dtype=torch.uint8, device=kv.device)

    for ti in range(num_tiles):
        tile = kv[:, ti * tile_size : (ti + 1) * tile_size].float()
        scale = (tile.abs().amax(dim=-1).clamp(min=1e-4) / 448.0).to(torch.float32)
        fp8 = (tile / scale.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
        result[:, ti * tile_size : (ti + 1) * tile_size] = fp8.view(torch.uint8)
        result[:, d_nope + ti * 4 : d_nope + (ti + 1) * 4] = (
            scale.view(torch.float32).view(torch.uint8).view(nt, 4)
        )

    # Bytes 528:656 are reserved padding in the stable packed-cache ABI.
    return result.view(nb, bs, 1, bpt)


def _assert_has_non_pow2_inline_scales(packed: torch.Tensor) -> None:
    scales = packed.reshape(-1, 656)[:, 512:528].contiguous().view(torch.float32)
    log2_scales = scales.float().log2()
    assert torch.any((log2_scales - log2_scales.round()).abs() > 1e-3)


def dequantize_kv_dsv3_2(packed: torch.Tensor) -> torch.Tensor:
    """Unpack DSv3.2 FP8 INLINE → bf16. Inverse of :func:`quantize_kv_dsv3_2`."""
    d_nope, d_rope, tile_size, num_tiles = 512, 64, 128, 4
    scale_bytes = num_tiles * 4
    nb, bs, _, _ = packed.shape
    nt = nb * bs
    p = packed.reshape(nt, -1)

    result = torch.zeros(nt, d_nope + d_rope, dtype=torch.bfloat16, device=p.device)
    for ti in range(num_tiles):
        fp8 = (
            p[:, ti * tile_size : (ti + 1) * tile_size]
            .view(torch.float8_e4m3fn)
            .float()
        )
        scale = (
            p[:, d_nope + ti * 4 : d_nope + (ti + 1) * 4]
            .contiguous()
            .view(torch.float32)
            .squeeze(-1)
        )
        result[:, ti * tile_size : (ti + 1) * tile_size] = (
            fp8 * scale.unsqueeze(-1)
        ).to(torch.bfloat16)
    rope_bytes = p[:, d_nope + scale_bytes :].contiguous()
    result[:, d_nope:] = rope_bytes.view(torch.bfloat16).reshape(nt, d_rope)
    return result.view(nb, bs, 1, d_nope + d_rope)


# PyTorch SDPA reference.


def _ref_sparse_attn(
    q: torch.Tensor,
    kv_dequant: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int,
    attn_sink: torch.Tensor | None = None,
    topk_length: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense SDPA over sparse-gathered KV."""
    num_tokens, num_heads, d_qk = q.shape
    topk = indices.shape[-1]

    kv_flat = kv_dequant.view(-1, d_qk).float()
    q_f = q.float()

    idx_fixed = indices.clamp(min=0)
    invalid = indices < 0
    if topk_length is not None:
        ar = torch.arange(topk, device=q.device).unsqueeze(0)
        invalid = invalid | (ar >= topk_length.unsqueeze(-1))

    gathered = kv_flat.index_select(0, idx_fixed.view(-1)).view(num_tokens, topk, d_qk)
    P = torch.einsum("thd,tkd->thk", q_f, gathered) * sm_scale
    P[invalid.unsqueeze(1).expand_as(P)] = float("-inf")

    lse_e = torch.logsumexp(P, dim=-1)
    lse_safe = lse_e.clone()
    lse_safe[lse_safe == float("-inf")] = float("+inf")
    weights = torch.exp(P - lse_safe.unsqueeze(-1))
    out_f = torch.einsum("thk,tkd->thd", weights, gathered[..., :d_v])

    LN2 = float(torch.log(torch.tensor(2.0)).item())
    lse_log2 = lse_e / LN2

    if attn_sink is not None:
        sink = attn_sink.float()
        factor = torch.sigmoid(lse_e.float() - sink.unsqueeze(0))
        out_f = out_f * factor.unsqueeze(-1)
        lse_log2 = torch.logaddexp(lse_e, sink.unsqueeze(0)) / LN2

    return out_f.to(torch.bfloat16), lse_log2


def _make_decode_scratch(
    num_tokens: int,
    num_heads: int,
    topk: int,
    d_v: int,
    device: torch.device,
    *,
    extra_topk: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    # BI is model-dependent: the DeepSeek family consumes 64 candidates per
    # iteration, DOTS3_SWA 32. num_splits must be derived with the same value
    # the kernel uses or the grid covers only part of each candidate list.
    bi = 32 if d_v == 1024 else 64
    num_splits = (topk + bi - 1) // bi + (extra_topk + bi - 1) // bi
    # The runtime-H decode kernels HPB-align the scratch head dim (the
    # dedicated num_heads=8 instantiation keeps the true count).
    from flashinfer.mla._sparse_mla_sm120._policy import _decode_scratch_heads

    scratch_heads = _decode_scratch_heads(num_heads)
    return (
        torch.empty(
            (num_tokens, scratch_heads, num_splits, d_v),
            dtype=torch.bfloat16,
            device=device,
        ),
        torch.empty(
            (num_tokens, scratch_heads, num_splits),
            dtype=torch.float32,
            device=device,
        ),
    )


def inputs(heads=13, main_pbs=61, extra_pbs=53, mixed=True):
    import pytest
    from flashinfer.utils import is_sm12x_supported

    if not torch.cuda.is_available() or not is_sm12x_supported(torch.device("cuda")):
        pytest.skip("Sparse-MLA SM120 requires SM12x.")
    torch.manual_seed(173)
    q = torch.randn(4, heads, 512, device="cuda", dtype=torch.bfloat16) * 0.3
    main = quantize_kv_dsv4_1(
        torch.randn(4, main_pbs, 1, 512, device="cuda", dtype=torch.bfloat16) * 0.3
    )
    extra = (quantize_kv_dsv4_1_fp4 if mixed else quantize_kv_dsv4_1)(
        torch.randn(4, extra_pbs, 1, 512, device="cuda", dtype=torch.bfloat16) * 0.3
    )
    virtual = torch.cat(
        [
            dequantize_kv_dsv4_1(main).reshape(-1, 512),
            (dequantize_kv_dsv4_1_fp4 if mixed else dequantize_kv_dsv4_1)(
                extra
            ).reshape(-1, 512),
        ]
    ).reshape(-1, 1, 1, 512)

    def pitched(cache):
        n, p, _, b = cache.shape
        storage = torch.empty(
            n, ((p * b + 511) // 512 + 1) * 512, device="cuda", dtype=torch.uint8
        )
        result = storage[:, : p * b].view_as(cache)
        result.copy_(cache)
        return result

    main, extra = pitched(main), pitched(extra)
    idx = torch.randint(4 * main_pbs, (4, 133), device="cuda", dtype=torch.int32)[
        :, :128
    ]
    exidx = torch.randint(4 * extra_pbs, (4, 82), device="cuda", dtype=torch.int32)[
        :, :77
    ]
    idx[:, 4:32] = -1
    exidx[:, 32:64] = 3
    lens = torch.tensor([0, 69, 0, 128], device="cuda", dtype=torch.int32)
    exlens = torch.tensor([75, 0, 0, 77], device="cuda", dtype=torch.int32)
    sink = torch.randn(heads, device="cuda")
    mi = idx.masked_fill(torch.arange(128, device="cuda")[None] >= lens[:, None], -1)
    ei = exidx.masked_fill(torch.arange(77, device="cuda")[None] >= exlens[:, None], -1)
    ref = _ref_sparse_attn(
        q,
        virtual,
        torch.cat([mi, torch.where(ei < 0, ei, ei + 4 * main_pbs)], -1),
        512**-0.5,
        512,
        attn_sink=sink,
    )
    kwargs = dict(
        topk_length=lens,
        attn_sink=sink,
        extra_kv_cache=extra,
        extra_indices=exidx,
        extra_topk_length=exlens,
    )
    return q, main, idx, kwargs, ref


# DSV4 NVFP4 references (distinct from DSV4.1 FP4 arithmetic).


_D_NOPE = 448
_D_ROPE = 64
_PACKED_NOPE_BYTES = 224
_ROPE_BYTES = 128
_DATA_BYTES_PER_TOKEN = 352
_SCALE_BYTES_PER_TOKEN = 32
_BYTES_PER_TOKEN = 384


def _split_cache(cache: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if cache.shape[1] == 1:
        page_size = cache.shape[2]
    else:
        page_size = cache.shape[1]
    flat = cache.reshape(cache.shape[0], page_size * _BYTES_PER_TOKEN)
    data = flat[:, : page_size * _DATA_BYTES_PER_TOKEN].reshape(
        cache.shape[0], page_size, _DATA_BYTES_PER_TOKEN
    )
    scales = flat[:, page_size * _DATA_BYTES_PER_TOKEN :].reshape(
        cache.shape[0], page_size, _SCALE_BYTES_PER_TOKEN
    )
    return data, scales


def _reference_rows(latent_kv: torch.Tensor) -> tuple[torch.Tensor, ...]:
    rows = latent_kv.reshape(-1, _D_NOPE + _D_ROPE)
    global_scale = torch.ones(1, dtype=torch.float32, device=latent_kv.device)
    packed, scales = flashinfer.nvfp4_kv_quantize(
        rows[:, :_D_NOPE].contiguous(), global_scale
    )
    rope = (
        rows[:, _D_NOPE:]
        .contiguous()
        .view(torch.uint8)
        .reshape(rows.shape[0], _ROPE_BYTES)
    )
    return packed, scales.view(torch.uint8), rope


def _dequantize_linear_nvfp4(
    packed: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    lut = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        dtype=torch.float32,
        device=packed.device,
    )
    codes = torch.stack((packed & 0xF, packed >> 4), dim=-1).reshape(
        packed.shape[0], -1
    )
    values = lut[codes.long()]
    scale_values = scales.view(torch.float8_e4m3fn).float()
    return values * scale_values.repeat_interleave(16, dim=-1)


def _dequantize_nvfp4_cache(cache: torch.Tensor) -> torch.Tensor:
    data, scales = _split_cache(cache)
    num_pages, page_size = data.shape[:2]
    nope = _dequantize_linear_nvfp4(
        data[..., :_PACKED_NOPE_BYTES].reshape(-1, _PACKED_NOPE_BYTES),
        scales[..., : _D_NOPE // 16].reshape(-1, _D_NOPE // 16),
    )
    rope = (
        data[..., _PACKED_NOPE_BYTES:]
        .contiguous()
        .view(torch.bfloat16)
        .reshape(-1, _D_ROPE)
        .float()
    )
    return torch.cat((nope, rope), dim=-1).reshape(
        num_pages, page_size, 1, _D_NOPE + _D_ROPE
    )


def _dequantize_nvfp4_query(q: torch.Tensor) -> torch.Tensor:
    q_flat = q.reshape(-1, _D_NOPE + _D_ROPE)
    global_scale = torch.ones(1, dtype=torch.float32, device=q.device)
    packed, scales = flashinfer.nvfp4_kv_quantize(
        q_flat[:, :_D_NOPE].contiguous(), global_scale
    )
    nope = _dequantize_linear_nvfp4(packed, scales)
    return torch.cat((nope, q_flat[:, _D_NOPE:].float()), dim=-1).reshape_as(q.float())


def _reference_sparse_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    *,
    topk_length: torch.Tensor | None = None,
    attn_sink: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens, num_heads, dim = q.shape
    topk = indices.shape[1]
    invalid = indices < 0
    if topk_length is not None:
        positions = torch.arange(topk, device=q.device).unsqueeze(0)
        invalid = invalid | (positions >= topk_length.unsqueeze(1))
    gathered = kv.reshape(-1, dim).index_select(
        0, indices.clamp_min(0).reshape(-1).long()
    )
    gathered = gathered.reshape(num_tokens, topk, dim)
    logits = torch.einsum("thd,tkd->thk", q, gathered) * sm_scale
    logits.masked_fill_(invalid.unsqueeze(1), float("-inf"))
    lse = torch.logsumexp(logits, dim=-1)
    safe_lse = torch.where(torch.isneginf(lse), torch.inf, lse)
    weights = torch.exp(logits - safe_lse.unsqueeze(-1))
    output = torch.einsum("thk,tkd->thd", weights, gathered)

    if attn_sink is not None:
        sink = attn_sink.float().unsqueeze(0)
        output *= torch.sigmoid(lse - sink).unsqueeze(-1)
        lse = torch.logaddexp(lse, sink)

    return output.to(torch.bfloat16), lse * math.log2(math.e)
