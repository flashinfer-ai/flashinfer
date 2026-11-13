import torch

from flashinfer import SfLayout


def to_float8(
    x: torch.Tensor, dtype=torch.float8_e4m3fn
) -> tuple[torch.Tensor, torch.Tensor]:
    finfo = torch.finfo(dtype)
    min_val, max_val = x.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-12)
    scale = finfo.max / amax
    x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
    return x_scl_sat.to(dtype), scale.float().reciprocal()


@torch.no_grad()
def _swizzle_mxfp8_scales(scales: torch.Tensor, sf_layout: SfLayout):
    layout = getattr(sf_layout, "value", sf_layout)
    if layout == SfLayout.layout_linear.value:
        return scales.contiguous().reshape(-1)

    m, num_cols = scales.shape
    if layout == SfLayout.layout_128x4.value:
        row_tile = 128
        padded_rows = (m + row_tile - 1) // row_tile * row_tile
        padded_cols = (num_cols + 3) // 4 * 4
        out = torch.zeros(
            padded_rows * padded_cols,
            dtype=torch.uint8,
            device=scales.device,
        )

        rows = torch.arange(m, dtype=torch.int64, device=scales.device).unsqueeze(1)
        cols = torch.arange(
            num_cols, dtype=torch.int64, device=scales.device
        ).unsqueeze(0)
        indices = (
            cols % 4
            + (cols // 4) * 512
            + (rows % 32) * 16
            + ((rows % 128) // 32) * 4
            + (rows // 128) * (128 * padded_cols)
        )
    elif layout == SfLayout.layout_8x4.value:
        row_tile = 8
        padded_rows = (m + row_tile - 1) // row_tile * row_tile
        padded_cols = (num_cols + 3) // 4 * 4
        num_k_tiles = padded_cols // 4
        out = torch.zeros(
            padded_rows * padded_cols,
            dtype=torch.uint8,
            device=scales.device,
        )

        rows = torch.arange(m, dtype=torch.int64, device=scales.device).unsqueeze(1)
        cols = torch.arange(
            num_cols, dtype=torch.int64, device=scales.device
        ).unsqueeze(0)
        indices = (
            (rows // 8) * (num_k_tiles * 32)
            + (cols // 4) * 32
            + (rows % 8) * 4
            + cols % 4
        )
    else:
        raise ValueError(f"Unsupported MXFP8 scale layout: {sf_layout}")

    out[indices.reshape(-1)] = scales.reshape(-1)
    return out


@torch.no_grad()
def mxfp8_quantize_reference(
    a,
    *,
    is_sf_swizzled_layout=True,
    alignment=32,
    sf_swizzle_layout=None,
):
    sf_vec_size = 32
    assert alignment % sf_vec_size == 0
    if sf_swizzle_layout is None:
        sf_swizzle_layout = (
            SfLayout.layout_128x4 if is_sf_swizzled_layout else SfLayout.layout_linear
        )

    x = a.reshape(-1, a.shape[-1]).to(torch.float32)
    m, k = x.shape
    assert k % sf_vec_size == 0
    padded_k = (k + alignment - 1) // alignment * alignment
    if padded_k != k:
        padded = x.new_zeros((m, padded_k))
        padded[:, :k] = x
        x = padded

    blocks = x.reshape(m, padded_k // sf_vec_size, sf_vec_size)
    block_scale = blocks.abs().amax(dim=-1) / 448.0

    scale_bits = block_scale.contiguous().view(torch.int32)
    exp = (scale_bits >> 23) & 0xFF
    mantissa = scale_bits & 0x7FFFFF
    round_up = (mantissa != 0) & ((exp != 0) | (mantissa > 0x400000))
    exp = (exp + round_up.to(exp.dtype)).clamp(max=254)
    scales = torch.where(block_scale > 0, exp, 0).to(torch.uint8)

    inv_scale = torch.ldexp(
        torch.ones_like(block_scale),
        127 - scales.to(torch.int32),
    )
    inv_scale.masked_fill_(scales == 0, 0)
    quantized = (blocks * inv_scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
    return quantized.reshape(*a.shape[:-1], padded_k), _swizzle_mxfp8_scales(
        scales,
        sf_swizzle_layout,
    )


def assert_mxfp8_quantize_exact(
    a,
    a_fp8,
    a_sf,
    *,
    is_sf_swizzled_layout=True,
    alignment=32,
    sf_swizzle_layout=None,
):
    ref_fp8, ref_sf = mxfp8_quantize_reference(
        a,
        is_sf_swizzled_layout=is_sf_swizzled_layout,
        alignment=alignment,
        sf_swizzle_layout=sf_swizzle_layout,
    )
    actual_fp8_values = (
        a_fp8.contiguous().view(torch.float8_e4m3fn)
        if a_fp8.dtype == torch.uint8
        else a_fp8
    )
    actual_fp8_bits = a_fp8.contiguous().view(torch.uint8)
    ref_fp8_bits = ref_fp8.contiguous().view(torch.uint8)
    assert actual_fp8_values.shape == ref_fp8.shape, (
        f"quantized output shape mismatch: actual={actual_fp8_values.shape}, "
        f"expected={ref_fp8.shape}"
    )
    if not torch.equal(actual_fp8_bits, ref_fp8_bits):
        mismatch = actual_fp8_bits != ref_fp8_bits
        mismatch_count = int(mismatch.sum().item())
        first_index = tuple(
            int(x) for x in torch.nonzero(mismatch, as_tuple=False)[0].cpu()
        )
        raise AssertionError(
            f"quantized output element mismatch: "
            f"{mismatch_count}/{actual_fp8_bits.numel()} elements differ; "
            f"first mismatch at index {first_index}: "
            f"actual={actual_fp8_values[first_index].float().item()} "
            f"(bits={int(actual_fp8_bits[first_index].item())}), "
            f"expected={ref_fp8[first_index].float().item()} "
            f"(bits={int(ref_fp8_bits[first_index].item())})"
        )

    assert a_sf.shape == ref_sf.shape, (
        f"scale factors shape mismatch: actual={a_sf.shape}, expected={ref_sf.shape}"
    )
    if not torch.equal(a_sf, ref_sf):
        mismatch = a_sf != ref_sf
        mismatch_count = int(mismatch.sum().item())
        first_index = tuple(
            int(x) for x in torch.nonzero(mismatch, as_tuple=False)[0].cpu()
        )
        raise AssertionError(
            f"scale factors element mismatch: {mismatch_count}/{a_sf.numel()} "
            f"elements differ; first mismatch at index {first_index}: "
            f"actual={int(a_sf[first_index].item())}, "
            f"expected={int(ref_sf[first_index].item())}"
        )
