"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import pytest
import torch

import flashinfer

# E2M1 lookup table for reference dequantization
E2M1_LUT = [
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
]


def reference_dequant(fp4_data, block_scales, global_scale_val, output_dtype):
    """Vectorized PyTorch reference dequantization."""
    M, K_packed = fp4_data.shape
    K = K_packed * 2

    lut = torch.tensor(E2M1_LUT, dtype=torch.float32)

    # Unpack FP4 nibbles: [M, K_packed] -> lo/hi [M, K_packed]
    fp4_bytes = fp4_data.cpu().to(torch.int32)
    fp4_lo = fp4_bytes & 0xF
    fp4_hi = (fp4_bytes >> 4) & 0xF

    # Interleave lo/hi to get [M, K] indices
    indices = torch.stack([fp4_lo, fp4_hi], dim=-1).reshape(M, K)
    values = lut[indices]

    # Convert block scales from FP8 E4M3 bytes to float: [M, K/16]
    scale_floats = block_scales.cpu().view(torch.float8_e4m3fn).float()

    # Expand scales to match each element: each scale covers 16 elements
    # [M, K/16] -> [M, K/16, 1] -> [M, K/16, 16] -> [M, K]
    scale_expanded = scale_floats.unsqueeze(-1).expand(M, K // 16, 16).reshape(M, K)

    output = values * scale_expanded * global_scale_val
    return output.to(output_dtype).to(fp4_data.device)


def get_compute_capability():
    props = torch.cuda.get_device_properties(0)
    return props.major * 10 + props.minor


SHAPES = [(128, 64), (256, 128), (1, 32), (2048, 2048)]
DTYPES = [torch.bfloat16, torch.float16]


def make_non_contiguous_last_dim_view(x):
    padded_shape = (x.size(0) + 1, x.size(1) + 1, x.size(2) + 1, x.size(3) + 2)
    padded = torch.empty(padded_shape, dtype=x.dtype, device=x.device)
    view = padded[1:, 1:, 1:, 1 : 1 + x.size(3)]
    view.copy_(x)
    assert not view.is_contiguous()
    assert view.stride(-1) == 1
    assert view.storage_offset() > 0
    return view


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_nvfp4_kv_dequant(shape, dtype):
    """Test dequantization kernel against PyTorch reference."""
    cc = get_compute_capability()
    if cc < 80:
        pytest.skip(f"SM{cc} does not support FP8 E4M3 (requires SM80+)")

    M, K = shape
    torch.manual_seed(42)

    # Generate random FP4 packed data (each byte holds 2 FP4 values, 0-15 per nibble)
    fp4_data = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device="cuda")

    # Generate random block scales as FP8 E4M3 bytes
    # Use values that are valid FP8 E4M3 (avoid NaN/Inf for stability)
    block_scales = torch.randint(1, 120, (M, K // 16), dtype=torch.uint8, device="cuda")

    global_scale_val = 0.5
    global_scale = torch.tensor([global_scale_val], dtype=torch.float32, device="cuda")

    # CUDA kernel output
    output = flashinfer.nvfp4_kv_dequantize(
        fp4_data, block_scales, global_scale, output_dtype=dtype
    )

    # Reference output
    ref = reference_dequant(fp4_data, block_scales, global_scale_val, dtype)

    torch.testing.assert_close(output.float(), ref.float(), atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("kv_layout", ["NHD", "HND"])
@pytest.mark.parametrize("block_table_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("non_contiguous", [False, True])
def test_nvfp4_kv_dequantize_paged(kv_layout, block_table_dtype, dtype, non_contiguous):
    """Test paged NVFP4 KV dequantization against PyTorch reference."""
    cc = get_compute_capability()
    if cc < 80:
        pytest.skip(f"SM{cc} does not support FP8 E4M3 (requires SM80+)")

    torch.manual_seed(42)

    num_pages = 8
    page_size = 3
    batch_size = 2
    max_seq_len = 7
    num_kv_heads = 2
    k_head_dim = 64
    v_head_dim = 128
    k_scale_dim = k_head_dim // 16
    v_scale_dim = v_head_dim // 16

    k_cache_nhd = torch.randint(
        0,
        256,
        (num_pages, page_size, num_kv_heads, k_head_dim // 2),
        dtype=torch.uint8,
        device="cuda",
    )
    v_cache_nhd = torch.randint(
        0,
        256,
        (num_pages, page_size, num_kv_heads, v_head_dim // 2),
        dtype=torch.uint8,
        device="cuda",
    )
    k_scales_nhd = torch.randint(
        1,
        120,
        (num_pages, page_size, num_kv_heads, k_scale_dim),
        dtype=torch.uint8,
        device="cuda",
    ).view(torch.float8_e4m3fn)
    v_scales_nhd = torch.randint(
        1,
        120,
        (num_pages, page_size, num_kv_heads, v_scale_dim),
        dtype=torch.uint8,
        device="cuda",
    ).view(torch.float8_e4m3fn)

    if kv_layout == "NHD":
        k_cache = k_cache_nhd
        v_cache = v_cache_nhd
        k_scales = k_scales_nhd
        v_scales = v_scales_nhd
    else:
        k_cache = k_cache_nhd.permute(0, 2, 1, 3).contiguous()
        v_cache = v_cache_nhd.permute(0, 2, 1, 3).contiguous()
        k_scales = k_scales_nhd.permute(0, 2, 1, 3).contiguous()
        v_scales = v_scales_nhd.permute(0, 2, 1, 3).contiguous()
    if non_contiguous:
        k_cache = make_non_contiguous_last_dim_view(k_cache)
        v_cache = make_non_contiguous_last_dim_view(v_cache)
        k_scales = make_non_contiguous_last_dim_view(k_scales)
        v_scales = make_non_contiguous_last_dim_view(v_scales)

    block_tables = torch.tensor(
        [[2, 5, 1], [6, 3, 0]], dtype=block_table_dtype, device="cuda"
    )
    seq_lens = torch.tensor([7, 4], dtype=torch.int32, device="cuda")
    k_scale_val = 0.5
    v_scale_val = 0.25
    k_scale = torch.tensor([k_scale_val], dtype=torch.float32, device="cuda")
    v_scale = torch.tensor([v_scale_val], dtype=torch.float32, device="cuda")

    output_k = torch.full(
        (batch_size, max_seq_len, num_kv_heads, k_head_dim),
        123.0,
        dtype=dtype,
        device="cuda",
    )
    output_v = torch.full(
        (batch_size, max_seq_len, num_kv_heads, v_head_dim),
        123.0,
        dtype=dtype,
        device="cuda",
    )

    flashinfer.nvfp4_kv_dequantize_paged(
        (k_cache, v_cache),
        (k_scales, v_scales),
        block_tables,
        seq_lens,
        k_scale,
        v_scale,
        output_k,
        output_v,
        kv_layout=kv_layout,
    )

    ref_k = torch.full_like(output_k, 123.0)
    ref_v = torch.full_like(output_v, 123.0)
    for batch_idx, seq_len in enumerate(seq_lens.cpu().tolist()):
        for token_idx in range(seq_len):
            page = int(block_tables[batch_idx, token_idx // page_size].item())
            entry = token_idx % page_size
            k_rows = k_cache_nhd[page, entry]
            v_rows = v_cache_nhd[page, entry]
            k_scale_rows = k_scales_nhd[page, entry]
            v_scale_rows = v_scales_nhd[page, entry]
            ref_k[batch_idx, token_idx] = reference_dequant(
                k_rows, k_scale_rows, k_scale_val, dtype
            )
            ref_v[batch_idx, token_idx] = reference_dequant(
                v_rows, v_scale_rows, v_scale_val, dtype
            )

    torch.testing.assert_close(output_k.float(), ref_k.float(), atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(output_v.float(), ref_v.float(), atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("kv_layout", ["NHD", "HND"])
def test_nvfp4_kv_dequantize_paged_long_context(kv_layout):
    """Exercise a long single-request cached-prefill style page walk."""
    cc = get_compute_capability()
    if cc < 80:
        pytest.skip(f"SM{cc} does not support FP8 E4M3 (requires SM80+)")

    page_size = 16
    batch_size = 1
    max_seq_len = 16674
    num_pages = (max_seq_len + page_size - 1) // page_size
    num_kv_heads = 4
    k_head_dim = 128
    v_head_dim = 128
    k_scale_dim = k_head_dim // 16
    v_scale_dim = v_head_dim // 16

    k_cache_nhd = torch.zeros(
        (num_pages, page_size, num_kv_heads, k_head_dim // 2),
        dtype=torch.uint8,
        device="cuda",
    )
    v_cache_nhd = torch.zeros(
        (num_pages, page_size, num_kv_heads, v_head_dim // 2),
        dtype=torch.uint8,
        device="cuda",
    )
    k_scales_nhd = torch.zeros(
        (num_pages, page_size, num_kv_heads, k_scale_dim),
        dtype=torch.uint8,
        device="cuda",
    ).view(torch.float8_e4m3fn)
    v_scales_nhd = torch.zeros(
        (num_pages, page_size, num_kv_heads, v_scale_dim),
        dtype=torch.uint8,
        device="cuda",
    ).view(torch.float8_e4m3fn)

    if kv_layout == "NHD":
        k_cache = k_cache_nhd
        v_cache = v_cache_nhd
        k_scales = k_scales_nhd
        v_scales = v_scales_nhd
    else:
        k_cache = k_cache_nhd.permute(0, 2, 1, 3).contiguous()
        v_cache = v_cache_nhd.permute(0, 2, 1, 3).contiguous()
        k_scales = k_scales_nhd.permute(0, 2, 1, 3).contiguous()
        v_scales = v_scales_nhd.permute(0, 2, 1, 3).contiguous()

    block_tables = torch.arange(num_pages, dtype=torch.int32, device="cuda").reshape(
        batch_size, num_pages
    )
    seq_lens = torch.tensor([max_seq_len], dtype=torch.int32, device="cuda")
    k_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    v_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    output_k = torch.full(
        (batch_size, max_seq_len, num_kv_heads, k_head_dim),
        123.0,
        dtype=torch.bfloat16,
        device="cuda",
    )
    output_v = torch.full(
        (batch_size, max_seq_len, num_kv_heads, v_head_dim),
        123.0,
        dtype=torch.bfloat16,
        device="cuda",
    )

    flashinfer.nvfp4_kv_dequantize_paged(
        (k_cache, v_cache),
        (k_scales, v_scales),
        block_tables,
        seq_lens,
        k_scale,
        v_scale,
        output_k,
        output_v,
        kv_layout=kv_layout,
    )
    torch.cuda.synchronize()

    assert torch.count_nonzero(output_k).item() == 0
    assert torch.count_nonzero(output_v).item() == 0


@pytest.mark.parametrize("kv_layout", ["NHD", "HND"])
@pytest.mark.parametrize(
    "batch_size,max_seq_len,num_kv_heads,block_table_stride",
    [
        (0, 3, 2, 1),
        (2, 0, 2, 0),
        (2, 3, 0, 1),
    ],
)
def test_nvfp4_kv_dequantize_paged_empty_output(
    kv_layout, batch_size, max_seq_len, num_kv_heads, block_table_stride
):
    """Empty output tensors should return before launching a zero-sized grid."""
    cc = get_compute_capability()
    if cc < 80:
        pytest.skip(f"SM{cc} does not support FP8 E4M3 (requires SM80+)")

    page_size = 4
    num_pages = 1
    k_head_dim = 64
    v_head_dim = 128
    k_scale_dim = k_head_dim // 16
    v_scale_dim = v_head_dim // 16

    if kv_layout == "NHD":
        k_cache_shape = (num_pages, page_size, num_kv_heads, k_head_dim // 2)
        v_cache_shape = (num_pages, page_size, num_kv_heads, v_head_dim // 2)
        k_scale_shape = (num_pages, page_size, num_kv_heads, k_scale_dim)
        v_scale_shape = (num_pages, page_size, num_kv_heads, v_scale_dim)
    else:
        k_cache_shape = (num_pages, num_kv_heads, page_size, k_head_dim // 2)
        v_cache_shape = (num_pages, num_kv_heads, page_size, v_head_dim // 2)
        k_scale_shape = (num_pages, num_kv_heads, page_size, k_scale_dim)
        v_scale_shape = (num_pages, num_kv_heads, page_size, v_scale_dim)

    k_cache = torch.empty(k_cache_shape, dtype=torch.uint8, device="cuda")
    v_cache = torch.empty(v_cache_shape, dtype=torch.uint8, device="cuda")
    k_scales = torch.empty(k_scale_shape, dtype=torch.uint8, device="cuda").view(
        torch.float8_e4m3fn
    )
    v_scales = torch.empty(v_scale_shape, dtype=torch.uint8, device="cuda").view(
        torch.float8_e4m3fn
    )
    block_tables = torch.zeros(
        (batch_size, block_table_stride), dtype=torch.int32, device="cuda"
    )
    seq_lens = torch.zeros((batch_size,), dtype=torch.int32, device="cuda")
    k_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    v_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    output_k = torch.empty(
        (batch_size, max_seq_len, num_kv_heads, k_head_dim),
        dtype=torch.bfloat16,
        device="cuda",
    )
    output_v = torch.empty(
        (batch_size, max_seq_len, num_kv_heads, v_head_dim),
        dtype=torch.bfloat16,
        device="cuda",
    )

    flashinfer.nvfp4_kv_dequantize_paged(
        (k_cache, v_cache),
        (k_scales, v_scales),
        block_tables,
        seq_lens,
        k_scale,
        v_scale,
        output_k,
        output_v,
        kv_layout=kv_layout,
    )
    torch.cuda.synchronize()


@pytest.mark.parametrize("kv_layout", ["NHD", "HND"])
@pytest.mark.parametrize("dtype", DTYPES)
def test_nvfp4_kv_dequantize_paged_stacked_cache(kv_layout, dtype):
    """Test stacked paged KV cache input for the paged NVFP4 dequant helper."""
    cc = get_compute_capability()
    if cc < 80:
        pytest.skip(f"SM{cc} does not support FP8 E4M3 (requires SM80+)")

    torch.manual_seed(7)

    num_pages = 5
    page_size = 4
    batch_size = 2
    max_seq_len = 6
    num_kv_heads = 2
    head_dim = 64
    scale_dim = head_dim // 16

    k_cache_nhd = torch.randint(
        0,
        256,
        (num_pages, page_size, num_kv_heads, head_dim // 2),
        dtype=torch.uint8,
        device="cuda",
    )
    v_cache_nhd = torch.randint(
        0,
        256,
        (num_pages, page_size, num_kv_heads, head_dim // 2),
        dtype=torch.uint8,
        device="cuda",
    )
    k_scales_nhd = torch.randint(
        1,
        120,
        (num_pages, page_size, num_kv_heads, scale_dim),
        dtype=torch.uint8,
        device="cuda",
    ).view(torch.float8_e4m3fn)
    v_scales_nhd = torch.randint(
        1,
        120,
        (num_pages, page_size, num_kv_heads, scale_dim),
        dtype=torch.uint8,
        device="cuda",
    ).view(torch.float8_e4m3fn)

    if kv_layout == "NHD":
        stacked_cache = torch.stack([k_cache_nhd, v_cache_nhd], dim=1).contiguous()
        stacked_scales = torch.stack([k_scales_nhd, v_scales_nhd], dim=1).contiguous()
    else:
        k_cache_hnd = k_cache_nhd.permute(0, 2, 1, 3).contiguous()
        v_cache_hnd = v_cache_nhd.permute(0, 2, 1, 3).contiguous()
        k_scales_hnd = k_scales_nhd.permute(0, 2, 1, 3).contiguous()
        v_scales_hnd = v_scales_nhd.permute(0, 2, 1, 3).contiguous()
        stacked_cache = torch.stack([k_cache_hnd, v_cache_hnd], dim=1).contiguous()
        stacked_scales = torch.stack([k_scales_hnd, v_scales_hnd], dim=1).contiguous()

    block_tables = torch.tensor([[1, 3], [4, 2]], dtype=torch.int32, device="cuda")
    seq_lens = torch.tensor([6, 3], dtype=torch.int32, device="cuda")
    k_scale_val = 0.75
    v_scale_val = 0.5
    k_scale = torch.tensor([k_scale_val], dtype=torch.float32, device="cuda")
    v_scale = torch.tensor([v_scale_val], dtype=torch.float32, device="cuda")
    output_k = torch.full(
        (batch_size, max_seq_len, num_kv_heads, head_dim),
        123.0,
        dtype=dtype,
        device="cuda",
    )
    output_v = torch.full_like(output_k, 123.0)

    flashinfer.nvfp4_kv_dequantize_paged(
        stacked_cache,
        stacked_scales,
        block_tables,
        seq_lens,
        k_scale,
        v_scale,
        output_k,
        output_v,
        kv_layout=kv_layout,
    )

    ref_k = torch.full_like(output_k, 123.0)
    ref_v = torch.full_like(output_v, 123.0)
    for batch_idx, seq_len in enumerate(seq_lens.cpu().tolist()):
        for token_idx in range(seq_len):
            page = int(block_tables[batch_idx, token_idx // page_size].item())
            entry = token_idx % page_size
            ref_k[batch_idx, token_idx] = reference_dequant(
                k_cache_nhd[page, entry],
                k_scales_nhd[page, entry],
                k_scale_val,
                dtype,
            )
            ref_v[batch_idx, token_idx] = reference_dequant(
                v_cache_nhd[page, entry],
                v_scales_nhd[page, entry],
                v_scale_val,
                dtype,
            )

    torch.testing.assert_close(output_k.float(), ref_k.float(), atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(output_v.float(), ref_v.float(), atol=1e-3, rtol=1e-3)


def test_nvfp4_kv_dequantize_paged_rejects_short_block_tables():
    """Reject page tables that cannot cover the requested output length."""
    cc = get_compute_capability()
    if cc < 80:
        pytest.skip(f"SM{cc} does not support FP8 E4M3 (requires SM80+)")

    num_pages = 2
    page_size = 4
    batch_size = 1
    max_seq_len = 5
    num_kv_heads = 1
    head_dim = 64
    scale_dim = head_dim // 16

    k_cache = torch.empty(
        (num_pages, page_size, num_kv_heads, head_dim // 2),
        dtype=torch.uint8,
        device="cuda",
    )
    v_cache = torch.empty_like(k_cache)
    k_scales = torch.empty(
        (num_pages, page_size, num_kv_heads, scale_dim),
        dtype=torch.uint8,
        device="cuda",
    ).view(torch.float8_e4m3fn)
    v_scales = torch.empty_like(k_scales)
    # max_seq_len=5 and page_size=4 require two page table columns.
    block_tables = torch.zeros((batch_size, 1), dtype=torch.int32, device="cuda")
    seq_lens = torch.tensor([max_seq_len], dtype=torch.int32, device="cuda")
    k_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    v_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    output_k = torch.empty(
        (batch_size, max_seq_len, num_kv_heads, head_dim),
        dtype=torch.bfloat16,
        device="cuda",
    )
    output_v = torch.empty_like(output_k)

    with pytest.raises(RuntimeError, match="block_tables column count insufficient"):
        flashinfer.nvfp4_kv_dequantize_paged(
            (k_cache, v_cache),
            (k_scales, v_scales),
            block_tables,
            seq_lens,
            k_scale,
            v_scale,
            output_k,
            output_v,
            kv_layout="NHD",
        )


def test_nvfp4_kv_dequantize_paged_rejects_non_scalar_scale():
    """Reject global scale tensors with more than one element."""
    cc = get_compute_capability()
    if cc < 80:
        pytest.skip(f"SM{cc} does not support FP8 E4M3 (requires SM80+)")

    num_pages = 1
    page_size = 1
    batch_size = 1
    max_seq_len = 1
    num_kv_heads = 1
    head_dim = 64
    scale_dim = head_dim // 16

    k_cache = torch.empty(
        (num_pages, page_size, num_kv_heads, head_dim // 2),
        dtype=torch.uint8,
        device="cuda",
    )
    v_cache = torch.empty_like(k_cache)
    k_scales = torch.empty(
        (num_pages, page_size, num_kv_heads, scale_dim),
        dtype=torch.uint8,
        device="cuda",
    ).view(torch.float8_e4m3fn)
    v_scales = torch.empty_like(k_scales)
    block_tables = torch.zeros((batch_size, 1), dtype=torch.int32, device="cuda")
    seq_lens = torch.tensor([max_seq_len], dtype=torch.int32, device="cuda")
    k_scale = torch.ones((2,), dtype=torch.float32, device="cuda")
    v_scale = torch.ones((1,), dtype=torch.float32, device="cuda")
    output_k = torch.empty(
        (batch_size, max_seq_len, num_kv_heads, head_dim),
        dtype=torch.bfloat16,
        device="cuda",
    )
    output_v = torch.empty_like(output_k)

    with pytest.raises(ValueError, match="k_scale and v_scale must be scalar tensors"):
        flashinfer.nvfp4_kv_dequantize_paged(
            (k_cache, v_cache),
            (k_scales, v_scales),
            block_tables,
            seq_lens,
            k_scale,
            v_scale,
            output_k,
            output_v,
            kv_layout="NHD",
        )


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_nvfp4_kv_quant(shape, dtype):
    """Test quantization kernel output shapes and basic validity."""
    cc = get_compute_capability()
    if cc < 100:
        pytest.skip(f"SM{cc} does not support NVFP4 quantization (requires SM100+)")

    M, K = shape
    torch.manual_seed(42)

    input_data = torch.randn((M, K), dtype=dtype, device="cuda")
    global_scale_val = 1.0
    global_scale = torch.tensor([global_scale_val], dtype=torch.float32, device="cuda")

    fp4_output, block_scales = flashinfer.nvfp4_kv_quantize(input_data, global_scale)

    # Check shapes
    assert fp4_output.shape == (M, K // 2)
    assert fp4_output.dtype == torch.uint8
    assert block_scales.shape == (M, K // 16)
    assert block_scales.dtype == torch.uint8

    # Check that FP4 values are in valid range (each nibble 0-15)
    assert (fp4_output <= 255).all()


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_nvfp4_kv_roundtrip(shape, dtype):
    """Test quantize -> dequantize roundtrip error is within FP4 precision."""
    cc = get_compute_capability()
    if cc < 100:
        pytest.skip(f"SM{cc} does not support NVFP4 quantization (requires SM100+)")

    M, K = shape
    torch.manual_seed(42)

    input_data = torch.randn((M, K), dtype=dtype, device="cuda")
    # Use global_scale=1.0 to avoid FP8 E4M3 block scale underflow
    global_scale_val = 1.0
    global_scale = torch.tensor([global_scale_val], dtype=torch.float32, device="cuda")

    # Quantize
    fp4_output, block_scales = flashinfer.nvfp4_kv_quantize(input_data, global_scale)

    # Dequantize
    reconstructed = flashinfer.nvfp4_kv_dequantize(
        fp4_output, block_scales, global_scale, output_dtype=dtype
    )

    # FP4 E2M1 has very limited precision (only 16 representable values),
    # so we check relative error with generous tolerance
    input_float = input_data.float()
    recon_float = reconstructed.float()

    # Compute per-element relative error where input is non-negligible
    mask = input_float.abs() > 1e-6
    if mask.any():
        rel_error = (input_float[mask] - recon_float[mask]).abs() / input_float[
            mask
        ].abs().clamp(min=1e-6)
        # FP4 quantization can have up to ~50% relative error for some values,
        # but on average should be much better
        assert rel_error.mean() < 0.5, (
            f"Mean relative error too high: {rel_error.mean():.4f}"
        )

    # Also check that the overall cosine similarity is reasonable
    cos_sim = torch.nn.functional.cosine_similarity(
        input_float.flatten().unsqueeze(0),
        recon_float.flatten().unsqueeze(0),
    )
    assert cos_sim > 0.8, f"Cosine similarity too low: {cos_sim:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
