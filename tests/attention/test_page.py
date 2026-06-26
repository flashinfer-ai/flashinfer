import pytest
import torch

import flashinfer
from flashinfer.fp4_quantization import e2m1_and_ufp8sf_scale_to_float


def _nvfp4_dequant_linear(packed, scales, global_scale):
    packed_flat = packed.reshape(-1, packed.shape[-1])
    scales_flat = scales.reshape(-1, scales.shape[-1]).view(torch.uint8)
    global_scale_tensor = torch.tensor(
        [global_scale], dtype=torch.float32, device=packed.device
    )
    dequant = e2m1_and_ufp8sf_scale_to_float(
        packed_flat,
        scales_flat,
        global_scale_tensor,
        sf_vec_size=16,
        is_sf_swizzled_layout=False,
    )
    return dequant.reshape(*packed.shape[:-1], packed.shape[-1] * 2)


def _assert_nvfp4_quantized_close(actual, expected):
    actual_f = actual.float().flatten()
    expected_f = expected.float().flatten()
    mask = expected_f.abs() > 1e-6
    if mask.any():
        rel_error = (actual_f[mask] - expected_f[mask]).abs() / expected_f[mask].abs()
        assert rel_error.mean() < 0.5
    cos_sim = torch.nn.functional.cosine_similarity(
        actual_f.unsqueeze(0), expected_f.unsqueeze(0)
    )
    assert cos_sim > 0.8


@pytest.mark.parametrize("contiguous", [True, False])
def test_append_paged_kv_cache(contiguous):
    nnz_kv = 100
    num_kv_heads = 32
    head_dim = 128

    if contiguous:
        k_append = torch.randn(nnz_kv, num_kv_heads, head_dim).half().to(0)
        v_append = torch.randn(nnz_kv, num_kv_heads, head_dim).half().to(0)
    else:
        kv_append = torch.randn(nnz_kv, 2, num_kv_heads, head_dim).half().to(0)
        k_append = kv_append[:, 0]
        v_append = kv_append[:, 1]
    # 45 + 8 + 25 + 22 = nnz_kv
    kv_append_length = torch.tensor([45, 8, 25, 22], dtype=torch.int32, device="cuda:0")
    kv_append_indptr = torch.cat(
        [torch.zeros(1).int().to(0), torch.cumsum(kv_append_length, dim=0)]
    ).int()

    max_num_pages = 1000
    page_size = 16
    paged_kv_cache = (
        torch.randn(max_num_pages, 2, page_size, num_kv_heads, head_dim).half().to(0)
    )
    num_pages_per_req = torch.tensor([3, 1, 2, 2], dtype=torch.int32, device="cuda:0")
    kv_page_indptr = torch.cat(
        [torch.zeros(1).int().to(0), torch.cumsum(num_pages_per_req, dim=0)]
    ).int()
    # use first 8 pages in the paged-kv
    kv_page_indices = torch.arange(8, dtype=torch.int32, device="cuda:0")
    # 45 = (3 - 1) * 16 + 13
    # 8 = (1 - 1) * 16 + 8
    # 25 = (2 - 1) * 16 + 9
    # 22 = (2 - 1) * 16 + 6
    kv_last_page_len = torch.tensor([13, 8, 9, 6], dtype=torch.int32, device="cuda:0")
    batch_indices, positions = flashinfer.get_batch_indices_positions(
        kv_append_indptr,
        flashinfer.get_seq_lens(kv_page_indptr, kv_last_page_len, page_size),
        nnz_kv,
    )

    flashinfer.append_paged_kv_cache(
        k_append,
        v_append,
        batch_indices,
        positions,
        paged_kv_cache,
        kv_page_indices,
        kv_page_indptr,
        kv_last_page_len,
    )


@pytest.mark.parametrize("kv_layout", ["NHD", "HND"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim", [128, 512])
@pytest.mark.parametrize("contiguous", [True, False])
def test_nvfp4_quantize_append_paged_kv_cache(kv_layout, dtype, head_dim, contiguous):
    cc = (
        torch.cuda.get_device_capability()[0] * 10
        + torch.cuda.get_device_capability()[1]
    )
    if cc < 80:
        pytest.skip(f"SM{cc} does not support FP8 E4M3 scale tensors")

    torch.manual_seed(42)
    nnz_kv = 100
    num_kv_heads = 32
    page_size = 16
    max_num_pages = 1000

    if contiguous:
        k_append = torch.randn(
            nnz_kv, num_kv_heads, head_dim, device="cuda:0", dtype=dtype
        )
        v_append = torch.randn_like(k_append)
    else:
        kv_append = torch.randn(
            nnz_kv, 2, num_kv_heads, head_dim, device="cuda:0", dtype=dtype
        )
        k_append = kv_append[:, 0]
        v_append = kv_append[:, 1]

    if kv_layout == "NHD":
        packed_shape = (max_num_pages, page_size, num_kv_heads, head_dim // 2)
        scale_shape = (max_num_pages, page_size, num_kv_heads, head_dim // 16)
    else:
        packed_shape = (max_num_pages, num_kv_heads, page_size, head_dim // 2)
        scale_shape = (max_num_pages, num_kv_heads, page_size, head_dim // 16)

    k_cache = torch.zeros(packed_shape, dtype=torch.uint8, device="cuda:0")
    v_cache = torch.zeros_like(k_cache)
    k_scales = torch.zeros(scale_shape, dtype=torch.float8_e4m3fn, device="cuda:0")
    v_scales = torch.zeros_like(k_scales)

    kv_append_length = torch.tensor([45, 8, 25, 22], dtype=torch.int32, device="cuda:0")
    kv_append_indptr = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device="cuda:0"),
            torch.cumsum(kv_append_length, dim=0),
        ]
    )
    num_pages_per_req = torch.tensor([3, 1, 2, 2], dtype=torch.int32, device="cuda:0")
    kv_page_indptr = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device="cuda:0"),
            torch.cumsum(num_pages_per_req, dim=0),
        ]
    )
    kv_page_indices = torch.arange(8, dtype=torch.int32, device="cuda:0")
    kv_last_page_len = torch.tensor([13, 8, 9, 6], dtype=torch.int32, device="cuda:0")
    batch_indices, positions = flashinfer.get_batch_indices_positions(
        kv_append_indptr,
        flashinfer.get_seq_lens(kv_page_indptr, kv_last_page_len, page_size),
        nnz_kv,
    )

    k_scale = 1.0
    v_scale = 1.0
    flashinfer.nvfp4_quantize_append_paged_kv_cache(
        k_append,
        v_append,
        batch_indices,
        positions,
        (k_cache, v_cache),
        (k_scales, v_scales),
        kv_page_indices,
        kv_page_indptr,
        kv_last_page_len,
        k_scale,
        v_scale,
        kv_layout=kv_layout,
    )

    k_dequant = _nvfp4_dequant_linear(k_cache, k_scales, k_scale)
    v_dequant = _nvfp4_dequant_linear(v_cache, v_scales, v_scale)

    batch_indices_cpu = batch_indices.cpu()
    positions_cpu = positions.cpu()
    kv_page_indptr_cpu = kv_page_indptr.cpu()
    kv_page_indices_cpu = kv_page_indices.cpu()
    for i in range(nnz_kv):
        batch_idx = int(batch_indices_cpu[i].item())
        pos = int(positions_cpu[i].item())
        page_offset = pos // page_size
        entry_idx = pos % page_size
        page_id = int(
            kv_page_indices_cpu[int(kv_page_indptr_cpu[batch_idx]) + page_offset].item()
        )
        if kv_layout == "NHD":
            k_actual = k_dequant[page_id, entry_idx]
            v_actual = v_dequant[page_id, entry_idx]
        else:
            k_actual = k_dequant[page_id, :, entry_idx]
            v_actual = v_dequant[page_id, :, entry_idx]
        _assert_nvfp4_quantized_close(k_actual, k_append[i])
        _assert_nvfp4_quantized_close(v_actual, v_append[i])


@pytest.mark.parametrize("kv_layout", ["NHD", "HND"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim", [128, 512])
@pytest.mark.parametrize("contiguous", [True, False])
@pytest.mark.parametrize("slot_dtype", [torch.int32, torch.int64])
def test_nvfp4_quantize_append_paged_kv_cache_with_slot_mapping(
    kv_layout, dtype, head_dim, contiguous, slot_dtype
):
    cc = (
        torch.cuda.get_device_capability()[0] * 10
        + torch.cuda.get_device_capability()[1]
    )
    if cc < 80:
        pytest.skip(f"SM{cc} does not support FP8 E4M3 scale tensors")

    torch.manual_seed(42)
    nnz_kv = 100
    padded_nnz_kv = 112
    num_kv_heads = 32
    page_size = 16
    max_num_pages = 1000

    if contiguous:
        k_append = torch.randn(
            padded_nnz_kv, num_kv_heads, head_dim, device="cuda:0", dtype=dtype
        )
        v_append = torch.randn_like(k_append)
    else:
        kv_append = torch.randn(
            padded_nnz_kv, 2, num_kv_heads, head_dim, device="cuda:0", dtype=dtype
        )
        k_append = kv_append[:, 0]
        v_append = kv_append[:, 1]

    if kv_layout == "NHD":
        packed_shape = (max_num_pages, page_size, num_kv_heads, head_dim // 2)
        scale_shape = (max_num_pages, page_size, num_kv_heads, head_dim // 16)
    else:
        packed_shape = (max_num_pages, num_kv_heads, page_size, head_dim // 2)
        scale_shape = (max_num_pages, num_kv_heads, page_size, head_dim // 16)

    k_cache = torch.zeros(packed_shape, dtype=torch.uint8, device="cuda:0")
    v_cache = torch.zeros_like(k_cache)
    k_scales = torch.zeros(scale_shape, dtype=torch.float8_e4m3fn, device="cuda:0")
    v_scales = torch.zeros_like(k_scales)

    slot_mapping = torch.randperm(
        max_num_pages * page_size, device="cuda:0", dtype=slot_dtype
    )[:nnz_kv]
    slot_mapping[3] = -1
    slot_mapping[71] = -1

    k_scale = torch.ones(1, dtype=torch.float32, device="cuda:0")
    v_scale = torch.ones(1, dtype=torch.float32, device="cuda:0")
    flashinfer.nvfp4_quantize_append_paged_kv_cache_with_slot_mapping(
        k_append,
        v_append,
        slot_mapping,
        (k_cache, v_cache),
        (k_scales, v_scales),
        k_scale,
        v_scale,
        kv_layout=kv_layout,
    )

    k_dequant = _nvfp4_dequant_linear(k_cache, k_scales, float(k_scale.item()))
    v_dequant = _nvfp4_dequant_linear(v_cache, v_scales, float(v_scale.item()))
    if kv_layout == "HND":
        k_dequant = k_dequant.transpose(1, 2).contiguous()
        v_dequant = v_dequant.transpose(1, 2).contiguous()
    k_dequant = k_dequant.reshape(max_num_pages * page_size, num_kv_heads, head_dim)
    v_dequant = v_dequant.reshape(max_num_pages * page_size, num_kv_heads, head_dim)

    valid = slot_mapping >= 0
    valid_slots = slot_mapping[valid].long()
    _assert_nvfp4_quantized_close(k_dequant[valid_slots], k_append[:nnz_kv][valid])
    _assert_nvfp4_quantized_close(v_dequant[valid_slots], v_append[:nnz_kv][valid])
