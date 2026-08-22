import math
from typing import List

import pytest
import torch

import flashinfer
from flashinfer.fp4_quantization import e2m1_and_ufp8sf_scale_to_float
from flashinfer.utils import get_compute_capability

CKV_DIM = 512
KPE_DIM = 64


def _skip_if_fp8_e4m3_unsupported():
    major, minor = get_compute_capability(torch.device("cuda:0"))
    if major < 8:
        pytest.skip(f"SM{major}{minor} does not support FP8 E4M3 tensors")


def calculate_last_page_len(kv_len: List[int], page_size: int):
    return [len % page_size if len % page_size != 0 else page_size for len in kv_len]


kv_len_configs = [
    [45],
    [4096],
    [45, 8, 25],
    [45, 8, 25, 22],
    [45, 8, 25, 22, 400],
    [45, 8, 25, 22, 100],
]


@pytest.mark.parametrize("kv_len", kv_len_configs)
@pytest.mark.parametrize("page_size", [1, 16, 64])
def test_append_mla_paged_kv_cache(kv_len, page_size):
    nnz_kv = sum(kv_len)
    ckv_append = torch.randn(nnz_kv, CKV_DIM, dtype=torch.float16, device="cuda:0")
    kpe_append = torch.randn(nnz_kv, KPE_DIM, dtype=torch.float16, device="cuda:0")
    num_pages_per_req = torch.tensor(
        [math.ceil(len / page_size) for len in kv_len],
        dtype=torch.int32,
        device="cuda:0",
    )
    kv_append_length = torch.tensor(kv_len, dtype=torch.int32, device="cuda:0")
    kv_append_indptr = torch.cat(
        [torch.zeros(1).int().to(0), torch.cumsum(kv_append_length, dim=0)]
    ).int()

    max_num_pages = sum(num_pages_per_req)
    ckv_cache = torch.zeros(
        max_num_pages, page_size, CKV_DIM, dtype=torch.float16, device="cuda:0"
    )
    kpe_cache = torch.zeros(
        max_num_pages, page_size, KPE_DIM, dtype=torch.float16, device="cuda:0"
    )
    kv_page_indptr = torch.cat(
        [torch.zeros(1).int().to(0), torch.cumsum(num_pages_per_req, dim=0)]
    ).int()
    kv_page_indices = torch.arange(
        sum(num_pages_per_req), dtype=torch.int32, device="cuda:0"
    )
    kv_last_page_len = torch.tensor(
        calculate_last_page_len(kv_len, page_size), dtype=torch.int32, device="cuda:0"
    )
    batch_indices, positions = flashinfer.get_batch_indices_positions(
        kv_append_indptr,
        flashinfer.get_seq_lens(kv_page_indptr, kv_last_page_len, page_size),
        nnz_kv,
    )
    flashinfer.append_paged_mla_kv_cache(
        ckv_append,
        kpe_append,
        batch_indices,
        positions,
        ckv_cache,
        kpe_cache,
        kv_page_indices,
        kv_page_indptr,
        kv_last_page_len,
    )

    ckv_cache = ckv_cache.view(-1, CKV_DIM)
    kpe_cache = kpe_cache.view(-1, KPE_DIM)

    acc_kv_len = 0
    acc_padding_kv_len = 0
    for i in range(len(kv_len)):
        assert torch.all(
            torch.isclose(
                kpe_append[acc_kv_len : acc_kv_len + kv_len[i]],
                kpe_cache[acc_padding_kv_len : acc_padding_kv_len + kv_len[i]],
            )
        )
        assert torch.all(
            torch.isclose(
                ckv_append[acc_kv_len : acc_kv_len + kv_len[i]],
                ckv_cache[acc_padding_kv_len : acc_padding_kv_len + kv_len[i]],
            )
        )
        assert bool(
            torch.all(
                ckv_cache[
                    acc_padding_kv_len + kv_len[i] : acc_padding_kv_len
                    + num_pages_per_req[i] * page_size
                ]
                == 0
            )
        )
        assert bool(
            torch.all(
                kpe_cache[
                    acc_padding_kv_len + kv_len[i] : acc_padding_kv_len
                    + num_pages_per_req[i] * page_size
                ]
                == 0
            )
        )
        acc_kv_len += kv_len[i]
        acc_padding_kv_len += num_pages_per_req[i] * page_size


def _reference_e2m1_encode(scaled: torch.Tensor) -> torch.Tensor:
    """Round-to-nearest-even E2M1 codes, matching nvfp4_append_quantize_e2m1."""
    sign = torch.where(torch.signbit(scaled), 8, 0).to(torch.uint8)
    mag = scaled.abs()
    code = torch.zeros_like(mag, dtype=torch.uint8)
    code = torch.where(
        mag > 0.25, torch.tensor(1, dtype=torch.uint8, device=mag.device), code
    )
    code = torch.where(
        mag >= 0.75, torch.tensor(2, dtype=torch.uint8, device=mag.device), code
    )
    code = torch.where(
        mag > 1.25, torch.tensor(3, dtype=torch.uint8, device=mag.device), code
    )
    code = torch.where(
        mag >= 1.75, torch.tensor(4, dtype=torch.uint8, device=mag.device), code
    )
    code = torch.where(
        mag > 2.5, torch.tensor(5, dtype=torch.uint8, device=mag.device), code
    )
    code = torch.where(
        mag >= 3.5, torch.tensor(6, dtype=torch.uint8, device=mag.device), code
    )
    code = torch.where(
        mag > 5.0, torch.tensor(7, dtype=torch.uint8, device=mag.device), code
    )
    return sign | code


def _reference_nvfp4_quantize_ckv(ckv: torch.Tensor, global_scale: float):
    """Bit-exact replica of the ckv path of nvfp4_quantize_append_paged_mla_kv_cache."""
    gs = torch.tensor(global_scale, dtype=torch.float32, device=ckv.device)
    inv_6gs = torch.tensor(
        6.0 * gs.item(), dtype=torch.float32, device=ckv.device
    ).reciprocal()
    n = ckv.shape[0]
    blocks = ckv.float().reshape(n, CKV_DIM // 16, 16)
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    sf = torch.where(amax > 0, amax * inv_6gs, torch.zeros_like(amax))
    sf_fp8 = sf.to(torch.float8_e4m3fn)
    sf_rounded = sf_fp8.float()
    output_scale = torch.where(
        (amax > 0) & (sf_rounded > 0),
        (sf_rounded * gs).reciprocal(),
        torch.zeros_like(sf_rounded),
    )
    codes = _reference_e2m1_encode(blocks * output_scale).reshape(n, CKV_DIM)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return packed, sf_fp8.reshape(n, CKV_DIM // 16)


def _reference_fp8_quantize_kpe(kpe: torch.Tensor, kpe_scale: float) -> torch.Tensor:
    inv_scale = torch.tensor(kpe_scale, dtype=torch.float32).reciprocal()
    return (kpe.float() * inv_scale.to(kpe.device)).to(torch.float8_e4m3fn)


def _build_mla_paged_inputs(kv_len, page_size, permute_pages, device="cuda:0"):
    nnz_kv = sum(kv_len)
    num_pages_per_req = torch.tensor(
        [math.ceil(len / page_size) for len in kv_len], dtype=torch.int32, device=device
    )
    kv_append_length = torch.tensor(kv_len, dtype=torch.int32, device=device)
    kv_append_indptr = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=device),
            torch.cumsum(kv_append_length, dim=0),
        ]
    ).int()
    max_num_pages = int(num_pages_per_req.sum().item())
    kv_page_indptr = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=device),
            torch.cumsum(num_pages_per_req, dim=0),
        ]
    ).int()
    if permute_pages:
        kv_page_indices = torch.randperm(
            max_num_pages, dtype=torch.int64, device=device
        ).int()
    else:
        kv_page_indices = torch.arange(max_num_pages, dtype=torch.int32, device=device)
    kv_last_page_len = torch.tensor(
        calculate_last_page_len(kv_len, page_size), dtype=torch.int32, device=device
    )
    batch_indices, positions = flashinfer.get_batch_indices_positions(
        kv_append_indptr,
        flashinfer.get_seq_lens(kv_page_indptr, kv_last_page_len, page_size),
        nnz_kv,
    )
    return (
        nnz_kv,
        max_num_pages,
        kv_page_indices,
        kv_page_indptr,
        kv_last_page_len,
        batch_indices,
        positions,
    )


@pytest.mark.parametrize("kv_len", kv_len_configs)
@pytest.mark.parametrize("page_size", [1, 16, 64])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("permute_pages", [False, True])
def test_nvfp4_quantize_append_paged_mla_kv_cache(
    kv_len, page_size, dtype, permute_pages
):
    torch.manual_seed(42)
    device = "cuda:0"
    (
        nnz_kv,
        max_num_pages,
        kv_page_indices,
        kv_page_indptr,
        kv_last_page_len,
        batch_indices,
        positions,
    ) = _build_mla_paged_inputs(kv_len, page_size, permute_pages, device)

    ckv_append = torch.randn(nnz_kv, CKV_DIM, dtype=dtype, device=device)
    kpe_append = 4.0 * torch.randn(nnz_kv, KPE_DIM, dtype=dtype, device=device)
    ckv_scale = float(ckv_append.abs().amax()) / (448.0 * 6.0)
    kpe_scale = float(kpe_append.abs().amax()) / 448.0

    ckv_cache = torch.zeros(
        max_num_pages, page_size, CKV_DIM // 2, dtype=torch.uint8, device=device
    )
    ckv_sf_cache = torch.zeros(
        max_num_pages,
        page_size,
        CKV_DIM // 16,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    kpe_cache = torch.zeros(
        max_num_pages, page_size, KPE_DIM, dtype=torch.float8_e4m3fn, device=device
    )

    flashinfer.nvfp4_quantize_append_paged_mla_kv_cache(
        ckv_append,
        kpe_append,
        batch_indices,
        positions,
        ckv_cache,
        ckv_sf_cache,
        kpe_cache,
        kv_page_indices,
        kv_page_indptr,
        kv_last_page_len,
        ckv_scale,
        kpe_scale,
    )

    ref_packed, ref_sf = _reference_nvfp4_quantize_ckv(ckv_append, ckv_scale)
    ref_kpe = _reference_fp8_quantize_kpe(kpe_append, kpe_scale)

    page_iters = kv_page_indptr[batch_indices].long() + positions.long() // page_size
    pages = kv_page_indices[page_iters].long()
    entries = positions.long() % page_size

    torch.testing.assert_close(ckv_cache[pages, entries], ref_packed, rtol=0, atol=0)
    torch.testing.assert_close(
        ckv_sf_cache[pages, entries].view(torch.uint8),
        ref_sf.view(torch.uint8),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        kpe_cache[pages, entries].view(torch.uint8),
        ref_kpe.view(torch.uint8),
        rtol=0,
        atol=0,
    )

    written = torch.zeros(max_num_pages, page_size, dtype=torch.bool, device=device)
    written[pages, entries] = True
    assert bool((ckv_cache[~written] == 0).all())
    assert bool((kpe_cache[~written].view(torch.uint8) == 0).all())


def test_nvfp4_quantize_append_paged_mla_kv_cache_dequant_close():
    _skip_if_fp8_e4m3_unsupported()
    torch.manual_seed(0)
    device = "cuda:0"
    kv_len, page_size = [45, 8, 25], 16
    (
        nnz_kv,
        max_num_pages,
        kv_page_indices,
        kv_page_indptr,
        kv_last_page_len,
        batch_indices,
        positions,
    ) = _build_mla_paged_inputs(kv_len, page_size, True, device)

    ckv_append = torch.randn(nnz_kv, CKV_DIM, dtype=torch.bfloat16, device=device)
    kpe_append = torch.randn(nnz_kv, KPE_DIM, dtype=torch.bfloat16, device=device)
    ckv_scale = float(ckv_append.abs().amax()) / (448.0 * 6.0)
    kpe_scale = float(kpe_append.abs().amax()) / 448.0

    ckv_cache = torch.zeros(
        max_num_pages, page_size, CKV_DIM // 2, dtype=torch.uint8, device=device
    )
    ckv_sf_cache = torch.zeros(
        max_num_pages,
        page_size,
        CKV_DIM // 16,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    kpe_cache = torch.zeros(
        max_num_pages, page_size, KPE_DIM, dtype=torch.float8_e4m3fn, device=device
    )
    flashinfer.nvfp4_quantize_append_paged_mla_kv_cache(
        ckv_append,
        kpe_append,
        batch_indices,
        positions,
        ckv_cache,
        ckv_sf_cache,
        kpe_cache,
        kv_page_indices,
        kv_page_indptr,
        kv_last_page_len,
        ckv_scale,
        kpe_scale,
    )

    page_iters = kv_page_indptr[batch_indices].long() + positions.long() // page_size
    pages = kv_page_indices[page_iters].long()
    entries = positions.long() % page_size

    # dequant via the shared utility pins the cache to the repo-wide NVFP4 format
    ckv_dequant = e2m1_and_ufp8sf_scale_to_float(
        ckv_cache[pages, entries],
        ckv_sf_cache[pages, entries].view(torch.uint8).reshape(-1),
        torch.tensor([ckv_scale], dtype=torch.float32, device=device),
        sf_vec_size=16,
        is_sf_swizzled_layout=False,
    ).to(device)
    rel = (ckv_dequant - ckv_append.float()).norm() / ckv_append.float().norm()
    assert rel < 0.15

    e2m1_lut = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32, device=device
    )
    packed = ckv_cache[pages, entries]
    codes = torch.stack([packed & 0xF, packed >> 4], dim=-1).reshape(nnz_kv, CKV_DIM)
    mags = e2m1_lut[(codes & 0x7).long()]
    signs = torch.where((codes & 0x8) > 0, -1.0, 1.0)
    sf = ckv_sf_cache[pages, entries].float().repeat_interleave(16, dim=-1)
    ckv_dequant_manual = signs * mags * sf * ckv_scale
    torch.testing.assert_close(ckv_dequant, ckv_dequant_manual, rtol=1e-6, atol=1e-8)

    kpe_dequant = kpe_cache[pages, entries].float() * kpe_scale
    rel_kpe = (kpe_dequant - kpe_append.float()).norm() / kpe_append.float().norm()
    assert rel_kpe < 0.05


@pytest.mark.parametrize("bad_scale", [0.0, -1.0, float("inf"), float("nan")])
def test_nvfp4_quantize_append_paged_mla_kv_cache_rejects_bad_scale(bad_scale):
    _skip_if_fp8_e4m3_unsupported()
    device = "cuda:0"
    (
        nnz_kv,
        max_num_pages,
        kv_page_indices,
        kv_page_indptr,
        kv_last_page_len,
        batch_indices,
        positions,
    ) = _build_mla_paged_inputs([4], 4, False, device)
    ckv_append = torch.randn(nnz_kv, CKV_DIM, dtype=torch.bfloat16, device=device)
    kpe_append = torch.randn(nnz_kv, KPE_DIM, dtype=torch.bfloat16, device=device)
    ckv_cache = torch.zeros(
        max_num_pages, 4, CKV_DIM // 2, dtype=torch.uint8, device=device
    )
    ckv_sf_cache = torch.zeros(
        max_num_pages, 4, CKV_DIM // 16, dtype=torch.float8_e4m3fn, device=device
    )
    kpe_cache = torch.zeros(
        max_num_pages, 4, KPE_DIM, dtype=torch.float8_e4m3fn, device=device
    )
    with pytest.raises(ValueError):
        flashinfer.nvfp4_quantize_append_paged_mla_kv_cache(
            ckv_append,
            kpe_append,
            batch_indices,
            positions,
            ckv_cache,
            ckv_sf_cache,
            kpe_cache,
            kv_page_indices,
            kv_page_indptr,
            kv_last_page_len,
            bad_scale,
            1.0,
        )


def test_dense_mla_decode_rejects_packed_uint8_cache():
    _skip_if_fp8_e4m3_unsupported()
    device = "cuda:0"
    query = torch.randn(1, 1, 128, CKV_DIM + KPE_DIM, device=device).to(
        torch.float8_e4m3fn
    )
    kv_cache = torch.zeros(1, 64, 352, dtype=torch.uint8, device=device)
    workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    block_tables = torch.zeros(1, 1, dtype=torch.int32, device=device)
    seq_lens = torch.ones(1, dtype=torch.int32, device=device)
    with pytest.raises(NotImplementedError, match="NVFP4"):
        flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
            query=query,
            kv_cache=kv_cache,
            workspace_buffer=workspace,
            qk_nope_head_dim=128,
            kv_lora_rank=CKV_DIM,
            qk_rope_head_dim=KPE_DIM,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=64,
        )


if __name__ == "__main__":
    test_append_mla_paged_kv_cache([45], 64)
    test_append_mla_paged_kv_cache([4096], 64)
    test_append_mla_paged_kv_cache([45, 8, 25], 64)
    test_append_mla_paged_kv_cache([45, 8, 25, 22], 64)
    test_append_mla_paged_kv_cache([45, 8, 25, 22, 400], 128)
    test_append_mla_paged_kv_cache([45, 8, 25, 22, 100], 16)
