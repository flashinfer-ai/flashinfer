import pytest
import torch

import flashinfer
from flashinfer.utils import is_sm90a_supported


@pytest.mark.parametrize("backend", ["fa2", "fa3"])
def test_mla_page_index_uint32_overflow_regression(backend):
    # Regression for the int64 widening in mla.cuh / mla_hopper.cuh
    # (`indices[q] * ckv_stride_page`). For a contiguous
    # [num_pages, page_size, head_dim_ckv] cache with page_size=32 and
    # head_dim_ckv=512, ckv_stride_page = 16384 elements. Any page index
    # >= 2^32 / 16384 = 262144 makes the multiplication overflow uint32 and
    # — pre-fix — silently wraps to the wrong page (no crash, wrong output).
    device = torch.device("cuda:0")
    if backend == "fa3" and not is_sm90a_supported(device):
        pytest.skip("fa3 backend requires SM90a")

    page_size, head_dim_ckv, head_dim_kpe, num_heads = 32, 512, 64, 128
    # 262144 * (32 * 512) = 2^32 exactly — the smallest index that overflows.
    OVERFLOW_START = 262144
    NUM_PAGES = 26  # matches the 26-page decode scenario from the original repro
    total_num_pages = OVERFLOW_START + NUM_PAGES  # 262170
    kv_len = NUM_PAGES * page_size

    # Big cache alone is ~9.66 GiB (bf16/fp16). Skip on small-memory runners.
    if torch.cuda.mem_get_info(device)[0] < 12 * (1 << 30):
        pytest.skip("needs ≥12 GiB free VRAM to force the 32-bit overflow")

    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    dtype = torch.float16
    sm_scale = 1.0 / ((128 + 64) ** 0.5)

    real_ckv = torch.randn(
        NUM_PAGES, page_size, head_dim_ckv, device=device, dtype=dtype
    )
    real_kpe = torch.randn(
        NUM_PAGES, page_size, head_dim_kpe, device=device, dtype=dtype
    )
    q_nope = torch.randn(1, num_heads, head_dim_ckv, device=device, dtype=dtype)
    q_pe = torch.randn(1, num_heads, head_dim_kpe, device=device, dtype=dtype)
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)

    def _run(ckv_cache, kpe_cache, page_indices):
        w = flashinfer.mla.BatchMLAPagedAttentionWrapper(workspace, backend=backend)
        w.plan(
            torch.tensor([0, 1], dtype=torch.int32, device=device),  # qo_indptr
            torch.tensor([0, len(page_indices)], dtype=torch.int32, device=device),
            page_indices,
            torch.tensor([kv_len], dtype=torch.int32, device=device),
            num_heads,
            head_dim_ckv,
            head_dim_kpe,
            page_size,
            False,
            sm_scale,
            dtype,
            dtype,
        )
        return w.run(q_nope, q_pe, ckv_cache, kpe_cache)

    # Overflow path: big contiguous cache; real data lives at [OVERFLOW_START, end).
    # stride(0) = page_size * head_dim_ckv = 16384 matches the reference below,
    # so only the page-index arithmetic differs between the two runs.
    ckv_big = torch.zeros(
        total_num_pages, page_size, head_dim_ckv, device=device, dtype=dtype
    )
    kpe_big = torch.zeros(
        total_num_pages, page_size, head_dim_kpe, device=device, dtype=dtype
    )
    ckv_big[OVERFLOW_START:] = real_ckv
    kpe_big[OVERFLOW_START:] = real_kpe
    big_indices = torch.arange(
        OVERFLOW_START, total_num_pages, dtype=torch.int32, device=device
    )
    out = _run(ckv_big, kpe_big, big_indices)
    del ckv_big, kpe_big
    torch.cuda.empty_cache()

    # Reference: same data, same stride(0), but page indices < overflow threshold.
    ref_indices = torch.arange(NUM_PAGES, dtype=torch.int32, device=device)
    ref = _run(real_ckv, real_kpe, ref_indices)

    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)
