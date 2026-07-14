"""Regression test for FlashInfer issue #3578: FP8-Q sliding-window prefill hang
on Hopper (SM90).

Root cause (confirmed by repro + source read): the FP8 consumer ``mma_fp8``
(attention/hopper/quantization/mainloop_mma.cuh) was missing the
``LEFT_SLIDING_WINDOW`` drain loop that the BF16 ``mma_f16`` has
(attention/hopper/mainloop_mma.cuh, added in PR #673). The FP8 producer loads
K/V down to ``swa_begin`` while the consumer stopped releasing pipeline stages
at ``swa_end + 1``, leaving the ``pipeline_k``/``pipeline_vt`` stages in the
``(swa_begin, swa_end]`` range unconsumed; the warp-specialized producer then
blocks on ``producer_acquire`` and ``cudaStreamSynchronize`` never returns
(reproduces at seq_len 257 with the 256-wide tiling).

The hang is reachable through the ragged and paged batch prefill wrappers, which
pass ``window_left`` into the shared ``mma_fp8`` consumer. The single-prefill API
(``single_prefill_with_kv_cache``) guards FP8 with ``assert window_left == -1``,
so it does not expose this path. These tests run under ``pytest.mark.timeout`` so
the deadlock surfaces as a test FAILURE rather than hanging CI, and check numerics
against the fp16 reference path (``mma_f16``, which already contains the drain
loop).
"""

from typing import Tuple

import pytest
import torch

import flashinfer
from flashinfer.utils import is_sm90a_supported

E4M3_MAX = 448.0
DT = torch.float8_e4m3fn
HQ, HKV, D = 32, 8, 128  # GQA, head_dim 128
SEQ_LENS = [257, 512, 1024]  # 257 = first hanging size at the 256-wide tiling
WINDOWS = [
    128,
    256,
    511,
]  # 128/256 truncate (swa_begin>0) for long seq; 511 ~ causal here


def per_head_symmetric_quant(
    x: torch.Tensor, quant_dtype: torch.dtype = DT
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-head symmetric FP8 quant (matches test_hopper_fp8_attention.py)."""
    x_max = x.abs().amax(dim=(0, 2)).to(torch.float32)
    s = torch.clamp(x_max / E4M3_MAX, min=1e-6)
    q = torch.clamp(x / s.view(1, -1, 1), min=-E4M3_MAX, max=E4M3_MAX).to(quant_dtype)
    return q, s


def _ws():
    return torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")


@pytest.mark.timeout(300)  # > cold-cache JIT compile; a real hang is immediate
@pytest.mark.parametrize("seq_len", SEQ_LENS)
@pytest.mark.parametrize("window_left", WINDOWS)
def test_fp8_ragged_prefill_sliding_window(seq_len, window_left):
    if not is_sm90a_supported(torch.device("cuda")):
        pytest.skip("SM90A is not supported")
    torch.manual_seed(0)
    qo_indptr = torch.tensor([0, seq_len], dtype=torch.int32, device="cuda")
    kv_indptr = torch.tensor([0, seq_len], dtype=torch.int32, device="cuda")

    q = torch.randn(seq_len, HQ, D, dtype=torch.half, device="cuda")
    k = torch.randn(seq_len, HKV, D, dtype=torch.half, device="cuda")
    v = torch.randn(seq_len, HKV, D, dtype=torch.half, device="cuda")

    ref_w = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(_ws(), "NHD", backend="fa3")
    ref_w.plan(
        qo_indptr, kv_indptr, HQ, HKV, D, D, causal=True, window_left=window_left
    )
    o_ref = ref_w.run(q, k, v)

    q8, sq = per_head_symmetric_quant(q)
    k8, sk = per_head_symmetric_quant(k)
    v8, sv = per_head_symmetric_quant(v)
    fp8_w = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(_ws(), "NHD", backend="fa3")
    fp8_w.plan(
        qo_indptr,
        kv_indptr,
        HQ,
        HKV,
        D,
        D,
        causal=True,
        window_left=window_left,
        q_data_type=DT,
        kv_data_type=DT,
        o_data_type=torch.half,
    )
    o_fp8 = fp8_w.run(q8, k8, v8, sq, sk, sv)
    torch.cuda.synchronize()  # without the fix the kernel deadlocks here

    mse = torch.mean((o_ref.float() - o_fp8.float()) ** 2).item()
    assert mse < 1.0, f"MSE too high: {mse}"


@pytest.mark.timeout(300)  # > cold-cache JIT compile; a real hang is immediate
@pytest.mark.parametrize("seq_len", SEQ_LENS)
@pytest.mark.parametrize("window_left", WINDOWS)
def test_fp8_paged_prefill_sliding_window(seq_len, window_left):
    if not is_sm90a_supported(torch.device("cuda")):
        pytest.skip("SM90A is not supported")
    torch.manual_seed(0)
    page_size = 32
    num_pages = (seq_len + page_size - 1) // page_size
    last = seq_len % page_size or page_size

    qo_indptr = torch.tensor([0, seq_len], dtype=torch.int32, device="cuda")
    kv_indptr = torch.tensor([0, num_pages], dtype=torch.int32, device="cuda")
    kv_indices = torch.arange(num_pages, dtype=torch.int32, device="cuda")
    kv_last = torch.tensor([last], dtype=torch.int32, device="cuda")

    q = torch.randn(seq_len, HQ, D, dtype=torch.half, device="cuda")
    pk = torch.randn(num_pages, page_size, HKV, D, dtype=torch.half, device="cuda")
    pv = torch.randn(num_pages, page_size, HKV, D, dtype=torch.half, device="cuda")

    ref_w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(_ws(), "NHD", backend="fa3")
    ref_w.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last,
        HQ,
        HKV,
        D,
        page_size,
        causal=True,
        window_left=window_left,
    )
    o_ref = ref_w.run(q, (pk, pv))

    q8, sq = per_head_symmetric_quant(q)
    k8, sk = per_head_symmetric_quant(pk.view(-1, HKV, D))
    v8, sv = per_head_symmetric_quant(pv.view(-1, HKV, D))
    k8 = k8.view(num_pages, page_size, HKV, D)
    v8 = v8.view(num_pages, page_size, HKV, D)
    fp8_w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(_ws(), "NHD", backend="fa3")
    fp8_w.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last,
        HQ,
        HKV,
        D,
        page_size,
        causal=True,
        window_left=window_left,
        q_data_type=DT,
        kv_data_type=DT,
        o_data_type=torch.half,
    )
    o_fp8 = fp8_w.run(q8, (k8, v8), sq, sk, sv)
    torch.cuda.synchronize()  # without the fix the kernel deadlocks here

    mse = torch.mean((o_ref.float() - o_fp8.float()) ** 2).item()
    assert mse < 1.0, f"MSE too high: {mse}"
