"""Focused FP8 KV accuracy coverage for Gemma 4's full-attention shape.

These tests exercise the Ampere FA2 large-head path enabled for FP8 KV cache:
16 query heads, 2 KV heads, and head dimension 512.  They use an independent
PyTorch FP32 oracle over the exact dequantized FP8 cache instead of comparing
one FlashInfer kernel with another.
"""

import math

import pytest
import torch

import flashinfer
from flashinfer.utils import get_compute_capability


DEVICE = torch.device("cuda:0")
Q_DTYPE = torch.bfloat16
KV_DTYPE = torch.float8_e4m3fn
NUM_QO_HEADS = 16
NUM_KV_HEADS = 2
HEAD_DIM = 512
PAGE_SIZE = 16
KV_LEN = 10_003
SM_SCALE = 1.0
K_SCALE = 0.02
V_SCALE = 0.02
LOG2_E = math.log2(math.e)


def _require_ampere_or_newer() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if get_compute_capability(DEVICE)[0] < 8:
        pytest.skip("FP8 KV head_dim=512 FA2 coverage requires SM80 or newer")


def _make_inputs(
    q_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    torch.manual_seed(42)
    num_pages = (KV_LEN + PAGE_SIZE - 1) // PAGE_SIZE

    # Gemma applies its query scaling before calling the attention backend.
    q = (
        torch.randn(
            q_len,
            NUM_QO_HEADS,
            HEAD_DIM,
            dtype=Q_DTYPE,
            device=DEVICE,
        )
        / 16
    )
    cache_shape = (num_pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM)
    k_cache = (torch.randn(cache_shape, dtype=Q_DTYPE, device=DEVICE) / K_SCALE).to(
        KV_DTYPE
    )
    v_cache = (torch.randn(cache_shape, dtype=Q_DTYPE, device=DEVICE) / V_SCALE).to(
        KV_DTYPE
    )
    return q, k_cache, v_cache, num_pages


def _reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return FP32 output and base-2 LSE for the causal query chunk."""
    q_len = q.shape[0]
    group_size = NUM_QO_HEADS // NUM_KV_HEADS
    k = k_cache.reshape(-1, NUM_KV_HEADS, HEAD_DIM)[:KV_LEN].float() * K_SCALE
    v = v_cache.reshape(-1, NUM_KV_HEADS, HEAD_DIM)[:KV_LEN].float() * V_SCALE

    q_positions = torch.arange(q_len, device=DEVICE) + KV_LEN - q_len
    k_positions = torch.arange(KV_LEN, device=DEVICE)
    causal_mask = k_positions.unsqueeze(0) <= q_positions.unsqueeze(1)

    ref_out = torch.empty(
        q_len, NUM_QO_HEADS, HEAD_DIM, dtype=torch.float32, device=DEVICE
    )
    ref_lse = torch.empty(q_len, NUM_QO_HEADS, dtype=torch.float32, device=DEVICE)

    for kv_head in range(NUM_KV_HEADS):
        head_lo = kv_head * group_size
        head_hi = head_lo + group_size
        q_group = q[:, head_lo:head_hi].float().permute(1, 0, 2)
        scores = torch.matmul(q_group, k[:, kv_head].t()) * SM_SCALE
        scores.masked_fill_(~causal_mask.unsqueeze(0), float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        ref_out[:, head_lo:head_hi] = torch.matmul(
            probs, v[:, kv_head].unsqueeze(0)
        ).permute(1, 0, 2)
        ref_lse[:, head_lo:head_hi] = (
            torch.logsumexp(scores, dim=-1).permute(1, 0) * LOG2_E
        )

    return ref_out, ref_lse


def _assert_matches_reference(
    actual_out: torch.Tensor,
    actual_lse: torch.Tensor,
    ref_out: torch.Tensor,
    ref_lse: torch.Tensor,
) -> None:
    expected_out = ref_out.to(Q_DTYPE)
    out_abs = (actual_out.float() - expected_out.float()).abs()
    lse_abs = (actual_lse.float() - ref_lse.float()).abs()
    diagnostics = (
        f"output max={out_abs.max().item():.6g}, "
        f"p99={torch.quantile(out_abs, 0.99).item():.6g}; "
        f"LSE max={lse_abs.max().item():.6g}, "
        f"p99={torch.quantile(lse_abs, 0.99).item():.6g}"
    )
    torch.testing.assert_close(
        actual_out,
        expected_out,
        rtol=2e-2,
        atol=2e-2,
        msg=diagnostics,
    )
    torch.testing.assert_close(
        actual_lse.float(),
        ref_lse,
        rtol=1e-3,
        atol=2e-2,
        msg=diagnostics,
    )


def _paged_metadata(
    q_len: int, num_pages: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    qo_indptr = torch.tensor([0, q_len], dtype=torch.int32, device=DEVICE)
    kv_indptr = torch.tensor([0, num_pages], dtype=torch.int32, device=DEVICE)
    kv_indices = torch.arange(num_pages, dtype=torch.int32, device=DEVICE)
    kv_last_page_len = torch.tensor(
        [(KV_LEN - 1) % PAGE_SIZE + 1], dtype=torch.int32, device=DEVICE
    )
    return qo_indptr, kv_indptr, kv_indices, kv_last_page_len


def test_gemma4_fp8_kv_head_dim_512_chunked_prefill_matches_torch() -> None:
    _require_ampere_or_newer()
    q_len = 17
    q, k_cache, v_cache, num_pages = _make_inputs(q_len)
    qo_indptr, kv_indptr, kv_indices, kv_last_page_len = _paged_metadata(
        q_len, num_pages
    )

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace, kv_layout="NHD", backend="fa2"
    )
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        NUM_QO_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        PAGE_SIZE,
        causal=True,
        sm_scale=SM_SCALE,
        q_data_type=Q_DTYPE,
        kv_data_type=KV_DTYPE,
    )
    out, lse = wrapper.run(
        q,
        (k_cache, v_cache),
        k_scale=K_SCALE,
        v_scale=V_SCALE,
        return_lse=True,
    )

    ref_out, ref_lse = _reference(q, k_cache, v_cache)
    _assert_matches_reference(out, lse, ref_out, ref_lse)


def test_gemma4_fp8_kv_head_dim_512_tensor_core_decode_matches_torch() -> None:
    _require_ampere_or_newer()
    q, k_cache, v_cache, num_pages = _make_inputs(1)
    _, kv_indptr, kv_indices, kv_last_page_len = _paged_metadata(1, num_pages)

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace,
        kv_layout="NHD",
        use_tensor_cores=True,
        backend="fa2",
    )
    wrapper.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        NUM_QO_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        PAGE_SIZE,
        pos_encoding_mode="NONE",
        logits_soft_cap=0.0,
        sm_scale=SM_SCALE,
        q_data_type=Q_DTYPE,
        kv_data_type=KV_DTYPE,
    )
    out, lse = wrapper.run(
        q,
        (k_cache, v_cache),
        k_scale=K_SCALE,
        v_scale=V_SCALE,
        return_lse=True,
    )

    ref_out, ref_lse = _reference(q, k_cache, v_cache)
    _assert_matches_reference(out, lse, ref_out, ref_lse)
