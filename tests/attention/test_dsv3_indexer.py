"""
Tests for MQA histogram top-K indexer.

Compares the FlashInfer fused and non-fused implementations against a
pure-PyTorch reference (dsa_topk_indexer).

Requires SM100a (Blackwell) GPU.
"""

import pytest
import torch

from flashinfer.dsv3_ops import mqa_topk_indexer
from flashinfer.utils import is_sm100a_supported

torch.manual_seed(0)


def _make_kv_cache(num_pages: int) -> torch.Tensor:
    """Build a random FP8 KV cache [num_pages, 64, 1, 132] uint8."""
    k_index_cache_fp8 = torch.empty(
        num_pages, 64, 1, 132, dtype=torch.uint8, device="cuda"
    )
    kv_flat = k_index_cache_fp8.view(num_pages, -1)
    kv_flat[:, : 64 * 128].view(torch.float8_e4m3fn).copy_(
        torch.randn(num_pages, 64 * 128, dtype=torch.float32, device="cuda").to(
            torch.float8_e4m3fn
        )
    )
    kv_flat[:, 64 * 128 :].view(torch.float32).copy_(
        torch.randn(num_pages, 64, dtype=torch.float32, device="cuda").abs()
    )
    return kv_flat.view(num_pages, 64, 1, 132)


def _make_dsa_test_data(batch_size: int, seq_len_range):
    """Build random test inputs for the MQA indexer.

    Returns:
        (q, k_cache, weights, seq_lens, block_table)
    """
    lo, hi = seq_len_range
    seq_lens = torch.randint(
        lo, hi + 1, (batch_size,), dtype=torch.int32, device="cuda"
    )
    num_pages_per_seq = ((seq_lens + 64 - 1) // 64).sum().item()
    max_num_pages = int((int(seq_lens.max().item()) + 64 - 1) // 64)
    # ensure max_num_pages is divisible by 2 (kernel requirement)
    max_num_pages = (max_num_pages + 1) // 2 * 2

    q = torch.randn(batch_size, 64, 128, dtype=torch.float32, device="cuda").to(
        torch.float8_e4m3fn
    )
    k_cache = _make_kv_cache(int(num_pages_per_seq))
    weights = torch.randn(batch_size, 64, dtype=torch.float32, device="cuda")
    block_table = torch.zeros(
        batch_size, max_num_pages, dtype=torch.int32, device="cuda"
    )

    page_offset = 0
    for b in range(batch_size):
        n = int((int(seq_lens[b].item()) + 64 - 1) // 64)
        block_table[b, :n] = torch.arange(
            page_offset, page_offset + n, dtype=torch.int32, device="cuda"
        )
        page_offset += n

    return q, k_cache, weights, seq_lens, block_table


# ---------------------------------------------------------------------------
# Reference implementation (pure PyTorch)
# Inlined from kernels_exp/indexer_ref.py
# ---------------------------------------------------------------------------


def _dequant_fp8_kv_cache(k_index_cache_fp8: torch.Tensor) -> torch.Tensor:
    """Dequantize FP8 KV cache [num_pages, 64, 1, 132] → [num_pages, 64, 128] float32."""
    k = k_index_cache_fp8.view(torch.uint8)
    num_pages, page_size, _, _ = k.shape
    head_dim = 128
    kv_flat = k.view(num_pages, page_size * 132)
    fp8_bytes = kv_flat[:, : page_size * head_dim].contiguous()
    fp8_tensor = fp8_bytes.view(num_pages, page_size, head_dim).view(
        torch.float8_e4m3fn
    )
    fp8_float = fp8_tensor.to(torch.float32)
    scale_bytes = kv_flat[:, page_size * head_dim :].contiguous()
    scale = scale_bytes.view(num_pages, page_size, 4).view(torch.float32)
    return fp8_float * scale


@torch.no_grad()
def _dsa_topk_indexer(q_fp8, k_cache_fp8, weights, seq_lens, block_table):
    """Pure-PyTorch reference implementation of the MQA top-K indexer."""
    batch_size, num_heads, head_dim = q_fp8.shape
    page_size = 64
    topk = 2048

    device = q_fp8.device
    q = q_fp8.to(torch.float32)
    K_all = _dequant_fp8_kv_cache(k_cache_fp8)  # [num_pages, page_size, head_dim]

    topk_indices = torch.full((batch_size, topk), -1, dtype=torch.int32, device=device)
    logits = []

    for b in range(batch_size):
        seq_len = int(seq_lens[b].item())
        if seq_len == 0:
            logits.append(torch.zeros(0, device=device))
            continue

        num_pages_for_seq = (seq_len + page_size - 1) // page_size
        page_indices = block_table[b, :num_pages_for_seq].to(torch.long)

        K_paged = K_all[page_indices]  # [num_pages_for_seq, page_size, head_dim]
        K = K_paged.reshape(-1, head_dim)[:seq_len]  # [seq_len, head_dim]

        q_b = q[b]  # [num_heads, head_dim]
        scores = q_b @ K.T  # [num_heads, seq_len]
        scores_relu = torch.relu(scores)
        w = weights[b]  # [num_heads]
        final_scores = (scores_relu * w[:, None]).sum(dim=0)  # [seq_len]

        logits.append(final_scores)

        actual_topk = min(topk, seq_len)
        _, topk_idx = torch.topk(final_scores, actual_topk)
        topk_indices[b, :actual_topk] = topk_idx.to(torch.int32)

    return topk_indices, logits


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("B", [1, 2, 4, 32, 64])
@pytest.mark.parametrize(
    "L", [31, 1024, 4096, 8192, 4096 * 8, 4096 * 10, 1 << 17, 1 << 18]
)
def test_mqa_topk_indexer(B, L):
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("Requires SM100a (Blackwell)")

    lo = int(L * 0.7)
    hi = min(1 << 19, int(L * 1.3))
    test_data = _make_dsa_test_data(B, (lo, hi))

    q, kv_cache, weights, seq_lens, block_table = test_data
    indices_ref, logits_ref = _dsa_topk_indexer(*test_data)
    indices_1, logits_1 = mqa_topk_indexer(
        q, kv_cache, weights, seq_lens, block_table, max_model_len=hi
    )

    for i in range(B):
        prefix = f"B={B}, L={L}, batch {i}"
        l = int(seq_lens[i])

        logit_ref = logits_ref[i][:l]
        logit_1 = logits_1[i][:l]

        diff1 = float((logit_1 - logit_ref).abs().max())
        assert diff1 <= 1, f"{prefix}: fused logit max-diff {diff1}"

        inds_ref = indices_ref[i][indices_ref[i] >= 0]
        inds_1 = indices_1[i][indices_1[i] >= 0]

        assert (inds_ref < l).all(), f"{prefix}: ref indices out of range"
        assert (inds_1 < l).all(), f"{prefix}: fused indices out of range"

        ref_topk = torch.sort(logit_ref[inds_ref])[0]
        topk_1 = torch.sort(logit_1[inds_1])[0]

        topk_diff = float((ref_topk - topk_1).abs().max())
        assert topk_diff <= 0.5, f"{prefix}: fused topk max-diff {topk_diff}"
