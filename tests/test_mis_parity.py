"""Parity test: MIS optimized kernel vs custom_mask reference.

Verifies that the MIS attention kernel (smem binary search) produces
identical results to an explicit custom mask implementation.
"""

import numpy as np
import pytest
import torch

import flashinfer
from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper


def build_custom_mask(prefix_len, item_lens, qo_len, kv_len):
    """Build explicit attention mask equivalent to MIS semantics."""
    offsets = [0]
    for l in item_lens:
        offsets.append(offsets[-1] + l)

    mask = torch.zeros(qo_len, kv_len, dtype=torch.bool, device="cuda")
    for q_idx in range(qo_len):
        kv_pos = prefix_len + q_idx
        for j in range(len(offsets) - 1):
            if offsets[j] <= q_idx < offsets[j + 1]:
                item_start_rel = offsets[j]
                break
        mask[q_idx, :prefix_len] = True
        kv_item_start = prefix_len + item_start_rel
        mask[q_idx, kv_item_start : kv_pos + 1] = True

    return mask.unsqueeze(0).reshape(-1)


def run_parity_check(
    prefix_len,
    item_lens,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    page_size,
    dtype=torch.float16,
    auto_max_item_len=False,
):
    total_items_tokens = sum(item_lens)
    qo_len = total_items_tokens
    kv_len = prefix_len + total_items_tokens

    offsets = [0]
    for l in item_lens:
        offsets.append(offsets[-1] + l)

    prefix_len_ptr = torch.tensor([prefix_len], dtype=torch.uint32, device="cuda")
    item_offsets = torch.tensor(offsets, dtype=torch.uint32, device="cuda")
    item_offsets_len = len(offsets)
    max_item_len_ptr = (
        None if auto_max_item_len
        else torch.tensor([max(item_lens)], dtype=torch.uint16, device="cuda")
    )

    torch.manual_seed(42)
    q = torch.randn(qo_len, num_qo_heads, head_dim, device="cuda", dtype=dtype)

    num_pages = (kv_len + page_size - 1) // page_size
    kv_data = torch.randn(
        num_pages, 2, page_size, num_kv_heads, head_dim, device="cuda", dtype=dtype
    )

    k_full = kv_data[:, 0].reshape(-1, num_kv_heads, head_dim)[:kv_len]
    v_full = kv_data[:, 1].reshape(-1, num_kv_heads, head_dim)[:kv_len]

    q_indptr = torch.tensor([0, qo_len], dtype=torch.int32, device="cuda")
    kv_indptr = torch.tensor([0, num_pages], dtype=torch.int32, device="cuda")
    kv_indices = torch.arange(num_pages, dtype=torch.int32, device="cuda")
    kv_last_page = torch.tensor(
        [(kv_len - 1) % page_size + 1], dtype=torch.int32, device="cuda"
    )
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda")

    wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD")
    wrapper.plan(
        q_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=True,
        pos_encoding_mode="NONE",
        prefix_len_ptr=prefix_len_ptr,
        item_offsets=item_offsets,
        item_offsets_len=item_offsets_len,
        max_item_len_ptr=max_item_len_ptr,
    )
    out_mis = wrapper.run(q, kv_data)

    custom_mask = build_custom_mask(prefix_len, item_lens, qo_len, kv_len)
    out_ref = flashinfer.prefill.single_prefill_with_kv_cache(
        q,
        k_full,
        v_full,
        causal=True,
        custom_mask=custom_mask,
    )

    max_diff = (out_mis - out_ref).abs().max().item()
    ref_scale = out_ref.abs().mean().item()
    rel_max = max_diff / (ref_scale + 1e-8)

    return rel_max


RNG = np.random.RandomState(123)

CONFIGS = [
    (256, [10] * 5),
    (256, [10] * 50),
    (256, [10] * 100),
    (512, [25] * 50),
    (512, [50] * 100),
    (1024, [128] * 10),
    (2048, [256] * 5),
    (4096, [512] * 4),
    (256, [1] * 50),
    (256, [2] * 100),
    (512, [20, 30, 25, 35, 40]),
    (256, [15, 25, 20, 30, 10] * 10),
    (512, list(range(20, 45, 5)) * 10),
    (1024, [40, 60, 50, 70, 55] * 10),
    (512, [25, 35, 30, 45, 20] * 20),
    (256, RNG.randint(10, 40, size=20).tolist()),
    (512, RNG.randint(15, 50, size=30).tolist()),
    (1024, RNG.randint(20, 60, size=50).tolist()),
    (512, sorted(RNG.randint(15, 45, size=20).tolist())),
]


@pytest.mark.parametrize("prefix_len, item_lens", CONFIGS)
def test_mis_parity(prefix_len, item_lens):
    rel_max = run_parity_check(
        prefix_len,
        item_lens,
        num_qo_heads=16,
        num_kv_heads=8,
        head_dim=128,
        page_size=16,
    )
    assert rel_max < 0.02, f"Relative max diff {rel_max:.6f} exceeds 2% threshold"


AUTO_MAX_CONFIGS = [
    (256, [10] * 5),
    (512, [20, 30, 25, 35, 40]),
    (256, [15, 25, 20, 30, 10] * 10),
    (1024, [128] * 10),
]


@pytest.mark.parametrize("prefix_len, item_lens", AUTO_MAX_CONFIGS)
def test_mis_parity_auto_max_item_len(prefix_len, item_lens):
    """Verify the auto-computed max_item_len_ptr path in plan()."""
    rel_max = run_parity_check(
        prefix_len,
        item_lens,
        num_qo_heads=16,
        num_kv_heads=8,
        head_dim=128,
        page_size=16,
        auto_max_item_len=True,
    )
    assert rel_max < 0.02, f"Relative max diff {rel_max:.6f} exceeds 2% threshold"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
