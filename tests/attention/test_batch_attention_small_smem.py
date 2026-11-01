"""Regression test for the persistent BatchAttention grid on small-smem GPUs.

``BatchAttention`` launches a cooperative persistent grid sized from the SM
count.  On GPUs with 100 KiB shared memory per SM (SM86/SM89/SM120),
``head_dim=128`` needs 69,632 B per CTA, so two CTAs cannot be co-resident and
the planned 2 x SM grid fails with ``cudaErrorCooperativeLaunchTooLarge`` even
for a single query.  The planner must fall back to one CTA per SM there while
keeping two CTAs per SM for ``head_dim=64``.
"""

import pytest
import torch

import flashinfer


def _max_smem_per_sm(device: torch.device) -> int:
    return int(
        torch.cuda.get_device_properties(device).shared_memory_per_multiprocessor
    )


@pytest.mark.parametrize("head_dim", [64, 128])
def test_batch_attention_persistent_grid_small_smem(head_dim: int):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    device = torch.device("cuda")
    if _max_smem_per_sm(device) < 100 * 1024:
        pytest.skip("persistent BatchAttention requires at least 100 KiB smem/SM")

    dtype = torch.bfloat16
    page_size = 8
    num_qo_heads, num_kv_heads = 8, 2
    qo_len, kv_len = 1, 1

    q_indptr = torch.tensor([0, qo_len], dtype=torch.int32, device=device)
    kv_indptr = torch.tensor([0, 1], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([kv_len], dtype=torch.int32, device=device)

    q = torch.randn(qo_len, num_qo_heads, head_dim, dtype=dtype, device=device)
    kv = torch.randn(
        1, 2, page_size, num_kv_heads, head_dim, dtype=dtype, device=device
    )

    wrapper = flashinfer.BatchAttention(kv_layout="NHD")
    wrapper.plan(
        q_indptr,
        kv_indptr,
        torch.arange(1, dtype=torch.int32, device=device),
        seq_lens,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        head_dim,
        page_size,
        causal=False,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    out, lse = wrapper.run(q, kv)
    torch.cuda.synchronize()

    assert out.shape == (qo_len, num_qo_heads, head_dim)
    assert torch.isfinite(out).all()
    assert torch.isfinite(lse).all()

    # With a single KV token, softmax has weight one, so each query head
    # returns the V vector of its GQA group.
    group_size = num_qo_heads // num_kv_heads
    v = kv[0, 1, 0]
    ref = v.repeat_interleave(group_size, dim=0)
    torch.testing.assert_close(out[0], ref, rtol=1e-2, atol=1e-2)
