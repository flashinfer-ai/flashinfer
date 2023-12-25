import numpy
import pytest
import torch

import flashinfer


@pytest.mark.parametrize("batch_size", [12, 17])
@pytest.mark.parametrize("unique_kv_len", [37, 17])
@pytest.mark.parametrize("shared_kv_len", [54, 97])
@pytest.mark.parametrize("num_heads", [8, 16])
@pytest.mark.parametrize("head_dim", [128])
def test_batch_decode_with_shared_prefix_padded_kv_cache(
    batch_size, unique_kv_len, shared_kv_len, num_heads, head_dim
):
    q = torch.randn(batch_size, num_heads, head_dim).to(0).half()
    k_shared = torch.randn(shared_kv_len, num_heads, head_dim).to(0).half()
    v_shared = torch.randn(shared_kv_len, num_heads, head_dim).to(0).half()
    k_unique = torch.randn(batch_size, unique_kv_len, num_heads, head_dim).to(0).half()
    v_unique = torch.randn(batch_size, unique_kv_len, num_heads, head_dim).to(0).half()

    o = flashinfer.ops.batch_decode_with_shared_prefix_padded_kv_cache(
        q, k_shared, v_shared, k_unique, v_unique
    )

    for i in range(batch_size):
        qi = q[i]
        ki = torch.cat([k_shared, k_unique[i]], dim=0)
        vi = torch.cat([v_shared, v_unique[i]], dim=0)
        o_ref_i = flashinfer.ops.single_decode_with_kv_cache(qi, ki, vi)
        o_i_np = o[i].cpu().numpy()
        o_ref_i_np = o_ref_i.cpu().numpy()
        numpy.testing.assert_allclose(o_i_np, o_ref_i_np, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    test_batch_decode_with_shared_prefix_padded_kv_cache(12, 37, 54, 8, 128)
