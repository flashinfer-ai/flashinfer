import pytest
import torch

import flashinfer


def _reference(q, k, v, sm_scale):
    # q: [H, D], k/v: [L, H, D] (NHD layout)
    s = torch.einsum("hd,lhd->hl", q.float(), k.float()) * sm_scale
    p = torch.softmax(s, dim=-1)
    return torch.einsum("hl,lhd->hd", p, v.float())


@pytest.mark.parametrize("kv_len", [257, 1024])
@pytest.mark.parametrize("head_multiplier", [1, 4])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_single_decode_num_kv_heads_exceeds_max_grid(
    kv_len, head_multiplier, head_dim, dtype
):
    """Regression test: num_kv_heads larger than blocks_per_sm * num_sm with
    kv_len > 256 (split-KV path) used to hit an integer divide by zero (SIGFPE)."""
    num_sm = torch.cuda.get_device_properties(0).multi_processor_count
    num_kv_heads = num_sm * head_multiplier + 4

    q = torch.randn(num_kv_heads, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(kv_len, num_kv_heads, head_dim, device="cuda", dtype=dtype)
    v = torch.randn_like(k)

    out = flashinfer.single_decode_with_kv_cache(q, k, v)
    ref = _reference(q, k, v, 1.0 / head_dim**0.5)

    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)
