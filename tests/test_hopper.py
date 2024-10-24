"""
Copyright (c) 2024 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import torch
import flashinfer
import flashinfer._prefill_sm90
import pytest

@pytest.mark.parametrize("seq_len", [11, 99, 1763, 9999])
@pytest.mark.parametrize("num_qo_heads", [1, 4, 8])
@pytest.mark.parametrize("num_kv_heads", [1, 4, 8])
@pytest.mark.parametrize("causal", [False])
@pytest.mark.parametrize("head_dim", [256])
def test_single_prefill(seq_len, num_qo_heads, num_kv_heads, causal, head_dim):
    if num_qo_heads % num_kv_heads != 0:
        pytest.skip("num_qo_heads must be divisible by num_kv_heads")
    q = torch.randn(seq_len, num_qo_heads, head_dim, dtype=torch.half, device="cuda")
    k = torch.randn(seq_len, num_kv_heads, head_dim, dtype=torch.half, device="cuda")
    v = torch.randn(seq_len, num_kv_heads, head_dim, dtype=torch.half, device="cuda")

    o_sm80, lse_sm80 = flashinfer.single_prefill_with_kv_cache_return_lse(q, k, v, causal=causal)
    sm_scale = 1.0 / (head_dim ** 0.5)
    o_sm90 = torch.zeros_like(q)
    o_sm90, lse_sm90 = flashinfer._prefill_sm90.single_prefill_with_kv_cache(
        q, k, v, o_sm90, causal, sm_scale
    )
    torch.testing.assert_close(lse_sm80.transpose(-1,-2).contiguous(), lse_sm90, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(o_sm80, o_sm90, rtol=1e-3, atol=1e-3)

def bench_single_prefill(seq_len):
    head_dim = 128
    num_qo_heads = num_kv_heads = 32    
    q = torch.randn(seq_len, num_qo_heads, head_dim, dtype=torch.half, device="cuda")
    k = torch.randn(seq_len, num_kv_heads, head_dim, dtype=torch.half, device="cuda")
    v = torch.randn(seq_len, num_kv_heads, head_dim, dtype=torch.half, device="cuda")

    sm_scale = 1.0 / (head_dim ** 0.5)

    import triton
    ms0 = triton.testing.do_bench(
        lambda: flashinfer.single_prefill_with_kv_cache_return_lse(q, k, v, causal=False), warmup=100, rep=1000
    )
    o = torch.zeros_like(q)
    ms = triton.testing.do_bench(
        lambda: flashinfer._prefill_sm90.single_prefill_with_kv_cache(
            q, k, v, o, False, sm_scale
        ), warmup=100, rep=1000
    )
    print(seq_len * seq_len * num_qo_heads * head_dim * 4 / ms0 / 1e9, seq_len * seq_len * num_qo_heads * head_dim * 4 / ms / 1e9)


if __name__ == "__main__":
    test_single_prefill(5, 8, 8, False, 128)
    # bench_single_prefill(16384)
