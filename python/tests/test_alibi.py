"""
Copyright (c) 2023 by FlashInfer team.

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

import numpy
import pytest
import torch

import flashinfer

from alibi_reference import alibi_attention


@pytest.mark.parametrize("seq_len", [1, 9, 81, 729, 33001])
@pytest.mark.parametrize("num_heads", [4, 8, 32])
@pytest.mark.parametrize("head_dim", [128, 256])
def test_single_decode_alibi(
    seq_len,
    num_heads,
    head_dim,
):
    q = torch.randn(num_heads, head_dim).to(0).half()
    k = torch.randn(seq_len, num_heads, head_dim).to(0).half()
    v = torch.randn(seq_len, num_heads, head_dim).to(0).half()

    o = flashinfer.single_decode_with_kv_cache(q, k, v, pos_encoding_mode="ALIBI")
    mask = torch.ones(1, seq_len, dtype=torch.bool).to(0)
    o_ref = alibi_attention(q.unsqueeze(0), k, v, mask).squeeze(0)
    numpy.testing.assert_allclose(
        o.cpu().numpy(), o_ref.cpu().numpy(), rtol=1e-3, atol=1e-3
    )


@pytest.mark.parametrize("q_len", [1, 17, 81, 987])
@pytest.mark.parametrize("kv_len", [1, 17, 81, 987, 31111])
@pytest.mark.parametrize("num_heads", [4, 8, 32])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("causal", [False, True])
def test_single_prefill_alibi(
    q_len,
    kv_len,
    num_heads,
    head_dim,
    causal,
):
    if causal and q_len > kv_len:
        pytest.skip("Causal attention requires q_len <= kv_len")
    q = torch.randn(q_len, num_heads, head_dim).to(0).half()
    k = torch.randn(kv_len, num_heads, head_dim).to(0).half()
    v = torch.randn(kv_len, num_heads, head_dim).to(0).half()

    o = flashinfer.single_prefill_with_kv_cache(
        q, k, v, causal=causal, pos_encoding_mode="ALIBI"
    )
    mask = torch.ones(q_len, kv_len, dtype=torch.bool).to(0)
    if causal:
        mask = torch.tril(mask, diagonal=kv_len - q_len)
    o_ref = alibi_attention(q, k, v, mask)
    numpy.testing.assert_allclose(
        o.cpu().numpy(), o_ref.cpu().numpy(), rtol=1e-2, atol=1e-2
    )


if __name__ == "__main__":
    test_single_decode_alibi(9, 32, 128)
    test_single_prefill_alibi(1, 64, 1, 128, False)
