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

import math

import pytest
import torch
from jit_utils import gen_decode_attention_modules, gen_prefill_attention_modules

import flashinfer


@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    flashinfer.jit.build_jit_specs(
        gen_decode_attention_modules(
            [torch.float16],  # q_dtypes
            [torch.float16],  # kv_dtypes
            [128, 256],  # head_dims
            [0],  # pos_encoding_modes
            [False],  # use_sliding_windows
            [False, True],  # use_logits_soft_caps
        )
        + gen_prefill_attention_modules(
            [torch.float16],  # q_dtypes
            [torch.float16],  # kv_dtypes
            [128, 256],  # head_dims
            [0],  # pos_encoding_modes
            [False],  # use_sliding_windows
            [False, True],  # use_logits_soft_caps
            [False],  # use_fp16_qk_reductions
        ),
        verbose=False,
    )
    yield


def attention_logits_soft_cap_torch(q, k, v, soft_cap):
    q_len, num_heads, head_dim = q.shape
    scores = torch.einsum("qhd,khd->qkh", q.float(), k.float())
    scores *= 1.0 / math.sqrt(head_dim)
    scores = soft_cap * torch.tanh(scores / soft_cap)
    attn = torch.softmax(scores, dim=1)
    return torch.einsum("ovh,vhd->ohd", attn, v.float()).to(q)


@pytest.mark.parametrize("seq_len", [1, 9, 81, 729, 33001])
@pytest.mark.parametrize("num_heads", [4, 8, 32])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("soft_cap", [1.0, 30.0, 50.0])
def test_single_decode_logits_soft_cap(
    seq_len,
    num_heads,
    head_dim,
    soft_cap,
):
    q = torch.randn(num_heads, head_dim, device="cuda:0", dtype=torch.float16)
    k = torch.randn(seq_len, num_heads, head_dim, device="cuda:0", dtype=torch.float16)
    v = torch.randn(seq_len, num_heads, head_dim, device="cuda:0", dtype=torch.float16)

    o = flashinfer.single_decode_with_kv_cache(q, k, v, logits_soft_cap=soft_cap)
    o_ref = attention_logits_soft_cap_torch(q.unsqueeze(0), k, v, soft_cap).squeeze(0)
    torch.testing.assert_close(o, o_ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("q_len", [1, 17, 81, 987])
@pytest.mark.parametrize("kv_len", [1, 17, 81, 987, 31111])
@pytest.mark.parametrize("num_heads", [4, 8, 32])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("soft_cap", [1.0, 30.0, 50.0])
def test_single_prefill_logits_soft_cap(
    q_len,
    kv_len,
    num_heads,
    head_dim,
    soft_cap,
):
    q = torch.randn(q_len, num_heads, head_dim, device="cuda:0", dtype=torch.float16)
    k = torch.randn(kv_len, num_heads, head_dim, device="cuda:0", dtype=torch.float16)
    v = torch.randn(kv_len, num_heads, head_dim, device="cuda:0", dtype=torch.float16)

    o = flashinfer.single_prefill_with_kv_cache(q, k, v, logits_soft_cap=soft_cap)
    o_ref = attention_logits_soft_cap_torch(q, k, v, soft_cap)
    torch.testing.assert_close(o, o_ref, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    test_single_decode_logits_soft_cap(9, 32, 128, 30.0)
    test_single_prefill_logits_soft_cap(64, 64, 1, 128, 30.0)
