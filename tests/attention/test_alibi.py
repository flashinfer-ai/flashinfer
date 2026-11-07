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

import pytest
import torch
from tests.test_helpers.alibi_reference import alibi_attention
from tests.test_helpers.jit_utils import (
    gen_decode_attention_modules,
    gen_prefill_attention_modules,
)

import flashinfer
from flashinfer.utils import has_flashinfer_jit_cache


@pytest.fixture(
    autouse=not has_flashinfer_jit_cache(),
    scope="module",
)
def warmup_jit():
    flashinfer.jit.build_jit_specs(
        gen_decode_attention_modules(
            [torch.float16],  # q_dtypes
            [torch.float16],  # kv_dtypes
            [128, 256],  # head_dims
            [0, 2],  # pos_encoding_modes
            [False],  # use_sliding_windows
            [False],  # use_logits_soft_caps
        )
        + gen_prefill_attention_modules(
            [torch.float16],  # q_dtypes
            [torch.float16],  # kv_dtypes
            [128, 256],  # head_dims
            [0, 2],  # pos_encoding_modes
            [False],  # use_sliding_windows
            [False],  # use_logits_soft_caps
            [False],  # use_fp16_qk_reductions
        )
        + gen_decode_attention_modules(
            [torch.float16],  # q_dtypes
            [torch.float16],  # kv_dtypes
            [128],  # head_dims
            [2],  # pos_encoding_modes
            [False],  # use_sliding_windows
            [True],  # use_logits_soft_caps
        )
        + gen_prefill_attention_modules(
            [torch.float16],  # q_dtypes
            [torch.float16],  # kv_dtypes
            [128],  # head_dims
            [2],  # pos_encoding_modes
            [False],  # use_sliding_windows
            [True],  # use_logits_soft_caps
            [False],  # use_fp16_qk_reductions
        ),
        verbose=False,
    )
    yield


@pytest.mark.parametrize("seq_len", [1, 9, 81, 729])
@pytest.mark.parametrize("num_heads", [4, 8, 32])
@pytest.mark.parametrize("head_dim", [128, 256])
def test_single_decode_alibi(
    seq_len,
    num_heads,
    head_dim,
):
    q = torch.randn(num_heads, head_dim, device="cuda:0", dtype=torch.float16)
    k = torch.randn(seq_len, num_heads, head_dim, device="cuda:0", dtype=torch.float16)
    v = torch.randn(seq_len, num_heads, head_dim, device="cuda:0", dtype=torch.float16)

    o = flashinfer.single_decode_with_kv_cache(q, k, v, pos_encoding_mode="ALIBI")
    mask = torch.ones(1, seq_len, dtype=torch.bool, device="cuda:0")
    o_ref = alibi_attention(q.unsqueeze(0), k, v, mask).squeeze(0)
    torch.testing.assert_close(o, o_ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("q_len", [1, 17, 81, 987])
@pytest.mark.parametrize("kv_len", [1, 17, 81, 987])
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
    q = torch.randn(q_len, num_heads, head_dim, device="cuda:0", dtype=torch.float16)
    k = torch.randn(kv_len, num_heads, head_dim, device="cuda:0", dtype=torch.float16)
    v = torch.randn(kv_len, num_heads, head_dim, device="cuda:0", dtype=torch.float16)

    o = flashinfer.single_prefill_with_kv_cache(
        q, k, v, causal=causal, pos_encoding_mode="ALIBI"
    )
    mask = torch.ones(q_len, kv_len, dtype=torch.bool, device="cuda:0")
    if causal:
        mask = torch.tril(mask, diagonal=kv_len - q_len)
    o_ref = alibi_attention(q, k, v, mask)
    torch.testing.assert_close(o, o_ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("attention_mode", ["decode", "prefill"])
def test_alibi_logits_soft_cap_sm_scale_invariance(attention_mode):
    head_dim = 128
    sm_scale = 0.125
    q = torch.ones(1, head_dim, device="cuda:0", dtype=torch.float16)
    k = torch.stack([torch.zeros_like(q), torch.full_like(q, 0.5)])
    v = torch.stack([torch.zeros_like(q), torch.ones_like(q)])
    kwargs = {
        "pos_encoding_mode": "ALIBI",
        "logits_soft_cap": 8.0,
    }

    if attention_mode == "decode":
        attention = flashinfer.single_decode_with_kv_cache
    else:
        attention = flashinfer.single_prefill_with_kv_cache
        q = q.unsqueeze(0)
        kwargs["backend"] = "fa2"

    o_scaled_logits = attention(q, k, v, sm_scale=sm_scale, **kwargs)
    o_scaled_query = attention(q * sm_scale, k, v, sm_scale=1.0, **kwargs)

    torch.testing.assert_close(o_scaled_logits, o_scaled_query, rtol=0, atol=0)


if __name__ == "__main__":
    test_single_decode_alibi(4096, 32, 128)
    test_single_prefill_alibi(128, 128, 8, 128, False)
