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
from alibi_reference import alibi_attention
from jit_utils import jit_decode_attention_func_args, jit_prefill_attention_func_args

import flashinfer


@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    if flashinfer.jit.has_prebuilt_ops:
        yield
    else:
        try:
            flashinfer.jit.parallel_load_modules(
                jit_decode_attention_func_args(
                    [torch.float16],  # q_dtypes
                    [torch.float16],  # kv_dtypes
                    [128, 256],  # head_dims
                    [0, 2],  # pos_encoding_modes
                    [False],  # use_sliding_windows
                    [False],  # use_logits_soft_caps
                )
                + jit_prefill_attention_func_args(
                    [torch.float16],  # q_dtypes
                    [torch.float16],  # kv_dtypes
                    [128, 256],  # head_dims
                    [0, 2],  # pos_encoding_modes
                    [False],  # use_sliding_windows
                    [False],  # use_logits_soft_caps
                    [False],  # use_fp16_qk_reductions
                )
            )
        except Exception as e:
            # abort the test session if warmup fails
            pytest.exit(str(e))
        finally:
            yield


@pytest.mark.parametrize("seq_len", [1, 9, 81, 729])
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
    torch.testing.assert_close(o, o_ref, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    test_single_decode_alibi(4096, 32, 128)
    test_single_prefill_alibi(128, 128, 8, 128, False)
