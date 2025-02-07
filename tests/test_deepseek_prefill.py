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

import flashinfer


def attention_ref(
    batch_size,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    sm_scale: float,
) -> torch.Tensor:
    qo_len = q.shape[0] // batch_size
    kv_len = k.shape[0] // batch_size
    num_qo_heads = q.shape[1]
    head_dim_qk = q.shape[2]
    head_dim_vo = v.shape[2]
    logits = (
        torch.einsum(
            "bmhd,bnhd->bhmn",
            q.view(batch_size, qo_len, num_qo_heads, head_dim_qk).float(),
            k.view(batch_size, kv_len, num_qo_heads, head_dim_qk).float(),
        )
        * sm_scale
    )

    if causal:
        mask = (
            torch.arange(kv_len - qo_len, kv_len).unsqueeze(1)
            >= torch.arange(0, kv_len).unsqueeze(0)
        ).to(q.device)
    else:
        mask = torch.ones(qo_len, kv_len).to(q.device)

    logits = logits.masked_fill(mask.unsqueeze(0).unsqueeze(0) == 0, float("-inf"))
    p = torch.softmax(logits, dim=-1)
    o_ref = (
        torch.einsum(
            "bhmn,bnhd->bmhd",
            p,
            v.view(batch_size, kv_len, num_qo_heads, head_dim_vo).float(),
        )
        .contiguous()
        .view(batch_size * qo_len, num_qo_heads, head_dim_vo)
        .to(q)
    )

    return o_ref


@pytest.mark.parametrize("kv_len", [5532, 7563])
@pytest.mark.parametrize("qo_len", [1832, 3928])
@pytest.mark.parametrize("num_kv_heads", [4, 32])
@pytest.mark.parametrize("num_qo_heads", [32])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("backend", ["fa2", "fa3"])
def test_single_prefill_with_kv_cache(
    kv_len,
    qo_len,
    num_kv_heads,
    num_qo_heads,
    causal,
    backend,
):
    head_dim_qk = 192
    head_dim_vo = 128
    q = torch.randn(qo_len, num_qo_heads, head_dim_qk).to(0).half()
    k = torch.zeros(kv_len, num_kv_heads, head_dim_qk).to(0).half()
    v = torch.randn(kv_len, num_kv_heads, head_dim_vo).to(0).half()

    o = flashinfer.single_prefill_with_kv_cache(q, k, v, causal=causal, backend=backend)

    sm_scale = 1.0 / (head_dim_qk**0.5)

    if num_qo_heads != num_kv_heads:
        k = k.repeat_interleave(num_qo_heads // num_kv_heads, dim=1)
        v = v.repeat_interleave(num_qo_heads // num_kv_heads, dim=1)

    o_ref = attention_ref(1, q, k, v, causal, sm_scale)
    torch.testing.assert_close(o, o_ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [12, 17])
@pytest.mark.parametrize("kv_len", [544, 977])
@pytest.mark.parametrize("qo_len", [377, 177])
@pytest.mark.parametrize("num_kv_heads", [4, 32])
@pytest.mark.parametrize("num_qo_heads", [32])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("backend", ["fa2", "fa3"])
def test_batch_prefill_with_ragged_kv_cache(
    batch_size,
    kv_len,
    qo_len,
    num_kv_heads,
    num_qo_heads,
    causal,
    backend,
):
    kv_layout = "NHD"
    head_dim_qk = 192
    head_dim_vo = 128
    q = (
        torch.randn(batch_size * qo_len, num_qo_heads, head_dim_qk).to(0).bfloat16()
    )  # half()
    q_indptr = torch.arange(0, batch_size + 1).to(0).int() * qo_len

    k = (
        torch.zeros(batch_size * kv_len, num_kv_heads, head_dim_qk).to(0).bfloat16()
    )  # half()
    v = (
        torch.randn(batch_size * kv_len, num_kv_heads, head_dim_vo).to(0).bfloat16()
    )  # half()
    kv_indptr = torch.arange(0, batch_size + 1).to(0).int() * kv_len

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8).to(0)
    wrapper = flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(
        workspace_buffer, kv_layout, backend=backend
    )
    wrapper.plan(
        q_indptr,
        kv_indptr,
        num_qo_heads,
        num_kv_heads,
        head_dim_qk,
        head_dim_vo=head_dim_vo,
        causal=causal,
        q_data_type=q.dtype,
        kv_data_type=k.dtype,
    )
    o = wrapper.run(q, k, v)

    sm_scale = 1.0 / (head_dim_qk**0.5)
    if num_qo_heads != num_kv_heads:
        k = k.repeat_interleave(num_qo_heads // num_kv_heads, dim=1)
        v = v.repeat_interleave(num_qo_heads // num_kv_heads, dim=1)

    o_ref = attention_ref(batch_size, q, k, v, causal, sm_scale)

    torch.testing.assert_close(o, o_ref, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    test_single_prefill_with_kv_cache(54, 37, 4, 32, False, "fa2")
    test_batch_prefill_with_ragged_kv_cache(12, 54, 37, 4, 4, False, "fa2")
