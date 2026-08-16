"""
Copyright (c) 2026 by FlashInfer team.

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

import flashinfer

LN2 = 0.6931471805599453


def _torch_attn_lse(q, k, v, causal, sm_scale):
    """fp32 torch-native attention + natural-log LSE. q [T,H,Dqk], k/v [S,H,*]."""
    qf = q.float().permute(1, 0, 2)
    kf = k.float().permute(1, 0, 2)
    vf = v.float().permute(1, 0, 2)
    scores = qf @ kf.transpose(-1, -2) * sm_scale
    if causal:
        T_, S = scores.shape[-2:]
        mask = torch.triu(
            torch.ones(T_, S, dtype=torch.bool, device=q.device), diagonal=S - T_ + 1
        )
        scores = scores.masked_fill(mask, float("-inf"))
    lse = torch.logsumexp(scores, dim=-1).permute(1, 0)
    out = (torch.softmax(scores, dim=-1) @ vf).permute(1, 0, 2)
    return out, lse


def _merge_state(o1, lse1, o2, lse2):
    """fp32 natural-log merge of two attention states (out, natural-log LSE)."""
    m = torch.maximum(lse1, lse2)
    e1 = torch.exp(lse1 - m).unsqueeze(-1)
    e2 = torch.exp(lse2 - m).unsqueeze(-1)
    return (o1.float() * e1 + o2.float() * e2) / (e1 + e2)


@pytest.mark.parametrize("prefix_len", [512, 4096])
def test_ragged_lse_base_on_e(prefix_len):
    """forward_return_lse yields base-2 LSE by default and natural-log LSE with
    return_lse_base_on_e=True; only the natural-log form merges correctly with
    base-e merge consumers (e.g. chunked-prefix attention merges)."""
    torch.manual_seed(0)
    dev = "cuda"
    num_heads, head_dim_qk, head_dim_vo = 16, 192, 128
    extend_len = 16
    sm_scale = head_dim_qk**-0.5

    q = torch.randn(extend_len, num_heads, head_dim_qk, dtype=torch.bfloat16, device=dev)
    k_pre = torch.randn(prefix_len, num_heads, head_dim_qk, dtype=torch.bfloat16, device=dev)
    v_pre = torch.randn(prefix_len, num_heads, head_dim_vo, dtype=torch.bfloat16, device=dev)
    k_ext = torch.randn(extend_len, num_heads, head_dim_qk, dtype=torch.bfloat16, device=dev)
    v_ext = torch.randn(extend_len, num_heads, head_dim_vo, dtype=torch.bfloat16, device=dev)

    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=dev)

    def make_wrapper():
        return flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(
            workspace, "NHD", backend="cutlass"
        )

    def begin(w, qo_tokens, kv_tokens):
        w.begin_forward(
            qo_indptr=torch.tensor([0, qo_tokens], dtype=torch.int32, device=dev),
            kv_indptr=torch.tensor([0, kv_tokens], dtype=torch.int32, device=dev),
            num_qo_heads=num_heads,
            num_kv_heads=num_heads,
            head_dim_qk=head_dim_qk,
            head_dim_vo=head_dim_vo,
            q_data_type=torch.bfloat16,
        )

    # Exact reference: one ragged call over the full sequence.
    w = make_wrapper()
    begin(w, extend_len, prefix_len + extend_len)
    k_full = torch.cat([k_pre, k_ext], 0)
    v_full = torch.cat([v_pre, v_ext], 0)
    o_ref, _ = w.forward_return_lse(q, k_full, v_full, causal=True, sm_scale=sm_scale)

    # Extend part (causal) and prefix chunk (non-causal), both with base-e LSE.
    w1 = make_wrapper()
    begin(w1, extend_len, extend_len)
    o_ext, lse_ext = w1.forward_return_lse(
        q, k_ext, v_ext, causal=True, sm_scale=sm_scale, return_lse_base_on_e=True
    )
    w2 = make_wrapper()
    begin(w2, extend_len, prefix_len)
    o_pre, lse_pre = w2.forward_return_lse(
        q, k_pre, v_pre, causal=False, sm_scale=sm_scale, return_lse_base_on_e=True
    )

    # 1) default (no flag): LSE must be base-2.
    _, lse_ext_default = w1.forward_return_lse(
        q, k_ext, v_ext, causal=True, sm_scale=sm_scale
    )
    _, lse_ext_true = _torch_attn_lse(q, k_ext, v_ext, True, sm_scale)
    default_vs_ln = (lse_ext_default - lse_ext_true).abs().max().item()
    default_vs_log2 = (lse_ext_default - lse_ext_true * math.log2(math.e)).abs().max().item()
    assert default_vs_log2 < 1e-3 and default_vs_ln > 0.1, (
        f"default LSE should be base-2 (vs log2 err={default_vs_log2:.2e}, "
        f"vs ln err={default_vs_ln:.2e})"
    )

    # 2) with the flag: LSE must be natural-log.
    flag_vs_ln = (lse_ext - lse_ext_true).abs().max().item()
    assert flag_vs_ln < 1e-2, f"flagged LSE should be natural-log (err={flag_vs_ln:.2e})"

    # 3) merge with natural-log LSE matches the full-sequence result; merging the
    #    raw base-2 LSE with a base-e merge consumer does not.
    o_merged_good = _merge_state(o_pre, lse_pre, o_ext, lse_ext)
    err_good = (o_merged_good - o_ref.float()).abs().max().item()
    o_merged_bad = _merge_state(o_pre, lse_pre / LN2, o_ext, lse_ext / LN2)
    err_bad = (o_merged_bad - o_ref.float()).abs().max().item()
    assert err_good < 1e-2, f"base-e merge should be exact (err={err_good:.2e})"
    assert err_bad > 10 * err_good, (
        f"base-2 LSE with a base-e merge should be visibly wrong "
        f"(err_bad={err_bad:.2e}, err_good={err_good:.2e})"
    )


if __name__ == "__main__":
    test_ragged_lse_base_on_e(512)
    test_ragged_lse_base_on_e(4096)
    print("PASS")
