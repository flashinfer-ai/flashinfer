"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

------------------------------------------------------------------------------
Correctness tests for the single-node MoE-LoRA delta builders
(``flashinfer.fused_moe.bgmv_moe_gemm1_lora_delta`` / ``...gemm2_lora_delta``).

These directly exercise the two new BGMV kernel modes added for this feature:
  * the expand kernel's ``finalize=False`` mode (per-pair, unweighted, plain store) — via FC1;
  * the shrink kernel's ``per_pair_input=True`` mode (per-pair activation read) — via FC2.

End-to-end wiring is covered by ``test_trtllm_gen_fused_moe.py::test_moe_lora_delta``.
"""

import os

os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")

import pytest
import torch

from flashinfer.fused_moe import (
    bgmv_moe_gemm1_lora_delta,
    bgmv_moe_gemm2_lora_delta,
    fill_w_ptr,
)

# BGMV MoE kernels are validated on SM90 (H100) and SM100/103 (Blackwell).
_SUPPORTED_SM = {90, 100, 103}


def _skip_if_unsupported_sm():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    major, minor = torch.cuda.get_device_capability()
    sm = major * 10 + minor
    if sm not in _SUPPORTED_SM:
        pytest.skip(f"BGMV MoE kernels not validated on SM{sm}")


def _build_w_ptr(weights, num_experts):
    """Build the [num_slices, num_experts] int64 base-pointer table + shared lora_stride
    from a list of LoRA weight banks ([max_loras, num_experts, *, *]) via fill_w_ptr.
    (The builders take w_ptr/stride as inputs; weight management lives here in the test.)"""
    device = weights[0].device
    w_ptr = torch.zeros(len(weights), num_experts, dtype=torch.int64, device=device)
    stride = 0
    for s, w in enumerate(weights):
        stride = fill_w_ptr(w_ptr, w, num_experts, s)
    return w_ptr, stride


# ============================================================
# PyTorch references (float32 accumulation)
# ============================================================


def _ref_fc1_delta(hidden, lora_a, lora_b, topk_ids, lora_ids, scale):
    """delta[t, j, :] = scale * concat_s( B_s[l,e] @ (A_s[l,e] @ x[t]) ), unweighted."""
    T, _ = hidden.shape
    k = topk_ids.shape[1]
    inter = lora_b[0].shape[2]
    out = torch.zeros(T, k, 2 * inter, dtype=torch.float32, device=hidden.device)
    hf = hidden.float()
    for t in range(T):
        lid = int(lora_ids[t])
        if lid < 0:
            continue
        for j in range(k):
            e = int(topk_ids[t, j])
            for s in range(2):
                a = lora_a[s][lid, e].float()  # [r, H]
                b = lora_b[s][lid, e].float()  # [I, r]
                out[t, j, s * inter : (s + 1) * inter] = scale * (b @ (a @ hf[t]))
    return out


def _ref_fc1_delta_fused(hidden, a_shared, b_fused, topk_ids, lora_ids, scale):
    """delta[t, j, :] = scale * B_fused[l,e] @ (A_shared[l,e] @ x[t]), unweighted."""
    T, _ = hidden.shape
    k = topk_ids.shape[1]
    two_i = b_fused.shape[2]
    out = torch.zeros(T, k, two_i, dtype=torch.float32, device=hidden.device)
    hf = hidden.float()
    for t in range(T):
        lid = int(lora_ids[t])
        if lid < 0:
            continue
        for j in range(k):
            e = int(topk_ids[t, j])
            a = a_shared[lid, e].float()  # [r, H]
            b = b_fused[lid, e].float()  # [2I, r]
            out[t, j] = scale * (b @ (a @ hf[t]))
    return out


def _ref_fc2_delta(
    act_perm, perm, lora_a, lora_b, topk_ids, topk_weights, lora_ids, scale
):
    """delta[t] = scale * sum_j w[t,j] * ( B[l,e] @ (A[l,e] @ a[t,j]) ),
    where a[t,j] = act_perm[perm[t*k+j]]."""
    T, k = topk_ids.shape
    hidden = lora_b[0].shape[2]
    out = torch.zeros(T, hidden, dtype=torch.float32, device=act_perm.device)
    af = act_perm.float()
    for t in range(T):
        lid = int(lora_ids[t])
        if lid < 0:
            continue
        for j in range(k):
            e = int(topk_ids[t, j])
            w = float(topk_weights[t, j])
            p = int(perm[t * k + j])
            a = lora_a[0][lid, e].float()  # [r, I]
            b = lora_b[0][lid, e].float()  # [H, r]
            out[t] += scale * w * (b @ (a @ af[p]))
    return out


def _make_routing(T, k, num_experts, max_loras, device):
    torch.manual_seed(0)
    # distinct experts per token (no duplicate expert within a token's top-k)
    topk_ids = torch.empty(T, k, dtype=torch.int64, device=device)
    for t in range(T):
        topk_ids[t] = torch.randperm(num_experts, device=device)[:k]
    topk_weights = torch.softmax(torch.randn(T, k, device=device), dim=-1)
    lora_ids = torch.randint(-1, max_loras, (T,), dtype=torch.int64, device=device)
    lora_ids[: max(1, T // 2)] = torch.randint(
        0, max_loras, (max(1, T // 2),), dtype=torch.int64, device=device
    )
    return topk_ids, topk_weights, lora_ids


@pytest.mark.parametrize("T", [4, 16])
@pytest.mark.parametrize("hidden", [768])
@pytest.mark.parametrize("inter", [768])
@pytest.mark.parametrize("rank", [16, 32])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("performant", [False, True])
def test_gemm1_lora_delta(T, hidden, inter, rank, dtype, performant):
    """Regular (2-slice, 2D w_ptr) and performant (shared-A + fused-B, 1D w_ptr) FC1 delta."""
    _skip_if_unsupported_sm()
    device = torch.device("cuda")
    torch.manual_seed(1)
    num_experts, k, max_loras = 8, 2, 4
    scale = 0.5

    hidden_states = torch.randn(T, hidden, dtype=dtype, device=device) * 0.1
    topk_ids, _, lora_ids = _make_routing(T, k, num_experts, max_loras, device)

    if performant:
        a_shared = (
            torch.randn(
                max_loras, num_experts, rank, hidden, dtype=dtype, device=device
            )
            * 0.02
        )
        b_fused = (
            torch.randn(
                max_loras, num_experts, 2 * inter, rank, dtype=dtype, device=device
            )
            * 0.02
        )
        w_ptr_a, stride_a = _build_w_ptr([a_shared], num_experts)
        w_ptr_b, stride_b = _build_w_ptr([b_fused], num_experts)
        w_ptr_a, w_ptr_b = (
            w_ptr_a.reshape(-1),
            w_ptr_b.reshape(-1),
        )  # drop slice dim -> [E]
        ref = _ref_fc1_delta_fused(
            hidden_states, a_shared, b_fused, topk_ids, lora_ids, scale
        )
    else:
        lora_a = [
            torch.randn(
                max_loras, num_experts, rank, hidden, dtype=dtype, device=device
            )
            * 0.02
            for _ in range(2)
        ]
        lora_b = [
            torch.randn(max_loras, num_experts, inter, rank, dtype=dtype, device=device)
            * 0.02
            for _ in range(2)
        ]
        w_ptr_a, stride_a = _build_w_ptr(lora_a, num_experts)
        w_ptr_b, stride_b = _build_w_ptr(lora_b, num_experts)
        ref = _ref_fc1_delta(hidden_states, lora_a, lora_b, topk_ids, lora_ids, scale)

    out = bgmv_moe_gemm1_lora_delta(
        hidden_states,
        w_ptr_a,
        stride_a,
        w_ptr_b,
        stride_b,
        topk_ids,
        lora_ids,
        rank,
        inter,
        lora_dtype=dtype,
        scale=scale,
        out_dtype=dtype,
    )

    assert out.shape == (T, k, 2 * inter)
    torch.testing.assert_close(out.float(), ref, atol=1e-2, rtol=1e-2)
    # no-adapter tokens must produce an exactly-zero delta (skip path stores 0)
    for t in range(T):
        if int(lora_ids[t]) < 0:
            assert out[t].abs().max().item() == 0.0


@pytest.mark.parametrize("T", [4, 16])
@pytest.mark.parametrize("hidden", [768])
@pytest.mark.parametrize("inter", [768])
@pytest.mark.parametrize("rank", [16, 32])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_gemm2_lora_delta(T, hidden, inter, rank, dtype):
    _skip_if_unsupported_sm()
    device = torch.device("cuda")
    torch.manual_seed(2)
    num_experts, k, max_loras = 8, 2, 4
    scale = 0.5
    P = T * k

    lora_a = [
        torch.randn(max_loras, num_experts, rank, inter, dtype=dtype, device=device)
        * 0.02
    ]
    lora_b = [
        torch.randn(max_loras, num_experts, hidden, rank, dtype=dtype, device=device)
        * 0.02
    ]
    topk_ids, topk_weights, lora_ids = _make_routing(
        T, k, num_experts, max_loras, device
    )

    # Synthetic permuted post-activation + a permutation map (all slots active).
    act_perm = torch.randn(P, inter, dtype=dtype, device=device) * 0.1
    perm = torch.randperm(P, dtype=torch.int64, device=device)

    w_ptr_a, stride_a = _build_w_ptr(lora_a, num_experts)
    w_ptr_b, stride_b = _build_w_ptr(lora_b, num_experts)
    out = bgmv_moe_gemm2_lora_delta(
        act_perm,
        perm,
        w_ptr_a,
        stride_a,
        w_ptr_b,
        stride_b,
        topk_ids,
        topk_weights,
        lora_ids,
        rank,
        hidden,
        lora_dtype=dtype,
        scale=scale,
        out_dtype=dtype,
    )
    ref = _ref_fc2_delta(
        act_perm, perm, lora_a, lora_b, topk_ids, topk_weights, lora_ids, scale
    )

    assert out.shape == (T, hidden)
    torch.testing.assert_close(out.float(), ref, atol=1e-2, rtol=1e-2)


def test_gemm2_inactive_slot_zeroed():
    """`expanded_idx_to_permuted_idx < 0` (inactive slot) must contribute nothing."""
    _skip_if_unsupported_sm()
    device = torch.device("cuda")
    torch.manual_seed(3)
    T, k, num_experts, max_loras, rank, hidden, inter = 4, 2, 8, 4, 32, 768, 768
    P = T * k

    lora_a = [
        torch.randn(
            max_loras, num_experts, rank, inter, dtype=torch.bfloat16, device=device
        )
        * 0.02
    ]
    lora_b = [
        torch.randn(
            max_loras, num_experts, hidden, rank, dtype=torch.bfloat16, device=device
        )
        * 0.02
    ]
    topk_ids, topk_weights, lora_ids = _make_routing(
        T, k, num_experts, max_loras, device
    )
    lora_ids[:] = 0  # all tokens have an adapter so only the perm<0 path zeroes
    act_perm = torch.randn(P, inter, dtype=torch.bfloat16, device=device) * 0.1
    perm = torch.arange(P, dtype=torch.int64, device=device)
    perm[0] = -1  # mark expanded slot 0 inactive

    w_ptr_a, stride_a = _build_w_ptr(lora_a, num_experts)
    w_ptr_b, stride_b = _build_w_ptr(lora_b, num_experts)
    out = bgmv_moe_gemm2_lora_delta(
        act_perm,
        perm,
        w_ptr_a,
        stride_a,
        w_ptr_b,
        stride_b,
        topk_ids,
        topk_weights,
        lora_ids,
        rank,
        hidden,
    )
    # Inactive (perm<0) slots must contribute zero rather than reading act_perm[-1].
    ref = _ref_fc2_delta_with_zeroed_inactive(
        act_perm, perm, lora_a, lora_b, topk_ids, topk_weights, lora_ids, 1.0
    )
    torch.testing.assert_close(out.float(), ref, atol=1e-2, rtol=1e-2)


def _ref_fc2_delta_with_zeroed_inactive(
    act_perm, perm, lora_a, lora_b, topk_ids, topk_weights, lora_ids, scale
):
    """Same as _ref_fc2_delta but inactive (perm<0) slots contribute zero."""
    T, k = topk_ids.shape
    hidden = lora_b[0].shape[2]
    inter = lora_a[0].shape[3]
    out = torch.zeros(T, hidden, dtype=torch.float32, device=act_perm.device)
    af = act_perm.float()
    for t in range(T):
        lid = int(lora_ids[t])
        if lid < 0:
            continue
        for j in range(k):
            p = int(perm[t * k + j])
            a_vec = af[p] if p >= 0 else torch.zeros(inter, device=act_perm.device)
            e = int(topk_ids[t, j])
            w = float(topk_weights[t, j])
            a = lora_a[0][lid, e].float()
            b = lora_b[0][lid, e].float()
            out[t] += scale * w * (b @ (a @ a_vec))
    return out
