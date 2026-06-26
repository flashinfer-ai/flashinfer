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

End-to-end wiring into ``trtllm_*_moe`` (FC1 delta injected as ``gemm1_lora_delta``,
FC2 delta added to the output) is documented in ``moe_lora_delta.py``; run it on a
Blackwell box following that snippet.
"""

import os

os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")

import pytest
import torch

import torch.nn.functional as F

from flashinfer import ActivationType, RoutingMethodType
from flashinfer.fused_moe import (
    WeightLayout,
    bgmv_moe_gemm1_lora_delta,
    bgmv_moe_gemm2_lora_delta,
    convert_to_block_layout,
    fill_w_ptr,
    trtllm_bf16_routed_moe,
)
from flashinfer.fused_moe.core import (
    _maybe_get_cached_w3_w1_permute_indices,
    get_w2_permute_indices_with_cache,
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


def _skip_if_not_blackwell():
    """The trtllm-gen routed MoE (the gemm1_lora_delta injection point) is SM100/103-only."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("trtllm-gen routed MoE requires SM100/SM103 (Blackwell)")


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


def _ref_fc2_delta(act_perm, perm, lora_a, lora_b, topk_ids, topk_weights, lora_ids, scale):
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
def test_gemm1_lora_delta(T, hidden, inter, rank, dtype):
    _skip_if_unsupported_sm()
    device = torch.device("cuda")
    torch.manual_seed(1)
    num_experts, k, max_loras = 8, 2, 4
    scale = 0.5

    hidden_states = torch.randn(T, hidden, dtype=dtype, device=device) * 0.1
    lora_a = [
        torch.randn(max_loras, num_experts, rank, hidden, dtype=dtype, device=device) * 0.02
        for _ in range(2)
    ]
    lora_b = [
        torch.randn(max_loras, num_experts, inter, rank, dtype=dtype, device=device) * 0.02
        for _ in range(2)
    ]
    topk_ids, _, lora_ids = _make_routing(T, k, num_experts, max_loras, device)

    w_ptr_a, stride_a = _build_w_ptr(lora_a, num_experts)
    w_ptr_b, stride_b = _build_w_ptr(lora_b, num_experts)
    out = bgmv_moe_gemm1_lora_delta(
        hidden_states, w_ptr_a, stride_a, w_ptr_b, stride_b, topk_ids, lora_ids,
        rank, inter, lora_dtype=dtype, scale=scale, out_dtype=dtype,
    )
    ref = _ref_fc1_delta(hidden_states, lora_a, lora_b, topk_ids, lora_ids, scale)

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

    lora_a = [torch.randn(max_loras, num_experts, rank, inter, dtype=dtype, device=device) * 0.02]
    lora_b = [torch.randn(max_loras, num_experts, hidden, rank, dtype=dtype, device=device) * 0.02]
    topk_ids, topk_weights, lora_ids = _make_routing(T, k, num_experts, max_loras, device)

    # Synthetic permuted post-activation + a permutation map (all slots active).
    act_perm = torch.randn(P, inter, dtype=dtype, device=device) * 0.1
    perm = torch.randperm(P, dtype=torch.int64, device=device)

    w_ptr_a, stride_a = _build_w_ptr(lora_a, num_experts)
    w_ptr_b, stride_b = _build_w_ptr(lora_b, num_experts)
    out = bgmv_moe_gemm2_lora_delta(
        act_perm, perm, w_ptr_a, stride_a, w_ptr_b, stride_b, topk_ids, topk_weights,
        lora_ids, rank, hidden, lora_dtype=dtype, scale=scale, out_dtype=dtype,
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

    lora_a = [torch.randn(max_loras, num_experts, rank, inter, dtype=torch.bfloat16, device=device) * 0.02]
    lora_b = [torch.randn(max_loras, num_experts, hidden, rank, dtype=torch.bfloat16, device=device) * 0.02]
    topk_ids, topk_weights, lora_ids = _make_routing(T, k, num_experts, max_loras, device)
    lora_ids[:] = 0  # all tokens have an adapter so only the perm<0 path zeroes
    act_perm = torch.randn(P, inter, dtype=torch.bfloat16, device=device) * 0.1
    perm = torch.arange(P, dtype=torch.int64, device=device)
    perm[0] = -1  # mark expanded slot 0 inactive

    w_ptr_a, stride_a = _build_w_ptr(lora_a, num_experts)
    w_ptr_b, stride_b = _build_w_ptr(lora_b, num_experts)
    out = bgmv_moe_gemm2_lora_delta(
        act_perm, perm, w_ptr_a, stride_a, w_ptr_b, stride_b, topk_ids, topk_weights,
        lora_ids, rank, hidden,
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


# ============================================================
# End-to-end: builders + the real trtllm-gen BF16 routed MoE (SM100/103)
#   FC1 delta injected via gemm1_lora_delta (pre-SwiGLU);
#   FC2 delta computed from the kernel's returned activation and added to the output.
# ============================================================


def _prep_bf16_shuffled_weights(gemm1, gemm2):
    """Reorder/shuffle + BlockMajorK layout, matching the BF16Moe kernel weight prep
    in tests/moe/test_trtllm_gen_fused_moe.py."""
    E = gemm1.shape[0]
    cache = {}
    g1s, g2s = [], []
    etm = 128
    for i in range(E):
        pi = _maybe_get_cached_w3_w1_permute_indices(
            cache, gemm1[i].view(torch.uint8), etm, is_gated_act_gemm=True
        )
        w1 = gemm1[i].view(torch.uint8)[pi.to(gemm1.device)].contiguous()
        pi2 = get_w2_permute_indices_with_cache(cache, gemm2[i].view(torch.uint8), etm)
        w2 = gemm2[i].view(torch.uint8)[pi2.to(gemm2.device)].contiguous()
        w1 = convert_to_block_layout(w1.view(torch.uint8), 128)
        w2 = convert_to_block_layout(w2.view(torch.uint8), 128)
        g1s.append(w1.view(torch.bfloat16))
        g2s.append(w2.view(torch.bfloat16))
    return torch.stack(g1s).contiguous(), torch.stack(g2s).contiguous()


def _pack_topk(idx, w):
    return (idx.to(torch.int32) << 16) | w.to(torch.bfloat16).view(torch.int16).to(
        torch.int32
    )


def _ref_moe_with_lora(
    hidden, gemm1, gemm2, topk_ids, topk_w, lora_ids, A_g, A_u, B_g, B_u, A_d, B_d, scale
):
    """Pure-PyTorch SwiGLU MoE forward with FC1 (gate/up) + FC2 (down) LoRA.
    SwiGLU convention matches trtllm: silu(2nd half) * 1st half."""
    T, H = hidden.shape
    k = topk_ids.shape[1]
    I = gemm2.shape[2]
    out = torch.zeros(T, H, dtype=torch.float32, device=hidden.device)
    hf = hidden.float()
    for t in range(T):
        lid = int(lora_ids[t])
        for j in range(k):
            e = int(topk_ids[t, j])
            w = float(topk_w[t, j])
            g = gemm1[e].float() @ hf[t]  # [2I]
            if lid >= 0:
                g[:I] += scale * (B_g[lid, e].float() @ (A_g[lid, e].float() @ hf[t]))
                g[I:] += scale * (B_u[lid, e].float() @ (A_u[lid, e].float() @ hf[t]))
            a = F.silu(g[I:]) * g[:I]  # [I]
            y = gemm2[e].float() @ a  # [H]
            if lid >= 0:
                y += scale * (B_d[lid, e].float() @ (A_d[lid, e].float() @ a))
            out[t] += w * y
    return out


@pytest.mark.parametrize("T", [16, 64])
@pytest.mark.parametrize("rank", [16, 32])
def test_e2e_fc1_fc2_lora_delta_bf16(T, rank):
    """FC1 delta -> trtllm_bf16_routed_moe(gemm1_lora_delta=...) -> FC2 delta added,
    vs a pure-PyTorch MoE-with-LoRA reference."""
    _skip_if_not_blackwell()
    device = torch.device("cuda")
    torch.manual_seed(0)
    H = I = 768
    E, k, L = 8, 2, 4
    scale = 0.5
    bf16 = torch.bfloat16

    hidden = torch.randn(T, H, device=device, dtype=bf16)
    gemm1 = torch.randn(E, 2 * I, H, device=device, dtype=bf16) / (H**0.5)
    gemm2 = torch.randn(E, H, I, device=device, dtype=bf16) / (I**0.5)

    # routing: softmax -> top-k -> renormalize so weights sum to 1 (Renormalize == identity)
    probs = torch.softmax(torch.randn(T, E, device=device), dim=-1)
    topk_w, topk_idx = torch.topk(probs, k, dim=-1)
    topk_w = (topk_w / topk_w.sum(-1, keepdim=True)).to(bf16)
    topk_idx = topk_idx.to(torch.int64)

    # LoRA weights (gate, up, down) + per-token adapter ids (some -1 = no adapter)
    A_g = torch.randn(L, E, rank, H, device=device, dtype=bf16) * 0.02
    A_u = torch.randn(L, E, rank, H, device=device, dtype=bf16) * 0.02
    B_g = torch.randn(L, E, I, rank, device=device, dtype=bf16) * 0.02
    B_u = torch.randn(L, E, I, rank, device=device, dtype=bf16) * 0.02
    A_d = torch.randn(L, E, rank, I, device=device, dtype=bf16) * 0.02
    B_d = torch.randn(L, E, H, rank, device=device, dtype=bf16) * 0.02
    lora_ids = torch.randint(-1, L, (T,), device=device, dtype=torch.int64)
    lora_ids[: max(1, T // 2)] = torch.randint(
        0, L, (max(1, T // 2),), device=device, dtype=torch.int64
    )

    # FC1 delta -> kernel  (caller builds the pointer tables from its LoRA weight banks)
    wpa1, sa1 = _build_w_ptr([A_g, A_u], E)
    wpb1, sb1 = _build_w_ptr([B_g, B_u], E)
    delta1 = bgmv_moe_gemm1_lora_delta(
        hidden, wpa1, sa1, wpb1, sb1, topk_idx, lora_ids, rank, I,
        scale=scale, out_dtype=bf16,
    )
    g1_shuf, g2_shuf = _prep_bf16_shuffled_weights(gemm1, gemm2)
    out_k, exp2perm, act = trtllm_bf16_routed_moe(
        _pack_topk(topk_idx, topk_w), hidden, g1_shuf, g2_shuf, E, k, None, None, I,
        0, E, None, use_shuffled_weight=True, weight_layout=WeightLayout.BlockMajorK,
        routing_method_type=RoutingMethodType.Renormalize, do_finalize=True,
        activation_type=ActivationType.Swiglu.value, gemm1_lora_delta=delta1,
    )

    # FC2 delta from the kernel's returned activation -> add to output
    wpa2, sa2 = _build_w_ptr([A_d], E)
    wpb2, sb2 = _build_w_ptr([B_d], E)
    delta2 = bgmv_moe_gemm2_lora_delta(
        act, exp2perm, wpa2, sa2, wpb2, sb2, topk_idx, topk_w, lora_ids, rank, H,
        scale=scale, out_dtype=torch.float32,
    )
    out_lora = out_k.float() + delta2.float()

    ref = _ref_moe_with_lora(
        hidden, gemm1, gemm2, topk_idx, topk_w, lora_ids,
        A_g, A_u, B_g, B_u, A_d, B_d, scale,
    )

    assert out_lora.shape == (T, H)
    # bf16 through two GEMMs + the kernel: relative max error well under 3%.
    rel = (out_lora - ref).abs().max() / ref.abs().max().clamp_min(1e-6)
    assert rel < 3e-2, f"e2e LoRA MoE relative max error too high: {rel.item()}"
    # sanity: LoRA must materially change the output vs the no-LoRA path
    base = _ref_moe_with_lora(
        hidden, gemm1, gemm2, topk_idx, topk_w,
        torch.full_like(lora_ids, -1), A_g, A_u, B_g, B_u, A_d, B_d, scale,
    )
    assert (ref - base).abs().max() > 0.02
