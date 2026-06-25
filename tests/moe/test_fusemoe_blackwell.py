# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Numerical-correctness tests for the contributed FuseMoE Blackwell kernel.

Runs on SM100 only. Compares the kernel output against:
  1. A streaming FP32 Python reference (treated as ground truth).
  2. ``flashinfer.trtllm_fp8_block_scale_moe`` (production CUDA kernel,
     same FP8 quantisation level).

Inputs are randomised FP8 hidden states + weights with random per-block
scales. Both the kernel and the reference allocate per-expert FP32
weight buffers one expert at a time so the test fits on a single GPU.
"""

import pytest
import torch

import flashinfer
from flashinfer.fusemoe_blackwell import DSV3_EP8_SHAPE
from flashinfer.utils import is_sm100a_supported


HIDDEN, INTERMEDIATE, NUM_EXPERTS, NUM_LOCAL_EXPERTS, TOP_K, N_GROUP, TOPK_GROUP = (
    DSV3_EP8_SHAPE
)
ROUTED_SCALING = 2.5
BLOCK = 128


# Skip the entire module on non-SM100 GPUs.
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_sm100a_supported(torch.device("cuda")),
    reason="fusemoe_blackwell_fp8_dsv3 requires SM100a (Blackwell).",
)


def _rand_fp8(shape, device, gen):
    x = torch.randn(*shape, device=device, dtype=torch.bfloat16, generator=gen) * 0.5
    return x.to(torch.float8_e4m3fn)


def _rand_scale(shape, device, gen):
    return (
        0.5 + torch.rand(*shape, device=device, dtype=torch.float32, generator=gen)
    ) * 0.05


def make_random_inputs(num_tokens, seed):
    device = "cuda"
    g = torch.Generator(device=device).manual_seed(seed)
    return dict(
        routing_logits=torch.randn(
            num_tokens, NUM_EXPERTS, device=device, dtype=torch.float32, generator=g
        ),
        routing_bias=torch.randn(
            NUM_EXPERTS, device=device, dtype=torch.bfloat16, generator=g
        )
        * 0.01,
        hidden_states=_rand_fp8((num_tokens, HIDDEN), device, g),
        hidden_states_scale=_rand_scale((HIDDEN // BLOCK, num_tokens), device, g),
        gemm1_weights=_rand_fp8(
            (NUM_LOCAL_EXPERTS, 2 * INTERMEDIATE, HIDDEN), device, g
        ),
        gemm1_weights_scale=_rand_scale(
            (NUM_LOCAL_EXPERTS, (2 * INTERMEDIATE) // BLOCK, HIDDEN // BLOCK),
            device,
            g,
        ),
        gemm2_weights=_rand_fp8((NUM_LOCAL_EXPERTS, HIDDEN, INTERMEDIATE), device, g),
        gemm2_weights_scale=_rand_scale(
            (NUM_LOCAL_EXPERTS, HIDDEN // BLOCK, INTERMEDIATE // BLOCK),
            device,
            g,
        ),
    )


@torch.no_grad()
def _route_dsv3(routing_logits, routing_bias):
    """Replica of the DSV3 routing in the trace JSON reference."""
    logits = routing_logits.to(torch.float32)
    bias = routing_bias.to(torch.float32).reshape(-1)
    s = torch.sigmoid(logits)
    s_with_bias = s + bias
    T, E = logits.shape
    group_size = E // N_GROUP
    top2_vals, _ = torch.topk(
        s_with_bias.view(T, N_GROUP, group_size),
        k=2,
        dim=2,
        largest=True,
        sorted=False,
    )
    group_scores = top2_vals.sum(dim=2)
    _, group_idx = torch.topk(
        group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=False
    )
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1.0)
    score_mask = group_mask.unsqueeze(2).expand(T, N_GROUP, group_size).reshape(T, E)
    neg_inf = torch.finfo(torch.float32).min
    scores_pruned = s_with_bias.masked_fill(score_mask == 0, neg_inf)
    _, topk_idx = torch.topk(scores_pruned, k=TOP_K, dim=1, largest=True, sorted=False)
    M = torch.zeros_like(s)
    M.scatter_(1, topk_idx, 1.0)
    raw_w = s * M
    weights_sum = raw_w.sum(dim=1, keepdim=True) + 1e-20
    weights = (raw_w / weights_sum) * ROUTED_SCALING
    return topk_idx, weights.gather(1, topk_idx)


@torch.no_grad()
def streaming_pyref(inp):
    """FP32 reference that dequantises one expert at a time."""
    T, H = inp["hidden_states"].shape
    device = inp["hidden_states"].device

    topk_idx, w_topk = _route_dsv3(inp["routing_logits"], inp["routing_bias"])

    A_fp32 = inp["hidden_states"].to(torch.float32)
    A_scale = inp["hidden_states_scale"].to(torch.float32).permute(1, 0).contiguous()
    A_full = A_scale.unsqueeze(-1).repeat(1, 1, BLOCK).reshape(T, H)
    A = A_fp32 * A_full
    del A_fp32, A_scale, A_full

    output = torch.zeros((T, H), dtype=torch.float32, device=device)
    w13, s13 = inp["gemm1_weights"], inp["gemm1_weights_scale"]
    w2, s2 = inp["gemm2_weights"], inp["gemm2_weights_scale"]

    for le in range(NUM_LOCAL_EXPERTS):
        sel_mask = (topk_idx == le).any(dim=1)
        if not sel_mask.any():
            continue
        token_idx = torch.nonzero(sel_mask, as_tuple=False).squeeze(1)
        A_e = A.index_select(0, token_idx)

        W13_e = w13[le].to(torch.float32) * (
            s13[le]
            .to(torch.float32)
            .repeat_interleave(BLOCK, dim=0)
            .repeat_interleave(BLOCK, dim=1)
        )
        G1 = A_e.matmul(W13_e.t())
        del W13_e
        X1, X2 = G1[:, :INTERMEDIATE], G1[:, INTERMEDIATE:]
        H1 = (X2 * torch.sigmoid(X2)) * X1
        del G1, X1, X2

        W2_e = w2[le].to(torch.float32) * (
            s2[le]
            .to(torch.float32)
            .repeat_interleave(BLOCK, dim=0)
            .repeat_interleave(BLOCK, dim=1)
        )
        O = H1.matmul(W2_e.t())
        del H1, W2_e

        match = (topk_idx.index_select(0, token_idx) == le).float()
        w_e = (w_topk.index_select(0, token_idx) * match).sum(dim=1)
        output.index_add_(0, token_idx, O * w_e.unsqueeze(1))
        del O

    return output.to(torch.bfloat16)


def run_kernel(inp):
    return flashinfer.fusemoe_blackwell_fp8_dsv3(
        routing_logits=inp["routing_logits"],
        routing_bias=inp["routing_bias"],
        hidden_states=inp["hidden_states"],
        hidden_states_scale=inp["hidden_states_scale"],
        gemm1_weights=inp["gemm1_weights"],
        gemm1_weights_scale=inp["gemm1_weights_scale"],
        gemm2_weights=inp["gemm2_weights"],
        gemm2_weights_scale=inp["gemm2_weights_scale"],
        top_k=TOP_K,
        n_group=N_GROUP,
        topk_group=TOPK_GROUP,
        local_expert_offset=0,
        routed_scaling_factor=ROUTED_SCALING,
    )


def run_trtllm_reference(inp):
    return flashinfer.trtllm_fp8_block_scale_moe(
        routing_logits=inp["routing_logits"],
        routing_bias=inp["routing_bias"],
        hidden_states=inp["hidden_states"],
        hidden_states_scale=inp["hidden_states_scale"],
        gemm1_weights=inp["gemm1_weights"],
        gemm1_weights_scale=inp["gemm1_weights_scale"],
        gemm2_weights=inp["gemm2_weights"],
        gemm2_weights_scale=inp["gemm2_weights_scale"],
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        n_group=N_GROUP,
        topk_group=TOPK_GROUP,
        intermediate_size=INTERMEDIATE,
        local_expert_offset=0,
        local_num_experts=NUM_LOCAL_EXPERTS,
        routed_scaling_factor=ROUTED_SCALING,
        routing_method_type=2,
    )


@pytest.mark.parametrize("num_tokens", [4, 32, 128])
def test_fusemoe_blackwell_matches_pyref(num_tokens):
    inp = make_random_inputs(num_tokens, seed=num_tokens)
    out = run_kernel(inp)
    ref = streaming_pyref(inp)
    diff = (out.float() - ref.float()).abs()
    # The trtllm production kernel hits ~0.0002 max_abs at this shape, so
    # 0.001 is a safe ceiling for a kernel passing the same FP8 quant noise.
    assert diff.max().item() < 1e-3, (
        f"contributed vs pyref max_abs={diff.max().item():.5f} exceeds 1e-3"
    )
    assert diff.mean().item() < 1e-4


@pytest.mark.parametrize("num_tokens", [4, 32, 128])
def test_fusemoe_blackwell_matches_trtllm(num_tokens):
    inp = make_random_inputs(num_tokens, seed=num_tokens + 1000)
    out = run_kernel(inp)
    ref = run_trtllm_reference(inp)
    diff = (out.float() - ref.float()).abs()
    # Both kernels operate at the same FP8 quant level, so they should
    # agree within the same tolerance as each does vs FP32 reference.
    assert diff.max().item() < 1e-3
    assert diff.mean().item() < 1e-4


def test_fusemoe_blackwell_rejects_bad_shape():
    """Calling at a non-verified shape should raise without
    experimental_shape=True."""
    device = "cuda"
    g = torch.Generator(device=device).manual_seed(0)
    # half-size shape (still passes divisibility but not in VERIFIED_SHAPES).
    H, I, E, LE = (
        HIDDEN // 2,
        INTERMEDIATE // 2,
        NUM_EXPERTS // 2,
        NUM_LOCAL_EXPERTS // 2,
    )
    inp = dict(
        routing_logits=torch.randn(
            8, E, device=device, dtype=torch.float32, generator=g
        ),
        routing_bias=torch.zeros(E, device=device, dtype=torch.bfloat16),
        hidden_states=_rand_fp8((8, H), device, g),
        hidden_states_scale=_rand_scale((H // BLOCK, 8), device, g),
        gemm1_weights=_rand_fp8((LE, 2 * I, H), device, g),
        gemm1_weights_scale=_rand_scale((LE, (2 * I) // BLOCK, H // BLOCK), device, g),
        gemm2_weights=_rand_fp8((LE, H, I), device, g),
        gemm2_weights_scale=_rand_scale((LE, H // BLOCK, I // BLOCK), device, g),
    )
    with pytest.raises(ValueError, match="verified-shape allowlist"):
        flashinfer.fusemoe_blackwell_fp8_dsv3(
            **inp,
            top_k=TOP_K,
            n_group=N_GROUP,
            topk_group=TOPK_GROUP,
            local_expert_offset=0,
            routed_scaling_factor=ROUTED_SCALING,
        )
