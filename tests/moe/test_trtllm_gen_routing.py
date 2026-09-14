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

Standalone tests for the trtllm-gen MoE routing stage (trtllm_gen_routing).

Routing correctness used to be covered only transitively, by multiplying every
routing method against the full fused-MoE quant/shape matrix. This file tests
the routing kernels directly against the host oracles in
trtllm_gen_fused_moe_utils.py, so the fused tests can pin routing to one or two
representative methods (see docs/design_docs/moe_routing_test_decomposition.md).

Notes on test construction:
- Logits are positive and tie-free by construction (per-row randperm / 32,
  exactly representable in bfloat16). The shared `routing_reference` oracle
  ranks the masked dense weight matrix with torch.topk, so zero entries would
  outrank negative routed weights (TopK/Sigmoid methods); positive logits keep
  the oracle valid. Tie-free values make strict id comparison meaningful.
- The kernel's within-expert ordering of the permuted buffer is not part of the
  contract; permutation outputs are checked via invariants (round-trip through
  permuted_idx_to_token_idx, per-expert padded segments, uniqueness) rather
  than element-wise equality.
"""

import math
import zlib

import pytest
import torch

from flashinfer.fused_moe import trtllm_gen_routing
from flashinfer.tllm_enums import RoutingMethodType
from flashinfer.utils import get_compute_capability

from tests.moe.trtllm_gen_fused_moe_utils import (
    routing_reference_default,
    routing_reference_minimax2,
    routing_reference_no_aux,
    routing_reference_renormalize,
    routing_reference_renormalize_naive,
    routing_reference_sigmoid_renorm,
    routing_reference_sqrt_softplus,
    routing_reference_topk,
)

# Weights come back bfloat16 (the routing dispatcher hard-codes its output
# dtype); references compute in float32 on the same inputs.
WEIGHT_ATOL = 1e-2
WEIGHT_RTOL = 2e-2


@pytest.fixture(autouse=True)
def require_supported_gpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    compute_capability = get_compute_capability(torch.device("cuda"))
    if compute_capability == (10, 7):
        pytest.skip("trtllm-gen routing is not implemented on SM107")
    if compute_capability[0] not in (10, 12):
        pytest.skip("trtllm-gen routing requires SM100/SM103/SM120/SM121")


def stable_seed(*parts):
    """Deterministic across processes (unlike hash(), which salts strings)."""
    return zlib.crc32(repr(parts).encode()) % (2**31)


def make_logits(num_tokens, num_experts, dtype, seed, hot_expert=None):
    """Positive, per-row-distinct logits, exactly representable in bfloat16.

    Values are a per-token permutation of {0, 1/32, ..., (num_experts-1)/32}
    (all < 8, where bf16 still resolves steps of 1/32). Optionally boosts one
    expert by +8 for every token so its value dominates the row while staying
    distinct from all others — a worst-case load-imbalance pattern for the
    padding/permutation logic.
    """
    assert num_experts <= 256 or dtype != torch.bfloat16, (
        "distinct-in-bf16 construction holds up to 256 experts; use float32 "
        "logits beyond that"
    )
    gen = torch.Generator().manual_seed(seed)
    perm = torch.argsort(torch.rand(num_tokens, num_experts, generator=gen), dim=1)
    logits = perm.float() / 32.0
    if hot_expert is not None:
        logits[:, hot_expert] += 8.0
    return logits.to(dtype).cuda()


def make_bias(num_experts, dtype, seed):
    """Distinct per-expert bias in [-2, 2), exactly representable in bf16."""
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_experts, generator=gen)
    return ((perm.float() - num_experts / 2) / 64.0).to(dtype).cuda()


def check_selection(result, permute_info, ref_scores, top_k):
    """Expert ids and weights must match the host oracle."""
    ids = result.topk_ids
    ref_ids = permute_info["topKIndices"].to(ids.device)
    ids_sorted, _ = torch.sort(ids.long(), dim=1)
    ref_sorted, _ = torch.sort(ref_ids.long(), dim=1)
    mismatched = (ids_sorted != ref_sorted).any(dim=1)
    assert not mismatched.any(), (
        f"expert selection mismatch on {int(mismatched.sum())}/{ids.shape[0]} tokens; "
        f"first bad token {int(mismatched.nonzero()[0])}: "
        f"got {ids_sorted[mismatched][0].tolist()}, "
        f"want {ref_sorted[mismatched][0].tolist()}"
    )

    # Gather the reference weight at the kernel-chosen expert so the check is
    # independent of intra-row ordering.
    ref_w = ref_scores.to(ids.device).float().gather(1, ids.long())
    torch.testing.assert_close(
        result.topk_weights.float(), ref_w, atol=WEIGHT_ATOL, rtol=WEIGHT_RTOL
    )


def check_permutation(result, permute_info, num_tokens, top_k):
    """Permutation outputs must be internally consistent with the oracle's
    per-expert padded segments, without assuming intra-expert ordering."""
    device = result.topk_ids.device
    padded_size = int(result.total_num_padded_tokens.item())
    assert padded_size == permute_info["permutedBufferSize"], (
        f"padded token count {padded_size} != "
        f"reference {permute_info['permutedBufferSize']}"
    )

    e2p = result.expanded_idx_to_permuted_idx.flatten().long()
    assert ((e2p >= 0) & (e2p < padded_size)).all(), "permuted idx out of range"
    assert e2p.unique().numel() == e2p.numel(), "permuted idx not unique"

    # Round-trip: the permuted slot must map back to the owning token.
    p2t = result.permuted_idx_to_token_idx.long()
    tokens = torch.arange(num_tokens, device=device).repeat_interleave(top_k)
    assert (p2t[e2p] == tokens).all(), "permuted->token round trip failed"

    # Each expanded (token, k) slot must land in its expert's padded segment.
    prefix = permute_info["paddedTokensPerExpertPrefixSum"].to(device).long()
    expert = result.topk_ids.flatten().long()
    in_segment = (e2p >= prefix[expert]) & (e2p < prefix[expert + 1])
    assert in_segment.all(), "permuted idx outside its expert's padded segment"

    # Per-expert counts.
    counts = torch.bincount(expert, minlength=prefix.numel() - 1)
    ref_counts = permute_info["numTokensPerExpert"].to(device).long()
    assert (counts == ref_counts).all(), "per-expert token counts mismatch"


def run_and_check(
    routing_method,
    reference,
    logits,
    top_k,
    tile_tokens_dim,
    routing_bias=None,
    **kwargs,
):
    num_tokens = logits.shape[0]
    permute_info, ref_scores = reference()
    result = trtllm_gen_routing(
        logits,
        routing_bias,
        routing_method,
        top_k,
        tile_tokens_dim=tile_tokens_dim,
        **kwargs,
    )
    check_selection(result, permute_info, ref_scores, top_k)
    check_permutation(result, permute_info, num_tokens, top_k)


# (method, reference fn) for the methods routed through routingCustom with no
# bias/group parameters. The reference gets (logits, top_k, num_experts,
# padding) except where noted.
CUSTOM_METHODS = [
    pytest.param(RoutingMethodType.Default, routing_reference_default, id="Default"),
    pytest.param(
        RoutingMethodType.Renormalize, routing_reference_renormalize, id="Renormalize"
    ),
    pytest.param(
        RoutingMethodType.RenormalizeNaive,
        routing_reference_renormalize_naive,
        id="RenormalizeNaive",
    ),
    pytest.param(RoutingMethodType.TopK, routing_reference_topk, id="TopK"),
    pytest.param(
        RoutingMethodType.SigmoidRenorm,
        lambda logits, top_k, num_experts, padding: routing_reference_sigmoid_renorm(
            logits, top_k, num_experts, padding, norm_topk_prob=True
        ),
        id="SigmoidRenorm",
    ),
    pytest.param(
        RoutingMethodType.Sigmoid,
        lambda logits, top_k, num_experts, padding: routing_reference_sigmoid_renorm(
            logits, top_k, num_experts, padding, norm_topk_prob=False
        ),
        id="Sigmoid",
    ),
]


@pytest.mark.parametrize("num_tokens", [1, 8, 150])
@pytest.mark.parametrize("num_experts,top_k", [(16, 1), (16, 4), (256, 8)])
@pytest.mark.parametrize("tile_tokens_dim", [8, 32])
@pytest.mark.parametrize("logits_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("hot_expert", [None, 3], ids=["uniform", "hot"])
@pytest.mark.parametrize("routing_method,reference", CUSTOM_METHODS)
def test_custom_routing_methods(
    routing_method,
    reference,
    num_tokens,
    num_experts,
    top_k,
    tile_tokens_dim,
    logits_dtype,
    hot_expert,
):
    seed = stable_seed(int(routing_method), num_tokens, num_experts, top_k)
    logits = make_logits(num_tokens, num_experts, logits_dtype, seed, hot_expert)
    run_and_check(
        routing_method,
        lambda: reference(logits, top_k, num_experts, tile_tokens_dim),
        logits,
        top_k,
        tile_tokens_dim,
    )


@pytest.mark.parametrize("num_tokens", [1, 8, 150])
@pytest.mark.parametrize(
    "num_experts,n_group,topk_group,top_k",
    [
        # Model-shaped configs, mirroring the routing_config list in
        # test_trtllm_gen_fused_moe.py::test_deepseekv3_routing — the fused
        # matrix only keeps a couple of representatives, routing variety is
        # covered here.
        (256, 8, 4, 8),  # DeepSeek-V3
        (128, 4, 2, 4),
        (96, 1, 1, 8),  # no-groups fast path (routingCustom SigmoidBias)
        (512, 1, 1, 22),  # nemotron_3_super (top_k kernel maximum)
        (384, 1, 1, 8),  # kimi_k2
        (160, 1, 1, 8),  # GLM4_MoE
        (72, 1, 1, 6),  # DSLite
    ],
)
@pytest.mark.parametrize("tile_tokens_dim", [8, 32])
@pytest.mark.parametrize("logits_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("bias_dtype", [torch.bfloat16, torch.float32])
def test_deepseekv3_routing(
    num_tokens,
    num_experts,
    n_group,
    topk_group,
    top_k,
    tile_tokens_dim,
    logits_dtype,
    bias_dtype,
):
    if num_experts > 256 and logits_dtype == torch.bfloat16:
        pytest.skip("tie-free logits construction needs float32 beyond 256 experts")
    routed_scaling = 2.5
    seed = stable_seed("dsv3", num_tokens, num_experts, n_group, top_k)
    logits = make_logits(num_tokens, num_experts, logits_dtype, seed)
    bias = make_bias(num_experts, bias_dtype, seed + 1)
    run_and_check(
        RoutingMethodType.DeepSeekV3,
        lambda: routing_reference_no_aux(
            logits, bias, top_k, n_group, topk_group, routed_scaling, tile_tokens_dim
        ),
        logits,
        top_k,
        tile_tokens_dim,
        routing_bias=bias,
        n_group=n_group,
        topk_group=topk_group,
        routed_scaling_factor=routed_scaling,
    )


@pytest.mark.parametrize("num_tokens", [1, 8, 150])
@pytest.mark.parametrize("num_experts,top_k", [(64, 4), (256, 8)])
@pytest.mark.parametrize("tile_tokens_dim", [8, 32])
@pytest.mark.parametrize("logits_dtype", [torch.float32, torch.bfloat16])
def test_minimax2_routing(
    num_tokens, num_experts, top_k, tile_tokens_dim, logits_dtype
):
    seed = stable_seed("minimax2", num_tokens, num_experts, top_k)
    logits = make_logits(num_tokens, num_experts, logits_dtype, seed)
    bias = make_bias(num_experts, torch.bfloat16, seed + 1)
    run_and_check(
        RoutingMethodType.MiniMax2,
        lambda: routing_reference_minimax2(
            logits, bias, top_k, num_experts, tile_tokens_dim, 1.0
        ),
        logits,
        top_k,
        tile_tokens_dim,
        routing_bias=bias,
        routed_scaling_factor=1.0,
    )


# ---------------------------------------------------------------------------
# SqrtSoftplus (DeepSeek-V4 family): sqrt(softplus(logit)) + bias -> ungrouped
# top-k -> renormalize the un-biased scores -> routed_scaling_factor.
#
# The token counts below cross every routingCustom kernel path for E=384/K=6:
# <=4 static block, <=16 dyn-block, 17..256 single-cluster split top-k
# (block-per-token scores), >256 large-batch histogram kernels.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("num_tokens", [1, 4, 8, 16, 17, 150, 256, 257, 1025])
@pytest.mark.parametrize(
    "num_experts,top_k",
    [
        (64, 4),
        (256, 6),  # DeepSeek-V4-Flash
        (384, 6),  # DeepSeek-V4.1-Flash / V4-Pro
    ],
)
@pytest.mark.parametrize("tile_tokens_dim", [8, 32])
@pytest.mark.parametrize("logits_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("bias_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("norm_topk_prob", [True, False])
def test_sqrt_softplus_routing(
    num_tokens,
    num_experts,
    top_k,
    tile_tokens_dim,
    logits_dtype,
    bias_dtype,
    norm_topk_prob,
):
    if num_experts > 256 and logits_dtype == torch.bfloat16:
        pytest.skip("tie-free logits construction needs float32 beyond 256 experts")
    if num_tokens > 256 and (tile_tokens_dim == 32 or bias_dtype == torch.float32):
        pytest.skip("large-batch kernels: keep one representative per shape")
    routed_scaling = 1.5  # DeepSeek-V4.1-Flash
    seed = stable_seed("sqrtsoftplus", num_tokens, num_experts, top_k, norm_topk_prob)
    logits = make_logits(num_tokens, num_experts, logits_dtype, seed)
    bias = make_bias(num_experts, bias_dtype, seed + 1)
    run_and_check(
        RoutingMethodType.SqrtSoftplus,
        lambda: routing_reference_sqrt_softplus(
            logits,
            bias,
            top_k,
            num_experts,
            tile_tokens_dim,
            routed_scaling,
            norm_topk_prob=norm_topk_prob,
        ),
        logits,
        top_k,
        tile_tokens_dim,
        routing_bias=bias,
        routed_scaling_factor=routed_scaling,
        norm_topk_prob=norm_topk_prob,
    )


def _sqrt_softplus(x):
    return math.sqrt(math.log1p(math.exp(x)))


def _softplus_inv(y):
    """x with softplus(x) == y."""
    return math.log(math.exp(y) - 1.0)


def _run_sqrt_softplus_vector(
    logits_row, bias_row, top_k, routed_scaling, norm_topk_prob=True
):
    """Route one adversarial row (padded to 16 experts) and return (ids, weights) by id."""
    num_experts = 16
    logits = torch.full((1, num_experts), -30.0)
    logits[0, : len(logits_row)] = torch.tensor(logits_row)
    bias = torch.zeros(num_experts)
    bias[: len(bias_row)] = torch.tensor(bias_row)
    result = trtllm_gen_routing(
        logits.cuda(),
        bias.cuda(),
        RoutingMethodType.SqrtSoftplus,
        top_k,
        routed_scaling_factor=routed_scaling,
        norm_topk_prob=norm_topk_prob,
    )
    ids = result.topk_ids[0].long().cpu()
    order = torch.argsort(ids)
    return ids[order].tolist(), result.topk_weights[0].float().cpu()[order]


def test_sqrt_softplus_adversarial_vectors():
    """Hand-built rows where sigmoid / softmax / biased-weight / wrong-order
    semantics select different experts or weights than the DeepSeek-V4 contract.
    Expected values are the analytic sqrt-softplus numbers (bf16 output)."""
    tol = dict(atol=1e-2, rtol=2e-2)

    # A. sqrt-softplus vs sigmoid: sqrtsoftplus(3)-sqrtsoftplus(0)=0.913 but
    #    sigmoid(3)-sigmoid(0)=0.453, so a +0.7 bias on expert 1 flips the winner
    #    only under sigmoid scoring (which would pick expert 1).
    ids, w = _run_sqrt_softplus_vector(
        [3.0, 0.0], [0.0, 0.7], top_k=1, routed_scaling=1.0
    )
    assert ids == [0]
    torch.testing.assert_close(w, torch.tensor([1.0]), **tol)

    # B. sqrt-softplus vs softmax: softmax scoring picks {0, 2} here, DSV4 {0, 1}.
    ids, w = _run_sqrt_softplus_vector(
        [2.0, 1.0, 0.0, -1.0], [0.0, 0.0, 0.25, 0.0], top_k=2, routed_scaling=1.0
    )
    assert ids == [0, 1]
    a, b = _sqrt_softplus(2.0), _sqrt_softplus(1.0)
    torch.testing.assert_close(w, torch.tensor([a / (a + b), b / (a + b)]), **tol)

    # C. the correction bias changes which expert enters the top-k (0,1 -> 0,2 -> 0,3).
    ids, _ = _run_sqrt_softplus_vector(
        [1.0, 0.9, 0.8, -2.0], [0.0] * 4, top_k=2, routed_scaling=1.0
    )
    assert ids == [0, 1]
    gap = (
        _sqrt_softplus(0.9)
        - _sqrt_softplus(0.8)
        + _sqrt_softplus(1.0)
        - _sqrt_softplus(0.8)
    ) / 2
    ids, _ = _run_sqrt_softplus_vector(
        [1.0, 0.9, 0.8, -2.0], [0.0, 0.0, gap, 0.0], top_k=2, routed_scaling=1.0
    )
    assert ids == [0, 2]
    ids, _ = _run_sqrt_softplus_vector(
        [1.0, 0.9, 0.8, -2.0], [0.0, 0.0, 0.0, 10.0], top_k=2, routed_scaling=1.0
    )
    assert ids == [0, 3]

    # D. final weights ignore the bias: equal logits, +5 bias on expert 0 -> equal
    #    weights (0.75 each at x1.5). A bias leak would give [1.1667, 0.3333].
    x2 = _softplus_inv(4.0)  # sqrt-softplus score 2.0
    ids, w = _run_sqrt_softplus_vector(
        [x2, x2], [5.0, 0.0], top_k=2, routed_scaling=1.5
    )
    assert ids == [0, 1]
    torch.testing.assert_close(w, torch.tensor([0.75, 0.75]), **tol)

    # D2. selection needs the bias but the weight is the tiny un-biased score:
    #     sqrt-softplus(-30) ~ 9.7e-7 must survive (key - bias would round it away).
    ids, w = _run_sqrt_softplus_vector(
        [-30.0, 0.0, -1.0, -2.0],
        [12.0, 0.0, 0.0, 0.0],
        top_k=2,
        routed_scaling=1.0,
        norm_topk_prob=False,
    )
    assert ids == [0, 1]
    assert 0 < w[0].item() < 2e-6
    torch.testing.assert_close(w[1], torch.tensor(_sqrt_softplus(0.0)), **tol)

    # E. renormalization with exact numbers: scores (2, 1) -> (2/3, 1/3).
    x1 = _softplus_inv(1.0)
    ids, w = _run_sqrt_softplus_vector(
        [x2, x1], [0.0, 0.0], top_k=2, routed_scaling=1.0
    )
    assert ids == [0, 1]
    torch.testing.assert_close(w, torch.tensor([2 / 3, 1 / 3]), **tol)

    # F. routed_scaling_factor=1.5 applies AFTER renormalization: (1.0, 0.5), sum 1.5.
    #    Scaling before renormalization would cancel (sum 1.0).
    ids, w = _run_sqrt_softplus_vector(
        [x2, x1], [0.0, 0.0], top_k=2, routed_scaling=1.5
    )
    torch.testing.assert_close(w, torch.tensor([1.0, 0.5]), **tol)
    #    norm_topk_prob=False keeps the raw scores: (2, 1) * 1.5.
    ids, w = _run_sqrt_softplus_vector(
        [x2, x1], [0.0, 0.0], top_k=2, routed_scaling=1.5, norm_topk_prob=False
    )
    torch.testing.assert_close(w, torch.tensor([3.0, 1.5]), **tol)

    # Numerics: a large positive logit must not overflow (naive log(1+exp(100)) is
    # inf) and a very negative logit pulled in by the bias must yield a finite ~0
    # weight rather than NaN.
    ids, w = _run_sqrt_softplus_vector(
        [100.0, 60.0, -100.0],
        [0.0, 0.0, 50.0],
        top_k=3,
        routed_scaling=1.0,
        norm_topk_prob=False,
    )
    assert ids == [0, 1, 2]
    torch.testing.assert_close(w[:2], torch.tensor([10.0, math.sqrt(60.0)]), **tol)
    assert torch.isfinite(w).all() and 0.0 <= w[2].item() < 1e-6


def test_sqrt_softplus_dsv41_geometry_fixture():
    """DeepSeek-V4.1-Flash geometry (E=384, K=6, x1.5) against a seeded fixture.

    The fixture pins the CPU oracle's expert ids (crc32) and the first rows so a
    future regression in either the oracle or the kernel is visible."""
    gen = torch.Generator().manual_seed(0x5190)
    T, E, K = 64, 384, 6
    logits = (torch.randn(T, E, generator=gen) * 2.0).cuda()
    bias = (torch.randn(E, generator=gen) * 0.5).to(torch.bfloat16).cuda()
    result = trtllm_gen_routing(
        logits, bias, RoutingMethodType.SqrtSoftplus, K, routed_scaling_factor=1.5
    )
    ids, order = torch.sort(result.topk_ids.long(), dim=1)
    w = result.topk_weights.float().gather(1, order)
    assert zlib.crc32(ids.to(torch.int32).cpu().numpy().tobytes()) == 2895044175
    assert ids[:4].tolist() == [
        [22, 138, 203, 226, 301, 302],
        [0, 57, 74, 138, 286, 333],
        [188, 196, 215, 287, 312, 313],
        [28, 42, 52, 65, 242, 244],
    ]
    torch.testing.assert_close(
        w[:4],
        torch.tensor(
            [
                [0.2312814, 0.2577158, 0.2590204, 0.3113003, 0.244737, 0.1959451],
                [0.2358731, 0.2165189, 0.2240225, 0.2810181, 0.2579508, 0.2846167],
                [0.2962734, 0.212483, 0.2642444, 0.2143541, 0.2315651, 0.28108],
                [0.2400168, 0.2203155, 0.2650456, 0.3130441, 0.2499551, 0.2116229],
            ],
            device=w.device,
        ),
        atol=WEIGHT_ATOL,
        rtol=WEIGHT_RTOL,
    )
    torch.testing.assert_close(
        w.sum(dim=1), torch.full((T,), 1.5, device=w.device), atol=2e-2, rtol=0.0
    )
    # Cross-check the whole batch against the host oracle.
    _, ref_scores = routing_reference_sqrt_softplus(
        logits.cpu(), bias.cpu(), K, E, 8, 1.5
    )
    ref_w = ref_scores.cuda().float().gather(1, ids)
    torch.testing.assert_close(w, ref_w, atol=WEIGHT_ATOL, rtol=WEIGHT_RTOL)


def test_sqrt_softplus_enum_round_trips_through_binding():
    """RoutingMethodType.SqrtSoftplus (11, appended after Unspecified=10) must
    survive Python enum -> int -> FFI binding range check -> C++ enum -> runner
    dispatch. One row separates every method family: sqrt-softplus + bias picks
    expert 1, sigmoid + bias picks expert 2, bias-ignoring methods pick expert 0
    (the raw argmax), so reaching any other branch is visible in the ids."""
    num_experts = 16
    logits = torch.full((1, num_experts), -30.0)
    logits[0, :3] = torch.tensor([3.0, 0.0, -1.0])
    bias = torch.zeros(num_experts)
    bias[:3] = torch.tensor([0.0, 1.0, 1.25])  # exact in bf16
    logits, bias = logits.cuda(), bias.to(torch.bfloat16).cuda()

    def selected(method, **kwargs):
        return int(trtllm_gen_routing(logits, bias, method, 1, **kwargs).topk_ids[0, 0])

    # Python deserialization of the wire value, then through the binding.
    method = RoutingMethodType(int(RoutingMethodType.SqrtSoftplus))
    assert method is RoutingMethodType.SqrtSoftplus
    assert method is not RoutingMethodType.Unspecified
    assert selected(method) == 1
    assert selected(11) == 1  # plain int, as a serialized config would carry it

    # Every other concrete method lands elsewhere on this row.
    assert selected(RoutingMethodType.DeepSeekV3, n_group=1, topk_group=1) == 2
    assert selected(RoutingMethodType.MiniMax2) == 2
    for bias_ignoring in (
        RoutingMethodType.Default,
        RoutingMethodType.Renormalize,
        RoutingMethodType.RenormalizeNaive,
        RoutingMethodType.TopK,
        RoutingMethodType.SigmoidRenorm,
        RoutingMethodType.Sigmoid,
        RoutingMethodType.TopKSigmoid,
        RoutingMethodType.Llama4,
    ):
        assert selected(bias_ignoring) == 0, bias_ignoring

    # The sentinel itself is still rejected on both sides of the binding.
    with pytest.raises(ValueError):
        trtllm_gen_routing(logits, bias, RoutingMethodType.Unspecified, 1)
    with pytest.raises(Exception, match="invalid routing_method_type"):
        trtllm_gen_routing(logits, bias, RoutingMethodType.SqrtSoftplus + 1, 1)


def test_sqrt_softplus_rejects_groups_and_fused_shared_experts():
    logits = make_logits(4, 16, torch.float32, 0)
    bias = make_bias(16, torch.bfloat16, 1)
    with pytest.raises(Exception, match="n_group <= 1"):
        trtllm_gen_routing(
            logits, bias, RoutingMethodType.SqrtSoftplus, 4, n_group=8, topk_group=4
        )
    with pytest.raises(Exception, match="fusing shared expert"):
        trtllm_gen_routing(
            logits, bias, RoutingMethodType.SqrtSoftplus, 4, num_fused_shared_experts=1
        )


@pytest.mark.parametrize("num_tokens", [1, 8, 150])
@pytest.mark.parametrize("num_experts", [16, 128])
@pytest.mark.parametrize("tile_tokens_dim", [8, 32])
@pytest.mark.parametrize("logits_dtype", [torch.float32, torch.bfloat16])
def test_llama4_routing(num_tokens, num_experts, tile_tokens_dim, logits_dtype):
    # Llama4 is top-1 -> sigmoid; routing_reference_no_aux with
    # use_routing_scales_on_input=True computes exactly sigmoid(logits) scores.
    top_k = 1
    seed = stable_seed("llama4", num_tokens, num_experts)
    logits = make_logits(num_tokens, num_experts, logits_dtype, seed)
    run_and_check(
        RoutingMethodType.Llama4,
        lambda: routing_reference_no_aux(
            logits,
            None,
            top_k,
            0,
            0,
            1.0,
            tile_tokens_dim,
            use_routing_scales_on_input=True,
        ),
        logits,
        top_k,
        tile_tokens_dim,
    )


@pytest.mark.parametrize(
    "routing_method,num_experts,top_k",
    [
        (RoutingMethodType.Renormalize, 256, 8),
        (RoutingMethodType.DeepSeekV3, 256, 8),
    ],
    ids=["Renormalize", "DeepSeekV3"],
)
def test_large_batch_smoke(routing_method, num_experts, top_k):
    """One large-batch case per kernel family; the dense grids stay small."""
    num_tokens, tile_tokens_dim = 4096, 64
    seed = stable_seed("large", int(routing_method))
    logits = make_logits(num_tokens, num_experts, torch.float32, seed)
    if routing_method == RoutingMethodType.DeepSeekV3:
        bias = make_bias(num_experts, torch.bfloat16, seed + 1)
        run_and_check(
            routing_method,
            lambda: routing_reference_no_aux(
                logits, bias, top_k, 8, 4, 2.5, tile_tokens_dim
            ),
            logits,
            top_k,
            tile_tokens_dim,
            routing_bias=bias,
            n_group=8,
            topk_group=4,
            routed_scaling_factor=2.5,
        )
    else:
        run_and_check(
            routing_method,
            lambda: routing_reference_renormalize(
                logits, top_k, num_experts, tile_tokens_dim
            ),
            logits,
            top_k,
            tile_tokens_dim,
        )


def test_invalid_group_args_rejected():
    """Grouped-routing argument validation at the FFI boundary.

    These combinations are rejected by the binding (mirroring
    FusedMoeLauncher::check_routing) rather than by the Python wrapper, so the
    exception is whatever TVM-FFI raises for a failed ICHECK; match on the
    message instead of the type.
    """
    logits = make_logits(4, 16, torch.float32, 0)
    bias = make_bias(16, torch.bfloat16, 1)

    def expect(message, top_k=4, **kwargs):
        with pytest.raises(Exception, match=message):
            trtllm_gen_routing(
                logits, bias, RoutingMethodType.DeepSeekV3, top_k, **kwargs
            )

    expect("n_group should not be zero")  # DeepSeekV3 defaults n_group=0
    expect("must be divisible by n_group", n_group=3, topk_group=1)
    expect("topk_group must be given", n_group=4, topk_group=0)
    expect("not be smaller than topk_group", n_group=2, topk_group=4)
    expect("less than total number of experts", n_group=4, topk_group=1, top_k=4)
    expect("topk_group <= 4", n_group=8, topk_group=5, top_k=2)

    # Non-grouped methods must not carry group parameters.
    with pytest.raises(Exception, match="only supports n_group <= 1"):
        trtllm_gen_routing(
            logits, None, RoutingMethodType.Renormalize, 4, n_group=2, topk_group=1
        )


@pytest.mark.parametrize(
    "routing_method,n_group,topk_group",
    [
        # Only the grouped DeepSeek kernel populates the fused shared-expert
        # slots. Every other method dispatches to routingCustom, which leaves
        # them untouched, so they must reject num_fused_shared_experts > 0
        # rather than hand back uninitialized slots.
        pytest.param(RoutingMethodType.DeepSeekV3, 1, 1, id="DeepSeekV3_no_groups"),
        pytest.param(RoutingMethodType.MiniMax2, 0, 0, id="MiniMax2"),
        pytest.param(RoutingMethodType.Renormalize, 0, 0, id="Renormalize"),
        pytest.param(RoutingMethodType.Llama4, 0, 0, id="Llama4"),
    ],
)
def test_unsupported_fused_shared_experts_rejected(routing_method, n_group, topk_group):
    """Methods whose kernel cannot fill the shared-expert slots must error."""
    num_experts = 16
    logits = make_logits(4, num_experts, torch.float32, 0)
    bias = make_bias(num_experts, torch.bfloat16, 1)
    top_k = 1 if routing_method == RoutingMethodType.Llama4 else 4
    with pytest.raises(Exception, match="fusing shared expert"):
        trtllm_gen_routing(
            logits,
            bias,
            routing_method,
            top_k,
            num_fused_shared_experts=1,
            n_group=n_group,
            topk_group=topk_group,
        )


def test_invalid_args_rejected():
    logits = make_logits(4, 16, torch.float32, 0)
    with pytest.raises(ValueError):
        trtllm_gen_routing(logits, None, RoutingMethodType.Renormalize, 17)  # top_k > E
    with pytest.raises(ValueError):
        trtllm_gen_routing(
            logits, None, RoutingMethodType.Renormalize, 4, tile_tokens_dim=12
        )  # not a power of two
    with pytest.raises(ValueError):
        trtllm_gen_routing(logits, None, RoutingMethodType.Unspecified, 4)


# ---------------------------------------------------------------------------
# Expert-parallel shards
#
# Under EP the kernels still *select* over all `num_experts` — only the
# permutation is restricted to the local shard. So the global host oracle stays
# valid for selection, and the expected permutation is its per-expert histogram
# re-padded over the local experts alone. These cases are the only coverage of
# the `+ local_expert_offset` id reconstruction and the `-1` masking of
# out-of-shard slots in flashinfer/fused_moe/trtllm_gen_routing.py.
# ---------------------------------------------------------------------------


def check_ep_shard(
    result,
    permute_info,
    num_tokens,
    top_k,
    tile_tokens_dim,
    local_expert_offset,
    local_num_experts,
):
    """Shard-aware counterpart to check_selection/check_permutation."""
    device = result.topk_ids.device
    ref_ids = permute_info["topKIndices"].to(device).long()
    in_shard = (ref_ids >= local_expert_offset) & (
        ref_ids < local_expert_offset + local_num_experts
    )
    # A shard that happens to own every routed expert would make the masking
    # assertions vacuous.
    assert (~in_shard).any(), (
        "test case is not a partial shard: every selected expert is local"
    )

    # Selection: in-shard slots keep their *global* expert id, out-of-shard
    # slots are masked to -1. Intra-row order is not contractual, and -1 sorts
    # below every real id, so compare row-sorted.
    got_ids = result.topk_ids.long()
    want_ids = torch.where(in_shard, ref_ids, torch.full_like(ref_ids, -1))
    got_sorted, _ = torch.sort(got_ids, dim=1)
    want_sorted, _ = torch.sort(want_ids, dim=1)
    mismatched = (got_sorted != want_sorted).any(dim=1)
    assert not mismatched.any(), (
        f"sharded expert selection mismatch on {int(mismatched.sum())}/{num_tokens} "
        f"tokens; first bad token {int(mismatched.nonzero()[0])}: "
        f"got {got_sorted[mismatched][0].tolist()}, "
        f"want {want_sorted[mismatched][0].tolist()}"
    )

    active = got_ids >= 0
    assert (active.sum(dim=1) == in_shard.sum(dim=1)).all(), (
        "number of unmasked slots per token disagrees with the oracle"
    )

    # Expected padded layout: the oracle's global histogram, re-padded over the
    # local experts only.
    ref_counts = permute_info["numTokensPerExpert"].to(device).long()
    local_counts = ref_counts[
        local_expert_offset : local_expert_offset + local_num_experts
    ]
    padded = ((local_counts + tile_tokens_dim - 1) // tile_tokens_dim) * tile_tokens_dim
    prefix = torch.cat(
        [torch.zeros(1, dtype=torch.long, device=device), padded.cumsum(0)]
    )
    padded_size = int(result.total_num_padded_tokens.item())
    assert padded_size == int(prefix[-1]), (
        f"sharded padded token count {padded_size} != reference {int(prefix[-1])}"
    )

    e2p = result.expanded_idx_to_permuted_idx.long()
    assert (e2p >= 0).eq(active).all(), (
        "expanded_idx_to_permuted_idx masking disagrees with topk_ids masking"
    )

    live = e2p[active]
    assert ((live >= 0) & (live < padded_size)).all(), "permuted idx out of range"
    assert live.unique().numel() == live.numel(), "permuted idx not unique"

    # Each live slot lands in its *local* expert's padded segment.
    local_expert = got_ids[active] - local_expert_offset
    assert ((local_expert >= 0) & (local_expert < local_num_experts)).all(), (
        "reconstructed id outside the local shard"
    )
    in_segment = (live >= prefix[local_expert]) & (live < prefix[local_expert + 1])
    assert in_segment.all(), "permuted idx outside its local expert's padded segment"

    # Round-trip: the permuted slot maps back to the owning token.
    p2t = result.permuted_idx_to_token_idx.long()
    tokens = torch.arange(num_tokens, device=device).unsqueeze(1).expand_as(e2p)[active]
    assert (p2t[live] == tokens).all(), "permuted->token round trip failed"

    counts = torch.bincount(local_expert, minlength=local_num_experts)
    assert (counts == local_counts).all(), "per-local-expert token counts mismatch"


def run_and_check_ep_shard(
    routing_method,
    reference,
    logits,
    top_k,
    tile_tokens_dim,
    local_expert_offset,
    local_num_experts,
    routing_bias=None,
    **kwargs,
):
    permute_info, _ = reference()
    result = trtllm_gen_routing(
        logits,
        routing_bias,
        routing_method,
        top_k,
        tile_tokens_dim=tile_tokens_dim,
        local_expert_offset=local_expert_offset,
        local_num_experts=local_num_experts,
        **kwargs,
    )
    check_ep_shard(
        result,
        permute_info,
        logits.shape[0],
        top_k,
        tile_tokens_dim,
        local_expert_offset,
        local_num_experts,
    )


# (num_experts, top_k, local_expert_offset, local_num_experts). Every case is a
# partial shard, including a first shard (offset 0) and a last shard.
EP_SHARD_CASES = [
    pytest.param(16, 4, 0, 8, id="E16_k4_rank0"),
    pytest.param(16, 4, 8, 8, id="E16_k4_rank1"),
    pytest.param(16, 4, 4, 4, id="E16_k4_ep4_mid"),
    pytest.param(256, 8, 64, 64, id="E256_k8_rank1"),
    pytest.param(256, 8, 192, 64, id="E256_k8_last"),
    pytest.param(96, 8, 32, 32, id="E96_k8_mid"),
]


@pytest.mark.parametrize("num_tokens", [8, 150])
@pytest.mark.parametrize("tile_tokens_dim", [8, 32])
@pytest.mark.parametrize(
    "num_experts,top_k,local_expert_offset,local_num_experts", EP_SHARD_CASES
)
def test_ep_shard_renormalize(
    num_experts,
    top_k,
    local_expert_offset,
    local_num_experts,
    num_tokens,
    tile_tokens_dim,
):
    """EP shards through the routingCustom kernel family."""
    seed = stable_seed("ep", num_experts, top_k, local_expert_offset, num_tokens)
    logits = make_logits(num_tokens, num_experts, torch.float32, seed)
    run_and_check_ep_shard(
        RoutingMethodType.Renormalize,
        lambda: routing_reference_renormalize(
            logits, top_k, num_experts, tile_tokens_dim
        ),
        logits,
        top_k,
        tile_tokens_dim,
        local_expert_offset,
        local_num_experts,
    )


# (num_experts, n_group, topk_group, top_k, local_expert_offset,
# local_num_experts) — n_group > 1 so these land on the grouped DeepSeek kernel
# rather than the routingCustom SigmoidBias fast path.
DEEPSEEK_EP_SHARD_CASES = [
    pytest.param(256, 8, 4, 8, 0, 64, id="E256_rank0"),
    pytest.param(256, 8, 4, 8, 192, 64, id="E256_last"),
    pytest.param(128, 4, 2, 4, 32, 32, id="E128_mid"),
]


@pytest.mark.parametrize("num_tokens", [8, 150])
@pytest.mark.parametrize("tile_tokens_dim", [8, 32])
@pytest.mark.parametrize(
    "num_experts,n_group,topk_group,top_k,local_expert_offset,local_num_experts",
    DEEPSEEK_EP_SHARD_CASES,
)
def test_ep_shard_deepseekv3(
    num_experts,
    n_group,
    topk_group,
    top_k,
    local_expert_offset,
    local_num_experts,
    num_tokens,
    tile_tokens_dim,
):
    """EP shards through the grouped DeepSeekV3 kernel family."""
    seed = stable_seed("ep_ds", num_experts, top_k, local_expert_offset, num_tokens)
    logits = make_logits(num_tokens, num_experts, torch.float32, seed)
    bias = make_bias(num_experts, torch.bfloat16, seed + 1)
    routed_scaling_factor = 2.5
    run_and_check_ep_shard(
        RoutingMethodType.DeepSeekV3,
        lambda: routing_reference_no_aux(
            logits,
            bias,
            top_k,
            n_group,
            topk_group,
            routed_scaling_factor,
            tile_tokens_dim,
        ),
        logits,
        top_k,
        tile_tokens_dim,
        local_expert_offset,
        local_num_experts,
        routing_bias=bias,
        n_group=n_group,
        topk_group=topk_group,
        routed_scaling_factor=routed_scaling_factor,
    )
