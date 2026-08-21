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
    major, _ = get_compute_capability(torch.device("cuda"))
    if major not in (10, 12):
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
