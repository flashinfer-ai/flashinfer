"""
Test for fused_topk_deepseek (DSv3 Fused Routing) Kernel

This test validates the fused_topk_deepseek kernel against a reference implementation,
accounting for numerical precision and tie-breaking differences.

================================================================================
DSv3 ROUTING ALGORITHM
================================================================================

1. Compute: sigmoid(scores) + bias for each expert (biased scores)
2. Group experts and compute group scores (sum of top-2 experts per group)
3. Select top-k groups based on group scores
4. From selected groups, select top-k experts based on biased scores
5. Normalize selected experts: sigmoid_scores / sum(sigmoid_scores) * scale

================================================================================
VALIDATION LOGIC FLOW
================================================================================

The test performs TWO stages of validation for each token:

STAGE 1: EXPERT SELECTION VALIDATION
-------------------------------------
Checks if the kernel selected the correct (or acceptably tied) experts.

1. Are kernel_experts == ref_experts (same set)?
   YES → ✅ VALID (status: "exact")
         Continue to Stage 2 to validate output values
   NO  → Continue to step 2

2. Are kernel_groups == ref_groups (same groups selected)?
   YES → Continue to step 3 (same groups, different experts)
   NO  → Continue to step 4 (different groups)

3. SAME GROUPS, DIFFERENT EXPERTS
   Check if the differing experts have tied scores:
   - Compute score_diff = max(diff_expert_scores) - min(diff_expert_scores)
   - If score_diff < expert_tie_threshold:
     → ✅ VALID (status: "tied_experts")
   - Else:
     → ❌ INVALID (status: "score_mismatch")

4. DIFFERENT GROUPS
   a) Are the groups tied?
      - Compute all group scores (sum of top-2 experts per group)
      - Check if differing groups have similar scores
      - If group_score_diff < group_tie_threshold:
        → Groups are tied, continue to step 4b
      - Else:
        → ❌ INVALID (status: "different_groups")

   b) Are the experts correct within kernel's groups?
      - Compute expected_experts = top-k experts from kernel's selected groups
      - If kernel_experts == expected_experts:
        → ✅ VALID (status: "tied_groups")
      - Else, check if differing experts have tied scores:
        - Compute score_diff for differing experts
        - If score_diff < expert_tie_threshold:
          → ✅ VALID (status: "tied_groups")
        - Else:
          → ❌ INVALID (status: "tied_groups_but_wrong_experts")

STAGE 2: OUTPUT VALUE VALIDATION
---------------------------------
For tokens where the SAME experts were selected (status: "exact"):
- Compare kernel output values vs reference output values
- Both are normalized scores: sigmoid_scores / sum(sigmoid_scores) * scale
- Check: abs(kernel_values - ref_values) within tolerance
  - If within tolerance → ✅ VALID
  - Else → ❌ INVALID (value mismatch)

For tokens where DIFFERENT experts were selected (even if acceptably):
- SKIP value validation
- Reason: Different experts → different normalization sum → different values
- The expert selection validation already confirmed correctness

Tolerance (data-type dependent):
- bfloat16: rtol=0.1, atol=0.1
- float16:  rtol=0.05, atol=0.05
- float32:  rtol=0.01, atol=0.01

================================================================================
KEY CONCEPTS
================================================================================

1. **Group Ties**: When two groups have similar group scores (within threshold),
   selecting either group is valid. The kernel may pick a different group than
   the reference due to tie-breaking.

2. **Expert Ties**: When experts have similar biased scores (within threshold),
   selecting any of them is valid. The kernel may pick different experts due
   to tie-breaking.

3. **Tied Groups → Verify Experts**: When different groups are selected due to
   ties, we must still verify that the kernel selected the correct top-k experts
   WITHIN its chosen groups (not compare across different groups).

4. **Float32 Internal Computation**: The kernel computes internally in float32
   even when inputs are float16/bfloat16. The reference must match this to
   ensure consistent group/expert selection.

================================================================================
THRESHOLDS (Data-Type Dependent)
================================================================================

                    Expert Tie      Group Tie
                    Threshold       Threshold
    bfloat16:       1.0             0.05
    float16:        0.5             0.02
    float32:        0.2             0.01

Group thresholds are higher because group scores are sums of 2 values,
accumulating more numerical error.

================================================================================
"""

import pytest
import torch

from flashinfer.dsv3_ops import fused_topk_deepseek

# from flashinfer.utils import get_compute_capability


class DSv3RoutingGroundTruth:
    """
    Computes and stores all ground truth data for DSv3 routing.
    Performs all computations in float32 to match kernel behavior.
    """

    def __init__(
        self, scores, bias, n_group, topk_group, topk, routed_scaling_factor, data_type
    ):
        self.num_tokens = scores.shape[0]
        self.num_experts = scores.shape[1]
        self.n_group = n_group
        self.topk_group = topk_group
        self.topk = topk
        self.routed_scaling_factor = routed_scaling_factor
        self.experts_per_group = self.num_experts // n_group
        self.device = scores.device

        # Set thresholds based on data type
        if data_type == torch.bfloat16:
            self.expert_tie_threshold = 1.0
            self.group_tie_threshold = 0.05
        elif data_type == torch.float16:
            self.expert_tie_threshold = 0.5
            self.group_tie_threshold = 0.02
        else:  # float32
            self.expert_tie_threshold = 0.2
            self.group_tie_threshold = 0.01

        # Convert to float32 to match kernel's internal computation
        scores_f32 = scores.to(torch.float32)
        bias_f32 = bias.to(torch.float32)

        # Compute sigmoid and biased scores
        self.sigmoid_scores = torch.sigmoid(scores_f32)
        self.biased_scores = self.sigmoid_scores + bias_f32

        # Reshape for group-wise operations
        scores_reshaped = self.biased_scores.view(
            self.num_tokens, n_group, self.experts_per_group
        )

        # Compute group scores (sum of top-2 experts per group)
        top2_per_group = torch.topk(
            scores_reshaped, k=2, dim=-1, largest=True, sorted=True
        )[0]
        self.group_scores = torch.sum(top2_per_group, dim=-1)

        # Reference group selection
        _, self.ref_group_indices = torch.topk(
            self.group_scores, k=topk_group, dim=-1, largest=True, sorted=True
        )

        # Identify tied groups for each token
        self.tied_group_sets = []
        for token_idx in range(self.num_tokens):
            tied_groups = set()
            group_scores_token = self.group_scores[token_idx]

            for g1 in range(n_group):
                for g2 in range(g1 + 1, n_group):
                    score_diff = abs(group_scores_token[g1] - group_scores_token[g2])
                    if score_diff < self.group_tie_threshold:
                        tied_groups.add(g1)
                        tied_groups.add(g2)

            self.tied_group_sets.append(tied_groups)

        # Compute reference expert selection and normalization
        self.ref_expert_indices = torch.zeros(
            self.num_tokens, topk, dtype=torch.long, device=self.device
        )
        self.ref_expert_values = torch.zeros(
            self.num_tokens, topk, dtype=torch.float32, device=self.device
        )

        for token_idx in range(self.num_tokens):
            # Create mask for selected groups
            group_mask = torch.zeros(n_group, dtype=torch.bool, device=self.device)
            group_mask[self.ref_group_indices[token_idx]] = True
            expert_mask = group_mask.repeat_interleave(self.experts_per_group)

            # Mask and select top-k experts
            masked_biased_scores = self.biased_scores[token_idx].masked_fill(
                ~expert_mask, float("-inf")
            )
            _, topk_idx = torch.topk(
                masked_biased_scores, k=topk, dim=-1, largest=True, sorted=True
            )

            # Normalize selected experts
            selected_sigmoid_scores = self.sigmoid_scores[token_idx][topk_idx]
            score_sum = selected_sigmoid_scores.sum() + 1e-20
            normalized_scores = (
                selected_sigmoid_scores / score_sum * routed_scaling_factor
            )

            # Sort by normalized scores
            sorted_vals, sorted_idx = torch.sort(normalized_scores, descending=True)
            self.ref_expert_values[token_idx] = sorted_vals
            self.ref_expert_indices[token_idx] = topk_idx[sorted_idx]

    def get_expert_group(self, expert_id):
        """Return which group an expert belongs to."""
        return expert_id // self.experts_per_group

    def is_valid_group_selection(self, token_idx, selected_groups):
        """Check if a set of selected groups is valid (exact match or tied)."""
        ref_groups = set(self.ref_group_indices[token_idx].tolist())
        selected_groups_set = set(selected_groups)

        if selected_groups_set == ref_groups:
            return True, "exact"

        if self.n_group > 1:
            diff_groups = selected_groups_set.symmetric_difference(ref_groups)
            tied_groups = self.tied_group_sets[token_idx]

            if diff_groups and diff_groups.issubset(tied_groups):
                return True, "tied_groups"

        return False, "different_groups"

    def is_valid_expert_selection(self, token_idx, selected_experts):
        """Check if a set of selected experts is valid (exact match or tied)."""
        ref_experts = set(self.ref_expert_indices[token_idx].tolist())
        selected_experts_set = set(selected_experts)

        if selected_experts_set == ref_experts:
            return True, "exact"

        # Check group-level validity
        selected_groups = set(self.get_expert_group(e) for e in selected_experts)
        ref_groups = set(self.ref_group_indices[token_idx].tolist())

        # If different groups selected
        if selected_groups != ref_groups:
            is_valid_groups, group_reason = self.is_valid_group_selection(
                token_idx, list(selected_groups)
            )
            if not is_valid_groups:
                # Groups are different and not tied - invalid
                return False, group_reason

            # Groups are tied - now check if kernel selected correct top-k within its groups
            expected_experts_in_kernel_groups = self._get_topk_experts_from_groups(
                token_idx, list(selected_groups)
            )

            # Check if kernel's selection matches expected experts (exact or tied)
            if selected_experts_set != expected_experts_in_kernel_groups:
                # Different experts - check if they have tied scores
                diff_experts = selected_experts_set.symmetric_difference(
                    expected_experts_in_kernel_groups
                )
                biased_scores_token = self.biased_scores[token_idx]
                diff_expert_scores = torch.tensor(
                    [biased_scores_token[e].item() for e in diff_experts]
                )
                score_range = diff_expert_scores.max() - diff_expert_scores.min()

                if score_range >= self.expert_tie_threshold:
                    # Experts are wrong (not tied) - invalid even though groups are tied
                    return (
                        False,
                        f"tied_groups_but_wrong_experts_score_diff={score_range:.6f}",
                    )

            # Groups are tied and experts are correct (or acceptably tied)
            return True, "tied_groups"

        # Same groups but different experts - check expert-level ties
        diff_experts = selected_experts_set.symmetric_difference(ref_experts)
        if diff_experts:
            biased_scores_token = self.biased_scores[token_idx]
            diff_expert_scores = torch.tensor(
                [biased_scores_token[e].item() for e in diff_experts]
            )
            score_range = diff_expert_scores.max() - diff_expert_scores.min()

            if score_range < self.expert_tie_threshold:
                return True, "tied_experts"
            else:
                return (
                    False,
                    f"score_diff={score_range:.6f}_threshold={self.expert_tie_threshold:.6f}",
                )

        return True, "exact"

    def _get_topk_experts_from_groups(self, token_idx, groups):
        """
        Get the expected top-k experts from specified groups.
        This computes what experts SHOULD be selected if these groups were chosen.
        """
        # Create mask for specified groups
        group_mask = torch.zeros(self.n_group, dtype=torch.bool, device=self.device)
        for g in groups:
            group_mask[g] = True
        expert_mask = group_mask.repeat_interleave(self.experts_per_group)

        # Mask and select top-k experts
        masked_biased_scores = self.biased_scores[token_idx].masked_fill(
            ~expert_mask, float("-inf")
        )
        _, topk_idx = torch.topk(
            masked_biased_scores, k=self.topk, dim=-1, largest=True, sorted=True
        )

        return set(topk_idx.tolist())


def test_ground_truth_excludes_unselected_groups_with_negative_scores():
    """Keep unselected groups excluded even when a selected expert scores below zero."""
    scores = torch.zeros((1, 8), dtype=torch.float32)
    bias = torch.tensor(
        [[2.5, 1.5, 0.5, -1.5, 0.0, -0.1, -0.2, -0.3]],
        dtype=torch.float32,
    )

    ground_truth = DSv3RoutingGroundTruth(
        scores,
        bias,
        n_group=2,
        topk_group=1,
        topk=4,
        routed_scaling_factor=1.0,
        data_type=torch.float32,
    )

    expected = {0, 1, 2, 3}
    assert set(ground_truth.ref_expert_indices[0].tolist()) == expected
    assert ground_truth._get_topk_experts_from_groups(0, [0]) == expected


def validate_expert_selection(ground_truth, topk_indices_kernel, topk_values_kernel):
    """Validate kernel outputs and provide detailed debug info for failures."""
    num_tokens = topk_indices_kernel.shape[0]
    tokens_with_different_experts = set()

    for token_idx in range(num_tokens):
        kernel_experts = topk_indices_kernel[token_idx].tolist()
        ref_experts = ground_truth.ref_expert_indices[token_idx].tolist()

        # Same experts - valid
        if set(kernel_experts) == set(ref_experts):
            continue

        # Different experts - mark for value comparison skip
        tokens_with_different_experts.add(token_idx)

        # Validate the selection
        is_valid, reason = ground_truth.is_valid_expert_selection(
            token_idx, kernel_experts
        )

        if not is_valid:
            return False, tokens_with_different_experts

    return True, tokens_with_different_experts


def validate_values(ground_truth, topk_values_kernel, tokens_to_skip, data_type):
    """Validate that output values match reference within tolerance."""
    # Set tolerance based on data type
    if data_type == torch.bfloat16:
        rtol, atol = 0.1, 0.1
    elif data_type == torch.float16:
        rtol, atol = 0.05, 0.05
    else:  # float32
        rtol, atol = 0.01, 0.01

    num_tokens = topk_values_kernel.shape[0]

    # Create mask for tokens to check
    tokens_to_check = torch.ones(num_tokens, dtype=torch.bool)
    for token_idx in tokens_to_skip:
        tokens_to_check[token_idx] = False

    if not tokens_to_check.any():
        return

    # Compare values
    ref_values = ground_truth.ref_expert_values[tokens_to_check].float()
    kernel_values = topk_values_kernel[tokens_to_check].float()

    try:
        torch.testing.assert_close(
            ref_values,
            kernel_values,
            rtol=rtol,
            atol=atol,
        )
    except AssertionError:
        # Find and report first mismatch
        for token_idx in range(num_tokens):
            if not tokens_to_check[token_idx]:
                continue

            ref_vals = ground_truth.ref_expert_values[token_idx].float()
            kernel_vals = topk_values_kernel[token_idx].float()

            if not torch.allclose(ref_vals, kernel_vals, rtol=rtol, atol=atol):
                diff = (kernel_vals - ref_vals).abs()
                max_diff = diff.max().item()
                max_diff_idx = diff.argmax().item()

                print(f"\n{'=' * 80}")
                print(f"VALUE MISMATCH - Token {token_idx}")
                print(f"{'=' * 80}")
                print(f"Tolerance: rtol={rtol}, atol={atol}")
                print(f"Max difference: {max_diff:.6f} at position {max_diff_idx}")
                print(f"\nReference values: {ref_vals.tolist()}")
                print(f"Kernel values:    {kernel_vals.tolist()}")
                print(f"Absolute diff:    {diff.tolist()}")
                print(
                    f"Expert indices:   {ground_truth.ref_expert_indices[token_idx].tolist()}"
                )
                break

        raise


def _is_supported_dsv3_config(num_experts, n_group, topk_group, topk):
    """Mirror the public validator so parametrized cases it rejects are skipped."""
    if n_group <= 0 or num_experts % n_group != 0:
        return False
    if topk_group <= 0 or topk_group > n_group:
        return False
    if topk <= 0 or topk > 8:
        return False
    experts_per_group = num_experts // n_group
    if topk > topk_group * experts_per_group:
        return False
    if n_group > 1:
        return (
            n_group <= 8
            and topk_group <= 4
            and 2 <= experts_per_group <= 32
            and experts_per_group * topk_group <= 128
        )
    return num_experts <= 384


@pytest.mark.parametrize("backend", ["default", "cake"])
@pytest.mark.parametrize(
    "num_experts,n_group,topk_group,topk",
    [
        pytest.param(256, 8, 4, 8, id="grouped-k8g4"),
        pytest.param(128, 4, 2, 4, id="grouped-general"),
        pytest.param(128, 1, 1, 1, id="single128"),
        pytest.param(384, 1, 1, 1, id="single384"),
    ],
)
@pytest.mark.parametrize("data_type", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("bias_type", [torch.float32, torch.float16, torch.bfloat16])
def test_dsv3_fused_routing_backend_correctness(
    backend, num_experts, n_group, topk_group, topk, data_type, bias_type
):
    """Exercise every Cake schedule and dtype pair alongside the default backend."""

    if backend == "cake" and torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
    ):
        pytest.skip("Cake fused routing requires SM100 or SM103")

    num_tokens = 7
    torch.manual_seed(42)
    scores = torch.randn(num_tokens, num_experts, device="cuda", dtype=data_type)
    bias = torch.randn(num_experts, device="cuda", dtype=bias_type)
    routed_scaling_factor = 1.0
    ground_truth = DSv3RoutingGroundTruth(
        scores,
        bias,
        n_group,
        topk_group,
        topk,
        routed_scaling_factor,
        data_type,
    )

    topk_values = torch.empty(num_tokens, topk, device="cuda", dtype=data_type)
    topk_indices = torch.empty(num_tokens, topk, device="cuda", dtype=torch.int32)
    routing_replay_out = torch.empty(num_tokens, topk, device="cuda", dtype=torch.int16)
    fused_topk_deepseek(
        scores,
        bias,
        n_group,
        topk_group,
        topk,
        routed_scaling_factor,
        topk_values,
        topk_indices,
        routing_replay_out=routing_replay_out,
        backend=backend,
    )

    for token_idx in range(num_tokens):
        assert set(routing_replay_out[token_idx].tolist()) == set(
            topk_indices[token_idx].tolist()
        )

    sorted_values, sorted_order = torch.sort(topk_values, dim=-1, descending=True)
    sorted_indices = topk_indices.gather(1, sorted_order)
    all_valid, tokens_with_different_experts = validate_expert_selection(
        ground_truth, sorted_indices, sorted_values
    )
    assert all_valid
    validate_values(
        ground_truth, sorted_values, tokens_with_different_experts, data_type
    )


@pytest.mark.parametrize("num_experts", [256, 384])
@pytest.mark.parametrize("topk", [2, 4, 8])
def test_dsv3_fused_routing_ungrouped_read_mask(num_experts, topk):
    """Regression: ``n_group == 1`` multi-warp shared-memory read mask.

    Stage one fills only the first ``topk`` of each 8-slot block, so a ``topk=8``
    warmup with zero bias followed by a query with bias ``-2`` leaves stale
    positive slot values that must be masked out; ``topk=8`` is the control.
    """
    num_tokens = 7
    routed_scaling_factor = 1.0
    device = "cuda"

    generator = torch.Generator(device=device).manual_seed(4867)
    ramp = torch.linspace(0.0, -1.0, num_experts, device=device, dtype=torch.float32)
    scores = torch.stack(
        [
            ramp[torch.randperm(num_experts, device=device, generator=generator)]
            for _ in range(num_tokens)
        ]
    )
    bias = torch.full((num_experts,), -2.0, device=device, dtype=torch.float32)

    ground_truth = DSv3RoutingGroundTruth(
        scores.clone(), bias.clone(), 1, 1, topk, routed_scaling_factor, torch.float32
    )

    # Warmup uses a zero bias, so every stale slot holds a positive value.
    warm_values = torch.empty(num_tokens, 8, device=device, dtype=torch.float32)
    warm_indices = torch.empty(num_tokens, 8, device=device, dtype=torch.int32)
    fused_topk_deepseek(
        scores,
        torch.zeros_like(bias),
        1,
        1,
        8,
        routed_scaling_factor,
        warm_values,
        warm_indices,
    )

    topk_values = torch.empty(num_tokens, topk, device=device, dtype=torch.float32)
    topk_indices = torch.empty(num_tokens, topk, device=device, dtype=torch.int32)
    fused_topk_deepseek(
        scores, bias, 1, 1, topk, routed_scaling_factor, topk_values, topk_indices
    )

    for token_idx in range(num_tokens):
        kernel_experts = set(topk_indices[token_idx].tolist())
        reference_experts = set(ground_truth.ref_expert_indices[token_idx].tolist())
        assert kernel_experts == reference_experts, (
            f"token {token_idx}: kernel {sorted(kernel_experts)} != "
            f"reference {sorted(reference_experts)}"
        )

    sorted_values, _ = torch.sort(topk_values, dim=-1, descending=True)
    torch.testing.assert_close(
        sorted_values, ground_truth.ref_expert_values, rtol=2e-6, atol=2e-6
    )


@pytest.mark.parametrize("num_tokens", [1, 8, 16, 64])
@pytest.mark.parametrize("num_experts", [256, 384])
@pytest.mark.parametrize("topk", [1, 2, 4, 8])
@pytest.mark.parametrize("n_group", [1, 2, 4, 8])
@pytest.mark.parametrize("topk_group", [1, 2, 4, 8])
@pytest.mark.parametrize("data_type", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("bias_type", [torch.float32, torch.float16, torch.bfloat16])
def test_dsv3_fused_routing_op(
    num_tokens, num_experts, topk, n_group, topk_group, data_type, bias_type
):
    """
    Test fused_topk_deepseek kernel against reference implementation.

    Validates:
    1. Expert selection equivalence (allowing for ties)
    2. Value correctness within numerical precision tolerance
    """

    # Skip invalid configurations
    if not _is_supported_dsv3_config(num_experts, n_group, topk_group, topk):
        pytest.skip("Invalid configuration for fused_topk_deepseek")

    # Generate random inputs
    torch.manual_seed(42)
    scores = torch.randn(num_tokens, num_experts, device="cuda", dtype=data_type)
    bias = torch.randn(num_experts, device="cuda", dtype=bias_type)
    routed_scaling_factor = 1.0

    # Compute ground truth
    ground_truth = DSv3RoutingGroundTruth(
        scores.clone(),
        bias.clone(),
        n_group,
        topk_group,
        topk,
        routed_scaling_factor,
        data_type,
    )

    # Run kernel
    topk_values = torch.empty(num_tokens, topk, device="cuda", dtype=data_type)
    topk_indices = torch.zeros(num_tokens, topk, device="cuda", dtype=torch.int32)

    fused_topk_deepseek(
        scores,
        bias,
        n_group,
        topk_group,
        topk,
        routed_scaling_factor,
        topk_values,
        topk_indices,
        launch_with_pdl=True,
    )

    # Sort kernel outputs for stable comparison
    sorted_vals, sorted_idx = torch.sort(topk_values, dim=-1, descending=True)
    topk_indices = topk_indices.gather(1, sorted_idx)

    # Validate expert selection
    all_valid, tokens_with_different_experts = validate_expert_selection(
        ground_truth, topk_indices, sorted_vals
    )

    if not all_valid:
        pytest.fail("Expert selection mismatch not due to acceptable ties")

    # Validate values
    validate_values(ground_truth, sorted_vals, tokens_with_different_experts, data_type)


@pytest.mark.parametrize("num_tokens", [1, 8, 64])
@pytest.mark.parametrize("num_experts", [256])
@pytest.mark.parametrize("topk", [1, 4, 8])
@pytest.mark.parametrize("n_group", [1, 8])
@pytest.mark.parametrize("topk_group", [1, 4])
@pytest.mark.parametrize("data_type", [torch.bfloat16, torch.float16])
def test_routing_replay_out_extended(
    num_tokens, num_experts, topk, n_group, topk_group, data_type
):
    """
    Test that routing_replay_out records the same expert IDs as topk_indices.

    Extended parametrization covering larger token counts (8, 64).
    """
    if not _is_supported_dsv3_config(num_experts, n_group, topk_group, topk):
        pytest.skip("Invalid configuration for fused_topk_deepseek")

    torch.manual_seed(42)
    device = "cuda"
    scores = torch.randn(num_tokens, num_experts, device=device, dtype=data_type)
    bias = torch.randn(num_experts, device=device, dtype=data_type)
    routed_scaling_factor = 1.0

    topk_values = torch.empty(num_tokens, topk, device=device, dtype=data_type)
    topk_indices = torch.zeros(num_tokens, topk, device=device, dtype=torch.int32)
    routing_replay_out = torch.full(
        (num_tokens, topk), -1, device=device, dtype=torch.int16
    )

    fused_topk_deepseek(
        scores,
        bias,
        n_group,
        topk_group,
        topk,
        routed_scaling_factor,
        topk_values,
        topk_indices,
        launch_with_pdl=True,
        routing_replay_out=routing_replay_out,
    )

    # routing_replay_out should contain the same expert IDs as topk_indices (per token)
    for t in range(num_tokens):
        replay_set = set(routing_replay_out[t].tolist())
        indices_set = set(topk_indices[t].tolist())
        assert replay_set == indices_set, (
            f"Token {t}: routing_replay_out experts {replay_set} "
            f"!= topk_indices experts {indices_set}"
        )

    # Verify None produces identical results (no side effects from replay)
    topk_values_no_replay = torch.empty(
        num_tokens, topk, device=device, dtype=data_type
    )
    topk_indices_no_replay = torch.zeros(
        num_tokens, topk, device=device, dtype=torch.int32
    )

    fused_topk_deepseek(
        scores,
        bias,
        n_group,
        topk_group,
        topk,
        routed_scaling_factor,
        topk_values_no_replay,
        topk_indices_no_replay,
        launch_with_pdl=True,
        routing_replay_out=None,
    )

    torch.testing.assert_close(topk_values, topk_values_no_replay)
    torch.testing.assert_close(topk_indices, topk_indices_no_replay)


@pytest.mark.parametrize(
    "num_experts,n_group,topk_group,topk",
    [
        pytest.param(16, 8, 4, 8, id="expert-capacity-two"),
        pytest.param(16, 8, 2, 4, id="two-extreme-groups"),
        pytest.param(64, 8, 4, 4, id="four-tracked-groups"),
        pytest.param(128, 8, 4, 8, id="eight-groups-at-warp-limit"),
        pytest.param(15, 5, 2, 6, id="five-groups-three-experts-per-group"),
    ],
)
@pytest.mark.parametrize("data_type", [torch.float32, torch.float16, torch.bfloat16])
def test_dsv3_fused_routing_grouped_small_capacity_numeric(
    num_experts, n_group, topk_group, topk, data_type
):
    """Verify expert IDs and weights at legal grouped-capacity boundaries.

    Widely separated biases avoid ties in expert and group selection.
    """
    num_tokens = 32
    experts_per_group = num_experts // n_group

    generator = torch.Generator(device="cuda").manual_seed(20240607)
    scores = torch.randn(
        num_tokens,
        num_experts,
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    ).to(data_type)

    # A fixed permutation ranks the groups and the experts inside a group keep a
    # strict order, so the biased scores are unique and widely spaced.
    group_rank = torch.randperm(n_group, device="cuda", generator=generator)
    expert_rank = (
        group_rank.repeat_interleave(experts_per_group) * experts_per_group
        + torch.arange(num_experts, device="cuda") % experts_per_group
    )
    bias = (-expert_rank.to(torch.float32) * 64.0).to(data_type)

    ground_truth = DSv3RoutingGroundTruth(
        scores.clone(), bias.clone(), n_group, topk_group, topk, 1.0, data_type
    )

    topk_values = torch.empty(num_tokens, topk, device="cuda", dtype=data_type)
    topk_indices = torch.zeros(num_tokens, topk, device="cuda", dtype=torch.int32)
    fused_topk_deepseek(
        scores,
        bias,
        n_group,
        topk_group,
        topk,
        1.0,
        topk_values,
        topk_indices,
        launch_with_pdl=True,
    )

    # Account for float32 normalization and the output dtype's rounding.
    rtol, atol = {
        torch.bfloat16: (6e-3, 6e-3),
        torch.float16: (8e-4, 8e-4),
        torch.float32: (1e-5, 1e-5),
    }[data_type]

    for token_idx in range(num_tokens):
        kernel_experts = set(topk_indices[token_idx].tolist())
        reference_experts = set(ground_truth.ref_expert_indices[token_idx].tolist())
        assert kernel_experts == reference_experts, (
            f"token {token_idx}: kernel {sorted(kernel_experts)} != "
            f"reference {sorted(reference_experts)}"
        )

    # Gather both outputs onto the expert axis so the comparison checks each
    # expert weight instead of each output slot.
    reference_weights = torch.zeros(
        num_tokens, num_experts, device="cuda", dtype=torch.float32
    )
    reference_weights.scatter_(
        1, ground_truth.ref_expert_indices, ground_truth.ref_expert_values
    )
    kernel_weights = torch.zeros(
        num_tokens, num_experts, device="cuda", dtype=torch.float32
    )
    kernel_weights.scatter_(1, topk_indices.long(), topk_values.float())
    torch.testing.assert_close(kernel_weights, reference_weights, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "num_experts,n_group,topk_group,topk,match",
    [
        pytest.param(
            256,
            0,
            4,
            8,
            r"n_group should be greater than or equal to 1",
            id="n-group-zero-icheck",
        ),
        pytest.param(
            256,
            16,
            4,
            8,
            r"n_group should be smaller than or equal to 8",
            id="n-group-above-warp-count-icheck",
        ),
        pytest.param(
            256,
            8,
            0,
            8,
            r"topk_group should be between 1 and n_group",
            id="topk-group-zero-icheck",
        ),
        pytest.param(
            64,
            8,
            5,
            8,
            r"unsupported configuration \(n_group=8, num_experts=64, topk_group=5\)",
            id="topk-group-above-tracked-groups-gate",
        ),
        pytest.param(
            8,
            8,
            4,
            4,
            r"unsupported configuration \(n_group=8, num_experts=8, topk_group=4\)",
            id="experts-per-group-below-top-2-gate",
        ),
        pytest.param(
            256,
            8,
            4,
            0,
            r"topk should be between 1 and 8",
            id="topk-zero-icheck",
        ),
        pytest.param(
            256,
            8,
            4,
            9,
            r"topk should be between 1 and 8",
            id="topk-above-8-icheck",
        ),
        pytest.param(
            32,
            8,
            1,
            8,
            r"topk should be smaller than or equal to the number of experts reachable",
            id="topk-above-reachable-icheck",
        ),
    ],
)
def test_dsv3_fused_routing_grouped_contract_native_bypass(
    num_experts, n_group, topk_group, topk, match
):
    """``skip_check=True`` still rejects unsupported grouped configurations.

    The public validator is bypassed, so the native binding and the launch gate
    have to reject each case themselves, and none of them may write to the
    output tensors: the sentinels below must survive the raise unchanged.
    """
    num_tokens = 2
    scores = torch.zeros(num_tokens, num_experts, device="cuda", dtype=torch.float32)
    bias = torch.zeros(num_experts, device="cuda", dtype=torch.float32)
    topk_values = torch.full(
        (num_tokens, topk), -7.0, device="cuda", dtype=torch.float32
    )
    topk_indices = torch.full((num_tokens, topk), -1, device="cuda", dtype=torch.int32)
    routing_replay_out = torch.full(
        (num_tokens, topk), -3, device="cuda", dtype=torch.int16
    )

    with pytest.raises(RuntimeError, match=match):
        fused_topk_deepseek(
            scores,
            bias,
            n_group,
            topk_group,
            topk,
            1.0,
            topk_values,
            topk_indices,
            skip_check=True,
            routing_replay_out=routing_replay_out,
        )

    torch.cuda.synchronize()
    assert torch.equal(topk_values, torch.full_like(topk_values, -7.0))
    assert torch.equal(topk_indices, torch.full_like(topk_indices, -1))
    assert torch.equal(routing_replay_out, torch.full_like(routing_replay_out, -3))


@pytest.mark.parametrize("num_tokens", [1, 7, 32])
@pytest.mark.parametrize("num_experts", [256])
@pytest.mark.parametrize("topk", [1, 4, 8])
@pytest.mark.parametrize("n_group", [1, 8])
@pytest.mark.parametrize("topk_group", [1, 4])
@pytest.mark.parametrize("data_type", [torch.bfloat16, torch.float16])
def test_routing_replay_out(
    num_tokens, num_experts, topk, n_group, topk_group, data_type
):
    """
    Test that routing_replay_out records the same expert IDs as topk_indices.

    The routing replay feature writes selected expert IDs (int16) into an
    optional output tensor during the fused routing kernel. This test verifies
    that routing_replay_out matches topk_indices (as sets per token), and that
    passing None produces identical routing results (no side effects).
    """
    if not _is_supported_dsv3_config(num_experts, n_group, topk_group, topk):
        pytest.skip("Invalid configuration for fused_topk_deepseek")

    torch.manual_seed(42)
    device = "cuda"
    scores = torch.randn(num_tokens, num_experts, device=device, dtype=data_type)
    bias = torch.randn(num_experts, device=device, dtype=data_type)
    routed_scaling_factor = 1.0

    topk_values = torch.empty(num_tokens, topk, device=device, dtype=data_type)
    topk_indices = torch.zeros(num_tokens, topk, device=device, dtype=torch.int32)
    routing_replay_out = torch.full(
        (num_tokens, topk), -1, device=device, dtype=torch.int16
    )

    fused_topk_deepseek(
        scores,
        bias,
        n_group,
        topk_group,
        topk,
        routed_scaling_factor,
        topk_values,
        topk_indices,
        launch_with_pdl=True,
        routing_replay_out=routing_replay_out,
    )

    # routing_replay_out should contain the same expert IDs as topk_indices (per token)
    for t in range(num_tokens):
        replay_set = set(routing_replay_out[t].tolist())
        indices_set = set(topk_indices[t].tolist())
        assert replay_set == indices_set, (
            f"Token {t}: routing_replay_out experts {replay_set} "
            f"!= topk_indices experts {indices_set}"
        )

    # Verify None produces identical results (no side effects from replay)
    topk_values_no_replay = torch.empty(
        num_tokens, topk, device=device, dtype=data_type
    )
    topk_indices_no_replay = torch.zeros(
        num_tokens, topk, device=device, dtype=torch.int32
    )

    fused_topk_deepseek(
        scores,
        bias,
        n_group,
        topk_group,
        topk,
        routed_scaling_factor,
        topk_values_no_replay,
        topk_indices_no_replay,
        launch_with_pdl=True,
        routing_replay_out=None,
    )

    torch.testing.assert_close(topk_values, topk_values_no_replay)
    torch.testing.assert_close(topk_indices, topk_indices_no_replay)
