"""
Test for NoAuxTc (DSv3 Fused Routing) Kernel

This test validates the NoAuxTc kernel against a reference implementation,
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

import torch
import pytest
from flashinfer.dsv3_ops import NoAuxTc
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
            group_mask = torch.zeros(n_group, dtype=torch.float32, device=self.device)
            group_mask[self.ref_group_indices[token_idx]] = 1.0
            expert_mask = group_mask.repeat_interleave(self.experts_per_group)

            # Mask and select top-k experts
            masked_biased_scores = self.biased_scores[token_idx] * expert_mask
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
        group_mask = torch.zeros(self.n_group, dtype=torch.float32, device=self.device)
        for g in groups:
            group_mask[g] = 1.0
        expert_mask = group_mask.repeat_interleave(self.experts_per_group)

        # Mask and select top-k experts
        masked_biased_scores = self.biased_scores[token_idx] * expert_mask
        _, topk_idx = torch.topk(
            masked_biased_scores, k=self.topk, dim=-1, largest=True, sorted=True
        )

        return set(topk_idx.tolist())


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
    Test NoAuxTc kernel against reference implementation.

    Validates:
    1. Expert selection equivalence (allowing for ties)
    2. Value correctness within numerical precision tolerance
    """

    # Skip invalid configurations
    if topk_group * n_group < topk or topk_group > n_group:
        pytest.skip(
            "Invalid configuration: topk_group * n_group < topk or topk_group > n_group"
        )
    if n_group > 1:
        if (
            topk > 8
            or num_experts / n_group > 32
            or num_experts / n_group * topk_group > 128
        ):
            pytest.skip("Invalid configuration: exceeds kernel limits for n_group > 1")
    else:
        if num_experts > 384 or topk > 8:
            pytest.skip("Invalid configuration: exceeds kernel limits for n_group = 1")

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

    NoAuxTc(
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
