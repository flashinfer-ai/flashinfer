"""
Regression test for issue #4267: FA2 attention bug with extreme negative logits.

Tests that attention correctly handles valid logits below the mask sentinel value.
Previously, math::inf was defined as 5e4 (a finite value), causing the running max
to clamp at -5e4. This led to exp() underflow and zero outputs for valid logits
below this threshold.

After the fix, math::inf uses true IEEE infinity, allowing correct handling of
extreme negative logits while maintaining proper masked semantics.
"""

import pytest
import torch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

torch_dtype_map = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def reference_prefill_attention(q, k, v, causal=False, scale=None):
    """
    Reference implementation of prefill attention using PyTorch SDPA.
    Uses FP64 for numerical stability in extreme-logit regime.
    """
    # Convert to FP64 for numerically stable computation
    q_fp64 = q.float().to(torch.float64)
    k_fp64 = k.float().to(torch.float64)
    v_fp64 = v.float().to(torch.float64)

    # Compute attention scale if not provided
    if scale is None:
        scale = q_fp64.shape[-1] ** -0.5

    # Scaled dot-product attention
    scores = torch.matmul(q_fp64, k_fp64.transpose(-2, -1)) * scale

    # Apply causal mask if needed
    if causal:
        seq_len = q_fp64.shape[1] if q_fp64.ndim == 3 else q_fp64.shape[0]
        kv_len = k_fp64.shape[1] if k_fp64.ndim == 3 else k_fp64.shape[0]
        mask = torch.triu(torch.ones(seq_len, kv_len, device=q.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))

    # Softmax and value projection
    attn_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, v_fp64)

    return output.to(q.dtype)


def create_test_case_with_target_logit(target_logit, dtype="bf16", num_heads=16, head_dim=128, kv_len=64, causal=False):
    """
    Create Q/K/V tensors such that the maximum QK logit equals target_logit.

    Strategy: All Q/K pairs produce logit = target_logit, except one pair that
    produces a slightly higher logit (target_logit + 5000) to avoid exact equality
    edge cases.

    Args:
        target_logit: Desired maximum QK logit value
        dtype: Data type for tensors
        num_heads: Number of attention heads
        head_dim: Dimension of each head
        kv_len: Length of key-value cache
        causal: Whether to use causal masking

    Returns:
        q, k, v tensors with the specified properties
    """
    device = "cuda"
    dtype_t = torch_dtype_map[dtype]

    # Q is all ones (shape: [1, num_heads, head_dim])
    q = torch.ones(1, num_heads, head_dim, dtype=dtype_t, device=device)

    # K produces target_logit for all positions, except position 0 which produces
    # logit = target_logit + 5000 (slightly higher to be the row maximum)
    k = torch.full((kv_len, 1, head_dim), target_logit / head_dim, dtype=dtype_t, device=device)
    k[0] = (target_logit + 5000.0) / head_dim

    # V is random
    v = torch.randn(kv_len, 1, head_dim, dtype=dtype_t, device=device)

    return q, k, v


def test_extreme_negative_logits_prefill_fp16():
    """Test FA2 prefill with FP16 and extreme negative logits."""
    pytest.importorskip("flashinfer")

    import flashinfer

    device = "cuda"
    dtype = "fp16"
    num_heads = 16
    head_dim = 128
    kv_len = 64

    # Test various logit values including those below the old -5e4 sentinel
    test_logits = [-1e4, -3e4, -5e4, -6e4, -1e5, -2e5]

    for target_logit in test_logits:
        q, k, v = create_test_case_with_target_logit(target_logit, dtype=dtype, num_heads=num_heads, head_dim=head_dim, kv_len=kv_len)

        # Compute FlashInfer output
        o_flashinfer = flashinfer.single_prefill_with_kv_cache(
            q, k, v, causal=False, backend="fa2"
        )

        # Compute reference output
        o_ref = reference_prefill_attention(q, k, v, causal=False)

        # Compute relative error
        rel_error = (o_flashinfer.float() - o_ref.float()).norm() / o_ref.norm()

        # Allow slightly higher tolerance for extremely negative logits due to numerical precision
        tolerance = 1e-3 if target_logit > -1e5 else 5e-3

        assert rel_error < tolerance, (
            f"Logit {target_logit}: relative error {rel_error:.6f} > {tolerance}. "
            f"FlashInfer norm: {o_flashinfer.float().norm():.6f}, "
            f"Reference norm: {o_ref.float().norm():.6f}"
        )

        print(f"✓ FP16 logit={target_logit:.0f}: rel_error={rel_error:.6f}")


def test_extreme_negative_logits_prefill_bf16():
    """Test FA2 prefill with BF16 and extreme negative logits."""
    pytest.importorskip("flashinfer")

    import flashinfer

    device = "cuda"
    dtype = "bf16"
    num_heads = 16
    head_dim = 128
    kv_len = 64

    # Test various logit values
    test_logits = [-1e4, -3e4, -5e4, -6e4, -1e5, -2e5, -2.45e5]

    for target_logit in test_logits:
        q, k, v = create_test_case_with_target_logit(target_logit, dtype=dtype, num_heads=num_heads, head_dim=head_dim, kv_len=kv_len)

        # Compute FlashInfer output
        o_flashinfer = flashinfer.single_prefill_with_kv_cache(
            q, k, v, causal=False, backend="fa2"
        )

        # Compute reference output
        o_ref = reference_prefill_attention(q, k, v, causal=False)

        # Compute relative error
        rel_error = (o_flashinfer.float() - o_ref.float()).norm() / o_ref.norm()

        # BF16 has less precision for extreme values
        tolerance = 1e-3 if target_logit > -1e5 else 1e-2

        assert rel_error < tolerance, (
            f"Logit {target_logit}: relative error {rel_error:.6f} > {tolerance}. "
            f"FlashInfer norm: {o_flashinfer.float().norm():.6f}, "
            f"Reference norm: {o_ref.float().norm():.6f}"
        )

        print(f"✓ BF16 logit={target_logit:.0f}: rel_error={rel_error:.6f}")


def test_extreme_negative_logits_with_masking():
    """Test that masked positions are still handled correctly with extreme logits."""
    pytest.importorskip("flashinfer")

    import flashinfer

    device = "cuda"
    dtype = "bf16"
    num_heads = 16
    head_dim = 128
    seq_len = 32

    # Create Q/K/V where some positions will be masked
    q = torch.randn(1, num_heads, head_dim, dtype=torch_dtype_map[dtype], device=device)

    # Scale K to produce extreme negative logits
    k_scale = -1e5 / head_dim
    k = torch.full((seq_len, 1, head_dim), k_scale, dtype=torch_dtype_map[dtype], device=device)

    # V is random
    v = torch.randn(seq_len, 1, head_dim, dtype=torch_dtype_map[dtype], device=device)

    # Test with causal mask (half the positions masked)
    o_flashinfer = flashinfer.single_prefill_with_kv_cache(
        q, k, v, causal=True, backend="fa2"
    )

    # Reference with causal mask
    o_ref = reference_prefill_attention(q, k, v, causal=True)

    rel_error = (o_flashinfer.float() - o_ref.float()).norm() / o_ref.norm()

    # Allow higher tolerance due to extreme values
    tolerance = 1e-2

    assert rel_error < tolerance, (
        f"Masked case: relative error {rel_error:.6f} > {tolerance}. "
        f"FlashInfer norm: {o_flashinfer.float().norm():.6f}, "
        f"Reference norm: {o_ref.float().norm():.6f}"
    )

    print(f"✓ Masked extreme logits: rel_error={rel_error:.6f}")


def test_below_old_sentinel_failure_boundary():
    """
    Test the specific failure boundary from issue #4267.

    The old sentinel was -5e4. This test verifies that logits just below
    this threshold are handled correctly.
    """
    pytest.importorskip("flashinfer")

    import flashinfer

    device = "cuda"
    num_heads = 16
    head_dim = 128
    kv_len = 64

    # Test around the old failure boundary
    test_cases = [
        (-4.5e4, "should_pass"),  # Above threshold
        (-5.0e4, "boundary"),     # At threshold
        (-5.5e4, "should_pass"),  # Just below threshold (FAILED in old code)
        (-6.0e4, "should_pass"),  # Below threshold (FAILED in old code)
        (-1.0e5, "should_pass"),  # Well below threshold (FAILED in old code)
    ]

    for target_logit, expected_behavior in test_cases:
        q, k, v = create_test_case_with_target_logit(target_logit, dtype="bf16", num_heads=num_heads, head_dim=head_dim, kv_len=kv_len)

        # Compute FlashInfer output
        o_flashinfer = flashinfer.single_prefill_with_kv_cache(
            q, k, v, causal=False, backend="fa2"
        )

        # Compute reference output
        o_ref = reference_prefill_attention(q, k, v, causal=False)

        # Check that output is not zero (the old bug symptom)
        output_norm = o_flashinfer.float().norm()

        assert output_norm > 1e-6, (
            f"Logit {target_logit}: Output norm is {output_norm:.6f} (near zero). "
            f"This indicates the bug from issue #4267 is NOT fixed."
        )

        # Verify correctness
        rel_error = (o_flashinfer.float() - o_ref.float()).norm() / o_ref.norm()
        tolerance = 1e-2

        assert rel_error < tolerance, (
            f"Logit {target_logit}: relative error {rel_error:.6f} > {tolerance}"
        )

        print(f"✓ Boundary test logit={target_logit:.0f} ({expected_behavior}): norm={output_norm:.4f}, rel_error={rel_error:.6f}")


if __name__ == "__main__":
    print("Running regression tests for issue #4267...")

    try:
        test_extreme_negative_logits_prefill_fp16()
        print("✓ test_extreme_negative_logits_prefill_fp16 PASSED")
    except Exception as e:
        print(f"✗ test_extreme_negative_logits_prefill_fp16 FAILED: {e}")

    try:
        test_extreme_negative_logits_prefill_bf16()
        print("✓ test_extreme_negative_logits_prefill_bf16 PASSED")
    except Exception as e:
        print(f"✗ test_extreme_negative_logits_prefill_bf16 FAILED: {e}")

    try:
        test_extreme_negative_logits_with_masking()
        print("✓ test_extreme_negative_logits_with_masking PASSED")
    except Exception as e:
        print(f"✗ test_extreme_negative_logits_with_masking FAILED: {e}")

    try:
        test_below_old_sentinel_failure_boundary()
        print("✓ test_below_old_sentinel_failure_boundary PASSED")
    except Exception as e:
        print(f"✗ test_below_old_sentinel_failure_boundary FAILED: {e}")

    print("\nAll tests completed!")
