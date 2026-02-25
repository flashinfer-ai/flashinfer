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
"""

import pytest
import torch
import math
import random

from flashinfer.gdn_decode import gated_delta_rule_decode_pretranspose
from flashinfer.utils import get_compute_capability

try:
    from .reference_delta_rule import decode_delta_rule
except ImportError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from reference_delta_rule import decode_delta_rule


def _skip_if_not_sm90_or_later():
    """Skip test if not Hopper (SM90+) or Blackwell (SM100+) architecture."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    cc = get_compute_capability(torch.device("cuda"))
    if cc[0] not in [9, 10, 11, 12]:
        pytest.skip(f"GDN decode requires SM90+ or SM100+, but got SM{cc[0]}{cc[1]}")


def _verify_pooled_decode_against_reference(
    batch_size: int,
    pool_size: int,
    num_heads: int,
    head_dim: int,
    state_indices: torch.Tensor,
    dtype_torch: torch.dtype,
    seed: int = 42,
):
    """
    Core verification logic: run pooled decode kernel and compare per-sample
    against the reference implementation.

    Returns (output, state_pool, initial_state_pool) for further assertions.
    """
    device = torch.device("cuda")

    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Inputs: q, k, v [B, 1, H, D]
    q = torch.randn(
        batch_size, 1, num_heads, head_dim, dtype=dtype_torch, device=device
    )
    k = torch.randn(
        batch_size, 1, num_heads, head_dim, dtype=dtype_torch, device=device
    )
    v = torch.randn(
        batch_size, 1, num_heads, head_dim, dtype=dtype_torch, device=device
    )
    k = torch.nn.functional.normalize(k, p=2.0, dim=-1)

    # GDN params
    A_log = torch.randn(num_heads, dtype=torch.float32, device=device) * 0.1
    dt_bias = torch.randn(num_heads, dtype=torch.float32, device=device) * 0.1
    a = torch.randn(batch_size, 1, num_heads, dtype=dtype_torch, device=device) * 0.1
    b = torch.randn(batch_size, 1, num_heads, dtype=dtype_torch, device=device) * 0.1

    # State pool: [pool_size, HV, V, K] (K-last layout)
    state_pool = torch.randn(
        pool_size, num_heads, head_dim, head_dim, dtype=torch.float32, device=device
    )
    initial_state_pool = state_pool.clone()

    # Run kernel
    output, _ = gated_delta_rule_decode_pretranspose(
        q=q,
        k=k,
        v=v,
        state=state_pool,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        state_indices=state_indices,
        use_qk_l2norm=True,
    )
    torch.cuda.synchronize()

    # Verify per-sample against reference
    valid_mask = state_indices >= 0
    invalid_mask = state_indices < 0

    # Case A: Padding slots (index < 0) must produce zero output
    if invalid_mask.any():
        padded_output = output[invalid_mask]
        assert torch.all(padded_output == 0), "Padding slots must produce zero output"

    # Case B: Valid slots — compare against reference per sample
    valid_indices_local = torch.where(valid_mask)[0].cpu().numpy()

    for i in valid_indices_local:
        pool_idx = state_indices[i].item()

        q_i = q[i].float()  # [1, H, D]
        k_i = k[i].float()
        v_i = v[i].float()
        a_i = a[i].float()  # [1, H]
        b_i = b[i].float()

        # Reference expects [B, H, K, V] (K-major), kernel has [B, H, V, K] (V-major)
        init_s_i = (
            initial_state_pool[pool_idx].transpose(-2, -1).contiguous().unsqueeze(0)
        )

        ref_o, ref_s = decode_delta_rule(
            q_i,
            k_i,
            v_i,
            init_s_i,
            A_log=A_log,
            a=a_i,
            dt_bias=dt_bias,
            b=b_i,
            scale_factor=1.0 / math.sqrt(head_dim),
            use_l2_norm=True,
        )

        # Verify output
        out_i = output[i].float()
        torch.testing.assert_close(
            out_i.squeeze(0), ref_o.squeeze(0).to(out_i.device), atol=1e-2, rtol=1e-2
        )

        # Verify state update (kernel: [H, V, K], ref: [1, H, K, V])
        ref_s_transposed = ref_s.squeeze(0).transpose(-2, -1)
        current_pool_state = state_pool[pool_idx]
        torch.testing.assert_close(
            current_pool_state, ref_s_transposed.to(device), atol=1e-2, rtol=1e-2
        )

    # Case C: Untouched pool slots must remain unchanged
    used_indices = state_indices[valid_mask].unique()
    touched_mask = torch.zeros(pool_size, dtype=torch.bool, device=device)
    if len(used_indices) > 0:
        touched_mask[used_indices.long()] = True

    untouched_states_final = state_pool[~touched_mask]
    untouched_states_initial = initial_state_pool[~touched_mask]

    if len(untouched_states_final) > 0:
        torch.testing.assert_close(
            untouched_states_final,
            untouched_states_initial,
            msg="Untouched pool states should not change",
        )

    return output, state_pool, initial_state_pool


# ============================================================================
# Test 1: Basic pooled decode with negative indices
# ============================================================================


@pytest.mark.parametrize("dtype", ["bfloat16"])
@pytest.mark.parametrize("batch_size", [1, 4, 8, 32, 127])
@pytest.mark.parametrize("pool_size_multiplier", [2])
def test_decode_pooled_with_negative_indices(
    dtype, batch_size, pool_size_multiplier, seed=42
):
    """
    Test pooled decode with state_indices, including negative indices for padding.
    20% of batch elements are randomly masked as padding (index = -1).
    Valid indices are randomly scattered across the pool.
    """
    _skip_if_not_sm90_or_later()

    device = torch.device("cuda")
    dtype_torch = getattr(torch, dtype)
    num_heads = 16
    head_dim = 128
    pool_size = batch_size * pool_size_multiplier

    # Create indices with ~20% padding
    random.seed(seed)
    torch.random.manual_seed(seed)
    state_indices = torch.arange(batch_size, dtype=torch.int32, device=device)
    mask = torch.rand(batch_size, device=device) < 0.2
    state_indices[mask] = -1

    # Map valid indices to random slots in pool
    valid_mask = state_indices >= 0
    num_valid = valid_mask.sum().item()
    if num_valid > 0:
        valid_slots = torch.randperm(pool_size, device=device)[:num_valid].to(
            torch.int32
        )
        state_indices[valid_mask] = valid_slots

    _verify_pooled_decode_against_reference(
        batch_size=batch_size,
        pool_size=pool_size,
        num_heads=num_heads,
        head_dim=head_dim,
        state_indices=state_indices,
        dtype_torch=dtype_torch,
        seed=seed,
    )


# ============================================================================
# Test 2: sglang-style pooled decode pattern
# Simulates exactly how sglang's GDNAttnBackend.forward_decode calls the kernel:
#   - Full pool passed as state (pool_size+1 slots, slot 0 is sentinel)
#   - cache_indices from scheduler, with PAD_SLOT_ID = -1 for padding
#   - .to(torch.int32) cast on cache_indices
# ============================================================================


@pytest.mark.parametrize("dtype", ["bfloat16"])
@pytest.mark.parametrize("batch_size", [1, 4, 8, 16, 32])
@pytest.mark.parametrize("pool_size", [128, 256])
def test_decode_pooled_sglang_pattern(dtype, batch_size, pool_size, seed=42):
    """
    Simulate sglang's forward_decode calling pattern:
      - ssm_states is the full pool [pool_size+1, HV, V, K] (extra sentinel slot)
      - cache_indices are int64, cast to int32 at call site
      - PAD_SLOT_ID = -1 for CUDA graph padding slots
      - The kernel should NOT gather/scatter — zero-copy pool access
    """
    _skip_if_not_sm90_or_later()

    device = torch.device("cuda")
    dtype_torch = getattr(torch, dtype)
    num_heads = 16
    head_dim = 128

    PAD_SLOT_ID = -1
    total_pool_size = pool_size + 1  # sglang adds +1 sentinel slot

    # sglang scheduler produces int64 cache_indices — each request has a unique slot
    num_valid = batch_size - max(1, batch_size // 4)
    cache_indices_int64 = torch.randperm(pool_size, device=device)[:num_valid].to(
        torch.int64
    )

    # Simulate CUDA graph padding: remaining slots are PAD_SLOT_ID
    num_padded = batch_size - num_valid
    padding = torch.full((num_padded,), PAD_SLOT_ID, dtype=torch.int64, device=device)
    cache_indices_int64 = torch.cat([cache_indices_int64, padding])

    # sglang casts to int32 at call site
    state_indices = cache_indices_int64.to(torch.int32)

    _verify_pooled_decode_against_reference(
        batch_size=batch_size,
        pool_size=total_pool_size,
        num_heads=num_heads,
        head_dim=head_dim,
        state_indices=state_indices,
        dtype_torch=dtype_torch,
        seed=seed,
    )


# ============================================================================
# Test 3: Pooled vs non-pooled equivalence
# When state_indices is identity [0, 1, ..., B-1] and pool_size == B,
# pooled decode should produce identical results to non-pooled decode.
# ============================================================================


@pytest.mark.parametrize("dtype", ["bfloat16"])
@pytest.mark.parametrize("batch_size", [1, 4, 16, 32])
def test_decode_pooled_vs_nonpooled_equivalence(dtype, batch_size, seed=42):
    """
    When state_indices = [0, 1, ..., B-1] (identity mapping) and pool_size == B,
    the pooled kernel should produce identical results to the non-pooled kernel.
    This verifies zero-copy mode doesn't introduce numerical differences.
    """
    _skip_if_not_sm90_or_later()

    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    device = torch.device("cuda")
    dtype_torch = getattr(torch, dtype)
    num_heads = 16
    head_dim = 128

    # Identical inputs
    q = torch.randn(
        batch_size, 1, num_heads, head_dim, dtype=dtype_torch, device=device
    )
    k = torch.randn(
        batch_size, 1, num_heads, head_dim, dtype=dtype_torch, device=device
    )
    v = torch.randn(
        batch_size, 1, num_heads, head_dim, dtype=dtype_torch, device=device
    )
    k = torch.nn.functional.normalize(k, p=2.0, dim=-1)

    A_log = torch.randn(num_heads, dtype=torch.float32, device=device) * 0.1
    dt_bias = torch.randn(num_heads, dtype=torch.float32, device=device) * 0.1
    a = torch.randn(batch_size, 1, num_heads, dtype=dtype_torch, device=device) * 0.1
    b = torch.randn(batch_size, 1, num_heads, dtype=dtype_torch, device=device) * 0.1

    # State: [B, HV, V, K] — same for both
    state_base = torch.randn(
        batch_size, num_heads, head_dim, head_dim, dtype=torch.float32, device=device
    )

    # Run non-pooled (state_indices=None)
    state_nonpooled = state_base.clone()
    output_nonpooled, _ = gated_delta_rule_decode_pretranspose(
        q=q,
        k=k,
        v=v,
        state=state_nonpooled,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        use_qk_l2norm=True,
        state_indices=None,
    )
    torch.cuda.synchronize()

    # Run pooled with identity indices
    state_pooled = state_base.clone()
    identity_indices = torch.arange(batch_size, dtype=torch.int32, device=device)
    output_pooled, _ = gated_delta_rule_decode_pretranspose(
        q=q,
        k=k,
        v=v,
        state=state_pooled,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        use_qk_l2norm=True,
        state_indices=identity_indices,
    )
    torch.cuda.synchronize()

    # Outputs should be identical (same compiled code path, same data)
    torch.testing.assert_close(
        output_pooled,
        output_nonpooled,
        atol=1e-5,
        rtol=1e-5,
        msg="Pooled decode with identity indices should match non-pooled decode",
    )

    # State should be identical
    torch.testing.assert_close(
        state_pooled,
        state_nonpooled,
        atol=1e-5,
        rtol=1e-5,
        msg="Pooled state update with identity indices should match non-pooled",
    )


# ============================================================================
# Test 4: All-padding batch (all negative indices)
# Output should be all zeros, pool state should be completely unchanged.
# ============================================================================


@pytest.mark.parametrize("dtype", ["bfloat16"])
@pytest.mark.parametrize("batch_size", [1, 4, 16, 32])
def test_decode_pooled_all_padding(dtype, batch_size, seed=42):
    """
    When ALL state_indices are negative (entire batch is padding),
    output must be all zeros and no pool state should be modified.
    This happens in CUDA graph when batch_size < max_bs.
    """
    _skip_if_not_sm90_or_later()

    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    device = torch.device("cuda")
    dtype_torch = getattr(torch, dtype)
    num_heads = 16
    head_dim = 128
    pool_size = batch_size * 2

    q = torch.randn(
        batch_size, 1, num_heads, head_dim, dtype=dtype_torch, device=device
    )
    k = torch.randn(
        batch_size, 1, num_heads, head_dim, dtype=dtype_torch, device=device
    )
    v = torch.randn(
        batch_size, 1, num_heads, head_dim, dtype=dtype_torch, device=device
    )

    A_log = torch.randn(num_heads, dtype=torch.float32, device=device) * 0.1
    dt_bias = torch.randn(num_heads, dtype=torch.float32, device=device) * 0.1
    a = torch.randn(batch_size, 1, num_heads, dtype=dtype_torch, device=device) * 0.1
    b = torch.randn(batch_size, 1, num_heads, dtype=dtype_torch, device=device) * 0.1

    state_pool = torch.randn(
        pool_size, num_heads, head_dim, head_dim, dtype=torch.float32, device=device
    )
    initial_state_pool = state_pool.clone()

    # ALL negative indices
    state_indices = torch.full((batch_size,), -1, dtype=torch.int32, device=device)

    output, _ = gated_delta_rule_decode_pretranspose(
        q=q,
        k=k,
        v=v,
        state=state_pool,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        state_indices=state_indices,
        use_qk_l2norm=True,
    )
    torch.cuda.synchronize()

    # All output must be zero
    assert torch.all(output == 0), (
        f"All-padding batch should produce all-zero output, "
        f"but got max abs = {output.abs().max().item()}"
    )

    # Entire pool must be unchanged
    torch.testing.assert_close(
        state_pool,
        initial_state_pool,
        msg="All-padding batch should not modify any pool state",
    )


if __name__ == "__main__":
    print("Running pooled decode tests...")

    print("\n=== Test 1: Negative indices ===")
    test_decode_pooled_with_negative_indices("bfloat16", 32, 2)
    print("PASS")

    print("\n=== Test 2: sglang pattern ===")
    test_decode_pooled_sglang_pattern("bfloat16", 16, 128)
    print("PASS")

    print("\n=== Test 3: Pooled vs non-pooled equivalence ===")
    test_decode_pooled_vs_nonpooled_equivalence("bfloat16", 16)
    print("PASS")

    print("\n=== Test 4: All padding ===")
    test_decode_pooled_all_padding("bfloat16", 16)
    print("PASS")

    print("\n✅ All pooled decode tests passed!")
    print("\nTo run full test suite:")
    print("  pytest tests/gdn/test_decode_pooled.py -v")
