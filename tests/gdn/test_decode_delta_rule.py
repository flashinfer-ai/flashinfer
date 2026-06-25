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

from __future__ import annotations

import math
import os
import sys
import random

import torch
import pytest


try:
    from .reference_delta_rule import decode_delta_rule, verify_delta_rule
except ImportError:
    # For direct script execution
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from reference_delta_rule import decode_delta_rule, verify_delta_rule

# Import the actual decode functions
from flashinfer.gdn_decode import (
    gated_delta_rule_decode_pretranspose,
    gated_delta_rule_decode,
    gated_delta_rule_mtp,
)
from flashinfer.utils import get_compute_capability

# Import BF16 state kernels (T=1 and MTP)
try:
    from flashinfer.gdn_kernels.gdn_decode_bf16_state import (
        gated_delta_rule as gdn_decode_bf16_state,
        gated_delta_rule_mtp as gdn_decode_bf16_state_mtp,
    )

    GDN_DECODE_BF16_STATE_AVAILABLE = True
except ImportError:
    GDN_DECODE_BF16_STATE_AVAILABLE = False


def _skip_if_not_sm90_or_later():
    """Skip test if not Hopper (SM90+) or Blackwell (SM100+) architecture."""
    cc = get_compute_capability(torch.device("cuda"))
    if cc[0] not in [9, 10, 11, 12]:
        pytest.skip(f"GDN decode requires SM90+ or SM100+, but got SM{cc[0]}{cc[1]}")


def _assert_close_large_tensor(
    actual: torch.Tensor,
    expected: torch.Tensor,
    atol: float,
    rtol: float,
    msg: str,
    timestep_dim: int | None = None,
):
    """Manual assert_close for large tensors that avoids RuntimeError in error formatting.

    torch.testing.assert_close crashes with RuntimeError when trying to format
    error messages for tensors with >1B elements. This function computes the
    comparison manually and reports per-timestep error diagnostics on failure.
    """
    # Compare per-slice to avoid allocating huge temporary tensors
    if timestep_dim is not None and actual.ndim > timestep_dim:
        T = actual.shape[timestep_dim]
        per_t_stats = []
        any_violation = False
        for t in range(T):
            diff_t = (
                actual.select(timestep_dim, t).float()
                - expected.select(timestep_dim, t).float()
            ).abs()
            tol_t = atol + rtol * expected.select(timestep_dim, t).float().abs()
            violations_t = diff_t > tol_t
            count = violations_t.sum().item()
            total = violations_t.numel()
            per_t_stats.append(
                (t, diff_t.max().item(), diff_t.mean().item(), count, total)
            )
            if count > 0:
                any_violation = True
            del diff_t, tol_t, violations_t

        if not any_violation:
            return

        lines = [msg]
        for t, t_max, t_mean, t_count, t_total in per_t_stats:
            lines.append(
                f"  t={t}: max_abs={t_max:.6f}, mean={t_mean:.6f}, "
                f"violations={t_count}/{t_total} ({100 * t_count / t_total:.4f}%)"
            )
        lines.append(f"  Tolerances: atol={atol}, rtol={rtol}")
        raise AssertionError("\n".join(lines))
    else:
        diff = (actual.float() - expected.float()).abs()
        tol = atol + rtol * expected.float().abs()
        violations = diff > tol
        if not violations.any():
            return
        num_violations = violations.sum().item()
        total = violations.numel()
        raise AssertionError(
            f"{msg}\n"
            f"  Max abs error: {diff.max().item():.6f}, "
            f"Violations: {num_violations}/{total} ({100 * num_violations / total:.4f}%), "
            f"Tolerances: atol={atol}, rtol={rtol}"
        )


# ============================================================================
# Test decode kernel with pretranspose version ([B*HV, V, K])
# Reference: fp32 h state (default); bf16 h state used only for gdn_decode_bf16_state.
# ============================================================================


def _test_decode_kernel_pretranspose(
    dtype: str,
    batch_size: int,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_size: int,
    scale: float,
    alpha: bool,
    beta: bool,
    seed: int | None = None,
):
    """Test single decode step"""
    _skip_if_not_sm90_or_later()

    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    max(num_q_heads, num_v_heads)
    # State and GDN parameters are based on num_v_heads (HV in kernel API)
    num_sab_heads = num_v_heads

    dtype_torch = getattr(torch, dtype)
    kv_dtype = torch.float32
    device = torch.device("cuda")

    with device:
        # Single token per batch (need T=1 dimension for kernel)
        q = torch.randn(batch_size, 1, num_q_heads, head_size, dtype=dtype_torch)
        k = torch.randn(batch_size, 1, num_k_heads, head_size, dtype=dtype_torch)
        v = torch.randn(batch_size, 1, num_v_heads, head_size, dtype=dtype_torch)

        # L2 norm k to avoid numerical instability
        k = torch.nn.functional.normalize(k, p=2.0, dim=-1)

        # Initial state (k-major layout: [B, HV, K, V] for reference, matches Triton)
        input_state_ref = torch.randn(
            batch_size, num_sab_heads, head_size, head_size, dtype=kv_dtype
        )

        # Kernel expects [B, HV, V, K] (v-major), so transpose for kernel
        input_state_kernel = input_state_ref.transpose(-2, -1).contiguous()

        # Create GDN-specific parameters
        # A_log: log decay parameter [HV]
        A_log = torch.randn(num_sab_heads, dtype=torch.float32, device=device) * 0.1

        # dt_bias: decay bias [HV]
        dt_bias = torch.randn(num_sab_heads, dtype=torch.float32, device=device) * 0.1

        # a: input-dependent decay [B, 1, HV]
        # Convert alpha to a: alpha = exp(-exp(A_log) * softplus(a + dt_bias))
        # For simplicity, use random values
        a = (
            torch.randn(batch_size, 1, num_sab_heads, dtype=dtype_torch, device=device)
            * 0.1
        )

        # b: update gate input [B, 1, HV]
        # Convert beta to b: beta = sigmoid(b)
        if beta:
            # Generate b such that sigmoid(b) gives reasonable beta values
            b_tensor = torch.randn(
                batch_size, 1, num_sab_heads, dtype=dtype_torch, device=device
            )
        else:
            # Set to high value so sigmoid(b) ≈ 1
            b_tensor = (
                torch.ones(
                    batch_size, 1, num_sab_heads, dtype=dtype_torch, device=device
                )
                * 10.0
            )

    # Call kernel
    our_state = input_state_kernel.clone()
    our_o, our_state = gated_delta_rule_decode_pretranspose(
        q=q,
        k=k,
        v=v,
        state=our_state,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b_tensor,
        scale=scale,
        use_qk_l2norm=True,
    )

    torch.cuda.synchronize()

    # Remove T dimension for comparison: [B, 1, H, D] -> [B, H, D]
    our_o = our_o.squeeze(1)

    # Reference: fp32 h state (default state_dtype)
    ref_o, ref_state = decode_delta_rule(
        q.squeeze(1).float(),  # [B, 1, H, K] -> [B, H, K]
        k.squeeze(1).float(),
        v.squeeze(1).float(),
        input_state_ref,  # [B, HV, K, V]
        A_log=A_log,
        a=a.squeeze(1),  # Remove T dimension: [B, 1, HV] -> [B, HV]
        dt_bias=dt_bias,
        b=b_tensor.squeeze(1),  # Remove T dimension: [B, 1, HV] -> [B, HV]
        scale_factor=scale,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        use_l2_norm=True,  # Match kernel behavior
    )

    ref_o = ref_o.to(dtype_torch)
    # Reference returns [B, HV, K, V], kernel returns [B, HV, V, K]
    # Transpose reference state to match kernel format for comparison
    ref_state = ref_state.transpose(-2, -1).contiguous().to(kv_dtype)

    atol_o = 5e-3
    rtol_o = 5e-3
    atol_kv = 5e-3
    rtol_kv = 5e-3

    # Compare outputs
    torch.testing.assert_close(our_o, ref_o, atol=atol_o, rtol=rtol_o)
    torch.testing.assert_close(our_state, ref_state, atol=atol_kv, rtol=rtol_kv)

    print(f"✓ Decode kernel test passed (batch={batch_size}, dtype={dtype})")


@pytest.mark.parametrize("beta", [True])
@pytest.mark.parametrize("alpha", [True])
@pytest.mark.parametrize("scale", [1.0])
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize(
    "num_q_heads, num_k_heads, num_v_heads",
    [(16, 16, 32)],
)
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
@pytest.mark.parametrize("dtype", ["bfloat16"])
def test_decode_kernel_basic_pretranspose(
    dtype: str,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_size: int,
    batch_size: int,
    scale: float | str,
    alpha: bool,
    beta: bool,
    seed: int = int(os.environ.get("SEED", "0")),
):
    scale_val = 1.0 / math.sqrt(head_size) if scale == "auto" else scale
    _test_decode_kernel_pretranspose(
        dtype,
        batch_size,
        num_q_heads,
        num_k_heads,
        num_v_heads,
        head_size,
        scale_val,
        alpha,
        beta,
        seed,
    )


# ============================================================================
# Test decode kernel with nontranspose version ([pool, HV, K, V])
# Reference: fp32 h state (default).
# ============================================================================


def _test_decode_kernel_nontranspose(
    dtype: str,
    batch_size: int,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_size: int,
    scale: float,
    alpha: bool,
    beta: bool,
    seed: int | None = None,
):
    """Test single decode step with nontranspose version (K-major layout)"""
    _skip_if_not_sm90_or_later()

    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    max(num_q_heads, num_v_heads)
    # State and GDN parameters are based on num_v_heads (HV in kernel API)
    num_sab_heads = num_v_heads

    dtype_torch = getattr(torch, dtype)
    kv_dtype = torch.float32
    device = torch.device("cuda")

    with device:
        # Single token per batch (need T=1 dimension for kernel)
        q = torch.randn(batch_size, 1, num_q_heads, head_size, dtype=dtype_torch)
        k = torch.randn(batch_size, 1, num_k_heads, head_size, dtype=dtype_torch)
        v = torch.randn(batch_size, 1, num_v_heads, head_size, dtype=dtype_torch)

        # L2 norm k to avoid numerical instability
        k = torch.nn.functional.normalize(k, p=2.0, dim=-1)

        # Initial state (k-major layout: [B, HV, K, V] for both reference and kernel)
        input_state = torch.randn(
            batch_size, num_sab_heads, head_size, head_size, dtype=kv_dtype
        )

        # Create GDN-specific parameters
        # A_log: log decay parameter [HV]
        A_log = torch.randn(num_sab_heads, dtype=torch.float32, device=device) * 0.1

        # dt_bias: decay bias [HV]
        dt_bias = torch.randn(num_sab_heads, dtype=torch.float32, device=device) * 0.1

        # a: input-dependent decay [B, 1, HV]
        a = (
            torch.randn(batch_size, 1, num_sab_heads, dtype=dtype_torch, device=device)
            * 0.1
        )

        # b: update gate input [B, 1, HV]
        if beta:
            # Generate b such that sigmoid(b) gives reasonable beta values
            b_tensor = torch.randn(
                batch_size, 1, num_sab_heads, dtype=dtype_torch, device=device
            )
        else:
            # Set to high value so sigmoid(b) ≈ 1
            b_tensor = (
                torch.ones(
                    batch_size, 1, num_sab_heads, dtype=dtype_torch, device=device
                )
                * 10.0
            )

    # Call kernel (nontranspose version uses K-major layout directly, no transpose needed)
    our_state = input_state.clone()
    our_o, our_state = gated_delta_rule_decode(
        q=q,
        k=k,
        v=v,
        state=our_state,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b_tensor,
        scale=scale,
        use_qk_l2norm=True,
    )

    torch.cuda.synchronize()

    # Remove T dimension for comparison: [B, 1, H, D] -> [B, H, D]
    our_o = our_o.squeeze(1)

    # Reference: fp32 h state (default state_dtype)
    ref_o, ref_state = decode_delta_rule(
        q.squeeze(1).float(),  # [B, 1, H, K] -> [B, H, K]
        k.squeeze(1).float(),
        v.squeeze(1).float(),
        input_state,  # [B, HV, K, V]
        A_log=A_log,
        a=a.squeeze(1),  # Remove T dimension: [B, 1, HV] -> [B, HV]
        dt_bias=dt_bias,
        b=b_tensor.squeeze(1),  # Remove T dimension: [B, 1, HV] -> [B, HV]
        scale_factor=scale,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        use_l2_norm=True,  # Match kernel behavior
    )

    ref_o = ref_o.to(dtype_torch)
    ref_state = ref_state.to(kv_dtype)

    atol_o = 5e-3
    rtol_o = 5e-3
    atol_kv = 5e-3
    rtol_kv = 5e-3

    # Compare outputs (no transpose needed, both use K-major layout)
    torch.testing.assert_close(our_o, ref_o, atol=atol_o, rtol=rtol_o)
    torch.testing.assert_close(our_state, ref_state, atol=atol_kv, rtol=rtol_kv)

    print(
        f"✓ Decode kernel (nontranspose) test passed (batch={batch_size}, dtype={dtype})"
    )


@pytest.mark.parametrize("beta", [True])
@pytest.mark.parametrize("alpha", [True])
@pytest.mark.parametrize("scale", [1.0])
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize(
    "num_q_heads, num_k_heads, num_v_heads",
    [(16, 16, 32)],
)
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
@pytest.mark.parametrize("dtype", ["bfloat16"])
def test_decode_kernel_basic_nontranspose(
    dtype: str,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_size: int,
    batch_size: int,
    scale: float | str,
    alpha: bool,
    beta: bool,
    seed: int = int(os.environ.get("SEED", "0")),
):
    scale_val = 1.0 / math.sqrt(head_size) if scale == "auto" else scale
    _test_decode_kernel_nontranspose(
        dtype,
        batch_size,
        num_q_heads,
        num_k_heads,
        num_v_heads,
        head_size,
        scale_val,
        alpha,
        beta,
        seed,
    )


# ============================================================================
# Test pretranspose kernel with pool + indices path
# Verifies that passing initial_state=[pool,HV,V,K] + initial_state_indices=[B]
# produces identical output and in-place state updates as the gather-run-scatter
# direct-state path, and that unselected pool slots are untouched.
# ============================================================================


def _test_decode_kernel_pretranspose_pool(
    dtype: str,
    batch_size: int,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_size: int,
    scale: float,
    pool_multiplier: int = 3,
    state_dtype: str = "bfloat16",
    seed: int | None = None,
):
    """Pool+indices path must match gather → direct-state → scatter reference."""
    _skip_if_not_sm90_or_later()

    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    num_sab_heads = num_v_heads
    pool_size = batch_size * pool_multiplier
    dtype_torch = getattr(torch, dtype)
    kv_dtype = getattr(torch, state_dtype)
    device = torch.device("cuda")

    with device:
        q = torch.randn(batch_size, 1, num_q_heads, head_size, dtype=dtype_torch)
        k = torch.nn.functional.normalize(
            torch.randn(batch_size, 1, num_k_heads, head_size, dtype=dtype_torch),
            p=2.0,
            dim=-1,
        )
        v = torch.randn(batch_size, 1, num_v_heads, head_size, dtype=dtype_torch)

        A_log = torch.randn(num_sab_heads, dtype=torch.float32) * 0.1
        dt_bias = torch.randn(num_sab_heads, dtype=torch.float32) * 0.1
        a = torch.randn(batch_size, 1, num_sab_heads, dtype=dtype_torch) * 0.1
        b = torch.randn(batch_size, 1, num_sab_heads, dtype=dtype_torch)

        # Pool in [pool, HV, V, K] K-last layout (same as the direct-state layout)
        pool = torch.randn(
            pool_size, num_sab_heads, head_size, head_size, dtype=kv_dtype
        )

        # Non-trivial indices: every pool_multiplier-th slot (non-contiguous)
        indices = (
            torch.arange(batch_size, dtype=torch.int32, device=device) * pool_multiplier
        )

    # ── Pool path (what we're testing) ──────────────────────────────────────
    pool_under_test = pool.clone()
    out_pool, _ = gated_delta_rule_decode_pretranspose(
        q=q,
        k=k,
        v=v,
        state=None,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        scale=scale,
        use_qk_l2norm=True,
        initial_state=pool_under_test,
        initial_state_indices=indices,
    )

    # ── Direct-state reference (gather → kernel) ─────────────────────────────
    gathered_state = pool[indices].clone()  # [B, HV, V, K]
    out_direct, updated_state = gated_delta_rule_decode_pretranspose(
        q=q,
        k=k,
        v=v,
        state=gathered_state,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        scale=scale,
        use_qk_l2norm=True,
    )

    atol = 5e-3
    rtol = 5e-3

    # Outputs must match
    torch.testing.assert_close(out_pool, out_direct, atol=atol, rtol=rtol)

    # Selected pool slots must match the state updated by the direct path
    torch.testing.assert_close(
        pool_under_test[indices], updated_state, atol=atol, rtol=rtol
    )

    # Non-selected pool slots must be exactly unchanged
    mask = torch.ones(pool_size, dtype=torch.bool, device=device)
    mask[indices] = False
    torch.testing.assert_close(pool_under_test[mask], pool[mask], atol=0.0, rtol=0.0)

    print(
        f"✓ Pool+indices pretranspose test passed "
        f"(batch={batch_size}, pool={pool_size}, dtype={dtype})"
    )


@pytest.mark.parametrize("state_dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("scale", [1.0])
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize("num_q_heads, num_k_heads, num_v_heads", [(16, 16, 32)])
@pytest.mark.parametrize("batch_size", [1, 4, 16, 32])
@pytest.mark.parametrize("dtype", ["bfloat16"])
def test_decode_kernel_pretranspose_pool(
    dtype: str,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_size: int,
    batch_size: int,
    scale: float,
    state_dtype: str,
    seed: int = int(os.environ.get("SEED", "0")),
):
    _test_decode_kernel_pretranspose_pool(
        dtype,
        batch_size,
        num_q_heads,
        num_k_heads,
        num_v_heads,
        head_size,
        scale,
        state_dtype=state_dtype,
        seed=seed,
    )


# ============================================================================
# Test pretranspose kernel pool + indices with negative indices (padding)
#
# Negative pool indices signal padding slots: the kernel must write zeros to
# output for those batch elements and leave the pool untouched.  The gather →
# direct-state reference cannot handle negative indices, so we compare valid
# slots against the Python reference and verify padding semantics directly.
#
# Only float32 state is tested because the bf16 fast-path kernel does not
# support negative indices.
# ============================================================================


def _test_decode_kernel_pretranspose_pool_negative_indices(
    dtype: str,
    batch_size: int,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_size: int,
    scale: float,
    pool_multiplier: int = 2,
    padding_fraction: float = 0.2,
    seed: int | None = None,
):
    """Pool+indices with negative indices must zero output for padding slots
    and match the Python reference for valid slots."""
    _skip_if_not_sm90_or_later()

    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    num_sab_heads = num_v_heads
    pool_size = batch_size * pool_multiplier
    dtype_torch = getattr(torch, dtype)
    device = torch.device("cuda")

    with device:
        q = torch.randn(batch_size, 1, num_q_heads, head_size, dtype=dtype_torch)
        k = torch.nn.functional.normalize(
            torch.randn(batch_size, 1, num_k_heads, head_size, dtype=dtype_torch),
            p=2.0,
            dim=-1,
        )
        v = torch.randn(batch_size, 1, num_v_heads, head_size, dtype=dtype_torch)

        A_log = torch.randn(num_sab_heads, dtype=torch.float32) * 0.1
        dt_bias = torch.randn(num_sab_heads, dtype=torch.float32) * 0.1
        a = torch.randn(batch_size, 1, num_sab_heads, dtype=dtype_torch) * 0.1
        b = torch.randn(batch_size, 1, num_sab_heads, dtype=dtype_torch)

        # Float32 state pool (only f32 CuTe DSL kernel supports negative indices)
        pool = torch.randn(
            pool_size, num_sab_heads, head_size, head_size, dtype=torch.float32
        )

        # Build indices with ~padding_fraction padding slots
        indices = torch.arange(batch_size, dtype=torch.int32, device=device)
        mask = torch.rand(batch_size, device=device) < padding_fraction
        # Ensure at least one valid and one padding slot when batch_size >= 2
        if batch_size >= 2:
            mask[0] = False  # first slot valid
            mask[-1] = True  # last slot padding
        indices[mask] = -1

        # Map valid indices to random non-contiguous pool slots
        valid_mask = indices >= 0
        num_valid = valid_mask.sum().item()
        if num_valid > 0:
            valid_slots = torch.randperm(pool_size, device=device)[:num_valid].to(
                torch.int32
            )
            indices[valid_mask] = valid_slots

    pool_under_test = pool.clone()
    out_pool, _ = gated_delta_rule_decode_pretranspose(
        q=q,
        k=k,
        v=v,
        state=None,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        scale=scale,
        use_qk_l2norm=True,
        initial_state=pool_under_test,
        initial_state_indices=indices,
    )
    torch.cuda.synchronize()

    # ── Padding slots must produce zero output ────────────────────────────────
    invalid_mask = indices < 0
    if invalid_mask.any():
        padded_output = out_pool[invalid_mask]
        assert torch.all(padded_output == 0), (
            f"Padding slots must produce zero output, "
            f"but got max abs = {padded_output.abs().max().item()}"
        )

    # ── Valid slots: compare per-sample against Python reference ──────────────
    valid_indices_local = torch.where(valid_mask)[0].cpu().numpy()
    pool_snapshot = pool.clone()  # original pool before kernel

    for i in valid_indices_local:
        pool_idx = indices[i].item()
        ref_o, ref_s = decode_delta_rule(
            q[i].squeeze(0).unsqueeze(0).float(),  # [1, H, K]
            k[i].squeeze(0).unsqueeze(0).float(),
            v[i].squeeze(0).unsqueeze(0).float(),
            pool_snapshot[pool_idx]
            .float()
            .transpose(-2, -1)
            .contiguous()
            .unsqueeze(0),  # [1, HV, K, V]
            A_log=A_log,
            a=a[i].squeeze(0).unsqueeze(0),  # [1, HV]
            dt_bias=dt_bias.float(),
            b=b[i].squeeze(0).unsqueeze(0),
            scale_factor=scale,
            use_l2_norm=True,
        )
        # Output
        torch.testing.assert_close(
            out_pool[i].float().squeeze(0),
            ref_o.squeeze(0).to(device),
            atol=1e-2,
            rtol=1e-2,
        )
        # State update (kernel: [HV, V, K], ref: [1, HV, K, V])
        torch.testing.assert_close(
            pool_under_test[pool_idx].float(),
            ref_s.squeeze(0).transpose(-2, -1).to(device),
            atol=1e-2,
            rtol=1e-2,
        )

    # ── Untouched pool slots must remain unchanged ────────────────────────────
    used_pool_indices = indices[valid_mask].unique()
    touched = torch.zeros(pool_size, dtype=torch.bool, device=device)
    if len(used_pool_indices) > 0:
        touched[used_pool_indices.long()] = True
    torch.testing.assert_close(
        pool_under_test[~touched], pool[~touched], atol=0.0, rtol=0.0
    )

    print(
        f"✓ Pool+indices negative-index test passed "
        f"(batch={batch_size}, pool={pool_size}, dtype={dtype})"
    )


def _test_decode_kernel_pretranspose_pool_all_padding(
    dtype: str,
    batch_size: int,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_size: int,
    scale: float,
    pool_multiplier: int = 2,
    seed: int | None = None,
):
    """When ALL indices are negative (entire batch is padding), output must be
    all zeros and the pool must remain completely unchanged."""
    _skip_if_not_sm90_or_later()

    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    num_sab_heads = num_v_heads
    pool_size = batch_size * pool_multiplier
    dtype_torch = getattr(torch, dtype)
    device = torch.device("cuda")

    with device:
        q = torch.randn(batch_size, 1, num_q_heads, head_size, dtype=dtype_torch)
        k = torch.randn(batch_size, 1, num_k_heads, head_size, dtype=dtype_torch)
        v = torch.randn(batch_size, 1, num_v_heads, head_size, dtype=dtype_torch)

        A_log = torch.randn(num_sab_heads, dtype=torch.float32) * 0.1
        dt_bias = torch.randn(num_sab_heads, dtype=torch.float32) * 0.1
        a = torch.randn(batch_size, 1, num_sab_heads, dtype=dtype_torch) * 0.1
        b = torch.randn(batch_size, 1, num_sab_heads, dtype=dtype_torch)

        pool = torch.randn(
            pool_size, num_sab_heads, head_size, head_size, dtype=torch.float32
        )
        indices = torch.full((batch_size,), -1, dtype=torch.int32, device=device)

    pool_under_test = pool.clone()
    out, _ = gated_delta_rule_decode_pretranspose(
        q=q,
        k=k,
        v=v,
        state=None,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        scale=scale,
        use_qk_l2norm=True,
        initial_state=pool_under_test,
        initial_state_indices=indices,
    )
    torch.cuda.synchronize()

    assert torch.all(out == 0), (
        f"All-padding batch must produce zero output, "
        f"but got max abs = {out.abs().max().item()}"
    )
    torch.testing.assert_close(
        pool_under_test,
        pool,
        atol=0.0,
        rtol=0.0,
        msg="All-padding batch must not modify any pool state",
    )

    print(
        f"✓ Pool+indices all-padding test passed "
        f"(batch={batch_size}, pool={pool_size}, dtype={dtype})"
    )


@pytest.mark.parametrize("scale", [1.0])
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize("num_q_heads, num_k_heads, num_v_heads", [(16, 16, 32)])
@pytest.mark.parametrize("batch_size", [1, 4, 8, 32, 127])
@pytest.mark.parametrize("dtype", ["bfloat16"])
def test_decode_kernel_pretranspose_pool_negative_indices(
    dtype: str,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_size: int,
    batch_size: int,
    scale: float,
    seed: int = int(os.environ.get("SEED", "0")),
):
    _test_decode_kernel_pretranspose_pool_negative_indices(
        dtype,
        batch_size,
        num_q_heads,
        num_k_heads,
        num_v_heads,
        head_size,
        scale,
        seed=seed,
    )


@pytest.mark.parametrize("scale", [1.0])
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize("num_q_heads, num_k_heads, num_v_heads", [(16, 16, 32)])
@pytest.mark.parametrize("batch_size", [1, 4, 16, 32])
@pytest.mark.parametrize("dtype", ["bfloat16"])
def test_decode_kernel_pretranspose_pool_all_padding(
    dtype: str,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_size: int,
    batch_size: int,
    scale: float,
    seed: int = int(os.environ.get("SEED", "0")),
):
    _test_decode_kernel_pretranspose_pool_all_padding(
        dtype,
        batch_size,
        num_q_heads,
        num_k_heads,
        num_v_heads,
        head_size,
        scale,
        seed=seed,
    )


# ============================================================================
# Test bf16 decode kernel with negative (padding) indices
#
# Verifies that the bf16 fast-path kernel handles negative indices correctly
# via the slot-0 null buffer pattern: negative indices are redirected to slot 0
# inside the kernel. Valid slots must produce correct output and updated state;
# the kernel must not crash.
# ============================================================================


def _test_decode_kernel_bf16_padding_indices(
    batch_size: int,
    num_q_heads: int,
    num_v_heads: int,
    head_size: int,
    scale: float,
    padding_fraction: float = 0.5,
    seed: int = 0,
):
    """bf16 kernel with mixed negative/valid indices must not crash and must
    produce correct output and state updates for valid slots."""
    _skip_if_not_sm90_or_later()

    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    pool_size = batch_size * 2 + 1  # slot 0 = null buffer; real slots start at 1
    device = torch.device("cuda")

    with device:
        q = torch.randn(batch_size, 1, num_q_heads, head_size, dtype=torch.bfloat16)
        k = torch.nn.functional.normalize(
            torch.randn(batch_size, 1, num_q_heads, head_size, dtype=torch.bfloat16),
            p=2.0,
            dim=-1,
        )
        v = torch.randn(batch_size, 1, num_v_heads, head_size, dtype=torch.bfloat16)

        A_log = torch.randn(num_v_heads, dtype=torch.float32) * 0.1
        dt_bias = torch.randn(num_v_heads, dtype=torch.float32) * 0.1
        a = torch.randn(batch_size, 1, num_v_heads, dtype=torch.bfloat16) * 0.1
        b = torch.randn(batch_size, 1, num_v_heads, dtype=torch.bfloat16)

        # Slot 0 = null buffer (zeros); real slots start from 1
        pool = torch.zeros(
            pool_size, num_v_heads, head_size, head_size, dtype=torch.bfloat16
        )
        pool[1:] = torch.randn(
            pool_size - 1, num_v_heads, head_size, head_size, dtype=torch.bfloat16
        )

        # Build indices: some slots are padding (-1), others map to real slots [1, pool_size)
        indices = torch.arange(1, batch_size + 1, dtype=torch.int32, device=device)
        mask = torch.rand(batch_size, device=device) < padding_fraction
        if batch_size >= 2:
            mask[0] = False  # ensure at least one valid
            mask[-1] = True  # ensure at least one padding
        indices[mask] = -1

    valid_mask = indices >= 0

    # ── Pool path under test ─────────────────────────────────────────────────
    pool_under_test = pool.clone()
    out_pool, _ = gated_delta_rule_decode_pretranspose(
        q=q,
        k=k,
        v=v,
        state=None,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        scale=scale,
        use_qk_l2norm=True,
        initial_state=pool_under_test,
        initial_state_indices=indices,
    )
    torch.cuda.synchronize()

    # ── Direct-state reference for valid slots only ──────────────────────────
    if valid_mask.any():
        valid_indices = indices[valid_mask]
        gathered = pool[valid_indices].clone()
        out_direct, updated = gated_delta_rule_decode_pretranspose(
            q=q[valid_mask],
            k=k[valid_mask],
            v=v[valid_mask],
            state=gathered,
            A_log=A_log,
            a=a[valid_mask],
            dt_bias=dt_bias,
            b=b[valid_mask],
            scale=scale,
            use_qk_l2norm=True,
        )
        atol, rtol = 5e-3, 5e-3
        torch.testing.assert_close(
            out_pool[valid_mask], out_direct, atol=atol, rtol=rtol
        )
        torch.testing.assert_close(
            pool_under_test[valid_indices], updated, atol=atol, rtol=rtol
        )

    # Non-selected real slots (slots 1..pool_size-1 not in valid_indices) must be untouched
    used = indices[valid_mask].to(device)
    unused_mask = torch.ones(pool_size, dtype=torch.bool, device=device)
    unused_mask[used] = False
    unused_mask[0] = False  # slot 0 may be modified (null buffer), don't check it
    torch.testing.assert_close(
        pool_under_test[unused_mask], pool[unused_mask], atol=0.0, rtol=0.0
    )

    # Slot 0 (null buffer) must have been written by padding slots.
    # Without the kernel-level fix, padding slots do an OOB write to gH[-1]
    # (before the pool base) leaving slot 0 untouched — this assertion catches that.
    if mask.any():
        assert not torch.equal(pool_under_test[0], pool[0]), (
            "Slot 0 (null buffer) should have been updated by padding slots; "
            "if it is unchanged the kernel fix is missing"
        )

    print(
        f"✓ bf16 padding indices test passed "
        f"(batch={batch_size}, valid={valid_mask.sum().item()}, padding={mask.sum().item()})"
    )


@pytest.mark.parametrize("scale", [1.0])
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize("num_q_heads, num_v_heads", [(16, 32)])
@pytest.mark.parametrize("batch_size", [1, 4, 16, 32])
def test_decode_kernel_bf16_padding_indices(
    batch_size: int,
    num_q_heads: int,
    num_v_heads: int,
    head_size: int,
    scale: float,
    seed: int = int(os.environ.get("SEED", "0")),
):
    _test_decode_kernel_bf16_padding_indices(
        batch_size, num_q_heads, num_v_heads, head_size, scale, seed=seed
    )


# ============================================================================
# Test verify kernel with MTP version (Multiple Token Processing)
# Reference: fp32 h state (default).
# ============================================================================


def _test_verify_kernel_mtp(
    dtype: str,
    batch_size: int,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_size: int,
    seq_len: int,  # T > 1 for MTP
    scale: float,
    alpha: bool,
    beta: bool,
    cache_intermediate_states: bool = True,
    disable_state_update: bool = True,
    seed: int = 0,
):
    """Test gated_delta_rule_mtp API (MTP version) against reference."""
    _skip_if_not_sm90_or_later()

    import math

    torch.manual_seed(seed)

    # Map dtype string to torch dtype
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map[dtype]

    B = batch_size
    T = seq_len
    H = num_q_heads
    HV = num_v_heads
    K = head_size
    V = head_size

    # Generate test inputs
    q = torch.randn(B, T, H, K, dtype=torch_dtype, device="cuda") * 0.1
    k = torch.randn(B, T, num_k_heads, K, dtype=torch_dtype, device="cuda") * 0.1
    v = torch.randn(B, T, HV, V, dtype=torch_dtype, device="cuda") * 0.1

    # GDN parameters
    A_log = (
        torch.randn(HV, dtype=torch.float32, device="cuda") * 0.1
        if alpha
        else torch.zeros(HV, dtype=torch.float32, device="cuda")
    )
    dt_bias = (
        torch.randn(HV, dtype=torch.float32, device="cuda") * 0.1
        if alpha
        else torch.zeros(HV, dtype=torch.float32, device="cuda")
    )
    a = (
        torch.randn(B, T, HV, dtype=torch_dtype, device="cuda") * 0.1
        if alpha
        else torch.zeros(B, T, HV, dtype=torch_dtype, device="cuda")
    )
    b_tensor = (
        torch.randn(B, T, HV, dtype=torch_dtype, device="cuda") * 0.1
        if beta
        else torch.zeros(B, T, HV, dtype=torch_dtype, device="cuda")
    )

    # Initial state: [pool_size, HV, V, K] for MTP version (K-last layout)
    pool_size = B
    initial_state = (
        torch.randn(pool_size, HV, V, K, dtype=torch.float32, device="cuda") * 0.01
    )
    initial_state_indices = torch.arange(B, dtype=torch.int32, device="cuda")

    # Intermediate states buffer (optional)
    if cache_intermediate_states:
        intermediate_states_buffer = torch.zeros(
            pool_size, T, HV, V, K, dtype=torch.float32, device="cuda"
        )
    else:
        intermediate_states_buffer = None

    # Scale factor
    scale_val = 1.0 / math.sqrt(K) if scale == "auto" else scale

    # Make copies for kernel and reference
    initial_state_kernel = initial_state.clone()
    initial_state_ref = initial_state.clone()

    # =========================================================================
    # Run kernel
    # =========================================================================
    output_kernel, final_state_kernel = gated_delta_rule_mtp(
        q=q,
        k=k,
        v=v,
        initial_state=initial_state_kernel,
        initial_state_indices=initial_state_indices,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b_tensor,
        scale=scale_val,
        output=None,
        intermediate_states_buffer=intermediate_states_buffer,
        disable_state_update=disable_state_update,
        use_qk_l2norm=True,
    )

    # =========================================================================
    # Run reference
    # =========================================================================
    # Reference expects [B, num_heads, K, V] state (K-major)
    # Convert from [pool_size, HV, V, K] to [B, HV, K, V]
    input_state_ref = initial_state_ref.transpose(
        -2, -1
    ).contiguous()  # [pool_size, HV, K, V]

    output_ref, final_state_ref, intermediate_states_ref = verify_delta_rule(
        q=q,
        k=k,
        v=v,
        state=input_state_ref,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b_tensor,
        scale_factor=scale_val,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        use_l2_norm=True,
        cache_intermediate_states=cache_intermediate_states,
    )

    # =========================================================================
    # Compare outputs
    # =========================================================================
    # Adjust tolerances for bfloat16
    if dtype == "bfloat16":
        atol_o = 1e-2
        rtol_o = 1e-2
        atol_s = 1e-2
        rtol_s = 1e-2
    else:
        atol_o = 1e-3
        rtol_o = 1e-3
        atol_s = 1e-3
        rtol_s = 1e-3

    # Compare outputs
    torch.testing.assert_close(
        output_kernel.float(),
        output_ref.float(),
        atol=atol_o,
        rtol=rtol_o,
        msg=f"Output mismatch for MTP kernel (B={B}, T={T}, dtype={dtype})",
    )

    # Compare intermediate states if cached
    if cache_intermediate_states and intermediate_states_buffer is not None:
        # Convert kernel's intermediate states from [pool_size, T, HV, V, K] to [B, T, HV, K, V]
        intermediate_states_kernel = intermediate_states_buffer.transpose(
            -2, -1
        )  # [pool_size, T, HV, K, V]

        # Use manual comparison to avoid RuntimeError from torch.testing.assert_close
        # when formatting error messages for tensors with >1B elements (e.g. [512, 5, 32, 128, 128])
        _assert_close_large_tensor(
            intermediate_states_kernel.float(),
            intermediate_states_ref.float(),
            atol=atol_s,
            rtol=rtol_s,
            msg=f"Intermediate states mismatch for MTP kernel (B={B}, T={T}, dtype={dtype})",
            timestep_dim=1,
        )

    # Compare final state if state update is enabled
    if not disable_state_update:
        # Kernel returns [pool_size, HV, V, K], reference returns [B, HV, K, V]
        final_state_ref_transposed = final_state_ref.transpose(-2, -1).contiguous()
        torch.testing.assert_close(
            final_state_kernel.float(),
            final_state_ref_transposed.float(),
            atol=atol_s,
            rtol=rtol_s,
            msg=f"Final state mismatch for MTP kernel (B={B}, T={T}, dtype={dtype})",
        )

    print(f"✓ MTP kernel test passed (batch={B}, seq_len={T}, dtype={dtype})")


@pytest.mark.parametrize("cache_intermediate_states", [True, False])
@pytest.mark.parametrize("beta", [True])
@pytest.mark.parametrize("alpha", [True])
@pytest.mark.parametrize("scale", [1.0])
@pytest.mark.parametrize("seq_len", [2, 4, 8])
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize(
    "num_q_heads, num_k_heads, num_v_heads",
    [(16, 16, 32)],
)
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("dtype", ["bfloat16"])
def test_verify_kernel_mtp(
    dtype: str,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_size: int,
    batch_size: int,
    seq_len: int,
    scale: float | str,
    alpha: bool,
    beta: bool,
    cache_intermediate_states: bool,
    seed: int = int(os.environ.get("SEED", "0")),
):
    scale_val = 1.0 / math.sqrt(head_size) if scale == "auto" else scale
    _test_verify_kernel_mtp(
        dtype,
        batch_size,
        num_q_heads,
        num_k_heads,
        num_v_heads,
        head_size,
        seq_len,
        scale_val,
        alpha,
        beta,
        cache_intermediate_states,
        seed,
    )


# ============================================================================
# Test MTP kernel with FP32 state, cache ON, state update ON (comprehensive)
# This tests the full production configuration: all BS and T values
# ============================================================================


@pytest.mark.parametrize("seq_len", [2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
@pytest.mark.parametrize("dtype", ["bfloat16"])
def test_mtp_fp32_state_with_cache_and_state_update(
    dtype: str,
    batch_size: int,
    seq_len: int,
    seed: int = int(os.environ.get("SEED", "0")),
):
    """
    Comprehensive MTP test with FP32 state, intermediate caching ON, state update ON.

    This tests the production configuration:
    - FP32 h state (not bf16)
    - cache_intermediate_states=True
    - disable_state_update=False (h is updated)
    - All batch sizes: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512
    - All sequence lengths: 2, 3, 4, 5, 6, 7, 8
    """
    scale_val = 1.0 / math.sqrt(128)  # head_size=128
    _test_verify_kernel_mtp(
        dtype=dtype,
        batch_size=batch_size,
        num_q_heads=16,
        num_k_heads=16,
        num_v_heads=32,
        head_size=128,
        seq_len=seq_len,
        scale=scale_val,
        alpha=True,
        beta=True,
        cache_intermediate_states=True,
        disable_state_update=False,  # State update ON
        seed=seed,
    )


# ============================================================================
# Test BF16 state kernel (T=1..4, bf16 state, K-last)
# Reference: bf16 h state only here (state_dtype=torch.bfloat16). Other kernels
# above use fp32 h state reference.
# ============================================================================


def _test_gdn_decode_bf16_state_kernel(
    dtype: str,
    batch_size: int,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_size: int,
    seq_len: int,
    scale: float,
    alpha: bool,
    beta: bool,
    seed: int | None = None,
):
    """Test BF16 state kernel with bf16 h state.

    Both kernel and reference use bf16 h state: reference runs with
    state_dtype=torch.bfloat16 (read h as fp32, compute in fp32, store h in bf16)
    so the comparison is apples-to-apples with the BF16 state kernel.
    """
    _skip_if_not_sm90_or_later()

    if not GDN_DECODE_BF16_STATE_AVAILABLE:
        pytest.skip("BF16 state kernel not available")

    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    assert seq_len >= 1, f"seq_len must be >= 1, got T={seq_len}"

    # State and GDN parameters are based on num_v_heads (HV in kernel API)
    num_sab_heads = num_v_heads

    dtype_torch = getattr(torch, dtype)
    device = torch.device("cuda")

    with device:
        # Generate inputs with T dimension
        q = torch.randn(batch_size, seq_len, num_q_heads, head_size, dtype=dtype_torch)
        k = torch.randn(batch_size, seq_len, num_k_heads, head_size, dtype=dtype_torch)
        v = torch.randn(batch_size, seq_len, num_v_heads, head_size, dtype=dtype_torch)

        # NOTE: Do NOT pre-normalize K here. Both the kernel (use_qk_l2norm_in_kernel=True)
        # and reference will apply L2 normalization internally after GQA expansion.

        # BF16 state kernel expects [B, HV, V, K] (K-fast layout) in BF16.
        # Use the same bf16 initial state for both kernel and reference so we
        # compare the bf16 h state path.
        input_state_kernel = torch.randn(
            batch_size, num_sab_heads, head_size, head_size, dtype=torch.bfloat16
        )

        # Reference uses [B, HV, K, V] layout; same bf16 values as kernel.
        input_state_ref_bf16 = input_state_kernel.transpose(-2, -1).contiguous()

        # Create GDN-specific parameters
        # A_log: log decay parameter [HV] - must be float32
        A_log = torch.randn(num_sab_heads, dtype=torch.float32, device=device) * 0.1

        # dt_bias: decay bias [HV] - must be float32 for BF16 state kernel
        dt_bias = torch.randn(num_sab_heads, dtype=torch.float32, device=device) * 0.1

        # a: input-dependent decay [B, T, HV]
        a = (
            torch.randn(
                batch_size, seq_len, num_sab_heads, dtype=dtype_torch, device=device
            )
            * 0.1
        )

        # b: update gate input [B, T, HV]
        if beta:
            b_tensor = torch.randn(
                batch_size, seq_len, num_sab_heads, dtype=dtype_torch, device=device
            )
        else:
            b_tensor = (
                torch.ones(
                    batch_size, seq_len, num_sab_heads, dtype=dtype_torch, device=device
                )
                * 10.0
            )

    # Call BF16 state kernel (T=1 uses gated_delta_rule, T>1 uses MTP).
    # The BF16 kernels are pool-only: treat the [B, HV, V, K] state tensor
    # as a pool of size B with sequential indices arange(B). Mathematically
    # identical to non-pool semantics.
    our_state = input_state_kernel.clone()
    indices = torch.arange(batch_size, dtype=torch.int32, device=device)
    kernel_fn = gdn_decode_bf16_state if seq_len == 1 else gdn_decode_bf16_state_mtp
    our_o = kernel_fn(
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        q=q,
        k=k,
        v=v,
        b=b_tensor,
        initial_state_source=our_state,
        initial_state_indices=indices,
        use_qk_l2norm_in_kernel=True,
        scale=scale,
    )

    torch.cuda.synchronize()

    # Reference implementation with bf16 h state (state_dtype=torch.bfloat16):
    # h is stored in bf16, read as fp32 for computation, written back in bf16.
    ref_state = input_state_ref_bf16.clone()
    ref_outputs = []

    for t in range(seq_len):
        ref_o_t, ref_state = decode_delta_rule(
            q[:, t].float(),  # [B, H, K]
            k[:, t].float(),
            v[:, t].float(),
            ref_state,  # [B, HV, K, V] bf16
            A_log=A_log,
            a=a[:, t],  # [B, HV]
            dt_bias=dt_bias,
            b=b_tensor[:, t],  # [B, HV]
            scale_factor=scale,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            use_l2_norm=True,
            state_dtype=torch.bfloat16,  # match kernel: h stored in bf16
        )
        ref_outputs.append(ref_o_t)

    # Stack reference outputs: [B, T, HV, V]
    ref_o = torch.stack(ref_outputs, dim=1).to(dtype_torch)

    # Tolerances for bf16 h state comparison
    atol_o = 0.001
    rtol_o = 0.005
    atol_kv = 0.016  # Accommodates 1 ULP for BF16 (~2.0) from parallel reductions
    rtol_kv = 0.005

    # Compare outputs
    torch.testing.assert_close(
        our_o.float(),
        ref_o.float(),
        atol=atol_o,
        rtol=rtol_o,
        msg=f"Output mismatch for BF16 state kernel (B={batch_size}, T={seq_len})",
    )

    # Compare states: both in bf16 (kernel [B, HV, V, K], ref [B, HV, K, V])
    ref_state_transposed = ref_state.transpose(-2, -1).contiguous()
    torch.testing.assert_close(
        our_state.float(),
        ref_state_transposed.float(),
        atol=atol_kv,
        rtol=rtol_kv,
        msg=f"State mismatch for BF16 state kernel (B={batch_size}, T={seq_len})",
    )

    print(
        f"✓ BF16 state kernel test passed (batch={batch_size}, T={seq_len}, dtype={dtype}, h_state=bf16)"
    )


@pytest.mark.parametrize("beta", [True])
@pytest.mark.parametrize("alpha", [True])
@pytest.mark.parametrize("scale", ["auto"])  # Use 1/sqrt(K) like compare_flashinfer.py
@pytest.mark.parametrize("seq_len", [1, 2, 3, 4])
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize(
    "num_q_heads, num_k_heads, num_v_heads",
    [(16, 16, 32)],
)
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16, 32, 64, 128])
@pytest.mark.parametrize("dtype", ["bfloat16"])
def test_gdn_decode_bf16_state_kernel(
    dtype: str,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_size: int,
    batch_size: int,
    seq_len: int,
    scale: float | str,
    alpha: bool,
    beta: bool,
    seed: int = int(os.environ.get("SEED", "0")),
):
    scale_val = 1.0 / math.sqrt(head_size) if scale == "auto" else scale
    _test_gdn_decode_bf16_state_kernel(
        dtype,
        batch_size,
        num_q_heads,
        num_k_heads,
        num_v_heads,
        head_size,
        seq_len,
        scale_val,
        alpha,
        beta,
        seed,
    )


@pytest.mark.parametrize("seq_len", [1, 2, 3, 4])
@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize(
    "num_q_heads, num_k_heads, num_v_heads",
    [(16, 16, 32)],
)
def test_pretranspose_api_uses_gdn_decode_bf16_state(
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_size: int,
    batch_size: int,
    seq_len: int,
    seed: int = int(os.environ.get("SEED", "0")),
):
    """Verify gated_delta_rule_decode_pretranspose dispatches to BF16 state kernel when state is bf16 and K=V=128.

    Calls the API with bf16 state and checks output/state match the direct kernel call.
    """
    _skip_if_not_sm90_or_later()
    if not GDN_DECODE_BF16_STATE_AVAILABLE:
        pytest.skip("BF16 state kernel not available")

    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    dtype = torch.bfloat16
    device = torch.device("cuda")
    scale = 1.0 / math.sqrt(head_size)
    num_sab_heads = num_v_heads

    q = torch.randn(
        batch_size, seq_len, num_q_heads, head_size, dtype=dtype, device=device
    )
    k = torch.randn(
        batch_size, seq_len, num_k_heads, head_size, dtype=dtype, device=device
    )
    v = torch.randn(
        batch_size, seq_len, num_v_heads, head_size, dtype=dtype, device=device
    )
    a = (
        torch.randn(batch_size, seq_len, num_sab_heads, dtype=dtype, device=device)
        * 0.1
    )
    b_tensor = torch.randn(
        batch_size, seq_len, num_sab_heads, dtype=dtype, device=device
    )
    A_log = torch.randn(num_sab_heads, dtype=torch.float32, device=device) * 0.1
    dt_bias = torch.randn(num_sab_heads, dtype=torch.float32, device=device) * 0.1

    # State [B, HV, V, K] in bf16 (Qwen-style K-last) so API uses improved backend
    state_api = torch.randn(
        batch_size,
        num_sab_heads,
        head_size,
        head_size,
        dtype=torch.bfloat16,
        device=device,
    )
    state_direct = state_api.clone()

    # Via API (should dispatch to gdn_decode_bf16_state)
    out_api, state_api = gated_delta_rule_decode_pretranspose(
        q=q,
        k=k,
        v=v,
        state=state_api,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b_tensor,
        scale=scale,
        use_qk_l2norm=True,
    )

    # Direct improved kernel (T=1 uses gdn_decode_bf16_state, T>1 uses MTP
    # variant). Pool-only: synthesize sequential indices arange(B).
    kernel_fn = gdn_decode_bf16_state if seq_len == 1 else gdn_decode_bf16_state_mtp
    indices = torch.arange(batch_size, dtype=torch.int32, device=device)
    out_direct = kernel_fn(
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        q=q,
        k=k,
        v=v,
        b=b_tensor,
        initial_state_source=state_direct,
        initial_state_indices=indices,
        use_qk_l2norm_in_kernel=True,
        scale=scale,
    )

    torch.testing.assert_close(out_api, out_direct, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(state_api, state_direct, atol=1e-2, rtol=1e-2)
    print(
        f"✓ API gdn_decode_bf16_state backend verified (batch={batch_size}, T={seq_len})"
    )


# ============================================================================
# Test BF16 state kernel (T=1)
# ============================================================================


def _test_gdn_decode_bf16_state_t1_kernel(
    dtype: str,
    batch_size: int,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_size: int,
    scale: float,
    alpha: bool,
    beta: bool,
    seed: int | None = None,
):
    """Test BF16 state kernel for T=1.

    Both kernel and reference use bf16 h state so the comparison is apples-to-apples.
    """
    _skip_if_not_sm90_or_later()

    if not GDN_DECODE_BF16_STATE_AVAILABLE:
        pytest.skip("BF16 state kernel not available")

    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    num_sab_heads = num_v_heads
    dtype_torch = getattr(torch, dtype)
    device = torch.device("cuda")

    with device:
        q = torch.randn(batch_size, 1, num_q_heads, head_size, dtype=dtype_torch)
        k = torch.randn(batch_size, 1, num_k_heads, head_size, dtype=dtype_torch)
        v = torch.randn(batch_size, 1, num_v_heads, head_size, dtype=dtype_torch)

        input_state_kernel = torch.randn(
            batch_size, num_sab_heads, head_size, head_size, dtype=torch.bfloat16
        )
        input_state_ref_bf16 = input_state_kernel.transpose(-2, -1).contiguous()

        A_log = torch.randn(num_sab_heads, dtype=torch.float32, device=device) * 0.1
        dt_bias = torch.randn(num_sab_heads, dtype=torch.float32, device=device) * 0.1
        a = (
            torch.randn(batch_size, 1, num_sab_heads, dtype=dtype_torch, device=device)
            * 0.1
        )

        if beta:
            b_tensor = torch.randn(
                batch_size, 1, num_sab_heads, dtype=dtype_torch, device=device
            )
        else:
            b_tensor = (
                torch.ones(
                    batch_size, 1, num_sab_heads, dtype=dtype_torch, device=device
                )
                * 10.0
            )

    # BF16 kernels are pool-only: treat [B, HV, V, K] as a pool of size B
    # with sequential indices arange(B). Mathematically identical to
    # non-pool semantics.
    our_state = input_state_kernel.clone()
    indices = torch.arange(batch_size, dtype=torch.int32, device=device)
    our_o = gdn_decode_bf16_state(
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        q=q,
        k=k,
        v=v,
        b=b_tensor,
        initial_state_source=our_state,
        initial_state_indices=indices,
        use_qk_l2norm_in_kernel=True,
        scale=scale,
    )

    torch.cuda.synchronize()

    ref_state = input_state_ref_bf16.clone()
    ref_o_t, ref_state = decode_delta_rule(
        q[:, 0].float(),
        k[:, 0].float(),
        v[:, 0].float(),
        ref_state,
        A_log=A_log,
        a=a[:, 0],
        dt_bias=dt_bias,
        b=b_tensor[:, 0],
        scale_factor=scale,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        use_l2_norm=True,
        state_dtype=torch.bfloat16,
    )
    ref_o = ref_o_t.unsqueeze(1).to(dtype_torch)

    atol_o = 0.001
    rtol_o = 0.005
    # State tolerances slightly higher: BF16 state accumulation at large batch
    # sizes can produce diffs up to ~0.016 (1 BF16 ULP at magnitude ~2)
    atol_kv = 0.02
    rtol_kv = 0.01

    torch.testing.assert_close(
        our_o.float(),
        ref_o.float(),
        atol=atol_o,
        rtol=rtol_o,
        msg=f"Output mismatch for BF16 state kernel (B={batch_size})",
    )

    ref_state_transposed = ref_state.transpose(-2, -1).contiguous()
    torch.testing.assert_close(
        our_state.float(),
        ref_state_transposed.float(),
        atol=atol_kv,
        rtol=rtol_kv,
        msg=f"State mismatch for BF16 state kernel (B={batch_size})",
    )

    print(f"  BF16 state T=1 PASS (batch={batch_size}, dtype={dtype})")


@pytest.mark.parametrize("beta", [True])
@pytest.mark.parametrize("alpha", [True])
@pytest.mark.parametrize("scale", ["auto"])
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize(
    "num_q_heads, num_k_heads, num_v_heads",
    [(16, 16, 32), (16, 16, 64)],
)
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
@pytest.mark.parametrize("dtype", ["bfloat16"])
def test_gdn_decode_bf16_state_t1_kernel(
    dtype: str,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_size: int,
    batch_size: int,
    scale: float | str,
    alpha: bool,
    beta: bool,
    seed: int = int(os.environ.get("SEED", "0")),
):
    scale_val = 1.0 / math.sqrt(head_size) if scale == "auto" else scale
    _test_gdn_decode_bf16_state_t1_kernel(
        dtype,
        batch_size,
        num_q_heads,
        num_k_heads,
        num_v_heads,
        head_size,
        scale_val,
        alpha,
        beta,
        seed,
    )


# ============================================================================
# Test BF16 state MTP kernel (T>=2)
# ============================================================================


def _test_gdn_decode_bf16_state_mtp_kernel(
    dtype: str,
    batch_size: int,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_size: int,
    seq_len: int,
    scale: float,
    cache_intermediate_states: bool,
    seed: int | None = None,
):
    """Test MTP BF16 state kernel for T>=2.

    Both kernel and reference use bf16 h state.
    Tests cache_intermediate_states and disable_state_update=True.
    """
    _skip_if_not_sm90_or_later()

    if not GDN_DECODE_BF16_STATE_AVAILABLE:
        pytest.skip("BF16 state kernel not available")

    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    num_sab_heads = num_v_heads
    dtype_torch = getattr(torch, dtype)
    device = torch.device("cuda")

    with device:
        q = torch.randn(batch_size, seq_len, num_q_heads, head_size, dtype=dtype_torch)
        k = torch.randn(batch_size, seq_len, num_k_heads, head_size, dtype=dtype_torch)
        v = torch.randn(batch_size, seq_len, num_v_heads, head_size, dtype=dtype_torch)

        pool_size = batch_size
        input_state_kernel = torch.randn(
            pool_size, num_sab_heads, head_size, head_size, dtype=torch.bfloat16
        )
        input_state_ref_bf16 = input_state_kernel.transpose(-2, -1).contiguous()

        A_log = torch.randn(num_sab_heads, dtype=torch.float32, device=device) * 0.1
        dt_bias = torch.randn(num_sab_heads, dtype=torch.float32, device=device) * 0.1
        a = (
            torch.randn(
                batch_size, seq_len, num_sab_heads, dtype=dtype_torch, device=device
            )
            * 0.1
        )
        b_tensor = torch.randn(
            batch_size, seq_len, num_sab_heads, dtype=dtype_torch, device=device
        )
        initial_state_indices = torch.arange(
            batch_size, dtype=torch.int32, device=device
        )

    if cache_intermediate_states:
        intermediate_states_buffer = torch.zeros(
            pool_size,
            seq_len,
            num_sab_heads,
            head_size,
            head_size,
            dtype=torch.bfloat16,
            device=device,
        )
    else:
        intermediate_states_buffer = None

    # Test with disable_state_update=True (MTP verify mode)
    our_state = input_state_kernel.clone()
    our_o = gdn_decode_bf16_state_mtp(
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        q=q,
        k=k,
        v=v,
        b=b_tensor,
        initial_state_source=our_state,
        initial_state_indices=initial_state_indices,
        intermediate_states_buffer=intermediate_states_buffer,
        disable_state_update=True,
        use_qk_l2norm_in_kernel=True,
        scale=scale,
    )

    torch.cuda.synchronize()

    # Reference: step through tokens with bf16 state
    ref_state = input_state_ref_bf16.clone()
    ref_outputs = []
    ref_intermediate_states = []

    for t in range(seq_len):
        ref_o_t, ref_state = decode_delta_rule(
            q[:, t].float(),
            k[:, t].float(),
            v[:, t].float(),
            ref_state,
            A_log=A_log,
            a=a[:, t],
            dt_bias=dt_bias,
            b=b_tensor[:, t],
            scale_factor=scale,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            use_l2_norm=True,
            state_dtype=torch.bfloat16,
        )
        ref_outputs.append(ref_o_t)
        if cache_intermediate_states:
            ref_intermediate_states.append(ref_state.clone())

    ref_o = torch.stack(ref_outputs, dim=1).to(dtype_torch)

    atol_o = 0.001
    rtol_o = 0.005

    torch.testing.assert_close(
        our_o.float(),
        ref_o.float(),
        atol=atol_o,
        rtol=rtol_o,
        msg=f"Output mismatch for MTP BF16 state kernel (B={batch_size}, T={seq_len})",
    )

    # With disable_state_update=True, initial state should be unchanged
    torch.testing.assert_close(
        our_state.float(),
        input_state_kernel.float(),
        atol=0,
        rtol=0,
        msg=f"State should be unchanged with disable_state_update=True (B={batch_size}, T={seq_len})",
    )

    # Check intermediate states buffer contents against reference
    if cache_intermediate_states and intermediate_states_buffer is not None:
        # intermediate_states_buffer: [pool_size, T, HV, V, K] (K-last layout, bf16)
        # ref intermediate states: [B, HV, K, V] per step (K-major layout, bf16)
        # Compare in batch chunks: at the wide_vec sweep's largest sizes the full
        # float32 upcast of both buffers (B*T*HV*V*K * 4B each) is several GiB
        # and trips an OOM inside torch.testing.assert_close that gets re-raised
        # as a generic RuntimeError, slipping past conftest's OOM-skip filter.
        atol_s = 0.02
        rtol_s = 0.01
        inter_bytes_per_b = seq_len * num_v_heads * head_size * head_size * 4
        chunk_b = max(1, min(batch_size, 256 * 1024 * 1024 // inter_bytes_per_b))
        for start in range(0, batch_size, chunk_b):
            stop = min(start + chunk_b, batch_size)
            ref_chunk = (
                torch.stack([s[start:stop] for s in ref_intermediate_states], dim=1)
                .transpose(-2, -1)
                .contiguous()
            )
            torch.testing.assert_close(
                intermediate_states_buffer[start:stop].float(),
                ref_chunk.float(),
                atol=atol_s,
                rtol=rtol_s,
                msg=f"Intermediate states mismatch for MTP BF16 state kernel (B[{start}:{stop}], T={seq_len})",
            )
            del ref_chunk

    print(
        f"  BF16 state MTP PASS (batch={batch_size}, T={seq_len}, "
        f"cache_intermediate={cache_intermediate_states})"
    )


@pytest.mark.parametrize("cache_intermediate_states", [True, False])
@pytest.mark.parametrize("seq_len", [2, 4, 8])
@pytest.mark.parametrize("scale", ["auto"])
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize(
    "num_q_heads, num_k_heads, num_v_heads",
    [(16, 16, 32)],
)
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("dtype", ["bfloat16"])
def test_gdn_decode_bf16_state_mtp_kernel(
    dtype: str,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_size: int,
    batch_size: int,
    seq_len: int,
    scale: float | str,
    cache_intermediate_states: bool,
    seed: int = int(os.environ.get("SEED", "0")),
):
    scale_val = 1.0 / math.sqrt(head_size) if scale == "auto" else scale
    _test_gdn_decode_bf16_state_mtp_kernel(
        dtype,
        batch_size,
        num_q_heads,
        num_k_heads,
        num_v_heads,
        head_size,
        seq_len,
        scale_val,
        cache_intermediate_states,
        seed,
    )


# ==============================================================================
# BF16 state MTP: wide-vector variant (gated_delta_rule_mtp_wide_vec)
# ==============================================================================
# Reuses _test_gdn_decode_bf16_state_mtp_kernel by monkey-patching the module's
# `gdn_decode_bf16_state_mtp` symbol for the scope of this test only. The
# parametrization is wider (B up to 256, T up to 8, HV in {32, 64}) because
# wide_vec's sweet spot is at large work-sizes; we want coverage where it
# matters. See results/bf16_mtp_optimization_apr18/wide_vec_design.md for the design.

try:
    from flashinfer.gdn_kernels.gdn_decode_bf16_state import (
        gated_delta_rule_mtp_wide_vec,
    )

    GDN_DECODE_BF16_STATE_WIDE_VEC_AVAILABLE = True
except ImportError:
    GDN_DECODE_BF16_STATE_WIDE_VEC_AVAILABLE = False


@pytest.mark.parametrize("tile_v", [32, 64, 128])
@pytest.mark.parametrize("cache_intermediate_states", [True, False])
@pytest.mark.parametrize("seq_len", [2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize(
    "num_q_heads, num_k_heads, num_v_heads",
    [(16, 16, 64)],
)
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16, 32, 64, 128, 256])
@pytest.mark.parametrize("dtype", ["bfloat16"])
def test_gdn_decode_bf16_state_wide_vec_mtp_kernel(
    monkeypatch,
    dtype: str,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_size: int,
    batch_size: int,
    seq_len: int,
    cache_intermediate_states: bool,
    tile_v: int,
    seed: int = int(os.environ.get("SEED", "0")),
):
    if not GDN_DECODE_BF16_STATE_WIDE_VEC_AVAILABLE:
        pytest.skip("wide_vec kernel not available")
    # Swap the module-level kernel symbol that _test_gdn_decode_bf16_state_mtp_kernel
    # looks up at call time. monkeypatch auto-restores after the test.
    # functools.partial binds the `tile_v` kwarg on each call.
    import functools

    monkeypatch.setattr(
        sys.modules[__name__],
        "gdn_decode_bf16_state_mtp",
        functools.partial(gated_delta_rule_mtp_wide_vec, tile_v=tile_v),
    )
    scale_val = 1.0 / math.sqrt(head_size)
    _test_gdn_decode_bf16_state_mtp_kernel(
        dtype,
        batch_size,
        num_q_heads,
        num_k_heads,
        num_v_heads,
        head_size,
        seq_len,
        scale_val,
        cache_intermediate_states,
        seed,
    )


# ==============================================================================
# BF16 state recovery: per-request variable K (accepted_steps)
# ==============================================================================
# State-recovery mode (disable_output=True, disable_state_update=False) with
# per-request K. Each request advances state by `accepted_steps[i]+1` tokens
# instead of the uniform T. Reference: per-request loop calling the uniform-T
# kernel with T = K_i tokens for each request independently.


@pytest.mark.parametrize("max_T", [2, 4, 8])
@pytest.mark.parametrize("batch_size", [1, 4, 16, 64, 256])
@pytest.mark.parametrize(
    "num_q_heads, num_k_heads, num_v_heads",
    [(16, 16, 64)],
)
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize("dtype", ["bfloat16"])
def test_gdn_decode_bf16_state_recovery_per_request_k(
    dtype: str,
    head_size: int,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    batch_size: int,
    max_T: int,
    seed: int = int(os.environ.get("SEED", "0")),
):
    """Per-request variable K state recovery (FLA equivalent)."""
    if not GDN_DECODE_BF16_STATE_AVAILABLE:
        pytest.skip("BF16 state kernel not available")
    _skip_if_not_sm90_or_later()

    try:
        from flashinfer.gdn_kernels.gdn_decode_bf16_state import (
            gated_delta_rule_mtp,
        )
    except ImportError:
        pytest.skip("gated_delta_rule_mtp not available")

    torch.manual_seed(seed)
    device = torch.device("cuda")
    B, T, HV, H, V, K = (
        batch_size,
        max_T,
        num_v_heads,
        num_q_heads,
        head_size,
        head_size,
    )

    # Random per-request K, each in [0, T-1] (0 means 1 token accepted).
    accepted_steps = torch.randint(0, T, (B,), dtype=torch.int32, device=device)

    pool_state = torch.randn(B, HV, V, K, dtype=torch.bfloat16, device=device)
    h0_indices = torch.arange(B, dtype=torch.int32, device=device)
    q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device=device)
    k = torch.randn(B, T, H, K, dtype=torch.bfloat16, device=device)
    v = torch.randn(B, T, HV, V, dtype=torch.bfloat16, device=device)
    A_log = torch.randn(HV, dtype=torch.float32, device=device)
    a = torch.randn(B, T, HV, dtype=torch.bfloat16, device=device)
    dt_bias = torch.randn(HV, dtype=torch.float32, device=device)
    b = torch.randn(B, T, HV, dtype=torch.bfloat16, device=device)

    # Per-request kernel call (single launch processing all B requests with
    # varying K via accepted_steps).
    pool_state_perreq = pool_state.clone()
    gated_delta_rule_mtp(
        q=q,
        k=k,
        v=v,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        initial_state_source=pool_state_perreq,
        initial_state_indices=h0_indices,
        accepted_steps=accepted_steps,
        disable_state_update=False,  # write final state
        disable_output=True,  # recovery: no output
        use_qk_l2norm_in_kernel=True,
        scale=K**-0.5,
        softplus_beta=1.0,
        softplus_threshold=20.0,
    )
    torch.cuda.synchronize()

    # Reference: independent uniform-T calls, one per request, with T=K_i.
    state_ref = pool_state.clone()
    for i in range(B):
        K_i = int(accepted_steps[i].item()) + 1
        pool_i = state_ref[i : i + 1].clone()
        gated_delta_rule_mtp(
            q=q[i : i + 1, :K_i].contiguous(),
            k=k[i : i + 1, :K_i].contiguous(),
            v=v[i : i + 1, :K_i].contiguous(),
            a=a[i : i + 1, :K_i].contiguous(),
            b=b[i : i + 1, :K_i].contiguous(),
            A_log=A_log,
            dt_bias=dt_bias,
            initial_state_source=pool_i,
            initial_state_indices=torch.tensor([0], dtype=torch.int32, device=device),
            disable_state_update=False,
            disable_output=True,
            use_qk_l2norm_in_kernel=True,
            scale=K**-0.5,
            softplus_beta=1.0,
            softplus_threshold=20.0,
        )
        torch.cuda.synchronize()
        state_ref[i] = pool_i[0]

    # BF16 epsilon is 2^-8 ≈ 0.0039. Noise scales with sqrt(N) where N is
    # accumulation count (B * K * K-dim reductions). At B=256, T=8, expected
    # noise envelope is ~0.05 — set tolerance just above.
    diff = (pool_state_perreq - state_ref).abs().max().item()
    assert diff <= 0.06, (
        f"per-request kernel deviates from per-request reference: "
        f"max abs diff = {diff:.6f} (B={B}, T={max_T}, HV={HV})"
    )


# ==============================================================================
# BF16 fused recovery+decode mode (recovery_steps > 0)
# ==============================================================================
# Fused mode collapses what would otherwise be two kernel calls:
#   Call A (state-only over K verified tokens, writes h_K)
#   Call B (full update over T-K speculated tokens with per-token output)
# into a single call with recovery_steps=K. The kernel:
#   - runs Phase A (no Q load / no output emission) for the first K tokens,
#   - writes h_K to the state pool asynchronously at the boundary (i_t=K-1),
#   - runs Phase B (full update with output) for the remaining T-K tokens,
#   - SKIPS the final h_T writeback (h_{K+T} is discarded — spec-decode
#     reject branch semantics: the new "good" state is h_K).
# Verifies bit-equivalence (within BF16 noise) against the two-call reference.


@pytest.mark.parametrize("recovery_steps", [1, 2, "T-1"])
@pytest.mark.parametrize("T", [4, 8])
@pytest.mark.parametrize("batch_size", [2, 8, 32])
@pytest.mark.parametrize(
    "num_q_heads, num_k_heads, num_v_heads",
    [(16, 16, 64)],
)
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize("dtype", ["bfloat16"])
def test_gdn_decode_bf16_state_fused_recovery_decode(
    dtype: str,
    head_size: int,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    batch_size: int,
    T: int,
    recovery_steps,
    seed: int = int(os.environ.get("SEED", "0")),
):
    """Fused recovery+decode (recovery_steps > 0) vs two-call reference."""
    if not GDN_DECODE_BF16_STATE_AVAILABLE:
        pytest.skip("BF16 state kernel not available")
    _skip_if_not_sm90_or_later()

    try:
        from flashinfer.gdn_kernels.gdn_decode_bf16_state import (
            gated_delta_rule_mtp,
        )
    except ImportError:
        pytest.skip("gated_delta_rule_mtp not available")

    # Normalize parametrization: "T-1" → T-1 (full prefix, last 1 decoded).
    K = (T - 1) if recovery_steps == "T-1" else recovery_steps
    if K >= T:
        pytest.skip(
            f"recovery_steps={K} >= T={T} (boundary at i_t=K-1 would equal final write)"
        )

    torch.manual_seed(seed)
    device = torch.device("cuda")
    B, HV, H, V, K_dim = batch_size, num_v_heads, num_q_heads, head_size, head_size

    # Shared inputs across fused and reference paths.
    h0_indices = torch.arange(B, dtype=torch.int32, device=device)
    q = torch.randn(B, T, H, K_dim, dtype=torch.bfloat16, device=device)
    k = torch.randn(B, T, H, K_dim, dtype=torch.bfloat16, device=device)
    v = torch.randn(B, T, HV, V, dtype=torch.bfloat16, device=device)
    A_log = torch.randn(HV, dtype=torch.float32, device=device)
    a = torch.randn(B, T, HV, dtype=torch.bfloat16, device=device)
    dt_bias = torch.randn(HV, dtype=torch.float32, device=device)
    b = torch.randn(B, T, HV, dtype=torch.bfloat16, device=device)
    pool_state_init = torch.randn(B, HV, V, K_dim, dtype=torch.bfloat16, device=device)

    common_kwargs = dict(
        A_log=A_log,
        dt_bias=dt_bias,
        initial_state_indices=h0_indices,
        use_qk_l2norm_in_kernel=True,
        scale=K_dim**-0.5,
        softplus_beta=1.0,
        softplus_threshold=20.0,
    )

    # === Fused single-call: recovery_steps=K, T total tokens ===
    pool_fused = pool_state_init.clone()
    out_fused = gated_delta_rule_mtp(
        q=q,
        k=k,
        v=v,
        a=a,
        b=b,
        initial_state_source=pool_fused,
        disable_state_update=False,  # writeback at i_t=K-1 (h_K)
        disable_output=False,  # emit output for i_t ∈ [K, T-1]
        recovery_steps=K,
        **common_kwargs,
    )

    # === Two-call reference ===
    # Call A: state-only over first K tokens → writes h_K to pool_ref.
    pool_ref = pool_state_init.clone()
    gated_delta_rule_mtp(
        q=q[:, :K].contiguous(),
        k=k[:, :K].contiguous(),
        v=v[:, :K].contiguous(),
        a=a[:, :K].contiguous(),
        b=b[:, :K].contiguous(),
        initial_state_source=pool_ref,
        disable_state_update=False,
        disable_output=True,
        recovery_steps=0,
        **common_kwargs,
    )
    # Call B: full update over remaining T-K tokens starting from h_K,
    # disable_state_update=True so we don't overwrite h_K with h_T.
    out_ref_tail = gated_delta_rule_mtp(
        q=q[:, K:].contiguous(),
        k=k[:, K:].contiguous(),
        v=v[:, K:].contiguous(),
        a=a[:, K:].contiguous(),
        b=b[:, K:].contiguous(),
        initial_state_source=pool_ref,  # contains h_K
        disable_state_update=True,
        disable_output=False,
        recovery_steps=0,
        **common_kwargs,
    )

    # Compare:
    # 1. State pool: fused-written h_K must equal Call A's h_K.
    state_diff = (pool_fused - pool_ref).abs().max().item()
    # 2. Output: fused's output[K:T] must equal Call B's full output [0:T-K].
    #    output[0:K] of fused is UNDEFINED (Phase A skipped writes) — don't check.
    out_diff = (out_fused[:, K:] - out_ref_tail).abs().max().item()

    # Wider tolerance than per-request K test: the state propagates through
    # MMAs, so we accumulate BF16 noise from K + (T-K) reductions. Empirically
    # ~0.06 covers worst case at B=32, T=8, K=4.
    assert state_diff <= 0.06, (
        f"fused h_K deviates from two-call h_K: max diff = {state_diff:.6f} "
        f"(B={B}, T={T}, recovery_steps={K})"
    )
    assert out_diff <= 0.06, (
        f"fused output[K:T] deviates from two-call output: max diff = {out_diff:.6f} "
        f"(B={B}, T={T}, recovery_steps={K})"
    )


# ==============================================================================
# BF16 fused recovery+decode with per-request K via accepted_steps
# ==============================================================================
# Unifies recovery and fused modes under a single per-request control tensor.
# When accepted_steps[i] is set and disable_output=False, disable_state_update=False:
#   - Phase A length per CTA = accepted_steps[i] + 1 (state-only iters)
#   - Boundary STG writes h_{accepted_steps[i]+1} to the state pool per request
#   - Phase B length per CTA = T - (accepted_steps[i] + 1) (output-emitting iters)
#   - Final h_T writeback is skipped (per-request fused semantics — discard h_T)
# This replaces the older scalar recovery_steps for fused mode. The scalar
# recovery_steps kwarg is ignored when accepted_steps is provided in fused
# flags.
# Reference: per-request two-call chain. For each request i:
#   Call A: gated_delta_rule_mtp(..., q=q[i, :K_i+1], disable_output=True,
#                                disable_state_update=False) → writes h_{K_i+1}
#   Call B: gated_delta_rule_mtp(..., q=q[i, K_i+1:T], disable_output=False,
#                                disable_state_update=True) → emits output[K_i+1:T]


@pytest.mark.parametrize("max_T", [4, 8])
@pytest.mark.parametrize("batch_size", [2, 8, 32])
@pytest.mark.parametrize(
    "num_q_heads, num_k_heads, num_v_heads",
    [(16, 16, 64)],
)
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize("dtype", ["bfloat16"])
def test_gdn_decode_bf16_state_fused_per_request_k(
    dtype: str,
    head_size: int,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    batch_size: int,
    max_T: int,
    seed: int = int(os.environ.get("SEED", "0")),
):
    """Fused recovery+decode with per-request K (via accepted_steps) vs
    per-request two-call reference."""
    if not GDN_DECODE_BF16_STATE_AVAILABLE:
        pytest.skip("BF16 state kernel not available")
    _skip_if_not_sm90_or_later()

    try:
        from flashinfer.gdn_kernels.gdn_decode_bf16_state import (
            gated_delta_rule_mtp,
        )
    except ImportError:
        pytest.skip("gated_delta_rule_mtp not available")

    torch.manual_seed(seed)
    device = torch.device("cuda")
    B, T, HV, H, V, K_dim = (
        batch_size,
        max_T,
        num_v_heads,
        num_q_heads,
        head_size,
        head_size,
    )

    # Per-request K in [0, T-2] so that Phase B has at least 1 output iter
    # per request. (K=T-1 would mean Phase B is empty for that request — also
    # a valid case, but excluded here to keep the output comparison non-trivial
    # for every request in the batch.)
    accepted_steps = torch.randint(0, T - 1, (B,), dtype=torch.int32, device=device)

    h0_indices = torch.arange(B, dtype=torch.int32, device=device)
    q = torch.randn(B, T, H, K_dim, dtype=torch.bfloat16, device=device)
    k = torch.randn(B, T, H, K_dim, dtype=torch.bfloat16, device=device)
    v = torch.randn(B, T, HV, V, dtype=torch.bfloat16, device=device)
    A_log = torch.randn(HV, dtype=torch.float32, device=device)
    a = torch.randn(B, T, HV, dtype=torch.bfloat16, device=device)
    dt_bias = torch.randn(HV, dtype=torch.float32, device=device)
    b = torch.randn(B, T, HV, dtype=torch.bfloat16, device=device)
    pool_state_init = torch.randn(B, HV, V, K_dim, dtype=torch.bfloat16, device=device)

    common_kwargs = dict(
        A_log=A_log,
        dt_bias=dt_bias,
        use_qk_l2norm_in_kernel=True,
        scale=K_dim**-0.5,
        softplus_beta=1.0,
        softplus_threshold=20.0,
    )

    # === Fused single-call: per-request accepted_steps controls Phase A length ===
    pool_fused = pool_state_init.clone()
    out_fused = gated_delta_rule_mtp(
        q=q,
        k=k,
        v=v,
        a=a,
        b=b,
        initial_state_source=pool_fused,
        initial_state_indices=h0_indices,
        accepted_steps=accepted_steps,
        disable_state_update=False,  # write h_{K_i} at boundary
        disable_output=False,  # emit output for i_t in [K_i+1, T-1]
        **common_kwargs,
    )

    # === Per-request two-call reference ===
    pool_ref = pool_state_init.clone()
    out_ref = torch.zeros_like(out_fused)
    for i in range(B):
        K_i = int(accepted_steps[i].item()) + 1  # Phase A length for request i
        # Call A: state-only over first K_i tokens → writes h_{K_i} into pool_ref[i].
        pool_i = pool_ref[i : i + 1].clone()
        gated_delta_rule_mtp(
            q=q[i : i + 1, :K_i].contiguous(),
            k=k[i : i + 1, :K_i].contiguous(),
            v=v[i : i + 1, :K_i].contiguous(),
            a=a[i : i + 1, :K_i].contiguous(),
            b=b[i : i + 1, :K_i].contiguous(),
            initial_state_source=pool_i,
            initial_state_indices=torch.tensor([0], dtype=torch.int32, device=device),
            disable_state_update=False,
            disable_output=True,
            **common_kwargs,
        )
        pool_ref[i] = pool_i[0]
        # Call B: full update over the remaining T-K_i tokens with disable_state_update=True
        # (don't overwrite h_{K_i}). Emits output[i, 0:T-K_i] which corresponds to
        # out_fused[i, K_i:T].
        out_b_i = gated_delta_rule_mtp(
            q=q[i : i + 1, K_i:].contiguous(),
            k=k[i : i + 1, K_i:].contiguous(),
            v=v[i : i + 1, K_i:].contiguous(),
            a=a[i : i + 1, K_i:].contiguous(),
            b=b[i : i + 1, K_i:].contiguous(),
            initial_state_source=pool_i,  # contains h_{K_i}
            initial_state_indices=torch.tensor([0], dtype=torch.int32, device=device),
            disable_state_update=True,
            disable_output=False,
            **common_kwargs,
        )
        out_ref[i, K_i:T] = out_b_i[0]
        torch.cuda.synchronize()

    # 1. State pool: per-request fused boundary STG writes h_{K_i} for each request.
    state_diff = (pool_fused - pool_ref).abs().max().item()
    # 2. Output: per-request out_fused[i, K_i:T] must match Call B's output for request i.
    #    out_fused[i, 0:K_i] is UNDEFINED (Phase A skipped writes for request i) — don't check.
    out_diff_max = 0.0
    for i in range(B):
        K_i = int(accepted_steps[i].item()) + 1
        diff_i = (out_fused[i, K_i:T] - out_ref[i, K_i:T]).abs().max().item()
        out_diff_max = max(out_diff_max, diff_i)

    assert state_diff <= 0.06, (
        f"per-request fused h_K deviates from two-call h_K: max diff = {state_diff:.6f} "
        f"(B={B}, T={max_T}, accepted_steps={accepted_steps.tolist()})"
    )
    assert out_diff_max <= 0.06, (
        f"per-request fused output[K_i:T] deviates from two-call output: max diff = "
        f"{out_diff_max:.6f} (B={B}, T={max_T}, accepted_steps={accepted_steps.tolist()})"
    )


if __name__ == "__main__":
    print("Running smoke tests...")
    print("\n=== Testing PRETRANSPOSE version ===")
    _test_decode_kernel_pretranspose(
        dtype="bfloat16",
        batch_size=4,
        num_q_heads=16,
        num_k_heads=16,
        num_v_heads=32,
        head_size=128,
        scale=1.0,
        alpha=True,
        beta=True,
        seed=42,
    )

    print("\n=== Testing NONTRANSPOSE version ===")
    _test_decode_kernel_nontranspose(
        dtype="bfloat16",
        batch_size=4,
        num_q_heads=16,
        num_k_heads=16,
        num_v_heads=32,
        head_size=128,
        scale=1.0,
        alpha=True,
        beta=True,
        seed=42,
    )

    print("\n=== Testing MTP (VERIFY) version ===")
    _test_verify_kernel_mtp(
        dtype="bfloat16",
        batch_size=4,
        num_q_heads=16,
        num_k_heads=16,
        num_v_heads=32,
        head_size=128,
        seq_len=4,  # T > 1 for MTP
        scale=1.0,
        alpha=True,
        beta=True,
        cache_intermediate_states=True,
        seed=42,
    )

    print("\n=== Testing Pool+indices with negative indices ===")
    _test_decode_kernel_pretranspose_pool_negative_indices(
        dtype="bfloat16",
        batch_size=8,
        num_q_heads=16,
        num_k_heads=16,
        num_v_heads=32,
        head_size=128,
        scale=1.0,
        seed=42,
    )

    print("\n=== Testing Pool+indices all-padding ===")
    _test_decode_kernel_pretranspose_pool_all_padding(
        dtype="bfloat16",
        batch_size=8,
        num_q_heads=16,
        num_k_heads=16,
        num_v_heads=32,
        head_size=128,
        scale=1.0,
        seed=42,
    )

    print("\n=== Testing BF16 state kernel (T=1,2,3,4) ===")
    if GDN_DECODE_BF16_STATE_AVAILABLE:
        for t in [1, 2, 3, 4]:
            _test_gdn_decode_bf16_state_kernel(
                dtype="bfloat16",
                batch_size=4,
                num_q_heads=16,
                num_k_heads=16,
                num_v_heads=32,
                head_size=128,
                seq_len=t,
                scale=1.0,
                alpha=True,
                beta=True,
                seed=42,
            )
    else:
        print("⚠ BF16 state kernel not available, skipping...")

    print("\n✅ All smoke tests passed!")
    print("\nTo run full test suite:")
    print(
        "  PRETRANSPOSE:       pytest test_decode_delta_rule.py::test_decode_kernel_basic_pretranspose -v"
    )
    print(
        "  NONTRANSPOSE:       pytest test_decode_delta_rule.py::test_decode_kernel_basic_nontranspose -v"
    )
    print(
        "  MTP (VERIFY):       pytest test_decode_delta_rule.py::test_verify_kernel_mtp -v"
    )
    print(
        "  gdn_decode_bf16_state:  pytest test_decode_delta_rule.py::test_gdn_decode_bf16_state_kernel -v"
    )
    print("  ALL: pytest test_decode_delta_rule.py -v")


# ============================================================================
# Tests for output_state_indices (separate read/write pool indices)
# ============================================================================


@pytest.mark.parametrize("state_dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("batch_size", [1, 4, 16])
def test_output_state_indices(batch_size: int, state_dtype: str):
    """Test that output_state_indices writes to different pool slots than read."""
    _skip_if_not_sm90_or_later()

    num_q_heads: int = 16
    num_k_heads: int = 16
    num_v_heads: int = 32
    head_size: int = 128

    seed: int = 42
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    num_sab_heads = num_v_heads
    pool_size = batch_size * 4  # plenty of room
    dtype_torch = torch.bfloat16
    kv_dtype = getattr(torch, state_dtype)
    device = torch.device("cuda")

    with device:
        q = torch.randn(batch_size, 1, num_q_heads, head_size, dtype=dtype_torch)
        k = torch.nn.functional.normalize(
            torch.randn(batch_size, 1, num_k_heads, head_size, dtype=dtype_torch),
            p=2.0,
            dim=-1,
        )
        v = torch.randn(batch_size, 1, num_v_heads, head_size, dtype=dtype_torch)
        A_log = torch.randn(num_sab_heads, dtype=torch.float32) * 0.1
        dt_bias = torch.randn(num_sab_heads, dtype=torch.float32) * 0.1
        a = torch.randn(batch_size, 1, num_sab_heads, dtype=dtype_torch) * 0.1
        b = torch.randn(batch_size, 1, num_sab_heads, dtype=dtype_torch)

        pool = torch.randn(
            pool_size, num_sab_heads, head_size, head_size, dtype=kv_dtype
        )

        # Read from first batch_size slots, write to second batch_size slots
        read_indices = torch.arange(batch_size, dtype=torch.int32, device=device)
        write_indices = torch.arange(
            batch_size, 2 * batch_size, dtype=torch.int32, device=device
        )

    pool_orig = pool.clone()
    pool_under_test = pool.clone()

    out, _ = gated_delta_rule_decode_pretranspose(
        q=q,
        k=k,
        v=v,
        state=None,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        scale=1.0,
        use_qk_l2norm=True,
        initial_state=pool_under_test,
        initial_state_indices=read_indices,
        output_state_indices=write_indices,
    )

    # Reference: direct state path (gather from read slots)
    gathered = pool_orig[read_indices].clone()
    out_ref, updated_ref = gated_delta_rule_decode_pretranspose(
        q=q,
        k=k,
        v=v,
        state=gathered,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        scale=1.0,
        use_qk_l2norm=True,
    )

    atol = 1e-3
    rtol = 1e-3

    # Outputs must match
    torch.testing.assert_close(out, out_ref, atol=atol, rtol=rtol)

    # Write slots must contain updated state
    torch.testing.assert_close(
        pool_under_test[write_indices], updated_ref, atol=atol, rtol=rtol
    )

    # Read slots must be unchanged (we wrote to different slots)
    torch.testing.assert_close(
        pool_under_test[read_indices], pool_orig[read_indices], atol=atol, rtol=rtol
    )

    # Other slots must be unchanged
    used_mask = torch.zeros(pool_size, dtype=torch.bool, device=device)
    used_mask[read_indices] = True
    used_mask[write_indices] = True
    torch.testing.assert_close(
        pool_under_test[~used_mask], pool_orig[~used_mask], atol=atol, rtol=rtol
    )


@pytest.mark.parametrize("state_dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("batch_size", [1, 4, 16])
def test_output_state_indices_same_as_input(batch_size: int, state_dtype: str):
    """output_state_indices == initial_state_indices must match existing pool behavior."""
    _skip_if_not_sm90_or_later()

    torch.random.manual_seed(42)
    torch.cuda.manual_seed(42)

    num_sab_heads = 32
    pool_size = batch_size * 3
    dtype_torch = torch.bfloat16
    kv_dtype = getattr(torch, state_dtype)
    device = torch.device("cuda")
    head_size = 128

    with device:
        q = torch.randn(batch_size, 1, 16, head_size, dtype=dtype_torch)
        k = torch.nn.functional.normalize(
            torch.randn(batch_size, 1, 16, head_size, dtype=dtype_torch),
            p=2.0,
            dim=-1,
        )
        v = torch.randn(batch_size, 1, num_sab_heads, head_size, dtype=dtype_torch)
        A_log = torch.randn(num_sab_heads, dtype=torch.float32) * 0.1
        dt_bias = torch.randn(num_sab_heads, dtype=torch.float32) * 0.1
        a = torch.randn(batch_size, 1, num_sab_heads, dtype=dtype_torch) * 0.1
        b = torch.randn(batch_size, 1, num_sab_heads, dtype=dtype_torch)
        pool = torch.randn(
            pool_size, num_sab_heads, head_size, head_size, dtype=kv_dtype
        )
        indices = torch.arange(batch_size, dtype=torch.int32, device=device) * 3

    # Without output_state_indices
    pool1 = pool.clone()
    out1, _ = gated_delta_rule_decode_pretranspose(
        q=q,
        k=k,
        v=v,
        state=None,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        scale=1.0,
        use_qk_l2norm=True,
        initial_state=pool1,
        initial_state_indices=indices,
    )

    # With output_state_indices == initial_state_indices
    pool2 = pool.clone()
    out2, _ = gated_delta_rule_decode_pretranspose(
        q=q,
        k=k,
        v=v,
        state=None,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        scale=1.0,
        use_qk_l2norm=True,
        initial_state=pool2,
        initial_state_indices=indices,
        output_state_indices=indices,
    )
    atol = 1e-3
    rtol = 1e-3

    torch.testing.assert_close(out1, out2, atol=atol, rtol=rtol)
    torch.testing.assert_close(pool1, pool2, atol=atol, rtol=rtol)


# ============================================================================
# BF16 split-pool MTP coverage (T>=2): exercises wide_vec and mtp_ilp4 split
# ============================================================================


@pytest.mark.parametrize("batch_size", [1, 8, 32])
@pytest.mark.parametrize("seq_len", [2, 4])
@pytest.mark.parametrize("cache_intermediate_states", [True, False])
def test_gdn_decode_bf16_state_mtp_split_pool(
    batch_size: int,
    seq_len: int,
    cache_intermediate_states: bool,
):
    """Direct gated_delta_rule_mtp split-pool test for the BF16 path.

    Verifies that the kernel:
      - produces bit-identical output (to bf16 noise floor) vs the
        single-pool dispatch on the same inputs;
      - leaves the read slots untouched in split mode;
      - writes the updated state into the write slots.

    Sweeps three batch sizes that hit each kernel:
      - B=1   → mtp_ilp4_kernel (wide_vec gates out at work_units=64)
      - B=8   → wide_vec        (work_units=512 ≥ 128)
      - B=32  → wide_vec        (work_units=2048)
    """
    _skip_if_not_sm90_or_later()
    if not GDN_DECODE_BF16_STATE_AVAILABLE:
        pytest.skip("BF16 state kernel not available")

    torch.random.manual_seed(0)
    torch.cuda.manual_seed(0)

    HQ = HK = 16
    HV = 64
    K = V = 128
    device = torch.device("cuda")
    dtype = torch.bfloat16

    with device:
        q = torch.randn(batch_size, seq_len, HQ, K, dtype=dtype)
        k = torch.nn.functional.normalize(
            torch.randn(batch_size, seq_len, HK, K, dtype=dtype), p=2.0, dim=-1
        )
        v = torch.randn(batch_size, seq_len, HV, V, dtype=dtype)
        a = torch.randn(batch_size, seq_len, HV, dtype=dtype) * 0.1
        b_t = torch.randn(batch_size, seq_len, HV, dtype=dtype)
        A_log = torch.randn(HV, dtype=torch.float32) * 0.1
        dt_bias = torch.randn(HV, dtype=torch.float32) * 0.1

        # Pool of size 2*B so write slots [B..2B) don't alias the read slots.
        pool = torch.randn(2 * batch_size, HV, V, K, dtype=torch.bfloat16)
        read_idx = torch.arange(batch_size, dtype=torch.int32, device=device)
        write_idx = torch.arange(
            batch_size, 2 * batch_size, dtype=torch.int32, device=device
        )

    pool_orig = pool.clone()
    scale = 1.0 / math.sqrt(K)

    # Single-pool reference: read==write (slots [0..B)).
    pool_single = pool_orig.clone()
    cache_single = (
        torch.zeros(batch_size, seq_len, HV, V, K, dtype=torch.bfloat16, device=device)
        if cache_intermediate_states
        else None
    )
    out_single = gdn_decode_bf16_state_mtp(
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        b=b_t,
        initial_state_source=pool_single,
        initial_state_indices=read_idx,
        intermediate_states_buffer=cache_single,
        disable_state_update=False,
        use_qk_l2norm_in_kernel=True,
        scale=scale,
    )

    # Split-pool: same reads from [0..B), writes to [B..2B).
    pool_split = pool_orig.clone()
    cache_split = (
        torch.zeros(batch_size, seq_len, HV, V, K, dtype=torch.bfloat16, device=device)
        if cache_intermediate_states
        else None
    )
    out_split = gdn_decode_bf16_state_mtp(
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        b=b_t,
        initial_state_source=pool_split,
        initial_state_indices=read_idx,
        output_state_indices=write_idx,
        intermediate_states_buffer=cache_split,
        disable_state_update=False,
        use_qk_l2norm_in_kernel=True,
        scale=scale,
    )

    atol = 2e-3
    rtol = 5e-3

    # Outputs must match between single- and split-pool dispatch.
    torch.testing.assert_close(out_single, out_split, atol=atol, rtol=rtol)

    # Split-pool: read slots unchanged.
    torch.testing.assert_close(
        pool_split[:batch_size], pool_orig[:batch_size], atol=0, rtol=0
    )

    if not cache_intermediate_states:
        # Final state writeback is enabled. Split-pool's write slots must
        # hold the same final state that single-pool wrote into the read
        # slots. (When cache_intermediate_states=True, wide_vec skips this
        # writeback entirely while mtp_ilp4 still performs it; the
        # pool-vs-cache-vs-skip behavior is asserted elsewhere — here we
        # only need to verify the split-pool routing is bit-equivalent to
        # single-pool dispatch when both write the pool.)
        torch.testing.assert_close(
            pool_split[batch_size : 2 * batch_size],
            pool_single[:batch_size],
            atol=atol,
            rtol=rtol,
        )


# ============================================================================
# OOB regression — intermediate_states indexing with pool_size > B
# (mirrors upstream PR #3145; covers wide_vec + mtp_ilp4 batch-scoped index)
# ============================================================================


@pytest.mark.parametrize("pool_size_multiplier", [1, 4])
@pytest.mark.parametrize("batch_size", [1, 8, 32])
@pytest.mark.parametrize("seq_len", [2, 4])
def test_gdn_decode_bf16_state_mtp_pool_larger_than_batch(
    batch_size: int,
    seq_len: int,
    pool_size_multiplier: int,
):
    """Catch the OOB bug fixed by upstream PR #3145.

    The ``intermediate_states_buffer`` is BATCH-scoped (shape
    ``[B, T, HV, V, K]``), but the kernel previously indexed it with the
    pool-scoped ``cache_idx`` (``initial_state_indices[i_n]``). When
    ``pool_size > B`` and the caller picks indices into the upper slots,
    that index expression goes OOB and corrupts memory or crashes.

    This test:
      1. Allocates a pool of size ``pool_size_multiplier * B``.
      2. Sets ``initial_state_indices = arange(B, 2*B)`` (pointing at the
         second block of slots) when ``pool_size_multiplier > 1`` —
         large enough that ``cache_idx >= B`` and the OOB triggers
         under the old bug.
      3. Allocates ``intermediate_states_buffer`` of shape
         ``[B, T, HV, V, K]``.
      4. Runs ``gated_delta_rule_mtp`` and checks the output equals a
         reference run where the same pool slots are gathered into a
         ``[B, ...]`` tensor and addressed with ``arange(B)``.

    Sweeps three batch sizes (B=1 hits ``mtp_ilp4_kernel``; B=8 / B=32 hit
    ``gdn_wide_vec_kernel``) so both kernels' indexing fix is exercised.
    """
    _skip_if_not_sm90_or_later()
    if not GDN_DECODE_BF16_STATE_AVAILABLE:
        pytest.skip("BF16 state kernel not available")

    torch.random.manual_seed(0)
    torch.cuda.manual_seed(0)

    HQ = HK = 16
    HV = 64
    K = V = 128
    device = torch.device("cuda")
    dtype = torch.bfloat16

    with device:
        q = torch.randn(batch_size, seq_len, HQ, K, dtype=dtype)
        k = torch.nn.functional.normalize(
            torch.randn(batch_size, seq_len, HK, K, dtype=dtype), p=2.0, dim=-1
        )
        v = torch.randn(batch_size, seq_len, HV, V, dtype=dtype)
        a = torch.randn(batch_size, seq_len, HV, dtype=dtype) * 0.1
        b_t = torch.randn(batch_size, seq_len, HV, dtype=dtype)
        A_log = torch.randn(HV, dtype=torch.float32) * 0.1
        dt_bias = torch.randn(HV, dtype=torch.float32) * 0.1

        pool_size = pool_size_multiplier * batch_size
        pool = torch.randn(pool_size, HV, V, K, dtype=torch.bfloat16)

        # Read indices point into the upper half of the pool when
        # pool_size_multiplier > 1, so cache_idx >= batch_size and the
        # pre-#3145 cache_idx-based intermediate_states index goes OOB.
        if pool_size_multiplier == 1:
            read_idx = torch.arange(batch_size, dtype=torch.int32, device=device)
        else:
            read_idx = torch.arange(
                batch_size, 2 * batch_size, dtype=torch.int32, device=device
            )

    scale = 1.0 / math.sqrt(K)

    # Run-under-test: realistic pool layout, batch-scoped cache buffer.
    pool_under_test = pool.clone()
    cache_under_test = torch.zeros(
        batch_size, seq_len, HV, V, K, dtype=torch.bfloat16, device=device
    )
    out_under_test = gdn_decode_bf16_state_mtp(
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        b=b_t,
        initial_state_source=pool_under_test,
        initial_state_indices=read_idx,
        intermediate_states_buffer=cache_under_test,
        disable_state_update=False,
        use_qk_l2norm_in_kernel=True,
        scale=scale,
    )

    # Reference: gather the same pool slots into a B-sized pool and run
    # with arange(B) indices. The two configurations are mathematically
    # equivalent — the kernel just sees different pool dimensions, but
    # the same per-batch state is loaded.
    ref_pool = pool[read_idx.long()].clone()
    ref_indices = torch.arange(batch_size, dtype=torch.int32, device=device)
    ref_cache = torch.zeros(
        batch_size, seq_len, HV, V, K, dtype=torch.bfloat16, device=device
    )
    out_ref = gdn_decode_bf16_state_mtp(
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        b=b_t,
        initial_state_source=ref_pool,
        initial_state_indices=ref_indices,
        intermediate_states_buffer=ref_cache,
        disable_state_update=False,
        use_qk_l2norm_in_kernel=True,
        scale=scale,
    )

    atol = 2e-3
    rtol = 5e-3

    # Output must match the reference (no OOB, correct indexing).
    torch.testing.assert_close(out_under_test, out_ref, atol=atol, rtol=rtol)

    # The cache must populate exactly the same way in both runs (proves
    # the kernel's batch-scoped indexing into the cache is correct).
    torch.testing.assert_close(cache_under_test, ref_cache, atol=atol, rtol=rtol)

    # When pool_size > B, the pool slots OUTSIDE the read indices must be
    # unchanged. The earlier OOB bug would have written into them.
    if pool_size_multiplier > 1:
        untouched_mask = torch.ones(pool_size, dtype=torch.bool, device=device)
        untouched_mask[read_idx.long()] = False
        torch.testing.assert_close(
            pool_under_test[untouched_mask],
            pool[untouched_mask],
            atol=0,
            rtol=0,
        )


# ==============================================================================
# BF16 state FLA-style per-token pool scatter (ssm_state_indices)
# ==============================================================================
# vLLM-compatible API: when `ssm_state_indices` (shape [B, T] int32) is passed,
# the kernel writes each h_{t+1} directly to pool[ssm_state_indices[i, t]],
# replacing the dense intermediate_states_buffer. Caller is responsible for
# pre-allocating B*T fresh pool slots and sizing the pool to >= B*(T+1) total
# slots. See results/2026-06-03/FLA_SCATTER_MODE_PLAN.md for the design.


@pytest.mark.parametrize("split_pool", [False, True])
@pytest.mark.parametrize("max_T", [2, 4, 8])
@pytest.mark.parametrize("batch_size", [1, 4, 16, 64, 256])
@pytest.mark.parametrize(
    "num_q_heads, num_k_heads, num_v_heads",
    [(16, 16, 64)],
)
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize("dtype", ["bfloat16"])
def test_gdn_decode_bf16_state_fla_scatter_vs_dense(
    dtype: str,
    head_size: int,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    batch_size: int,
    max_T: int,
    split_pool: bool,
    seed: int = int(os.environ.get("SEED", "0")),
):
    """FLA-style per-token scatter must produce bit-identical per-token
    states to the dense-buffer reference. ``pool_fla[ssm_state_indices[i,t]]``
    must equal ``intermediate_buffer[i, t]`` for every (i, t)."""
    if not GDN_DECODE_BF16_STATE_AVAILABLE:
        pytest.skip("BF16 state kernel not available")
    _skip_if_not_sm90_or_later()

    try:
        from flashinfer.gdn_kernels.gdn_decode_bf16_state import (
            gated_delta_rule_mtp,
        )
    except ImportError:
        pytest.skip("gated_delta_rule_mtp not available")

    torch.manual_seed(seed)
    device = torch.device("cuda")
    B, T, HV, H, V, K = (
        batch_size,
        max_T,
        num_v_heads,
        num_q_heads,
        head_size,
        head_size,
    )

    # Pool sized for h0 (B slots) + per-token scatter destinations (B*T slots).
    # Split-pool case adds B more output slots.
    pool_size = B * (T + 1) + (B if split_pool else 0)
    pool_init = torch.randn(pool_size, HV, V, K, dtype=torch.bfloat16, device=device)

    h0_idx = torch.arange(B, dtype=torch.int32, device=device)
    ssm_idx = torch.arange(B, B + B * T, dtype=torch.int32, device=device).reshape(B, T)
    out_idx = None
    if split_pool:
        out_idx = torch.arange(
            B * (T + 1), B * (T + 2), dtype=torch.int32, device=device
        )

    q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device=device)
    k = torch.randn(B, T, H, K, dtype=torch.bfloat16, device=device)
    v = torch.randn(B, T, HV, V, dtype=torch.bfloat16, device=device)
    A_log = torch.randn(HV, dtype=torch.float32, device=device)
    a = torch.randn(B, T, HV, dtype=torch.bfloat16, device=device)
    dt_bias = torch.randn(HV, dtype=torch.float32, device=device)
    b = torch.randn(B, T, HV, dtype=torch.bfloat16, device=device)

    common = dict(
        q=q,
        k=k,
        v=v,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        initial_state_indices=h0_idx,
        use_qk_l2norm_in_kernel=True,
        scale=K**-0.5,
        softplus_beta=1.0,
        softplus_threshold=20.0,
    )
    if split_pool:
        common["output_state_indices"] = out_idx

    # FLA mode: per-token scatter to pool slots
    pool_fla = pool_init.clone()
    gated_delta_rule_mtp(
        **common,
        initial_state_source=pool_fla,
        ssm_state_indices=ssm_idx,
    )
    torch.cuda.synchronize()

    # Reference: dense buffer
    pool_dense = pool_init.clone()
    intermediate_buffer = torch.empty(
        B, T, HV, V, K, dtype=torch.bfloat16, device=device
    )
    gated_delta_rule_mtp(
        **common,
        initial_state_source=pool_dense,
        intermediate_states_buffer=intermediate_buffer,
    )
    torch.cuda.synchronize()

    # Bit-equivalence at every (i, t).
    max_diff = 0.0
    worst_cell = None
    for i in range(B):
        for t in range(T):
            slot = int(ssm_idx[i, t].item())
            diff = (
                (pool_fla[slot].float() - intermediate_buffer[i, t].float())
                .abs()
                .max()
                .item()
            )
            if diff > max_diff:
                max_diff = diff
                worst_cell = (i, t, slot)
    assert max_diff <= 0.06, (
        f"FLA pool scatter deviates from dense buffer: "
        f"max abs diff = {max_diff:.6f} at (i, t, slot)={worst_cell} "
        f"(B={B}, T={T}, split_pool={split_pool})"
    )

    # h_0 slots: under same_pool (split_pool=False), the kernel skips the
    # final-state writeback so the initial state is preserved bit-exactly.
    if not split_pool:
        torch.testing.assert_close(
            pool_fla[h0_idx.long()],
            pool_init[h0_idx.long()],
            atol=0,
            rtol=0,
            msg="FLA mode under same_pool must preserve h_0 slots bit-exactly",
        )

    # Under split-pool, FLA mode writes h_T (the final-step state) to
    # output_state_indices[i] as the final-state writeback. Reference: the
    # dense path's last-step intermediate buffer entry. (We cannot compare
    # to pool_dense[out_idx] directly because the wide_vec dense path skips
    # the final writeback when caching is on — "buffer[:, T-1] IS h_T".)
    if split_pool:
        out_h_T_fla = pool_fla[out_idx.long()]  # [B, HV, V, K]
        out_h_T_ref = intermediate_buffer[:, T - 1]  # [B, HV, V, K]
        out_diff = (out_h_T_fla.float() - out_h_T_ref.float()).abs().max().item()
        assert out_diff <= 0.06, (
            f"split-pool final-state writeback (out_idx slot) "
            f"deviates from dense buffer's h_T: max diff = {out_diff:.6f}"
        )


@pytest.mark.parametrize("max_T", [4, 8])
@pytest.mark.parametrize("batch_size", [1, 4, 64])
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize("dtype", ["bfloat16"])
def test_gdn_decode_bf16_state_fla_scatter_with_accepted_steps(
    dtype: str,
    head_size: int,
    batch_size: int,
    max_T: int,
    seed: int = int(os.environ.get("SEED", "0")),
):
    """FLA scatter + per-request K (accepted_steps). Only the first
    ``accepted_steps[i]+1`` slots per request should be written; slots
    beyond must retain their pre-call values."""
    if not GDN_DECODE_BF16_STATE_AVAILABLE:
        pytest.skip("BF16 state kernel not available")
    _skip_if_not_sm90_or_later()

    from flashinfer.gdn_kernels.gdn_decode_bf16_state import gated_delta_rule_mtp

    torch.manual_seed(seed)
    device = torch.device("cuda")
    B, T = batch_size, max_T
    H, HV, K, V = 16, 64, head_size, head_size

    pool_size = B * (T + 1)
    pool_init = torch.randn(pool_size, HV, V, K, dtype=torch.bfloat16, device=device)
    h0_idx = torch.arange(B, dtype=torch.int32, device=device)
    ssm_idx = torch.arange(B, B + B * T, dtype=torch.int32, device=device).reshape(B, T)
    accepted_steps = torch.randint(0, T, (B,), dtype=torch.int32, device=device)

    pool_fla = pool_init.clone()
    gated_delta_rule_mtp(
        q=torch.randn(B, T, H, K, dtype=torch.bfloat16, device=device),
        k=torch.randn(B, T, H, K, dtype=torch.bfloat16, device=device),
        v=torch.randn(B, T, HV, V, dtype=torch.bfloat16, device=device),
        a=torch.randn(B, T, HV, dtype=torch.bfloat16, device=device),
        b=torch.randn(B, T, HV, dtype=torch.bfloat16, device=device),
        A_log=torch.randn(HV, dtype=torch.float32, device=device),
        dt_bias=torch.randn(HV, dtype=torch.float32, device=device),
        initial_state_source=pool_fla,
        initial_state_indices=h0_idx,
        ssm_state_indices=ssm_idx,
        accepted_steps=accepted_steps,
        use_qk_l2norm_in_kernel=True,
        scale=K**-0.5,
    )
    torch.cuda.synchronize()

    # Slots with t > accepted_steps[i] must be UNCHANGED from pool_init.
    for i in range(B):
        K_i = int(accepted_steps[i].item()) + 1
        for t in range(K_i, T):
            slot = int(ssm_idx[i, t].item())
            torch.testing.assert_close(
                pool_fla[slot],
                pool_init[slot],
                atol=0,
                rtol=0,
                msg=(
                    f"slot beyond accepted_steps was clobbered: "
                    f"req {i}, t={t}, K_i={K_i}, slot={slot}"
                ),
            )

    # h_0 slots must also be unchanged (FLA + same_pool).
    torch.testing.assert_close(
        pool_fla[h0_idx.long()],
        pool_init[h0_idx.long()],
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize("max_T", [4, 8])
@pytest.mark.parametrize("batch_size", [4, 64])
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize("dtype", ["bfloat16"])
def test_gdn_decode_bf16_state_fla_scatter_state_only(
    dtype: str,
    head_size: int,
    batch_size: int,
    max_T: int,
    seed: int = int(os.environ.get("SEED", "0")),
):
    """FLA scatter + disable_output=True (state-only mode). Per-token states
    must be scattered correctly even when no output is materialized."""
    if not GDN_DECODE_BF16_STATE_AVAILABLE:
        pytest.skip("BF16 state kernel not available")
    _skip_if_not_sm90_or_later()

    from flashinfer.gdn_kernels.gdn_decode_bf16_state import gated_delta_rule_mtp

    torch.manual_seed(seed)
    device = torch.device("cuda")
    B, T = batch_size, max_T
    H, HV, K, V = 16, 64, head_size, head_size

    pool_size = B * (T + 1)
    pool_init = torch.randn(pool_size, HV, V, K, dtype=torch.bfloat16, device=device)
    h0_idx = torch.arange(B, dtype=torch.int32, device=device)
    ssm_idx = torch.arange(B, B + B * T, dtype=torch.int32, device=device).reshape(B, T)

    common = dict(
        q=torch.randn(B, T, H, K, dtype=torch.bfloat16, device=device),
        k=torch.randn(B, T, H, K, dtype=torch.bfloat16, device=device),
        v=torch.randn(B, T, HV, V, dtype=torch.bfloat16, device=device),
        a=torch.randn(B, T, HV, dtype=torch.bfloat16, device=device),
        b=torch.randn(B, T, HV, dtype=torch.bfloat16, device=device),
        A_log=torch.randn(HV, dtype=torch.float32, device=device),
        dt_bias=torch.randn(HV, dtype=torch.float32, device=device),
        initial_state_indices=h0_idx,
        use_qk_l2norm_in_kernel=True,
        scale=K**-0.5,
        disable_output=True,  # state-only
    )

    pool_fla = pool_init.clone()
    gated_delta_rule_mtp(
        **common,
        initial_state_source=pool_fla,
        ssm_state_indices=ssm_idx,
    )
    torch.cuda.synchronize()

    # Reference: dense buffer in same state-only mode.
    pool_dense = pool_init.clone()
    intermediate_buffer = torch.empty(
        B, T, HV, V, K, dtype=torch.bfloat16, device=device
    )
    gated_delta_rule_mtp(
        **common,
        initial_state_source=pool_dense,
        intermediate_states_buffer=intermediate_buffer,
    )
    torch.cuda.synchronize()

    # Per-token states equal.
    for i in range(B):
        for t in range(T):
            slot = int(ssm_idx[i, t].item())
            diff = (
                (pool_fla[slot].float() - intermediate_buffer[i, t].float())
                .abs()
                .max()
                .item()
            )
            assert diff <= 0.06, (
                f"state-only FLA mode mismatch at (i={i}, t={t}): {diff:.6f}"
            )


def test_gdn_decode_bf16_state_fla_scatter_validation():
    """Wrapper validation rejects illegal ssm_state_indices combinations."""
    if not GDN_DECODE_BF16_STATE_AVAILABLE:
        pytest.skip("BF16 state kernel not available")
    _skip_if_not_sm90_or_later()

    from flashinfer.gdn_kernels.gdn_decode_bf16_state import gated_delta_rule_mtp

    device = torch.device("cuda")
    torch.manual_seed(0)
    B, T = 2, 4
    H, HV, K, V = 8, 8, 128, 128

    def mk_args(B=B, T=T):
        return dict(
            A_log=torch.randn(HV, dtype=torch.float32, device=device),
            a=torch.randn(B, T, HV, dtype=torch.bfloat16, device=device),
            dt_bias=torch.randn(HV, dtype=torch.float32, device=device),
            q=torch.randn(B, T, H, K, dtype=torch.bfloat16, device=device),
            k=torch.randn(B, T, H, K, dtype=torch.bfloat16, device=device),
            v=torch.randn(B, T, HV, V, dtype=torch.bfloat16, device=device),
            b=torch.randn(B, T, HV, dtype=torch.bfloat16, device=device),
            initial_state_source=torch.zeros(
                B * (T + 1), HV, V, K, dtype=torch.bfloat16, device=device
            ),
            initial_state_indices=torch.arange(B, dtype=torch.int32, device=device),
        )

    ssm_idx = torch.arange(B, B + B * T, dtype=torch.int32, device=device).reshape(B, T)

    # Mutex with intermediate_states_buffer
    ibuf = torch.zeros(B, T, HV, V, K, dtype=torch.bfloat16, device=device)
    with pytest.raises(AssertionError, match="mutually exclusive"):
        gated_delta_rule_mtp(
            **mk_args(), ssm_state_indices=ssm_idx, intermediate_states_buffer=ibuf
        )

    # Mutex with disable_state_update
    with pytest.raises(AssertionError, match="state writes"):
        gated_delta_rule_mtp(
            **mk_args(), ssm_state_indices=ssm_idx, disable_state_update=True
        )

    # Mutex with recovery_steps > 0
    with pytest.raises(AssertionError, match="not yet supported"):
        gated_delta_rule_mtp(**mk_args(), ssm_state_indices=ssm_idx, recovery_steps=1)

    # T = 1 not supported (MVP exclusion)
    bad_ssm = torch.zeros(B, 1, dtype=torch.int32, device=device)
    with pytest.raises(AssertionError, match="T >= 2"):
        gated_delta_rule_mtp(**mk_args(T=1), ssm_state_indices=bad_ssm)

    # Wrong dtype
    with pytest.raises(AssertionError, match="int32"):
        gated_delta_rule_mtp(**mk_args(), ssm_state_indices=ssm_idx.to(torch.int64))

    # Wrong shape
    bad_shape = torch.zeros(B, T + 1, dtype=torch.int32, device=device)
    with pytest.raises(AssertionError, match="shape"):
        gated_delta_rule_mtp(**mk_args(), ssm_state_indices=bad_shape)


# ============================================================================
# Additional FLA-scatter coverage: padded pool (fallback), non-contiguous
# slots (FLA index pattern), and FP32 FLA scatter (mirrors BF16 vs_dense).
# ============================================================================


@pytest.mark.parametrize("max_T", [2, 4, 8])
@pytest.mark.parametrize("batch_size", [4, 16])
def test_gdn_decode_bf16_state_fla_scatter_padded_pool(
    batch_size: int,
    max_T: int,
):
    """FLA scatter on a vLLM-style padded pool (stride[0] > HV*V*K).

    Exercises the `per_token_pool_scatter_flat=False` fallback branch
    (4D slot-slice) — the contiguous-pool flat path is mutex with padded
    layouts, so this is the only path that can run for vLLM's actual
    production pool (conv state co-allocated into each slot's stride).
    Without coverage here, the slot-slice fallback would have zero tests.
    """
    if not GDN_DECODE_BF16_STATE_AVAILABLE:
        pytest.skip("BF16 state kernel not available")
    _skip_if_not_sm90_or_later()

    from flashinfer.gdn_kernels.gdn_decode_bf16_state import gated_delta_rule_mtp

    torch.manual_seed(0)
    device = torch.device("cuda")
    B, T = batch_size, max_T
    H, HV, K, V = 16, 64, 128, 128

    pool_size = B * (T + 1)
    # vLLM-style padded layout: extra HV-aligned padding per slot.
    inner = HV * V * K
    pad_elts = 16384  # HV-row aligned
    pad_hv_rows = pad_elts // (V * K)
    big = torch.empty(
        pool_size, HV + pad_hv_rows, V, K, dtype=torch.bfloat16, device=device
    )
    pool_padded = big[:, :HV, :, :]
    pool_padded.copy_(torch.randn(pool_size, HV, V, K, dtype=torch.bfloat16) * 0.1)
    pool_padded._owner = big  # keep allocation alive
    assert pool_padded.stride() == (inner + pad_elts, V * K, K, 1)
    assert not pool_padded.is_contiguous()

    # Mirror the same data into a contiguous reference pool for diff.
    pool_contig_ref = pool_padded.contiguous()
    assert pool_contig_ref.stride() == (HV * V * K, V * K, K, 1)

    h0_idx = torch.arange(B, dtype=torch.int32, device=device)
    ssm_idx = torch.arange(B, B + B * T, dtype=torch.int32, device=device).reshape(B, T)

    q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device=device)
    k = torch.randn(B, T, H, K, dtype=torch.bfloat16, device=device)
    v = torch.randn(B, T, HV, V, dtype=torch.bfloat16, device=device)
    a = torch.randn(B, T, HV, dtype=torch.bfloat16, device=device)
    b = torch.randn(B, T, HV, dtype=torch.bfloat16, device=device)
    A_log = torch.randn(HV, dtype=torch.float32, device=device)
    dt_bias = torch.randn(HV, dtype=torch.float32, device=device)

    common = dict(
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        b=b,
        initial_state_indices=h0_idx,
        ssm_state_indices=ssm_idx,
        use_qk_l2norm_in_kernel=True,
        scale=K**-0.5,
    )

    # Reference: contiguous pool → flat path
    pool_ref = pool_contig_ref.clone()
    gated_delta_rule_mtp(**common, initial_state_source=pool_ref)
    torch.cuda.synchronize()

    # Under test: padded pool → slot-slice fallback
    pool_under_test = pool_padded.clone()
    gated_delta_rule_mtp(**common, initial_state_source=pool_under_test)
    torch.cuda.synchronize()

    # Per-token scatter destinations must match between flat and fallback
    # paths bit-exactly — they write the same h_{t+1} values.
    for i in range(B):
        for t in range(T):
            slot = int(ssm_idx[i, t].item())
            torch.testing.assert_close(
                pool_under_test[slot].contiguous(),
                pool_ref[slot],
                atol=0,
                rtol=0,
                msg=f"padded-pool fallback diverged at (i={i}, t={t}, slot={slot})",
            )


@pytest.mark.parametrize("max_T", [4, 8])
@pytest.mark.parametrize("batch_size", [4, 16])
def test_gdn_decode_bf16_state_fla_scatter_random_slots(
    batch_size: int,
    max_T: int,
):
    """FLA scatter with NON-CONTIGUOUS, scattered slot indices.

    vLLM's free-list allocator can hand out scattered slots (not the
    monotonic `arange(B, B+B*T)` pattern every other test uses). This
    test feeds a random permutation of free slots to confirm the kernel
    treats each `ssm_state_indices[i, t]` independently.
    """
    if not GDN_DECODE_BF16_STATE_AVAILABLE:
        pytest.skip("BF16 state kernel not available")
    _skip_if_not_sm90_or_later()

    from flashinfer.gdn_kernels.gdn_decode_bf16_state import gated_delta_rule_mtp

    torch.manual_seed(7)
    device = torch.device("cuda")
    B, T = batch_size, max_T
    H, HV, K, V = 16, 64, 128, 128

    # Generous pool so we can scatter slots arbitrarily.
    pool_size = max(64, B * (T + 1) * 2)
    pool_init = (
        torch.randn(pool_size, HV, V, K, dtype=torch.bfloat16, device=device) * 0.1
    )
    h0_idx = torch.arange(B, dtype=torch.int32, device=device)

    # Random permutation: pick B*T fresh slots (disjoint from h0_idx), shuffle.
    free_slots = torch.randperm(pool_size - B, device=device)[: B * T] + B
    ssm_idx = free_slots.to(torch.int32).reshape(B, T)
    # Sanity: all distinct, all >= B, all < pool_size.
    assert ssm_idx.unique().numel() == B * T
    assert ssm_idx.min().item() >= B
    assert ssm_idx.max().item() < pool_size

    q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device=device)
    k = torch.randn(B, T, H, K, dtype=torch.bfloat16, device=device)
    v = torch.randn(B, T, HV, V, dtype=torch.bfloat16, device=device)
    a = torch.randn(B, T, HV, dtype=torch.bfloat16, device=device)
    b = torch.randn(B, T, HV, dtype=torch.bfloat16, device=device)
    A_log = torch.randn(HV, dtype=torch.float32, device=device)
    dt_bias = torch.randn(HV, dtype=torch.float32, device=device)
    ibuf = torch.zeros(B, T, HV, V, K, dtype=torch.bfloat16, device=device)

    common = dict(
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        b=b,
        initial_state_indices=h0_idx,
        use_qk_l2norm_in_kernel=True,
        scale=K**-0.5,
    )

    # Dense reference (same input).
    pool_dense = pool_init.clone()
    gated_delta_rule_mtp(
        **common, initial_state_source=pool_dense, intermediate_states_buffer=ibuf
    )

    # FLA with scattered slots.
    pool_fla = pool_init.clone()
    gated_delta_rule_mtp(
        **common, initial_state_source=pool_fla, ssm_state_indices=ssm_idx
    )
    torch.cuda.synchronize()

    # Each scattered slot must hold exactly h_{t+1} for its (i, t).
    for i in range(B):
        for t in range(T):
            slot = int(ssm_idx[i, t].item())
            diff = (pool_fla[slot].float() - ibuf[i, t].float()).abs().max().item()
            assert diff == 0, (
                f"scattered slot mismatch at (i={i}, t={t}, slot={slot}): {diff}"
            )

    # All unused slots (not h0_idx, not ssm_idx) must equal pool_init.
    used = set(h0_idx.tolist()) | set(ssm_idx.flatten().tolist())
    for s in range(pool_size):
        if s in used:
            continue
        diff = (pool_fla[s] - pool_init[s]).abs().max().item()
        assert diff == 0, f"unused slot {s} clobbered: {diff}"


@pytest.mark.parametrize("max_T", [2, 4, 8])
@pytest.mark.parametrize("batch_size", [1, 4, 16, 64, 128])
def test_gdn_decode_fp32_state_fla_scatter_vs_dense(
    batch_size: int,
    max_T: int,
):
    """FP32 FLA bit-equivalence with the dense `intermediate_states_buffer`
    path. Exercises the Int64 widen on `fla_idx` — at B=128/T=8 the max
    flat_idx is 73,727 which would overflow Int32 byte-addressing without
    the widen (the smoke test that motivated the fix is now a regression
    test).
    """
    _skip_if_not_sm90_or_later()
    from flashinfer.gdn_decode import gated_delta_rule_mtp as fp32_mtp

    torch.manual_seed(0)
    device = torch.device("cuda")
    B, T = batch_size, max_T
    H, HV, K, V = 16, 64, 128, 128

    pool_size = B * (T + 1)
    pool_init = (
        torch.randn(pool_size, HV, V, K, dtype=torch.float32, device=device) * 0.1
    )
    h0_idx = torch.arange(B, dtype=torch.int32, device=device)
    ssm_idx = torch.arange(B, B + B * T, dtype=torch.int32, device=device).reshape(B, T)

    q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device=device)
    k = torch.randn(B, T, H, K, dtype=torch.bfloat16, device=device)
    v = torch.randn(B, T, HV, V, dtype=torch.bfloat16, device=device)
    a = torch.randn(B, T, HV, dtype=torch.bfloat16, device=device)
    b = torch.randn(B, T, HV, dtype=torch.bfloat16, device=device)
    A_log = torch.randn(HV, dtype=torch.float32, device=device)
    dt_bias = torch.randn(HV, dtype=torch.float32, device=device)

    common = dict(
        q=q,
        k=k,
        v=v,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        initial_state_indices=h0_idx,
        scale=K**-0.5,
        disable_state_update=False,
        use_qk_l2norm=True,
    )

    # Dense reference.
    pool_dense = pool_init.clone()
    ibuf = torch.zeros(B, T, HV, V, K, dtype=torch.float32, device=device)
    fp32_mtp(**common, initial_state=pool_dense, intermediate_states_buffer=ibuf)

    # FLA scatter.
    pool_fla = pool_init.clone()
    fp32_mtp(**common, initial_state=pool_fla, ssm_state_indices=ssm_idx)
    torch.cuda.synchronize()

    # Per-token states must match bit-exactly.
    for i in range(B):
        for t in range(T):
            slot = int(ssm_idx[i, t].item())
            diff = (pool_fla[slot].float() - ibuf[i, t].float()).abs().max().item()
            assert diff == 0, (
                f"FP32 FLA mismatch at (i={i}, t={t}, slot={slot}): {diff}"
            )
