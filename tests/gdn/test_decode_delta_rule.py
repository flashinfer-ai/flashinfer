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

    # Call BF16 state kernel (T=1 uses gated_delta_rule, T>1 uses MTP)
    our_state = input_state_kernel.clone()
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

    # Direct improved kernel (T=1 uses gdn_decode_bf16_state, T>1 uses MTP variant)
    kernel_fn = gdn_decode_bf16_state if seq_len == 1 else gdn_decode_bf16_state_mtp
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

    our_state = input_state_kernel.clone()
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
    [(16, 16, 32)],
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
        # Stack ref: [B, T, HV, K, V], transpose to [B, T, HV, V, K] for comparison
        ref_inter = torch.stack(ref_intermediate_states, dim=1)  # [B, T, HV, K, V]
        ref_inter_transposed = ref_inter.transpose(
            -2, -1
        ).contiguous()  # [B, T, HV, V, K]

        atol_s = 0.02
        rtol_s = 0.01
        torch.testing.assert_close(
            intermediate_states_buffer.float(),
            ref_inter_transposed.float(),
            atol=atol_s,
            rtol=rtol_s,
            msg=f"Intermediate states mismatch for MTP BF16 state kernel (B={batch_size}, T={seq_len})",
        )

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
