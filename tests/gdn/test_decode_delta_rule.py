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


def _skip_if_not_sm90_or_later():
    """Skip test if not Hopper (SM90+) or Blackwell (SM100+) architecture."""
    cc = get_compute_capability(torch.device("cuda"))
    if cc[0] not in [9, 10, 11, 12]:
        pytest.skip(f"GDN decode requires SM90+ or SM100+, but got SM{cc[0]}{cc[1]}")


# ============================================================================
# Test decode kernel with pretranspose version ([B*HV, V, K])
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
        dt_bias = torch.randn(num_sab_heads, dtype=dtype_torch, device=device) * 0.1

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

    # Reference implementation (remove T=1 dimension)
    # Now passes raw GDN parameters, will compute g and beta internally
    # Reference uses [B, HV, K, V] state (matches Triton)
    ref_o, ref_state = decode_delta_rule(
        q.squeeze(1).float(),  # [B, 1, H, K] -> [B, H, K]
        k.squeeze(1).float(),
        v.squeeze(1).float(),
        input_state_ref,  # Use [B, HV, K, V] state for reference
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
        dt_bias = torch.randn(num_sab_heads, dtype=dtype_torch, device=device) * 0.1

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

    # Reference implementation (remove T=1 dimension)
    # Reference uses [B, HV, K, V] state (matches both Triton and nontranspose kernel)
    ref_o, ref_state = decode_delta_rule(
        q.squeeze(1).float(),  # [B, 1, H, K] -> [B, H, K]
        k.squeeze(1).float(),
        v.squeeze(1).float(),
        input_state,  # Use [B, HV, K, V] state for reference
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
# Test verify kernel with MTP version (Multiple Token Processing)
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
        disable_state_update=True,  # Don't update state for testing
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

        torch.testing.assert_close(
            intermediate_states_kernel.float(),
            intermediate_states_ref.float(),
            atol=atol_s,
            rtol=rtol_s,
            msg=f"Intermediate states mismatch for MTP kernel (B={B}, T={T}, dtype={dtype})",
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

    print("\n✅ All smoke tests passed!")
    print("\nTo run full test suite:")
    print(
        "  PRETRANSPOSE: pytest test_decode_delta_rule.py::test_decode_kernel_basic_pretranspose -v"
    )
    print(
        "  NONTRANSPOSE: pytest test_decode_delta_rule.py::test_decode_kernel_basic_nontranspose -v"
    )
    print(
        "  MTP (VERIFY):  pytest test_decode_delta_rule.py::test_verify_kernel_mtp -v"
    )
    print("  ALL: pytest test_decode_delta_rule.py -v")
