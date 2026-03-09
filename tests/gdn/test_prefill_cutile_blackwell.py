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

Accuracy tests for cuTile GDN prefill kernel on Blackwell (SM100, B200).
Compares cuTile vs FLA Triton GDN baseline. Both use [B, T, H, K] format.

Usage:
    pytest tests/gdn/test_prefill_cutile_blackwell.py -v
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

def get_compute_capability(device):
    return torch.cuda.get_device_capability(device)

# ---------------------------------------------------------------------------
# cuTile import: from flashinfer.gdn_kernels.cutile_gdn_prefill
# ---------------------------------------------------------------------------
try:
    from flashinfer.gdn_kernels.cutile_gdn_prefill import chunk_gated_delta_rule_cutile as _ct_fwd
    _CUTILE_AVAILABLE = True
except ImportError:
    _CUTILE_AVAILABLE = False

# ---------------------------------------------------------------------------
# FLA Triton baseline import
# ---------------------------------------------------------------------------
try:
    from fla.ops.gated_delta_rule.chunk import (
        chunk_gated_delta_rule_fwd as _fla_fwd,
    )
    _FLA_AVAILABLE = True
except ImportError:
    _FLA_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _skip_if_not_sm100():
    """Skip test if not SM100 architecture (Blackwell B200)."""
    cc = get_compute_capability(torch.device("cuda"))
    if cc[0] != 10:
        pytest.skip(
            f"cuTile GDN prefill requires SM100 (Blackwell), but got SM{cc[0]}{cc[1]}"
        )


def _skip_if_cutile_unavailable():
    if not _CUTILE_AVAILABLE:
        pytest.skip(
            "cutile_gdn_prefill not available. Set CUTILE_ROOT to sglang repo path."
        )


def _skip_if_fla_unavailable():
    if not _FLA_AVAILABLE:
        pytest.skip("fla (flash-linear-attention) not installed.")


# ---------------------------------------------------------------------------
# Core test function
# ---------------------------------------------------------------------------

def _test_cutile_vs_fla(
    B: int,
    T: int,
    H: int,
    K: int,
    V: int,
    dtype: torch.dtype,
    use_initial_state: bool = False,
    seed: int = 42,
):
    """Compare cuTile and FLA Triton outputs for one config."""
    _skip_if_not_sm100()
    _skip_if_cutile_unavailable()
    _skip_if_fla_unavailable()

    assert T % 64 == 0, "T must be a multiple of block size 64"

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    device = torch.device("cuda")

    # Inputs in batch-first [B, T, H, K/V] format
    q = torch.randn(B, T, H, K, dtype=dtype, device=device)
    k = torch.randn(B, T, H, K, dtype=dtype, device=device)
    v = torch.randn(B, T, H, V, dtype=dtype, device=device)
    # g: logsigmoid gate values (float32 for numerical stability)
    g = F.logsigmoid(torch.randn(B, T, H, device=device, dtype=torch.float32))
    beta = torch.sigmoid(torch.randn(B, T, H, device=device, dtype=torch.float32))
    scale = K ** -0.5

    # L2-normalize q and k (same preprocessing used at inference)
    q_n = F.normalize(q.float(), p=2, dim=-1).to(dtype)
    k_n = F.normalize(k.float(), p=2, dim=-1).to(dtype)

    # Always provide initial state tensors; use zeros when not testing initial state
    # (cuTile ct.launch can't accept None tensors; zeros == no initial state)
    if use_initial_state:
        h0 = torch.randn(B, H, K, V, dtype=dtype, device=device)
    else:
        h0 = torch.zeros(B, H, K, V, dtype=dtype, device=device)
    idx = torch.arange(B, dtype=torch.int32, device=device)

    # ---- cuTile ----
    out_ct, _, _ = _ct_fwd(
        q_n, k_n, v, g, beta, scale,
        h0.clone(), idx,
        use_qk_l2norm_in_kernel=False,
    )
    torch.cuda.synchronize()

    # ---- FLA Triton ----
    # FLA accepts None (no initial state) or a tensor; use same h0 for parity
    fla_h0 = h0.clone() if use_initial_state else None
    ret_fla = _fla_fwd(
        q_n, k_n, v, g, beta, scale,
        initial_state=fla_h0,
        output_final_state=False,
    )
    # FLA returns (g_cumsum, o, v_new, final_state) — output is at index 1
    out_fla = ret_fla[1] if isinstance(ret_fla, (tuple, list)) and len(ret_fla) >= 2 else ret_fla
    torch.cuda.synchronize()

    # Both outputs are [B, T, H, V]
    if dtype == torch.bfloat16:
        atol, rtol = 1e-2, 1e-2
    else:
        atol, rtol = 1e-3, 1e-3

    torch.testing.assert_close(out_ct, out_fla, atol=atol, rtol=rtol,
                               msg=f"Mismatch for B={B},T={T},H={H},K={K},V={V}")


# ---------------------------------------------------------------------------
# Parametrized tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("use_initial_state", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("B,T,H,K,V", [
    # Small configs (sanity check)
    (1, 64,  4, 64,  64),
    (1, 128, 4, 64,  64),
    (2, 128, 4, 64,  64),
    # Medium configs
    (1, 512,  4, 128, 128),
    (2, 512,  4, 128, 128),
    (4, 512,  4, 128, 128),
    # Standard GDN configs (K=V=256, H=4)
    (1, 512,  4, 256, 256),
    (1, 1024, 4, 256, 256),
    (2, 1024, 4, 256, 256),
    (4, 1024, 4, 256, 256),
    (8, 1024, 4, 256, 256),
    # Large configs
    (1,  2048, 4, 256, 256),
    (4,  2048, 4, 256, 256),
    (8,  2048, 4, 256, 256),
    (16, 1024, 4, 256, 256),
])
def test_cutile_vs_fla_accuracy(B, T, H, K, V, dtype, use_initial_state):
    """cuTile GDN prefill output matches FLA Triton baseline on Blackwell."""
    _test_cutile_vs_fla(B, T, H, K, V, dtype, use_initial_state)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("B,T,H,K,V", [
    # Key production-scale configs targeting >1ms total kernel time
    (4,  4096, 4, 256, 256),
    (8,  2048, 4, 256, 256),
    (16, 1024, 4, 256, 256),
])
def test_cutile_large_configs(B, T, H, K, V, dtype):
    """Accuracy on large Blackwell workloads (>1ms kernel time)."""
    _test_cutile_vs_fla(B, T, H, K, V, dtype, use_initial_state=True)
