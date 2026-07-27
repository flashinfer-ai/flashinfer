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

Regression test for vLLM-style padded pool layouts on the bf16 GDN decode
path. vLLM packs conv-state and ssm-state on the same allocation page, so
the SSM pool tensor handed to flashinfer has a per-slot stride larger than
HV*V*K. Before the fix, the bf16 wrapper unconditionally executed
``initial_state_source.reshape(pool_size * HV, V, K)``, which silently
materialized a ``.contiguous()`` clone of the entire 8 GiB pool every call.

After the fix, the kernel reads the 4D ``[pool_size, HV, V, K]`` pool
directly via cute strides — no reshape, no clone. This test verifies
correctness: numerical results from the strided pool must match those from
a tightly-allocated pool with the same data.
"""

from __future__ import annotations

import random

import pytest
import torch

from flashinfer.gdn_decode import gated_delta_rule_decode_pretranspose
from flashinfer.utils import get_compute_capability


def _skip_if_not_sm90_or_later() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    cc = get_compute_capability(torch.device("cuda"))
    if cc[0] not in [9, 10, 11, 12]:
        pytest.skip(f"GDN decode requires SM90+ or SM100+, but got SM{cc[0]}{cc[1]}")


@pytest.mark.parametrize(
    "B,batch_label",
    [
        (1, "mtp_ilp4"),  # work_units = B*HV = 32 < 128 → mtp_ilp4 path
        (4, "mtp_ilp4"),  # 128, but tile_v=32 forced None at T=1 → mtp_ilp4 fallback
        (16, "wide_vec"),  # 512 → wide_vec tile_v=64
        (32, "wide_vec"),  # 1024 → wide_vec tile_v=128
    ],
)
@pytest.mark.parametrize(
    "pool_layout",
    [
        "tight",  # plain torch.empty (no padding) — sanity baseline
        "padded",  # vLLM-style: stride[0] = HV*V*K + 16384 (conv padding)
    ],
)
def test_decode_bf16_pool_strided(B: int, batch_label: str, pool_layout: str) -> None:
    """Verify bf16 decode is correct when the SSM pool has a non-contiguous
    per-slot stride (vLLM's actual production layout)."""
    _skip_if_not_sm90_or_later()

    seed = 20260505 + B + (1 if pool_layout == "padded" else 0)
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    T, H, HV, K, V = 1, 16, 32, 128, 128
    pool_size = max(8, B * 2)
    device = torch.device("cuda")
    state_dtype = torch.bfloat16
    qkv_dtype = torch.bfloat16

    with device:
        q = torch.randn(B, T, H, K, dtype=qkv_dtype) * 0.05
        k = torch.nn.functional.normalize(
            torch.randn(B, T, H, K, dtype=qkv_dtype), p=2.0, dim=-1
        )
        v = torch.randn(B, T, HV, V, dtype=qkv_dtype) * 0.05
        A_log = torch.randn(HV, dtype=torch.float32) * 0.1
        dt_bias = torch.randn(HV, dtype=torch.bfloat16) * 0.1
        a = torch.randn(B, T, HV, dtype=qkv_dtype) * 0.05
        b = torch.randn(B, T, HV, dtype=qkv_dtype) * 0.05

        # Reference data shared across both pools.
        ref_data = torch.randn(pool_size, HV, V, K, dtype=state_dtype) * 0.1

        if pool_layout == "padded":
            # vLLM-style: stride[0] = HV*V*K + 16384 elts (conv padding).
            inner = HV * V * K
            pad_elts = 16384
            assert pad_elts % (V * K) == 0, "pad must be HV-row aligned for this test"
            pad_hv_rows = pad_elts // (V * K)
            big = torch.empty(pool_size, HV + pad_hv_rows, V, K, dtype=state_dtype)
            pool = big[:, :HV, :, :]
            pool.copy_(ref_data)
            assert pool.stride() == (inner + pad_elts, V * K, K, 1)
            assert not pool.is_contiguous()
            pool._owner = big  # keep allocation alive
        else:
            pool = torch.empty(pool_size, HV, V, K, dtype=state_dtype)
            pool.copy_(ref_data)
            assert pool.is_contiguous()

        # Spread B reads across the pool so we exercise non-trivial cache_idx.
        indices = (
            torch.arange(B, dtype=torch.int32, device=device) * 2 + 1
        ) % pool_size

        # Path under test: pool with possibly non-contiguous slot stride.
        # Use a fresh allocation backing the same data so updated-slot diffs
        # don't alias the reference data.
        if pool_layout == "padded":
            pad_elts2 = 16384
            pad_hv_rows2 = pad_elts2 // (V * K)
            big2 = torch.empty(pool_size, HV + pad_hv_rows2, V, K, dtype=state_dtype)
            pool_under_test = big2[:, :HV, :, :]
            pool_under_test.copy_(ref_data)
            pool_under_test._owner = big2
            assert not pool_under_test.is_contiguous()
        else:
            pool_under_test = ref_data.clone().contiguous()

    out_under_test, _ = gated_delta_rule_decode_pretranspose(
        q=q,
        k=k,
        v=v,
        state=None,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        scale=K**-0.5,
        use_qk_l2norm=True,
        initial_state=pool_under_test,
        initial_state_indices=indices,
    )
    torch.cuda.synchronize()

    # Reference path: pass the equivalent tight pool to the same kernel.
    pool_reference = ref_data.clone().contiguous()
    out_reference, _ = gated_delta_rule_decode_pretranspose(
        q=q,
        k=k,
        v=v,
        state=None,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        scale=K**-0.5,
        use_qk_l2norm=True,
        initial_state=pool_reference,
        initial_state_indices=indices,
    )
    torch.cuda.synchronize()

    atol, rtol = 5e-3, 5e-3
    # Output must match.
    torch.testing.assert_close(out_under_test, out_reference, atol=atol, rtol=rtol)

    # Updated slots in the pool must match.
    torch.testing.assert_close(
        pool_under_test[indices], pool_reference[indices], atol=atol, rtol=rtol
    )

    # Untouched slots must be untouched (proves we didn't accidentally clone
    # the whole pool, which would mask cross-slot aliasing).
    untouched_mask = torch.ones(pool_size, dtype=torch.bool, device=device)
    untouched_mask[indices] = False
    torch.testing.assert_close(
        pool_under_test[untouched_mask],
        ref_data[untouched_mask],
        atol=0.0,
        rtol=0.0,
    )
