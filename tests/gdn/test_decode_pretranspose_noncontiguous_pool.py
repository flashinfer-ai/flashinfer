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


@pytest.mark.parametrize("page_gap", [2, 3])
def test_decode_pretranspose_pool_noncontiguous_state(page_gap: int) -> None:
    _skip_if_not_sm90_or_later()

    seed = 20260309 + page_gap
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    B, T, H, HV, K, V = 8, 1, 16, 32, 128, 128
    pool_size = B * 3
    device = torch.device("cuda")
    qkv_dtype = torch.bfloat16

    with device:
        q = torch.randn(B, T, H, K, dtype=qkv_dtype)
        k = torch.nn.functional.normalize(
            torch.randn(B, T, H, K, dtype=qkv_dtype), p=2.0, dim=-1
        )
        v = torch.randn(B, T, HV, V, dtype=qkv_dtype)

        A_log = torch.randn(HV, dtype=torch.float32) * 0.1
        dt_bias = torch.randn(HV, dtype=torch.float32) * 0.1
        a = torch.randn(B, T, HV, dtype=qkv_dtype) * 0.1
        b = torch.randn(B, T, HV, dtype=qkv_dtype)

        # Build a non-contiguous [pool, HV, V, K] view with page stride on dim-0.
        pool_storage = torch.randn(pool_size, page_gap, HV, V, K, dtype=torch.float32)
        pool_source = pool_storage[:, page_gap - 1]
        assert not pool_source.is_contiguous()

        indices = (torch.arange(B, dtype=torch.int32, device=device) * 2) % pool_size

    # Pool path under test: initial_state is a non-contiguous view.
    pool_under_test_storage = pool_storage.clone()
    pool_under_test = pool_under_test_storage[:, page_gap - 1]
    out_pool, _ = gated_delta_rule_decode_pretranspose(
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
        initial_state_indices=indices,
    )
    torch.cuda.synchronize()

    # Gather + direct-state reference path.
    gathered_state = pool_source[indices].clone()
    out_direct, updated_state = gated_delta_rule_decode_pretranspose(
        q=q,
        k=k,
        v=v,
        state=gathered_state,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        scale=1.0,
        use_qk_l2norm=True,
    )
    torch.cuda.synchronize()

    atol = 5e-3
    rtol = 5e-3
    torch.testing.assert_close(out_pool, out_direct, atol=atol, rtol=rtol)
    torch.testing.assert_close(
        pool_under_test[indices], updated_state, atol=atol, rtol=rtol
    )

    untouched = torch.ones(pool_size, dtype=torch.bool, device=device)
    untouched[indices] = False
    torch.testing.assert_close(
        pool_under_test[untouched], pool_source[untouched], atol=0.0, rtol=0.0
    )
