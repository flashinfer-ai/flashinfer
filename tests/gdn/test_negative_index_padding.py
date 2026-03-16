# Copyright (c) 2025 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for negative-index padding in BF16 decode and MTP kernels."""

from __future__ import annotations

import os
import random

import pytest
import torch

try:
    from .reference_delta_rule import decode_delta_rule, verify_delta_rule
except ImportError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from reference_delta_rule import decode_delta_rule, verify_delta_rule

from flashinfer.gdn_decode import gated_delta_rule_mtp
from flashinfer.utils import get_compute_capability

try:
    from flashinfer.gdn_kernels.gdn_decode_bf16_state import (
        gated_delta_rule as gdn_decode_klast_bf16_state,
    )

    GDN_DECODE_KLAST_BF16_STATE_AVAILABLE = True
except ImportError:
    GDN_DECODE_KLAST_BF16_STATE_AVAILABLE = False


def _skip_if_not_sm90_or_later():
    cc = get_compute_capability(torch.device("cuda"))
    if cc[0] not in [9, 10, 11, 12]:
        pytest.skip(f"GDN decode requires SM90+, but got SM{cc[0]}{cc[1]}")


def _test_bf16_decode_negative_indices(
    B: int,
    T: int,
    num_q_heads: int = 16,
    num_k_heads: int = 16,
    num_v_heads: int = 32,
    head_size: int = 128,
    scale: float = 1.0,
    pool_multiplier: int = 2,
    padding_fraction: float = 0.5,
    seed: int = 42,
):
    """BF16 decode kernel must zero output for negative-index (padding) slots
    and correctly process valid slots."""
    _skip_if_not_sm90_or_later()
    if not GDN_DECODE_KLAST_BF16_STATE_AVAILABLE:
        pytest.skip("BF16 state kernel not available")

    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    K = head_size
    V = head_size
    HV = num_v_heads
    pool_size = B * pool_multiplier
    device = torch.device("cuda")

    with device:
        q = torch.randn(B, T, num_q_heads, K, dtype=torch.bfloat16)
        k = torch.randn(B, T, num_k_heads, K, dtype=torch.bfloat16)
        v = torch.randn(B, T, HV, V, dtype=torch.bfloat16)
        a = torch.randn(B, T, HV, dtype=torch.bfloat16) * 0.1
        b = torch.randn(B, T, HV, dtype=torch.bfloat16)
        A_log = torch.randn(HV, dtype=torch.float32) * 0.1
        dt_bias = torch.randn(HV, dtype=torch.float32) * 0.1
        pool = torch.randn(pool_size, HV, V, K, dtype=torch.bfloat16)

        indices = torch.arange(B, dtype=torch.int32, device=device) % pool_size
        mask = torch.rand(B, device=device) < padding_fraction
        if B >= 2:
            mask[0] = False
            mask[-1] = True
        indices[mask] = -1

        valid_mask = indices >= 0
        num_valid = valid_mask.sum().item()
        if num_valid > 0:
            valid_slots = torch.randperm(pool_size, device=device)[:num_valid].to(
                torch.int32
            )
            indices[valid_mask] = valid_slots

    pool_under_test = pool.clone()
    output = gdn_decode_klast_bf16_state(
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        b=b,
        initial_state_source=pool_under_test,
        initial_state_indices=indices,
        scale=scale,
        use_qk_l2norm_in_kernel=True,
    )
    torch.cuda.synchronize()

    invalid_mask = indices < 0
    if invalid_mask.any():
        padded_output = output[invalid_mask]
        assert torch.all(padded_output == 0), (
            f"Padding slots must produce zero output, "
            f"but got max abs = {padded_output.abs().max().item()}"
        )

    valid_indices_local = torch.where(valid_mask)[0].cpu().numpy()
    pool_snapshot = pool.clone()

    for i in valid_indices_local:
        pool_idx = indices[i].item()
        for t in range(T):
            ref_state = (
                pool_snapshot[pool_idx].float().transpose(-2, -1).contiguous()
            )
            ref_o, ref_s = decode_delta_rule(
                q[i, t].unsqueeze(0).float(),
                k[i, t].unsqueeze(0).float(),
                v[i, t].unsqueeze(0).float(),
                ref_state.unsqueeze(0),
                A_log=A_log,
                a=a[i, t].unsqueeze(0),
                dt_bias=dt_bias.float(),
                b=b[i, t].unsqueeze(0),
                scale_factor=scale,
                use_l2_norm=True,
            )
            pool_snapshot[pool_idx] = ref_s.squeeze(0).transpose(-2, -1).to(
                pool_snapshot.dtype
            )

            torch.testing.assert_close(
                output[i, t].float(),
                ref_o.squeeze(0).to(device),
                atol=5e-2,
                rtol=5e-2,
            )

        torch.testing.assert_close(
            pool_under_test[pool_idx].float(),
            pool_snapshot[pool_idx].float(),
            atol=5e-2,
            rtol=5e-2,
        )

    used_pool_indices = indices[valid_mask].unique()
    touched = torch.zeros(pool_size, dtype=torch.bool, device=device)
    if len(used_pool_indices) > 0:
        touched[used_pool_indices.long()] = True
    torch.testing.assert_close(
        pool_under_test[~touched], pool[~touched], atol=0.0, rtol=0.0
    )


def _test_bf16_decode_all_padding(
    B: int,
    T: int,
    num_v_heads: int = 32,
    head_size: int = 128,
    seed: int = 42,
):
    """When ALL indices are negative, output must be all zeros and pool unchanged."""
    _skip_if_not_sm90_or_later()
    if not GDN_DECODE_KLAST_BF16_STATE_AVAILABLE:
        pytest.skip("BF16 state kernel not available")

    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    K = head_size
    V = head_size
    HV = num_v_heads
    pool_size = B * 2
    device = torch.device("cuda")

    with device:
        q = torch.randn(B, T, 16, K, dtype=torch.bfloat16)
        k = torch.randn(B, T, 16, K, dtype=torch.bfloat16)
        v = torch.randn(B, T, HV, V, dtype=torch.bfloat16)
        a = torch.randn(B, T, HV, dtype=torch.bfloat16) * 0.1
        b = torch.randn(B, T, HV, dtype=torch.bfloat16)
        A_log = torch.randn(HV, dtype=torch.float32) * 0.1
        dt_bias = torch.randn(HV, dtype=torch.float32) * 0.1
        pool = torch.randn(pool_size, HV, V, K, dtype=torch.bfloat16)
        indices = torch.full((B,), -1, dtype=torch.int32, device=device)

    pool_under_test = pool.clone()
    output = gdn_decode_klast_bf16_state(
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        b=b,
        initial_state_source=pool_under_test,
        initial_state_indices=indices,
        use_qk_l2norm_in_kernel=True,
    )
    torch.cuda.synchronize()

    assert torch.all(output == 0), (
        f"All-padding batch must produce zero output, "
        f"but got max abs = {output.abs().max().item()}"
    )
    torch.testing.assert_close(
        pool_under_test, pool, atol=0.0, rtol=0.0,
        msg="All-padding batch must not modify any pool state",
    )


@pytest.mark.parametrize("B", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("T", [1, 2, 3, 4])
def test_bf16_decode_negative_indices(
    B: int,
    T: int,
    seed: int = int(os.environ.get("SEED", "42")),
):
    _test_bf16_decode_negative_indices(B=B, T=T, seed=seed)


@pytest.mark.parametrize("B", [1, 4, 8])
@pytest.mark.parametrize("T", [1, 2, 4])
def test_bf16_decode_all_padding(
    B: int,
    T: int,
    seed: int = int(os.environ.get("SEED", "42")),
):
    _test_bf16_decode_all_padding(B=B, T=T, seed=seed)


def _test_mtp_negative_indices(
    B: int,
    T: int,
    num_q_heads: int = 16,
    num_k_heads: int = 16,
    num_v_heads: int = 32,
    head_size: int = 128,
    scale: float = 1.0,
    pool_multiplier: int = 2,
    padding_fraction: float = 0.5,
    seed: int = 42,
):
    """MTP kernel must zero output for negative-index (padding) slots."""
    _skip_if_not_sm90_or_later()

    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    K = head_size
    V = head_size
    H = num_q_heads
    HV = num_v_heads
    pool_size = B * pool_multiplier
    device = torch.device("cuda")

    with device:
        q = torch.randn(B, T, H, K, dtype=torch.bfloat16)
        k = torch.randn(B, T, H, K, dtype=torch.bfloat16)
        v = torch.randn(B, T, HV, V, dtype=torch.bfloat16)
        a = torch.randn(B, T, HV, dtype=torch.bfloat16) * 0.1
        b = torch.randn(B, T, HV, dtype=torch.bfloat16)
        A_log = torch.randn(HV, dtype=torch.float32) * 0.1
        dt_bias = torch.randn(HV, dtype=torch.float32) * 0.1
        initial_state = torch.randn(pool_size, HV, V, K, dtype=torch.float32)

        indices = torch.arange(B, dtype=torch.int32, device=device) % pool_size
        mask = torch.rand(B, device=device) < padding_fraction
        if B >= 2:
            mask[0] = False
            mask[-1] = True
        indices[mask] = -1

        valid_mask = indices >= 0
        num_valid = valid_mask.sum().item()
        if num_valid > 0:
            valid_slots = torch.randperm(pool_size, device=device)[:num_valid].to(
                torch.int32
            )
            indices[valid_mask] = valid_slots

    state_under_test = initial_state.clone()

    output_prealloc = torch.ones(B, T, HV, V, dtype=torch.bfloat16, device=device)
    output, _ = gated_delta_rule_mtp(
        q=q,
        k=k,
        v=v,
        initial_state=state_under_test,
        initial_state_indices=indices,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        scale=scale,
        output=output_prealloc,
        use_qk_l2norm=True,
    )
    torch.cuda.synchronize()

    invalid_mask = indices < 0
    if invalid_mask.any():
        padded_output = output[invalid_mask]
        assert torch.all(padded_output == 0), (
            f"MTP padding slots must produce zero output, "
            f"but got max abs = {padded_output.abs().max().item()}"
        )

    valid_indices_local = torch.where(valid_mask)[0].cpu().numpy()
    state_snapshot = initial_state.clone()

    for i in valid_indices_local:
        pool_idx = indices[i].item()
        for t in range(T):
            ref_state = (
                state_snapshot[pool_idx].float().transpose(-2, -1).contiguous()
            )
            ref_o, ref_s = decode_delta_rule(
                q[i, t].unsqueeze(0).float(),
                k[i, t].unsqueeze(0).float(),
                v[i, t].unsqueeze(0).float(),
                ref_state.unsqueeze(0),
                A_log=A_log,
                a=a[i, t].unsqueeze(0),
                dt_bias=dt_bias.float(),
                b=b[i, t].unsqueeze(0),
                scale_factor=scale,
                use_l2_norm=True,
            )
            state_snapshot[pool_idx] = ref_s.squeeze(0).transpose(-2, -1)

            torch.testing.assert_close(
                output[i, t].float(),
                ref_o.squeeze(0).to(device),
                atol=5e-2,
                rtol=5e-2,
            )

        torch.testing.assert_close(
            state_under_test[pool_idx].float(),
            state_snapshot[pool_idx].float(),
            atol=5e-2,
            rtol=5e-2,
        )


def _test_mtp_all_padding(
    B: int,
    T: int,
    num_v_heads: int = 32,
    head_size: int = 128,
    seed: int = 42,
):
    """When ALL indices are negative, MTP output must be all zeros."""
    _skip_if_not_sm90_or_later()

    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    K = head_size
    V = head_size
    HV = num_v_heads
    pool_size = B * 2
    device = torch.device("cuda")

    with device:
        q = torch.randn(B, T, 16, K, dtype=torch.bfloat16)
        k = torch.randn(B, T, 16, K, dtype=torch.bfloat16)
        v = torch.randn(B, T, HV, V, dtype=torch.bfloat16)
        a = torch.randn(B, T, HV, dtype=torch.bfloat16) * 0.1
        b = torch.randn(B, T, HV, dtype=torch.bfloat16)
        A_log = torch.randn(HV, dtype=torch.float32) * 0.1
        dt_bias = torch.randn(HV, dtype=torch.float32) * 0.1
        initial_state = torch.randn(pool_size, HV, V, K, dtype=torch.float32)
        indices = torch.full((B,), -1, dtype=torch.int32, device=device)

    state_under_test = initial_state.clone()
    output_prealloc = torch.ones(B, T, HV, V, dtype=torch.bfloat16, device=device)
    output, _ = gated_delta_rule_mtp(
        q=q,
        k=k,
        v=v,
        initial_state=state_under_test,
        initial_state_indices=indices,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        output=output_prealloc,
        use_qk_l2norm=True,
    )
    torch.cuda.synchronize()

    assert torch.all(output == 0), (
        f"MTP all-padding batch must produce zero output, "
        f"but got max abs = {output.abs().max().item()}"
    )
    torch.testing.assert_close(
        state_under_test, initial_state, atol=0.0, rtol=0.0,
        msg="MTP all-padding batch must not modify any pool state",
    )


@pytest.mark.parametrize("B", [4, 8, 16])
@pytest.mark.parametrize("T", [2, 3, 4])
def test_mtp_negative_indices(
    B: int,
    T: int,
    seed: int = int(os.environ.get("SEED", "42")),
):
    _test_mtp_negative_indices(B=B, T=T, seed=seed)


@pytest.mark.parametrize("B", [4, 8])
@pytest.mark.parametrize("T", [2, 4])
def test_mtp_all_padding(
    B: int,
    T: int,
    seed: int = int(os.environ.get("SEED", "42")),
):
    _test_mtp_all_padding(B=B, T=T, seed=seed)
