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

"""Tests for the unified GDN decode API (gated_delta_rule_decode_unified)."""

import random

import torch
import pytest

from flashinfer.gdn_decode import (
    gated_delta_rule_decode_unified,
    _gated_delta_rule_decode_pretranspose_impl,
    _gated_delta_rule_decode_kv_impl,
    _gated_delta_rule_mtp_impl,
)
from flashinfer.utils import get_compute_capability


def _skip_if_not_sm90_or_later():
    cc = get_compute_capability(torch.device("cuda"))
    if cc[0] not in [9, 10, 11, 12]:
        pytest.skip(f"GDN requires SM90+, got SM{cc[0]}{cc[1]}")


def _make_qkv_state_and_params(
    B: int,
    T: int,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_size: int,
    state_dtype: torch.dtype,
    q_dtype: torch.dtype,
    seed: int | None = None,
):
    """Create q, k, v, state (VK layout), A_log, a, dt_bias, b on CUDA."""
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    device = torch.device("cuda")
    q = torch.randn(B, T, num_q_heads, head_size, dtype=q_dtype, device=device) * 0.1
    k = torch.randn(B, T, num_k_heads, head_size, dtype=q_dtype, device=device) * 0.1
    k = torch.nn.functional.normalize(k, p=2.0, dim=-1)
    v = torch.randn(B, T, num_v_heads, head_size, dtype=q_dtype, device=device) * 0.1
    # VK state: [B, HV, V, K]
    state_vk = (
        torch.randn(
            B, num_v_heads, head_size, head_size, dtype=state_dtype, device=device
        )
        * 0.1
    )
    A_log = torch.randn(num_v_heads, dtype=torch.float32, device=device) * 0.1
    dt_bias = torch.randn(num_v_heads, dtype=torch.float32, device=device) * 0.1
    a = torch.randn(B, T, num_v_heads, dtype=q_dtype, device=device) * 0.1
    b = torch.randn(B, T, num_v_heads, dtype=q_dtype, device=device) * 0.1
    return q, k, v, state_vk, A_log, a, dt_bias, b


# -----------------------------------------------------------------------------
# Unified vs legacy: same inputs, compare outputs
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size", [2, 8])
@pytest.mark.parametrize("state_dtype", [torch.bfloat16])
def test_unified_vk_bf16_t1_no_pool_matches_pretranspose(
    batch_size: int, state_dtype: torch.dtype
):
    """Unified API (VK, bf16, T=1, no pool) should match gated_delta_rule_decode_pretranspose."""
    _skip_if_not_sm90_or_later()
    B, T = batch_size, 1
    num_q, num_k, num_v, head_size = 16, 16, 32, 128
    seed = 42
    q, k, v, state_vk, A_log, a, dt_bias, b = _make_qkv_state_and_params(
        B, T, num_q, num_k, num_v, head_size, state_dtype, torch.bfloat16, seed=seed
    )
    state_legacy = state_vk.clone()
    state_unified = state_vk.clone()

    out_legacy, _ = _gated_delta_rule_decode_pretranspose_impl(
        q=q,
        k=k,
        v=v,
        state=state_legacy,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        scale=None,
        use_qk_l2norm=True,
    )
    out_unified, _ = gated_delta_rule_decode_unified(
        q=q,
        k=k,
        v=v,
        state=state_unified,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        state_layout="VK",
        scale=None,
        use_qk_l2norm=True,
    )

    torch.testing.assert_close(out_unified, out_legacy, atol=1e-4, rtol=1e-3)
    torch.testing.assert_close(state_unified, state_legacy, atol=1e-4, rtol=1e-3)


@pytest.mark.parametrize("batch_size", [2, 8])
def test_unified_vk_fp32_t1_no_pool_matches_pretranspose(batch_size: int):
    """Unified API (VK, fp32, T=1, no pool) should match gated_delta_rule_decode_pretranspose."""
    _skip_if_not_sm90_or_later()
    B, T = batch_size, 1
    num_q, num_k, num_v, head_size = 16, 16, 32, 128
    seed = 43
    q, k, v, state_vk, A_log, a, dt_bias, b = _make_qkv_state_and_params(
        B, T, num_q, num_k, num_v, head_size, torch.float32, torch.bfloat16, seed=seed
    )
    state_legacy = state_vk.clone()
    state_unified = state_vk.clone()

    out_legacy, _ = _gated_delta_rule_decode_pretranspose_impl(
        q=q,
        k=k,
        v=v,
        state=state_legacy,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        scale=None,
        use_qk_l2norm=True,
    )
    out_unified, _ = gated_delta_rule_decode_unified(
        q=q,
        k=k,
        v=v,
        state=state_unified,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        state_layout="VK",
        scale=None,
        use_qk_l2norm=True,
    )

    torch.testing.assert_close(out_unified, out_legacy, atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(state_unified, state_legacy, atol=5e-3, rtol=5e-3)


@pytest.mark.parametrize("batch_size", [2, 8])
def test_unified_kv_fp32_t1_matches_decode_kv(batch_size: int):
    """Unified API (KV, fp32, T=1) should match gated_delta_rule_decode (KV layout)."""
    _skip_if_not_sm90_or_later()
    B, T = batch_size, 1
    num_q, num_k, num_v, head_size = 16, 16, 32, 128
    seed = 44
    q, k, v, state_vk, A_log, a, dt_bias, b = _make_qkv_state_and_params(
        B, T, num_q, num_k, num_v, head_size, torch.float32, torch.bfloat16, seed=seed
    )
    # KV layout: [B, HV, K, V]
    state_kv = state_vk.permute(0, 1, 3, 2).contiguous()
    state_legacy = state_kv.clone()
    state_unified = state_kv.clone()

    out_legacy, _ = _gated_delta_rule_decode_kv_impl(
        q=q,
        k=k,
        v=v,
        state=state_legacy,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        scale=None,
        use_qk_l2norm=True,
    )
    out_unified, _ = gated_delta_rule_decode_unified(
        q=q,
        k=k,
        v=v,
        state=state_unified,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        state_layout="KV",
        scale=None,
        use_qk_l2norm=True,
    )

    torch.testing.assert_close(out_unified, out_legacy, atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(state_unified, state_legacy, atol=5e-3, rtol=5e-3)


@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize("T", [2, 3])
def test_unified_vk_fp32_mtp_matches_mtp(batch_size: int, T: int):
    """Unified API (VK, fp32, T>1, pool) should match gated_delta_rule_mtp."""
    _skip_if_not_sm90_or_later()
    B = batch_size
    num_q, num_k, num_v, head_size = 16, 16, 32, 128
    pool_size = B + 2
    seed = 45
    device = torch.device("cuda")
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    q = torch.randn(B, T, num_q, head_size, dtype=torch.bfloat16, device=device) * 0.1
    k = torch.randn(B, T, num_k, head_size, dtype=torch.bfloat16, device=device) * 0.1
    k = torch.nn.functional.normalize(k, p=2.0, dim=-1)
    v = torch.randn(B, T, num_v, head_size, dtype=torch.bfloat16, device=device) * 0.1
    state_pool = (
        torch.randn(
            pool_size, num_v, head_size, head_size, dtype=torch.float32, device=device
        )
        * 0.1
    )
    state_indices = torch.arange(B, dtype=torch.int32, device=device)
    A_log = torch.randn(num_v, dtype=torch.float32, device=device) * 0.1
    dt_bias = torch.randn(num_v, dtype=torch.float32, device=device) * 0.1
    a = torch.randn(B, T, num_v, dtype=torch.bfloat16, device=device) * 0.1
    b = torch.randn(B, T, num_v, dtype=torch.bfloat16, device=device) * 0.1

    pool_legacy = state_pool.clone()
    pool_unified = state_pool.clone()

    out_legacy, _ = _gated_delta_rule_mtp_impl(
        q=q,
        k=k,
        v=v,
        initial_state=pool_legacy,
        initial_state_indices=state_indices,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        scale=None,
        disable_state_update=False,
        use_qk_l2norm=True,
    )
    out_unified, _ = gated_delta_rule_decode_unified(
        q=q,
        k=k,
        v=v,
        state=pool_unified,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        state_layout="VK",
        state_indices=state_indices,
        scale=None,
        disable_state_update=False,
        use_qk_l2norm=True,
    )

    torch.testing.assert_close(out_unified, out_legacy, atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(pool_unified, pool_legacy, atol=5e-3, rtol=5e-3)


def test_unified_vk_fp32_mtp_with_intermediate_buffer_matches_mtp():
    """Unified API with intermediate_states_buffer should match _gated_delta_rule_mtp_impl."""
    _skip_if_not_sm90_or_later()
    B, T, num_q, num_k, num_v, head_size = 4, 2, 16, 16, 32, 128
    pool_size = B + 2
    cache_steps = T
    seed = 46
    device = torch.device("cuda")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    q = torch.randn(B, T, num_q, head_size, dtype=torch.bfloat16, device=device) * 0.1
    k = torch.randn(B, T, num_k, head_size, dtype=torch.bfloat16, device=device) * 0.1
    k = torch.nn.functional.normalize(k, p=2.0, dim=-1)
    v = torch.randn(B, T, num_v, head_size, dtype=torch.bfloat16, device=device) * 0.1
    state_pool = (
        torch.randn(
            pool_size, num_v, head_size, head_size, dtype=torch.float32, device=device
        )
        * 0.1
    )
    state_indices = torch.arange(B, dtype=torch.int32, device=device)
    A_log = torch.randn(num_v, dtype=torch.float32, device=device) * 0.1
    dt_bias = torch.randn(num_v, dtype=torch.float32, device=device) * 0.1
    a = torch.randn(B, T, num_v, dtype=torch.bfloat16, device=device) * 0.1
    b = torch.randn(B, T, num_v, dtype=torch.bfloat16, device=device) * 0.1
    intermed_buf = torch.zeros(
        pool_size,
        cache_steps,
        num_v,
        head_size,
        head_size,
        dtype=torch.float32,
        device=device,
    )

    pool_legacy = state_pool.clone()
    pool_unified = state_pool.clone()
    intermed_legacy = intermed_buf.clone()
    intermed_unified = intermed_buf.clone()

    out_legacy, _ = _gated_delta_rule_mtp_impl(
        q=q,
        k=k,
        v=v,
        initial_state=pool_legacy,
        initial_state_indices=state_indices,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        scale=None,
        disable_state_update=False,
        use_qk_l2norm=True,
        intermediate_states_buffer=intermed_legacy,
    )
    out_unified, _ = gated_delta_rule_decode_unified(
        q=q,
        k=k,
        v=v,
        state=pool_unified,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        state_layout="VK",
        state_indices=state_indices,
        scale=None,
        disable_state_update=False,
        use_qk_l2norm=True,
        intermediate_states_buffer=intermed_unified,
    )

    torch.testing.assert_close(out_unified, out_legacy, atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(pool_unified, pool_legacy, atol=5e-3, rtol=5e-3)


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("pool_size", [1, 4])
def test_unified_vk_fp32_mtp_edge_pool_size_and_b1(batch_size: int, pool_size: int):
    """Edge cases: B=1 and/or pool_size=1 (unified MTP path)."""
    _skip_if_not_sm90_or_later()
    B, T = batch_size, 2
    num_q, num_k, num_v, head_size = 16, 16, 32, 128
    seed = 47
    device = torch.device("cuda")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    q = torch.randn(B, T, num_q, head_size, dtype=torch.bfloat16, device=device) * 0.1
    k = torch.randn(B, T, num_k, head_size, dtype=torch.bfloat16, device=device) * 0.1
    k = torch.nn.functional.normalize(k, p=2.0, dim=-1)
    v = torch.randn(B, T, num_v, head_size, dtype=torch.bfloat16, device=device) * 0.1
    state_pool = (
        torch.randn(
            pool_size, num_v, head_size, head_size, dtype=torch.float32, device=device
        )
        * 0.1
    )
    state_indices = torch.arange(B, dtype=torch.int32, device=device)
    A_log = torch.randn(num_v, dtype=torch.float32, device=device) * 0.1
    dt_bias = torch.randn(num_v, dtype=torch.float32, device=device) * 0.1
    a = torch.randn(B, T, num_v, dtype=torch.bfloat16, device=device) * 0.1
    b = torch.randn(B, T, num_v, dtype=torch.bfloat16, device=device) * 0.1

    pool_legacy = state_pool.clone()
    pool_unified = state_pool.clone()

    out_legacy, _ = _gated_delta_rule_mtp_impl(
        q=q,
        k=k,
        v=v,
        initial_state=pool_legacy,
        initial_state_indices=state_indices,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        scale=None,
        disable_state_update=False,
        use_qk_l2norm=True,
    )
    out_unified, _ = gated_delta_rule_decode_unified(
        q=q,
        k=k,
        v=v,
        state=pool_unified,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        state_layout="VK",
        state_indices=state_indices,
        scale=None,
        disable_state_update=False,
        use_qk_l2norm=True,
    )

    torch.testing.assert_close(out_unified, out_legacy, atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(pool_unified, pool_legacy, atol=5e-3, rtol=5e-3)


# -----------------------------------------------------------------------------
# Error handling: unsupported combinations
# -----------------------------------------------------------------------------


def test_unified_invalid_state_layout_raises():
    _skip_if_not_sm90_or_later()
    B, T, H, HV, K, V = 2, 1, 16, 32, 128, 128
    device = torch.device("cuda")
    q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device=device)
    k = torch.randn(B, T, H, K, dtype=torch.bfloat16, device=device)
    v = torch.randn(B, T, HV, V, dtype=torch.bfloat16, device=device)
    state = torch.randn(B, HV, V, K, dtype=torch.bfloat16, device=device)
    A_log = torch.randn(HV, dtype=torch.float32, device=device)
    a = torch.randn(B, T, HV, dtype=torch.bfloat16, device=device)
    dt_bias = torch.randn(HV, dtype=torch.float32, device=device)
    b = torch.randn(B, T, HV, dtype=torch.bfloat16, device=device)

    with pytest.raises(ValueError, match="state_layout must be"):
        gated_delta_rule_decode_unified(
            q, k, v, state, A_log, a, dt_bias, b, state_layout="invalid"
        )


def test_unified_kv_with_state_indices_raises():
    _skip_if_not_sm90_or_later()
    B, T, H, HV, K, V = 2, 1, 16, 32, 128, 128
    device = torch.device("cuda")
    q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device=device)
    k = torch.randn(B, T, H, K, dtype=torch.bfloat16, device=device)
    v = torch.randn(B, T, HV, V, dtype=torch.bfloat16, device=device)
    state = torch.randn(B, HV, K, V, dtype=torch.float32, device=device)
    state_indices = torch.arange(B, dtype=torch.int32, device=device)
    A_log = torch.randn(HV, dtype=torch.float32, device=device)
    a = torch.randn(B, T, HV, dtype=torch.bfloat16, device=device)
    dt_bias = torch.randn(HV, dtype=torch.float32, device=device)
    b = torch.randn(B, T, HV, dtype=torch.bfloat16, device=device)

    with pytest.raises(NotImplementedError, match="state_indices.*KV"):
        gated_delta_rule_decode_unified(
            q,
            k,
            v,
            state,
            A_log,
            a,
            dt_bias,
            b,
            state_layout="KV",
            state_indices=state_indices,
        )


def test_unified_vk_fp32_t_gt_1_without_pool_raises():
    _skip_if_not_sm90_or_later()
    B, T, H, HV, K, V = 2, 3, 16, 32, 128, 128
    device = torch.device("cuda")
    q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device=device)
    k = torch.randn(B, T, H, K, dtype=torch.bfloat16, device=device)
    v = torch.randn(B, T, HV, V, dtype=torch.bfloat16, device=device)
    # state as per-batch (no pool) but T>1 -> should require pool
    state = torch.randn(B, HV, V, K, dtype=torch.float32, device=device)
    A_log = torch.randn(HV, dtype=torch.float32, device=device)
    a = torch.randn(B, T, HV, dtype=torch.bfloat16, device=device)
    dt_bias = torch.randn(HV, dtype=torch.float32, device=device)
    b = torch.randn(B, T, HV, dtype=torch.bfloat16, device=device)

    with pytest.raises(ValueError, match="state_indices and state as pool"):
        gated_delta_rule_decode_unified(
            q,
            k,
            v,
            state,
            A_log,
            a,
            dt_bias,
            b,
            state_layout="VK",
        )


def test_unified_kv_t_gt_1_raises():
    _skip_if_not_sm90_or_later()
    B, T, H, HV, K, V = 2, 2, 16, 32, 128, 128
    device = torch.device("cuda")
    q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device=device)
    k = torch.randn(B, T, H, K, dtype=torch.bfloat16, device=device)
    v = torch.randn(B, T, HV, V, dtype=torch.bfloat16, device=device)
    state = torch.randn(B, HV, K, V, dtype=torch.float32, device=device)
    A_log = torch.randn(HV, dtype=torch.float32, device=device)
    a = torch.randn(B, T, HV, dtype=torch.bfloat16, device=device)
    dt_bias = torch.randn(HV, dtype=torch.float32, device=device)
    b = torch.randn(B, T, HV, dtype=torch.bfloat16, device=device)

    with pytest.raises(ValueError, match="state_layout='KV' only supports T=1"):
        gated_delta_rule_decode_unified(
            q, k, v, state, A_log, a, dt_bias, b, state_layout="KV"
        )
