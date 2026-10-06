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
"""

from __future__ import annotations

import importlib

import pytest
import torch

from flashinfer.utils import get_compute_capability

pytestmark = [pytest.mark.long_running, pytest.mark.solo]


def _skip_if_not_sm90_or_later() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    cc = get_compute_capability(torch.device("cuda"))
    if cc[0] not in (9, 10, 11, 12):
        pytest.skip(f"GDN decode requires SM90+ or SM100+, but got SM{cc[0]}{cc[1]}")


def _load_mtp_modules():
    try:
        api = importlib.import_module("flashinfer.gdn_decode")
        kernel = importlib.import_module("flashinfer.gdn_kernels.gdn_decode_mtp")
    except ImportError as exc:
        pytest.skip(f"CuTe DSL MTP kernel is unavailable: {exc}")
    return api, kernel


def _clear_mtp_compile_caches(kernel) -> None:
    kernel._get_compiled_mtp_kernel.cache_clear()
    kernel._get_compiled_mtp_kernel_inline.cache_clear()


def _skip_if_low_vram(required_bytes: int) -> None:
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    free, _ = torch.cuda.mem_get_info()
    if free < int(required_bytes * 1.2):
        pytest.skip(
            f"Requires ~{required_bytes / 1024**3:.1f}GB free VRAM, "
            f"only {free / 1024**3:.1f}GB available"
        )


@pytest.mark.parametrize(
    ("batch_size", "num_v_heads"),
    [(1, 1), (129, 1)],
    ids=["inline", "warp-specialized"],
)
def test_mtp_pool_reuses_compile_across_outer_layouts(
    batch_size: int,
    num_v_heads: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pool capacity and leading stride must not produce new cubins."""
    _skip_if_not_sm90_or_later()
    api, kernel = _load_mtp_modules()

    B, T, H, HV, K, V = batch_size, 2, 1, num_v_heads, 128, 128
    device = torch.device("cuda")
    torch.manual_seed(0)

    q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device=device)
    k = torch.nn.functional.normalize(
        torch.randn(B, T, H, K, dtype=torch.bfloat16, device=device),
        p=2.0,
        dim=-1,
    )
    v = torch.randn(B, T, HV, V, dtype=torch.bfloat16, device=device)
    A_log = torch.randn(HV, dtype=torch.float32, device=device) * 0.1
    dt_bias = torch.randn(HV, dtype=torch.float32, device=device) * 0.1
    a = torch.randn(B, T, HV, dtype=torch.bfloat16, device=device) * 0.1
    b = torch.randn(B, T, HV, dtype=torch.bfloat16, device=device)
    pool_indices = torch.arange(B, dtype=torch.int32, device=device)

    pool_compile_calls = 0
    pool_compile_runners = []
    original_compile = kernel.cute.compile

    def counted_compile(*args, **kwargs):
        nonlocal pool_compile_calls
        if kwargs.get("use_pool_indexing", False):
            pool_compile_calls += 1
            pool_compile_runners.append(args[0])
        return original_compile(*args, **kwargs)

    _clear_mtp_compile_caches(kernel)
    monkeypatch.setattr(kernel.cute, "compile", counted_compile)

    common = dict(
        q=q,
        k=k,
        v=v,
        initial_state_indices=pool_indices,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        scale=1.0,
        intermediate_states_buffer=None,
        disable_state_update=False,
        use_qk_l2norm=True,
    )

    try:
        inner_strides = None
        for pool_size, page_gap in ((B + 1, 2), (B + 3, 2), (B + 3, 3)):
            storage = torch.randn(
                pool_size,
                page_gap,
                HV,
                V,
                K,
                dtype=torch.float32,
                device=device,
            )
            pool = storage[:, 0]
            assert not pool.is_contiguous()
            if inner_strides is None:
                inner_strides = pool.stride()[1:]
            else:
                assert pool.stride()[1:] == inner_strides

            direct_state = pool[pool_indices].clone()
            pool_output, pool_state = api.gated_delta_rule_mtp(
                initial_state=pool, **common
            )
            direct_output, direct_state = api.gated_delta_rule_mtp(
                initial_state=direct_state, **common
            )
            torch.cuda.synchronize()

            torch.testing.assert_close(pool_output, direct_output, atol=1e-3, rtol=1e-3)
            torch.testing.assert_close(
                pool_state[pool_indices], direct_state, atol=1e-3, rtol=1e-3
            )

        assert pool_compile_calls == 1
        expected_runner = (
            kernel.run_gdn_verify_kernel_mtp_inline
            if B * HV <= 128
            else kernel.run_gdn_verify_kernel_mtp
        )
        assert pool_compile_runners == [expected_runner]
    finally:
        _clear_mtp_compile_caches(kernel)


@pytest.mark.parametrize(
    "getter_name",
    ["_get_compiled_mtp_kernel_inline", "_get_compiled_mtp_kernel"],
)
def test_mtp_pool_cache_keeps_inner_strides_static(getter_name: str) -> None:
    """Layouts with different inner strides must keep separate cache entries."""
    _, kernel = _load_mtp_modules()
    getter = getattr(kernel, getter_name)
    common = dict(
        T=2,
        H=1,
        HV=2,
        K=128,
        V=128,
        cache_steps=2,
        disable_state_update=False,
        use_pool_indexing=True,
        scale=1.0,
        use_qk_l2norm=True,
        tile_v=8,
        vec_size=4,
        dtype_key=(torch.float32, torch.float32, torch.int32),
        ilp_rows=2,
        use_smem_v=False,
        use_packed_fma=False,
        per_token_pool_scatter=False,
    )

    getter.cache_clear()
    try:
        compact = getter(
            pool_inner_strides_key=(128 * 128, 128, 1),
            **common,
        )
        padded_inner = getter(
            pool_inner_strides_key=(129 * 128, 128, 1),
            **common,
        )
        assert compact is not padded_inner
    finally:
        getter.cache_clear()


def test_mtp_pool_rejects_misaligned_slot_stride() -> None:
    """Each pool slot must preserve the alignment required by vec4 copies."""
    _skip_if_not_sm90_or_later()
    api, _ = _load_mtp_modules()

    B, T, H, HV, K, V = 1, 2, 1, 1, 128, 128
    device = torch.device("cuda")
    pool = torch.empty_strided(
        (2, HV, V, K),
        (HV * V * K + 1, V * K, K, 1),
        dtype=torch.float32,
        device=device,
    )
    assert not pool.is_contiguous()

    with pytest.raises(
        AssertionError,
        match=r"stride\(0\) must be a multiple of 4 FP32 elements",
    ):
        api.gated_delta_rule_mtp(
            q=torch.empty(B, T, H, K, dtype=torch.bfloat16, device=device),
            k=torch.empty(B, T, H, K, dtype=torch.bfloat16, device=device),
            v=torch.empty(B, T, HV, V, dtype=torch.bfloat16, device=device),
            initial_state=pool,
            initial_state_indices=torch.zeros(B, dtype=torch.int32, device=device),
            A_log=torch.empty(HV, dtype=torch.float32, device=device),
            a=torch.empty(B, T, HV, dtype=torch.bfloat16, device=device),
            dt_bias=torch.empty(HV, dtype=torch.float32, device=device),
            b=torch.empty(B, T, HV, dtype=torch.bfloat16, device=device),
            disable_state_update=False,
        )


def test_mtp_pool_int64_dynamic_leading_stride() -> None:
    """Large dynamic slot offsets must use 64-bit address arithmetic."""
    _skip_if_not_sm90_or_later()
    api, kernel = _load_mtp_modules()

    T, H, HV, K, V = 2, 16, 32, 128, 128
    slot_stride = (HV + 1) * V * K
    overflow_pool_idx = (2**31 + slot_stride - 1) // slot_stride
    route_cases = (
        ("inline", 1, overflow_pool_idx),
        ("warp-specialized", 5, overflow_pool_idx + 1),
    )
    pool_size = max(first_idx + B for _, B, first_idx in route_cases)
    required_elements = (pool_size - 1) * slot_stride + HV * V * K
    required_bytes = required_elements * torch.float32.itemsize
    _skip_if_low_vram(required_bytes)

    device = torch.device("cuda")
    pool = torch.empty_strided(
        (pool_size, HV, V, K),
        (slot_stride, V * K, K, 1),
        dtype=torch.float32,
        device=device,
    )
    assert not pool.is_contiguous()
    assert pool.stride(0) == slot_stride
    assert overflow_pool_idx * slot_stride >= 2**31

    _clear_mtp_compile_caches(kernel)
    try:
        for route, B, first_idx in route_cases:
            torch.manual_seed(20260730 + B)
            pool_indices = torch.arange(
                first_idx,
                first_idx + B,
                dtype=torch.int32,
                device=device,
            )
            for pool_idx in pool_indices.tolist():
                pool[pool_idx].normal_(0.0, 0.01)

            q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device=device) * 0.05
            k = torch.nn.functional.normalize(
                torch.randn(B, T, H, K, dtype=torch.bfloat16, device=device),
                p=2.0,
                dim=-1,
            )
            v = torch.randn(B, T, HV, V, dtype=torch.bfloat16, device=device) * 0.05
            A_log = torch.randn(HV, dtype=torch.float32, device=device) * 0.1
            dt_bias = torch.randn(HV, dtype=torch.float32, device=device) * 0.1
            a = torch.randn(B, T, HV, dtype=torch.bfloat16, device=device) * 0.1
            b = torch.randn(B, T, HV, dtype=torch.bfloat16, device=device)

            direct_state = pool[pool_indices].clone()
            direct_indices = torch.arange(B, dtype=torch.int32, device=device)
            common = dict(
                q=q,
                k=k,
                v=v,
                A_log=A_log,
                a=a,
                dt_bias=dt_bias,
                b=b,
                scale=1.0,
                intermediate_states_buffer=None,
                disable_state_update=False,
                use_qk_l2norm=True,
            )

            pool_output, pool_state = api.gated_delta_rule_mtp(
                initial_state=pool,
                initial_state_indices=pool_indices,
                **common,
            )
            direct_output, direct_state = api.gated_delta_rule_mtp(
                initial_state=direct_state,
                initial_state_indices=direct_indices,
                **common,
            )
            torch.cuda.synchronize()

            torch.testing.assert_close(
                pool_output,
                direct_output,
                atol=1e-3,
                rtol=1e-3,
                msg=f"{route} output mismatch",
            )
            torch.testing.assert_close(
                pool_state[pool_indices],
                direct_state,
                atol=1e-3,
                rtol=1e-3,
                msg=f"{route} state mismatch",
            )
    finally:
        _clear_mtp_compile_caches(kernel)
