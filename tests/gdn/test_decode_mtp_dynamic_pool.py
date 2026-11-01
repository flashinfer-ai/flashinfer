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

pytestmark = [pytest.mark.long_running, pytest.mark.gpu_exclusive]


def _skip_if_not_sm90_or_later() -> None:
    """Skip devices outside the supported GDN decode architectures."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    cc = get_compute_capability(torch.device("cuda"))
    if cc[0] not in (9, 10, 11, 12):
        pytest.skip(f"GDN decode requires SM90+ or SM100+, but got SM{cc[0]}{cc[1]}")


def _load_mtp_modules():
    """Load the MTP API and kernel, skipping only a missing optional CuTe DSL."""
    pytest.importorskip("cutlass")
    api = importlib.import_module("flashinfer.gdn_decode")
    kernel = importlib.import_module("flashinfer.gdn_kernels.gdn_decode_mtp")
    return api, kernel


def _clear_mtp_compile_caches(kernel) -> None:
    """Clear both in-process specialization caches between tests."""
    kernel._get_compiled_mtp_kernel.cache_clear()
    kernel._get_compiled_mtp_kernel_inline.cache_clear()


def _skip_if_low_vram(required_bytes: int) -> None:
    """Leave allocation headroom for the large-offset test and its reference."""
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    free, _ = torch.cuda.mem_get_info()
    if free < int(required_bytes * 1.2):
        pytest.skip(
            f"Requires ~{required_bytes / 1024**3:.1f}GB free VRAM, "
            f"only {free / 1024**3:.1f}GB available"
        )


@pytest.fixture
def mtp_modules(monkeypatch):
    """Isolate specialization counts and disable disk-cache hits by default."""
    _skip_if_not_sm90_or_later()
    api, kernel = _load_mtp_modules()
    monkeypatch.setenv("FLASHINFER_CUTE_DSL_DISABLE_CACHE", "1")
    _clear_mtp_compile_caches(kernel)
    try:
        yield api, kernel
    finally:
        _clear_mtp_compile_caches(kernel)


def _make_mtp_inputs(batch_size, num_v_heads):
    """Build deterministic MTP inputs with nontrivial pool-slot indices."""
    B, T, H, HV, K, V = batch_size, 2, 1, num_v_heads, 128, 128
    device = torch.device("cuda")
    torch.manual_seed(0)
    return dict(
        q=torch.randn(B, T, H, K, dtype=torch.bfloat16, device=device) * 0.1,
        k=torch.nn.functional.normalize(
            torch.randn(B, T, H, K, dtype=torch.bfloat16, device=device),
            p=2.0,
            dim=-1,
        ),
        v=torch.randn(B, T, HV, V, dtype=torch.bfloat16, device=device) * 0.1,
        A_log=torch.randn(HV, dtype=torch.float32, device=device) * 0.1,
        dt_bias=torch.randn(HV, dtype=torch.float32, device=device) * 0.1,
        a=torch.randn(B, T, HV, dtype=torch.bfloat16, device=device) * 0.1,
        b=torch.randn(B, T, HV, dtype=torch.bfloat16, device=device),
        initial_state_indices=torch.arange(B, 0, -1, dtype=torch.int32, device=device),
        scale=1.0,
        intermediate_states_buffer=None,
        disable_state_update=False,
        use_qk_l2norm=True,
    )


def _check_pool_call(api, kernel, pool, common, expected_misses):
    """Count only the pool call; the compact reference has its own specialization."""
    B, _, HV, _ = common["v"].shape
    getter, other_getter = (
        (kernel._get_compiled_mtp_kernel_inline, kernel._get_compiled_mtp_kernel)
        if B * HV <= 128
        else (kernel._get_compiled_mtp_kernel, kernel._get_compiled_mtp_kernel_inline)
    )
    indices = common["initial_state_indices"]
    pool_before = pool.clone()
    direct_state = pool_before[indices].clone()
    before = getter.cache_info()
    other_before = other_getter.cache_info()
    pool_output, pool_state = api.gated_delta_rule_mtp(initial_state=pool, **common)
    after = getter.cache_info()
    assert after.misses - before.misses == expected_misses
    assert after.hits - before.hits == 1 - expected_misses
    assert after.currsize - before.currsize == expected_misses
    assert other_getter.cache_info() == other_before
    assert pool_state is pool

    direct_args = dict(common)
    direct_args["initial_state_indices"] = torch.arange(
        B, dtype=indices.dtype, device=indices.device
    )
    direct_output, direct_state = api.gated_delta_rule_mtp(
        initial_state=direct_state, **direct_args
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(pool_output, direct_output, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(pool_state[indices], direct_state, atol=1e-3, rtol=1e-3)
    untouched = torch.ones(pool.shape[0], dtype=torch.bool, device=pool.device)
    untouched[indices] = False
    torch.testing.assert_close(pool[untouched], pool_before[untouched], atol=0, rtol=0)


@pytest.mark.parametrize("batch_size", [1, 65], ids=["inline", "warp-specialized"])
def test_mtp_pool_specializes_capacity_and_slot_stride(mtp_modules, batch_size):
    """Each capacity/stride pair compiles once and is reused on subsequent calls."""
    api, kernel = mtp_modules
    HV, V, K = 2, 128, 128
    common = _make_mtp_inputs(batch_size, HV)
    inner_strides = None
    for pool_size, page_gap, expected_misses in (
        (batch_size + 1, 2, 1),
        (batch_size + 3, 2, 1),
        (batch_size + 3, 2, 0),
        (batch_size + 3, 3, 1),
        (batch_size + 1, 2, 0),
    ):
        storage = (
            torch.randn(
                pool_size, page_gap, HV, V, K, dtype=torch.float32, device="cuda"
            )
            * 0.01
        )
        pool = storage[:, 0]
        assert not pool.is_contiguous()
        if inner_strides is None:
            inner_strides = pool.stride()[1:]
        assert pool.stride()[1:] == inner_strides
        padding_before = storage[:, 1:].clone()
        _check_pool_call(api, kernel, pool, common, expected_misses)
        torch.testing.assert_close(storage[:, 1:], padding_before, atol=0, rtol=0)


@pytest.mark.parametrize("batch_size", [1, 65], ids=["inline", "warp-specialized"])
def test_mtp_compact_pool_reuses_compile_across_capacities(mtp_modules, batch_size):
    """The compact 3D path must retain its existing dynamic-capacity ABI."""
    api, kernel = mtp_modules
    HV, V, K = 2, 128, 128
    common = _make_mtp_inputs(batch_size, HV)
    for pool_size, expected_misses in ((batch_size + 1, 1), (batch_size + 3, 0)):
        pool = (
            torch.randn(pool_size, HV, V, K, dtype=torch.float32, device="cuda")
            * 0.01
        )
        assert pool.is_contiguous()
        _check_pool_call(api, kernel, pool, common, expected_misses)


@pytest.mark.parametrize("batch_size", [1, 65], ids=["inline", "warp-specialized"])
def test_mtp_pool_cache_keeps_inner_strides_static(mtp_modules, batch_size):
    """Changing only the head stride must compile and execute a distinct layout."""
    api, kernel = mtp_modules
    HV, V, K = 2, 128, 128
    common = _make_mtp_inputs(batch_size, HV)
    slot_stride = (HV + 1) * V * K
    for head_stride, expected_misses in (
        (V * K, 1),
        ((V + 1) * K, 1),
        ((V + 1) * K, 0),
        (V * K, 0),
    ):
        # Hold capacity and the slot stride fixed to isolate inner-stride keying.
        storage = (
            torch.randn(batch_size + 2, slot_stride, dtype=torch.float32, device="cuda")
            * 0.01
        )
        pool = storage.as_strided(
            (batch_size + 2, HV, V, K), (slot_stride, head_stride, K, 1)
        )
        assert not pool.is_contiguous()
        before = storage.clone()
        _check_pool_call(api, kernel, pool, common, expected_misses)
        padding = torch.ones(slot_stride, dtype=torch.bool, device="cuda")
        for head in range(HV):
            padding[head * head_stride : head * head_stride + V * K] = False
        torch.testing.assert_close(
            storage[:, padding], before[:, padding], atol=0, rtol=0
        )


@pytest.mark.parametrize("batch_size", [1, 65], ids=["inline", "warp-specialized"])
def test_mtp_pool_reloads_disk_cache_for_each_capacity(
    mtp_modules, batch_size, monkeypatch, tmp_path
):
    """Each capacity has its own artifact, reused after clearing the memory cache."""
    from flashinfer.jit import env as jit_env

    api, kernel = mtp_modules
    monkeypatch.setenv("FLASHINFER_CUTE_DSL_DISABLE_CACHE", "0")
    monkeypatch.setattr(jit_env, "FLASHINFER_JIT_DIR", tmp_path)
    HV, V, K = 2, 128, 128
    common = _make_mtp_inputs(batch_size, HV)
    for pool_size, artifact_count in ((batch_size + 1, 2), (batch_size + 3, 3)):
        storage = (
            torch.randn(pool_size, 2, HV, V, K, dtype=torch.float32, device="cuda")
            * 0.01
        )
        _check_pool_call(api, kernel, storage[:, 0], common, expected_misses=1)
        # One compact reference plus one strided specialization per capacity.
        assert len(list(tmp_path.glob("gdn_decode_mtp_*/*.o"))) == artifact_count
    artifacts = sorted(tmp_path.glob("gdn_decode_mtp_*/*.o"))

    _clear_mtp_compile_caches(kernel)
    build_and_load = kernel.build_and_load_cute_dsl_kernel

    def load_only(module_name, kernel_name, compile_fn, **kwargs):
        """Require actual disk reloads without falling back to compilation."""

        def unexpected_compile():
            pytest.fail(f"Expected a disk-cache hit for {kernel_name}")

        return build_and_load(module_name, kernel_name, unexpected_compile, **kwargs)

    monkeypatch.setattr(kernel, "build_and_load_cute_dsl_kernel", load_only)
    for pool_size, expected_misses in (
        (batch_size + 3, 1),
        (batch_size + 1, 1),
        (batch_size + 3, 0),
    ):
        storage = (
            torch.randn(pool_size, 2, HV, V, K, dtype=torch.float32, device="cuda")
            * 0.01
        )
        _check_pool_call(api, kernel, storage[:, 0], common, expected_misses)
    assert sorted(tmp_path.glob("gdn_decode_mtp_*/*.o")) == artifacts


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


def test_mtp_pool_int64_slot_offsets() -> None:
    """Large slot offsets must use 64-bit arithmetic even with a static stride."""
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
