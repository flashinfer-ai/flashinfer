# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-side tests for the SM120 PRIMS FMHA API and JIT specialization."""

import inspect
import pytest
import torch

from flashinfer.attention.cute_dsl.sm120_fmha import (
    _prepare_skip_softmax_threshold,
    sm120_fmha_fp8_paged_prefill,
    sm120_fmha_fp8_ragged_prefill,
)


def test_skip_softmax_threshold_float_expansion_and_validation():
    q = torch.empty(1)
    assert _prepare_skip_softmax_threshold(None, q, 3) is None
    zero = _prepare_skip_softmax_threshold(0.0, q, 3)
    assert zero is not None
    assert zero.shape == (3,)
    assert zero.dtype == torch.float32
    assert torch.equal(zero, torch.zeros(3))
    threshold = _prepare_skip_softmax_threshold(2.0, q, 3)
    assert torch.equal(threshold, torch.full((3,), 2.0))
    with pytest.raises(ValueError, match="finite and >= 0"):
        _prepare_skip_softmax_threshold(-1.0, q, 3)
    with pytest.raises(ValueError, match="finite and >= 0"):
        _prepare_skip_softmax_threshold(float("nan"), q, 3)
    with pytest.raises(TypeError, match="None, a Python float, or a torch.Tensor"):
        _prepare_skip_softmax_threshold(object(), q, 3)


def test_skip_softmax_threshold_rejects_cpu_tensor():
    q = torch.empty(1)
    with pytest.raises(ValueError, match="must be a CUDA tensor"):
        _prepare_skip_softmax_threshold(torch.zeros(2), q, 2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_skip_softmax_threshold_tensor_metadata_validation():
    q = torch.empty(1, device="cuda")
    threshold = torch.zeros(2, device="cuda", dtype=torch.float32)
    assert _prepare_skip_softmax_threshold(threshold, q, 2) is threshold
    with pytest.raises(ValueError, match="dtype torch.float32"):
        _prepare_skip_softmax_threshold(threshold.half(), q, 2)
    with pytest.raises(ValueError, match=r"shape \(2,\)"):
        _prepare_skip_softmax_threshold(threshold[:1], q, 2)
    noncontiguous = torch.zeros(4, device="cuda", dtype=torch.float32)[::2]
    with pytest.raises(ValueError, match="must be contiguous"):
        _prepare_skip_softmax_threshold(noncontiguous, q, 2)


def test_skip_softmax_is_exposed_only_by_direct_sm120_apis():
    from flashinfer.prefill import (
        BatchPrefillWithPagedKVCacheWrapper,
        BatchPrefillWithRaggedKVCacheWrapper,
    )

    ragged = inspect.signature(sm120_fmha_fp8_ragged_prefill).parameters
    paged = inspect.signature(sm120_fmha_fp8_paged_prefill).parameters
    assert "skip_softmax_threshold" in ragged
    assert "skip_softmax_threshold" in paged
    assert "max_seqlen_k" not in ragged
    assert "max_seqlen_kv" not in paged
    assert "skip_softmax_threshold_scale_factor" not in ragged
    assert "skip_softmax_threshold_scale_factor" not in paged

    ragged_wrapper = inspect.signature(
        BatchPrefillWithRaggedKVCacheWrapper.run
    ).parameters
    paged_wrapper = inspect.signature(
        BatchPrefillWithPagedKVCacheWrapper.run
    ).parameters
    assert "skip_softmax_threshold_scale_factor" not in ragged_wrapper
    assert "skip_softmax_threshold_scale_factor" in paged_wrapper


def test_skip_softmax_has_a_distinct_jit_specialization(monkeypatch):
    pytest.importorskip("cutlass.experimental")

    from flashinfer.cute_dsl.attention.fmha.sm120.compile import (
        compile_sm120_fmha_fp8_ragged_kernel,
    )
    from flashinfer.jit import cute_dsl_core

    observed_names = []

    def fake_build_and_load(module_name, kernel_name, compile_fn, **kwargs):
        observed_names.append((module_name, kernel_name))
        return kernel_name

    compile_sm120_fmha_fp8_ragged_kernel.cache_clear()
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device=None: (12, 0)
    )
    monkeypatch.setattr(
        cute_dsl_core, "build_and_load_cute_dsl_kernel", fake_build_and_load
    )

    common = dict(
        in_dtype=torch.float8_e4m3fn,
        out_dtype=torch.bfloat16,
        num_qo_heads=8,
        num_kv_heads=8,
        head_dim=128,
        is_causal=False,
        kv_tile=128,
        q_tile=128,
        device=torch.device("cuda"),
    )
    dense = compile_sm120_fmha_fp8_ragged_kernel(**common)
    sparse = compile_sm120_fmha_fp8_ragged_kernel(**common, enable_skip_softmax=True)

    assert dense != sparse
    assert observed_names[0][1].endswith("_skip0")
    assert observed_names[1][1].endswith("_skip1")
    compile_sm120_fmha_fp8_ragged_kernel.cache_clear()
