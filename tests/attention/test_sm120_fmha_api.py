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


def test_skip_softmax_threshold_load_follows_pdl_dependency_wait():
    from flashinfer.cute_dsl.attention.fmha.sm120 import (
        SM120FusedMultiHeadAttentionFP8ForwardTMA,
    )

    kernel_source = inspect.getsource(SM120FusedMultiHeadAttentionFP8ForwardTMA.kernel)
    wait = kernel_source.index("cute.arch.griddepcontrol_wait()")
    threshold_load = kernel_source.index("skip_softmax_threshold[request_idx]")
    assert wait < threshold_load


def test_page_size_1_supported_and_can_implement_paged():
    cutlass = pytest.importorskip("cutlass")
    pytest.importorskip("cutlass.experimental")

    from flashinfer.cute_dsl.attention.fmha.sm120 import (
        SM120FusedMultiHeadAttentionFP8ForwardTMA,
    )

    assert 1 in SM120FusedMultiHeadAttentionFP8ForwardTMA.SUPPORTED_PAGE_SIZES
    for head_dim in (32, 64, 128, 256):
        for q_tile in (64, 128):
            for kv_tile in (64, 128):
                assert SM120FusedMultiHeadAttentionFP8ForwardTMA.can_implement_paged(
                    cutlass.Float8E4M3FN,
                    cutlass.BFloat16,
                    q_shape=(2, 1, 24, head_dim),
                    k_shape=(2, 1, 4, head_dim),
                    num_tokens_per_page=1,
                    kv_tile=kv_tile,
                    q_tile=q_tile,
                )
    for bad_page_size in (0, 2, 4, 8, 256):
        assert not SM120FusedMultiHeadAttentionFP8ForwardTMA.can_implement_paged(
            cutlass.Float8E4M3FN,
            cutlass.BFloat16,
            q_shape=(1, 1, 8, 128),
            k_shape=(1, 1, 2, 128),
            num_tokens_per_page=bad_page_size,
        )


def test_page_size_1_has_a_distinct_jit_specialization(monkeypatch):
    pytest.importorskip("cutlass.experimental")

    from flashinfer.cute_dsl.attention.fmha.sm120.compile import (
        compile_sm120_fmha_fp8_paged_kernel,
    )
    from flashinfer.jit import cute_dsl_core

    observed_names = []

    def fake_build_and_load(module_name, kernel_name, compile_fn, **kwargs):
        observed_names.append((module_name, kernel_name))
        return kernel_name

    compile_sm120_fmha_fp8_paged_kernel.cache_clear()
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device=None: (12, 0)
    )
    monkeypatch.setattr(
        cute_dsl_core, "build_and_load_cute_dsl_kernel", fake_build_and_load
    )

    common = dict(
        in_dtype=torch.float8_e4m3fn,
        out_dtype=torch.bfloat16,
        num_qo_heads=24,
        num_kv_heads=4,
        head_dim=256,
        is_causal=True,
        kv_tile=128,
        q_tile=128,
        device=torch.device("cuda"),
        balanced_scheduler=True,
    )
    ps1 = compile_sm120_fmha_fp8_paged_kernel(**common, num_tokens_per_page=1)
    ps16 = compile_sm120_fmha_fp8_paged_kernel(**common, num_tokens_per_page=16)

    assert ps1 != ps16
    assert "_page1_" in observed_names[0][1]
    assert "_page16_" in observed_names[1][1]
    compile_sm120_fmha_fp8_paged_kernel.cache_clear()


def test_validate_hnd_paged_pool_page_size_1():
    from flashinfer.attention.cute_dsl.sm120_fmha import _validate_hnd_paged_pool

    # Standalone HND pool with page_size=1
    standalone = torch.empty((16, 4, 1, 256), dtype=torch.float8_e4m3fn)
    _validate_hnd_paged_pool("k_pool", standalone)

    # Plane view from combined [num_pages, 2, Hkv, 1, D] cache
    combined = torch.empty((16, 2, 4, 1, 256), dtype=torch.float8_e4m3fn)
    k_view, v_view = combined.unbind(dim=1)
    _validate_hnd_paged_pool("k_pool", k_view)
    _validate_hnd_paged_pool("v_pool", v_view)

    # NHD -> HND zero-copy squeeze/unsqueeze view for page_size=1
    nhd_combined = torch.empty((16, 2, 1, 4, 256), dtype=torch.float8_e4m3fn)
    k_nhd, v_nhd = nhd_combined.unbind(dim=1)
    _validate_hnd_paged_pool("k_pool", k_nhd.squeeze(1).unsqueeze(2))
    _validate_hnd_paged_pool("v_pool", v_nhd.squeeze(1).unsqueeze(2))

    # Misaligned outer page stride (<16B multiple) must be rejected
    raw = torch.empty(16 * (4 * 32 + 8), dtype=torch.float8_e4m3fn)
    misaligned = torch.as_strided(raw, (16, 4, 1, 32), (4 * 32 + 8, 32, 32, 1))
    with pytest.raises(ValueError, match="16-byte aligned page stride"):
        _validate_hnd_paged_pool("k_pool", misaligned)


def test_plan_paged_accepts_page_size_1_and_nhd_layout_only_for_page_size_1(
    monkeypatch,
):
    from flashinfer.attention.cute_dsl.sm120_fmha import (
        SM120PrimsBatchPrefillBackend,
        _validate_sm120_prims_plan_options,
    )
    from flashinfer.cute_dsl.attention.fmha import sm120 as sm120_mod

    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device=None: (12, 0)
    )
    compiled_calls = []
    monkeypatch.setattr(
        sm120_mod,
        "compile_sm120_fmha_fp8_paged_kernel",
        lambda **kwargs: compiled_calls.append(kwargs),
    )

    backend = SM120PrimsBatchPrefillBackend(torch.device("cpu"))
    qo = torch.tensor([0, 4], dtype=torch.int32)
    seqlens = torch.tensor([8], dtype=torch.int32)
    block_tables = torch.arange(8, dtype=torch.int32).unsqueeze(0)

    backend.plan_paged(
        qo_indptr=qo,
        qo_indptr_host=qo,
        seqlens_kv=seqlens,
        seqlens_kv_host=seqlens,
        block_tables=block_tables,
        q_dtype=torch.float8_e4m3fn,
        kv_dtype=torch.float8_e4m3fn,
        o_dtype=torch.bfloat16,
        num_qo_heads=24,
        num_kv_heads=4,
        head_dim_qk=256,
        head_dim_vo=256,
        page_size=1,
        causal=True,
        sm_scale=None,
        disable_split_kv=True,
    )
    assert backend._page_size == 1
    assert compiled_calls and compiled_calls[-1]["num_tokens_per_page"] == 1

    with pytest.raises(ValueError, match=r"\{1, 16, 32, 64, 128\}"):
        backend.plan_paged(
            qo_indptr=qo,
            qo_indptr_host=qo,
            seqlens_kv=seqlens,
            seqlens_kv_host=seqlens,
            block_tables=block_tables,
            q_dtype=torch.float8_e4m3fn,
            kv_dtype=torch.float8_e4m3fn,
            o_dtype=torch.bfloat16,
            num_qo_heads=24,
            num_kv_heads=4,
            head_dim_qk=256,
            head_dim_vo=256,
            page_size=8,
            causal=True,
            sm_scale=None,
            disable_split_kv=True,
        )

    # NHD layout is rejected when required_kv_layout="HND" (page_size > 1)
    with pytest.raises(ValueError, match="requires kv_layout='HND'"):
        _validate_sm120_prims_plan_options(
            kv_layout="NHD",
            required_kv_layout="HND",
            custom_mask=None,
            packed_custom_mask=None,
            pos_encoding_mode="NONE",
            use_fp16_qk_reduction=False,
            window_left=-1,
            logits_soft_cap=0.0,
            prefix_len_ptr=None,
            token_pos_in_items_ptr=None,
            max_item_len_ptr=None,
            max_sequence_kv=None,
        )


def test_paged_wrapper_run_nhd_page_size_1_reshapes_zero_copy_and_validates():
    from flashinfer.attention.cute_dsl.sm120_fmha import _validate_hnd_paged_pool
    from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper

    captured = []

    class _StubPrimsBackend:
        _page_size = 1

        def run_paged(self, q, k, v, out, **kwargs):
            _validate_hnd_paged_pool("k_pool", k)
            _validate_hnd_paged_pool("v_pool", v)
            captured.append((k, v))
            return out

    wrapper = BatchPrefillWithPagedKVCacheWrapper.__new__(
        BatchPrefillWithPagedKVCacheWrapper
    )
    wrapper._kv_layout = "NHD"
    wrapper._backend = "cute-dsl-prims"
    wrapper._batch_invariant = None
    wrapper._cached_q_data_type = torch.float8_e4m3fn
    wrapper._cached_kv_data_type = torch.float8_e4m3fn
    wrapper._cached_o_data_type = torch.bfloat16
    wrapper._qo_indptr_last = 2
    wrapper._window_left = -1
    wrapper._prims_backend = _StubPrimsBackend()

    q = torch.empty((2, 24, 256), dtype=torch.float8_e4m3fn)

    # 3D token-granular (k_pool, v_pool) tuple
    k3 = torch.empty((8, 4, 256), dtype=torch.float8_e4m3fn)
    v3 = torch.empty((8, 4, 256), dtype=torch.float8_e4m3fn)
    wrapper.run(q, (k3, v3), enable_pdl=False)
    k_seen, v_seen = captured[-1]
    assert k_seen.shape == (8, 4, 1, 256)
    assert k_seen.data_ptr() == k3.data_ptr()
    assert v_seen.data_ptr() == v3.data_ptr()

    # 4D combined token-granular cache [num_pages, 2, Hkv, D]
    kv4 = torch.empty((8, 2, 4, 256), dtype=torch.float8_e4m3fn)
    wrapper.run(q, kv4, enable_pdl=False)
    k_seen, v_seen = captured[-1]
    assert k_seen.shape == (8, 4, 1, 256)
    assert k_seen.stride() == (2 * 4 * 256, 256, 256, 1)
    assert k_seen.data_ptr() == kv4[:, 0].data_ptr()
    assert v_seen.data_ptr() == kv4[:, 1].data_ptr()

    # 5D combined NHD cache [num_pages, 2, 1, Hkv, D]
    kv5 = torch.empty((8, 2, 1, 4, 256), dtype=torch.float8_e4m3fn)
    wrapper.run(q, kv5, enable_pdl=False)
    k_seen, v_seen = captured[-1]
    assert k_seen.shape == (8, 4, 1, 256)
    assert k_seen.data_ptr() == kv5[:, 0].data_ptr()

    # Invalid NHD page_size != 1 pool shape is rejected
    bad_k = torch.empty((8, 16, 4, 256), dtype=torch.float8_e4m3fn)
    bad_v = torch.empty((8, 16, 4, 256), dtype=torch.float8_e4m3fn)
    with pytest.raises(ValueError, match=r"\[num_pages, 1, num_kv_heads, head_dim\]"):
        wrapper.run(q, (bad_k, bad_v), enable_pdl=False)


def _plan_options(**overrides):
    options = dict(
        kv_layout="HND",
        required_kv_layout="HND",
        custom_mask=None,
        packed_custom_mask=None,
        pos_encoding_mode="NONE",
        use_fp16_qk_reduction=False,
        window_left=-1,
        logits_soft_cap=0.0,
        prefix_len_ptr=None,
        token_pos_in_items_ptr=None,
        max_item_len_ptr=None,
        max_sequence_kv=None,
    )
    options.update(overrides)
    return options


def test_plan_options_accept_sliding_window_and_keep_other_rejections():
    from flashinfer.attention.cute_dsl.sm120_fmha import (
        _validate_sm120_prims_plan_options,
    )

    for window_left in (-1, 0, 127, 1023):
        _validate_sm120_prims_plan_options(**_plan_options(window_left=window_left))
    with pytest.raises(ValueError, match="window_left == -1"):
        _validate_sm120_prims_plan_options(**_plan_options(window_left=-2))
    with pytest.raises(TypeError, match="integer window_left"):
        _validate_sm120_prims_plan_options(**_plan_options(window_left=True))
    with pytest.raises(NotImplementedError, match="logits_soft_cap"):
        _validate_sm120_prims_plan_options(
            **_plan_options(window_left=127, logits_soft_cap=30.0)
        )
    with pytest.raises(NotImplementedError, match="custom masks"):
        _validate_sm120_prims_plan_options(
            **_plan_options(
                window_left=127, custom_mask=torch.ones(4, dtype=torch.bool)
            )
        )


def test_sinks_validation():
    from flashinfer.attention.cute_dsl.sm120_fmha import _validate_sinks

    q = torch.empty((3, 8, 64), dtype=torch.float8_e4m3fn)
    _validate_sinks(q, None)
    _validate_sinks(q, torch.zeros(8))
    with pytest.raises(ValueError, match=r"shape \(8,\)"):
        _validate_sinks(q, torch.zeros(4))
    with pytest.raises(ValueError, match=r"shape \(8,\)"):
        _validate_sinks(q, torch.zeros(1, 8))
    with pytest.raises(ValueError, match="dtype torch.float32"):
        _validate_sinks(q, torch.zeros(8, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="must be contiguous"):
        _validate_sinks(q, torch.zeros(16)[::2])
    with pytest.raises(TypeError, match="torch.Tensor"):
        _validate_sinks(q, [0.0] * 8)


def test_window_and_sinks_have_distinct_jit_specializations(monkeypatch):
    pytest.importorskip("cutlass.experimental")

    from flashinfer.cute_dsl.attention.fmha.sm120.compile import (
        compile_sm120_fmha_fp8_paged_kernel,
        compile_sm120_fmha_fp8_ragged_kernel,
    )
    from flashinfer.jit import cute_dsl_core

    observed_names = []

    def fake_build_and_load(module_name, kernel_name, compile_fn, **kwargs):
        observed_names.append(kernel_name)
        return kernel_name

    compile_sm120_fmha_fp8_ragged_kernel.cache_clear()
    compile_sm120_fmha_fp8_paged_kernel.cache_clear()
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device=None: (12, 0)
    )
    monkeypatch.setattr(
        cute_dsl_core, "build_and_load_cute_dsl_kernel", fake_build_and_load
    )
    common = dict(
        in_dtype=torch.float8_e4m3fn,
        out_dtype=torch.bfloat16,
        num_qo_heads=64,
        num_kv_heads=8,
        head_dim=64,
        is_causal=True,
        kv_tile=128,
        q_tile=128,
        device=torch.device("cuda"),
        balanced_scheduler=True,
    )
    variants = [
        {},
        {"sliding_window": True},
        {"with_sinks": True},
        {"sliding_window": True, "with_sinks": True},
    ]
    ragged = [compile_sm120_fmha_fp8_ragged_kernel(**common, **v) for v in variants]
    paged = [
        compile_sm120_fmha_fp8_paged_kernel(**common, num_tokens_per_page=16, **v)
        for v in variants
    ]
    for names in (ragged, paged):
        assert len(set(names)) == len(variants)
        # The default kernels keep their pre-existing compile cache keys.
        assert "_swa" not in names[0] and "_sinks" not in names[0]
        assert names[1].endswith("_swa")
        assert names[2].endswith("_sinks")
        assert names[3].endswith("_swa_sinks")
    compile_sm120_fmha_fp8_ragged_kernel.cache_clear()
    compile_sm120_fmha_fp8_paged_kernel.cache_clear()


def test_direct_apis_expose_window_left_and_sinks():
    for fn in (sm120_fmha_fp8_ragged_prefill, sm120_fmha_fp8_paged_prefill):
        params = inspect.signature(fn).parameters
        assert params["window_left"].default == -1
        assert params["sinks"].default is None


def test_paged_kv_split_bounds_cover_only_the_window():
    from flashinfer.attention.cute_dsl.sm120_fmha import _paged_kv_split_bounds

    tile, sms = 128, 188
    # Gemma-style local layer: chunked prefill of 256 tokens on a 32K prefix.
    q_lens, kv_lens, window_left = [256], [32768], 1023
    bounds = _paged_kv_split_bounds(
        q_lens, kv_lens, 32, True, tile, sms, 0, 128, 2, window_left
    )
    begin = (kv_lens[0] - q_lens[0] - window_left) // tile
    visible = (kv_lens[0] - q_lens[0]) // tile
    assert bounds is not None and len(bounds[0]) > 2
    assert bounds[0][0] == begin and bounds[0][-1] == kv_lens[0] // tile
    assert all(a < b for a, b in zip(bounds[0], bounds[0][1:], strict=False))
    assert max(bounds[0][1:-1]) <= visible
    # The unwindowed split of the same request spreads over the whole prefix.
    full = _paged_kv_split_bounds(q_lens, kv_lens, 32, True, tile, sms, 0, 128, 2)
    assert full[0][0] == 0 and full[0][1] < begin
    # A window covering every key leaves the split unchanged.
    assert (
        _paged_kv_split_bounds(
            q_lens, kv_lens, 32, True, tile, sms, 0, 128, 2, kv_lens[0]
        )
        == full
    )
    # Fixed split: cut at multiples of the step, inside the window only.
    fixed = _paged_kv_split_bounds(
        q_lens, kv_lens, 32, True, tile, sms, 512, 128, 2, window_left
    )
    assert fixed == [[begin, 248, 252, 256]]
    # A fresh causal prompt still has no prefix every row sees: no split.
    assert (
        _paged_kv_split_bounds([4096], [4096], 64, True, tile, sms, 0, 128, 8, 127)
        is None
    )
    # GPT-OSS decode-like rows with a 128-token window touch two tiles: they may split
    # those, but never reach outside the window.
    small = _paged_kv_split_bounds([4], [32768], 64, True, tile, sms, 0, 128, 4, 127)
    assert small is None or small[0][0] == (32768 - 4 - 127) // tile


def test_paged_wrapper_forwards_run_time_sinks_to_prims():
    from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper

    captured = []

    class _StubPrimsBackend:
        _page_size = 16

        def run_paged(self, q, k, v, out, **kwargs):
            captured.append(kwargs)
            return out

    wrapper = BatchPrefillWithPagedKVCacheWrapper.__new__(
        BatchPrefillWithPagedKVCacheWrapper
    )
    wrapper._kv_layout = "HND"
    wrapper._backend = "cute-dsl-prims"
    wrapper._batch_invariant = None
    wrapper._cached_q_data_type = torch.float8_e4m3fn
    wrapper._cached_kv_data_type = torch.float8_e4m3fn
    wrapper._cached_o_data_type = torch.bfloat16
    wrapper._qo_indptr_last = 2
    wrapper._window_left = 127
    wrapper._prims_backend = _StubPrimsBackend()

    q = torch.empty((2, 8, 64), dtype=torch.float8_e4m3fn)
    cache = torch.empty((4, 2, 1, 16, 64), dtype=torch.float8_e4m3fn)
    sinks = torch.zeros(8)
    wrapper.run(q, cache, sinks=sinks)
    assert captured[-1]["sinks"] is sinks
    wrapper.run(q, cache)
    assert captured[-1]["sinks"] is None
    with pytest.raises(NotImplementedError, match="run-time window_left"):
        wrapper.run(q, cache, window_left=-1)
