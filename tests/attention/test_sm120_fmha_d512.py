# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-side and SM120 end-to-end tests for head_dim=512 FP8 prefill FMHA.

Host-side tests (tile selection, config/SMEM validation, compile cache keys,
Split-KV planning) run on CPU without a GPU. SM120 GPU tests verify ragged and
paged HND (page_size in {1, 16, 32, 64, 128}) correctness, Gemma 4 GQA shapes,
GQA packing, chunked prefill, empty/all-masked rows, forced Split-KV, and
64-bit Q/O pointer offsets past the signed Int32 2^31 boundary.

Run on an SM120 GPU:
    pytest tests/attention/test_sm120_fmha_d512.py -v
"""

import math
import tracemalloc
from importlib.metadata import PackageNotFoundError, version

import pytest
import torch
from packaging.version import Version

import flashinfer
from types import SimpleNamespace

# The SM120 host module imports the optional CuTe DSL at module level.
pytest.importorskip("cutlass.cute", reason="nvidia-cutlass-dsl is not installed")

from flashinfer.attention.cute_dsl import sm120_fmha as sm120_fmha_mod
from flashinfer.attention.cute_dsl.sm120_fmha import (
    SM120PrimsBatchPrefillBackend,
    _kv_split_scratch,
    _paged_kv_split_bounds,
    _resolve_balanced_scheduler,
    _select_tile_sizes,
    _use_balanced_scheduler,
    _validate_hnd_paged_pool,
    sm120_fmha_fp8_paged_prefill,
    sm120_fmha_fp8_ragged_prefill,
)
from flashinfer.cute_dsl.availability import is_cute_dsl_experimental_available
from flashinfer.utils import ceil_div, get_compute_capability


def _has_required_cutlass_dsl() -> bool:
    try:
        installed_version = version("nvidia-cutlass-dsl")
    except PackageNotFoundError:
        return False
    return Version(installed_version) >= Version("4.7.0")


_HAS_SM120 = (
    torch.cuda.is_available()
    and get_compute_capability(torch.device("cuda")) == (12, 0)
    and is_cute_dsl_experimental_available()
    and _has_required_cutlass_dsl()
)

_requires_sm120 = pytest.mark.skipif(
    not _HAS_SM120,
    reason="requires SM120 GPU (compute capability 12.0) and nvidia-cutlass-dsl>=4.7.0",
)


# ---------------------------------------------------------------------------
# CPU-only unit tests (tile sizing, validation, SMEM budget, compile keys)
# ---------------------------------------------------------------------------


def test_select_tile_sizes_d512_and_legacy():
    for head_dim in (32, 64, 128, 256):
        assert _select_tile_sizes(head_dim) == (128, 128)
    assert _select_tile_sizes(512) == (64, 64)


def test_validate_config_accepts_d512(monkeypatch):
    monkeypatch.setattr(sm120_fmha_mod, "_check_cutlass_dsl_version", lambda: None)
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device=None: (12, 0)
    )
    backend = SM120PrimsBatchPrefillBackend(torch.device("cpu"))
    for out_dtype in (torch.float16, torch.bfloat16):
        backend._validate_config(
            q_dtype=torch.float8_e4m3fn,
            kv_dtype=torch.float8_e4m3fn,
            o_dtype=out_dtype,
            num_qo_heads=32,
            num_kv_heads=4,
            head_dim_qk=512,
            head_dim_vo=512,
        )

    with pytest.raises(ValueError, match="head dimensions"):
        backend._validate_config(
            q_dtype=torch.float8_e4m3fn,
            kv_dtype=torch.float8_e4m3fn,
            o_dtype=torch.bfloat16,
            num_qo_heads=32,
            num_kv_heads=4,
            head_dim_qk=512,
            head_dim_vo=256,
        )


def test_d512_can_implement_and_smem_budget():
    pytest.importorskip("cutlass.experimental")
    import cutlass

    from flashinfer.cute_dsl.attention.fmha.sm120.fmha_prefill_fp8_tma import (
        SM120FusedMultiHeadAttentionFP8ForwardTMA as Kernel,
    )

    # Valid D=512 ragged config (64x64).
    assert Kernel._can_implement(
        cutlass.Float8E4M3FN,
        cutlass.BFloat16,
        q_shape=(1, 1, 32, 512),
        k_shape=(1, 1, 4, 512),
        kv_tile=64,
        q_tile=64,
    )
    # Unsupported D=512 tile shapes must be rejected.
    for bad_q_tile, bad_kv_tile in ((128, 128), (128, 64), (64, 128)):
        assert not Kernel._can_implement(
            cutlass.Float8E4M3FN,
            cutlass.BFloat16,
            q_shape=(1, 1, 32, 512),
            k_shape=(1, 1, 4, 512),
            kv_tile=bad_kv_tile,
            q_tile=bad_q_tile,
        )

    # Valid D=512 paged configs, including page_size=128 > kv_tile=64.
    for page_size in (1, 16, 32, 64, 128):
        assert Kernel.can_implement_paged(
            cutlass.Float8E4M3FN,
            cutlass.BFloat16,
            q_shape=(1, 1, 32, 512),
            k_shape=(1, 1, 4, 512),
            kv_tile=64,
            q_tile=64,
            num_tokens_per_page=page_size,
        )
    for bad_page_size in (24, 48, 96):
        assert not Kernel.can_implement_paged(
            cutlass.Float8E4M3FN,
            cutlass.BFloat16,
            q_shape=(1, 1, 32, 512),
            k_shape=(1, 1, 4, 512),
            kv_tile=64,
            q_tile=64,
            num_tokens_per_page=bad_page_size,
        )

    # Verify Split-D attributes and dynamic SMEM budget (81,952 B <= 100,352 B).
    inst = Kernel(
        in_dtype=cutlass.Float8E4M3FN,
        out_dtype=cutlass.BFloat16,
        is_causal=True,
        head_tile=512,
        kv_tile=64,
        q_tile=64,
        use_paged_kv=True,
        num_tokens_per_page=128,
    )
    assert inst.split_d is True
    assert inst.num_warps == 12
    assert inst.num_compute_warps == 8
    assert inst.head_tile_per_warp == 256
    assert inst.kv_pipeline_stages == 2
    assert inst.tma_page_rows == 64
    assert inst.kv_page_chunks == 1
    assert inst.smem_elems == 65536
    assert inst.split_d_scratch_elems == 4096
    dyn_smem_bytes = (
        inst.smem_elems * cutlass.Float8E4M3FN.bytes
        + inst.split_d_scratch_elems * cutlass.Float32.bytes
        + inst.kv_pipeline_stages * 2 * cutlass.Int64.bytes
    )
    assert dyn_smem_bytes == 81952


def test_paged_kv_split_bounds_tile64():
    bounds = _paged_kv_split_bounds(
        q_lens=[16],
        kv_lens=[8192],
        num_qo_heads=32,
        causal=True,
        tile=64,
        num_ctas_per_wave=170,
        fixed_split_tokens=1024,
        max_chunks=16,
        gqa_pack_size=8,
        q_tile=64,
    )
    assert bounds is not None
    assert len(bounds) == 1
    req_bounds = bounds[0]
    assert len(req_bounds) == 9  # 8 chunks -> 9 boundary points
    assert req_bounds[0] == 0
    assert req_bounds[-1] == 128
    assert all(req_bounds[i] < req_bounds[i + 1] for i in range(len(req_bounds) - 1))


def test_d512_compile_cache_keys_distinct(monkeypatch):
    pytest.importorskip("cutlass.experimental")

    from flashinfer.cute_dsl.attention.fmha.sm120.compile import (
        compile_sm120_fmha_fp8_paged_kernel,
        compile_sm120_fmha_fp8_ragged_kernel,
    )
    from flashinfer.jit import cute_dsl_core

    observed = []

    def fake_build_and_load(module_name, kernel_name, compile_fn, **kwargs):
        observed.append((module_name, kernel_name))
        return kernel_name

    compile_sm120_fmha_fp8_ragged_kernel.cache_clear()
    compile_sm120_fmha_fp8_paged_kernel.cache_clear()
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device=None: (12, 0)
    )
    monkeypatch.setattr(
        cute_dsl_core, "build_and_load_cute_dsl_kernel", fake_build_and_load
    )

    k_rag_512 = compile_sm120_fmha_fp8_ragged_kernel(
        in_dtype=torch.float8_e4m3fn,
        out_dtype=torch.bfloat16,
        num_qo_heads=32,
        num_kv_heads=4,
        head_dim=512,
        is_causal=True,
        kv_tile=64,
        q_tile=64,
        device=torch.device("cpu"),
    )
    k_rag_256 = compile_sm120_fmha_fp8_ragged_kernel(
        in_dtype=torch.float8_e4m3fn,
        out_dtype=torch.bfloat16,
        num_qo_heads=32,
        num_kv_heads=4,
        head_dim=256,
        is_causal=True,
        kv_tile=128,
        q_tile=128,
        device=torch.device("cpu"),
    )
    k_pag_512_p128 = compile_sm120_fmha_fp8_paged_kernel(
        in_dtype=torch.float8_e4m3fn,
        out_dtype=torch.bfloat16,
        num_qo_heads=32,
        num_kv_heads=4,
        head_dim=512,
        is_causal=True,
        kv_tile=64,
        q_tile=64,
        num_tokens_per_page=128,
        device=torch.device("cpu"),
    )
    assert "_d512_" in k_rag_512 and "_kt64_qt64" in k_rag_512
    assert k_rag_512 != k_rag_256
    assert "_d512_" in k_pag_512_p128 and "_page128_" in k_pag_512_p128
    assert len(observed) == 3
    compile_sm120_fmha_fp8_ragged_kernel.cache_clear()
    compile_sm120_fmha_fp8_paged_kernel.cache_clear()


def test_d512_rejects_skip_softmax_threshold(monkeypatch):
    """Verify non-None skip_softmax_threshold is rejected on ragged, paged, and compile paths for D=512."""
    pytest.importorskip("cutlass.experimental")
    from flashinfer.cute_dsl.attention.fmha.sm120.compile import (
        compile_sm120_fmha_fp8_paged_kernel,
        compile_sm120_fmha_fp8_ragged_kernel,
    )

    monkeypatch.setattr(sm120_fmha_mod, "_check_cutlass_dsl_version", lambda: None)
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device=None: (12, 0)
    )

    q = torch.empty((4, 8, 512), dtype=torch.float8_e4m3fn)
    k = torch.empty((4, 2, 512), dtype=torch.float8_e4m3fn)
    v = torch.empty((4, 2, 512), dtype=torch.float8_e4m3fn)
    o = torch.empty((4, 8, 512), dtype=torch.bfloat16)
    cu = torch.tensor([0, 4], dtype=torch.int32)

    for bad_threshold in (0.0, 2.0, torch.zeros((1,), dtype=torch.float32)):
        with pytest.raises(NotImplementedError, match="skip_softmax_threshold"):
            sm120_fmha_fp8_ragged_prefill(
                q, k, v, o, cu, cu, skip_softmax_threshold=bad_threshold
            )

    k_pool = torch.empty((2, 2, 16, 512), dtype=torch.float8_e4m3fn)
    v_pool = torch.empty((2, 2, 16, 512), dtype=torch.float8_e4m3fn)
    bt = torch.tensor([[0, 1]], dtype=torch.int32)
    seqlens_kv = torch.tensor([16], dtype=torch.int32)
    for bad_threshold in (0.0, 2.0, torch.zeros((1,), dtype=torch.float32)):
        with pytest.raises(NotImplementedError, match="skip_softmax_threshold"):
            sm120_fmha_fp8_paged_prefill(
                q,
                k_pool,
                v_pool,
                o,
                bt,
                seqlens_kv,
                cu,
                skip_softmax_threshold=bad_threshold,
            )

    with pytest.raises(NotImplementedError, match="head_dim=512"):
        compile_sm120_fmha_fp8_ragged_kernel(
            in_dtype=torch.float8_e4m3fn,
            out_dtype=torch.bfloat16,
            num_qo_heads=8,
            num_kv_heads=2,
            head_dim=512,
            is_causal=False,
            kv_tile=64,
            q_tile=64,
            device=torch.device("cpu"),
            enable_skip_softmax=True,
        )
    with pytest.raises(NotImplementedError, match="head_dim=512"):
        compile_sm120_fmha_fp8_paged_kernel(
            in_dtype=torch.float8_e4m3fn,
            out_dtype=torch.bfloat16,
            num_qo_heads=8,
            num_kv_heads=2,
            head_dim=512,
            is_causal=False,
            kv_tile=64,
            q_tile=64,
            num_tokens_per_page=16,
            device=torch.device("cpu"),
            enable_skip_softmax=True,
        )


def test_validate_hnd_paged_pool_base_alignment_d512():
    """Verify 16-byte base-address validation for D=512 HND paged pools."""
    for page_size in (1, 16):
        combined = torch.empty((4, 2, 4, page_size, 512), dtype=torch.float8_e4m3fn)
        k_view, v_view = combined.unbind(dim=1)
        _validate_hnd_paged_pool("k_pool", k_view)
        _validate_hnd_paged_pool("v_pool", v_view)

        numel = 4 * 4 * page_size * 512
        raw = torch.empty(numel + 16, dtype=torch.float8_e4m3fn)
        misaligned = raw[1 : 1 + numel].view(4, 4, page_size, 512)
        with pytest.raises(ValueError, match="base address must be 16-byte aligned"):
            _validate_hnd_paged_pool("k_pool", misaligned)


def test_causal_balanced_scheduler_respects_cuda_grid_y_limit(monkeypatch):
    """Verify the scheduler fallback at CUDA's grid.y limit and that backend dispatch reuses the
    planned scheduler flag even if the environment changes after plan()."""
    pytest.importorskip("cutlass.experimental")
    from flashinfer.cute_dsl.attention.fmha import sm120 as sm120_mod

    monkeypatch.delenv("PRIMS_FMHA_DISABLE_BALANCED_SCHEDULING", raising=False)
    monkeypatch.setattr(sm120_fmha_mod, "_check_cutlass_dsl_version", lambda: None)
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device=None: (12, 0)
    )
    # _ensure_kernel probes CUDA Graph capture, which raises on CPU-only PyTorch.
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    # D=512, Q64, GQA 16:1 -> gqa_pack_size=16 -> 4 Q tokens per CTA.
    # 65,535 tiles (262,140 tokens) fits grid.y; 65,536 tiles (262,144 tokens) falls back to grid.x.
    assert _use_balanced_scheduler(
        True, max_seqlen_q=262_140, q_tile=64, gqa_pack_size=16
    )
    assert not _use_balanced_scheduler(
        True, max_seqlen_q=262_144, q_tile=64, gqa_pack_size=16
    )
    # Ragged (gqa_pack_size=1, Q64): 65,535 * 64 fits; 65,535 * 64 + 1 falls back.
    assert _use_balanced_scheduler(True, max_seqlen_q=65_535 * 64, q_tile=64)
    assert not _use_balanced_scheduler(True, max_seqlen_q=65_535 * 64 + 1, q_tile=64)
    # The unbalanced grid puts head groups on grid.y.
    with pytest.raises(ValueError, match="grid.y"):
        _resolve_balanced_scheduler(
            False, False, max_seqlen_q=1, q_tile=64, num_qo_heads=65_536
        )
    assert not _resolve_balanced_scheduler(
        None, False, max_seqlen_q=1, q_tile=64, num_qo_heads=65_536, gqa_pack_size=2
    )
    # An explicit balanced flag still has to fit its Q tiles on grid.y.
    with pytest.raises(ValueError, match="grid.y"):
        _resolve_balanced_scheduler(
            True, True, max_seqlen_q=65_536 * 64, q_tile=64, num_qo_heads=8
        )

    paged_calls = []
    ragged_calls = []
    monkeypatch.setattr(
        sm120_mod,
        "compile_sm120_fmha_fp8_paged_kernel",
        lambda **kwargs: paged_calls.append(kwargs) or (lambda *a, **kw: None),
    )
    monkeypatch.setattr(
        sm120_mod,
        "compile_sm120_fmha_fp8_ragged_kernel",
        lambda **kwargs: ragged_calls.append(kwargs) or (lambda *a, **kw: None),
    )

    backend = SM120PrimsBatchPrefillBackend(torch.device("cpu"))
    # Paged: 262,144-token causal D=512 GQA 16:1 -> warmup, _ensure_kernel, and dispatch agree on balanced=False.
    qo_long = torch.tensor([0, 262_144], dtype=torch.int32)
    kv_long = torch.tensor([262_144], dtype=torch.int32)
    bt = torch.zeros((1, 1), dtype=torch.int32)
    backend.plan_paged(
        qo_indptr=qo_long,
        qo_indptr_host=qo_long,
        seqlens_kv=kv_long,
        seqlens_kv_host=kv_long,
        block_tables=bt,
        q_dtype=torch.float8_e4m3fn,
        kv_dtype=torch.float8_e4m3fn,
        o_dtype=torch.bfloat16,
        num_qo_heads=16,
        num_kv_heads=1,
        head_dim_qk=512,
        head_dim_vo=512,
        page_size=16,
        causal=True,
        sm_scale=None,
        disable_split_kv=True,
    )
    assert backend._gqa_pack_size == 16
    assert backend._balanced_scheduler is False
    assert paged_calls[-1]["balanced_scheduler"] is False
    q_small = torch.empty((4, 16, 512), dtype=torch.float8_e4m3fn)
    o_small = torch.empty((4, 16, 512), dtype=torch.bfloat16)
    k_pool = torch.empty((1, 1, 16, 512), dtype=torch.float8_e4m3fn)
    v_pool = torch.empty((1, 1, 16, 512), dtype=torch.float8_e4m3fn)
    run_scales = dict(enable_pdl=False, q_scale=None, k_scale=None, v_scale=None)
    for return_lse in (False, True):
        backend.run_paged(
            q_small,
            k_pool,
            v_pool,
            o_small,
            return_lse=return_lse,
            lse=None,
            **run_scales,
        )
        assert paged_calls[-1]["balanced_scheduler"] is False
        assert paged_calls[-1]["with_lse"] is return_lse

    # Boundary that fits (262,140 tokens -> 65,535 Q tiles) retains balanced_scheduler=True.
    qo_fit = torch.tensor([0, 262_140], dtype=torch.int32)
    kv_fit = torch.tensor([262_140], dtype=torch.int32)
    backend.plan_paged(
        qo_indptr=qo_fit,
        qo_indptr_host=qo_fit,
        seqlens_kv=kv_fit,
        seqlens_kv_host=kv_fit,
        block_tables=bt,
        q_dtype=torch.float8_e4m3fn,
        kv_dtype=torch.float8_e4m3fn,
        o_dtype=torch.bfloat16,
        num_qo_heads=16,
        num_kv_heads=1,
        head_dim_qk=512,
        head_dim_vo=512,
        page_size=16,
        causal=True,
        sm_scale=None,
        disable_split_kv=True,
    )
    assert backend._balanced_scheduler is True
    assert paged_calls[-1]["balanced_scheduler"] is True
    # Changing the environment after plan() must not change the dispatched specialization.
    monkeypatch.setenv("PRIMS_FMHA_DISABLE_BALANCED_SCHEDULING", "1")
    backend.run_paged(
        q_small, k_pool, v_pool, o_small, return_lse=False, lse=None, **run_scales
    )
    assert paged_calls[-1]["balanced_scheduler"] is True
    # Direct callers without a planned flag still follow the environment.
    sm120_fmha_fp8_paged_prefill(
        q_small,
        k_pool,
        v_pool,
        o_small,
        bt,
        torch.tensor([16], dtype=torch.int32),
        torch.tensor([0, 4], dtype=torch.int32),
        is_causal=True,
        max_seqlen_q=262_140,
        gqa_pack_size=16,
    )
    assert paged_calls[-1]["balanced_scheduler"] is False
    monkeypatch.delenv("PRIMS_FMHA_DISABLE_BALANCED_SCHEDULING")

    # Ragged: 65,536 Q tiles (65,535 * 64 + 1 tokens) falls back to balanced_scheduler=False.
    qo_rag_long = torch.tensor([0, 65_535 * 64 + 1], dtype=torch.int32)
    backend.plan_ragged(
        qo_indptr=qo_rag_long,
        kv_indptr=qo_rag_long,
        qo_indptr_host=qo_rag_long,
        kv_indptr_host=qo_rag_long,
        q_dtype=torch.float8_e4m3fn,
        kv_dtype=torch.float8_e4m3fn,
        o_dtype=torch.bfloat16,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=512,
        head_dim_vo=512,
        causal=True,
        sm_scale=None,
    )
    assert backend._balanced_scheduler is False
    assert ragged_calls[-1]["balanced_scheduler"] is False
    q_rag = torch.empty((4, 8, 512), dtype=torch.float8_e4m3fn)
    kv_rag = torch.empty((4, 2, 512), dtype=torch.float8_e4m3fn)
    o_rag = torch.empty((4, 8, 512), dtype=torch.bfloat16)
    lse_rag = torch.empty((4, 8), dtype=torch.float32)
    for lse in (None, lse_rag):
        backend.run_ragged(q_rag, kv_rag, kv_rag, o_rag, lse=lse, **run_scales)
        assert ragged_calls[-1]["balanced_scheduler"] is False
        assert ragged_calls[-1]["with_lse"] is (lse is not None)

    qo_rag_fit = torch.tensor([0, 65_535 * 64], dtype=torch.int32)
    backend.plan_ragged(
        qo_indptr=qo_rag_fit,
        kv_indptr=qo_rag_fit,
        qo_indptr_host=qo_rag_fit,
        kv_indptr_host=qo_rag_fit,
        q_dtype=torch.float8_e4m3fn,
        kv_dtype=torch.float8_e4m3fn,
        o_dtype=torch.bfloat16,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=512,
        head_dim_vo=512,
        causal=True,
        sm_scale=None,
    )
    assert backend._balanced_scheduler is True
    monkeypatch.setenv("PRIMS_FMHA_DISABLE_BALANCED_SCHEDULING", "1")
    backend.run_ragged(q_rag, kv_rag, kv_rag, o_rag, lse=None, **run_scales)
    assert ragged_calls[-1]["balanced_scheduler"] is True


def test_split_kv_respects_cuda_grid_z_limit(monkeypatch):
    """Verify Split-KV workspace sizing and planning enforce CUDA's grid.z <= 65,535 limit."""
    monkeypatch.setattr(sm120_fmha_mod, "_check_cutlass_dsl_version", lambda: None)
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device=None: (12, 0)
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device=None: SimpleNamespace(multi_processor_count=170),
    )

    # 8,192 requests x 262,144 KV tokens need 4,096 chunks each. The rejection counts them
    # without materializing the ~33M cut points.
    tracemalloc.start()
    try:
        with pytest.raises(ValueError, match="increase fixed_split_size"):
            _paged_kv_split_bounds(
                [1] * 8192,
                [262_144] * 8192,
                1,
                False,
                64,
                170,
                16,
                1 << 30,
                1,
                q_tile=64,
            )
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 16 << 20
    # Accepted splits keep their boundaries: cuts every 2 tiles, padded with empty ranges.
    assert _paged_kv_split_bounds(
        [1, 1], [300, 130], 1, False, 64, 170, 128, 1 << 30, 1, q_tile=64
    ) == [[0, 2, 4, 5], [0, 2, 3, 3]]

    # 128 noncausal requests x 32,768 KV tokens, D=512 (tile=64), page_size=16:
    # fixed_split_size=1 -> 512 chunks -> 128 * 512 = 65,536 > 65,535 -> rejected.
    q_lens = [1] * 128
    kv_lens = [32_768] * 128
    with pytest.raises(ValueError, match="increase fixed_split_size"):
        _paged_kv_split_bounds(
            q_lens, kv_lens, 1, False, 64, 170, 16, 1 << 30, 1, q_tile=64
        )

    backend = SM120PrimsBatchPrefillBackend(torch.device("cpu"))
    qo_indptr = torch.arange(0, 129, dtype=torch.int32)
    seqlens_kv = torch.full((128,), 32_768, dtype=torch.int32)
    with pytest.raises(ValueError, match="increase fixed_split_size"):
        backend.paged_workspace_size(
            qo_indptr_host=qo_indptr,
            seqlens_kv_host=seqlens_kv,
            num_qo_heads=1,
            num_kv_heads=1,
            head_dim=512,
            o_dtype=torch.bfloat16,
            page_size=16,
            causal=False,
            fixed_split_size=1,
            disable_split_kv=False,
            use_cuda_graph=False,
        )

    # A 2-tile split (fixed_split_size=8 pages = 128 tokens) produces 256 chunks -> 32,768 <= 65,535.
    ws_bytes = backend.paged_workspace_size(
        qo_indptr_host=qo_indptr,
        seqlens_kv_host=seqlens_kv,
        num_qo_heads=1,
        num_kv_heads=1,
        head_dim=512,
        o_dtype=torch.bfloat16,
        page_size=16,
        causal=False,
        fixed_split_size=8,
        disable_split_kv=False,
        use_cuda_graph=False,
    )
    assert ws_bytes > 0

    # 16,384 requests leave room for 65,535 // 16,384 = 3 chunks each. One active Q request with
    # long, equal K/V lengths would otherwise take the 128-chunk maximum.
    batch = 16_384
    auto_args = ([1] + [0] * (batch - 1), [32_768] * batch, 1, False, 64, 170, 0)
    bounds = _paged_kv_split_bounds(*auto_args, 1 << 30, 1, q_tile=64)
    assert len(bounds[0]) - 1 == 3
    with monkeypatch.context() as m:
        m.setattr(sm120_fmha_mod, "_CUDA_MAX_GRID_Z", 1 << 31)
        bounds = _paged_kv_split_bounds(*auto_args, 1 << 30, 1, q_tile=64)
        assert len(bounds[0]) - 1 == 128
    auto_ws = backend.paged_workspace_size(
        qo_indptr_host=torch.tensor([0] + [1] * batch, dtype=torch.int32),
        seqlens_kv_host=torch.full((batch,), 32_768, dtype=torch.int32),
        num_qo_heads=1,
        num_kv_heads=1,
        head_dim=512,
        o_dtype=torch.bfloat16,
        page_size=16,
        causal=False,
        fixed_split_size=-1,
        disable_split_kv=False,
        use_cuda_graph=False,
    )
    assert auto_ws == _kv_split_scratch(1, 1, 512, torch.bfloat16, 3)[1]
    # Workspace sizing rejects a batch the launch grid cannot hold, even without a split.
    with pytest.raises(ValueError, match="grid.z"):
        backend.paged_workspace_size(
            qo_indptr_host=torch.zeros(65_537, dtype=torch.int32),
            seqlens_kv_host=torch.zeros(65_536, dtype=torch.int32),
            num_qo_heads=1,
            num_kv_heads=1,
            head_dim=512,
            o_dtype=torch.bfloat16,
            page_size=16,
            causal=False,
            fixed_split_size=-1,
            disable_split_kv=True,
            use_cuda_graph=False,
        )
    # Batch size > 32,767 cannot even split into 2 chunks without exceeding grid.z.
    assert (
        _paged_kv_split_bounds(
            [1] * 40_000, [4096] * 40_000, 1, False, 64, 170, 0, 1 << 30, 1, q_tile=64
        )
        is None
    )


def _stub_sm120_compile(monkeypatch):
    """Patch the SM120 checks and kernel compilers so host dispatch runs on CPU."""
    from flashinfer.cute_dsl.attention.fmha import sm120 as sm120_mod

    monkeypatch.setattr(sm120_fmha_mod, "_check_cutlass_dsl_version", lambda: None)
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device=None: (12, 0)
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device=None: SimpleNamespace(multi_processor_count=170),
    )
    calls = []
    for name in (
        "compile_sm120_fmha_fp8_paged_kernel",
        "compile_sm120_fmha_fp8_ragged_kernel",
    ):
        monkeypatch.setattr(
            sm120_mod,
            name,
            lambda **kwargs: calls.append(kwargs) or (lambda *a, **kw: None),
        )
    return calls


def test_paged_metadata_consistency_and_empty_batch(monkeypatch):
    """Verify paged batch metadata must agree and an empty batch returns without a launch."""
    pytest.importorskip("cutlass.experimental")
    calls = _stub_sm120_compile(monkeypatch)
    q = torch.empty((4, 8, 512), dtype=torch.float8_e4m3fn)
    o = torch.empty((4, 8, 512), dtype=torch.bfloat16)
    pool = torch.empty((1, 2, 16, 512), dtype=torch.float8_e4m3fn)
    bt = torch.zeros((2, 1), dtype=torch.int32)
    kv = torch.tensor([16, 16], dtype=torch.int32)
    cu_q = torch.tensor([0, 2, 4], dtype=torch.int32)

    with pytest.raises(ValueError, match="cu_seqlens_q"):
        sm120_fmha_fp8_paged_prefill(q, pool, pool, o, bt, kv, cu_q[:2])
    with pytest.raises(ValueError, match="seqlens_kv"):
        sm120_fmha_fp8_paged_prefill(q, pool, pool, o, bt, kv[:1], cu_q)
    sm120_fmha_fp8_paged_prefill(q, pool, pool, o, bt, kv, cu_q)
    assert len(calls) == 1

    # B=0: no launch, and Split-KV ranges are not divided by the empty batch.
    empty_q = q[:0]
    lse = torch.empty((0, 8), dtype=torch.float32)
    sm120_fmha_fp8_paged_prefill(
        empty_q,
        pool,
        pool,
        o[:0],
        bt[:0],
        kv[:0],
        cu_q[:1],
        lse=lse,
        kv_tile_ranges=torch.empty((0, 2), dtype=torch.int32),
    )
    sm120_fmha_fp8_ragged_prefill(empty_q, q, q, o[:0], cu_q[:1], cu_q[:1], lse=lse)
    assert len(calls) == 1

    backend = SM120PrimsBatchPrefillBackend(torch.device("cpu"))
    plan = dict(
        qo_indptr=cu_q[:1],
        qo_indptr_host=cu_q[:1],
        seqlens_kv=kv[:0],
        seqlens_kv_host=kv[:0],
        block_tables=bt[:0],
        q_dtype=torch.float8_e4m3fn,
        kv_dtype=torch.float8_e4m3fn,
        o_dtype=torch.bfloat16,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=512,
        head_dim_vo=512,
        page_size=16,
        causal=True,
        sm_scale=None,
        float_workspace_buffer=torch.empty(1 << 20, dtype=torch.uint8),
    )
    backend.plan_paged(**plan)
    assert backend._max_seqlen_q == 0 and backend._kv_split is None
    with pytest.raises(ValueError, match="block_tables rows"):
        backend.plan_paged(**{**plan, "block_tables": bt})
    backend.plan_ragged(
        qo_indptr=cu_q[:1],
        kv_indptr=cu_q[:1],
        qo_indptr_host=cu_q[:1],
        kv_indptr_host=cu_q[:1],
        q_dtype=torch.float8_e4m3fn,
        kv_dtype=torch.float8_e4m3fn,
        o_dtype=torch.bfloat16,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=512,
        head_dim_vo=512,
        causal=True,
        sm_scale=None,
    )
    assert backend._max_seqlen_q == 0


@pytest.mark.parametrize(
    "sm_scale", [0.0, -0.05, float("nan"), float("inf"), 1e-50, 3.63, 1e39]
)
def test_rejects_non_positive_sm_scale(monkeypatch, sm_scale):
    """Verify sm_scale * log2(e) must be finite and positive in FP32: the online softmax takes
    the row max of unscaled logits, and a zero scale turns masked -inf logits into NaN."""
    pytest.importorskip("cutlass.experimental")
    calls = _stub_sm120_compile(monkeypatch)
    q = torch.empty((4, 8, 512), dtype=torch.float8_e4m3fn)
    kv = torch.empty((4, 2, 512), dtype=torch.float8_e4m3fn)
    o = torch.empty((4, 8, 512), dtype=torch.bfloat16)
    cu = torch.tensor([0, 4], dtype=torch.int32)
    pool = torch.empty((1, 2, 16, 512), dtype=torch.float8_e4m3fn)
    bt = torch.zeros((1, 1), dtype=torch.int32)
    with pytest.raises(ValueError, match="sm_scale"):
        sm120_fmha_fp8_ragged_prefill(q, kv, kv, o, cu, cu, sm_scale=sm_scale)
    with pytest.raises(ValueError, match="sm_scale"):
        sm120_fmha_fp8_paged_prefill(
            q, pool, pool, o, bt, cu[1:], cu, sm_scale=sm_scale
        )
    assert not calls


def test_paged_rejects_empty_kv_tile_ranges_and_skips_empty_q(monkeypatch):
    """Verify a non-empty batch needs at least one K/V chunk, and zero Q tokens launch nothing."""
    pytest.importorskip("cutlass.experimental")
    calls = _stub_sm120_compile(monkeypatch)
    pool = torch.empty((1, 2, 16, 512), dtype=torch.float8_e4m3fn)
    bt = torch.zeros((2, 1), dtype=torch.int32)
    kv_lens = torch.tensor([16, 16], dtype=torch.int32)
    q = torch.empty((4, 8, 512), dtype=torch.float8_e4m3fn)
    o = torch.empty((4, 8, 512), dtype=torch.bfloat16)
    cu = torch.tensor([0, 2, 4], dtype=torch.int32)
    with pytest.raises(ValueError, match="num_chunks >= 1"):
        sm120_fmha_fp8_paged_prefill(
            q,
            pool,
            pool,
            o[:0],
            bt,
            kv_lens,
            cu,
            kv_tile_ranges=torch.empty((0, 2), dtype=torch.int32),
        )
    empty_q = torch.empty((0, 8, 512), dtype=torch.float8_e4m3fn)
    empty_o = torch.empty((0, 8, 512), dtype=torch.bfloat16)
    zero_cu = torch.zeros(3, dtype=torch.int32)
    sm120_fmha_fp8_paged_prefill(empty_q, pool, pool, empty_o, bt, kv_lens, zero_cu)
    sm120_fmha_fp8_ragged_prefill(empty_q, q[:, :2], q[:, :2], empty_o, zero_cu, cu)
    assert not calls


# ---------------------------------------------------------------------------
# Reference and data-generation helpers for SM120 GPU tests
# ---------------------------------------------------------------------------


def _make_fp8(shape, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randint(
        -2, 3, shape, dtype=torch.float32, device="cuda", generator=g
    ).to(torch.float8_e4m3fn)


def _ref_attention(q_b, k_b, v_b, sm_scale, is_causal, v_scale=1.0):
    """Exact FP32 reference for one request; returns (O, LSE_log2)."""
    sq, hq, d = q_b.shape
    skv, hkv, _ = k_b.shape
    if sq == 0:
        return (
            torch.empty((0, hq, d), dtype=torch.float32, device=q_b.device),
            torch.empty((0, hq), dtype=torch.float32, device=q_b.device),
        )
    if skv == 0:
        return (
            torch.zeros((sq, hq, d), dtype=torch.float32, device=q_b.device),
            torch.full((sq, hq), float("-inf"), dtype=torch.float32, device=q_b.device),
        )
    qf = q_b.float().transpose(0, 1)  # (Hq, sq, D)
    kf = k_b.float().transpose(0, 1)  # (Hkv, skv, D)
    vf = v_b.float().transpose(0, 1)
    if hq != hkv:
        kf = kf.repeat_interleave(hq // hkv, dim=0)
        vf = vf.repeat_interleave(hq // hkv, dim=0)
    scores = torch.einsum("hqd,hkd->hqk", qf, kf) * sm_scale
    if is_causal:
        q_offset = skv - sq
        q_pos = torch.arange(sq, device=q_b.device)[:, None] + q_offset
        k_pos = torch.arange(skv, device=q_b.device)[None, :]
        scores = scores.masked_fill(k_pos > q_pos, float("-inf"))
    all_masked = torch.isneginf(scores).all(dim=-1)  # (Hq, sq)
    safe_scores = scores.masked_fill(all_masked.unsqueeze(-1), 0.0)
    probs = torch.softmax(safe_scores, dim=-1)
    probs = probs.masked_fill(all_masked.unsqueeze(-1), 0.0)
    out = torch.einsum("hqk,hkd->hqd", probs, vf).transpose(0, 1) * v_scale
    lse = torch.logsumexp(scores, dim=-1).transpose(0, 1) * math.log2(math.e)
    return out, lse


@_requires_sm120
@pytest.mark.parametrize("bound", ["at", "above"])
def test_sm120_d512_scale_upper_bound(bound):
    """Verify the largest accepted scale still returns V for a single key with logit 511.

    The softmax subtracts the row max after scaling in FP32; far above the bound (for example
    sm_scale=1e8) that rounding makes every weight underflow and the output silently zero.
    The bound does not cover FP8 P saturation; see the large-logit xfail below.
    """
    max_scale = sm120_fmha_mod._max_scale_log2(512) / math.log2(math.e)
    sm_scale = max_scale * (1 - 1e-6 if bound == "at" else 1 + 1e-6)
    h, d = 4, 512
    q = torch.ones((1, h, d), device="cuda").to(torch.float8_e4m3fn)
    k = torch.ones((1, h, d), device="cuda")
    k[..., 0] = 0  # q . k = 511
    k = k.to(torch.float8_e4m3fn)
    v = _make_fp8((1, h, d), seed=1)
    o = torch.empty((1, h, d), dtype=torch.bfloat16, device="cuda")
    lse = torch.empty((1, h), dtype=torch.float32, device="cuda")
    cu = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    run = lambda: sm120_fmha_fp8_ragged_prefill(  # noqa: E731
        q, k, v, o, cu, cu, sm_scale=sm_scale, lse=lse
    )
    if bound == "above":
        with pytest.raises(ValueError, match="at most 3.621 for head_dim=512"):
            run()
        return
    run()
    torch.testing.assert_close(o.float(), v.float(), atol=0, rtol=0)
    torch.testing.assert_close(
        lse, torch.full_like(lse, 511 * sm_scale * math.log2(math.e)), atol=0, rtol=1e-5
    )


@_requires_sm120
@pytest.mark.parametrize("page_size", [None, 16])
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="Known limitation shared with the D<=256 SM120 FP8 kernels: the softmax "
    "subtracts the unscaled row max after scaling in FP32, and for this large positive "
    "logit the rounding pushes the pre-scaled weight past FP8 P's range (O = 0.219, not 1).",
)
def test_sm120_d512_large_logit_saturates_fp8_p(page_size):
    """Document that a single key with every Q/K component 384 (logit 75,497,472) at
    sm_scale=1 does not return V."""
    h, d = 4, 512
    q = torch.full((1, h, d), 384.0, device="cuda").to(torch.float8_e4m3fn)
    v = torch.ones((1, h, d), device="cuda").to(torch.float8_e4m3fn)
    o = torch.empty((1, h, d), dtype=torch.bfloat16, device="cuda")
    cu = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    if page_size is None:
        sm120_fmha_fp8_ragged_prefill(q, q, v, o, cu, cu, sm_scale=1.0)
    else:
        k_pool, v_pool, _, block_tables, *_ = _make_varlen_paged_kv([q], [v], page_size)
        sm120_fmha_fp8_paged_prefill(
            q,
            k_pool,
            v_pool,
            o,
            block_tables,
            torch.tensor([1], dtype=torch.int32, device="cuda"),
            cu,
            sm_scale=1.0,
        )
    torch.testing.assert_close(o.float(), v.float(), atol=1e-2, rtol=1e-2)


@_requires_sm120
@pytest.mark.parametrize("page_size", [None, 16])
def test_sm120_d512_subnormal_scale_matches_reference(page_size):
    """Verify a positive scale that is subnormal in FP32 (1e-39) is accepted and correct."""
    hq, hkv, d, q_len, kv_len = 8, 2, 512, 4, 100
    q = _make_fp8((q_len, hq, d), seed=2)
    k = _make_fp8((kv_len, hkv, d), seed=3)
    v = _make_fp8((kv_len, hkv, d), seed=4)
    ref_o, ref_lse = _ref_attention(q, k, v, 1e-39, False)
    o = torch.empty((q_len, hq, d), dtype=torch.bfloat16, device="cuda")
    lse = torch.empty((q_len, hq), dtype=torch.float32, device="cuda")
    cu_q = torch.tensor([0, q_len], dtype=torch.int32, device="cuda")
    if page_size is None:
        cu_k = torch.tensor([0, kv_len], dtype=torch.int32, device="cuda")
        sm120_fmha_fp8_ragged_prefill(q, k, v, o, cu_q, cu_k, sm_scale=1e-39, lse=lse)
    else:
        k_pool, v_pool, _, block_tables, *_ = _make_varlen_paged_kv([k], [v], page_size)
        sm120_fmha_fp8_paged_prefill(
            q,
            k_pool,
            v_pool,
            o,
            block_tables,
            torch.tensor([kv_len], dtype=torch.int32, device="cuda"),
            cu_q,
            sm_scale=1e-39,
            lse=lse,
        )
    torch.testing.assert_close(o.float(), ref_o, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(lse, ref_lse, atol=1e-4, rtol=1e-5)


@_requires_sm120
@pytest.mark.parametrize("num_chunks", [None, 2])
def test_sm120_d512_paged_zero_page_pool(num_chunks):
    """Verify a pool without pages (all K/V empty) writes O=0 and LSE=-inf for nonempty Q."""
    hq, hkv, d, b = 8, 2, 512, 2
    q = _make_fp8((5, hq, d))
    pool = torch.empty((0, hkv, 16, d), dtype=torch.float8_e4m3fn, device="cuda")
    rows = 5 * (num_chunks or 1)
    o = torch.full((rows, hq, d), 7.0, dtype=torch.bfloat16, device="cuda")
    lse = torch.zeros((rows, hq), dtype=torch.float32, device="cuda")
    ranges = (
        None
        if num_chunks is None
        else torch.zeros((num_chunks * b, 2), dtype=torch.int32, device="cuda")
    )
    sm120_fmha_fp8_paged_prefill(
        q,
        pool,
        pool,
        o,
        torch.empty((b, 0), dtype=torch.int32, device="cuda"),
        torch.zeros(b, dtype=torch.int32, device="cuda"),
        torch.tensor([0, 3, 5], dtype=torch.int32, device="cuda"),
        lse=lse,
        kv_tile_ranges=ranges,
    )
    torch.cuda.synchronize()
    assert torch.equal(o, torch.zeros_like(o))
    assert torch.isneginf(lse).all()


def _make_varlen_paged_kv(k_list, v_list, page_size, seed=0):
    """Pack per-request K/V lists into a shuffled combined HND paged pool."""
    hkv, d = k_list[0].shape[1], k_list[0].shape[2]
    pages_per_req = [ceil_div(k.shape[0], page_size) for k in k_list]
    max_pages = max(max(pages_per_req, default=1), 1)
    total_pages = max(sum(pages_per_req), 1)

    g = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(total_pages, generator=g).tolist()

    kv_pool = torch.zeros(
        (total_pages, 2, hkv, page_size, d),
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    block_tables = torch.zeros(
        (len(k_list), max_pages), dtype=torch.int32, device="cuda"
    )
    page_indptr = [0]
    page_indices = []
    last_page_len = []

    cursor = 0
    for req_idx, (k_b, v_b) in enumerate(zip(k_list, v_list, strict=True)):
        skv = k_b.shape[0]
        n_pages = ceil_div(skv, page_size)
        for p in range(n_pages):
            phys = perm[cursor]
            cursor += 1
            block_tables[req_idx, p] = phys
            page_indices.append(phys)
            tok0 = p * page_size
            tok1 = min(tok0 + page_size, skv)
            kv_pool[phys, 0, :, : tok1 - tok0, :].copy_(k_b[tok0:tok1].transpose(0, 1))
            kv_pool[phys, 1, :, : tok1 - tok0, :].copy_(v_b[tok0:tok1].transpose(0, 1))
        page_indptr.append(len(page_indices))
        last_page_len.append(0 if skv == 0 else ((skv - 1) % page_size) + 1)

    k_pool, v_pool = kv_pool.unbind(dim=1)
    return (
        k_pool,
        v_pool,
        kv_pool,
        block_tables,
        torch.tensor(page_indptr, dtype=torch.int32, device="cuda"),
        torch.tensor(page_indices, dtype=torch.int32, device="cuda"),
        torch.tensor(last_page_len, dtype=torch.int32, device="cuda"),
    )


# ---------------------------------------------------------------------------
# SM120 GPU Correctness Tests
# ---------------------------------------------------------------------------


@_requires_sm120
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("page_size", [None, 16, 128])
def test_sm120_d512_structured_exact_tight_tolerance(is_causal, page_size):
    """Structured low-entropy inputs verify Split-D half-sum & lane mapping tightly."""
    d = 512
    hq, hkv = 8, 2
    sq, skv = 130, 193
    sm_scale = 1.0 / math.sqrt(d)

    # Distinct patterns in D[0:256] and D[256:512] catch any missing/duplicated half-sum.
    q = torch.zeros((sq, hq, d), dtype=torch.float32, device="cuda")
    k = torch.zeros((skv, hkv, d), dtype=torch.float32, device="cuda")
    v = torch.zeros((skv, hkv, d), dtype=torch.float32, device="cuda")
    q[:, :, :256] = 0.25
    q[:, :, 256:] = -0.125
    k[:, :, :256] = (torch.arange(skv, device="cuda") % 3 - 1).float()[
        :, None, None
    ] * 0.25
    k[:, :, 256:] = (torch.arange(skv, device="cuda") % 5 - 2).float()[
        :, None, None
    ] * 0.25
    v[:, :, :256] = ((torch.arange(skv, device="cuda") % 7) - 3).float()[
        :, None, None
    ] * 0.25
    v[:, :, 256:] = ((torch.arange(skv, device="cuda") % 5) - 2).float()[
        :, None, None
    ] * 0.5
    q_f8, k_f8, v_f8 = (
        q.to(torch.float8_e4m3fn),
        k.to(torch.float8_e4m3fn),
        v.to(torch.float8_e4m3fn),
    )

    o = torch.empty((sq, hq, d), dtype=torch.bfloat16, device="cuda")
    lse = torch.empty((sq, hq), dtype=torch.float32, device="cuda")
    cu_q = torch.tensor([0, sq], dtype=torch.int32, device="cuda")

    if page_size is None:
        cu_k = torch.tensor([0, skv], dtype=torch.int32, device="cuda")
        sm120_fmha_fp8_ragged_prefill(
            q_f8,
            k_f8,
            v_f8,
            o,
            cu_q,
            cu_k,
            is_causal=is_causal,
            sm_scale=sm_scale,
            lse=lse,
        )
    else:
        k_pool, v_pool, _, bt, _, _, _ = _make_varlen_paged_kv(
            [k_f8], [v_f8], page_size, seed=7
        )
        seqlens_kv = torch.tensor([skv], dtype=torch.int32, device="cuda")
        sm120_fmha_fp8_paged_prefill(
            q_f8,
            k_pool,
            v_pool,
            o,
            bt,
            seqlens_kv,
            cu_q,
            is_causal=is_causal,
            sm_scale=sm_scale,
            lse=lse,
        )

    ref_o, ref_lse = _ref_attention(q_f8, k_f8, v_f8, sm_scale, is_causal)
    torch.testing.assert_close(o.float(), ref_o, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(lse, ref_lse, atol=2e-3, rtol=2e-3)


@_requires_sm120
@pytest.mark.parametrize(
    "hq,hkv",
    [
        (8, 8),  # MHA
        (8, 2),  # Gemma 4 E4B (G=4)
        (16, 2),  # Gemma 4 26B-A4B (G=8)
        (32, 4),  # Gemma 4 31B (G=8)
        (16, 1),  # Gemma 4 12B (G=16)
    ],
)
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
def test_sm120_d512_ragged_gemma4_shapes(hq, hkv, is_causal, out_dtype):
    """Variable-length ragged D=512 across Gemma 4 GQA shapes and non-64-multiple lengths."""
    d = 512
    q_lens = [1, 15, 63, 64, 65, 130]
    kv_lens = [31, 64, 63, 127, 129, 255]
    sm_scale = 1.0 / math.sqrt(d)

    q_list = [_make_fp8((sq, hq, d), seed=10 + i) for i, sq in enumerate(q_lens)]
    k_list = [_make_fp8((sk, hkv, d), seed=20 + i) for i, sk in enumerate(kv_lens)]
    v_list = [_make_fp8((sk, hkv, d), seed=30 + i) for i, sk in enumerate(kv_lens)]

    q = torch.cat(q_list, dim=0)
    k = torch.cat(k_list, dim=0)
    v = torch.cat(v_list, dim=0)
    cu_q = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(q_lens), 0).tolist()),
        dtype=torch.int32,
        device="cuda",
    )
    cu_k = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(kv_lens), 0).tolist()),
        dtype=torch.int32,
        device="cuda",
    )

    o = torch.empty((q.shape[0], hq, d), dtype=out_dtype, device="cuda")
    lse = torch.empty((q.shape[0], hq), dtype=torch.float32, device="cuda")
    sm120_fmha_fp8_ragged_prefill(
        q, k, v, o, cu_q, cu_k, is_causal=is_causal, sm_scale=sm_scale, lse=lse
    )

    refs, lses = [], []
    for qb, kb, vb in zip(q_list, k_list, v_list, strict=True):
        ro, rl = _ref_attention(qb, kb, vb, sm_scale, is_causal)
        refs.append(ro)
        lses.append(rl)
    torch.testing.assert_close(o.float(), torch.cat(refs), atol=0.2, rtol=0.2)
    torch.testing.assert_close(lse, torch.cat(lses), atol=2e-3, rtol=2e-3)


@_requires_sm120
@pytest.mark.parametrize("page_size", [1, 16, 32, 64, 128])
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("gqa_pack_size", [1, 8])
def test_sm120_d512_paged_all_page_sizes_and_gqa_pack(
    page_size, is_causal, gqa_pack_size
):
    """Paged HND D=512 across page_size in {1, 16, 32, 64, 128} and GQA packing."""
    d = 512
    hq, hkv = 32, 4
    q_lens = [1, 16, 17, 65, 127]
    kv_lens = [1, 63, 128, 193, 384]
    sm_scale = 1.0 / math.sqrt(d)

    q_list = [_make_fp8((sq, hq, d), seed=100 + i) for i, sq in enumerate(q_lens)]
    k_list = [_make_fp8((sk, hkv, d), seed=200 + i) for i, sk in enumerate(kv_lens)]
    v_list = [_make_fp8((sk, hkv, d), seed=300 + i) for i, sk in enumerate(kv_lens)]

    q = torch.cat(q_list, dim=0)
    cu_q = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(q_lens), 0).tolist()),
        dtype=torch.int32,
        device="cuda",
    )
    seqlens_kv = torch.tensor(kv_lens, dtype=torch.int32, device="cuda")
    k_pool, v_pool, _, bt, _, _, _ = _make_varlen_paged_kv(
        k_list, v_list, page_size, seed=42
    )

    o = torch.empty((q.shape[0], hq, d), dtype=torch.bfloat16, device="cuda")
    lse = torch.empty((q.shape[0], hq), dtype=torch.float32, device="cuda")
    sm120_fmha_fp8_paged_prefill(
        q,
        k_pool,
        v_pool,
        o,
        bt,
        seqlens_kv,
        cu_q,
        is_causal=is_causal,
        sm_scale=sm_scale,
        lse=lse,
        gqa_pack_size=gqa_pack_size,
    )

    refs, lses = [], []
    for qb, kb, vb in zip(q_list, k_list, v_list, strict=True):
        ro, rl = _ref_attention(qb, kb, vb, sm_scale, is_causal)
        refs.append(ro)
        lses.append(rl)
    torch.testing.assert_close(o.float(), torch.cat(refs), atol=0.2, rtol=0.2)
    torch.testing.assert_close(lse, torch.cat(lses), atol=2e-3, rtol=2e-3)


@_requires_sm120
@pytest.mark.parametrize("is_causal", [False, True])
def test_sm120_d512_empty_requests_and_all_masked_rows(is_causal):
    """Verify empty requests (qo=0, kv=0) and causal all-masked rows (qo > kv) yield O=0, LSE=-inf."""
    d = 512
    hq, hkv = 16, 2
    sm_scale = 1.0 / math.sqrt(d)

    # Req 0: qo=0, kv=64; Req 1: qo=17, kv=0; Req 2: qo=80, kv=30 (rows 0..49 all-masked if causal);
    # Req 3: qo=65, kv=130 (normal chunked prefill).
    q_lens = [0, 17, 80, 65]
    kv_lens = [64, 0, 30, 130]
    q_list = [_make_fp8((sq, hq, d), seed=500 + i) for i, sq in enumerate(q_lens)]
    k_list = [_make_fp8((sk, hkv, d), seed=600 + i) for i, sk in enumerate(kv_lens)]
    v_list = [_make_fp8((sk, hkv, d), seed=700 + i) for i, sk in enumerate(kv_lens)]

    q = torch.cat(q_list, dim=0)
    k = torch.cat(k_list, dim=0)
    v = torch.cat(v_list, dim=0)
    cu_q = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(q_lens), 0).tolist()),
        dtype=torch.int32,
        device="cuda",
    )
    cu_k = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(kv_lens), 0).tolist()),
        dtype=torch.int32,
        device="cuda",
    )

    o = torch.full(
        (q.shape[0], hq, d), float("nan"), dtype=torch.bfloat16, device="cuda"
    )
    lse = torch.full((q.shape[0], hq), float("nan"), dtype=torch.float32, device="cuda")
    sm120_fmha_fp8_ragged_prefill(
        q, k, v, o, cu_q, cu_k, is_causal=is_causal, sm_scale=sm_scale, lse=lse
    )

    refs, lses = [], []
    for qb, kb, vb in zip(q_list, k_list, v_list, strict=True):
        ro, rl = _ref_attention(qb, kb, vb, sm_scale, is_causal)
        refs.append(ro)
        lses.append(rl)
    ref_o = torch.cat(refs)
    ref_lse = torch.cat(lses)

    assert not torch.isnan(o).any()
    assert not torch.isnan(lse).any()
    # Compare finite LSE entries tightly, verify -inf mask matches identically,
    # and require exact zero outputs on empty / fully-masked rows.
    inf_mask = torch.isneginf(ref_lse)
    assert torch.equal(torch.isneginf(lse), inf_mask)
    if inf_mask.any():
        assert torch.equal(o[inf_mask], torch.zeros_like(o[inf_mask]))
    if (~inf_mask).any():
        torch.testing.assert_close(
            o[~inf_mask].float(), ref_o[~inf_mask], atol=0.2, rtol=0.2
        )
        torch.testing.assert_close(
            lse[~inf_mask], ref_lse[~inf_mask], atol=2e-3, rtol=2e-3
        )


@_requires_sm120
@pytest.mark.parametrize("page_size", [1, 16, 64, 128])
@pytest.mark.parametrize("num_kv_chunks", [2, 4, 8])
def test_sm120_d512_paged_split_kv_and_public_wrapper(page_size, num_kv_chunks):
    """Verify forced Split-KV via public BatchPrefillWithPagedKVCacheWrapper and direct merge_states."""
    d = 512
    hq, hkv = 32, 4
    q_lens = [15, 65]
    kv_lens = [512, 1024]
    q_scale, k_scale, v_scale = 0.75, 1.25, 0.5
    sm_scale = (q_scale * k_scale) / math.sqrt(d)

    q_list = [_make_fp8((sq, hq, d), seed=800 + i) for i, sq in enumerate(q_lens)]
    k_list = [_make_fp8((sk, hkv, d), seed=900 + i) for i, sk in enumerate(kv_lens)]
    v_list = [_make_fp8((sk, hkv, d), seed=950 + i) for i, sk in enumerate(kv_lens)]

    q = torch.cat(q_list, dim=0)
    cu_q = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(q_lens), 0).tolist()),
        dtype=torch.int32,
        device="cuda",
    )
    (
        k_pool,
        v_pool,
        kv_pool,
        bt,
        page_indptr,
        page_indices,
        last_page_len,
    ) = _make_varlen_paged_kv(k_list, v_list, page_size, seed=77)

    # 1. Public BatchPrefillWithPagedKVCacheWrapper with forced fixed_split_size.
    workspace = torch.empty(128 << 20, dtype=torch.uint8, device="cuda")
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace, "HND", backend="cute-dsl-prims"
    )
    fixed_split_pages = max(1, ceil_div(max(kv_lens) // num_kv_chunks, page_size))
    wrapper.plan(
        cu_q,
        page_indptr,
        page_indices,
        last_page_len,
        hq,
        hkv,
        d,
        page_size,
        causal=True,
        q_data_type=torch.float8_e4m3fn,
        kv_data_type=torch.float8_e4m3fn,
        o_data_type=torch.bfloat16,
        fixed_split_size=fixed_split_pages,
    )
    out, out_lse = wrapper.run_return_lse(
        q,
        kv_pool,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        enable_pdl=True,
    )

    refs, lses = [], []
    for qb, kb, vb in zip(q_list, k_list, v_list, strict=True):
        ro, rl = _ref_attention(qb, kb, vb, sm_scale, is_causal=True, v_scale=v_scale)
        refs.append(ro)
        lses.append(rl)
    ref_o = torch.cat(refs)
    ref_lse = torch.cat(lses)
    torch.testing.assert_close(out.float(), ref_o, atol=0.2, rtol=0.2)
    torch.testing.assert_close(out_lse, ref_lse, atol=2e-3, rtol=2e-3)

    # 2. Direct forced num_kv_chunks with explicit kv_tile_ranges + merge_states.
    seqlens_kv = torch.tensor(kv_lens, dtype=torch.int32, device="cuda")
    ranges = []
    for c in range(num_kv_chunks):
        for sk in kv_lens:
            n_tiles = ceil_div(sk, 64)
            t0 = (n_tiles * c) // num_kv_chunks
            t1 = (n_tiles * (c + 1)) // num_kv_chunks
            ranges.append([t0, t1])
    kv_tile_ranges = torch.tensor(ranges, dtype=torch.int32, device="cuda")

    total_q = q.shape[0]
    split_o = torch.empty(
        (total_q * num_kv_chunks, hq, d), dtype=torch.bfloat16, device="cuda"
    )
    split_lse = torch.empty(
        (total_q * num_kv_chunks, hq), dtype=torch.float32, device="cuda"
    )
    sm120_fmha_fp8_paged_prefill(
        q,
        k_pool,
        v_pool,
        split_o,
        bt,
        seqlens_kv,
        cu_q,
        is_causal=True,
        sm_scale=sm_scale,
        v_scale=v_scale,
        lse=split_lse,
        kv_tile_ranges=kv_tile_ranges,
    )
    merged_o, merged_lse = flashinfer.merge_states(
        split_o.view(total_q, num_kv_chunks, hq, d),
        split_lse.view(total_q, num_kv_chunks, hq),
    )
    torch.testing.assert_close(merged_o.float(), ref_o, atol=0.2, rtol=0.2)
    torch.testing.assert_close(merged_lse, ref_lse, atol=2e-3, rtol=2e-3)


@_requires_sm120
def test_sm120_d512_64bit_output_offset_past_int32_boundary():
    """Verify 64-bit O/Q indexing when partial O exceeds 2^31 elements (Int32 overflow)."""
    # total_q = 20_000, num_kv_chunks = 8, Hq = 32, D = 512 ->
    # 20_000 * 8 * 32 * 512 = 2,621,440,000 elements (> 2^31 - 1 = 2,147,483,647).
    d = 512
    hq, hkv = 32, 4
    num_chunks = 8
    page_size = 16
    req_q_len = 8
    req_kv_len = 128
    num_reqs = 2500  # total_q = 20_000
    total_q = num_reqs * req_q_len
    assert total_q * num_chunks * hq * d > (1 << 31)

    sm_scale = 1.0 / math.sqrt(d)
    q = torch.zeros((total_q, hq, d), dtype=torch.float8_e4m3fn, device="cuda")
    # Populate the last 2 requests (crossing the 2^31 element boundary) with non-trivial data.
    tail_reqs = 2
    tail_q = _make_fp8((tail_reqs * req_q_len, hq, d), seed=1234)
    q[-tail_reqs * req_q_len :].copy_(tail_q)

    # Compact shared pool of 8 pages (128 tokens) reused across requests.
    pages_per_req = req_kv_len // page_size
    k_ref = _make_fp8((req_kv_len, hkv, d), seed=2345)
    v_ref = _make_fp8((req_kv_len, hkv, d), seed=3456)
    kv_pool = torch.empty(
        (pages_per_req, 2, hkv, page_size, d),
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    kv_pool[:, 0].copy_(
        k_ref.view(pages_per_req, page_size, hkv, d).permute(0, 2, 1, 3)
    )
    kv_pool[:, 1].copy_(
        v_ref.view(pages_per_req, page_size, hkv, d).permute(0, 2, 1, 3)
    )
    k_pool, v_pool = kv_pool.unbind(dim=1)
    bt = (
        torch.arange(pages_per_req, dtype=torch.int32, device="cuda")
        .unsqueeze(0)
        .expand(num_reqs, pages_per_req)
        .contiguous()
    )
    seqlens_kv = torch.full((num_reqs,), req_kv_len, dtype=torch.int32, device="cuda")
    cu_q = torch.arange(0, total_q + 1, req_q_len, dtype=torch.int32, device="cuda")

    # 2 KV tiles (128 / 64): put tile 0 in chunk 0, tile 1 in chunk 7 (highest offset > 2^31),
    # and empty ranges in chunks 1..6.
    kv_tile_ranges = torch.zeros(
        (num_chunks * num_reqs, 2), dtype=torch.int32, device="cuda"
    )
    kv_tile_ranges[:num_reqs, 0] = 0
    kv_tile_ranges[:num_reqs, 1] = 1
    kv_tile_ranges[(num_chunks - 1) * num_reqs :, 0] = 1
    kv_tile_ranges[(num_chunks - 1) * num_reqs :, 1] = 2

    split_o = torch.empty(
        (total_q * num_chunks, hq, d), dtype=torch.bfloat16, device="cuda"
    )
    split_lse = torch.empty(
        (total_q * num_chunks, hq), dtype=torch.float32, device="cuda"
    )
    sm120_fmha_fp8_paged_prefill(
        q,
        k_pool,
        v_pool,
        split_o,
        bt,
        seqlens_kv,
        cu_q,
        is_causal=True,
        sm_scale=sm_scale,
        lse=split_lse,
        kv_tile_ranges=kv_tile_ranges,
    )

    # Merge and verify only the tail requests whose split_o offsets exceed 2^31 elements.
    tail_rows = tail_reqs * req_q_len
    tail_split_o = split_o[-tail_rows * num_chunks :].view(tail_rows, num_chunks, hq, d)
    tail_split_lse = split_lse[-tail_rows * num_chunks :].view(
        tail_rows, num_chunks, hq
    )
    merged_o, merged_lse = flashinfer.merge_states(tail_split_o, tail_split_lse)

    refs, lses = [], []
    for r in range(tail_reqs):
        qb = tail_q[r * req_q_len : (r + 1) * req_q_len]
        ro, rl = _ref_attention(qb, k_ref, v_ref, sm_scale, is_causal=True)
        refs.append(ro)
        lses.append(rl)
    torch.testing.assert_close(merged_o.float(), torch.cat(refs), atol=0.2, rtol=0.2)
    torch.testing.assert_close(merged_lse, torch.cat(lses), atol=2e-3, rtol=2e-3)


def _ref_merge_rows(v, s, rows):
    """Reference merge of the given rows of (n, chunks, h, d) partials with log2 LSEs."""
    out, lse = [], []
    for r in rows:
        s_r = s[r] * math.log(2)
        w = torch.softmax(s_r, dim=0)
        out.append((w[..., None] * v[r].float()).sum(0))
        lse.append(torch.logsumexp(s_r, dim=0) / math.log(2))
    return torch.stack(out), torch.stack(lse)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("shape", [(8, 3, 0, 512), (0, 3, 4, 512), (0, 0, 0, 64)])
def test_merge_states_empty_output_is_a_no_op(shape):
    """Verify merge_states returns empty outputs for zero tokens or zero heads instead of
    building an empty grid (or dividing by a zero head tile)."""
    v = torch.empty(shape, dtype=torch.bfloat16, device="cuda")
    s = torch.empty(shape[:3], dtype=torch.float32, device="cuda")
    merged_o, merged_lse = flashinfer.merge_states(v, s)
    torch.cuda.synchronize()
    assert merged_o.shape == (shape[0], shape[2], shape[3])
    assert merged_lse.shape == (shape[0], shape[2])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("num_heads", [65_535, 65_536])
def test_merge_states_many_heads_respects_grid_y_limit(num_heads):
    """Verify merge_states with more index sets than tokens and more than 65,535 heads
    stays within CUDA's grid.y limit."""
    gen = torch.Generator(device="cuda").manual_seed(0)
    v = torch.randn(
        (1, 2, num_heads, 64), dtype=torch.bfloat16, device="cuda", generator=gen
    )
    s = torch.randn((1, 2, num_heads), device="cuda", generator=gen)
    merged_o, merged_lse = flashinfer.merge_states(v, s)
    ref_o, ref_lse = _ref_merge_rows(v, s, [0])
    torch.testing.assert_close(merged_o.float(), ref_o, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(merged_lse, ref_lse, atol=1e-3, rtol=1e-3)


@_requires_sm120
@pytest.mark.parametrize("num_heads", [33, 40, 64])
def test_merge_states_d512_more_than_32_heads(num_heads):
    """Verify merge_states launches a legal block for D=512 with more than 32 heads."""
    d = 512
    gen = torch.Generator(device="cuda").manual_seed(num_heads)
    v = torch.randn(
        (8, 3, num_heads, d), dtype=torch.bfloat16, device="cuda", generator=gen
    )
    s = torch.randn((8, 3, num_heads), device="cuda", generator=gen)
    merged_o, merged_lse = flashinfer.merge_states(v, s)
    ref_o, ref_lse = _ref_merge_rows(v, s, range(8))
    torch.testing.assert_close(merged_o.float(), ref_o, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(merged_lse, ref_lse, atol=1e-3, rtol=1e-3)


@_requires_sm120
def test_sm120_d512_paged_split_kv_40_heads():
    """Verify the wrapper's Split-KV merge for D=512 with 40 Q heads (40 * 32 merge threads)."""
    d = 512
    hq, hkv, page_size = 40, 8, 16
    q_lens, kv_lens = [7, 33], [384, 640]
    sm_scale = 1.0 / math.sqrt(d)
    q_list = [_make_fp8((sq, hq, d), seed=1300 + i) for i, sq in enumerate(q_lens)]
    k_list = [_make_fp8((sk, hkv, d), seed=1400 + i) for i, sk in enumerate(kv_lens)]
    v_list = [_make_fp8((sk, hkv, d), seed=1500 + i) for i, sk in enumerate(kv_lens)]
    q = torch.cat(q_list, dim=0)
    cu_q = torch.tensor([0, q_lens[0], sum(q_lens)], dtype=torch.int32, device="cuda")
    _, _, kv_pool, _, page_indptr, page_indices, last_page_len = _make_varlen_paged_kv(
        k_list, v_list, page_size, seed=78
    )
    workspace = torch.empty(128 << 20, dtype=torch.uint8, device="cuda")
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace, "HND", backend="cute-dsl-prims"
    )
    wrapper.plan(
        cu_q,
        page_indptr,
        page_indices,
        last_page_len,
        hq,
        hkv,
        d,
        page_size,
        causal=True,
        q_data_type=torch.float8_e4m3fn,
        kv_data_type=torch.float8_e4m3fn,
        o_data_type=torch.bfloat16,
        fixed_split_size=8,
    )
    assert wrapper._prims_backend._kv_split is not None
    out, out_lse = wrapper.run_return_lse(q, kv_pool)
    refs, lses = [], []
    for qb, kb, vb in zip(q_list, k_list, v_list, strict=True):
        ro, rl = _ref_attention(qb, kb, vb, sm_scale, is_causal=True)
        refs.append(ro)
        lses.append(rl)
    torch.testing.assert_close(out.float(), torch.cat(refs), atol=0.2, rtol=0.2)
    torch.testing.assert_close(out_lse, torch.cat(lses), atol=2e-3, rtol=2e-3)


@_requires_sm120
def test_sm120_d512_skip_softmax_rejected_on_gpu():
    """Verify non-None skip_softmax_threshold raises NotImplementedError on SM120 GPU while None succeeds."""
    d = 512
    hq, hkv = 8, 2
    sq, skv = 64, 128
    q = _make_fp8((sq, hq, d), seed=1301)
    k = _make_fp8((skv, hkv, d), seed=1302)
    v = _make_fp8((skv, hkv, d), seed=1303)
    o = torch.empty((sq, hq, d), dtype=torch.bfloat16, device="cuda")
    cu_q = torch.tensor([0, sq], dtype=torch.int32, device="cuda")
    cu_k = torch.tensor([0, skv], dtype=torch.int32, device="cuda")
    t_cuda = torch.tensor([2.0], dtype=torch.float32, device="cuda")

    for bad_threshold in (0.0, 2.0, t_cuda):
        with pytest.raises(NotImplementedError, match="skip_softmax_threshold"):
            sm120_fmha_fp8_ragged_prefill(
                q, k, v, o, cu_q, cu_k, skip_softmax_threshold=bad_threshold
            )

    k_pool, v_pool, _, bt, _, _, _ = _make_varlen_paged_kv([k], [v], 16, seed=1304)
    seqlens_kv = torch.tensor([skv], dtype=torch.int32, device="cuda")
    for bad_threshold in (0.0, 2.0, t_cuda):
        with pytest.raises(NotImplementedError, match="skip_softmax_threshold"):
            sm120_fmha_fp8_paged_prefill(
                q,
                k_pool,
                v_pool,
                o,
                bt,
                seqlens_kv,
                cu_q,
                skip_softmax_threshold=bad_threshold,
            )

    # Dense path (skip_softmax_threshold=None) continues to succeed.
    sm120_fmha_fp8_ragged_prefill(q, k, v, o, cu_q, cu_k, skip_softmax_threshold=None)
    sm120_fmha_fp8_paged_prefill(
        q, k_pool, v_pool, o, bt, seqlens_kv, cu_q, skip_softmax_threshold=None
    )


@_requires_sm120
def test_sm120_d512_unbalanced_causal_and_long_grid_x(monkeypatch):
    """Verify unbalanced causal D=512 (Q tiles on grid.x) and >65,535 Q-tile launch on SM120."""
    d = 512
    hq, hkv = 16, 1
    sq, skv = 65, 130
    sm_scale = 1.0 / math.sqrt(d)

    q = _make_fp8((sq, hq, d), seed=1401)
    k = _make_fp8((skv, hkv, d), seed=1402)
    v = _make_fp8((skv, hkv, d), seed=1403)
    cu_q = torch.tensor([0, sq], dtype=torch.int32, device="cuda")
    cu_k = torch.tensor([0, skv], dtype=torch.int32, device="cuda")
    ref_o, ref_lse = _ref_attention(q, k, v, sm_scale, is_causal=True)

    # 1. Explicit unbalanced causal (PRIMS_FMHA_DISABLE_BALANCED_SCHEDULING=1) on ragged + paged.
    monkeypatch.setenv("PRIMS_FMHA_DISABLE_BALANCED_SCHEDULING", "1")
    o_rag = torch.empty((sq, hq, d), dtype=torch.bfloat16, device="cuda")
    lse_rag = torch.empty((sq, hq), dtype=torch.float32, device="cuda")
    sm120_fmha_fp8_ragged_prefill(
        q, k, v, o_rag, cu_q, cu_k, is_causal=True, sm_scale=sm_scale, lse=lse_rag
    )
    torch.testing.assert_close(o_rag.float(), ref_o, atol=0.2, rtol=0.2)
    torch.testing.assert_close(lse_rag, ref_lse, atol=2e-3, rtol=2e-3)

    k_pool, v_pool, _, bt, _, _, _ = _make_varlen_paged_kv([k], [v], 16, seed=1404)
    seqlens_kv = torch.tensor([skv], dtype=torch.int32, device="cuda")
    o_pag = torch.empty((sq, hq, d), dtype=torch.bfloat16, device="cuda")
    lse_pag = torch.empty((sq, hq), dtype=torch.float32, device="cuda")
    sm120_fmha_fp8_paged_prefill(
        q,
        k_pool,
        v_pool,
        o_pag,
        bt,
        seqlens_kv,
        cu_q,
        is_causal=True,
        sm_scale=sm_scale,
        lse=lse_pag,
        gqa_pack_size=16,
    )
    torch.testing.assert_close(o_pag.float(), ref_o, atol=0.2, rtol=0.2)
    torch.testing.assert_close(lse_pag, ref_lse, atol=2e-3, rtol=2e-3)

    # 2. Automatic fallback when q_tiles = ceil_div(262_144, 64 // 16) = 65,536 > 65,535
    #    with balanced scheduling enabled by default (actually launches 65,536 CTAs on grid.x).
    monkeypatch.delenv("PRIMS_FMHA_DISABLE_BALANCED_SCHEDULING", raising=False)
    o_long = torch.empty((sq, hq, d), dtype=torch.bfloat16, device="cuda")
    lse_long = torch.empty((sq, hq), dtype=torch.float32, device="cuda")
    sm120_fmha_fp8_paged_prefill(
        q,
        k_pool,
        v_pool,
        o_long,
        bt,
        seqlens_kv,
        cu_q,
        is_causal=True,
        sm_scale=sm_scale,
        max_seqlen_q=262_144,
        lse=lse_long,
        gqa_pack_size=16,
    )
    torch.testing.assert_close(o_long.float(), ref_o, atol=0.2, rtol=0.2)
    torch.testing.assert_close(lse_long, ref_lse, atol=2e-3, rtol=2e-3)
