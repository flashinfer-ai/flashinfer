"""CPU-side packaging and public-routing tests for Cake FMHA."""

from __future__ import annotations

from pathlib import Path

import flashinfer
import flashinfer.cake_fmha as cake_api
import flashinfer.decode as decode
import flashinfer.prefill as prefill
import pytest
import torch
from flashinfer.jit.cake_fmha import (
    CAKE_FMHA_FLASHINFER_MATRIX_REVISION,
    CAKE_FMHA_MANIFEST_SHA256,
    gen_cake_fmha_compat_module,
    get_cake_fmha_compat_uri,
    get_cake_fmha_manifest,
)


def test_cake_fmha_manifest_is_authenticated_and_complete() -> None:
    manifest = get_cake_fmha_manifest()
    assert manifest["product"] == "cake_fmha"
    assert manifest["flashinfer_matrix_revision"] == (
        CAKE_FMHA_FLASHINFER_MATRIX_REVISION
    )
    assert manifest["publication"]["promotion_ready"] is True
    assert manifest["capability"]["complete"] is True
    assert manifest["capability"]["cake_coverage_ratio"] == 1.0
    assert manifest["capability"]["upstream_valid_cases"] == 57_280
    assert manifest["capability"]["cake_covered_cases"] == 57_280
    assert manifest["components"]["compat_v1"]["launch_binding"] == (
        "cake_fmha_launch_compat_v1"
    )
    assert len(CAKE_FMHA_MANIFEST_SHA256) == 64


def test_cake_fmha_public_manifest_is_defensive_copy() -> None:
    public_manifest = cake_api.cake_fmha_manifest()
    public_manifest["product"] = "mutated"
    assert cake_api.cake_fmha_manifest()["product"] == "cake_fmha"


def test_cake_fmha_jit_spec_uses_versioned_standalone_sources(monkeypatch) -> None:
    import flashinfer.jit.core as jit_core

    monkeypatch.setattr(jit_core, "check_cuda_arch", lambda: None)
    for target in ("sm100a", "sm103a"):
        spec = gen_cake_fmha_compat_module(target)
        assert spec.name == get_cake_fmha_compat_uri(target)
        source_names = {Path(source).name for source in spec.sources}
        assert source_names == {
            "cake_fmha_compat_v1.cu",
            "cake_fmha_compat_v1_binding.cu",
            "cake_fmha_jit_binding.cu",
        }


def test_cake_decode_public_entrypoint_forces_cake_backend(monkeypatch) -> None:
    observed = {}

    def fake_decode(*args, **kwargs):
        observed.update(kwargs)
        return "decode-result"

    monkeypatch.setattr(decode, "trtllm_batch_decode_with_kv_cache", fake_decode)
    assert cake_api.cake_batch_decode_with_kv_cache("query") == "decode-result"
    assert observed["backend"] == "cake"


def test_cake_context_public_entrypoint_forces_cake_backend(monkeypatch) -> None:
    observed = {}

    def fake_context(*args, **kwargs):
        observed.update(kwargs)
        return "context-result"

    monkeypatch.setattr(prefill, "trtllm_batch_context_with_kv_cache", fake_context)
    assert cake_api.cake_batch_context_with_kv_cache("query") == "context-result"
    assert observed["backend"] == "cake"


def test_cake_public_symbols_are_top_level() -> None:
    assert flashinfer.cake_batch_decode_with_kv_cache is (
        cake_api.cake_batch_decode_with_kv_cache
    )
    assert flashinfer.cake_batch_context_with_kv_cache is (
        cake_api.cake_batch_context_with_kv_cache
    )
    assert flashinfer.cake_fmha_manifest is cake_api.cake_fmha_manifest


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_cake_decode_bf16_matches_flashinfer_reference() -> None:
    from tests.attention.test_trtllm_gen_attention_decode import (
        _test_trtllm_batch_decode,
    )

    _test_trtllm_batch_decode(
        backend="cake",
        kv_layout="HND",
        batch_size=2,
        q_len_per_req=1,
        page_size=16,
        num_kv_heads=2,
        head_grp_size=2,
        window_left=-1,
        q_dtype="bf16",
        o_dtype="bf16",
        kv_dtype="bf16",
        enable_pdl=False,
        enable_sink=False,
        max_in_kv_len=31,
        head_dim=128,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_cake_context_bf16_separate_tables_matches_reference(monkeypatch) -> None:
    from tests.attention.test_trtllm_gen_attention_prefill import (
        _test_trtllm_batch_prefill,
    )

    original = prefill.trtllm_batch_context_with_kv_cache

    def cake_context(*args, **kwargs):
        return original(*args, backend="cake", **kwargs)

    monkeypatch.setattr(prefill, "trtllm_batch_context_with_kv_cache", cake_context)
    _test_trtllm_batch_prefill(
        kv_layout="NHD",
        batch_size=2,
        page_size=32,
        num_kv_heads=2,
        head_grp_size=2,
        causal=False,
        window_left=-1,
        q_dtype="bf16",
        o_dtype="bf16",
        kv_dtype="bf16",
        enable_pdl=False,
        enable_sink=False,
        max_q_len=7,
        max_kv_len=31,
        device_scale=False,
        head_dim=128,
        uses_shared_paged_kv_idx=False,
    )
