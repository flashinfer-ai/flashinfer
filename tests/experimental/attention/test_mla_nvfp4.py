from __future__ import annotations

import pytest
import torch

from flashinfer.experimental.sm12x.attention._shared.mla import api, traits
from flashinfer.experimental.sm12x.attention._shared.mla.kernel import (
    run_unified_decode,
)
from flashinfer.experimental.sm12x.attention._shared.mla.prefill_mg import (
    run_unified_prefill_mg,
)
from flashinfer.experimental.sm12x.attention._shared.mla.smem import make_smem_layout
from flashinfer.experimental.sm12x.attention._shared.mla.traits import (
    ComputeMode,
    ModelType,
    ScaleFormat,
)


def test_nvfp4_decode_allocates_bf16_q_staging() -> None:
    nvfp4_traits = traits.make_unified_traits(
        ModelType.GLM_NSA,
        ComputeMode.BF16,
        ScaleFormat.NVFP4_E4M3,
        fp8_rope=False,
    )
    layout = make_smem_layout(nvfp4_traits)

    expected_bytes = nvfp4_traits.hpb * nvfp4_traits.q_nope_stride * 2
    assert layout.q_fp8_bytes == expected_bytes
    assert layout.q_sc_off >= layout.q_fp8_off + expected_bytes


def test_nvfp4_decode_rejects_unknown_record_width() -> None:
    with pytest.raises(ValueError, match="must be 368 or 432 bytes"):
        _run_invalid_nvfp4_decode(record_bytes=400, fp8_rope=None)


def test_nvfp4_decode_rejects_fp8_rope_layout_mismatch() -> None:
    with pytest.raises(ValueError, match="disagrees with fp8_rope_override"):
        _run_invalid_nvfp4_decode(record_bytes=368, fp8_rope=False)


def test_nvfp4_mg_prefill_rejects_unknown_record_width() -> None:
    with pytest.raises(ValueError, match="must be 368 or 432 bytes"):
        _run_invalid_nvfp4_mg_prefill(record_bytes=400, fp8_rope=None)


def test_nvfp4_mg_prefill_rejects_explicit_layout_mismatch() -> None:
    with pytest.raises(ValueError, match="disagrees with fp8_rope"):
        _run_invalid_nvfp4_mg_prefill(record_bytes=368, fp8_rope=False)


def test_api_uses_traits_fp8_rope_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(traits, "KV_FP8_ROPE_ENABLED", True)
    assert api._resolve_kv_fp8_rope(None) is True

    monkeypatch.setattr(traits, "KV_FP8_ROPE_ENABLED", False)
    assert api._resolve_kv_fp8_rope(None) is False


def _run_invalid_nvfp4_decode(*, record_bytes: int, fp8_rope: bool | None) -> None:
    rows = 1
    topk = 4
    run_unified_decode(
        q_all=torch.empty((rows, 8, 576), dtype=torch.bfloat16),
        swa_k_cache=torch.empty((1, 1, record_bytes), dtype=torch.uint8),
        swa_indices=torch.zeros((rows, topk), dtype=torch.int32),
        swa_topk_lengths=torch.full((rows,), topk, dtype=torch.int32),
        workspace=object(),
        sm_scale=0.1,
        swa_page_size=64,
        scale_format_override=ScaleFormat.NVFP4_E4M3,
        fp8_rope_override=fp8_rope,
    )


def _run_invalid_nvfp4_mg_prefill(*, record_bytes: int, fp8_rope: bool | None) -> None:
    rows = 1
    topk = 4
    run_unified_prefill_mg(
        q=torch.empty((rows, 16, 576), dtype=torch.bfloat16),
        kv_cache=torch.empty((1, 1, record_bytes), dtype=torch.uint8),
        topk_indices=torch.zeros((rows, topk), dtype=torch.int32),
        sm_scale=0.1,
        page_block_size=1,
        model_type=ModelType.GLM_NSA,
        scale_format=ScaleFormat.NVFP4_E4M3,
        fp8_rope=fp8_rope,
    )
