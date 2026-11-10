"""Two-level (per-token second-level scale) nvfp4_ds_mla record tests.

``per_token_scale=True`` stores ``s_t = token_amax/(6*448)`` as fp32 at record
bytes [292, 296) and encodes every group scale relative to it, so the largest
group's E4M3 scale byte lands at the top of the E4M3 range by construction.
These tests pin the record ABI, the positioning invariant that motivates the
mode (the largest group scale is placed near the top of the E4M3 range), and
the accuracy win over the implicit-1.0 static writer at shallow-layer
magnitudes. Smaller groups within the same token can still be subnormal.
"""

from __future__ import annotations

import pytest
import torch

from b12x.attention._shared.mla.kv_cache import (
    clear_nvfp4_mla_fp8_rope_kv_cache_kernel_cache,
    concat_and_cache_nvfp4_mla_fp8_rope,
)

from b12x.testing.reference.helpers import dequantize_nvfp4_mla_nope
from ..conftest import require_b12x as require_sm120


_RECORD_BYTES = 368
_NOPE_BYTES = 256
_GROUP_SCALES_OFFSET = 256
_ROPE_SCALE_OFFSET = 288
_LATENT_SCALE_OFFSET = 292
_PAD_OFFSET = 296
_ROPE_OFFSET = 304
_PAGE_SIZE = 64
_V_HEAD_DIM = 512
_PE_DIM = 64
# Exact-constant contract shared with the writer (kv_cache._TWO_LEVEL_RCP).
_TWO_LEVEL_RCP = torch.tensor(1.0 / (6.0 * 448.0), dtype=torch.float32)
# Smallest normal E4M3 value; group scales at or below this are subnormal.
_E4M3_MIN_NORMAL = 2.0**-6


@pytest.fixture(scope="module", autouse=True)
def _isolate_writer_kernel_cache():
    clear_nvfp4_mla_fp8_rope_kv_cache_kernel_cache()
    yield
    clear_nvfp4_mla_fp8_rope_kv_cache_kernel_cache()


def _write_records(
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    *,
    per_token_scale: bool,
) -> torch.Tensor:
    num_tokens = kv_c.shape[0]
    cache = torch.zeros(
        (2, _PAGE_SIZE, _RECORD_BYTES), dtype=torch.uint8, device=kv_c.device
    )
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=kv_c.device)
    concat_and_cache_nvfp4_mla_fp8_rope(
        kv_c,
        k_pe,
        cache,
        slot_mapping,
        per_token_scale=per_token_scale,
    )
    return cache.reshape(-1, _RECORD_BYTES)[:num_tokens].cpu()


def _dequantize_two_level(records: torch.Tensor) -> torch.Tensor:
    """Host reference: e2m1 * e4m3(group scale) * fp32 s_t at [292, 296)."""
    values, _ = dequantize_nvfp4_mla_nope(
        records,
        nope_bytes=_NOPE_BYTES,
        group_scales_offset=_GROUP_SCALES_OFFSET,
        group_scales_end=_ROPE_SCALE_OFFSET,
        latent_scale_offset=_LATENT_SCALE_OFFSET,
    )
    return values


def _dequantize_static(records: torch.Tensor) -> torch.Tensor:
    """Host reference for the implicit-1.0 static writer (no second level)."""
    values, _ = dequantize_nvfp4_mla_nope(
        records,
        nope_bytes=_NOPE_BYTES,
        group_scales_offset=_GROUP_SCALES_OFFSET,
        group_scales_end=_ROPE_SCALE_OFFSET,
    )
    return values


def _layered_inputs(
    device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Token magnitudes spanning the GLM-5.2 per-layer kv_c range (~240x).

    Token 0 mimics a shallow layer (max_abs ~0.02 -- the E4M3-subnormal
    regime under the implicit-1.0 writer), the last a deep layer (~5.2).
    """
    torch.manual_seed(0x145)
    base = torch.randn((6, _V_HEAD_DIM), dtype=torch.float32)
    base = base / base.abs().amax(dim=-1, keepdim=True)
    mags = torch.tensor([0.02, 0.08, 0.35, 1.0, 2.6, 5.2]).unsqueeze(-1)
    kv_c = (base * mags).to(device=device, dtype=dtype)
    k_pe = torch.randn((6, _PE_DIM), dtype=torch.float32).to(device=device, dtype=dtype)
    return kv_c, k_pe


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@torch.inference_mode()
def test_per_token_records_store_exact_scale_and_keep_abi(
    dtype: torch.dtype,
) -> None:
    device = require_sm120()
    kv_c, k_pe = _layered_inputs(device, dtype)
    records = _write_records(kv_c, k_pe, per_token_scale=True)
    static_records = _write_records(kv_c, k_pe, per_token_scale=False)

    # s_t == token_amax * f32(1/2688) bit-exactly (same exact-constant
    # contract as the rope scale).
    token_amax = kv_c.float().abs().amax(dim=-1).cpu()
    expected = token_amax * _TWO_LEVEL_RCP
    stored = (
        records[:, _LATENT_SCALE_OFFSET:_PAD_OFFSET]
        .contiguous()
        .view(torch.float32)
        .reshape(-1)
    )
    assert torch.equal(stored, expected)

    # Remaining pad stays zero; static mode keeps the full pad zero.
    assert (records[:, _PAD_OFFSET:_ROPE_OFFSET] == 0).all()
    assert (static_records[:, _LATENT_SCALE_OFFSET:_ROPE_OFFSET] == 0).all()

    # The RoPE lane (scale + payload) is mode-independent.
    assert torch.equal(
        records[:, _ROPE_SCALE_OFFSET:_LATENT_SCALE_OFFSET],
        static_records[:, _ROPE_SCALE_OFFSET:_LATENT_SCALE_OFFSET],
    )
    assert torch.equal(records[:, _ROPE_OFFSET:], static_records[:, _ROPE_OFFSET:])


@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
@torch.inference_mode()
def test_per_token_scale_positions_largest_group_scale_near_e4m3_max(
    dtype: torch.dtype,
) -> None:
    device = require_sm120()
    kv_c, k_pe = _layered_inputs(device, dtype)
    records = _write_records(kv_c, k_pe, per_token_scale=True)
    static_records = _write_records(kv_c, k_pe, per_token_scale=False)

    group_scales = (
        records[:, _GROUP_SCALES_OFFSET:_ROPE_SCALE_OFFSET]
        .contiguous()
        .view(torch.float8_e4m3fn)
        .float()
    )
    # Positioning invariant: every token's largest group scale sits near the
    # top of the E4M3 range (>= 256 given e4m3 rounding).
    assert (group_scales.amax(dim=-1) >= 256.0).all()

    # The static writer's shallow-magnitude token (amax 0.02) demonstrates the
    # defect this mode removes: its group scales are E4M3 subnormals.
    static_scales = (
        static_records[:, _GROUP_SCALES_OFFSET:_ROPE_SCALE_OFFSET]
        .contiguous()
        .view(torch.float8_e4m3fn)
        .float()
    )
    assert (static_scales[0].amax() <= _E4M3_MIN_NORMAL) or (
        static_scales[0][static_scales[0] > 0].amin() < _E4M3_MIN_NORMAL
    )

    # Accuracy: two-level reconstruction must beat implicit-1.0 on the
    # shallow tokens and never lose on the deep ones (small slack for
    # rcp.approx in the pack path).
    reference = kv_c.float().cpu()
    err_dynamic = (_dequantize_two_level(records) - reference).abs().amax(dim=-1)
    err_static = (_dequantize_static(static_records) - reference).abs().amax(dim=-1)
    assert (err_dynamic[:2] < err_static[:2]).all()
    assert (err_dynamic <= err_static * 1.05 + 1e-6).all()

    # Absolute bound: NVFP4 worst-case relative step at two-level positioning.
    rel = err_dynamic / kv_c.float().abs().amax(dim=-1).cpu()
    assert (rel < 0.25).all()


@torch.inference_mode()
def test_zero_token_writes_zero_scale_and_zero_payload() -> None:
    device = require_sm120()
    kv_c = torch.zeros((2, _V_HEAD_DIM), dtype=torch.bfloat16, device=device)
    k_pe = torch.zeros((2, _PE_DIM), dtype=torch.bfloat16, device=device)
    records = _write_records(kv_c, k_pe, per_token_scale=True)
    assert (records[:, :_ROPE_SCALE_OFFSET] == 0).all()
    assert (records[:, _LATENT_SCALE_OFFSET:_PAD_OFFSET] == 0).all()
    assert torch.equal(_dequantize_two_level(records), torch.zeros((2, _V_HEAD_DIM)))
