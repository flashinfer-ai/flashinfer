# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

from flashinfer.gemm import fp8_linear, fp8_linear_swiglu, fp8_qkv_qknorm_rope
from flashinfer.prims_ts.gemm import PrimsTsGemmConfig
from flashinfer.prims_ts.gemm.api import _fp8_mma_k
from flashinfer.utils import get_compute_capability


def _require_gemm_gpu():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if get_compute_capability(torch.device("cuda")) not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("requires SM100, SM103, or SM107")


def _fp8_problem(m=128, n=512, k=128):
    torch.manual_seed(7)
    a = (torch.randn((m, k), device="cuda") / 5).to(torch.float8_e4m3fn)
    w = (torch.randn((n, k), device="cuda") / 5).to(torch.float8_e4m3fn)
    a_scale = torch.rand((m,), device="cuda", dtype=torch.float32) + 0.25
    w_scale = torch.rand((n,), device="cuda", dtype=torch.float32) + 0.25
    return a, w, a_scale, w_scale


def test_fp8_linear_per_token_per_channel_and_bias():
    _require_gemm_gpu()
    a, w, a_scale, w_scale = _fp8_problem()
    bias = torch.randn((w.shape[0],), device="cuda", dtype=torch.bfloat16)
    actual = fp8_linear(a, w, a_scale, w_scale, bias)
    expected = (
        F.linear(a.float(), w.float()) * a_scale[:, None] * w_scale[None, :]
        + bias.float()
    )
    torch.testing.assert_close(actual.float(), expected, atol=0.2, rtol=2e-2)


def _linear_config(arch, operand_format="fp8_e4m3", has_bias=False, **kwargs):
    return PrimsTsGemmConfig(
        arch, operand_format, "bf16", "linear", has_bias, None, None, **kwargs
    )


@pytest.mark.parametrize(
    "arch,operand_format,fp8_mma_k,expected",
    [
        (100, "fp8_e4m3", None, 32),
        (103, "fp8_e4m3", None, 32),
        (107, "fp8_e4m3", None, 64),
        (107, "fp8_e4m3", 32, 32),
        (100, "nvfp4_e2m1", None, 32),
        (107, "nvfp4_e2m1", None, 32),
    ],
)
def test_fp8_mma_k_defaults(arch, operand_format, fp8_mma_k, expected):
    config = _linear_config(arch, operand_format, fp8_mma_k=fp8_mma_k)
    assert _fp8_mma_k(config) == expected


@pytest.mark.parametrize(
    "arch,operand_format,fp8_mma_k,match",
    [
        (100, "fp8_e4m3", 64, "requires SM107"),
        (103, "fp8_e4m3", 64, "requires SM107"),
        (107, "fp8_e4m3", 128, "32 or 64"),
        (107, "nvfp4_e2m1", 32, "requires FP8"),
    ],
)
def test_fp8_mma_k_rejects_unsupported_config(arch, operand_format, fp8_mma_k, match):
    config = _linear_config(arch, operand_format, fp8_mma_k=fp8_mma_k)
    with pytest.raises(ValueError, match=match):
        _fp8_mma_k(config)


@pytest.mark.parametrize("fp8_mma_k", [32, 64])
def test_fp8_linear_mma_k_on_sm107(fp8_mma_k):
    _require_gemm_gpu()
    if get_compute_capability(torch.device("cuda")) != (10, 7):
        pytest.skip("64 B FP8 MMA K requires SM107")
    a, w, a_scale, w_scale = _fp8_problem(m=256, n=512, k=512)
    bias = torch.randn((w.shape[0],), device="cuda", dtype=torch.bfloat16)
    config = _linear_config(107, has_bias=True, fp8_mma_k=fp8_mma_k)
    actual = fp8_linear(a, w, a_scale, w_scale, bias, config=config)
    expected = (
        F.linear(a.float(), w.float()) * a_scale[:, None] * w_scale[None, :]
        + bias.float()
    )
    torch.testing.assert_close(actual.float(), expected, atol=0.2, rtol=2e-2)


def test_fp8_linear_qkv_scale_matches_three_scaled_output_groups():
    _require_gemm_gpu()
    m, n, k = 128, 1536, 128
    a, w, a_scale, w_scale = _fp8_problem(m, n, k)
    qkv_scale = torch.tensor((0.5, 1.25, 2.0), device="cuda", dtype=torch.float32)
    major, minor = get_compute_capability(torch.device("cuda"))
    config = PrimsTsGemmConfig(
        arch=major * 10 + minor,
        operand_format="fp8_e4m3",
        output_format="bf16",
        epilogue="linear",
        has_bias=False,
        head_dim=None,
        is_neox=None,
        tile_m=256,
        tile_n=256,
        tile_k=128,
        cluster_shape=(2, 2, 1),
        scheduler="clc_dynamic",
        has_qkv_scale=True,
    )
    actual = fp8_linear(a, w, a_scale, w_scale, qkv_scale=qkv_scale, config=config)
    expected = F.linear(a.float(), w.float()) * a_scale[:, None] * w_scale[None, :]
    expected = expected.view(m, 3, n // 3) * qkv_scale.view(1, 3, 1)
    torch.testing.assert_close(
        actual.float(), expected.reshape_as(actual), atol=0.25, rtol=3e-2
    )


def test_fp8_swiglu_interleaved_gate_activation():
    _require_gemm_gpu()
    a, w, a_scale, w_scale = _fp8_problem()
    actual = fp8_linear_swiglu(a, w, a_scale, w_scale)
    linear = F.linear(a.float(), w.float()) * a_scale[:, None] * w_scale[None, :]
    expected = linear[:, 0::2] * F.silu(linear[:, 1::2])
    torch.testing.assert_close(actual.float(), expected, atol=0.3, rtol=3e-2)


def test_fp8_output_scale_ownership_and_preallocated_payload():
    _require_gemm_gpu()
    a, w, a_scale, w_scale = _fp8_problem()
    payload, scale = fp8_linear(a, w, a_scale, w_scale, out_dtype=torch.float8_e4m3fn)
    assert payload.dtype == torch.float8_e4m3fn
    assert scale.dtype == torch.float32 and scale.shape == (1,)
    expected = (
        F.linear(a.float(), w.float()) * a_scale[:, None] * w_scale[None, :] * scale
    ).to(torch.float8_e4m3fn)
    torch.testing.assert_close(payload.float(), expected.float(), atol=0, rtol=0)
    supplied_scale = torch.ones((1,), device="cuda", dtype=torch.float32)
    supplied_payload = torch.empty_like(payload)
    actual = fp8_linear(
        a,
        w,
        a_scale,
        w_scale,
        out_dtype=torch.float8_e4m3fn,
        output_quant_scale=supplied_scale,
        out=supplied_payload,
    )
    assert actual is supplied_payload


def test_fp8_qkv_qknorm_rope_matches_torch_oracle():
    _require_gemm_gpu()
    m, n, k, head_dim = 128, 1536, 128, 128
    a, w, a_scale, w_scale = _fp8_problem(m, n, k)
    q_weight = torch.rand((head_dim,), device="cuda", dtype=torch.bfloat16)
    k_weight = torch.rand((head_dim,), device="cuda", dtype=torch.bfloat16)
    angles = torch.randn((m, head_dim // 2), device="cuda")
    cos_sin = torch.cat((angles.cos(), angles.sin()), dim=-1).contiguous()
    positions = torch.arange(m, device="cuda", dtype=torch.int64)
    actual = fp8_qkv_qknorm_rope(
        a,
        w,
        a_scale,
        w_scale,
        q_weight,
        k_weight,
        cos_sin,
        positions,
        num_q_heads=4,
        num_kv_heads=4,
        head_dim=head_dim,
    )
    base = (
        (F.linear(a.float(), w.float()) * a_scale[:, None] * w_scale[None, :])
        .bfloat16()
        .float()
    )
    shaped = base.view(m, 3, 4, head_dim)
    cos, sin = cos_sin[:, None, :64], cos_sin[:, None, 64:]
    for part, norm_weight in ((0, q_weight), (1, k_weight)):
        value = shaped[:, part]
        value = (
            value
            * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)
            * norm_weight.float()
        )
        pairs = value.view(m, 4, 64, 2)
        shaped[:, part] = torch.stack(
            (
                pairs[..., 0] * cos - pairs[..., 1] * sin,
                pairs[..., 1] * cos + pairs[..., 0] * sin,
            ),
            -1,
        ).flatten(-2)
    torch.testing.assert_close(
        actual.float(), shaped.reshape_as(base), atol=0.3, rtol=3e-2
    )
