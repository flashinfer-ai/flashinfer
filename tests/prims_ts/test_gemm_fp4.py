# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from flashinfer.gemm import fp4_linear, fp4_linear_swiglu, fp4_qkv_qknorm_rope
from flashinfer.prims_ts.gemm import (
    prepare_fp4_linear,
    prepare_fp4_linear_swiglu,
    prepare_fp4_qkv_qknorm_rope,
)
from flashinfer.prims_ts.gemm.api import PreparedFp4Linear, _launch, _load_kernel
from flashinfer.prims_ts.gemm.config import PrimsTsGemmConfig
from flashinfer.utils import get_compute_capability
from tests.prims_ts.gemm_fp4_reference import (
    dequantize_nvfp4,
    dequantize_nvfp4_128x4,
    linear_scale_to_128x4,
)


def _require_gemm_gpu():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if get_compute_capability(torch.device("cuda")) not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("requires SM100, SM103, or SM107")


def _require_sm103(reason):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
        pytest.skip(reason)


def _fp4_problem(m=8192, n=8192, k=8192):
    torch.manual_seed(11)
    a = torch.randint(0, 256, (m, k // 2), device="cuda", dtype=torch.uint8)
    w = torch.randint(0, 256, (n, k // 2), device="cuda", dtype=torch.uint8)
    a_scale = (torch.rand((m, k // 16), device="cuda") / 4).to(torch.float8_e4m3fn)
    w_scale = (torch.rand((n, k // 16), device="cuda") / 4).to(torch.float8_e4m3fn)
    return a, a_scale, w, w_scale


def _swizzled(a_scale, w_scale):
    return linear_scale_to_128x4(a_scale), linear_scale_to_128x4(w_scale)


def _overlap_config(mma_k=96, **kwargs):
    return PrimsTsGemmConfig(
        103,
        "nvfp4_e2m1",
        "bf16",
        "linear",
        False,
        None,
        None,
        tile_k=256,
        nvfp4_mma_k=mma_k,
        tmem_overlap=True,
        **kwargs,
    )


def test_fp4_linear_rejects_logical_scales():
    _require_gemm_gpu()
    a, a_scale, w, w_scale = _fp4_problem()
    with pytest.raises(ValueError, match="128x4"):
        fp4_linear(a, a_scale, 1.0, w, w_scale, 1.0)


def test_fp4_linear_bf16_matches_block_scaled_reference():
    _require_gemm_gpu()
    a, a_scale, w, w_scale = _fp4_problem()
    a_sf, w_sf = _swizzled(a_scale, w_scale)
    actual = fp4_linear(a, a_sf, 1.0, w, w_sf, 1.0)
    prepared = prepare_fp4_linear(w, w_sf, 1.0, 1.0)
    prepared_actual = prepared(a, a_sf)
    expected = F.linear(
        dequantize_nvfp4(a, a_scale, 1.0), dequantize_nvfp4(w, w_scale, 1.0)
    )
    torch.testing.assert_close(actual.float(), expected, atol=0.5, rtol=4e-2)
    torch.testing.assert_close(prepared_actual, actual)


def test_fp4_swiglu_and_packed_output_contract():
    _require_gemm_gpu()
    a, a_scale, w, w_scale = _fp4_problem()
    a_sf, w_sf = _swizzled(a_scale, w_scale)
    actual = fp4_linear_swiglu(a, a_sf, 1.0, w, w_sf, 1.0)
    prepared = prepare_fp4_linear_swiglu(w, w_sf, 1.0, 1.0)
    prepared_actual = prepared(a, a_sf)
    linear = F.linear(
        dequantize_nvfp4(a, a_scale, 1.0), dequantize_nvfp4(w, w_scale, 1.0)
    )
    expected = linear[:, 0::2] * F.silu(linear[:, 1::2])
    torch.testing.assert_close(actual.float(), expected, atol=1.0, rtol=6e-2)
    torch.testing.assert_close(prepared_actual, actual)
    payload, scales, global_scale = fp4_linear_swiglu(
        a, a_sf, 1.0, w, w_sf, 1.0, out_dtype=torch.uint8
    )
    prepared_fp4 = prepare_fp4_linear_swiglu(w, w_sf, 1.0, 1.0, out_dtype=torch.uint8)
    prepared_payload, prepared_scales, prepared_global_scale = prepared_fp4(a, a_sf)
    assert payload.shape == (a.shape[0], w.shape[0] // 4)
    assert scales.dtype == torch.uint8
    assert global_scale.shape == (1,)
    assert prepared_global_scale.shape == (1,)
    torch.testing.assert_close(prepared_payload, payload)
    torch.testing.assert_close(prepared_scales, scales)
    # The default encode scale of 1 caps |output| at 448 * 6 = 2688, and this
    # problem reaches about 4e3 (the exact max depends on the device's RNG
    # stream), so measure accuracy the way a caller would: map the global
    # absmax onto the E4M3 x E2M1 range.
    encode_scale = (448.0 * 6.0 / expected.abs().amax()).reshape(1)
    payload, scales = fp4_linear_swiglu(
        a,
        a_sf,
        1.0,
        w,
        w_sf,
        1.0,
        out_dtype=torch.uint8,
        output_quant_scale=encode_scale,
    )
    dequantized = dequantize_nvfp4_128x4(payload, scales, encode_scale)
    block_error = (
        (dequantized - expected).reshape(a.shape[0], -1, 16).abs()
        / expected.reshape(a.shape[0], -1, 16).abs().amax(-1, keepdim=True).clamp(min=1)
    ).amax()
    # E2M1 rounds to within half a step, at most 1/6 of the block absmax,
    # widened by up to 1/16 for E4M3 rounding of the block scale (~0.177).
    # Encoding against the unrounded FP32 scale instead measures about 0.225.
    assert block_error < 0.2
    with pytest.raises(ValueError, match="fp4_linear_swiglu"):
        fp4_linear(a, a_sf, 1.0, w, w_sf, 1.0, out_dtype=torch.uint8)


def test_fp4_qkv_qknorm_rope_bf16():
    _require_gemm_gpu()
    m, n, k, head_dim = 8192, 9216, 8192, 128
    num_heads = n // (3 * head_dim)
    a, a_scale, w, w_scale = _fp4_problem(m, n, k)
    a_sf, w_sf = _swizzled(a_scale, w_scale)
    q_weight = torch.rand((head_dim,), device="cuda", dtype=torch.bfloat16)
    k_weight = torch.rand((head_dim,), device="cuda", dtype=torch.bfloat16)
    qkv_scale = torch.tensor([0.5, 1.5, 2.0], device="cuda")
    angles = torch.randn((m, head_dim // 2), device="cuda")
    cos_sin = torch.cat((angles.cos(), angles.sin()), dim=-1).contiguous()
    positions = torch.arange(m, device="cuda", dtype=torch.int64)
    actual = fp4_qkv_qknorm_rope(
        a,
        a_sf,
        1.0,
        w,
        w_sf,
        1.0,
        q_weight,
        k_weight,
        cos_sin,
        positions,
        num_q_heads=num_heads,
        num_kv_heads=num_heads,
        head_dim=head_dim,
        qkv_scale=qkv_scale,
    )
    prepared = prepare_fp4_qkv_qknorm_rope(
        w,
        w_sf,
        1.0,
        1.0,
        q_weight,
        k_weight,
        num_q_heads=num_heads,
        num_kv_heads=num_heads,
        head_dim=head_dim,
        qkv_scale=qkv_scale,
    )
    prepared_actual = prepared(a, a_sf, cos_sin=cos_sin, positions=positions)
    base = (
        F.linear(dequantize_nvfp4(a, a_scale, 1.0), dequantize_nvfp4(w, w_scale, 1.0))
        .bfloat16()
        .float()
    )
    shaped = base.view(m, 3, num_heads, head_dim)
    shaped.mul_(qkv_scale.view(1, 3, 1, 1))
    half = head_dim // 2
    cos, sin = cos_sin[:, None, :half], cos_sin[:, None, half:]
    for part, norm_weight in ((0, q_weight), (1, k_weight)):
        value = shaped[:, part]
        value = (
            value
            * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)
            * norm_weight.float()
        )
        pairs = value.view(m, num_heads, half, 2)
        shaped[:, part] = torch.stack(
            (
                pairs[..., 0] * cos - pairs[..., 1] * sin,
                pairs[..., 1] * cos + pairs[..., 0] * sin,
            ),
            -1,
        ).flatten(-2)
    torch.testing.assert_close(
        actual.float(), shaped.reshape_as(base), atol=1.0, rtol=8e-2
    )
    torch.testing.assert_close(prepared_actual, actual)


@pytest.mark.parametrize("scheduler", ["static", "clc_dynamic"])
@pytest.mark.parametrize("tile_n", [128, 256])
@pytest.mark.parametrize("epilogue", ["linear", "swiglu"])
def test_nvfp4_multicast_scale_b_multiple_pair_rows(scheduler, tile_n, epilogue):
    _require_gemm_gpu()
    # Six K tiles wrap the five-stage operand ring. A 4x4 cluster has two
    # pair rows: SFB must deliver every box to both CTAs in the remote row,
    # or its full barrier stalls and prevents cluster-wide stage reuse.
    m, n, k = 1024, 2048, 1536
    a, a_scale, b, b_scale = _fp4_problem(m, n, k)
    sfa, sfb = _swizzled(a_scale, b_scale)
    logical_n = n // 2 if epilogue == "swiglu" else n
    out = torch.empty((m, logical_n), device="cuda", dtype=torch.bfloat16)
    major, minor = get_compute_capability(torch.device("cuda"))
    config = PrimsTsGemmConfig(
        major * 10 + minor,
        "nvfp4_e2m1",
        "bf16",
        epilogue,
        False,
        None,
        None,
        tile_n=tile_n,
        tile_k=256,
        cluster_shape=(4, 4, 1),
        scheduler=scheduler,
    )
    for _ in range(2):
        _launch(config, a, b, out, (m, n, k), sfa=sfa, sfb=sfb)
        torch.cuda.synchronize()
    expected = F.linear(
        dequantize_nvfp4(a, a_scale, 1.0), dequantize_nvfp4(b, b_scale, 1.0)
    )
    if epilogue == "swiglu":
        expected = expected[:, 0::2] * F.silu(expected[:, 1::2])
    torch.testing.assert_close(out.float(), expected, atol=1.0, rtol=6e-2)


@pytest.mark.parametrize("tile_n", [128, 256])
@pytest.mark.parametrize("tile_k", [768, 256])
@pytest.mark.parametrize("k", [256, 768, 1024, 1536, 7680, 8192])
def test_nvfp4_3x(tile_n, tile_k, k):
    _require_sm103("MMA-K=96 requires SM103")
    m, n = 512, 512
    a, a_scale, b, b_scale = _fp4_problem(m, n, k)
    sfa, sfb = _swizzled(a_scale, b_scale)
    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    config = PrimsTsGemmConfig(
        103,
        "nvfp4_e2m1",
        "bf16",
        "linear",
        False,
        None,
        None,
        tile_n=tile_n,
        tile_k=tile_k,
        nvfp4_mma_k=96,
    )
    _launch(config, a, b, out, (m, n, k), sfa=sfa, sfb=sfb)
    expected = dequantize_nvfp4(a, a_scale, 1.0) @ dequantize_nvfp4(b, b_scale, 1.0).T
    torch.testing.assert_close(out.float(), expected, atol=2e-3, rtol=4e-3)


@pytest.mark.parametrize("tile_k", [768, 256])
@pytest.mark.parametrize("scheduler", ["clc_dynamic", "static"])
def test_nvfp4_3x_persistent_tail_and_graph(tile_k, scheduler):
    _require_sm103("MMA-K=96 requires SM103")
    # Many output tiles exercise persistent reuse and both AB/SF ring wraps;
    # M is partial and K ends in the second 256-element box of a period.
    m, n, k = 4097, 1024, 1280
    a, a_scale, b, b_scale = _fp4_problem(m, n, k)
    sfa, sfb = _swizzled(a_scale, b_scale)
    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    config = PrimsTsGemmConfig(
        103,
        "nvfp4_e2m1",
        "bf16",
        "linear",
        False,
        None,
        None,
        tile_k=tile_k,
        nvfp4_mma_k=96,
        scheduler=scheduler,
        cluster_shape=(2, 1, 1),
    )
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    config = replace(config, has_bias=True)

    def run():
        _launch(config, a, b, out, (m, n, k), sfa=sfa, sfb=sfb, bias=bias, scale=0.375)

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    # Replay with changed data catches state retained across launches.
    a.bitwise_xor_(0x88)
    for _ in range(3):
        graph.replay()
    # The low-level scalar epilogue scales the biased accumulator.
    expected = (
        (dequantize_nvfp4(a, a_scale, 1.0) @ dequantize_nvfp4(b, b_scale, 1.0).T)
        + bias.float()
    ) * 0.375
    torch.testing.assert_close(out.float(), expected, atol=2e-3, rtol=4e-3)


@pytest.mark.parametrize(
    "arch,operand,tile_k",
    [(100, "nvfp4_e2m1", 256), (103, "fp8_e4m3", 256), (103, "nvfp4_e2m1", 512)],
)
def test_nvfp4_3x_rejects_unsupported_config(arch, operand, tile_k):
    config = PrimsTsGemmConfig(
        arch,
        operand,
        "bf16",
        "linear",
        False,
        None,
        None,
        tile_k=tile_k,
        nvfp4_mma_k=96,
    )
    with pytest.raises(ValueError, match="MMA-K=96"):
        _load_kernel(config)


@pytest.mark.parametrize(
    "epilogue,out_dtype",
    [
        ("linear", torch.bfloat16),
        ("swiglu", torch.bfloat16),
        ("swiglu", torch.float8_e4m3fn),
        ("swiglu", torch.uint8),
        ("qkv_qknorm_rope", torch.bfloat16),
    ],
)
def test_nvfp4_3x_prepared_epilogues(epilogue, out_dtype):
    _require_sm103("MMA-K=96 requires SM103")
    m, n, k = 257, 1536, 1024
    a, a_scale, b, b_scale = _fp4_problem(m, n, k)
    sfa, sfb = _swizzled(a_scale, b_scale)
    options = dict(epilogue=epilogue, out_dtype=out_dtype)
    inputs = {}
    if epilogue == "qkv_qknorm_rope":
        options.update(
            q_norm=torch.rand(128, device="cuda", dtype=torch.bfloat16),
            k_norm=torch.rand(128, device="cuda", dtype=torch.bfloat16),
            head_dim=128,
            is_neox=False,
        )
        angles = torch.randn(m, 64, device="cuda")
        inputs.update(
            cos_sin=torch.cat((angles.cos(), angles.sin()), dim=-1),
            positions=torch.arange(m, device="cuda", dtype=torch.int64),
        )
    reference = PreparedFp4Linear(b, sfb, 1.0, 1.0, **options)(a, sfa, **inputs)
    prepared = PreparedFp4Linear(b, sfb, 1.0, 1.0, mma_k=96, **options)
    actual = prepared(a, sfa, **inputs)
    if isinstance(actual, tuple):
        torch.testing.assert_close(actual[0], reference[0], rtol=0, atol=0)
        torch.testing.assert_close(actual[2], reference[2], rtol=0, atol=0)
        # M=257 leaves padding rows in the 128x4 scale allocation. Those
        # bytes are unspecified; compare the scales used by logical rows.
        torch.testing.assert_close(
            dequantize_nvfp4_128x4(*actual),
            dequantize_nvfp4_128x4(*reference),
            rtol=0,
            atol=0,
        )
    else:
        torch.testing.assert_close(
            actual.float(), reference.float(), atol=2e-3, rtol=4e-3
        )


@pytest.mark.parametrize("mma_k", [64, 96])
@pytest.mark.parametrize("k", [256, 7680, 8192])
@pytest.mark.parametrize(
    "scheduler,cluster_n", [("clc_dynamic", 2), ("clc_dynamic", 1), ("static", 1)]
)
def test_nvfp4_overlap_persistent_graph(mma_k, k, scheduler, cluster_n):
    _require_sm103("Overlap comparison requires SM103")
    # More tiles than resident CTA pairs, a partial M tile, AB/SF ring reuse,
    # and both exact and partial K scheduling periods.
    m, n = 4097, 1024
    a, asf, b, bsf = _fp4_problem(m, n, k)
    sfa, sfb = _swizzled(asf, bsf)
    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    config = replace(
        _overlap_config(mma_k, scheduler=scheduler, cluster_shape=(2, cluster_n, 1)),
        has_bias=True,
    )

    def run():
        _launch(config, a, b, out, (m, n, k), sfa=sfa, sfb=sfb, bias=bias, scale=0.375)

    def check():
        expected = (
            dequantize_nvfp4(a, asf, 1.0) @ dequantize_nvfp4(b, bsf, 1.0).T
            + bias.float()
        ) * 0.375
        torch.testing.assert_close(out.float(), expected, atol=2e-3, rtol=4e-3)

    run()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    a.bitwise_xor_(0x88)
    for _ in range(3):
        graph.replay()
    check()


@pytest.mark.parametrize("mma_k", [64, 96])
def test_nvfp4_overlap_prepared_and_config_isolation(mma_k):
    _require_sm103("Overlap comparison requires SM103")
    a, asf, b, bsf = _fp4_problem(513, 512, 1024)
    sfa, sfb = _swizzled(asf, bsf)
    expected = dequantize_nvfp4(a, asf, 1.0) @ dequantize_nvfp4(b, bsf, 1.0).T
    plain = prepare_fp4_linear(b, sfb, 1.0, 1.0, mma_k=mma_k)
    overlap = prepare_fp4_linear(b, sfb, 1.0, 1.0, mma_k=mma_k, tmem_overlap=True)
    assert plain.config != overlap.config
    assert plain._module is not overlap._module
    assert plain._module.num_epilogue_warps == 4
    assert overlap._module.num_epilogue_warps == 8
    assert overlap._module.acc_stages == 2
    for prepared in (plain, overlap, plain, overlap):
        torch.testing.assert_close(
            prepared(a, sfa).float(), expected, atol=2e-3, rtol=4e-3
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"tile_n": 128},
        {"tile_k": 768},
        {"epilogue_warps": 4},
        {"epilogue_warps": 7},
        {"operand_format": "fp8_e4m3", "nvfp4_mma_k": 64},
        {"epilogue": "qkv_qknorm_rope", "head_dim": 128, "is_neox": False},
        {"output_format": "fp8_e4m3"},
    ],
)
def test_nvfp4_overlap_rejects_unsupported_config(overrides):
    with pytest.raises(ValueError, match="overlap|epilogue_warps"):
        _load_kernel(replace(_overlap_config(), **overrides))


@pytest.mark.parametrize("mma_k", [64, 96])
@pytest.mark.parametrize("k", [256, 3072])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.uint8])
def test_nvfp4_swiglu_overlap_matches_plain(mma_k, k, out_dtype):
    _require_sm103("Overlap comparison requires SM103")
    # Overlap changes only the accumulator schedule: SwiGLU output, including
    # the packed NVFP4 payload and its scales, must be bit-identical.
    m, n = 4097, 2048
    a, asf, b, bsf = _fp4_problem(m, n, k)
    sfa, sfb = _swizzled(asf, bsf)
    kwargs = dict(out_dtype=out_dtype, mma_k=mma_k)
    if out_dtype == torch.uint8:
        kwargs["output_quant_scale"] = torch.tensor([0.25], device="cuda")
    plain = prepare_fp4_linear_swiglu(b, sfb, 1.0, 1.0, **kwargs)
    overlap = prepare_fp4_linear_swiglu(b, sfb, 1.0, 1.0, tmem_overlap=True, **kwargs)
    assert overlap._module.acc_stages == 2
    assert plain.config != overlap.config
    expected, actual = plain(a, sfa), overlap(a, sfa)
    if out_dtype == torch.uint8:
        # Scale rows past M in the last 128-row block are padding that neither
        # kernel writes, so compare the payload and the decoded real rows.
        torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
        scale = kwargs["output_quant_scale"]
        torch.testing.assert_close(
            dequantize_nvfp4_128x4(actual[0], actual[1], scale),
            dequantize_nvfp4_128x4(expected[0], expected[1], scale),
            rtol=0,
            atol=0,
        )
    else:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        linear = F.linear(dequantize_nvfp4(a, asf, 1.0), dequantize_nvfp4(b, bsf, 1.0))
        reference = linear[:, 0::2] * F.silu(linear[:, 1::2])
        torch.testing.assert_close(actual.float(), reference, atol=1.0, rtol=6e-2)


@pytest.mark.parametrize("k", [256, 3072])
@pytest.mark.parametrize("has_qkv_scale", [False, True])
def test_nvfp4_qkv_overlap_matches_plain(k, has_qkv_scale):
    _require_sm103("Overlap comparison requires SM103")
    # The early release happens after the QKNorm sum pass; output must be
    # bit-identical to the single-accumulator schedule.
    m, head_dim, num_heads = 4097, 128, 4
    n = 3 * num_heads * head_dim
    a, asf, b, bsf = _fp4_problem(m, n, k)
    sfa, sfb = _swizzled(asf, bsf)
    q_weight = torch.rand((head_dim,), device="cuda", dtype=torch.bfloat16)
    k_weight = torch.rand((head_dim,), device="cuda", dtype=torch.bfloat16)
    qkv_scale = torch.tensor([0.5, 1.5, 2.0], device="cuda") if has_qkv_scale else None
    angles = torch.randn((m, head_dim // 2), device="cuda")
    cos_sin = torch.cat((angles.cos(), angles.sin()), dim=-1).contiguous()
    positions = torch.arange(m, device="cuda", dtype=torch.int64)
    kwargs = dict(
        num_q_heads=num_heads,
        num_kv_heads=num_heads,
        head_dim=head_dim,
        qkv_scale=qkv_scale,
    )
    overlap = prepare_fp4_qkv_qknorm_rope(
        b, sfb, 1.0, 1.0, q_weight, k_weight, tmem_overlap=True, **kwargs
    )
    plain = prepare_fp4_qkv_qknorm_rope(b, sfb, 1.0, 1.0, q_weight, k_weight, **kwargs)
    assert overlap._module.acc_stages == 2
    expected = plain(a, sfa, cos_sin=cos_sin, positions=positions)
    actual = overlap(a, sfa, cos_sin=cos_sin, positions=positions)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_nvfp4_overlap_epilogue_warp_config_isolation():
    base = replace(_overlap_config(), tmem_overlap=False)
    wide = replace(base, epilogue_warps=8)
    assert base != wide
    assert _load_kernel(base).num_epilogue_warps == 4
    assert _load_kernel(wide).num_epilogue_warps == 8
