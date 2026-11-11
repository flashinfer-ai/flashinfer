# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0.
# https://www.apache.org/licenses/LICENSE-2.0
"""Native E2M1 GEMM: analytical values, random-input references, runtime alpha, graph replay."""

import pytest
import torch
from flashinfer.experimental.deepgemm_fp4_gemm import fp4_gemm as _runtime
from flashinfer.fp4_gemm import prepare_fp4_gemm

_SHIPPED = [
    (m, n, k) for m in (16, 128, 512, 4096) for n, k in ((4608, 5120), (5120, 2304))
]
_SHIPPED += [(256, 128, 2048), (256, 128, 256)]
_HELD_OUT = [
    (1, 4608, 5120),
    (64, 5120, 2304),
    (100, 4608, 5120),
    (384, 5120, 2304),
    (768, 4608, 5120),
    (2048, 1536, 1024),
    (4096, 7168, 4096),
]
_FP4_LUT = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


def _device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    try:
        _runtime.device_facts(0)
    except RuntimeError as error:
        pytest.skip(str(error))
    return torch.device("cuda", 0)


def _ceil_to_ue8m0(x):
    bits = x.abs().float().view(torch.int32)
    exp = ((bits >> 23) & 0xFF) + (bits & 0x7FFFFF).bool().int()
    return (exp.clamp(1, 254) << 23).view(torch.float32)


def _pack_ue8m0(sf):
    """[mn, k/32] power-of-two FP32 scales -> [k/128, align(mn, 4)] packed int32 words."""
    mn, words = sf.shape
    aligned_mn, aligned_words = -(-mn // 4) * 4, -(-words // 4) * 4
    bytes_ = torch.zeros(aligned_mn, aligned_words, dtype=torch.uint8, device=sf.device)
    bytes_[:mn, :words] = (sf.view(torch.int32) >> 23).to(torch.uint8)
    return (
        bytes_.view(-1)
        .view(torch.int32)
        .view(aligned_mn, aligned_words // 4)
        .t()
        .contiguous()
    )


def _quantize_fp4(x):
    """[rows, k] FP32 -> packed E2M1 bytes, packed UE8M0 words and the dequantized FP32 operand."""
    rows, k = x.shape
    sf = _ceil_to_ue8m0(x.view(rows, k // 32, 32).abs().amax(dim=2).clamp(1e-4) / 6.0)
    scaled = (x.view(rows, k // 32, 32) / sf.unsqueeze(2)).view(rows, k)
    boundaries = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=x.device)
    codes = torch.bucketize(scaled.abs().clamp_max(6.0), boundaries).to(torch.uint8)
    codes |= ((scaled < 0) & (codes != 0)).to(torch.uint8) << 3
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
    lut = torch.tensor(_FP4_LUT, dtype=torch.float32, device=x.device)
    return packed, _pack_ue8m0(sf), lut[codes.long()] * sf.repeat_interleave(32, dim=1)


def _case(m, n, k, device, seed, alpha):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    a = torch.randn(m, k, dtype=torch.float32, device=device, generator=generator)
    b = torch.randn(n, k, dtype=torch.float32, device=device, generator=generator)
    a_packed, sfa, a_ref = _quantize_fp4(a)
    b_packed, sfb, b_ref = _quantize_fp4(b)
    return a_packed, b_packed, sfa, sfb, (a_ref @ b_ref.t()) * alpha


@pytest.mark.parametrize("m,n,k", _SHIPPED + _HELD_OUT)
@pytest.mark.parametrize("alpha", (1.0, -0.75))
def test_random_inputs_match_reference(m, n, k, alpha):
    device = _device()
    a, b, sfa, sfb, expected = _case(
        m, n, k, device, seed=m * 7 + n * 3 + k, alpha=alpha
    )
    plan = prepare_fp4_gemm(a, b, sfa, sfb, m=m, alpha=alpha)
    assert tuple(plan.output.shape) == (m, n)
    out = plan.run()
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), expected, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("m,n,k", _SHIPPED)
def test_analytical_values_alpha_stream_replay(m, n, k):
    device = _device()
    _arch, num_sms = _runtime.device_facts(0)
    route = _runtime.select_route(m, n, k, num_sms=num_sms)
    geometry = _runtime.route_geometry(route, m, n, k)
    alpha = 0.5
    a = torch.full((m, k // 2), 0x22, dtype=torch.uint8, device=device)
    b = torch.full((n, k // 2), 0x22, dtype=torch.uint8, device=device)
    # Constant scale 1 has exponent 127 in every packed byte.
    sfa = torch.full(
        (geometry["sfa_words"], geometry["sfa_mn"]),
        0x7F7F7F7F,
        dtype=torch.int32,
        device=device,
    )
    sfb = torch.full(
        (geometry["sfb_words"], geometry["sfb_mn"]),
        0x7F7F7F7F,
        dtype=torch.int32,
        device=device,
    )
    plan = prepare_fp4_gemm(
        a, b, sfa, sfb, m=m, alpha=alpha, num_stages=_runtime.route_stages(route)
    )
    assert plan.route == route
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        plan.run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            plan.run()
    stream.synchronize()
    for negative in (False, True, False):
        with torch.cuda.stream(stream):
            a.fill_(0xAA if negative else 0x22)
            plan.storage.fill_(float("nan"))
            graph.replay()
        stream.synchronize()
        expected = torch.full_like(plan.output, (-1 if negative else 1) * k * alpha)
        torch.testing.assert_close(plan.output, expected, atol=0, rtol=0)


def test_logical_m_within_storage():
    device = _device()
    a, b, sfa, sfb, expected = _case(512, 4608, 5120, device, seed=11, alpha=1.0)
    plan = prepare_fp4_gemm(a, b, sfa[:, :300].contiguous(), sfb, m=300)
    assert tuple(plan.output.shape) == (300, 4608)
    torch.testing.assert_close(plan.run().float(), expected[:300], atol=1e-2, rtol=1e-2)


def test_rejects_unrasterable_shapes_and_foreign_knobs():
    device = _device()
    a = torch.zeros(16, 128, dtype=torch.uint8, device=device)
    b = torch.zeros(128, 128, dtype=torch.uint8, device=device)
    sfa = torch.zeros(2, 16, dtype=torch.int32, device=device)
    sfb = torch.zeros(2, 128, dtype=torch.int32, device=device)
    with pytest.raises(NotImplementedError):
        prepare_fp4_gemm(a, b, sfa, sfb, m=16)  # one 128-column N tile: no CTA pair
    b = torch.zeros(256, 128, dtype=torch.uint8, device=device)
    sfb = torch.zeros(2, 256, dtype=torch.int32, device=device)
    with pytest.raises(NotImplementedError):
        prepare_fp4_gemm(a, b, sfa, sfb, m=16, num_stages=4)
    with pytest.raises(NotImplementedError):
        prepare_fp4_gemm(a, b, sfa, sfb, m=16, block_n=256)
