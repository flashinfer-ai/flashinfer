"""Mixed E4M3/E2M1 GEMM: analytical values, random-input references, graph replay."""

import pytest
import torch
from flashinfer.experimental.deepgemm_mixed_gemm import mixed_gemm as _runtime
from flashinfer.fp8_fp4_gemm import prepare_fp8_fp4_gemm

_SHIPPED = [
    (m, n, k, 32)
    for m in (1, 16, 128, 512, 4096)
    for n, k in ((4608, 5120), (5120, 2304))
]
# The shipped K=128 smoke row (256x224x128, gran_k_a=128) is excluded: its one-word A-scale tensor map is
# rejected by cuTensorMapEncodeTiled in the shipped programs as well.
_SHIPPED += [(4096, 7168, 4096, 128), (256, 256, 256, 32)]
_HELD_OUT = [
    (3, 4608, 5120, 32),
    (33, 2304, 4096, 32),
    (100, 5120, 2304, 32),
    (384, 5120, 2304, 32),
    (768, 4608, 5120, 32),
    (2048, 1536, 1024, 32),
    (4096, 7168, 4096, 32),
    (512, 7168, 4096, 128),
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
    """[mn, k] power-of-two FP32 scales -> [ceil(k/4), align(mn,4)] packed int32 words."""
    mn, k = sf.shape
    aligned_mn, aligned_k = -(-mn // 4) * 4, -(-k // 4) * 4
    bytes_ = torch.zeros(aligned_mn, aligned_k, dtype=torch.uint8, device=sf.device)
    bytes_[:mn, :k] = (sf.view(torch.int32) >> 23).to(torch.uint8)
    return (
        bytes_.view(-1)
        .view(torch.int32)
        .view(aligned_mn, aligned_k // 4)
        .t()
        .contiguous()
    )


def _quantize(m, n, k, gran_k_a, device, seed):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    a = torch.randn(m, k, dtype=torch.float32, device=device, generator=generator)
    b = torch.randn(n, k, dtype=torch.float32, device=device, generator=generator)
    sfa = _ceil_to_ue8m0(
        a.view(m, k // gran_k_a, gran_k_a).abs().amax(dim=2).clamp(1e-4) / 448.0
    )
    a_fp8 = (
        (a.view(m, k // gran_k_a, gran_k_a) / sfa.unsqueeze(2))
        .to(torch.float8_e4m3fn)
        .view(m, k)
        .contiguous()
    )
    sfb = _ceil_to_ue8m0(b.view(n, k // 32, 32).abs().amax(dim=2).clamp(1e-4) / 6.0)
    scaled = (b.view(n, k // 32, 32) / sfb.unsqueeze(2)).view(n, k)
    boundaries = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=device)
    codes = torch.bucketize(scaled.abs().clamp_max(6.0), boundaries).to(torch.uint8)
    codes |= ((scaled < 0) & (codes != 0)).to(torch.uint8) << 3
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
    lut = torch.tensor(_FP4_LUT, dtype=torch.float32, device=device)
    a_ref = a_fp8.float() * sfa.repeat_interleave(gran_k_a, dim=1)
    b_ref = lut[codes.long()] * sfb.repeat_interleave(32, dim=1)
    return a_fp8, packed, _pack_ue8m0(sfa), _pack_ue8m0(sfb), a_ref @ b_ref.t()


@pytest.mark.parametrize("m,n,k,gran_k_a", _SHIPPED + _HELD_OUT)
def test_random_inputs_match_reference(m, n, k, gran_k_a):
    device = _device()
    a, b, sfa, sfb, expected = _quantize(
        m, n, k, gran_k_a, device, seed=m * 7 + n * 3 + k
    )
    plan = prepare_fp8_fp4_gemm(a, b, sfa, sfb, gran_k_a=gran_k_a)
    assert tuple(plan.output.shape) == (m, n)
    out = plan.run()
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), expected, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("m,n,k,gran_k_a", _SHIPPED)
def test_analytical_values_packing_stream_replay(m, n, k, gran_k_a):
    device = _device()
    arch, num_sms = _runtime.device_facts(0)
    route = _runtime.select_route(m, n, k, num_sms=num_sms, gran_k_a=gran_k_a)
    geometry = _runtime.route_geometry(route, m, n, k, gran_k_a)
    a = torch.full((m, k), 0x38, dtype=torch.uint8, device=device).view(
        torch.float8_e4m3fn
    )
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
    plan = prepare_fp8_fp4_gemm(a, b, sfa, sfb, m=m, gran_k_a=gran_k_a)
    assert plan.route == route
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        plan.run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            plan.run()
    stream.synchronize()
    for sign in (1, -1, 1):
        with torch.cuda.stream(stream):
            a.view(torch.uint8).fill_(0x38 if sign > 0 else 0xB8)
            plan.output.fill_(float("nan"))
            graph.replay()
        stream.synchronize()
        torch.testing.assert_close(
            plan.output, torch.full_like(plan.output, sign * k), atol=0, rtol=0
        )


def test_logical_m_within_storage():
    device = _device()
    a, b, sfa, sfb, expected = _quantize(512, 4608, 5120, 32, device, seed=11)
    plan = prepare_fp8_fp4_gemm(a, b, sfa[:, :300].contiguous(), sfb, m=300)
    assert tuple(plan.output.shape) == (300, 4608)
    torch.testing.assert_close(plan.run().float(), expected[:300], atol=1e-2, rtol=1e-2)


def test_rejects_unrasterable_shapes():
    device = _device()
    a = torch.zeros(1, 128, dtype=torch.float8_e4m3fn, device=device)
    b = torch.zeros(128, 64, dtype=torch.uint8, device=device)
    sfa = torch.zeros(1, 4, dtype=torch.int32, device=device)
    sfb = torch.zeros(1, 128, dtype=torch.int32, device=device)
    with pytest.raises(NotImplementedError):
        prepare_fp8_fp4_gemm(a, b, sfa, sfb)
    with pytest.raises(ValueError):
        prepare_fp8_fp4_gemm(a, b, sfa, sfb, gran_k_a=64)
