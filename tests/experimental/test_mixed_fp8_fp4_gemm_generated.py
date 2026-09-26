"""Analytical mixed E4M3/E2M1 values and changed-input current-stream replay."""

import pytest
import torch
from flashinfer.experimental.deepgemm_mixed_gemm import mixed_gemm as _runtime
from flashinfer.fp8_fp4_gemm import prepare_fp8_fp4_gemm

_CASES = [
    (m, n, k, None, 128, 32)
    for m in (1, 16, 128, 512, 4096)
    for n, k in ((4608, 5120), (5120, 2304))
]
_CASES += [
    (256, 256, 256, None, 128, 32),
    (4096, 7168, 4096, None, 224, 128),
    (256, 224, 128, "bk128_s6", 128, 128),
    (4096, 7168, 4096, "bk128_s6", 128, 128),
    (256, 128, 256, "bk256_s4", 128, 128),
]


def _skip_unless_exported():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    try:
        arch = _runtime.device_arch(torch.device("cuda"))
    except RuntimeError as error:
        pytest.skip(str(error))
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    if sms not in _runtime.supported_num_sms(arch):
        pytest.skip(
            f"The exported {arch} schedules cover {_runtime.supported_num_sms(arch)} SMs, "
            f"this device has {sms}"
        )
    return arch


def _route_config(arch, m, n, k, variant, block_n, gran_k_a):
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    key = _runtime.route_key(
        dict(
            M=m,
            N=n,
            K=k,
            num_sms=sms,
            variant=variant,
            block_n=block_n,
            gran_k_a=gran_k_a,
        )
    )
    return _runtime._catalog()["arches"][arch]["routes"][key]["config"]


@pytest.mark.parametrize("m,n,k,variant,block_n,gran_k_a", _CASES)
def test_mixed_values_packing_stream_replay(m, n, k, variant, block_n, gran_k_a):
    arch = _skip_unless_exported()
    # The catalogued route fixes the physical A/output rows and the packed scale
    # geometry (source-selected routes bind logical M; generic routes pad to 256).
    cfg = _route_config(arch, m, n, k, variant, block_n, gran_k_a)
    a = torch.full((cfg["input_m"], k), 0x38, dtype=torch.uint8, device="cuda").view(
        torch.float8_e4m3fn
    )
    b = torch.full((n, k // 2), 0x22, dtype=torch.uint8, device="cuda")
    # Constant scale1 has exponent127 in every packed byte, whichever layout the
    # route declares (native per-granularity bytes or the BK256 broadcast words).
    sfa = torch.full(
        (cfg["sfa_words"], cfg["sfa_mn"]), 0x7F7F7F7F, dtype=torch.int32, device="cuda"
    )
    sfb = torch.full(
        (cfg["sfb_words"], cfg["sfb_mn"]), 0x7F7F7F7F, dtype=torch.int32, device="cuda"
    )
    plan = prepare_fp8_fp4_gemm(
        a, b, sfa, sfb, m=m, variant=variant, block_n=block_n, gran_k_a=gran_k_a
    )
    assert tuple(plan.output.shape) == (m, n)
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
            plan.storage.fill_(float("nan"))
            graph.replay()
        stream.synchronize()
        torch.testing.assert_close(
            plan.output, torch.full_like(plan.output, sign * k), atol=0, rtol=0
        )
