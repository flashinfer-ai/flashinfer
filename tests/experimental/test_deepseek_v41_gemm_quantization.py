# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.

import pytest
import torch

from flashinfer.deepseek_v41 import deepseek_v41_quantize_gemm
from .test_deepseek_v41 import reference


@pytest.mark.parametrize(
    "m,k",
    [
        (0, 1280),
        (1, 128),
        (1, 1280),
        (2, 5120),
        (4, 8192),
        (17, 2304),
        (129, 6144),
        (3, 32768),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_gemm_quant_exact_bytes_scale_layout_padding_and_changed_graph(m, k, dtype):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
    ):
        pytest.skip("GEMM quantization requires SM100/SM103")
    torch.manual_seed(41723 + m + k)
    x = torch.randn(m, k, device="cuda", dtype=dtype)
    data = torch.empty(m, k, device="cuda", dtype=torch.float8_e4m3fn)
    stride = (m + 3) // 4 * 4
    storage = torch.full(
        (k // 128, stride), 0x45454545, device="cuda", dtype=torch.int32
    )
    scales = storage.T[:m]
    if not m:
        # Empty tensor default strides do not encode the declared packed
        # layout; explicitly request its zero MN stride for this no-op case.
        scales = torch.empty_strided(
            (0, k // 128), (1, 0), device="cuda", dtype=torch.int32
        )

    def run():
        return deepseek_v41_quantize_gemm(x, data=data, scales=scales)

    def verify():
        if m:
            expected_data, sf = reference(x, "swa_mxfp8")
            expected_sf = (
                sf.reshape(m, k // 128, 4)
                .contiguous()
                .view(torch.int32)
                .reshape(m, k // 128)
            )
            torch.testing.assert_close(
                data.view(torch.uint8), expected_data, atol=0, rtol=0
            )
            torch.testing.assert_close(scales, expected_sf, atol=0, rtol=0)
        torch.testing.assert_close(
            storage[:, m:], torch.full_like(storage[:, m:], 0x45454545), atol=0, rtol=0
        )

    run()
    verify()
    if not m:
        allocated = deepseek_v41_quantize_gemm(x)
        assert allocated[0].shape == (0, k)
        assert allocated[1].shape == (0, k // 128)
        assert allocated[1].stride() == (1, 0)
        return
    allocated = deepseek_v41_quantize_gemm(x)
    torch.testing.assert_close(
        allocated[0].view(torch.uint8), data.view(torch.uint8), atol=0, rtol=0
    )
    torch.testing.assert_close(allocated[1], scales, atol=0, rtol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for factor in (0.125, 16.0, 1.0):
        x.copy_(torch.randn_like(x) * factor)
        x[0, :32].zero_()
        graph.replay()
        verify()
    torch.cuda.set_sync_debug_mode("error")
    try:
        run()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    with pytest.raises(ValueError, match="overlap"):
        deepseek_v41_quantize_gemm(
            x,
            data=x.view(torch.float8_e4m3fn).flatten()[: m * k].view(m, k),
            scales=scales,
        )


def test_gemm_quantizer_deepgemm_scale_interop():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
    ):
        pytest.skip("DeepGEMM interop requires SM100/SM103")
    deep_gemm = pytest.importorskip("deep_gemm")
    torch.manual_seed(41991)
    x = torch.randn(3, 5120, device="cuda", dtype=torch.bfloat16)
    _, actual = deepseek_v41_quantize_gemm(x)
    _, raw_sf = reference(x, "swa_mxfp8")
    reference_sf = deep_gemm.transform_sf_into_required_layout(
        raw_sf.view(torch.float8_e8m0fnu).float(), 3, 5120, (1, 32)
    )
    assert (
        actual.shape == reference_sf.shape and actual.stride() == reference_sf.stride()
    )
    torch.testing.assert_close(actual, reference_sf, atol=0, rtol=0)
