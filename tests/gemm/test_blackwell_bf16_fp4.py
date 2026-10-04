# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Numerical tests of the public source-built GEMM backends on SM100/SM103."""

import pytest
import torch

from flashinfer import mm_bf16_fp4, prepare_bf16_fp4_weights

# Smallest M whose 16-row activation-tile count exceeds the CUDA Y grid limit.
LARGE_M = 65535 * 16 + 16


@pytest.fixture(scope="module")
def blackwell_device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    device = torch.device("cuda")
    if torch.cuda.get_device_capability(device) not in ((10, 0), (10, 3)):
        pytest.skip("the source-built backends require SM100 or SM103")
    return device


def _make_weights(n, k, device, generator):
    # E2M1 stores a sign bit and the magnitudes below. Construct codes directly
    # so this GEMM test does not depend on a separate quantizer's correctness.
    codes = torch.randint(
        0, 16, (n, k), dtype=torch.uint8, device=device, generator=generator
    )
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    magnitudes = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32, device=device
    )
    values = magnitudes[(codes & 7).long()]

    # Every scale and dequantized value is exactly representable in FP16.
    scale_values = torch.tensor(
        [0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0],
        dtype=torch.float32,
        device=device,
    )
    scale_indices = torch.randint(
        len(scale_values), (n, k // 16), device=device, generator=generator
    )
    scales = scale_values[scale_indices]
    reference_weights = torch.where((codes & 8) != 0, -values, values) * (
        scales.repeat_interleave(16, dim=1)
    )

    # Canonical 128x4 scale layout: [N/128, K_sf/4, N%32, N/32%4, K_sf%4].
    # Build it from padded logical rows without using backend layout helpers.
    n_blocks = (n + 127) // 128
    k_blocks = (k // 16 + 3) // 4
    padded_scales = torch.zeros(
        (n_blocks * 128, k_blocks * 4), dtype=torch.uint8, device=device
    )
    padded_scales[:n, : k // 16] = scales.to(torch.float8_e4m3fn).view(torch.uint8)
    canonical_scales = (
        padded_scales.reshape(n_blocks, 4, 32, k_blocks, 4)
        .permute(0, 3, 2, 1, 4)
        .contiguous()
        .reshape(-1)
    )
    return packed, canonical_scales, reference_weights


def _reference(a, reference_weights, alpha_value, out_dtype):
    previous_precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        reference = a.float() @ reference_weights.T
    finally:
        torch.set_float32_matmul_precision(previous_precision)
    if alpha_value is not None:
        reference = reference * alpha_value
    return reference.to(out_dtype)


def _case(device, backend, shape, out_dtype, alpha_value, seed):
    m, n, k = shape
    generator = torch.Generator(device=device).manual_seed(seed)
    a = (
        torch.randn((m, k), dtype=torch.bfloat16, device=device, generator=generator)
        * 0.125
    )
    packed, scales, reference_weights = _make_weights(n, k, device, generator)
    alpha = (
        None
        if alpha_value is None
        else torch.tensor([alpha_value], dtype=torch.float32, device=device)
    )
    prepared = prepare_bf16_fp4_weights(packed, scales, alpha, backend=backend)
    return a, prepared, _reference(a, reference_weights, alpha_value, out_dtype)


@pytest.mark.parametrize(
    "backend,shape,out_dtype,alpha_value,caller_out,enable_pdl",
    [
        pytest.param(
            "blackwell-native",
            (16, 1024, 1024),
            torch.bfloat16,
            None,
            False,
            False,
            id="native-m16-bf16",
        ),
        pytest.param(
            "blackwell-tiled",
            (16, 1024, 1024),
            torch.bfloat16,
            None,
            False,
            False,
            id="tiled-m16-bf16",
        ),
        pytest.param(
            "blackwell-native",
            (16, 1024, 1024),
            torch.float16,
            0.75,
            True,
            True,
            id="native-m16-fp16-alpha-out-pdl",
        ),
        pytest.param(
            "blackwell-tiled",
            (16, 1024, 1024),
            torch.bfloat16,
            0.75,
            True,
            True,
            id="tiled-m16-alpha-out-pdl",
        ),
        pytest.param(
            "blackwell-native",
            (7, 65, 48),
            torch.bfloat16,
            -0.5,
            True,
            False,
            id="native-ragged-m-n-k48",
        ),
        pytest.param(
            "blackwell-native",
            (1, 129, 16),
            torch.float16,
            None,
            False,
            True,
            id="native-m1-ragged-n-k16",
        ),
        pytest.param(
            "blackwell-native",
            (129, 192, 128),
            torch.bfloat16,
            0.75,
            False,
            True,
            id="native-large-m-tail",
        ),
        pytest.param(
            "blackwell-tiled",
            (7, 4160, 192),
            torch.bfloat16,
            -0.5,
            True,
            False,
            id="tiled-short-k-ragged-m",
        ),
        pytest.param(
            "blackwell-tiled",
            (13, 192, 48),
            torch.bfloat16,
            None,
            False,
            True,
            id="tiled-ragged-m-k48",
        ),
        # Routes the earlier parametrization never reached.
        pytest.param(
            "blackwell-tiled",
            (33, 128, 16),
            torch.bfloat16,
            0.5,
            False,
            True,
            id="tiled-k16-persistent",
        ),
        pytest.param(
            "blackwell-tiled",
            (5, 4096, 32),
            torch.bfloat16,
            None,
            True,
            False,
            id="tiled-k32",
        ),
        pytest.param(
            "blackwell-tiled",
            (40, 256, 256),
            torch.bfloat16,
            -0.25,
            False,
            True,
            id="tiled-m64-route",
        ),
        pytest.param(
            "blackwell-tiled",
            (17, 64, 80),
            torch.bfloat16,
            None,
            False,
            False,
            id="tiled-generic-fallback-k80",
        ),
        pytest.param(
            "blackwell-tiled",
            (17, 3072, 3072),
            torch.bfloat16,
            0.75,
            False,
            True,
            id="tiled-m32-long-k",
        ),
        pytest.param(
            "blackwell-native",
            (33, 128, 192),
            torch.float16,
            0.5,
            False,
            False,
            id="native-fp16-cp-async-alpha",
        ),
        pytest.param(
            "blackwell-native",
            (768, 2112, 2048),
            torch.bfloat16,
            0.75,
            False,
            True,
            id="native-m768-n2112-k2048",
        ),
        pytest.param(
            "blackwell-native",
            (1, 4096, 4096),
            torch.bfloat16,
            None,
            False,
            True,
            id="native-m1-n4096-k4096",
        ),
        pytest.param(
            "blackwell-native",
            (LARGE_M, 64, 16),
            torch.bfloat16,
            None,
            False,
            True,
            id="native-large-m-flat-grid",
        ),
        pytest.param(
            "blackwell-tiled",
            (LARGE_M, 64, 80),
            torch.bfloat16,
            0.5,
            False,
            False,
            id="tiled-large-m-flat-grid",
        ),
    ],
)
def test_blackwell_bf16_fp4_numerical(
    blackwell_device, backend, shape, out_dtype, alpha_value, caller_out, enable_pdl
):
    m, n, _k = shape
    a, prepared, reference = _case(
        blackwell_device, backend, shape, out_dtype, alpha_value, 42
    )
    out = (
        torch.full((m, n), float("nan"), dtype=out_dtype, device=blackwell_device)
        if caller_out
        else None
    )
    actual = mm_bf16_fp4(
        a,
        *prepared,
        backend=backend,
        out_dtype=out_dtype,
        out=out,
        enable_pdl=enable_pdl,
    )
    if caller_out:
        assert actual is out

    # Compare to the represented FP4 weights, not to unquantized source weights.
    # Thus the standard BF16/FP16 pointwise tolerance need not include FP4 error.
    torch.testing.assert_close(actual, reference, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("backend", ["blackwell-native", "blackwell-tiled"])
def test_blackwell_bf16_fp4_many_distinct_activations(blackwell_device, backend):
    """Every call encodes its own descriptors; distinct activation and output buffers never exhaust a cache."""

    m, n, k = 3, 64, 256
    generator = torch.Generator(device=blackwell_device).manual_seed(7)
    packed, scales, reference_weights = _make_weights(n, k, blackwell_device, generator)
    prepared = prepare_bf16_fp4_weights(packed, scales, None, backend=backend)
    activations = [
        torch.randn(
            (m, k), dtype=torch.bfloat16, device=blackwell_device, generator=generator
        )
        * 0.125
        for _ in range(4200)
    ]
    outputs = [mm_bf16_fp4(a, *prepared, backend=backend) for a in activations]
    torch.cuda.synchronize()
    # Every 700th call, plus the two calls on either side of the 4096-call mark
    # where the previous binding's fixed-size descriptor cache ran out.
    for index in (*range(0, 4200, 700), 4096, 4199):
        torch.testing.assert_close(
            outputs[index],
            _reference(activations[index], reference_weights, None, torch.bfloat16),
            atol=1e-2,
            rtol=1e-2,
        )


@pytest.mark.parametrize(
    "backend,shape",
    [
        pytest.param("blackwell-native", (16, 1024, 1024), id="native-m16"),
        pytest.param("blackwell-tiled", (16, 1024, 1024), id="tiled-m16"),
        pytest.param("blackwell-native", (1, 4096, 4096), id="native-split-k"),
        pytest.param("blackwell-native", (768, 2112, 2048), id="native-m768"),
    ],
)
def test_blackwell_bf16_fp4_cuda_graph_capture(blackwell_device, backend, shape):
    """A first call inside CUDA Graph capture is legal and replays bitwise."""

    m, n, _k = shape
    a, prepared, reference = _case(
        blackwell_device, backend, shape, torch.bfloat16, 0.75, 11
    )
    out = torch.full(
        (m, n), float("nan"), dtype=torch.bfloat16, device=blackwell_device
    )
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        mm_bf16_fp4(a, *prepared, backend=backend, out=out, enable_pdl=True)
    torch.cuda.synchronize()
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, reference, atol=1e-2, rtol=1e-2)
    replayed = out.clone()
    a.copy_(a * 0.5)
    graph.replay()
    torch.cuda.synchronize()
    assert not torch.equal(out, replayed)
    torch.testing.assert_close(
        out, (reference.float() * 0.5).to(torch.bfloat16), atol=1e-2, rtol=1e-2
    )
