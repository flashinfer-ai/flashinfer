# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

import importlib

import pytest
import torch

from flashinfer.norm import rmsnorm_fp4quant
from flashinfer.api_logging import ExperimentalWarning

sm120 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="Triton RMSNorm NVFP4 requires SM120",
)


def _codes(x):
    # Independent enumeration: ties choose the even significand code.
    order = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7])
    levels = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    nearest = (x.abs()[..., None] - levels[order]).abs().argmin(-1)
    return (order[nearest] | (torch.signbit(x).long() << 3)).to(torch.uint8)


def _decode(q, sf, k, swizzled, global_scale):
    raw = q.cpu().view(torch.uint8).reshape(-1, k // 2)
    m = raw.shape[0]
    codes = torch.stack((raw & 15, raw >> 4), -1).reshape(m, k)
    scales = sf.cpu().view(torch.uint8)
    if swizzled:
        mt, kt = (m + 127) // 128, (k // 16 + 3) // 4
        matrix = scales.reshape(mt, kt, 32, 4, 4).permute(0, 3, 2, 1, 4)
        matrix = matrix.reshape(mt * 128, kt * 4)
        scales = matrix[:m, : k // 16].contiguous()
    else:
        scales = scales.reshape(m, k // 16)
    levels = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    values = levels[(codes & 7).long()] * torch.where(codes & 8 != 0, -1.0, 1.0)
    decoded = (
        values.reshape(m, k // 16, 16)
        * scales.view(torch.float8_e4m3fn).float()[..., None]
    )
    return codes, scales, decoded.reshape(m, k) / global_scale


@sm120
@pytest.mark.parametrize(
    "shape", [(1, 64), (7, 80), (33, 4112), (129, 7168), (2, 8192), (2, 3, 4096)]
)
@pytest.mark.parametrize("swizzled", [False, True])
@pytest.mark.parametrize("g", [None, 0.25, 1.0, 32.0])
def test_reference_and_preallocated(shape, swizzled, g):
    torch.manual_seed(73)
    x = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(shape[-1], dtype=torch.bfloat16, device="cuda")
    scale = None if g is None else torch.tensor([g], device="cuda")
    kwargs = dict(global_scale=scale, is_sf_swizzled_layout=swizzled, backend="triton")
    q, sf = rmsnorm_fp4quant(x, w, **kwargs)
    assert q.dtype == torch.float4_e2m1fn_x2
    assert sf.dtype == torch.float8_e4m3fn
    out = torch.empty_like(q)
    out_sf = torch.full_like(sf.view(torch.uint8), 205).view(sf.dtype)
    actual_q, actual_sf = rmsnorm_fp4quant(x, w, out, out_sf, **kwargs)
    assert actual_q is out and actual_sf is out_sf
    assert torch.equal(q.view(torch.uint8), out.view(torch.uint8))
    k = shape[-1]
    codes, scales, decoded = _decode(out, out_sf, k, swizzled, g or 1.0)
    # CPU FP64 RMSNorm is independent of the GPU reduction implementation.
    ref_x = x.cpu().double().reshape(-1, k)
    ref = (
        ref_x
        * torch.rsqrt(ref_x.square().mean(-1, keepdim=True) + 1e-6)
        * w.cpu().double()
    ).float()
    blocks = ref.reshape(-1, k // 16, 16)
    expected_sf = (
        (blocks.abs().amax(-1) / 6 * (g or 1.0)).clamp_max(448).to(torch.float8_e4m3fn)
    )
    inv = torch.where(expected_sf.float() > 0, (g or 1.0) / expected_sf.float(), 0.0)
    expected_codes = _codes(blocks * inv[..., None]).reshape_as(codes)
    assert (codes != expected_codes).float().mean() < 0.005
    assert (scales != expected_sf.view(torch.uint8)).float().mean() < 0.005
    assert torch.linalg.vector_norm(decoded - ref) / torch.linalg.vector_norm(ref) < 0.2
    if swizzled:
        m = ref.shape[0]
        matrix = (
            out_sf.cpu()
            .view(torch.uint8)
            .reshape((m + 127) // 128, (k // 16 + 3) // 4, 32, 4, 4)
        )
        matrix = matrix.permute(0, 3, 2, 1, 4).reshape(
            ((m + 127) // 128) * 128, ((k // 16 + 3) // 4) * 4
        )
        assert (matrix[m:] == 205).all()
        assert (matrix[:m, k // 16 :] == 205).all()


@sm120
@pytest.mark.parametrize(
    "pattern", ["zero_input", "zero_weight", "outlier", "underflow", "saturation"]
)
def test_special_values(pattern):
    torch.manual_seed(109)
    x = torch.randn(3, 256, device="cuda", dtype=torch.bfloat16)
    w = torch.ones(256, device="cuda", dtype=torch.bfloat16)
    if pattern == "zero_input":
        x.zero_()
    elif pattern == "zero_weight":
        w.zero_()
    elif pattern == "outlier":
        x[:, 0] = 1000
        w[17] = -2
    elif pattern == "underflow":
        w.fill_(1e-8)
    else:
        w.fill_(10000)
    q, sf = rmsnorm_fp4quant(x, w, backend="triton")
    _, _, decoded = _decode(q, sf, 256, False, 1.0)
    assert torch.isfinite(decoded).all()
    if pattern in ("zero_input", "zero_weight", "underflow"):
        assert torch.count_nonzero(decoded) == 0
    if pattern == "saturation":
        assert (sf.float() == 448).all()
        assert decoded.abs().max() <= 2688


@sm120
def test_graph_reads_updated_global_scale_and_current_stream():
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x = torch.randn(7, 80, device="cuda", dtype=torch.bfloat16)
        w = torch.ones(80, device="cuda", dtype=torch.bfloat16)
        g = torch.ones(1, device="cuda")
        q, sf = rmsnorm_fp4quant(x, w, global_scale=g, backend="triton")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            rmsnorm_fp4quant(x, w, q, sf, global_scale=g, backend="triton")
        for value in (0.25, 16.0):
            g.fill_(value)
            graph.replay()
            expected_q, expected_sf = rmsnorm_fp4quant(
                x, w, global_scale=g, backend="triton"
            )
            torch.testing.assert_close(
                q.view(torch.uint8), expected_q.view(torch.uint8), rtol=0, atol=0
            )
            torch.testing.assert_close(sf.float(), expected_sf.float(), rtol=0, atol=0)
    stream.synchronize()


@sm120
@pytest.mark.parametrize("shape", [(0, 80), (2, 0, 80)])
def test_empty(shape):
    x = torch.empty(shape, device="cuda", dtype=torch.bfloat16)
    w = torch.ones(80, device="cuda", dtype=torch.bfloat16)
    for swizzled in (False, True):
        q, sf = rmsnorm_fp4quant(x, w, backend="triton", is_sf_swizzled_layout=swizzled)
        assert q.shape == (*shape[:-1], 40)
        assert q.numel() == sf.numel() == 0


@sm120
@pytest.mark.parametrize(
    "case",
    [
        "dtype",
        "stride",
        "weight",
        "scale",
        "block_size",
        "format",
        "pdl",
        "eps",
        "output_dtype",
        "output_shape",
        "alias",
    ],
)
def test_invalid_inputs(case):
    x = torch.randn(2, 128, device="cuda", dtype=torch.bfloat16)
    w = torch.ones(128, device="cuda", dtype=torch.bfloat16)
    kwargs = dict(backend="triton")
    if case == "dtype":
        x = x.half()
    elif case == "stride":
        x = x[:, ::2]
        w = w[:64]
    elif case == "weight":
        w = w[:64]
    elif case == "scale":
        kwargs["global_scale"] = torch.ones(1, device="cpu")
    elif case == "block_size":
        kwargs["block_size"] = 32
    elif case == "format":
        kwargs["scale_format"] = "ue8m0"
    elif case == "pdl":
        kwargs["enable_pdl"] = True
    elif case == "eps":
        kwargs["eps"] = float("nan")
    elif case == "output_dtype":
        kwargs["y_fp4"] = torch.empty(2, 64, dtype=torch.uint8, device="cuda")
    elif case == "output_shape":
        kwargs["block_scale"] = torch.empty(
            16, dtype=torch.float8_e4m3fn, device="cuda"
        )
    else:
        kwargs["y_fp4"] = (
            x.view(torch.uint8)
            .flatten()[:128]
            .reshape(2, 64)
            .view(torch.float4_e2m1fn_x2)
        )
    with pytest.raises(ValueError):
        rmsnorm_fp4quant(x, w, **kwargs)


def test_default_dispatch_and_warning(monkeypatch):
    cute = importlib.import_module("flashinfer.cute_dsl")
    exp = importlib.import_module(
        "flashinfer.experimental.triton_rmsnorm_fp4quant.backend"
    )
    log = importlib.import_module("flashinfer.api_logging")
    calls = []
    monkeypatch.setattr(
        cute, "rmsnorm_fp4quant", lambda *args: calls.append("cute") or (None, None)
    )
    monkeypatch.setattr(
        exp, "run", lambda *args: calls.append("triton") or (None, None)
    )
    monkeypatch.setattr(log, "_WARNED_EXPERIMENTAL_BACKENDS", set())
    monkeypatch.delenv("FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS", raising=False)
    x = torch.empty(1, 64)
    w = torch.empty(64)
    rmsnorm_fp4quant(x, w)
    rmsnorm_fp4quant(x, w, backend="auto")
    monkeypatch.setenv("FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS", "1")
    rmsnorm_fp4quant(x, w, backend="auto")
    monkeypatch.delenv("FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS")
    with pytest.warns(ExperimentalWarning):
        rmsnorm_fp4quant(x, w, backend="triton")
    assert calls == ["cute", "cute", "cute", "triton"]
    with pytest.raises(ValueError, match="Unknown"):
        rmsnorm_fp4quant(x, w, backend="unknown")


@sm120
def test_native_packing_ties_and_signed_zero():
    import triton
    import triton.language as tl
    from flashinfer.experimental.triton_rmsnorm_fp4quant.kernel import _pack_e2m1

    @triton.jit
    def encode(X, Q, N: tl.constexpr, B: tl.constexpr):
        i = tl.arange(0, B)
        a = tl.load(X + 2 * i, 2 * i < N, other=0)
        b = tl.load(X + 2 * i + 1, 2 * i + 1 < N, other=0)
        tl.store(Q + i, _pack_e2m1(a, b), i < N // 2)

    endpoints = torch.tensor(
        [
            0.0,
            0.25,
            0.5,
            0.75,
            1.0,
            1.25,
            1.5,
            1.75,
            2.0,
            2.5,
            3.0,
            3.5,
            4.0,
            5.0,
            6.0,
            7.0,
            100.0,
        ]
    )
    lower = torch.nextafter(endpoints, torch.full_like(endpoints, -float("inf")))
    upper = torch.nextafter(endpoints, torch.full_like(endpoints, float("inf")))
    values = torch.cat(
        (
            endpoints,
            lower,
            upper,
            -endpoints,
            -lower,
            -upper,
            torch.randn(4096, generator=torch.Generator().manual_seed(812)) * 5,
        )
    )
    q = torch.empty(values.numel() // 2, device="cuda", dtype=torch.uint8)
    encode[(1,)](
        values.cuda(), q, values.numel(), triton.next_power_of_2(q.numel()), num_warps=4
    )
    codes = _codes(values)
    torch.testing.assert_close(
        q.cpu(), codes[0::2] | (codes[1::2] << 4), rtol=0, atol=0
    )


@sm120
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("block_size", [16, 32])
@pytest.mark.parametrize("shape", [(3, 128), (2, 3, 128)])
def test_existing_cute_modes_unchanged(dtype, block_size, shape):
    from flashinfer.cute_dsl.rmsnorm_fp4quant import rmsnorm_fp4quant as direct

    torch.manual_seed(73)
    x = torch.randn(shape, device="cuda", dtype=dtype)
    w = torch.ones(shape[-1], device="cuda", dtype=dtype)
    expected = direct(x, w, block_size=block_size, enable_pdl=False)
    actual = rmsnorm_fp4quant(x, w, block_size=block_size, enable_pdl=False)
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(
            a.view(torch.uint8), b.view(torch.uint8), rtol=0, atol=0
        )
