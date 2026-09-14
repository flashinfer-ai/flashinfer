# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.

import gc

import pytest
import torch

from flashinfer.deepseek_v41 import deepseek_v41_woa, deepseek_v41_woa_plan


@pytest.fixture(scope="module", autouse=True)
def sm100():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("Frost WOA requires SM100")


def reference(x, weight, scales):
    # Decode semantically on CPU, independently of the kernel's bit shifts.
    scale = torch.exp2(scales.cpu().view(torch.uint8).double() - 127)
    decoded = (
        (weight.cpu().double().reshape(256, 32, 128, 32) * scale[:, None, :, None])
        .bfloat16()
        .reshape(8, 1024, 4096)
        .double()
    )
    return (
        torch.bmm(x.cpu().double().transpose(0, 1), decoded.transpose(1, 2))
        .transpose(0, 1)
        .contiguous()
    )


def check_close(actual, expected):
    value = actual.cpu().double()
    delta = value - expected
    assert torch.isfinite(value).all()
    # The same source-BF16/FP64 gates used for the model-shaped component.
    assert delta.norm() <= 0.003 * expected.norm()
    assert delta.abs().max() <= 0.006 * expected.abs().max()


@pytest.fixture(scope="module")
def tensors():
    generator = torch.Generator().manual_seed(419222)
    weight = (
        torch.randn((8192, 4096), generator=generator).to(torch.float8_e4m3fn).cuda()
    )
    scales = torch.randint(
        122, 131, (256, 128), generator=generator, dtype=torch.uint8
    ).cuda()
    x = torch.randn((1, 8, 4096), generator=generator).bfloat16().cuda()
    return x, weight, scales


@pytest.mark.parametrize("scale_dtype", [torch.uint8, torch.float8_e8m0fnu])
def test_source_arithmetic_and_graph_replay(tensors, scale_dtype):
    original_x, weight, codes = tensors
    x = original_x.clone()
    scales = codes.view(scale_dtype)
    weight_saved, scale_saved = weight.clone(), codes.clone()
    plan = deepseek_v41_woa_plan(weight, scales, backend="frost")
    out = torch.full((1, 8, 1024), torch.nan, device=x.device, dtype=x.dtype)
    assert deepseek_v41_woa(x, plan, out=out) is out
    check_close(out, reference(x, weight, scales))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        deepseek_v41_woa(x, plan, out=out)
    for new_x in (
        original_x.neg(),
        original_x.roll(1, -1),
        torch.zeros_like(x),
        original_x,
    ):
        x.copy_(new_x)
        out.fill_(torch.nan)
        graph.replay()
        check_close(out, reference(x, weight, scales))
        torch.testing.assert_close(x, new_x, rtol=0, atol=0)
    torch.testing.assert_close(
        weight.view(torch.uint8), weight_saved.view(torch.uint8), rtol=0, atol=0
    )
    torch.testing.assert_close(codes, scale_saved, rtol=0, atol=0)


def test_independent_outputs_plan_lifetime_and_stream(tensors):
    x, original_weight, original_scales = tensors
    weight, scales = original_weight.clone(), original_scales.clone()
    plan = deepseek_v41_woa_plan(weight, scales, backend="frost")
    del weight, scales
    gc.collect()
    first = deepseek_v41_woa(x, plan)
    saved = first.clone()
    second = deepseek_v41_woa(-x, plan)
    assert first.data_ptr() != second.data_ptr()
    torch.testing.assert_close(first, saved, rtol=0, atol=0)
    check_close(second, reference(-x, original_weight, original_scales))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        third = deepseek_v41_woa(x, plan)
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(third, saved, rtol=0, atol=0)


@pytest.mark.parametrize(
    "code,w,xvalue",
    [
        (1, 0.5, 1.0),
        (1, 2.0**-8, 1.0),
        (1, 3 * 2.0**-9, 2.0**120),
        (127, 1.0, 2.0**-133),
        (254, 1.0, 2.0**-133),
        (254, 2.0**-9, 1.0),
        (1, 0.5, 2.0**120),
        (127, 448.0, -(2.0**20)),
    ],
)
def test_finite_exponent_and_bf16_subnormal_boundaries(code, w, xvalue):
    # Single nonzero product per output makes the rounding order observable.
    host_weight = torch.zeros((8192, 4096), dtype=torch.bfloat16)
    host_weight[:, 0] = w
    weight = host_weight.to(torch.float8_e4m3fn).cuda()
    scales = torch.full((256, 128), code, device="cuda", dtype=torch.uint8)
    x = torch.zeros((1, 8, 4096), device="cuda", dtype=torch.bfloat16)
    x[..., 0] = xvalue
    plan = deepseek_v41_woa_plan(weight, scales, backend="frost")
    expected = reference(x, weight, scales).bfloat16()
    actual = deepseek_v41_woa(x, plan).cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("code", [0, 255])
def test_reject_invalid_scale_code(tensors, code):
    _, weight, original = tensors
    scales = original.clone()
    scales[-1, -1] = code
    with pytest.raises(ValueError, match="codes 1..254"):
        deepseek_v41_woa_plan(weight, scales, backend="frost")


@pytest.mark.parametrize("backend", ["auto", "cutedsl", "triton", "cudnn"])
def test_explicit_frost_backend(tensors, backend):
    _, weight, scales = tensors
    with pytest.raises(ValueError, match="backend='frost'"):
        deepseek_v41_woa_plan(weight, scales, backend=backend)


@pytest.mark.parametrize("storage", ["input", "weight", "scales"])
def test_reject_output_alias(tensors, storage):
    x, weight, scales = tensors
    plan = deepseek_v41_woa_plan(weight, scales, backend="frost")
    tensor = {"input": x, "weight": weight, "scales": scales}[storage]
    out = tensor.view(torch.bfloat16).reshape(-1)[:8192].view(1, 8, 1024)
    with pytest.raises(ValueError, match="overlap"):
        deepseek_v41_woa(x, plan, out=out)


def test_reject_invalid_tensor_contract(tensors):
    x, weight, scales = tensors
    for invalid in (
        weight[:4096],
        weight.bfloat16(),
        weight.cpu(),
        weight.t().contiguous().t(),
    ):
        with pytest.raises(ValueError, match="weight"):
            deepseek_v41_woa_plan(invalid, scales, backend="frost")
    for invalid in (
        scales[:128],
        scales.float(),
        scales.cpu(),
        scales.t().contiguous().t(),
    ):
        with pytest.raises(ValueError, match="scales"):
            deepseek_v41_woa_plan(weight, invalid, backend="frost")
    plan = deepseek_v41_woa_plan(weight, scales, backend="frost")
    for invalid in (
        x[..., :2048],
        x.float(),
        x.cpu(),
        x.transpose(1, 2).contiguous().transpose(1, 2),
    ):
        with pytest.raises(ValueError, match="input"):
            deepseek_v41_woa(invalid, plan)
    for out in (
        torch.empty((1, 8, 1024), device="cuda", dtype=torch.float32),
        torch.empty((1, 8, 1024), dtype=torch.bfloat16),
        torch.empty((1, 8, 2048), device="cuda", dtype=torch.bfloat16),
        torch.empty((1, 8, 2048), device="cuda", dtype=torch.bfloat16)[..., ::2],
    ):
        with pytest.raises(ValueError, match="output"):
            deepseek_v41_woa(x, plan, out=out)
    with pytest.raises(TypeError, match="plan"):
        deepseek_v41_woa(x, object())


def test_prepare_rejected_during_capture(tensors):
    x, weight, scales = tensors
    plan = deepseek_v41_woa_plan(weight, scales, backend="frost")
    out = deepseek_v41_woa(x, plan)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        with pytest.raises(RuntimeError, match="outside CUDA Graph"):
            deepseek_v41_woa_plan(weight, scales, backend="frost")
        deepseek_v41_woa(x, plan, out=out)
    out.fill_(torch.nan)
    graph.replay()
    check_close(out, reference(x, weight, scales))
