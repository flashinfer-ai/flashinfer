# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Canonical weights must remain reusable while native W4A16 computes correctly."""

import pytest
import torch

from flashinfer import mm_bf16_fp4, prepare_bf16_fp4_weights
from flashinfer.autotuner import autotune
from flashinfer.utils import get_compute_capability

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or get_compute_capability(torch.device("cuda")) not in ((12, 0), (12, 1)),
    reason="Native W4A16 requires SM120/121",
)

BACKEND = "cute-dsl-native"
FP4_VALUES = (0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6)


def make_case(m, n, k, seed=42):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    a = torch.randn((m, k), generator=generator, device="cuda", dtype=torch.bfloat16)
    b = torch.randint(
        256, (n, k // 2), generator=generator, device="cuda", dtype=torch.uint8
    )
    pn, pk = ((n + 127) // 128) * 128, ((k // 16 + 3) // 4) * 4
    linear = torch.rand((pn, pk), generator=generator, device="cuda").to(
        torch.float8_e4m3fn
    )
    # Construct physical blocks from logical scales independently of the
    # kernel's scalar-address calculation, including nonzero padding.
    sf = linear.reshape(pn // 128, 4, 32, pk // 4, 4)
    sf = sf.permute(0, 3, 2, 1, 4).contiguous().view(-1)
    lut = torch.tensor(FP4_VALUES, device="cuda", dtype=torch.float32)
    codes = torch.stack((b & 15, b >> 4), dim=-1).long().reshape(n, k)
    weight = lut[codes] * linear[:n, : k // 16].float().repeat_interleave(16, dim=1)
    return a, b, sf, weight


@pytest.mark.parametrize(
    "m,n,k",
    [
        (1, 1, 16),
        (2, 127, 64),
        (3, 129, 80),
        (4, 193, 192),
        (8, 129, 1024),
        (17, 129, 80),
        (31, 513, 192),
        (33, 127, 4112),
        (65, 129, 192),
        (129, 257, 384),
    ],
)
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
def test_native_matches_reference_without_preparation(m, n, k, out_dtype):
    """Nibble decoding, scale interleaving and N/K tails preserve the linear map."""
    a, b, sf, weight = make_case(m, n, k)
    alpha = torch.tensor([0.375], device="cuda")
    out = mm_bf16_fp4(a, b, sf, alpha, backend=BACKEND, out_dtype=out_dtype)
    ref = (a.float() @ weight.T) * alpha
    torch.testing.assert_close(out.float(), ref, atol=2e-3, rtol=8e-3)


def test_prepare_preserves_both_buffer_objects():
    """Optional preparation must not allocate or rewrite a second representation."""
    a, b, sf, _ = make_case(1, 129, 80)
    alpha = torch.tensor([0.5], device="cuda")
    original_b, original_sf = b.clone(), sf.clone()
    prepared = prepare_bf16_fp4_weights(b, sf, alpha, backend=BACKEND)
    assert all(
        left is right for left, right in zip(prepared, (b, sf, alpha), strict=True)
    )
    mm_bf16_fp4(a, *prepared, backend=BACKEND)
    torch.testing.assert_close(b, original_b, atol=0, rtol=0)
    torch.testing.assert_close(
        sf.view(torch.uint8), original_sf.view(torch.uint8), atol=0, rtol=0
    )


def test_all_codes_and_nonunit_scales_with_one_hot_inputs():
    """Select one weight at a time so cancellation cannot hide packing mistakes."""
    a, b, sf, weight = make_case(16, 129, 64)
    a.zero_()
    a[torch.arange(16, device="cuda"), torch.arange(16, device="cuda")] = 1
    b[:, :8] = torch.tensor(
        [16 * (i + 1) + i for i in range(0, 16, 2)], device="cuda", dtype=torch.uint8
    )
    # Replace the corresponding reference codes without using any decode helper.
    first_scale = weight.new_empty((129,))
    sf_logical = sf.reshape(2, 1, 32, 4, 4).permute(0, 3, 2, 1, 4).reshape(256, 4)
    first_scale.copy_(sf_logical[:129, 0].float())
    expected = (
        torch.tensor(FP4_VALUES, device="cuda", dtype=torch.float32)[:, None]
        * first_scale
    )
    out = mm_bf16_fp4(a, b, sf, backend=BACKEND)
    torch.testing.assert_close(out, expected.to(out.dtype), atol=0, rtol=0)


def test_all_finite_e4m3_scales_preserve_fp4_values():
    """BF16 conversion must preserve zero, subnormal, negative and maximum scales."""
    a = torch.eye(16, 64, device="cuda", dtype=torch.bfloat16)
    codes = torch.arange(256, device="cuda")
    scale_bits = codes[(codes & 127) != 127].to(torch.uint8)
    scales = scale_bits.view(torch.float8_e4m3fn)
    n = scales.numel()
    linear = torch.ones((256, 4), device="cuda", dtype=torch.float32).to(scales.dtype)
    linear[:n, 0] = scales
    sf = linear.reshape(2, 4, 32, 1, 4).permute(0, 3, 2, 1, 4).contiguous().view(-1)
    b = torch.zeros((n, 32), device="cuda", dtype=torch.uint8)
    b[:, :8] = torch.tensor(
        [16 * (i + 1) + i for i in range(0, 16, 2)], device="cuda", dtype=torch.uint8
    )
    expected = torch.tensor(FP4_VALUES, device="cuda")[:, None] * scales.float()
    out = mm_bf16_fp4(a, b, sf, backend=BACKEND)
    torch.testing.assert_close(out, expected.to(out.dtype), atol=0, rtol=0)


@pytest.mark.parametrize("enable_pdl", [True, False])
@pytest.mark.parametrize(
    "m,n,k", [(3, 513, 512), (1, 129, 2048), (33, 129, 80), (65, 129, 192)]
)
def test_graph_replay_reads_live_alpha_and_inputs_on_current_stream(
    enable_pdl, m, n, k
):
    """Captured execution must retain device pointers, including mutable alpha."""
    a, b, sf, weight = make_case(m, n, k)
    alpha = torch.tensor([0.5], device="cuda")
    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        mm_bf16_fp4(a, b, sf, alpha, backend=BACKEND, out=out, enable_pdl=enable_pdl)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        result = mm_bf16_fp4(
            a, b, sf, alpha, backend=BACKEND, out=out, enable_pdl=enable_pdl
        )
    assert result is out
    with torch.cuda.stream(stream):
        a.mul_(0.5)
        alpha.fill_(-1.75)
        b.bitwise_xor_(0x88)
        graph.replay()
    stream.synchronize()
    ref = (a.float() @ -weight.T) * alpha
    torch.testing.assert_close(out.float(), ref, atol=2e-3, rtol=8e-3)
    with torch.cuda.stream(stream):
        sf.view(torch.uint8).bitwise_xor_(0x80)
        graph.replay()
    stream.synchronize()
    torch.testing.assert_close(out.float(), -ref, atol=2e-3, rtol=8e-3)


@pytest.mark.parametrize(
    "m,n,k,alpha_value,out_dtype",
    [
        (1, 1, 16, None, torch.bfloat16),
        (1, 129, 192, None, torch.bfloat16),
        (1, 129, 384, None, torch.bfloat16),
        (1, 129, 1152, None, torch.bfloat16),
        (1, 129, 2048, None, torch.bfloat16),
        (3, 129, 80, None, torch.bfloat16),
        (16, 513, 512, None, torch.bfloat16),
        (16, 513, 4112, None, torch.bfloat16),
        (1, 129, 1152, 0.375, torch.bfloat16),
        (2, 256, 1024, -0.5, torch.bfloat16),
        (4, 129, 768, 0.375, torch.bfloat16),
        (2, 256, 1024, 0.375, torch.float16),
        (17, 129, 80, None, torch.bfloat16),
        (33, 513, 192, -0.5, torch.float16),
        (65, 129, 192, None, torch.bfloat16),
        (129, 257, 384, -0.5, torch.float16),
    ],
)
def test_autotuned_tactic_matches_reference(m, n, k, alpha_value, out_dtype):
    """Every tactic must reload shared operands when its staging buffer wraps."""
    from flashinfer.gemm.kernels.native_bf16_fp4.runner import get_runner

    a, b, sf, weight = make_case(m, n, k)
    out = torch.empty((m, n), device="cuda", dtype=out_dtype)
    alpha = (
        torch.tensor([alpha_value], device="cuda") if alpha_value is not None else None
    )
    inputs = [a, b, sf, alpha, out, True]
    ref = a.float() @ weight.T
    if alpha is not None:
        ref = ref * alpha
    runner = get_runner()
    for tactic in runner.get_valid_tactics(inputs, None):
        runner(inputs, tactic=tactic)
        torch.testing.assert_close(out.float(), ref, atol=2e-3, rtol=8e-3)
    with autotune(True):
        result = mm_bf16_fp4(a, b, sf, alpha, backend=BACKEND, out_dtype=out_dtype)
    torch.testing.assert_close(result.float(), ref, atol=2e-3, rtol=8e-3)


def test_invalid_weight_layout_is_rejected():
    """A prepared int32 tile bank must never be interpreted as canonical bytes."""
    a, b, sf, _ = make_case(1, 128, 128)
    with pytest.raises(ValueError, match="contiguous uint8 weights"):
        mm_bf16_fp4(a, b.view(torch.int32), sf, backend=BACKEND)


def test_output_alias_is_rejected():
    a, b, sf, _ = make_case(1, 128, 128)
    with pytest.raises(ValueError, match="must not overlap"):
        mm_bf16_fp4(a, b, sf, backend=BACKEND, out=a)


@pytest.mark.parametrize("unaligned", ["activation", "weight", "scale"])
@pytest.mark.parametrize("m", [3, 65])
def test_unaligned_canonical_buffers_use_valid_tactics(unaligned, m):
    """A valid narrow-aligned buffer must not reach a 16-byte asynchronous load."""
    from flashinfer.gemm.kernels.native_bf16_fp4.runner import get_runner

    a, b, sf, weight = make_case(m, 129, 128)
    tensors = {"activation": a, "weight": b, "scale": sf}
    original = tensors[unaligned]
    offset = 4 if unaligned == "weight" else 1
    storage = torch.empty(
        original.numel() + offset, device="cuda", dtype=original.dtype
    )
    shifted = storage[offset:].view(original.shape)
    shifted.copy_(original)
    tensors[unaligned] = shifted
    a, b, sf = (tensors[key] for key in ("activation", "weight", "scale"))
    out = torch.empty((m, 129), device="cuda", dtype=torch.bfloat16)
    runner = get_runner()
    inputs = [a, b, sf, None, out, True]
    assert all(
        tactic[0] not in ("staged", "tiled")
        for tactic in runner.get_valid_tactics(inputs, None)
    )
    result = mm_bf16_fp4(a, b, sf, backend=BACKEND, out=out)
    torch.testing.assert_close(
        result.float(), a.float() @ weight.T, atol=2e-3, rtol=8e-3
    )


@pytest.mark.parametrize("exponent", [20, -30])
def test_staging_preserves_bf16_activation_range(exponent):
    """Native BF16 MMA must not narrow activations through FP16."""
    a, b, sf, weight = make_case(16, 129, 128)
    a.zero_()
    indices = torch.arange(16, device="cuda")
    a[indices, indices] = 2.0**exponent
    result = mm_bf16_fp4(a, b, sf, backend=BACKEND)
    expected = (a.float() @ weight.T).to(torch.bfloat16)
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result, expected, atol=0, rtol=0)


@pytest.mark.parametrize("splits", [1, 2])
def test_tiled_traversal_preserves_outputs_with_m_and_n_tails(splits):
    """Changing independent block traversal must preserve every output bit."""
    from flashinfer.gemm.kernels.native_bf16_fp4.runner import _compile

    m, n, k = 193, 257, 384
    a, b, sf, weight = make_case(m, n, k)
    alpha = torch.tensor([-0.375], device="cuda")
    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    partial = (
        torch.empty((splits, m, n), device="cuda", dtype=torch.float32)
        if splits > 1
        else None
    )
    outputs = []
    for raster_m in (False, True):
        tactic = ("tiled", 128, 8, splits, 2, 64, raster_m)
        compiled = _compile(m, n, k, out.dtype, True, True, tactic)
        compiled(a.view(torch.int32), b.view(torch.int32), sf, alpha, out, partial)
        outputs.append(out.clone())
    assert torch.equal(outputs[0].view(torch.int16), outputs[1].view(torch.int16))
    torch.testing.assert_close(
        out.float(), (a.float() @ weight.T) * alpha, atol=2e-3, rtol=8e-3
    )
