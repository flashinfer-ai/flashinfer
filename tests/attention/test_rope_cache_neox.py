"""
Copyright (c) 2024 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import pytest
import torch
import flashinfer


def cos_sin_cache(max_pos, rotary_dim, theta=10000.0):
    inv = 1.0 / (
        theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
    )
    freqs = torch.arange(max_pos, dtype=torch.float32)[:, None] * inv[None, :]
    return torch.cat(
        (freqs.cos(), freqs.sin()), dim=-1
    ).cuda()  # [max_pos, rotary_dim]: cos half, sin half


def reference(x, pos, cache, head_size, is_neox):
    rd = cache.shape[-1]
    cos, sin = cache[pos].chunk(2, dim=-1)  # [nnz, rd / 2]
    cos, sin = cos[:, None, :], sin[:, None, :]
    x = x.view(x.shape[0], -1, head_size).float()
    r = x[..., :rd]
    if is_neox:
        a, b = r[..., : rd // 2], r[..., rd // 2 :]
        out = torch.cat((a * cos - b * sin, b * cos + a * sin), dim=-1)
    else:
        a, b = r[..., 0::2], r[..., 1::2]
        out = torch.stack((a * cos - b * sin, b * cos + a * sin), dim=-1).flatten(-2)
    return torch.cat((out, x[..., rd:]), dim=-1).flatten(1)


@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("is_neox", [True, False])
@pytest.mark.parametrize(
    "hs,rd",
    [(d, d) for d in (2, 4, 8, 16, 24, 32, 48, 64, 80, 96, 128, 256, 512)]
    + [(128, 48), (128, 80), (128, 96), (80, 48), (96, 80)],
)
@pytest.mark.parametrize("nnz", [1, 21, 65])
def test_rope_cache_unaligned_neox(hs, rd, dtype, is_neox, nnz, inplace):
    if not is_neox and hs % 16 != 0:
        pytest.skip("Small interleaved vector tails are outside the NeoX fix")
    generator = torch.Generator(device="cuda").manual_seed(7)
    q = torch.randn(nnz, 4 * hs, device="cuda", dtype=dtype, generator=generator)
    k = torch.randn(nnz, 2 * hs, device="cuda", dtype=dtype, generator=generator)
    pos = torch.arange(nnz, device="cuda", dtype=torch.int64)
    cache = cos_sin_cache(nnz, rd)
    expected = [reference(x, pos, cache, hs, is_neox) for x in (q, k)]
    previous = None
    for _ in range(3):
        actual = [q.clone(), k.clone()]
        if inplace:
            flashinfer.rope.apply_rope_with_cos_sin_cache_inplace(
                pos, *actual, hs, cache, is_neox=is_neox
            )
        else:
            actual = flashinfer.rope.apply_rope_with_cos_sin_cache(
                pos, *actual, hs, cache, is_neox=is_neox
            )
        for output, ref in zip(actual, expected, strict=True):
            torch.testing.assert_close(
                output.float(), ref, atol=4 * torch.finfo(dtype).eps, rtol=0
            )
        if previous is not None:
            for output, prev in zip(actual, previous, strict=True):
                assert torch.equal(output, prev)
        previous = actual


@pytest.mark.parametrize("rotary_dim", [2, 4, 8, 16, 24, 32, 48, 80, 96])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("quant_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_rope_quantize_neox_vector_widths(rotary_dim, dtype, quant_dtype):
    torch.manual_seed(7)
    nnz, head_size = 21, rotary_dim + 16
    q = torch.randn(nnz, 4, head_size, device="cuda", dtype=dtype)
    k = torch.randn(nnz, 2, head_size, device="cuda", dtype=dtype)
    pos = torch.arange(nnz, device="cuda", dtype=torch.int64)
    cache = cos_sin_cache(nnz, rotary_dim)
    outputs = [torch.empty_like(x, dtype=quant_dtype) for x in (q, k)]
    flashinfer.rope.rope_quantize_fp8(
        q[..., :rotary_dim],
        k[..., :rotary_dim],
        q[..., rotary_dim:],
        k[..., rotary_dim:],
        cache,
        pos,
        is_neox=True,
        q_rope_out=outputs[0][..., :rotary_dim],
        k_rope_out=outputs[1][..., :rotary_dim],
        q_nope_out=outputs[0][..., rotary_dim:],
        k_nope_out=outputs[1][..., rotary_dim:],
        quant_scale_q=1.25,
        quant_scale_kv=0.75,
        enable_pdl=False,
        backend="cuda",
    )
    for x, output, scale in zip((q, k), outputs, (1.25, 0.75), strict=True):
        expected = (
            reference(x.flatten(1), pos, cache, head_size, True) * scale
        ).view_as(x)
        torch.testing.assert_close(output.float(), expected, atol=0.125, rtol=0.125)
        torch.testing.assert_close(
            output[..., rotary_dim:].float(),
            (x[..., rotary_dim:].float() * scale).to(quant_dtype).float(),
            atol=0,
            rtol=0,
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_rope_neox_96_inplace_warp_boundary(dtype):
    # Six 16-element vectors per head previously let heads straddle warp boundaries.
    # More rows and repetitions expose the intermittent in-place partner overwrite.
    torch.manual_seed(7)
    nnz, hs = 257, 96
    q = torch.randn(nnz, 4 * hs, device="cuda", dtype=dtype)
    k = torch.randn(nnz, 2 * hs, device="cuda", dtype=dtype)
    pos = torch.arange(nnz, device="cuda")
    cache = cos_sin_cache(nnz, hs)
    expected = [reference(x, pos, cache, hs, True) for x in (q, k)]
    for _ in range(100):
        actual = [q.clone(), k.clone()]
        flashinfer.rope.apply_rope_with_cos_sin_cache_inplace(
            pos, *actual, hs, cache, is_neox=True
        )
        for output, ref in zip(actual, expected, strict=True):
            torch.testing.assert_close(
                output.float(), ref, atol=4 * torch.finfo(dtype).eps, rtol=0
            )
