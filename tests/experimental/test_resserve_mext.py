# SPDX-FileCopyrightText: Copyright (c) 2026 Rong Shuo
# SPDX-License-Identifier: Apache-2.0
"""ResServe MEXT: quantization, FP32 fold, and changed-input graph replay."""

import pytest
import torch

from flashinfer.resserve import mm_nvfp4_residue, nvfp4_quantize_residue


def _decode(packed, scales):
    rows, kp = packed.shape
    k = 2 * kp
    table = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device=packed.device,
    )
    values = torch.stack((packed & 15, packed >> 4), dim=-1).long().reshape(rows, k)
    r = torch.arange(rows, device=packed.device)[:, None]
    c = torch.arange(k // 16, device=packed.device)[None, :]
    offsets = (
        (r // 128) * ((k + 63) // 64) * 512
        + (c // 4) * 512
        + (r % 32) * 16
        + ((r % 128) // 32) * 4
        + c % 4
    )
    sf = scales.reshape(-1).view(torch.float8_e4m3fn).float()[offsets]
    return table[values] * sf.repeat_interleave(16, dim=1)


@pytest.fixture(autouse=True)
def sm100():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("ResServe MEXT requires SM100")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "m,k,n",
    [
        (1, 64, 128),
        (7, 128, 128),
        (65, 256, 128),
        (129, 1024, 128),
        (7, 128, 8),
        (7, 128, 24),
        (65, 256, 136),
    ],
)
def test_residue_and_fold(dtype, m, k, n):
    torch.manual_seed(42)
    x = torch.randn(m, k, device="cuda", dtype=dtype)
    scales = torch.tensor([64.0], device="cuda")
    p, sf = nvfp4_quantize_residue(x, scales)
    decoded = _decode(p, sf)
    base = decoded[0::2] / scales[0]
    residue = decoded[1::2] / scales[0]
    base_error = (x.float() - base).square().mean()
    assert (x.float() - base - residue).square().mean() < base_error * 0.1
    # Arbitrary packed weights exercise both signs and all representable values.
    w = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda")
    ws = (
        (
            torch.rand(((n + 127) // 128) * ((k + 63) // 64) * 512, device="cuda")
            * 0.75
            + 0.25
        )
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    alpha = scales.reciprocal()
    y = mm_nvfp4_residue(p, w, sf, ws, alpha, out_dtype=dtype)
    ref = (base + residue) @ _decode(w, ws).T
    torch.testing.assert_close(y.float(), ref, atol=0.08, rtol=0.02)


def test_graph_changed_inputs():
    x = torch.randn(7, 128, device="cuda", dtype=torch.bfloat16)
    s = torch.tensor([64.0], device="cuda")
    for _ in range(3):
        nvfp4_quantize_residue(x, s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        p, sf = nvfp4_quantize_residue(x, s)
    for factor in [0.0, 1.0, -2.0]:
        x.copy_(torch.randn_like(x) * factor)
        graph.replay()
        ep, esf = nvfp4_quantize_residue(x, s)
        torch.testing.assert_close(p, ep, atol=0, rtol=0)
        torch.testing.assert_close(sf, esf, atol=0, rtol=0)


def test_reject_wrong_scale_dtype():
    x = torch.zeros(1, 64, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="dtype"):
        nvfp4_quantize_residue(x, torch.ones(1, device="cuda", dtype=torch.float16))


@pytest.mark.parametrize("n", [136, 264])
def test_strided_base_weight_graph(n):
    torch.manual_seed(17)
    x = torch.randn(7, 256, device="cuda", dtype=torch.bfloat16)
    scale = torch.tensor([64.0], device="cuda")
    a, sa = nvfp4_quantize_residue(x, scale)
    storage = torch.randint(0, 256, (n, 192), device="cuda", dtype=torch.uint8)
    weight = storage[:, :128]
    ws = (
        (torch.rand(((n + 127) // 128) * 4 * 512, device="cuda") * 0.75 + 0.25)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    alpha = scale.reciprocal()

    def run():
        return mm_nvfp4_residue(a, weight, sa, ws, alpha)

    for _ in range(3):
        run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = run()
    for _ in range(2):
        storage.random_(0, 256)
        graph.replay()
        contiguous = mm_nvfp4_residue(a, weight.contiguous(), sa, ws, alpha)
        decoded = _decode(a, sa)
        ref = (decoded[0::2] / scale[0] + decoded[1::2] / scale[0]) @ _decode(
            weight, ws
        ).T
        torch.testing.assert_close(out, contiguous, atol=0, rtol=0)
        torch.testing.assert_close(out.float(), ref, atol=0.08, rtol=0.02)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("k", [16, 48, 80])
def test_partial_scale_tile(dtype, k):
    """Cover partial K scale groups and both sides of the row-tile boundary."""
    torch.manual_seed(3)
    m = 65
    x = torch.randn(m, k, device="cuda", dtype=dtype)
    scale = torch.tensor([64.0], device="cuda")
    packed, sf = nvfp4_quantize_residue(x, scale)
    decoded = _decode(packed, sf) / scale
    base, residue = decoded[0::2], decoded[1::2]
    assert (x.float() - base - residue).square().mean() < (
        x.float() - base
    ).square().mean() * 0.1
    rows = torch.arange(2 * m, device="cuda")[:, None]
    cols = torch.arange(k // 16, device="cuda")[None, :]
    offsets = (
        (rows // 128) * ((k + 63) // 64) * 512
        + (cols // 4) * 512
        + (rows % 32) * 16
        + ((rows % 128) // 32) * 4
        + cols % 4
    )
    padding = torch.ones(sf.numel(), device="cuda", dtype=torch.bool)
    padding[offsets.flatten()] = False
    assert (sf.reshape(-1)[padding] == 0).all()


def test_reject_independent_residue_scale():
    x = torch.zeros(1, 64, device="cuda", dtype=torch.bfloat16)
    scale = torch.ones(1, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="global_scale must contain one element"):
        nvfp4_quantize_residue(x, scale.repeat(2))
    a, sa = nvfp4_quantize_residue(x, scale)
    w = torch.zeros(128, 32, device="cuda", dtype=torch.uint8)
    ws = torch.zeros(512, device="cuda", dtype=torch.uint8)
    with pytest.raises(ValueError, match="alpha must have one element"):
        mm_nvfp4_residue(a, w, sa, ws, scale.repeat(2))


def _launch_args(m2, n=128, k_packed=32):
    activation = torch.zeros(m2, k_packed, device="cuda", dtype=torch.uint8)
    weight = torch.zeros(n, k_packed, device="cuda", dtype=torch.uint8)
    scale = torch.zeros(512, device="cuda", dtype=torch.uint8)
    alpha = torch.ones(1, device="cuda")
    return weight, activation, scale, scale, alpha


@pytest.mark.parametrize("fp4_format", ["nvfp4", "mxfp4"])
@pytest.mark.parametrize("store_mode", ["tma", "tma_mext", "direct_mext"])
def test_reject_narrow_tile_spanning_rows(fp4_format, store_mode):
    """80 activation rows span a second width-32 tile, which has no SFB addressing."""
    from flashinfer.experimental.resserve_mext.host import mext_fold_gemm

    with pytest.raises(ValueError, match="narrow tma tiles"):
        mext_fold_gemm(
            *_launch_args(80),
            mma_tiler_mn=(128, 32),
            store_mode=store_mode,
            fp4_format=fp4_format,
        )


def test_narrow_tma_tile_matches_reference():
    """Width 8 is legal for one token row and matches the decoded reference."""
    from flashinfer.experimental.resserve_mext.host import mext_fold_gemm

    torch.manual_seed(42)
    m, k, n = 1, 64, 64
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    scales = torch.tensor([64.0], device="cuda")
    packed, sf = nvfp4_quantize_residue(x, scales)
    decoded = _decode(packed, sf) / scales
    base, residue = decoded[0::2], decoded[1::2]
    weight = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda")
    weight_scale = (
        (
            torch.rand(((n + 127) // 128) * ((k + 63) // 64) * 512, device="cuda")
            * 0.75
            + 0.25
        )
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    result = mext_fold_gemm(
        weight,
        packed,
        weight_scale,
        sf,
        scales.reciprocal(),
        mma_tiler_mn=(128, 8),
        cluster_shape_mn=(1, 1),
    )
    reference = (base + residue) @ _decode(weight, weight_scale).T
    torch.testing.assert_close(result.float(), reference, atol=0.08, rtol=0.02)


@pytest.mark.parametrize("store_mode", ["tma_mext", "direct_mext"])
def test_reject_mfold_narrow_tile(store_mode):
    from flashinfer.experimental.resserve_mext.host import mext_fold_gemm

    # 16 physical rows fit in one width-32 tile, but M-fold keeps the addressed widths.
    with pytest.raises(ValueError, match="narrow tma tiles"):
        mext_fold_gemm(*_launch_args(16), mma_tiler_mn=(128, 32), store_mode=store_mode)


@pytest.mark.parametrize("m,n", [(64, 32), (64, 160), (72, 160)])
def test_direct_store_tails(m, n, monkeypatch):
    from flashinfer.experimental.resserve_mext import host

    k = 256
    a = torch.full((2 * m, k // 2), 0x22, device="cuda", dtype=torch.uint8)
    w = torch.full((n, k // 2), 0x22, device="cuda", dtype=torch.uint8)
    sa = torch.full(
        (((2 * m + 127) // 128) * 4 * 512,), 56, device="cuda", dtype=torch.uint8
    )
    sb = torch.full(
        (((n + 127) // 128) * 4 * 512,), 56, device="cuda", dtype=torch.uint8
    )
    rows = torch.arange(n, device="cuda")[:, None]
    cols = torch.arange(k // 16, device="cuda")[None, :]
    offsets = (
        (rows // 128) * 4 * 512
        + (cols // 4) * 512
        + (rows % 32) * 16
        + ((rows % 128) // 32) * 4
        + cols % 4
    )
    # Vary scales within each tile and across the output-tile boundary.
    sb[offsets] = (56 + 8 * (rows % 2)).expand_as(offsets).to(torch.uint8)
    alpha = torch.ones(1, device="cuda")
    storage = torch.full((m * n + 4096,), -123, device="cuda", dtype=torch.bfloat16)
    out = storage[2048:-2048].view(m, n)
    original_empty = torch.empty

    def guarded_empty(*shape, **kwargs):
        if shape == (m, n):
            return out
        return original_empty(*shape, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(host.torch, "empty", guarded_empty)
        for packed, value in [(0x22, 1.0), (0x11, 0.5)]:
            a.fill_(packed)
            result = host.mext_fold_gemm(w, a, sb, sa, alpha, store_mode="direct_mext")
            torch.cuda.synchronize()
            expected = (
                2 * k * value * (1 + torch.arange(n, device="cuda") % 2)
            ).expand(m, n)
            torch.testing.assert_close(result.float(), expected.float(), atol=0, rtol=0)
            assert (storage[:2048] == -123).all()
            assert (storage[-2048:] == -123).all()


@pytest.mark.parametrize("store_mode", ["tma_mext", "direct_mext"])
def test_reject_unpadded_tile8(store_mode):
    from flashinfer.experimental.resserve_mext.host import mext_fold_gemm

    a = torch.zeros(130, 128, device="cuda", dtype=torch.uint8)
    w = torch.zeros(128, 128, device="cuda", dtype=torch.uint8)
    scale = torch.empty(0, device="cuda", dtype=torch.uint8)
    alpha = torch.ones(1, device="cuda")
    with pytest.raises(ValueError, match="padded to a multiple of 8"):
        mext_fold_gemm(w, a, scale, scale, alpha, store_mode=store_mode)
