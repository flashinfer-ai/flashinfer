# SPDX-FileCopyrightText: Copyright (c) 2026 Rong Shuo
# SPDX-License-Identifier: Apache-2.0
"""ResInfer KEXT: channel order, scale padding, pack modes and upstream GEMM."""

import pytest
import torch

from flashinfer import mm_fp4
from flashinfer.quantization import nvfp4_quantize
from flashinfer.resinfer import nvfp4_quantize_kext, nvfp4_quantize_residue


@pytest.fixture(autouse=True)
def sm100():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("ResInfer KEXT requires SM100")


def _masks(k, count):
    selected = torch.rand(k // 8, 8, device="cuda").argsort(dim=1)[:, :count]
    selected = selected.sort(dim=1).values
    masks = (1 << selected).sum(dim=1).to(torch.uint8)
    indices = (selected + torch.arange(k // 8, device="cuda")[:, None] * 8).flatten()
    return masks, indices


def _decode(packed, scales):
    rows, kp = packed.shape
    k = 2 * kp
    table = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device=packed.device,
    )
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).long().reshape(rows, k)
    r = torch.arange(rows, device=packed.device)[:, None]
    c = torch.arange(k // 16, device=packed.device)[None, :]
    offsets = (
        (r // 128) * ((k + 63) // 64) * 512
        + (c // 4) * 512
        + (r % 32) * 16
        + ((r % 128) // 32) * 4
        + c % 4
    )
    sf = scales.flatten().view(torch.float8_e4m3fn).float()[offsets]
    return table[codes] * sf.repeat_interleave(16, dim=1), sf, offsets


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("count", [1, 2, 4])
@pytest.mark.parametrize(
    "m,k", [(1, 256), (7, 768), (129, 1536), (1024, 512), (4, 8192)]
)
def test_selected_residue(dtype, count, m, k):
    torch.manual_seed(42)
    x = torch.randn(m, k, device="cuda", dtype=dtype)
    masks, selected = _masks(k, count)
    scale = torch.tensor([64.0], device="cuda")
    pair, pair_sf = nvfp4_quantize_residue(x, scale)
    mext_base = _decode(pair, pair_sf)[0][0::2] / scale
    outputs = []
    for mode in (8, 16, 0):
        p, sf = nvfp4_quantize_kext(x, scale, masks, count, elts_per_thread=mode)
        decoded, block_sf, offsets = _decode(p, sf)
        decoded /= scale
        base, residue = decoded[:, :k], decoded[:, k:]
        torch.testing.assert_close(base, mext_base, atol=0, rtol=0)
        target = (x.float() - base)[:, selected]
        # E2M1's largest adjacent gap is 2; nearest rounding error is at most
        # one local scale. The small slack covers donor reciprocal approximations.
        bound = block_sf[:, k // 16 :].repeat_interleave(16, dim=1) / scale
        assert ((residue - target).abs() <= bound * 1.01 + 1e-5).all()
        assert (residue - target).square().mean() < target.square().mean() * 0.1
        padding = torch.ones(sf.numel(), device="cuda", dtype=torch.bool)
        padding[offsets.flatten()] = False
        assert (sf[padding] == 0).all()
        outputs.append(decoded)
    torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)
    torch.testing.assert_close(outputs[0], outputs[2], atol=0, rtol=0)


@pytest.mark.parametrize("mode", [0, 8, 16])
def test_graph_changed_input_masks_scale(mode):
    x = torch.randn(7, 768, device="cuda", dtype=torch.bfloat16)
    masks, _ = _masks(768, 2)
    scale = torch.tensor([64.0], device="cuda")

    def run():
        return nvfp4_quantize_kext(x, scale, masks, 2, elts_per_thread=mode)

    for _ in range(3):
        run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        p, sf = run()
    for factor in (0.0, 1.0, -2.0):
        x.copy_(torch.randn_like(x) * factor)
        masks.copy_(_masks(768, 2)[0])
        scale.fill_(32.0 if factor < 0 else 64.0)
        graph.replay()
        ep, esf = run()
        torch.testing.assert_close(p, ep, atol=0, rtol=0)
        torch.testing.assert_close(sf, esf, atol=0, rtol=0)
        if factor == 0:
            assert (_decode(p, sf)[0] == 0).all()


@pytest.mark.parametrize("count", [1, 2, 4])
def test_upstream_gemm(count):
    torch.manual_seed(71)
    m, n, k = 7, 128, 512
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    masks, selected = _masks(k, count)
    a_scale = torch.tensor([64.0], device="cuda")
    w_scale = torch.tensor([32.0], device="cuda")
    # Exactly the public ResInfer weight extension order, before quantization.
    extended_weight = torch.cat((w, w[:, selected]), dim=1)
    wp, ws = nvfp4_quantize(extended_weight, w_scale, backend="cute-dsl")
    ap, ass = nvfp4_quantize_kext(x, a_scale, masks, count)
    alpha = (a_scale * w_scale).reciprocal()
    actual = mm_fp4(ap, wp.T, ass, ws, alpha, backend="cute-dsl")
    reference = (_decode(ap, ass)[0] @ _decode(wp, ws)[0].T) * alpha
    torch.testing.assert_close(actual.float(), reference, atol=0.03, rtol=0.01)


@pytest.mark.parametrize("invalid", ["shape", "mask", "ratio", "scale", "mode"])
def test_reject_invalid_metadata(invalid):
    x = torch.zeros(1, 256, device="cuda", dtype=torch.bfloat16)
    masks = torch.ones(32, device="cuda", dtype=torch.uint8)
    scale = torch.ones(1, device="cuda")
    ratio, mode = 1, 0
    if invalid == "shape":
        x = x[:, :128].contiguous()
    elif invalid == "mask":
        masks = masks[:-1]
    elif invalid == "ratio":
        ratio = 3
    elif invalid == "scale":
        scale = scale.half()
    else:
        mode = 32
    with pytest.raises(ValueError):
        nvfp4_quantize_kext(x, scale, masks, ratio, elts_per_thread=mode)


@pytest.mark.parametrize("mode", [8, 16])
def test_missing_mask_bits_do_not_read_outside_local_residue(mode):
    # Malformed masks are outside the numerical contract. Still exercise the
    # local-array guard added during the port: __ffs(0) must not index at -1.
    x = torch.randn(1, 256, device="cuda", dtype=torch.bfloat16)
    masks = torch.zeros(32, device="cuda", dtype=torch.uint8)
    scale = torch.tensor([64.0], device="cuda")
    p, sf = nvfp4_quantize_kext(x, scale, masks, 4, elts_per_thread=mode)
    decoded = _decode(p, sf)[0]
    assert torch.isfinite(decoded).all()
    assert (decoded[:, 256:] == 0).all()
