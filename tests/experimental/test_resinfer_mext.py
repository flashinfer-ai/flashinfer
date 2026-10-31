# SPDX-FileCopyrightText: Copyright (c) 2026 Rong Shuo
# SPDX-License-Identifier: Apache-2.0
"""ResInfer MEXT: quantization, FP32 fold, and changed-input graph replay."""

import pytest
import torch

from flashinfer.resinfer import mm_nvfp4_residue, nvfp4_quantize_residue


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
    sf = scales.view(torch.float8_e4m3fn).float()[offsets]
    return table[values] * sf.repeat_interleave(16, dim=1)


@pytest.fixture(autouse=True)
def sm100():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("ResInfer MEXT requires SM100")


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
@pytest.mark.parametrize("dual", [False, True])
def test_residue_and_fold(dtype, m, k, n, dual):
    torch.manual_seed(42)
    x = torch.randn(m, k, device="cuda", dtype=dtype)
    scales = torch.tensor([64.0, 512.0] if dual else [64.0], device="cuda")
    p, sf = nvfp4_quantize_residue(x, scales)
    decoded = _decode(p, sf)
    base = decoded[0::2] / scales[0]
    residue = decoded[1::2] / scales[-1]
    base_error = (x.float() - base).square().mean()
    assert (x.float() - base - residue).square().mean() < base_error * 0.1
    # Arbitrary packed weights exercise both signs and all representable values.
    w = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda")
    ws = (
        torch.ones(((n + 127) // 128) * ((k + 63) // 64) * 512, device="cuda")
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    alpha = scales.reciprocal()
    y = mm_nvfp4_residue(p, w, sf, ws, alpha, out_dtype=dtype)
    ref = (base + residue) @ _decode(w, ws).T
    torch.testing.assert_close(y.float(), ref, atol=0.08, rtol=0.02)


def test_graph_changed_inputs():
    x = torch.randn(7, 128, device="cuda", dtype=torch.bfloat16)
    s = torch.tensor([64.0, 512.0], device="cuda")
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
