from __future__ import annotations

import torch

from b12x._lib.fp6 import (
    FLOAT6_E2M3_MAX,
    FLOAT6_E3M2_MAX,
    SF_VEC_SIZE_FP6,
    _encode_fp6_nearest,
    dequant_mxfp6_torch,
    fp6_quantize_values_torch,
    pack_fp6_codes_tensor,
    quantize_grouped_mxfp6_torch,
    unpack_fp6_packed_tensor,
)


def test_fp6_quantize_clamps_to_format_max() -> None:
    x = torch.tensor(
        [FLOAT6_E3M2_MAX * 2.0, -FLOAT6_E2M3_MAX * 2.0],
        dtype=torch.float32,
    )
    q3 = fp6_quantize_values_torch(x[:1], fmt="e3m2")
    q2 = fp6_quantize_values_torch(x[1:], fmt="e2m3")
    assert float(q3.abs().max()) <= FLOAT6_E3M2_MAX + 1e-3
    assert float(q2.abs().max()) <= FLOAT6_E2M3_MAX + 1e-3


def test_fp6_pack_dequant_roundtrip_single_block() -> None:
    torch.manual_seed(0)
    k = SF_VEC_SIZE_FP6
    values = torch.randn(k, dtype=torch.float32)
    q = fp6_quantize_values_torch(values, fmt="e3m2")
    codes = torch.zeros(k, dtype=torch.uint8)
    for i in range(k):
        codes[i] = _encode_fp6_nearest(float(q[i].item()), "e3m2")
    packed = pack_fp6_codes_tensor(codes.unsqueeze(0)).squeeze(0)
    ue = torch.tensor([127], dtype=torch.uint8)
    scales = ue.view(1, 1).expand(1, 1)
    out = dequant_mxfp6_torch(
        packed.unsqueeze(0),
        scales,
        num_fp6=k,
        fmt="e3m2",
        global_scale=torch.tensor(1.0),
        scales_swizzled=False,
    ).squeeze(0)
    torch.testing.assert_close(out, q, rtol=0.0, atol=1e-2)


def test_quantize_grouped_mxfp6_torch_shapes() -> None:
    groups, rows, cols = 2, 64, 128
    x = torch.randn(groups, rows, cols, dtype=torch.bfloat16)
    row_counts = torch.tensor([rows, rows // 2], dtype=torch.int32)
    gs = torch.ones(groups, dtype=torch.float32)
    packed, scale_view = quantize_grouped_mxfp6_torch(x, row_counts, gs, fmt="e3m2")
    assert packed.shape == (rows, cols * 3 // 4, groups)
    assert scale_view.ndim >= 4
    roundtrip = unpack_fp6_packed_tensor(packed[:, :, 0], num_fp6=cols)
    assert roundtrip.shape == (rows, cols)


def test_quantize_grouped_mxfp6_e2m3_vs_e3m2_differ_on_large_values() -> None:
    x = torch.full((1, 32, 128), 20.0, dtype=torch.bfloat16)
    row_counts = torch.tensor([32], dtype=torch.int32)
    gs = torch.tensor([1.0], dtype=torch.float32)
    _, sf_e3 = quantize_grouped_mxfp6_torch(x, row_counts, gs, fmt="e3m2")
    _, sf_e2 = quantize_grouped_mxfp6_torch(x, row_counts, gs, fmt="e2m3")
    assert not torch.equal(sf_e3, sf_e2)
