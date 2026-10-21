"""Compressed preparation preserves the native scale conversion and row ordering."""

import numpy as np
import pytest
import torch

from b12x._lib.quant.csf_blocks import Nvfp4CompactBlockPlane, upload_csf_block_payload
from b12x._lib.quant.nvfp4_csf import Nvfp4CsfDecoder
from b12x.loader._csf_blocks import (
    concatenate_csf_blocks,
    slice_csf_blocks,
    transform_csf_blocks,
)
from ..conftest import require_b12x
from .test_csf_blocks import planes_and_native


def logical_grid(swizzled, rows, columns):
    experts, pr, pc = swizzled.shape
    return (
        swizzled.reshape(experts, pr // 128, pc // 4, 32, 4, 4)
        .permute(0, 1, 4, 3, 2, 5)
        .reshape(experts, pr, pc)[:, :rows, :columns]
        .contiguous()
    )


def convert_sources(sources, table, rotation):
    converted = []
    for source in sources:
        source = transform_csf_blocks(source, table)
        if rotation:
            source = concatenate_csf_blocks(
                (
                    slice_csf_blocks(source, row_range=(rotation, source.rows)),
                    slice_csf_blocks(source, row_range=(0, rotation)),
                )
            )
        converted.append(source)
    return tuple(converted)


@pytest.mark.parametrize("factor", [1.0, 2.0, 8.0])
@pytest.mark.parametrize("rotation", [0, 320])
@pytest.mark.parametrize("columns", [1, 10])
def test_nvfp4_packed_conversion_matches_native_preparation(factor, rotation, columns):
    from b12x.moe._shared.kernels.w4a16.prepare import (
        _permute_packed_scales,
        _process_nvfp4_packed_scales,
    )

    device = require_b12x()
    rows, experts = 640, 2
    sources, swizzled = planes_and_native(rows, columns, experts, dense=False)
    grid = logical_grid(swizzled, rows, columns).to(device).view(torch.float8_e4m3fn)
    alphabet = torch.arange(256, dtype=torch.uint8, device=device).view(
        torch.float8_e4m3fn
    )
    table = (
        _process_nvfp4_packed_scales(
            alphabet[:, None].expand(256, 4).contiguous(), scale_factor=factor
        )
        .view(torch.uint8)[:, 0]
        .cpu()
        .numpy()
    )
    converted = convert_sources(sources, table, rotation)
    reference = torch.stack(
        [
            _process_nvfp4_packed_scales(
                _permute_packed_scales(
                    torch.roll(plane.view(torch.uint8), -rotation, 0)
                    .view(torch.float8_e4m3fn)
                    .T.contiguous(),
                    size_k=columns * 16,
                    size_n=rows,
                    group_size=16,
                ),
                scale_factor=factor,
            ).view(torch.uint8)
            for plane in grid
        ]
    )
    payload = upload_csf_block_payload(
        converted, device, native_codes=True, group_record_bytes=3, destination=1
    )
    plane = Nvfp4CompactBlockPlane(payload, layout=1)
    outputs = tuple(torch.empty_like(reference) for _ in range(2))
    decoder = Nvfp4CsfDecoder.prepare(plane, plane, *outputs)
    decoder.decode_all(*outputs)
    for output in outputs:
        torch.testing.assert_close(output, reference, rtol=0, atol=0)


@pytest.mark.parametrize(
    "destination,rows,columns,group_rows",
    [
        (1, 128, 1, 128),
        (1, 576, 160, 576),
        (1, 5120, 18, 5120),
        (2, 576, 18, 288),
        (2, 1024, 160, 512),
        (2, 5120, 9, 5120),
        (3, 1152, 160, 576),
        (3, 5120, 18, 5120),
        (3, 384, 6, 192),
    ],
)
@pytest.mark.parametrize("rotate", [False, True])
def test_mxfp4_clamp_and_swizzle_match_native_preparation(
    destination, rows, columns, group_rows, rotate
):
    from b12x.moe._shared.kernels.w4a16.prepare import _pack_e8m0_k32_scales
    from b12x.moe.fused_moe._impl import (
        _e8m0_scale_to_w4a8_n64_sfb_inplace,
        _e8m0_scale_to_w4a8_sfb_inplace,
    )

    device = require_b12x()
    sources, swizzled = planes_and_native(
        rows, columns, 2, dense=False, quantization="mxfp4"
    )
    grid = logical_grid(swizzled, rows, columns).to(device)
    rotation = rows // 2 if rotate else 0
    table = np.minimum(np.arange(256), 247).astype(np.uint8)
    converted = convert_sources(sources, table, rotation)
    if destination == 1:
        reference = _pack_e8m0_k32_scales(
            grid,
            size_k=columns * 32,
            size_n=rows,
            row_rotation=rotation,
            allow_k_tail=True,
        )
    elif destination == 2:
        reference = _e8m0_scale_to_w4a8_sfb_inplace(
            grid.clone(),
            weight_E=2,
            rows=rows,
            k_dim=columns * 32,
            row_rotation=rotation,
            gated_half_rows=group_rows if group_rows != rows else None,
        )
    else:
        reference = _e8m0_scale_to_w4a8_n64_sfb_inplace(
            grid.clone(),
            weight_E=2,
            rows=rows,
            k_dim=columns * 32,
            row_rotation=rotation,
            group_rows=group_rows,
        )
    payload = upload_csf_block_payload(
        converted,
        device,
        native_codes=True,
        group_record_bytes=3,
        destination=destination,
        group_rows=group_rows,
    )
    plane = Nvfp4CompactBlockPlane(payload, layout=int(destination == 1))
    shape = (
        (2, plane.columns, plane.rows)
        if destination == 1
        else (2, plane.rows, plane.columns)
    )
    outputs = tuple(
        torch.empty(shape, dtype=torch.uint8, device=device) for _ in range(2)
    )
    decoder = Nvfp4CsfDecoder.prepare(plane, plane, *outputs)
    decoder.decode_all(*outputs)
    for output in outputs:
        torch.testing.assert_close(
            output.reshape(-1), reference.view(torch.uint8).reshape(-1), rtol=0, atol=0
        )
