"""GPU checkpoint regrouping agrees with independent CPU block transforms."""

import numpy as np
import pytest
import torch

from b12x.loader._csf_blocks import (
    concatenate_csf_blocks,
    slice_csf_blocks,
    transform_csf_blocks,
)
from b12x.loader._csf_source import upload_csf_planes, stage_csf_sources
from b12x._lib.quant._csf_prepare import prepare_csf_payload, validate_source
from b12x._lib.quant.csf_blocks import upload_csf_block_payload, Nvfp4CompactBlockPlane
from b12x._lib.quant.nvfp4_csf import Nvfp4CsfDecoder
from .test_csf_blocks import planes_and_native
from ..conftest import require_b12x


@pytest.mark.parametrize("quantization", ["nvfp4", "mxfp4"])
@pytest.mark.parametrize(
    "destination,rows,columns,join,row_range,column_range,rotation,group_rows",
    [
        (0, 640, 160, 2, None, None, 0, None),
        (0, 640, 160, 2, (320, 640), None, 320, None),
        (0, 640, 160, 2, (480, 640), None, 0, None),
        (0, 256, 40, 1, None, (30, 40), 0, None),
        (0, 128, 9, 1, None, (1, 8), 0, None),
        (1, 128, 1, 1, None, None, 0, None),
        (1, 640, 10, 1, None, None, 320, None),
        (2, 576, 18, 1, None, None, 288, 288),
        (3, 1152, 160, 1, None, None, 576, 576),
    ],
)
def test_gpu_gather_matches_cpu_native_payload(
    quantization,
    destination,
    rows,
    columns,
    join,
    row_range,
    column_range,
    rotation,
    group_rows,
):
    device = require_b12x()
    planes, _ = planes_and_native(
        rows, columns, 2 * join, dense=False, quantization=quantization
    )
    source = upload_csf_planes(
        planes, device, join=join, row_range=row_range, column_range=column_range
    )
    table = (
        np.minimum(np.arange(256), 247).astype(np.uint8)
        if quantization == "mxfp4"
        else None
    )
    selected = tuple(
        slice_csf_blocks(p, row_range=row_range, column_range=column_range)
        for p in planes
    )
    selected = tuple(
        concatenate_csf_blocks(selected[i : i + join])
        for i in range(0, len(selected), join)
    )
    if table is not None:
        selected = tuple(transform_csf_blocks(p, table) for p in selected)
    if rotation:
        selected = tuple(
            concatenate_csf_blocks(
                (
                    slice_csf_blocks(p, row_range=(rotation, p.rows)),
                    slice_csf_blocks(p, row_range=(0, rotation)),
                )
            )
            for p in selected
        )
    expected = upload_csf_block_payload(
        selected,
        device,
        native_codes=True,
        group_record_bytes=3,
        destination=destination,
        group_rows=group_rows,
    )
    actual = prepare_csf_payload(
        source,
        destination=destination,
        rotation=rotation,
        group_rows=group_rows,
        value_table=None if table is None else torch.from_numpy(table).to(device),
    )
    assert actual.storage.numel() == expected.storage.numel()
    outputs = []
    for payload in (actual, expected):
        plane = Nvfp4CompactBlockPlane(payload, layout=int(destination == 1))
        shape = (
            (2, plane.columns, plane.rows)
            if destination == 1
            else (2, plane.rows, plane.columns)
        )
        first = torch.empty(shape, dtype=torch.uint8, device=device)
        second = torch.empty_like(first)
        Nvfp4CsfDecoder.prepare(plane, plane, first, second).decode_all(first, second)
        outputs.append(first)
    torch.testing.assert_close(*outputs, rtol=0, atol=0)


@pytest.mark.parametrize("quantization", ["nvfp4", "mxfp4"])
@pytest.mark.parametrize(
    "join,row_range,column_range", [(1, None, (3, 13)), (2, (32, 192), None)]
)
def test_slab_source_expands_to_same_native_bytes(
    quantization, join, row_range, column_range
):
    from b12x.loader._csf_blocks import logical_csf_components

    device = require_b12x()
    rows, columns = 256, 16
    planes, _ = planes_and_native(
        rows, columns, 2 * join, dense=True, quantization=quantization
    )
    bits = 4 if quantization == "nvfp4" else 1
    fields = 8 // bits
    slabs = []
    for plane in planes:
        bases, offsets, positions, values = logical_csf_components(plane)
        codes = np.bitwise_or.reduce(
            offsets.reshape(rows, -1, fields) << (np.arange(fields) * bits), axis=-1
        ).astype(np.uint8)
        fixed = np.concatenate(
            (bases.reshape(-1, 16), codes.reshape(rows // 16, -1)), axis=1
        )
        words = positions.astype(np.uint32) | (values.astype(np.uint32) << 24)
        slabs.append((torch.from_numpy(fixed), torch.from_numpy(words)))
    slab = stage_csf_sources(
        slabs,
        rows=rows,
        columns=columns,
        quantization=quantization,
        device=device,
        encoding="csf",
        join=join,
        row_range=row_range,
        column_range=column_range,
    )
    blocks = upload_csf_planes(
        planes, device, join=join, row_range=row_range, column_range=column_range
    )
    outputs = []
    for source in (slab, blocks):
        validate_source(source)
        plane = Nvfp4CompactBlockPlane(prepare_csf_payload(source))
        output = torch.empty(
            (2, plane.rows, plane.columns), dtype=torch.uint8, device=device
        )
        Nvfp4CsfDecoder.prepare(
            plane, plane, output, torch.empty_like(output)
        ).decode_all(output, torch.empty_like(output))
        outputs.append(output)
    torch.testing.assert_close(*outputs, rtol=0, atol=0)


@pytest.mark.parametrize(
    "corruption", ["bases", "offsets", "page_bases", "exceptions", "starts"]
)
def test_gpu_validation_rejects_corrupt_blocks(corruption):
    device = require_b12x()
    planes, _ = planes_and_native(128, 8, 2, dense=False)
    source = upload_csf_planes(planes, device)
    if corruption == "starts":
        source.starts[0] = 1
    else:
        index = ("bases", "codes", "page_bases", "offsets", "exceptions").index(
            corruption
        )
        tensor = source.components[index].reshape(-1)
        tensor[0] = {
            "bases": 255,
            "offsets": 65535,
            "page_bases": 0xFFFFFFFE,
            "exceptions": 255,
        }[corruption]
    with pytest.raises(ValueError, match="CSF"):
        validate_source(source)
