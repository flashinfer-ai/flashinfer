"""Byte-level checks for slicing compressed UE8M0 scales across TP ranks."""

import numpy as np
import pytest
import torch

from b12x.moe.checkpoints.exact_mxfp4 import slice_scale_plane


def decode(fixed, exceptions, rows, columns):
    selectors = (columns + 7) // 8
    stream = fixed.numpy().reshape(rows // 16, 16 * (1 + selectors))
    bits = np.unpackbits(
        stream[:, 16:].reshape(rows, selectors), axis=1, bitorder="little"
    )[:, :columns]
    result = (stream[:, :16].reshape(rows, 1) + bits).astype(np.uint8)
    words = exceptions.numpy()
    result.flat[words & 0xFFFFFF] = words >> 24
    return result


@pytest.mark.parametrize("columns", [9, 18, 36, 72, 160])
@pytest.mark.parametrize("row_slice", [(0, 64), (16, 48), (48, 64)])
def test_slices_preserve_all_scale_bytes(columns, row_slice):
    rng = np.random.default_rng(78)
    rows = 64
    bits = rng.integers(0, 2, (rows, columns), dtype=np.uint8)
    bases = rng.integers(0, 254, (rows // 16, 16), dtype=np.uint8)
    fixed = torch.from_numpy(
        np.concatenate(
            (
                bases,
                np.packbits(bits, axis=1, bitorder="little").reshape(rows // 16, -1),
            ),
            1,
        )
    )
    positions = np.unique(rng.integers(0, rows * columns, 50)).astype(np.uint32)
    values = rng.integers(0, 256, len(positions), dtype=np.uint32)
    exceptions = torch.from_numpy(positions | values << 24)
    reference = decode(fixed, exceptions, rows, columns)
    for c0, c1 in ((0, columns), (1, columns), (columns // 2, columns)):
        sliced = slice_scale_plane(
            fixed, exceptions, rows, columns, row_slice, (c0, c1)
        )
        actual = decode(*sliced, row_slice[1] - row_slice[0], c1 - c0)
        assert np.array_equal(actual, reference[row_slice[0] : row_slice[1], c0:c1])


def test_rejects_unaligned_rows_and_bad_padding():
    fixed = torch.zeros((4, 48), dtype=torch.uint8)
    exceptions = torch.empty(0, dtype=torch.uint32)
    with pytest.raises(ValueError, match="16-row"):
        slice_scale_plane(fixed, exceptions, 64, 9, (1, 17), (0, 9))
    fixed[0, 17] = 128
    with pytest.raises(ValueError, match="unused selector"):
        slice_scale_plane(fixed, exceptions, 64, 9, (0, 64), (0, 9))
