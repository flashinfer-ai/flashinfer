"""Compressed TP partitions retain every scale byte and exception value."""

import pytest
import torch

from b12x.loader._nvfp4_csf import slice_nvfp4_csf_plane


def decode(fixed, exceptions, rows, columns):
    bases = fixed[:, :16].reshape(rows, 1).to(torch.int16)
    codes = fixed[:, 16:].reshape(rows, columns // 2)
    offsets = torch.stack((codes & 15, codes >> 4), dim=-1).reshape(rows, columns)
    values = (bases + offsets).to(torch.uint8)
    words = exceptions.to(torch.int64)
    values.view(-1)[words & 0xFFFFFF] = (words >> 24).to(torch.uint8)
    return values


def plane():
    generator = torch.Generator().manual_seed(53)
    fixed = torch.randint(0, 256, (4, 80), dtype=torch.uint8, generator=generator)
    fixed[:, :16] %= 241
    positions = [0, 1, 7, 8, 127, 128, 255, 256, 257, 383, 384, 510, 511]
    exceptions = torch.tensor(
        [((241 + i) << 24) | position for i, position in enumerate(positions)],
        dtype=torch.uint32,
    )
    return fixed, exceptions


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("axis", [0, 1])
def test_partitions_reconstruct_original_scale_bytes(world, axis):
    fixed, exceptions = plane()
    original = decode(fixed, exceptions, 64, 8)
    parts = []
    for rank in range(world):
        extent = (64, 8)[axis] // world
        selected = (rank * extent, (rank + 1) * extent)
        kwargs = {"row_range" if axis == 0 else "column_range": selected}
        f, e = slice_nvfp4_csf_plane(fixed, exceptions, rows=64, columns=8, **kwargs)
        parts.append(
            decode(f, e, extent if axis == 0 else 64, extent if axis == 1 else 8)
        )
        assert f.is_contiguous() and e.is_contiguous()
    assert torch.equal(torch.cat(parts, dim=axis), original)


def test_combined_slice_and_empty_exceptions():
    fixed, exceptions = plane()
    for words in (exceptions, exceptions[:0]):
        f, e = slice_nvfp4_csf_plane(
            fixed, words, rows=64, columns=8, row_range=(16, 48), column_range=(2, 6)
        )
        assert torch.equal(decode(f, e, 32, 4), decode(fixed, words, 64, 8)[16:48, 2:6])


@pytest.mark.parametrize("positions", [[2, 1], [3, 3], [512]])
def test_rejects_corrupt_exception_positions(positions):
    fixed, _ = plane()
    with pytest.raises(ValueError, match="unique, sorted and in range"):
        slice_nvfp4_csf_plane(
            fixed, torch.tensor(positions, dtype=torch.uint32), rows=64, columns=8
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"row_range": (1, 16)},
        {"row_range": (0, 80)},
        {"row_range": (16, 16)},
        {"column_range": (1, 4)},
        {"column_range": (-2, 4)},
        {"column_range": (0, 10)},
    ],
)
def test_rejects_invalid_partition(kwargs):
    with pytest.raises(ValueError, match="aligned, nonempty and in range"):
        slice_nvfp4_csf_plane(*plane(), rows=64, columns=8, **kwargs)


def test_rejects_overflowing_row_base():
    fixed, exceptions = plane()
    fixed[0, 0] = 241
    with pytest.raises(ValueError, match="row bases"):
        slice_nvfp4_csf_plane(fixed, exceptions, rows=64, columns=8)
