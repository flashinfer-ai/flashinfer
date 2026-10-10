"""Compressed TP partitions retain every scale byte and exception value."""

import pytest
import torch

from b12x._lib.quant._nvfp4_csf_prepare import convert_nvfp4_csf_storage
from ..conftest import require_b12x


def converted(fixed, exceptions, **kwargs):
    device = require_b12x()
    batch = convert_nvfp4_csf_storage(
        fixed[None].to(device),
        exceptions.view(torch.uint8).to(device),
        torch.tensor([0, exceptions.numel()], dtype=torch.int64, device=device),
        **kwargs,
    )
    r, c = batch.rows, batch.columns
    native = batch.fixed.cpu()
    bases = native[:, :, :128].reshape(1, r // 128, 32, 4).transpose(2, 3).reshape(r, 1)
    packed = native[:, :, 128:].reshape(1, r // 128, c // 4, 32, 4, 2)
    codes = packed.permute(0, 1, 4, 3, 2, 5).reshape(r, c // 2)
    values = (
        bases.to(torch.int16)
        + torch.stack((codes & 15, codes >> 4), dim=-1).reshape(r, c)
    ).to(torch.uint8)
    words = batch.exceptions.cpu().view(torch.uint32).to(torch.int64)
    values.view(-1)[words & 0xFFFFFF] = (words >> 24).to(torch.uint8)
    return values[: batch.logical_rows or r, : batch.logical_columns or c]


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
        parts.append(converted(fixed, exceptions, rows=64, columns=8, **kwargs))
    assert torch.equal(torch.cat(parts, dim=axis), original)


def test_combined_slice_and_empty_exceptions():
    fixed, exceptions = plane()
    for words in (exceptions, exceptions[:0]):
        actual = converted(
            fixed, words, rows=64, columns=8, row_range=(16, 48), column_range=(2, 6)
        )
        assert torch.equal(actual, decode(fixed, words, 64, 8)[16:48, 2:6])


@pytest.mark.parametrize("positions", [[2, 1], [3, 3], [512]])
def test_rejects_corrupt_exception_positions(positions):
    fixed, _ = plane()
    with pytest.raises(ValueError, match="unique, sorted and in range"):
        converted(
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
        converted(*plane(), rows=64, columns=8, **kwargs)


def test_rejects_overflowing_row_base():
    fixed, exceptions = plane()
    fixed[0, 0] = 241
    with pytest.raises(ValueError, match="row bases"):
        converted(fixed, exceptions, rows=64, columns=8)


@pytest.mark.parametrize("join", [1, 2])
def test_empty_expert_segments_keep_exact_exception_partitions(join):
    device = require_b12x()
    fixed, exceptions = plane()
    count = exceptions.numel()
    segments = 4
    starts = [0, 0, count, count, 2 * count]
    batch = convert_nvfp4_csf_storage(
        torch.stack([fixed] * segments).to(device),
        torch.cat([exceptions, exceptions]).view(torch.uint8).to(device),
        torch.tensor(starts, dtype=torch.int64, device=device),
        rows=64,
        columns=8,
        join=join,
    )
    assert batch.task_offsets.cpu().tolist() == (
        [[0, 0], [0, count], [count, count], [count, 2 * count]]
        if join == 1
        else [[0, count], [count, 2 * count]]
    )
    words = batch.exceptions.cpu().view(torch.uint32).to(torch.int64)
    expected = torch.cat([exceptions, exceptions]).to(torch.int64)
    if join == 2:
        expected += 64 * 8
    assert torch.equal(words, expected)


@pytest.mark.parametrize("offsets", [[1, 13], [0, 12], [0, -1, 13], [0, 14, 13]])
def test_invalid_exception_offsets_are_rejected_before_conversion(offsets):
    device = require_b12x()
    fixed, exceptions = plane()
    with pytest.raises(ValueError, match="offsets"):
        convert_nvfp4_csf_storage(
            torch.stack([fixed] * (len(offsets) - 1)).to(device),
            exceptions.view(torch.uint8).to(device),
            torch.tensor(offsets, dtype=torch.int64, device=device),
            rows=64,
            columns=8,
        )


def test_layout_preparation_and_repacking_do_not_copy_payloads_to_cpu(monkeypatch):
    from b12x._lib.quant.nvfp4_csf import make_nvfp4_csf_batch, repack_nvfp4_csf_batch

    device = require_b12x()
    fixed, exceptions = plane()
    lut = torch.arange(256, dtype=torch.uint8, device=device)

    def unexpected(*args, **kwargs):
        raise AssertionError("CSF preparation must remain on CUDA")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "cpu", unexpected)
        patch.setattr(torch.Tensor, "numpy", unexpected)
        batch = make_nvfp4_csf_batch(
            [fixed], [exceptions], rows=64, columns=8, device=device
        )
        packed = repack_nvfp4_csf_batch(batch, row_rotation=32, value_lut=lut)
    assert packed.layout == 1
    assert packed.logical_rows == 64
    assert packed.exceptions.numel() == exceptions.nbytes
