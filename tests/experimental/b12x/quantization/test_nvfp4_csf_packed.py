"""The stage-readable packed CSF index rebuilds exactly what the W4A16 expansion pass writes."""

import numpy as np
import pytest
import torch

from b12x._lib.quant.nvfp4_csf import (
    Nvfp4CsfDecoder,
    make_nvfp4_csf_batch,
    repack_nvfp4_csf_batch,
)
from b12x._lib.quant.nvfp4_csf_packed import (
    PackedCsfPlane,
    build_packed_csf_scales,
    expand_packed_csf_scales,
    packed_slab_position,
)
from b12x.moe._shared.kernels.w4a16.prepare import _process_nvfp4_packed_scales
from ..conftest import require_b12x


def _logical_scales(experts, rows, columns, seed, outliers=0.03):
    """E4M3 scale bytes with a narrow per-row range and some out-of-window values.

    Two rows hold zero and subnormal scales, which W4A16's value table does not
    re-bias like normal ones: one as out-of-window values, one within its window.
    """
    rng = np.random.default_rng(seed)
    base = rng.integers(0x28, 0x48, size=(experts, rows, 1))
    scales = base + rng.integers(0, 14, size=(experts, rows, columns))
    hot = rng.random((experts, rows, columns)) < outliers
    scales[hot] = rng.integers(0x40, 0x7E, size=int(hot.sum()))
    scales[:, 5, ::7] = rng.integers(0, 8, size=scales[:, 5, ::7].shape)
    scales[:, 9] = 4 + np.arange(columns) % 13
    return scales.astype(np.uint8)


def _value_table(kind, device):
    """W4A16's E4M3 re-bias table, or a permutation no constant folds."""
    if kind == "permutation":
        table = np.random.default_rng(7).permutation(256).astype(np.uint8)
        return torch.from_numpy(table).to(device)
    alphabet = torch.arange(256, dtype=torch.uint8, device=device).view(torch.float8_e4m3fn)
    alphabet = alphabet[:, None].expand(256, 4).contiguous().to(torch.bfloat16)
    packed = _process_nvfp4_packed_scales(alphabet, scale_factor=2.0)
    return packed.view(torch.uint8)[:, 0].contiguous()


def _native_batch(source, device):
    """Byte-window planes (row base, 4-bit offsets, exceptions) of logical scales."""
    experts, rows, columns = source.shape
    fixed, exceptions = [], []
    for plane in source:
        base = np.minimum(plane.min(axis=1), 240).astype(np.uint8)
        offsets = plane.astype(np.int16) - base[:, None]
        outside = offsets > 15
        offsets[outside] = 0
        offsets = offsets.astype(np.uint8)
        packed = offsets[:, ::2] | (offsets[:, 1::2] << 4)
        fixed.append(
            np.concatenate((base.reshape(-1, 16), packed.reshape(rows // 16, -1)), 1)
        )
        position = np.flatnonzero(outside).astype(np.uint32)
        exceptions.append(position | (plane.ravel()[position].astype(np.uint32) << 24))
    return make_nvfp4_csf_batch(fixed, exceptions, rows=rows, columns=columns, device=device)


def test_packed_slab_positions_invert_the_plane_permutation():
    perm = (
        np.arange(64).reshape(8, 8).T.reshape(-1).reshape(-1, 4)[:, [0, 2, 1, 3]].reshape(-1)
    )
    perm = np.concatenate((perm, perm + 64))
    positions = packed_slab_position(torch.arange(128)).numpy()
    assert np.array_equal(perm[positions], np.arange(128))


@pytest.mark.parametrize("table", ["w4a16", "permutation"])
@pytest.mark.parametrize("rows,columns,rotation", [(256, 64, 128), (512, 16, 0), (128, 256, 0)])
def test_index_matches_the_expansion_pass(rows, columns, rotation, table):
    device = require_b12x()
    experts = 5
    lut = _value_table(table, device)
    first = repack_nvfp4_csf_batch(
        _native_batch(_logical_scales(experts, rows, columns, 1), device),
        row_rotation=rotation,
        value_lut=lut,
    )
    second = repack_nvfp4_csf_batch(
        _native_batch(_logical_scales(experts, 256, 32, 2), device),
        row_rotation=0,
        value_lut=lut,
    )
    out13 = torch.empty(experts, columns, rows, dtype=torch.uint8, device=device)
    out2 = torch.empty(experts, 32, 256, dtype=torch.uint8, device=device)
    outputs = (out13.view(torch.float8_e4m3fn), out2.view(torch.float8_e4m3fn))
    decoder = Nvfp4CsfDecoder.prepare(first, second, *outputs)
    decoder.decode(torch.arange(experts, dtype=torch.int32, device=device), *outputs)
    torch.cuda.synchronize()
    stored = []
    for batch, expected in ((first, out13), (second, out2)):
        scales = build_packed_csf_scales(batch)
        stored.append(scales)
        assert torch.equal(expand_packed_csf_scales(scales), expected)
        assert scales.max_atom_words <= 128
        if table == "w4a16":
            # The table folds into the bases: most words carry no exception.
            exception_bytes = scales.storage.numel() - scales.words_offset - 512
            assert exception_bytes < 0.25 * experts * batch.rows * batch.columns
    # Calls too large for stage-wise reads expand routed experts from the storage.
    again = tuple(torch.full_like(out, 0xFF) for out in (out13, out2))
    views = tuple(out.view(torch.float8_e4m3fn) for out in again)
    expander = Nvfp4CsfDecoder.prepare(*(PackedCsfPlane.of(s) for s in stored), *views)
    expander.decode(torch.tensor([[3, 1]], dtype=torch.int32, device=device), *views)
    torch.cuda.synchronize()
    for out, expected in zip(again, (out13, out2), strict=True):
        assert torch.equal(out[[1, 3]], expected[[1, 3]])


@pytest.mark.parametrize("packed_index", [0, 1])
def test_pair_decodes_each_projection_storage_layout(packed_index):
    device = require_b12x()
    lut = _value_table("w4a16", device)
    planes = [repack_nvfp4_csf_batch(
        _native_batch(_logical_scales(3, 256, 16, seed), device),
        row_rotation=0, value_lut=lut,
    ) for seed in (9, 13)]
    expected = [torch.empty(3, 16, 256, dtype=torch.float8_e4m3fn, device=device) for _ in planes]
    ids = torch.arange(3, device=device, dtype=torch.int32)
    Nvfp4CsfDecoder.prepare(*planes, *expected).decode(ids, *expected)
    planes[packed_index] = PackedCsfPlane.of(build_packed_csf_scales(planes[packed_index]))
    actual = [torch.empty_like(t) for t in expected]
    Nvfp4CsfDecoder.prepare(*planes, *actual).decode(ids, *actual)
    torch.cuda.synchronize()
    for output, reference in zip(actual, expected):
        assert torch.equal(output.view(torch.uint8), reference.view(torch.uint8))
