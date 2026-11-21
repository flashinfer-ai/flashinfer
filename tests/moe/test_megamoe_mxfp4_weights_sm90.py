"""Check the complete compact and predecoded weight tiles against MXFP4 values."""

import pytest
import torch


def _logical_weights(packed, scales, interleave):
    words = packed.cpu().reshape(*packed.shape[:2], -1, 4)
    codes = torch.cat((words >> 4, words & 15), dim=-1).flatten(2)
    lattice = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6])
    values = lattice[(codes & 7).long()] * torch.where(codes < 8, 1.0, -1.0)
    values *= torch.exp2(scales.cpu().float() - 127).repeat_interleave(32, -1)
    values = values.to(torch.float8_e4m3fn).view(torch.uint8)
    if interleave:
        e, n, k = values.shape
        gate, up = values.chunk(2, dim=1)
        values = torch.stack(
            (gate.reshape(e, -1, 8, k), up.reshape(e, -1, 8, k)), dim=2
        ).reshape(e, n, k)
    return values


def _restore_tiles(tiles, storage):
    tiles = tiles.cpu().long()
    row = torch.arange(128)[:, None]
    col = torch.arange(128)[None, :]
    stored_col = col ^ ((row & 7) * 16)
    if storage == "fp8":
        return (
            tiles[..., (row * 128 + stored_col).flatten()]
            .reshape(*tiles.shape[:2], 128, 128)
            .to(torch.uint8)
        )
    half, warp = row // 64, (row % 64) // 16
    lane = (row % 8) * 4 + (col % 16) // 4
    pair = (row % 16) // 8
    group, side, byte = col // 32, (col % 32) // 16, col % 4
    offset = half * 4096 + warp * 1024 + lane * 16
    offset = offset + (group // 2) * 512 + ((group % 2) * 2 + pair) * 4
    words = tiles.reshape(*tiles.shape[:2], -1, 4)
    words = sum(words[..., i] << (8 * i) for i in range(4))
    payload = words[..., (offset // 4).flatten()].reshape(*tiles.shape[:2], 128, 128)
    sf_offset = 8192 + half * 128 + warp * 32 + (lane // 4) * 4
    sf = words[..., (sf_offset // 4).flatten()].reshape(*tiles.shape[:2], 128, 128)
    exponent = ((sf >> (group * 4 + pair * 16)) & 15) - 6
    magnitude = (payload >> ((side * 4 + byte) * 4)) & 7
    sign = ((payload >> (3 + byte * 8 + side * 4)) & 1) << 7
    table = torch.tensor([0, 0x30, 0x38, 0x3C, 0x40, 0x44, 0x48, 0x4C])
    decoded = torch.where(magnitude != 0, table[magnitude] + exponent * 8, 0)
    return (decoded | sign).to(torch.uint8)


@pytest.mark.parametrize("storage", ["mxfp4_compact", "fp8"])
def test_mxfp4_weight_storage_exact(storage):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("requires an SM90 GPU")
    from flashinfer.megamoe import transform_mxfp4_weights_for_mega_moe

    generator = torch.Generator().manual_seed(1729)
    raw = []
    expected = []
    for n, k, interleave in ((256, 256, True), (256, 128, False)):
        packed = torch.randint(
            256, (2, n, k // 2), generator=generator, dtype=torch.uint8
        )
        scales = torch.randint(
            122, 134, (2, n, k // 32), generator=generator, dtype=torch.uint8
        )
        packed[:, :, :16] = 0x88
        scales[:, :, 0] = 0
        expected.append(_logical_weights(packed, scales, interleave))
        raw.append((packed.cuda(), scales.cuda()))
    transformed = transform_mxfp4_weights_for_mega_moe(*raw, storage=storage)
    for result, reference in zip(transformed, expected, strict=True):
        e, n, k = reference.shape
        restored = _restore_tiles(result, storage)
        restored = restored.reshape(e, n // 128, k // 128, 128, 128)
        restored = restored.permute(0, 1, 3, 2, 4).reshape(e, n, k)
        assert torch.equal(restored, reference)
