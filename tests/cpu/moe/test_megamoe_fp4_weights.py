"""Independent checks of the public transforms and decoder byte addresses."""

import pytest
import torch

from flashinfer.megamoe.weights import (
    transform_mxfp4_weights_for_mega_moe,
    transform_nvfp4_weights_for_mega_moe,
)


@pytest.mark.cpu_only
@pytest.mark.parametrize("weight_format", ["nvfp4", "mxfp4"])
def test_decoder_addresses_preserve_nibbles_and_scales(weight_format):
    group = 16 if weight_format == "nvfp4" else 32
    transform = (
        transform_nvfp4_weights_for_mega_moe
        if group == 16
        else transform_mxfp4_weights_for_mega_moe
    )
    generator = torch.Generator().manual_seed(19)
    raw = []
    for n, k in ((512, 512), (512, 256)):
        raw.append(
            (
                torch.randint(
                    0, 256, (2, n, k // 2), dtype=torch.uint8, generator=generator
                ),
                torch.randint(
                    0, 255, (2, n, k // group), dtype=torch.uint8, generator=generator
                ),
            )
        )
    outputs = transform(*raw)
    for layer, ((packed, scales), output) in enumerate(zip(raw, outputs, strict=True)):
        e, n, half_k = packed.shape
        kb = half_k // 64
        if group == 16:
            rows = output.reshape(e, n, kb, 80)
            payload, sf = rows[..., :64], rows[..., 64:72]
            assert torch.count_nonzero(rows[..., 72:]) == 0
        else:
            tiles = output.reshape(e, n // 128, kb, 8704)
            payload = tiles[..., :8192].reshape(e, n // 128, kb, 4, 128, 16)
            payload = payload.permute(0, 1, 4, 2, 3, 5).reshape(e, n, kb, 64)
            sf = tiles[..., 8192:].reshape(e, n // 128, kb, 128, 4)
            sf = sf.permute(0, 1, 3, 2, 4).reshape(e, n, kb, 4)
        # Literal decoder mapping: PRMT selects low/high nibbles in order,
        # while sign bits come from byte bit 7 and byte bit 3 respectively.
        words = payload.reshape(e, n, kb, 16, 4)
        magnitudes = torch.stack((words & 7, (words >> 4) & 7), dim=-1).flatten(-2)
        signs = torch.cat((words >> 7, (words >> 3) & 1), dim=-1)
        decoded = magnitudes | (signs << 3)
        original_words = packed.reshape(e, n, kb, 16, 4)
        expected = torch.cat((original_words >> 4, original_words & 15), dim=-1)
        if layer == 0:
            # Map each physical interleaved gate/up row back to its source row.
            physical = torch.arange(n)
            source = physical // 16 * 8 + physical % 8 + (physical % 16 >= 8) * (n // 2)
            expected = expected[:, source]
            scales = scales[:, source]
        assert torch.equal(decoded, expected)
        assert torch.equal(sf.reshape(e, n, -1), scales)


@pytest.mark.cpu_only
def test_weight_transform_rejects_mismatched_scales():
    p1 = torch.zeros(1, 512, 256, dtype=torch.uint8)
    p2 = torch.zeros(1, 512, 128, dtype=torch.uint8)
    with pytest.raises(ValueError, match="matching scales"):
        transform_mxfp4_weights_for_mega_moe(
            (p1, torch.zeros(1, 512, 8, dtype=torch.uint8)),
            (p2, torch.zeros(1, 512, 8, dtype=torch.uint8)),
        )
