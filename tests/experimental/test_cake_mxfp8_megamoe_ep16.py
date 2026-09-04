"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Tests for the experimental Cake MXFP8 MegaMoE EP16 backend.

import torch

from flashinfer.experimental.cake_mxfp8_megamoe_ep16.backend import (
    _pack_scale_n128_k128,
    _pack_scale_n256_k128,
)
from flashinfer.experimental.cake_mxfp8_megamoe_ep16.jit import _read_manifest
from flashinfer.moe_ep import (
    CakeMxfp8MegaMoeEp16,
    preprocess_cake_mxfp8_megamoe_ep16_weights,
)


def test_public_entry_points_are_experimental() -> None:
    assert CakeMxfp8MegaMoeEp16.is_experimental
    assert preprocess_cake_mxfp8_megamoe_ep16_weights.is_experimental


def test_generated_source_closure() -> None:
    _, manifest = _read_manifest()
    sequence = manifest["sequences"][0]
    assert sequence["arch"] == "sm_103a"
    assert len(sequence["translation_units"]["devices"]) == 3


def test_pack_scale_n256_k128() -> None:
    scales = torch.arange(256 * 8, dtype=torch.int32).view(1, 256, 8)
    packed = _pack_scale_n256_k128(scales)
    for row in range(256):
        for block_column in range(8):
            tile_k, u = divmod(block_column, 4)
            d, row_in_128 = divmod(row, 128)
            a, row32 = divmod(row_in_128, 32)
            offset = tile_k * 1024 + d * 512 + row32 * 16 + a * 4 + u
            assert packed[offset] == scales[0, row, block_column]


def test_pack_scale_n128_k128() -> None:
    scales = torch.arange(128 * 8, dtype=torch.int32).view(1, 128, 8)
    packed = _pack_scale_n128_k128(scales)
    for row in range(128):
        for block_column in range(8):
            tile_k, u = divmod(block_column, 4)
            a, row32 = divmod(row, 32)
            offset = tile_k * 512 + row32 * 16 + a * 4 + u
            assert packed[offset] == scales[0, row, block_column]
