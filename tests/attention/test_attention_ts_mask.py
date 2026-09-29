# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for shared PrimTS attention masking predicates."""

import pytest

pytest.importorskip(
    "cutlass",
    minversion="4.7.0",
    reason="PrimTS attention tests require nvidia-cutlass-dsl==4.7.0",
)

from flashinfer.attention.prims_ts.kernels.mask import (
    kv_tile_is_fully_visible,
    kv_tile_needs_right_mask,
)


@pytest.mark.parametrize(
    ("tile_offset_k", "tile_size_kv", "visible_begin", "visible_end", "expected"),
    (
        pytest.param(0, 128, 0, 128, True, id="exact-interval"),
        pytest.param(128, 128, 0, 256, True, id="exact-right-boundary"),
        pytest.param(0, 128, 0, 129, True, id="interior-with-short-remainder"),
        pytest.param(128, 128, 0, 255, False, id="crosses-right-boundary"),
        pytest.param(0, 128, 1, 256, False, id="crosses-window-left-boundary"),
        pytest.param(128, 128, 128, 256, True, id="starts-at-window-boundary"),
        pytest.param(128, 128, 160, 160, False, id="empty-intersection"),
    ),
)
def test_kv_tile_is_fully_visible(
    tile_offset_k: int,
    tile_size_kv: int,
    visible_begin: int,
    visible_end: int,
    expected: bool,
) -> None:
    assert (
        kv_tile_is_fully_visible(
            tile_offset_k,
            tile_size_kv,
            visible_begin,
            visible_end,
        )
        is expected
    )


@pytest.mark.parametrize(
    ("tile_offset_k", "tile_size_kv", "visible_end", "expected"),
    (
        pytest.param(0, 128, 128, False, id="exact-boundary"),
        pytest.param(128, 128, 255, True, id="crosses-boundary"),
    ),
)
def test_kv_tile_needs_right_mask(
    tile_offset_k: int,
    tile_size_kv: int,
    visible_end: int,
    expected: bool,
) -> None:
    assert (
        kv_tile_needs_right_mask(tile_offset_k, tile_size_kv, visible_end) is expected
    )
