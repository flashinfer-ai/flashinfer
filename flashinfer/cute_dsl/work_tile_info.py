# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FlashInfer-owned work tile for persistent CuTe DSL schedulers.

``cutlass.utils.WorkTileInfo`` is one process-global class shared by every
CuTe DSL user in the process, and CUTLASS DSL 4.7 Task Scheduling rewrites it
when it is first imported (the rewritten ``__init__`` accepts only a flat
``(m, n, l)`` triple of scalars). Schedulers whose tile coordinate is nested,
such as ``(m, 0, (batch, head))``, or has four axes, such as
``(block, head, batch, split)``, therefore must not inherit from or construct
that shared class. This type keeps the general contract instead: any
``cute.Coord`` is stored unchanged and round-trips through MLIR values.
"""

import cutlass.cute as cute
from cutlass.cutlass_dsl import Boolean, extract_mlir_values, new_from_mlir_values


class WorkTileInfo:
    """Tile coordinate plus validity flag of one persistent-scheduler step."""

    def __init__(self, tile_idx: cute.Coord, is_valid_tile: Boolean):
        self._tile_idx = tile_idx
        self._is_valid_tile = Boolean(is_valid_tile)

    def __extract_mlir_values__(self) -> list:
        return extract_mlir_values(self._tile_idx) + extract_mlir_values(
            self._is_valid_tile
        )

    def __new_from_mlir_values__(self, values: list) -> "WorkTileInfo":
        # The validity flag is one i1 value; everything before it is tile_idx.
        return type(self)(
            new_from_mlir_values(self._tile_idx, values[:-1]),
            new_from_mlir_values(self._is_valid_tile, values[-1:]),
        )

    @property
    @cute.jit
    def tile_idx(self) -> cute.Coord:
        return self._tile_idx

    @property
    @cute.jit
    def is_valid_tile(self) -> Boolean:
        return self._is_valid_tile
