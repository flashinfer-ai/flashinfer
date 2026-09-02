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

"""Require the installed CUTLASS DSL wheel for Prims-TS kernels."""

from __future__ import annotations

import importlib

_BOOTSTRAPPED = False
_BOOTSTRAP_ERROR: BaseException | None = None


def _install_work_tile_info_compatibility() -> None:
    """Keep Task Scheduling scalarization compatible with nested CuTe coords.

    CUTLASS DSL 4.7 Task Scheduling globally replaces ``WorkTileInfo``'s
    generic coordinate storage with three ``Int32`` fields. That representation
    is required by Prims-TS loop-carried scheduling, but other CUTLASS kernels
    use nested coordinates such as ``(m, 0, (batch, head))``. Preserve the
    scalar representation for flat triples and fall back to the upstream
    generic representation when any top-level coordinate is nested.
    """

    import cutlass
    import cutlass.cute as cute
    from cutlass.cutlass_dsl import extract_mlir_values, new_from_mlir_values
    from cutlass.utils import static_persistent_tile_scheduler as scheduler

    work_tile_info = scheduler.WorkTileInfo
    if getattr(work_tile_info, "_flashinfer_nested_coord_compatible", False):
        return

    @cute.jit
    def _work_tile_info_init_compatible(self, tile_idx, is_valid_tile) -> None:
        if cutlass.const_expr(any(isinstance(idx, (tuple, list)) for idx in tile_idx)):
            self._tile_idx = tile_idx
            self._tile_idx_num_values = None
        else:
            m_idx, n_idx, l_idx = tile_idx
            self._m_idx = cutlass.Int32(m_idx)
            self._n_idx = cutlass.Int32(n_idx)
            self._l_idx = cutlass.Int32(l_idx)
        self._is_valid_tile = cutlass.Boolean(is_valid_tile)

    def _work_tile_info_extract_compatible(self) -> list:
        if hasattr(self, "_tile_idx"):
            tile_idx_values = extract_mlir_values(self._tile_idx)
            self._tile_idx_num_values = len(tile_idx_values)
            return tile_idx_values + extract_mlir_values(self._is_valid_tile)
        return [
            self._m_idx.ir_value(),
            self._n_idx.ir_value(),
            self._l_idx.ir_value(),
            self._is_valid_tile.ir_value(),
        ]

    def _work_tile_info_new_from_mlir_values_compatible(self, values: list):
        if hasattr(self, "_tile_idx"):
            if self._tile_idx_num_values is None:
                raise ValueError(
                    "WorkTileInfo reconstruction requires tile_idx width recorded "
                    "during extraction"
                )
            num_tile_idx_values = self._tile_idx_num_values
            if len(values) != num_tile_idx_values + 1:
                raise ValueError(
                    "expected "
                    f"{num_tile_idx_values + 1} MLIR values for WorkTileInfo, "
                    f"got {len(values)}"
                )
            new_tile_idx = new_from_mlir_values(
                self._tile_idx, values[:num_tile_idx_values]
            )
            new_is_valid_tile = new_from_mlir_values(
                self._is_valid_tile, values[num_tile_idx_values:]
            )
            return work_tile_info(new_tile_idx, new_is_valid_tile)
        if len(values) != 4:
            raise ValueError(
                f"expected 4 MLIR values for WorkTileInfo, got {len(values)}"
            )
        return work_tile_info(
            tuple(cutlass.Int32(value) for value in values[:3]),
            cutlass.Boolean(values[3]),
        )

    @cute.jit
    def _work_tile_info_tile_idx_compatible(self):
        if cutlass.const_expr(hasattr(self, "_tile_idx")):
            return self._tile_idx
        return (self._m_idx, self._n_idx, self._l_idx)

    @cute.jit
    def _work_tile_info_is_valid_tile_compatible(self):
        return self._is_valid_tile

    @cute.jit
    def _work_tile_info_update_from_compatible(self, other) -> None:
        if cutlass.const_expr(hasattr(self, "_tile_idx")):
            self._tile_idx = other.tile_idx
        else:
            m_idx, n_idx, l_idx = other.tile_idx
            self._m_idx = cutlass.Int32(m_idx)
            self._n_idx = cutlass.Int32(n_idx)
            self._l_idx = cutlass.Int32(l_idx)
        self._is_valid_tile = cutlass.Boolean(other.is_valid_tile)

    work_tile_info.__init__ = _work_tile_info_init_compatible
    work_tile_info.__extract_mlir_values__ = _work_tile_info_extract_compatible
    work_tile_info.__new_from_mlir_values__ = (
        _work_tile_info_new_from_mlir_values_compatible
    )
    work_tile_info.tile_idx = property(_work_tile_info_tile_idx_compatible)
    work_tile_info.is_valid_tile = property(_work_tile_info_is_valid_tile_compatible)
    work_tile_info.update_from = _work_tile_info_update_from_compatible
    work_tile_info._flashinfer_nested_coord_compatible = True


def _has_required_modules() -> bool:
    required = (
        "cutlass",
        "cutlass.cute",
        "cutlass.experimental.primitives",
        "cutlass.experimental.task_scheduling",
    )
    for name in required:
        importlib.import_module(name)
    _install_work_tile_info_compatibility()
    return True


def ensure_cutlass_dsl_experimental() -> bool:
    """Return whether Prims-TS dependencies are importable from installed wheels."""

    global _BOOTSTRAPPED, _BOOTSTRAP_ERROR

    if _BOOTSTRAPPED:
        return True
    if _BOOTSTRAP_ERROR is not None:
        return False

    try:
        importlib.invalidate_caches()
        _has_required_modules()
    except BaseException as exc:
        _BOOTSTRAP_ERROR = exc
        return False

    _BOOTSTRAPPED = True
    return True


def require_cutlass_dsl_experimental() -> None:
    if ensure_cutlass_dsl_experimental():
        return
    raise RuntimeError(
        "Prims-TS requires the CUTLASS DSL wheel. Install the pinned "
        "release-branch wheel before importing flashinfer.prims_ts."
    ) from _BOOTSTRAP_ERROR


def get_cutlass_dsl_bootstrap_error() -> BaseException | None:
    return _BOOTSTRAP_ERROR
