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

"""CPU unit tests for MaskedScheduler cluster counting (issue #9629)."""

import ast
from pathlib import Path
from typing import Tuple

import pytest


def _load_masked_scheduler_namespace():
    """Extract and compile MaskedSchedulerParams and MaskedScheduler without GPU/CuTe runtime."""
    src_path = (
        Path(__file__).resolve().parents[3]
        / "flashinfer"
        / "gemm"
        / "kernels"
        / "grouped_gemm_masked_blackwell.py"
    )
    tree = ast.parse(src_path.read_text(encoding="utf-8"), filename=str(src_path))
    target_names = {"with_byte", "MaskedSchedulerParams", "MaskedScheduler"}
    nodes = [
        n
        for n in tree.body
        if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in target_names
    ]

    class _StripDSLDecorators(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            node.decorator_list = [
                d
                for d in node.decorator_list
                if isinstance(d, ast.Name) and d.id in ("staticmethod", "property")
            ]
            node.returns = None
            for arg in node.args.args + node.args.kwonlyargs:
                arg.annotation = None
            return self.generic_visit(node)

    mod_ast = ast.Module(
        body=[_StripDSLDecorators().visit(n) for n in nodes], type_ignores=[]
    )
    ast.fix_missing_locations(mod_ast)

    class FakeLayout:
        def __init__(self, shape):
            self.shape = shape

    class FakeCute:
        @staticmethod
        def ceil_div(a, b, **kw):
            if isinstance(a, tuple):
                return (
                    tuple((x + y - 1) // y for x, y in zip(a, b, strict=False))
                    + a[len(b) :]
                )
            return (a + b - 1) // b

        @staticmethod
        def zipped_divide(c, tiler):
            return {(0, (None, None, None)): FakeLayout(c)}

        @staticmethod
        def make_layout(shape, **kw):
            return FakeLayout(shape)

        @staticmethod
        def size(x, **kw):
            if isinstance(x, FakeLayout):
                return x.shape[0] * x.shape[1] * x.shape[2]
            res = 1
            for v in x:
                res *= v
            return res

    class FakeCutlass:
        @staticmethod
        def const_expr(x):
            return x

        Boolean = bool

    ns = {
        "cute": FakeCute,
        "cutlass": FakeCutlass,
        "Int32": int,
        "Uint8": int,
        "Uint64": int,
        "WorkTileInfo": lambda tile_idx, is_valid: (tile_idx, bool(is_valid)),
        "extract_mlir_values": lambda x: [x],
        "new_from_mlir_values": lambda obj, vals: vals[0],
    }
    exec(compile(mod_ast, str(src_path), "exec"), ns)
    return ns


class _FakeMaskedM(list):
    @property
    def shape(self):
        return (len(self),)


def _collect_scheduled_tiles(
    masked_m_vals,
    n_tiles: int,
    cluster_shape_mn: Tuple[int, int],
    *,
    is_swap_ab: bool = False,
    num_persistent_clusters: int = 2,
):
    ns = _load_masked_scheduler_namespace()
    c_tiles = (
        (n_tiles, 4, len(masked_m_vals))
        if is_swap_ab
        else (4, n_tiles, len(masked_m_vals))
    )
    params = ns["MaskedSchedulerParams"](
        masked_m=_FakeMaskedM(masked_m_vals),
        dst_signals=1,
        c=c_tiles,
        c_tiler=(128, 128),
        cluster_shape_mnk=(*cluster_shape_mn, 1),
        is_swap_ab=is_swap_ab,
    )
    grid_dim = (*cluster_shape_mn, num_persistent_clusters)
    valid_by_cta = {}
    dsm_by_cluster = {}
    for bidx in range(cluster_shape_mn[0]):
        for bidy in range(cluster_shape_mn[1]):
            coords = []
            for bidz in range(grid_dim[2]):
                sched = ns["MaskedScheduler"].create(
                    params, (bidx, bidy, bidz), grid_dim
                )
                dsm_pending = 0
                dsm_counter = 0
                tile_idx, is_valid = sched.initial_work_tile_info()
                while is_valid:
                    coords.append(tile_idx)
                    dsm_counter += 1
                    sched.advance_to_next_work()
                    (tile_idx, is_valid), dsm_pending = sched.get_current_work(
                        dsm_pending_packed=dsm_pending,
                        dsm_counter=dsm_counter,
                        num_c_stage=2,
                    )
                if (bidx, bidy) == (0, 0):
                    dsm_by_cluster[bidz] = dsm_pending
            valid_by_cta[(bidx, bidy)] = sorted(
                coords, key=lambda t: (t[2], t[0], t[1])
            )
    return valid_by_cta, dsm_by_cluster


@pytest.mark.cpu_only
def test_masked_scheduler_cluster_2x1_counts_m_clusters():
    """With cluster_shape_mn=(2, 1), per-expert M work is ceil(masked_m / 256), not ceil(masked_m / 128)."""
    # Expert 0 has 384 rows (3 M-tiles -> 2 M-clusters), Expert 1 has 128 rows (1 M-tile -> 1 M-cluster).
    # With N = 2 tiles (2 N-clusters), Expert 0 takes 4 clusters and Expert 1 takes 2 clusters (6 total).
    tiles_by_cta, _ = _collect_scheduled_tiles(
        [384, 128], n_tiles=2, cluster_shape_mn=(2, 1)
    )
    assert tiles_by_cta[(0, 0)] == [
        (0, 0, 0),
        (0, 1, 0),
        (2, 0, 0),
        (2, 1, 0),
        (0, 0, 1),
        (0, 1, 1),
    ]
    assert tiles_by_cta[(1, 0)] == [
        (1, 0, 0),
        (1, 1, 0),
        (3, 0, 0),
        (3, 1, 0),
        (1, 0, 1),
        (1, 1, 1),
    ]


@pytest.mark.cpu_only
def test_masked_scheduler_cluster_1x2_counts_n_clusters():
    """With cluster_shape_mn=(1, 2), per-expert N work is num_clusters_n = ceil(n_tiles / 2)."""
    # Expert 0 has 256 rows (2 M-tiles), Expert 1 has 128 rows (1 M-tile).
    # With N = 4 tiles and cluster_n = 2, num_clusters_n = 2.
    # Expert 0 takes 2 * 2 = 4 clusters; Expert 1 takes 1 * 2 = 2 clusters (6 total).
    tiles_by_cta, _ = _collect_scheduled_tiles(
        [256, 128], n_tiles=4, cluster_shape_mn=(1, 2)
    )
    assert tiles_by_cta[(0, 0)] == [
        (0, 0, 0),
        (0, 2, 0),
        (1, 0, 0),
        (1, 2, 0),
        (0, 0, 1),
        (0, 2, 1),
    ]
    assert tiles_by_cta[(0, 1)] == [
        (0, 1, 0),
        (0, 3, 0),
        (1, 1, 0),
        (1, 3, 0),
        (0, 1, 1),
        (0, 3, 1),
    ]


@pytest.mark.cpu_only
def test_masked_scheduler_swap_ab_cluster_2x2():
    """Under is_swap_ab=True, mode 0 is N and mode 1 is M; both must count clusters."""
    # Expert 0 has 384 rows (3 M-tiles -> 2 M-clusters), Expert 1 has 0 rows (0 M-clusters).
    # N has 4 tiles -> 2 N-clusters. Total clusters = 2 * 2 = 4, covering 16 CTA tiles.
    tiles_by_cta, dsm_by_cluster = _collect_scheduled_tiles(
        [384, 0], n_tiles=4, cluster_shape_mn=(2, 2), is_swap_ab=True
    )
    for cta_tiles in tiles_by_cta.values():
        assert len(cta_tiles) == 4
    all_tiles = sorted([t for cta_tiles in tiles_by_cta.values() for t in cta_tiles])
    expected_tiles = sorted([(n_t, m_t, 0) for n_t in range(4) for m_t in range(4)])
    assert all_tiles == expected_tiles
    # Both persistent clusters must record completion signals for both experts after 2 steps.
    for bidz in (0, 1):
        assert (dsm_by_cluster[bidz] & 0xFF) == 3
        assert ((dsm_by_cluster[bidz] >> 8) & 0xFF) == 3
