# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Production SFA-first placement, independent of comparison input fixtures."""

import pytest

from cutlass.experimental.task_scheduling.memory import SmemAllocator
from flashinfer.prims_ts.batched_gemm import batched_gemm_run as runner
from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
    make_config,
    validate_config,
)
from flashinfer.prims_ts.batched_gemm.batched_gemm_kernel import (
    _build_schedule_validate,
)
from flashinfer.prims_ts.batched_gemm.routed_config import routed_tile256_options
from flashinfer.prims_ts.batched_gemm.routed_smem import (
    RoutedTile256SmemAllocator,
    make_smem_allocator,
)
from tests.prims_ts.routing_matrix_utils import CGAS, matrix_options
from tests.prims_ts.test_batched_gemm_eight_warp_fp4_fc1 import eight_warp_fc1_options
from tests.prims_ts.test_batched_gemm_rolled_fp4_fc1 import rolled_fc1_options
from tests.prims_ts.test_batched_gemm_routed_geometry import install_sfa_first


def placement_options(tile_m, cga, output):
    if output == "bf16":
        options = matrix_options(tile_m, cga, routed=True)
    elif output == "fc1-w4":
        options = rolled_fc1_options(tile_m, cga, routed=True, c_stages=2)
    else:
        options = eight_warp_fc1_options(tile_m, cga, routed=True)
    options.update(routed_sfa_first=1, routed_low_smem=0)
    return options


def layout_snapshot(allocator):
    return (
        {a.name: (a.offset, a.size_bytes, a.alignment) for a in allocator._allocations},
        [(r.name, r.pipeline_config.num_stages) for r in allocator._barrier_resources],
        [(g.name, g.pipeline_config.num_stages) for g in allocator._barrier_groups],
        allocator.total_smem_bytes,
        allocator.barrier_smem_bytes,
    )


def test_sfa_first_cli_and_default_allocator():
    parser = runner.build_arg_parser()
    assert runner._parse_overrides(parser.parse_args([]))["routed_sfa_first"] == 0
    overrides = runner._parse_overrides(parser.parse_args(["--routed-sfa-first", "1"]))
    assert overrides["routed_sfa_first"] == 1
    assert type(make_smem_allocator(make_config())) is SmemAllocator
    compact = make_config(**routed_tile256_options(max_active_clusters=2))
    assert isinstance(make_smem_allocator(compact), RoutedTile256SmemAllocator)


@pytest.mark.parametrize("value", [-1, 2, "1"])
def test_sfa_first_rejects_nonbinary(value):
    options = placement_options(512, (2, 1), "fc1-w8")
    options["routed_sfa_first"] = value
    with pytest.raises(ValueError, match="routed_sfa_first"):
        validate_config(make_config(**options))


@pytest.mark.parametrize("compact", [False, True])
def test_sfa_first_rejects_incompatible_allocator(compact):
    options = (
        routed_tile256_options(max_active_clusters=2)
        if compact
        else matrix_options(512, (2, 1), routed=False)
    )
    options["routed_sfa_first"] = 1
    with pytest.raises(ValueError, match="routed_sfa_first requires"):
        validate_config(make_config(**options))


@pytest.mark.parametrize("tile_m", [256, 512])
@pytest.mark.parametrize("cga", CGAS, ids=lambda c: f"{c[0]}x{c[1]}")
@pytest.mark.parametrize("output", ["bf16", "fc1-w4", "fc1-w8"])
def test_sfa_first_matches_measured_layout(monkeypatch, tile_m, cga, output):
    options = placement_options(tile_m, cga, output)
    cfg = make_config(**options)
    validate_config(cfg, problem_mnk=(12288, 1280, 2304))
    _, _, placed, _ = _build_schedule_validate(cfg)
    options["routed_sfa_first"] = 0
    default_cfg = make_config(**options)
    _, _, default, _ = _build_schedule_validate(default_cfg)
    # Only this compatibility audit uses the archived benchmark helper.
    # Numerical tests below compile through the production option alone.
    with monkeypatch.context() as patch:
        install_sfa_first(patch)
        _, _, measured, _ = _build_schedule_validate(default_cfg)
    assert layout_snapshot(placed) == layout_snapshot(measured)
    sfa = [a for a in placed._allocations if a.name.endswith("_sf_ldg")]
    assert len(sfa) == 1 and sfa[0].offset == 0
    original = {a.name: a for a in default._allocations}
    assert original[sfa[0].name].offset > 0
    assert placed.total_smem_bytes == default.total_smem_bytes
    assert layout_snapshot(placed)[1:] == layout_snapshot(default)[1:]
    for a in placed._allocations:
        assert a.offset % a.alignment == 0
        assert a.size_bytes == original[a.name].size_bytes
    regions = sorted((a.offset, a.offset + a.size_bytes) for a in placed._allocations)
    assert all(
        left[1] <= right[0] for left, right in zip(regions, regions[1:], strict=False)
    )


@pytest.mark.arch_rubin
@pytest.mark.timeout(240)
@pytest.mark.parametrize("tile_m", [256, 512])
@pytest.mark.parametrize(
    "cga", [(2, 1), (2, 4), (4, 2)], ids=lambda c: f"{c[0]}x{c[1]}"
)
@pytest.mark.parametrize("output", ["bf16", "fc1-w4", "fc1-w8"])
def test_sfa_first_nonzero_gpu(tile_m, cga, output):
    assert runner.reference_check(
        num_experts=8,
        num_tokens=1031,
        top_k=8,
        problem_n=1280,
        problem_k=2304,
        routing="uniform",
        repeat_launches=3,
        output_guard_elements=256,
        require_mixed_cga_runtime=cga != (2, 1),
        **placement_options(tile_m, cga, output),
    )


@pytest.mark.arch_rubin
@pytest.mark.timeout(240)
def test_sfa_first_fused_static_gpu():
    options = routed_tile256_options(max_active_clusters=2)
    options.update(routed_low_smem=0, routed_sfa_first=1)
    assert runner.reference_check(
        num_experts=3,
        num_tokens=1031,
        top_k=2,
        problem_n=512,
        problem_k=2304,
        routing="uniform",
        repeat_launches=3,
        output_guard_elements=256,
        **options,
    )
