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

"""Two independent four-warp groups in the rolled NVFP4/SwiGLU epilogue."""

import pytest

from flashinfer.prims_ts.batched_gemm import batched_gemm_run as runner
from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
    ActKind,
    DType,
    TileScheduler,
    make_config,
    validate_config,
)
from flashinfer.prims_ts.batched_gemm.batched_gemm_kernel import (
    _build_batched_gemm_task_manager,
    _make_pipeline_configs,
)
from tests.prims_ts.routing_matrix_utils import CGAS
from tests.prims_ts.test_batched_gemm_rolled_fp4_fc1 import rolled_fc1_options


def eight_warp_fc1_options(tile_m, cga, *, routed, output_cols=64, c_stages=2):
    # Keep the combined B/SFB loader for routed tile256 at either output width.
    options = rolled_fc1_options(tile_m, cga, routed=routed, c_stages=c_stages)
    options.update(
        dense_epilogue_warps=8,
        epilogue_regs=144 if routed else 160,
        epi_tile_n=2 * output_cols,
        tmem_ldst_max_num_regs=output_cols,
        gated_weight_group_size=64 if output_cols == 64 else 1,
    )
    if routed:
        # A 24-warp CTA starts at 80 registers/thread. Redistribute the
        # allocated 61440 registers as 8x144 epilogue + 16x48 loader/control.
        options.update(load_a_regs=48, load_sfa_regs=48)
    return options


@pytest.mark.parametrize("tile_m", [256, 512])
@pytest.mark.parametrize("cga", CGAS, ids=lambda c: f"{c[0]}x{c[1]}")
@pytest.mark.parametrize("routed", [False, True], ids=["dense", "routed"])
def test_eight_warp_fc1_host_schedule(tile_m, cga, routed):
    options = eight_warp_fc1_options(tile_m, cga, routed=routed)
    cfg = make_config(**options)
    validate_config(cfg, problem_mnk=(12288, 1280, 2304))
    assert cfg.warp_layout.num_epilogue_warps == 8
    assert cfg.non_swap_epilogue_subtile_cnt == tile_m // 256
    assert cfg.non_swap_epilogue_cols_per_call == 256
    assert cfg.num_bytes_c_smem_scratch == 2 * 2 * 128 * 64 // 2
    assert cfg.num_stages_tmem_acc == (2 if tile_m == 256 else 1)
    pipes = _make_pipeline_configs(cfg)
    assert pipes["tmem_c"].consumer_group.size == 8 * 32 * 2
    _build_batched_gemm_task_manager(verbose=False, **options)


@pytest.mark.arch_rubin
@pytest.mark.timeout(240)
@pytest.mark.parametrize("tile_m", [256, 512])
@pytest.mark.parametrize("cga", CGAS, ids=lambda c: f"{c[0]}x{c[1]}")
@pytest.mark.parametrize("routed", [False, True], ids=["dense", "routed"])
def test_eight_warp_fc1_geometry_gpu(tile_m, cga, routed):
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
        **eight_warp_fc1_options(tile_m, cga, routed=routed),
    )


@pytest.mark.parametrize("tile_m", [256, 512])
@pytest.mark.parametrize("routed", [False, True], ids=["dense", "routed"])
def test_eight_warp_fc1_interleaved_schedule(tile_m, routed):
    options = eight_warp_fc1_options(
        tile_m, (2, 4), routed=routed, output_cols=32, c_stages=3
    )
    cfg = make_config(**options)
    validate_config(cfg)
    assert cfg.non_swap_epilogue_subtile_cnt == tile_m // 128
    assert cfg.num_bytes_c_smem_scratch == 2 * 3 * 128 * 32 // 2
    _build_batched_gemm_task_manager(verbose=False, **options)


@pytest.mark.parametrize("regs", [160, 168])
def test_eight_warp_fc1_rejects_register_overcommit(regs):
    options = eight_warp_fc1_options(512, (2, 1), routed=True)
    options["epilogue_regs"] = regs
    # 160 fits the physical register file but exceeds the 24-warp CTA's
    # initial allocation; 168 also exceeds the physical register file.
    with pytest.raises(ValueError, match="register budget|registers needed"):
        _build_batched_gemm_task_manager(verbose=False, **options)


@pytest.mark.parametrize(
    "override",
    [
        {"dense_rolled_epilogue": 0},
        {"dense_preload_epilogue": 1},
        {"dense_preload_store_prefix_fragments": 2},
        {"use_tma_store": 0},
        {"use_max_tmem_overlap": 1},
        {"dtype_c": int(DType.BF16)},
        {"act_kind": int(ActKind.GEGLU)},
    ],
)
def test_eight_warp_fc1_rejects_unsupported_output(override):
    options = eight_warp_fc1_options(256, (2, 1), routed=False, output_cols=32)
    options.update(override)
    with pytest.raises(ValueError, match="epilogue"):
        validate_config(make_config(**options))


@pytest.mark.arch_rubin
@pytest.mark.timeout(240)
@pytest.mark.parametrize("tile_m", [256, 512])
@pytest.mark.parametrize("routed", [False, True], ids=["dense", "routed"])
def test_eight_warp_fc1_interleaved_gpu(tile_m, routed):
    # Three C slots do not divide the per-tile call count. Both groups must
    # preserve independent ring state across repeated ragged persistent work.
    assert runner.reference_check(
        num_experts=8,
        num_tokens=2579 if tile_m == 512 else 1543,
        top_k=4,
        problem_n=1280,
        problem_k=2304,
        routing="uniform",
        repeat_launches=3,
        output_guard_elements=256,
        require_mixed_cga_runtime=True,
        **eight_warp_fc1_options(
            tile_m, (2, 4), routed=routed, output_cols=32, c_stages=3
        ),
    )


@pytest.mark.arch_rubin
@pytest.mark.timeout(240)
@pytest.mark.parametrize("output_cols,c_stages", [(64, 1), (64, 2), (32, 3)])
def test_eight_warp_fc1_static_ring_gpu(output_cols, c_stages):
    options = eight_warp_fc1_options(
        256, (2, 1), routed=True, output_cols=output_cols, c_stages=c_stages
    )
    options.update(
        tile_scheduler=int(TileScheduler.STATIC_PERSISTENT),
        static_persistent_clusters=2,
        num_stages_workid=2,
    )
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


@pytest.mark.arch_rubin
@pytest.mark.timeout(240)
@pytest.mark.parametrize("tile_m", [256, 512])
def test_eight_warp_fc1_second_group_out_of_bounds_gpu(tile_m):
    # N=1408 leaves 64 post-gating columns in the last tile: group zero
    # stores them, group one must suppress both C and SF-C stores while
    # advancing its own TMA queue/ring and participating in TMEM release.
    # Native routing and tile512's generic expert lookup require N%256=0.
    # Use the dense single-expert path to exercise the shared output code.
    options = eight_warp_fc1_options(tile_m, (2, 1), routed=False)
    options["dense_single_expert"] = 1
    assert runner.reference_check(
        num_experts=1,
        num_tokens=1031,
        top_k=1,
        problem_n=1408,
        problem_k=2304,
        routing="uniform",
        repeat_launches=3,
        output_guard_elements=256,
        **options,
    )
