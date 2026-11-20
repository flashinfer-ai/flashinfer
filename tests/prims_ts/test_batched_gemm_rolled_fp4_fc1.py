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

"""Opt-in rolled NVFP4/SwiGLU epilogues, with separate CGA specializations."""

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
)
from flashinfer.prims_ts.batched_gemm.routed_config import (
    routed_tile256_fc1_options,
)
from tests.prims_ts.routing_matrix_utils import CGAS
from tests.prims_ts.test_batched_gemm_native_fp4_fc1 import fc1_options


def rolled_fc1_options(tile_m, cga, *, routed, output_cols=64, c_stages=3):
    options = fc1_options(tile_m, cga, routed=routed)
    if tile_m == 256 and routed and output_cols == 64:
        # The older separate-weight loader has 24 warps and an 80-register
        # initial limit, insufficient for x64 TMEM loads (ptxas needs >=82).
        # Use the optimized shared B/SFB preset for the wider output step.
        options = routed_tile256_fc1_options(max_active_clusters=2)
        flexible = cga != (2, 1)
        options.update(
            tile_scheduler=int(TileScheduler.PERSISTENT),
            static_persistent_clusters=0,
            num_stages_workid=3,
            cluster_dim_x=cga[0],
            cluster_dim_y=cga[1],
            fallback_cluster_dim_x=2 if flexible else 0,
            fallback_cluster_dim_y=1,
            dense_specialize_tma_cga=int(flexible),
            dense_local_barrier=int(flexible),
            routed_low_smem=int(not flexible),
        )
    options.update(
        dense_rolled_epilogue=1,
        epi_tile_n=2 * output_cols,
        tmem_ldst_max_num_regs=output_cols,
        gated_weight_group_size=64 if output_cols == 64 else 1,
        epilogue_regs=168 if output_cols == 64 else 192,
        num_stages_c_smem=c_stages,
        num_stages_tmem_acc=2 if tile_m == 256 else 1,
    )
    return options


def test_rolled_fc1_cli_is_opt_in():
    parser = runner.build_arg_parser()
    assert runner._parse_overrides(parser.parse_args([]))["dense_rolled_epilogue"] == 0
    overrides = runner._parse_overrides(
        parser.parse_args(["--dense-rolled-epilogue", "1"])
    )
    options = fc1_options(512, (2, 4), routed=True)
    options["dense_rolled_epilogue"] = overrides["dense_rolled_epilogue"]
    validate_config(make_config(**options))
    assert options["dense_rolled_epilogue"] == 1


@pytest.mark.parametrize("tile_m", [256, 512])
@pytest.mark.parametrize("cga", CGAS, ids=lambda c: f"{c[0]}x{c[1]}")
@pytest.mark.parametrize("routed", [False, True], ids=["dense", "routed"])
@pytest.mark.parametrize("rolled", [0, 1], ids=["unrolled", "rolled"])
def test_rolled_fc1_host_schedule(tile_m, cga, routed, rolled):
    options = rolled_fc1_options(tile_m, cga, routed=routed)
    options["dense_rolled_epilogue"] = rolled
    cfg = make_config(**options)
    validate_config(cfg, problem_mnk=(12288, 1280, 2304))
    assert cfg.num_stages_tmem_acc == (2 if tile_m == 256 else 1)
    assert cfg.non_swap_epilogue_subtile_cnt == tile_m // 128
    assert cfg.uses_non_swap_c_tma_ring
    assert cfg.fallback_cluster_shape == (None if cga == (2, 1) else (2, 1, 1))
    _build_batched_gemm_task_manager(verbose=False, **options)


@pytest.mark.parametrize("c_stages", [1, 2, 3, 4])
@pytest.mark.parametrize("rolled", [0, 1])
def test_rolled_fc1_static_host_schedule(c_stages, rolled):
    options = routed_tile256_fc1_options(max_active_clusters=2)
    # The existing preset remains opt-in, as does the generic config default.
    assert make_config(**options).dense_rolled_epilogue == 0
    options.update(dense_rolled_epilogue=rolled, num_stages_c_smem=c_stages)
    cfg = make_config(**options)
    validate_config(cfg)
    assert cfg.num_stages_tmem_acc == 2
    _build_batched_gemm_task_manager(verbose=False, **options)


@pytest.mark.parametrize(
    "override,error",
    [
        ({"dense_rolled_epilogue": 2}, "dense_rolled_epilogue"),
        ({"use_tma_store": 0}, "dense_rolled_epilogue"),
        ({"dtype_c": int(DType.BF16)}, "dense_rolled_epilogue"),
        ({"act_kind": int(ActKind.GEGLU)}, "dense_rolled_epilogue"),
        (
            {"use_max_tmem_overlap": 1, "dense_rolled_mma": 0},
            "dense_rolled_epilogue",
        ),
        ({"dense_dkg_store_order": 1}, "dense_dkg_store_order"),
        ({"dense_preload_epilogue": 1}, "dense_preload_epilogue"),
    ],
)
def test_rolled_fc1_rejects_unsupported_epilogues(override, error):
    # Avoid the independent U64/G64 layout guard masking epilogue validation.
    options = rolled_fc1_options(256, (2, 1), routed=False, output_cols=32)
    options.update(override)
    with pytest.raises(ValueError, match=error):
        validate_config(make_config(**options))


@pytest.mark.arch_rubin
@pytest.mark.timeout(240)
@pytest.mark.parametrize("tile_m", [256, 512])
@pytest.mark.parametrize("cga", CGAS, ids=lambda c: f"{c[0]}x{c[1]}")
@pytest.mark.parametrize("routed", [False, True], ids=["dense", "routed"])
def test_rolled_fc1_geometry_gpu(tile_m, cga, routed):
    # Nonzero Kaiming inputs, partial M/N clusters, K-ring wraparound and
    # both M fragments of tile512. Three scratch slots do not divide the
    # two/four output calls, so persistent tiles must preserve ring state.
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
        **rolled_fc1_options(tile_m, cga, routed=routed),
    )


@pytest.mark.arch_rubin
@pytest.mark.timeout(240)
@pytest.mark.parametrize("tile_m", [256, 512])
@pytest.mark.parametrize("routed", [False, True], ids=["dense", "routed"])
def test_rolled_fc1_interleaved_weights_gpu(tile_m, routed):
    # Exercise the original U/G interleave, 32 output columns per call,
    # single-slot C scratch and ragged routing (top-k < expert count).
    assert runner.reference_check(
        num_experts=8,
        # Tile512 needs three padded M tiles per expert to exceed one wave
        # for both physical CGA shapes on a 212-SM Rubin.
        num_tokens=2579 if tile_m == 512 else 1543,
        top_k=4,
        problem_n=1280,
        problem_k=2304,
        routing="uniform",
        repeat_launches=3,
        output_guard_elements=256,
        require_mixed_cga_runtime=True,
        **rolled_fc1_options(  # 2x4 preferred, 2x1 fallback.
            tile_m, (2, 4), routed=routed, output_cols=32, c_stages=1
        ),
    )


@pytest.mark.arch_rubin
@pytest.mark.timeout(240)
@pytest.mark.parametrize("c_stages", [1, 2, 3])
def test_rolled_fc1_static_persistent_ring_gpu(c_stages):
    options = routed_tile256_fc1_options(max_active_clusters=2)
    options.update(dense_rolled_epilogue=1, num_stages_c_smem=c_stages)
    # Two resident clusters force repeated work handoffs with two accumulator
    # stages, including C rings that do not divide the output-call count.
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


@pytest.mark.parametrize("tile_m", [256, 512])
@pytest.mark.parametrize("rolled", [0, 1])
def test_rolled_fc1_accumulator_release_follows_stores(tile_m, rolled):
    from cutlass.experimental.task_scheduling.enums import ScheduleStage
    from flashinfer.prims_ts.batched_gemm.batched_gemm_kernel import (
        _build_schedule_validate,
    )

    options = rolled_fc1_options(tile_m, (2, 4), routed=True)
    options["dense_rolled_epilogue"] = rolled
    cfg = make_config(**options)
    tasks, _, _, _ = _build_schedule_validate(cfg, num_k_tiles=9)
    epilogue = next(task for task in tasks if task.name == "EpilogueTask0")
    tail = epilogue.tail_schedule_list
    labels = [entry[-1] for entry in tail]
    # Verify that the option selects one runtime-loop call, rather than
    # silently retaining the unrolled calls, and preserves the TMEM lifetime.
    assert labels.count("store_dense_epilogue_loop") == rolled
    assert labels.count("store_epilogue") == (
        0 if rolled else cfg.non_swap_epilogue_subtile_cnt
    )
    store_positions = [
        i
        for i, label in enumerate(labels)
        if label in ("store_dense_epilogue_loop", "store_epilogue")
    ]
    wait_positions = [
        i
        for i, (resource, stage, _, _) in enumerate(tail)
        if resource.name == "TmemC" and stage == ScheduleStage.ConsumerWait
    ]
    release_positions = [
        i
        for i, (resource, stage, _, _) in enumerate(tail)
        if resource.name == "TmemC" and stage == ScheduleStage.ConsumerRelease
    ]
    assert len(wait_positions) == len(release_positions) == 1
    assert wait_positions[0] < min(store_positions)
    assert max(store_positions) < release_positions[0]
