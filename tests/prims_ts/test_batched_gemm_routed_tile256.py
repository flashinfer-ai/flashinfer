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

"""Native routed tile256, fused ring ownership and static/CLC queue contracts."""

import pytest

from flashinfer.prims_ts.batched_gemm import batched_gemm_run as runner
from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
    make_config,
    validate_config,
    TileScheduler,
)
from flashinfer.prims_ts.batched_gemm.batched_gemm_cga import (
    static_queue_consumer_threads,
)
from flashinfer.prims_ts.batched_gemm.batched_gemm_kernel import (
    _build_batched_gemm_task_manager,
)
from flashinfer.prims_ts.batched_gemm.routed_config import routed_tile256_options
from flashinfer.prims_ts.batched_gemm.routed_smem import RoutedTile256SmemAllocator


PROFILES = [
    ("static", "shared", 0),
    ("static", "separate", 0),
    ("clc", "shared", 0),
    ("clc", "separate", 0),
    ("clc", "separate", 1),
]


def options(scheduler="static", weights="auto", throttle=None, rolled=0):
    return routed_tile256_options(
        scheduler=scheduler,
        weight_pipeline=weights,
        work_throttle=throttle,
        rolled_mma=rolled,
        max_active_clusters=106,
    )


@pytest.mark.parametrize("scheduler,weights,throttle", PROFILES)
def test_routed_tile256_schedule_contract(scheduler, weights, throttle):
    opts = options(scheduler, weights, throttle)
    cfg = make_config(**opts)
    assert cfg.tmem_sfa_cols == 4 and cfg.tmem_sf_col_stride("a") == 2
    assert cfg.tmem_sfb_cols == 32
    assert cfg.warp_layout.load_sfa_warp_idx == cfg.warp_layout.load_a_warp_idx
    assert cfg.warp_layout.load_sfb_warp_idx == cfg.warp_layout.load_b_warp_idx
    assert static_queue_consumer_threads(cfg, 0) == 352
    assert static_queue_consumer_threads(cfg, 1) == 320
    allocator = RoutedTile256SmemAllocator(cfg)
    barriers = allocator.barrier_regions()
    assert ("TmaWeights" in barriers) == (weights == "shared")
    assert ("SmemSfB" in barriers) == (weights == "separate")
    # Run the real framework's schedule/resource/deadlock checks.
    _build_batched_gemm_task_manager(verbose=False, **opts)


@pytest.mark.parametrize(
    "overrides",
    [
        {"sm_arch": 100},
        {"tile_k": 128},
        {"route_act": 1},
        {"num_load_a_warps": 2, "num_load_sfa_warps": 2},
        {"num_stages_smem_sfa": 7},
        {"routed_native_sfa": 0},
        {"num_stages_b": 6, "num_stages_smem_sfb": 6},
        {"num_stages_tmem_acc": 1},
        {"use_early_exit": 1},
        {"use_work_throttle": 1},
        {"num_stages_workid": 3},
        {"static_persistent_clusters": 0},
        {"fused_a_single_commit": 0},
    ],
)
def test_routed_tile256_rejects_incompatible_layout(overrides):
    with pytest.raises(ValueError):
        validate_config(make_config(**(options() | overrides)))


def test_routed_tile256_cli_and_presets():
    parser = runner.build_arg_parser()
    args = parser.parse_args(
        [
            "--tile-scheduler",
            "static-persistent",
            "--routed-native-sfa",
            "1",
            "--routed-low-smem",
            "1",
            "--fused-a-single-commit",
            "1",
            "--static-persistent-clusters",
            "106",
        ]
    )
    assert (
        args.routed_native_sfa
        == args.routed_low_smem
        == args.fused_a_single_commit
        == 1
    )
    assert args.static_persistent_clusters == 106
    assert options()["tile_scheduler"] == int(TileScheduler.STATIC_PERSISTENT)
    assert options("clc")["use_work_throttle"] == 1


@pytest.mark.parametrize("n,k", [(384, 768), (512, 384)])
def test_routed_tile256_rejects_partial_hidden_tiles(n, k):
    with pytest.raises(ValueError, match="routed_native_sfa requires problem_"):
        runner._runtime_config(make_config(**options()), k, out_hidden=n)


@pytest.mark.arch_rubin
@pytest.mark.timeout(180)
@pytest.mark.parametrize(
    "scheduler,weights,throttle,rolled",
    [(*profile, 0) for profile in PROFILES]
    + [("static", "shared", 0, 1), ("clc", "separate", 1, 1)],
)
def test_routed_tile256_small_correctness(
    monkeypatch, scheduler, weights, throttle, rolled
):
    # Small irregular routes exercise masked rows and repeated persistent work.
    # A narrow SF range keeps the integer reference exactly representable.
    monkeypatch.setattr(
        runner,
        "_random_sf_bytes",
        lambda shape, *, cfg, device, layout_kind: runner._random_uint8(
            shape, low=0x28, high=0x41, device=device
        ),
    )
    result = runner.reference_check(
        num_experts=3,
        num_tokens=257,
        top_k=2,
        problem_n=512,
        problem_k=768,
        routing="uniform",
        repeat_launches=3,
        output_guard_elements=256,
        **options(scheduler, weights, throttle, rolled),
    )
    assert result and result.max_abs == 0


@pytest.mark.arch_rubin
@pytest.mark.timeout(180)
@pytest.mark.parametrize(
    "scheduler,weights", [("static", "separate"), ("clc", "shared")]
)
@pytest.mark.parametrize("empty_experts", [False, True])
def test_routed_tile256_ring_wrap_and_empty_experts(
    monkeypatch, scheduler, weights, empty_experts
):
    monkeypatch.setattr(
        runner,
        "_random_sf_bytes",
        lambda shape, *, cfg, device, layout_kind: runner._random_uint8(
            shape, low=0x28, high=0x41, device=device
        ),
    )
    opts = options(scheduler, weights, 0)
    if scheduler == "static":
        # Force repeated work and both queue slots even on a large GPU.
        opts["static_persistent_clusters"] = 2
    result = runner.reference_check(
        num_experts=8 if empty_experts else 3,
        num_tokens=1 if empty_experts else 1031,
        top_k=1 if empty_experts else 2,
        problem_n=512,
        problem_k=2304,  # Nine K tiles wrap the eight-stage operand rings.
        routing="uniform",
        repeat_launches=3,
        output_guard_elements=256,
        **opts,
    )
    assert result and result.max_abs == 0
