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

"""Rubin BatchedGemm architecture-feature regressions."""

from dataclasses import replace

import cutlass
import cutlass.cute as cute
import pytest
import torch
from cutlass import Int32
from cutlass.cute.runtime import make_ptr
from cutlass.experimental import primitives as prims

from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
    ActKind,
    BatchMode,
    BiasType,
    CtaRasterOrder,
    DType,
    RouteImpl,
    SfLayout,
    TileScheduler,
    compute_warp_layout,
    make_config,
    resolve_problem_config,
    tmem_sf_col_stride,
    tmem_sf_cols,
    tmem_sf_vec4_groups_per_mma,
    uniform_pipeline_stage_overrides,
    validate_config,
)


def _fp4_config(**overrides):
    values = dict(
        sm_arch=107,
        batch_mode=int(BatchMode.BATCH_M),
        transpose_mma_output=0,
        route_act=1,
        route_sfs_act=int(RouteImpl.AUTO),
        sf_layout_b=int(SfLayout.R8c4),
        umma_cta_count=1,
        tile_n=8,
        tile_k=256,
        mma_m=128,
        mma_n=8,
        mma_k=64,
        epi_tile_n=8,
        dtype_a=int(DType.E2M1),
        dtype_b=int(DType.E2M1),
        dtype_c=int(DType.BF16),
        act_kind=0,
        tile_scheduler=int(TileScheduler.STATIC),
        num_stages_a=5,
        num_stages_b=5,
        num_stages_smem_sfa=5,
        num_stages_smem_sfb=5,
        num_stages_tmem_acc=1,
        num_stages_tmem_sfa=5,
        num_stages_tmem_sfb=5,
        num_stages_c_smem=1,
        use_unroll_loop_2x_for_mma=0,
        use_max_tmem_overlap=0,
    )
    values.update(overrides)
    return make_config(**values)


def test_rubin_architecture_limits():
    blackwell = _fp4_config(sm_arch=100)
    rubin = _fp4_config(sm_arch=107)

    assert blackwell.smem_capacity_bytes == 227 * 1024
    assert rubin.smem_capacity_bytes == 327 * 1024
    assert blackwell.tmem_capacity_columns == 512
    assert rubin.tmem_capacity_columns == 576
    assert rubin.arch_name == "sm_107"


def test_runtime_architecture_detection(monkeypatch):
    from flashinfer.prims_ts.batched_gemm import (
        batched_gemm_run,
    )

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (10, 7))
    monkeypatch.setattr(
        batched_gemm_run, "is_prims_ts_arch_available", lambda major, minor: True
    )
    assert batched_gemm_run._resolve_runtime_sm_arch({"tile_n": 64}) == {
        "tile_n": 64,
        "sm_arch": 107,
    }
    assert batched_gemm_run._resolve_runtime_sm_arch({"sm_arch": 100}) == {
        "sm_arch": 100
    }


def test_runtime_architecture_detection_rejects_missing_native_target(monkeypatch):
    from flashinfer.prims_ts.batched_gemm import batched_gemm_run

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (10, 7))
    monkeypatch.setattr(
        batched_gemm_run, "is_prims_ts_arch_available", lambda major, minor: False
    )

    with pytest.raises(RuntimeError, match="cannot natively target SM107"):
        batched_gemm_run._resolve_runtime_sm_arch({"tile_n": 64})


@pytest.mark.parametrize(
    "arch_list, expected",
    [
        (["sm_80", "sm_100"], False),
        (["sm_80", "sm_100", "sm_107"], True),
        (["sm_80", "sm_107a"], True),
    ],
)
def test_pytorch_sm107_codegen_detection(monkeypatch, arch_list, expected):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
        _torch_supports_cuda_arch,
    )

    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: arch_list)
    assert _torch_supports_cuda_arch(10, 7) is expected


def test_nvfp4_mmak128_requires_rubin():
    validate_config(_fp4_config(mma_k=128))

    with pytest.raises(ValueError, match="requires sm_arch=107"):
        validate_config(_fp4_config(sm_arch=100, mma_k=128))


def test_mmak128_scale_factor_tmem_grouping():
    assert tmem_sf_vec4_groups_per_mma(64, 16) == 1
    assert tmem_sf_vec4_groups_per_mma(128, 16) == 2
    assert tmem_sf_col_stride(128, 64, 16) == 4
    assert tmem_sf_col_stride(128, 128, 16) == 8

    # Two packed vec4 words share each K128 MMA base, so the complete stage
    # footprint is unchanged from K64 for the same tile shape.
    assert tmem_sf_cols(128, 256, 16, 64) == 16
    assert tmem_sf_cols(128, 256, 16, 128) == 16
    assert tmem_sf_cols(256, 256, 16, 128) == 32


def test_rubin_tmem_footprint_uses_exclusive_allocation():
    cfg = _fp4_config(
        batch_mode=int(BatchMode.BATCH_N),
        transpose_mma_output=1,
        route_act=0,
        umma_cta_count=2,
        tile_n=256,
        mma_m=256,
        mma_n=256,
        epi_tile_n=64,
        sf_layout_b=int(SfLayout.R128c4),
        tile_scheduler=int(TileScheduler.PERSISTENT),
        num_stages_smem_sfa=6,
        num_stages_smem_sfb=7,
        num_stages_tmem_sfa=6,
        num_stages_tmem_sfb=7,
        use_shuffled_matrix=1,
        act_kind=int(ActKind.RELU2),
    )
    validate_config(cfg)
    assert cfg.tmem_required_cols == 576
    assert cfg.tmem_alloc_cols == 576

    cfg_544 = replace(cfg, num_stages_tmem_sfb=6)
    validate_config(cfg_544)
    assert cfg_544.tmem_required_cols == 544
    assert cfg_544.tmem_alloc_cols == 544

    with pytest.raises(ValueError, match="exceeds the 512-column"):
        validate_config(replace(cfg, sm_arch=100))


def test_rubin_large_smem_budget_is_used_by_task_manager():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_kernel import (
        build_batched_gemm_task_manager,
    )

    overrides = dict(
        sm_arch=107,
        batch_mode=int(BatchMode.BATCH_M),
        transpose_mma_output=0,
        route_act=1,
        route_sfs_act=int(RouteImpl.AUTO),
        sf_layout_b=int(SfLayout.R8c4),
        umma_cta_count=1,
        tile_n=8,
        tile_k=512,
        mma_m=128,
        mma_n=8,
        mma_k=64,
        epi_tile_n=8,
        dtype_a=int(DType.E2M1),
        dtype_b=int(DType.E2M1),
        dtype_c=int(DType.BF16),
        act_kind=0,
        tile_scheduler=int(TileScheduler.STATIC),
        num_stages_a=7,
        num_stages_b=7,
        num_stages_smem_sfa=7,
        num_stages_smem_sfb=7,
        num_stages_tmem_acc=1,
        num_stages_tmem_sfa=7,
        num_stages_tmem_sfb=7,
        num_stages_c_smem=1,
        use_unroll_loop_2x_for_mma=0,
        use_max_tmem_overlap=0,
    )
    manager = build_batched_gemm_task_manager(verbose=False, **overrides)
    total_smem = (
        manager._smem_allocator.total_smem_bytes
        + manager._smem_allocator.barrier_smem_bytes
    )
    assert 256 * 1024 < total_smem <= 327 * 1024

    with pytest.raises(ValueError, match="exceeds hardware capacity"):
        build_batched_gemm_task_manager(verbose=False, **{**overrides, "sm_arch": 100})


@cute.kernel
def _rubin_tmem_high_column_round_trip(output: cute.Pointer):
    """Write/read column 544 inside an exclusive 576-column allocation."""
    holding = cute.arch.alloc_smem(Int32, 1, 16)
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    if warp_idx == 0:
        cute.arch.alloc_tmem(576, holding, is_two_cta=False, arch="sm_107")
        cute.arch.relinquish_tmem_alloc_permit(is_two_cta=False)
    cute.arch.barrier()

    tmem = cute.arch.retrieve_tmem_ptr(Int32, 16, holding)
    high_col = prims.make_tmem_ptr(Int32(holding[0]) + Int32(544), Int32)
    tidx, _, _ = cute.arch.thread_idx()
    prims.tcgen05_st("32x32b", high_col, tidx + Int32(1234))
    cute.arch.fence_view_async_tmem_store()
    prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
    loaded = prims.tcgen05_ld("32x32b", high_col)
    cute.arch.fence_view_async_tmem_load()
    prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
    output[tidx] = loaded
    cute.arch.barrier()
    if warp_idx == 0:
        cute.arch.dealloc_tmem(tmem, 576, is_two_cta=False, arch="sm_107")


@cute.jit
def _launch_rubin_tmem_high_column_round_trip(output: cute.Pointer):
    _rubin_tmem_high_column_round_trip(output).launch(grid=(1, 1, 1), block=(32, 1, 1))


@pytest.mark.arch_rubin
def test_rubin_tmem_576_high_column_round_trip():
    output = torch.zeros(32, dtype=torch.int32, device="cuda")
    output_ptr = make_ptr(
        Int32,
        output.data_ptr(),
        cutlass.AddressSpace.gmem,
        assumed_align=16,
    )
    _launch_rubin_tmem_high_column_round_trip(output_ptr)
    torch.cuda.synchronize()
    torch.testing.assert_close(output.cpu(), torch.arange(32, dtype=torch.int32) + 1234)


RUBIN_FP4 = dict(
    sm_arch=107,
    batch_mode=int(BatchMode.BATCH_N),
    transpose_mma_output=1,
    route_act=0,
    route_sfs_act=int(RouteImpl.AUTO),
    umma_cta_count=2,
    tile_k=256,
    mma_m=256,
    dtype_a=int(DType.E2M1),
    dtype_b=int(DType.E2M1),
    dtype_c=int(DType.BF16),
    act_kind=0,
    epilogue_regs=128,
    mma_regs=48,
    load_regs=48,
    load_sf_regs=48,
    copy_sf_regs=48,
    workid_regs=48,
    padding_regs=48,
    use_unroll_loop_2x_for_mma=0,
    tile_scheduler=int(TileScheduler.PERSISTENT),
    num_stages_a=5,
    num_stages_b=5,
    num_stages_smem_sfa=5,
    num_stages_smem_sfb=5,
    num_stages_tmem_sfa=5,
    num_stages_tmem_sfb=5,
    num_stages_c_smem=1,
)


RUBIN_ROUTE_ACT_FP4 = dict(
    sm_arch=107,
    umma_cta_count=2,
    tile_n=128,
    mma_m=256,
    mma_n=128,
    mma_k=64,
    epi_tile_n=64,
    dtype_a=int(DType.E2M1),
    dtype_b=int(DType.E2M1),
    dtype_c=int(DType.BF16),
    sf_layout_a=int(SfLayout.R128c4),
    sf_layout_b=int(SfLayout.R128c4),
    act_kind=0,
    tile_scheduler=int(TileScheduler.PERSISTENT),
    num_stages_a=4,
    num_stages_b=4,
    num_stages_smem_sfa=4,
    num_stages_smem_sfb=4,
    num_stages_tmem_acc=2,
    num_stages_tmem_sfa=4,
    num_stages_tmem_sfb=4,
    num_stages_c_smem=1,
    epilogue_regs=128,
    mma_regs=48,
    load_regs=48,
    load_sf_regs=48,
    copy_sf_regs=48,
    workid_regs=48,
    padding_regs=48,
    use_unroll_loop_2x_for_mma=0,
    use_max_tmem_overlap=0,
)


RUBIN_ROUTE_ACT_CASES = (
    pytest.param(
        BatchMode.BATCH_M,
        RouteImpl.TMA,
        RouteImpl.NONE,
        0,
        0,
        256,
        id="batch-m-act-tma-sf-preexpanded",
    ),
    pytest.param(
        BatchMode.BATCH_N,
        RouteImpl.TMA,
        RouteImpl.NONE,
        0,
        0,
        256,
        id="batch-n-act-tma-sf-preexpanded",
    ),
    pytest.param(
        BatchMode.BATCH_M,
        RouteImpl.LDGSTS,
        RouteImpl.LDGSTS,
        0,
        0,
        256,
        id="batch-m-act-ldgsts-sf-ldgsts-separate-copy",
    ),
    pytest.param(
        BatchMode.BATCH_M,
        RouteImpl.TMA,
        RouteImpl.LDGSTS,
        1,
        0,
        256,
        id="batch-m-act-tma-sf-ldgsts-fused-copy",
    ),
    pytest.param(
        BatchMode.BATCH_M,
        RouteImpl.LDGSTS,
        RouteImpl.TMA,
        0,
        0,
        512,
        id="batch-m-act-ldgsts-sf-tma-separate-copy",
    ),
    pytest.param(
        BatchMode.BATCH_M,
        RouteImpl.TMA,
        RouteImpl.TMA,
        0,
        0,
        512,
        id="batch-m-act-tma-sf-tma-separate-copy",
    ),
    pytest.param(
        BatchMode.BATCH_N,
        RouteImpl.LDGSTS,
        RouteImpl.LDGSTS,
        0,
        0,
        256,
        id="batch-n-act-ldgsts-sf-ldgsts-separate-copy",
    ),
    pytest.param(
        BatchMode.BATCH_N,
        RouteImpl.TMA,
        RouteImpl.LDGSTS,
        1,
        0,
        256,
        id="batch-n-act-tma-sf-ldgsts-fused-copy",
    ),
    pytest.param(
        BatchMode.BATCH_N,
        RouteImpl.LDGSTS,
        RouteImpl.TMA,
        0,
        0,
        512,
        id="batch-n-act-ldgsts-sf-tma-separate-copy",
    ),
    pytest.param(
        BatchMode.BATCH_N,
        RouteImpl.TMA,
        RouteImpl.TMA,
        0,
        0,
        512,
        id="batch-n-act-tma-sf-tma-separate-copy",
    ),
    pytest.param(
        BatchMode.BATCH_N,
        RouteImpl.LDGSTS,
        RouteImpl.LDGSTS,
        1,
        1,
        256,
        id="batch-n-fused-operand-sf-loads",
    ),
)


@pytest.mark.parametrize("wide_cga", [False, True])
def test_rubin_routed_sfa_copy_barrier_census(wide_cga):
    from cutlass.experimental.task_scheduling.enums import SignalingThreads
    from flashinfer.prims_ts.batched_gemm.batched_gemm_kernel import (
        _build_schedule_validate,
        _make_pipeline_configs,
    )

    cfg = make_config(
        **RUBIN_ROUTE_ACT_FP4,
        tile_k=512,
        batch_mode=int(BatchMode.BATCH_M),
        transpose_mma_output=0,
        route_act=int(RouteImpl.TMA),
        route_sfs_act=int(RouteImpl.TMA),
        cluster_dim_x=4 if wide_cga else 2,
        cluster_dim_y=2 if wide_cga else 1,
    )
    compute_warp_layout(cfg)
    tasks, *_ = _build_schedule_validate(cfg)
    copy = next(task for task in tasks if task.name == "CopySfATask")
    assert copy.run_only_on_cta_id is None
    pcfgs = _make_pipeline_configs(cfg)
    assert pcfgs["tmem_sfa"].producer_group.size == 256
    assert pcfgs["tmem_sfa"].producer_signaling_threads == SignalingThreads.All
    if not wide_cga:
        # Every task except MMA and WorkId executes in both CTAs.
        expected = sum(
            task.num_warps * 32 * (1 if task.run_only_on_cta_id == 0 else 2)
            for task in tasks
        )
        assert pcfgs["workid"].consumer_group.size == expected


@pytest.mark.parametrize("dim_x,dim_y", [(2, 1), (4, 2), (4, 4)])
def test_rubin_fused_load_clc_barrier_census(dim_x, dim_y):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_cga import (
        CgaPolicy,
        host_clc_consumer_threads,
    )
    from flashinfer.prims_ts.batched_gemm.batched_gemm_kernel import (
        _build_schedule_validate,
    )

    cfg = make_config(
        **RUBIN_ROUTE_ACT_FP4,
        tile_k=256,
        batch_mode=int(BatchMode.BATCH_N),
        transpose_mma_output=1,
        route_act=int(RouteImpl.LDGSTS),
        route_sfs_act=int(RouteImpl.LDGSTS),
        fuse_sf_copy_to_mma=1,
        fuse_operand_sf_loads=1,
        num_load_sfb_warps=4,
        cluster_dim_x=dim_x,
        cluster_dim_y=dim_y,
    )
    compute_warp_layout(cfg)
    # Four epilogue, four gather and one padding warp per CTA. Fused
    # A/SFA multicast runs in one Y row, MMA in each atom leader, WorkId once.
    assert host_clc_consumer_threads(cfg, dim_x, dim_y) == 32 * (
        9 * dim_x * dim_y + dim_x + dim_x * dim_y // 2 + 1
    )
    tasks, *_ = _build_schedule_validate(cfg)
    load = next(task for task in tasks if task.name == "FusedLoadASfATask")
    expected_policy = CgaPolicy.MCAST_A if dim_y > 1 else CgaPolicy.ALL
    assert getattr(load, "cga_policy", int(CgaPolicy.ALL)) == int(expected_policy)


@pytest.mark.arch_rubin
@pytest.mark.timeout(180)
@pytest.mark.parametrize("tile_n", [16, 256])
@pytest.mark.parametrize("unit_sf", [False, True])
def test_rubin_clustered_gather_pipeline_wrap(tile_n, unit_sf, monkeypatch):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import reference_check

    monkeypatch.setenv("BATCHED_GEMM_TS_RANDOMIZE_SF", "none" if unit_sf else "all")
    cfg = {
        **RUBIN_ROUTE_ACT_FP4,
        "tile_n": tile_n,
        "mma_n": tile_n,
        "epi_tile_n": min(tile_n, 64),
        "sf_layout_b": int(SfLayout.R8c4 if tile_n == 16 else SfLayout.R128c4),
        "use_max_tmem_overlap": int(tile_n == 256),
        "fuse_sf_copy_to_mma": int(tile_n == 256),
        "num_stages_tmem_acc": 1 if tile_n == 256 else 2,
    }
    assert reference_check(
        num_experts=2,
        num_tokens=512,
        top_k=1,
        problem_n=512,
        problem_k=2048,
        tile_k=256,
        batch_mode=int(BatchMode.BATCH_M),
        transpose_mma_output=0,
        route_act=int(RouteImpl.LDGSTS),
        route_sfs_act=int(RouteImpl.LDGSTS),
        **cfg,
    )


@pytest.mark.arch_rubin
@pytest.mark.timeout(300)
@pytest.mark.parametrize(
    (
        "batch_mode",
        "route_act",
        "route_sfs_act",
        "fuse_sf_copy_to_mma",
        "fuse_operand_sf_loads",
        "tile_k",
    ),
    RUBIN_ROUTE_ACT_CASES,
)
def test_rubin_route_act_mixed_large_cga_modes(
    batch_mode,
    route_act,
    route_sfs_act,
    fuse_sf_copy_to_mma,
    fuse_operand_sf_loads,
    tile_k,
):
    """Numerically cover each valid routed transport family under 4x2/2x1 CGAs."""
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import reference_check

    is_batch_n = batch_mode == BatchMode.BATCH_N
    cfg = {**RUBIN_ROUTE_ACT_FP4, "tile_k": tile_k}
    result = reference_check(
        num_experts=2,
        num_tokens=512,
        top_k=1,
        problem_n=8192 if is_batch_n else 4096,
        # Four or more K tiles exercise steady-state barriers and, for fused
        # B/SFB LDGSTS, the delayed cp.async commit/drain window.
        problem_k=max(1024, tile_k * 4),
        batch_mode=int(batch_mode),
        transpose_mma_output=int(is_batch_n),
        route_act=int(route_act),
        route_sfs_act=int(route_sfs_act),
        fuse_sf_copy_to_mma=fuse_sf_copy_to_mma,
        fuse_operand_sf_loads=fuse_operand_sf_loads,
        cluster_dim_x=4,
        cluster_dim_y=2,
        fallback_cluster_dim_x=2,
        fallback_cluster_dim_y=1,
        num_load_sfb_warps=4 if fuse_operand_sf_loads else 0,
        require_mixed_cga_runtime=True,
        **cfg,
    )
    assert result


@pytest.mark.arch_rubin
@pytest.mark.timeout(300)
@pytest.mark.parametrize(
    ("batch_mode", "fuse_operand_sf_loads"),
    (
        pytest.param(BatchMode.BATCH_M, 0, id="batch-m-unfused"),
        pytest.param(BatchMode.BATCH_N, 1, id="batch-n-fused-loads"),
    ),
)
def test_rubin_route_act_16cta_mixed_cga(batch_mode, fuse_operand_sf_loads):
    """Stress routed LDGSTS atom-leader synchronization in a 4x4 physical CGA."""
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import reference_check

    is_batch_n = batch_mode == BatchMode.BATCH_N
    cfg = {**RUBIN_ROUTE_ACT_FP4, "tile_k": 256}
    result = reference_check(
        num_experts=2,
        num_tokens=512,
        top_k=1,
        problem_n=8192 if is_batch_n else 4096,
        problem_k=1024,
        batch_mode=int(batch_mode),
        transpose_mma_output=int(is_batch_n),
        route_act=int(RouteImpl.LDGSTS),
        route_sfs_act=int(RouteImpl.LDGSTS),
        fuse_sf_copy_to_mma=fuse_operand_sf_loads,
        fuse_operand_sf_loads=fuse_operand_sf_loads,
        cluster_dim_x=4,
        cluster_dim_y=4,
        fallback_cluster_dim_x=2,
        fallback_cluster_dim_y=2,
        num_load_sfb_warps=4 if fuse_operand_sf_loads else 0,
        require_mixed_cga_runtime=True,
        **cfg,
    )
    assert result


@pytest.mark.arch_rubin
@pytest.mark.timeout(180)
@pytest.mark.parametrize("mma_k", [64, 128])
def test_rubin_fc2_raster_along_n_mixed_cga(mma_k):
    """Check N-raster decoding with both 4x2 and 2x1 clusters executing work."""
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import reference_check

    cfg = {
        **RUBIN_FP4,
        "batch_mode": int(BatchMode.BATCH_M),
        "transpose_mma_output": 0,
        "num_stages_a": 6,
        "num_stages_b": 6,
        "num_stages_smem_sfa": 6,
        "num_stages_smem_sfb": 6,
        "num_stages_tmem_sfa": 1,
        "num_stages_tmem_sfb": 1,
    }
    assert reference_check(
        num_experts=2,
        num_tokens=512,
        top_k=1,
        problem_n=8192,
        problem_k=2048,
        tile_n=256,
        mma_n=256,
        mma_k=mma_k,
        epi_tile_n=64,
        sf_layout_b=int(SfLayout.R128c4),
        num_stages_tmem_acc=1,
        use_max_tmem_overlap=1,
        fuse_sf_copy_to_mma=1,
        use_tma_store=1,
        cta_raster_order=int(CtaRasterOrder.ALONG_N),
        cluster_dim_x=4,
        cluster_dim_y=2,
        fallback_cluster_dim_x=2,
        fallback_cluster_dim_y=1,
        require_mixed_cga_runtime=True,
        **cfg,
    )


@pytest.mark.arch_rubin
@pytest.mark.timeout(180)
@pytest.mark.parametrize("mma_k", [64, 128])
@pytest.mark.parametrize("tile_n", [64, 128, 256])
def test_rubin_nvfp4_mmak_tile_n_scale_factor_layouts(mma_k, tile_n):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import reference_check

    tile256 = tile_n == 256
    result = reference_check(
        num_experts=2,
        num_tokens=256,
        top_k=1,
        tile_n=tile_n,
        mma_n=tile_n,
        mma_k=mma_k,
        epi_tile_n=64 if tile256 else tile_n,
        sf_layout_b=int(SfLayout.R8c4 if tile_n < 128 else SfLayout.R128c4),
        num_stages_tmem_acc=1 if tile256 else 2,
        use_max_tmem_overlap=int(tile256),
        fuse_sf_copy_to_mma=int(tile256),
        use_tma_store=int(tile256),
        **RUBIN_FP4,
    )
    assert result


@pytest.mark.arch_rubin
@pytest.mark.timeout(240)
@pytest.mark.parametrize("mma_k", [64, 128])
def test_rubin_nvfp4_tile256_overlap_uses_high_tmem_columns(mma_k):
    """Exercise both accumulator sets with fused scale factors above column 512."""
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import reference_check

    assert reference_check(
        num_experts=2,
        num_tokens=2048,
        top_k=1,
        problem_n=4096,
        problem_k=2048,
        tile_n=256,
        mma_n=256,
        mma_k=mma_k,
        epi_tile_n=32,
        sf_layout_b=int(SfLayout.R128c4),
        num_stages_tmem_acc=1,
        use_max_tmem_overlap=1,
        fuse_sf_copy_to_mma=1,
        use_tma_store=1,
        **RUBIN_FP4,
    )


@pytest.mark.arch_rubin
@pytest.mark.timeout(180)
def test_rubin_nvfp4_mmak128_four_experts():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import reference_check

    result = reference_check(
        num_experts=4,
        num_tokens=512,
        top_k=1,
        sm_arch=107,
        batch_mode=int(BatchMode.BATCH_M),
        transpose_mma_output=0,
        route_act=1,
        route_sfs_act=int(RouteImpl.AUTO),
        sf_layout_b=int(SfLayout.R8c4),
        umma_cta_count=1,
        tile_n=8,
        tile_k=256,
        mma_m=128,
        mma_n=8,
        mma_k=128,
        epi_tile_n=8,
        dtype_a=int(DType.E2M1),
        dtype_b=int(DType.E2M1),
        dtype_c=int(DType.BF16),
        act_kind=0,
        tile_scheduler=int(TileScheduler.STATIC),
        num_stages_tmem_acc=1,
        use_unroll_loop_2x_for_mma=0,
        use_max_tmem_overlap=0,
        **uniform_pipeline_stage_overrides(5),
    )
    assert result


@pytest.mark.arch_rubin
@pytest.mark.timeout(180)
def test_rubin_nvfp4_mmak128_tile_k512():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import reference_check

    result = reference_check(
        num_experts=2,
        num_tokens=256,
        top_k=1,
        sm_arch=107,
        batch_mode=int(BatchMode.BATCH_M),
        transpose_mma_output=0,
        route_act=1,
        route_sfs_act=int(RouteImpl.AUTO),
        sf_layout_b=int(SfLayout.R8c4),
        umma_cta_count=1,
        tile_n=8,
        tile_k=512,
        mma_m=128,
        mma_n=8,
        mma_k=128,
        epi_tile_n=8,
        dtype_a=int(DType.E2M1),
        dtype_b=int(DType.E2M1),
        dtype_c=int(DType.BF16),
        act_kind=0,
        tile_scheduler=int(TileScheduler.STATIC),
        num_stages_tmem_acc=1,
        use_unroll_loop_2x_for_mma=0,
        use_max_tmem_overlap=0,
        **uniform_pipeline_stage_overrides(4),
    )
    assert result


RUBIN_2X_MMAK_DTYPES = (
    pytest.param(DType.MXE4M3, DType.MXE4M3, id="mxfp8xmxfp8"),
    pytest.param(DType.MXE2M1, DType.MXE4M3, id="mxfp4xmxfp8"),
    pytest.param(DType.E4M3, DType.E4M3, id="fp8xfp8"),
)


@pytest.mark.arch_rubin
@pytest.mark.timeout(300)
@pytest.mark.parametrize(("dtype_a", "dtype_b"), RUBIN_2X_MMAK_DTYPES)
@pytest.mark.parametrize("mma_k", (32, 64))
@pytest.mark.parametrize("tile_k", (128, 256, 512))
@pytest.mark.parametrize("tile_n", (8, 256))
def test_rubin_2x_mmak_dtype_matrix(dtype_a, dtype_b, mma_k, tile_k, tile_n):
    """Cover K32 compatibility and Rubin K64 across dtype and K-box layouts."""
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import reference_check

    result = reference_check(
        num_experts=2,
        num_tokens=max(256, tile_n),
        top_k=1,
        problem_n=128,
        problem_k=tile_k,
        sm_arch=107,
        batch_mode=int(BatchMode.BATCH_N),
        transpose_mma_output=1,
        route_act=int(RouteImpl.NONE),
        route_sfs_act=int(RouteImpl.AUTO),
        dtype_a=int(dtype_a),
        dtype_b=int(dtype_b),
        dtype_c=int(DType.BF16),
        sf_layout_a=int(SfLayout.R128c4),
        sf_layout_b=int(SfLayout.R128c4 if tile_n == 256 else SfLayout.R8c4),
        umma_cta_count=1,
        tile_n=tile_n,
        tile_k=tile_k,
        mma_m=128,
        mma_n=tile_n,
        mma_k=mma_k,
        epi_tile_n=min(tile_n, 64),
        act_kind=int(ActKind.NONE),
        tile_scheduler=int(TileScheduler.STATIC),
        use_tma_oob_opt=1,
        use_unroll_loop_2x_for_mma=0,
        use_max_tmem_overlap=0,
        **uniform_pipeline_stage_overrides(1, tmem_acc_stages=1),
    )
    assert result


@pytest.mark.arch_rubin
@pytest.mark.timeout(300)
def test_rubin_mixed_mx_2x_mmak_2cta_fused_utccp():
    """Mirror trtllm-gen's 2-CTA K64 mixed-MX fused-UTCCP example."""
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import reference_check

    assert reference_check(
        num_experts=4,
        num_tokens=256,
        top_k=1,
        problem_n=128,
        problem_k=1024,
        sm_arch=107,
        batch_mode=int(BatchMode.BATCH_N),
        transpose_mma_output=1,
        route_act=int(RouteImpl.NONE),
        route_sfs_act=int(RouteImpl.AUTO),
        dtype_a=int(DType.MXE2M1),
        dtype_b=int(DType.MXE4M3),
        dtype_c=int(DType.BF16),
        sf_layout_a=int(SfLayout.R128c4),
        sf_layout_b=int(SfLayout.R128c4),
        umma_cta_count=2,
        tile_n=256,
        tile_k=256,
        mma_m=256,
        mma_n=256,
        mma_k=64,
        epi_tile_n=64,
        act_kind=int(ActKind.NONE),
        tile_scheduler=int(TileScheduler.PERSISTENT),
        fuse_sf_copy_to_mma=1,
        use_max_tmem_overlap=1,
        use_tma_store=1,
        use_tma_oob_opt=1,
        use_unroll_loop_2x_for_mma=0,
        epilogue_regs=144,
        mma_regs=96,
        load_regs=96,
        load_sf_regs=96,
        copy_sf_regs=96,
        workid_regs=96,
        padding_regs=96,
        **uniform_pipeline_stage_overrides(3, tmem_acc_stages=1),
    )


@pytest.mark.arch_rubin
@pytest.mark.timeout(180)
def test_rubin_large_smem_descriptor_runtime():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import reference_check

    result = reference_check(
        num_experts=2,
        num_tokens=256,
        top_k=1,
        sm_arch=107,
        batch_mode=int(BatchMode.BATCH_M),
        transpose_mma_output=0,
        route_act=1,
        route_sfs_act=int(RouteImpl.AUTO),
        sf_layout_b=int(SfLayout.R8c4),
        umma_cta_count=1,
        tile_n=8,
        tile_k=512,
        mma_m=128,
        mma_n=8,
        mma_k=64,
        epi_tile_n=8,
        dtype_a=int(DType.E2M1),
        dtype_b=int(DType.E2M1),
        dtype_c=int(DType.BF16),
        act_kind=0,
        tile_scheduler=int(TileScheduler.STATIC),
        num_stages_tmem_acc=1,
        use_unroll_loop_2x_for_mma=0,
        use_max_tmem_overlap=0,
        **uniform_pipeline_stage_overrides(7),
    )
    assert result


@pytest.mark.arch_rubin
@pytest.mark.timeout(180)
@pytest.mark.parametrize("batch_mode", [BatchMode.BATCH_M, BatchMode.BATCH_N])
@pytest.mark.parametrize("route_act", [RouteImpl.TMA, RouteImpl.LDGSTS])
def test_rubin_explicit_routed_loader_width(batch_mode, route_act):
    """Explicit loader width must agree with copy partitioning and arrivals."""
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import reference_check

    is_batch_n = batch_mode == BatchMode.BATCH_N
    operand = "b" if is_batch_n else "a"
    assert reference_check(
        num_experts=2,
        num_tokens=512,
        top_k=1,
        problem_n=512,
        problem_k=2048,
        batch_mode=int(batch_mode),
        transpose_mma_output=int(is_batch_n),
        route_act=int(route_act),
        route_sfs_act=int(RouteImpl.LDGSTS),
        tile_k=256,
        **{f"num_load_{operand}_warps": 8},
        **RUBIN_ROUTE_ACT_FP4,
    )


def _run_rubin_config(
    cfg,
    *,
    num_tokens=512,
    problem_n=512,
    k_tiles=4,
    problem_k=None,
    require_mixed_cga_runtime=False,
):
    """Run a simplified SM107 tactic against the numerical reference."""
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import reference_check

    if problem_k is None:
        problem_k = k_tiles * cfg["tile_k"]
    assert reference_check(
        num_experts=2,
        num_tokens=num_tokens,
        top_k=1,
        problem_n=problem_n,
        problem_k=problem_k,
        require_mixed_cga_runtime=require_mixed_cga_runtime,
        seed=123,
        **cfg,
    )


def _assert_unroll_engaged(cfg, problem_k):
    resolved = resolve_problem_config(make_config(**cfg), problem_k)
    assert resolved.use_unroll_loop_2x_for_mma == 1


RUBIN_NVFP4_FC1 = {
    "sm_arch": 107,
    "batch_mode": int(BatchMode.BATCH_N),
    "transpose_mma_output": 1,
    "dtype_a": int(DType.E2M1),
    "dtype_b": int(DType.E2M1),
    "dtype_c": int(DType.E2M1),
    "sf_layout_a": int(SfLayout.R128c4),
    "sf_layout_b": int(SfLayout.LINEAR),
    "sf_layout_c": int(SfLayout.R8c4),
    "mma_k": 128,
    "act_kind": int(ActKind.SWIGLU),
    "bias_type": int(BiasType.M),
    "use_global_scales": 1,
    "use_tma_store": 1,
    "tile_scheduler": int(TileScheduler.PERSISTENT),
    "num_stages_tmem_acc": 2,
    **uniform_pipeline_stage_overrides(4),
}


RUBIN_MX_FC1 = {
    "sm_arch": 107,
    "batch_mode": int(BatchMode.BATCH_N),
    "transpose_mma_output": 1,
    "dtype_a": int(DType.MXE2M1),
    "dtype_b": int(DType.MXE4M3),
    "dtype_c": int(DType.MXE4M3),
    "sf_layout_a": int(SfLayout.R128c4),
    "sf_layout_b": int(SfLayout.LINEAR),
    "sf_layout_c": int(SfLayout.R8c4),
    "mma_k": 64,
    "act_kind": int(ActKind.SWIGLU),
    "bias_type": int(BiasType.M),
    "use_tma_store": 1,
    "tile_scheduler": int(TileScheduler.PERSISTENT),
    "num_stages_tmem_acc": 2,
    **uniform_pipeline_stage_overrides(4),
}


RUBIN_NVFP4_ROUTED_FC1_CASES = (
    pytest.param(
        {
            "route_act": int(RouteImpl.LDGSTS),
            "route_sfs_act": int(RouteImpl.LDGSTS),
            "tile_n": 8,
            "tile_k": 512,
            "mma_m": 128,
            "mma_n": 8,
            "epi_tile_n": 8,
            "umma_cta_count": 1,
        },
        id="ll-n8-ldgsts-k512-swiglu",
    ),
    pytest.param(
        {
            "route_act": int(RouteImpl.TMA),
            "route_sfs_act": int(RouteImpl.AUTO),
            "tile_n": 64,
            "tile_k": 512,
            "mma_m": 256,
            "mma_n": 64,
            "epi_tile_n": 32,
            "umma_cta_count": 2,
            "cluster_dim_x": 2,
            "num_load_b_warps": 4,
            "num_load_sfb_warps": 8,
        },
        id="ht-n64-tma-auto-k512-swiglu-sfb8",
    ),
    pytest.param(
        {
            "route_act": int(RouteImpl.LDGSTS),
            "route_sfs_act": int(RouteImpl.LDGSTS),
            "tile_n": 128,
            "tile_k": 256,
            "mma_m": 256,
            "mma_n": 128,
            "epi_tile_n": 64,
            "umma_cta_count": 2,
            "cluster_dim_x": 2,
            "fuse_sf_copy_to_mma": 1,
            "fuse_operand_sf_loads": 1,
            "num_load_b_warps": 4,
            "num_load_sfb_warps": 4,
            "num_stages_tmem_sfa": 1,
            "num_stages_tmem_sfb": 1,
        },
        id="ht-n128-ldgsts-fusedsf-k256-swiglu",
    ),
    pytest.param(
        {
            "route_act": int(RouteImpl.TMA),
            "route_sfs_act": int(RouteImpl.LDGSTS),
            "tile_n": 256,
            "tile_k": 256,
            "mma_m": 256,
            "mma_n": 256,
            "epi_tile_n": 64,
            "umma_cta_count": 2,
            "cluster_dim_x": 2,
            "act_kind": int(ActKind.RELU2),
            "fuse_sf_copy_to_mma": 1,
            "use_max_tmem_overlap": 1,
            "num_stages_tmem_acc": 1,
            "num_stages_tmem_sfa": 1,
            "num_stages_tmem_sfb": 1,
            "num_load_b_warps": 8,
            "num_load_sfb_warps": 4,
            "epilogue_regs": 128,
        },
        id="ht-n256-tma-overlap-k256-relu2",
    ),
)


@pytest.mark.arch_rubin
@pytest.mark.timeout(300)
@pytest.mark.parametrize("case", RUBIN_NVFP4_ROUTED_FC1_CASES)
def test_rubin_nvfp4_routed_fc1_batch_n_mmak128(case):
    _run_rubin_config({**RUBIN_NVFP4_FC1, **case})


RUBIN_MX_ROUTED_FC1_CASES = (
    pytest.param(
        {
            "route_act": int(RouteImpl.LDGSTS),
            "route_sfs_act": int(RouteImpl.LDGSTS),
            "tile_n": 8,
            "tile_k": 512,
            "mma_m": 128,
            "mma_n": 8,
            "epi_tile_n": 8,
            "umma_cta_count": 1,
        },
        id="mxfp4-ll-n8-ldgsts-k512-swiglu",
    ),
    pytest.param(
        {
            "route_act": int(RouteImpl.TMA),
            "route_sfs_act": int(RouteImpl.LDGSTS),
            "tile_n": 64,
            "tile_k": 256,
            "mma_m": 256,
            "mma_n": 64,
            "epi_tile_n": 64,
            "umma_cta_count": 2,
            "cluster_dim_x": 2,
        },
        id="mxfp4-ht-n64-tma-k256-swiglu",
    ),
    pytest.param(
        {
            "route_act": int(RouteImpl.LDGSTS),
            "route_sfs_act": int(RouteImpl.LDGSTS),
            "tile_n": 128,
            "tile_k": 256,
            "mma_m": 256,
            "mma_n": 128,
            "epi_tile_n": 32,
            "umma_cta_count": 2,
            "cluster_dim_x": 2,
            "sf_layout_c": int(SfLayout.R128c4),
            "fuse_sf_copy_to_mma": 1,
            "fuse_operand_sf_loads": 1,
            "num_load_b_warps": 4,
            "num_load_sfb_warps": 4,
            "num_stages_tmem_sfa": 1,
            "num_stages_tmem_sfb": 1,
        },
        id="mxfp4-ht-n128-ldgsts-fusedsf-k256-swiglu",
    ),
    pytest.param(
        {
            "dtype_a": int(DType.MXE4M3),
            "route_act": int(RouteImpl.LDGSTS),
            "route_sfs_act": int(RouteImpl.LDGSTS),
            "tile_n": 128,
            "tile_k": 256,
            "mma_m": 256,
            "mma_n": 128,
            "epi_tile_n": 32,
            "umma_cta_count": 2,
            "cluster_dim_x": 2,
            "act_kind": int(ActKind.RELU2),
            "sf_layout_c": int(SfLayout.R128c4),
            "fuse_sf_copy_to_mma": 1,
            "fuse_operand_sf_loads": 1,
            "num_load_b_warps": 4,
            "num_load_sfb_warps": 4,
            "num_stages_tmem_sfa": 1,
            "num_stages_tmem_sfb": 1,
        },
        id="mxfp8-ht-n128-ldgsts-fusedsf-k256-relu2",
    ),
    pytest.param(
        {
            "dtype_a": int(DType.MXE4M3),
            "route_act": int(RouteImpl.TMA),
            "route_sfs_act": int(RouteImpl.LDGSTS),
            "tile_n": 256,
            "tile_k": 256,
            "mma_m": 256,
            "mma_n": 256,
            "epi_tile_n": 64,
            "umma_cta_count": 2,
            "cluster_dim_x": 2,
            "fuse_sf_copy_to_mma": 1,
            "use_max_tmem_overlap": 1,
            "num_stages_tmem_acc": 1,
            "num_stages_tmem_sfa": 1,
            "num_stages_tmem_sfb": 1,
            "num_load_b_warps": 8,
            "num_load_sfb_warps": 4,
        },
        id="mxfp8-ht-n256-tma-overlap-k256-swiglu",
    ),
)


@pytest.mark.arch_rubin
@pytest.mark.timeout(300)
@pytest.mark.parametrize("case", RUBIN_MX_ROUTED_FC1_CASES)
def test_rubin_mx_routed_fc1_batch_n_mmak64(case):
    _run_rubin_config({**RUBIN_MX_FC1, **case})


RUBIN_NVFP4_FC2 = {
    "sm_arch": 107,
    "batch_mode": int(BatchMode.BATCH_N),
    "transpose_mma_output": 1,
    "route_act": int(RouteImpl.NONE),
    "route_sfs_act": int(RouteImpl.AUTO),
    "dtype_a": int(DType.E2M1),
    "dtype_b": int(DType.E2M1),
    "dtype_c": int(DType.BF16),
    "sf_layout_a": int(SfLayout.R128c4),
    "sf_layout_b": int(SfLayout.R128c4),
    "mma_k": 128,
    "act_kind": int(ActKind.NONE),
    "bias_type": int(BiasType.M),
    "use_global_scales": 1,
    "use_tma_store": 1,
    "tile_scheduler": int(TileScheduler.PERSISTENT),
    "num_stages_tmem_acc": 2,
    **uniform_pipeline_stage_overrides(4),
}


RUBIN_MX_FC2 = {
    "sm_arch": 107,
    "batch_mode": int(BatchMode.BATCH_N),
    "transpose_mma_output": 1,
    "route_act": int(RouteImpl.NONE),
    "route_sfs_act": int(RouteImpl.AUTO),
    "dtype_a": int(DType.MXE2M1),
    "dtype_b": int(DType.MXE4M3),
    "dtype_c": int(DType.BF16),
    "sf_layout_a": int(SfLayout.R128c4),
    "sf_layout_b": int(SfLayout.R128c4),
    "mma_k": 64,
    "act_kind": int(ActKind.NONE),
    "bias_type": int(BiasType.M),
    "use_tma_store": 1,
    "tile_scheduler": int(TileScheduler.PERSISTENT),
    "num_stages_tmem_acc": 2,
    **uniform_pipeline_stage_overrides(4),
}


RUBIN_CLUSTER4_CASES = (
    pytest.param(
        {
            **RUBIN_NVFP4_FC1,
            "route_act": int(RouteImpl.TMA),
            "route_sfs_act": int(RouteImpl.LDGSTS),
            "tile_n": 256,
            "tile_k": 256,
            "mma_m": 256,
            "mma_n": 256,
            "epi_tile_n": 32,
            "umma_cta_count": 2,
            "cluster_dim_x": 4,
            "cluster_dim_y": 1,
            "fallback_cluster_dim_x": 2,
            "fallback_cluster_dim_y": 1,
            "act_kind": int(ActKind.RELU2),
            "fuse_sf_copy_to_mma": 1,
            "use_max_tmem_overlap": 1,
            "num_stages_a": 4,
            "num_stages_b": 4,
            "num_stages_smem_sfa": 13,
            "num_stages_smem_sfb": 9,
            "num_stages_tmem_sfa": 1,
            "num_stages_tmem_sfb": 1,
            "num_stages_tmem_acc": 1,
            "num_stages_c_smem": 1,
            "num_load_b_warps": 8,
            "num_load_sfb_warps": 4,
            "epilogue_regs": 96,
        },
        id="nvfp4-fc1-n256-tma-overlap-relu2",
    ),
    pytest.param(
        {
            **RUBIN_MX_FC2,
            "tile_n": 256,
            "tile_k": 128,
            "mma_m": 256,
            "mma_n": 256,
            "epi_tile_n": 64,
            "umma_cta_count": 2,
            "cluster_dim_x": 4,
            "cluster_dim_y": 1,
            "fallback_cluster_dim_x": 2,
            "fallback_cluster_dim_y": 1,
            "fuse_sf_copy_to_mma": 1,
            "use_max_tmem_overlap": 1,
            "num_stages_a": 8,
            "num_stages_b": 8,
            "num_stages_smem_sfa": 8,
            "num_stages_smem_sfb": 9,
            "num_stages_tmem_sfa": 1,
            "num_stages_tmem_sfb": 1,
            "num_stages_tmem_acc": 1,
        },
        id="mxfp4-mxfp8-fc2-n256-k128-overlap",
    ),
)


@pytest.mark.parametrize("case", RUBIN_CLUSTER4_CASES)
def test_rubin_cluster4x1_fallback2x1_schedule(case):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_kernel import (
        _build_schedule_validate,
    )

    cfg = make_config(**case)
    compute_warp_layout(cfg)
    assert cfg.preferred_cluster_shape == (4, 1, 1)
    assert cfg.fallback_cluster_shape == (2, 1, 1)
    _build_schedule_validate(cfg)


@pytest.mark.arch_rubin
@pytest.mark.timeout(300)
@pytest.mark.parametrize("case", RUBIN_CLUSTER4_CASES)
def test_rubin_cluster4x1_fallback2x1_mixed_cga(case):
    _run_rubin_config(
        case,
        problem_n=16384,
        problem_k=2048,
        require_mixed_cga_runtime=True,
    )


def _rubin_mx_tile_k1024_config(*, mma_k=64, tile_n=32, umma_cta_count=2):
    return {
        **RUBIN_MX_FC1,
        "mma_k": mma_k,
        "route_act": int(RouteImpl.LDGSTS),
        "route_sfs_act": int(RouteImpl.LDGSTS),
        "tile_n": tile_n,
        "tile_k": 1024,
        "mma_m": 128 * umma_cta_count,
        "mma_n": tile_n,
        "epi_tile_n": tile_n,
        "umma_cta_count": umma_cta_count,
        "cluster_dim_x": umma_cta_count,
        "num_stages_a": 3,
        "num_stages_b": 3,
        "num_stages_smem_sfa": 4,
        "num_stages_smem_sfb": 2,
        "num_stages_tmem_sfa": 5,
        "num_stages_tmem_sfb": 5,
        "num_stages_tmem_acc": 2,
        "num_stages_c_smem": 2,
    }


def test_rubin_mx_tile_k1024_uses_rubin_smem():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_kernel import (
        build_batched_gemm_task_manager,
    )

    cfg = _rubin_mx_tile_k1024_config()
    manager = build_batched_gemm_task_manager(
        num_experts=2,
        num_tokens=512,
        top_k=1,
        verbose=False,
        **cfg,
    )
    total_smem = (
        manager._smem_allocator.total_smem_bytes
        + manager._smem_allocator.barrier_smem_bytes
    )
    assert 227 * 1024 < total_smem <= 327 * 1024

    # K64 is Rubin-only, so validate the same physical allocation against the
    # SM100 budget directly rather than changing to K32 (which expands MXFP4
    # in SMEM and no longer represents the shipped K64 tactic).
    manager._smem_capacity_bytes = 227 * 1024
    with pytest.raises(ValueError, match="exceeds hardware capacity"):
        manager._verify_smem_capacity()


@pytest.mark.arch_rubin
@pytest.mark.timeout(300)
def test_rubin_mx_fc1_tile_k1024():
    _run_rubin_config(_rubin_mx_tile_k1024_config(), problem_k=4096)


RUBIN_UNROLL2X_CASES = (
    pytest.param(
        {
            **RUBIN_NVFP4_FC2,
            "tile_n": 32,
            "tile_k": 256,
            "mma_m": 128,
            "mma_n": 32,
            "epi_tile_n": 32,
            "umma_cta_count": 1,
            "sf_layout_b": int(SfLayout.R8c4),
            "use_unroll_loop_2x_for_mma": 1,
        },
        id="nvfp4-fc2-ll-n32-k256",
    ),
    pytest.param(
        {
            **RUBIN_NVFP4_FC1,
            "route_act": int(RouteImpl.LDGSTS),
            "route_sfs_act": int(RouteImpl.LDGSTS),
            "tile_n": 16,
            "tile_k": 512,
            "mma_m": 128,
            "mma_n": 16,
            "epi_tile_n": 16,
            "umma_cta_count": 1,
            "use_unroll_loop_2x_for_mma": 1,
        },
        id="nvfp4-fc1-ll-n16-ldgsts-k512-swiglu",
    ),
    pytest.param(
        {
            **RUBIN_MX_FC1,
            "route_act": int(RouteImpl.TMA),
            "route_sfs_act": int(RouteImpl.LDGSTS),
            "tile_n": 192,
            "tile_k": 256,
            "mma_m": 256,
            "mma_n": 192,
            "epi_tile_n": 32,
            "umma_cta_count": 2,
            "cluster_dim_x": 2,
            "num_load_b_warps": 8,
            "num_load_sfb_warps": 4,
            "use_unroll_loop_2x_for_mma": 1,
        },
        id="mxfp4-fc1-ht-n192-tma-k256-swiglu",
    ),
)


@pytest.mark.arch_rubin
@pytest.mark.timeout(300)
@pytest.mark.parametrize("case", RUBIN_UNROLL2X_CASES)
def test_rubin_unroll2x(case):
    problem_k = 4 * case["tile_k"]
    _assert_unroll_engaged(case, problem_k)
    _run_rubin_config(case, problem_k=problem_k)
