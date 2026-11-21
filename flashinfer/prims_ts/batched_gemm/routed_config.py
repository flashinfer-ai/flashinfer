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

"""Routed NVFP4 configurations for Rubin."""

from .config_types import (
    ActKind,
    BatchMode,
    CtaRasterOrder,
    DType,
    RouteImpl,
    SfLayout,
    TileScheduler,
)


def routed_tile256_options(
    *,
    scheduler="static",
    rolled_mma=0,
    weight_pipeline="auto",
    work_throttle=None,
    max_active_clusters=0,
):
    """Return explicit options; the caller supplies the resident cluster budget.

    Static/shared weights is the measured E256 preset. CLC/separate weights
    with work throttling is the E384 preset. Both multicast partitioned SFB.
    """
    if scheduler not in ("static", "clc"):
        raise ValueError("scheduler must be static or clc")
    if weight_pipeline == "auto":
        weight_pipeline = "shared" if scheduler == "static" else "separate"
    if weight_pipeline not in ("shared", "separate"):
        raise ValueError("weight_pipeline must be auto, shared or separate")
    if work_throttle is None:
        work_throttle = int(scheduler == "clc" and weight_pipeline == "separate")
    if work_throttle not in (0, 1) or rolled_mma not in (0, 1):
        raise ValueError("work_throttle and rolled_mma must be 0 or 1")
    if work_throttle and (scheduler == "static" or weight_pipeline == "shared"):
        raise ValueError("work throttling requires CLC with separate weight pipelines")
    if scheduler == "static" and max_active_clusters <= 0:
        raise ValueError(
            "static scheduling requires a positive max_active_clusters budget"
        )
    options = dict(
        sm_arch=107,
        batch_mode=int(BatchMode.BATCH_M),
        transpose_mma_output=0,
        route_act=int(RouteImpl.LDGSTS),
        route_sfs_act=int(RouteImpl.LDGSTS),
        dtype_a=int(DType.E2M1),
        dtype_b=int(DType.E2M1),
        dtype_c=int(DType.BF16),
        act_kind=int(ActKind.NONE),
        sf_layout_a=int(SfLayout.R128c4),
        sf_layout_b=int(SfLayout.R128c4),
        umma_cta_count=2,
        cluster_dim_x=2,
        cluster_dim_y=1,
        tile_n=256,
        tile_k=256,
        mma_m=256,
        mma_n=256,
        mma_k=128,
        epi_tile_n=64,
        tmem_ldst_max_num_regs=64,
        num_stages_a=8,
        num_stages_b=8,
        num_stages_smem_sfa=8,
        num_stages_smem_sfb=8,
        num_stages_tmem_acc=2,
        num_stages_tmem_sfa=1,
        num_stages_tmem_sfb=1,
        num_stages_c_smem=1,
        num_stages_workid=2 if scheduler == "static" else 3,
        tile_scheduler=int(
            TileScheduler.STATIC_PERSISTENT
            if scheduler == "static"
            else TileScheduler.PERSISTENT
        ),
        static_persistent_clusters=max_active_clusters if scheduler == "static" else 0,
        cta_raster_order=int(CtaRasterOrder.ALONG_N),
        use_tma_oob_opt=0,
        use_tma_store=1,
        use_max_tmem_overlap=0,
        use_early_exit=0,
        use_pdl=0,
        use_global_scales=0,
        use_unroll_loop_2x_for_mma=0,
        fuse_sf_copy_to_mma=1,
        fuse_operand_sf_loads=1,
        fused_a_single_commit=1,
        routed_native_sfa=1,
        routed_low_smem=1,
        dense_partition_tma_loads=1,
        dense_split_tma_pipeline=int(weight_pipeline == "shared"),
        use_work_throttle=work_throttle,
        dense_rolled_mma=rolled_mma,
        num_load_a_warps=4,
        num_load_sfa_warps=4,
        epilogue_regs=168,
        load_a_regs=72,
        load_sfa_regs=72,
    )
    for name in (
        "mma_regs",
        "load_regs",
        "load_sf_regs",
        "copy_sf_regs",
        "sync_regs",
        "workid_regs",
        "padding_regs",
    ):
        options[name] = 48
    return options


def routed_options(*, collective_tile_m=256, cluster_shape=(2, 1), **preset_options):
    """Extend the tile256 presets to fixed wider CGAs and two M fragments.

    ``collective_tile_m`` is the row coverage of one two-CTA MMA atom, not
    one CTA. Tile512 uses one accumulator stage and five operand stages to
    fit both M fragments; it keeps the generic shared-memory allocator.
    """
    if collective_tile_m not in (256, 512):
        raise ValueError("collective_tile_m must be 256 or 512")
    cluster_x, cluster_y = cluster_shape
    if cluster_x not in (2, 4) or cluster_y not in (1, 2, 4):
        raise ValueError("cluster_shape must be 2xY or 4xY with Y=1/2/4")
    options = routed_tile256_options(**preset_options)
    options.update(cluster_dim_x=cluster_x, cluster_dim_y=cluster_y)
    if collective_tile_m == 512:
        options.update(
            dense_m_repeats=2,
            dense_b_reuse=1,
            dense_async_sf=1,
            dense_rolled_epilogue=1,
            dense_preload_epilogue=1,
            dense_dkg_store_order=1,
            dense_epilogue_warps=8,
            epilogue_regs=144,
            num_stages_a=5,
            num_stages_b=5,
            num_stages_smem_sfa=5,
            num_stages_smem_sfb=5,
            num_stages_tmem_acc=1,
            routed_low_smem=0,
        )
    return options


def routed_tile256_fc1_options(*, max_active_clusters):
    """E256 FC1 preset: eight TMA-A/four SFA warps and gated NVFP4 output.

    Keep the static shared-weight mainloop's eight operand stages and two
    accumulator stages. Two 64-column output boxes drain each accumulator
    through a persistent two-stage C ring, using four epilogue warps.
    """
    options = routed_tile256_options(max_active_clusters=max_active_clusters)
    options.update(
        route_act=int(RouteImpl.TMA),
        num_load_a_warps=8,
        fuse_operand_sf_loads=0,
        fused_a_single_commit=0,
        dense_fuse_tma_loads=1,
        dtype_c=int(DType.E2M1),
        act_kind=int(ActKind.SWIGLU),
        sf_layout_c=int(SfLayout.R128c4),
        use_global_scales=1,
        epi_tile_n=128,
        num_stages_c_smem=2,
        gated_weight_group_size=64,
    )
    return options
