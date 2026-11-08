# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Mixed-weight tasks share stage boundaries without sharing weight formats."""

import pytest
import torch

pytestmark = pytest.mark.arch_rubin


@pytest.fixture(autouse=True)
def _require_rubin_worker():
    # The root architecture hook is absent with --confcutdir=tests/moe_ep.
    if torch.cuda.is_available() and torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("Requires SM107")


def _mainloop(tile, *, shared):
    from flashinfer.moe_ep.kernel_src.sm107.next_cutedsl_megamoe.shim import (
        block_scaled,
    )

    block_scaled._configure_dsl_arch()
    import cutlass
    from cutlass.cute.nvgpu import OperandMajorMode
    from sources.api import ImplDesc, ProblemDesc
    from sources.helpers.cute_py_helpers import (
        Tcgen05MmaInstruction,
        make_tcgen05_tmem_plan,
    )
    from sources.kernel_src.rubin.inference.mega.block_scaled_swap_ab_fc12_mainloop import (
        BlockScaledSwapAbFc12Mainloop,
    )

    instruction = Tcgen05MmaInstruction(
        a_type=cutlass.Float4E2M1FN,
        b_type=cutlass.Float8E4M3FN,
        instruction_mnk=(*tile[:2], 64),
        participates=2 if tile[0] == 256 else 1,
        sfa_type=cutlass.Float8E8M0FNU,
        sfb_type=cutlass.Float8E8M0FNU,
        sf_vec_size=32,
    )
    problem = ProblemDesc(
        dict(
            quant_kind="mxfp4_mxfp8",
            shared_quant_kind="mxfp8_e4m3" if shared else None,
            a_major_mode=OperandMajorMode.K,
            b_major_mode=OperandMajorMode.K,
            hidden_size=4096,
            intermediate_gateup_size=4096,
        )
    )
    implementation = ImplDesc(
        dict(
            mma_instruction_mnk=instruction.instruction_mnk,
            mma_tiler_mnk=tile,
            mma_k_mode="2x",
            cluster_shape_mn=(4, 1),
            use_2cta_instrs=instruction.participates == 2,
            mainloop_smem_budget_bytes=200 * 1024,
            num_accumulator_consumer_warps_per_cta=4,
            tmem_plan=make_tcgen05_tmem_plan(instruction, "sm_107", tile),
        )
    )
    return BlockScaledSwapAbFc12Mainloop(problem, implementation)


@pytest.mark.parametrize("tile_m", [128, 256])
@pytest.mark.parametrize("tile_n", [64, 128, 256])
def test_shared_weight_stage_boundaries(tile_m, tile_n):
    routed = _mainloop((tile_m, tile_n, 128), shared=False)
    mixed = _mainloop((tile_m, tile_n, 128), shared=True)
    shared = mixed.shared_mainloop
    assert shared is not None
    assert mixed.num_ab_pipeline_stages == shared.num_ab_pipeline_stages
    assert mixed.num_ab_pipeline_stages <= routed.num_ab_pipeline_stages
    assert mixed.a_smem_region.dtype.width == 4
    assert shared.a_smem_region.dtype.width == 8
    assert mixed.a_storage_region == shared.a_smem_region

    for stage in range(mixed.num_ab_pipeline_stages):
        packed_start = stage * mixed.a_smem_region.stride[-1] // 2
        shared_start = stage * shared.a_smem_region.stride[-1]
        assert packed_start == shared_start
        assert packed_start + routed.a_smem_region.stride[-1] // 2 <= (
            shared_start + shared.a_smem_region.stride[-1]
        )
    assert mixed.a_smem_region.nbytes <= mixed.a_storage_region.nbytes
    assert mixed.ab_stage_tma_bytes == routed.ab_stage_tma_bytes
    assert shared.ab_stage_tma_bytes > mixed.ab_stage_tma_bytes
    assert mixed.num_tmem_alloc_cols == shared.num_tmem_alloc_cols
    assert mixed.num_accumulator_stages == shared.num_accumulator_stages
    assert routed.shared_mainloop is None
    assert routed.a_storage_region == routed.a_smem_region


@pytest.mark.parametrize("schedule", ["grouped", "phase_interleave"])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("mixed", [False, True])
def test_shared_schedule_workspace(schedule, shared, mixed):
    kernel = _kernel(schedule, shared=shared, mixed=mixed)
    routed_slots = kernel.token_comm.max_fc1_ready_slot_count
    assert kernel.scheduler.expert_count == 4
    assert kernel.shared_counter_offset == routed_slots
    regions = kernel._device_workspace
    expected_slots = routed_slots + (2 if shared else 0)
    assert regions.region(kernel.fc1_done_counter_region).shape == (expected_slots,)
    if shared:
        assert regions.region(kernel.shared_ready_region).shape == (2,)
        assert regions.region(kernel.shared_ready_region).reset == "tail_reset"
    assert regions.region(kernel.fc1_done_counter_region).reset == "tail_reset"


@pytest.mark.parametrize("schedule", ["grouped", "phase_interleave"])
@pytest.mark.parametrize("shared", [False, True])
def test_mixed_cga_workspace_covers_all_fallback_clusters(schedule, shared):
    kernel = _kernel(schedule, shared=shared, mixed=True)
    cfg = kernel.mixed_cga_config
    slots = kernel._device_workspace.region(
        "non_clc_mixed_cga.fallback_group_tokens"
    ).shape[0]
    assert slots == cfg.total_cta_cnt // 2
    assert slots > cfg.fallback_cluster_count
    assert kernel._mainloop.is_mixed_cga
    if shared:
        assert kernel._mainloop.shared_mainloop.is_mixed_cga


def _kernel(schedule, *, shared, mixed=False, variant="inference", tile_n=128):
    from flashinfer.moe_ep.kernel_src.sm107.next_cutedsl_megamoe.shim import (
        block_scaled,
    )

    block_scaled._configure_dsl_arch()
    import cutlass
    from cutlass.cute.nvgpu import OperandMajorMode
    from sources.api import ImplDesc, ProblemDesc
    from sources.kernel_src.rubin.inference.mega.block_scaled_swap_ab_mega_moe_kernel import (
        BlockScaledSwapAbMegaMoeKernel,
    )
    from sources.quant_def import CombineFormat

    problem = ProblemDesc(
        dict(
            expert_count=8,
            intermediate_gateup_size=512,
            hidden_size=256,
            quant_kind="mxfp4_mxfp8",
            shared_quant_kind="mxfp8_e4m3" if shared else None,
            a_major_mode=OperandMajorMode.K,
            b_major_mode=OperandMajorMode.K,
            combine_format=CombineFormat.parse("bf16"),
            gate_up_clamp=None,
            world_size=2,
            topk=2,
            topk_index_dtype=cutlass.Int32,
            max_tokens_per_rank=256,
            apply_topk_at_fc1=False,
        )
    )
    implementation = ImplDesc(
        dict(
            mma_instruction_mnk=(256, tile_n, 64),
            mma_tiler_mnk=(256, tile_n, 128),
            mma_k_mode="2x",
            cluster_shape_mn=(4, 1),
            use_2cta_instrs=True,
            schedule_policy=(schedule, None),
            token_padding_block=128,
            sf_padding_block=128,
            work_id_mode="atomic_counter",
            fc2_use_bulk=True,
            epi_flag_batches=(4, 2),
            launch_cluster_count=4,
            token_in_flag_batch=1,
            token_back_mode="epi_warps",
            reduce_topk_in_kernel=False,
        )
    )
    if mixed:
        implementation = ImplDesc(
            {
                **implementation,
                "fallback_cluster_shape_mn": (2, 1),
                "preferred_cluster_count": 2,
                "fallback_cluster_count": 4,
            }
        )
    if variant == "genphase":
        from sources.kernel_src.rubin.inference.mega.block_scaled_swap_ab_mega_moe_kernel_gen_specialized import (
            BlockScaledSwapAbGenphaseMoeKernel,
        )

        return BlockScaledSwapAbGenphaseMoeKernel(problem, implementation)
    return BlockScaledSwapAbMegaMoeKernel(problem, implementation)


@pytest.mark.parametrize("tile_n", [64, 128, 256])
def test_shared_genphase_stage_plan(tile_n):
    routed = _kernel("grouped", shared=False, variant="genphase", tile_n=tile_n)
    kernel = _kernel("grouped", shared=True, variant="genphase", tile_n=tile_n)
    mixed = kernel._mainloop
    shared = mixed.shared_mainloop
    assert shared is not None
    assert mixed.num_weight_ab_stages == shared.num_weight_ab_stages
    assert mixed.num_token_ab_stages == shared.num_token_ab_stages
    assert mixed.num_weight_ab_stages <= routed._mainloop.num_weight_ab_stages
    assert mixed.uses_asymmetric_ab_stages == (tile_n == 256)
    assert mixed.dynamic_gather == (tile_n == 64)
    assert mixed.a_storage_region == shared.a_smem_region
    for stage in range(mixed.num_weight_ab_stages):
        assert (
            stage * mixed.a_smem_region.stride[-1] // 2
            == stage * shared.a_smem_region.stride[-1]
        )
    slots = kernel._device_workspace.region(kernel.fc1_done_counter_region).shape[0]
    assert slots == kernel.shared_counter_offset + (256 + tile_n - 1) // tile_n
    assert kernel.name() != routed.name()
