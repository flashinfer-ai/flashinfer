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

"""Coverage for register-capped TMEM loads and byte-derived C TMA swizzles."""

import pytest
import torch

from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
    ActKind,
    BatchMode,
    DType,
    SfLayout,
    TileScheduler,
    compute_warp_layout,
    make_config,
    uniform_pipeline_stage_overrides,
    validate_config,
)
from flashinfer.utils import is_sm100a_supported


def test_shuffled_matrix_indices_match_trtllm_gen_layout():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        shuffled_matrix_row_indices,
    )

    assert shuffled_matrix_row_indices(64) == (
        0,
        8,
        1,
        9,
        2,
        10,
        3,
        11,
        4,
        12,
        5,
        13,
        6,
        14,
        7,
        15,
    )
    assert shuffled_matrix_row_indices(128) == (
        0,
        8,
        16,
        24,
        1,
        9,
        17,
        25,
        2,
        10,
        18,
        26,
        3,
        11,
        19,
        27,
        4,
        12,
        20,
        28,
        5,
        13,
        21,
        29,
        6,
        14,
        22,
        30,
        7,
        15,
        23,
        31,
    )


requires_sm100a = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_sm100a_supported(torch.device("cuda")),
    reason="Prims-TS batched GEMM requires Blackwell sm_100a+",
)


@pytest.mark.parametrize(
    (
        "dtype_c",
        "epi_tile_n",
        "expected_cols",
        "expected_passes",
        "expected_swizzle_bytes",
    ),
    (
        (DType.BF16, 8, 8, 1, 16),
        (DType.FP16, 16, 16, 1, 32),
        (DType.BF16, 32, 32, 1, 64),
        (DType.FP16, 64, 64, 1, 128),
        (DType.BF16, 128, 64, 2, 128),
        (DType.E4M3, 16, 16, 1, 16),
        (DType.MXE4M3, 32, 32, 1, 32),
        (DType.E4M3, 64, 64, 1, 64),
        (DType.MXE4M3, 128, 128, 1, 128),
        (DType.E4M3, 256, 128, 2, 128),
        (DType.E2M1, 32, 32, 1, 16),
        (DType.MXE2M1, 64, 64, 1, 32),
        (DType.E2M1, 128, 128, 1, 64),
        (DType.MXE2M1, 256, 256, 1, 128),
    ),
)
def test_non_swap_epilogue_tma_width_is_derived_from_contiguous_n(
    dtype_c,
    epi_tile_n,
    expected_cols,
    expected_passes,
    expected_swizzle_bytes,
):
    cfg = make_config(
        batch_mode=int(BatchMode.BATCH_M),
        transpose_mma_output=0,
        dtype_c=int(dtype_c),
        tile_n=256,
        epi_tile_n=epi_tile_n,
        use_tma_store=1,
    )

    assert cfg.tma_store_cols == expected_cols
    assert cfg.tma_store_passes == expected_passes
    assert cfg.c_tma_swizzle_bytes == expected_swizzle_bytes
    assert cfg.c_tma_swizzle_bytes <= 128


@pytest.mark.parametrize("dtype_c", (DType.BF16, DType.FP16))
def test_non_swap_gated_tma_load_width_uses_pre_activation_columns(dtype_c):
    cfg = make_config(
        batch_mode=int(BatchMode.BATCH_M),
        transpose_mma_output=0,
        act_kind=int(ActKind.SWIGLU),
        dtype_c=int(dtype_c),
        tile_n=32,
        epi_tile_n=32,
        use_tma_store=1,
        use_shuffled_matrix=1,
    )

    validate_config(cfg, problem_mnk=(128, 128, 512))
    assert cfg.tma_store_cols == 16
    assert cfg.non_swap_tma_accumulator_cols == 32
    assert cfg.non_swap_tmem_load_num_regs == 32
    assert cfg.non_swap_gated_output_cols_per_call == 16


def test_non_swap_fp4_tma_rejects_sub_16_byte_transaction():
    cfg = make_config(
        batch_mode=int(BatchMode.BATCH_M),
        transpose_mma_output=0,
        act_kind=int(ActKind.SWIGLU),
        dtype_c=int(DType.E2M1),
        tile_n=32,
        epi_tile_n=32,
        use_tma_store=1,
        use_shuffled_matrix=1,
    )

    with pytest.raises(ValueError, match="complete 16-byte transaction"):
        validate_config(cfg, problem_mnk=(128, 128, 512))


def test_non_swap_gated_epilogue_streams_more_than_two_total_tmem_fragments():
    cfg = make_config(
        batch_mode=int(BatchMode.BATCH_M),
        transpose_mma_output=0,
        act_kind=int(ActKind.SWIGLU),
        dtype_a=int(DType.BF16),
        dtype_b=int(DType.BF16),
        dtype_c=int(DType.BF16),
        sf_layout_b=int(SfLayout.R128c4),
        mma_k=16,
        tile_k=128,
        tile_n=128,
        epi_tile_n=128,
        tmem_ldst_max_num_regs=32,
        use_tma_store=1,
    )

    validate_config(cfg, problem_mnk=(128, 256, 512))
    assert cfg.non_swap_epilogue_fragments_per_group == 2
    assert cfg.non_swap_gated_steps_per_epi_tile == 2
    assert cfg.non_swap_gated_output_cols_per_call == 32


def test_non_swap_gated_epilogue_rejects_more_than_32_outputs_per_call():
    cfg = make_config(
        batch_mode=int(BatchMode.BATCH_M),
        transpose_mma_output=0,
        act_kind=int(ActKind.SWIGLU),
        dtype_c=int(DType.BF16),
        tile_n=128,
        epi_tile_n=128,
        tmem_ldst_max_num_regs=64,
    )

    with pytest.raises(ValueError, match="at most 32 output columns per call"):
        validate_config(cfg, problem_mnk=(128, 256, 512))


@pytest.mark.parametrize("dtype_c", (DType.BF16, DType.FP16))
def test_non_swap_nvfp4_gated_16bit_tma_uses_per_warpgroup_scratch(dtype_c):
    cfg = make_config(
        batch_mode=int(BatchMode.BATCH_M),
        transpose_mma_output=0,
        act_kind=int(ActKind.SWIGLU),
        dtype_a=int(DType.E2M1),
        dtype_b=int(DType.E2M1),
        dtype_c=int(dtype_c),
        tile_n=256,
        epi_tile_m=128,
        epi_tile_n=64,
        umma_cta_count=2,
        tile_k=256,
        mma_m=256,
        mma_n=256,
        sf_layout_b=int(SfLayout.R128c4),
        tile_scheduler=int(TileScheduler.PERSISTENT),
        num_stages_tmem_acc=1,
        use_max_tmem_overlap=1,
        fuse_sf_copy_to_mma=1,
        num_stages_c_smem=1,
        use_tma_store=1,
    )

    assert cfg.uses_16bit_gated_tma_c_scratch
    assert cfg.non_swap_tmem_load_num_regs == 32
    assert cfg.non_swap_epilogue_fragments_per_group == 2
    assert cfg.tma_store_cols == 32
    assert cfg.c_tma_swizzle_bytes == 64
    assert cfg.num_bytes_c_tma_store_per_group == 8192
    assert cfg.num_bytes_c_smem_scratch == 16384
    assert not cfg.aliases_c_scratch_with_ab


@pytest.mark.parametrize(
    (
        "dtype_c",
        "expected_swizzle_bytes",
        "expected_store_rows",
        "expected_row_passes",
    ),
    (
        (DType.BF16, 128, 64, 2),
        (DType.FP16, 128, 64, 2),
        (DType.E4M3, 128, 128, 1),
        (DType.MXE4M3, 128, 128, 1),
        (DType.E2M1, 64, 128, 1),
        (DType.MXE2M1, 64, 128, 1),
    ),
)
@pytest.mark.parametrize("epi_tile_n", (8, 16, 32, 64, 128, 256))
def test_swap_epilogue_tma_swizzle_is_derived_from_contiguous_m(
    dtype_c,
    expected_swizzle_bytes,
    expected_store_rows,
    expected_row_passes,
    epi_tile_n,
):
    cfg = make_config(
        batch_mode=int(BatchMode.BATCH_N),
        transpose_mma_output=1,
        dtype_c=int(dtype_c),
        tile_n=256,
        epi_tile_m=128,
        epi_tile_n=epi_tile_n,
        use_tma_store=1,
    )

    assert cfg.tma_store_cols == epi_tile_n
    assert cfg.tma_store_passes == 1
    assert cfg.c_tma_swizzle_bytes == expected_swizzle_bytes
    assert cfg.c_tma_swizzle_bytes == min(cfg.epi_tile_m * cfg.dtype_c_bits // 8, 128)
    assert cfg.swap_tma_store_tile_rows == cfg.epi_tile_m
    assert cfg.swap_tma_store_rows == expected_store_rows
    assert cfg.swap_tma_store_row_passes == expected_row_passes
    assert cfg.swap_tma_store_rows * cfg.dtype_c_bits // 8 <= 128
    assert cfg.swap_tma_store_offsets_are_col_linear == (
        dtype_c in (DType.BF16, DType.FP16)
    )


@pytest.mark.parametrize("dtype_c", (DType.BF16, DType.FP16))
@pytest.mark.parametrize("epi_tile_m", (16, 32, 64, 128))
@pytest.mark.parametrize("epi_tile_n", (8, 16, 32, 64, 128, 256))
def test_swap_16bit_cached_smem_offsets_match_general_formula(
    dtype_c,
    epi_tile_m,
    epi_tile_n,
):
    cfg = make_config(
        batch_mode=int(BatchMode.BATCH_N),
        transpose_mma_output=1,
        dtype_c=int(dtype_c),
        tile_n=256,
        epi_tile_m=epi_tile_m,
        epi_tile_n=epi_tile_n,
        use_tma_store=1,
    )

    def smem_element_offset(row, col):
        box_rows = cfg.swap_tma_store_rows
        row_block, row_in_block = divmod(row, box_rows)
        box_stride = cfg.epi_tile_n * box_rows
        offset_bytes = (row_block * box_stride + col * box_rows + row_in_block) * 2
        swizzle_row = offset_bytes // 128
        if cfg.c_tma_swizzle_bytes % 128 == 0:
            swizzle_mask = swizzle_row % 8 * 16
        elif cfg.c_tma_swizzle_bytes % 64 == 0:
            swizzle_mask = swizzle_row % 4 * 16
        elif cfg.c_tma_swizzle_bytes % 32 == 0:
            swizzle_mask = swizzle_row % 2 * 16
        else:
            swizzle_mask = 0
        return (offset_bytes ^ swizzle_mask) // 2

    assert cfg.swap_tma_store_offsets_are_col_linear
    step = cfg.swap_tma_store_col_group_step_elems
    assert step == 8 * cfg.swap_tma_store_rows
    for row in range(cfg.swap_tma_store_tile_rows):
        for base_col in range(8):
            base_offset = smem_element_offset(row, base_col)
            for group in range(cfg.epi_tile_n // 8):
                assert smem_element_offset(row, base_col + group * 8) == (
                    base_offset + group * step
                )


@pytest.mark.parametrize(
    ("dtype_c", "expected_swizzle_bytes"),
    (
        (DType.BF16, 128),
        (DType.E4M3, 64),
    ),
)
def test_swap_deepseek_two_epilogue_store_covers_warpgroup_m_share(
    dtype_c,
    expected_swizzle_bytes,
):
    """Store every M row staged by each split DeepSeek epilogue warpgroup.

    Both the SMEM group stride and the per-warpgroup global M base advance by
    ``tile_m // 2`` rows. A smaller store tile leaves part of that warpgroup's
    staged output in SMEM without ever committing it to C.
    """
    cfg = make_config(
        batch_mode=int(BatchMode.BATCH_N),
        transpose_mma_output=1,
        use_deepseek_fp8=1,
        dtype_c=int(dtype_c),
        mma_m=64,
        epi_tile_m=64,
        epi_tile_n=8,
        use_tma_store=1,
    )

    assert cfg.has_deepseek_fp8_two_epilogue
    assert cfg.swap_tma_store_tile_rows == cfg.tile_m // 2
    assert cfg.c_tma_swizzle_bytes == expected_swizzle_bytes
    assert cfg.swap_tma_store_rows * cfg.swap_tma_store_row_passes == cfg.tile_m // 2
    assert cfg.swap_tma_store_rows * cfg.dtype_c_bits // 8 <= 128


@pytest.mark.parametrize(
    ("dtype_c", "expected_swizzle_bytes"),
    (
        (DType.BF16, 128),
        (DType.E4M3, 64),
        (DType.MXE4M3, 64),
        (DType.E2M1, 32),
        (DType.MXE2M1, 32),
    ),
)
def test_swap_gated_tma_uses_physical_output_m_width(
    dtype_c,
    expected_swizzle_bytes,
):
    cfg = make_config(
        batch_mode=int(BatchMode.BATCH_N),
        transpose_mma_output=1,
        act_kind=int(ActKind.SWIGLU),
        dtype_c=int(dtype_c),
        epi_tile_m=128,
        epi_tile_n=8,
        use_tma_store=1,
    )

    assert cfg.swap_tma_store_tile_rows == cfg.epi_tile_m // 2
    assert cfg.c_tma_swizzle_bytes == expected_swizzle_bytes
    assert cfg.swap_tma_store_rows == cfg.swap_tma_store_tile_rows
    assert cfg.swap_tma_store_row_passes == 1


@pytest.mark.parametrize(
    (
        "max_regs",
        "epi_tile_n",
        "expected_load_regs",
        "expected_load_cols",
        "expected_loads",
    ),
    (
        (32, 16, 8, 16, 1),
        (32, 32, 16, 32, 1),
        (32, 64, 32, 64, 1),
        (32, 128, 32, 64, 2),
        (64, 128, 64, 128, 1),
    ),
)
def test_swap_tmem_load_obeys_register_cap(
    max_regs,
    epi_tile_n,
    expected_load_regs,
    expected_load_cols,
    expected_loads,
):
    cfg = make_config(
        batch_mode=int(BatchMode.BATCH_N),
        transpose_mma_output=1,
        epi_tile_n=epi_tile_n,
        tmem_ldst_max_num_regs=max_regs,
    )

    assert cfg.tmem_load_num_regs == expected_load_regs
    assert cfg.swap_tmem_load_cols == expected_load_cols
    assert cfg.tmem_overlap_loads == expected_loads


@pytest.mark.timeout(180)
@pytest.mark.parametrize("tmem_ldst_max_num_regs", (32, 64))
def test_swap_ht256_register_capped_overlap_schedule_builds(
    tmem_ldst_max_num_regs,
):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_kernel import (
        build_batched_gemm_task_manager,
    )

    task_manager = build_batched_gemm_task_manager(
        num_experts=2,
        num_tokens=2048,
        top_k=1,
        batch_mode=int(BatchMode.BATCH_N),
        transpose_mma_output=1,
        tile_n=256,
        tile_k=256,
        epi_tile_m=128,
        epi_tile_n=128,
        mma_m=256,
        mma_n=256,
        mma_k=64,
        umma_cta_count=2,
        dtype_a=int(DType.E2M1),
        dtype_b=int(DType.E2M1),
        dtype_c=int(DType.BF16),
        **uniform_pipeline_stage_overrides(4),
        tile_scheduler=int(TileScheduler.PERSISTENT),
        num_stages_tmem_acc=1,
        sf_layout_a=int(SfLayout.R128c4),
        sf_layout_b=int(SfLayout.R128c4),
        fuse_sf_copy_to_mma=1,
        use_max_tmem_overlap=1,
        use_tma_oob_opt=1,
        use_tma_store=1,
        tmem_ldst_max_num_regs=tmem_ldst_max_num_regs,
        epilogue_regs=176,
        mma_regs=80,
        load_regs=80,
        load_sf_regs=80,
        workid_regs=80,
        padding_regs=80,
        verbose=False,
    )

    assert {task.name for task in task_manager.tasks} >= {
        "EpilogueTask0",
        "MmaTask0",
        "WorkScheduleTask",
    }


@requires_sm100a
@pytest.mark.xdist_group("isolated_cuda")
@pytest.mark.timeout(240)
@pytest.mark.parametrize("tile_n", (16, 32), ids=("s32b", "s64b"))
def test_non_swap_bf16_tma_swizzle_uses_128b_band_index(tile_n):
    """Exercise sub-128-byte swizzles whose logical rows share a 128B band."""
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import reference_check

    assert reference_check(
        num_experts=1,
        num_tokens=128,
        top_k=1,
        batch_mode=int(BatchMode.BATCH_M),
        transpose_mma_output=0,
        route_act=0,
        act_kind=0,
        tile_scheduler=int(TileScheduler.STATIC),
        tile_n=tile_n,
        tile_k=128,
        epi_tile_m=128,
        epi_tile_n=tile_n,
        mma_m=128,
        mma_n=tile_n,
        mma_k=16,
        umma_cta_count=1,
        dtype_a=int(DType.BF16),
        dtype_b=int(DType.BF16),
        dtype_c=int(DType.BF16),
        sf_layout_b=int(SfLayout.R8c4),
        **uniform_pipeline_stage_overrides(4),
        num_stages_tmem_acc=2,
        use_max_tmem_overlap=0,
        use_tma_store=1,
        epilogue_regs=160,
        mma_regs=24,
        load_regs=24,
        padding_regs=24,
        workid_regs=24,
    )


@requires_sm100a
@pytest.mark.xdist_group("isolated_cuda")
@pytest.mark.timeout(240)
def test_non_swap_bf16_tma_ring_reuse_across_persistent_tiles():
    """Reuse both C stages when each persistent tile commits one TMA box."""
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import reference_check

    assert reference_check(
        num_experts=7,
        num_tokens=4097,
        top_k=3,
        problem_n=64,
        problem_k=128,
        batch_mode=int(BatchMode.BATCH_M),
        transpose_mma_output=0,
        route_act=0,
        act_kind=int(ActKind.NONE),
        tile_scheduler=int(TileScheduler.PERSISTENT),
        tile_n=64,
        tile_k=128,
        epi_tile_m=128,
        epi_tile_n=64,
        mma_m=128,
        mma_n=64,
        mma_k=16,
        umma_cta_count=1,
        dtype_a=int(DType.BF16),
        dtype_b=int(DType.BF16),
        dtype_c=int(DType.BF16),
        sf_layout_b=int(SfLayout.R8c4),
        **uniform_pipeline_stage_overrides(4),
        num_stages_tmem_acc=2,
        num_stages_c_smem=2,
        use_max_tmem_overlap=0,
        use_tma_store=1,
        repeat_launches=5,
        output_guard_elements=256,
        epilogue_regs=160,
        mma_regs=24,
        load_regs=24,
        padding_regs=24,
        workid_regs=24,
    )


@requires_sm100a
@pytest.mark.xdist_group("isolated_cuda")
@pytest.mark.timeout(240)
@pytest.mark.parametrize(
    ("dtype_c", "tile_n"),
    (
        pytest.param(DType.E4M3, 32, id="fp8-s32b"),
        pytest.param(DType.BF16, 16, id="bf16-s32b"),
        pytest.param(DType.BF16, 32, id="bf16-s64b"),
    ),
)
def test_swap_tma_swizzle_uses_contiguous_m_width(dtype_c, tile_n):
    """Use the M-derived swizzle regardless of the logical N stripe width."""
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import RouteImpl
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import reference_check

    uses_fp8_output = dtype_c == DType.E4M3
    assert reference_check(
        num_experts=2,
        num_tokens=128,
        top_k=1,
        problem_n=128,
        problem_k=128,
        batch_mode=int(BatchMode.BATCH_N),
        transpose_mma_output=1,
        route_act=int(RouteImpl.TMA if uses_fp8_output else RouteImpl.NONE),
        act_kind=int(ActKind.NONE),
        tile_scheduler=int(TileScheduler.STATIC),
        tile_n=tile_n,
        tile_k=128,
        epi_tile_m=128,
        epi_tile_n=tile_n,
        mma_m=64,
        mma_n=tile_n,
        mma_k=32,
        umma_cta_count=1,
        dtype_a=int(DType.E4M3),
        dtype_b=int(DType.E4M3),
        dtype_c=int(dtype_c),
        **uniform_pipeline_stage_overrides(4),
        num_stages_tmem_acc=2,
        **({"num_stages_c_smem": 1} if not uses_fp8_output else {}),
        use_tma_store=1,
        use_tma_oob_opt=1,
        use_global_scales=1,
        use_deepseek_fp8=1,
        **({"num_load_b_warps": 2} if uses_fp8_output else {}),
        epilogue_regs=160,
        mma_regs=48,
        load_regs=48,
        padding_regs=48,
        workid_regs=48,
        load_sfab_regs=48,
    )


@requires_sm100a
@pytest.mark.xdist_group("isolated_cuda")
@pytest.mark.timeout(240)
@pytest.mark.parametrize("tmem_ldst_max_num_regs", (32, 64))
def test_swap_ht256_multi_work_tile_tma_handoff(tmem_ldst_max_num_regs):
    """Exercise both TMEM windows with one or two swapAB loads per stripe."""
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import reference_check

    assert reference_check(
        num_experts=2,
        num_tokens=2048,
        top_k=1,
        problem_n=256,
        problem_k=256,
        batch_mode=int(BatchMode.BATCH_N),
        transpose_mma_output=1,
        route_act=0,
        act_kind=0,
        tile_n=256,
        tile_k=256,
        epi_tile_m=128,
        epi_tile_n=128,
        mma_m=256,
        mma_n=256,
        mma_k=64,
        umma_cta_count=2,
        dtype_a=int(DType.E2M1),
        dtype_b=int(DType.E2M1),
        dtype_c=int(DType.BF16),
        **uniform_pipeline_stage_overrides(4),
        tile_scheduler=int(TileScheduler.PERSISTENT),
        num_stages_tmem_acc=1,
        sf_layout_a=int(SfLayout.R128c4),
        sf_layout_b=int(SfLayout.R128c4),
        fuse_sf_copy_to_mma=1,
        use_max_tmem_overlap=1,
        use_tma_oob_opt=1,
        use_tma_store=1,
        tmem_ldst_max_num_regs=tmem_ldst_max_num_regs,
        epilogue_regs=176,
        mma_regs=80,
        load_regs=80,
        load_sf_regs=80,
        workid_regs=80,
        padding_regs=80,
    )


@requires_sm100a
@pytest.mark.xdist_group("isolated_cuda")
@pytest.mark.timeout(240)
@pytest.mark.parametrize("tmem_ldst_max_num_regs", (32, 64))
@pytest.mark.parametrize("use_tma_store", (0, 1), ids=("direct", "tma"))
def test_swap_gated_bf16_multi_pass_tma(
    use_tma_store,
    tmem_ldst_max_num_regs,
):
    """Compare gated direct stores with the M-derived s128 TMA store."""
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import reference_check

    assert reference_check(
        num_experts=2,
        num_tokens=256,
        top_k=1,
        problem_n=256,
        problem_k=256,
        batch_mode=int(BatchMode.BATCH_N),
        transpose_mma_output=1,
        route_act=1,
        act_kind=int(ActKind.SWIGLU),
        has_gemm1_clamp_limit=1,
        gemm1_clamp_limit_value=2.0,
        tile_n=128,
        tile_k=256,
        epi_tile_m=128,
        epi_tile_n=128,
        mma_m=128,
        mma_n=128,
        mma_k=64,
        umma_cta_count=1,
        dtype_a=int(DType.E2M1),
        dtype_b=int(DType.E2M1),
        dtype_c=int(DType.BF16),
        **uniform_pipeline_stage_overrides(5),
        tile_scheduler=int(TileScheduler.STATIC),
        num_stages_tmem_acc=1,
        sf_layout_a=int(SfLayout.R128c4),
        sf_layout_b=int(SfLayout.R128c4),
        use_max_tmem_overlap=0,
        use_tma_oob_opt=1,
        use_tma_store=use_tma_store,
        tmem_ldst_max_num_regs=tmem_ldst_max_num_regs,
        epilogue_regs=128,
        mma_regs=48,
        load_regs=48,
        load_sf_regs=48,
        copy_sf_regs=48,
        padding_regs=48,
        load_b_regs=48,
    )


def test_swap_tma_config_validates_at_both_register_caps():
    """Keep config validation in the new suite independent of CUDA availability."""
    for max_regs in (32, 64):
        cfg = make_config(
            batch_mode=int(BatchMode.BATCH_N),
            transpose_mma_output=1,
            umma_cta_count=2,
            tile_n=256,
            tile_k=256,
            epi_tile_m=128,
            epi_tile_n=128,
            mma_m=256,
            mma_n=256,
            mma_k=64,
            tile_scheduler=int(TileScheduler.PERSISTENT),
            num_stages_tmem_acc=1,
            sf_layout_a=int(SfLayout.R128c4),
            sf_layout_b=int(SfLayout.R128c4),
            fuse_sf_copy_to_mma=1,
            use_max_tmem_overlap=1,
            use_tma_store=1,
            tmem_ldst_max_num_regs=max_regs,
        )
        compute_warp_layout(cfg)
        validate_config(cfg)
