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

"""Schedule-verification regressions for BatchedGemm TS schedules."""

import sys

import pytest


def _fused_operand_sf_config(**overrides):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        ActKind,
        BatchMode,
        DType,
        RouteImpl,
        SfLayout,
        TileScheduler,
        compute_warp_layout,
        make_config,
    )

    values = dict(
        batch_mode=int(BatchMode.BATCH_N),
        umma_cta_count=2,
        dtype_a=int(DType.E2M1),
        dtype_b=int(DType.E2M1),
        dtype_c=int(DType.E2M1),
        epi_tile_n=32,
        act_kind=int(ActKind.SWIGLU),
        fuse_operand_sf_loads=1,
        fuse_sf_copy_to_mma=1,
        mma_k=64,
        mma_m=256,
        mma_n=128,
        num_load_sfb_warps=4,
        num_stages_a=3,
        num_stages_b=3,
        num_stages_smem_sfa=3,
        num_stages_smem_sfb=3,
        route_act=int(RouteImpl.LDGSTS),
        route_sfs_act=int(RouteImpl.LDGSTS),
        sf_layout_b=int(SfLayout.LINEAR),
        tile_k=512,
        tile_n=128,
        tile_scheduler=int(TileScheduler.PERSISTENT),
        transpose_mma_output=1,
        use_early_exit=1,
    )
    values.update(overrides)
    cfg = make_config(**values)
    compute_warp_layout(cfg)
    return cfg


@pytest.mark.cpu_only
def test_persistent_cta_swizzle_config_and_cli_mapping():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        CtaRasterOrder,
        TileScheduler,
        make_config,
        validate_config,
    )
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
        _parse_overrides,
        build_arg_parser,
    )

    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--tile-scheduler",
            "persistent",
            "--cta-swizzle-type",
            "rasterizeAlongN",
        ]
    )
    overrides = _parse_overrides(args)
    cfg = make_config(**overrides)

    validate_config(cfg)
    assert cfg.cta_raster_order == int(CtaRasterOrder.ALONG_N)
    assert cfg.tile_scheduler == int(TileScheduler.PERSISTENT)
    assert not cfg.raster_along_m


@pytest.mark.cpu_only
def test_clc_fast_drain_skip_is_suffix_only_for_slow_token_axis():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        BatchMode,
        CtaRasterOrder,
        TileScheduler,
        make_config,
    )

    persistent = int(TileScheduler.PERSISTENT)
    along_m = int(CtaRasterOrder.ALONG_M)
    along_n = int(CtaRasterOrder.ALONG_N)
    swap_along_m = make_config(
        batch_mode=int(BatchMode.BATCH_N),
        cta_raster_order=along_m,
    )
    swap_along_n = make_config(
        batch_mode=int(BatchMode.BATCH_N),
        cta_raster_order=along_n,
        tile_scheduler=persistent,
    )
    batch_m_along_m = make_config(
        batch_mode=int(BatchMode.BATCH_M),
        transpose_mma_output=0,
        cta_raster_order=along_m,
    )
    batch_m_along_n = make_config(
        batch_mode=int(BatchMode.BATCH_M),
        transpose_mma_output=0,
        cta_raster_order=along_n,
        tile_scheduler=persistent,
    )

    assert swap_along_m.clc_fast_drain_skip_is_suffix
    assert not swap_along_n.clc_fast_drain_skip_is_suffix
    assert not batch_m_along_m.clc_fast_drain_skip_is_suffix
    assert batch_m_along_n.clc_fast_drain_skip_is_suffix


@pytest.mark.cpu_only
def test_along_n_cta_swizzle_rejects_non_persistent_scheduler():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        CtaRasterOrder,
        make_config,
        validate_config,
    )

    cfg = make_config(cta_raster_order=int(CtaRasterOrder.ALONG_N))

    with pytest.raises(ValueError, match="requires persistent scheduling"):
        validate_config(cfg)


@pytest.mark.cpu_only
@pytest.mark.parametrize("epi_tile_m", (64, 256))
@pytest.mark.parametrize("use_tma_store", (0, 1))
def test_batch_m_requires_full_m_epilogue_tile(epi_tile_m, use_tma_store):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        BatchMode,
        DType,
        make_config,
        validate_config,
    )

    cfg = make_config(
        batch_mode=int(BatchMode.BATCH_M),
        transpose_mma_output=0,
        dtype_a=int(DType.BF16),
        dtype_b=int(DType.BF16),
        dtype_c=int(DType.BF16),
        mma_k=16,
        mma_n=128,
        tile_k=128,
        tile_n=128,
        epi_tile_m=epi_tile_m,
        epi_tile_n=128,
        use_tma_store=use_tma_store,
    )

    with pytest.raises(ValueError, match="epi_tile_m must equal tile_m"):
        validate_config(cfg)


@pytest.mark.cpu_only
@pytest.mark.parametrize("use_tma_store", (0, 1))
def test_batch_m_accepts_full_m_epilogue_tile(use_tma_store):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        BatchMode,
        DType,
        make_config,
        validate_config,
    )

    cfg = make_config(
        batch_mode=int(BatchMode.BATCH_M),
        transpose_mma_output=0,
        dtype_a=int(DType.BF16),
        dtype_b=int(DType.BF16),
        dtype_c=int(DType.BF16),
        mma_k=16,
        mma_n=128,
        tile_k=128,
        tile_n=128,
        epi_tile_m=128,
        epi_tile_n=128,
        use_tma_store=use_tma_store,
    )

    validate_config(cfg)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("dtype_name", "mma_k", "tile_k", "act_name", "problem_n", "accepted"),
    (
        ("E2M1", 64, 256, "SWIGLU", 496, False),
        ("E2M1", 64, 256, "SWIGLU", 512, True),
        ("E4M3", 32, 128, "RELU2", 127, False),
        ("E4M3", 32, 128, "RELU2", 128, True),
        ("BF16", 16, 128, "NONE", 127, False),
        ("BF16", 16, 128, "NONE", 128, True),
    ),
)
def test_batch_m_output_rows_require_16_byte_alignment(
    dtype_name, mma_k, tile_k, act_name, problem_n, accepted
):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        ActKind,
        BatchMode,
        DType,
        SfLayout,
        make_config,
        uniform_pipeline_stage_overrides,
        validate_config,
    )

    dtype = int(getattr(DType, dtype_name))
    cfg = make_config(
        batch_mode=int(BatchMode.BATCH_M),
        transpose_mma_output=0,
        dtype_a=dtype,
        dtype_b=dtype,
        dtype_c=dtype,
        tile_n=128,
        tile_k=tile_k,
        mma_m=256,
        mma_n=128,
        mma_k=mma_k,
        umma_cta_count=2,
        epi_tile_m=128,
        epi_tile_n=64,
        tile_scheduler=1,
        num_stages_tmem_acc=2,
        use_unroll_loop_2x_for_mma=0,
        act_kind=int(getattr(ActKind, act_name)),
        route_act=0,
        use_tma_store=0,
        use_global_scales=1,
        sf_layout_b=int(SfLayout.R128c4),
        sf_layout_c=int(SfLayout.R8c4),
        **uniform_pipeline_stage_overrides(2),
    )

    if accepted:
        validate_config(cfg, problem_mnk=(128, problem_n, 512))
    else:
        with pytest.raises(ValueError, match="output rows must be 16-byte aligned"):
            validate_config(cfg, problem_mnk=(128, problem_n, 512))


@pytest.mark.cpu_only
def test_batch_n_does_not_require_16_byte_output_rows():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        ActKind,
        DType,
        make_config,
        validate_config,
    )

    cfg = make_config(
        dtype_a=int(DType.BF16),
        dtype_b=int(DType.BF16),
        dtype_c=int(DType.BF16),
        mma_k=16,
        act_kind=int(ActKind.SWIGLU),
    )

    validate_config(cfg, problem_mnk=(128, 504, 512))


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("argv", "expected"),
    (
        ([], 32),
        (["--tmem-ldst-max-num-regs", "32"], 32),
        (["--tmem-ldst-max-num-regs", "64"], 64),
    ),
)
def test_tmem_ldst_max_num_regs_cli_mapping(argv, expected):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import make_config
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
        _parse_overrides,
        build_arg_parser,
    )

    args = build_arg_parser().parse_args(argv)
    cfg = make_config(**_parse_overrides(args))

    assert cfg.tmem_ldst_max_num_regs == expected


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("value", "expected"),
    (("true", 1), ("1", 1), ("false", 0), ("0", 0)),
)
def test_use_shuffled_matrix_cli_accepts_boolean_spelling(value, expected):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
        _parse_overrides,
        build_arg_parser,
    )

    args = build_arg_parser().parse_args(["--use-shuffled-matrix", value])

    assert _parse_overrides(args)["use_shuffled_matrix"] == expected


@pytest.mark.cpu_only
@pytest.mark.parametrize("distribution", ("default", "allZeros"))
def test_data_distribution_cli_matches_trtllm_gen_spelling(distribution):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import build_arg_parser

    args = build_arg_parser().parse_args(["--dist", distribution])

    assert args.dist == distribution


@pytest.mark.cpu_only
def test_all_zero_input_data_does_not_modify_routing_metadata():
    import torch

    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
        _zero_input_data_tensors,
    )

    token_data = torch.tensor([1, 2], dtype=torch.uint8)
    weight_data = torch.tensor([3, 4], dtype=torch.uint8)
    scale_factors = torch.tensor([5, 6], dtype=torch.uint8)
    route_map = torch.tensor([1, 0], dtype=torch.int32)

    _zero_input_data_tensors(token_data, weight_data, scale_factors)

    assert not torch.count_nonzero(token_data)
    assert not torch.count_nonzero(weight_data)
    assert not torch.count_nonzero(scale_factors)
    assert torch.equal(route_map, torch.tensor([1, 0], dtype=torch.int32))


@pytest.mark.cpu_only
def test_nvfp4_instruction_descriptor_is_architecture_specific():
    from cutlass.experimental import primitives as prims

    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        MX_MMA_FORMAT_E2M1,
        MX_MMA_FORMAT_E4M3,
    )
    from flashinfer.prims_ts.batched_gemm.tmem_c_resources import (
        _build_fp8_instr_desc,
        _build_mx_instr_desc,
        _build_nvfp4_instr_desc,
        _mx_dtype_from_format,
    )

    blackwell_desc = _build_nvfp4_instr_desc(
        n_dim=256, m_dim=256, mma_k=64, is_rubin=False
    )
    assert int(blackwell_desc.a_format) == 5
    assert int(blackwell_desc.b_format) == 5
    assert int(blackwell_desc) & 0xFFFFFFFF == 0x10401680

    rubin_k64_desc = _build_nvfp4_instr_desc(
        n_dim=256, m_dim=256, mma_k=64, is_rubin=True
    )
    assert int(rubin_k64_desc.a_format) == 1
    assert int(rubin_k64_desc.b_format) == 1
    assert int(rubin_k64_desc.k_dim) == 0
    assert int(rubin_k64_desc) & 0xFFFFFFFF == 0x10400480

    rubin_k128_desc = _build_nvfp4_instr_desc(
        n_dim=256, m_dim=256, mma_k=128, is_rubin=True
    )
    assert int(rubin_k128_desc.a_format) == 1
    assert int(rubin_k128_desc.b_format) == 1
    assert int(rubin_k128_desc.k_dim) == 2
    assert int(rubin_k128_desc) & 0xFFFFFFFF == 0x10400488

    # OCP MXFP4 uses kind::mxf8f6f4, whose E2M1 encoding remains 5.
    mx_e2m1 = _mx_dtype_from_format(MX_MMA_FORMAT_E2M1)
    mx_desc = prims.Tcgen05MxInstrDesc.build(
        a_dtype=mx_e2m1,
        b_dtype=mx_e2m1,
        scale_format=1,
        n_dim=256,
        m_dim=256,
    )
    assert int(mx_desc.a_format) == 5
    assert int(mx_desc.b_format) == 5

    fp8_k32 = _build_fp8_instr_desc(n_dim=256, m_dim=128, mma_k=32)
    fp8_k64 = _build_fp8_instr_desc(n_dim=256, m_dim=128, mma_k=64)
    assert int(fp8_k32.k_dim) == 0
    assert int(fp8_k64.k_dim) == 1
    assert (int(fp8_k32) ^ int(fp8_k64)) & 0xFFFFFFFF == 1 << 29

    mx_k32 = _build_mx_instr_desc(
        a_format=MX_MMA_FORMAT_E2M1,
        b_format=MX_MMA_FORMAT_E4M3,
        n_dim=256,
        m_dim=128,
        mma_k=32,
    )
    mx_k64 = _build_mx_instr_desc(
        a_format=MX_MMA_FORMAT_E2M1,
        b_format=MX_MMA_FORMAT_E4M3,
        n_dim=256,
        m_dim=128,
        mma_k=64,
    )
    assert int(mx_k32.k_dim) == 0
    assert int(mx_k64.k_dim) == 1
    assert (int(mx_k32) ^ int(mx_k64)) & 0xFFFFFFFF == 1 << 31


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    (
        "mma_k",
        "smem_bits_a",
        "smem_bits_b",
        "tile_k",
        "b_stage_bytes",
        "expected_a",
        "expected_b",
    ),
    (
        # Existing BF16, FP8/MX, and NVFP4 paths. The split-B rows use half
        # the full tile_n=64 stage size to pin the per-CTA group stride.
        (
            16,
            16,
            16,
            128,
            8192,
            (0, 2, 4, 6, 1024, 1026, 1028, 1030),
            (0, 2, 4, 6, 256, 258, 260, 262),
        ),
        (
            32,
            8,
            8,
            256,
            8192,
            (0, 2, 4, 6, 1024, 1026, 1028, 1030),
            (0, 2, 4, 6, 256, 258, 260, 262),
        ),
        (
            64,
            4,
            4,
            512,
            8192,
            (0, 2, 4, 6, 1024, 1026, 1028, 1030),
            (0, 2, 4, 6, 256, 258, 260, 262),
        ),
        (128, 4, 4, 512, 8192, (0, 4, 1024, 1028), (0, 4, 256, 260)),
        # Rubin 2x mmaK: uniform FP8/MXFP8 and mixed native-MXFP4/MXFP8.
        (64, 8, 8, 256, 8192, (0, 4, 1024, 1028), (0, 4, 256, 260)),
        (
            64,
            4,
            8,
            512,
            16384,
            (0, 2, 4, 6, 1024, 1026, 1028, 1030),
            (0, 4, 256, 260, 512, 516, 768, 772),
        ),
    ),
)
def test_mma_smem_descriptor_k_stepping(
    mma_k,
    smem_bits_a,
    smem_bits_b,
    tile_k,
    b_stage_bytes,
    expected_a,
    expected_b,
):
    from flashinfer.prims_ts.batched_gemm.tmem_c_resources import (
        _mma_smem_desc_increment,
    )

    assert (
        tuple(
            _mma_smem_desc_increment(
                idx,
                mma_k=mma_k,
                smem_bits=smem_bits_a,
                tile_k=tile_k,
            )
            for idx in range(tile_k // mma_k)
        )
        == expected_a
    )
    assert (
        tuple(
            _mma_smem_desc_increment(
                idx,
                mma_k=mma_k,
                smem_bits=smem_bits_b,
                tile_k=tile_k,
                stage_bytes=b_stage_bytes,
            )
            for idx in range(tile_k // mma_k)
        )
        == expected_b
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("dtype_a_name", "dtype_b_name", "sm_arch", "mma_k", "mma_m"),
    (
        ("MXE4M3", "MXE4M3", 100, 32, 128),
        ("MXE4M3", "MXE4M3", 107, 64, 128),
        ("MXE2M1", "MXE4M3", 107, 64, 128),
        ("E4M3", "E4M3", 100, 32, 128),
        ("E4M3", "E4M3", 107, 64, 128),
    ),
)
def test_validation_accepts_dtype_specific_mma_k(
    dtype_a_name, dtype_b_name, sm_arch, mma_k, mma_m
):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        DType,
        make_config,
        validate_config,
    )

    cfg = make_config(
        sm_arch=sm_arch,
        dtype_a=int(getattr(DType, dtype_a_name)),
        dtype_b=int(getattr(DType, dtype_b_name)),
        dtype_c=int(DType.BF16),
        tile_k=256,
        mma_k=mma_k,
        mma_m=mma_m,
        tile_n=8,
        mma_n=8,
        epi_tile_n=8,
    )
    validate_config(cfg)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("dtype_a_name", "dtype_b_name", "sm_arch", "mma_k", "mma_m", "match"),
    (
        ("MXE4M3", "MXE4M3", 100, 64, 128, "MX mma_k=64 requires sm_arch=107"),
        ("MXE4M3", "MXE4M3", 107, 128, 128, "MX MMA requires mma_k=32 or 64"),
        (
            "MXE2M1",
            "MXE2M1",
            107,
            64,
            128,
            "MxFp4NvFp4 mma_k=128 path is not implemented",
        ),
        ("E4M3", "E4M3", 100, 64, 128, "FP8 mma_k=64 requires sm_arch=107"),
        (
            "E4M3",
            "E4M3",
            107,
            64,
            64,
            "FP8 mma_k=64 requires mma_m to be a multiple of 128",
        ),
        ("BF16", "BF16", 107, 64, 128, "BF16 MMA requires mma_k=16"),
        (
            "MXE2M1",
            "BF16",
            107,
            64,
            128,
            "CastA BF16 MMA requires mma_k=16",
        ),
    ),
)
def test_validation_rejects_dtype_specific_mma_k(
    dtype_a_name, dtype_b_name, sm_arch, mma_k, mma_m, match
):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        DType,
        make_config,
        validate_config,
    )

    cfg = make_config(
        sm_arch=sm_arch,
        dtype_a=int(getattr(DType, dtype_a_name)),
        dtype_b=int(getattr(DType, dtype_b_name)),
        dtype_c=int(DType.BF16),
        tile_k=256,
        mma_k=mma_k,
        mma_m=mma_m,
        tile_n=8,
        mma_n=8,
        epi_tile_n=8,
    )
    with pytest.raises(ValueError, match=match):
        validate_config(cfg)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("dtype_a_name", "dtype_b_name", "sm_arch", "mma_k", "tile_k"),
    (
        ("MXE2M1", "MXE4M3", 107, 64, 64),
        ("MXE2M1", "MXE4M3", 107, 64, 192),
        ("MXE4M3", "MXE4M3", 100, 32, 96),
    ),
)
def test_validation_rejects_mx_tile_k_partial_sf_block(
    dtype_a_name, dtype_b_name, sm_arch, mma_k, tile_k
):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        DType,
        make_config,
        validate_config,
    )

    cfg = make_config(
        sm_arch=sm_arch,
        dtype_a=int(getattr(DType, dtype_a_name)),
        dtype_b=int(getattr(DType, dtype_b_name)),
        dtype_c=int(DType.BF16),
        tile_k=tile_k,
        mma_k=mma_k,
        tile_n=8,
        mma_n=8,
        epi_tile_n=8,
    )
    with pytest.raises(
        ValueError, match="MX block scaling requires tile_k.*multiple of 128"
    ):
        validate_config(cfg)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("dtype_a_name", "dtype_b_name", "problem_k", "accepted"),
    (
        ("MXE2M1", "MXE4M3", 512, True),
        ("MXE2M1", "MXE4M3", 576, False),
        ("MXE2M1", "MXE4M3", 640, True),
        ("MXE2M1", "MXE4M3", 704, False),
        ("MXE2M1", "MXE4M3", 1152, True),
        ("E2M1", "E2M1", 544, False),
        ("E2M1", "E2M1", 576, True),
    ),
)
def test_validation_requires_complete_input_scale_factor_packs(
    dtype_a_name, dtype_b_name, problem_k, accepted
):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        DType,
        make_config,
        resolve_problem_config,
        validate_config,
    )

    is_mx = dtype_a_name.startswith("MX")
    cfg = make_config(
        sm_arch=107,
        dtype_a=int(getattr(DType, dtype_a_name)),
        dtype_b=int(getattr(DType, dtype_b_name)),
        dtype_c=int(DType.BF16),
        tile_k=256,
        mma_k=64,
        tile_n=8,
        mma_n=8,
        epi_tile_n=8,
    )
    resolved = resolve_problem_config(cfg, problem_k)
    if accepted:
        validate_config(resolved)
    else:
        alignment = 128 if is_mx else 64
        with pytest.raises(
            ValueError,
            match=rf"problem_k={problem_k}.*pad K to a multiple of {alignment}",
        ):
            validate_config(resolved)


@pytest.mark.cpu_only
def test_problem_mnk_drives_input_scale_factor_k_validation():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        DType,
        make_config,
        validate_config,
    )

    cfg = make_config(
        sm_arch=107,
        dtype_a=int(DType.MXE2M1),
        dtype_b=int(DType.MXE4M3),
        dtype_c=int(DType.BF16),
        tile_k=256,
        mma_k=64,
        tile_n=8,
        mma_n=8,
        epi_tile_n=8,
    )
    with pytest.raises(ValueError, match=r"problem_k=576.*multiple of 128"):
        validate_config(cfg, problem_mnk=(128, 5120, 576))


@pytest.mark.cpu_only
@pytest.mark.parametrize(("tile_n", "tma_bytes"), ((8, 32), (16, 64)))
def test_small_mx_sfb_stage_stride_is_tma_aligned(tile_n, tma_bytes):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        DType,
        make_config,
    )

    cfg = make_config(
        sm_arch=107,
        dtype_a=int(DType.MXE2M1),
        dtype_b=int(DType.MXE4M3),
        dtype_c=int(DType.BF16),
        tile_k=128,
        mma_k=64,
        tile_n=tile_n,
        mma_n=tile_n,
        epi_tile_n=tile_n,
    )
    assert cfg.num_bytes_sfb_per_stage == 64
    assert cfg.num_bytes_sfb_tma_per_stage == tma_bytes
    assert cfg.smem_sfb_stage_stride_bytes == 128


@pytest.mark.cpu_only
def test_rubin_mixed_mx_2x_mmak_uses_native_fp4_smem():
    from cutlass.experimental import cuda

    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        DType,
        make_config,
        tmem_sf_cols,
        tmem_sf_vec4_groups_per_mma,
        validate_config,
    )
    from flashinfer.prims_ts.batched_gemm.batched_gemm_kernel import (
        _tma_box_k_for_dtype,
        _tma_format_for_dtype,
        _tma_swizzle_for_fastest_dim_bytes,
    )
    from flashinfer.prims_ts.batched_gemm.smem_ab_resources import (
        _native_mx_fp4_smem_desc_params,
    )

    common = dict(
        dtype_a=int(DType.MXE2M1),
        dtype_b=int(DType.MXE4M3),
        dtype_c=int(DType.BF16),
        tile_k=256,
        tile_n=8,
        mma_m=128,
        mma_n=8,
        epi_tile_n=8,
    )
    legacy = make_config(sm_arch=100, mma_k=32, **common)
    rubin = make_config(sm_arch=107, mma_k=64, **common)

    validate_config(legacy)
    validate_config(rubin)
    assert not legacy.uses_native_mx_fp4_operands
    assert rubin.uses_native_mx_fp4_operands
    assert (legacy.dtype_a_tma_bits, legacy.dtype_a_smem_bits) == (4, 8)
    assert (rubin.dtype_a_tma_bits, rubin.dtype_a_smem_bits) == (4, 4)
    assert legacy.dtype_b_smem_bits == rubin.dtype_b_smem_bits == 8
    assert legacy.num_bytes_a_per_stage == 2 * rubin.num_bytes_a_per_stage
    assert (
        _tma_format_for_dtype(legacy.dtype_a, legacy.uses_native_mx_fp4_operands)
        == cuda.TensorMapDataFormat.B4X16_P64
    )
    assert (
        _tma_format_for_dtype(rubin.dtype_a, rubin.uses_native_mx_fp4_operands)
        == cuda.TensorMapDataFormat.B4X16
    )
    assert _tma_box_k_for_dtype(legacy, legacy.dtype_a) == 128
    assert (
        _tma_box_k_for_dtype(rubin, rubin.dtype_a, rubin.uses_native_mx_fp4_operands)
        == 256
    )
    assert _tma_swizzle_for_fastest_dim_bytes(128) == cuda.TensorMapSwizzle.s128b
    assert _tma_swizzle_for_fastest_dim_bytes(64) == cuda.TensorMapSwizzle.s64b
    assert _native_mx_fp4_smem_desc_params(128, 128) == (8192, 512, 4)
    assert _native_mx_fp4_smem_desc_params(128, 256) == (16384, 1024, 2)
    assert _native_mx_fp4_smem_desc_params(128, 512) == (16384, 1024, 2)

    # K64 consumes two adjacent block-32 scales per instruction but retains
    # the same packed vec4 group and total TMEM footprint as K32.
    assert tmem_sf_vec4_groups_per_mma(32, 32) == 1
    assert tmem_sf_vec4_groups_per_mma(64, 32) == 1
    assert tmem_sf_cols(128, 256, 32, 32) == tmem_sf_cols(128, 256, 32, 64)


@pytest.mark.cpu_only
@pytest.mark.timeout(180)
def test_trtllm_equiv_task_manager_smem_matches_kernel_allocation():
    """Cover the configuration in run_prims_ts_fp4_fc2_trtllm_equiv.sh."""
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        BatchMode,
        BiasType,
        CtaRasterOrder,
        DType,
        SfLayout,
        TileScheduler,
        make_config,
    )
    from flashinfer.prims_ts.batched_gemm.batched_gemm_kernel import (
        _build_schedule_validate,
        _make_tmem_ptr_smem_allocation,
    )

    cfg = make_config(
        route_act=0,
        act_kind=0,
        dtype_a=int(DType.E2M1),
        dtype_b=int(DType.E2M1),
        dtype_c=int(DType.BF16),
        sf_layout_a=int(SfLayout.R128c4),
        sf_layout_b=int(SfLayout.R128c4),
        sf_layout_c=int(SfLayout.R128c4),
        umma_cta_count=2,
        tile_n=256,
        tile_k=256,
        mma_m=256,
        mma_n=256,
        mma_k=64,
        epi_tile_n=64,
        tile_scheduler=int(TileScheduler.PERSISTENT),
        cta_raster_order=int(CtaRasterOrder.ALONG_N),
        batch_mode=int(BatchMode.BATCH_M),
        transpose_mma_output=0,
        num_stages_a=5,
        num_stages_b=5,
        num_stages_smem_sfa=6,
        num_stages_smem_sfb=6,
        num_stages_tmem_acc=1,
        num_stages_tmem_sfa=5,
        num_stages_tmem_sfb=1,
        num_stages_c_smem=1,
        use_tma_oob_opt=1,
        use_tma_store=1,
        use_unroll_loop_2x_for_mma=0,
        fuse_sf_copy_to_mma=1,
        use_max_tmem_overlap=1,
        use_early_exit=1,
        use_pdl=0,
        use_per_token_sf_b=0,
        per_token_sf_dtype=int(DType.FP32),
        bias_type=int(BiasType.NONE),
        epilogue_regs=176,
        mma_regs=80,
        load_regs=80,
        load_sf_regs=80,
        copy_sf_regs=80,
        workid_regs=80,
        padding_regs=80,
        tmem_ldst_max_num_regs=32,
    )
    _, _, smem_allocator, _ = _build_schedule_validate(cfg, num_k_tiles=16)

    assert smem_allocator.total_smem_bytes + smem_allocator.barrier_smem_bytes == 217616
    assert [
        (r.name, r.pipeline_config.num_stages)
        for r in smem_allocator._barrier_resources
    ] == [
        ("SmemA", 5),
        ("SmemB", 5),
        ("SmemSfA", 6),
        ("SmemSfB", 6),
        ("WorkQueue", 3),
        ("TmemC", 1),
        ("WorkThrottle", 3),
    ]

    tmem_ptr = _make_tmem_ptr_smem_allocation()
    assert (tmem_ptr.size_bytes, tmem_ptr.alignment, tmem_ptr.count) == (8, 8, 1)


@pytest.mark.cpu_only
@pytest.mark.timeout(180)
def test_sm107_task_manager_smem_matches_kernel_capacity_check():
    """Validation includes the standalone SMEM slots allocated by the kernel."""
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        ActKind,
        BatchMode,
        DType,
        RouteImpl,
        SfLayout,
        TileScheduler,
    )
    from flashinfer.prims_ts.batched_gemm.batched_gemm_kernel import (
        build_batched_gemm_task_manager,
    )

    overrides = dict(
        sm_arch=107,
        route_act=int(RouteImpl.TMA),
        route_sfs_act=int(RouteImpl.LDGSTS),
        act_kind=int(ActKind.SWIGLU),
        dtype_a=int(DType.E2M1),
        dtype_b=int(DType.E2M1),
        dtype_c=int(DType.E2M1),
        sf_layout_a=int(SfLayout.R128c4),
        sf_layout_b=int(SfLayout.LINEAR),
        sf_layout_c=int(SfLayout.R8c4),
        umma_cta_count=2,
        cluster_dim_x=2,
        cluster_dim_y=1,
        tile_n=64,
        tile_k=512,
        epi_tile_n=64,
        mma_m=256,
        mma_n=64,
        mma_k=128,
        tile_scheduler=int(TileScheduler.PERSISTENT),
        batch_mode=int(BatchMode.BATCH_N),
        transpose_mma_output=1,
        use_shuffled_matrix=1,
        num_stages_a=6,
        num_stages_b=6,
        num_stages_smem_sfa=17,
        num_stages_smem_sfb=5,
        num_stages_tmem_acc=2,
        num_stages_tmem_sfa=4,
        num_stages_tmem_sfb=10,
        num_stages_c_smem=2,
        num_stages_workid=3,
        use_tma_oob_opt=1,
        use_tma_store=1,
        use_unroll_loop_2x_for_mma=1,
        use_early_exit=1,
        use_clc_fast_drain=1,
        use_work_throttle=1,
        use_global_scales=1,
    )

    with pytest.raises(
        ValueError,
        match=r"SMEM usage \(334856 B",
    ):
        build_batched_gemm_task_manager(
            num_experts=96,
            num_tokens=1024,
            top_k=6,
            verbose=False,
            **overrides,
        )

    manager = build_batched_gemm_task_manager(
        num_experts=96,
        num_tokens=1024,
        top_k=6,
        verbose=False,
        **{**overrides, "use_work_throttle": 0},
    )
    assert (
        manager._smem_allocator.total_smem_bytes
        + manager._smem_allocator.barrier_smem_bytes
        == 334808
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("max_regs", "epi_tile_n", "expected_overlap_loads"),
    ((32, 64, 2), (64, 64, 1), (64, 128, 2)),
)
def test_non_swap_ht256_overlap_uses_epilogue_tile_width(
    max_regs, epi_tile_n, expected_overlap_loads
):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        SfLayout,
        TileScheduler,
        compute_warp_layout,
        make_config,
        validate_config,
    )

    cfg = make_config(
        umma_cta_count=2,
        tile_n=256,
        tile_k=256,
        epi_tile_m=128,
        epi_tile_n=epi_tile_n,
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

    assert cfg.warp_layout.num_epilogue_warps == 4
    assert cfg.non_swap_tmem_overlap_loads == expected_overlap_loads
    max_swizzled_cols = 128 * 8 // cfg.dtype_c_bits
    assert cfg.non_swap_tma_store_cols == min(cfg.epi_tile_n, max_swizzled_cols)
    assert cfg.epi_tile_n % cfg.non_swap_tma_store_cols == 0
    tma_store_bytes = cfg.tile_m * cfg.non_swap_tma_store_cols * cfg.dtype_c_bits // 8
    assert cfg.num_bytes_c_per_stage % tma_store_bytes == 0


@pytest.mark.cpu_only
@pytest.mark.timeout(240)
def test_schedule_checker_reports_no_persistent_c_scratch_ab_alias_race():
    """Persistent multi-stage work IDs keep C scratch separate from A/B.

    This uses a reduced BF16 K=tileK schedule to keep the checker tractable:
    five tasks and two persistent work-tile iterations. The production config
    disables C-scratch/A-B aliasing for multi-stage work-id scheduling, and the
    checker should not report the aliasing race in that schedule.
    """
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        DType,
        TileScheduler,
        make_config,
    )
    from flashinfer.prims_ts.batched_gemm.batched_gemm_kernel import (
        _build_schedule_validate,
    )
    from cutlass.experimental.task_scheduling.exhaustive_checker import (
        build_alias_info,
        check_all_interleavings,
    )
    from cutlass.experimental.task_scheduling.resources import WorkQueue

    cfg = make_config(
        tile_scheduler=int(TileScheduler.PERSISTENT),
        num_stages_workid=3,
        dtype_a=int(DType.BF16),
        dtype_b=int(DType.BF16),
        dtype_c=int(DType.BF16),
        tile_k=64,
        mma_k=16,
        tile_n=16,
        use_tma_store=1,
        use_early_exit=1,
    )
    assert not cfg.aliases_c_scratch_with_ab

    tasks, _dep_graph, _smem_allocator, _tmem_allocator = _build_schedule_validate(
        cfg,
        num_k_tiles=1,
    )
    assert len(tasks) == 5

    resources = []
    seen = set()
    for task in tasks:
        for resource in task.src_resources + task.dst_resources:
            if id(resource) not in seen:
                seen.add(id(resource))
                resources.append(resource)

    alias_map, prod_map, cons_map, overlap_descs = build_alias_info(resources)
    resource_by_id = {id(resource): resource.name for resource in resources}
    assert not any(
        resource_by_id.get(resource_id) == "SmemA"
        and any(resource_by_id.get(alias_id) == "GmemC" for alias_id in aliases)
        for resource_id, aliases in alias_map.items()
    )

    num_tiles = (
        2 if any(isinstance(resource, WorkQueue) for resource in resources) else 1
    )
    result = check_all_interleavings(
        tasks,
        alias_map=alias_map,
        prod_alias_map=prod_map,
        cons_alias_map=cons_map,
        overlap_descs=overlap_descs,
        num_tiles=num_tiles,
        max_states=1_000_000,
        verbose=False,
    )

    assert result.is_safe
    assert not any(
        race.writer_task == "LoadATask"
        and race.writer_resource == "SmemA"
        and race.victim_task == "EpilogueTask0"
        and race.victim_resource == "GmemC"
        for race in result.race_states
    )


@pytest.mark.cpu_only
def test_validation_accepts_mma_unroll():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        DType,
        make_config,
        validate_config,
    )

    cfg = make_config(
        dtype_a=int(DType.BF16),
        dtype_b=int(DType.BF16),
        dtype_c=int(DType.BF16),
        tile_k=64,
        mma_k=16,
        tile_n=16,
        use_unroll_loop_2x_for_mma=1,
    )
    validate_config(cfg)


@pytest.mark.cpu_only
def test_validation_accepts_safe_fused_operand_sf_pipeline():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import validate_config

    validate_config(_fused_operand_sf_config())


@pytest.mark.cpu_only
@pytest.mark.parametrize("operand", ("sfa", "sfb"))
def test_validation_rejects_fused_sf_copy_with_non_r128c4_layout(operand):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        BatchMode,
        DType,
        RouteImpl,
        SfLayout,
        make_config,
        validate_config,
    )

    overrides = {
        "sf_layout_a": int(SfLayout.R128c4),
        "sf_layout_b": int(SfLayout.R128c4),
    }
    overrides[f"sf_layout_{operand[-1]}"] = int(SfLayout.LINEAR)
    overrides.update(
        batch_mode=int(BatchMode.BATCH_M if operand == "sfa" else BatchMode.BATCH_N),
        route_act=int(RouteImpl.TMA),
        route_sfs_act=int(RouteImpl.TMA),
        transpose_mma_output=int(operand == "sfb"),
    )
    cfg = make_config(
        dtype_a=int(DType.E2M1),
        dtype_b=int(DType.E2M1),
        dtype_c=int(DType.BF16),
        fuse_sf_copy_to_mma=1,
        mma_k=64,
        mma_n=128,
        tile_k=512,
        tile_n=128,
        **overrides,
    )

    with pytest.raises(ValueError, match=f"R128c4 SMEM layout for {operand.upper()}"):
        validate_config(cfg)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("overrides", "match"),
    (
        (
            {"num_load_sfa_warps": 2},
            "num_load_sfa_warps supports one warp for a non-routed TMA loader",
        ),
        (
            {"num_load_sfb_warps": 2},
            "Fused B/SF loader requires equal warp counts",
        ),
        (
            {
                "num_stages_a": 1,
                "num_stages_b": 1,
                "num_stages_smem_sfb": 4,
            },
            "delayed SFB commit depth",
        ),
    ),
)
def test_validation_rejects_unsafe_fused_operand_sf_pipeline(overrides, match):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import validate_config

    with pytest.raises(ValueError, match=match):
        validate_config(_fused_operand_sf_config(**overrides))


@pytest.mark.cpu_only
@pytest.mark.parametrize(("num_load_sfb_warps", "rejected"), ((2, False), (4, True)))
def test_validation_rejects_tma_routed_sf_warps_without_gather4_rows(
    num_load_sfb_warps, rejected
):
    # Sweep v6 tile-8 FC1 hang: LoadSfB warps 2 and 3 own no gather4 row group.
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        make_config,
        validate_config,
    )
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
        _parse_overrides,
        build_arg_parser,
    )

    argv = (
        "--sm-arch 107 --route-act ldgsts --route-sfs-act tma "
        "--tile-scheduler persistent --act-kind relu2 --sf-layout-b linear "
        "--mma-k 128 --dtype-c e2m1 "
        "--num-stages-a 6 --num-stages-b 15 --num-stages-smem-sfa 6 "
        "--num-stages-smem-sfb 4 --num-stages-tmem-acc 10 --use-tma-store 1 "
        "--use-early-exit 1 --use-clc-fast-drain 1 "
        f"--num-load-sfb-warps {num_load_sfb_warps}"
    ).split()
    cfg = make_config(**_parse_overrides(build_arg_parser().parse_args(argv)))

    if rejected:
        with pytest.raises(ValueError, match="own at least one gather4 row group"):
            validate_config(cfg)
    else:
        validate_config(cfg)


@pytest.mark.cpu_only
def test_pdl_early_exit_derives_required_count_wait():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        DType,
        make_config,
        validate_config,
    )

    cfg = make_config(
        dtype_a=int(DType.BF16),
        dtype_b=int(DType.BF16),
        dtype_c=int(DType.BF16),
        mma_k=16,
        tile_k=64,
        use_early_exit=1,
        use_pdl=1,
    )

    validate_config(cfg)
    from flashinfer.prims_ts.batched_gemm.batched_gemm_kernel import (
        _pdl_wait_completed_before_tasks,
    )

    assert _pdl_wait_completed_before_tasks(cfg)


def _skip_early_exit_pdl_config(**overrides):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import DType, make_config

    values = dict(
        dtype_a=int(DType.BF16),
        dtype_b=int(DType.BF16),
        dtype_c=int(DType.BF16),
        mma_k=16,
        tile_k=64,
        use_early_exit=1,
        use_pdl=1,
        skip_early_exit_pdl=1,
    )
    values.update(overrides)
    return make_config(**values)


@pytest.mark.cpu_only
def test_skip_early_exit_pdl_drops_entry_wait():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import validate_config
    from flashinfer.prims_ts.batched_gemm.batched_gemm_kernel import (
        _pdl_wait_completed_before_tasks,
    )

    cfg = _skip_early_exit_pdl_config()
    validate_config(cfg)
    assert not _pdl_wait_completed_before_tasks(cfg)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "overrides",
    ({"use_pdl": 0}, {"use_early_exit": 0}),
    ids=("no_pdl", "no_early_exit"),
)
def test_skip_early_exit_pdl_requires_pdl_and_early_exit(overrides):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import validate_config

    with pytest.raises(ValueError, match="skip_early_exit_pdl requires"):
        validate_config(_skip_early_exit_pdl_config(**overrides))


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("dtype_c_name", "match"),
    (
        ("FP32", "Unsupported dtype_c output store"),
        ("E4M3", "dtype_c=e4m3 plain FP8 output requires dtype_a=dtype_b=e4m3"),
    ),
)
def test_validation_rejects_unsupported_plain_output_store_dtype(dtype_c_name, match):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        DType,
        make_config,
        validate_config,
    )

    cfg = make_config(
        dtype_a=int(DType.BF16),
        dtype_b=int(DType.BF16),
        dtype_c=int(getattr(DType, dtype_c_name)),
        tile_k=64,
        mma_k=16,
        tile_n=16,
    )
    with pytest.raises(ValueError, match=match):
        validate_config(cfg)


@pytest.mark.cpu_only
def test_runner_uses_fp16_for_fp16_plain_output_store_dtype():
    import cutlass
    import torch

    from flashinfer.prims_ts.batched_gemm import batched_gemm_run
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        DType,
        make_config,
        validate_config,
    )

    cfg = make_config(
        dtype_a=int(DType.BF16),
        dtype_b=int(DType.BF16),
        dtype_c=int(DType.FP16),
        tile_k=64,
        mma_k=16,
        tile_n=16,
    )

    validate_config(cfg)
    assert batched_gemm_run._plain_output_torch_dtype(cfg) == torch.float16
    assert batched_gemm_run._plain_output_cutlass_dtype(cfg) == cutlass.Float16


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "act_kind_name, expected",
    (
        ("NONE", False),
        ("SWIGLU", True),
        ("GEGLU", True),
        ("RELU2", False),
        ("SILU", False),
        ("SITU", True),
    ),
)
def test_runner_gated_activation_classification_matches_kernel(act_kind_name, expected):
    from flashinfer.prims_ts.batched_gemm import batched_gemm_run
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import ActKind

    assert (
        batched_gemm_run._is_gated_act_kind(int(getattr(ActKind, act_kind_name)))
        is expected
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "batch_mode_name, act_kind_name, expected_shape",
    (
        ("BATCH_N", "NONE", (128, 64)),
        ("BATCH_M", "NONE", (128, 64)),
        ("BATCH_N", "SWIGLU", (64, 64)),
        ("BATCH_M", "SWIGLU", (128, 32)),
    ),
)
def test_runner_logical_output_shape_for_gated_activation_abi(
    batch_mode_name, act_kind_name, expected_shape
):
    from flashinfer.prims_ts.batched_gemm import batched_gemm_run
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        ActKind,
        BatchMode,
        make_config,
    )

    batch_mode = getattr(BatchMode, batch_mode_name)
    cfg = make_config(
        batch_mode=int(batch_mode),
        transpose_mma_output=int(batch_mode == BatchMode.BATCH_N),
        act_kind=int(getattr(ActKind, act_kind_name)),
    )
    is_gated = batched_gemm_run._is_gated_act_kind(cfg.act_kind)
    logical_shape = batched_gemm_run._logical_output_shape(cfg, 128, 64, is_gated)

    assert logical_shape == expected_shape


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "stage_name",
    (
        "num_stages_a",
        "num_stages_b",
        "num_stages_tmem_acc",
    ),
)
def test_validation_rejects_zero_required_pipeline_stage_counts(stage_name):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        DType,
        make_config,
        validate_config,
    )

    kwargs = dict(
        dtype_a=int(DType.BF16),
        dtype_b=int(DType.BF16),
        dtype_c=int(DType.BF16),
        tile_k=64,
        mma_k=16,
        tile_n=16,
    )
    kwargs[stage_name] = 0
    cfg = make_config(**kwargs)

    with pytest.raises(ValueError, match=f"{stage_name} must be positive"):
        validate_config(cfg)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("num_stages_a", "num_stages_b"),
    ((3, 4), (4, 3)),
)
def test_validation_rejects_unequal_2cta_gather_pipeline_stages(
    num_stages_a, num_stages_b
):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        DType,
        make_config,
        validate_config,
    )

    cfg = make_config(
        umma_cta_count=2,
        route_act=2,
        dtype_a=int(DType.BF16),
        dtype_b=int(DType.BF16),
        dtype_c=int(DType.BF16),
        tile_k=128,
        mma_k=16,
        tile_n=64,
        mma_n=64,
        mma_m=256,
        num_stages_a=num_stages_a,
        num_stages_b=num_stages_b,
    )

    with pytest.raises(
        ValueError, match="2-CTA gather requires num_stages_a == num_stages_b"
    ):
        validate_config(cfg)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "stage_name",
    (
        "num_stages_c_smem",
        "num_stages_smem_sfa",
        "num_stages_smem_sfb",
        "num_stages_tmem_sfa",
        "num_stages_tmem_sfb",
        "num_stages_workid",
    ),
)
def test_validation_accepts_zero_inactive_pipeline_stage_counts(stage_name):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        DType,
        make_config,
        validate_config,
    )

    kwargs = dict(
        dtype_a=int(DType.BF16),
        dtype_b=int(DType.BF16),
        dtype_c=int(DType.BF16),
        tile_k=64,
        mma_k=16,
        tile_n=16,
    )
    kwargs[stage_name] = 0
    cfg = make_config(**kwargs)

    validate_config(cfg)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("stage_name", "overrides"),
    (
        ("num_stages_c_smem", {"use_tma_store": 1}),
        ("num_stages_smem_sfa", {}),
        ("num_stages_smem_sfb", {}),
        ("num_stages_tmem_sfa", {}),
        ("num_stages_tmem_sfb", {}),
    ),
)
def test_validation_rejects_zero_active_sf_and_store_stage_counts(
    stage_name, overrides
):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        DType,
        SfLayout,
        make_config,
        validate_config,
    )

    kwargs = dict(
        dtype_a=int(DType.E2M1),
        dtype_b=int(DType.E2M1),
        dtype_c=int(DType.BF16),
        tile_k=512,
        mma_k=64,
        tile_n=8,
        epi_tile_n=8,
        sf_layout_a=int(SfLayout.R128c4),
        sf_layout_b=int(SfLayout.LINEAR),
    )
    kwargs.update(overrides)
    kwargs[stage_name] = 0
    cfg = make_config(**kwargs)

    with pytest.raises(ValueError, match=f"{stage_name} must be positive"):
        validate_config(cfg)


@pytest.mark.cpu_only
def test_validation_rejects_zero_persistent_workid_stages():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        DType,
        TileScheduler,
        make_config,
        validate_config,
    )

    cfg = make_config(
        dtype_a=int(DType.BF16),
        dtype_b=int(DType.BF16),
        dtype_c=int(DType.BF16),
        tile_k=64,
        mma_k=16,
        tile_n=16,
        tile_scheduler=int(TileScheduler.PERSISTENT),
        num_stages_workid=0,
    )

    with pytest.raises(ValueError, match="num_stages_workid must be positive"):
        validate_config(cfg)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("epi_tile_n", "expected"),
    (
        (0, "positive"),
        (12, "multiple of 8"),
        (24, "divide tile_n"),
    ),
)
def test_validation_rejects_invalid_epi_tile_n(epi_tile_n, expected):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        DType,
        make_config,
        validate_config,
    )

    cfg = make_config(
        dtype_a=int(DType.BF16),
        dtype_b=int(DType.BF16),
        dtype_c=int(DType.BF16),
        tile_k=64,
        mma_k=16,
        tile_n=64,
        mma_n=64,
        epi_tile_n=epi_tile_n,
    )
    with pytest.raises(ValueError, match=expected):
        validate_config(cfg)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("overrides", "expected"),
    (
        ({"mma_m": 96}, "mma_m must be 64/128/256"),
        ({"mma_n": 12}, "mma_n must be 8/16/32/64/128/192/256"),
    ),
)
def test_validation_rejects_invalid_mma_shape(overrides, expected):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        DType,
        make_config,
        validate_config,
    )

    kwargs = dict(
        dtype_a=int(DType.BF16),
        dtype_b=int(DType.BF16),
        dtype_c=int(DType.BF16),
        tile_k=64,
        mma_k=16,
        tile_n=64,
        mma_m=128,
        mma_n=64,
        epi_tile_n=64,
    )
    kwargs.update(overrides)
    cfg = make_config(**kwargs)
    with pytest.raises(ValueError, match=expected):
        validate_config(cfg)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("overrides", "expected"),
    (
        ({"tile_n": 16, "mma_n": 32}, "tile_n must be a multiple of mma_n"),
        ({"mma_m": 256}, "cluster-wide tile_m must be a multiple of mma_m"),
        (
            {"umma_cta_count": 2, "mma_n": 8},
            "umma_cta_count=2 requires mma_n >= 16",
        ),
        ({"umma_cta_count": 2}, "umma_cta_count=2 requires mma_m=256"),
    ),
)
def test_validation_rejects_incompatible_tile_and_mma_shapes(overrides, expected):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        DType,
        make_config,
        validate_config,
    )

    kwargs = dict(
        dtype_a=int(DType.BF16),
        dtype_b=int(DType.BF16),
        dtype_c=int(DType.BF16),
        tile_k=64,
        mma_k=16,
        tile_n=64,
        mma_m=128,
        mma_n=64,
        epi_tile_n=16,
    )
    kwargs.update(overrides)
    cfg = make_config(**kwargs)
    with pytest.raises(ValueError, match=expected):
        validate_config(cfg)


@pytest.mark.cpu_only
def test_validation_rejects_nvfp4_mma_m64():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        DType,
        make_config,
        validate_config,
    )

    cfg = make_config(
        dtype_a=int(DType.E2M1),
        dtype_b=int(DType.E2M1),
        dtype_c=int(DType.BF16),
        tile_k=256,
        mma_k=64,
        tile_n=16,
        mma_m=64,
        mma_n=16,
        epi_tile_n=16,
    )
    with pytest.raises(ValueError, match="NVFP4 MMA does not support mma_m=64"):
        validate_config(cfg)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("batch_mode_name", "transpose_mma_output"),
    (("BATCH_N", 0), ("BATCH_M", 1)),
)
def test_validation_rejects_transpose_incompatible_with_batch_mode(
    batch_mode_name, transpose_mma_output
):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        BatchMode,
        make_config,
        validate_config,
    )

    cfg = make_config(
        batch_mode=int(getattr(BatchMode, batch_mode_name)),
        transpose_mma_output=transpose_mma_output,
    )
    with pytest.raises(ValueError, match="transpose_mma_output must match.*swap_ab"):
        validate_config(cfg)


@pytest.mark.cpu_only
def test_validation_rejects_non_binary_transpose_mma_output():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        make_config,
        validate_config,
    )

    cfg = make_config(transpose_mma_output=2)
    with pytest.raises(ValueError, match="transpose_mma_output must be 0 or 1"):
        validate_config(cfg)


@pytest.mark.cpu_only
def test_validation_rejects_tile_k_not_multiple_of_mma_k():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        DType,
        make_config,
        validate_config,
    )

    cfg = make_config(
        dtype_a=int(DType.BF16),
        dtype_b=int(DType.BF16),
        dtype_c=int(DType.BF16),
        tile_k=20,
        mma_k=16,
        tile_n=16,
    )
    with pytest.raises(ValueError, match="tile_k must be a multiple of mma_k"):
        validate_config(cfg)


@pytest.mark.cpu_only
@pytest.mark.parametrize("tile_k", (64, 128, 320))
def test_validation_rejects_nvfp4_tile_k_not_multiple_of_256(tile_k):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        DType,
        make_config,
        validate_config,
    )

    cfg = make_config(
        dtype_a=int(DType.E2M1),
        dtype_b=int(DType.E2M1),
        dtype_c=int(DType.BF16),
        tile_k=tile_k,
        mma_k=64,
        tile_n=16,
    )
    with pytest.raises(ValueError, match="NVFP4 requires tile_k.*multiple of 256"):
        validate_config(cfg)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "case",
    (
        "swap_explicit_none",
        "swap_auto_none",
        "swap_no_activation_route",
        "non_swap_explicit_none",
    ),
)
def test_validation_rejects_non_routed_linear_input_scale_factors(case):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        BatchMode,
        DType,
        RouteImpl,
        SfLayout,
        make_config,
        validate_config,
    )

    values = dict(
        batch_mode=int(BatchMode.BATCH_N),
        dtype_a=int(DType.E2M1),
        dtype_b=int(DType.E2M1),
        dtype_c=int(DType.BF16),
        mma_k=64,
        route_act=int(RouteImpl.LDGSTS),
        route_sfs_act=int(RouteImpl.NONE),
        sf_layout_b=int(SfLayout.LINEAR),
        tile_k=512,
        tile_n=64,
        transpose_mma_output=1,
    )
    expected_operand = "b"
    if case == "swap_auto_none":
        values["route_sfs_act"] = int(RouteImpl.AUTO)
    elif case == "swap_no_activation_route":
        values["route_act"] = int(RouteImpl.NONE)
    elif case == "non_swap_explicit_none":
        values.update(
            batch_mode=int(BatchMode.BATCH_M),
            sf_layout_a=int(SfLayout.LINEAR),
            sf_layout_b=int(SfLayout.R8c4),
            transpose_mma_output=0,
        )
        expected_operand = "a"

    cfg = make_config(**values)
    with pytest.raises(
        ValueError,
        match=rf"sf_layout_{expected_operand}=linear requires routed input scale factors",
    ):
        validate_config(cfg)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "case",
    (
        "ldgsts_routed_linear",
        "tma_routed_linear",
        "preexpanded_8x4",
        "casta_linear_b",
    ),
)
def test_validation_accepts_supported_linear_and_preexpanded_sf_layouts(case):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        BatchMode,
        DType,
        RouteImpl,
        SfLayout,
        make_config,
        validate_config,
    )

    values = dict(
        batch_mode=int(BatchMode.BATCH_N),
        dtype_a=int(DType.E2M1),
        dtype_b=int(DType.E2M1),
        dtype_c=int(DType.BF16),
        mma_k=64,
        route_act=int(RouteImpl.LDGSTS),
        route_sfs_act=int(RouteImpl.LDGSTS),
        sf_layout_b=int(SfLayout.LINEAR),
        tile_k=512,
        tile_n=64,
        transpose_mma_output=1,
    )
    if case == "tma_routed_linear":
        values.update(
            route_act=int(RouteImpl.TMA),
            route_sfs_act=int(RouteImpl.TMA),
        )
    elif case == "preexpanded_8x4":
        values.update(
            route_sfs_act=int(RouteImpl.NONE),
            sf_layout_b=int(SfLayout.R8c4),
        )
    elif case == "casta_linear_b":
        values.update(
            batch_mode=int(BatchMode.BATCH_N),
            dtype_a=int(DType.MXE2M1),
            dtype_b=int(DType.BF16),
            mma_k=16,
            num_stages_a=1,
            num_stages_tmem_acc=1,
            route_act=int(RouteImpl.NONE),
            route_sfs_act=int(RouteImpl.AUTO),
            tile_k=256,
            tile_n=16,
            transpose_mma_output=1,
        )

    validate_config(make_config(**values))


@pytest.mark.cpu_only
def test_validation_rejects_bf16_kbox_tma_a_tile_k_not_multiple_of_64():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        DType,
        make_config,
        validate_config,
    )

    cfg = make_config(
        dtype_a=int(DType.BF16),
        dtype_b=int(DType.BF16),
        dtype_c=int(DType.BF16),
        tile_k=16,
        mma_k=16,
        tile_n=16,
    )
    assert cfg.use_bf16_kbox_tma_a
    with pytest.raises(ValueError, match="BF16 k-box TMA A path"):
        validate_config(cfg)


@pytest.mark.cpu_only
def test_validation_rejects_casta_tile_k_not_256():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        DType,
        make_config,
        validate_config,
    )

    cfg = make_config(
        dtype_a=int(DType.MXE2M1),
        dtype_b=int(DType.BF16),
        dtype_c=int(DType.BF16),
        tile_k=128,
        mma_k=16,
        tile_n=16,
    )
    assert cfg.has_cast_a
    with pytest.raises(ValueError, match="CastA MXFP4 input requires tile_k=256"):
        validate_config(cfg)


@pytest.mark.cpu_only
@pytest.mark.parametrize("dtype_c_name", ("BF16", "FP16"))
def test_validation_accepts_casta_plain_output_store(dtype_c_name):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        DType,
        make_config,
        validate_config,
    )

    cfg = make_config(
        dtype_a=int(DType.MXE2M1),
        dtype_b=int(DType.BF16),
        dtype_c=int(getattr(DType, dtype_c_name)),
        tile_k=256,
        mma_k=16,
        tile_n=16,
        num_stages_a=1,
        num_stages_tmem_acc=1,
    )

    assert cfg.has_cast_a
    validate_config(cfg)


@pytest.mark.cpu_only
def test_validation_rejects_casta_output_quantization():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        ActKind,
        BatchMode,
        DType,
        make_config,
        validate_config,
    )

    cfg = make_config(
        batch_mode=int(BatchMode.BATCH_N),
        act_kind=int(ActKind.SWIGLU),
        dtype_a=int(DType.MXE2M1),
        dtype_b=int(DType.BF16),
        dtype_c=int(DType.E2M1),
        tile_k=256,
        mma_k=16,
        tile_n=16,
    )

    assert cfg.has_cast_a
    with pytest.raises(
        ValueError, match="CastA MXFP4 input supports dtype_c=bf16/fp16"
    ):
        validate_config(cfg)


@pytest.mark.cpu_only
def test_validation_accepts_silu_activation():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        ActKind,
        DType,
        make_config,
        validate_config,
    )

    cfg = make_config(
        dtype_a=int(DType.BF16),
        dtype_b=int(DType.BF16),
        dtype_c=int(DType.BF16),
        tile_k=64,
        mma_k=16,
        tile_n=16,
        act_kind=int(ActKind.SILU),
    )
    validate_config(cfg)


@pytest.mark.cpu_only
def test_clustered_swap_ab_ldgsts_splits_b_smem_across_ctas():
    """Each CTA stages its routed B rows and uses the matching K-group stride."""
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        DType,
        RouteImpl,
        make_config,
    )

    cfg = make_config(
        umma_cta_count=2,
        route_act=int(RouteImpl.LDGSTS),
        dtype_a=int(DType.E2M1),
        dtype_b=int(DType.E2M1),
        dtype_c=int(DType.E2M1),
        tile_n=128,
        mma_n=128,
        epi_tile_n=32,
        tile_k=512,
        mma_k=64,
    )

    assert cfg.is_swap_ab
    assert cfg.has_gather
    assert cfg.split_b_across_ctas
    assert cfg.num_bytes_b_per_stage == 32_768
    assert cfg.num_bytes_b_smem_per_stage == 16_384


@pytest.mark.cpu_only
@pytest.mark.parametrize("tile_n", (128, 256))
def test_clustered_swap_ab_tma_gather_splits_b_smem_across_ctas(tile_n):
    """TMA gather4 stages the same per-CTA B slice as the LDGSTS gather."""
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        DType,
        RouteImpl,
        make_config,
    )

    cfg = make_config(
        sm_arch=107,
        umma_cta_count=2,
        route_act=int(RouteImpl.TMA),
        route_sfs_act=int(RouteImpl.LDGSTS),
        dtype_a=int(DType.E2M1),
        dtype_b=int(DType.E2M1),
        dtype_c=int(DType.E2M1),
        tile_n=tile_n,
        mma_m=256,
        mma_n=tile_n,
        epi_tile_n=64,
        tile_k=256,
        mma_k=128,
        fuse_sf_copy_to_mma=1,
        use_max_tmem_overlap=0,
    )

    assert cfg.is_swap_ab
    assert cfg.has_tma_route
    assert not cfg.use_tile256_tmem_overlap
    assert cfg.split_b_across_ctas
    assert (
        cfg.num_bytes_b_smem_per_stage
        == cfg.num_bytes_b_per_stage // cfg.umma_cta_count
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("kind", "tile_n", "tile_k", "mma_k", "expected"),
    (
        ("mx", 128, 128, 32, ((0, 0),)),
        ("mx", 128, 256, 32, ((0, 0), (32, 4))),
        ("mx", 128, 512, 32, ((0, 0), (32, 4), (64, 8), (96, 12))),
        ("mx", 256, 128, 32, ((0, 0), (32, 4))),
        ("mx", 256, 256, 32, ((0, 0), (32, 8), (64, 4), (96, 12))),
        (
            "mx",
            256,
            512,
            32,
            (
                (0, 0),
                (32, 8),
                (64, 16),
                (96, 24),
                (128, 4),
                (160, 12),
                (192, 20),
                (224, 28),
            ),
        ),
        ("nvfp4", 128, 256, 64, ((0, 0), (32, 4), (64, 8), (96, 12))),
        (
            "nvfp4",
            256,
            256,
            64,
            (
                (0, 0),
                (32, 8),
                (64, 16),
                (96, 24),
                (128, 4),
                (160, 12),
                (192, 20),
                (224, 28),
            ),
        ),
        ("nvfp4", 128, 256, 128, ((0, 0), (32, 4), (64, 8), (96, 12))),
        (
            "nvfp4",
            256,
            256,
            128,
            (
                (0, 0),
                (32, 8),
                (64, 16),
                (96, 24),
                (128, 4),
                (160, 12),
                (192, 20),
                (224, 28),
            ),
        ),
    ),
)
def test_r128c4_sfb_utccp_offsets_cover_tmem(kind, tile_n, tile_k, mma_k, expected):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        DType,
        SfLayout,
        TMEM_SF_UTCCP_COLS_PER_COPY,
        make_config,
        sfb_r128c4_utccp_offsets,
    )

    dtype_a, dtype_b = (
        (DType.MXE2M1, DType.MXE4M3) if kind == "mx" else (DType.E2M1, DType.E2M1)
    )
    cfg = make_config(
        sm_arch=107,
        dtype_a=int(dtype_a),
        dtype_b=int(dtype_b),
        dtype_c=int(DType.BF16),
        sf_layout_a=int(SfLayout.R128c4),
        sf_layout_b=int(SfLayout.R128c4),
        tile_n=tile_n,
        mma_n=tile_n,
        epi_tile_n=64,
        tile_k=tile_k,
        mma_k=mma_k,
    )
    num_copies = cfg.tmem_sfb_cols // TMEM_SF_UTCCP_COLS_PER_COPY
    offsets = tuple(
        sfb_r128c4_utccp_offsets(cfg, copy_idx) for copy_idx in range(num_copies)
    )

    assert cfg.uses_n_major_r128c4_sfb_utccp
    assert offsets == expected
    written_cols = [
        col
        for _, tmem_delta in offsets
        for col in range(tmem_delta, tmem_delta + TMEM_SF_UTCCP_COLS_PER_COPY)
    ]
    assert len(written_cols) == len(set(written_cols))
    assert set(written_cols) == set(range(cfg.tmem_sfb_cols))


@pytest.mark.cpu_only
def test_ldgsts_routed_sf_does_not_use_tma_descriptors():
    """Only the non-routed operand needs a TensorMap on the LDGSTS path."""
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        BatchMode,
        DType,
        RouteImpl,
        make_config,
        uses_routed_sfa_tma_desc,
        uses_routed_sfb_tma_desc,
        uses_sfa_tma_desc,
        uses_sfb_tma_desc,
    )

    common = dict(
        route_act=int(RouteImpl.TMA),
        route_sfs_act=int(RouteImpl.LDGSTS),
        dtype_a=int(DType.MXE4M3),
        dtype_b=int(DType.MXE4M3),
        dtype_c=int(DType.MXE4M3),
        tile_k=128,
        mma_k=32,
    )

    swap_ab_cfg = make_config(**common)
    assert swap_ab_cfg.has_routed_sfs
    assert swap_ab_cfg.uses_ldgsts_routed_sfs
    assert not uses_routed_sfa_tma_desc(swap_ab_cfg)
    assert not uses_routed_sfb_tma_desc(swap_ab_cfg)
    assert uses_sfa_tma_desc(swap_ab_cfg)
    assert not uses_sfb_tma_desc(swap_ab_cfg)

    non_swap_ab_cfg = make_config(
        **common,
        batch_mode=int(BatchMode.BATCH_M),
        transpose_mma_output=0,
    )
    assert non_swap_ab_cfg.has_routed_sfs
    assert non_swap_ab_cfg.uses_ldgsts_routed_sfs
    assert not uses_routed_sfa_tma_desc(non_swap_ab_cfg)
    assert not uses_routed_sfb_tma_desc(non_swap_ab_cfg)
    assert not uses_sfa_tma_desc(non_swap_ab_cfg)
    assert uses_sfb_tma_desc(non_swap_ab_cfg)

    tma_swap_ab_cfg = make_config(
        **{**common, "route_sfs_act": int(RouteImpl.TMA), "tile_k": 512},
    )
    assert uses_routed_sfb_tma_desc(tma_swap_ab_cfg)
    assert not uses_routed_sfa_tma_desc(tma_swap_ab_cfg)
    assert uses_sfa_tma_desc(tma_swap_ab_cfg)
    assert uses_sfb_tma_desc(tma_swap_ab_cfg)

    tma_non_swap_ab_cfg = make_config(
        **{**common, "route_sfs_act": int(RouteImpl.TMA), "tile_k": 512},
        batch_mode=int(BatchMode.BATCH_M),
        transpose_mma_output=0,
    )
    assert uses_routed_sfa_tma_desc(tma_non_swap_ab_cfg)
    assert not uses_routed_sfb_tma_desc(tma_non_swap_ab_cfg)
    assert uses_sfa_tma_desc(tma_non_swap_ab_cfg)
    assert uses_sfb_tma_desc(tma_non_swap_ab_cfg)

    non_routed_cfg = make_config(
        **{
            **common,
            "route_act": int(RouteImpl.NONE),
            "route_sfs_act": int(RouteImpl.NONE),
        },
    )
    assert uses_sfa_tma_desc(non_routed_cfg)
    assert uses_sfb_tma_desc(non_routed_cfg)


@pytest.mark.cpu_only
def test_validation_accepts_nvfp4_tile256_split_sfb_descriptor():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        DType,
        RouteImpl,
        SfLayout,
        make_config,
        validate_config,
    )

    cfg = make_config(
        sm_arch=107,
        umma_cta_count=2,
        route_act=int(RouteImpl.NONE),
        route_sfs_act=int(RouteImpl.NONE),
        dtype_a=int(DType.E2M1),
        dtype_b=int(DType.E2M1),
        dtype_c=int(DType.BF16),
        sf_layout_a=int(SfLayout.R128c4),
        sf_layout_b=int(SfLayout.R128c4),
        tile_n=256,
        mma_m=256,
        mma_n=256,
        epi_tile_n=64,
        tile_k=256,
        mma_k=128,
        fuse_sf_copy_to_mma=1,
        use_max_tmem_overlap=0,
    )

    validate_config(cfg)
    assert cfg.uses_split_r128c4_sfb_tma_desc


@pytest.mark.cpu_only
def test_validation_accepts_tile256_routed_sfb_tma_descriptor():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        DType,
        RouteImpl,
        SfLayout,
        make_config,
        validate_config,
    )

    cfg = make_config(
        sm_arch=107,
        umma_cta_count=2,
        cluster_dim_x=2,
        route_act=int(RouteImpl.TMA),
        route_sfs_act=int(RouteImpl.TMA),
        dtype_a=int(DType.E2M1),
        dtype_b=int(DType.E2M1),
        dtype_c=int(DType.BF16),
        sf_layout_a=int(SfLayout.R128c4),
        sf_layout_b=int(SfLayout.LINEAR),
        tile_n=256,
        tile_k=512,
        mma_m=256,
        mma_n=256,
        mma_k=128,
        epi_tile_n=64,
        num_stages_a=3,
        num_stages_b=3,
        num_stages_smem_sfa=3,
        num_stages_smem_sfb=3,
        num_stages_tmem_acc=1,
        num_stages_tmem_sfa=3,
        num_stages_tmem_sfb=3,
        fuse_sf_copy_to_mma=0,
        use_max_tmem_overlap=0,
    )

    validate_config(cfg)


@pytest.mark.cpu_only
@pytest.mark.timeout(180)
def test_deepseek_fp8_uses_two_epilogue_warpgroups():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        DType,
        make_config,
    )
    from flashinfer.prims_ts.batched_gemm.batched_gemm_kernel import (
        _build_schedule_validate,
    )

    cfg = make_config(
        dtype_a=int(DType.E4M3),
        dtype_b=int(DType.E4M3),
        dtype_c=int(DType.E4M3),
        tile_n=8,
        tile_k=128,
        mma_m=64,
        mma_n=8,
        mma_k=32,
        epi_tile_n=8,
        use_deepseek_fp8=1,
    )
    tasks, _dep_graph, _smem_allocator, _tmem_allocator = _build_schedule_validate(
        cfg,
        num_k_tiles=1,
    )
    by_name = {task.name: task for task in tasks}

    assert by_name["EpilogueTask0DsFp8"].warp_idx == 0
    assert by_name["EpilogueTask0DsFp8"].num_warps == 8
    assert by_name["MmaTask0"].warp_idx == 8
    assert by_name["LoadATask"].warp_idx == 9
    assert by_name["LoadBTask"].warp_idx == 10
    assert by_name["LoadSfAbTask"].warp_idx == 11
    assert cfg.warp_layout.threads_per_cta == 384


@pytest.mark.cpu_only
@pytest.mark.timeout(180)
def test_persistent_deepseek_fp8_schedule_builds():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        DType,
        TileScheduler,
    )
    from flashinfer.prims_ts.batched_gemm.batched_gemm_kernel import (
        build_batched_gemm_task_manager,
    )

    task_manager = build_batched_gemm_task_manager(
        num_experts=2,
        num_tokens=256,
        top_k=1,
        tile_scheduler=int(TileScheduler.PERSISTENT),
        use_early_exit=1,
        dtype_a=int(DType.E4M3),
        dtype_b=int(DType.E4M3),
        dtype_c=int(DType.E4M3),
        tile_n=8,
        tile_k=128,
        mma_m=64,
        mma_n=8,
        mma_k=32,
        epi_tile_n=8,
        use_deepseek_fp8=1,
    )

    assert {task.name for task in task_manager.tasks} >= {
        "LoadSfAbTask",
        "EpilogueTask0DsFp8",
        "WorkScheduleTask",
    }


@pytest.mark.cpu_only
@pytest.mark.timeout(180)
@pytest.mark.parametrize(
    ("tile_n", "num_load_b_warps", "load_sfab_warp_idx", "padding_warp_idx"),
    (
        (32, 2, 12, 13),
        (128, 4, 14, 15),
    ),
)
def test_deepseek_fp8_tma_route_uses_generated_load_b_warp_count(
    tile_n,
    num_load_b_warps,
    load_sfab_warp_idx,
    padding_warp_idx,
):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        DType,
        RouteImpl,
        make_config,
    )
    from flashinfer.prims_ts.batched_gemm.batched_gemm_kernel import (
        _build_schedule_validate,
    )

    cfg = make_config(
        dtype_a=int(DType.E4M3),
        dtype_b=int(DType.E4M3),
        dtype_c=int(DType.E4M3),
        route_act=int(RouteImpl.TMA),
        tile_n=tile_n,
        tile_k=128,
        mma_m=64,
        mma_n=tile_n,
        mma_k=32,
        epi_tile_n=tile_n,
        num_stages_a=6,
        num_stages_b=6,
        num_stages_smem_sfa=6,
        num_stages_smem_sfb=6,
        num_stages_tmem_acc=1,
        use_deepseek_fp8=1,
        num_load_b_warps=num_load_b_warps,
    )
    tasks, _dep_graph, _smem_allocator, _tmem_allocator = _build_schedule_validate(
        cfg,
        num_k_tiles=1,
    )
    by_name = {task.name: task for task in tasks}

    assert by_name["EpilogueTask0DsFp8"].warp_idx == 0
    assert by_name["EpilogueTask0DsFp8"].num_warps == 8
    assert by_name["MmaTask0"].warp_idx == 8
    assert by_name["LoadATask"].warp_idx == 9
    assert by_name["LoadBTask"].warp_idx == 10
    assert by_name["LoadBTask"].num_warps == num_load_b_warps
    assert by_name["LoadSfAbTask"].warp_idx == load_sfab_warp_idx
    assert by_name["PaddingTask"].warp_idx == padding_warp_idx
    assert cfg.warp_layout.threads_per_cta == 512


@pytest.mark.cpu_only
def test_validation_accepts_clc_fast_drain_for_persistent_early_exit():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        DType,
        TileScheduler,
        make_config,
        validate_config,
    )

    cfg = make_config(
        dtype_a=int(DType.BF16),
        dtype_b=int(DType.BF16),
        dtype_c=int(DType.BF16),
        tile_k=64,
        mma_k=16,
        tile_n=16,
        tile_scheduler=int(TileScheduler.PERSISTENT),
        use_early_exit=1,
        use_clc_fast_drain=1,
    )
    validate_config(cfg)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("tile_scheduler", "use_early_exit"),
    ((0, 1), (1, 0)),
)
def test_validation_rejects_clc_fast_drain_without_persistent_early_exit(
    tile_scheduler, use_early_exit
):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        DType,
        make_config,
        validate_config,
    )

    cfg = make_config(
        dtype_a=int(DType.BF16),
        dtype_b=int(DType.BF16),
        dtype_c=int(DType.BF16),
        tile_k=64,
        mma_k=16,
        tile_n=16,
        tile_scheduler=tile_scheduler,
        use_early_exit=use_early_exit,
        use_clc_fast_drain=1,
    )
    with pytest.raises(ValueError, match="requires persistent scheduling"):
        validate_config(cfg)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "removed_option",
    (
        "skip_tmem_dealloc",
        "skip_tmem_dealloc_barrier",
        "skip_tmem_dealloc_instruction",
        "use_delayed_ldgsts_sfb_commit",
        "use_predicated_ldgsts_sfb",
        "skip_ldgsts_sfb_payload",
        "skip_copy_sfb_payload",
        "debug_print",
        "num_experts",
        "top_k",
        "static_num_k_tiles",
        "max_num_ctas_in_token_dim",
        "early_exit_max_token_ctas",
        "acc_dtype_bytes",
    ),
)
def test_config_rejects_removed_non_config_options(removed_option):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        DType,
        make_config,
    )

    with pytest.raises(TypeError, match=removed_option):
        make_config(
            dtype_a=int(DType.BF16),
            dtype_b=int(DType.BF16),
            dtype_c=int(DType.BF16),
            tile_k=64,
            mma_k=16,
            tile_n=16,
            **{removed_option: 1},
        )


@pytest.mark.cpu_only
def test_validation_rejects_ldg_plus_sts_routes():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        DType,
        make_config,
        validate_config,
    )

    cfg = make_config(
        dtype_a=int(DType.BF16),
        dtype_b=int(DType.BF16),
        dtype_c=int(DType.BF16),
        tile_k=64,
        mma_k=16,
        tile_n=16,
        route_act=3,
    )
    with pytest.raises(ValueError, match="Unsupported route"):
        validate_config(cfg)

    cfg = make_config(
        dtype_a=int(DType.BF16),
        dtype_b=int(DType.BF16),
        dtype_c=int(DType.BF16),
        tile_k=64,
        mma_k=16,
        tile_n=16,
        route_sfs_act=3,
    )
    with pytest.raises(ValueError, match="Unsupported route"):
        validate_config(cfg)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("batch_mode", "register_override"),
    (
        ("m", {"load_b_regs": 64}),
        ("n", {"load_a_regs": 64}),
        ("m", {"load_sfb_regs": 64}),
        ("n", {"load_sfa_regs": 64}),
    ),
)
def test_register_override_applies_to_non_routed_operand(batch_mode, register_override):
    """Canonical operand budgets apply independently of transport selection."""
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        BatchMode,
        DType,
        RouteImpl,
        make_config,
        validate_config,
    )

    is_batch_n = batch_mode == "n"
    cfg = make_config(
        dtype_a=int(DType.E2M1),
        dtype_b=int(DType.E2M1),
        dtype_c=int(DType.BF16),
        tile_n=16,
        tile_k=256,
        mma_m=128,
        mma_n=16,
        mma_k=64,
        umma_cta_count=1,
        batch_mode=int(BatchMode.BATCH_N if is_batch_n else BatchMode.BATCH_M),
        transpose_mma_output=int(is_batch_n),
        route_act=int(RouteImpl.TMA),
        route_sfs_act=int(RouteImpl.LDGSTS),
        **register_override,
    )

    validate_config(cfg)
    for name, budget in register_override.items():
        assert getattr(cfg.warp_layout, name) == budget


@pytest.mark.cpu_only
@pytest.mark.parametrize("batch_mode", ("m", "n"))
def test_validation_accepts_register_overrides_for_routed_operand(batch_mode):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        BatchMode,
        DType,
        RouteImpl,
        make_config,
        validate_config,
    )

    is_batch_n = batch_mode == "n"
    routed_registers = (
        {"load_b_regs": 64, "load_sfb_regs": 64}
        if is_batch_n
        else {"load_a_regs": 64, "load_sfa_regs": 64}
    )
    cfg = make_config(
        dtype_a=int(DType.E2M1),
        dtype_b=int(DType.E2M1),
        dtype_c=int(DType.BF16),
        tile_n=16,
        tile_k=256,
        mma_m=128,
        mma_n=16,
        mma_k=64,
        umma_cta_count=1,
        batch_mode=int(BatchMode.BATCH_N if is_batch_n else BatchMode.BATCH_M),
        transpose_mma_output=int(is_batch_n),
        route_act=int(RouteImpl.TMA),
        route_sfs_act=int(RouteImpl.LDGSTS),
        **routed_registers,
    )

    validate_config(cfg)


@pytest.mark.cpu_only
def test_validation_accepts_minimal_plain_fp8_per_token_sfb():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        ActKind,
        BatchMode,
        DType,
        RouteImpl,
        make_config,
        validate_config,
    )

    cfg = make_config(
        dtype_a=int(DType.E4M3),
        dtype_b=int(DType.E4M3),
        dtype_c=int(DType.E4M3),
        tile_k=32,
        mma_k=32,
        tile_n=16,
        route_act=int(RouteImpl.TMA),
        act_kind=int(ActKind.SWIGLU),
        use_per_token_sf_b=1,
        per_token_sf_dtype=int(DType.FP32),
    )
    validate_config(cfg)

    cfg = make_config(
        dtype_a=int(DType.E4M3),
        dtype_b=int(DType.E4M3),
        dtype_c=int(DType.E4M3),
        tile_k=32,
        mma_k=32,
        tile_n=16,
        route_act=int(RouteImpl.TMA),
        act_kind=int(ActKind.SWIGLU),
        use_per_token_sf_a=1,
        per_token_sf_dtype=int(DType.FP32),
    )
    validate_config(cfg)

    cfg = make_config(
        dtype_a=int(DType.E4M3),
        dtype_b=int(DType.E4M3),
        dtype_c=int(DType.E4M3),
        tile_k=32,
        mma_k=32,
        tile_n=16,
        route_act=int(RouteImpl.TMA),
        act_kind=int(ActKind.SWIGLU),
        use_per_token_sf_a=1,
        use_per_token_sf_b=1,
        per_token_sf_dtype=int(DType.FP32),
    )
    validate_config(cfg)

    cfg = make_config(
        dtype_a=int(DType.E4M3),
        dtype_b=int(DType.E4M3),
        dtype_c=int(DType.E4M3),
        tile_k=32,
        mma_k=32,
        tile_n=16,
        route_act=int(RouteImpl.TMA),
        act_kind=int(ActKind.SWIGLU),
        batch_mode=int(BatchMode.BATCH_M),
        transpose_mma_output=0,
        use_per_token_sf_b=1,
        per_token_sf_dtype=int(DType.BF16),
    )
    validate_config(cfg)


@pytest.mark.cpu_only
def test_validation_accepts_nvfp4_per_token_sfb_only():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        ActKind,
        BatchMode,
        DType,
        RouteImpl,
        SfLayout,
        make_config,
        validate_config,
    )

    nvfp4_kwargs = dict(
        dtype_a=int(DType.E2M1),
        dtype_b=int(DType.E2M1),
        dtype_c=int(DType.BF16),
        tile_n=8,
        tile_k=512,
        epi_tile_n=8,
        mma_m=128,
        mma_n=8,
        mma_k=64,
        sf_layout_a=int(SfLayout.R128c4),
        sf_layout_b=int(SfLayout.R8c4),
        sf_layout_c=int(SfLayout.R8c4),
        use_global_scales=1,
    )

    cfg = make_config(
        **nvfp4_kwargs,
        use_per_token_sf_b=1,
        per_token_sf_dtype=int(DType.FP32),
    )
    assert cfg.has_per_token_sf_b
    validate_config(cfg)

    cfg = make_config(
        **nvfp4_kwargs,
        bias_type=1,
        use_per_token_sf_b=1,
        per_token_sf_dtype=int(DType.FP32),
    )
    validate_config(cfg)

    cfg = make_config(
        **{
            **nvfp4_kwargs,
            "dtype_c": int(DType.E2M1),
            "route_act": int(RouteImpl.TMA),
            "act_kind": int(ActKind.SWIGLU),
        },
        use_per_token_sf_b=1,
        per_token_sf_dtype=int(DType.FP32),
    )
    assert cfg.has_per_token_sf_b
    assert cfg.has_output_block_scaling
    validate_config(cfg)

    cfg = make_config(
        **nvfp4_kwargs,
        use_per_token_sf_a=1,
        per_token_sf_dtype=int(DType.FP32),
    )
    with pytest.raises(ValueError, match="Per-token sf_a is only valid"):
        validate_config(cfg)

    no_swap_ab_kwargs = {
        **nvfp4_kwargs,
        "batch_mode": int(BatchMode.BATCH_M),
        "transpose_mma_output": 0,
    }
    cfg = make_config(
        **no_swap_ab_kwargs,
        use_per_token_sf_b=1,
        per_token_sf_dtype=int(DType.FP32),
    )
    validate_config(cfg)

    cfg = make_config(
        dtype_a=int(DType.BF16),
        dtype_b=int(DType.BF16),
        dtype_c=int(DType.BF16),
        tile_k=64,
        mma_k=16,
        tile_n=16,
        use_per_token_sf_b=1,
    )
    with pytest.raises(ValueError, match="Per-token sf_b"):
        validate_config(cfg)


@pytest.mark.cpu_only
@pytest.mark.timeout(180)
def test_nvfp4_per_token_sfb_schedule_validates():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        ActKind,
        DType,
        RouteImpl,
        SfLayout,
    )
    from flashinfer.prims_ts.batched_gemm.batched_gemm_kernel import (
        build_batched_gemm_task_manager,
    )

    build_batched_gemm_task_manager(
        num_experts=64,
        num_tokens=1,
        top_k=8,
        dtype_a=int(DType.E2M1),
        dtype_b=int(DType.E2M1),
        dtype_c=int(DType.E2M1),
        route_act=int(RouteImpl.TMA),
        act_kind=int(ActKind.SWIGLU),
        tile_n=8,
        tile_k=512,
        epi_tile_n=8,
        mma_m=128,
        mma_n=8,
        mma_k=64,
        sf_layout_a=int(SfLayout.R128c4),
        sf_layout_b=int(SfLayout.LINEAR),
        sf_layout_c=int(SfLayout.R8c4),
        use_global_scales=1,
        use_per_token_sf_b=1,
        per_token_sf_dtype=int(DType.FP32),
    )


@pytest.mark.cpu_only
def test_benchmark_preserves_supported_fast_drain_and_mma_unroll_knobs():
    from flashinfer.prims_ts.batched_gemm.tools.bench import _ts_kwargs

    kwargs = {
        "use_clc_fast_drain": 1,
        "use_unroll_loop_2x_for_mma": 1,
    }

    result = _ts_kwargs(kwargs)

    assert result == kwargs
    assert result is not kwargs


@pytest.mark.cpu_only
def test_fp4_json_variant_forwards_per_token_sfb_without_skip():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        BatchMode,
        BiasType,
        DType,
        make_config,
        validate_config,
    )
    from flashinfer.prims_ts.batched_gemm.tools import bench

    options = {
        "tileN": 8,
        "tileK": 512,
        "tileScheduler": "static",
        "numStagesMma": 1,
        "numStages": 5,
        "epilogueTileN": 8,
        "mmaM": 128,
        "mmaN": 8,
        "sfLayoutA": "128x4",
        "sfLayoutB": "linear",
        "sfLayoutC": "8x4",
        "dtypeC": "e2m1",
        "transpose_mma_output": True,
        "usePerTokenSfB": True,
        "perTokenSfDtype": "fp32",
        "biasType": "none",
    }
    variant = bench._with_ts_skip_reason(
        bench._fp4_json_variant(
            config_index=0,
            config_comment="FC1_LL",
            combo_index=0,
            options=options,
        )
    )

    assert variant.ts_skip_reason is None
    assert variant.kwargs["dtype_a"] == int(DType.E2M1)
    assert variant.kwargs["dtype_b"] == int(DType.E2M1)
    assert variant.kwargs["dtype_c"] == int(DType.E2M1)
    assert variant.kwargs["batch_mode"] == int(BatchMode.BATCH_N)
    assert variant.kwargs["transpose_mma_output"] == 1
    assert variant.kwargs["use_shuffled_matrix"] == 1
    assert variant.kwargs["use_per_token_sf_a"] == 0
    assert variant.kwargs["use_per_token_sf_b"] == 1
    assert variant.kwargs["per_token_sf_dtype"] == int(DType.FP32)
    assert variant.kwargs["bias_type"] == int(BiasType.NONE)

    non_transposed = bench._fp4_json_variant(
        config_index=0,
        config_comment="FC1_LL",
        combo_index=1,
        options={**options, "transpose_mma_output": False},
    )
    assert non_transposed.kwargs["batch_mode"] == int(BatchMode.BATCH_N)
    assert non_transposed.kwargs["transpose_mma_output"] == 0
    with pytest.raises(ValueError, match="must match.*swap_ab"):
        validate_config(make_config(**non_transposed.kwargs))

    unshuffled = bench._fp4_json_variant(
        config_index=0,
        config_comment="FC1_LL",
        combo_index=2,
        options={**options, "useShuffledMatrix": False},
    )
    assert unshuffled.kwargs["use_shuffled_matrix"] == 0


@pytest.mark.cpu_only
def test_trtllm_gen_json_transpose_option_is_normalized(tmp_path):
    import json

    from flashinfer.prims_ts.batched_gemm.tools import bench

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "templates": {},
                "configs": [
                    {
                        "_comment": "FC1_LL",
                        bench.TRTLLM_GEN_TRANSPOSE_MMA_OUTPUT_KEY: False,
                    }
                ],
            }
        )
    )

    [(_config_index, _comment, _combo_index, options)] = bench._expanded_json_options(
        config_path
    )
    assert options["transpose_mma_output"] is False
    assert bench.TRTLLM_GEN_TRANSPOSE_MMA_OUTPUT_KEY not in options


@pytest.mark.cpu_only
def test_fp4_json_variant_forwards_fp16_dtype_c():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import DType
    from flashinfer.prims_ts.batched_gemm.tools import bench

    variant = bench._fp4_json_variant(
        config_index=0,
        config_comment="FC2_LL",
        combo_index=0,
        options={
            "tileN": 8,
            "tileK": 512,
            "tileScheduler": "static",
            "numStagesMma": 1,
            "numStages": 5,
            "epilogueTileN": 8,
            "mmaM": 128,
            "mmaN": 8,
            "dtypeC": "fp16",
        },
    )

    assert variant.kwargs["dtype_c"] == int(DType.FP16)


@pytest.mark.cpu_only
def test_bf16_json_variant_forwards_fp16_dtype_c():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import DType
    from flashinfer.prims_ts.batched_gemm.tools import bench

    variant = bench._bf16_json_variant(
        config_index=0,
        config_comment="FC2_HT",
        combo_index=0,
        options={
            "tileN": 16,
            "tileK": 128,
            "tileScheduler": "static",
            "numStages": 4,
            "epilogueTileN": 16,
            "mmaM": 128,
            "mmaN": 16,
            "dtypeC": "fp16",
        },
    )

    assert variant.kwargs["dtype_a"] == int(DType.BF16)
    assert variant.kwargs["dtype_b"] == int(DType.BF16)
    assert variant.kwargs["dtype_c"] == int(DType.FP16)


@pytest.mark.cpu_only
def test_generated_command_treats_fp16_dtype_c_as_plain_output():
    from pathlib import Path

    from flashinfer.prims_ts.batched_gemm.tools import bench

    variant = bench._bf16_json_variant(
        config_index=0,
        config_comment="FC2_HT",
        combo_index=0,
        options={
            "tileN": 16,
            "tileK": 128,
            "tileScheduler": "static",
            "numStages": 4,
            "epilogueTileN": 16,
            "mmaM": 128,
            "mmaN": 16,
            "dtypeC": "fp16",
        },
    )

    cmd = bench._trtllm_gen_command(
        binary=Path("batched_gemm"),
        variant=variant,
        num_tokens=128,
        num_experts=1,
        top_k=1,
        warmup_iters=1,
        bench_iters=1,
        num_rotated_buffers=0,
        use_ccache=False,
        use_cuda_graph=False,
    )

    assert "DeepSeekR1_TP1_EP1_MoE_output_tokens128" in cmd
    assert cmd[cmd.index("-dtypeC") + 1] == "fp16"
    assert "-eltwiseActType" not in cmd


@pytest.mark.cpu_only
def test_runner_cli_accepts_batch_mode(monkeypatch):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import BatchMode

    import flashinfer.prims_ts.batched_gemm.batched_gemm_run as batched_gemm_run

    captured = {}

    def fake_validate_schedule(**kwargs):
        captured.update(kwargs)
        return True

    monkeypatch.setattr(
        batched_gemm_run,
        "validate_schedule",
        fake_validate_schedule,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "batched_gemm_run.py",
            "--validate-only",
            "--batch-mode",
            "m",
            "--transpose-mma-output",
            "0",
        ],
    )

    batched_gemm_run.main()

    assert captured["batch_mode"] == int(BatchMode.BATCH_M)
    assert captured["transpose_mma_output"] == 0


@pytest.mark.cpu_only
def test_validation_rejects_deepseek_fp8_fused_gated_activation():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        ActKind,
        DType,
        make_config,
        validate_config,
    )

    cfg = make_config(
        dtype_a=int(DType.E4M3),
        dtype_b=int(DType.E4M3),
        dtype_c=int(DType.BF16),
        tile_k=128,
        mma_k=32,
        tile_n=16,
        act_kind=int(ActKind.SWIGLU),
        use_deepseek_fp8=1,
    )
    with pytest.raises(ValueError, match="fused gated activation"):
        validate_config(cfg)


@pytest.mark.cpu_only
def test_validation_rejects_deepseek_fp8_split_tma_c_scratch_stages():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        DType,
        make_config,
        validate_config,
    )

    cfg = make_config(
        dtype_a=int(DType.E4M3),
        dtype_b=int(DType.E4M3),
        dtype_c=int(DType.BF16),
        tile_k=128,
        mma_k=32,
        mma_m=64,
        use_deepseek_fp8=1,
        use_tma_store=1,
        num_stages_c_smem=2,
    )
    with pytest.raises(ValueError, match="num_stages_c_smem=1"):
        validate_config(cfg)


@pytest.mark.cpu_only
def test_validation_rejects_deepseek_fp8_multi_epilogue_subtiles():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        DType,
        make_config,
        validate_config,
    )

    cfg = make_config(
        dtype_a=int(DType.E4M3),
        dtype_b=int(DType.E4M3),
        dtype_c=int(DType.BF16),
        tile_n=128,
        epi_tile_n=64,
        tile_k=128,
        mma_k=32,
        mma_m=64,
        use_deepseek_fp8=1,
    )
    with pytest.raises(ValueError, match="epi_subtile_cnt=1"):
        validate_config(cfg)


@pytest.mark.cpu_only
def test_deepseek_fp8_epi_subtile_validation_uses_computed_warp_layout():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        DType,
        compute_warp_layout,
        make_config,
        validate_config,
    )

    cfg = make_config(
        dtype_a=int(DType.E4M3),
        dtype_b=int(DType.E4M3),
        dtype_c=int(DType.BF16),
        tile_n=256,
        epi_tile_n=128,
        tile_k=128,
        mma_k=32,
        mma_m=128,
        mma_n=256,
        use_deepseek_fp8=1,
    )
    compute_warp_layout(cfg)
    assert cfg.warp_layout.num_epilogue_warps == 8
    validate_config(cfg)


@pytest.mark.cpu_only
def test_validation_rejects_plain_fp8_output_without_activation_side():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        DType,
        make_config,
        validate_config,
    )

    cfg = make_config(
        dtype_a=int(DType.E4M3),
        dtype_b=int(DType.E4M3),
        dtype_c=int(DType.E4M3),
        tile_k=32,
        mma_k=32,
        tile_n=16,
    )
    with pytest.raises(ValueError, match="Plain FP8 C output"):
        validate_config(cfg)
