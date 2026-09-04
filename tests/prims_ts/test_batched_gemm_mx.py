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

"""Tests for MXFP4/MXFP8 BatchedGemm TS kernels.

These cover the MX family where operands use UE8M0 block scaling
with 32-value K blocks:
  - MXE2M1 x MXE4M3 -> BF16 FC2
  - MXE4M3 x MXE4M3 -> BF16 FC2
  - MXE2M1 x MXE4M3 -> MXE2M1 FC1 epilogue quantization
  - MXE2M1 x MXE4M3 -> MXE4M3 FC1 epilogue quantization

Requires: CUDA GPU with SM100A+.
"""

from collections import Counter

import pytest

import torch

from flashinfer.utils import is_sm100a_supported

from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
    uniform_pipeline_stage_overrides,
)

pytestmark = [
    pytest.mark.xdist_group("isolated_cuda"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required"),
    pytest.mark.skipif(
        (torch.cuda.is_available() and not is_sm100a_supported(torch.device("cuda"))),
        reason="FP4 cvt (cvt.e2m1x2.f32) requires Blackwell sm_100+",
    ),
]


def _finalize_tmem(cfg):
    """Exercise config construction while leaving TMEM columns derived."""
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        compute_warp_layout,
        make_config,
    )

    cfg = dict(cfg)
    tmp = make_config(**cfg)
    compute_warp_layout(tmp)
    return cfg


def _mx_fc1_base(*, dtype_a):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        ActKind,
        BatchMode,
        BiasType,
        DType,
        RouteImpl,
        SfLayout,
        TileScheduler,
    )

    return dict(
        batch_mode=int(BatchMode.BATCH_N),
        transpose_mma_output=1,
        route_act=int(RouteImpl.TMA),
        route_sfs_act=int(RouteImpl.LDGSTS),
        act_kind=int(ActKind.SWIGLU),
        tile_m=128,
        mma_m=128,
        mma_k=32,
        dtype_a=int(dtype_a),
        dtype_b=int(DType.MXE4M3),
        dtype_c=int(DType.MXE4M3),
        sf_bits=8,
        sf_block_size_a=32,
        sf_block_size_b=32,
        sf_block_size_c=32,
        sf_layout_a=int(SfLayout.R128c4),
        sf_layout_b=int(SfLayout.LINEAR),
        sf_layout_c=int(SfLayout.R8c4),
        use_tma_store=1,
        use_tma_oob_opt=1,
        tile_scheduler=int(TileScheduler.PERSISTENT),
        num_stages_tmem_acc=2,
        bias_type=int(BiasType.M),
        has_gemm1_clamp_limit=1,
        epilogue_regs=160,
        mma_regs=48,
        load_regs=48,
        load_sf_regs=48,
        copy_sf_regs=48,
        workid_regs=48,
        padding_regs=48,
        gather_regs=48,
    )


def _mx_fc2_base(*, dtype_a, dtype_b):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        BatchMode,
        BiasType,
        DType,
        SfLayout,
    )

    return dict(
        batch_mode=int(BatchMode.BATCH_N),
        transpose_mma_output=1,
        route_act=0,
        route_sfs_act=0,
        act_kind=0,
        tile_m=128,
        mma_m=128,
        mma_k=32,
        dtype_a=int(dtype_a),
        dtype_b=int(dtype_b),
        dtype_c=int(DType.BF16),
        sf_bits=8,
        sf_block_size_a=32,
        sf_block_size_b=32,
        sf_layout_a=int(SfLayout.R128c4),
        sf_layout_b=int(SfLayout.R8c4),
        epilogue_regs=128,
        mma_regs=48,
        load_regs=48,
        load_sf_regs=48,
        copy_sf_regs=48,
        workid_regs=48,
        padding_regs=48,
        bias_type=int(BiasType.M),
        use_tma_oob_opt=1,
    )


def _mxfp4_bf16_base(*, has_activation_epilogue):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        ActKind,
        BatchMode,
        BiasType,
        DType,
        RouteImpl,
        SfLayout,
        TileScheduler,
    )

    return dict(
        batch_mode=int(BatchMode.BATCH_N),
        transpose_mma_output=1,
        route_act=(
            int(RouteImpl.LDGSTS) if has_activation_epilogue else int(RouteImpl.NONE)
        ),
        route_sfs_act=int(RouteImpl.NONE),
        act_kind=(
            int(ActKind.SWIGLU) if has_activation_epilogue else int(ActKind.NONE)
        ),
        tile_m=128,
        tile_n=16,
        tile_k=256,
        epi_tile_n=16,
        mma_m=128,
        mma_n=16,
        mma_k=16,
        dtype_a=int(DType.MXE2M1),
        dtype_b=int(DType.BF16),
        dtype_c=int(DType.BF16),
        sf_bits=8,
        sf_block_size_a=32,
        sf_layout_a=int(SfLayout.R128c4),
        sf_layout_b=int(SfLayout.R8c4),
        **uniform_pipeline_stage_overrides(3),
        num_stages_tmem_acc=2,
        tile_scheduler=int(TileScheduler.PERSISTENT),
        use_tma_store=1,
        use_tma_oob_opt=1,
        bias_type=int(BiasType.M),
        **({"has_gemm1_clamp_limit": 1} if has_activation_epilogue else {}),
        epilogue_regs=128,
        mma_regs=48,
        load_regs=48,
        load_sf_regs=48,
        cast_a_regs=160,
        copy_sf_regs=48,
        workid_regs=48,
        padding_regs=48,
        gather_regs=48,
    )


def _run_mx_fc1(cfg, *, num_tokens=128, problem_n=128, problem_k=None):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
        reference_check,
    )

    cfg = _finalize_tmem(cfg)
    result = reference_check(
        num_experts=2,
        num_tokens=num_tokens,
        top_k=1,
        problem_n=problem_n,
        problem_k=problem_k or cfg["tile_k"],
        seed=123,
        **cfg,
    )
    assert result


def _run_mx_fc2(cfg, *, num_tokens=128, problem_n=None):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
        reference_check,
    )

    cfg = _finalize_tmem(cfg)
    result = reference_check(
        num_experts=2,
        num_tokens=num_tokens,
        top_k=1,
        problem_n=problem_n,
        problem_k=cfg["tile_k"],
        **cfg,
    )
    assert result


def _run_mxfp4_bf16(cfg, *, num_tokens=128, problem_n=128, problem_k=256):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
        reference_check,
    )

    cfg = _finalize_tmem(cfg)
    result = reference_check(
        num_experts=2,
        num_tokens=num_tokens,
        top_k=1,
        problem_n=problem_n,
        problem_k=problem_k,
        seed=123,
        **cfg,
    )
    assert result


def _mx_generated_json_rows():
    """Return compact aliases for all rows in the MX JSON."""
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        DType,
        SfLayout,
    )

    rows = []
    fc1_ll_shapes = [
        (128, 8, 8, 8, 512, 1, 3),
        (128, 8, 8, 8, 256, 1, 6),
        (128, 16, 16, 16, 256, 1, 5),
        (256, 16, 16, 16, 256, 2, 5),
        (128, 32, 32, 32, 256, 1, 5),
        (256, 32, 32, 32, 256, 2, 5),
    ]
    fc2_ll_shapes = [
        (128, 8, 8, 8, 512, 1, 3),
        (128, 8, 8, 8, 256, 1, 6),
        (128, 16, 16, 16, 256, 1, 5),
        (256, 16, 16, 16, 256, 2, 6),
        (128, 32, 32, 32, 256, 1, 5),
        (256, 32, 32, 32, 256, 2, 5),
    ]
    for dtype_a in (DType.MXE2M1, DType.MXE4M3):
        for use_unroll in (1, 0):
            for shape in fc1_ll_shapes:
                rows.append(
                    (
                        "MxFp4xMxFp8_FC1_LowLatency",
                        {
                            "dtype_a": dtype_a,
                            "sf_layout_b": SfLayout.LINEAR,
                            "sf_layout_c": SfLayout.R8c4,
                            "use_unroll": use_unroll,
                            "use_max_tmem_overlap": 0,
                            "mma_m": shape[0],
                            "mma_n": shape[1],
                            "tile_n": shape[2],
                            "epi_tile_n": shape[3],
                            "tile_k": shape[4],
                            "cluster_m": shape[5],
                            "stages": shape[6],
                            "num_stages_tmem_acc": 2,
                        },
                    )
                )
        for use_unroll in (1, 0):
            for mma_n, tile_n, epi_n, tile_k, stages, sf_layout_c in (
                (64, 64, 64, 256, 4, SfLayout.R8c4),
                (64, 64, 64, 128, 7, SfLayout.R8c4),
                (128, 128, 32, 256, 4, SfLayout.R128c4),
                (128, 128, 32, 128, 7, SfLayout.R128c4),
            ):
                rows.append(
                    (
                        "MxFp4xMxFp8_FC1_HighThroughput",
                        {
                            "dtype_a": dtype_a,
                            "sf_layout_b": SfLayout.LINEAR,
                            "sf_layout_c": sf_layout_c,
                            "use_unroll": use_unroll,
                            "use_max_tmem_overlap": 0,
                            "mma_m": 256,
                            "mma_n": mma_n,
                            "tile_n": tile_n,
                            "epi_tile_n": epi_n,
                            "tile_k": tile_k,
                            "cluster_m": 2,
                            "stages": stages,
                            "num_stages_tmem_acc": 2,
                            "num_load_b_warps": 8 if tile_n == 128 else 1,
                            "num_load_sfb_warps": 4 if tile_n == 128 else 1,
                            "epilogue_regs": 152 if tile_n == 128 else 168,
                            "non_epilogue_regs": 48,
                        },
                    )
                )
        for tile_k, stages in ((128, 6), (256, 3)):
            rows.append(
                (
                    "MxFp4xMxFp8_FC1_HighThroughput_tileN_256",
                    {
                        "dtype_a": dtype_a,
                        "sf_layout_b": SfLayout.LINEAR,
                        "sf_layout_c": SfLayout.R128c4,
                        "use_unroll": 0,
                        "use_max_tmem_overlap": 1,
                        "mma_m": 256,
                        "mma_n": 256,
                        "tile_n": 256,
                        "epi_tile_n": 64,
                        "tile_k": tile_k,
                        "cluster_m": 2,
                        "stages": stages,
                        "num_stages_tmem_acc": 1,
                        "num_load_b_warps": 8,
                        "num_load_sfb_warps": 4,
                        "epilogue_regs": 112,
                        "non_epilogue_regs": 88,
                        "load_b_regs": 56,
                        "load_sfb_regs": 56,
                    },
                )
            )
        for shape in fc2_ll_shapes:
            rows.append(
                (
                    "MxFp4xMxFp8_FC2_LowLatency",
                    {
                        "dtype_a": dtype_a,
                        "sf_layout_b": SfLayout.R8c4,
                        "use_unroll": 0,
                        "use_max_tmem_overlap": 0,
                        "mma_m": shape[0],
                        "mma_n": shape[1],
                        "tile_n": shape[2],
                        "epi_tile_n": shape[3],
                        "tile_k": shape[4],
                        "cluster_m": shape[5],
                        "stages": shape[6],
                        "num_stages_tmem_acc": 2,
                    },
                )
            )
        for mma_n, tile_n, tile_k, stages, sf_layout_b in (
            (64, 64, 256, 4, SfLayout.R8c4),
            (64, 64, 128, 7, SfLayout.R8c4),
            (128, 128, 256, 4, SfLayout.R128c4),
            (128, 128, 128, 7, SfLayout.R128c4),
        ):
            rows.append(
                (
                    "MxFp4xMxFp8_FC2_HighThroughput",
                    {
                        "dtype_a": dtype_a,
                        "sf_layout_b": sf_layout_b,
                        "use_unroll": 0,
                        "use_max_tmem_overlap": 0,
                        "mma_m": 256,
                        "mma_n": mma_n,
                        "tile_n": tile_n,
                        "epi_tile_n": 64,
                        "tile_k": tile_k,
                        "cluster_m": 2,
                        "stages": stages,
                        "num_stages_tmem_acc": 2,
                        "epilogue_regs": 168,
                        "non_epilogue_regs": 96,
                    },
                )
            )
        rows.append(
            (
                "MxFp4xMxFp8_FC2_HighThroughput_tileN_256",
                {
                    "dtype_a": dtype_a,
                    "sf_layout_b": SfLayout.R128c4,
                    "use_unroll": 0,
                    "use_max_tmem_overlap": 1,
                    "mma_m": 256,
                    "mma_n": 256,
                    "tile_n": 256,
                    "epi_tile_n": 64,
                    "tile_k": 128,
                    "cluster_m": 2,
                    "stages": 5,
                    "num_stages_tmem_acc": 1,
                    "epilogue_regs": 144,
                    "non_epilogue_regs": 96,
                },
            )
        )
    return rows


def _mxfp4_bf16_generated_json_rows():
    """Return compact aliases for rows in the MXFP4-weight BF16 JSON."""
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        TileScheduler,
    )

    rows = []
    fc2_shapes = [
        (8, 8, 8, 128, 1, 0),
        (16, 16, 16, 128, 1, 0),
        (16, 16, 16, 256, 2, 0),
        (32, 32, 32, 128, 1, 0),
        (32, 32, 32, 256, 2, 0),
        (64, 64, 64, 128, 1, 0),
        (64, 64, 64, 256, 2, 0),
        (128, 128, 128, 256, 2, 0),
    ]
    fc1_shapes = [
        (8, 8, 8, 128, 1, 0),
        (16, 16, 16, 128, 1, 0),
        (16, 16, 16, 256, 2, 0),
        (32, 32, 32, 128, 1, 0),
        (32, 32, 32, 256, 2, 0),
        (64, 64, 64, 128, 1, 0),
        (64, 64, 64, 256, 2, 0),
        (128, 128, 128, 256, 2, 0),
    ]
    for tile_n, mma_n, epi_n, mma_m, cluster_m, use_unroll in fc2_shapes:
        rows.append(
            (
                "MxFp4xBf16_FC2",
                {
                    "has_activation_epilogue": False,
                    "tile_n": tile_n,
                    "mma_n": mma_n,
                    "epi_tile_n": epi_n,
                    "mma_m": mma_m,
                    "cluster_m": cluster_m,
                    "use_unroll": use_unroll,
                    "tile_scheduler": int(
                        TileScheduler.STATIC
                        if tile_n == 128
                        else TileScheduler.PERSISTENT
                    ),
                },
            )
        )
    for tile_n, mma_n, epi_n, mma_m, cluster_m, use_unroll in fc1_shapes:
        rows.append(
            (
                "MxFp4xBf16_FC1_SwiGLU",
                {
                    "has_activation_epilogue": True,
                    "tile_n": tile_n,
                    "mma_n": mma_n,
                    "epi_tile_n": epi_n,
                    "mma_m": mma_m,
                    "cluster_m": cluster_m,
                    "use_unroll": use_unroll,
                    "tile_scheduler": int(TileScheduler.PERSISTENT),
                },
            )
        )
    return rows


def _mxfp4_bf16_json_row_to_cfg(row):
    cfg = _mxfp4_bf16_base(has_activation_epilogue=row["has_activation_epilogue"])
    cfg.update(
        tile_n=row["tile_n"],
        mma_n=row["mma_n"],
        epi_tile_n=row["epi_tile_n"],
        mma_m=row["mma_m"],
        cluster_m=row["cluster_m"],
        # Generated MX rows may request the unroll-2x MMA path. TS parses that
        # option but rejects it in kernel validation until a matching MMA task
        # exists, so generated-row construction tests use the same boundary
        # mapping as the benchmark driver.
        use_unroll_loop_2x_for_mma=0,
        tile_scheduler=row["tile_scheduler"],
    )
    if row["tile_scheduler"] == 0:
        cfg["num_stages_workid"] = 1
    if row["tile_n"] == 128:
        cfg["num_stages_tmem_acc"] = 1
    return cfg


def _mx_json_row_to_cfg(comment, row):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        DType,
        TileScheduler,
    )

    dtype_a = row["dtype_a"]
    if "FC1" in comment:
        cfg = _mx_fc1_base(dtype_a=dtype_a)
    else:
        cfg = _mx_fc2_base(
            dtype_a=dtype_a,
            dtype_b=DType.MXE4M3,
        )
    non_epilogue_regs = row.get("non_epilogue_regs")
    if non_epilogue_regs is not None:
        cfg.update(
            mma_regs=non_epilogue_regs,
            load_regs=non_epilogue_regs,
            load_sf_regs=non_epilogue_regs,
            copy_sf_regs=non_epilogue_regs,
            workid_regs=non_epilogue_regs,
            padding_regs=non_epilogue_regs,
            gather_regs=non_epilogue_regs,
        )
    cfg.update(
        tile_n=row["tile_n"],
        mma_n=row["mma_n"],
        epi_tile_n=row["epi_tile_n"],
        tile_k=row["tile_k"],
        **uniform_pipeline_stage_overrides(row["stages"]),
        num_stages_tmem_acc=row["num_stages_tmem_acc"],
        mma_m=row["mma_m"],
        cluster_m=row["cluster_m"],
        sf_layout_b=int(row["sf_layout_b"]),
        # Generated MX rows may request the unroll-2x MMA path. TS parses that
        # option but rejects it in kernel validation until a matching MMA task
        # exists, so generated-row construction tests use the same boundary
        # mapping as the benchmark driver.
        use_unroll_loop_2x_for_mma=0,
        use_max_tmem_overlap=row["use_max_tmem_overlap"],
        use_tma_store=1,
        tile_scheduler=int(TileScheduler.PERSISTENT),
    )
    if "sf_layout_c" in row:
        cfg["sf_layout_c"] = int(row["sf_layout_c"])
    if row["use_max_tmem_overlap"]:
        cfg["num_stages_tmem_acc"] = 1
    for key in (
        "num_load_b_warps",
        "num_load_sfb_warps",
        "epilogue_regs",
        "load_b_regs",
        "load_sfb_regs",
    ):
        if key in row:
            cfg[key] = row[key]
    return cfg


def test_generated_mx_json_rows_construct_ts_configs():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        compute_warp_layout,
        make_config,
        validate_config,
    )

    rows = _mx_generated_json_rows()
    assert Counter(comment for comment, _ in rows) == {
        "MxFp4xMxFp8_FC1_LowLatency": 24,
        "MxFp4xMxFp8_FC1_HighThroughput": 16,
        "MxFp4xMxFp8_FC1_HighThroughput_tileN_256": 4,
        "MxFp4xMxFp8_FC2_LowLatency": 12,
        "MxFp4xMxFp8_FC2_HighThroughput": 8,
        "MxFp4xMxFp8_FC2_HighThroughput_tileN_256": 2,
    }
    for comment, row in rows:
        cfg = _finalize_tmem(_mx_json_row_to_cfg(comment, row))
        concrete = make_config(**cfg)
        compute_warp_layout(concrete)
        validate_config(concrete)


def test_generated_mxfp4_bf16_json_rows_construct_ts_configs():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        compute_warp_layout,
        make_config,
        validate_config,
    )

    rows = _mxfp4_bf16_generated_json_rows()
    assert Counter(comment for comment, _ in rows) == {
        "MxFp4xBf16_FC2": 8,
        "MxFp4xBf16_FC1_SwiGLU": 8,
    }
    for _, row in rows:
        cfg = _finalize_tmem(_mxfp4_bf16_json_row_to_cfg(row))
        concrete = make_config(**cfg)
        compute_warp_layout(concrete)
        validate_config(concrete)


class TestMxFp4Bf16Fc2:
    def test_tile8_persistent_correctness(self):
        cfg = {
            **_mxfp4_bf16_base(has_activation_epilogue=False),
            "tile_n": 8,
            "mma_n": 8,
            "epi_tile_n": 8,
            "mma_m": 128,
            "cluster_m": 1,
        }
        _run_mxfp4_bf16(cfg, num_tokens=128, problem_n=128)

    def test_tile16_cluster2_persistent_correctness(self):
        cfg = {
            **_mxfp4_bf16_base(has_activation_epilogue=False),
            "tile_n": 16,
            "mma_n": 16,
            "epi_tile_n": 16,
            "mma_m": 256,
            "cluster_m": 2,
        }
        _run_mxfp4_bf16(cfg, num_tokens=128, problem_n=128)

    def test_tile128_static_correctness(self):
        from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
            TileScheduler,
        )

        cfg = {
            **_mxfp4_bf16_base(has_activation_epilogue=False),
            "tile_n": 128,
            "mma_n": 128,
            "epi_tile_n": 128,
            "mma_m": 256,
            "cluster_m": 2,
            "tile_scheduler": int(TileScheduler.STATIC),
            "num_stages_tmem_acc": 1,
        }
        _run_mxfp4_bf16(cfg, num_tokens=128, problem_n=256)


class TestMxFp4Bf16Fc1:
    """MXFP4-weight / BF16-activation FC1 with fused SwiGLU (CastA + fusedAct).

    Exercises the Kaiming-init + MXFP4 weight quantization on the CastA path: the
    weights are MXE2M1 (packed E2M1 + UE8M0 scale factors), so this covers the
    block-scaled Kaiming quantizer for a non-NVFP4 dtype with a fused activation.
    """

    def test_tile8_swiglu_persistent_correctness(self):
        cfg = {
            **_mxfp4_bf16_base(has_activation_epilogue=True),
            "tile_n": 8,
            "mma_n": 8,
            "epi_tile_n": 8,
            "mma_m": 128,
            "cluster_m": 1,
        }
        _run_mxfp4_bf16(cfg, num_tokens=128, problem_n=256)

    def test_tile8_swiglu_fp16_output_correctness(self):
        from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
            DType,
        )

        cfg = {
            **_mxfp4_bf16_base(has_activation_epilogue=True),
            "dtype_c": int(DType.FP16),
            "tile_n": 8,
            "mma_n": 8,
            "epi_tile_n": 8,
            "mma_m": 128,
            "cluster_m": 1,
        }
        _run_mxfp4_bf16(cfg, num_tokens=128, problem_n=256)

    def test_tile16_cluster2_swiglu_persistent_correctness(self):
        cfg = {
            **_mxfp4_bf16_base(has_activation_epilogue=True),
            "tile_n": 16,
            "mma_n": 16,
            "epi_tile_n": 16,
            "mma_m": 256,
            "cluster_m": 2,
        }
        _run_mxfp4_bf16(cfg, num_tokens=128, problem_n=256)


class TestMxFc2LowLatency:
    def test_mxfp4_mxfp8_tile8_k512_persistent(self):
        from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
            DType,
            TileScheduler,
        )

        cfg = {
            **_mx_fc2_base(
                dtype_a=DType.MXE2M1,
                dtype_b=DType.MXE4M3,
            ),
            "tile_n": 8,
            "mma_n": 8,
            "epi_tile_n": 8,
            "tile_k": 512,
            **uniform_pipeline_stage_overrides(3),
            "tile_scheduler": int(TileScheduler.PERSISTENT),
            "num_stages_tmem_acc": 2,
        }
        _run_mx_fc2(cfg)

    def test_mxfp4_mxfp8_tile8_static_8x4_sfb(self):
        from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
            DType,
        )

        cfg = {
            **_mx_fc2_base(
                dtype_a=DType.MXE2M1,
                dtype_b=DType.MXE4M3,
            ),
            "tile_n": 8,
            "mma_n": 8,
            "epi_tile_n": 8,
            "tile_k": 256,
            **uniform_pipeline_stage_overrides(6),
            "tile_scheduler": 0,
            "num_stages_tmem_acc": 1,
        }
        _run_mx_fc2(cfg)

    def test_mxfp8_mxfp8_tile8_static_8x4_sfb(self):
        from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
            DType,
        )

        cfg = {
            **_mx_fc2_base(
                dtype_a=DType.MXE4M3,
                dtype_b=DType.MXE4M3,
            ),
            "tile_n": 8,
            "mma_n": 8,
            "epi_tile_n": 8,
            "tile_k": 256,
            **uniform_pipeline_stage_overrides(6),
            "tile_scheduler": 0,
            "num_stages_tmem_acc": 1,
        }
        _run_mx_fc2(cfg)

    def test_mxfp4_mxfp8_tile16_cluster_persistent(self):
        from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
            DType,
            TileScheduler,
        )

        cfg = {
            **_mx_fc2_base(
                dtype_a=DType.MXE2M1,
                dtype_b=DType.MXE4M3,
            ),
            "tile_n": 16,
            "mma_n": 16,
            "epi_tile_n": 16,
            "tile_k": 256,
            "mma_m": 256,
            "cluster_m": 2,
            **uniform_pipeline_stage_overrides(6),
            "tile_scheduler": int(TileScheduler.PERSISTENT),
            "num_stages_tmem_acc": 2,
        }
        _run_mx_fc2(cfg, num_tokens=256)

    def test_mxfp8_mxfp8_tile32_cluster_persistent(self):
        from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
            DType,
            TileScheduler,
        )

        cfg = {
            **_mx_fc2_base(
                dtype_a=DType.MXE4M3,
                dtype_b=DType.MXE4M3,
            ),
            "tile_n": 32,
            "mma_n": 32,
            "epi_tile_n": 32,
            "tile_k": 256,
            "mma_m": 256,
            "cluster_m": 2,
            **uniform_pipeline_stage_overrides(5),
            "tile_scheduler": int(TileScheduler.PERSISTENT),
            "num_stages_tmem_acc": 2,
        }
        _run_mx_fc2(cfg, num_tokens=256)


class TestMxFc2HighThroughput:
    def test_mxfp4_mxfp8_tile64_k128_cluster_persistent(self):
        from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
            DType,
            SfLayout,
            TileScheduler,
        )

        cfg = {
            **_mx_fc2_base(
                dtype_a=DType.MXE2M1,
                dtype_b=DType.MXE4M3,
            ),
            "tile_n": 64,
            "mma_n": 64,
            "epi_tile_n": 64,
            "tile_k": 128,
            "mma_m": 256,
            "cluster_m": 2,
            "sf_layout_b": int(SfLayout.R8c4),
            **uniform_pipeline_stage_overrides(7),
            "tile_scheduler": int(TileScheduler.PERSISTENT),
            "num_stages_tmem_acc": 2,
            "epilogue_regs": 168,
            "mma_regs": 96,
            "load_regs": 96,
            "load_sf_regs": 96,
            "copy_sf_regs": 96,
            "workid_regs": 96,
            "padding_regs": 96,
            "gather_regs": 96,
        }
        _run_mx_fc2(cfg, num_tokens=256)

    def test_mxfp4_mxfp8_tile128_cluster_persistent(self):
        """Regression for the generated FC2 HT R128c4 SFB/TmemSfAb path."""
        from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
            DType,
            SfLayout,
            TileScheduler,
        )

        cfg = {
            **_mx_fc2_base(
                dtype_a=DType.MXE2M1,
                dtype_b=DType.MXE4M3,
            ),
            "tile_n": 128,
            "mma_n": 128,
            "epi_tile_n": 64,
            "tile_k": 256,
            "mma_m": 256,
            "cluster_m": 2,
            "sf_layout_b": int(SfLayout.R128c4),
            **uniform_pipeline_stage_overrides(3),
            "tile_scheduler": int(TileScheduler.PERSISTENT),
            "num_stages_tmem_acc": 2,
        }
        _run_mx_fc2(cfg, num_tokens=128, problem_n=128)

    def test_mxfp8_mxfp8_tile256_k128_tmem_overlap(self):
        from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
            DType,
            SfLayout,
            TileScheduler,
        )

        cfg = {
            **_mx_fc2_base(
                dtype_a=DType.MXE4M3,
                dtype_b=DType.MXE4M3,
            ),
            "tile_n": 256,
            "mma_n": 256,
            "epi_tile_n": 64,
            "tile_k": 128,
            "mma_m": 256,
            "cluster_m": 2,
            "sf_layout_b": int(SfLayout.R128c4),
            **uniform_pipeline_stage_overrides(5),
            "tile_scheduler": int(TileScheduler.PERSISTENT),
            "num_stages_tmem_acc": 1,
            "use_max_tmem_overlap": 1,
            "epilogue_regs": 144,
            "mma_regs": 96,
            "load_regs": 96,
            "load_sf_regs": 96,
            "copy_sf_regs": 96,
            "workid_regs": 96,
            "padding_regs": 96,
            "gather_regs": 96,
        }
        _run_mx_fc2(cfg, num_tokens=128, problem_n=128)


class TestMxFc1LowLatency:
    def test_mxfp4_mxfp8_tile8_k512_persistent(self):
        from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
            DType,
        )

        cfg = {
            **_mx_fc1_base(dtype_a=DType.MXE2M1),
            "tile_n": 8,
            "mma_n": 8,
            "epi_tile_n": 8,
            "tile_k": 512,
            **uniform_pipeline_stage_overrides(3),
            "mma_m": 128,
            "cluster_m": 1,
        }
        _run_mx_fc1(cfg, problem_k=512)

    def test_mxfp4_mxfp4_tile8_k512_persistent(self):
        from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
            DType,
        )

        cfg = {
            **_mx_fc1_base(dtype_a=DType.MXE2M1),
            "dtype_c": int(DType.MXE2M1),
            "tile_n": 8,
            "mma_n": 8,
            "epi_tile_n": 8,
            "tile_k": 512,
            **uniform_pipeline_stage_overrides(3),
            "mma_m": 128,
            "cluster_m": 1,
        }
        _run_mx_fc1(cfg, problem_k=512)

    def test_mxfp4_mxfp8_tile16_cluster_persistent(self):
        from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
            DType,
        )

        cfg = {
            **_mx_fc1_base(dtype_a=DType.MXE2M1),
            "tile_n": 16,
            "mma_n": 16,
            "epi_tile_n": 16,
            "tile_k": 256,
            **uniform_pipeline_stage_overrides(5),
            "mma_m": 256,
            "cluster_m": 2,
        }
        _run_mx_fc1(cfg, num_tokens=256)

    def test_mxfp8_mxfp8_tile32_cluster_persistent(self):
        from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
            DType,
        )

        cfg = {
            **_mx_fc1_base(dtype_a=DType.MXE4M3),
            "tile_n": 32,
            "mma_n": 32,
            "epi_tile_n": 32,
            "tile_k": 256,
            **uniform_pipeline_stage_overrides(5),
            "mma_m": 256,
            "cluster_m": 2,
        }
        _run_mx_fc1(cfg, num_tokens=256)


class TestMxFc1HighThroughput:
    @staticmethod
    def _run_tile256_tma_overlap(
        *,
        dtype_a,
        tile_k,
        stages,
        num_tokens=128,
        problem_n=256,
    ):
        from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
            SfLayout,
        )

        cfg = {
            **_mx_fc1_base(dtype_a=dtype_a),
            "tile_n": 256,
            "mma_n": 256,
            "epi_tile_n": 64,
            "tile_k": tile_k,
            **uniform_pipeline_stage_overrides(stages),
            "mma_m": 256,
            "cluster_m": 2,
            "sf_layout_c": int(SfLayout.R128c4),
            "num_load_b_warps": 8,
            "num_load_sfb_warps": 4,
            "num_stages_tmem_acc": 1,
            "use_max_tmem_overlap": 1,
            "use_tma_store": 1,
            "epilogue_regs": 112,
            "mma_regs": 88,
            "load_regs": 88,
            "load_sf_regs": 88,
            "load_b_regs": 56,
            "load_sfb_regs": 56,
            "copy_sf_regs": 88,
            "workid_regs": 88,
            "padding_regs": 88,
            "gather_regs": 88,
        }
        _run_mx_fc1(
            cfg,
            num_tokens=num_tokens,
            problem_n=problem_n,
            problem_k=tile_k,
        )

    def test_mxfp4_mxfp8_tile64_k128_cluster_persistent(self):
        from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
            DType,
            SfLayout,
        )

        cfg = {
            **_mx_fc1_base(dtype_a=DType.MXE2M1),
            "tile_n": 64,
            "mma_n": 64,
            "epi_tile_n": 64,
            "tile_k": 128,
            **uniform_pipeline_stage_overrides(7),
            "mma_m": 256,
            "cluster_m": 2,
            "sf_layout_c": int(SfLayout.R8c4),
            "epilogue_regs": 168,
        }
        _run_mx_fc1(cfg, num_tokens=256, problem_k=128)

    def test_mxfp8_mxfp8_tile128_k128_cluster_persistent(self):
        from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
            DType,
            SfLayout,
        )

        cfg = {
            **_mx_fc1_base(dtype_a=DType.MXE4M3),
            "tile_n": 128,
            "mma_n": 128,
            "epi_tile_n": 32,
            "tile_k": 128,
            **uniform_pipeline_stage_overrides(7),
            "mma_m": 256,
            "cluster_m": 2,
            "sf_layout_c": int(SfLayout.R128c4),
            "num_load_b_warps": 8,
            "num_load_sfb_warps": 4,
            "epilogue_regs": 152,
        }
        _run_mx_fc1(cfg, num_tokens=256, problem_k=128)

    def test_mxfp4_mxfp8_tile128_k256_cluster_second_kbox_regression(self):
        """Regression for clustered routed-B MX descriptor stride at tileK=256."""
        from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
            DType,
            SfLayout,
        )

        cfg = {
            **_mx_fc1_base(dtype_a=DType.MXE2M1),
            "tile_n": 128,
            "mma_n": 128,
            "epi_tile_n": 32,
            "tile_k": 256,
            **uniform_pipeline_stage_overrides(3),
            "mma_m": 256,
            "cluster_m": 2,
            "sf_layout_c": int(SfLayout.R128c4),
            "num_load_b_warps": 8,
            "num_load_sfb_warps": 4,
            "epilogue_regs": 152,
        }
        _run_mx_fc1(cfg, num_tokens=256, problem_n=512, problem_k=1024)

    def test_mxfp8_mxfp8_tile256_k128_tma_overlap(self):
        from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
            DType,
        )

        self._run_tile256_tma_overlap(
            dtype_a=DType.MXE4M3,
            tile_k=128,
            stages=6,
        )

    def test_mxfp8_mxfp8_tile256_k256_tma_overlap(self):
        from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
            DType,
        )

        self._run_tile256_tma_overlap(
            dtype_a=DType.MXE4M3,
            tile_k=256,
            stages=3,
        )

    def test_mxfp4_mxfp8_tile256_k256_tma_overlap_multi_n_regression(self):
        """Regression for the tile256 max-overlap handoff bug.

        The first 64-column epilogue sub-tile is loaded from the shared
        middle TMEM window.  TS used Python locals assigned inside staged
        control flow for that T2R index remap; PyIR kept the original local
        values live afterward, so the generated PTX silently used the normal
        0,1,2,3 order and left columns 192:256 of each 256-token output tile
        stale/zero for multi-N FC1 shapes.
        """
        from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
            DType,
        )

        self._run_tile256_tma_overlap(
            dtype_a=DType.MXE2M1,
            tile_k=256,
            stages=3,
            num_tokens=1024,
            problem_n=6144,
        )


class TestMxFc1QuantizedEpilogue:
    def test_mxfp4_mxfp8_to_mxfp8_tile8_tma_store(self):
        from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
            ActKind,
            BatchMode,
            DType,
            RouteImpl,
            SfLayout,
        )
        from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
            reference_check,
        )

        cfg = _finalize_tmem(
            dict(
                batch_mode=int(BatchMode.BATCH_N),
                transpose_mma_output=1,
                route_act=int(RouteImpl.TMA),
                route_sfs_act=int(RouteImpl.LDGSTS),
                act_kind=int(ActKind.SWIGLU),
                tile_m=128,
                tile_n=8,
                tile_k=256,
                epi_tile_n=8,
                mma_m=128,
                mma_n=8,
                mma_k=32,
                dtype_a=int(DType.MXE2M1),
                dtype_b=int(DType.MXE4M3),
                dtype_c=int(DType.MXE4M3),
                sf_bits=8,
                sf_block_size_a=32,
                sf_block_size_b=32,
                sf_block_size_c=32,
                sf_layout_a=int(SfLayout.R128c4),
                sf_layout_b=int(SfLayout.LINEAR),
                sf_layout_c=int(SfLayout.R8c4),
                **uniform_pipeline_stage_overrides(6),
                tile_scheduler=0,
                num_stages_tmem_acc=1,
                use_tma_store=1,
                epilogue_regs=128,
                mma_regs=48,
                load_regs=48,
                load_sf_regs=48,
                copy_sf_regs=48,
                workid_regs=48,
                padding_regs=48,
                gather_regs=48,
            )
        )
        result = reference_check(
            num_experts=2,
            num_tokens=128,
            top_k=1,
            problem_n=128,
            problem_k=256,
            seed=123,
            **cfg,
        )
        assert result
