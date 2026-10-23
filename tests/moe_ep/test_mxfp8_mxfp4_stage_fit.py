# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4.stage_fit import (
    OperandStageBytes,
    compute_stage_fit,
)


def test_compute_stage_fit_accounts_for_both_scale_planes() -> None:
    operand = OperandStageBytes(
        weight_a=16 * 1024,
        activation_b=32 * 1024,
        weight_sfa=1024,
        activation_sfb=2048,
    )
    fit = compute_stage_fit(
        smem_capacity_bytes=228 * 1024,
        occupancy=1,
        operand_bytes=operand,
        fixed_bytes_per_cta=24 * 1024,
        minimum_stages=2,
        maximum_stages=4,
    )

    assert fit.stages == 4
    assert fit.bytes_per_stage == 51 * 1024
    assert fit.used_bytes == 228 * 1024
    assert fit.remaining_bytes == 0


def test_compute_stage_fit_uses_per_cta_occupancy_budget() -> None:
    operand = OperandStageBytes(4096, 8192, 256, 512)
    fit = compute_stage_fit(
        smem_capacity_bytes=128 * 1024,
        occupancy=2,
        operand_bytes=operand,
        fixed_bytes_per_cta=8192,
        minimum_stages=2,
    )

    assert fit.available_bytes == 64 * 1024
    assert fit.stages == 4


def test_compute_stage_fit_rejects_non_fitting_pipeline() -> None:
    operand = OperandStageBytes(16 * 1024, 32 * 1024, 1024, 2048)
    with pytest.raises(ValueError, match="does not fit"):
        compute_stage_fit(
            smem_capacity_bytes=64 * 1024,
            occupancy=1,
            operand_bytes=operand,
            fixed_bytes_per_cta=16 * 1024,
            minimum_stages=2,
        )
