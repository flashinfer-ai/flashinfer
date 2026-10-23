# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused host checks for the MXFP8 x MXFP4 FC1 epilogue contract."""

import pytest

cutlass = pytest.importorskip("cutlass")

from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4.fc1_epilogue import (  # noqa: E402
    Fc1Epilogue,
    Fc1EpilogueConfig,
    handoff_storage_bytes,
    validate_fc1_epilogue_contract,
)


def test_fc1_epilogue_dtype_contract() -> None:
    validate_fc1_epilogue_contract(
        acc_dtype=cutlass.Float32,
        output_dtype=cutlass.Float8E4M3FN,
        scale_dtype=cutlass.Float8E8M0FNU,
        sf_vec_size=32,
    )

    with pytest.raises(ValueError, match="accumulators"):
        validate_fc1_epilogue_contract(
            acc_dtype=cutlass.BFloat16,
            output_dtype=cutlass.Float8E4M3FN,
            scale_dtype=cutlass.Float8E8M0FNU,
            sf_vec_size=32,
        )
    with pytest.raises(ValueError, match="sf_vec_size=32"):
        validate_fc1_epilogue_contract(
            acc_dtype=cutlass.Float32,
            output_dtype=cutlass.Float8E4M3FN,
            scale_dtype=cutlass.Float8E8M0FNU,
            sf_vec_size=16,
        )


def test_fc1_epilogue_rejects_negative_clamp() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        Fc1EpilogueConfig(gate_up_clamp=-1.0)


def test_fc1_epilogue_policy_keeps_routing_outside_fc1() -> None:
    epilogue = Fc1Epilogue(Fc1EpilogueConfig(gate_up_clamp=10.0, fast_math=False))
    assert epilogue.config.gate_up_clamp == 10.0
    assert epilogue.config.fast_math is False
    # No score/top-k policy exists on the FC1 component.  Routing belongs to
    # FC2/output reduction per README.md.
    assert not hasattr(epilogue.config, "routing_weight")
    assert not hasattr(epilogue.config, "apply_topk")


@pytest.mark.parametrize(
    ("tokens", "intermediate", "expected"),
    [
        (0, 32, (0, 0)),
        (1, 32, (32, 1)),
        (17, 256, (4352, 136)),
    ],
)
def test_handoff_workspace_is_e4m3_plus_one_e8m0_per_k32(
    tokens, intermediate, expected
) -> None:
    assert (
        handoff_storage_bytes(token_count=tokens, intermediate_size=intermediate)
        == expected
    )


def test_handoff_workspace_rejects_partial_scale_blocks() -> None:
    with pytest.raises(ValueError, match="multiple of 32"):
        handoff_storage_bytes(token_count=4, intermediate_size=48)
