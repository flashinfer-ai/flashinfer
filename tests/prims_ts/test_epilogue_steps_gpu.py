# Copyright (c) 2026 FlashInfer contributors.
# SPDX-License-Identifier: Apache-2.0

"""Explicit schedules on real nonzero inputs, persistent tiles and graph replay."""

import pytest
import torch

from flashinfer.prims_ts.batched_gemm.epilogue_schedule import (
    EpilogueStep as Step,
    epilogue_steps_for_prefix,
    fc1_streaming_steps,
)


def full_tile_steps():
    return (Step(load_fragments=1, stage_fragments=1),) * 8 + (
        Step(release_tmem=True),
        Step(emit_fragments=8),
    )


@pytest.mark.arch_rubin
@pytest.mark.timeout(300)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "warps,stages,prefix,ahead",
    [
        (4, 2, 8, 0),
        (4, 1, 2, 0),
        (4, 2, 1, 1),
        (8, 1, 2, 1),
        (8, 2, 0, 1),
        (8, 2, 8, 0),
        (4, 1, -1, 1),
        (8, 2, -1, 1),
    ],
)
def test_fused_step_replay(monkeypatch, warps, stages, prefix, ahead):
    from tests.prims_ts.test_moe_fused_finalize import test_fused_finalize_replay

    steps = (
        full_tile_steps()
        if prefix == 8
        else epilogue_steps_for_prefix(max(0, prefix), read_ahead=ahead)
    )
    if prefix == -1:
        steps = (
            Step(load_fragments=4, stage_fragments=4, read_ahead=ahead),
            Step(emit_fragments=4),
        ) * 2 + (Step(release_tmem=True),)
    test_fused_finalize_replay(
        monkeypatch, "BlockMajorK", [1024, 0], 0, stages, warps, 0, -1, steps
    )


@pytest.mark.arch_rubin
@pytest.mark.timeout(300)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "warps,stages,prefix,ahead",
    [
        (4, 1, 1, 1),
        (4, 2, 2, 0),
        (8, 1, 0, 1),
        (8, 2, 2, 1),
    ],
)
def test_plain_step_reference(
    warps,
    stages,
    prefix,
    ahead,
    mode="dense-bf16",
    layout="BlockMajorK",
    dtype="BF16",
    steps=(),
):
    from tests.prims_ts.test_batched_gemm_rubin_shuffled import shuffled_profile
    from flashinfer.prims_ts.batched_gemm import batched_gemm_run as runner
    from flashinfer.tllm_enums import WeightLayout
    from flashinfer.prims_ts.batched_gemm.config_types import DType

    options = shuffled_profile(mode, (2, 2), WeightLayout[layout])
    options.update(
        dense_epilogue_warps=warps,
        epilogue_regs=248 if warps == 4 else 168,
        num_stages_c_smem=stages,
        num_stages_b=5,
        num_stages_smem_sfb=5,
        dense_preload_store_prefix_fragments=0,
        epilogue_steps=steps
        or epilogue_steps_for_prefix(prefix, finalize=False, read_ahead=ahead),
        use_global_scales=1,
        dtype_c=int(DType[dtype]),
    )
    assert runner.reference_check(
        num_experts=8,
        num_tokens=4099,
        top_k=4,
        problem_n=1280,
        problem_k=2304,
        routing="uniform",
        scale_c_value=0.7,
        repeat_launches=3,
        output_guard_elements=256,
        **options,
    )


@pytest.mark.arch_rubin
@pytest.mark.timeout(300)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "mode,dtype", [("dense-bf16", "FP16"), ("routed-bf16", "BF16")]
)
def test_plain_step_other_layouts(monkeypatch, mode, dtype):
    from flashinfer.prims_ts.batched_gemm import batched_gemm_run as runner

    def bounded_sf(shape, *, cfg, device, layout_kind):
        return runner._random_uint8(shape, low=0x28, high=0x41, device=device)

    monkeypatch.setattr(runner, "_random_sf_bytes", bounded_sf)
    test_plain_step_reference(4, 2, 1, 1, mode, "MajorK", dtype)


@pytest.mark.arch_rubin
@pytest.mark.timeout(300)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_plain_step_grouped_emission():
    # Two TMA transactions must commit on the same fixed lane, and one wait
    # must retire both source slots before the following pair overwrites them.
    steps = (
        Step(load_fragments=2, stage_fragments=2, read_ahead=1),
        Step(emit_fragments=2),
    ) * 4 + (Step(release_tmem=True),)
    test_plain_step_reference(8, 2, 0, 0, steps=steps)


@pytest.mark.arch_rubin
@pytest.mark.timeout(300)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "warps,stages,prefix,ahead,layout",
    [
        (8, 2, 4, 0, "BlockMajorK"),
        (4, 1, 0, 1, "MajorK"),
        (8, 1, 0, 1, "MajorK"),
        (4, 2, 2, 0, "BlockMajorK"),
        (8, 1, -1, 0, "BlockMajorK"),
        (8, 2, -2, 0, "BlockMajorK"),
        (4, 1, -2, 0, "MajorK"),
    ],
)
def test_fc1_step_reference(warps, stages, prefix, ahead, layout, streaming=0):
    from tests.prims_ts.test_batched_gemm_rubin_shuffled import shuffled_profile
    from flashinfer.prims_ts.batched_gemm import batched_gemm_run as runner
    from flashinfer.tllm_enums import WeightLayout

    from flashinfer.utils import get_compute_capability

    if get_compute_capability(torch.device("cuda")) != (10, 7):
        pytest.skip("Native FC1 steps require Rubin")
    options = shuffled_profile("fc1", (2, 2), WeightLayout[layout], group_size=1)
    options.update(
        dense_preload_epilogue=1,
        dense_epilogue_warps=warps,
        epilogue_regs=248 if warps == 4 else (168 if ahead else 144),
        num_stages_c_smem=stages,
        epilogue_steps=epilogue_steps_for_prefix(
            max(0, prefix), finalize=False, gated=True, warps=warps, read_ahead=ahead
        ),
        use_global_scales=1,
        dense_streaming_c=streaming,
    )
    if prefix == -1:
        options["epilogue_steps"] = (
            Step(load_fragments=1, stage_fragments=1),
            Step(load_fragments=1, stage_fragments=1),
            Step(emit_fragments=2),
        ) * (16 // warps) + (Step(release_tmem=True),)
    if prefix == -2:
        options["epilogue_steps"] = fc1_streaming_steps(warps=warps)
    if warps == 8 and ahead:
        # Read-ahead needs an extra live N64 fragment. Donate registers from
        # the eight A-loader warps so the complete CTA remains admissible.
        options["load_a_regs"] = 32
        options["load_sfa_regs"] = 32
    problem = dict(
        num_experts=8,
        num_tokens=4099,
        top_k=4,
        problem_n=1280,
        problem_k=2304,
        routing="uniform",
        scale_c_value=0.7,
        repeat_launches=3,
        output_guard_elements=256,
    )
    if prefix != -2:
        assert runner.reference_check(**problem, **options)
    else:
        # The paired lowering changes scheduling, not activation/quantization.
        # Compare only routed rows: inactive padding has no output contract.
        result, output = runner.reference_check(
            **problem, **options, return_output=True
        )
        legacy_options = dict(options, dense_preload_epilogue=0, epilogue_steps=())
        legacy_result, legacy_output = runner.reference_check(
            **problem, **legacy_options, return_output=True
        )
        assert result and legacy_result
        cfg = runner.make_config(**options)
        layout = runner._make_token_layout(
            num_tokens=problem["num_tokens"],
            num_experts=problem["num_experts"],
            top_k=problem["top_k"],
            tile_size=cfg.token_tile_size,
            cluster_dim_in_token=cfg.token_cluster_dim,
            routing=problem["routing"],
        )
        valid = torch.tensor(layout.expanded_to_expert, device="cuda") >= 0
        assert valid.sum().item() == problem["num_tokens"] * problem["top_k"]
        torch.testing.assert_close(output[valid], legacy_output[valid], rtol=0, atol=0)


@pytest.mark.arch_rubin
@pytest.mark.timeout(300)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "warps,stages,layout,tactic",
    [
        (8, 2, "BlockMajorK", [1024, 0]),
        (4, 1, "MajorK", [512, 0]),
    ],
)
@pytest.mark.parametrize("streaming", [False, True])
def test_fc1_step_moe_replay(monkeypatch, warps, stages, layout, tactic, streaming):
    from tests.prims_ts.test_moe_fused_finalize import test_fused_finalize_replay

    test_fused_finalize_replay(
        monkeypatch,
        layout,
        tactic,
        0,
        2,
        4,
        0,
        -1,
        fc1_overrides=dict(
            dense_preload_epilogue=1,
            dense_epilogue_warps=warps,
            num_stages_c_smem=stages,
            epilogue_regs=248 if warps == 4 else 144,
            epilogue_steps=fc1_streaming_steps(warps=warps)
            if streaming
            else epilogue_steps_for_prefix(
                0, finalize=False, gated=True, warps=warps, read_ahead=1
            ),
        ),
    )


@pytest.mark.arch_rubin
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fc1_step_streaming_reference():
    test_fc1_step_reference(8, 2, 4, 0, "BlockMajorK", streaming=1)
