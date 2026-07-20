# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
"""moe declarative execution model: MoESpec / ExecutionPlan planner contracts
(CPU-only). Ported from b12x tests/test_moe_execution_model.py; exercises the
public plan_weights/plan_execution verbs plus the _shared.execution vocabulary.
"""

from __future__ import annotations

import pytest
import torch

from flashinfer.experimental.sm12x.moe import fused_moe
from flashinfer.experimental.sm12x.moe.fused_moe import _impl as tp_moe_impl
from flashinfer.experimental.sm12x.moe._shared.execution import (
    GemmEngine,
    MoERegime,
    OperandEncoding,
    PreparedWeightLayout,
    RouteLayout,
    ScaleEncoding,
    WorkAvailability,
    WorkScheduler,
    WeightPreparationTransform,
    WeightStoragePolicy,
    lower_moe_execution,
    make_moe_spec,
)

plan_sm12x_fp4_moe_weights = fused_moe.plan_weights
plan_tp_moe_execution = fused_moe.plan_execution


def _weight_plan(
    quant_modes: str | tuple[str, ...],
    *,
    source_format: str,
    k: int = 128,
    n: int = 128,
    w4a16_layout: PreparedWeightLayout | None = None,
):
    return plan_sm12x_fp4_moe_weights(
        quant_modes=quant_modes,
        source_format=source_format,
        activation="silu",
        params_dtype=torch.bfloat16,
        num_experts=32,
        hidden_size=k,
        intermediate_size=n,
        w4a16_layout=w4a16_layout,
    )


@pytest.mark.parametrize(
    (
        "quant_mode",
        "source_format",
        "activation_encoding",
        "source_scale",
        "compute_scale",
    ),
    [
        (
            "nvfp4",
            "modelopt_nvfp4",
            OperandEncoding.FP4_E2M1,
            ScaleEncoding.E4M3_K16,
            ScaleEncoding.E4M3_K16,
        ),
        (
            "w4a8_nvfp4",
            "modelopt_nvfp4",
            OperandEncoding.MXFP8_E4M3,
            ScaleEncoding.E4M3_K16,
            ScaleEncoding.E8M0_K32_X_E4M3_K16_RESIDUAL,
        ),
        (
            "w4a8_mx",
            "fp4_e8m0_k32",
            OperandEncoding.MXFP8_E4M3,
            ScaleEncoding.E8M0_K32,
            ScaleEncoding.E8M0_K32,
        ),
        (
            "w4a16",
            "compressed_tensors",
            OperandEncoding.BF16,
            ScaleEncoding.E4M3_K16,
            ScaleEncoding.E4M3_K16,
        ),
    ],
)
def test_moe_spec_separates_source_and_compute_formats(
    quant_mode: str,
    source_format: str,
    activation_encoding: OperandEncoding,
    source_scale: ScaleEncoding,
    compute_scale: ScaleEncoding,
) -> None:
    spec = make_moe_spec(
        quant_mode=quant_mode,
        source_format=source_format,
        activation="silu",
        io_dtype="bfloat16",
        w13_layout="w13",
    )

    assert spec.activation_encoding is activation_encoding
    assert spec.source_weight_scale is source_scale
    assert spec.weight_scale is compute_scale


def test_moe_spec_rejects_invalid_source_quant_pair() -> None:
    with pytest.raises(ValueError, match="incompatible"):
        make_moe_spec(
            quant_mode="w4a8_mx",
            source_format="modelopt_nvfp4",
            activation="silu",
            io_dtype="bfloat16",
            w13_layout="w13",
        )


def test_same_w4a8_numerics_lower_to_queue_or_grid() -> None:
    spec = make_moe_spec(
        quant_mode="w4a8_mx",
        source_format="fp4_e8m0_k32",
        activation="silu",
        io_dtype="bfloat16",
        w13_layout="w13",
    )

    queued = lower_moe_execution(
        spec,
        regime=MoERegime.MATERIALIZED_FUSED,
        scheduler=WorkScheduler.ATOMIC_QUEUE,
        tile_m=32,
        tile_n=128,
    )
    grid = lower_moe_execution(
        spec,
        regime=MoERegime.MATERIALIZED_FUSED,
        scheduler=WorkScheduler.PERSISTENT_GRID,
        tile_m=32,
        tile_n=128,
    )

    assert queued.gemm_engine is GemmEngine.MXFP8_QMMA
    assert grid.gemm_engine is GemmEngine.MXFP8_QMMA
    assert queued.work_availability is WorkAvailability.PRECOMPUTED
    assert queued.scheduler is WorkScheduler.ATOMIC_QUEUE
    assert queued.uses_explicit_task_queue
    assert queued.grid_addressable
    assert grid.work_availability is WorkAvailability.PRECOMPUTED
    assert grid.scheduler is WorkScheduler.PERSISTENT_GRID
    assert grid.grid_addressable
    assert grid.weight_layout is PreparedWeightLayout.QMMA_REPACKED
    assert queued.weight_layout is PreparedWeightLayout.QMMA_REPACKED


def test_materialized_fused_work_can_use_grid_or_load_balancing_queue() -> None:
    spec = make_moe_spec(
        quant_mode="nvfp4",
        source_format="modelopt_nvfp4",
        activation="silu",
        io_dtype="bfloat16",
        w13_layout="w13",
    )

    grid = lower_moe_execution(
        spec,
        regime=MoERegime.MATERIALIZED_FUSED,
        scheduler=WorkScheduler.PERSISTENT_GRID,
        tile_m=128,
        tile_n=128,
    )
    queued = lower_moe_execution(
        spec,
        regime=MoERegime.MATERIALIZED_FUSED,
        scheduler=WorkScheduler.ATOMIC_QUEUE,
        tile_m=128,
        tile_n=128,
    )

    assert grid.work_availability is WorkAvailability.PRECOMPUTED
    assert queued.work_availability is WorkAvailability.PRECOMPUTED
    assert grid.grid_addressable and queued.grid_addressable
    assert not grid.uses_explicit_task_queue
    assert queued.uses_explicit_task_queue
    assert grid.gemm_engine is queued.gemm_engine is GemmEngine.NVFP4_MMA


def test_workspace_plan_maps_kernel_family_to_execution_contract() -> None:
    dynamic_weights = _weight_plan("nvfp4", source_format="modelopt_nvfp4")
    dynamic = plan_tp_moe_execution(
        num_tokens=64,
        num_topk=4,
        device=torch.device("cpu"),
        weight_plan=dynamic_weights,
        quant_mode="nvfp4",
    )
    w4a16_weights = _weight_plan("w4a16", source_format="fp4_e8m0_k32")
    w4a16 = plan_tp_moe_execution(
        num_tokens=64,
        num_topk=4,
        device=torch.device("cpu"),
        weight_plan=w4a16_weights,
        quant_mode="w4a16",
    )

    assert dynamic.implementation == "dynamic"
    assert dynamic.execution.regime is MoERegime.MATERIALIZED_FUSED
    assert dynamic.execution.route_layout is RouteLayout.APPEND_ONLY_EXPERT
    assert dynamic.execution.work_availability is WorkAvailability.PRECOMPUTED
    assert dynamic.execution.scheduler is WorkScheduler.ATOMIC_QUEUE
    assert dynamic.execution.uses_explicit_task_queue
    assert dynamic.execution.tile_m == 16
    assert w4a16.implementation == "w4a16"
    assert w4a16.spec.source_weight_scale is ScaleEncoding.E8M0_K32
    assert w4a16.execution.route_layout is RouteLayout.SORTED_PADDED_EXPERT
    assert w4a16.execution.scheduler is WorkScheduler.PERSISTENT_GRID
    assert w4a16.execution.route_block_rows is not None


def test_workspace_plan_uses_weight_plan_source_contract() -> None:
    weights = _weight_plan(
        "w4a8_mx",
        source_format="fp4_e8m0_k32",
        k=256,
        n=128,
    )
    plan = plan_tp_moe_execution(
        num_tokens=4,
        num_topk=2,
        device=torch.device("cpu"),
        weight_plan=weights,
        quant_mode="w4a8_mx",
    )

    assert plan.spec.source_format == "fp4_e8m0_k32"
    assert plan.spec.source_weight_scale is ScaleEncoding.E8M0_K32


def test_native_w4a8_m1_alone_selects_fixed_materialized_regime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """M=1 specializes unified dynamic; neighboring decode sizes do not."""

    monkeypatch.setenv("FLASHINFER_EXP_SM12X_DYNAMIC_WORK_SOURCE", "materialized_queue")
    monkeypatch.setenv("FLASHINFER_EXP_SM12X_DYNAMIC_W4A8_MATERIALIZED", "1")
    common = dict(
        quant_mode="w4a8_mx",
        activation="silu",
        num_experts=256,
        k=6144,
        n=1024,
        w4a8_repacked=True,
        share_input_across_experts=True,
        deterministic_output=False,
    )

    assert tp_moe_impl._w4a8_dynamic_materialized_enabled(
        num_tokens=1,
        routed_rows=8,
        **common,
    )
    assert not tp_moe_impl._w4a8_dynamic_materialized_enabled(
        num_tokens=2,
        routed_rows=16,
        **common,
    )
    assert not tp_moe_impl._w4a8_dynamic_materialized_enabled(
        num_tokens=1,
        routed_rows=8,
        **(common | {"w4a8_repacked": False}),
    )
    assert not tp_moe_impl._w4a8_dynamic_materialized_enabled(
        num_tokens=1,
        routed_rows=40,
        **common,
    )


def test_source_native_w4a16_and_nvfp4_share_one_allocation() -> None:
    plan = _weight_plan(
        ("nvfp4", "w4a16"),
        source_format="modelopt_nvfp4",
    )

    assert plan.storage_policy is WeightStoragePolicy.KEEP_SOURCE
    assert plan.transforms == {
        WeightPreparationTransform.RUNTIME_ALPHAS,
        WeightPreparationTransform.W4A16_NATIVE,
    }
    assert plan.required_weight_layout("nvfp4") is None
    assert plan.required_weight_layout("w4a16") is PreparedWeightLayout.SOURCE_NATIVE


def test_planner_rejects_source_plus_model_sized_repack() -> None:
    with pytest.raises(ValueError, match="both source-native weights"):
        _weight_plan(
            ("nvfp4", "w4a16"),
            source_format="modelopt_nvfp4",
            w4a16_layout=PreparedWeightLayout.MMA_PACKED,
        )


def test_planner_rejects_two_incompatible_repacks() -> None:
    with pytest.raises(ValueError, match="multiple incompatible model-sized"):
        _weight_plan(
            ("w4a8_mx", "w4a16"),
            source_format="fp4_e8m0_k32",
            k=256,
            n=128,
            w4a16_layout=PreparedWeightLayout.MMA_PACKED,
        )


def test_storage_policy_has_no_keep_both_state() -> None:
    assert "keep_both" not in {policy.value for policy in WeightStoragePolicy}


def test_non_128_aligned_e8m0_w4a16_shards_stay_source_native() -> None:
    """2048/TP6 = 352 and 3072/TP16 = 192 have no packed tile configs; the
    planner must route them through the native layout instead of failing at
    kernel-config time. 128-aligned shards keep the packed fast path."""
    for shard in (352, 192):
        plan = _weight_plan("w4a16", source_format="fp4_e8m0_k32", n=shard)
        assert WeightPreparationTransform.W4A16_NATIVE in plan.transforms
        assert (
            plan.required_weight_layout("w4a16") is PreparedWeightLayout.SOURCE_NATIVE
        )

    aligned = _weight_plan("w4a16", source_format="fp4_e8m0_k32", n=256)
    assert WeightPreparationTransform.W4A16_PACKED in aligned.transforms
    assert aligned.required_weight_layout("w4a16") is PreparedWeightLayout.MMA_PACKED
