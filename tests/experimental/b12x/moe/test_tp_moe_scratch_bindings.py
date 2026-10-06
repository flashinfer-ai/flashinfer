from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import b12x.moe.fused_moe._impl as tp_moe_impl
from b12x.moe.fused_moe._impl import (
    B12XFP4ExpertWeights,
    TPMoEFP4Binding,
    TPMoERouteBinding,
    TPMoEScratchCaps,
    TPMoESparseFP4Binding,
    build_tp_moe_route_binding,
    build_tp_moe_sparse_fp4_binding,
    plan_b12x_fp4_moe_weights,
    plan_tp_moe_scratch,
    prepare_b12x_fp4_moe_weights,
)
from b12x.moe._shared.execution import PreparedWeightLayout
from b12x.moe._shared.kernels.w4a8.weights import repack_w4a8_weights
from b12x.moe.fused_moe._tuning import MoeDecodeConfig
def _weight_plan(
    quant_mode: str = "nvfp4",
    *,
    source_format: str = "modelopt_nvfp4",
    experts: int = 8,
    k: int = 128,
    n: int = 64,
    activation: str = "silu",
    w4a16_layout: PreparedWeightLayout | None = None,
):
    return plan_b12x_fp4_moe_weights(
        quant_modes=quant_mode,
        source_format=source_format,
        activation=activation,
        params_dtype=torch.bfloat16,
        num_experts=experts,
        hidden_size=k,
        intermediate_size=n,
        w4a16_layout=w4a16_layout,
    )


def _caps(**overrides) -> TPMoEScratchCaps:
    weight_plan = overrides.pop("weight_plan", None) or _weight_plan()
    values = dict(
        device="cpu",
        max_tokens=4,
        num_topk=2,
        weight_plan=weight_plan,
        quant_mode=next(iter(weight_plan.quant_modes)),
    )
    values.update(overrides)
    return TPMoEScratchCaps(**values)


def _clear_moe_force_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "B12X_MOE_FORCE_A8",
        "B12X_FORCE_MOE_A8",
        "B12X_MOE_FORCE_A16",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("weight_layout", ["iq2_xs", "iq2_xxs", "q8_0", "trellis_t256"])
def test_prefill_route_reduction_does_not_change_other_codec_epilogues(weight_layout):
    from b12x.moe._shared.kernels.w4a16.host import prefill_fused_sum_eligible

    assert not prefill_fused_sum_eligible(
        dtype=torch.bfloat16,
        m=4096,
        full_rotation=False,
        weight_layout=weight_layout,
        collect_activation_amax=False,
        enabled=True,
    )


def test_w4a16_prefill_reduction_freezes_caller_scratch_contract(monkeypatch):
    from b12x.moe.fused_moe._tuning import MoeDecodeConfig

    monkeypatch.setattr(tp_moe_impl, "get_num_sm", lambda _device: 188)
    weight_plan = _weight_plan(
        "w4a16",
        source_format="fp4_e8m0_k32",
        experts=896,
        k=7168,
        n=192,
        activation="situ",
    )
    kwargs = dict(
        max_tokens=4096,
        core_token_counts=(4096,),
        num_topk=16,
        route_num_experts=896,
        device="cpu",
        weight_plan=weight_plan,
        quant_mode="w4a16",
        decode_config=MoeDecodeConfig(
            backend="w4a16",
            route_planner="internal",
            max_active_clusters=None,
            w4a16_route_mode="packed",
        ),
    )
    monkeypatch.setenv("B12X_W4A16_PREFILL_FUSED_SUM", "0")
    materialized_caps = TPMoEScratchCaps(**kwargs)
    monkeypatch.setenv("B12X_W4A16_PREFILL_FUSED_SUM", "1")
    fused_caps = TPMoEScratchCaps(**kwargs)
    calibrated_caps = TPMoEScratchCaps(**kwargs, collect_activation_amax=True)
    monkeypatch.setenv("B12X_W4A16_PREFILL_FUSED_SUM", "0")
    materialized, fused, calibrated = [
        plan_tp_moe_scratch(caps, prewarm_launches=False)
        for caps in (materialized_caps, fused_caps, calibrated_caps)
    ]
    specs = {spec.name: spec for spec in fused._core_workspace_plan.tensor_specs}
    assert specs["intermediate_cache13"].shape == (max(4096 * 16 * 384, 4096 * 7168),)
    assert specs["prefill_sum_accum"].shape == (4096 * 7168,)
    assert specs["prefill_sum_accum"].dtype == torch.float32
    assert fused._core_workspace_plan.prefill_fused_sum_fp32
    assert not materialized._core_workspace_plan.prefill_fused_sum_fp32
    assert not calibrated._core_workspace_plan.prefill_fused_sum_fp32
    assert (
        fused.layout.core_workspace_nbytes < materialized.layout.core_workspace_nbytes
    )
    assert calibrated.layout.core_workspace_nbytes == materialized.layout.core_workspace_nbytes


def test_dynamic_deterministic_output_is_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("B12X_DYNAMIC_DETERMINISTIC_OUTPUT", raising=False)

    assert not tp_moe_impl._dynamic_deterministic_output_enabled(
        quant_mode="nvfp4",
        device=torch.device("cuda"),
    )

    monkeypatch.setenv("B12X_DYNAMIC_DETERMINISTIC_OUTPUT", "1")

    assert tp_moe_impl._dynamic_deterministic_output_enabled(
        quant_mode="nvfp4",
        device=torch.device("cuda"),
    )
    assert not tp_moe_impl._dynamic_deterministic_output_enabled(
        quant_mode="w4a16",
        device=torch.device("cuda"),
    )
    assert not tp_moe_impl._dynamic_deterministic_output_enabled(
        quant_mode="nvfp4",
        device=torch.device("cpu"),
    )


def test_moe_force_envs_do_not_override_explicit_quant_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_moe_force_env(monkeypatch)
    monkeypatch.setenv("B12X_MOE_FORCE_A8", "1")
    monkeypatch.setenv("B12X_FORCE_MOE_A8", "1")
    monkeypatch.setenv("B12X_MOE_FORCE_A16", "1")

    assert tp_moe_impl.default_moe_quant_mode() == "nvfp4"
    assert tp_moe_impl._normalize_quant_mode(None) == "nvfp4"
    assert tp_moe_impl._normalize_quant_mode("nvfp4") == "nvfp4"
    assert tp_moe_impl._normalize_quant_mode("w4a16") == "w4a16"
    assert (
        tp_moe_impl._normalize_quant_mode_for_source(
            "w4a16",
            "fp4_e8m0_k32",
        )
        == "w4a16"
    )
    caps = _caps(
        weight_plan=_weight_plan(
            "w4a16",
            source_format="fp4_e8m0_k32",
        )
    )

    assert caps.quant_mode == "w4a16"


def test_explicit_w4a8_mx_binds_prepared_metadata() -> None:
    weight_plan = _weight_plan(
        "w4a8_mx",
        source_format="fp4_e8m0_k32",
        k=256,
        n=128,
    )
    plan = plan_tp_moe_scratch(
        _caps(
            weight_plan=weight_plan,
            route_num_experts=0,
            swiglu_limit=7.0,
        )
    )
    scratch = _scratch_for_plan(plan)
    prepared_w4a8 = SimpleNamespace(
        num_experts=8,
        hidden_size=256,
        intermediate_size=128,
        params_dtype=torch.bfloat16,
        w13_rp=torch.empty((1,), dtype=torch.int32),
        w13_sfb=torch.empty((1,), dtype=torch.int32),
        w2_rp=torch.empty((1,), dtype=torch.int32),
        w2_sfb=torch.empty((1,), dtype=torch.int32),
    )
    runtime_tensors = _runtime_tensors(k=256, n=128)
    experts = _experts(runtime_tensors, weight_plan, prepared_w4a8)

    binding = plan.bind(
        scratch=scratch,
        **_binding_args(runtime_tensors, experts),
    )

    assert binding.quant_mode == "w4a8_mx"
    assert binding.weight_E == 8
    assert binding.n == 128
    assert binding.experts is experts
    assert experts.representation_for("w4a8_mx") is prepared_w4a8
    assert binding.swiglu_limit == 7.0


def test_explicit_w4a8_mx_prepares_native_e8m0_source_in_place() -> None:
    experts, k, n = 8, 256, 128
    w1_rows = 2 * n
    w1_source = (
        torch.arange(experts * w1_rows * (k // 2), dtype=torch.int64)
        .remainder(256)
        .to(torch.uint8)
        .reshape(experts, w1_rows, k // 2)
    )
    w2_source = (
        torch.arange(experts * k * (n // 2), dtype=torch.int64)
        .remainder(256)
        .to(torch.uint8)
        .reshape(experts, k, n // 2)
    )
    w1_scale = (
        torch.arange(experts * w1_rows * (k // 32), dtype=torch.int64)
        .remainder(256)
        .to(torch.uint8)
        .reshape(experts, w1_rows, k // 32)
    )
    w2_scale = (
        torch.arange(experts * k * (n // 32), dtype=torch.int64)
        .remainder(256)
        .to(torch.uint8)
        .reshape(experts, k, n // 32)
    )
    w1_original = w1_source.clone()
    w2_original = w2_source.clone()
    w1_scale_original = w1_scale.clone()
    w2_scale_original = w2_scale.clone()

    w1_expected = torch.cat([w1_original[:, n:], w1_original[:, :n]], dim=1)
    w1_scale_expected = torch.cat(
        [w1_scale_original[:, n:], w1_scale_original[:, :n]], dim=1
    )
    expected_w13_rp, expected_w13_sfb = repack_w4a8_weights(
        w1_expected.contiguous(),
        w1_scale_expected.clamp(max=247).contiguous(),
    )
    expected_w2_rp, expected_w2_sfb = repack_w4a8_weights(
        w2_original.contiguous(),
        w2_scale_original.clamp(max=247).contiguous(),
    )

    weight_plan = plan_b12x_fp4_moe_weights(
        quant_modes="w4a8_mx",
        source_format="fp4_e8m0_k32",
        activation="silu",
        params_dtype=torch.bfloat16,
        num_experts=experts,
        hidden_size=k,
        intermediate_size=n,
        w13_layout="w31",
    )
    prepared = prepare_b12x_fp4_moe_weights(
        plan=weight_plan,
        w1_fp4=w1_source,
        w1_blockscale=w1_scale,
        w1_global_scale=torch.ones((experts,), dtype=torch.float32),
        a1_gscale=torch.ones((experts,), dtype=torch.float32),
        w2_fp4=w2_source,
        w2_blockscale=w2_scale,
        w2_global_scale=torch.ones((experts,), dtype=torch.float32),
        a2_gscale=torch.ones((experts,), dtype=torch.float32),
        params_dtype=torch.bfloat16,
    )
    w4a8 = prepared.representation_for("w4a8_mx")

    assert w4a8 is not None
    assert w4a8.num_experts == experts
    assert w4a8.hidden_size == k
    assert w4a8.intermediate_size == n
    assert (
        w4a8.w13_rp.untyped_storage().data_ptr()
        == w1_source.untyped_storage().data_ptr()
    )
    assert (
        w4a8.w2_rp.untyped_storage().data_ptr()
        == w2_source.untyped_storage().data_ptr()
    )
    assert (
        w4a8.w13_sfb.untyped_storage().data_ptr()
        == w1_scale.untyped_storage().data_ptr()
    )
    assert (
        w4a8.w2_sfb.untyped_storage().data_ptr()
        == w2_scale.untyped_storage().data_ptr()
    )
    assert torch.equal(w4a8.w13_rp, expected_w13_rp)
    assert torch.equal(w4a8.w13_sfb, expected_w13_sfb)
    assert torch.equal(w4a8.w2_rp, expected_w2_rp)
    assert torch.equal(w4a8.w2_sfb, expected_w2_sfb)


def _runtime_tensors(
    m: int = 3,
    topk: int = 2,
    *,
    experts: int = 8,
    k: int = 128,
    n: int = 64,
):
    a = torch.empty((m, k), dtype=torch.bfloat16)
    a1_gscale = torch.ones((experts,), dtype=torch.float32)
    w1_fp4 = torch.empty((experts, 2 * n, k // 2), dtype=torch.uint8)
    w1_blockscale = torch.empty((experts, 1, 1), dtype=torch.uint8)
    w1_alphas = torch.ones((experts,), dtype=torch.float32)
    a2_gscale = torch.ones((experts,), dtype=torch.float32)
    w2_fp4 = torch.empty((experts, k, n // 2), dtype=torch.uint8)
    w2_blockscale = torch.empty((experts, 1, 1), dtype=torch.uint8)
    w2_alphas = torch.ones((experts,), dtype=torch.float32)
    topk_weights = torch.empty((m, topk), dtype=torch.float32)
    topk_ids = torch.empty((m, topk), dtype=torch.int32)
    return {
        "a": a,
        "a1_gscale": a1_gscale,
        "w1_fp4": w1_fp4,
        "w1_blockscale": w1_blockscale,
        "w1_alphas": w1_alphas,
        "a2_gscale": a2_gscale,
        "w2_fp4": w2_fp4,
        "w2_blockscale": w2_blockscale,
        "w2_alphas": w2_alphas,
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
    }


def _experts(
    tensors: dict[str, torch.Tensor],
    weight_plan=None,
    payload: object | None = None,
) -> B12XFP4ExpertWeights:
    if weight_plan is None:
        weight_plan = _weight_plan(
            experts=int(tensors["w1_fp4"].shape[0]),
            k=int(tensors["w2_fp4"].shape[1]),
            n=int(tensors["w2_fp4"].shape[2]) * 2,
        )
    representation = None
    for mode in weight_plan.quant_modes:
        layout = weight_plan.required_weight_layout(mode)
        if layout is None:
            continue
        if payload is None:
            payload = SimpleNamespace()
        if mode == "w4a8_mx":
            defaults = {
                "w13_rp": tensors["w1_fp4"],
                "w13_sfb": tensors["w1_blockscale"],
                "w2_rp": tensors["w2_fp4"],
                "w2_sfb": tensors["w2_blockscale"],
            }
        else:
            defaults = {
                "w13": tensors["w1_fp4"],
                "w13_scale": tensors["w1_blockscale"],
                "w13_global_scale": tensors["w1_alphas"],
                "w2": tensors["w2_fp4"],
                "w2_scale": tensors["w2_blockscale"],
                "w2_global_scale": tensors["w2_alphas"],
                "weight_layout": (
                    "modelopt"
                    if layout is PreparedWeightLayout.SOURCE_NATIVE
                    else "packed"
                ),
                "scale_format": (
                    "e8m0_k32"
                    if weight_plan.source_format == "fp4_e8m0_k32"
                    else "e4m3_k16"
                ),
            }
        defaults.update(
            num_experts=weight_plan.num_experts,
            hidden_size=weight_plan.hidden_size,
            intermediate_size=weight_plan.intermediate_size,
            params_dtype=torch.bfloat16,
            is_gated=True,
        )
        for name, value in defaults.items():
            if not hasattr(payload, name):
                setattr(payload, name, value)
        assert representation is None
        representation = tp_moe_impl._PreparedWeightRepresentation(
            quant_mode=mode,
            layout=layout,
            value=payload,
        )
    canonical_w1 = tensors["w1_fp4"]
    canonical_s1 = tensors["w1_blockscale"]
    canonical_w2 = tensors["w2_fp4"]
    canonical_s2 = tensors["w2_blockscale"]
    if representation is not None and weight_plan.discards_source_parameters:
        value = representation.value
        canonical_w1 = getattr(value, "w13_rp", getattr(value, "w13", None))
        canonical_s1 = getattr(value, "w13_sfb", getattr(value, "w13_scale", None))
        canonical_w2 = getattr(value, "w2_rp", getattr(value, "w2", None))
        canonical_s2 = getattr(value, "w2_sfb", getattr(value, "w2_scale", None))
    return B12XFP4ExpertWeights(
        plan=weight_plan,
        a1_gscale=tensors["a1_gscale"],
        w1_fp4=canonical_w1,
        w1_blockscale=canonical_s1,
        w1_alphas=tensors["w1_alphas"],
        a2_gscale=tensors["a2_gscale"],
        w2_fp4=canonical_w2,
        w2_blockscale=canonical_s2,
        w2_alphas=tensors["w2_alphas"],
        representation=representation,
    )


def _binding_args(
    tensors: dict[str, torch.Tensor],
    experts: B12XFP4ExpertWeights,
) -> dict[str, object]:
    return {
        "a": tensors["a"],
        "experts": experts,
        "topk_weights": tensors["topk_weights"],
        "topk_ids": tensors["topk_ids"],
    }


def _scratch_for_plan(plan):
    return tuple(
        torch.empty(shape, dtype=dtype, device=plan.scratch_specs()[idx].device)
        for idx, (shape, dtype) in enumerate(plan.shapes_and_dtypes())
    )


def test_tp_moe_scratch_plan_exposes_one_opaque_scratch_spec() -> None:
    plan = plan_tp_moe_scratch(_caps())

    specs = plan.scratch_specs()
    assert len(specs) == 1
    assert specs[0].name == "tp_moe.scratch"
    assert specs[0].dtype == torch.uint8
    assert specs[0].shape == plan.shapes_and_dtypes()[0][0]
    assert specs[0].nbytes == specs[0].shape[0]
    assert plan.layout.route_workspace_nbytes > 0
    assert plan.layout.core_workspace_nbytes > 0
    assert plan.layout.total_nbytes == specs[0].nbytes


def test_tp_moe_scratch_plan_can_skip_route_scratch() -> None:
    caps = _caps(route_num_experts=0)
    plan = plan_tp_moe_scratch(caps)

    assert plan.layout.route_workspace_nbytes == 0
    assert plan.layout.core_workspace_nbytes > 0
    assert plan.scratch_specs()[0].name == "tp_moe.scratch"


def test_w4a8_mx_tp6_prefill_scratch_uses_repacked_n128_extent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GLM TP6 plans logical N=352 but launches its repacked N=384 extent."""

    monkeypatch.setattr(tp_moe_impl, "get_num_sm", lambda _device: 188)
    monkeypatch.setattr(tp_moe_impl, "_current_compute_capability", lambda: (12, 0))
    weight_plan = _weight_plan(
        "w4a8_mx",
        source_format="fp4_e8m0_k32",
        experts=256,
        k=6144,
        n=352,
    )
    execution = tp_moe_impl.plan_tp_moe_execution(
        num_tokens=4096,
        num_topk=8,
        device="cpu",
        weight_plan=weight_plan,
        quant_mode="w4a8_mx",
    )
    core = tp_moe_impl._plan_core_workspace(
        execution.implementation,
        execution.quant_mode,
        execution.state_E,
        execution.weight_E,
        execution.k,
        execution.n,
        execution.num_topk,
        execution.device,
        execution.dtype,
        routed_rows=execution.routed_rows,
        max_rows=execution.max_rows,
        activation=execution.activation,
        dynamic_physical_tiles=execution.dynamic_physical_tiles,
        dynamic_task_capacity=execution.dynamic_task_capacity,
        source_format=weight_plan.source_format,
    )
    intermediate = next(
        spec for spec in core.tensor_specs if spec.name == "materialized_intermediate"
    )
    available_bytes = tp_moe_impl._tensor_numel(
        intermediate.shape
    ) * tp_moe_impl._dtype_nbytes(intermediate.dtype)
    kernel_n = tp_moe_impl._dynamic_kernel_intermediate_size(
        execution.n, execution.quant_mode
    )
    tile_m, _ = tp_moe_impl._select_dynamic_tile_mn(
        execution.routed_rows,
        kernel_n,
        execution.quant_mode,
        num_experts=execution.state_E,
        activation=execution.activation,
        compute_capability=(12, 0),
    )
    required_bytes = (
        execution.dynamic_physical_tiles * tile_m * (kernel_n + kernel_n // 32)
    )

    assert execution.n == 352
    assert kernel_n == 384
    assert available_bytes >= required_bytes
    assert available_bytes < (
        required_bytes + execution.k * tp_moe_impl._dtype_nbytes(intermediate.dtype)
    )


def test_nvfp4_mid_atom_gate_boundary_caps_dynamic_tile_m(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("B12X_DYNAMIC_TILE_MN", raising=False)

    assert tp_moe_impl._select_dynamic_tile_mn(
        80 * 512,
        192,
        "nvfp4",
        num_experts=512,
        activation="silu",
    ) == (32, 128)
    assert tp_moe_impl._select_dynamic_tile_mn(
        80 * 512,
        256,
        "nvfp4",
        num_experts=512,
        activation="silu",
    ) == (64, 128)
    assert tp_moe_impl._select_dynamic_tile_mn(
        80 * 512,
        192,
        "nvfp4",
        num_experts=512,
        activation="relu2",
    ) == (128, 128)
    assert tp_moe_impl._select_dynamic_tile_mn(
        96 * 512,
        192,
        "nvfp4",
        num_experts=512,
        activation="silu",
    ) == (128, 128)


def test_w4a16_scratch_plan_uses_route_pack_capacity_buckets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tp_moe_impl, "get_num_sm", lambda _device: 120)

    weight_plan = _weight_plan(
        "w4a16",
        experts=256,
        k=4096,
        n=7168,
        w4a16_layout=PreparedWeightLayout.MMA_PACKED,
    )
    base_caps = dict(
        device="cpu",
        weight_plan=weight_plan,
        quant_mode="w4a16",
        num_topk=8,
        route_num_experts=0,
    )
    plan_4080 = plan_tp_moe_scratch(
        TPMoEScratchCaps(max_tokens=4080, core_token_counts=(4080,), **base_caps)
    )
    plan_4096 = plan_tp_moe_scratch(
        TPMoEScratchCaps(max_tokens=4096, core_token_counts=(4096,), **base_caps)
    )
    plan_topk6 = plan_tp_moe_scratch(
        TPMoEScratchCaps(
            max_tokens=4080,
            core_token_counts=(4080,),
            **{**base_caps, "num_topk": 6},
        )
    )

    assert plan_4080.layout.core_token_counts[0] == 4096
    assert plan_4096.layout.core_token_counts[0] == 4096
    assert plan_topk6.layout.core_token_counts[0] == 4096
    assert 4080 not in plan_4080.layout.core_token_counts
    assert 4080 not in plan_topk6.layout.core_token_counts
    assert plan_4080.shapes_and_dtypes() == plan_4096.shapes_and_dtypes()


def test_trellis_scratch_plan_preserves_exact_fixed_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trellis must not pay the generic W4A16 compile-bucket memory cost."""
    monkeypatch.setattr(tp_moe_impl, "get_num_sm", lambda _device: 188)
    weight_plan = plan_b12x_fp4_moe_weights(
        quant_modes="w4a16",
        source_format="exl3",
        trellis_codebook="mcg",
        activation="silu",
        params_dtype=torch.bfloat16,
        num_experts=256,
        hidden_size=6144,
        intermediate_size=512,
        trellis_bits=3,
        trellis_tile_config=(64, 256, 64, 256),
    )
    plan = plan_tp_moe_scratch(
        TPMoEScratchCaps(
            max_tokens=3072,
            core_token_counts=(3072,),
            num_topk=8,
            route_num_experts=0,
            device="cpu",
            weight_plan=weight_plan,
            quant_mode="w4a16",
            w4a16_block_size_m=64,
        )
    )

    assert plan.layout.core_token_counts[0] == 3072
    assert 4096 not in plan.layout.core_token_counts
    assert plan.layout.route_workspace_nbytes == 0
    # This GLM-5.2 Trellis geometry currently needs 1054.16 MiB. Keep a small
    # alignment margin while rejecting the much larger generic 4096-token
    # bucket that this fixed-capacity path exists to avoid.
    min_fixed_capacity_bytes = 1000 * (1 << 20)
    max_fixed_capacity_bytes = 1060 * (1 << 20)
    assert (
        min_fixed_capacity_bytes
        < plan.layout.core_workspace_nbytes
        < max_fixed_capacity_bytes
    )
    assert plan.layout.total_nbytes == plan.layout.core_workspace_nbytes


def test_trellis_scratch_plan_resolves_default_route_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tp_moe_impl, "get_num_sm", lambda _device: 188)
    monkeypatch.setattr(
        tp_moe_impl,
        "_plan_full_rotation_w4a16_launches",
        lambda **_kwargs: ((), ()),
    )
    weight_plan = plan_b12x_fp4_moe_weights(
        quant_modes="w4a16",
        source_format="exl3",
        trellis_codebook="mcg",
        activation="silu",
        params_dtype=torch.bfloat16,
        num_experts=256,
        hidden_size=6144,
        intermediate_size=512,
        trellis_bits=3,
        trellis_tile_config=(64, 256, 64, 256),
    )
    caps = TPMoEScratchCaps(
        max_tokens=3072,
        core_token_counts=(3072,),
        num_topk=8,
        route_num_experts=0,
        device="cpu",
        weight_plan=weight_plan,
        quant_mode="w4a16",
        decode_config=MoeDecodeConfig(
            backend="w4a16", route_planner="internal", max_active_clusters=None,
        ),
    )

    plan = plan_tp_moe_scratch(caps)

    assert plan._core_workspace_plan.route_block_size_m == 64
    assert plan.layout.core_token_counts[0] == 3072
    assert 4096 not in plan.layout.core_token_counts


@pytest.mark.parametrize(
    "shared_memory, expected_rows", [(101376, 48), (65536, 16), (131072, 64)]
)
def test_trellis_route_block_respects_fixed_tile_shared_memory(
    monkeypatch: pytest.MonkeyPatch, shared_memory: int, expected_rows: int
) -> None:
    """Packed projection tiles constrain route size before launches are compiled."""
    monkeypatch.setattr(
        torch.cuda, "get_device_properties",
        lambda _device: SimpleNamespace(shared_memory_per_block_optin=shared_memory),
    )
    plan = plan_b12x_fp4_moe_weights(
        quant_modes="w4a16", source_format="exl3", trellis_codebook="lut_e4m3",
        activation="silu", params_dtype=torch.bfloat16,
        num_experts=2, hidden_size=512, intermediate_size=256,
        trellis_bits=2, trellis_tile_config=(128, 128, 128, 128),
    )
    caps = TPMoEScratchCaps(
        max_tokens=129, num_topk=2, device="cuda:0", weight_plan=plan,
        quant_mode="w4a16",
        decode_config=MoeDecodeConfig(
            backend="w4a16", route_planner="internal", max_active_clusters=None,
        ),
    )
    assert tp_moe_impl._resolve_trellis_route_block_size(caps) == expected_rows


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_trellis_launch_planner_compiles_fixed_launch_matrix() -> None:
    """The real planner must cover every fixed decode and route-pack variant."""
    weight_plan = plan_b12x_fp4_moe_weights(
        quant_modes="w4a16",
        source_format="exl3",
        trellis_codebook="mcg",
        activation="silu",
        params_dtype=torch.bfloat16,
        num_experts=8,
        hidden_size=128,
        intermediate_size=128,
        trellis_bits=3,
        trellis_tile_config=(64, 128, 64, 128),
    )
    plan = plan_tp_moe_scratch(
        TPMoEScratchCaps(
            max_tokens=4,
            core_token_counts=(4,),
            num_topk=2,
            route_num_experts=8,
            device="cuda",
            weight_plan=weight_plan,
            quant_mode="w4a16",
            w4a16_block_size_m=8,
        )
    )

    assert tuple(tokens for tokens, _launch in plan._prewarmed_fused_launches) == (
        1,
        2,
        3,
        4,
    )
    assert {
        (ids_dtype, mapped)
        for ids_dtype, mapped, _launch in plan._prewarmed_topk_sum_launches
    } == {
        (torch.int32, False),
        (torch.int32, True),
        (torch.int64, False),
        (torch.int64, True),
    }


def test_trellis_scratch_plan_prewarms_without_forcing_runtime_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fused_launches = tuple((tokens, object()) for tokens in range(1, 5))
    sums = tuple(
        (ids_dtype, mapped, object())
        for ids_dtype in (torch.int32, torch.int64)
        for mapped in (False, True)
    )
    monkeypatch.setattr(tp_moe_impl, "get_num_sm", lambda _device: 188)
    monkeypatch.setattr(
        tp_moe_impl,
        "_plan_full_rotation_w4a16_launches",
        lambda **_kwargs: (fused_launches, sums),
    )
    weight_plan = plan_b12x_fp4_moe_weights(
        quant_modes="w4a16",
        source_format="exl3",
        trellis_codebook="mcg",
        activation="silu",
        params_dtype=torch.bfloat16,
        num_experts=8,
        hidden_size=128,
        intermediate_size=128,
        trellis_bits=3,
        trellis_tile_config=(64, 128, 64, 128),
    )
    plan = plan_tp_moe_scratch(
        TPMoEScratchCaps(
            max_tokens=4,
            core_token_counts=(4,),
            num_topk=2,
            route_num_experts=0,
            device="cpu",
            weight_plan=weight_plan,
            quant_mode="w4a16",
            w4a16_block_size_m=8,
        )
    )
    tensors = _runtime_tensors(n=128)
    captured = {}

    def _capture_binding(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(
        tp_moe_impl,
        "_build_tp_moe_fp4_binding_from_views",
        _capture_binding,
    )
    output_expert_map = torch.arange(8, dtype=torch.int32)
    plan.bind(
        scratch=_scratch_for_plan(plan),
        **_binding_args(tensors, _experts(tensors, weight_plan)),
        output_expert_map=output_expert_map,
    )

    assert plan._prewarmed_fused_launches == fused_launches
    assert plan._prewarmed_topk_sum_launches == sums
    assert captured["fused_launch"] is None
    assert captured["topk_sum_launch"] is None


@pytest.mark.parametrize("tokens", (1, 2))
def test_w4a16_decode_scratch_covers_direct_topk_geometry(
    monkeypatch: pytest.MonkeyPatch,
    tokens: int,
) -> None:
    from b12x.moe._shared.kernels.w4a16.host import (
        max_packed_route_slots,
        packed_gemm_scratch_elements,
        select_route_block_size_m,
    )

    monkeypatch.setattr(tp_moe_impl, "get_num_sm", lambda _device: 120)
    topk = 8
    hidden_size = 7168
    intermediate_size = 256
    weight_plan = _weight_plan(
        "w4a16",
        experts=1,
        k=hidden_size,
        n=intermediate_size,
        w4a16_layout=PreparedWeightLayout.MMA_PACKED,
    )
    plan = plan_tp_moe_scratch(
        _caps(
            weight_plan=weight_plan,
            max_tokens=tokens,
            num_topk=topk,
            core_token_counts=(tokens,),
            route_num_experts=0,
        )
    )
    specs = {spec.name: spec for spec in plan._core_workspace_plan.tensor_specs}
    block_size = select_route_block_size_m(tokens, topk, weight_plan.num_experts)
    packed_slots = max_packed_route_slots(
        tokens * topk,
        block_size,
        weight_plan.num_experts,
    )
    direct_slots = tokens * topk * block_size

    assert direct_slots > packed_slots
    assert tp_moe_impl._tensor_numel(specs["fc1_c_tmp"].shape) >= (
        packed_gemm_scratch_elements(
            size_n=2 * intermediate_size,
            route_slots=direct_slots,
            moe_block_size=block_size,
            sms=120,
        )
    )
    assert tp_moe_impl._tensor_numel(specs["fc2_c_tmp"].shape) >= (
        packed_gemm_scratch_elements(
            size_n=hidden_size,
            route_slots=direct_slots,
            moe_block_size=block_size,
            sms=120,
        )
    )


def test_w4a16_topk6_bucket_binds_with_planned_scratch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tp_moe_impl, "get_num_sm", lambda _device: 120)
    weight_plan = _weight_plan(
        "w4a16",
        w4a16_layout=PreparedWeightLayout.MMA_PACKED,
    )
    plan = plan_tp_moe_scratch(
        _caps(
            weight_plan=weight_plan,
            max_tokens=15,
            num_topk=6,
            core_token_counts=(15,),
            route_num_experts=0,
        )
    )
    scratch = _scratch_for_plan(plan)

    tensors = _runtime_tensors(m=15, topk=6)
    binding = plan.bind(
        scratch=scratch,
        **_binding_args(tensors, _experts(tensors, weight_plan)),
    )

    assert binding.implementation == "w4a16"
    assert binding.routed_rows_capacity is not None
    assert binding.routed_rows_capacity >= 15 * 6


def test_w4a16_materialize_can_prewarm_activation_amax_variant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}
    fused = object()

    def _fake_w4a16_prewarm(
        workspace, *, token_counts, collect_activation_amax=False, **_kwargs
    ) -> None:
        captured["collect_activation_amax"] = bool(collect_activation_amax)
        workspace.planned_fused_moe_launches = {
            (
                "packed",
                "e4m3_k16",
                int(token_count),
                bool(collect_activation_amax),
            ): fused
            for token_count in token_counts
        }
        workspace.planned_topk_sum_launches = {
            int(token_count): object() for token_count in token_counts
        }
        workspace.planned_collect_activation_amax = bool(collect_activation_amax)

    monkeypatch.setattr(tp_moe_impl, "get_num_sm", lambda _device: 120)
    monkeypatch.setattr(
        tp_moe_impl,
        "_prewarm_w4a16_planned_launches",
        _fake_w4a16_prewarm,
    )
    pool = tp_moe_impl.allocate_tp_moe_workspace_pool(frozen=True)
    weight_plan = _weight_plan(
        "w4a16",
        w4a16_layout=PreparedWeightLayout.MMA_PACKED,
    )

    tp_moe_impl.materialize_tp_moe_arena_workspaces(
        pool,
        caps=_caps(
            max_tokens=4,
            weight_plan=weight_plan,
            core_token_counts=(4,),
            route_num_experts=0,
            collect_activation_amax=True,
        ),
    )

    workspace = next(iter(pool.workspaces.values()))
    selected, topk_sum = tp_moe_impl._w4a16_preplanned_launches(
        workspace,
        token_count=4,
        weight_layout="packed",
        scale_format="e4m3_k16",
        collect_activation_amax=True,
    )

    assert captured["collect_activation_amax"] is True
    assert workspace.planned_collect_activation_amax is True
    assert selected is fused
    assert topk_sum is not None


def test_w4a16_scratch_binding_carries_activation_amax_to_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    weight_plan = _weight_plan(
        "w4a16",
        w4a16_layout=PreparedWeightLayout.MMA_PACKED,
    )
    plan = plan_tp_moe_scratch(_caps(weight_plan=weight_plan, route_num_experts=0))
    scratch = _scratch_for_plan(plan)
    tensors = _runtime_tensors()
    activation_amax = torch.zeros((3, 8, 2), dtype=torch.float32)
    output = torch.empty_like(tensors["a"])
    payload = SimpleNamespace(
        num_experts=8,
        hidden_size=128,
        intermediate_size=64,
        params_dtype=torch.bfloat16,
        is_gated=True,
        weight_layout="packed",
        scale_format="e4m3_k16",
    )
    binding = plan.bind(
        scratch=scratch,
        **_binding_args(tensors, _experts(tensors, weight_plan, payload)),
        output=output,
        activation_amax=activation_amax,
        layer_idx=2,
    )
    assert binding.expert_counts is not None
    assert tuple(binding.expert_counts.shape) == (8,)
    calls = {}

    import b12x.moe._shared.kernels.w4a16.kernel as w4a16_kernel

    def _fake_run_w4a16(*args, **kwargs):
        calls.update(kwargs)
        return kwargs["output"]

    monkeypatch.setattr(w4a16_kernel, "run_w4a16_moe", _fake_run_w4a16)

    result = tp_moe_impl.b12x_moe_fp4(binding=binding)

    assert result is output
    assert calls["activation_amax"] is activation_amax
    assert calls["layer_idx"] == 2
    assert calls["expert_counts"] is binding.expert_counts


def test_activation_amax_is_w4a16_only() -> None:
    plan = plan_tp_moe_scratch(_caps())
    scratch = _scratch_for_plan(plan)
    tensors = _runtime_tensors()
    activation_amax = torch.zeros((1, 8, 2), dtype=torch.float32)
    binding = plan.bind(
        scratch=scratch,
        **_binding_args(tensors, _experts(tensors, plan.caps.weight_plan)),
        activation_amax=activation_amax,
        layer_idx=0,
    )

    with pytest.raises(NotImplementedError, match="only supported for W4A16"):
        tp_moe_impl.b12x_moe_fp4(binding=binding)


def test_tp_moe_scratch_plan_binding_maps_caller_owned_scratch() -> None:
    plan = plan_tp_moe_scratch(_caps())
    scratch = _scratch_for_plan(plan)
    tensors = _runtime_tensors()

    binding = plan.bind(
        scratch=scratch,
        **_binding_args(tensors, _experts(tensors, plan.caps.weight_plan)),
    )

    assert isinstance(binding, TPMoEFP4Binding)
    assert binding.row_counts is not None
    assert (
        binding.row_counts.untyped_storage().data_ptr()
        == scratch[0].untyped_storage().data_ptr()
    )
    assert binding.a is tensors["a"]
    assert binding.topk_ids is tensors["topk_ids"]


def test_non_w4a16_plan_rejects_expert_maps() -> None:
    plan = plan_tp_moe_scratch(_caps())
    scratch = _scratch_for_plan(plan)
    tensors = _runtime_tensors()
    route_expert_map = torch.arange(8, dtype=torch.int32)

    with pytest.raises(ValueError, match="only supported for W4A16"):
        plan.bind(
            scratch=scratch,
            **_binding_args(tensors, _experts(tensors, plan.caps.weight_plan)),
            route_expert_map=route_expert_map,
        )


def test_packed_w4a16_plan_binds_global_route_map() -> None:
    weight_plan = _weight_plan(
        "w4a16",
        w4a16_layout=PreparedWeightLayout.MMA_PACKED,
    )
    plan = plan_tp_moe_scratch(
        _caps(
            weight_plan=weight_plan,
            route_num_experts=12,
        )
    )
    scratch = _scratch_for_plan(plan)
    tensors = _runtime_tensors()
    route_expert_map = torch.full((12,), -1, dtype=torch.int32)
    route_expert_map[:8] = torch.arange(8, dtype=torch.int32)

    binding = plan.bind(
        scratch=scratch,
        **_binding_args(tensors, _experts(tensors, weight_plan)),
        route_expert_map=route_expert_map,
    )

    assert binding.route_expert_map is route_expert_map
    assert binding.weight_E == 8
    assert plan._core_workspace_plan.route_E == 12


def test_tp_moe_scratch_plan_binds_caller_owned_scratch() -> None:
    plan = plan_tp_moe_scratch(_caps())
    scratch = _scratch_for_plan(plan)
    tensors = _runtime_tensors()

    binding = plan.bind(
        scratch=scratch,
        **_binding_args(tensors, _experts(tensors, plan.caps.weight_plan)),
    )

    assert isinstance(binding, TPMoEFP4Binding)
    assert not hasattr(binding, "workspace")
    assert not hasattr(binding, "scratch")
    assert binding.row_counts is not None
    assert binding.token_map is not None
    assert binding.packed_input is not None
    assert binding.a is tensors["a"]
    assert binding.topk_ids is tensors["topk_ids"]


def test_tp_moe_fp4_binding_rehydrates_micro_workspace_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = plan_tp_moe_scratch(_caps())
    scratch = _scratch_for_plan(plan)
    tensors = _runtime_tensors()
    output = torch.empty_like(tensors["a"])
    binding = plan.bind(
        scratch=scratch,
        **_binding_args(tensors, _experts(tensors, plan.caps.weight_plan)),
        output=output,
    )
    calls = {}

    monkeypatch.setattr(tp_moe_impl, "current_cuda_stream", lambda: None)
    monkeypatch.setattr(tp_moe_impl, "_get_weight_views", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        tp_moe_impl,
        "_launch_micro",
        lambda **kwargs: calls.update(kwargs),
    )

    result = tp_moe_impl.b12x_moe_fp4(binding=binding)

    assert result is output
    assert isinstance(calls["workspace"], tp_moe_impl.TPMicroWorkspace)
    assert calls["workspace"].active_expert_count is binding.active_expert_count
    assert calls["workspace"].micro_intermediate is binding.micro_intermediate


@pytest.mark.parametrize("layout", ["vector", "scalar", "strided", "float64"])
def test_dynamic_scale_binding_maps_views_and_refresh_observes_live_values(
    monkeypatch: pytest.MonkeyPatch, layout: str,
) -> None:
    plan = plan_tp_moe_scratch(_caps(max_tokens=400, route_num_experts=0))
    scratch = _scratch_for_plan(plan)
    tensors = _runtime_tensors(m=400)
    for name in ("a1_gscale", "a2_gscale"):
        if layout == "scalar":
            tensors[name] = torch.ones(1)
        elif layout == "strided":
            tensors[name] = torch.ones(16)[::2]
        elif layout == "float64":
            tensors[name] = tensors[name].double()
    experts = _experts(tensors, plan.caps.weight_plan)
    output = torch.empty_like(tensors["a"])

    def reject_copy(*_args, **_kwargs):
        raise AssertionError("binding must not copy scale data")

    with monkeypatch.context() as guard:
        guard.setattr(torch.Tensor, "copy_", reject_copy)
        binding = plan.bind(
            scratch=scratch, **_binding_args(tensors, experts), output=output,
        )
    workspace = SimpleNamespace(
        input_gs=binding.input_gs, down_input_scale=binding.down_input_scale,
        weight_E=8, input_gs_src_ptr=0, down_input_scale_src_ptr=0,
    )
    assert (binding.input_gs.data_ptr() == experts.a1_gscale.data_ptr()) == (
        layout == "vector"
    )
    for factor in (2, 3):
        experts.a1_gscale.mul_(factor)
        experts.a2_gscale.mul_(factor + 1)
        with monkeypatch.context() as guard:
            if layout == "vector":
                guard.setattr(torch.Tensor, "copy_", reject_copy)
            tp_moe_impl._refresh_dynamic_workspace_scales(
                workspace, experts.a1_gscale, experts.a2_gscale,
                input_scales_static=False,
            )
        torch.testing.assert_close(
            binding.input_gs, experts.a1_gscale.expand(8).float(),
        )
        torch.testing.assert_close(
            binding.down_input_scale, experts.a2_gscale.expand(8).float(),
        )


def test_tp_moe_fp4_binding_rehydrates_dynamic_workspace_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    caps = _caps(max_tokens=400, route_num_experts=0)
    plan = plan_tp_moe_scratch(caps)
    scratch = _scratch_for_plan(plan)
    tensors = _runtime_tensors(m=400)
    output = torch.empty_like(tensors["a"])
    binding = plan.bind(
        scratch=scratch,
        **_binding_args(tensors, _experts(tensors, plan.caps.weight_plan)),
        output=output,
    )
    calls = {}

    monkeypatch.setattr(tp_moe_impl, "current_cuda_stream", lambda: None)
    monkeypatch.setattr(tp_moe_impl, "_get_weight_views", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        tp_moe_impl,
        "_launch_dynamic",
        lambda **kwargs: calls.update(kwargs),
    )

    result = tp_moe_impl.b12x_moe_fp4(binding=binding)

    assert result is output
    assert binding.route_output is not None
    assert isinstance(calls["workspace"], tp_moe_impl.TPDynamicWorkspace)
    assert calls["workspace"].route_output is binding.route_output
    assert calls["workspace"].input_gs is binding.input_gs
    assert calls["workspace"].task_ready is binding.task_ready


def test_tp_moe_scratch_plan_bind_does_not_materialize_workspace_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fail_materialize(*_args, **_kwargs) -> None:
        raise AssertionError("bind must not materialize or prewarm workspaces")

    monkeypatch.setattr(
        tp_moe_impl,
        "materialize_tp_moe_arena_workspaces",
        _fail_materialize,
    )
    monkeypatch.setattr(
        tp_moe_impl,
        "_prewarm_w4a16_planned_launches",
        _fail_materialize,
    )
    monkeypatch.setattr(
        tp_moe_impl,
        "_materialize_workspace_from_core_arena",
        _fail_materialize,
    )
    monkeypatch.setattr(tp_moe_impl, "get_num_sm", lambda _device: 120)
    weight_plan = _weight_plan(
        "w4a16",
        w4a16_layout=PreparedWeightLayout.MMA_PACKED,
    )
    plan = plan_tp_moe_scratch(_caps(weight_plan=weight_plan, route_num_experts=0))
    scratch = _scratch_for_plan(plan)
    tensors = _runtime_tensors()

    binding = plan.bind(
        scratch=scratch,
        **_binding_args(tensors, _experts(tensors, weight_plan)),
    )

    assert isinstance(binding, TPMoEFP4Binding)
    assert not hasattr(binding, "workspace")
    assert not hasattr(binding, "scratch")
    assert binding.intermediate_cache13 is not None
    assert binding.intermediate_cache2 is not None
    assert binding.packed_route_indices is not None


def test_tp_moe_plan_bind_fp4_returns_common_binding_type() -> None:
    plan = plan_tp_moe_scratch(_caps())
    scratch = _scratch_for_plan(plan)
    tensors = _runtime_tensors()

    binding = plan.bind(
        scratch=scratch,
        **_binding_args(tensors, _experts(tensors, plan.caps.weight_plan)),
    )

    assert isinstance(binding, TPMoEFP4Binding)
    assert not hasattr(binding, "workspace")
    assert not hasattr(binding, "scratch")
    assert binding.row_counts is not None
    assert binding.token_map is not None
    assert binding.a is tensors["a"]
    assert binding.topk_ids is tensors["topk_ids"]


def test_tp_moe_route_builder_returns_common_binding_type() -> None:
    hidden_states = torch.empty((3, 128), dtype=torch.bfloat16)
    gate_weight = torch.empty((8, 128), dtype=torch.bfloat16)

    binding = build_tp_moe_route_binding(
        hidden_states=hidden_states,
        top_k=2,
        gate_weight=gate_weight,
    )

    assert isinstance(binding, TPMoERouteBinding)
    assert not hasattr(binding, "workspace")
    assert binding.scratch is None
    assert binding.hidden_states is hidden_states
    assert binding.gate_weight is gate_weight


def test_tp_moe_sparse_fp4_builder_returns_common_binding_type() -> None:
    scratch = tp_moe_impl.TPMoEWorkspacePool()
    tensors = _runtime_tensors()
    experts = _experts(tensors)

    binding = build_tp_moe_sparse_fp4_binding(
        scratch=scratch,
        hidden_states=tensors["a"],
        experts=experts,
        routing=tp_moe_impl.B12XTopKRouting(
            topk_weights=tensors["topk_weights"],
            topk_ids=tensors["topk_ids"],
        ),
    )

    assert isinstance(binding, TPMoESparseFP4Binding)
    assert not hasattr(binding, "workspace")
    assert binding.scratch is scratch
    assert binding.hidden_states is tensors["a"]
    assert binding.experts is experts


def test_tp_moe_fp4_binding_run_uses_function_binding_argument(monkeypatch) -> None:
    plan = plan_tp_moe_scratch(_caps())
    scratch = _scratch_for_plan(plan)
    tensors = _runtime_tensors()
    binding = plan.bind(
        scratch=scratch,
        **_binding_args(tensors, _experts(tensors, plan.caps.weight_plan)),
    )
    calls = {}
    sentinel = object()

    def fake_moe_fp4(**kwargs):
        calls.update(kwargs)
        return sentinel

    monkeypatch.setattr(tp_moe_impl, "b12x_moe_fp4", fake_moe_fp4)

    assert binding.run() is sentinel
    assert calls["binding"] is binding


def test_tp_moe_route_binding_run_uses_function_binding_argument(monkeypatch) -> None:
    hidden_states = torch.empty((3, 128), dtype=torch.bfloat16)
    gate_weight = torch.empty((8, 128), dtype=torch.bfloat16)
    binding = build_tp_moe_route_binding(
        hidden_states=hidden_states,
        top_k=2,
        gate_weight=gate_weight,
    )
    calls = {}
    sentinel = object()

    def fake_route(**kwargs):
        calls.update(kwargs)
        return sentinel

    monkeypatch.setattr(tp_moe_impl, "b12x_route_experts_fast", fake_route)

    assert binding.run() is sentinel
    assert calls["binding"] is binding


def test_tp_moe_sparse_fp4_binding_run_uses_function_binding_argument(
    monkeypatch,
) -> None:
    scratch = tp_moe_impl.TPMoEWorkspacePool()
    tensors = _runtime_tensors()
    binding = build_tp_moe_sparse_fp4_binding(
        scratch=scratch,
        hidden_states=tensors["a"],
        experts=_experts(tensors),
        routing=tp_moe_impl.B12XTopKRouting(
            topk_weights=tensors["topk_weights"],
            topk_ids=tensors["topk_ids"],
        ),
    )
    calls = {}
    sentinel = object()

    def fake_sparse(**kwargs):
        calls.update(kwargs)
        return sentinel

    monkeypatch.setattr(tp_moe_impl, "b12x_sparse_moe_fp4", fake_sparse)

    assert binding.run() is sentinel
    assert calls["binding"] is binding


def test_tp_moe_fp4_binding_owns_runtime_tensors() -> None:
    plan = plan_tp_moe_scratch(_caps())
    scratch = _scratch_for_plan(plan)
    tensors = _runtime_tensors()
    binding = plan.bind(
        scratch=scratch,
        **_binding_args(tensors, _experts(tensors, plan.caps.weight_plan)),
    )

    with pytest.raises(TypeError):
        tp_moe_impl.b12x_moe_fp4(tensors["a"], binding=binding)


def test_tp_moe_route_binding_owns_runtime_tensors() -> None:
    hidden_states = torch.empty((3, 128), dtype=torch.bfloat16)
    gate_weight = torch.empty((8, 128), dtype=torch.bfloat16)
    binding = build_tp_moe_route_binding(
        hidden_states=hidden_states,
        top_k=2,
        gate_weight=gate_weight,
    )

    with pytest.raises(TypeError):
        tp_moe_impl.b12x_route_experts_fast(hidden_states, binding=binding)


def test_tp_moe_sparse_fp4_binding_owns_runtime_tensors() -> None:
    scratch = tp_moe_impl.TPMoEWorkspacePool()
    tensors = _runtime_tensors()
    experts = _experts(tensors)
    binding = build_tp_moe_sparse_fp4_binding(
        scratch=scratch,
        hidden_states=tensors["a"],
        experts=experts,
        routing=tp_moe_impl.B12XTopKRouting(
            topk_weights=tensors["topk_weights"],
            topk_ids=tensors["topk_ids"],
        ),
    )

    with pytest.raises(TypeError):
        tp_moe_impl.b12x_sparse_moe_fp4(
            tensors["a"],
            experts=experts,
            binding=binding,
        )


def test_tp_moe_fp4_entrypoint_requires_tensors_or_binding() -> None:
    with pytest.raises(TypeError):
        tp_moe_impl.b12x_moe_fp4()


def test_tp_moe_route_entrypoint_requires_inputs_or_binding() -> None:
    with pytest.raises(TypeError):
        tp_moe_impl.b12x_route_experts_fast()


def test_tp_moe_sparse_fp4_entrypoint_requires_inputs_or_binding() -> None:
    with pytest.raises(TypeError):
        tp_moe_impl.b12x_sparse_moe_fp4()
