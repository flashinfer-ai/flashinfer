from __future__ import annotations

import pytest
import torch

from b12x._lib.intrinsics import swizzle_block_scale
from b12x.moe.fused_moe._impl import (
    plan_b12x_fp4_moe_weights,
    prepare_b12x_fp4_moe_weights,
)
from b12x.moe._shared.execution import PreparedWeightLayout
from b12x.moe._shared.kernels.w4a16.host import (
    reorder_w13_to_gate_up,
    unswizzle_expert_scales,
)
from b12x.moe._shared.kernels.w4a16.prepare import (
    prepare_w4a16_compressed_tensors_weights,
    prepare_w4a16_fp4_e8m0_k32_weights,
    prepare_w4a16_modelopt_nvfp4_weights as prepare_w4a16_weights,
    prepare_w4a16_packed_weights,
)


def _positive_fp8(shape: tuple[int, ...]) -> torch.Tensor:
    return (torch.rand(shape, device="cuda") * 1.75 + 0.03125).to(torch.float8_e4m3fn)


def _e8m0_scales(shape: tuple[int, ...]) -> torch.Tensor:
    storage = torch.randint(0, 256, shape, dtype=torch.uint8, device="cuda")
    flat = storage.flatten()
    if flat.numel() >= 2:
        flat[0] = 0
        flat[1] = 255
    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    return storage.view(e8m0_dtype) if e8m0_dtype is not None else storage


def _expected_e8m0_k32_packed(
    scale_bytes: torch.Tensor,
    *,
    size_k: int,
    size_n: int,
    row_rotation: int | None = None,
) -> torch.Tensor:
    source = scale_bytes
    if row_rotation is not None:
        source = torch.cat(
            [source[:, row_rotation:], source[:, :row_rotation]],
            dim=1,
        )
    scale_perm: list[int] = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    packed = torch.empty(
        (int(source.shape[0]), int(size_k) // 32, int(size_n)),
        dtype=torch.uint8,
        device=source.device,
    )
    for expert in range(int(source.shape[0])):
        expert_packed = source[expert].T.contiguous()
        expert_packed = expert_packed.reshape(-1, len(scale_perm))[:, scale_perm]
        expert_packed = expert_packed.reshape(-1, int(size_n)).contiguous()
        expert_packed = (
            expert_packed.view(-1, 4)[:, [0, 2, 1, 3]]
            .reshape_as(expert_packed)
            .contiguous()
        )
        packed[expert].copy_(expert_packed)
    return packed


@pytest.mark.parametrize("w13_layout", ["w13", "w31"])
def test_prepare_fp4_moe_weights_modelopt_runtime_alphas_accepts_w13_layout(
    w13_layout: str,
) -> None:
    w1_global_scale = torch.tensor([[2.0], [4.0]], dtype=torch.float32)
    w2_global_scale = torch.tensor([3.0, 5.0], dtype=torch.float32)
    a1_gscale = torch.tensor([0.5, 0.25], dtype=torch.float32)
    a2_gscale = torch.tensor([1.0 / 3.0, 0.2], dtype=torch.float32)
    plan = plan_b12x_fp4_moe_weights(
        quant_modes="nvfp4",
        source_format="modelopt_nvfp4",
        activation="silu",
        params_dtype=torch.bfloat16,
        num_experts=2,
        hidden_size=128,
        intermediate_size=64,
        w13_layout=w13_layout,
    )

    prepared = prepare_b12x_fp4_moe_weights(
        plan=plan,
        w1_global_scale=w1_global_scale,
        a1_gscale=a1_gscale,
        w2_global_scale=w2_global_scale,
        a2_gscale=a2_gscale,
        w1_fp4=torch.empty((2, 128, 64), dtype=torch.uint8),
        w1_blockscale=torch.empty((2, 1), dtype=torch.uint8),
        w2_fp4=torch.empty((2, 128, 32), dtype=torch.uint8),
        w2_blockscale=torch.empty((2, 1), dtype=torch.uint8),
        params_dtype=torch.bfloat16,
    )

    w1_runtime = prepared.w1_alphas
    w2_runtime = prepared.w2_alphas
    assert w1_runtime.shape == (2,)
    assert w2_runtime.shape == (2,)
    assert prepared.w13_layout == w13_layout
    torch.testing.assert_close(w1_runtime, torch.tensor([4.0, 16.0]))
    torch.testing.assert_close(w2_runtime, torch.tensor([9.0, 25.0]))


def _reorder_swizzled_w13_to_gate_up(
    w13: torch.Tensor,
    w13_blockscale: torch.Tensor,
    *,
    intermediate_size: int,
    hidden_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    w13_scale = unswizzle_expert_scales(
        w13_blockscale,
        rows=intermediate_size * 2,
        cols=hidden_size,
    )
    w13_reordered, w13_scale_reordered = reorder_w13_to_gate_up(
        w13,
        w13_scale,
        intermediate_size=intermediate_size,
    )
    return w13_reordered, swizzle_block_scale(w13_scale_reordered)


def _make_case(
    *,
    experts: int = 3,
    hidden_size: int = 128,
    intermediate_size: int = 128,
    activation: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    is_gated = activation == "silu"
    w13_rows = intermediate_size * (2 if is_gated else 1)
    w13 = torch.randint(
        0,
        256,
        (experts, w13_rows, hidden_size // 2),
        dtype=torch.uint8,
        device="cuda",
    )
    w2 = torch.randint(
        0,
        256,
        (experts, hidden_size, intermediate_size // 2),
        dtype=torch.uint8,
        device="cuda",
    )
    w13_blockscale = swizzle_block_scale(
        _positive_fp8((experts, w13_rows, hidden_size // 16))
    )
    w2_blockscale = swizzle_block_scale(
        _positive_fp8((experts, hidden_size, intermediate_size // 16))
    )
    return w13, w13_blockscale, w2, w2_blockscale


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("activation", ["relu2", "silu"])
@pytest.mark.parametrize("params_dtype", [torch.bfloat16, torch.float16])
def test_packed_weight_preparation_shapes_and_dtypes(
    activation: str,
    params_dtype: torch.dtype,
) -> None:
    torch.manual_seed(20260514)
    experts, hidden_size, intermediate_size = 3, 128, 128
    w13_rows = intermediate_size * (2 if activation == "silu" else 1)
    w13, w13_blockscale, w2, w2_blockscale = _make_case(
        experts=experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
    )
    w13_global_scale = (torch.rand(experts, device="cuda") * 0.5 + 0.25).to(torch.float32)
    w2_global_scale = (torch.rand(experts, device="cuda") * 0.5 + 0.25).to(torch.float32)

    prepared = prepare_w4a16_weights(
        w13,
        w13_blockscale,
        w13_global_scale,
        w2,
        w2_blockscale,
        w2_global_scale,
        activation=activation,
        params_dtype=params_dtype,
    )

    assert prepared.w13.shape == (
        experts,
        hidden_size // 16,
        (w13_rows // 64) * 128,
    )
    assert prepared.w2.shape == (
        experts,
        intermediate_size // 16,
        (hidden_size // 64) * 128,
    )
    assert prepared.w13_scale.dtype == torch.float8_e4m3fn
    assert prepared.w2_scale.dtype == torch.float8_e4m3fn
    assert prepared.w13_global_scale.dtype == torch.float32
    assert prepared.w2_global_scale.dtype == torch.float32
    assert prepared.workspace.dtype == torch.int32
    assert prepared.params_dtype == params_dtype
    assert prepared.source_format == "modelopt_nvfp4"
    assert prepared.w13.is_contiguous()
    assert prepared.w2.is_contiguous()
    assert prepared.w13_scale.is_contiguous()
    assert prepared.w2_scale.is_contiguous()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("activation", ["relu2", "silu"])
@pytest.mark.parametrize("params_dtype", [torch.bfloat16, torch.float16])
def test_modelopt_nvfp4_preparation_packs_runtime_weights(
    activation: str,
    params_dtype: torch.dtype,
) -> None:
    torch.manual_seed(20260523)
    experts, hidden_size, intermediate_size = 3, 128, 128
    w13, w13_blockscale, w2, w2_blockscale = _make_case(
        experts=experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
    )
    w13_global_scale = (torch.rand(experts, device="cuda") * 0.5 + 0.25).to(torch.float32)
    w2_global_scale = (torch.rand(experts, device="cuda") * 0.5 + 0.25).to(torch.float32)

    actual = prepare_w4a16_weights(
        w13,
        w13_blockscale,
        w13_global_scale,
        w2,
        w2_blockscale,
        w2_global_scale,
        activation=activation,
        params_dtype=params_dtype,
    )

    assert actual.weight_layout == "packed"
    assert actual.source_format == "modelopt_nvfp4"
    assert actual.w13.dtype == torch.int32
    assert actual.w2.dtype == torch.int32
    assert actual.w13.data_ptr() != w13.data_ptr()
    assert actual.w2.data_ptr() != w2.data_ptr()
    assert actual.w13.shape != w13.shape
    assert actual.w2.shape != w2.shape
    assert actual.w13_scale.dtype == torch.float8_e4m3fn
    assert actual.w2_scale.dtype == torch.float8_e4m3fn


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("reuse_input_storage", [False, True])
def test_modelopt_nvfp4_w31_layout_skips_second_gated_w13_reorder(
    reuse_input_storage: bool,
) -> None:
    torch.manual_seed(20260524)
    experts, hidden_size, intermediate_size = 3, 128, 128
    w13, w13_blockscale, w2, w2_blockscale = _make_case(
        experts=experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation="silu",
    )
    w13_w31, w13_w31_blockscale = _reorder_swizzled_w13_to_gate_up(
        w13,
        w13_blockscale,
        intermediate_size=intermediate_size,
        hidden_size=hidden_size,
    )
    w13_global_scale = (torch.rand(experts, device="cuda") * 0.5 + 0.25).to(
        torch.float32
    )
    w2_global_scale = (torch.rand(experts, device="cuda") * 0.5 + 0.25).to(
        torch.float32
    )

    expected = prepare_w4a16_weights(
        w13,
        w13_blockscale,
        w13_global_scale,
        w2,
        w2_blockscale,
        w2_global_scale,
        activation="silu",
    )
    actual = prepare_w4a16_weights(
        w13_w31,
        w13_w31_blockscale,
        w13_global_scale,
        w2.clone() if reuse_input_storage else w2,
        w2_blockscale,
        w2_global_scale,
        activation="silu",
        w13_layout="w31",
        reuse_input_storage=reuse_input_storage,
    )

    assert actual.source_format == "modelopt_nvfp4"
    assert actual.w13_layout == "w31"
    if reuse_input_storage:
        assert actual.w13.data_ptr() == w13_w31.data_ptr()
    for name in (
        "w13",
        "w13_scale",
        "w13_global_scale",
        "w2",
        "w2_scale",
        "w2_global_scale",
    ):
        assert torch.equal(getattr(actual, name), getattr(expected, name)), name


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_legacy_modelopt_source_format_is_not_accepted() -> None:
    torch.manual_seed(20260523)
    experts = 3
    w13, w13_blockscale, w2, w2_blockscale = _make_case(
        experts=experts,
        activation="relu2",
    )
    w13_global_scale = (torch.rand(experts, device="cuda") * 0.5 + 0.25).to(torch.float32)
    w2_global_scale = (torch.rand(experts, device="cuda") * 0.5 + 0.25).to(torch.float32)

    with pytest.raises(ValueError, match="modelopt_nvfp4"):
        prepare_w4a16_packed_weights(
            w13,
            w13_blockscale,
            w13_global_scale,
            w2,
            w2_blockscale,
            w2_global_scale,
            activation="relu2",
            source_format="modelopt",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("activation", ["relu2", "silu"])
@pytest.mark.parametrize("params_dtype", [torch.bfloat16, torch.float16])
def test_compressed_tensors_source_matches_reciprocal_modelopt_nvfp4_contract(
    activation: str,
    params_dtype: torch.dtype,
) -> None:
    torch.manual_seed(20260520)
    experts = 3
    w13, w13_blockscale, w2, w2_blockscale = _make_case(
        experts=experts,
        activation=activation,
    )
    w13_weight_global_scale = (torch.rand(experts, device="cuda") * 0.5 + 1.25).to(
        torch.float32
    )
    w2_weight_global_scale = (torch.rand(experts, device="cuda") * 0.5 + 1.25).to(
        torch.float32
    )

    modelopt = prepare_w4a16_weights(
        w13,
        w13_blockscale,
        (1.0 / w13_weight_global_scale).to(torch.float32),
        w2,
        w2_blockscale,
        (1.0 / w2_weight_global_scale).to(torch.float32),
        activation=activation,
        params_dtype=params_dtype,
    )
    compressed_tensors = prepare_w4a16_compressed_tensors_weights(
        w13,
        w13_blockscale,
        w13_weight_global_scale,
        w2,
        w2_blockscale,
        w2_weight_global_scale,
        activation=activation,
        params_dtype=params_dtype,
    )

    for name in (
        "w13",
        "w13_scale",
        "w13_global_scale",
        "w2",
        "w2_scale",
        "w2_global_scale",
    ):
        assert torch.equal(getattr(modelopt, name), getattr(compressed_tensors, name)), name


def test_mxfp4_native_source_format_is_removed() -> None:
    with pytest.raises(ValueError, match="mxfp4_native.*removed"):
        prepare_w4a16_packed_weights(source_format="mxfp4_native")


def test_fp4_e8m0_k32_is_rejected_by_w4a4_quant_mode() -> None:
    with pytest.raises(ValueError, match="quant_mode='w4a16'"):
        plan_b12x_fp4_moe_weights(
            quant_modes="nvfp4",
            source_format="fp4_e8m0_k32",
            activation="relu2",
            params_dtype=torch.bfloat16,
            num_experts=1,
            hidden_size=4,
            intermediate_size=4,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("activation", ["relu2", "silu"])
@pytest.mark.parametrize("params_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("reuse_input_storage", [False, True])
def test_fp4_e8m0_k32_scales_clamp_high_bytes_and_keep_scale_count(
    activation: str,
    params_dtype: torch.dtype,
    reuse_input_storage: bool,
) -> None:
    torch.manual_seed(20260524)
    experts, hidden_size, intermediate_size = 3, 128, 128
    w13_rows = intermediate_size * (2 if activation == "silu" else 1)
    w13, _, w2, _ = _make_case(
        experts=experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
    )
    w13_e8m0_scale = _e8m0_scales((experts, w13_rows, hidden_size // 32))
    w2_e8m0_scale = _e8m0_scales((experts, hidden_size, intermediate_size // 32))
    w13_scale_bytes = w13_e8m0_scale.view(torch.uint8)
    w2_scale_bytes = w2_e8m0_scale.view(torch.uint8)
    w13_source_scale_bytes = w13_scale_bytes.clone()
    w2_source_scale_bytes = w2_scale_bytes.clone()
    w13_clamped_scale_bytes = w13_source_scale_bytes.clamp(max=247)
    w2_clamped_scale_bytes = w2_source_scale_bytes.clamp(max=247)
    w13_global_scale = (torch.rand(experts, device="cuda") * 0.5 + 0.25).to(torch.float32)
    w2_global_scale = (torch.rand(experts, device="cuda") * 0.5 + 0.25).to(torch.float32)

    actual = prepare_w4a16_fp4_e8m0_k32_weights(
        w13.clone() if reuse_input_storage else w13,
        w13_e8m0_scale,
        w13_global_scale,
        w2.clone() if reuse_input_storage else w2,
        w2_e8m0_scale,
        w2_global_scale,
        activation=activation,
        params_dtype=params_dtype,
        reuse_input_storage=reuse_input_storage,
    )

    expected_w13_packed = _expected_e8m0_k32_packed(
        w13_clamped_scale_bytes,
        size_k=hidden_size,
        size_n=w13_rows,
        row_rotation=intermediate_size if activation == "silu" else None,
    )
    expected_w2_packed = _expected_e8m0_k32_packed(
        w2_clamped_scale_bytes,
        size_k=intermediate_size,
        size_n=hidden_size,
    )

    assert actual.source_format == "fp4_e8m0_k32"
    assert actual.scale_format == "e8m0_k32"
    assert actual.weight_layout == "packed"
    assert actual.w13_scale.numel() == experts * (hidden_size // 32) * w13_rows
    assert actual.w2_scale.numel() == experts * (intermediate_size // 32) * hidden_size
    if reuse_input_storage:
        assert actual.w13_scale.data_ptr() == w13_e8m0_scale.data_ptr()
        assert actual.w2_scale.data_ptr() == w2_e8m0_scale.data_ptr()
        assert int(w13_scale_bytes.max().item()) <= 247
        assert int(w2_scale_bytes.max().item()) <= 247
    else:
        assert actual.w13_scale.data_ptr() != w13_e8m0_scale.data_ptr()
        assert actual.w2_scale.data_ptr() != w2_e8m0_scale.data_ptr()
        assert int(w13_scale_bytes.max().item()) == 255
        assert int(w2_scale_bytes.max().item()) == 255
    assert actual.w13_scale.view(torch.uint8).shape == expected_w13_packed.shape
    assert actual.w2_scale.view(torch.uint8).shape == expected_w2_packed.shape
    assert torch.equal(actual.w13_scale.view(torch.uint8), expected_w13_packed)
    assert torch.equal(actual.w2_scale.view(torch.uint8), expected_w2_packed)
    assert int(actual.w13_scale.view(torch.uint8).max().item()) <= 247
    assert int(actual.w2_scale.view(torch.uint8).max().item()) <= 247
    assert torch.equal(
        torch.sort(actual.w13_scale.view(torch.uint8).flatten()).values,
        torch.sort(w13_clamped_scale_bytes.flatten()).values,
    )
    assert torch.equal(
        torch.sort(actual.w2_scale.view(torch.uint8).flatten()).values,
        torch.sort(w2_clamped_scale_bytes.flatten()).values,
    )
    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    if e8m0_dtype is not None:
        assert actual.w13_scale.dtype == e8m0_dtype
        assert actual.w2_scale.dtype == e8m0_dtype

    w13_dispatch_scale = w13_source_scale_bytes.view(w13_e8m0_scale.dtype)
    w2_dispatch_scale = w2_source_scale_bytes.view(w2_e8m0_scale.dtype)
    dispatched = prepare_w4a16_packed_weights(
        w13,
        w13_dispatch_scale,
        w13_global_scale,
        w2,
        w2_dispatch_scale,
        w2_global_scale,
        activation=activation,
        params_dtype=params_dtype,
        source_format="fp4_e8m0_k32",
    )
    assert dispatched.scale_format == "e8m0_k32"
    assert torch.equal(dispatched.w13_scale.view(torch.uint8), expected_w13_packed)
    assert torch.equal(dispatched.w2_scale.view(torch.uint8), expected_w2_packed)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("activation", ["relu2", "silu"])
def test_packed_weight_preparation_can_reuse_input_storage(activation: str) -> None:
    torch.manual_seed(20260521)
    experts, hidden_size, intermediate_size = 3, 128, 128
    w13, w13_blockscale, w2, w2_blockscale = _make_case(
        experts=experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
    )
    w13_global_scale = (torch.rand(experts, device="cuda") * 0.5 + 0.25).to(torch.float32)
    w2_global_scale = (torch.rand(experts, device="cuda") * 0.5 + 0.25).to(torch.float32)

    expected = prepare_w4a16_weights(
        w13.clone(),
        w13_blockscale,
        w13_global_scale,
        w2.clone(),
        w2_blockscale,
        w2_global_scale,
        activation=activation,
    )
    actual = prepare_w4a16_weights(
        w13,
        w13_blockscale,
        w13_global_scale,
        w2,
        w2_blockscale,
        w2_global_scale,
        activation=activation,
        reuse_input_storage=True,
    )

    assert actual.w13.data_ptr() == w13.data_ptr()
    assert actual.w2.data_ptr() == w2.data_ptr()
    for name in (
        "w13",
        "w13_scale",
        "w13_global_scale",
        "w2",
        "w2_scale",
        "w2_global_scale",
    ):
        assert torch.equal(getattr(actual, name), getattr(expected, name)), name


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("activation", ["relu2", "silu"])
def test_integration_packed_preparation_uses_raw_a16_alpha_contract(
    monkeypatch: pytest.MonkeyPatch,
    activation: str,
) -> None:
    torch.manual_seed(20260522)
    monkeypatch.setenv("B12X_MOE_FORCE_A16", "1")
    experts, hidden_size, intermediate_size = 3, 128, 128
    w13, w13_blockscale, w2, w2_blockscale = _make_case(
        experts=experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
    )
    w13_alphas = (torch.rand(experts, device="cuda") * 0.5 + 0.25).to(torch.float32)
    w2_alphas = (torch.rand(experts, device="cuda") * 0.5 + 0.25).to(torch.float32)
    a1_gscale = (torch.rand(experts, device="cuda") * 0.5 + 0.75).to(torch.float32)
    a2_gscale = (torch.rand(experts, device="cuda") * 0.5 + 0.75).to(torch.float32)

    expected = prepare_w4a16_weights(
        w13.clone(),
        w13_blockscale,
        w13_alphas,
        w2.clone(),
        w2_blockscale,
        w2_alphas,
        activation=activation,
    )
    plan = plan_b12x_fp4_moe_weights(
        quant_modes="w4a16",
        source_format="modelopt_nvfp4",
        activation=activation,
        params_dtype=torch.bfloat16,
        num_experts=experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        w4a16_layout=PreparedWeightLayout.MMA_PACKED,
    )
    prepared = prepare_b12x_fp4_moe_weights(
        plan=plan,
        w1_fp4=w13,
        w1_blockscale=w13_blockscale,
        w1_global_scale=w13_alphas,
        a1_gscale=a1_gscale,
        w2_fp4=w2,
        w2_blockscale=w2_blockscale,
        w2_global_scale=w2_alphas,
        a2_gscale=a2_gscale,
        params_dtype=torch.bfloat16,
    )
    actual = prepared.representation_for("w4a16")

    assert actual is not None
    assert actual.w13.data_ptr() == w13.data_ptr()
    assert actual.w2.data_ptr() == w2.data_ptr()
    for name in (
        "w13",
        "w13_scale",
        "w13_global_scale",
        "w2",
        "w2_scale",
        "w2_global_scale",
    ):
        assert torch.equal(getattr(actual, name), getattr(expected, name)), name


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("activation", ["relu2", "silu"])
def test_integration_modelopt_nvfp4_preparation_uses_raw_weight_global_scales(
    activation: str,
) -> None:
    torch.manual_seed(20260523)
    experts, hidden_size, intermediate_size = 3, 128, 128
    w13, w13_blockscale, w2, w2_blockscale = _make_case(
        experts=experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
    )
    w13_alphas = (torch.rand(experts, device="cuda") * 0.5 + 0.25).to(
        torch.float32
    )
    w2_alphas = (torch.rand(experts, device="cuda") * 0.5 + 0.25).to(
        torch.float32
    )
    a1_gscale = (torch.rand(experts, device="cuda") * 0.5 + 0.75).to(torch.float32)
    a2_gscale = (torch.rand(experts, device="cuda") * 0.5 + 0.75).to(torch.float32)

    plan = plan_b12x_fp4_moe_weights(
        quant_modes=("nvfp4", "w4a16"),
        source_format="modelopt_nvfp4",
        activation=activation,
        params_dtype=torch.bfloat16,
        num_experts=experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    prepared = prepare_b12x_fp4_moe_weights(
        plan=plan,
        w1_fp4=w13,
        w1_blockscale=w13_blockscale,
        w1_global_scale=w13_alphas,
        a1_gscale=a1_gscale,
        w2_fp4=w2,
        w2_blockscale=w2_blockscale,
        w2_global_scale=w2_alphas,
        a2_gscale=a2_gscale,
        params_dtype=torch.bfloat16,
    )
    actual = prepared.representation_for("w4a16")

    assert actual is not None
    torch.testing.assert_close(prepared.w1_alphas, w13_alphas / a1_gscale)
    torch.testing.assert_close(prepared.w2_alphas, w2_alphas / a2_gscale)
    assert actual.weight_layout == "modelopt"
    assert actual.source_format == "modelopt_nvfp4"
    assert actual.w13.untyped_storage().data_ptr() == w13.untyped_storage().data_ptr()
    assert actual.w2.untyped_storage().data_ptr() == w2.untyped_storage().data_ptr()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_modelopt_nvfp4_gate_up_source_does_not_rotate_gated_w13_twice() -> None:
    torch.manual_seed(20260524)
    experts, hidden_size, intermediate_size = 3, 128, 128
    w13, w13_blockscale, w2, w2_blockscale = _make_case(
        experts=experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation="silu",
    )
    w13_global_scale = (torch.rand(experts, device="cuda") * 0.5 + 0.25).to(
        torch.float32
    )
    w2_global_scale = (torch.rand(experts, device="cuda") * 0.5 + 0.25).to(
        torch.float32
    )

    half = intermediate_size
    b12x_w13 = torch.cat([w13[:, half:], w13[:, :half]], dim=1).contiguous()
    b12x_w13_blockscale = torch.cat(
        [w13_blockscale[:, half:], w13_blockscale[:, :half]], dim=1
    ).contiguous()

    expected = prepare_w4a16_weights(
        w13,
        w13_blockscale,
        w13_global_scale,
        w2,
        w2_blockscale,
        w2_global_scale,
        activation="silu",
    )
    actual = prepare_w4a16_weights(
        b12x_w13,
        b12x_w13_blockscale,
        w13_global_scale,
        w2,
        w2_blockscale,
        w2_global_scale,
        activation="silu",
        w13_layout="gate_up",
    )

    assert actual.source_format == "modelopt_nvfp4"
    for name in (
        "w13",
        "w13_scale",
        "w13_global_scale",
        "w2",
        "w2_scale",
        "w2_global_scale",
    ):
        assert torch.equal(getattr(actual, name), getattr(expected, name)), name


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("activation", ["relu2", "silu"])
def test_integration_modelopt_nvfp4_preparation_can_reuse_input_storage(
    activation: str,
) -> None:
    torch.manual_seed(20260524)
    experts, hidden_size, intermediate_size = 3, 128, 128
    w13, w13_blockscale, w2, w2_blockscale = _make_case(
        experts=experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
    )
    w13_alphas = (torch.rand(experts, device="cuda") * 0.5 + 0.25).to(
        torch.float32
    )
    w2_alphas = (torch.rand(experts, device="cuda") * 0.5 + 0.25).to(
        torch.float32
    )
    a1_gscale = (torch.rand(experts, device="cuda") * 0.5 + 0.75).to(torch.float32)
    a2_gscale = (torch.rand(experts, device="cuda") * 0.5 + 0.75).to(torch.float32)

    expected = prepare_w4a16_weights(
        w13.clone(),
        w13_blockscale,
        w13_alphas,
        w2.clone(),
        w2_blockscale,
        w2_alphas,
        activation=activation,
    )
    plan = plan_b12x_fp4_moe_weights(
        quant_modes="w4a16",
        source_format="modelopt_nvfp4",
        activation=activation,
        params_dtype=torch.bfloat16,
        num_experts=experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        w4a16_layout=PreparedWeightLayout.MMA_PACKED,
    )
    actual_prepared = prepare_b12x_fp4_moe_weights(
        plan=plan,
        w1_fp4=w13,
        w1_blockscale=w13_blockscale,
        w1_global_scale=w13_alphas,
        a1_gscale=a1_gscale,
        w2_fp4=w2,
        w2_blockscale=w2_blockscale,
        w2_global_scale=w2_alphas,
        a2_gscale=a2_gscale,
        params_dtype=torch.bfloat16,
    )
    actual = actual_prepared.representation_for("w4a16")

    assert expected is not None
    assert actual is not None
    assert actual.w13.data_ptr() == w13.data_ptr()
    assert actual.w2.data_ptr() == w2.data_ptr()
    for name in (
        "w13",
        "w13_scale",
        "w13_global_scale",
        "w2",
        "w2_scale",
        "w2_global_scale",
    ):
        assert torch.equal(getattr(actual, name), getattr(expected, name)), name
