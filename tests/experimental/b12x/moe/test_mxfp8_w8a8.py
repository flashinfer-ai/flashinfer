"""Native MXFP8 (``w8a8_mx``) fused MoE: preparation contract and kernel path.

Two lanes live here:

*CPU (always runs).* Lossless weight preparation plus the planner/execution
vocabulary.  The load-bearing property of this recipe is that the checkpoint's
E4M3 bytes and UE8M0 K/32 scale bytes reach the MMA operand buffers unchanged
(there is no FP4/FP6 requantization to hide behind), so the scale swizzle is
checked byte-for-byte against an independently derived index mapping rather
than against ``swizzle_block_scale`` itself.

*GPU (SM120/SM121).* True numerical execution against the pure-torch oracle on
the Qwen step5500 MTP expert geometry (hidden 2560, intermediate 640, top-k
10), plus the shapes the grouped kernel must survive: experts with no routes,
partial 128-row route tiles, and CUDA-graph replay with kernel resolution
frozen.
"""

from __future__ import annotations

import pytest

import torch

from b12x.preparation import FrozenMapping

from ..conftest import require_b12x

# hidden_size / moe_intermediate_size / num_experts_per_tok are the checkpoint
# values; only the expert count is reduced so the suite fits a bounded device
# memory window.
HIDDEN_SIZE = 2560
INTERMEDIATE_SIZE = 640
TOP_K = 10
NUM_EXPERTS = 16

_SF_BLOCK = 32
# Corpus-wide MoE agreement bar (see test_cute_migration_moe_standard_corpus).
_MIN_COS = 0.999
_MAX_NORMALIZED_RMSE = 0.03


def _finite_e4m3_bytes(shape, generator) -> torch.Tensor:
    """Uniform bytes over every finite E4M3 encoding.

    0x7F and 0xFF are the E4M3 NaN encodings: a byte-preservation fixture may
    not use them, because their arithmetic is undefined and the NaN payload
    encodings are the very thing a numerical fixture must exclude.
    """
    finite = torch.tensor(
        [b for b in range(256) if b not in (0x7F, 0xFF)], dtype=torch.uint8
    )
    index = torch.randint(
        0, finite.numel(), tuple(shape), generator=generator, dtype=torch.int64
    )
    return finite[index]


def _finite_ue8m0_grid(rows: int, cols: int) -> torch.Tensor:
    """Deterministic grid covering every finite UE8M0 byte 0x00..0xFE."""
    return (
        (torch.arange(rows * cols, dtype=torch.int32) % 255)
        .to(torch.uint8)
        .reshape(rows, cols)
    )


def _expected_swizzled(grid: torch.Tensor) -> torch.Tensor:
    """Independently derived canonical MMA block-scale layout.

    ``swizzle_block_scale`` pads rows to 128 and scale columns to 4, then
    regroups ``[m_tile, g4, r32, k_tile, c4]`` into ``[m_tile, k_tile, r32,
    g4, c4]`` storage order.  Recomputing the destination offset here (instead
    of calling the production helper) makes this a real check of where every
    scale byte lands.
    """
    rows, cols = int(grid.shape[0]), int(grid.shape[1])
    rows_padded = ((rows + 127) // 128) * 128
    cols_padded = ((cols + 3) // 4) * 4
    k_tiles = cols_padded // 4
    out = torch.zeros(rows_padded * cols_padded, dtype=torch.uint8)
    for m_tile in range(rows_padded // 128):
        for k_tile in range(k_tiles):
            for r32 in range(32):
                for g4 in range(4):
                    for c4 in range(4):
                        row = m_tile * 128 + g4 * 32 + r32
                        col = k_tile * 4 + c4
                        if row >= rows or col >= cols:
                            continue
                        out[
                            ((((m_tile * k_tiles + k_tile) * 32 + r32) * 4 + g4) * 4)
                            + c4
                        ] = grid[row, col]
    return out.reshape(rows_padded, cols_padded)


def _checkpoint(experts: int, k: int, i: int, device):
    """Checkpoint-native MXFP8 tensors produced by real weight quantization.

    The payload bytes and UE8M0 scale bytes come from the in-tree MXFP8
    quantizer applied to BF16 weights, so the fixture carries the tensor
    statistics a numerical comparison needs (finite E4M3 payloads, block
    scales near the operand amax) instead of uniform random bytes, whose
    extreme UE8M0 exponents overflow the GEMM to NaN for reasons that have
    nothing to do with the kernel.
    """
    from b12x.gemm._shared.wo_mxfp8 import quantize_mxfp8_rows_torch

    generator = torch.Generator(device="cpu").manual_seed(7)

    def _quantized(rows: int, cols: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize a random BF16 matrix to MXFP8 codes and UE8M0 scales."""
        source = (
            (torch.randn(rows, cols, generator=generator, dtype=torch.float32) * 0.2)
            .to(torch.bfloat16)
            .to(device)
        )
        # The row quantizer takes 128-multiple K; UE8M0 blocks are 32 wide
        # and independent, so quantizing a zero-extended row and keeping the
        # first `cols` codes and `cols // 32` scales is exact.
        padded_cols = -(-cols // 128) * 128
        source = torch.nn.functional.pad(source, (0, padded_cols - cols))
        quantized = quantize_mxfp8_rows_torch(source)
        return (
            quantized.values.view(torch.uint8)[:, :cols].contiguous(),
            quantized.scale_rows.reshape(rows, padded_cols // _SF_BLOCK)
            .view(torch.uint8)[:, : cols // _SF_BLOCK]
            .contiguous(),
        )

    w13_values, w13_scales, w2_values, w2_scales = [], [], [], []
    for _ in range(experts):
        values, scales = _quantized(2 * i, k)
        w13_values.append(values)
        w13_scales.append(scales)
        values, scales = _quantized(k, i)
        w2_values.append(values)
        w2_scales.append(scales)
    return (
        torch.stack(w13_values),
        torch.stack(w13_scales),
        torch.stack(w2_values),
        torch.stack(w2_scales),
    )


def _prepare_experts(fused_moe, experts: int, k: int, i: int, device):
    """Plan and prepare synthetic MXFP8 experts through the public API."""
    w13, w13_scale, w2, w2_scale = _checkpoint(experts, k, i, device)
    plan = fused_moe.plan_weights(
        source=fused_moe.PackedSource(
            format=fused_moe.PackedSourceFormat.MXFP8_E8M0_K32,
            w13_layout=fused_moe.W13Layout.W13,
        ),
        activation=fused_moe.ActivationSpec(
            mode=fused_moe.ActivationMode.A8,
            nonlinearity="silu",
            io_dtype=torch.bfloat16,
        ),
        geometry=fused_moe.MoEGeometry(
            num_experts=experts, hidden_size=k, intermediate_size=i
        ),
    )
    # Distinct, non-unit per-expert scales on both sides: the runtime alphas
    # are weight_scale_2 / a_gscale, so a fixture that used unit activation
    # scales would pass even if the division were dropped.
    weight_scale_2 = torch.linspace(
        0.25, 0.75, experts, dtype=torch.float32, device=device
    )
    down_scale_2 = torch.linspace(1.5, 3.0, experts, dtype=torch.float32, device=device)
    a1_gscale = torch.linspace(0.5, 2.0, experts, dtype=torch.float32, device=device)
    a2_gscale = torch.linspace(4.0, 6.0, experts, dtype=torch.float32, device=device)
    experts_obj = fused_moe.prepare_weights(
        plan=plan,
        weights=fused_moe.PackedWeights(
            w13=w13,
            w2=w2,
            w13_block_scales=w13_scale,
            w2_block_scales=w2_scale,
            w13_global_scales=weight_scale_2,
            w2_global_scales=down_scale_2,
            input_scale=a1_gscale,
            intermediate_scale=a2_gscale,
            immutable_input_scales=True,
        ),
    )
    return experts_obj, {
        "w13": w13,
        "w13_scale": w13_scale,
        "w2": w2,
        "w2_scale": w2_scale,
        "w1_alpha": experts_obj._impl.w1_alphas,
        "w2_alpha": experts_obj._impl.w2_alphas,
        "a1_gscale": a1_gscale,
        "a2_gscale": a2_gscale,
    }


def _routes(experts: int, tokens: int, top_k: int, device, *, offset: int = 0):
    """Build deterministic top-k routes and weights."""
    ids = torch.stack(
        tuple(
            (torch.arange(tokens, device=device) + offset + step) % experts
            for step in range(top_k)
        ),
        dim=1,
    ).to(torch.int32)
    weights = torch.full(
        (tokens, top_k), 1.0 / top_k, dtype=torch.float32, device=device
    )
    return ids, weights


# ---------------------------------------------------------------------------
# CPU: canonical vocabulary and lossless preparation
# ---------------------------------------------------------------------------


def test_plan_weights_mxfp8_a8_selects_the_w8a8_contract() -> None:
    """MXFP8 with A8 activations plans the w8a8_mx recipe."""
    from b12x.moe import fused_moe

    plan = fused_moe.plan_weights(
        source=fused_moe.PackedSource(
            format=fused_moe.PackedSourceFormat.MXFP8_E8M0_K32,
            w13_layout=fused_moe.W13Layout.W13,
        ),
        activation=fused_moe.ActivationSpec(
            mode=fused_moe.ActivationMode.A8,
            nonlinearity="silu",
            io_dtype=torch.bfloat16,
        ),
        geometry=fused_moe.MoEGeometry(
            num_experts=NUM_EXPERTS,
            hidden_size=HIDDEN_SIZE,
            intermediate_size=INTERMEDIATE_SIZE,
        ),
    )
    assert plan._impl.quant_modes == frozenset({"w8a8_mx"})
    assert plan.prepared_format.weights is fused_moe.WeightEncoding.FP8_E4M3
    assert plan.prepared_format.scales is fused_moe.ScaleEncoding.E8M0_K32
    assert plan.prepared_format.packing is fused_moe.WeightPacking.SOURCE_NATIVE
    spec = plan._impl.specs[0]
    assert spec.weight_encoding.value == "mxfp8_e4m3"
    assert spec.activation_encoding.value == "mxfp8_e4m3"
    assert spec.weight_scale.value == "e8m0_k32"


@pytest.mark.parametrize("mode", ["A16", "A4"])
def test_plan_weights_mxfp8_rejects_narrower_activations(mode: str) -> None:
    """MXFP8 weights reject activation modes narrower than A8."""
    from b12x.moe import fused_moe

    with pytest.raises(ValueError):
        fused_moe.plan_weights(
            source=fused_moe.PackedSource(
                format=fused_moe.PackedSourceFormat.MXFP8_E8M0_K32
            ),
            activation=fused_moe.ActivationSpec(
                mode=getattr(fused_moe.ActivationMode, mode),
                nonlinearity="silu",
                io_dtype=torch.bfloat16,
            ),
            geometry=fused_moe.MoEGeometry(
                num_experts=4, hidden_size=256, intermediate_size=128
            ),
        )


def test_prepare_w8a8_keeps_weight_bytes_and_swizzles_scales_exactly() -> None:
    """Preparation keeps E4M3 bytes and swizzles scales like the canonical helper."""
    from b12x.moe._shared.kernels.w8a8 import prepare_w8a8_mxfp8_weights

    experts, k, i = 2, 256, 128
    generator = torch.Generator(device="cpu").manual_seed(11)
    w13 = _finite_e4m3_bytes((experts, 2 * i, k), generator)
    w2 = _finite_e4m3_bytes((experts, k, i), generator)
    w13_scale = torch.stack(
        [_finite_ue8m0_grid(2 * i, k // _SF_BLOCK) for _ in range(experts)]
    )
    w2_scale = torch.stack(
        [_finite_ue8m0_grid(k, i // _SF_BLOCK) for _ in range(experts)]
    )

    prepared = prepare_w8a8_mxfp8_weights(
        w13_values=w13,
        w13_scale=w13_scale,
        w13_scale_2=torch.tensor([2.0, 4.0]),
        w2_values=w2,
        w2_scale=w2_scale,
        w2_scale_2=torch.tensor([8.0, 16.0]),
        a1_gscale=torch.tensor([0.5, 1.0]),
        a2_gscale=torch.tensor([2.0, 4.0]),
        num_experts=experts,
        hidden_size=k,
        intermediate_size=i,
    )

    assert torch.equal(prepared.w13_values, w13)
    assert torch.equal(prepared.w2_values, w2)
    assert torch.equal(
        prepared.w13_sf_swizzled,
        torch.stack([_expected_swizzled(w13_scale[e]) for e in range(experts)]),
    )
    assert torch.equal(
        prepared.w2_sf_swizzled,
        torch.stack([_expected_swizzled(w2_scale[e]) for e in range(experts)]),
    )
    torch.testing.assert_close(
        prepared.w13_alpha, torch.tensor([4.0, 4.0]), rtol=0, atol=0
    )
    torch.testing.assert_close(
        prepared.w2_alpha, torch.tensor([4.0, 4.0]), rtol=0, atol=0
    )


def test_prepare_w8a8_swizzles_non_128_fc1_halves_independently() -> None:
    """A mid-atom gate half gets its own atom-aligned scale grid.

    With I=96 the gate half starts at row 96 of the first 128-row atom, so a
    single swizzle of the stacked grid would put gate rows inside the up
    atom.  Each half must land where an independently swizzled [I, K/32] grid
    places it.
    """
    from b12x.moe._shared.kernels.w8a8 import prepare_w8a8_mxfp8_weights

    experts, k, i = 2, 256, 96
    w13_scale = torch.stack(
        [_finite_ue8m0_grid(2 * i, k // _SF_BLOCK).roll(e) for e in range(experts)]
    )
    prepared = prepare_w8a8_mxfp8_weights(
        w13_values=torch.zeros(experts, 2 * i, k, dtype=torch.uint8),
        w13_scale=w13_scale,
        w13_scale_2=torch.ones(experts),
        w2_values=torch.zeros(experts, k, i, dtype=torch.uint8),
        w2_scale=torch.full((experts, k, i // _SF_BLOCK), 127, dtype=torch.uint8),
        w2_scale_2=torch.ones(experts),
        num_experts=experts,
        hidden_size=k,
        intermediate_size=i,
    )
    assert torch.equal(
        prepared.w13_sf_swizzled,
        torch.stack(
            [
                torch.stack([_expected_swizzled(w13_scale[e, :i]) for e in range(experts)]),
                torch.stack([_expected_swizzled(w13_scale[e, i:]) for e in range(experts)]),
            ]
        ),
    )


def test_prepare_w8a8_accepts_float8_payload_and_scale_storage() -> None:
    """Float8 payload and scale storage dtypes are accepted as byte views."""
    from b12x.moe._shared.kernels.w8a8 import prepare_w8a8_mxfp8_weights

    experts, k, i = 1, 128, 128
    codes = _finite_e4m3_bytes(
        (experts, 2 * i, k), torch.Generator(device="cpu").manual_seed(3)
    )
    w13 = codes.view(torch.float8_e4m3fn)
    prepared = prepare_w8a8_mxfp8_weights(
        w13_values=w13,
        w13_scale=torch.zeros(
            experts, 2 * i, k // _SF_BLOCK, dtype=torch.float8_e8m0fnu
        ),
        w13_scale_2=torch.ones(experts),
        w2_values=torch.zeros(experts, k, i, dtype=torch.float8_e4m3fn),
        w2_scale=torch.zeros(experts, k, i // _SF_BLOCK, dtype=torch.uint8),
        w2_scale_2=torch.ones(experts),
        num_experts=experts,
        hidden_size=k,
        intermediate_size=i,
    )
    assert prepared.w13_values.dtype is torch.uint8
    assert torch.equal(prepared.w13_values, codes.reshape(experts, 2 * i, k))


def test_prepare_w8a8_rejects_ue8m0_nan_and_preserves_top_exponent() -> None:
    """UE8M0 0xFF is rejected while 0xFE survives unchanged."""
    from b12x.moe._shared.kernels.w8a8 import prepare_w8a8_mxfp8_weights

    experts, k, i = 1, 128, 128
    args = {
        "w13_values": torch.zeros(experts, 2 * i, k, dtype=torch.uint8),
        "w13_scale_2": torch.ones(experts),
        "w2_values": torch.zeros(experts, k, i, dtype=torch.uint8),
        "w2_scale_2": torch.ones(experts),
        "num_experts": experts,
        "hidden_size": k,
        "intermediate_size": i,
    }
    bad = torch.full((experts, 2 * i, k // _SF_BLOCK), 127, dtype=torch.uint8)
    bad[0, 3, 1] = 0xFF
    with pytest.raises(ValueError, match="NaN"):
        prepare_w8a8_mxfp8_weights(
            **args,
            w13_scale=bad,
            w2_scale=torch.full((experts, k, i // _SF_BLOCK), 127, dtype=torch.uint8),
        )
    cleared = bad.clone()
    cleared[0, 3, 1] = 127
    prepared = prepare_w8a8_mxfp8_weights(
        **args,
        w13_scale=cleared,
        w2_scale=torch.full((experts, k, i // _SF_BLOCK), 254, dtype=torch.uint8),
    )
    assert int(prepared.w2_sf_swizzled.max().item()) == 254


def test_prepare_w8a8_rejects_non_finite_runtime_alpha() -> None:
    """Non-finite runtime alphas are rejected."""
    from b12x.moe._shared.kernels.w8a8 import prepare_w8a8_mxfp8_weights

    experts, k, i = 1, 128, 128
    with pytest.raises(ValueError, match="non-finite"):
        prepare_w8a8_mxfp8_weights(
            w13_values=torch.zeros(experts, 2 * i, k, dtype=torch.uint8),
            w13_scale=torch.full(
                (experts, 2 * i, k // _SF_BLOCK), 127, dtype=torch.uint8
            ),
            w13_scale_2=torch.ones(experts),
            w2_values=torch.zeros(experts, k, i, dtype=torch.uint8),
            w2_scale=torch.full((experts, k, i // _SF_BLOCK), 127, dtype=torch.uint8),
            w2_scale_2=torch.ones(experts),
            a1_gscale=torch.zeros(experts),
            a2_gscale=torch.ones(experts),
            num_experts=experts,
            hidden_size=k,
            intermediate_size=i,
        )


@pytest.mark.parametrize(
    "field, value, error",
    [
        ("w13_values", torch.zeros(1, 256, 128, dtype=torch.float32), TypeError),
        ("w13_values", torch.zeros(1, 255, 128, dtype=torch.uint8), ValueError),
        ("w13_scale", torch.zeros(1, 256, 3, dtype=torch.uint8), ValueError),
        ("w2_scale", torch.zeros(1, 128, 4, dtype=torch.int32), TypeError),
    ],
)
def test_prepare_w8a8_rejects_malformed_inputs(field, value, error) -> None:
    """Malformed payload, scale, or geometry inputs are rejected."""
    from b12x.moe._shared.kernels.w8a8 import prepare_w8a8_mxfp8_weights

    experts, k, i = 1, 128, 128
    kwargs = {
        "w13_values": torch.zeros(experts, 2 * i, k, dtype=torch.uint8),
        "w13_scale": torch.zeros(experts, 2 * i, k // _SF_BLOCK, dtype=torch.uint8),
        "w13_scale_2": torch.ones(experts),
        "w2_values": torch.zeros(experts, k, i, dtype=torch.uint8),
        "w2_scale": torch.zeros(experts, k, i // _SF_BLOCK, dtype=torch.uint8),
        "w2_scale_2": torch.ones(experts),
        "num_experts": experts,
        "hidden_size": k,
        "intermediate_size": i,
    }
    kwargs[field] = value
    with pytest.raises(error):
        prepare_w8a8_mxfp8_weights(**kwargs)


def test_w8a8_weight_plan_lowers_to_the_mxfp8_engine() -> None:
    """The w8a8_mx weight plan lowers to the MXFP8 GEMM engine."""
    from b12x.moe._shared.execution import (
        GemmEngine,
        MoERegime,
        PreparedWeightLayout,
        WeightPreparationTransform,
        WeightStoragePolicy,
        lower_moe_execution,
        make_moe_spec,
        plan_moe_weight_preparation,
        validate_moe_source_quant,
    )

    validate_moe_source_quant(source_format="mxfp8_e8m0_k32", quant_mode="w8a8_mx")
    with pytest.raises(ValueError):
        validate_moe_source_quant(source_format="mxfp8_e8m0_k32", quant_mode="w6a8_mx")
    with pytest.raises(ValueError):
        validate_moe_source_quant(source_format="mxfp6_e2m3", quant_mode="w8a8_mx")

    spec = make_moe_spec(
        quant_mode="w8a8_mx",
        source_format="mxfp8_e8m0_k32",
        activation="silu",
        io_dtype="bfloat16",
        w13_layout="w13",
    )
    plan = plan_moe_weight_preparation(
        spec,
        num_experts=NUM_EXPERTS,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
    )
    assert plan.transforms == frozenset({WeightPreparationTransform.W8A8_MXFP8})
    assert plan.storage_policy is WeightStoragePolicy.TRANSFER_SOURCE
    assert plan.required_weight_layout("w8a8_mx") is PreparedWeightLayout.SOURCE_NATIVE
    execution = lower_moe_execution(
        spec,
        regime=MoERegime.MATERIALIZED_FUSED,
        tile_m=128,
        tile_n=128,
        required_weight_layout=plan.required_weight_layout("w8a8_mx"),
    )
    assert execution.gemm_engine is GemmEngine.MXFP8_QMMA
    assert plan.supports(quant_mode="w8a8_mx", execution=execution)


def test_dynamic_kernel_w8a8_shares_mxf8_geometry_without_fp6_packing() -> None:
    """w8a8_mx shares the MXF8 geometry without FP6 packing."""
    from b12x.moe._shared.kernels.dynamic import MoEDynamicKernelBackend

    backend = MoEDynamicKernelBackend(
        32, (128, 128), quant_recipe="w8a8_mx", activation="silu"
    )
    assert not backend.is_w6a8  # 3:4-packed B staging stays FP6-only
    assert backend.is_w4a8 is False
    assert backend.tile_shape_mnk == (128, 128, 128)
    assert backend.mxfp6_fmt_a is None and backend.mxfp6_fmt_b is None
    assert backend._act_container_fmt == "e4m3"

    with pytest.raises(ValueError, match="sf_vec_size"):
        MoEDynamicKernelBackend(16, (128, 128), quant_recipe="w8a8_mx")
    with pytest.raises(ValueError, match="mma_tiler_mn"):
        MoEDynamicKernelBackend(32, (64, 128), quant_recipe="w8a8_mx")
    with pytest.raises(ValueError, match="swap_ab"):
        MoEDynamicKernelBackend(32, (128, 128), quant_recipe="w8a8_mx", swap_ab=True)
    with pytest.raises(ValueError, match="direct_routing"):
        MoEDynamicKernelBackend(
            32, (128, 128), quant_recipe="w8a8_mx", direct_routing=True
        )
    with pytest.raises(ValueError, match="only valid for w6a8_mx"):
        MoEDynamicKernelBackend(
            32, (128, 128), quant_recipe="w8a8_mx", mxfp6_fmt_b="e2m3"
        )


def test_w8a8_tuning_requires_dynamic_backend_and_m128_tile() -> None:
    """Tuning validation admits only the dynamic backend and M128 tile."""
    from b12x.moe.fused_moe._tuning import (
        MoeDecodeConfig,
        MoeDecodeQuery,
        validate_moe_decode_config,
    )

    def query(**overrides) -> MoeDecodeQuery:
        """Build a w8a8_mx decode query with overrides."""
        values = {
            "quant_mode": "w8a8_mx",
            "quant_modes": ("w8a8_mx",),
            "source_format": "mxfp8_e8m0_k32",
            "activation": "silu",
            "io_dtype": "bfloat16",
            "num_experts": NUM_EXPERTS,
            "hidden_size": HIDDEN_SIZE,
            "intermediate_size": INTERMEDIATE_SIZE,
            "top_k": TOP_K,
            "num_tokens": 8,
            "routed_rows": 8 * TOP_K,
            "route_num_experts": NUM_EXPERTS,
            "route_logits_dtype": None,
            "apply_router_weight_on_input": False,
            "collect_activation_amax": False,
            "deterministic_output": None,
            "swiglu_limit": None,
            "swiglu_alpha": None,
            "swiglu_beta": None,
            "w13_layout": "w13",
            "weight_layouts": ("source_native",),
            "w4a16_weight_layout": None,
            "w4a16_scale_format": None,
            "w4a16_block_size_m": None,
            "fast_math": True,
            "numerical_recipe": None,
            "controls": FrozenMapping(),
        }
        values.update(overrides)
        return MoeDecodeQuery(**values)

    validate_moe_decode_config(
        query(),
        MoeDecodeConfig(
            backend="dynamic",
            route_planner="internal",
            max_active_clusters=None,
            dynamic_tile_m=128,
            dynamic_route_mode="grouped",
        ),
        None,
    )
    with pytest.raises(ValueError, match="dynamic backend"):
        validate_moe_decode_config(
            query(),
            MoeDecodeConfig(
                backend="micro", route_planner="internal", max_active_clusters=None
            ),
            None,
        )
    with pytest.raises(ValueError, match="M128"):
        validate_moe_decode_config(
            query(),
            MoeDecodeConfig(
                backend="dynamic",
                route_planner="internal",
                max_active_clusters=None,
                dynamic_tile_m=16,
                dynamic_route_mode="grouped",
            ),
            None,
        )


def _ue8m0_byte_from_asm(block_max: torch.Tensor, gs: torch.Tensor) -> torch.Tensor:
    """Transcription of ``fp6_block_ue8m0_exact``'s inline PTX.

    ``mul.f32 t = max_abs*gs; div.rn.f32 ratio = t/448;`` then
    ``byte = exponent_field + (mantissa != 0)``, clamped to ``[0, 255]``, and
    0 when ``ratio <= 0``.
    """
    ratio = (block_max.to(torch.float32) * gs) / 448.0
    bits = ratio.contiguous().view(torch.int32)
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & 0x007FFFFF
    byte = (exponent + (mantissa != 0).to(torch.int32)).clamp(0, 255)
    return torch.where(ratio > 0, byte, torch.zeros_like(byte)).to(torch.uint8)


def test_mxfp8_scaled_roundtrip_matches_the_device_ue8m0_rule() -> None:
    """The MoE oracle quantizer must agree with the kernel's PTX byte rule.

    Sweeps every float8-E4M3 payload code against block amaxes landing
    exactly on, one ulp below, and one ulp above a power-of-two ceiling — the
    region where ``ceil(log2(...))`` and exact IEEE bit extraction disagree,
    which would silently shift a whole block by 2x.
    """
    from b12x._lib.intrinsics import quant_dequant_mxfp8_scaled_torch

    codes = (
        torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).to(torch.float32)
    )
    codes = torch.nan_to_num(codes, nan=0.0, posinf=0.0, neginf=0.0)
    powers = torch.tensor([2.0**e for e in (-9, -4, 0, 5, 9)], dtype=torch.float32)
    deltas = torch.tensor([1.0, 1.0 + 2**-23, 1.0 - 2**-23], dtype=torch.float32)
    scales = torch.tensor([1.0, 0.5, 8.0, 448.0], dtype=torch.float32)

    for gs in scales:
        blocks = (
            (
                powers.view(-1, 1, 1)
                * deltas.view(1, -1, 1)
                * codes.view(1, 1, -1)[..., :32]
            )
            .reshape(-1, 32)
            .contiguous()
        )
        block_max = blocks.abs().amax(dim=-1, keepdim=True)
        byte = _ue8m0_byte_from_asm(block_max, gs)
        scale = torch.where(
            byte == 0,
            torch.zeros_like(byte, dtype=torch.float32),
            torch.exp2(byte.to(torch.float32) - 127.0),
        )
        # A byte-0 (all-zero block) scale means the operand is zero, not that
        # the payload is an unrepresentable inf/NaN.
        inv = torch.where(scale > 0, gs / scale, torch.zeros_like(scale))
        payload = (
            (blocks * inv)
            .clamp(-448.0, 448.0)
            .to(torch.float8_e4m3fn)
            .to(torch.float32)
        )
        torch.testing.assert_close(
            quant_dequant_mxfp8_scaled_torch(blocks, gs),
            payload * scale,
            rtol=0,
            atol=0,
        )


# ---------------------------------------------------------------------------
# GPU: numerical execution, degenerate routes, tails, graph replay
# ---------------------------------------------------------------------------


# 640 is the TP=1 MTP shard (128-aligned, shared FC1 descriptor); 320 the TP=2
# shard and 96 a sub-128 shard, both on the split up/gate FC1 descriptors
# with a TMA zero-filled tail tile.
@pytest.mark.parametrize("i", [INTERMEDIATE_SIZE, 320, 96])
@pytest.mark.parametrize("tokens", [1, 5, 16])
def test_w8a8_matches_oracle_on_model_geometry(tokens: int, i: int) -> None:
    """Native MXFP8 experts reproduce the pure-torch oracle.

    tokens=1 is the sparsest grouped tile, 5 a partial 128-row tile, 16 the
    first run that fills two tiles.
    """
    require_b12x()
    from b12x.moe import fused_moe
    from b12x.moe._shared.kernels.reference import (
        compare_to_reference,
        moe_reference_w8a8_mx,
    )

    from b12x.testing.reference.helpers import make_tp_moe_fp4_binding

    device = torch.device("cuda")
    experts, k = NUM_EXPERTS, HIDDEN_SIZE
    prepared, native = _prepare_experts(fused_moe, experts, k, i, device)
    a = (torch.randn(tokens, k, device=device) * 0.25).to(torch.bfloat16)
    topk_ids, topk_weights = _routes(experts, tokens, TOP_K, device)
    output = torch.empty(tokens, k, dtype=torch.bfloat16, device=device)

    with make_tp_moe_fp4_binding(
        a=a,
        experts=prepared,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        output=output,
        quant_mode="w8a8_mx",
    ) as binding:
        actual = fused_moe.run(binding=binding)

    reference = moe_reference_w8a8_mx(
        a.float(),
        native["w13"],
        native["w13_scale"],
        native["w1_alpha"],
        native["w2"],
        native["w2_scale"],
        native["w2_alpha"],
        topk_ids,
        topk_weights,
        experts,
        k,
        i,
        a1_gscale=native["a1_gscale"],
        a2_gscale=native["a2_gscale"],
    )
    assert bool(torch.isfinite(actual.float()).all().item())
    metrics = compare_to_reference(actual.float(), reference)
    assert metrics.cos >= _MIN_COS, metrics
    rms = float(reference.square().mean().sqrt().item())
    assert rms > 0.0, "the fixture must produce signal"
    assert metrics.rmse / rms <= _MAX_NORMALIZED_RMSE, metrics


def test_w8a8_ignores_experts_without_routes() -> None:
    """Empty experts must contribute exactly nothing.

    Every token routes to the first TOP_K experts, so the remaining experts
    own no rows; their FC1 payload is poisoned with the E4M3 NaN bit pattern,
    which would leak into the output if a task were issued for an empty expert.
    """
    require_b12x()
    from b12x.moe import fused_moe

    from b12x.testing.reference.helpers import make_tp_moe_fp4_binding

    device = torch.device("cuda")
    experts, k, i = NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE
    prepared, _ = _prepare_experts(fused_moe, experts, k, i, device)
    payload = prepared._impl.w1_fp4
    payload.view(torch.uint8)[experts - 1] = 0xFF

    tokens = 4
    a = (torch.randn(tokens, k, device=device) * 0.25).to(torch.bfloat16)
    topk_ids, topk_weights = _routes(experts, tokens, TOP_K, device)
    output = torch.empty(tokens, k, dtype=torch.bfloat16, device=device)

    with make_tp_moe_fp4_binding(
        a=a,
        experts=prepared,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        output=output,
        quant_mode="w8a8_mx",
    ) as binding:
        actual = fused_moe.run(binding=binding).float()

    assert bool(torch.isfinite(actual).all().item())
    assert float(actual.abs().sum().item()) > 0.0


def _capture_and_replay(fused_moe, binding, *, label: str):
    """Warm, capture under frozen resolution, replay, and report the misses."""
    from b12x._lib.compiler import compile_cache_info
    from b12x._lib.runtime_control import kernel_resolution_guard

    fused_moe.run(binding=binding)
    torch.cuda.synchronize()
    misses_before = compile_cache_info()["compile_misses"]
    with kernel_resolution_guard(label):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = fused_moe.run(binding=binding)
        allocated_before = torch.cuda.memory_stats()["allocation.all.allocated"]
        for _ in range(3):
            captured.fill_(float("nan"))
            graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated_before
    misses_after = compile_cache_info()["compile_misses"]
    result = captured.clone()
    del graph
    return result, misses_before, misses_after


@pytest.mark.parametrize("i", [INTERMEDIATE_SIZE, 320])
def test_w8a8_replays_under_cuda_graph_with_frozen_resolution(i: int) -> None:
    """A captured w8a8 launch must replay without compiling and stay correct.

    The default reduction is ``OutputReduction.ATOMIC_SCATTER``: expert
    partials are added into the caller's BF16 output with
    ``scatter_add_bf16x2`` in whatever order the materialized queue hands out
    tasks, so eager and replay are two different floating-point summation
    orders and may not agree bit-for-bit.  What capture must guarantee is
    (a) zero kernel resolution inside the capture and (b) that the replayed
    result is still numerically correct, both asserted here.  Bit-exact replay
    is asserted in the deterministic-reduction test below.
    """
    require_b12x()
    from b12x.moe import fused_moe
    from b12x.moe._shared.kernels.reference import (
        compare_to_reference,
        moe_reference_w8a8_mx,
    )

    from b12x.testing.reference.helpers import make_tp_moe_fp4_binding

    device = torch.device("cuda")
    experts, k = NUM_EXPERTS, HIDDEN_SIZE
    prepared, native = _prepare_experts(fused_moe, experts, k, i, device)
    tokens = 8
    a = (torch.randn(tokens, k, device=device) * 0.25).to(torch.bfloat16)
    topk_ids, topk_weights = _routes(experts, tokens, TOP_K, device)
    output = torch.empty(tokens, k, dtype=torch.bfloat16, device=device)

    with make_tp_moe_fp4_binding(
        a=a,
        experts=prepared,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        output=output,
        quant_mode="w8a8_mx",
    ) as binding:
        captured, misses_before, misses_after = _capture_and_replay(
            fused_moe, binding, label="w8a8 graph replay test"
        )

    assert misses_after == misses_before, (
        "no kernel may compile during or after warm capture: "
        f"{misses_before} -> {misses_after}"
    )
    reference = moe_reference_w8a8_mx(
        a.float(),
        native["w13"],
        native["w13_scale"],
        native["w1_alpha"],
        native["w2"],
        native["w2_scale"],
        native["w2_alpha"],
        topk_ids,
        topk_weights,
        experts,
        k,
        i,
        a1_gscale=native["a1_gscale"],
        a2_gscale=native["a2_gscale"],
    )
    assert bool(torch.isfinite(captured.float()).all().item())
    metrics = compare_to_reference(captured.float(), reference)
    assert metrics.cos >= _MIN_COS, metrics
    rms = float(reference.square().mean().sqrt().item())
    assert rms > 0.0, "the fixture must produce signal"
    assert metrics.rmse / rms <= _MAX_NORMALIZED_RMSE, metrics


def test_w8a8_deterministic_reduction_replays_bit_exact() -> None:
    """With the fixed-order reduction, capture and replay must be exact.

    ``deterministic_output=True`` lowers to
    ``OutputReduction.ROUTE_BUFFER_TOPK_SUM``: each routed pair writes its own
    row and a fixed-order top-k sum produces the output, removing the atomic
    ordering that makes the default reduction merely accurate, not repeatable.
    """
    require_b12x()
    from b12x.moe import fused_moe

    from b12x.testing.reference.helpers import make_tp_moe_fp4_binding

    device = torch.device("cuda")
    experts, k, i = NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE
    prepared, _ = _prepare_experts(fused_moe, experts, k, i, device)
    tokens = 8
    a = (torch.randn(tokens, k, device=device) * 0.25).to(torch.bfloat16)
    topk_ids, topk_weights = _routes(experts, tokens, TOP_K, device)
    output = torch.empty(tokens, k, dtype=torch.bfloat16, device=device)

    with make_tp_moe_fp4_binding(
        a=a,
        experts=prepared,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        output=output,
        quant_mode="w8a8_mx",
        deterministic_output=True,
    ) as binding:
        eager = fused_moe.run(binding=binding).clone()
        torch.cuda.synchronize()
        captured, misses_before, misses_after = _capture_and_replay(
            fused_moe, binding, label="w8a8 deterministic replay test"
        )

    assert misses_after == misses_before, (
        f"kernel resolution inside capture: {misses_before} -> {misses_after}"
    )
    assert bool(torch.isfinite(captured.float()).all().item())
    torch.testing.assert_close(captured, eager, rtol=0, atol=0)


@pytest.mark.parametrize("cols", [64, 96])
def test_mxfp8_scaled_roundtrip_supports_per_row_globals(cols):
    from b12x._lib.intrinsics import quant_dequant_mxfp8_scaled_torch

    x = torch.linspace(-4.0, 7.0, 3 * cols).reshape(3, cols)
    scales = torch.tensor([0.5, 2.0, 3.0])
    expected = torch.cat([
        quant_dequant_mxfp8_scaled_torch(row.unsqueeze(0), scale)
        for row, scale in zip(x, scales)
    ])
    actual = quant_dequant_mxfp8_scaled_torch(x, scales)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_w6a8_scale_validation_preserves_strided_input_support():
    from b12x.moe._shared.kernels.w6a8.weights import _validate_e8m0_scale_grid

    backing = torch.arange(64, dtype=torch.uint8).reshape(2, 4, 8)
    scales = backing[:, :, ::2]
    assert not scales.is_contiguous()
    actual = _validate_e8m0_scale_grid(
        scales, name="scales", num_experts=2, rows=4, k=128,
    )
    assert actual.is_contiguous()
    torch.testing.assert_close(actual, scales, rtol=0, atol=0)


def test_w8a8_prepared_capacity_reuses_launches_for_live_counts():
    require_b12x()
    from b12x.moe import fused_moe as moe
    from b12x.moe._shared.kernels.reference import compare_to_reference, moe_reference_w8a8_mx
    from b12x.preparation import PreparationSession, PreparedCall
    from b12x._lib.runtime_control import kernel_resolution_guard

    device = torch.device("cuda")
    experts, hidden, intermediate, capacity, topk = 16, 2560, 320, 32, 10
    prepared, native = _prepare_experts(moe, experts, hidden, intermediate, device)
    plan = moe.plan_execution(
        experts=prepared,
        capacity=moe.ExecutionCapacity(max_tokens=capacity, top_k=topk),
        routing=moe.RoutingSpec(deterministic_output=True),
    )
    x = torch.randn(capacity, hidden, device=device).to(torch.bfloat16) * 0.25
    ids, weights = _routes(experts, capacity, topk, device)
    output = torch.empty_like(x)

    def primer(state):
        scratch = tuple(torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
                        for spec in state.scratch.scratch_specs())
        binding = state.bind(a=x, experts=prepared, topk_ids=ids,
                             topk_weights=weights, output=output, scratch=scratch)
        return PreparedCall(run=lambda: state.run(binding), owners=(scratch, binding))

    with PreparationSession(device=device, autotune=False, compile_workers=0) as session:
        session.prepare((plan.request(name="w8a8-live-counts", prepare_call=primer),))
        session.freeze()
        scratch = tuple(torch.empty(spec.shape, dtype=spec.dtype, device=device)
                        for spec in plan.scratch_specs())
        for rows in (1, 17, 32):
            reference = moe_reference_w8a8_mx(
                x[:rows].float(), native["w13"], native["w13_scale"], native["w1_alpha"],
                native["w2"], native["w2_scale"], native["w2_alpha"], ids[:rows], weights[:rows],
                experts, hidden, intermediate,
                a1_gscale=native["a1_gscale"], a2_gscale=native["a2_gscale"],
            )
            with kernel_resolution_guard():
                binding = moe.bind(plan, a=x[:rows], experts=prepared,
                                   topk_ids=ids[:rows], topk_weights=weights[:rows],
                                   output=output[:rows], scratch=scratch)
                actual = moe.run(binding=binding)
            metrics = compare_to_reference(actual.float(), reference)
            assert metrics.cos >= _MIN_COS, metrics
            assert torch.isfinite(actual).all() and torch.count_nonzero(actual)
