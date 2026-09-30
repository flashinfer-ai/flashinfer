"""NVFP4/MXFP8 prepared GEMM: numerical references, grouped scales, and replay."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from b12x._lib.runtime_control import kernel_resolution_guard
from b12x._lib.utils import convert_sf_from_mma_layout
from b12x._lib.intrinsics import (
    fp4_quantize_values_torch,
    quantize_grouped_nvfp4_torch,
    swizzle_block_scale,
)
from b12x.testing.reference.gemm import nvfp4_gemm_reference
from b12x.gemm import blockscaled
from b12x.preparation import DeviceIdentity
from b12x.gemm._shared.wo_mxfp8 import (
    dequantize_mxfp8_rows_torch,
    pack_fp8_block_scaled_weight_mxfp8,
    quantize_mxfp8_rows_torch,
)

from ._blockscaled import prepared
from ..conftest import require_b12x


def test_regime_plan_combines_static_shapes_with_dynamic_capacity() -> None:
    query = blockscaled.BlockscaledQuery(
        recipe="mxfp8",
        num_tokens=128,
        in_features=128,
        padded_in_features=128,
        out_features=64,
        expected_m=None,
    )

    plan = blockscaled.plan_regimes(query, exact_m=(1, 2, 4, 8))

    assert plan.token_counts == (1, 2, 4, 8, 128)
    assert dict(plan.capacity_metadata) == {
        "max_rows": 128,
        "exact_m": (1, 2, 4, 8),
    }


@pytest.mark.parametrize("workspace_form", ["provided", "owned"])
@pytest.mark.parametrize("n,k", [(2560, 2560), (2560, 6144), (6144, 2560)])
def test_mxfp8_capacity_lowering_keeps_prefill_tile_hint(workspace_form, n, k):
    """The prepared capacity, not a live row count, selects the native tile."""
    from b12x.gemm.blockscaled._preparation import _dense_lowering
    from b12x.gemm.blockscaled._tuning import BlockscaledConfig

    query = blockscaled.BlockscaledQuery(
        recipe="mxfp8", num_tokens=6019, in_features=k,
        padded_in_features=k, out_features=n, workspace_form=workspace_form,
    )
    device = SimpleNamespace(identity=DeviceIdentity(
        vendor="nvidia", compute_capability=(12, 0), sm_count=188,
        product_name="RTX PRO 6000 Blackwell Max-Q",
    ))
    lowering = _dense_lowering(query, BlockscaledConfig(mode="quantized"), device)

    assert query.expected_m is None  # The public plan still accepts shorter rows.
    assert lowering.m == 6019
    assert lowering.expected_m == 6019
    assert lowering.mma_tiler_mn == (128, 64)
    assert not lowering.policy.large_m_unroll


@pytest.mark.parametrize("recipe,hint", [("mxfp8", 4), ("nvfp4", None), ("nvfp4", 4)])
def test_capacity_lowering_preserves_explicit_hints_and_nvfp4(recipe, hint):
    from b12x.gemm.blockscaled._preparation import _dense_lowering
    from b12x.gemm.blockscaled._tuning import BlockscaledConfig

    query = blockscaled.BlockscaledQuery(
        recipe=recipe, num_tokens=6019, in_features=2560,
        padded_in_features=2560, out_features=2560, expected_m=hint,
        activation_scale_available=recipe == "nvfp4",
    )
    device = SimpleNamespace(identity=DeviceIdentity(
        vendor="nvidia", compute_capability=(12, 0), sm_count=188,
        product_name="RTX PRO 6000 Blackwell Max-Q",
    ))
    lowering = _dense_lowering(query, BlockscaledConfig(mode="quantized"), device)
    assert query.expected_m == hint
    assert lowering.expected_m == hint


@pytest.mark.parametrize("capacity", [128, 2047, 2048, 6019])
def test_mxfp8_capacity_lowering_preserves_short_prefill_program(capacity):
    """A large prefill specialization does not replace the dynamic short tile."""
    from b12x.gemm.blockscaled._preparation import (
        _dense_lowering, _lower_dense_with_hint, _short_dense_lowering,
    )
    from b12x.gemm.blockscaled._tuning import BlockscaledConfig

    query = blockscaled.BlockscaledQuery(
        recipe="mxfp8", num_tokens=capacity, in_features=2560,
        padded_in_features=2560, out_features=6144, workspace_form="provided",
    )
    device = SimpleNamespace(identity=DeviceIdentity(
        vendor="nvidia", compute_capability=(12, 0), sm_count=188,
        product_name="RTX PRO 6000 Blackwell Max-Q",
    ))
    config = BlockscaledConfig(mode="quantized")
    dynamic = _lower_dense_with_hint(query, config, device, None)
    short = _short_dense_lowering(query, config, device)
    if capacity < 2048:
        assert short is None
        assert _dense_lowering(query, config, device) == dynamic
    else:
        assert short == dynamic
        assert short.mma_tiler_mn == (64, 128)
        assert short.expected_m is None
        assert _dense_lowering(query, config, device).mma_tiler_mn == (128, 64)


def _quantize_mxfp4_rows(
    source: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent row-major MXFP4 quantizer for dense correctness tests."""

    rows, width = map(int, source.shape)
    blocked = source.float().view(rows, width // 32, 32)
    block_max = blocked.abs().amax(dim=-1, keepdim=True)
    safe_scale = torch.where(
        block_max > 0,
        block_max / 6.0,
        torch.ones_like(block_max),
    )
    exponent = torch.ceil(torch.log2(safe_scale)).clamp(-127, 127)
    scale_byte = torch.where(
        block_max > 0,
        exponent + 127,
        torch.zeros_like(exponent),
    ).to(torch.uint8)
    scale = torch.where(
        block_max > 0,
        torch.exp2(exponent),
        torch.zeros_like(exponent),
    )
    values = fp4_quantize_values_torch(
        torch.where(
            scale > 0,
            blocked / scale.clamp_min(1e-30),
            torch.zeros_like(blocked),
        ).view(rows, width)
    )
    magnitudes = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    nibbles = torch.zeros_like(values, dtype=torch.uint8)
    for code, magnitude in enumerate(magnitudes):
        nibbles = torch.where(
            values.abs() == magnitude,
            torch.full_like(nibbles, code),
            nibbles,
        )
    nibbles |= (values < 0).to(torch.uint8) << 3
    pairs = nibbles.view(rows, width // 2, 2)
    packed = (pairs[..., 0] | (pairs[..., 1] << 4)).contiguous()
    return packed, scale_byte.squeeze(-1).contiguous()


def _dequantize_mxfp4_rows(
    packed: torch.Tensor,
    scale_byte: torch.Tensor,
) -> torch.Tensor:
    magnitudes = torch.tensor(
        (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0),
        device=packed.device,
    )
    nibbles = torch.stack((packed & 0xF, packed >> 4), dim=-1).flatten(1)
    values = magnitudes[(nibbles & 0x7).long()]
    values *= torch.where((nibbles & 0x8) != 0, -1.0, 1.0)
    scale = torch.where(
        scale_byte == 0,
        0.0,
        torch.exp2(scale_byte.float() - 127.0),
    ).repeat_interleave(32, dim=1)
    return values * scale




def _make_quantized_operand(
    shape: tuple[int, int, int],
    *,
    dtype: torch.dtype,
) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    source = torch.randn(shape, device="cuda", dtype=dtype) / 4
    row_counts = torch.full(
        (shape[0],), shape[1], dtype=torch.int32, device=source.device
    )
    tensor_amax = source.abs().max().to(torch.float32)
    global_scale = torch.tensor(
        [torch.finfo(torch.float8_e4m3fn).max * 6.0 / tensor_amax],
        dtype=torch.float32,
        device=source.device,
    )
    packed, scales = quantize_grouped_nvfp4_torch(source, row_counts, global_scale)
    return (packed, scales), global_scale


def _prepared_nvfp4(lhs, rhs, lhs_scale, rhs_scale, *, c_dtype="bfloat16", **options):
    alpha = (1.0 / (lhs_scale[0] * rhs_scale[0])).view(1)
    return prepared(
        lhs, rhs, alpha=alpha, ab_dtype="float4_e2m1fn", sf_dtype="float8_e4m3fn",
        c_dtype=c_dtype, sf_vec_size=16, **options,
    )


def _mm_nvfp4(lhs, rhs, lhs_scale, rhs_scale, *, plan, c_dtype="bfloat16", out=None):
    alpha = (1.0 / (lhs_scale[0] * rhs_scale[0])).view(1)
    return blockscaled.mm(
        lhs, rhs, plan=plan, out=out, alpha=alpha, out_dtype=getattr(torch, c_dtype),
    )


@pytest.mark.parametrize(
    ("m", "n", "k", "c_dtype"),
    [
        (128, 128, 128, "bfloat16"),
        (256, 512, 128, "bfloat16"),
        (512, 256, 256, "bfloat16"),
        (128, 128, 128, "float16"),
    ],
)
def test_mm_nvfp4_matches_quantized_reference(m, n, k, c_dtype) -> None:
    require_b12x()
    torch.manual_seed(42)

    lhs, lhs_scale = _make_quantized_operand((1, m, k), dtype=torch.bfloat16)
    rhs, rhs_scale = _make_quantized_operand((1, n, k), dtype=torch.bfloat16)

    with _prepared_nvfp4(lhs, rhs, lhs_scale, rhs_scale, c_dtype=c_dtype) as plan:
        actual = _mm_nvfp4(lhs, rhs, lhs_scale, rhs_scale, c_dtype=c_dtype, plan=plan)

    oracle = nvfp4_gemm_reference(
        lhs, rhs, lhs_scale, rhs_scale, k=k, dtype=getattr(torch, c_dtype),
    )[0]

    torch.testing.assert_close(actual[:, :, 0], oracle, rtol=0, atol=0)


def test_mm_mxfp8_grouped_batches_use_their_own_scales() -> None:
    require_b12x()
    torch.manual_seed(29)

    # Pin BK64 to exercise packed-scale addressing across grouped WO-A batches.
    m, n, k = 64, 1024, 512
    groups = 4
    group_multipliers = torch.tensor(
        [1.0, 2.0, 4.0, 0.5], device="cuda", dtype=torch.bfloat16
    ).view(1, 1, groups)
    a = torch.randn((m, k, groups), device="cuda", dtype=torch.bfloat16) / 4
    a_q = quantize_mxfp8_rows_torch(a * group_multipliers)
    b_values = (
        torch.randn((groups * n, k), device="cuda", dtype=torch.bfloat16) / 32
    ).to(torch.float8_e4m3fn)
    b_scales = (
        torch.tensor([1.0, 2.0, 4.0, 0.5], device="cuda", dtype=torch.float32)
        .view(groups, 1, 1)
        .expand(groups, n // 128, k // 128)
        .reshape(groups * (n // 128), k // 128)
        .contiguous()
    )
    b_q = pack_fp8_block_scaled_weight_mxfp8(
        b_values, b_scales, m=n, k=k, num_groups=groups
    )
    assert not torch.equal(a_q.scale_rows[0], a_q.scale_rows[1])
    assert not torch.equal(b_q.scale_rows[0], b_q.scale_rows[1])

    lhs = (a_q.values, a_q.scale_mma)
    rhs = (b_q.values, b_q.scale_mma)
    with prepared(
        lhs, rhs, ab_dtype="float8_e4m3fn", sf_dtype="float8_e8m0fnu",
        c_dtype="bfloat16", sf_vec_size=32, mma_tiler_mn=(128, 128),
        expected_m=2048, sfb_k_replicated=True, _tile_k_override=64,
    ) as plan:
        out = blockscaled.mm(lhs, rhs, plan=plan)
    # Accumulate in FP64 so FP32 rounding cannot cross a BF16 midpoint.
    a_deq = dequantize_mxfp8_rows_torch(a_q.values, a_q.scale_rows).to(torch.float64)
    b_deq = dequantize_mxfp8_rows_torch(b_q.values, b_q.scale_rows).to(torch.float64)
    ref = torch.einsum("mkl,nkl->mnl", a_deq, b_deq).to(torch.bfloat16)

    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_mm_pair_replays_under_cuda_graph() -> None:
    require_b12x()
    torch.manual_seed(1234)

    gate_m, gate_n, gate_k = 32, 2048, 512
    down_m, down_n, down_k = 32, 1024, 2048

    gate_lhs, gate_ls = _make_quantized_operand(
        (1, gate_m, gate_k), dtype=torch.bfloat16
    )
    gate_rhs, gate_rs = _make_quantized_operand(
        (1, gate_n, gate_k), dtype=torch.bfloat16
    )
    down_lhs, down_ls = _make_quantized_operand(
        (1, down_m, down_k), dtype=torch.bfloat16
    )
    down_rhs, down_rs = _make_quantized_operand(
        (1, down_n, down_k), dtype=torch.bfloat16
    )

    graph_gate = torch.empty((gate_m, gate_n, 1), device="cuda", dtype=torch.bfloat16)
    graph_down = torch.empty((down_m, down_n, 1), device="cuda", dtype=torch.bfloat16)
    with (
        _prepared_nvfp4(
            gate_lhs, gate_rhs, gate_ls, gate_rs, out=graph_gate, freeze=False,
        ) as gate_plan,
        _prepared_nvfp4(
            down_lhs, down_rhs, down_ls, down_rs, out=graph_down, freeze=False,
        ) as down_plan,
        kernel_resolution_guard("prepared NVFP4 pair"),
    ):
        _mm_nvfp4(gate_lhs, gate_rhs, gate_ls, gate_rs, plan=gate_plan, out=graph_gate)
        _mm_nvfp4(down_lhs, down_rhs, down_ls, down_rs, plan=down_plan, out=graph_down)
        eager_gate, eager_down = graph_gate.clone(), graph_down.clone()
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                _mm_nvfp4(gate_lhs, gate_rhs, gate_ls, gate_rs, plan=gate_plan, out=graph_gate)
                _mm_nvfp4(down_lhs, down_rhs, down_ls, down_rs, plan=down_plan, out=graph_down)
            pointers = (graph_gate.data_ptr(), graph_down.data_ptr())
            allocated = torch.cuda.memory_allocated()
            for _ in range(3):
                graph_gate.fill_(float("nan"))
                graph_down.fill_(float("nan"))
                graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated() == allocated
            assert (graph_gate.data_ptr(), graph_down.data_ptr()) == pointers
            torch.testing.assert_close(graph_gate, eager_gate, rtol=0, atol=0)
            torch.testing.assert_close(graph_down, eager_down, rtol=0, atol=0)
        finally:
            graph.reset()


def test_mm_serialized_mxfp4_matches_independent_dequantized_reference() -> None:
    """Non-unit E8M0 scales catch scale-fragment ordering regressions."""

    require_b12x()
    torch.manual_seed(20260823)
    m, n, k = 6, 128, 256
    lhs_source = torch.randn((m, k), device="cuda") * 0.3
    rhs_source = torch.randn((n, k), device="cuda") * 0.3
    lhs_values, lhs_scale_rows = _quantize_mxfp4_rows(lhs_source)
    rhs_values, rhs_scale_rows = _quantize_mxfp4_rows(rhs_source)
    lhs_scale_storage = swizzle_block_scale(lhs_scale_rows)
    rhs_scale_storage = swizzle_block_scale(rhs_scale_rows)

    lhs_dequant = _dequantize_mxfp4_rows(lhs_values, lhs_scale_rows)
    rhs_dequant = _dequantize_mxfp4_rows(rhs_values, rhs_scale_rows)
    expected = (lhs_dequant.to(torch.bfloat16) @ rhs_dequant.to(torch.bfloat16).T).to(
        torch.bfloat16
    )
    with prepared(
        (lhs_values, lhs_scale_storage), (rhs_values, rhs_scale_storage),
        ab_dtype="float4_e2m1fn", sf_dtype="float8_e8m0fnu", c_dtype="bfloat16",
        sf_vec_size=32, expected_m=m,
    ) as plan:
        actual = blockscaled.mm(
            (lhs_values, lhs_scale_storage), (rhs_values, rhs_scale_storage),
            plan=plan, ab_dtype="float4_e2m1fn", sf_dtype="float8_e8m0fnu",
            c_dtype="bfloat16", sf_vec_size=32,
        )
        launch = torch.compile(
            lambda a, sa, b, sb: blockscaled.mm_mxfp4(a, sa, b, sb, plan=plan),
            fullgraph=True,
        )
        wrapped = launch(lhs_values, lhs_scale_storage, rhs_values, rhs_scale_storage)
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(wrapped, actual, rtol=0, atol=0)
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                graph_output = launch(lhs_values, lhs_scale_storage, rhs_values, rhs_scale_storage)
            output_ptr = graph_output.data_ptr()
            allocated = torch.cuda.memory_allocated()
            for _ in range(3):
                graph_output.fill_(float("nan"))
                graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated() == allocated
            assert graph_output.data_ptr() == output_ptr
            torch.testing.assert_close(graph_output, expected, rtol=0, atol=0)
        finally:
            graph.reset()


def test_mm_serialized_nvfp4_and_block_fp8_match_native_views() -> None:
    require_b12x()
    torch.manual_seed(20260824)
    m, n, k = 6, 128, 256

    lhs, lhs_global_scale = _make_quantized_operand((1, m, k), dtype=torch.bfloat16)
    rhs, rhs_global_scale = _make_quantized_operand((1, n, k), dtype=torch.bfloat16)
    alpha = (1.0 / (lhs_global_scale[0] * rhs_global_scale[0])).view(1)
    with _prepared_nvfp4(lhs, rhs, lhs_global_scale, rhs_global_scale) as plan:
        native_nvfp4 = _mm_nvfp4(lhs, rhs, lhs_global_scale, rhs_global_scale, plan=plan)[:, :, 0]
    serialized_lhs = (
        lhs[0][:, :, 0], convert_sf_from_mma_layout(lhs[1], m=m, k=k, num_groups=1),
    )
    serialized_rhs = (
        rhs[0][:, :, 0], convert_sf_from_mma_layout(rhs[1], m=n, k=k, num_groups=1),
    )
    with _prepared_nvfp4(serialized_lhs, serialized_rhs, lhs_global_scale, rhs_global_scale) as plan:
        serialized_nvfp4 = blockscaled.mm(
            serialized_lhs, serialized_rhs, plan=plan, alpha=alpha,
            ab_dtype="float4_e2m1fn", sf_dtype="float8_e4m3fn",
            c_dtype="bfloat16", sf_vec_size=16,
        )
        launch = torch.compile(
            lambda a, sa, b, sb, gain: blockscaled.mm_nvfp4(a, sa, b, sb, gain, plan=plan),
            fullgraph=True,
        )
        wrapped_nvfp4 = launch(*serialized_lhs, *serialized_rhs, alpha)
        torch.testing.assert_close(serialized_nvfp4, native_nvfp4, rtol=0, atol=0)
        torch.testing.assert_close(wrapped_nvfp4, serialized_nvfp4, rtol=0, atol=0)

    lhs_fp8 = torch.randn((m, k), device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    rhs_fp8 = torch.randn((n, k), device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    lhs_scale = torch.rand((m, k // 128), device="cuda", dtype=torch.float32)
    rhs_scale = torch.rand((n // 128, k // 128), device="cuda", dtype=torch.float32)
    native_lhs = (lhs_fp8.unsqueeze(-1), lhs_scale)
    native_rhs = (rhs_fp8.unsqueeze(-1), rhs_scale)
    options = dict(
        ab_dtype="float8_e4m3fn", sf_dtype="float32", c_dtype="bfloat16",
        sf_vec_size=128, block_fp8=True,
    )
    with prepared(native_lhs, native_rhs, expected_m=m, **options) as plan:
        native_block_fp8 = blockscaled.mm(native_lhs, native_rhs, plan=plan)[:, :, 0]
    with prepared((lhs_fp8, lhs_scale), (rhs_fp8, rhs_scale), expected_m=m, **options) as plan:
        serialized_block_fp8 = blockscaled.mm(
            (lhs_fp8, lhs_scale), (rhs_fp8, rhs_scale), plan=plan, **options,
        )
        wrapped_block_fp8 = blockscaled.mm_block_fp8(
            lhs_fp8, lhs_scale, rhs_fp8, rhs_scale, plan=plan, out_dtype=torch.bfloat16,
        )
        torch.testing.assert_close(serialized_block_fp8, native_block_fp8, rtol=0, atol=0)
        torch.testing.assert_close(wrapped_block_fp8, native_block_fp8, rtol=0, atol=0)


def test_nvfp4_a16_preserves_logical_k_inside_padded_weight() -> None:
    from dataclasses import replace

    device = require_b12x()
    m, n, logical_k, stored_k = 4, 128, 136, 256
    source = torch.randn((m, logical_k), dtype=torch.bfloat16, device=device)
    codes = torch.randint(0, 256, (n, stored_k // 2), dtype=torch.uint8, device=device)
    scales = torch.full(
        (n, stored_k // 16), 0.5, dtype=torch.float8_e4m3fn, device=device
    )
    gain = torch.tensor([0.03125], dtype=torch.float32, device=device)
    weight = blockscaled.pack_weight(
        codes,
        swizzle_block_scale(scales),
        recipe="nvfp4",
        global_scale=gain,
    )
    weight = replace(weight, in_features=logical_k)
    with prepared(
        source, weight, activation_mode="a16",
        override=blockscaled.BlockscaledConfig(mode="a16", tile_n=64, tile_k=64, split_k=2),
    ) as plan:
        output = blockscaled.mm(source, weight, plan=plan)
    lut = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
        dtype=torch.float32,
        device=device,
    )
    decoded = torch.stack(
        (lut[(codes & 15).long()], lut[(codes >> 4).long()]), -1
    ).flatten(-2)
    expected = source.float() @ (decoded[:, :logical_k] * 0.5 * gain).T
    cosine = torch.nn.functional.cosine_similarity(
        output.float().flatten(), expected.flatten(), dim=0
    )
    assert cosine.item() >= 0.999


@pytest.mark.parametrize("recipe", [
    "mxfp8", "mxfp8_fp16", "mxfp8_prequantized", "tensor_fp8", "block_fp8", "mxfp4", "nvfp4",
])
def test_implicit_plan_reuses_capacity_under_compile_and_graph(recipe):
    """Calls without a plan reuse heuristic programs across row counts and replay."""
    require_b12x()
    n, k = 128, 256
    if recipe in ("mxfp4", "nvfp4"):
        a = torch.randint(0, 256, (4, k // 2), device="cuda", dtype=torch.uint8)
        b = torch.randint(0, 256, (n, k // 2), device="cuda", dtype=torch.uint8)
        lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6], device="cuda")
        def unpack(value):
            return lut[torch.stack((value & 15, value >> 4), -1).long()].flatten(1)
        reference = unpack(a) @ unpack(b).T
        group = 32 if recipe == "mxfp4" else 16
        dtype = torch.uint8 if recipe == "mxfp4" else torch.float8_e4m3fn
        scale = 127 if recipe == "mxfp4" else 1
        sa = swizzle_block_scale(torch.full((4, k // group), scale, device="cuda").to(dtype))
        sb = swizzle_block_scale(torch.full((n, k // group), scale, device="cuda").to(dtype))
        alpha = torch.ones(1, device="cuda")
        def call(x):
            if recipe == "mxfp4":
                return blockscaled.mm_mxfp4(x, sa, b, sb)
            return blockscaled.mm_nvfp4(x, sa, b, sb, alpha)
    else:
        a = torch.randn((4, k), device="cuda", dtype=torch.bfloat16)
        b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        if recipe.startswith("mxfp8"):
            weight = blockscaled.pack_weight(b, torch.full((n, k // 32), 127, device="cuda", dtype=torch.uint8))
            if recipe == "mxfp8_fp16":
                a = torch.randint(-4, 5, (4, k), device="cuda").to(torch.float16)
            elif recipe == "mxfp8_prequantized":
                a = a.to(torch.float8_e4m3fn)
        else:
            a = a.to(torch.float8_e4m3fn)
            weight = blockscaled.pack_weight(b, torch.ones(1, device="cuda"))
        reference = a.float() @ b.float().T
        if recipe == "block_fp8":
            sa = torch.ones((4, k // 128), device="cuda")
            sb = torch.ones((n // 128, k // 128), device="cuda")
            def call(x):
                return blockscaled.mm_block_fp8(x, sa[:x.shape[0]], b, sb)
        else:
            def call(x):
                if recipe == "mxfp8_prequantized":
                    scales = torch.full((x.shape[0], k // 32), 127, device=x.device, dtype=torch.uint8)
                    return blockscaled.mm((x, scales), weight)
                return blockscaled.mm(x, weight, expected_m=x.shape[0])

    actual = call(a[:3])
    torch.testing.assert_close(actual.float(), reference[:3].to(actual.dtype).float(), atol=.125, rtol=.02)
    launch = torch.compile(call, fullgraph=True)
    compiled = launch(a)
    with kernel_resolution_guard("legacy capacity replay"):
        for rows in (3, 4):
            actual = call(a[:rows])
            torch.testing.assert_close(actual.float(), reference[:rows].to(actual.dtype).float(), atol=.125, rtol=.02)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            replayed = launch(a)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(replayed, compiled, atol=0, rtol=0)
