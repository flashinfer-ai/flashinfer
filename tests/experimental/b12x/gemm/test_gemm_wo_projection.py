from __future__ import annotations

import pytest
import torch

from b12x.gemm._shared.wo_mxfp8 import (
    FP8_E4M3_MAX,
    MXFP8Rows,
    WO_A_INPUT_QUANT_GROUP_SIZE,
    WOProjectionScratchCaps,
    dequantize_mxfp8_rows_torch,
    empty_dense_gemm_mnl_view,
    empty_mxfp8_rows_bases,
    empty_mxfp8_rows_for_dense_gemm,
    mxfp8_rows_from_bases,
    pack_fp8_block_scaled_weight_mxfp8,
    pack_mxfp8_scales_for_dense_gemm,
    pack_wo_projection_fp8_block_scaled_weights_mxfp8,
    quantize_mxfp8_rows_torch,
    quantize_wo_a_input_inv_rope_mxfp8,
    quantize_wo_a_input_mxfp8,
    quantize_wo_b_input_mxfp8,
    quantize_wo_projection_weights_mxfp8_torch,
    wo_a_dense_gemm_mxfp8,
    wo_b_dense_gemm_mxfp8,
)

from b12x.testing.reference.helpers import require_b12x
from b12x.gemm import wo_projection as wo
from b12x.preparation import PreparationSession, PreparedCall


def _assert_close_bf16(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(
        actual,
        expected.to(actual.dtype),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("rows", range(1, 9))
@pytest.mark.parametrize("block_size", (32, 128))
def test_decode_wo_b_column_tiles_match_independent_quantized_oracle(rows, block_size):
    """Column tiling preserves MXFP8 group-major quantization and BF16 output."""
    from b12x._lib.dense_gemm import dense_gemm_fused_quant_a

    require_b12x()
    torch.manual_seed(41732 + rows)
    source = (torch.randn(2, rows, 1024, device="cuda") / 4).bfloat16()
    source = source.permute(1, 2, 0)
    # Include an all-zero quantization block and distinct scales per group.
    source[:, :32, 0] = 0
    values = torch.randn(5120, 2048, device="cuda").to(torch.float8_e4m3fn)
    exponents = torch.randint(
        118, 124, (5120 // block_size, 2048 // block_size), device="cuda"
    ).byte()
    packed = pack_fp8_block_scaled_weight_mxfp8(
        values,
        exponents.view(torch.float8_e8m0fnu),
        m=5120,
        k=2048,
        num_groups=1,
        block_size=(block_size, block_size),
    )
    grouped = source.permute(0, 2, 1).reshape(rows, 64, 32).float()
    amax = grouped.abs().amax(-1)
    scales = torch.exp2(
        torch.ceil(
            torch.log2(torch.where(amax > 0, amax / 448, torch.ones_like(amax)))
        ).clamp(-127, 127)
    )
    quantized = (grouped / scales[..., None]).clamp(-448, 448).to(
        torch.float8_e4m3fn
    ).double() * scales[..., None].double()
    w_scales = (
        torch.exp2(exponents.double() - 127)
        .repeat_interleave(block_size, 0)
        .repeat_interleave(block_size, 1)
    )
    expected = quantized.reshape(rows, 2048) @ (values.double() * w_scales).T
    # FP32 tensor-core accumulation may round an exact BF16 midpoint to either
    # neighbor. Permit one BF16 ULP, not a generic relative-error threshold.
    rounded = expected.bfloat16()
    ulp = (
        torch.nextafter(rounded, torch.full_like(rounded, float("inf"))).double()
        - rounded.double()
    ).abs()
    results = []
    for tile in (0, 64, 128):
        actual = dense_gemm_fused_quant_a(
            source,
            packed.values.reshape(5120, 2048, 1),
            packed.scale_mma,
            expected_m=rows,
            sfb_k_replicated=block_size == 128,
            rhs_values_tiled=packed.values_tiled,
            mma_tiler_mn=(16, tile) if tile else None,
            a_inner_span=1024,
        )
        actual = actual[..., 0].clone()
        assert bool(torch.isfinite(actual).all())
        assert bool(torch.count_nonzero(actual))
        assert bool(((actual.double() - rounded.double()).abs() <= ulp).all())
        results.append(actual)
    for actual in results[1:]:
        torch.testing.assert_close(actual, results[0], atol=0, rtol=0)


@pytest.fixture
def wo_session():
    with PreparationSession(device=require_b12x(), autotune=False, compile_workers=2) as session:
        yield session


def _make_wo_projection_binding(source_tgd, weights, *, session, config=None):
    tokens, groups, group_width = source_tgd.shape
    plan = wo.plan(wo.Caps(device=source_tgd.device, max_tokens=tokens, groups=groups,
                           group_width=group_width, rank=weights.rank, hidden=weights.hidden,
                           dtype=source_tgd.dtype), override=config)
    kwargs = dict(source_tgd=source_tgd, weights=weights)
    def prepare(state):
        scratch = tuple(torch.empty(spec.shape, dtype=spec.dtype, device=source_tgd.device)
                        for spec in state._scratch_state.scratch_specs())
        binding = state.bind(scratch=scratch, **kwargs)
        return PreparedCall(run=lambda: state.run(binding), owners=(scratch, binding))
    session.prepare((plan.request(name="wo", prepare_call=prepare),))
    scratch = tuple(torch.empty(spec.shape, dtype=spec.dtype, device=source_tgd.device)
                    for spec in plan.scratch_specs())
    return wo.bind(plan, scratch=scratch, **kwargs)


def _make_inv_rope_binding(o, positions, cos_sin_cache, weights, *, session,
                           heads_per_group, nope_dim, rope_dim):
    plan = wo.plan(wo.Caps(device=o.device, max_tokens=o.shape[0], groups=weights.groups,
                           group_width=weights.group_width, rank=weights.rank, hidden=weights.hidden,
                           dtype=o.dtype), invocation=dict(operation="inv_rope",
        heads_per_group=heads_per_group, nope_dim=nope_dim, rope_dim=rope_dim,
        positions_dtype=str(positions.dtype).removeprefix("torch."),
        cos_sin_dtype=str(cos_sin_cache.dtype).removeprefix("torch.")))
    kwargs = dict(o=o, positions=positions, cos_sin_cache=cos_sin_cache, weights=weights,
                  heads_per_group=heads_per_group, nope_dim=nope_dim, rope_dim=rope_dim)
    def prepare(state):
        scratch = tuple(torch.empty(spec.shape, dtype=spec.dtype, device=o.device)
                        for spec in state._scratch_state.scratch_specs())
        binding = state.bind_inv_rope(scratch=scratch, **kwargs)
        return PreparedCall(run=lambda: state.run_inv_rope(binding), owners=(scratch, binding))
    session.prepare((plan.request(name="wo-inv-rope", prepare_call=prepare),))
    scratch = tuple(torch.empty(spec.shape, dtype=spec.dtype, device=o.device)
                    for spec in plan.scratch_specs())
    return wo.bind_inv_rope(plan, scratch=scratch, **kwargs)


def _run_binding(*, binding):
    return wo.run(binding=binding, plan=binding.plan)


def _sglang_wo_a_input_quant_reference(source_tgd: torch.Tensor) -> MXFP8Rows:
    tokens, groups, group_width = source_tgd.shape
    chunks = group_width // WO_A_INPUT_QUANT_GROUP_SIZE
    blocked = source_tgd.float().reshape(
        tokens,
        groups,
        chunks,
        WO_A_INPUT_QUANT_GROUP_SIZE,
    )
    max_abs = blocked.abs().amax(dim=-1)
    quant_scale = torch.where(
        max_abs > 0,
        max_abs / FP8_E4M3_MAX,
        torch.ones_like(max_abs),
    )
    scale_u8 = (torch.ceil(torch.log2(quant_scale)).clamp(-127, 127) + 127).to(
        torch.uint8
    )
    # MXFP8 stores only the UE8M0 power-of-two scale, so values must be
    # quantized with that representable scale rather than the pre-rounded
    # max/FP8_MAX value.
    quant_scale_e8m0 = torch.exp2(scale_u8.to(torch.float32) - 127.0)
    values_tgd = (
        (blocked / quant_scale_e8m0[..., None])
        .clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
        .to(torch.float8_e4m3fn)
        .reshape(tokens, groups, group_width)
    )
    values_grouped = values_tgd.permute(1, 0, 2).contiguous()
    values = values_grouped.as_strided(
        (tokens, group_width, groups),
        (group_width, 1, tokens * group_width),
    )
    scale_rows_u8 = scale_u8.repeat_interleave(
        WO_A_INPUT_QUANT_GROUP_SIZE // 32,
        dim=2,
    )
    scale_rows = scale_rows_u8.permute(1, 0, 2).contiguous().view(torch.float8_e8m0fnu)
    scale_mma = pack_mxfp8_scales_for_dense_gemm(
        scale_rows,
        m=tokens,
        k=group_width,
        num_groups=groups,
    )
    return MXFP8Rows(values=values, scale_rows=scale_rows, scale_mma=scale_mma)


def test_pack_mxfp8_scales_round_trips_grouped_rows() -> None:
    require_b12x()

    groups, m, k = 3, 5, 256
    sf_k = k // 32
    scale_u8 = (
        torch.arange(groups * m * sf_k, device="cuda", dtype=torch.int32) % 16 + 120
    ).to(torch.uint8)
    scale = scale_u8.view(torch.float8_e8m0fnu).reshape(groups, m, sf_k)

    packed = pack_mxfp8_scales_for_dense_gemm(
        scale,
        m=m,
        k=k,
        num_groups=groups,
    )
    round_trip = (
        packed.permute(5, 2, 1, 0, 4, 3).contiguous().reshape(groups, 128, sf_k)
    )

    torch.testing.assert_close(
        round_trip[:, :m, :].view(torch.uint8),
        scale.view(torch.uint8),
        rtol=0,
        atol=0,
    )
    assert bool((round_trip[:, m:, :].view(torch.uint8) == 127).all().item())


def test_pack_fp8_block_scaled_weight_expands_grouped_scales() -> None:
    require_b12x()
    torch.manual_seed(20260523)

    groups, m, k = 2, 129, 256
    m_tiles = 2
    k_tiles = 2
    raw_weight = (
        torch.randn((groups * m, k), device="cuda", dtype=torch.bfloat16) / 3
    ).to(torch.float8_e4m3fn)
    block_scale_u8 = (
        torch.arange(groups * m_tiles * k_tiles, device="cuda", dtype=torch.int32) % 8
        + 124
    ).to(torch.uint8)
    block_scale = block_scale_u8.view(torch.float8_e8m0fnu).reshape(
        groups * m_tiles,
        k_tiles,
    )

    packed = pack_fp8_block_scaled_weight_mxfp8(
        raw_weight,
        block_scale,
        m=m,
        k=k,
        num_groups=groups,
    )
    expected_scale_rows_u8 = (
        block_scale_u8.reshape(groups, m_tiles, k_tiles)[:, :, None, :, None]
        .expand(groups, m_tiles, 128, k_tiles, 4)
        .reshape(groups, m_tiles * 128, k_tiles * 4)[:, :m, : k // 32]
        .contiguous()
    )

    assert packed.values.shape == (m, k, groups)
    assert packed.values.stride() == (k, 1, m * k)
    torch.testing.assert_close(
        packed.scale_rows.view(torch.uint8),
        expected_scale_rows_u8,
        rtol=0,
        atol=0,
    )

    deq = dequantize_mxfp8_rows_torch(packed.values, packed.scale_rows)
    expected_values = raw_weight.view(groups, m, k).permute(1, 2, 0)
    expected = (
        (
            expected_values.float().reshape(m, k // 32, 32, groups).permute(3, 0, 1, 2)
            * expected_scale_rows_u8.view(torch.float8_e8m0fnu).float()[..., None]
        )
        .permute(1, 2, 3, 0)
        .reshape(m, k, groups)
    )
    torch.testing.assert_close(deq, expected, rtol=0, atol=0)


def test_pack_fp8_block_scaled_weight_accepts_float_scales() -> None:
    require_b12x()
    torch.manual_seed(20260524)

    groups, m, k = 4, 1024, 4096
    m_tiles = m // 128
    k_tiles = k // 128
    raw_weight = (
        torch.randn((groups * m, k), device="cuda", dtype=torch.bfloat16) / 8
    ).to(torch.float8_e4m3fn)
    scale_u8 = (
        torch.arange(groups * m_tiles * k_tiles, device="cuda", dtype=torch.int32) % 4
        + 125
    ).to(torch.uint8)
    scale_e8m0 = scale_u8.view(torch.float8_e8m0fnu).reshape(
        groups * m_tiles,
        k_tiles,
    )
    scale_float = scale_e8m0.float()

    packed_e8m0 = pack_fp8_block_scaled_weight_mxfp8(
        raw_weight,
        scale_e8m0,
        m=m,
        k=k,
        num_groups=groups,
    )
    packed_float = pack_fp8_block_scaled_weight_mxfp8(
        raw_weight,
        scale_float,
        m=m,
        k=k,
        num_groups=groups,
    )

    torch.testing.assert_close(
        packed_float.scale_rows.view(torch.uint8),
        packed_e8m0.scale_rows.view(torch.uint8),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        packed_float.scale_mma.view(torch.uint8),
        packed_e8m0.scale_mma.view(torch.uint8),
        rtol=0,
        atol=0,
    )


def test_quantize_mxfp8_rows_dequantizes_on_gpu() -> None:
    require_b12x()
    torch.manual_seed(20260522)

    source = (
        torch.randn((3, 128, 2), device="cuda", dtype=torch.bfloat16) / 4
    ).contiguous()
    q = quantize_mxfp8_rows_torch(source)
    deq = dequantize_mxfp8_rows_torch(q.values, q.scale_rows)

    assert q.values.shape == source.shape
    assert q.values.dtype == torch.float8_e4m3fn
    assert q.scale_rows.shape == (2, 3, 4)
    assert q.scale_mma.shape == (32, 4, 1, 4, 1, 2)
    assert bool(torch.isfinite(deq).all().item())
    max_abs = (deq.float() - source.float()).abs().max().item()
    assert max_abs < 0.05


def test_wo_activation_quant_kernels_match_gpu_reference() -> None:
    require_b12x()
    torch.manual_seed(31000)

    tokens, groups, group_width, rank = 3, 4, 512, 64
    source_tgd = (
        torch.randn((tokens, groups, group_width), device="cuda", dtype=torch.bfloat16)
        / 4
    ).contiguous()
    actual_a = quantize_wo_a_input_mxfp8(source_tgd)
    expected_a = _sglang_wo_a_input_quant_reference(source_tgd)
    torch.cuda.synchronize()

    torch.testing.assert_close(
        actual_a.values.view(torch.uint8),
        expected_a.values.view(torch.uint8),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        actual_a.scale_rows.view(torch.uint8),
        expected_a.scale_rows.view(torch.uint8),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        actual_a.scale_mma.view(torch.uint8),
        expected_a.scale_mma.view(torch.uint8),
        rtol=0,
        atol=0,
    )

    source_trg = empty_dense_gemm_mnl_view(
        tokens,
        rank,
        groups,
        device="cuda",
        dtype=torch.bfloat16,
    )
    source_trg.copy_(
        torch.randn((tokens, rank, groups), device="cuda", dtype=torch.bfloat16) / 4
    )
    actual_b = quantize_wo_b_input_mxfp8(source_trg)
    expected_b = quantize_mxfp8_rows_torch(
        source_trg.permute(0, 2, 1).contiguous().reshape(tokens, rank * groups)
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(
        actual_b.values.view(torch.uint8),
        expected_b.values.view(torch.uint8),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        actual_b.scale_rows.view(torch.uint8),
        expected_b.scale_rows.view(torch.uint8),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        actual_b.scale_mma.view(torch.uint8),
        expected_b.scale_mma.view(torch.uint8),
        rtol=0,
        atol=0,
    )


def test_wo_a_inv_rope_input_quant_uses_mxfp8_32_column_groups() -> None:
    require_b12x()
    torch.manual_seed(31005)

    tokens = 3
    groups = 2
    heads_per_group = 4
    head_dim = 128
    nope_dim = 96
    rope_dim = 32
    group_width = heads_per_group * head_dim
    source_tgd = (
        torch.randn((tokens, groups, group_width), device="cuda", dtype=torch.bfloat16)
        / 8
    ).contiguous()
    source_tgd[:, :, 0::WO_A_INPUT_QUANT_GROUP_SIZE] = 3.5
    source_tgd[:, :, 32::WO_A_INPUT_QUANT_GROUP_SIZE] /= 16

    o = source_tgd.reshape(tokens, groups, heads_per_group, head_dim).reshape(
        tokens,
        groups * heads_per_group,
        head_dim,
    )
    positions = torch.arange(tokens, device="cuda", dtype=torch.long)
    cos_sin_cache = torch.zeros((tokens, rope_dim), device="cuda", dtype=torch.float32)
    cos_sin_cache[:, : rope_dim // 2] = 1

    actual = quantize_wo_a_input_inv_rope_mxfp8(
        o,
        positions,
        cos_sin_cache,
        groups=groups,
        heads_per_group=heads_per_group,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
    )
    expected = _sglang_wo_a_input_quant_reference(source_tgd)
    torch.cuda.synchronize()

    torch.testing.assert_close(
        actual.values.view(torch.uint8),
        expected.values.view(torch.uint8),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        actual.scale_rows.view(torch.uint8),
        expected.scale_rows.view(torch.uint8),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        actual.scale_mma.view(torch.uint8),
        expected.scale_mma.view(torch.uint8),
        rtol=0,
        atol=0,
    )


def test_wo_b_dense_gemm_ignores_poisoned_activation_scale_padding() -> None:
    require_b12x()
    torch.manual_seed(31006)

    tokens, rank, groups, hidden = 1, 1024, 4, 128
    width = rank * groups
    source = (
        torch.randn((tokens, rank, groups), device="cuda", dtype=torch.bfloat16) / 4
    ).contiguous()
    quantized = []
    bases = []
    for poison in (0, 254):
        values, scale_rows, scale_physical = empty_mxfp8_rows_bases(
            tokens,
            width,
            num_groups=1,
            device="cuda",
            initialize_scales=False,
        )
        values.view(torch.uint8).fill_(poison)
        scale_rows.fill_(poison)
        scale_physical.fill_(poison)
        out = mxfp8_rows_from_bases(
            values,
            scale_rows,
            scale_physical,
            tokens,
            width,
            num_groups=1,
        )
        quantize_wo_b_input_mxfp8(source, out=out)
        quantized.append(out)
        bases.append((values, scale_rows, scale_physical))

    weight = empty_mxfp8_rows_for_dense_gemm(hidden, width, device="cuda")
    weight.values.fill_(1)
    actual_0 = wo_b_dense_gemm_mxfp8(quantized[0], weight, expected_m=1).clone()
    actual_1 = wo_b_dense_gemm_mxfp8(quantized[1], weight, expected_m=1).clone()
    torch.cuda.synchronize()

    # Both quantizers overwrite every logical value and row scale. Most of the
    # physical scale tile is M padding at tokens=1 and intentionally remains
    # poisoned; the dense kernel must not let it affect the logical output row.
    assert torch.equal(bases[0][0].view(torch.uint8), bases[1][0].view(torch.uint8))
    assert torch.equal(bases[0][1], bases[1][1])
    assert bool((bases[0][2] != bases[1][2]).any().item())
    assert bool(torch.isfinite(actual_0).all().item())
    assert bool((actual_0 != 0).any().item())
    torch.testing.assert_close(actual_0, actual_1, rtol=0, atol=0)


def test_wo_a_dense_gemm_ignores_poisoned_activation_scale_padding() -> None:
    require_b12x()
    torch.manual_seed(31008)

    tokens, groups, heads_per_group = 1, 2, 4
    nope_dim, rope_dim = 96, 32
    head_dim = nope_dim + rope_dim
    group_width = heads_per_group * head_dim
    rank = 64
    o = (
        torch.randn(
            (tokens, groups * heads_per_group, head_dim),
            device="cuda",
            dtype=torch.bfloat16,
        )
        / 4
    ).contiguous()
    positions = torch.zeros((tokens,), device="cuda", dtype=torch.long)
    cos_sin_cache = torch.zeros((1, rope_dim), device="cuda", dtype=torch.float32)
    cos_sin_cache[:, : rope_dim // 2] = 1
    quantized = []
    bases = []
    for poison in (0, 254):
        values, scale_rows, scale_physical = empty_mxfp8_rows_bases(
            tokens,
            group_width,
            num_groups=groups,
            device="cuda",
            initialize_scales=False,
        )
        values.view(torch.uint8).fill_(poison)
        scale_rows.fill_(poison)
        scale_physical.fill_(poison)
        out = mxfp8_rows_from_bases(
            values,
            scale_rows,
            scale_physical,
            tokens,
            group_width,
            num_groups=groups,
        )
        quantize_wo_a_input_inv_rope_mxfp8(
            o,
            positions,
            cos_sin_cache,
            groups=groups,
            heads_per_group=heads_per_group,
            nope_dim=nope_dim,
            rope_dim=rope_dim,
            out=out,
        )
        quantized.append(out)
        bases.append((values, scale_rows, scale_physical))

    weight = empty_mxfp8_rows_for_dense_gemm(
        rank, group_width, num_groups=groups, device="cuda"
    )
    weight.values.fill_(1)
    actual_0 = wo_a_dense_gemm_mxfp8(quantized[0], weight, expected_m=1).clone()
    actual_1 = wo_a_dense_gemm_mxfp8(quantized[1], weight, expected_m=1).clone()
    torch.cuda.synchronize()

    assert torch.equal(bases[0][0].view(torch.uint8), bases[1][0].view(torch.uint8))
    assert torch.equal(bases[0][1], bases[1][1])
    assert bool((bases[0][2] != bases[1][2]).any().item())
    assert bool(torch.isfinite(actual_0).all().item())
    assert bool((actual_0 != 0).any().item())
    torch.testing.assert_close(actual_0, actual_1, rtol=0, atol=0)


def test_wo_a_dense_gemm_mxfp8_matches_quantized_gpu_reference() -> None:
    require_b12x()
    torch.manual_seed(31001)

    tokens, groups, group_width, rank = 2, 2, 128, 64
    x_tgd = (
        torch.randn((tokens, groups, group_width), device="cuda", dtype=torch.bfloat16)
        / 4
    )
    wo_a_grd = (
        torch.randn((groups, rank, group_width), device="cuda", dtype=torch.bfloat16)
        / group_width**0.5
    )
    x_tdg = quantize_wo_a_input_mxfp8(x_tgd)
    wo_a_rdg = quantize_mxfp8_rows_torch(wo_a_grd.permute(1, 2, 0).contiguous())

    actual = wo_a_dense_gemm_mxfp8(x_tdg, wo_a_rdg)
    torch.cuda.synchronize()

    x_deq = dequantize_mxfp8_rows_torch(x_tdg.values, x_tdg.scale_rows).permute(0, 2, 1)
    wo_a_deq = dequantize_mxfp8_rows_torch(
        wo_a_rdg.values,
        wo_a_rdg.scale_rows,
    ).permute(2, 0, 1)
    expected = torch.einsum("tgd,grd->trg", x_deq, wo_a_deq)

    _assert_close_bf16(actual, expected)


def test_two_gemm_wo_projection_group_major_path_matches_quantized_reference() -> None:
    require_b12x()
    torch.manual_seed(31002)

    tokens, groups, group_width, rank, hidden = 2, 2, 128, 64, 128
    x_tgd = (
        torch.randn((tokens, groups, group_width), device="cuda", dtype=torch.bfloat16)
        / 4
    )
    wo_a_grd = (
        torch.randn((groups, rank, group_width), device="cuda", dtype=torch.bfloat16)
        / group_width**0.5
    )
    wo_b_hgr = (
        torch.randn((hidden, groups * rank), device="cuda", dtype=torch.bfloat16)
        / (groups * rank) ** 0.5
    ).contiguous()

    x_tdg = quantize_wo_a_input_mxfp8(x_tgd)
    wo_a_rdg = quantize_mxfp8_rows_torch(wo_a_grd.permute(1, 2, 0).contiguous())
    tmp_trg = wo_a_dense_gemm_mxfp8(x_tdg, wo_a_rdg)
    tmp_q = quantize_wo_b_input_mxfp8(tmp_trg)

    wo_b_q = quantize_mxfp8_rows_torch(wo_b_hgr)
    actual = wo_b_dense_gemm_mxfp8(tmp_q, wo_b_q)
    torch.cuda.synchronize()

    tmp_deq = dequantize_mxfp8_rows_torch(tmp_q.values, tmp_q.scale_rows)
    wo_b_deq = dequantize_mxfp8_rows_torch(wo_b_q.values, wo_b_q.scale_rows)
    expected = tmp_deq @ wo_b_deq.T

    _assert_close_bf16(actual[:, :, 0], expected)


def test_two_gemm_wo_projection_singleton_group_matches_quantized_reference(wo_session) -> None:
    """TP8 collapses DSV4's eight output groups to one local WO group."""

    require_b12x()
    torch.manual_seed(31005)

    tokens, groups, group_width, rank, hidden = 3, 1, 512, 128, 128
    x_tgd = (
        torch.randn((tokens, groups, group_width), device="cuda", dtype=torch.bfloat16)
        / 4
    )
    wo_a_grd = (
        torch.randn((groups, rank, group_width), device="cuda", dtype=torch.bfloat16)
        / group_width**0.5
    )
    wo_b_hgr = (
        torch.randn((hidden, groups * rank), device="cuda", dtype=torch.bfloat16)
        / (groups * rank) ** 0.5
    )

    weights = quantize_wo_projection_weights_mxfp8_torch(wo_a_grd, wo_b_hgr)
    binding = _make_wo_projection_binding(x_tgd, weights, session=wo_session)
    actual = _run_binding(binding=binding)
    torch.cuda.synchronize()

    x_q = quantize_wo_a_input_mxfp8(x_tgd)
    x_deq = dequantize_mxfp8_rows_torch(x_q.values, x_q.scale_rows)
    wo_a_deq = dequantize_mxfp8_rows_torch(
        weights.wo_a.values,
        weights.wo_a.scale_rows,
    )
    tmp = (x_deq @ wo_a_deq.T).to(torch.bfloat16).unsqueeze(-1)
    tmp_q = quantize_wo_b_input_mxfp8(tmp)
    tmp_deq = dequantize_mxfp8_rows_torch(tmp_q.values, tmp_q.scale_rows)
    wo_b_deq = dequantize_mxfp8_rows_torch(
        weights.wo_b.values,
        weights.wo_b.scale_rows,
    )
    expected = tmp_deq @ wo_b_deq.T

    assert weights.wo_a.values.ndim == 2
    assert x_q.values.ndim == 2
    _assert_close_bf16(actual, expected)


def test_two_gemm_wo_projection_replays_under_graph(wo_session) -> None:
    require_b12x()
    torch.manual_seed(31003)

    tokens, groups, group_width, rank, hidden = 1, 2, 128, 64, 128
    x_tgd = (
        torch.randn((tokens, groups, group_width), device="cuda", dtype=torch.bfloat16)
        / 4
    )
    wo_a_grd = (
        torch.randn((groups, rank, group_width), device="cuda", dtype=torch.bfloat16)
        / group_width**0.5
    )
    wo_b_hgr = (
        torch.randn((hidden, groups * rank), device="cuda", dtype=torch.bfloat16)
        / (groups * rank) ** 0.5
    )

    weights = quantize_wo_projection_weights_mxfp8_torch(wo_a_grd, wo_b_hgr)
    binding = _make_wo_projection_binding(x_tgd, weights, session=wo_session)

    def run_once() -> torch.Tensor:
        return _run_binding(binding=binding)

    eager = run_once().clone()
    torch.cuda.synchronize()

    run_once()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_once()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(binding.output[:, :, 0], eager, rtol=0, atol=0)
    graph.reset()


def test_inv_rope_fused_wo_replays_under_graph_with_uninitialized_scale_padding(wo_session):
    require_b12x()
    torch.manual_seed(31007)

    tokens = 1
    groups = 2
    heads_per_group = 4
    nope_dim = 96
    rope_dim = 32
    head_dim = nope_dim + rope_dim
    group_width = heads_per_group * head_dim
    rank, hidden = 64, 128
    o = (
        torch.randn(
            (tokens, groups * heads_per_group, head_dim),
            device="cuda",
            dtype=torch.bfloat16,
        )
        / 4
    ).contiguous()
    positions = torch.zeros((tokens,), device="cuda", dtype=torch.long)
    cos_sin_cache = torch.zeros((4, rope_dim), device="cuda", dtype=torch.float32)
    cos_sin_cache[:, : rope_dim // 2] = 1
    wo_a = (
        torch.randn((groups, rank, group_width), device="cuda", dtype=torch.bfloat16)
        / group_width**0.5
    )
    wo_b = (
        torch.randn((hidden, groups * rank), device="cuda", dtype=torch.bfloat16)
        / (groups * rank) ** 0.5
    )
    weights = quantize_wo_projection_weights_mxfp8_torch(wo_a, wo_b)

    binding = _make_inv_rope_binding(o, positions, cos_sin_cache, weights,
        session=wo_session, heads_per_group=heads_per_group, nope_dim=nope_dim, rope_dim=rope_dim)
    wo_session.freeze()
    def run_once():
        return wo.run_inv_rope(binding=binding, plan=binding.plan)

    # Warm all compilation/allocation before capture, then retain the graph-owned
    # output and verify replay observes changed inputs without allocating or
    # depending on stale scale padding.
    run_once()
    run_once()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run_once()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()

    o.copy_(torch.randn_like(o) / 4)
    graph.replay()
    torch.cuda.synchronize()
    replayed = captured.clone()
    expected = run_once().clone()
    torch.cuda.synchronize()

    assert bool(torch.isfinite(replayed).all().item())
    assert bool((replayed != 0).any().item())
    torch.testing.assert_close(replayed, expected, rtol=0, atol=0)
    graph.reset()


@pytest.mark.parametrize("tokens", [1, 8, 16])
def test_wo_prepared_decode_tiles_and_prefill(tokens, wo_session):
    require_b12x()
    torch.manual_seed(31008)
    groups, width, rank, hidden = 2, 4096, 1024, 5120
    source = torch.randn(tokens, groups, width, device="cuda", dtype=torch.bfloat16).mul_(0.25)
    a = torch.randn(groups, rank, width, device="cuda", dtype=torch.bfloat16) / width**0.5
    b = torch.randn(hidden, groups * rank, device="cuda", dtype=torch.bfloat16) / (groups * rank)**0.5
    weights = quantize_wo_projection_weights_mxfp8_torch(a, b)
    bindings = [_make_wo_projection_binding(source, weights, session=wo_session,
                 config=wo.WoProjectionConfig(decode_tile_n=tile))
                for tile in ((64, 128) if tokens <= 8 else (0,))]
    x_q = quantize_wo_a_input_mxfp8(source)
    tmp = wo_a_dense_gemm_mxfp8(x_q, weights.wo_a)
    tmp_q = quantize_wo_b_input_mxfp8(tmp)
    expected = wo_b_dense_gemm_mxfp8(tmp_q, weights.wo_b)[:, :, 0]
    wo_session.freeze()
    outputs = []
    for binding in bindings:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _run_binding(binding=binding)
        graph.replay()
        torch.cuda.synchronize()
        actual = binding.output[:, :, 0].clone()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        outputs.append(actual)
        graph.reset()
    for output in outputs[1:]:
        torch.testing.assert_close(output, outputs[0], rtol=0, atol=0)


def test_wo_projection_block_scaled_weight_pack_runs_graph(wo_session) -> None:
    require_b12x()
    torch.manual_seed(31004)

    tokens, groups, group_width, rank, hidden = 1, 2, 128, 128, 128
    x_tgd = (
        torch.randn((tokens, groups, group_width), device="cuda", dtype=torch.bfloat16)
        / 4
    )
    wo_a_weight = (
        torch.randn((groups * rank, group_width), device="cuda", dtype=torch.bfloat16)
        / 8
    ).to(torch.float8_e4m3fn)
    wo_b_weight = (
        torch.randn((hidden, groups * rank), device="cuda", dtype=torch.bfloat16) / 8
    ).to(torch.float8_e4m3fn)
    wo_a_scale = torch.full(
        (groups * (rank // 128), group_width // 128),
        127,
        dtype=torch.uint8,
        device="cuda",
    ).view(torch.float8_e8m0fnu)
    wo_b_scale = torch.full(
        (hidden // 128, groups * rank // 128),
        127,
        dtype=torch.uint8,
        device="cuda",
    ).view(torch.float8_e8m0fnu)

    weights = pack_wo_projection_fp8_block_scaled_weights_mxfp8(
        wo_a_weight,
        wo_a_scale,
        wo_b_weight,
        wo_b_scale,
        groups=groups,
        group_width=group_width,
        rank=rank,
        hidden=hidden,
    )
    binding = _make_wo_projection_binding(x_tgd, weights, session=wo_session)

    eager = _run_binding(binding=binding).clone()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _run_binding(binding=binding)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()

    assert bool(torch.isfinite(binding.output).all().item())
    torch.testing.assert_close(binding.output[:, :, 0], eager, rtol=0, atol=0)
    graph.reset()


def test_wo_projection_block32_pack_preserves_k_scale_bytes(wo_session) -> None:
    require_b12x()
    torch.manual_seed(41004)

    tokens, groups, group_width, rank, hidden = 3, 2, 128, 128, 128
    x_tgd = (
        torch.randn((tokens, groups, group_width), device="cuda", dtype=torch.bfloat16)
        / 4
    )
    wo_a_weight = (
        torch.randn((groups * rank, group_width), device="cuda", dtype=torch.bfloat16)
        / 8
    ).to(torch.float8_e4m3fn)
    wo_b_weight = (
        torch.randn((hidden, groups * rank), device="cuda", dtype=torch.bfloat16) / 8
    ).to(torch.float8_e4m3fn)
    wo_a_scale = torch.randint(
        121,
        130,
        (groups * (rank // 32), group_width // 32),
        dtype=torch.uint8,
        device="cuda",
    ).view(torch.float8_e8m0fnu)
    wo_b_scale = torch.randint(
        121,
        130,
        (hidden // 32, groups * rank // 32),
        dtype=torch.uint8,
        device="cuda",
    ).view(torch.float8_e8m0fnu)

    weights = pack_wo_projection_fp8_block_scaled_weights_mxfp8(
        wo_a_weight,
        wo_a_scale,
        wo_b_weight,
        wo_b_scale,
        groups=groups,
        group_width=group_width,
        rank=rank,
        hidden=hidden,
        block_size=(32, 32),
    )
    assert not weights.sfb_k_replicated

    actual = _run_binding(binding=_make_wo_projection_binding(x_tgd, weights, session=wo_session))
    x_q = quantize_wo_a_input_mxfp8(x_tgd)
    x_deq = dequantize_mxfp8_rows_torch(x_q.values, x_q.scale_rows)
    wo_a_deq = dequantize_mxfp8_rows_torch(weights.wo_a.values, weights.wo_a.scale_rows)
    tmp = torch.einsum("mkg,rkg->mrg", x_deq, wo_a_deq).to(torch.bfloat16)
    tmp_q = quantize_wo_b_input_mxfp8(tmp)
    tmp_deq = dequantize_mxfp8_rows_torch(tmp_q.values, tmp_q.scale_rows)
    wo_b_deq = dequantize_mxfp8_rows_torch(weights.wo_b.values, weights.wo_b.scale_rows)
    expected = tmp_deq @ wo_b_deq.T
    torch.testing.assert_close(actual, expected.to(actual.dtype), rtol=0, atol=0)


def test_wo_dense_gemms_sfb_k_reuse_is_byte_identical() -> None:
    require_b12x()
    torch.manual_seed(31007)

    tokens, groups, group_width, rank, hidden = 5, 2, 256, 128, 256
    x_tgd = (
        torch.randn((tokens, groups, group_width), device="cuda", dtype=torch.bfloat16)
        / 4
    )
    wo_a_weight = (
        torch.randn((groups * rank, group_width), device="cuda", dtype=torch.bfloat16)
        / 8
    ).to(torch.float8_e4m3fn)
    wo_b_weight = (
        torch.randn((hidden, groups * rank), device="cuda", dtype=torch.bfloat16) / 8
    ).to(torch.float8_e4m3fn)
    # Distinct per-128-block UE8M0 scales so k reuse would diverge if the
    # replication assumption were wrong.
    wo_a_scale = torch.randint(
        120,
        134,
        (groups * (rank // 128), group_width // 128),
        dtype=torch.uint8,
        device="cuda",
    ).view(torch.float8_e8m0fnu)
    wo_b_scale = torch.randint(
        120,
        134,
        (hidden // 128, groups * rank // 128),
        dtype=torch.uint8,
        device="cuda",
    ).view(torch.float8_e8m0fnu)

    weights = pack_wo_projection_fp8_block_scaled_weights_mxfp8(
        wo_a_weight,
        wo_a_scale,
        wo_b_weight,
        wo_b_scale,
        groups=groups,
        group_width=group_width,
        rank=rank,
        hidden=hidden,
    )
    assert weights.sfb_k_replicated

    x_tdg = quantize_wo_a_input_mxfp8(x_tgd)
    tmp_base = wo_a_dense_gemm_mxfp8(x_tdg, weights.wo_a)
    tmp_reuse = wo_a_dense_gemm_mxfp8(x_tdg, weights.wo_a, sfb_k_replicated=True)
    torch.cuda.synchronize()
    torch.testing.assert_close(tmp_reuse, tmp_base, rtol=0, atol=0)

    tmp_q = quantize_wo_b_input_mxfp8(tmp_base)
    out_base = wo_b_dense_gemm_mxfp8(tmp_q, weights.wo_b)
    out_reuse = wo_b_dense_gemm_mxfp8(tmp_q, weights.wo_b, sfb_k_replicated=True)
    torch.cuda.synchronize()
    torch.testing.assert_close(out_reuse, out_base, rtol=0, atol=0)


def _wo_b_fused_vs_unfused(
    tokens: int, rank: int, groups: int, hidden: int
) -> tuple[torch.Tensor, torch.Tensor]:
    from b12x.gemm._shared.wo_mxfp8 import wo_b_dense_gemm_fused_quant_mxfp8

    tmp = empty_dense_gemm_mnl_view(
        tokens, rank, groups, device="cuda", dtype=torch.bfloat16
    )
    tmp.copy_(
        torch.randn((tokens, rank, groups), device="cuda", dtype=torch.bfloat16) / 4
    )
    wo_b_bf16 = (
        torch.randn((hidden, groups * rank), device="cuda", dtype=torch.bfloat16)
        / (groups * rank) ** 0.5
    )
    wo_b = quantize_mxfp8_rows_torch(wo_b_bf16)

    tmp_q = quantize_wo_b_input_mxfp8(tmp)
    unfused = wo_b_dense_gemm_mxfp8(tmp_q, wo_b, expected_m=tokens)
    fused = wo_b_dense_gemm_fused_quant_mxfp8(tmp, wo_b, expected_m=tokens)
    torch.cuda.synchronize()
    return fused, unfused


def test_wo_b_fused_quant_matches_unfused_small_shapes() -> None:
    require_b12x()
    torch.manual_seed(31008)

    for tokens in (1, 3, 8):
        fused, unfused = _wo_b_fused_vs_unfused(tokens, rank=128, groups=2, hidden=256)
        torch.testing.assert_close(fused, unfused, rtol=0, atol=0)
    # Singleton group takes the contiguous-source path.
    fused, unfused = _wo_b_fused_vs_unfused(2, rank=256, groups=1, hidden=256)
    torch.testing.assert_close(fused, unfused, rtol=0, atol=0)


def test_wo_b_fused_quant_matches_unfused_split_k_serving_shape() -> None:
    require_b12x()
    torch.manual_seed(31009)

    # DS4-Flash TP2 WO-B: N=4096, K=4096 -> the decode policy picks 2-way
    # split-K; the fused path must produce byte-identical output through the
    # FP32-partials reduce.
    for tokens in (1, 4):
        fused, unfused = _wo_b_fused_vs_unfused(
            tokens, rank=1024, groups=4, hidden=4096
        )
        torch.testing.assert_close(fused, unfused, rtol=0, atol=0)


def test_wo_a_fused_quant_matches_unfused_small_shapes() -> None:
    require_b12x()
    torch.manual_seed(31010)
    from b12x.gemm._shared.wo_mxfp8 import wo_a_dense_gemm_fused_quant_mxfp8

    for tokens, groups, group_width, rank in (
        (1, 2, 256, 128),
        (3, 2, 256, 128),
        (8, 1, 512, 128),
    ):
        x_tgd = (
            torch.randn(
                (tokens, groups, group_width), device="cuda", dtype=torch.bfloat16
            )
            / 4
        )
        wo_a_grd = (
            torch.randn(
                (groups, rank, group_width), device="cuda", dtype=torch.bfloat16
            )
            / group_width**0.5
        )
        wo_a_source = wo_a_grd.permute(1, 2, 0).contiguous()
        if groups == 1:
            wo_a_source = wo_a_source[:, :, 0]
        wo_a = quantize_mxfp8_rows_torch(wo_a_source)

        x_q = quantize_wo_a_input_mxfp8(x_tgd)
        unfused = wo_a_dense_gemm_mxfp8(x_q, wo_a, expected_m=tokens)
        fused = wo_a_dense_gemm_fused_quant_mxfp8(x_tgd, wo_a, expected_m=tokens)
        torch.cuda.synchronize()
        torch.testing.assert_close(fused, unfused, rtol=0, atol=0)


def test_wo_a_fused_quant_inv_rope_matches_unfused() -> None:
    require_b12x()
    torch.manual_seed(31011)
    from b12x.gemm._shared.wo_mxfp8 import wo_a_dense_gemm_fused_quant_mxfp8

    tokens, groups, heads_per_group = 3, 2, 4
    head_dim, nope_dim, rope_dim = 128, 96, 32
    group_width = heads_per_group * head_dim
    rank = 128
    o = (
        torch.randn(
            (tokens, groups * heads_per_group, head_dim),
            device="cuda",
            dtype=torch.bfloat16,
        )
        / 8
    ).contiguous()
    positions = torch.arange(7, 7 + tokens, device="cuda", dtype=torch.int64)
    max_pos = 64
    inv_freq = 1.0 / (
        10000.0
        ** (torch.arange(0, rope_dim, 2, device="cuda", dtype=torch.float32) / rope_dim)
    )
    t = torch.arange(max_pos, device="cuda", dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    wo_a_grd = (
        torch.randn((groups, rank, group_width), device="cuda", dtype=torch.bfloat16)
        / group_width**0.5
    )
    wo_a = quantize_mxfp8_rows_torch(wo_a_grd.permute(1, 2, 0).contiguous())
    source_tgd = o.reshape(tokens, groups, group_width)

    for cos_sin_dtype in (torch.float32, torch.bfloat16):
        cos_sin = torch.cat([freqs.cos(), freqs.sin()], dim=-1).to(cos_sin_dtype)
        x_q = quantize_wo_a_input_inv_rope_mxfp8(
            o,
            positions,
            cos_sin,
            groups=groups,
            heads_per_group=heads_per_group,
            nope_dim=nope_dim,
            rope_dim=rope_dim,
        )
        unfused = wo_a_dense_gemm_mxfp8(x_q, wo_a, expected_m=tokens)
        fused = wo_a_dense_gemm_fused_quant_mxfp8(
            source_tgd,
            wo_a,
            positions=positions,
            cos_sin_cache=cos_sin,
            head_dim=head_dim,
            nope_dim=nope_dim,
            rope_dim=rope_dim,
            expected_m=tokens,
        )
        torch.cuda.synchronize()
        # The fused de-rotation reproduces the Triton quantizer's FP32 math
        # bit-for-bit (also verified at the DS4-Flash TP2 serving shape).
        torch.testing.assert_close(fused, unfused, rtol=0, atol=0)


def test_wo_inv_rope_route_fused_small_m_matches_reference_shapes(wo_session) -> None:
    require_b12x()
    torch.manual_seed(31012)

    # DS4-Flash TP2 serving shape at decode M: the fused route must replay
    # under graph capture and stay finite/consistent between eager and replay.
    tokens, groups, heads_per_group = 1, 4, 8
    head_dim, nope_dim, rope_dim = 512, 448, 64
    group_width = heads_per_group * head_dim
    rank, hidden = 1024, 4096
    o = (
        torch.randn(
            (tokens, groups * heads_per_group, head_dim),
            device="cuda",
            dtype=torch.bfloat16,
        )
        / 8
    ).contiguous()
    positions = torch.randint(0, 4096, (tokens,), device="cuda", dtype=torch.int64)
    inv_freq = 1.0 / (
        10000.0
        ** (torch.arange(0, rope_dim, 2, device="cuda", dtype=torch.float32) / rope_dim)
    )
    t = torch.arange(4096, device="cuda", dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    cos_sin = torch.cat([freqs.cos(), freqs.sin()], dim=-1).to(torch.bfloat16)
    wo_a_grd = (
        torch.randn((groups, rank, group_width), device="cuda", dtype=torch.bfloat16)
        / group_width**0.5
    )
    wo_b_hgr = (
        torch.randn((hidden, groups * rank), device="cuda", dtype=torch.bfloat16)
        / (groups * rank) ** 0.5
    )
    weights = quantize_wo_projection_weights_mxfp8_torch(wo_a_grd, wo_b_hgr)

    binding = _make_inv_rope_binding(o, positions, cos_sin, weights,
        session=wo_session, heads_per_group=heads_per_group, nope_dim=nope_dim, rope_dim=rope_dim)
    wo_session.freeze()
    def run():
        return wo.run_inv_rope(binding=binding, plan=binding.plan)

    eager = run().clone()
    torch.cuda.synchronize()
    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()

    assert bool(torch.isfinite(actual).all().item())
    torch.testing.assert_close(actual, eager, rtol=0, atol=0)
    graph.reset()


@pytest.mark.parametrize(
    "quant_path",
    ("grouped", "inv-rope-fp32", "inv-rope-bf16", "group-major"),
)
def test_wo_quant_cute_paths_match_triton_bit_exact(quant_path: str) -> None:
    """Give every compiled WO-quant specialization its own launch range.

    The grouped specializations intentionally share one generated CUDA symbol.
    Keeping them in separate parametrized tests lets the migration corpus bind
    each compile-spec hash to an unambiguous Nsight range instead of pretending
    that a symbol name alone identifies a specialization.
    """

    require_b12x()
    torch.manual_seed(31013)

    tokens, groups, heads_per_group = 130, 2, 4
    head_dim, nope_dim, rope_dim = 128, 96, 32
    group_width = heads_per_group * head_dim
    rank = 256

    def _assert_rows_equal(actual: MXFP8Rows, expected: MXFP8Rows) -> None:
        torch.testing.assert_close(
            actual.values.view(torch.uint8),
            expected.values.view(torch.uint8),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            actual.scale_rows.view(torch.uint8),
            expected.scale_rows.view(torch.uint8),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            actual.scale_mma.view(torch.uint8),
            expected.scale_mma.view(torch.uint8),
            rtol=0,
            atol=0,
        )

    from b12x.gemm.wo_projection._quant_cute import (
        quantize_wo_group_major_rows_cute,
        quantize_wo_grouped_rows_cute,
    )

    if quant_path == "group-major":
        source = empty_dense_gemm_mnl_view(
            tokens, rank, groups, device="cuda", dtype=torch.bfloat16
        )
        source.copy_(torch.randn_like(source) / 4)
        output = empty_mxfp8_rows_for_dense_gemm(
            tokens, rank * groups, num_groups=1, device="cuda"
        )

        def launch() -> None:
            quantize_wo_group_major_rows_cute(
                source,
                output.values,
                output.scale_rows.view(torch.uint8),
                output.scale_mma.view(torch.uint8),
                m=tokens,
                groups=groups,
                rank=rank,
            )

        def reference() -> MXFP8Rows:
            return quantize_wo_b_input_mxfp8(source)

    else:
        output = empty_mxfp8_rows_for_dense_gemm(
            tokens, group_width, num_groups=groups, device="cuda"
        )
        if quant_path == "grouped":
            source = (
                torch.randn(
                    (tokens, groups, group_width),
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                / 4
            ).contiguous()

            def launch() -> None:
                quantize_wo_grouped_rows_cute(
                    source.reshape(tokens, groups * group_width),
                    output.values,
                    output.scale_rows.view(torch.uint8),
                    output.scale_mma.view(torch.uint8),
                    m=tokens,
                    groups=groups,
                    group_width=group_width,
                )

            def reference() -> MXFP8Rows:
                return quantize_wo_a_input_mxfp8(source)

        else:
            source = (
                torch.randn(
                    (tokens, groups * heads_per_group, head_dim),
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                / 8
            ).contiguous()
            positions = torch.randint(
                0, 512, (tokens,), device="cuda", dtype=torch.int64
            )
            inv_freq = 1.0 / (
                10000.0
                ** (
                    torch.arange(0, rope_dim, 2, device="cuda", dtype=torch.float32)
                    / rope_dim
                )
            )
            freqs = torch.outer(
                torch.arange(512, device="cuda", dtype=torch.float32), inv_freq
            )
            cos_sin_dtype = (
                torch.float32 if quant_path == "inv-rope-fp32" else torch.bfloat16
            )
            cos_sin = torch.cat([freqs.cos(), freqs.sin()], dim=-1).to(cos_sin_dtype)

            def launch() -> None:
                quantize_wo_grouped_rows_cute(
                    source.reshape(tokens, groups * group_width),
                    output.values,
                    output.scale_rows.view(torch.uint8),
                    output.scale_mma.view(torch.uint8),
                    m=tokens,
                    groups=groups,
                    group_width=group_width,
                    positions=positions,
                    cos_sin_cache=cos_sin,
                    head_dim=head_dim,
                    nope_dim=nope_dim,
                    rope_dim=rope_dim,
                )

            def reference() -> MXFP8Rows:
                return quantize_wo_a_input_inv_rope_mxfp8(
                    source,
                    positions,
                    cos_sin,
                    groups=groups,
                    heads_per_group=heads_per_group,
                    nope_dim=nope_dim,
                    rope_dim=rope_dim,
                )

    launch()
    torch.cuda.synchronize()
    _assert_rows_equal(output, reference())

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()

    # Prove replay reads live inputs while preserving every captured address.
    source.copy_(torch.randn_like(source) / 7)
    graph.replay()
    torch.cuda.synchronize()
    _assert_rows_equal(output, reference())
