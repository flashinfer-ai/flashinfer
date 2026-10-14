"""Host-side checks of the NVFP4-transport all-reduce delivery: window plans, program tables, codec and fused-norm references (no GPU)."""

from __future__ import annotations

import importlib
import math
from fractions import Fraction

import pytest
import torch

from flashinfer.jit import fp4_allreduce as loader
from tests.comm import fp4_allreduce_reference as codec

EXPECTED_REGION_BYTES = {
    ("allreduce", 2): (341_311_488, 339_738_624),
    ("allreduce", 4): (341_311_488,),
    ("rmsnorm", 2): (341_311_488,),
}
FAMILY_WORLD_SIZES = [("allreduce", 2), ("allreduce", 4), ("rmsnorm", 2)]


def _api():
    # ``flashinfer.comm`` re-exports the ``fp4_allreduce`` function under the module's name.
    return importlib.import_module("flashinfer.comm.fp4_allreduce")


@pytest.mark.parametrize(("family", "world_size"), FAMILY_WORLD_SIZES)
def test_window_plan_matches_the_binding_contract(family: str, world_size: int) -> None:
    api = _api()
    plan = api.window_plan(world_size, family=family)
    table = loader.program_table(world_size, family)
    assert plan.family == family
    assert plan.table_entries == table["table_entries"]
    assert plan.ctrl_words == table["ctrl_words"]
    assert plan.row_multiple == table["row_multiple"]
    assert (
        tuple(region.bytes for region in plan.regions)
        == EXPECTED_REGION_BYTES[(family, world_size)]
    )
    for region in plan.regions:
        assert region.bytes % api.PAGE_BYTES == 0
        words = region.bytes // 4
        end = 0
        for fill in region.fills:
            assert fill.words > 0 and fill.offset >= end, (
                "fills must be ordered and disjoint"
            )
            assert fill.offset + fill.words <= words, "a fill runs past its region"
            if fill.period:
                assert 0 < fill.split < fill.period and fill.words % fill.period == 0
            end = fill.offset + fill.words
    if family == "rmsnorm":
        (window,) = plan.regions
        assert window.name == "rmsnorm" and len(window.fills) == 9
        # The fused-norm window is the one-shot window: three buffers of 2 * capacity slot rows.
        oneshot = api.window_plan(2).regions[0]
        assert window.bytes == oneshot.bytes and window.fills == oneshot.fills
        assert (
            plan.row_multiple == 4 and plan.ctrl_words == 4 and plan.table_entries == 2
        )
    elif world_size == 2:
        oneshot, records = plan.regions
        assert (oneshot.name, records.name) == ("oneshot", "nvfp4")
        assert len(oneshot.fills) == 9 and len(records.fills) == 3
        assert {fill.period for fill in records.fills} == {codec.RECORD_BYTES // 4}
        assert {fill.split for fill in records.fills} == {codec.VALUE_BYTES // 4}
        # Each rotation buffer of the one-shot window holds 2 * capacity slot rows of 3472 bytes.
        assert oneshot.fills[0].words == 2 * loader.CAPACITY * codec.VALUE_BYTES // 4
    else:
        (twoshot,) = plan.regions
        assert twoshot.name == "twoshot" and len(twoshot.fills) == 18
        owner_rows = -(-loader.CAPACITY // 4)
        assert twoshot.fills[0].words == 4 * owner_rows * codec.VALUE_BYTES // 4


def test_window_plan_rejects_undeliverable_requests() -> None:
    api = _api()
    with pytest.raises(ValueError):
        api.window_plan(8)
    with pytest.raises(ValueError):
        api.window_plan(2, capacity=1024)
    with pytest.raises(ValueError):
        api.window_plan(2, family="layernorm")
    with pytest.raises(ValueError):
        api.window_plan(4, family="rmsnorm")


@pytest.mark.parametrize("world_size", [2, 4])
def test_program_table_lists_every_generated_program(world_size: int) -> None:
    table = loader.program_table(world_size)
    variants = 1 if world_size == 2 else 3
    assert table["family"] == "allreduce"
    assert table["variants"] == variants
    kernels = table["kernels"]
    assert len(kernels) == 2 * variants * world_size
    symbols = [kernel["symbol"] for kernel in kernels]
    assert len(set(symbols)) == len(symbols)
    source_dir = loader._source_dir()
    binding = (source_dir / table["binding"]).read_text()
    for kernel in kernels:
        assert kernel["input_format"] in loader.INPUT_FORMATS
        assert 0 <= kernel["rank"] < world_size and 0 <= kernel["variant"] < variants
        assert kernel["symbol"].endswith(f"_r{kernel['rank']}_cap{loader.CAPACITY}")
        source = (source_dir / kernel["source"]).read_text()
        assert f"{kernel['symbol']}(" in source
        assert f'extern "C" __global__ void {kernel["symbol"]}(' in binding
    assert set(symbols) == set(loader.kernel_names(world_size))
    assert (
        loader.kernel_name(world_size, "nvfp4", world_size - 1, variants - 1) in symbols
    )
    with pytest.raises(KeyError):
        loader.kernel_name(world_size, "bf16", world_size, 0)


def test_rmsnorm_program_table_lists_every_generated_program() -> None:
    table = loader.program_table(2, "rmsnorm")
    assert table["family"] == "rmsnorm" and table["variants"] == 1
    assert table["variant_rule"] is None
    formats = {tuple(pair) for pair in table["formats"]}
    assert formats == {
        ("bf16", "bf16"),
        ("nvfp4", "bf16"),
        ("bf16", "nvfp4"),
        ("bf16", "mxfp8"),
    }
    kernels = table["kernels"]
    assert len(kernels) == 2 * len(formats)
    symbols = [kernel["symbol"] for kernel in kernels]
    assert len(set(symbols)) == len(symbols)
    assert {(k["input_format"], k["output_format"]) for k in kernels} == formats
    source_dir = loader._source_dir()
    binding = (source_dir / table["binding"]).read_text()
    assert table["binding"] != loader.program_table(2)["binding"]
    for kernel in kernels:
        assert kernel["symbol"].endswith(f"_r{kernel['rank']}_cap{loader.CAPACITY}")
        assert kernel["variant"] == 0 and 0 <= kernel["rank"] < 2
        source = (source_dir / kernel["source"]).read_text()
        assert f"{kernel['symbol']}(" in source
        assert f'extern "C" __global__ void {kernel["symbol"]}(' in binding
    assert set(symbols) == set(loader.kernel_names(2, "rmsnorm"))
    assert set(loader.kernel_names(family="rmsnorm")) == set(symbols)
    assert loader.kernel_name(2, "bf16", 1, 0, "rmsnorm", "mxfp8") in symbols
    assert loader.kernel_name(2, "nvfp4", 0, 0, "rmsnorm") in symbols
    with pytest.raises(KeyError):
        loader.kernel_name(2, "nvfp4", 0, 0, "rmsnorm", "mxfp8")
    with pytest.raises(ValueError):
        loader.program_table(4, "rmsnorm")
    with pytest.raises(ValueError):
        loader.program_table(2, "layernorm")
    assert loader.world_sizes("rmsnorm") == (2,) and loader.world_sizes() == (2, 4)
    # Launch geometry per input format, carried by the binding as well.
    assert (
        set(table["block"])
        == set(table["dynamic_smem_bytes"])
        == set(loader.INPUT_FORMATS)
    )
    for fmt in loader.INPUT_FORMATS:
        assert table["block"][fmt] % 32 == 0 and 0 < table["block"][fmt] <= 1024
        assert 0 < table["dynamic_smem_bytes"][fmt] <= 227 * 1024
    assert (
        f"kThreads[kInputFormats] = {{{table['block']['bf16']}, {table['block']['nvfp4']}}};"
        in binding
    )
    assert (
        "kDynamicSmemBytes[kInputFormats] = "
        f"{{{table['dynamic_smem_bytes']['bf16']}, {table['dynamic_smem_bytes']['nvfp4']}}};"
        in binding
    )
    assert table["eps"] == loader.RMSNORM_EPS == 1e-6
    assert table["weight_bias"] == loader.RMSNORM_WEIGHT_BIAS == 1.0
    assert (
        table["row_multiple"] == 4
        and table["ctrl_words"] == 4
        and table["table_entries"] == 2
    )
    assert table["nvfp4_value_bytes"] == codec.VALUE_BYTES
    assert table["nvfp4_scales_per_row"] == codec.NVFP4_OUTPUT_SCALES_PER_ROW
    assert table["mxfp8_value_bytes"] == codec.HIDDEN
    assert table["mxfp8_scales_per_row"] == codec.MXFP8_OUTPUT_SCALES_PER_ROW


@pytest.mark.parametrize(
    ("rows", "grid", "expected"),
    [
        (64, 148, 1),
        (1024, 148, 1),
        (4096, 148, 1),
        (148 * 40, 148, 1),
        (148 * 41, 148, 0),
        (8192, 148, 0),
        (148 * 64, 148, 0),
        (148 * 65, 148, 2),
        (16384, 148, 2),
    ],
)
def test_phase_b_variant_rule(rows: int, grid: int, expected: int) -> None:
    assert loader.phase_b_variant(rows, grid, 4) == expected
    assert loader.phase_b_variant(rows, grid, 2) == 0
    assert loader.phase_b_variant(rows, grid, 2, "rmsnorm") == 0


def test_target_arch_routes_sm100_sm103_sm107() -> None:
    assert loader._ARCH_BY_CAPABILITY == {
        (10, 0): "sm_100a",
        (10, 3): "sm_103a",
        (10, 7): "sm_107a",
    }
    assert set(loader._NVCC_FLAGS) == {"sm_100a", "sm_103a", "sm_107a"}


def test_e2m1_rounding_table() -> None:
    values = torch.tensor(
        [
            0.0,
            0.24,
            0.25,
            0.26,
            0.75,
            1.0,
            1.25,
            1.26,
            1.75,
            2.5,
            2.51,
            3.5,
            5.0,
            5.01,
            6.0,
            7.0,
            -0.1,
            -0.3,
            math.nan,
        ],
        dtype=torch.float32,
    )
    codes = codec.e2m1_rn_satfinite(values)
    assert codes.tolist() == [0, 0, 0, 1, 2, 2, 2, 3, 4, 4, 5, 6, 6, 7, 7, 7, 8, 9, 7]
    assert codec.canon_neg0_codes(codes).tolist()[16] == 0
    assert codec.e2m1_to_f32(torch.arange(16, dtype=torch.uint8)).tolist() == list(
        codec._E2M1_VALUES
    )


def test_e4m3_saturates_and_keeps_nan() -> None:
    values = torch.tensor(
        [448.0, 449.0, 1e9, -1e9, 0.0, 0.1, math.nan], dtype=torch.float32
    )
    codes = codec.e4m3_rn_satfinite(values)
    decoded = codec.e4m3_to_f32(codes)
    assert decoded[:5].tolist() == [448.0, 448.0, 448.0, -448.0, 0.0]
    assert abs(decoded[5].item() - 0.1) <= 0.1 / 8
    assert codes[6].item() == 0x7F


def test_fma_is_correctly_rounded() -> None:
    generator = torch.Generator().manual_seed(7)
    a = torch.randn(256, generator=generator) * 1e3
    b = torch.randn(256, generator=generator)
    c = torch.randn(256, generator=generator) * 1e-3
    result = codec.f32_fma_rn(a, b, c)
    up = torch.nextafter(result, torch.full_like(result, math.inf))
    down = torch.nextafter(result, torch.full_like(result, -math.inf))
    for i in range(a.numel()):
        exact = Fraction(a[i].item()) * Fraction(b[i].item()) + Fraction(c[i].item())
        error = abs(exact - Fraction(result[i].item()))
        assert error <= abs(exact - Fraction(up[i].item()))
        assert error <= abs(exact - Fraction(down[i].item()))


def test_pack_unpack_roundtrip() -> None:
    codes = (
        (torch.arange(2 * codec.HIDDEN) % 16).to(torch.uint8).reshape(2, codec.HIDDEN)
    )
    packed = codec.pack_e2m1_codes(codes)
    assert packed.shape == (2, codec.VALUE_BYTES)
    assert torch.equal(codec.unpack_e2m1_bytes(packed), codes)


def test_codec_reproduces_an_exact_case() -> None:
    x = torch.full((2, codec.HIDDEN), 2688.0, dtype=torch.bfloat16)
    assert torch.equal(codec.reference_allreduce([x, x]), torch.full_like(x, 5376.0))
    assert torch.equal(
        codec.reference_allreduce([x, x, x, x]), torch.full_like(x, 10752.0)
    )
    with pytest.raises(ValueError):
        codec.reference_allreduce([x, x, x])


def test_generate_rank_input_is_deterministic_with_outliers() -> None:
    x = codec.generate_rank_input(8, 1, 123, "cpu")
    assert x.shape == (8, codec.HIDDEN) and x.dtype == torch.bfloat16
    assert torch.equal(x, codec.generate_rank_input(8, 1, 123, "cpu"))
    assert not torch.equal(x, codec.generate_rank_input(8, 0, 123, "cpu"))
    assert int((x.abs() == 40.0 * codec.SOURCE_MAGNITUDES[1]).sum().item()) >= 30
    plain = codec.generate_rank_input(8, 1, 123, "cpu", outliers=False)
    assert int((plain.abs() == 40.0 * codec.SOURCE_MAGNITUDES[1]).sum().item()) == 0
    assert int((plain != x).sum().item()) <= codec.OUTLIER_COUNT


def test_nvfp4_records_layout() -> None:
    x = codec.generate_rank_input(4, 0, 7, "cpu")
    scale = codec.tensor_global_scale(x)
    assert scale > 0
    records = codec.encode_nvfp4_records(x, scale)
    assert records.shape == (4, codec.RECORD_BYTES) and records.dtype == torch.uint8
    values, scales = codec.split_nvfp4_records(records)
    assert values.shape == (4, codec.VALUE_BYTES) and scales.shape == (
        4,
        codec.SCALE_BYTES,
    )
    out = codec.reference_allreduce_nvfp4([records, records], [scale, scale])
    assert out.shape == (4, codec.HIDDEN) and out.dtype == torch.bfloat16
    assert torch.isfinite(out.float()).all()


def test_fused_norm_entry_point_is_typed() -> None:
    api = _api()
    spec = api.Fp4AllReduceFusedNormSpec(
        weight=torch.ones((codec.HIDDEN,)),
        output=api.Fp4AllReduceFusedNormOutput.MXFP8,
    )
    assert spec.output.value == "mxfp8" and spec.eps == loader.RMSNORM_EPS
    assert spec.global_scale is None
    assert {member.value for member in api.Fp4AllReduceFusedNormOutput} == {
        "bf16",
        "nvfp4",
        "mxfp8",
    }
    assert api.delivered_families() == ("allreduce", "rmsnorm")
    result = api.Fp4AllReduceFusedNormResult(
        output=api.Fp4AllReduceFusedNormOutput.BF16,
        residual_out=torch.zeros(1),
        out=torch.zeros(1),
    )
    assert result.values is None and result.scales is None
    assert api.padded_scale_rows(1) == 128 and api.padded_scale_rows(1024) == 1024
    assert (
        api.fused_norm_scale_bytes(1000, api.Fp4AllReduceFusedNormOutput.NVFP4)
        == 1024 * 384
    )
    assert (
        api.fused_norm_scale_bytes(4096, api.Fp4AllReduceFusedNormOutput.MXFP8)
        == 4096 * 192
    )
    with pytest.raises(ValueError):
        api.fused_norm_scale_bytes(64, api.Fp4AllReduceFusedNormOutput.BF16)
    with pytest.raises(TypeError):
        api.fp4_allreduce_rmsnorm(torch.zeros(1), torch.zeros(1), spec, object())
    with pytest.raises(TypeError):
        api.fp4_allreduce_rmsnorm_nvfp4(
            torch.zeros(1), torch.zeros(1), torch.zeros(1), spec, object()
        )


def test_fused_norm_reference_on_cpu() -> None:
    rows = 64
    seed = 321
    partials = [codec.generate_rank_input(rows, rank, seed, "cpu") for rank in range(2)]
    residual = codec.generate_residual(rows, seed, "cpu")
    weight = codec.generate_weight(seed, "cpu")
    assert residual.shape == (rows, codec.HIDDEN) and weight.shape == (codec.HIDDEN,)
    assert torch.equal(residual, codec.generate_residual(rows, seed, "cpu"))
    assert float(weight.float().abs().max()) < 1.0
    ref = codec.reference_fused_rmsnorm(partials, residual, weight)
    assert ref.rsqrt_form == codec.RSQRT_FORM_IEEE
    assert ref.y_f32.shape == (rows, codec.HIDDEN) and ref.rinv.shape == (rows,)
    assert torch.isfinite(ref.y_f32).all()
    # residual_out is bf16(z) of the float32 codec sum plus the residual, element by element.
    acc = codec._codec_sum(partials)
    z = codec.f32_add_rn(acc, residual.to(torch.float32))
    assert torch.equal(ref.residual_out, z.to(torch.bfloat16))
    # The row norm: mean of z^2 over the row within a few float32 ulps of the exact value.
    exact = (z.to(torch.float64) ** 2).sum(dim=1)
    assert torch.allclose(ref.ss.to(torch.float64), exact, rtol=1e-6, atol=0.0)
    # The reference agrees with itself and flags a wrong weight bias.
    report = codec.check_normalised_output(ref.y_bf16, ref)
    assert report["passed"] and report["max_ulp"] == 0
    biased = codec.reference_fused_rmsnorm(
        partials, residual, weight, weight_bias=codec.NEGATIVE_CONTROL_WEIGHT_BIAS
    )
    assert torch.equal(biased.residual_out, ref.residual_out)
    assert codec.check_normalised_output(ref.y_bf16, biased)["violations"] > 0
    # NVFP4 records of both ranks give the records reference, with the same residual path.
    scales = [codec.tensor_global_scale(x) for x in partials]
    records = [
        codec.encode_nvfp4_records(x, s) for x, s in zip(partials, scales, strict=True)
    ]
    ref_records = codec.reference_fused_rmsnorm(
        residual_bf16=residual,
        weight_bf16=weight,
        records_by_rank=records,
        global_scales=scales,
    )
    assert ref_records.y_f32.shape == ref.y_f32.shape
    with pytest.raises(ValueError):
        codec.reference_fused_rmsnorm(partials + partials, residual, weight)
    with pytest.raises(ValueError):
        codec.reference_fused_rmsnorm(partials, residual, weight[:-1])


def test_bf16_ulp_distance_and_global_scale() -> None:
    a = torch.tensor([1.0, -1.0, 0.0, 2.0], dtype=torch.bfloat16)
    # One bfloat16 step away from zero for positive values, toward zero for negative ones (sign-magnitude bits).
    bits = a.view(torch.int16).to(torch.int32)
    neighbour = (bits + torch.where(a < 0, -1, 1)).to(torch.int16).view(torch.bfloat16)
    assert codec.bf16_ulp_distance(a, a).tolist() == [0, 0, 0, 0]
    assert codec.bf16_ulp_distance(a, neighbour).tolist() == [1, 1, 1, 1]
    assert codec.bf16_ulp_distance(
        torch.tensor([0.0], dtype=torch.bfloat16),
        torch.tensor([-0.0], dtype=torch.bfloat16),
    ).tolist() == [0]
    y = torch.full((2, codec.HIDDEN), 3.0, dtype=torch.bfloat16)
    y[0, 0] = 24.0
    scale = codec.output_global_scale(y)
    assert scale.dtype == torch.float32 and tuple(scale.shape) == (1,)
    assert scale.item() == 2688.0 / 24.0


def test_unswizzle_scales_128x4_matches_the_offset_rule() -> None:
    rows, per_row = 200, 384
    padded = codec.padded_scale_rows(rows)
    assert padded == 256
    kt = per_row // 4
    flat = torch.zeros((padded * per_row,), dtype=torch.uint8)
    # Mark three scales through the offset rule and read them back in block order.
    marks = [(0, 0, 11), (37, 5, 22), (rows - 1, per_row - 1, 33)]
    for m, k, value in marks:
        offset = (
            (m // 128) * kt * 512
            + (k // 4) * 512
            + (m % 32) * 16
            + ((m % 128) // 32) * 4
            + (k % 4)
        )
        flat[offset] = value
    block_order = codec.unswizzle_scales_128x4(flat, rows, per_row)
    assert block_order.shape == (rows, per_row)
    for m, k, value in marks:
        assert int(block_order[m, k]) == value
    assert int(block_order.sum()) == sum(value for _, _, value in marks)
    with pytest.raises(ValueError):
        codec.unswizzle_scales_128x4(flat[:-1], rows, per_row)
