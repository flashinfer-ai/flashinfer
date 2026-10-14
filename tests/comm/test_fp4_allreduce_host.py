"""Host-side checks of the NVFP4-transport all-reduce delivery: window plans, program table, codec reference (no GPU)."""

from __future__ import annotations

import importlib
import math
from fractions import Fraction

import pytest
import torch

from flashinfer.jit import fp4_allreduce as loader
from tests.comm import fp4_allreduce_reference as codec

EXPECTED_REGION_BYTES = {2: (341_311_488, 339_738_624), 4: (341_311_488,)}


def _api():
    # ``flashinfer.comm`` re-exports the ``fp4_allreduce`` function under the module's name.
    return importlib.import_module("flashinfer.comm.fp4_allreduce")


@pytest.mark.parametrize("world_size", [2, 4])
def test_window_plan_matches_the_binding_contract(world_size: int) -> None:
    api = _api()
    plan = api.window_plan(world_size)
    table = loader.program_table(world_size)
    assert plan.table_entries == table["table_entries"]
    assert plan.ctrl_words == table["ctrl_words"]
    assert plan.row_multiple == table["row_multiple"]
    assert (
        tuple(region.bytes for region in plan.regions)
        == EXPECTED_REGION_BYTES[world_size]
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
    if world_size == 2:
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
        api.window_plan(2, family="rmsnorm")


@pytest.mark.parametrize("world_size", [2, 4])
def test_program_table_lists_every_generated_program(world_size: int) -> None:
    table = loader.program_table(world_size)
    variants = 1 if world_size == 2 else 3
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
        eps=1e-6,
        output=api.Fp4AllReduceFusedNormOutput.MXFP8,
    )
    assert spec.output.value == "mxfp8"
    assert {member.value for member in api.Fp4AllReduceFusedNormOutput} == {
        "bf16",
        "nvfp4",
        "mxfp8",
    }
    assert api.delivered_families() == ("allreduce",)
    with pytest.raises(TypeError):
        api.fp4_allreduce_rmsnorm(torch.zeros(1), torch.zeros(1), spec, object())
