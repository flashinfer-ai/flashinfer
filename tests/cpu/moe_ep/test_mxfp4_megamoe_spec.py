# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Device-free conformance of the pure/mixed MXFP4 MegaMoE classes."""

import tracemalloc

import pytest

from flashinfer.batch_invariant import class_name, fingerprint
from flashinfer.batch_invariant.fp_interface import term
from flashinfer.batch_invariant.fp_interface.term import sexpr
from flashinfer.moe_ep.batch_invariant import mxfp4_spec as spec

pytestmark = pytest.mark.cpu_only

CLASSES = tuple(
    (arch, kind, provider)
    for arch, kind in (
        ("sm_100", "mxfp4_mxfp8"),
        ("sm_103", "mxfp4_mxfp8"),
        ("sm_107", "mxfp4_mxfp8"),
        ("sm_107", "mxfp4"),
    )
    for provider in ("standard", "genphase")
)
SHAPE = dict(hidden=256, intermediate=256, top_k=3, num_experts=4)
GOLDEN = (
    "megamoe-mxfp4-6b8c5ec3e767d110",
    "megamoe-mxfp4-106e805da60e707c",
    "megamoe-mxfp4-2d61281c8d7f06f8",
    "megamoe-mxfp4-aee9d55651fe1b92",
    "megamoe-mxfp4-a89b13a7a890085e",
    "megamoe-mxfp4-0823bbee93bb8619",
    "megamoe-mxfp4-557b3a47edd82b1d",
    "megamoe-mxfp4-37ed39d8e175efd2",
)


@pytest.mark.parametrize("arch,kind,provider", CLASSES)
@pytest.mark.parametrize("k", (0, 1, 32, 64, 128, 192, 256, 384))
def test_mma_native_step_initialization_and_padding(arch, kind, provider, k):
    fp = spec.OrderedTermAlgebra()
    cls = spec.numerical_class(arch, kind, provider)
    kwargs = dict(arch=arch, quant_kind=kind, implementation=provider)
    flat = spec.mma_fold(fp, k, "boundary", **kwargs)
    assert flat is spec.mma_fold(fp, k, "boundary", blocked=True, **kwargs)
    text = sexpr(flat)
    steps = ((k + cls.stage_k - 1) // cls.stage_k) * cls.stage_k // cls.instruction_k
    assert text.count("op:mma.init.") == bool(k)
    assert text.count("op:mma.acc.") == max(0, steps - 1)
    assert text.count(f"op:operand.{kind}.k{cls.instruction_k}") == steps
    assert "bf16.k16" not in text and "op:mma.init.nvfp4." not in text
    if k == 1:
        assert f"boundary {steps - 1} 0 weight.e2m1.ue8m0.block32" in text
    if kind == "mxfp4" and k:
        assert "mxf4nvf4.block32.k128" in text


def test_paired_amax_keeps_logical_half_order_and_input_association():
    fp = spec.OrderedTermAlgebra()
    values = tuple(fp.op("x", i) for i in range(32))
    low, high = spec.half_amax(fp, values[:16]), spec.half_amax(fp, values[16:])
    combined = spec.combine_half_amax(fp, low, high)
    assert combined is spec.block_amax(fp, values)
    assert combined is not spec.combine_half_amax(fp, high, low)
    text = sexpr(combined)
    assert text.count("op:max.xorsign.abs.f32") == 30
    assert text.count("op:abs.bits.f32") == 2
    assert text.count("op:max.f32") == 1
    pure = spec.input_block_amax(fp, values, "mxfp4")
    mixed = spec.input_block_amax(fp, values, "mxfp4_mxfp8")
    assert pure is not mixed
    assert sexpr(pure).count("op:max.f32") == 64
    assert sexpr(mixed).count("op:max.f32") == 65
    assert "max.xorsign.abs" not in sexpr(pure)


@pytest.mark.parametrize("kind", ("mxfp4", "mxfp4_mxfp8"))
def test_quantization_scale_roundtrip_normal_floor_and_soft_mask_are_distinct(kind):
    fp = spec.OrderedTermAlgebra()
    values = tuple(fp.op("x", i) for i in range(32))
    amax = fp.op("amax")
    normal = spec.quantization_scale_value(fp, amax, kind, staging=True)
    fc1 = spec.quantization_scale_value(fp, amax, kind)
    for value in (normal, fc1):
        text = sexpr(value)
        assert "cvt.rp.satfinite.ue8m0.f32" in text
        assert "cvt.rn.bf16.ue8m0" in text and "widen.bf16.f32" in text
        assert ("3e2aaaab" if kind == "mxfp4" else "3b124925") in text
    assert "00800000" in sexpr(normal) and "00800000" not in sexpr(fc1)
    direct = spec.quantization_encoding_scale(fp, normal, staging=True)
    masked = spec.quantization_encoding_scale(fp, fc1)
    assert "7149f2ca" not in sexpr(direct)
    assert "7149f2ca" in sexpr(masked)
    assert "div" not in sexpr(masked)
    scaled, scale = spec.quantize_block_precast(fp, values, kind)
    low, high = spec.half_amax(fp, values[:16]), spec.half_amax(fp, values[16:])
    lo, sf0 = spec.quantize_half_precast(fp, values[:16], low, high, quant_kind=kind)
    hi, sf1 = spec.quantize_half_precast(fp, values[16:], low, high, quant_kind=kind)
    assert scaled == lo + hi and scale is sf0 is sf1
    assert scale.op == "op:cvt.rp.satfinite.ue8m0.f32"


@pytest.mark.parametrize("early", (False, True))
@pytest.mark.parametrize("arch,kind,provider", CLASSES)
def test_full_composition_blocked_lowering_and_ordered_hash(
    arch, kind, provider, early
):
    kwargs = dict(
        **SHAPE,
        arch=arch,
        quant_kind=kind,
        implementation=provider,
        apply_routing_weights_before_fc2=early,
        gate_up_clamp=-2,
        valid_routes=(True, False, False),
    )
    terms = spec.component_terms(**kwargs)
    blocked = spec.component_terms(**kwargs, blocked=True)
    assert len(terms) == 15
    assert all(blocked[name] is value for name, value in terms.items())
    for value in terms.values():
        assert spec._fingerprint_component(value) == term.fingerprint(value)
        assert "fc1_alpha" not in sexpr(value) and "fc2_alpha" not in sexpr(value)
    final = sexpr(terms["finalize"])
    assert final.count("op:add.rn.f32" if early else "op:fma.rn.f32") == 3
    assert final.count("op:bf16_bits 0000") == 2
    assert terms["fc2"].op == "op:cvt.rn.bf16.f32"


@pytest.mark.parametrize("arch,kind,provider", CLASSES)
def test_public_registry_and_arithmetic_dimensions_are_isolated(arch, kind, provider):
    kwargs = dict(**SHAPE, arch=arch, quant_kind=kind, implementation=provider)
    receipt = spec.spec_fingerprint(**kwargs)
    assert receipt == GOLDEN[CLASSES.index((arch, kind, provider))]
    assert fingerprint("megamoe_mxfp4", **kwargs) == receipt
    assert class_name("megamoe_mxfp4") == "megamoe.mxfp4"
    if kind == "mxfp4_mxfp8" and provider == "standard":
        assert fingerprint("megamoe_mxfp4", **SHAPE, arch=arch) == receipt
    for change in (
        dict(apply_routing_weights_before_fc2=False),
        dict(gate_up_clamp=0),
        dict(hidden=384),
        dict(intermediate=384),
        dict(top_k=2),
        dict(num_experts=8),
    ):
        assert spec.spec_fingerprint(**(kwargs | change)) != receipt
    assert spec.spec_fingerprint(
        **(kwargs | dict(gate_up_clamp=-2))
    ) == spec.spec_fingerprint(**(kwargs | dict(gate_up_clamp=2)))


def test_architecture_kind_and_provider_all_enter_receipt():
    receipts = {
        spec.spec_fingerprint(
            **SHAPE, arch=arch, quant_kind=kind, implementation=provider
        )
        for arch, kind, provider in CLASSES
    }
    assert len(receipts) == 8


def test_runtime_hash_rejects_shared_serializer_aliases():
    fp = spec.OrderedTermAlgebra()
    # The shared serializer aliases this operator. The local incremental hash
    # supports only the declared generic ordered atoms, without changing it.
    with pytest.raises(ValueError, match="ordered primitives"):
        spec._fingerprint_component(fp.op("mma_nvfp4_k64", fp.const(0)))


def test_large_runtime_receipt_has_bounded_memory_and_no_global_retention():
    before = len(term._TABLE)
    tracemalloc.start()
    receipt = spec.spec_fingerprint(65536, 65536, 32, 256, arch="sm_100")
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert receipt.startswith("megamoe-mxfp4-")
    assert len(term._TABLE) == before
    assert peak < 16 * 1024**2


@pytest.mark.parametrize(
    "changes",
    (
        dict(hidden=64),
        dict(hidden=0),
        dict(intermediate=64),
        dict(top_k=0),
        dict(num_experts=0),
        dict(arch="sm_90"),
        dict(quant_kind="mxfp4"),
        dict(quant_kind="nvfp4"),
        dict(implementation="legacy"),
        dict(apply_routing_weights_before_fc2=1),
    ),
)
def test_invalid_numerical_classes_are_rejected(changes):
    with pytest.raises(ValueError):
        spec.spec_fingerprint(**(SHAPE | dict(arch="sm_100") | changes))
