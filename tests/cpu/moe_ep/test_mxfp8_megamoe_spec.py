# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Host conformance of E4M3/E5M2 MegaMoE arithmetic and receipts."""

import tracemalloc

import pytest

from flashinfer.batch_invariant import fingerprint, class_name
from flashinfer.batch_invariant.fp_interface import term
from flashinfer.batch_invariant.fp_interface.term import sexpr
from flashinfer.moe_ep.batch_invariant import mxfp8_spec as spec

pytestmark = pytest.mark.cpu_only
SHAPE = dict(hidden=256, intermediate=256, top_k=3, num_experts=4)
CLASSES = tuple(
    (a, p, k)
    for a in ("sm_100", "sm_103", "sm_107")
    for p in ("legacy", "standard", "genphase")
    for k in spec.KINDS
    if (a, p) != ("sm_107", "legacy")
)
GOLDEN = (
    "fea4671bbdf7cd20",
    "722f496aa889b500",
    "afe194d9d256f5e9",
    "cbc7be0c55fc8936",
    "82a58012463b1333",
    "b56df098589b2dcb",
    "fe8c332e765a062c",
    "ffc3352117a115db",
    "0bd30ebfe192e065",
    "f74cecb86d9be110",
    "aea481c6f8e65a3b",
    "953221b48a9e7cac",
    "c3235421adf6e226",
    "a64f0fd46fc946e8",
    "159a26a0f8cbc532",
    "f891cbef7895999a",
)


@pytest.mark.parametrize("arch,provider,kind", CLASSES)
@pytest.mark.parametrize("k", (0, 1, 32, 64, 128, 192, 256))
def test_native_mma_steps_padding_and_operand_orientation(arch, provider, kind, k):
    fp = spec.OrderedTermAlgebra()
    cls = spec.numerical_class(arch, kind, provider)
    kwargs = dict(arch=arch, quant_kind=kind, implementation=provider)
    flat = spec.mma_fold(fp, k, "input_w13.gate", **kwargs)
    assert flat is spec.mma_fold(fp, k, "input_w13.gate", blocked=True, **kwargs)
    text = sexpr(flat)
    steps = ((k + 127) // 128) * 128 // cls.instruction_k
    assert text.count("op:mma.init.") == bool(k)
    assert text.count("op:mma.acc.") == max(0, steps - 1)
    assert text.count("op:operand.") == steps
    if k:
        assert (
            "A.activation_B.weight" if provider == "legacy" else "A.weight_B.activation"
        ) in text
        assert f"m256n{256 if provider == 'legacy' else 128}.cta2" in text
        assert "A.weight_B.activation" in sexpr(
            spec.mma_fold(fp, k, "handoff_w2", **kwargs)
        )
        assert cls.instruction_k == (64 if arch == "sm_107" else 32)
    if k == 1:
        assert f"input_w13.gate {steps - 1} 0 " in text


def test_legacy_activation_keeps_its_distinct_association_and_nan_clamp():
    fp = spec.OrderedTermAlgebra()
    gate, up, weight = (fp.op("input", n) for n in ("gate", "up", "weight"))
    legacy = spec.legacy_fc1_activation(fp, gate, up, weight=weight, clamp=2)
    next_ = spec.next_fc1_activation(fp, gate, up, weight=weight, clamp=2)
    assert legacy is not next_
    text = sexpr(legacy)
    assert "min.NaN.f32" in text and "max.NaN.f32" in text
    assert "min.xorsign.abs.f32" not in text
    assert legacy.args[1] is weight
    assert legacy.args[0].args[1].op == "op:max.NaN.f32"
    assert spec.legacy_fc1_activation(
        fp, gate, up, clamp=-2
    ) is spec.legacy_fc1_activation(fp, gate, up, clamp=2)


@pytest.mark.parametrize(
    "kind,constant", (("mxfp8_e4m3", "3b124925"), ("mxfp8_e5m2", "37924925"))
)
def test_legacy_max_tree_norm_one_and_next_mask_remain_separate(kind, constant):
    fp = spec.OrderedTermAlgebra()
    values = tuple(fp.op("x", i) for i in range(32))
    maximum = spec.legacy_amax(fp, values)
    assert sexpr(maximum).count("op:max.NaN.f32") == 32
    assert sexpr(maximum).count("op:abs.bits.f32") == 32
    assert maximum.args[0] is fp.const(0.0)
    assert maximum is not spec.legacy_amax(fp, values[::-1])
    legacy_scale = spec.quantization_scale_value(fp, maximum, kind, legacy=True)
    next_scale = spec.quantization_scale_value(fp, maximum, kind)
    assert constant in sexpr(legacy_scale)
    assert legacy_scale is not next_scale
    assert sexpr(legacy_scale).count("op:mul.rn.f32") == 2
    legacy_enc = spec.quantization_encoding_scale(fp, legacy_scale, legacy=True)
    assert legacy_enc.op == "op:min.NaN.f32"
    assert legacy_enc.args[0].args[0] is fp.const(1)
    assert "7149f2ca" not in sexpr(legacy_enc)
    assert "7149f2ca" in sexpr(spec.quantization_encoding_scale(fp, next_scale))
    assert "00800000" in sexpr(
        spec.quantization_scale_value(fp, maximum, kind, staging=True)
    )
    with pytest.raises(ValueError, match="normal scales"):
        spec.input_quantize_amax_precast(fp, values, maximum, kind, normal_scales=False)


PLACEMENTS = tuple(
    (a, p, k, early)
    for a, p, k in CLASSES
    for early in ((True,) if p == "legacy" else (False, True))
)


@pytest.mark.parametrize("arch,provider,kind,early", PLACEMENTS)
def test_compositional_trace_and_incremental_hash(arch, provider, kind, early):
    kwargs = dict(
        **SHAPE,
        arch=arch,
        quant_kind=kind,
        implementation=provider,
        apply_routing_weights_before_fc2=early,
        gate_up_clamp=-2,
        valid_routes=(False, True, False),
    )
    terms = spec.component_terms(**kwargs)
    blocked = spec.component_terms(**kwargs, blocked=True)
    assert len(terms) == 15
    assert all(blocked[name] is value for name, value in terms.items())
    assert "00800000" in sexpr(terms["input_scale_value"])
    for value in terms.values():
        assert spec._fingerprint_component(value) == term.fingerprint(value)
    final = sexpr(terms["finalize"])
    assert final.count("op:add.rn.f32" if early else "op:fma.rn.f32") == 3
    assert final.count("op:bf16_bits 0000") == 2


@pytest.mark.parametrize("arch,provider,kind", CLASSES)
def test_public_registry_golden_and_class_dimensions(arch, provider, kind):
    kwargs = dict(**SHAPE, arch=arch, quant_kind=kind, implementation=provider)
    receipt = spec.spec_fingerprint(**kwargs)
    assert receipt == "megamoe-mxfp8-" + GOLDEN[CLASSES.index((arch, provider, kind))]
    assert fingerprint("megamoe_mxfp8", **kwargs) == receipt
    assert class_name("megamoe_mxfp8") == "megamoe.mxfp8"
    if provider == "standard" and kind == "mxfp8_e4m3":
        assert fingerprint("megamoe_mxfp8", **SHAPE, arch=arch) == receipt
    assert spec.spec_fingerprint(**(kwargs | dict(gate_up_clamp=0))) != receipt
    if provider == "legacy":
        with pytest.raises(ValueError, match="legacy requires early"):
            spec.spec_fingerprint(
                **(kwargs | dict(apply_routing_weights_before_fc2=False))
            )
    else:
        assert (
            spec.spec_fingerprint(
                **(kwargs | dict(apply_routing_weights_before_fc2=False))
            )
            != receipt
        )
    assert spec.spec_fingerprint(
        **(kwargs | dict(gate_up_clamp=-2))
    ) == spec.spec_fingerprint(**(kwargs | dict(gate_up_clamp=2)))


def test_classes_are_isolated_and_large_receipts_do_not_retain_global_terms():
    assert len(set(GOLDEN)) == 16
    before = len(term._TABLE)
    tracemalloc.start()
    value = spec.spec_fingerprint(
        65536, 65536, 32, 256, arch="sm_100", implementation="legacy"
    )
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert value.startswith("megamoe-mxfp8-") and len(term._TABLE) == before
    assert peak < 16 * 1024**2
    fp = spec.OrderedTermAlgebra()
    with pytest.raises(ValueError, match="generic ordered primitives"):
        spec._fingerprint_component(fp.op("mma_mxfp8_k32", fp.const(0)))


@pytest.mark.parametrize(
    "changes",
    (
        dict(hidden=64),
        dict(intermediate=0),
        dict(top_k=0),
        dict(num_experts=0),
        dict(quant_kind="mxfp4"),
        dict(arch="sm_90"),
        dict(arch="sm_107", implementation="legacy"),
        dict(apply_routing_weights_before_fc2=1),
    ),
)
def test_invalid_class_dimensions_are_rejected(changes):
    with pytest.raises(ValueError):
        spec.spec_fingerprint(**(SHAPE | dict(arch="sm_100") | changes))
