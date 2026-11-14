# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Device-free checks of NVFP4 MegaMoE's arithmetic classes and receipts."""

from itertools import product
import tracemalloc

import pytest

from flashinfer.batch_invariant import class_name, fingerprint
from flashinfer.batch_invariant.fp_interface import term
from flashinfer.batch_invariant.fp_interface.term import sexpr
from flashinfer.moe_ep.batch_invariant import nvfp4_spec as spec

pytestmark = pytest.mark.cpu_only

PROVIDERS = tuple(
    (arch, implementation)
    for arch, implementation in product(
        ("sm_100", "sm_103", "sm_107"), ("legacy", "standard", "genphase")
    )
    if (arch, implementation) != ("sm_107", "legacy")
)
SHAPE = dict(hidden=256, intermediate=256, top_k=3, num_experts=4)
SCALES = dict(has_fc1_alpha=True, has_fc2_alpha=True, has_fc1_norm_const=True)
GOLDEN = {
    ("sm_100", "legacy"): "megamoe-nvfp4-bcac6de7438b948f",
    ("sm_100", "standard"): "megamoe-nvfp4-f237e3bf1578ed79",
    ("sm_100", "genphase"): "megamoe-nvfp4-a0ce8ca7cce18d7d",
    ("sm_103", "legacy"): "megamoe-nvfp4-8c01cadbbc2aa8f2",
    ("sm_103", "standard"): "megamoe-nvfp4-ad0f4aaabc847716",
    ("sm_103", "genphase"): "megamoe-nvfp4-b0a0ffbc1ce53a3c",
    ("sm_107", "standard"): "megamoe-nvfp4-0ecd95dfb6d0f8d8",
    ("sm_107", "genphase"): "megamoe-nvfp4-fb5fd1b8b4eb98d0",
}


@pytest.mark.parametrize("arch,implementation", PROVIDERS)
@pytest.mark.parametrize("k", (0, 1, 64, 128, 192, 256, 320))
def test_actual_mma_atoms_init_order_and_physical_stage_padding(
    arch, implementation, k
):
    fp = spec.OrderedTermAlgebra()
    cls = spec.numerical_class(arch, implementation)
    flat = spec.mma_fold(fp, k, "boundary", arch=arch, implementation=implementation)
    blocked = spec.mma_fold(
        fp, k, "boundary", arch=arch, implementation=implementation, blocked=True
    )
    assert flat is blocked
    text = sexpr(flat)
    count = ((k + cls.stage_k - 1) // cls.stage_k) * cls.stage_k // cls.instruction_k
    assert text.count("op:mma.init.nvfp4") == bool(k)
    assert text.count("op:mma.acc.nvfp4") == max(0, count - 1)
    assert text.count(f"op:operand.nvfp4.k{cls.instruction_k}") == count
    assert "bf16.k16" not in text
    if k == 64 and cls.stage_k == 256:
        assert f"boundary {count - 1} 0 e2m1.e4m3.block16" in text


def test_clamp_and_dequant_are_ordered_distinct_from_bf16():
    fp = spec.OrderedTermAlgebra()
    gate, up, alpha, weight = (fp.op("input", name) for name in ("g", "u", "a", "w"))
    result = spec.fc1_activation(fp, gate, up, alpha=alpha, weight=weight, clamp=2)
    expression = sexpr(result)
    assert "min.f32 (op:mul.rn.f32 (op:input g) (op:input a))" in expression
    assert "min.xorsign.abs.f32 (op:mul.rn.f32 (op:input u) (op:input a))" in expression
    assert ".NaN.f32" not in expression
    assert "ex2.approx.ftz.f32" in expression
    assert result.args[1] is weight
    assert spec.fc1_activation(fp, gate, up, clamp=-2) is spec.fc1_activation(
        fp, gate, up, clamp=2
    )
    assert spec.fc1_activation(fp, gate, up, clamp=-0.0) is spec.fc1_activation(
        fp, gate, up, clamp=0.0
    )
    assert fp.mul(gate, alpha) is not fp.mul(alpha, gate)


def test_block16_requantization_keeps_amax_order_scale_and_cap_mask():
    fp = spec.OrderedTermAlgebra()
    values = tuple(fp.op("x", i) for i in range(16))
    norm = fp.op("norm")
    amax = spec.block_amax(fp, values)
    assert sexpr(amax).count("max.xorsign.abs.f32") == 15
    assert amax.op == "op:abs.bits.f32"
    assert spec.block_amax(fp, values[::-1]) is not amax
    scale = spec.quantization_scale(fp, amax, norm=norm)
    assert scale.args[0].args[1] is norm
    assert scale.args[0].args[0] is fp.mul(amax, fp.const(1 / 6))
    encoding = spec.quantization_encoding_scale(fp, scale, norm=norm)
    text = sexpr(encoding)
    assert "rcp.approx.ftz.f32" in text
    assert "7f7fffff" in text and "7149f2ca" in text
    assert "div" not in text
    pre, sf = spec.quantize_block_precast(fp, values, norm=norm)
    out, sf2 = spec.quantize_block(fp, values, norm=norm)
    assert sf is sf2 is scale
    assert all(
        v.op == "op:cvt.rn.satfinite.e2m1.f32" and v.args == (p,)
        for v, p in zip(out, pre, strict=True)
    )
    input_amax = spec.input_block_amax(fp, values)
    assert sexpr(input_amax).count("op:max.f32") == 32
    assert "max.xorsign.abs" not in sexpr(input_amax)
    assert spec.input_quantize_block(fp, values, norm=norm)[1] is not scale
    with pytest.raises(ValueError, match="16"):
        spec.quantize_block(fp, values[:8])


@pytest.mark.parametrize("early", (False, True))
@pytest.mark.parametrize("arch,implementation", PROVIDERS)
def test_all_components_have_same_flat_and_blocked_trace(early, arch, implementation):
    kwargs = dict(
        **SHAPE,
        **SCALES,
        arch=arch,
        implementation=implementation,
        apply_routing_weights_before_fc2=early,
        gate_up_clamp=-2,
        valid_routes=(False, True, False),
    )
    flat = spec.component_terms(**kwargs)
    blocked = spec.component_terms(**kwargs, blocked=True)
    assert len(flat) == 13
    assert all(blocked[name] is value for name, value in flat.items())
    final = sexpr(flat["finalize"])
    assert final.count("op:add.rn.f32" if early else "op:fma.rn.f32") == 3
    assert final.count("op:bf16_bits 0000") == 2
    for value in flat.values():
        assert spec._fingerprint_component(value) == term.fingerprint(value)


@pytest.mark.parametrize("arch,implementation", PROVIDERS)
def test_public_registry_golden_and_class_isolation(arch, implementation):
    kwargs = dict(**SHAPE, **SCALES, arch=arch, implementation=implementation)
    receipt = spec.spec_fingerprint(**kwargs)
    assert receipt == GOLDEN[arch, implementation]
    assert fingerprint("megamoe_nvfp4", **kwargs) == receipt
    assert (
        fingerprint("megamoe_nvfp4", **SHAPE, arch=arch, implementation=implementation)
        == receipt
    )
    assert class_name("megamoe_nvfp4") == "megamoe.nvfp4"
    assert len(set(GOLDEN.values())) == 8
    for change in (
        dict(apply_routing_weights_before_fc2=False),
        dict(gate_up_clamp=0),
        dict(input_norm_const=0.5),
        dict(has_fc1_alpha=False),
        dict(has_fc2_alpha=False),
        dict(has_fc1_norm_const=False),
    ):
        assert spec.spec_fingerprint(**(kwargs | change)) != receipt
    assert spec.spec_fingerprint(
        **(kwargs | dict(gate_up_clamp=-2))
    ) == spec.spec_fingerprint(**(kwargs | dict(gate_up_clamp=2)))


def test_large_runtime_fingerprint_has_bounded_memory_and_no_global_retention():
    before = len(term._TABLE)
    tracemalloc.start()
    receipt = spec.spec_fingerprint(
        65536, 65536, 32, 256, arch="sm_107", implementation="standard", **SCALES
    )
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert receipt.startswith("megamoe-nvfp4-")
    assert len(term._TABLE) == before
    assert peak < 16 * 1024**2


@pytest.mark.parametrize(
    "change",
    (
        dict(hidden=0),
        dict(hidden=32),
        dict(intermediate=32),
        dict(top_k=0),
        dict(num_experts=0),
        dict(arch="sm_90"),
        dict(arch="sm_107", implementation="legacy"),
        dict(implementation="auto"),
        dict(input_norm_const=0),
        dict(input_norm_const=float("inf")),
        dict(has_fc1_alpha=1),
        dict(apply_routing_weights_before_fc2=1),
    ),
)
def test_invalid_class_dimensions_are_rejected(change):
    with pytest.raises(ValueError):
        spec.spec_fingerprint(
            **(SHAPE | dict(arch="sm_100", implementation="legacy") | change)
        )
