# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Device-free conformance of MegaMoE's ordered arithmetic specification."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import struct
import sys
from types import SimpleNamespace

import pytest

from flashinfer.batch_invariant import CLASSES, class_name, fingerprint
from flashinfer.batch_invariant.fp_interface import term as shared_term
from flashinfer.batch_invariant.fp_interface.term import (
    fingerprint as term_fingerprint,
    sexpr,
    size,
)
from flashinfer.moe_ep.batch_invariant.spec import (
    ALLOWED_PLANS,
    ALLOWED_PLAN_CONFIGS,
    OrderedTermAlgebra,
    PlanConfig,
    _fingerprint_component,
    component_terms,
    f32_bits,
    fc1_activation,
    finalizer_term,
    mma_fold,
    mma_plan_fold,
    plan_component_terms,
    plan_config_compatible,
    spec_fingerprint,
    specification,
)

pytestmark = pytest.mark.cpu_only

SHAPE = dict(hidden=96, intermediate=128, top_k=3, num_experts=4)
ARCHES = ("sm_100", "sm_103", "sm_107")


def test_ordered_atoms_preserve_operands_and_signed_zero():
    fp = OrderedTermAlgebra()
    a, b, c = (fp.op("input", name) for name in ("a", "b", "c"))
    assert fp.add(a, b) is not fp.add(b, a)
    assert fp.mul(a, b) is not fp.mul(b, a)
    assert fp.fma(a, b, c) is not fp.fma(b, a, c)
    assert fp.add(fp.add(a, b), c) is not fp.add(a, fp.add(b, c))
    assert f32_bits(0.0) == "00000000"
    assert f32_bits(-0.0) == "80000000"
    assert fp.const(0.0) is not fp.const(-0.0)
    assert f32_bits(float("inf")) == "7f800000"


def test_activation_records_actual_primitive_chain_and_roundings():
    fp = OrderedTermAlgebra()
    gate, up, weight = (fp.op("input", name) for name in ("gate", "up", "weight"))
    late = sexpr(fc1_activation(fp, gate, up))
    assert late == (
        "(op:cvt.rn.bf16.f32 "
        "(op:mul.rn.f32 (op:mul.rn.f32 (op:input up) (op:input gate)) "
        "(op:rcp.approx.ftz.f32 (op:add.rn.f32 "
        "(op:ex2.approx.ftz.f32 (op:mul.rn.f32 (op:input gate) "
        "(op:f32_bits bfb8aa3b))) (op:f32_bits 3f800000)))))"
    )
    early = fc1_activation(fp, gate, up, weight=weight)
    assert early.args[0].op == "op:mul.rn.f32"
    assert early.args[0].args[1] is weight
    assert early.args[0].args[0] is fc1_activation(fp, gate, up).args[0]
    clamped = sexpr(fc1_activation(fp, gate, up, clamp=2.0))
    assert "(op:min.NaN.f32 (op:input gate) (op:f32_bits 40000000))" in clamped
    assert (
        "(op:max.NaN.f32 (op:min.NaN.f32 (op:input up) "
        "(op:f32_bits 40000000)) (op:f32_bits c0000000))"
    ) in clamped


@pytest.mark.parametrize("k", (0, 32, 64, 96, 128, 160, 256))
def test_blocked_mma_trace_matches_flat_fold_and_keeps_padding(k):
    fp = OrderedTermAlgebra()
    flat = mma_fold(fp, k)
    assert mma_fold(fp, k, blocked=True) is flat
    expression = sexpr(flat)
    steps = ((k + 63) // 64) * 4
    assert expression.count("op:operand.k16") == steps
    assert expression.count("op:mma.init.bf16.k16") == bool(k)
    assert expression.count("op:mma.acc.bf16.k16") == max(0, steps - 1)
    if k == 32:
        assert "(op:operand.k16 fc1.gate 2 0)" in expression
        assert "(op:operand.k16 fc1.gate 3 0)" in expression
    if k == 0:
        assert flat is fp.const(0.0)


@pytest.mark.parametrize("early", (False, True))
@pytest.mark.parametrize("arch", ARCHES)
def test_all_six_plan_lowerings_match_one_canonical_composition(early, arch):
    canonical = component_terms(
        **SHAPE, apply_routing_weights_before_fc2=early, arch=arch
    )
    assert len(ALLOWED_PLANS) == 6
    for plan in ALLOWED_PLANS:
        lowered = plan_component_terms(
            **SHAPE, apply_routing_weights_before_fc2=early, arch=arch, plan=plan
        )
        assert lowered.keys() == canonical.keys()
        assert all(lowered[name] is term for name, term in canonical.items())
    with pytest.raises(ValueError, match="unvalidated"):
        plan_component_terms(**SHAPE, plan=("unknown", "epi_warps"))


@pytest.mark.parametrize("early", (False, True))
@pytest.mark.parametrize("arch", ARCHES)
@pytest.mark.parametrize(
    "hidden,intermediate",
    ((32, 64), (96, 128), (128, 128), (160, 256), (224, 256), (256, 256)),
)
def test_all_compatible_tile_lowerings_preserve_the_canonical_components(
    early, arch, hidden, intermediate
):
    canonical = component_terms(hidden, intermediate, 3, 4, early, arch=arch)
    assert len(ALLOWED_PLAN_CONFIGS) == 12
    for config in ALLOWED_PLAN_CONFIGS:
        if not plan_config_compatible(hidden, intermediate, config):
            with pytest.raises(ValueError, match="canonical K64 padding"):
                plan_component_terms(
                    hidden,
                    intermediate,
                    3,
                    4,
                    early,
                    arch=arch,
                    plan=ALLOWED_PLANS[0],
                    plan_config=config,
                )
            continue
        for plan in ALLOWED_PLANS:
            lowered = plan_component_terms(
                hidden,
                intermediate,
                3,
                4,
                early,
                arch=arch,
                plan=plan,
                plan_config=config,
            )
            assert all(lowered[name] is term for name, term in canonical.items())


def test_larger_k_stage_does_not_append_zero_mma_instructions():
    fp = OrderedTermAlgebra()
    for k, stage in ((96, 128), (224, 256), (256, 256), (0, 256)):
        assert mma_plan_fold(fp, k, stage_k=stage) is mma_fold(fp, k)
    # Both cases would append four all-zero K16 instructions. The model must
    # reject them instead of assuming a floating-point zero-MMA identity.
    for k, stage in ((64, 128), (160, 256)):
        with pytest.raises(ValueError, match="canonical K64 padding"):
            mma_plan_fold(fp, k, stage_k=stage)
    assert plan_config_compatible(96, 128, PlanConfig((256, 128, 128)))
    assert plan_config_compatible(
        224, 256, PlanConfig((128, 256, 256), (1, 1, 1), False)
    )
    assert not plan_config_compatible(96, 64, PlanConfig((256, 128, 128)))


@pytest.mark.parametrize(
    "options",
    (
        {"mma_tiler_mnk": (64, 128, 64)},
        {"mma_tiler_mnk": (256, 64, 64)},
        {"mma_tiler_mnk": (256, 256, 32)},
        {"mma_tiler_mnk": (128, 256, 64)},
        {"cluster_shape_mnk": (1, 1, 1)},
        {"use_2cta_instrs": 1},
        {"mma_tiler_mnk": [256, 256, 64]},
    ),
)
def test_invalid_tile_geometries_fail_closed(options):
    with pytest.raises(ValueError):
        PlanConfig(**options)


def test_existing_v1_fingerprints_remain_unchanged():
    path = Path(__file__).with_name("fixtures") / "megamoe_bf16_v1_fingerprints.json"
    saved = json.loads(path.read_text())
    assert len(saved) == 90
    assert {case["args"]["arch"] for case in saved[:60]} == {"sm_100", "sm_103"}
    assert all(case["args"]["arch"] == "sm_107" for case in saved[60:])
    for case in saved:
        assert spec_fingerprint(**case["args"]) == case["fingerprint"]


@pytest.mark.parametrize("early", (False, True))
@pytest.mark.parametrize("clamp", (None, -0.0, -2.0, float("inf"), float("nan")))
@pytest.mark.parametrize("valid_routes", (None, (False, True, False)))
def test_incremental_component_hash_matches_shared_sexpr(early, clamp, valid_routes):
    terms = component_terms(
        **SHAPE,
        apply_routing_weights_before_fc2=early,
        gate_up_clamp=clamp,
        valid_routes=valid_routes,
    )
    for term in terms.values():
        assert _fingerprint_component(term) == term_fingerprint(term)


def test_incremental_hash_keeps_operand_association_and_mma_initialization():
    fp = OrderedTermAlgebra()
    a, b, c = (fp.op("ref.f32", name) for name in ("a", "b", "c"))
    operands = fp.op("operand.k16", "fc1.gate", 0, 16)
    pairs = (
        (fp.add(a, b), fp.add(b, a)),
        (fp.mul(a, b), fp.mul(b, a)),
        (fp.fma(a, b, c), fp.fma(b, a, c)),
        (fp.add(fp.add(a, b), c), fp.add(a, fp.add(b, c))),
        (fp.fma(a, b, c), fp.add(fp.mul(a, b), c)),
        (fp.const(0.0), fp.const(-0.0)),
        (
            fp.op("mma.init.bf16.k16", operands),
            fp.op("mma.acc.bf16.k16", fp.const(0.0), operands),
        ),
    )
    for left, right in pairs:
        assert _fingerprint_component(left) == term_fingerprint(left)
        assert _fingerprint_component(right) == term_fingerprint(right)
        assert _fingerprint_component(left) != _fingerprint_component(right)
    for k in (0, 32, 64, 96, 128, 160, 256):
        term = mma_fold(fp, k)
        assert _fingerprint_component(term) == term_fingerprint(term)
    # This serializer is specific to MegaMoE's ordered atoms. Shared algebra
    # operators retain their own normalization and serialization semantics.
    with pytest.raises(ValueError, match="unvalidated MegaMoE component primitive"):
        _fingerprint_component(shared_term.TermAlgebra().const(0.0))


def test_fingerprint_construction_does_not_grow_the_shared_interner():
    before = len(shared_term._TABLE)
    for hidden in (32, 160, 224, 1024):
        for early in (False, True):
            args = dict(hidden=hidden, intermediate=192, top_k=3, num_experts=12)
            specification(**args, apply_routing_weights_before_fc2=early)
            spec_fingerprint(**args, apply_routing_weights_before_fc2=early)
    assert len(shared_term._TABLE) == before


def test_large_fingerprint_has_bounded_peak_and_retained_memory():
    # A fresh interpreter prevents existing symbolic nodes or receipt cache
    # hits from concealing allocations. Trace only fingerprint work, not imports.
    code = """
import gc
import json
import tracemalloc
from flashinfer.batch_invariant.fp_interface import term
from flashinfer.moe_ep.batch_invariant.spec import specification, spec_fingerprint
gc.collect()
before = len(term._TABLE)
tracemalloc.start()
receipts = []
for early in (False, True):
    args = dict(hidden=65536, intermediate=65536, top_k=32, num_experts=256,
                apply_routing_weights_before_fc2=early, gate_up_clamp=2.0)
    spec = specification(**args)
    assert set(spec['components']) == {'fc1_gate', 'fc1_up', 'activation', 'fc2', 'finalize'}
    receipts.append(spec_fingerprint(**args))
gc.collect()
current, peak = tracemalloc.get_traced_memory()
assert len(term._TABLE) == before, (before, len(term._TABLE))
assert peak < 32 * 1024**2, peak
assert current < 2 * 1024**2, current
# Captured from the original v1 serializer for this real 4096-step FC2 fold.
assert receipts[1] == 'megamoe-bf16-e438134b1e5b5353', receipts
print(json.dumps(dict(peak=peak, retained=current, receipts=receipts)))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("early", (False, True))
def test_masked_slots_are_zero_steps_instead_of_deleted_leaves(early):
    fp = OrderedTermAlgebra()
    term = finalizer_term(fp, 3, 4, early, valid_routes=(False, True, False))
    expression = sexpr(term)
    primitive = "op:add.rn.f32" if early else "op:fma.rn.f32"
    assert expression.count(primitive) == 3
    assert "(op:ref.bf16 fc2 0)" not in expression
    assert "(op:ref.bf16 fc2 2)" not in expression
    assert "(op:ref.bf16 fc2 1)" in expression
    if not early:
        assert "(op:input.f32 route_weight 0)" not in expression
        assert "(op:input.f32 route_weight 2)" not in expression
    symbolic = sexpr(finalizer_term(fp, 3, 4, early))
    assert "op:select.lazy" in symbolic
    assert "(op:route.valid (op:input.expert_id 1) 4)" in symbolic
    with pytest.raises(ValueError, match="one bool"):
        finalizer_term(fp, 3, 4, early, valid_routes=(True,))


def test_composition_binds_handoffs_without_expanding_all_gemm_operands():
    terms = component_terms(7168, 2048, 8, 256)
    assert set(terms) == {"fc1_gate", "fc1_up", "activation", "fc2", "finalize"}
    # FC2 references the BF16 handoff specification; it does not duplicate the
    # entire FC1 expression for every intermediate feature and route.
    assert sum(size(term) for term in terms.values()) < 10000
    spec = specification(7168, 2048, 8, 256)
    assert ("fc2.handoff_w2", "activation", "route_expert.w2") in spec["composition"]
    assert "batch" not in spec and "capacity" not in spec and "plan" not in spec


def test_fingerprint_identifies_arithmetic_instead_of_launch_configuration():
    base = spec_fingerprint(**SHAPE)
    variants = (
        {"hidden": 64},
        {"intermediate": 64},
        {"top_k": 2},
        {"num_experts": 8},
        {"apply_routing_weights_before_fc2": False},
        {"gate_up_clamp": 2.0},
        {"gate_up_clamp": 0.0},
        {"arch": "sm_103"},
        {"arch": "sm_107"},
    )
    fingerprints = [spec_fingerprint(**{**SHAPE, **override}) for override in variants]
    assert base not in fingerprints
    assert len(set(fingerprints)) == len(fingerprints)
    with pytest.raises(TypeError):
        spec_fingerprint(**SHAPE, batch_size=32)
    with pytest.raises(TypeError):
        fingerprint("megamoe_bf16", **SHAPE, arch="sm_100", capacity=257)


@pytest.mark.parametrize(
    "arch,cc", (("sm_100", (10, 0)), ("sm_103", (10, 3)), ("sm_107", (10, 7)))
)
def test_public_fingerprint_registry_and_default_options(arch, cc):
    info = CLASSES["megamoe_bf16"]
    assert set(info.dims) == set(SHAPE)
    assert {
        "apply_routing_weights_before_fc2",
        "gate_up_clamp",
        "arch",
        "activation",
        "swiglu_alpha",
        "swiglu_beta",
        "situ_beta",
        "situ_linear_beta",
        "per_token_scale",
        "combine_dtype",
        "output_mode",
    } == set(info.optional)
    assert class_name("megamoe_bf16") == "megamoe.bf16"
    assert fingerprint("megamoe_bf16", **SHAPE, arch=arch) == spec_fingerprint(
        **SHAPE, arch=arch
    )
    assert fingerprint("megamoe_bf16", **SHAPE, arch=cc) == spec_fingerprint(
        **SHAPE, arch=arch
    )


def test_architectures_share_ordered_trace_but_have_separate_receipts():
    terms = [component_terms(**SHAPE, arch=arch) for arch in ARCHES]
    assert terms[0] == terms[1] == terms[2]
    specs = [specification(**SHAPE, arch=arch) for arch in ARCHES]
    assert all(spec["version"] == 1 for spec in specs)
    assert {spec["arch"] for spec in specs} == set(ARCHES)
    assert len({spec_fingerprint(**SHAPE, arch=arch) for arch in ARCHES}) == 3
    # Equal syntax permits a different fixed hardware interpretation on each
    # architecture. It does not assert cross-architecture bitwise equality.
    assert specs[0]["components"] == specs[1]["components"] == specs[2]["components"]


def test_fingerprint_cache_uses_effective_clamp_bits_including_nan_payloads():
    positive_zero = spec_fingerprint(**SHAPE, gate_up_clamp=0.0)
    negative_zero = spec_fingerprint(**SHAPE, gate_up_clamp=-0.0)
    assert positive_zero == negative_zero
    assert spec_fingerprint(**SHAPE, gate_up_clamp=2.0) == spec_fingerprint(
        **SHAPE, gate_up_clamp=-2.0
    )
    assert component_terms(**SHAPE, gate_up_clamp=2.0) == component_terms(
        **SHAPE, gate_up_clamp=-2.0
    )
    # Host floats that round to the same device constant name one calculation.
    assert spec_fingerprint(**SHAPE, gate_up_clamp=2.0) == spec_fingerprint(
        **SHAPE, gate_up_clamp=2.0 + 2.0**-30
    )
    nans = [
        struct.unpack(">f", bytes.fromhex(bits))[0] for bits in ("7fc00001", "7fc00002")
    ]
    assert [f32_bits(value) for value in nans] == ["7fc00001", "7fc00002"]
    assert spec_fingerprint(**SHAPE, gate_up_clamp=nans[0]) != spec_fingerprint(
        **SHAPE, gate_up_clamp=nans[1]
    )


@pytest.mark.parametrize(
    "override",
    (
        {"hidden": 0},
        {"hidden": 33},
        {"intermediate": 32},
        {"top_k": 0},
        {"top_k": True},
        {"num_experts": 0},
        {"apply_routing_weights_before_fc2": 1},
        {"arch": "sm_120"},
    ),
)
def test_unsupported_specifications_fail_before_device_work(override):
    with pytest.raises(ValueError):
        spec_fingerprint(**{**SHAPE, **override})


@pytest.mark.parametrize("arch", ARCHES)
def test_public_fingerprint_is_process_stable_and_does_not_import_kernel_tree(arch):
    code = """
import sys
from flashinfer.batch_invariant import fingerprint
value = fingerprint('megamoe_bf16', hidden=96, intermediate=128, top_k=3,
                    num_experts=4, arch=sys.argv[1])
prefix = 'flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe'
assert not any(name == prefix or name.startswith(prefix + '.') for name in sys.modules)
print(value)
"""
    result = subprocess.run(
        [sys.executable, "-c", code, arch], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip().splitlines()[-1] == spec_fingerprint(
        **SHAPE, arch=arch
    )


@pytest.mark.parametrize("clamp_argument", ("gate_up_clamp", "activation_clamp"))
@pytest.mark.parametrize("original,override", ((1.0, 2.0), (1.0, -2.0)))
def test_autotune_records_the_effective_launch_clamp_receipt(
    monkeypatch, tmp_path, clamp_argument, original, override
):
    import torch

    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe.shim import (
        autotune,
        bf16,
        knob_cache,
    )

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", str(tmp_path / "knobs.json"))
    frontend = bf16.MegaMoEBf16Frontend(
        bf16.MegaMoEBf16Config(
            rank=0,
            world_size=1,
            num_tokens_per_rank=32,
            num_topk=2,
            num_total_experts=4,
            hidden=64,
            intermediate=64,
            gate_up_clamp=original,
            batch_invariant=True,
        )
    )
    buffer = SimpleNamespace(_frontend=frontend)

    def fake_launch(*_args, **kwargs):
        # The real launch updates the existing frontend before compiling or
        # running. Exercise the same mutation without invoking a GPU benchmark.
        frontend.set_gate_up_clamp(kwargs[clamp_argument])

    def fake_sweep(frontend, launch, candidates, *, on_winner, **_kwargs):
        winner = candidates[0]
        frontend.apply_knobs(winner)
        launch()
        on_winner(winner, 1e-6)
        return winner

    monkeypatch.setattr(bf16, "bf16_mega_moe", fake_launch)
    monkeypatch.setattr(autotune, "autotune_knobs", fake_sweep)
    winner = autotune.autotune_bf16_mega_moe(
        None, None, None, buffer, num_tokens=1, **{clamp_argument: override}
    )
    key = dict(
        dtype="bf16",
        world_size=1,
        hidden=64,
        intermediate=64,
        num_experts=4,
        topk=2,
        max_tokens=32,
        live_tokens=1,
        apply_routing_weights_before_fc2=True,
        batch_invariant=True,
    )
    assert knob_cache.lookup_knobs(**key, gate_up_clamp=override) == winner
    assert knob_cache.lookup_knobs(**key, gate_up_clamp=original) is None
