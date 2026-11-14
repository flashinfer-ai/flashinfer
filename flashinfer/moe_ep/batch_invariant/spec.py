# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Ordered, compositional dataflow of the SM100/103/107 BF16 MegaMoE class.

FC1 and FC2 are folds over K16 MMA steps, including the zero-filled steps of
their last K64 stage. The first instruction resets the accumulator: it is an
independent primitive, rather than an assumed ``mma(+0, ...)`` identity.
Activation, BF16 handoff, BF16 combine storage and slot-order finalization are
separate components, linked by named references. This keeps the specification
linear in H + I + top-k instead of expanding the FC1 computation inside every
operand slice of every FC2 output.

All primitive arguments remain ordered, including add/mul and their NaN
operands. Approximate exponential, reciprocal and MMA steps are opaque
hardware primitives. Equality of these terms does not prove that generated
CUDA implements the terms, that remote writes complete, or that instruction
variants have identical hardware semantics. Device conformance and the
dispatch/communication audit remain necessary. Each architecture has its own
fixed interpretation of these primitives and its own receipt; matching traces
across architectures does not assert matching output bits.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Sequence

from ...batch_invariant.fp_interface.term import Term, TermAlgebra
from . import numerical_variants as _numerical

SPEC_VERSION = 1
MMA_K = 16
STAGE_K = 64
LOG2_E = 1.4426950408889634
ALLOWED_PLANS = tuple(
    (schedule, carrier)
    for schedule in ("static", "atomic_counter")
    for carrier in ("epi_warps", "standalone_warps", "reuse_dispatch_warps")
)


@dataclass(frozen=True)
class PlanConfig:
    """Physical MMA geometry, excluded from the numerical receipt.

    One cluster owns ``M`` token rows. Each CTA owns 128 rows, with one or
    two CTAs per instruction. ``K`` is a staging width, not a new padding
    rule: a shape must also pass :func:`plan_config_compatible`.
    """

    mma_tiler_mnk: tuple[int, int, int] = (256, 256, 64)
    cluster_shape_mnk: tuple[int, int, int] = (2, 1, 1)
    use_2cta_instrs: bool = True

    def __post_init__(self) -> None:
        for name in ("mma_tiler_mnk", "cluster_shape_mnk"):
            value = getattr(self, name)
            if (
                type(value) is not tuple
                or len(value) != 3
                or any(type(axis) is not int for axis in value)
            ):
                raise ValueError(f"{name} must be a tuple of three integers")
        m, n, k = self.mma_tiler_mnk
        if m not in (128, 256) or n not in (128, 256) or k not in (64, 128, 256):
            raise ValueError(
                f"unvalidated batch-invariant geometry: {self.mma_tiler_mnk}"
            )
        if type(self.use_2cta_instrs) is not bool or self.use_2cta_instrs != (m == 256):
            raise ValueError("use_2cta_instrs must be True exactly for M=256")
        if self.cluster_shape_mnk != (m // 128, 1, 1):
            raise ValueError("cluster_shape_mnk must be (M // 128, 1, 1)")


DEFAULT_PLAN_CONFIG = PlanConfig()
ALLOWED_PLAN_CONFIGS = tuple(
    PlanConfig((m, n, k), (m // 128, 1, 1), m == 256)
    for m in (128, 256)
    for n in (128, 256)
    for k in (64, 128, 256)
)


def plan_config_compatible(hidden: int, intermediate: int, config: PlanConfig) -> bool:
    """Whether full physical K stages preserve the existing canonical padding.

    The old trace has ``ceil(K / 64) * 4`` K16 instructions. A larger stage
    may regroup those instructions but must not append additional zero MMA
    updates. SMEM capacity and hardware/lowering conformance are separate
    implementation checks.
    """
    if not isinstance(config, PlanConfig):
        return False
    if (
        type(hidden) is not int
        or hidden <= 0
        or hidden % 32
        or type(intermediate) is not int
        or intermediate <= 0
        or intermediate % 64
    ):
        return False
    stage_k = config.mma_tiler_mnk[2]
    return all(((k + 63) // 64 * 64) % stage_k == 0 for k in (hidden, intermediate))


def f32_bits(value: float) -> str:
    """The literal's binary32 representation, including signed zero and infinity."""
    value = float(value)
    try:
        packed = struct.pack(">f", value)
    except OverflowError:
        packed = struct.pack(">f", math.copysign(math.inf, value))
    return packed.hex()


class OrderedTermAlgebra(TermAlgebra):
    """The shared scalar chain's symbolic backend, without commutative rewriting."""

    def const(self, value: float) -> Term:
        return self.op("f32_bits", f32_bits(value))

    def add(self, a: Term, b: Term) -> Term:
        return self.op("add.rn.f32", a, b)

    def mul(self, a: Term, b: Term) -> Term:
        return self.op("mul.rn.f32", a, b)

    def fma(self, a: Term, b: Term, c: Term) -> Term:
        return self.op("fma.rn.f32", a, b, c)

    def ex2(self, a: Term) -> Term:
        return self.op("ex2.approx.ftz.f32", a)

    def rcp(self, a: Term) -> Term:
        return self.op("rcp.approx.ftz.f32", a)

    def fmin_nan(self, a: Term, b: Term) -> Term:
        return self.op("min.NaN.f32", a, b)

    def fmax_nan(self, a: Term, b: Term) -> Term:
        return self.op("max.NaN.f32", a, b)

    def cvt_out(self, a: Term, fmt_name: str) -> Term:
        if fmt_name != "bf16":
            raise ValueError("MegaMoE's specified output conversion is BF16 only")
        return self.op("cvt.rn.bf16.f32", a)

    def widen(self, a: Term) -> Term:
        return self.op("widen.bf16.f32", a)


class _FingerprintTermAlgebra(OrderedTermAlgebra):
    def __init__(self):
        # Runtime receipts must not retain every shape in the shared interner.
        # Public symbolic builders keep their existing global identity semantics.
        self._table = {}


_COMPONENT_OPS = frozenset(
    "op:" + name
    for name in (
        "f32_bits",
        "bf16_bits",
        "add.rn.f32",
        "mul.rn.f32",
        "fma.rn.f32",
        "ex2.approx.ftz.f32",
        "rcp.approx.ftz.f32",
        "min.NaN.f32",
        "max.NaN.f32",
        "cvt.rn.bf16.f32",
        "widen.bf16.f32",
        "operand.k16",
        "mma.init.bf16.k16",
        "mma.acc.bf16.k16",
        "ref.f32",
        "ref.bf16",
        "input.f32",
        "input.expert_id",
        "route.valid",
        "select.lazy",
    )
)


def _fingerprint_component(term: Term) -> str:
    """Hash the shared S-expression bytes without retaining expanded prefixes.

    MegaMoE uses only generic ordered ``op:`` nodes, with none of the shared
    serializer's commutative or pretty-print aliases. Keep that boundary
    explicit rather than changing serialization for other operator classes.
    The stack holds nodes and delimiters, never each accumulator's full text.
    """
    digest = hashlib.sha256()
    stack: list[Any] = [term]
    while stack:
        item = stack.pop()
        if not isinstance(item, Term):
            digest.update(str(item).encode())
            continue
        if item.op not in _COMPONENT_OPS:
            raise ValueError(f"unvalidated MegaMoE component primitive: {item.op}")
        digest.update(("(" + item.op + " ").encode())
        stack.append(")")
        for index, arg in enumerate(reversed(item.args)):
            if index:
                stack.append(" ")
            stack.append(arg)
    return digest.hexdigest()[:16]


def fc1_activation_accumulator(fp, gate, up, weight=None, clamp=None):
    """The shared scalar SwiGLU chain before its single BF16 handoff conversion.

    ``fp`` is the ordered symbolic backend or the explicit device backend.
    The operation order is also used by the real BI epilogue; vector packing
    of independent output elements is a device placement of this scalar chain.
    The kernel constructor interprets the clamp as a magnitude, so negative
    clamp arguments name the same chain as their positive counterparts.
    """
    if clamp is not None:
        clamp = abs(clamp)
        limit = fp.const(clamp)
        gate = fp.fmin_nan(gate, limit)
        up = fp.fmax_nan(fp.fmin_nan(up, limit), fp.const(-clamp))
    product = fp.mul(up, gate)
    exponent = fp.mul(gate, fp.const(-LOG2_E))
    sigmoid = fp.rcp(fp.add(fp.ex2(exponent), fp.const(1.0)))
    activation = fp.mul(product, sigmoid)
    if weight is not None:
        activation = fp.mul(activation, weight)
    return activation


def fc1_activation(fp, gate, up, weight=None, clamp=None):
    """The shared activation chain including the BF16 FC1-to-FC2 handoff."""
    return fp.cvt_out(fc1_activation_accumulator(fp, gate, up, weight, clamp), "bf16")


def _operand_slice(fp, operand: str, k: int, step: int):
    return fp.op("operand.k16", operand, step, min(MMA_K, max(0, k - step * MMA_K)))


def mma_fold(fp, k: int, operand: str = "fc1.gate", *, blocked: bool = False):
    """Flat specification or independent K64-blocked lowering of one MMA fold.

    The blocked trace models the actual stage loop's four K16 updates. Its
    equivalence to a device ``cute.gemm`` remains a hardware/lowering hypothesis.
    ``k=0`` is useful in the model; public operator dimensions are positive.
    """
    if type(k) is not int or k < 0:
        raise ValueError("MMA fold dimension must be a nonnegative integer")
    stages = (k + STAGE_K - 1) // STAGE_K
    accumulator = fp.const(0.0)
    if blocked:
        for stage in range(stages):
            for inner in range(STAGE_K // MMA_K):
                step = stage * (STAGE_K // MMA_K) + inner
                operands = _operand_slice(fp, operand, k, step)
                if stage == 0 and inner == 0:
                    accumulator = fp.op("mma.init.bf16.k16", operands)
                else:
                    accumulator = fp.op("mma.acc.bf16.k16", accumulator, operands)
    else:
        for step in range(stages * (STAGE_K // MMA_K)):
            operands = _operand_slice(fp, operand, k, step)
            if step == 0:
                accumulator = fp.op("mma.init.bf16.k16", operands)
            else:
                accumulator = fp.op("mma.acc.bf16.k16", accumulator, operands)
    return accumulator


def mma_plan_fold(fp, k: int, operand: str = "fc1.gate", *, stage_k: int):
    """Independent physical-stage lowering, with the canonical padding intact."""
    if type(k) is not int or k < 0:
        raise ValueError("MMA fold dimension must be a nonnegative integer")
    if type(stage_k) is not int or stage_k not in (64, 128, 256):
        raise ValueError("stage_k must be 64, 128, or 256")
    if ((k + 63) // 64 * 64) % stage_k:
        raise ValueError(
            "physical K stage would add MMA steps beyond canonical K64 padding"
        )
    accumulator = fp.const(0.0)
    for stage in range((k + stage_k - 1) // stage_k):
        for inner in range(stage_k // MMA_K):
            step = stage * (stage_k // MMA_K) + inner
            operands = _operand_slice(fp, operand, k, step)
            if stage == 0 and inner == 0:
                accumulator = fp.op("mma.init.bf16.k16", operands)
            else:
                accumulator = fp.op("mma.acc.bf16.k16", accumulator, operands)
    return accumulator


def finalizer_term(
    fp,
    top_k: int,
    num_experts: int,
    apply_routing_weights_before_fc2: bool = True,
    *,
    valid_routes: Sequence[bool] | None = None,
):
    """Fixed slot-order finalization, including masked slots' zero operations.

    A symbolic lazy select prevents an invalid route from loading a stale
    BF16 wire value. A concrete mask instantiates that select for model tests;
    it never removes the slot's add/FMA operation. Routing-slot permutations
    are deliberately not a placement of this specification.
    """
    if valid_routes is not None and (
        len(valid_routes) != top_k or any(type(v) is not bool for v in valid_routes)
    ):
        raise ValueError("valid_routes must contain one bool per top-k slot")
    zero = fp.const(0.0)
    wire_zero = fp.op("bf16_bits", "0000")
    accumulator = zero
    for slot in range(top_k):
        wire = fp.op("ref.bf16", "fc2", slot)
        score = fp.op("input.f32", "route_weight", slot)
        if valid_routes is None:
            valid = fp.op("route.valid", fp.op("input.expert_id", slot), num_experts)
            wire = fp.op("select.lazy", valid, wire, wire_zero)
            score = fp.op("select.lazy", valid, score, zero)
        elif not valid_routes[slot]:
            wire, score = wire_zero, zero
        value = fp.widen(wire)
        if apply_routing_weights_before_fc2:
            accumulator = fp.add(accumulator, value)
        else:
            accumulator = fp.fma(value, score, accumulator)
    return fp.cvt_out(accumulator, "bf16")


def _validate_spec(hidden, intermediate, top_k, num_experts, early, arch):
    if type(hidden) is not int or hidden <= 0 or hidden % 32:
        raise ValueError("hidden must be a positive multiple of 32")
    if type(intermediate) is not int or intermediate <= 0 or intermediate % 64:
        raise ValueError("intermediate must be a positive multiple of 64")
    if type(top_k) is not int or not 1 <= top_k <= 32:
        raise ValueError("top_k must be in [1, 32]")
    if type(num_experts) is not int or num_experts <= 0:
        raise ValueError("num_experts must be a positive integer")
    if type(early) is not bool:
        raise ValueError("apply_routing_weights_before_fc2 must be a bool")
    if arch not in ("sm_100", "sm_103", "sm_107"):
        raise ValueError(
            "batch-invariant BF16 MegaMoE requires sm_100, sm_103 or sm_107"
        )


def component_terms(
    hidden: int,
    intermediate: int,
    top_k: int,
    num_experts: int,
    apply_routing_weights_before_fc2: bool = True,
    gate_up_clamp: float | None = None,
    arch: str = "sm_100",
    *,
    blocked: bool = False,
    valid_routes: Sequence[bool] | None = None,
    fp: OrderedTermAlgebra | None = None,
    **numerical_kwargs,
) -> dict[str, Term]:
    """Canonical components with references bound by ``COMPOSITION`` below."""
    _validate_spec(
        hidden, intermediate, top_k, num_experts, apply_routing_weights_before_fc2, arch
    )
    if not _numerical.is_default(**numerical_kwargs):
        return _numerical.component_terms(
            "bf16",
            (hidden, intermediate, top_k, num_experts),
            dict(
                arch=arch,
                apply_routing_weights_before_fc2=apply_routing_weights_before_fc2,
                gate_up_clamp=gate_up_clamp,
                blocked=blocked,
                valid_routes=valid_routes,
            ),
            numerical_kwargs,
            fp=fp,
        )
    fp = fp or OrderedTermAlgebra()
    return {
        "fc1_gate": mma_fold(fp, hidden, "fc1.gate", blocked=blocked),
        "fc1_up": mma_fold(fp, hidden, "fc1.up", blocked=blocked),
        "activation": fc1_activation(
            fp,
            fp.op("ref.f32", "fc1_gate"),
            fp.op("ref.f32", "fc1_up"),
            (
                fp.op("input.f32", "route_weight")
                if apply_routing_weights_before_fc2
                else None
            ),
            gate_up_clamp,
        ),
        "fc2": fp.cvt_out(
            mma_fold(fp, intermediate, "fc2.handoff_w2", blocked=blocked), "bf16"
        ),
        "finalize": finalizer_term(
            fp,
            top_k,
            num_experts,
            apply_routing_weights_before_fc2,
            valid_routes=valid_routes,
        ),
    }


def plan_component_terms(
    hidden: int,
    intermediate: int,
    top_k: int,
    num_experts: int,
    apply_routing_weights_before_fc2: bool = True,
    gate_up_clamp: float | None = None,
    arch: str = "sm_100",
    *,
    plan: tuple[str, str],
    plan_config: PlanConfig = DEFAULT_PLAN_CONFIG,
    valid_routes: Sequence[bool] | None = None,
    fp: OrderedTermAlgebra | None = None,
    **numerical_kwargs,
) -> dict[str, Term]:
    """A physical plan's lowering; M/N change ownership, K regroups K16 steps.

    Identical trees rely on the same per-element interpretation of the
    opaque MMA atoms for the admitted instruction shapes. This is not a
    proof of the corresponding CUDA memory protocol or hardware opcodes.
    """
    if plan not in ALLOWED_PLANS:
        raise ValueError(f"unvalidated MegaMoE batch-invariant plan: {plan!r}")
    if not plan_config_compatible(hidden, intermediate, plan_config):
        raise ValueError("plan geometry is incompatible with canonical K64 padding")
    fp = fp or OrderedTermAlgebra()
    terms = component_terms(
        hidden,
        intermediate,
        top_k,
        num_experts,
        apply_routing_weights_before_fc2,
        gate_up_clamp,
        arch,
        valid_routes=valid_routes,
        fp=fp,
        **numerical_kwargs,
    )
    stage_k = plan_config.mma_tiler_mnk[2]
    terms["fc1_gate"] = mma_plan_fold(fp, hidden, "fc1.gate", stage_k=stage_k)
    terms["fc1_up"] = mma_plan_fold(fp, hidden, "fc1.up", stage_k=stage_k)
    terms["fc2"] = fp.cvt_out(
        mma_plan_fold(fp, intermediate, "fc2.handoff_w2", stage_k=stage_k), "bf16"
    )
    return terms


# The component references are bindings, not additional independent inputs.
# Every route instantiates both FC1 folds and FC2 with the same logical source
# token and global expert; activation is FC2's BF16 row, and finalization reads
# the BF16 FC2 output associated with the original ordered top-k slot.
COMPOSITION = (
    ("fc1_gate", "input_row", "route_expert.w13_gate"),
    ("fc1_up", "input_row", "route_expert.w13_up"),
    ("activation", "fc1_gate", "fc1_up", "route_weight"),
    ("fc2.handoff_w2", "activation", "route_expert.w2"),
    ("ref.bf16:fc2:slot", "fc2", "original_route_slot"),
    ("finalize", "ordered_route_slots", "output_row"),
)


def specification(
    hidden: int,
    intermediate: int,
    top_k: int,
    num_experts: int,
    apply_routing_weights_before_fc2: bool = True,
    gate_up_clamp: float | None = None,
    arch: str = "sm_100",
    **numerical_kwargs,
) -> dict[str, Any]:
    """The compositional arithmetic receipt, excluding batch and placement."""
    if not _numerical.is_default(**numerical_kwargs):
        return _numerical.specification(
            "bf16",
            (hidden, intermediate, top_k, num_experts),
            dict(
                arch=arch,
                apply_routing_weights_before_fc2=apply_routing_weights_before_fc2,
                gate_up_clamp=gate_up_clamp,
            ),
            numerical_kwargs,
        )
    terms = component_terms(
        hidden,
        intermediate,
        top_k,
        num_experts,
        apply_routing_weights_before_fc2,
        gate_up_clamp,
        arch,
        fp=_FingerprintTermAlgebra(),
    )
    return {
        "class": "megamoe.bf16",
        "version": SPEC_VERSION,
        "hidden": hidden,
        "intermediate": intermediate,
        "top_k": top_k,
        "num_experts": num_experts,
        "arch": arch,
        "apply_routing_weights_before_fc2": apply_routing_weights_before_fc2,
        "clamp_bits": None if gate_up_clamp is None else f32_bits(abs(gate_up_clamp)),
        "composition": COMPOSITION,
        "components": {
            name: _fingerprint_component(term) for name, term in terms.items()
        },
    }


@lru_cache(maxsize=256)
def _cached_spec_fingerprint(
    hidden: int,
    intermediate: int,
    top_k: int,
    num_experts: int,
    apply_routing_weights_before_fc2: bool,
    clamp_bits: str | None,
    arch: str,
) -> str:
    clamp = (
        None
        if clamp_bits is None
        else struct.unpack(">f", bytes.fromhex(clamp_bits))[0]
    )
    spec = specification(
        hidden,
        intermediate,
        top_k,
        num_experts,
        apply_routing_weights_before_fc2,
        clamp,
        arch,
    )
    encoded = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
    return "megamoe-bf16-" + hashlib.sha256(encoded).hexdigest()[:16]


def spec_fingerprint(
    hidden: int,
    intermediate: int,
    top_k: int,
    num_experts: int,
    apply_routing_weights_before_fc2: bool = True,
    gate_up_clamp: float | None = None,
    arch: str = "sm_100",
    **numerical_kwargs,
) -> str:
    """Hash the arithmetic specification, rather than a launch configuration.

    The receipt is scoped to the same architecture/software/compiler and EP
    topology. Operands, ordered routing IDs and weights are runtime inputs;
    batch size, capacity, worker assignment and return carrier are placements.
    Cache keys use the binary32 bits of the effective clamp magnitude, matching
    the kernel constructor and avoiding repeated symbolic construction on a
    prepared layer's forward path. Other numerical constants retain their sign.
    """
    _validate_spec(
        hidden, intermediate, top_k, num_experts, apply_routing_weights_before_fc2, arch
    )
    if not _numerical.is_default(**numerical_kwargs):
        return _numerical.spec_fingerprint(
            "bf16",
            (hidden, intermediate, top_k, num_experts),
            dict(
                arch=arch,
                apply_routing_weights_before_fc2=apply_routing_weights_before_fc2,
                gate_up_clamp=gate_up_clamp,
            ),
            numerical_kwargs,
        )
    return _cached_spec_fingerprint(
        hidden,
        intermediate,
        top_k,
        num_experts,
        apply_routing_weights_before_fc2,
        None if gate_up_clamp is None else f32_bits(abs(gate_up_clamp)),
        arch,
    )
