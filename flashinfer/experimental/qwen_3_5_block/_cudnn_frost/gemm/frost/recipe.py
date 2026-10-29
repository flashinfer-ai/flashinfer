# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What one compiled gemm needs per call, settled when it compiles.

Operand roles, majors, packing factors, alignment requirements, output shape
rules, which outputs are reductions and what order the kernel takes its
parameters in are all fixed by the time cute hands back a launchable. A call
carries M, N, K, the strides and the pointers, and nothing else. This module
writes the first set down, so that no call re-derives them.

One reading of it RUNS: ``CompiledFusedGemm.lowered``, the closure ``_lower``
captures it into, where every check is a loop over tuples flat enough to need no
attribute lookup. The other only EXPLAINS -- ``CompiledFusedGemm.explain``, which
walks the operand structure through :func:`check_shapes` and
:func:`check_alignment` to name what is wrong with a call the first one refused,
and never launches anything. Two executors would have been two answers to what
the graph computes, and a differential between them cannot catch a misconception
they share, which is how the axis-order bug survived one.

So the rules here are written twice and the launch is written once: the fast
form pays per call and answers a bool, the readable form runs only on a call
that has already failed. If they ever disagree, ``explain`` finds nothing and
says so rather than returning quietly.

The field that makes one loop serve six flavors is :attr:`GemmRecipe.arg_plan`:
what differs between plain, aux, multi-output, multi-GEMM and block scale is
only which buffers the launch passes and in what order, so that is data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.buffers import init_word, is_contiguous, strided_fill_plan

from .dtypes import DTYPE_BYTES, _aux_align_reqs, _output_align_reqs, _pow2_floor, tensor_alignment
from .fusion_ir import FusionChain

# An operand's three axes, named by role rather than by position.
AX_BATCH, AX_MN, AX_K = 0, 1, 2

# cuDNN's matmul ABI declares A as [b, m, k] and B as [b, k, n]; this engine's
# own direct-call API takes B the way the kernel reads it, (b, n, k). Both
# describe the same memory, so an operand arrives in one of two axis orders and
# the description alone cannot always say which -- at N == K the two are
# identical tuples.
#
# The tie-break is the backend's own rule: a graph's tensor descriptor DEFINES
# the tensor and the variant pack supplies only a pointer. So a buffer whose
# (shape, stride) is the declaration is read as the declaration, which is what
# the backend computes from it (measured: a matmul whose B matches the declared
# [b, K, N] agrees with the backend to bf16 tolerance, and differs from the
# (b, N, K) reading by 65). Anything else is the caller's own labelling of the
# same memory, which is the direct-call order.
_DECLARED_AXES = {"a": (0, 1, 2), "b": (0, 2, 1)}
KERNEL_AXES = (0, 1, 2)

# How an output axis follows from the problem size.
CONST, FROM_M, FROM_N = 0, 1, 2

REDUCTION_INIT_VALUE = {
    "fp32": {
        "add": 0.0,
        "amax": 0.0,
        "max": -float("inf"),
        "min": float("inf"),
        "avg": 0.0,
        "norm1": 0.0,
        "norm2": 0.0,
        "mul": 1.0,
        "mul_no_zeros": 1.0,
    },
    "int32": {
        "add": 0,
        "amax": 0,
        "max": -(2**31),
        "min": 2**31 - 1,
        "norm1": 0,
    },
}


def contiguous_modulus(dtype: str, contiguous_is_k: bool) -> tuple:
    """``(stored_modulus, packing)`` for one operand's TMA 16-byte rule.

    TMA encodes the contiguous dimension in 16-byte units, so the LOGICAL extent
    must divide ``128 // bits``. A shape carries the STORED count, and fp4 packs
    two elements per slot along K -- so a buffer is checked against the quotient
    while a user wrote their shape against the product. Both the graph-time gate
    and the per-call one read this, which is the one number that could drift.
    """
    bits = 4 if dtype == "fp4_e2m1" else DTYPE_BYTES[dtype] * 8
    pack = 2 if (dtype == "fp4_e2m1" and contiguous_is_k) else 1
    return (128 // bits) // pack, pack


def expected_shape(rule, m: int, n: int) -> tuple:
    """One output's runtime shape, from the rule the build recorded."""
    return tuple(v if s == CONST else (m if s == FROM_M else n // v) for s, v in rule)


def _output_rule(spec, chain: FusionChain, *, physical_mn_swapped: bool = False) -> tuple:
    """How one output's shape follows from (M, N), as a per-axis rule.

    The single answer to a question that used to be asked in two places and
    disagreed: an fp4 dense output is (batch, M, N/2), and a hand-written launch
    path that pinned N instead rejected a legal call.
    """
    batch = chain.matmul.batch
    if spec.is_quant_scale:
        rule = tuple((CONST, int(d)) for d in spec.dim)
        qi = int(spec.source.rsplit("_", 1)[1])
        if physical_mn_swapped and chain.quants[qi].scale_reorder == "F8_128x4":
            # Reordered scales are physical atoms, unchanged by swapping the
            # logical quantization axis. Compact scale tables do transpose.
            return rule
    elif not spec.is_reduction:
        rule = ((CONST, batch), (FROM_M, 0), (FROM_N, 2 if spec.dtype == "fp4_e2m1" else 1))
    else:
        red_idx = int(spec.source.rsplit("_", 1)[1])
        if chain.reductions[red_idx].grouped_by_moe:
            rule = tuple((CONST, int(d)) for d in spec.dim)
        else:
            # A reduced axis collapses to 1; the rest follow the problem size.
            rule = (
                (CONST, 1 if spec.dim[0] == 1 else batch),
                (CONST, 1) if spec.dim[1] == 1 else (FROM_M, 0),
                (CONST, 1) if spec.dim[2] == 1 else (FROM_N, 1),
            )
    # The transformed IR is [B, new-M, new-N], but the user's unchanged buffer
    # still presents [B, old-M, old-N] = [B, new-N, new-M].
    return (rule[0], rule[2], rule[1]) if physical_mn_swapped else rule


@dataclass(frozen=True)
class Operand:
    """One A or B operand: what a call must satisfy, with the rest baked."""

    index: int  # position in the operand list the engine hands over
    role: str  # names this operand in a rejection message
    major: str
    kpack: int  # 2 when fp4 stores two elements per slot
    batch: int  # the extent the graph declares on the batch axis
    is_b: bool  # its non-K extent is N rather than M
    modulus: int  # the contiguous STORED extent must divide this
    pack: int  # ... and multiplying by this recovers the logical extent
    contiguous_role: int  # AX_K for a k-major operand, else AX_MN
    declared: tuple  # (batch, mn, k) axis positions, graph order
    declared_layout: tuple  # the (dim, stride) that order comes with
    dc: int  # where stride 1 lands in the graph's order
    kc: int  # ... and in the caller's

    def axes(self, shape, stride, graph_order: bool) -> "tuple | None":
        """Which axis holds which role, or None when the layout is not the major.

        ``graph_order`` says the pack described this slot FROM the graph (a bare
        address has no geometry of its own), so it is the declaration by
        construction. A buffer that reports the declaration is read as the
        declaration too -- that is what the backend computes from it, and a
        caller must not get a different answer for having landed on a python
        plan. Everything else is the caller's own labelling of the same memory.
        """
        if graph_order or (tuple(shape), tuple(stride)) == self.declared_layout:
            return self.declared if stride[self.dc] == 1 else None
        return KERNEL_AXES if stride[self.kc] == 1 else None


@dataclass(frozen=True)
class Output:
    index: int
    role: str
    rule: tuple
    align: int
    raw: bool  # the kernel takes only its pointer
    init: Any = None  # reduction identity, seeded before the kernel runs


@dataclass(frozen=True)
class Aux:
    index: int
    role: str
    align: int
    ref: Any  # TensorRef, for the fake-shape reshape


@dataclass(frozen=True)
class ScaleFactor:
    index: int
    role: str
    is_a: bool
    operand_at: int  # the A or B operand this one scales, as an index into inputs


@dataclass(frozen=True)
class GemmRecipe:
    inputs: tuple  # every A operand then every B operand, in kernel slot order
    outputs: tuple
    aux: tuple
    sf: tuple
    roles: tuple  # what occupies each operand position, for a missing-buffer message
    a_at: int  # M and K are read off inputs[a_at]
    b_at: int  # N off inputs[b_at]
    has_output_specs: bool
    block_size: "int | None"
    workspace_bytes: int
    multi_gemm: bool
    device: int  # the GPU whose SMEM depth / cluster count / SM the kernel is baked for
    batch: int  # the kernel's batch extent, pinned by the checks above the launch
    # The launch call as data. ``arg_plan`` is one entry per positional argument
    # after ``problem_size``: an operand index, plus the aux TensorRef whose fake
    # shape it reshapes to (None -- every other role -- means permute(1, 2, 0)).
    # ``stride_ins`` gives the positions in ``inputs`` whose permuted strides ride
    # in ``problem_size``, ahead of every output's. This is the one field that says
    # how the six flavors differ, which is why they differ in a table and not in
    # six launchers.
    arg_plan: tuple
    stride_ins: tuple
    # ``(leader, followers)`` groups whose strides the launch collapses to one,
    # as POSITIONS in ``inputs``. Positions and not operand indices, because one
    # buffer can occupy two roles -- ``matmul(A, A)`` binds a single slot as both
    # operands -- and a role is what carries an axis order.
    shared_layout: tuple
    # ``(output index, identity as a dtype-packed word, the dtype's byte width)``
    # per reduction output. The width is checked before the seed is written: the
    # word is 32 bits and the count is the buffer's numel, so a narrower element
    # would put the fill past the end of the caller's allocation.
    seeds: tuple

    @property
    def a(self) -> Operand:
        return self.inputs[self.a_at]

    @property
    def b(self) -> Operand:
        return self.inputs[self.b_at]

    def problem(self, operands, graph_order=None) -> tuple:
        """``((M, N, K), axes-per-input)``, located rather than assumed.

        Reading M off axis 1 assumes the caller laid the buffer out the way the
        kernel reads it, which a bare device address does not: the pack
        describes that one from the graph's declaration, which orders B the
        other way round. ``graph_order`` is the pack's per-operand answer to which
        it was, or None when every operand is the caller's own.
        """
        axes, bad = [], []
        rank = []
        for op in self.inputs:
            v = operands[op.index]
            if len(v.shape) != 3:
                # Everything below indexes three named axes, so a buffer with a
                # different rank has to be answered here and not by an
                # IndexError three frames down.
                rank.append(f"{op.role}: expected a rank-3 buffer, got shape={tuple(v.shape)}")
                axes.append(KERNEL_AXES)
                continue
            borrowed = bool(graph_order and graph_order[op.index])
            ax = op.axes(v.shape, v.stride(), borrowed)
            if ax is None:
                want = op.dc if borrowed else op.kc
                bad.append(
                    f"{op.role}: graph declares {op.major}-major (dim {want} contiguous) but the buffer has shape={tuple(v.shape)}, stride={tuple(v.stride())}"
                )
                ax = KERNEL_AXES
            axes.append(ax)
        if rank:
            raise ValueError("the kernel reads three axes off every operand: " + "; ".join(rank))
        if bad:
            raise ValueError("runtime operand layout does not match the layout the kernel was compiled for: " + "; ".join(bad))
        a, b = self.a, self.b
        a_ax, b_ax = axes[self.a_at], axes[self.b_at]
        a_shape, b_shape = operands[a.index].shape, operands[b.index].shape
        return (a_shape[a_ax[AX_MN]], b_shape[b_ax[AX_MN]], a_shape[a_ax[AX_K]] * a.kpack), tuple(axes)


def _tma_reject(recipe: GemmRecipe, operands, axes) -> "str | None":
    """TMA encodes the contiguous dimension in 16-byte units; a misaligned
    extent silently mis-strides every row past the first."""
    bad = []
    for op, ax in zip(recipe.inputs, axes):
        role = op.contiguous_role
        extent = operands[op.index].shape[ax[role]]
        if extent % op.modulus:
            # Report the LOGICAL extent and modulus: fp4 stores two elements per
            # slot, so "K % 32" is the rule a user wrote their shape against.
            name = "K" if role == AX_K else ("N" if op.is_b else "M")
            bad.append(f"{op.role} ({op.major}-major) requires {name} % {op.modulus * op.pack} == 0, got {name}={extent * op.pack}")
    if bad:
        return "TMA input contiguous dimensions must be 16-byte aligned: " + "; ".join(bad)
    return None


def _shape_reject(recipe: GemmRecipe, operands, axes, mnk) -> "str | None":
    """Every operand must agree with the M/N/K read off the first A and B --
    the kernel walks the inferred K on all of them."""
    m, n, k = mnk
    bad = []
    for op, ax in zip(recipe.inputs, axes):
        shape = operands[op.index].shape
        want = (op.batch, n if op.is_b else m, k // op.kpack)
        got = (shape[ax[AX_BATCH]], shape[ax[AX_MN]], shape[ax[AX_K]])
        if got != want:
            bad.append(f"{op.role}: expected (batch, M|N, K) = {want}, got {got}")
    if bad:
        return f"runtime operand shapes disagree with the inferred problem size (M={m}, N={n}, K={k}): " + "; ".join(bad)
    return None


def _output_shape_reject(recipe: GemmRecipe, operands, mnk) -> "str | None":
    m, n, _ = mnk
    bad = []
    for out in recipe.outputs:
        shape = tuple(operands[out.index].shape)
        want = expected_shape(out.rule, m, n)
        if shape != want:
            bad.append(f"{out.role}: expected {want}, got {shape}")
    if bad:
        return "runtime tensors must be rank-3 with shapes matching the graph: " + "; ".join(bad)
    return None


def _align_reject(recipe: GemmRecipe, operands) -> "str | None":
    """Every buffer's alignment must meet the width its role's access uses.

    Inputs and scale factors are TMA-loaded, so only the base pointer is at
    stake; outputs and aux are stored and loaded directly, so their strides and
    contiguous extent bound the vector too.
    """
    bad = []
    for item in recipe.inputs + recipe.sf:
        ptr = int(operands[item.index].data_ptr())
        align = _pow2_floor(ptr)
        if align < 16:
            bad.append(f"{item.role}: alignment {align}B < required 16B (ptr=0x{ptr:x})")
    for item in recipe.outputs + recipe.aux:
        v = operands[item.index]
        ptr = int(v.data_ptr())
        align = tensor_alignment(tuple(v.shape), tuple(v.stride()), v.element_size(), ptr=ptr)
        if align < item.align:
            bad.append(f"{item.role}: alignment {align}B < required {item.align}B (ptr=0x{ptr:x})")
    if bad:
        return "runtime tensor alignment is below the kernel's compiled requirement: " + "; ".join(bad)
    return None


def moe_weight_sf_reject(named_planes) -> str | None:
    """Validate F8_128x4 weight SF before using its runtime expert stride.

    Entries are (role, buffer, required_plane_bytes, num_experts). Each inner
    plane is a dense byte blob, possibly larger than the blocked payload;
    its axes may be permuted. Padding between planes is allowed. Requiring
    an aligned pitch at least as large as the plane's span keeps every expert
    read inside its own observed plane, including the last expert.
    """
    for role, buf, required, experts in named_planes:
        shape, strides = tuple(buf.shape), tuple(buf.stride())
        reason = None
        if len(shape) != 3 or len(strides) != 3 or shape[0] != experts or any(s < 1 for s in shape):
            reason = f"expected rank-3 shape [{experts}, ..., ...], got {shape}"
        elif buf.element_size() != 1:
            reason = "expected byte-sized scale factors"
        else:
            span = 1
            for stride, extent in sorted((st, s) for s, st in zip(shape[1:], strides[1:]) if s > 1):
                if stride != span:
                    reason = f"each expert plane must be a dense byte run, got shape {shape} stride {strides}"
                    break
                span *= extent
            if reason is None:
                if span < required:
                    reason = f"each expert plane has {span}B but the kernel reads {required}B"
                elif strides[0] < span or strides[0] % 16:
                    reason = f"expert stride {strides[0]}B must be a multiple of 16 and at least the {span}B plane span"
        if reason is not None:
            return f"block-scale F8_128x4 weight scale factors must have packed expert planes: {role}: {reason}"
    return None


def _sf_blob_reject(recipe: GemmRecipe, operands, axes, mnk) -> "str | None":
    """A block-scale SF reaches the kernel as a base pointer plus a layout the
    template re-synthesizes from M/N/K, so a blob that is not one dense byte run
    of at least the required size is read out of bounds with no fault."""
    m, n, k = mnk
    k4 = ((k // recipe.block_size) + 3) // 4
    bad = []
    for sf in recipe.sf:
        v = operands[sf.index]
        if len(v.shape) != 3:
            # It reaches the kernel through the same rank-3 relabelling as every
            # other head, so a different rank is answered here rather than by
            # the permute three frames down.
            bad.append(f"{sf.role}: expected a rank-3 buffer, got shape={tuple(v.shape)}")
            continue
        op = recipe.inputs[sf.operand_at]
        rows = m if sf.is_a else n
        batch = operands[op.index].shape[axes[sf.operand_at][AX_BATCH]]
        required = 512 * k4 * ((rows + 127) // 128) * int(batch)
        span = 1 + sum((int(s) - 1) * int(st) for s, st in zip(v.shape, v.stride()))
        if int(v.numel()) != span:
            bad.append(f"{sf.role} shape {tuple(v.shape)} stride {tuple(v.stride())} is not a dense byte run")
            continue
        have = int(v.numel()) * v.element_size()
        if have < required:
            bad.append(
                f"{sf.role} is {have}B but the kernel reads {required}B ({required // 512} atoms of 128 rows x 4 SF-K) — was it produced by to_blocked()?"
            )
    if bad:
        return "block-scale F8_128x4 scale factors must be a packed blob: " + "; ".join(bad)
    return None


def _shared_layout_reject(recipe: GemmRecipe, operands) -> "str | None":
    """Block-scale multi-GEMM sends ONE A stride triple for every A operand, so
    operands that share it must actually be laid out alike."""
    bad = []
    for lead, followers in recipe.shared_layout:
        want = tuple(operands[recipe.inputs[lead].index].stride())
        for j in followers:
            got = tuple(operands[recipe.inputs[j].index].stride())
            if got != want:
                bad.append(f"{recipe.inputs[j].role} has stride {got} where {recipe.inputs[lead].role} has {want}")
    if bad:
        return "this kernel sends one stride triple per operand pool, so the operands in a pool must share a layout: " + "; ".join(bad)
    return None


def _seed_reject(recipe: GemmRecipe, operands) -> "str | None":
    """A reduction output is seeded with its identity before the kernel runs, so
    the seed's own preconditions are the call's -- and they are checked before
    the first byte is written, because a seed that fails halfway has already
    scribbled on a caller's buffer."""
    bad = []
    for index, _word, elem_bytes in recipe.seeds:
        v = operands[index]
        got = v.element_size()
        if got != elem_bytes:
            bad.append(f"{recipe.roles[index]}: the reduction accumulates in {elem_bytes}-byte elements but this buffer stores {got}-byte ones")
            continue
        shape, stride = tuple(v.shape), tuple(v.stride())
        if not is_contiguous(shape, stride) and strided_fill_plan(shape, stride) is None:
            bad.append(f"{recipe.roles[index]}: shape {shape} stride {stride} writes some element twice")
    if bad:
        return "a reduction output must be seedable: " + "; ".join(bad)
    return None


def _raise_first(reasons) -> None:
    for reason in reasons:
        if reason is not None:
            raise ValueError(reason)


def check_shapes(recipe: GemmRecipe, operands, mnk, axes) -> None:
    """Do the extents agree with the problem size this call will run?

    The kernel's M/N/K are symbolic, so one plan serves many problem sizes and
    the call's own extents are what has to hold together: every operand against
    the M/N/K read off the first A and B, every output against the shape rule
    the build recorded, and a block-scale blob against the size the template
    re-synthesizes.
    """
    reasons = [
        _shape_reject(recipe, operands, axes, mnk),
        _output_shape_reject(recipe, operands, mnk),
        _shared_layout_reject(recipe, operands),
        _seed_reject(recipe, operands),
    ]
    if recipe.block_size:
        reasons.append(_sf_blob_reject(recipe, operands, axes, mnk))
    _raise_first(reasons)


def check_alignment(recipe: GemmRecipe, operands, axes) -> None:
    """Does every buffer meet the width its role's accesses were compiled for?

    Two rules, both about 16 bytes and neither about shape: TMA encodes the
    contiguous dimension in 16-byte units, so its extent has a modulus; and each
    buffer's base (plus, for a stored output, its stride and contiguous extent)
    bounds the widest vector the kernel can issue. Below either, the kernel does
    not fault -- it mis-strides or reads past the end.
    """
    _raise_first([_tma_reject(recipe, operands, axes), _align_reject(recipe, operands)])


def _declared_layout(tensor) -> tuple:
    """The ``(dim, stride)`` the graph declared, or ``((), ())`` when it cannot say.

    An empty pair never equals a runtime description, so an operand whose
    declaration is unreadable is simply read in the direct-call order.
    """
    try:
        return tuple(int(d) for d in tensor.get_dim()), tuple(int(s) for s in tensor.get_stride())
    except Exception:  # noqa: BLE001 -- an analyzer-synthesized ref has no dims
        return (), ()


def _operand(index: int, role: str, tensor, *, major: str, dtype: str, batch: int, is_b: bool, declared_is_b: bool | None = None) -> Operand:
    declared = _DECLARED_AXES["b" if (is_b if declared_is_b is None else declared_is_b) else "a"]
    contiguous = AX_K if major == "k" else AX_MN
    modulus, pack = contiguous_modulus(dtype, contiguous == AX_K)
    return Operand(
        index=index,
        role=role,
        major=major,
        kpack=2 if dtype == "fp4_e2m1" else 1,
        batch=batch,
        is_b=is_b,
        modulus=modulus,
        pack=pack,
        contiguous_role=contiguous,
        declared_layout=_declared_layout(tensor),
        declared=declared,
        dc=declared[contiguous],
        kc=KERNEL_AXES[contiguous],
    )


def build(compiled) -> GemmRecipe:
    """Read one compiled gemm into the table its call path runs off."""
    chain: FusionChain = compiled.chain
    mm = chain.matmul
    binding = compiled.binding
    swap_ab = compiled.config.swap_ab
    order = {}
    for i, t in enumerate(binding.bound_tensors()):
        order.setdefault(id(t), i)

    inputs = [
        _operand(
            order[id(t)],
            f"A operand[{i}]",
            t,
            major=mm.a_major,
            dtype=mm.a_dtype,
            batch=mm.a_batch,
            is_b=False,
            declared_is_b=swap_ab,
        )
        for i, t in enumerate(binding.a_operands)
    ] + [
        _operand(
            order[id(t)],
            f"B operand[{i}]",
            t,
            major=mm.b_major,
            dtype=mm.b_dtype,
            batch=mm.b_batch,
            is_b=True,
            declared_is_b=not swap_ab,
        )
        for i, t in enumerate(binding.b_operands)
    ]

    out_reqs = _output_align_reqs(chain, compiled.tma_slots, vec_bytes=compiled.vec_bytes_epi)
    aux_reqs = _aux_align_reqs(chain, vec_bytes=compiled.vec_bytes_epi)
    outputs, seeds = [], []
    for i, (spec, t) in enumerate(zip(chain.outputs, binding.outputs)):
        init = None
        if spec.is_reduction:
            red = chain.reductions[int(spec.source.rsplit("_", 1)[1])]
            init = REDUCTION_INIT_VALUE[red.compute_dtype][red.mode]
            seeds.append((order[id(t)], init_word(red.compute_dtype, init), DTYPE_BYTES[red.compute_dtype]))
        outputs.append(
            Output(
                index=order[id(t)],
                role=spec.source,
                rule=_output_rule(spec, chain, physical_mn_swapped=swap_ab),
                align=out_reqs[i],
                raw=bool(spec.is_reduction or spec.is_quant_scale),
                init=init,
            )
        )

    aux = tuple(
        Aux(index=order[id(t)], role=f"aux {compiled.aux_names[i]!r}", align=aux_reqs[compiled.aux_names[i]], ref=ref)
        for i, (t, ref) in enumerate(zip(binding.aux, chain.aux_tensors))
    )
    na = len(binding.a_operands)
    sf = tuple(
        [ScaleFactor(index=order[id(t)], role=f"SFA[{i}]", is_a=True, operand_at=i) for i, t in enumerate(binding.sfa_operands)]
        + [ScaleFactor(index=order[id(t)], role=f"SFB[{j}]", is_a=False, operand_at=na + j) for j, t in enumerate(binding.sfb_operands)]
    )

    roles = [f"bound tensor {i}" for i in range(len(binding.bound_tensors()))]
    for item in (*inputs, *outputs, *aux, *sf):
        roles[item.index] = item.role

    # The kernel's signature, in the order the launchers pass it: every distinct
    # A, every distinct B, their scale factors, then the outputs and the aux --
    # except under a TMA-store epilogue, where the single dense output binds the
    # template's trailing TMA-only parameter and so goes last.
    heads = [(op.index, None) for op in inputs] + [(s.index, None) for s in sf]
    outs = [(o.index, None) for o in outputs]
    auxs = [(x.index, x.ref) for x in aux]
    # Taps, aux, then the TMA-C slots in slot order -- an output on the TMA
    # surface binds a trailing TMA-only kernel parameter.
    slots = compiled.tma_slots
    taps = [o for i, o in enumerate(outs) if i not in slots]
    tmas = [outs[i] for i in sorted(slots) if i < len(outs)]
    arg_plan = tuple(heads + taps + auxs + tmas)

    # Block-scale multi-GEMM sends ONE A and ONE B stride triple and requires the
    # rest to match it; every other flavor sends each operand's own.
    grouped = bool(chain.is_multi_gemm and compiled.block_scale)
    stride_ins = (0, na) if grouped else tuple(range(len(inputs)))
    shared_layout = ()
    if grouped:
        pools = (tuple(range(na)), tuple(range(na, len(inputs))))
        shared_layout = tuple((pool[0], pool[1:]) for pool in pools if len(pool) > 1)

    return GemmRecipe(
        inputs=tuple(inputs),
        outputs=tuple(outputs),
        aux=aux,
        sf=sf,
        roles=tuple(roles),
        a_at=0,
        b_at=na,
        has_output_specs=bool(chain.output_specs),
        block_size=chain.block_scale.block_size if compiled.block_scale else None,
        workspace_bytes=int(getattr(compiled, "workspace_bytes", 0) or 0),
        multi_gemm=bool(chain.is_multi_gemm),
        device=int(compiled.device),
        batch=int(outputs[0].rule[0][1] if chain.output_specs else max(mm.a_batch, mm.b_batch)),
        arg_plan=arg_plan,
        stride_ins=stride_ins,
        shared_layout=shared_layout,
        seeds=tuple(seeds),
    )
