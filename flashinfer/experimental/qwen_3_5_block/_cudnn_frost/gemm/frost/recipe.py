# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What one compiled gemm needs per call, settled when it compiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from math import prod

from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.buffers import DeviceView, dtype_name, init_word, is_contiguous, strided_fill_plan

from .dtypes import DTYPE_BYTES, _aux_align_reqs, _output_align_reqs, _pow2_floor, tensor_alignment
from .fusion_ir import FusionChain
from .layouts import fold_batch_layout

# An operand's three axes, named by role rather than by position.
AX_BATCH, AX_MN, AX_K = 0, 1, 2

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
    """``(stored_modulus, packing)`` for one operand's TMA 16-byte rule."""
    bits = 4 if dtype == "fp4_e2m1" else DTYPE_BYTES[dtype] * 8
    pack = 2 if (dtype == "fp4_e2m1" and contiguous_is_k) else 1
    return (128 // bits) // pack, pack


def expected_shape(rule, m: int, n: int) -> tuple:
    """One output's runtime shape, from the rule the build recorded."""
    return tuple(v if s == CONST else (m if s == FROM_M else n // v) for s, v in rule)


def _output_rule(spec, chain: FusionChain, *, physical_mn_swapped: bool = False) -> tuple:
    """How one output's shape follows from (M, N), as a per-axis rule."""
    batch = chain.matmul.batch
    if spec.is_quant_scale:
        rule = tuple((CONST, int(d)) for d in spec.dim)
        qi = int(spec.source.rsplit("_", 1)[1])
        if physical_mn_swapped and chain.quants[qi].scale_reorder == "F8_128x4":
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
    graph_order: bool = False  # normalized operands always bind in original graph-input coordinates

    def axes(self, shape, stride, graph_order: bool) -> "tuple | None":
        """Which axis holds which role, or None when the layout is not the major."""
        if self.graph_order or graph_order or (tuple(shape), tuple(stride)) == self.declared_layout:
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
class PerKScaleRule:
    """Prepared binding contract in caller coordinates, independent of SMEM layout."""

    axis: int
    dtype: str
    name: str = ""

    def check(self, buffer, k: int) -> None:
        expected = (1, 1, k) if self.axis == 2 else (1, k, 1)
        if tuple(buffer.shape) != expected:
            raise ValueError(f"per-K mainloop scale {self.name!r}: expected shape {expected}, got {tuple(buffer.shape)}")
        if buffer.stride()[self.axis] != 1:
            raise ValueError(f"per-K mainloop scale {self.name!r}: K axis {self.axis} must have unit stride")
        if dtype_name(buffer) != self.dtype:
            raise ValueError(f"per-K mainloop scale {self.name!r}: expected dtype {self.dtype}, got {dtype_name(buffer)}")


def _per_k_rule(ref, swapped: bool) -> PerKScaleRule | None:
    if ref.bcast_mode != "per_k":
        return None
    axis = 3 - ref.k_axis if swapped else ref.k_axis
    dtype = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}[ref.dtype]
    return PerKScaleRule(axis, dtype, ref.name)


@dataclass(frozen=True)
class Aux:
    index: int
    role: str
    align: int
    ref: Any  # TensorRef, for the fake-shape reshape
    mainloop_scale: PerKScaleRule | None = None
    transposed: bool = False


@dataclass(frozen=True)
class ScaleFactor:
    index: int
    role: str
    is_a: bool
    operand_at: int  # the A or B operand this one scales, as an index into inputs


def _batch_shape(shape, prefix):
    """Pin each prepared batch axis while allowing dynamic matrix extents."""
    shape = tuple(map(int, shape))
    if len(shape) != len(prefix) + 2 or shape[:-2] != prefix:
        raise ValueError(f"matrix batch view expected original batch axes {prefix}, got shape={shape}")
    return (prod(prefix), *shape[-2:])


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
    arg_plan: tuple
    stride_ins: tuple
    shared_layout: tuple
    seeds: tuple
    matrix_views: tuple = ()

    @property
    def a(self) -> Operand:
        return self.inputs[self.a_at]

    @property
    def b(self) -> Operand:
        return self.inputs[self.b_at]

    def operand_shape(self, index, shape):
        """Project a caller shape using this recipe's prepared batch axes."""
        for slot, prefix in self.matrix_views:
            if slot == index:
                return _batch_shape(shape, prefix)
        return shape

    def operand_views(self, operands, *, check_outputs=True):
        """Apply prepared metadata views, without allocating or copying device storage.

        Workspace sizing can omit write-layout validation; launch and diagnostic
        paths validate every output before any kernel or seed write.
        """
        if not self.matrix_views:
            return operands
        views = list(operands)
        outputs = {o.index for o in self.outputs} if check_outputs else ()
        for index, prefix in self.matrix_views:
            value = views[index]
            shape = _batch_shape(value.shape, prefix)
            fold_batch_layout(value.shape, value.stride(), writable=index in outputs)
            if tuple(value.shape) == shape:
                continue  # identity view preserves singleton pitches too
            if hasattr(value, "flatten_batch"):
                views[index] = value.flatten_batch()  # native strided metadata
            elif isinstance(value, DeviceView):
                views[index] = value.reshape(shape)  # contiguous by contract
            else:
                views[index] = value.view(shape)  # Torch view cannot copy
        return views

    def problem(self, operands, graph_order=None) -> tuple:
        """``((M, N, K), axes-per-input)``, located rather than assumed."""
        axes, bad = [], []
        rank = []
        for op in self.inputs:
            v = operands[op.index]
            if len(v.shape) != 3:
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
    """Every buffer's alignment must meet the width its role's access uses."""
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
    """Validate F8_128x4 weight SF before using its runtime expert stride."""
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
    """Validate scale-factor storage layout and capacity."""
    m, n, k = mnk
    k4 = ((k // recipe.block_size) + 3) // 4
    bad = []
    for sf in recipe.sf:
        v = operands[sf.index]
        if len(v.shape) != 3:
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
    """Validate reduction outputs before identity initialization."""
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
    """Do the extents agree with the problem size this call will run?"""
    reasons = [
        _shape_reject(recipe, operands, axes, mnk),
        _output_shape_reject(recipe, operands, mnk),
        _shared_layout_reject(recipe, operands),
        _seed_reject(recipe, operands),
    ]
    if recipe.block_size:
        reasons.append(_sf_blob_reject(recipe, operands, axes, mnk))
    _raise_first(reasons)


def check_split_k_shape(k: int, n: int, batch: int, slices: int, cta_k_elems: int, reducer_elems: int = 1) -> None:
    """Validate split-K extents before any writes, for either reduction mode."""
    tiles = (k + cta_k_elems - 1) // cta_k_elems
    if tiles < slices:
        raise ValueError(
            f"cudnn.frost gemm: split_k_slices={slices} exceeds the {tiles} K tile(s) of K={k}; " "the plan was built for a deeper K — rebuild for this shape"
        )
    if batch * slices > 65535:
        raise ValueError(f"cudnn.frost gemm: batch={batch} * split_k_slices={slices} exceeds the CUDA grid.z limit of 65535")
    if n % reducer_elems:
        raise ValueError(
            f"cudnn.frost gemm: N={n} is not a multiple of the reducer chunk ({reducer_elems} elements) " "this plan was built with — rebuild for this shape"
        )


def check_alignment(recipe: GemmRecipe, operands, axes) -> None:
    """Does every buffer meet the width its role's accesses were compiled for?"""
    _raise_first([_tma_reject(recipe, operands, axes), _align_reject(recipe, operands)])


def _declared_layout(tensor) -> tuple:
    """The ``(dim, stride)`` the graph declared, or ``((), ())`` when it cannot say."""
    try:
        return tuple(int(d) for d in tensor.get_dim()), tuple(int(s) for s in tensor.get_stride())
    except Exception:  # noqa: BLE001 -- an analyzer-synthesized ref has no dims
        return (), ()


def _operand(
    index: int,
    role: str,
    tensor,
    *,
    major: str,
    dtype: str,
    batch: int,
    is_b: bool,
    declared_is_b: bool | None = None,
    transposed=False,
    graph_order=False,
    matrix_batch=None,
) -> Operand:
    """Prepare one operand contract in the kernel ABI and original caller axes."""
    declared = _DECLARED_AXES["b" if (is_b if declared_is_b is None else declared_is_b) else "a"]
    if transposed:
        declared = (declared[0], declared[2], declared[1])
    layout = _declared_layout(tensor)
    if matrix_batch is not None:
        layout = fold_batch_layout(*layout)
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
        declared_layout=layout,
        declared=declared,
        dc=declared[contiguous],
        kc=KERNEL_AXES[contiguous],
        graph_order=graph_order,
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

    matrix_batches = {id(t): prefix for t, prefix in binding.matrix_batches}
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
            transposed=bool(binding.a_transposed and binding.a_transposed[i]),
            graph_order=bool(binding.a_transposed),
            matrix_batch=matrix_batches.get(id(t)),
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
            transposed=bool(binding.b_transposed and binding.b_transposed[i]),
            graph_order=bool(binding.b_transposed),
            matrix_batch=matrix_batches.get(id(t)),
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
        Aux(
            index=order[id(t)],
            role=f"aux {compiled.aux_names[i]!r}",
            align=aux_reqs[compiled.aux_names[i]],
            ref=ref,
            mainloop_scale=_per_k_rule(ref, swap_ab ^ bool(binding.aux_transposed and binding.aux_transposed[i])),
            transposed=bool(binding.aux_transposed and binding.aux_transposed[i]),
        )
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

    heads = [(op.index, None) for op in inputs] + [(s.index, None) for s in sf]
    outs = [(o.index, None) for o in outputs]
    auxs = [(x.index, x.ref) for x in aux]
    slots = compiled.tma_slots
    taps = [o for i, o in enumerate(outs) if i not in slots]
    tmas = [outs[i] for i in sorted(slots) if i < len(outs)]
    arg_plan = tuple(heads + taps + auxs + tmas)

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
        matrix_views=tuple((order[tid], prefix) for tid, prefix in matrix_batches.items() if tid in order),
    )
