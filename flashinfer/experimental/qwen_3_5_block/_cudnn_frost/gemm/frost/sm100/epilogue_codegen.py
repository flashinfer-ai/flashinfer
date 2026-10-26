# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Codegen: FusionChain -> string snippets that fill the kernel template's
hook slots (aux_views + per-vector epilogue). compiler.py merges them via
string replacement at the `# FUSION_HOOK:*` markers."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import gcd

from ..dtypes import DTYPE_BYTES, DTYPE_TO_CUTLASS, _allowed_vsize, _aux_align_reqs, _compute_output_vec_bytes, _output_align_reqs, dense_output_layout
from ..fusion_ir import (
    BlockQuantizeSpec,
    Dtype,
    FusionChain,
    FusionOp,
    MoeSwapAbSpec,
    ReductionSpec,
    TensorRef,
    gemm_index,
    is_gemm_source,
)


@dataclass(frozen=True)
class EpilogueSnippets:
    aux_views: str  # inserted at INJECT_AUX_VIEWS
    epilogue: str  # inserted at INJECT_EPILOGUE
    kernel_params: list[str]  # extra kernel-signature param decls
    host_args: list[str]  # extra host-side arg names for .launch
    # Tap plumbing: one entry per tap-slot output (STG: every output; TMA:
    # all but the first) numbered 0..N-1 (templates reference mC_tap_<i>).
    tap_kernel_params: list[str] = field(default_factory=list)
    tap_host_params: list[str] = field(default_factory=list)
    tap_host_pass: list[str] = field(default_factory=list)
    tap_compile_fakes: list[str] = field(default_factory=list)
    tap_compile_pass: list[str] = field(default_factory=list)
    tap_ptr_binds: list[str] = field(default_factory=list)
    tap_constants: list[str] = field(default_factory=list)  # vec_bytes_tap_<i> assignments
    # Mainloop-fusion transforms (INJECT_MAINLOOP_A/B in the 12-warp templates):
    # given ml_vec_<a|b>, compute the op chain in fp32 and define ml_out_<a|b>
    # (cast back) which the template stores in place. "pass" = no fusion.
    mainloop_transform_a: str = "pass"
    mainloop_transform_b: str = "pass"


def _aux_ptr_var(name: str) -> str:
    return f"_aux_{name}_ptr"


def _aux_prefetch_var(name: str) -> str:
    return f"_aux_{name}_pre"


def _compute_cast(var: str, dtype: Dtype, tag: str) -> tuple[list[str], str]:
    """Cast a running vector/scalar to the op's compute dtype."""
    new = f"_c_{tag}"
    return [f"{new} = ({var}).to({DTYPE_TO_CUTLASS[dtype]})"], new


def _compute_literal(dtype: Dtype, value: float | int) -> str:
    if dtype == "int32":
        return f"cutlass.Int32({int(value)})"
    return f"cutlass.Float32({float(value)})"


# Aux load expressions (string forms used inside the inner loop)


def _index_term(var: str, stride: int) -> str:
    if stride == 1:
        return var
    return f"{var} * {stride}"


def _aux_index_expr(aux: TensorRef, *, row_var: str = "row", col_var: str = "col_j") -> str:
    """Linear element offset for the current epilogue location. Extent-1
    dims are broadcast and don't contribute to the offset."""
    if len(aux.dim) == 1:
        axes = ((aux.dim[0], aux.stride[0], col_var),)
    elif len(aux.dim) == 2:
        axes = (
            (aux.dim[0], aux.stride[0], row_var),
            (aux.dim[1], aux.stride[1], col_var),
        )
    elif len(aux.dim) == 3:
        lead_var = "group_idx" if aux.grouped_by_moe else "tile_l"
        axes = (
            (aux.dim[0], aux.stride[0], lead_var),
            (aux.dim[1], aux.stride[1], row_var),
            (aux.dim[2], aux.stride[2], col_var),
        )
    else:
        raise ValueError(f"unsupported aux rank {len(aux.dim)} for {aux.name!r}")

    terms = [_index_term(var, stride) for dim, stride, var in axes if dim != 1]
    return " + ".join(terms) if terms else "0"


def _elem_coord(e: int) -> tuple[str, str]:
    """The (row, col) of element ``e``. Every arm hands the snippet the SAME
    row-per-lane fragment, so these are the fragment's own coordinates."""
    return ("row", f"col_j + {e}")


def tma_out_value(j: int) -> str:
    """The variable holding the j-th TMA-stored output's value. Slot 0 keeps the
    legacy `vec_out`, so a single-TMA-output render is unchanged."""
    return "vec_out" if j == 0 else f"_tma_out_{j}"


def tma_out_ready_marker(j: int) -> str:
    """Codegen marker immediately after TMA output ``j`` is ready."""
    return f"# @@FROST_TMA_OUT_{j}_READY@@"


def _aux_reads_row(aux: TensorRef) -> bool:
    return len(aux.dim) >= 2 and aux.dim[-2] != 1


def _bounded_aux_prelude(chain: FusionChain, row_bound: str, col_bound: str, on_tma_arm: bool) -> list[str]:
    """The TMA arm hands the snippet a SUBTILE base with neither an N nor an M
    bound -- the store is clipped by the descriptor's global extent, so nothing
    downstream needs one, but a `per_col` / `per_elem` LDG at `col_j + k` is a
    real read past the aux allocation (N=144, cta_tile_n=128, epi_n=32: the
    subtile at col=224 reads columns 224..255, and at row M-1 that is 80
    elements past a `per_elem` tensor's end). Take the whole-chunk vector load
    when the window is inside both extents -- the case on every subtile but the
    last -- and otherwise read element-wise at a CLAMPED coordinate: the lanes
    past the extent land on the last valid element, whose value is never stored."""
    lines: list[str] = []
    for aux in chain.aux_tensors:
        if aux.bcast_mode not in ("per_col", "per_elem") or (not on_tma_arm and aux.stride[-1] == 1):
            continue
        n, ptr = aux.name, _aux_ptr_var(aux.name)
        lines.append(f"_auxt_{n} = cute.make_rmem_tensor(vsize, {DTYPE_TO_CUTLASS[aux.dtype]})")
        lines.append(f"_auxv_{n} = _auxt_{n}.load().to_vector()")
        cond = [f"col_j + vsize <= {col_bound}", str(aux.stride[-1] == 1)]
        if _aux_reads_row(aux):
            cond.append(f"row < {row_bound}")
        lines.append(f"if {' & '.join(f'({c})' for c in cond)}:")
        lines.append(f"    _auxv_{n} = ({ptr} + {_aux_index_expr(aux)}).load(count=vsize, alignment=ALIGN_AUX_{n})")
        lines.append("else:")
        row_var = f"_auxr_{n}"
        if _aux_reads_row(aux):
            lines.append(f"    {row_var} = cute.math.min(cutlass.Int32(row), cutlass.Int32({row_bound}) - 1)")
        else:
            row_var = "row"
        lines.append("    for _auxk in cutlass.range_constexpr(vsize):")
        lines.append(f"        _auxc_{n} = cute.math.min(cutlass.Int32(col_j) + _auxk, cutlass.Int32({col_bound}) - 1)")
        idx = _aux_index_expr(aux, row_var=row_var, col_var=f"_auxc_{n}")
        lines.append(f"        _auxt_{n}[_auxk] = ({ptr} + {idx}).load()")
        lines.append(f"    _auxv_{n} = _auxt_{n}.load().to_vector()")
    return lines


def _scatter_aux_prelude(chain: FusionChain) -> list[str]:
    """Pointwise nodes consume SCATTER's logical output coordinates."""
    swapped = isinstance(chain.moe, MoeSwapAbSpec)
    row = "row" if swapped else "moe_scatter_row(token_index, token_ks, row, group_end, M, moe_top_k)"
    col = "moe_scatter_row(token_index, token_ks, col_j + _auxk, group_end, N, moe_top_k)" if swapped else "col_j + _auxk"
    lines = []
    for aux in chain.aux_tensors:
        if aux.bcast_mode == "per_row":
            idx = _aux_index_expr(aux, row_var=f"cute.math.min(cutlass.Int32({row}), cutlass.Int32(M) - 1)")
            lines.append(f"{_aux_prefetch_var(aux.name)} = ({_aux_ptr_var(aux.name)} + {idx}).load()")
        if aux.bcast_mode not in ("per_col", "per_elem"):
            continue
        n = aux.name
        idx = _aux_index_expr(
            aux, row_var=f"cute.math.min(cutlass.Int32({row}), cutlass.Int32(M) - 1)", col_var=f"cute.math.min(cutlass.Int32({col}), cutlass.Int32(N) - 1)"
        )
        lines += [
            f"_auxt_{n} = cute.make_rmem_tensor(vsize, {DTYPE_TO_CUTLASS[aux.dtype]})",
            "for _auxk in cutlass.range_constexpr(vsize):",
            f"    _auxt_{n}[_auxk] = ({_aux_ptr_var(n)} + {idx}).load()",
            f"_auxv_{n} = _auxt_{n}.load().to_vector()",
        ]
    return lines


def _aux_load_expr(aux: TensorRef, compute_dtype: Dtype, like_var: str, *, bounded: bool = False) -> str:
    """Expression yielding aux value(s) as a length-`vsize` vector in the op's
    compute dtype, matching ``like_var``'s dtype."""
    idx = _aux_index_expr(aux)
    cast = DTYPE_TO_CUTLASS[compute_dtype]
    if aux.bcast_mode == "scalar":
        # scalar prefetched in aux_views, broadcast to vec
        return f"cutlass.full_like({like_var}, {_aux_prefetch_var(aux.name)}.to({cast}))"
    if aux.bcast_mode == "per_row":
        # per-row scalar prefetched in aux_views, broadcast to vec
        return f"cutlass.full_like({like_var}, {_aux_prefetch_var(aux.name)}.to({cast}))"
    if aux.bcast_mode in ("per_col", "per_elem"):
        if bounded:
            return f"_auxv_{aux.name}.to({cast})"
        return f"({_aux_ptr_var(aux.name)} + {idx}).load(count=vsize, " f"alignment=ALIGN_AUX_{aux.name}).to({cast})"
    raise AssertionError(f"unknown bcast_mode {aux.bcast_mode!r}")


# Per-op emitter

# exp(y) = exp2(y * LOG2E); tanh(x) = 1 - 2/(exp2(2x*LOG2E) + 1).
_LOG2E = "cutlass.Float32(1.4426950408889634)"
_TWO_LOG2E = "cutlass.Float32(2.8853900817779268)"


def _tanh_expr(v: str) -> str:
    """Vector tanh via exp2/rcp: 1 - 2/(exp2(2x*log2e) + 1). Native
    ``cute.math.tanh`` aborts under vector lowering; exp2/rcp with fastmath
    lower fine and saturate cleanly (exp2→inf ⇒ tanh→1; exp2→0 ⇒ tanh→-1)."""
    one = f"cutlass.full_like({v}, cutlass.Float32(1.0))"
    two = f"cutlass.full_like({v}, cutlass.Float32(2.0))"
    e2 = f"cute.math.exp2({v} * cutlass.full_like({v}, {_TWO_LOG2E}), fastmath=True)"
    return f"({one} - {two} * cute.math.rcp({e2} + {one}, approx=True, ftz=True))"


def _fl(var: str, c: float) -> str:
    return f"cutlass.full_like({var}, cutlass.Float32({float(c)}))"


def _step01(d_expr: str, tag: str) -> tuple[list[str], str]:
    """0/1 vector for [d >= 0], exact at the boundary: min(1, relu(floor(d)+1))."""
    d, v = f"_sd{tag}", f"_st{tag}"
    lines = [
        f"{d} = ({d_expr})",
        f"{v} = cute.math.min(cute.math.max(cute.math.floor({d}) + {_fl(d, 1.0)}, {_fl(d, 0.0)}), {_fl(d, 1.0)})",
    ]
    return lines, v


def _nonzero01(x_expr: str, tag: str) -> tuple[list[str], str]:
    """0/1 vector for [x != 0]."""
    x, v = f"_nz{tag}", f"_nzv{tag}"
    lines = [f"{x} = ({x_expr})"]
    l1, sp = _step01(x, f"{tag}p")
    l2, sn = _step01(f"-{x}", f"{tag}n")
    lines += l1 + l2 + [f"{v} = {_fl(x, 1.0)} - {sp} * {sn}"]
    return lines, v


def _sigmoid_lines(x_var: str, tag: str) -> tuple[list[str], str]:
    v = f"_sg{tag}"
    return [
        f"{v} = cute.math.rcp({_fl(x_var, 1.0)} + cute.math.exp2(-{x_var} * cutlass.full_like({x_var}, {_LOG2E}), fastmath=True), approx=True, ftz=True)"
    ], v


_CMP_OPS = ("cmp_eq", "cmp_neq", "cmp_gt", "cmp_ge", "cmp_lt", "cmp_le")
_LOGIC_OPS = ("logical_and", "logical_or")
_BACKWARD_OPS = (
    "relu_backward",
    "leaky_relu_backward",
    "swish_backward",
    "sigmoid_backward",
    "tanh_backward",
    "elu_backward",
    "gelu_backward",
    "gelu_tanh_backward",
    "softplus_backward",
)


def _emit_binary_ext(op: FusionOp, a_expr: str, b_expr: str, idx: int, new: str) -> tuple[list[str], str]:
    """cmp / logical / mod / activation-backward binary ops on (a, b) = for
    backward ops (loss, activation input)."""
    a, b = f"_ba{idx}", f"_bb{idx}"
    lines = [f"{a} = ({a_expr})", f"{b} = ({b_expr})"]
    attrs = dict(op.attrs)

    if op.op in _CMP_OPS:
        if op.op == "cmp_le":
            step, v = _step01(f"{b} - {a}", f"_{idx}")
            return lines + step + [f"{new} = {v}"], new
        if op.op == "cmp_ge":
            step, v = _step01(f"{a} - {b}", f"_{idx}")
            return lines + step + [f"{new} = {v}"], new
        if op.op == "cmp_lt":
            step, v = _step01(f"{a} - {b}", f"_{idx}")
            return lines + step + [f"{new} = {_fl(a, 1.0)} - {v}"], new
        if op.op == "cmp_gt":
            step, v = _step01(f"{b} - {a}", f"_{idx}")
            return lines + step + [f"{new} = {_fl(a, 1.0)} - {v}"], new
        l1, v1 = _step01(f"{a} - {b}", f"_{idx}e1")
        l2, v2 = _step01(f"{b} - {a}", f"_{idx}e2")
        if op.op == "cmp_eq":
            return lines + l1 + l2 + [f"{new} = {v1} * {v2}"], new
        return lines + l1 + l2 + [f"{new} = {_fl(a, 1.0)} - {v1} * {v2}"], new

    if op.op in _LOGIC_OPS:
        la, na = _nonzero01(a, f"_{idx}a")
        lb, nb = _nonzero01(b, f"_{idx}b")
        if op.op == "logical_and":
            return lines + la + lb + [f"{new} = {na} * {nb}"], new
        return lines + la + lb + [f"{new} = {na} + {nb} - {na} * {nb}"], new

    if op.op == "mod":
        q = f"_mq{idx}"
        lines.append(f"{q} = {a} / {b}")
        lp, sp = _step01(q, f"_{idx}mp")
        ln, sn = _step01(f"-{q}", f"_{idx}mn")
        return (
            lines
            + lp
            + ln
            + [
                f"_mt{idx} = cute.math.floor(cute.math.abs({q})) * ({sp} - {sn})",
                f"{new} = {a} - {b} * _mt{idx}",
            ],
            new,
        )

    if op.op in ("relu_backward", "leaky_relu_backward"):
        slope = attrs.get("negative_slope")
        lower = attrs.get("lower_clip", 0.0)
        upper = attrs.get("upper_clip")
        l1, s1 = _step01(f"{_fl(b, lower)} - {b}", f"_{idx}r1")
        lines += l1 + [f"_rm{idx} = {_fl(b, 1.0)} - {s1}"]
        m = f"_rm{idx}"
        if upper is not None:
            l2, s2 = _step01(f"{b} - {_fl(b, upper)}", f"_{idx}r2")
            lines += l2 + [f"_rmu{idx} = {m} * ({_fl(b, 1.0)} - {s2})"]
            m = f"_rmu{idx}"
        if slope is not None:
            lines.append(f"_rf{idx} = {m} + {_fl(b, slope)} * ({_fl(b, 1.0)} - {m})")
            m = f"_rf{idx}"
        return lines + [f"{new} = {a} * {m}"], new

    if op.op == "sigmoid_backward":
        ls, sg = _sigmoid_lines(b, f"_{idx}")
        return lines + ls + [f"{new} = {a} * {sg} * ({_fl(b, 1.0)} - {sg})"], new

    if op.op == "tanh_backward":
        lines.append(f"_tb{idx} = {_tanh_expr(b)}")
        return lines + [f"{new} = {a} * ({_fl(b, 1.0)} - _tb{idx} * _tb{idx})"], new

    if op.op == "softplus_backward":
        ls, sg = _sigmoid_lines(b, f"_{idx}")
        return lines + ls + [f"{new} = {a} * {sg}"], new

    if op.op == "elu_backward":
        lm, m = _step01(f"-{b}", f"_{idx}")
        return (
            lines
            + lm
            + [
                f"_ee{idx} = cute.math.exp2(cute.math.min({b}, {_fl(b, 0.0)}) * cutlass.full_like({b}, {_LOG2E}), fastmath=True)",
                f"{new} = {a} * (({_fl(b, 1.0)} - {m}) + {m} * _ee{idx})",
            ],
            new,
        )

    if op.op == "gelu_backward":
        return (
            lines
            + [
                f"_gc{idx} = {_fl(b, 0.5)} * ({_fl(b, 1.0)} + cute.math.erf({b} * {_fl(b, 0.7071067811865476)}))",
                f"_gp{idx} = {b} * cute.math.exp2(-{b} * {b} * {_fl(b, 0.7213475204444817)}, fastmath=True) * {_fl(b, 0.3989422804014327)}",
                f"{new} = {a} * (_gc{idx} + _gp{idx})",
            ],
            new,
        )

    if op.op == "gelu_tanh_backward":
        lines += [
            f"_gu{idx} = {_fl(b, 0.7978845608028654)} * ({b} + {_fl(b, 0.044715)} * {b} * {b} * {b})",
            f"_gt{idx} = {_tanh_expr(f'_gu{idx}')}",
            f"_gd{idx} = {_fl(b, 0.7978845608028654)} * ({_fl(b, 1.0)} + {_fl(b, 0.134145)} * {b} * {b})",
        ]
        return (
            lines + [f"{new} = {a} * ({_fl(b, 0.5)} * ({_fl(b, 1.0)} + _gt{idx}) + {_fl(b, 0.5)} * {b} * ({_fl(b, 1.0)} - _gt{idx} * _gt{idx}) * _gd{idx})"],
            new,
        )

    if op.op == "swish_backward":
        beta = attrs.get("swish_beta", 1.0)
        lines.append(f"_sx{idx} = {b} * {_fl(b, beta)}")
        ls, sg = _sigmoid_lines(f"_sx{idx}", f"_{idx}")
        return lines + ls + [f"{new} = {a} * ({sg} + _sx{idx} * {sg} * ({_fl(b, 1.0)} - {sg}))"], new

    raise AssertionError(f"unhandled binary-ext op {op.op!r}")


def _emit_op(
    op: FusionOp,
    prev: str,
    idx: int,
    aux_loads: dict[str, str],
    other_in_chain: str | None = None,
    third_in_chain: str | None = None,
    vsize: int = 32,
) -> tuple[list[str], str]:
    """Emit lines computing this op, given the previous-step var name.
    ``other_in_chain`` = second operand var for fan-in binary ops (else None).
    Returns (new_lines, new_var_name)."""
    new = f"_op_{idx}"

    if op.op == "identity":
        return [], prev

    if op.op == "relu":
        ra = dict(op.attrs)
        if not ra:
            return [f"{new} = cute.math.max({prev}, cutlass.full_like({prev}, {_compute_literal(op.compute_dtype, 0)}))"], new
        slope, lower, upper = ra.get("negative_slope"), ra.get("lower_clip"), ra.get("upper_clip")
        rl = [f"_r{idx} = {prev}"]
        rv = f"_r{idx}"
        if slope is not None:
            rl.append(f"_rk{idx} = cute.math.max({rv}, {_fl(rv, 0.0)}) + {_fl(rv, slope)} * cute.math.min({rv}, {_fl(rv, 0.0)})")
            cur = f"_rk{idx}"
            if lower is not None:
                rl.append(f"_rl{idx} = cute.math.max({cur}, {_fl(rv, lower)})")
                cur = f"_rl{idx}"
        else:
            rl.append(f"_rl{idx} = cute.math.max({rv}, {_fl(rv, lower if lower is not None else 0.0)})")
            cur = f"_rl{idx}"
        if upper is not None:
            rl.append(f"_ru{idx} = cute.math.min({cur}, {_fl(rv, upper)})")
            cur = f"_ru{idx}"
        return rl + [f"{new} = {cur}"], new

    if op.op == "leaky_relu":
        slope = dict(op.attrs).get("negative_slope", 0.0)
        return [
            f"_lr{idx} = {prev}",
            f"{new} = cute.math.max(_lr{idx}, {_fl(f'_lr{idx}', 0.0)}) + {_fl(f'_lr{idx}', slope)} * cute.math.min(_lr{idx}, {_fl(f'_lr{idx}', 0.0)})",
        ], new

    if op.op == "elu":
        return [
            f"_el{idx} = {prev}",
            f"{new} = cute.math.max(_el{idx}, {_fl(f'_el{idx}', 0.0)}) + cute.math.exp2(cute.math.min(_el{idx}, {_fl(f'_el{idx}', 0.0)}) * cutlass.full_like(_el{idx}, {_LOG2E}), fastmath=True) - {_fl(f'_el{idx}', 1.0)}",
        ], new

    if op.op == "softplus":
        return [
            f"_sp{idx} = {prev}",
            f"{new} = cute.math.max(_sp{idx}, {_fl(f'_sp{idx}', 0.0)}) + cute.math.log({_fl(f'_sp{idx}', 1.0)} + cute.math.exp2(-cute.math.abs(_sp{idx}) * cutlass.full_like(_sp{idx}, {_LOG2E}), fastmath=True))",
        ], new

    if op.op == "tan":
        return [f"{new} = cute.math.sin({prev}) * cute.math.rcp(cute.math.cos({prev}), approx=True, ftz=True)"], new

    if op.op == "logical_not":
        ll, nz = _nonzero01(prev, f"_{idx}")
        return ll + [f"{new} = {_fl(nz, 1.0)} - {nz}"], new

    if op.op == "gen_index":
        axis = dict(op.attrs).get("axis")
        if axis == 1:
            return [f"{new} = cutlass.full_like({prev}, cutlass.Float32(row))"], new
        gl = [f"_gi{idx} = cute.make_rmem_tensor({vsize}, cutlass.Float32)"]
        for k in range(vsize):
            _row, _col = _elem_coord(k)
            gl.append(f"_gi{idx}[{k}] = cutlass.Float32({_row if axis == 1 else _col})")
        gl.append(f"{new} = _gi{idx}.load().to_vector()")
        return gl, new

    if op.op == "binary_select":
        assert other_in_chain is not None and third_in_chain is not None
        bl, nz = _nonzero01(third_in_chain, f"_{idx}")
        return bl + [f"{new} = {other_in_chain} + ({prev} - {other_in_chain}) * {nz}"], new

    if op.op in _CMP_OPS or op.op in _LOGIC_OPS or op.op in _BACKWARD_OPS or op.op == "mod":
        if op.parent_idx_b is not None:
            assert other_in_chain is not None
            rhs = other_in_chain
        else:
            assert op.aux is not None
            rhs = f"({aux_loads[op.aux]})"
        lhs, rhs = (prev, rhs) if op.aux_on_rhs else (rhs, prev)
        return _emit_binary_ext(op, lhs, rhs, idx, new)

    if op.op == "tanh":
        return [f"{new} = {_tanh_expr(prev)}"], new

    if op.op == "exp":
        # exp(x) = exp2(x * log2e); vector exp2.approx (MUFU).
        return [f"{new} = cute.math.exp2({prev} * cutlass.full_like({prev}, {_LOG2E}), fastmath=True)"], new

    if op.op == "abs":
        return [f"{new} = cute.math.abs({prev})"], new

    if op.op == "neg":
        return [f"{new} = -{prev}"], new

    if op.op == "cos":
        return [f"{new} = cute.math.cos({prev})"], new

    if op.op == "sin":
        return [f"{new} = cute.math.sin({prev})"], new

    if op.op == "ceil":
        return [f"{new} = cute.math.ceil({prev})"], new

    if op.op == "floor":
        return [f"{new} = cute.math.floor({prev})"], new

    if op.op == "erf":
        return [f"{new} = cute.math.erf({prev})"], new

    if op.op == "log":
        return [f"{new} = cute.math.log({prev})"], new

    if op.op == "reciprocal":
        return [f"{new} = cute.math.rcp({prev}, approx=True, ftz=True)"], new

    if op.op == "rsqrt":
        return [f"{new} = cute.math.rsqrt({prev})"], new

    if op.op == "sqrt":
        return [f"{new} = cute.math.sqrt({prev})"], new

    if op.op == "sigmoid":
        # sigmoid(x) = 1/(1+exp(-x)) — vector exp2.approx + rcp.approx (MUFU).
        return [
            f"{new} = cute.math.rcp(cutlass.full_like({prev}, cutlass.Float32(1.0)) + "
            f"cute.math.exp2(-{prev} * cutlass.full_like({prev}, {_LOG2E}), fastmath=True), approx=True, ftz=True)"
        ], new

    if op.op == "swish":
        beta = dict(op.attrs).get("swish_beta")
        if beta is not None:
            return [
                f"_swb{idx} = {prev} * {_fl(prev, beta)}",
                f"{new} = {prev} * cute.math.rcp(cutlass.full_like({prev}, cutlass.Float32(1.0)) + "
                f"cute.math.exp2(-_swb{idx} * cutlass.full_like({prev}, {_LOG2E}), fastmath=True), approx=True, ftz=True)",
            ], new
        # swish/SiLU = x * sigmoid(x).
        return [
            f"{new} = {prev} * cute.math.rcp(cutlass.full_like({prev}, cutlass.Float32(1.0)) + "
            f"cute.math.exp2(-{prev} * cutlass.full_like({prev}, {_LOG2E}), fastmath=True), approx=True, ftz=True)"
        ], new

    if op.op == "gelu_tanh":
        # 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3))); native vector tanh.approx.
        return [
            f"_g_x{idx} = {prev}",
            f"_g_inner{idx} = cutlass.full_like(_g_x{idx}, cutlass.Float32(0.7978845608028654)) * "
            f"(_g_x{idx} + cutlass.full_like(_g_x{idx}, cutlass.Float32(0.044715)) * _g_x{idx} * _g_x{idx} * _g_x{idx})",
            f"_g_tanh{idx} = {_tanh_expr(f'_g_inner{idx}')}",
            f"{new} = cutlass.full_like(_g_x{idx}, cutlass.Float32(0.5)) * _g_x{idx} * (cutlass.full_like(_g_x{idx}, cutlass.Float32(1.0)) + _g_tanh{idx})",
        ], new

    if op.op == "gelu":
        # 0.5 * x * (1 + erf(x / sqrt(2)))
        return [
            f"_e_x{idx} = {prev}",
            f"_e_erf{idx} = cute.math.erf(_e_x{idx} * cutlass.full_like(_e_x{idx}, {_compute_literal(op.compute_dtype, 0.7071067811865475)}))",
            f"{new} = cutlass.full_like(_e_x{idx}, {_compute_literal(op.compute_dtype, 0.5)}) * _e_x{idx} * "
            f"(cutlass.full_like(_e_x{idx}, {_compute_literal(op.compute_dtype, 1)}) + _e_erf{idx})",
        ], new

    if op.op in {"add", "mul", "sub", "div", "max", "min", "pow", "add_square"}:
        py_op = {"add": "+", "mul": "*", "sub": "-", "div": "/"}.get(op.op)
        if op.parent_idx_b is not None:
            # fan-in: second operand is another in-chain op result
            assert other_in_chain is not None
            if op.op == "max":
                return [f"{new} = cute.math.max({prev}, {other_in_chain})"], new
            if op.op == "min":
                return [f"{new} = cute.math.min({prev}, {other_in_chain})"], new
            if op.op == "pow":
                if op.aux_on_rhs:
                    return [f"{new} = cute.math.pow({prev}, {other_in_chain})"], new
                return [f"{new} = cute.math.pow({other_in_chain}, {prev})"], new
            if op.op == "add_square":
                if op.aux_on_rhs:
                    return [f"{new} = {prev} + {other_in_chain} * {other_in_chain}"], new
                return [f"{new} = {other_in_chain} + {prev} * {prev}"], new
            assert py_op is not None
            if op.aux_on_rhs:
                return [f"{new} = {prev} {py_op} {other_in_chain}"], new
            return [f"{new} = {other_in_chain} {py_op} {prev}"], new
        assert op.aux is not None
        aux_expr = aux_loads[op.aux]
        if op.op == "max":
            return [f"{new} = cute.math.max({prev}, ({aux_expr}))"], new
        if op.op == "min":
            return [f"{new} = cute.math.min({prev}, ({aux_expr}))"], new
        if op.op == "pow":
            if op.aux_on_rhs:
                return [f"{new} = cute.math.pow({prev}, ({aux_expr}))"], new
            return [f"{new} = cute.math.pow(({aux_expr}), {prev})"], new
        if op.op == "add_square":
            if op.aux_on_rhs:
                return [
                    f"_sq_aux_{idx} = ({aux_expr})",
                    f"{new} = {prev} + _sq_aux_{idx} * _sq_aux_{idx}",
                ], new
            return [
                f"_sq_aux_{idx} = ({aux_expr})",
                f"{new} = _sq_aux_{idx} + {prev} * {prev}",
            ], new
        assert py_op is not None
        if op.aux_on_rhs:
            return [f"{new} = {prev} {py_op} ({aux_expr})"], new
        return [f"{new} = ({aux_expr}) {py_op} {prev}"], new

    raise AssertionError(f"unhandled op {op.op!r}")


# Top-level codegen


def _emit_round(var: str, out_dtype: Dtype, tag: str) -> tuple[list[str], str]:
    """Round a register value to a tensor's declared dtype (no-op for fp32)."""
    if out_dtype == "fp32":
        return [], var
    new = f"_r_{tag}"
    return [f"{new} = ({var}).to({DTYPE_TO_CUTLASS[out_dtype]})"], new


def _store_cast_expr(var: str, dtype: Dtype) -> str:
    if dtype == "uint8":
        # GEMM exposes uint8 tensor raw pointers as Int8; bitcast preserves the byte payload.
        return f"({var}).to(cutlass.Uint8).bitcast(cutlass.Int8)"
    return f"({var}).to({DTYPE_TO_CUTLASS[dtype]})"


def _dense_store_offset(i: int, is_fp4: bool, batch: int) -> str:
    """Element/byte offset for dense output ``i``'s vector store: every dense
    output carries its OWN runtime strides (out_stride_{m,n,l}_<i>). The
    vector store spans col_j..col_j+vsize-1, so the N stride must be 1
    (n-major, validated at runtime); fp4 offsets are in packed bytes."""
    col = "(col_j >> 1)" if is_fp4 else "col_j"
    base = f"row * out_stride_m_{i} + {col}"
    return f"(tile_l * out_stride_l_{i} + {base})" if batch > 1 else f"({base})"


def _emit_mmajor_scatter(
    tap_idx: int,
    i: int,
    source_var: str,
    dtype: Dtype,
    batch: int,
    vsize: int,
    *,
    row_pred: str | None = None,
    converted: bool = False,
    col_bound: str = "N",
) -> list[str]:
    """Per-element scatter for an M-major (or arbitrarily strided) dense
    output: vsize scalar stores through the output's own runtime strides."""
    tap_var = f"_tap_{tap_idx}"
    l_term = f"tile_l * out_stride_l_{i} + " if batch > 1 else ""
    lines = [f"{tap_var} = {source_var if converted else _store_cast_expr(source_var, dtype)}"]
    for e in range(vsize):
        store = (
            f"(gC_tap_{tap_idx}_ptr + {l_term}row * out_stride_m_{i} + "
            f"(col_j + {e}) * out_stride_n_{i}).store({tap_var}[{e} : {e} + 1], "
            f"alignment={DTYPE_BYTES[dtype]})"
        )
        if row_pred is not None:
            lines.append(f"if ({row_pred}) & (col_j + {e} < {col_bound}):")
            lines.append(f"    {store}")
        else:
            lines.append(store)
    return lines


def _tap_store_elems(chain: FusionChain, dtype: Dtype, dim, stride, vsize: int) -> int:
    """Elements per STG store for a dense tap = min(compute ``vsize``, the tap's
    OWN declared-layout alignment). The store width is derived from the tensor,
    decoupling it from the (possibly larger) compute ``vsize`` a block-scale quant
    pins to its block size."""
    if dtype == "fp4_e2m1":
        return vsize  # packed 2/byte → the whole-chunk store is already <= 16B
    dim, stride = dense_output_layout(chain, dtype, dim, stride)
    return min(vsize, _allowed_vsize(chain, dtype, dim, stride))


def _tap_vec_bytes(chain: FusionChain, dtype: Dtype, dim, stride, vsize: int) -> int:
    """Byte width (== required alignment) of one dense tap vector store."""
    if dtype == "fp4_e2m1":
        return max(vsize // 2, 4)
    return _tap_store_elems(chain, dtype, dim, stride, vsize) * DTYPE_BYTES[dtype]


def _emit_moe_combine_store(tap_idx, i, src, spec, chain, vsize):
    swapped = isinstance(chain.moe, MoeSwapAbSpec)
    if not swapped and spec.major == "n":
        # One lane owns consecutive features for the SAME routed row. Only
        # those addresses can be packed; swapAB's register vector instead
        # contains unrelated token destinations and keeps scalar scatter-add.
        width = min(vsize, 16 // DTYPE_BYTES[spec.dtype])
        lines = [
            "if row < group_end:",
            "    _combine_token = token_index[row]",
            "    _combine_slot = token_ks[row]",
            "    if (_combine_token >= 0) & (_combine_token < top_k_scores.shape[0]) & (_combine_slot >= 0) & (_combine_slot < moe_top_k):",
            "        _combine_scale = top_k_scores[_combine_token, _combine_slot]",
            f"        _combine_values = {src} * cutlass.full_like({src}, _combine_scale)",
            f"        _combine_ptr = gC_tap_{tap_idx}_ptr + cutlass.Int64(_combine_token) * out_stride_m_{i} + cutlass.Int64(col_j) * out_stride_n_{i}",
        ]
        for e in range(0, vsize, width):
            lines += [
                f"        if col_j + {e + width} <= N:",
                f"            moe_combine_add_vector(_combine_ptr + {e}, _combine_values[{e}:{e + width}], {DTYPE_TO_CUTLASS[spec.dtype]}, {width})",
            ]
        return lines
    lines = []
    for e in range(vsize):
        routed = f"col_j + {e}" if swapped else "row"
        feature = "row" if swapped else f"col_j + {e}"
        bound = "M" if swapped else "N"
        offset = (
            f"cutlass.Int64({feature}) * out_stride_m_{i} + cutlass.Int64(_combine_token) * out_stride_n_{i}"
            if swapped
            else f"cutlass.Int64(_combine_token) * out_stride_m_{i} + cutlass.Int64({feature}) * out_stride_n_{i}"
        )
        lines += [
            f"if ({routed} < group_end) & ({feature} < {bound}):",
            f"    _combine_token = token_index[{routed}]",
            f"    _combine_slot = token_ks[{routed}]",
            "    if (_combine_token >= 0) & (_combine_token < top_k_scores.shape[0]) & (_combine_slot >= 0) & (_combine_slot < moe_top_k):",
            f"        _combine_value = {src}[{e}] * top_k_scores[_combine_token, _combine_slot]",
            f"        moe_combine_add(gC_tap_{tap_idx}_ptr + {offset}, _combine_value, {DTYPE_TO_CUTLASS[spec.dtype]})",
        ]
    return lines


def _emit_moe_scatter_store(tap_idx, i, src, spec, chain, vsize, row_pred, col_bound):
    """Store FC2 fragments into distinct token/top-k slots without atomics."""
    swapped = isinstance(chain.moe, MoeSwapAbSpec)
    lines = [f"_scatter_value = {_store_cast_expr(src, spec.dtype)}"]
    if not swapped:
        lines.append("_scatter_row = moe_scatter_row(token_index, token_ks, row, group_end, M, moe_top_k)")
        pred = "_scatter_row < M"
        if row_pred:
            pred = f"({row_pred}) & ({pred})"
        if spec.major == "n":
            lines += _emit_tap_store(
                tap_idx,
                "_scatter_value",
                spec.dtype,
                chain,
                spec.dim,
                spec.stride,
                vsize,
                f"(_scatter_row * out_stride_m_{i} + col_j)",
                row_pred=pred,
                converted=True,
                col_bound=col_bound,
            )
            return lines
    for e in range(vsize):
        if swapped:
            lines.append(f"_scatter_col = moe_scatter_row(token_index, token_ks, col_j + {e}, group_end, N, moe_top_k)")
            pred = "(row < M) & (_scatter_col < N)"
            offset = f"row * out_stride_m_{i} + _scatter_col * out_stride_n_{i}"
        else:
            pred = f"(_scatter_row < M) & (col_j + {e} < N)"
            offset = f"_scatter_row * out_stride_m_{i} + (col_j + {e}) * out_stride_n_{i}"
        if row_pred:
            pred = f"({row_pred}) & ({pred})"
        lines += [f"if {pred}:", f"    (gC_tap_{tap_idx}_ptr + {offset}).store(_scatter_value[{e}:{e+1}], alignment={DTYPE_BYTES[spec.dtype]})"]
    return lines


def _emit_tap_store(
    tap_idx: int,
    source_var: str,
    tap_dtype: Dtype,
    chain: FusionChain,
    dim,
    stride,
    vsize: int,
    offset_expr: str = "linear_idx",
    *,
    row_pred: str | None = None,
    converted: bool = False,
    col_bound: str = "N",
) -> list[str]:
    """Store one N-major tap vector: ``offset_expr`` in ``_tap_store_elems``-wide
    chunks (a wide dtype co-materialized with a block-quant splits into <=32B
    sub-stores). An M-major output goes through `_emit_mmajor_scatter`."""
    tap_var = f"_tap_{tap_idx}"
    store_elems = _tap_store_elems(chain, tap_dtype, dim, stride, vsize)
    lines = [f"{tap_var} = {source_var if converted else _store_cast_expr(source_var, tap_dtype)}"]
    # The STG arm sits inside `row < M` / `col_j + vsize <= N`; the TMA arm has
    # neither -- its store is clipped by the descriptor's global extent instead.
    # `_tap_store_elems` divides both N and `col_j`, so a sub-chunk is wholly
    # inside N or wholly outside: the whole-chunk predicate loses no column.
    for _s in range(0, vsize, store_elems) if store_elems < vsize else (None,):
        span = f"{tap_var}" if _s is None else f"{tap_var}[{_s} : {_s + store_elems}]"
        off = f"{offset_expr}" if _s is None else f"{offset_expr} + {_s}"
        width = vsize if _s is None else store_elems
        store = f"(gC_tap_{tap_idx}_ptr + {off}).store({span}, alignment=VEC_BYTES_TAP_{tap_idx})"
        if row_pred is not None:
            lines.append(f"if ({row_pred}) & (col_j + {(_s or 0) + width} <= {col_bound}):")
            lines.append(f"    {store}")
        else:
            lines.append(store)
    return lines


def _reduction_output_offset_expr(red_idx: int, red: ReductionSpec, value_idx: str) -> str:
    """Runtime-stride offset for a reduction output. Outputs are permuted to
    internal `(M, N, B)` order; the runtime wrapper passes matching strides."""
    b_extent, m_extent, n_extent = red.dim
    if red.grouped_by_moe:
        batch = "cutlass.Int64(group_idx)"
    else:
        batch = "cutlass.Int64(0)" if b_extent == 1 else "tile_l"
    m = "cutlass.Int64(0)" if m_extent == 1 else "row"
    n = "cutlass.Int64(0)" if n_extent == 1 else f"(col_j + {value_idx})"
    return f"(({m}) * red_stride_m_{red_idx} + " f"({n}) * red_stride_n_{red_idx} + " f"({batch}) * red_stride_l_{red_idx})"


def _emit_reduction_local_combine(
    red_idx: int,
    red: ReductionSpec,
    src: str,
    vsize: int,
) -> tuple[list[str], str]:
    """Combine the vector's elements into one register value so a single
    atomic per vector covers the whole chunk (valid only when every element
    maps to the same output address, i.e. the N axis is reduced)."""
    acc = f"_red_{red_idx}_acc"
    is_int = red.compute_dtype == "int32"
    zero = "cutlass.Int32(0)" if is_int else "cutlass.Float32(0.0)"

    def elem(i: int) -> str:
        v = f"{src}[{i}]"
        if red.mode in ("amax", "norm1"):
            return f"cute.math.abs({v})"
        if red.mode == "norm2":
            return f"{v} * {v}"
        return v

    if red.mode in ("add", "avg", "norm1", "norm2"):
        init, combine = zero, "add"
    elif red.mode in ("mul", "mul_no_zeros"):
        init, combine = "cutlass.Float32(1.0)", "mul"
    elif red.mode in ("max", "amax"):
        init, combine = elem(0), "max"
    else:
        assert red.mode == "min"
        init, combine = elem(0), "min"

    lines = [f"{acc} = {init}"]
    start = 0 if combine in ("add", "mul") else 1
    for i in range(start, vsize):
        if combine == "add":
            lines.append(f"{acc} = {acc} + {elem(i)}")
        elif combine == "mul":
            if red.mode == "mul_no_zeros":
                lines.append(f"if {src}[{i}] != cutlass.Float32(0.0):")
                lines.append(f"    {acc} = {acc} * {src}[{i}]")
            else:
                lines.append(f"{acc} = {acc} * {src}[{i}]")
        elif is_int:
            op = ">" if combine == "max" else "<"
            lines.append(f"if {elem(i)} {op} {acc}:")
            lines.append(f"    {acc} = {elem(i)}")
        else:
            lines.append(f"{acc} = cute.math.{combine}({acc}, {elem(i)})")
    return lines, acc


def _emit_reduction_atomic(
    tap_idx: int,
    red_idx: int,
    red: ReductionSpec,
    source_var: str,
    chain: FusionChain,
    vsize: int,
    row_pred: str | None = None,
    *,
    col_bound: str = "N",
    chunk_elems: int | None = None,
) -> list[str]:
    """A reduction is an atomic RMW, so an out-of-extent element cannot be
    clamped onto a valid one -- it has to be SKIPPED. The STG arm inherits
    `row < M` / `col_j + vsize <= N` from the drain; the TMA arm has neither, so
    it re-applies them here. Each subchunk divides the column boundary alignment,
    so its local fold never mixes real columns with OOB ones. N-grouped MoE
    uses both the offset-value promise and the token capacity S."""
    body = _emit_reduction_atomic_body(tap_idx, red_idx, red, source_var, chain, vsize)
    if row_pred is None:
        return body
    width = chunk_elems or vsize
    lines = [f"if ({row_pred}) & (col_j + {vsize} <= {col_bound}):"] + [f"    {ln}" for ln in body]
    if width == vsize:
        return lines
    # Keep the vector fold (one atomic when reducing N) for every full chunk.
    # Only the boundary chunk needs narrower contributions.
    lines.append(f"elif {row_pred}:")
    for start in range(0, vsize, width):
        src = f"{source_var}[{start}:{start + width}]"
        body = _emit_reduction_atomic_body(tap_idx, red_idx, red, src, chain, width, col_offset=start)
        lines.append(f"    if col_j + {start + width} <= {col_bound}:")
        lines.extend(f"        {ln}" for ln in body)
    return lines


def _emit_reduction_atomic_body(
    tap_idx: int,
    red_idx: int,
    red: ReductionSpec,
    source_var: str,
    chain: FusionChain,
    vsize: int,
    *,
    col_offset: int = 0,
) -> list[str]:
    src = f"_red_{red_idx}_src"
    lines = [f"{src} = ({source_var}).to({DTYPE_TO_CUTLASS[red.compute_dtype]})"]
    if red.mode == "avg":
        if chain.has_moe:
            extents = ["M", "N"]
            if red.grouped_by_moe:
                extents[1 if isinstance(chain.moe, MoeSwapAbSpec) else 0] = "cutlass.Int32(group_end) - cutlass.Int32(group_begin)"
            count = " * ".join(f"cutlass.Float32({extent})" for dim, extent in zip(red.dim[1:], extents) if dim == 1)
            count = count or "cutlass.Float32(1.0)"
            lines.append(f"_red_{red_idx}_inv = cutlass.Float32(1.0) / ({count})")
        else:
            matmul = chain.matmul
            count = (matmul.N if red.dim[2] == 1 else 1) * (matmul.M if red.dim[1] == 1 else 1) * (matmul.batch if red.dim[0] == 1 else 1)
            lines.append(f"_red_{red_idx}_inv = cutlass.Float32({1.0 / count})")
    if red.dim[2] == 1:
        combine_lines, acc = _emit_reduction_local_combine(red_idx, red, src, vsize)
        lines.extend(combine_lines)
        offset = _reduction_output_offset_expr(red_idx, red, "0")
        ptr = f"gC_tap_{tap_idx}_ptr + {offset}"
        if red.compute_dtype == "int32":
            op = {"amax": "max", "norm1": "add", "add": "add", "max": "max", "min": "min"}[red.mode]
            lines.append(f'nvvm.atomicrmw("{op}", {ptr}, {acc}, mem_order="relaxed", syncscope="gpu")')
            return lines
        if red.mode == "amax":
            lines.append(f'cute.arch.atomic_fmax({ptr}, {acc}, sign_bit=False, sem="relaxed", scope="gpu")')
        elif red.mode == "max":
            lines.append(f'cute.arch.atomic_fmax({ptr}, {acc}, sem="relaxed", scope="gpu")')
        elif red.mode == "min":
            bits = f"_red_{red_idx}_bits"
            lines.extend(
                [
                    f"{bits} = ({acc}).bitcast(cutlass.Int32)",
                    f"if {bits} < cutlass.Int32(0):",
                    f"    cute.arch.atomic_max({ptr}, cutlass.Uint32({bits}), " f'sem="relaxed", scope="gpu")',
                    "else:",
                    f"    cute.arch.atomic_min({ptr}, {bits}, " f'sem="relaxed", scope="gpu")',
                ]
            )
        elif red.mode in ("mul", "mul_no_zeros"):
            t = f"_red_{red_idx}_cas"
            lines.extend(
                [
                    f"if {acc} != cutlass.Float32(1.0):",
                    f"    {t}_mo = (({ptr})).load().bitcast(cutlass.Int32)",
                    f"    {t}_go = cutlass.Boolean(True)",
                    f"    while {t}_go:",
                    f"        {t}_nb = (({t}_mo).bitcast(cutlass.Float32) * {acc}).bitcast(cutlass.Int32)",
                    f'        {t}_pv = cute.arch.atomic_cas({ptr}, cmp={t}_mo, val={t}_nb, sem="relaxed", scope="gpu")',
                    f"        if {t}_pv == {t}_mo:",
                    f"            {t}_go = cutlass.Boolean(False)",
                    f"        else:",
                    f"            {t}_mo = {t}_pv",
                ]
            )
        else:
            val = f"{acc} * _red_{red_idx}_inv" if red.mode == "avg" else acc
            lines.append(f'nvvm.atomicrmw("add", {ptr}, {val}, mem_order="relaxed", syncscope="gpu")')
        return lines
    for i in range(vsize):
        val = f"{src}[{i}]"
        offset = _reduction_output_offset_expr(red_idx, red, str(col_offset + i))
        ptr = f"gC_tap_{tap_idx}_ptr + {offset}"
        if red.compute_dtype == "int32":
            if red.mode == "amax":
                val = f"cute.math.abs({val})"
                op = "max"
            elif red.mode == "norm1":
                val = f"cute.math.abs({val})"
                op = "add"
            elif red.mode in {"add", "max", "min"}:
                op = red.mode
            else:
                raise AssertionError(f"unhandled int32 reduction mode {red.mode!r}")
            lines.append(f'nvvm.atomicrmw("{op}", {ptr}, ' f'{val}, mem_order="relaxed", syncscope="gpu")')
            continue
        if red.mode == "amax":
            lines.append(f"cute.arch.atomic_fmax({ptr}, " f'cute.math.abs({val}), sign_bit=False, sem="relaxed", scope="gpu")')
            continue
        if red.mode == "norm1":
            lines.append(f'nvvm.atomicrmw("add", {ptr}, cute.math.abs({val}), mem_order="relaxed", syncscope="gpu")')
            continue
        if red.mode == "norm2":
            lines.append(f'nvvm.atomicrmw("add", {ptr}, {val} * {val}, mem_order="relaxed", syncscope="gpu")')
            continue
        if red.mode == "avg":
            lines.append(f'nvvm.atomicrmw("add", {ptr}, {val} * _red_{red_idx}_inv, mem_order="relaxed", syncscope="gpu")')
            continue
        if red.mode in ("mul", "mul_no_zeros"):
            t = f"_red_{red_idx}_{i}"
            pad = ""
            if red.mode == "mul_no_zeros":
                lines.append(f"if {val} != cutlass.Float32(0.0):")
                pad = "    "
            lines.extend(
                [
                    f"{pad}{t}_mo = (({ptr})).load().bitcast(cutlass.Int32)",
                    f"{pad}{t}_go = cutlass.Boolean(True)",
                    f"{pad}while {t}_go:",
                    f"{pad}    {t}_nb = (({t}_mo).bitcast(cutlass.Float32) * {val}).bitcast(cutlass.Int32)",
                    f'{pad}    {t}_pv = cute.arch.atomic_cas({ptr}, cmp={t}_mo, val={t}_nb, sem="relaxed", scope="gpu")',
                    f"{pad}    if {t}_pv == {t}_mo:",
                    f"{pad}        {t}_go = cutlass.Boolean(False)",
                    f"{pad}    else:",
                    f"{pad}        {t}_mo = {t}_pv",
                ]
            )
            continue
        if red.mode == "max":
            lines.append(f'cute.arch.atomic_fmax({ptr}, {val}, sem="relaxed", scope="gpu")')
            continue
        if red.mode == "min":
            bits = f"_red_{red_idx}_{i}_bits"
            lines.extend(
                [
                    f"{bits} = ({val}).bitcast(cutlass.Int32)",
                    f"if {bits} < cutlass.Int32(0):",
                    f"    cute.arch.atomic_max({ptr}, cutlass.Uint32({bits}), " f'sem="relaxed", scope="gpu")',
                    "else:",
                    f"    cute.arch.atomic_min({ptr}, {bits}, " f'sem="relaxed", scope="gpu")',
                ]
            )
            continue
        else:
            assert red.mode == "add"
            op = "add"
        lines.append(f'nvvm.atomicrmw("{op}", {ptr}, ' f'{val}, mem_order="relaxed", syncscope="gpu")')
    return lines


def _quant_output_max(dtype: Dtype) -> str:
    if dtype == "fp8_e4m3":
        return "cutlass.Float32(448.0)"
    if dtype == "fp8_e5m2":
        return "cutlass.Float32(57344.0)"
    if dtype == "fp4_e2m1":
        return "cutlass.Float32(6.0)"
    raise ValueError(f"block quantize output dtype {dtype!r} is not supported by codegen")


def _scale_store_dtype(scale_dtype: Dtype) -> str:
    """The DSL type a quantized scale is STORED as — one source of truth for the
    scale tap's element type, its zero-init, and the value written. E5M3 has no
    DSL float type and a raw_ptr store to a Uint8 tensor is rejected, so it rides
    the Int8 byte carrier (the same one packed FP4 data uses)."""
    return "cutlass.Int8" if scale_dtype == "fp8_e5m3" else DTYPE_TO_CUTLASS[scale_dtype]


def _f8_128x4_row_scale_index_expr(row: str, scale_col: str, n_col_quads: str, *, atom_base: str | None = None) -> str:
    """Physical byte index for one logical row scale in an F8_128x4 blob."""
    prefix = f"{atom_base} + " if atom_base is not None else ""
    return f"{prefix}(({row} // 128) * {n_col_quads} + ({scale_col} // 4)) * 512 + " f"({row} % 32) * 16 + (({row} % 128) // 32) * 4 + ({scale_col} % 4)"


def _e8m0_f32(bits: str, *, reciprocal: bool = False) -> str:
    """Widen a scale or its exact reciprocal without flushing subnormals.

    E8M0 byte 0 is 2**-127; byte 255 is NaN. The reciprocal exponent
    is 254-byte, with byte 254 producing the FP32 subnormal 2**-127.
    """
    exponent = f"(254 - {bits})" if reciprocal else bits
    special = f"({bits} >= 254)" if reciprocal else f"(({bits} == 0) | ({bits} == 255))"
    return f"(({exponent} << 23) | ({special}.to(cutlass.Int32) << 22)).bitcast(cutlass.Float32)"


def _emit_scale_quantize(p: str, sfx: str, src: str, scale_var: str, back_var: str, quant: BlockQuantizeSpec) -> list[str]:
    """Quantize one fp32 scale to ``quant.scale_dtype`` and read the STORED
    value back as fp32 — the data is divided by what was actually written, not
    by the pre-rounding scale, so a dequantize reproduces it exactly.

    E4M3 round-trips through the DSL ``.to()``. The other two reach the cvt unit
    through the helpers :func:`compiler._quant_device_imports` emits (which
    documents why their ``.to()`` is not usable), and read the byte back as
    the exact exponent encoding for ue8m0, including its minimum scale at
    byte 0, or through the paired widening helper for ue5m3."""
    scale_dtype = _scale_store_dtype(quant.scale_dtype)
    if quant.scale_dtype == "fp8_e8m0":
        return [
            f"{p}_qb{sfx} = _frost_cvt_f32_to_e8m0_bits({src})",
            f"{scale_var} = (({p}_qb{sfx}).to(cutlass.Int8)).bitcast({scale_dtype})",
            f"{back_var} = {_e8m0_f32(f'{p}_qb{sfx}')}",
        ]
    if quant.scale_dtype == "fp8_e5m3":
        return [
            f"{p}_qb{sfx} = _frost_cvt_f32_to_e5m3_bits({src})",
            f"{scale_var} = ({p}_qb{sfx}).to({scale_dtype})",
            f"{back_var} = _frost_e5m3_bits_to_f32({p}_qb{sfx})",
        ]
    return [
        f"{scale_var} = ({src}).to({scale_dtype})",
        f"{back_var} = ({scale_var}).to(cutlass.Float32)",
    ]


def _emit_scale_quantize_pair(p: str, a: tuple[str, str, str, str], b: tuple[str, str, str, str], quant: BlockQuantizeSpec) -> list[str]:
    """:func:`_emit_scale_quantize` for two scales at once. Each tuple is
    ``(suffix, src, scale_var, back_var)``. The cvt unit is x2, so a scale
    dtype that reaches it through the generated helpers converts both columns
    in one instruction; e4m3 goes through the DSL cast and falls back to two."""
    scale_dtype = _scale_store_dtype(quant.scale_dtype)
    if quant.scale_dtype not in ("fp8_e8m0", "fp8_e5m3"):
        return _emit_scale_quantize(p, *a, quant) + _emit_scale_quantize(p, *b, quant)
    (sa, srca, scalea, backa), (sb, srcb, scaleb, backb) = a, b
    fn = "_frost_cvt_f32_to_e8m0_bits" if quant.scale_dtype == "fp8_e8m0" else "_frost_cvt_f32_to_e5m3_bits"
    lines = [
        f"{p}_qw{sa} = {fn}_x2({srca}, {srcb})",
        f"{p}_qb{sa} = {p}_qw{sa} & 0xFF",
        f"{p}_qb{sb} = ({p}_qw{sa} >> 8) & 0xFF",
    ]
    if quant.scale_dtype == "fp8_e8m0":
        return lines + [
            f"{scalea} = (({p}_qb{sa}).to(cutlass.Int8)).bitcast({scale_dtype})",
            f"{backa} = {_e8m0_f32(f'{p}_qb{sa}')}",
            f"{scaleb} = (({p}_qb{sb}).to(cutlass.Int8)).bitcast({scale_dtype})",
            f"{backb} = {_e8m0_f32(f'{p}_qb{sb}')}",
        ]
    return lines + [
        f"{scalea} = ({p}_qb{sa}).to({scale_dtype})",
        f"{scaleb} = ({p}_qb{sb}).to({scale_dtype})",
        f"{backa}, {backb} = _frost_e5m3_bits_to_f32_x2({p}_qb{sa}, {p}_qb{sb})",
    ]


def _emit_block_quant_col(
    quant: BlockQuantizeSpec,
    quant_idx: int,
    source_var: str,
    output_dtype: Dtype,
    out_var: str,
    scale_tap_idx: int,
    chain: FusionChain,
    batch_index_expr: str,
    vsize: int,
    row_pred: str | None = None,
) -> list[str]:
    """Emit M-axis (col) block quantize for one epilogue vector. A warp
    (block 32) or half-warp (block 16) of lanes holds one block of rows, so
    each column's block amax is a (half-)warp redux; lane ``l`` keeps and
    stores the scale byte(s) of column(s) ``col_j + k*G + l % G``. The
    compiler gates the row guards to be reduction-uniform."""
    p = f"_q{quant_idx}"
    swapped = isinstance(chain.moe, MoeSwapAbSpec)
    col_bound = "group_end" if swapped else "N"
    scale_dtype = _scale_store_dtype(quant.scale_dtype)
    G = 16 if quant.block_size == 16 else 32
    if vsize % G != 0:
        raise NotImplementedError(
            f"col block-quantize: store vector {vsize} must be a multiple of the {G}-lane "
            f"block group — every lane stores column `col_j + lane % {G}`, so a narrower "
            f"chunk would store scales for columns it never computed"
        )
    n_groups = vsize // G
    lines: list[str] = [
        f"{p}_lane = tidx % 32",
        f"{p}_src = ({source_var}).to(cutlass.Float32)",
        f"{p}_out = cute.make_rmem_tensor({vsize}, cutlass.Float32)",
        f"{p}_rl = cute.arch.rcp_approx({_quant_output_max(output_dtype)})",
        f"{p}_scale_mine = cute.make_rmem_tensor({n_groups}, {scale_dtype})",
    ]
    for k in range(n_groups):
        lines.append(f"{p}_scale_mine[{k}] = (cutlass.Float32(0.0)).to({scale_dtype})")

    # Keep only four columns' reduction/scale temporaries live at once.  It is
    # important that this is one generated constexpr loop with reused names,
    # rather than Python-codegen unrolling into q0..q31 SSA values: the latter
    # makes the backend retain most of the 32-column scale state concurrently.
    lines.append(f"for {p}_vi in cutlass.range_constexpr(0, {vsize}, 4):")
    for lane_in_batch in range(4):
        if quant.block_size == 16:
            # 16-row blocks: each half-warp reduces its own block (the
            # cta_tile_m=64 1-CTA layout only ever runs the low half).
            lines.extend(
                [
                    f"    {p}_a{lane_in_batch} = cutlass.Float32(0.0)",
                    f"    if {p}_lane < 16:",
                    f'        {p}_a{lane_in_batch} = cute.arch.warp_redux_sync({p}_src[{p}_vi + {lane_in_batch}], "fmax", '
                    "mask_and_clamp=0x0000FFFF, abs=True)",
                    "    else:",
                    f'        {p}_a{lane_in_batch} = cute.arch.warp_redux_sync({p}_src[{p}_vi + {lane_in_batch}], "fmax", '
                    "mask_and_clamp=0xFFFF0000, abs=True)",
                ]
            )
        else:
            lines.append(f'    {p}_a{lane_in_batch} = cute.arch.warp_redux_sync({p}_src[{p}_vi + {lane_in_batch}], "fmax", abs=True)')
    for lane_in_batch in range(0, 4, 2):
        other = lane_in_batch + 1
        lines.append(
            f"    {p}_s{lane_in_batch}, {p}_s{other} = cute.arch.mul_packed_f32x2("
            f'({p}_a{lane_in_batch}, {p}_a{other}), ({p}_rl, {p}_rl), rnd="rn", ftz=False)'
        )
        lines.extend(
            "    " + line
            for line in _emit_scale_quantize_pair(
                p,
                (f"b{lane_in_batch}", f"{p}_s{lane_in_batch}", f"{p}_q{lane_in_batch}", f"{p}_u{lane_in_batch}"),
                (f"b{other}", f"{p}_s{other}", f"{p}_q{other}", f"{p}_u{other}"),
                quant,
            )
        )
    # The deterministic FP32->scale converter is a hardware x2 operation (the
    # native x4 form is stochastic-only), but the four rounded scale values can
    # still share one vector reciprocal/min pipeline.  Keeping that pipeline as
    # TensorSSA gives the backend the same four-wide scheduling opportunity as
    # the specialized Rubin epilogue without changing RP/SATFINITE semantics.
    if quant.scale_dtype == "fp8_e8m0":
        lines.extend(
            [
                f"    {p}_up4 = cute.make_rmem_tensor(4, cutlass.Float32)",
                *(f"    {p}_up4[{i}] = {_e8m0_f32(f'{p}_qbb{i}', reciprocal=True)}" for i in range(4)),
                f"    {p}_iv = {p}_up4.load()",
            ]
        )
    else:
        lines.extend(
            [
                f"    {p}_up4 = cute.make_rmem_tensor(4, cutlass.Float32)",
                *(f"    {p}_up4[{i}] = {p}_u{i}" for i in range(4)),
                f"    {p}_upv = {p}_up4.load()",
                f"    {p}_iv = cute.math.min(cute.math.rcp({p}_upv, approx=True, ftz=True), " f"cutlass.full_like({p}_upv, cutlass.Float32(3.402823466e38)))",
            ]
        )
    for lane_in_batch in range(0, 4, 2):
        other = lane_in_batch + 1
        lines.extend(
            [
                f"    {p}_out[{p}_vi + {lane_in_batch}], {p}_out[{p}_vi + {other}] = cute.arch.mul_packed_f32x2("
                f"({p}_src[{p}_vi + {lane_in_batch}], {p}_src[{p}_vi + {other}]), "
                f'({p}_iv[{lane_in_batch}], {p}_iv[{other}]), rnd="rn", ftz=False)',
                f"    if ({p}_lane % {G}) == (({p}_vi + {lane_in_batch}) % {G}):",
                f"        {p}_scale_mine[({p}_vi + {lane_in_batch}) // {G}] = {p}_q{lane_in_batch}",
                f"    if ({p}_lane % {G}) == (({p}_vi + {other}) % {G}):",
                f"        {p}_scale_mine[({p}_vi + {other}) // {G}] = {p}_q{other}",
            ]
        )
    # The scale is rounded up and the reciprocal comes from that stored scale,
    # so every data value is already inside the destination format's finite
    # range.  An explicit elementwise min/max here is redundant and lowers to
    # hundreds of FSETP/FMNMX instructions for a dual row/col quant epilogue.
    lines.extend(
        [
            f"{p}_vec = {p}_out.load().to_vector()",
            (
                f"{out_var} = ({p}_vec).to(cutlass.Float4E2M1FN).bitcast(cutlass.Int8)"
                if output_dtype == "fp4_e2m1"
                else f"{out_var} = ({p}_vec).to({DTYPE_TO_CUTLASS[output_dtype]})"
            ),
        ]
    )
    if quant.grouped_by_moe and swapped:
        lines.extend(
            [
                f"{p}_mb = row // {quant.block_size}",
                f"{p}_mcb = ((M // {quant.block_size}) + 3) // 4",
                f"{p}_base = start_sf_block_n * {p}_mcb * 512",
            ]
        )
    elif quant.grouped_by_moe:
        lines.extend(
            [
                f"{p}_mb = (row - group_begin) // {quant.block_size}",
                f"{p}_mcb = (((group_end - group_begin) // {quant.block_size}) + 3) // 4",
                f"{p}_base = (group_begin // {4 * quant.block_size}) * ((N + 127) // 128) * 512",
            ]
        )
    else:
        lines.append(f"{p}_mb = row // {quant.block_size}")
    for k in range(n_groups):
        lines.append(f"{p}_n{k} = col_j + {k * G} + ({p}_lane % {G})")
        if quant.scale_reorder == "F8_128x4":
            mcb = f"{p}_mcb" if quant.grouped_by_moe else str((chain.matmul.M // quant.block_size + 3) // 4)
            base = f"{p}_base + " if quant.grouped_by_moe else ""
            token = f"({p}_n{k} - group_begin)" if quant.grouped_by_moe and swapped else f"{p}_n{k}"
            lines.append(
                f"{p}_sidx{k} = {batch_index_expr} * quant_scale_stride_l_{quant_idx} + {base}"
                f"(({token} // 128) * {mcb} + ({p}_mb // 4)) * 512 + "
                f"({token} % 32) * 16 + (({token} % 128) // 32) * 4 + ({p}_mb % 4)"
            )
        else:
            lines.append(
                f"{p}_sidx{k} = {batch_index_expr} * quant_scale_stride_l_{quant_idx} + "
                f"{p}_mb * quant_scale_stride_m_{quant_idx} + {p}_n{k} * quant_scale_stride_n_{quant_idx}"
            )
        # The scale byte is a SIDE STORE: the STG arm sits inside `row < M`,
        # the TMA arm has no row bound of its own.
        _st = f"(gC_tap_{scale_tap_idx}_ptr + {p}_sidx{k}).store({p}_scale_mine[{k}], alignment=1)"
        if row_pred is None:
            lines.append(_st)
        else:
            # `_output_store_mode` forces N % chunk == 0, so a chunk is wholly
            # inside N or wholly past it -- the chunk-level column test is exact.
            col_pred = f"{p}_n{k} < {col_bound}" if swapped else f"col_j + {vsize} <= N"
            lines.append(f"if ({row_pred}) & ({col_pred}):")
            lines.append(f"    {_st}")
    return lines


def _emit_block_quant(
    quant: BlockQuantizeSpec,
    quant_idx: int,
    source_var: str,
    output_dtype: Dtype,
    out_var: str,
    scale_tap_idx: int,
    chain: FusionChain,
    batch_index_expr: str = "tile_l",
    vsize: int = 32,
    row_pred: str | None = None,
) -> list[str]:
    """Emit block quantize for one epilogue vector, binding the quantized
    vector to ``out_var`` and storing the scale byte(s) through the
    ``scale_tap_idx`` tap pointer. Row/N-axis: the chunk holds
    ``vsize / block_size`` whole blocks (the compiler gates divisibility),
    each with a thread-local amax and its own scale. Col/M-axis (``axis ==
    1``) dispatches to :func:`_emit_block_quant_col`."""
    if quant.axis == 1:
        return _emit_block_quant_col(
            quant,
            quant_idx,
            source_var,
            output_dtype,
            out_var,
            scale_tap_idx,
            chain,
            batch_index_expr,
            vsize,
            row_pred,
        )
    p = f"_q{quant_idx}"
    swapped = isinstance(chain.moe, MoeSwapAbSpec)
    col_bound = "group_end" if swapped else "N"
    bs = quant.block_size
    if vsize % bs != 0:
        raise NotImplementedError(f"row block-quantize: store vector {vsize} must be a multiple of block_size {bs}")
    n_sub = vsize // bs
    lines: list[str] = [
        f"{p}_src = ({source_var}).to(cutlass.Float32)",
        f"{p}_frg = cute.TensorSSA({p}_src.ir_value(), ({bs}, {n_sub}), cutlass.Float32)",
        f"{p}_abs = cute.math.absf({p}_frg)",
        f"{p}_out = cute.make_rmem_tensor({vsize}, cutlass.Float32)",
        f"{p}_rl = cute.arch.rcp_approx({_quant_output_max(output_dtype)})",
    ]
    for k in range(n_sub):
        base = k * bs
        lines.append(f"{p}_amax{k} = {p}_abs[None, {k}].reduce(cute.ReductionOp.MAX, cutlass.Float32(0.0), 0)")
        lines.append(f"{p}_sf{k} = {p}_amax{k} * {p}_rl")
        lines.extend(_emit_scale_quantize(p, str(k), f"{p}_sf{k}", f"{p}_scale{k}", f"{p}_up{k}", quant))
        if quant.scale_dtype == "fp8_e8m0":
            lines.append(f"{p}_inv{k} = {_e8m0_f32(f'{p}_qb{k}', reciprocal=True)}")
        else:
            lines.append(f"{p}_inv{k} = cute.math.min(cute.arch.rcp_approx({p}_up{k}), cutlass.Float32(3.402823466e38))")
        for e in range(0, bs, 2):
            lines.append(
                f"{p}_out[{base + e}], {p}_out[{base + e + 1}] = cute.arch.mul_packed_f32x2("
                f'({p}_src[{base + e}], {p}_src[{base + e + 1}]), ({p}_inv{k}, {p}_inv{k}), rnd="rn", ftz=False)'
            )
    # See the col-quant path above: scale-up rounding already bounds the data,
    # and the direct narrowing is the same contract used by the specialized
    # grouped-GEMM quant epilogues.
    lines.extend(
        [
            f"{p}_vec = {p}_out.load().to_vector()",
            (
                f"{out_var} = ({p}_vec).to(cutlass.Float4E2M1FN).bitcast(cutlass.Int8)"
                if output_dtype == "fp4_e2m1"
                else f"{out_var} = ({p}_vec).to({DTYPE_TO_CUTLASS[output_dtype]})"
            ),
        ]
    )
    if quant.grouped_by_moe and swapped:
        lines.extend(
            [
                f"{p}_local_row = row",
                f"{p}_ncb = (((group_end - group_begin) // {bs}) + 3) // 4",
                f"{p}_base = (group_begin // {4 * bs}) * ((M + 127) // 128) * 512",
            ]
        )
    elif quant.grouped_by_moe:
        # Slot 6 is the block-scaled MoE scheduler's prefix sum of preceding
        # groups' ceil(group_rows/128) scale atoms. Restart the row address at
        # the group-local row and that atom base.
        lines.extend(
            [
                f"{p}_local_row = row - group_begin",
                f"{p}_ncb = ((N // {bs}) + 3) // 4",
                f"{p}_base = start_sf_block_m * {p}_ncb * 512",
            ]
        )
    for k in range(n_sub):
        col = "(col_j - group_begin)" if quant.grouped_by_moe and swapped else "col_j"
        lines.append(f"{p}_scol{k} = {col} // {bs} + {k}")
        if quant.scale_reorder == "F8_128x4":
            if quant.grouped_by_moe:
                sidx = _f8_128x4_row_scale_index_expr(f"{p}_local_row", f"{p}_scol{k}", f"{p}_ncb", atom_base=f"{p}_base")
                lines.append(f"{p}_sidx{k} = {batch_index_expr} * quant_scale_stride_l_{quant_idx} + {sidx}")
            else:
                lines.extend(
                    [
                        f"{p}_ncb{k} = ((N // {bs}) + 3) // 4",
                        (
                            f"{p}_sidx{k} = {batch_index_expr} * quant_scale_stride_l_{quant_idx} + "
                            f"((row // 128) * {p}_ncb{k} + ({p}_scol{k} // 4)) * 512 + "
                            f"(row % 32) * 16 + ((row % 128) // 32) * 4 + ({p}_scol{k} % 4)"
                        ),
                    ]
                )
        else:
            lines.append(
                f"{p}_sidx{k} = {batch_index_expr} * quant_scale_stride_l_{quant_idx} + "
                f"row * quant_scale_stride_m_{quant_idx} + {p}_scol{k} * quant_scale_stride_n_{quant_idx}"
            )
        # The scale byte is a SIDE STORE: the STG arm sits inside `row < M`,
        # the TMA arm has no row bound of its own.
        _st = f"(gC_tap_{scale_tap_idx}_ptr + {p}_sidx{k}).store({p}_scale{k}, alignment=1)"
        if row_pred is None:
            lines.append(_st)
        else:
            # `_output_store_mode` forces N % chunk == 0, so a chunk is wholly
            # inside N or wholly past it -- the chunk-level column test is exact.
            width = (k + 1) * bs if swapped else vsize
            lines.append(f"if ({row_pred}) & (col_j + {width} <= {col_bound}):")
            lines.append(f"    {_st}")
    return lines


def _tap_fake_shape(tap, chain: FusionChain | None = None) -> str:
    if chain is not None and chain.has_moe and chain.moe.mode == "combine":
        return "(sym_m, cute.sym_int(), 1)" if isinstance(chain.moe, MoeSwapAbSpec) else "(cute.sym_int(), sym_n, 1)"
    if tap.is_quant_scale:
        if chain is None or not chain.quants:
            raise AssertionError("quant scale tap requires FusionChain context")
        q = chain.quants[int(tap.source.rsplit("_", 1)[1])]
        swapped = isinstance(chain.moe, MoeSwapAbSpec)
        if q.grouped_by_moe and ((q.axis == 1) == swapped):
            # The scheduler computes the segmented row prefix at runtime.
            # Capacity belongs to the caller's allocation, not the JIT shape.
            return "(cute.sym_int64(), cute.sym_int64(), 1)"
        b, m, n = q.scale_dim or (
            chain.matmul.batch,
            chain.matmul.M,
            chain.matmul.N // q.block_size,
        )
        logical_n = chain.matmul.N // q.block_size
        m_expr = "1" if m == 1 else ("sym_m" if m == chain.matmul.M else str(m))
        n_expr = "1" if n == 1 else (f"(sym_n // {q.block_size})" if n == logical_n else str(n))
        l_expr = "1" if b == 1 else ("sym_l" if b == chain.matmul.batch else str(b))
        return f"({m_expr}, {n_expr}, {l_expr})"
    if tap.dim is None:
        l_expr = "1" if (chain is not None and chain.has_moe) else "sym_l"
        if tap.dtype == "fp4_e2m1":
            return f"(sym_m, (sym_n // 2), {l_expr})"
        return f"(sym_m, sym_n, {l_expr})"
    b, m, n = tap.dim
    m_expr = "1" if m == 1 else "sym_m"
    n_expr = "1" if n == 1 else "sym_n"
    l_expr = "1" if b == 1 else "sym_l"
    if tap.is_reduction and chain is not None:
        red_idx = int(tap.source.rsplit("_", 1)[1])
        if chain.reductions[red_idx].grouped_by_moe:
            l_expr = "sym_g"
    return f"({m_expr}, {n_expr}, {l_expr})"


# Per-op temp-var index base for mainloop transforms, kept distinct per operand
# and far from the epilogue's 0-based indices so snippets sharing one JIT
# function never collide on a var name (cute type-checks the whole body).
_MAINLOOP_IDX_BASE = {"a": 900, "b": 800}


# Mainloop identity-cast fast paths (more intrinsics can be added).
_MAINLOOP_IDENTITY_CAST_INTRINSICS: dict[tuple[Dtype, Dtype], str] = {
    ("int8", "bf16"): "cute.arch.cvt_i8_bf16_intrinsic",
}


def generate_mainloop(chain: FusionChain, operand: str = "a") -> str:
    """Emit the mainloop-fusion transform snippet for one operand
    (INJECT_MAINLOOP_A/B). Contract: the template loaded ``ml_vec_<operand>``
    from SMEM; this defines ``ml_out_<operand>`` = the op chain applied in fp32,
    cast back to the operand dtype, which the template stores in place."""
    ops = chain.mainloop_a_ops if operand == "a" else chain.mainloop_b_ops
    if not ops:
        return "pass"
    # dtype-preserving: result rounded back to the operand's own dtype in place
    src_dtype = chain.matmul.a_dtype if operand == "a" else chain.matmul.b_dtype
    ab_dtype = DTYPE_TO_CUTLASS[src_dtype]
    vec_var = f"ml_vec_{operand}"
    out_var = f"ml_out_{operand}"
    f32_var = f"ml_f32_{operand}"
    base = _MAINLOOP_IDX_BASE[operand]
    load_dtype = chain.mainloop_a_load_dtype if operand == "a" else chain.mainloop_b_load_dtype
    identity_cast_intrinsic = (
        _MAINLOOP_IDENTITY_CAST_INTRINSICS.get((load_dtype, src_dtype)) if len(ops) == 1 and ops[0].op == "identity" and load_dtype is not None else None
    )
    if identity_cast_intrinsic is not None:
        cvt = f"{identity_cast_intrinsic}({vec_var}.ir_value(), ml_vec_elems)"
        return f"{out_var} = cutlass.Vector({cvt}, dtype={ab_dtype})"
    # Scalar-aux loads for binary mainloop ops: broadcast the scalar (loaded
    # from its GMEM ptr — a kernel param) to a fp32 vector. Scalar only.
    aux_loads: dict[str, str] = {}
    for op in ops:
        if op.aux is not None:
            ptr = f"{op.aux}.iterator.raw_ptr()"
            aux_loads[op.aux] = f"cutlass.full_like({f32_var}, ({ptr} + 0).load().to(cutlass.Float32))"
    lines: list[str] = [f"{f32_var} = {vec_var}.to(cutlass.Float32)"]
    result_var: dict[int, str] = {-1: f32_var}
    for i, op in enumerate(ops):
        parent = op.resolved_parent_idx(i)
        parent_var = result_var[parent]
        emit_lines, cur = _emit_op(op, parent_var, base + i, aux_loads)
        lines.extend(emit_lines)
        result_var[i] = cur
    terminal_var = result_var[len(ops) - 1]
    # int -> fp8 fold workaround (foot-gun #3): int-loaded + fp8 MMA folds the
    # int->fp32 into the fp32->fp8 narrowing → invalid direct int->fp8 cast
    # (NaN). Break the def-use chain with `+ 0.0` (a two-step .to() does NOT
    # help — must be an arithmetic op).
    if load_dtype in ("int8", "uint8") and src_dtype in ("fp8_e4m3", "fp8_e5m2"):
        terminal_var = f"({terminal_var} + cutlass.full_like({terminal_var}, 0.0))"
    lines.append(f"{out_var} = ({terminal_var}).to({ab_dtype})")
    return "\n".join(lines)


def _emit_splitk_partial_store(vsize: int) -> list[str]:
    """Split-K kernel-1 STG fallback: store the raw fp32 accumulator chunk to
    partials[tile_l][M][N]; tile_l = grid z = batch*split."""
    chunk = min(vsize, 4)  # 4 fp32 = one 16-byte STG
    lines = [f"_split_k_off = (tile_l * M + row) * N + col_j"]
    for s in range(0, vsize, chunk):
        lines.append(f"(gSplitK_partials_ptr + _split_k_off + {s}).store(vec_f32[{s} : {s + chunk}], alignment={chunk * 4})")
    return lines


def generate(
    chain: FusionChain,
    *,
    vec_bytes_epi: int = 32,
    output_elem_bytes: int = 2,
    tma_slots: "frozenset[int]" = frozenset(),
    packed_lanes: bool = False,
    split_k_slices: int = 1,
) -> EpilogueSnippets:
    """Produce the two hook-site snippets, the extra kernel param list, and
    all per-tap plumbing. ``vec_bytes_epi`` / ``output_elem_bytes`` (from the
    compiler) fix the inner-loop chunk size: each tap stores
    ``vsize = vec_bytes_epi // output_elem_bytes`` elements per chunk."""
    vsize = vec_bytes_epi // output_elem_bytes
    scatter = chain.has_moe and chain.moe.mode == "scatter"
    row_bound = "group_end" if chain.has_moe and not isinstance(chain.moe, MoeSwapAbSpec) else "M"
    col_bound = "group_end" if isinstance(chain.moe, MoeSwapAbSpec) else "N"
    # aux_views snippet. `row` is defined by the template just before this hook
    # (M-aware: differs for MMA_M=64 vs MMA_M>=128) — we just consume it.
    aux_lines: list[str] = []
    aux_row = f"cute.math.min(cutlass.Int32(row), cutlass.Int32({row_bound}) - 1)"
    for aux in chain.aux_tensors:
        aux_lines.append(f"{_aux_ptr_var(aux.name)} = {aux.name}.iterator.raw_ptr()")
        if aux.bcast_mode == "scalar":
            aux_lines.append(f"{_aux_prefetch_var(aux.name)} = " f"({_aux_ptr_var(aux.name)} + {_aux_index_expr(aux)}).load()")
        elif aux.bcast_mode == "per_row" and not scatter:
            aux_lines.append(f"{_aux_prefetch_var(aux.name)} = " f"({_aux_ptr_var(aux.name)} + {_aux_index_expr(aux, row_var=aux_row)}).load()")
        # per_col / per_elem load inside the inner loop.

    aux_views = "\n".join(aux_lines) if aux_lines else "pass"

    # epilogue snippet (interleaves op chain with tap stores)
    # ANY slot on the TMA surface means the kernel renders the TMA arm, so the
    # bounds the arm does not supply have to be emitted -- keying this on slot 0
    # leaves them off whenever slot 0 is the one that fell back (an fp4 data
    # output beside a bf16 one, say).
    on_tma_arm = bool(tma_slots)
    # The template binds `vec_f32_<g>` in the STG arm only, where the chunk is a
    # slice of the subtile. The TMA arm takes the whole subtile, and when it is
    # rendered the STG arm is deleted outright -- so emit the bindings here
    # rather than adding a second template marker to all 8 parity groups.
    tma_vec_bindings = [f"vec_f32_{g} = c_rmem_vecs[{g}]" for g in range(1, chain.num_gemms)] if on_tma_arm else []
    # The TMA arm carries no row bound of its own; reuse the one its STG
    # sibling applies -- the routed group's end on MoE, the problem M elsewhere.
    # The TMA arm has no row predicate of its own -- the descriptor's global
    # extent clips the store -- so every SIDE EFFECT placed on it carries one.
    # Under the packed `lane < 16` layout half the lanes hold nothing, and a
    # reduction's atomic RMW cannot be clipped after the fact.
    store_row_pred = None
    bounded = scatter or on_tma_arm or (isinstance(chain.moe, MoeSwapAbSpec) and bool(chain.quants))
    if bounded:
        store_row_pred = f"row < {row_bound}"
        if packed_lanes:
            store_row_pred = f"row_active & ({store_row_pred})"
    _aux_pre = _scatter_aux_prelude(chain) if scatter else _bounded_aux_prelude(chain, row_bound, col_bound, bounded)
    body_lines: list[str] = tma_vec_bindings + _aux_pre

    # Per-op result var name lookup (handles `identity` pass-throughs).
    result_var: dict[int, str] = {}

    # Round each GEMM's fp32 accumulator to the matmul out_dtype before any op
    # reads it (no-op when fp32). GEMM 0 binds legacy ``vec_f32`` (so every
    # non-multi-GEMM template is unchanged); GEMMs >0 bind ``vec_f32_<g>``.
    gemm_var: dict[int, str] = {}
    for g in range(chain.num_gemms):
        src = "vec_f32" if g == 0 else f"vec_f32_{g}"
        tag = "mm" if g == 0 else f"mm{g}"
        round_lines, var = _emit_round(src, chain.matmul.out_dtype, tag)
        body_lines.extend(round_lines)
        gemm_var[g] = var

    def _parent_value(ref: int) -> str:
        """Var name for an op input: a GEMM output (ref < 0) or a prior op."""
        if is_gemm_source(ref):
            return gemm_var[gemm_index(ref)]
        return result_var[ref]

    for i, op in enumerate(chain.ops):
        if op.op == "aux_load":
            aux_ref = chain.aux_by_name(op.aux)
            body_lines.append(f"_op_{i} = {_aux_load_expr(aux_ref, op.compute_dtype, 'vec_f32', bounded=bounded or aux_ref.stride[-1] != 1)}")
            round_lines, cur = _emit_round(f"_op_{i}", op.out_dtype, str(i))
            body_lines.extend(round_lines)
            result_var[i] = cur
            continue
        parent = op.resolved_parent_idx(i)
        parent_raw = _parent_value(parent)
        cast_lines, parent_var = _compute_cast(parent_raw, op.compute_dtype, f"{i}_a")
        body_lines.extend(cast_lines)
        aux_loads = {aux.name: _aux_load_expr(aux, op.compute_dtype, parent_var, bounded=bounded or aux.stride[-1] != 1) for aux in chain.aux_tensors}
        other_in_chain = _parent_value(op.parent_idx_b) if op.parent_idx_b is not None else None
        if other_in_chain is not None:
            cast_lines, other_in_chain = _compute_cast(other_in_chain, op.compute_dtype, f"{i}_b")
            body_lines.extend(cast_lines)
        third_in_chain = _parent_value(op.parent_idx_c) if op.parent_idx_c is not None else None
        if third_in_chain is not None:
            cast_lines, third_in_chain = _compute_cast(third_in_chain, op.compute_dtype, f"{i}_c")
            body_lines.extend(cast_lines)
        lines, cur = _emit_op(op, parent_var, i, aux_loads, other_in_chain, third_in_chain, vsize=vec_bytes_epi // output_elem_bytes)
        if scatter and op.op == "gen_index":
            token_axis = 2 if isinstance(chain.moe, MoeSwapAbSpec) else 1
            if dict(op.attrs).get("axis") == token_axis:
                if token_axis == 1:
                    lines = [f"{cur} = cutlass.full_like({parent_var}, cutlass.Float32(moe_scatter_row(token_index, token_ks, row, group_end, M, moe_top_k)))"]
                else:
                    lines = [
                        f"_gi{i} = cute.make_rmem_tensor(vsize, cutlass.Float32)",
                        "for _sk in cutlass.range_constexpr(vsize):",
                        f"    _gi{i}[_sk] = cutlass.Float32(moe_scatter_row(token_index, token_ks, col_j + _sk, group_end, N, moe_top_k))",
                        f"{cur} = _gi{i}.load().to_vector()",
                    ]

        body_lines.extend(lines)
        # Round to the op's out_dtype (no-op for fp32) so every consumer —
        # downstream ops and outputs alike — sees the declared-dtype value.
        round_lines, cur = _emit_round(cur, op.out_dtype, str(i))
        body_lines.extend(round_lines)
        result_var[i] = cur

    # Dense outputs, uniform in slot order. The store MODE is per slot: a slot
    # in ``tma_slots`` binds the template's ``vec_out`` (the TMA staging consumes
    # it) and occupies the trailing TMA-C parameter; every other output rides a
    # tap slot. `_tap_of` is the ONE place that numbering is decided -- the tap
    # index, the reduction index and the quant-scale index all read it, so they
    # cannot drift apart. A quant-carrying spec emits the block-quantize
    # (quantized vector + scale byte) in place of a plain cast.
    specs = chain.output_specs
    quant_batch_expr = "0" if chain.has_moe else "tile_l"
    _tap_of: dict[int, int] = {}
    for _oi in range(len(chain.outputs)):
        if _oi not in tma_slots:
            _tap_of[_oi] = len(_tap_of)

    def _scale_tap_idx(qi: int) -> int:
        return _tap_of[len(specs) + len(chain.reductions) + qi]

    output_order = list(range(len(specs)))
    if on_tma_arm and len(tma_slots) > 1:
        # The compiler stores each output at its ready marker.  Retire outputs
        # backed by an exclusive source first, while register-heavy quantizers
        # stay last.  Among quantizers retire M-axis/column reduction state
        # before the cheaper row path; on a shared 64-column drain this avoids
        # carrying the heavier path across the row output's TMA store.
        source_uses: dict[int, int] = {}
        for output_spec in specs:
            source_uses[output_spec.source_ref] = source_uses.get(output_spec.source_ref, 0) + 1
        for reduction in chain.reductions:
            source_uses[reduction.source_ref] = source_uses.get(reduction.source_ref, 0) + 1
        output_order.sort(
            key=lambda oi: (
                specs[oi].quant_idx is not None,
                0 if specs[oi].quant_idx is not None and chain.quants[specs[oi].quant_idx].axis == 1 else 1,
                source_uses[specs[oi].source_ref],
            )
        )
    for si in output_order:
        spec = specs[si]
        src = _parent_value(spec.source_ref)
        converted = spec.quant_idx is not None
        if converted:
            qv = f"_quant_data_{si}"
            body_lines.extend(
                _emit_block_quant(
                    chain.quants[spec.quant_idx],
                    spec.quant_idx,
                    src,
                    spec.dtype,
                    qv,
                    _scale_tap_idx(spec.quant_idx),
                    chain,
                    quant_batch_expr,
                    vsize,
                    store_row_pred,
                )
            )
            src = qv
        # Quantized values (including packed FP4) are already converted. All
        # dense outputs use the same layout/store dispatch after this point.
        if si in tma_slots:
            _tma_j = sorted(tma_slots).index(si)
            _ov = tma_out_value(_tma_j)
            body_lines.append(f"{_ov} = {src if converted else _store_cast_expr(src, spec.dtype)}")
            body_lines.append(tma_out_ready_marker(_tma_j))
            continue
        tap_idx = _tap_of[si]
        if chain.has_moe and chain.moe.mode == "combine":
            body_lines.extend(_emit_moe_combine_store(tap_idx, si, src, spec, chain, vsize))
            continue
        if chain.has_moe and chain.moe.mode == "scatter":
            body_lines.extend(_emit_moe_scatter_store(tap_idx, si, src, spec, chain, vsize, store_row_pred, col_bound))
            continue
        if spec.major == "m":
            body_lines.extend(
                _emit_mmajor_scatter(tap_idx, si, src, spec.dtype, chain.matmul.batch, vsize, row_pred=store_row_pred, converted=converted, col_bound=col_bound)
            )
            continue
        offset_expr = _dense_store_offset(si, spec.dtype == "fp4_e2m1", chain.matmul.batch)
        body_lines.extend(
            _emit_tap_store(
                tap_idx, src, spec.dtype, chain, spec.dim, spec.stride, vsize, offset_expr, row_pred=store_row_pred, converted=converted, col_bound=col_bound
            )
        )

    # The STG chunk divides every group boundary, including S. Its width is
    # also checked on runtime shape reuse, even when a dense output uses TMA.
    red_chunk = gcd(vsize, _compute_output_vec_bytes(chain) // DTYPE_BYTES[chain.output_dtype]) if isinstance(chain.moe, MoeSwapAbSpec) else vsize
    if isinstance(chain.moe, MoeSwapAbSpec) and chain.quants:
        red_chunk = gcd(red_chunk, chain.moe.offset_multiple, chain.matmul.N)
    for red_idx, red in enumerate(chain.reductions):
        red_source = _parent_value(red.source_ref)
        body_lines.extend(
            _emit_reduction_atomic(
                _tap_of[len(specs) + red_idx], red_idx, red, red_source, chain, vsize, store_row_pred, col_bound=col_bound, chunk_elems=red_chunk
            )
        )

    if tma_slots and any(red.mode in ("mul", "mul_no_zeros") for red in chain.reductions):
        body_lines.append("nvvm.bar_warp_sync(0xFFFFFFFF)")

    # Split-K partial store handling
    if split_k_slices > 1:
        if tma_slots:
            body_lines = ["vec_out = vec_f32", tma_out_ready_marker(0)]
        else:
            body_lines = _emit_splitk_partial_store(vsize)

    epilogue = "\n".join(body_lines)

    # aux kernel params
    kernel_params = [f"{aux.name}: cute.Tensor" for aux in chain.aux_tensors]
    host_args = [aux.name for aux in chain.aux_tensors]

    # tap plumbing: every output that is NOT on the TMA-C surface, in tap order.
    _slot_of = {t: o for o, t in _tap_of.items()}
    taps = [chain.outputs[_slot_of[i]] for i in range(len(_tap_of))]
    tap_kernel_params = [f"mC_tap_{i}: cute.Tensor" for i in range(len(taps))]
    tap_host_params = [f"c_tap_{i}: cute.Tensor" for i in range(len(taps))]
    tap_host_pass = [f"c_tap_{i}" for i in range(len(taps))]
    tap_compile_fakes: list[str] = []
    # Per-slot true store alignment (matches _alignment_reject's contract); 16 is
    # a false claim for reduction / quant-scale / M-major taps.
    _out_reqs = _output_align_reqs(chain, tma_slots, vec_bytes=vec_bytes_epi)
    for i, tap in enumerate(taps):
        # Byte-carrier taps: packed FP4 data, and an E5M3 scale (the DSL has no
        # E5M3 float type, and a raw_ptr store to a Uint8 tensor is rejected —
        # Int8 is the 8-bit carrier this package already uses for FP4).
        _fp4_data_tap = not tap.is_reduction and not tap.is_quant_scale and tap.dtype == "fp4_e2m1"
        _fake_dt = "cutlass.Int8" if _fp4_data_tap else _scale_store_dtype(tap.dtype) if tap.is_quant_scale else DTYPE_TO_CUTLASS[tap.dtype]
        # Every tap is consumed as a raw pointer (gC_tap_i_ptr) plus explicit
        # out_/red_/quant_scale_stride_* scalars, so the only genuine layout
        # contract is stride_n == 1 on an N-major DENSE tap (_dense_store_offset).
        # M-major dense scatters, and reductions / quant scales index purely by
        # runtime strides (and legitimately carry stride 0 for broadcast modes).
        # Tap i is output slot _slot_of[i]; a slot outside chain.outputs would
        # silently wrap the _out_reqs lookup.
        _si = _slot_of[i]
        assert 0 <= _si < len(_out_reqs), f"tap {i} maps to output slot {_si}, outside chain.outputs ({len(_out_reqs)})"
        _n_major_dense = (not tap.is_reduction) and (not tap.is_quant_scale) and _si < len(specs) and specs[_si].major == "n"
        _stride = "(cute.sym_int64(), 1, cute.sym_int64())" if _n_major_dense else "(cute.sym_int64(), cute.sym_int64(), cute.sym_int64())"
        tap_compile_fakes.append(
            f"fake_c_tap_{i} = cute.runtime.make_fake_tensor(\n"
            f"    {_fake_dt},\n"
            f"    {_tap_fake_shape(tap, chain)},\n"
            f"    stride={_stride},\n"
            f"    assumed_align={_out_reqs[_si]},\n"
            f")"
        )
    tap_compile_pass = [f"fake_c_tap_{i}" for i in range(len(taps))]
    tap_ptr_binds: list[str] = []
    for i in range(len(taps)):
        tap_ptr_binds.append(f"gC_tap_{i}_ptr = mC_tap_{i}.iterator.raw_ptr()")
        tap_ptr_binds.append(f"VEC_BYTES_TAP_{i} = vec_bytes_tap_{i}")
    # vsize is the shared COMPUTE chunk; each tap's STORE width is capped at
    # MAX_MEM_ACCESS_BYTES (a wide dtype under a block-quant splits into sub-stores).
    tap_constants = []
    for i, tap in enumerate(taps):
        # reduction / quant-scale side outputs store scalar (constant unused);
        # only a DENSE tap's store width comes from its own layout alignment.
        vb = DTYPE_BYTES[tap.dtype] if (tap.is_reduction or tap.is_quant_scale) else _tap_vec_bytes(chain, tap.dtype, tap.dim, tap.stride, vsize)
        tap_constants.append(f"vec_bytes_tap_{i} = {vb}")
    # Each per-col / per-elem aux is loaded at ITS OWN alignment (not the output's
    # VEC_BYTES): min(the aux tensor's alignment, the chunk it reads = vsize elems).
    for aux in chain.aux_tensors:
        if aux.bcast_mode in ("per_col", "per_elem"):
            _aalign = _aux_align_reqs(chain, vec_bytes=vec_bytes_epi)[aux.name]
            tap_constants.append(f"ALIGN_AUX_{aux.name} = {_aalign}")

    mainloop_transform_a = generate_mainloop(chain, "a")
    mainloop_transform_b = generate_mainloop(chain, "b")

    return EpilogueSnippets(
        aux_views=aux_views,
        epilogue=epilogue,
        mainloop_transform_a=mainloop_transform_a,
        mainloop_transform_b=mainloop_transform_b,
        kernel_params=kernel_params,
        host_args=host_args,
        tap_kernel_params=tap_kernel_params,
        tap_host_params=tap_host_params,
        tap_host_pass=tap_host_pass,
        tap_compile_fakes=tap_compile_fakes,
        tap_compile_pass=tap_compile_pass,
        tap_ptr_binds=tap_ptr_binds,
        tap_constants=tap_constants,
    )
