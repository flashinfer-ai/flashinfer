# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Shared epilogue emitters and active-family entry points."""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass, field

from .arch_family import active_family
from .dtypes import DTYPE_BYTES, DTYPE_TO_CUTLASS, _allowed_vsize, dense_output_layout
from .fusion_ir import BlockQuantizeSpec, Dtype, FusionChain, FusionOp, MoeSwapAbSpec, ReductionSpec, TensorRef, gemm_index, is_gemm_source


@dataclass(frozen=True)
class EpilogueSnippets:
    aux_views: str
    epilogue: str
    kernel_params: list[str]
    host_args: list[str]
    tap_kernel_params: list[str] = field(default_factory=list)
    tap_host_params: list[str] = field(default_factory=list)
    tap_host_pass: list[str] = field(default_factory=list)
    tap_compile_fakes: list[str] = field(default_factory=list)
    tap_compile_pass: list[str] = field(default_factory=list)
    tap_ptr_binds: list[str] = field(default_factory=list)
    tap_constants: list[str] = field(default_factory=list)
    mainloop_transform_a: str = "pass"
    mainloop_transform_b: str = "pass"


def _aux_ptr_var(name: str) -> str:
    return f"_aux_{name}_ptr"


def _aux_prefetch_var(name: str) -> str:
    return f"_aux_{name}_pre"


def _compute_cast(var: str, dtype: Dtype, tag: str, *, source_dtype: Dtype | None = None) -> tuple[list[str], str]:
    """Cast a running vector to the op's compute dtype."""
    new = f"_c_{tag}"
    value = f"({var})"
    if dtype == "int32" and source_dtype in ("int8", "uint8", "int32"):
        value = f"cutlass.Vector({value}.ir_value(), dtype={DTYPE_TO_CUTLASS[source_dtype]})"
    lines = [f"{new} = {value}.to({DTYPE_TO_CUTLASS[dtype]})"]
    if dtype == "fp32" and source_dtype in ("int8", "uint8", "int32"):
        lines.append(f"{new} = {new} + cutlass.full_like({new}, 0.0)")
    return lines, new


def _compute_literal(dtype: Dtype, value: float | int) -> str:
    """Format a literal in the requested compute dtype."""
    if dtype == "int32":
        return f"cutlass.Int32({int(value)})"
    return f"cutlass.Float32({float(value)})"


def _elem_coord(e: int) -> tuple[str, str]:
    """Return an epilogue element coordinate."""
    return ("row", f"col_j + {e}")


def tma_out_value(j: int) -> str:
    """Return the variable for a TMA output."""
    return "vec_out" if j == 0 else f"_tma_out_{j}"


def tma_out_ready_marker(j: int) -> str:
    """Codegen marker immediately after TMA output ``j`` is ready."""
    return f"# @@FROST_TMA_OUT_{j}_READY@@"


def _aux_reads_row(aux: TensorRef) -> bool:
    return len(aux.dim) >= 2 and aux.dim[-2] != 1


_LOG2E = "cutlass.Float32(1.4426950408889634)"

_TWO_LOG2E = "cutlass.Float32(2.8853900817779268)"


def _tanh_expr(v: str) -> str:
    """Generate vector tanh."""
    one = f"cutlass.full_like({v}, cutlass.Float32(1.0))"
    two = f"cutlass.full_like({v}, cutlass.Float32(2.0))"
    e2 = f"cute.math.exp2({v} * cutlass.full_like({v}, {_TWO_LOG2E}), fastmath=True)"
    return f"({one} - {two} * cute.math.rcp({e2} + {one}, approx=True, ftz=True))"


def _fl(var: str, c: float) -> str:
    return f"cutlass.full_like({var}, cutlass.Float32({float(c)}))"


def _step01(d_expr: str, tag: str) -> tuple[list[str], str]:
    """Generate a nonnegative indicator vector."""
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

_CMP_PY_OPS = {"cmp_eq": "==", "cmp_neq": "!=", "cmp_gt": ">", "cmp_ge": ">=", "cmp_lt": "<", "cmp_le": "<="}

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
    """Generate comparison, logical, modulo, and activation-gradient operations."""
    a, b = f"_ba{idx}", f"_bb{idx}"
    lines = [f"{a} = ({a_expr})", f"{b} = ({b_expr})"]
    attrs = dict(op.attrs)

    if op.op in _CMP_OPS:
        if op.compute_dtype == "fp32":
            # Compare natively (arith.cmpf: ordered OEQ/OGT/OGE/OLT/OLE, unordered UNE) to keep IEEE
            # semantics. The step form below gets NaN operands and equal infinities wrong: a - b is NaN
            # there, and the NaN-ignoring max/min turn that NaN into 0. Only int32 compute still reaches
            # the step form, and it does not build there (the DSL's floor takes floating-point types only).
            return lines + [f"{new} = cutlass._mlir_helpers.vector.where({a} {_CMP_PY_OPS[op.op]} {b}, {_fl(a, 1.0)}, {_fl(a, 0.0)})"], new
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
            # cuDNN RELU_BWD: the slope applies only where x <= lower (s1); x >= upper has zero gradient.
            lines.append(f"_rf{idx} = {m} + {_fl(b, slope)} * {s1}")
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
    """Emit lines computing this op, given the previous-step var name."""
    new = f"_op_{idx}"

    if op.op == "identity":
        return [], prev

    # relu (every slope / clip form), leaky_relu and elu keep a NaN input NaN, as torch does and as the frontend
    # requests for RELU_FWD (CUDNN_PROPAGATE_NAN). The DSL's default max/min (maxnum/minnum) return the non-NaN
    # operand, which turned NaN into 0 or lower_clip; propagate_nan=True lowers to arith.maximumf/minimumf. The
    # binary max/min ops keep the default, which matches native cuDNN (eng10) MAX/MIN.
    if op.op == "relu":
        ra = dict(op.attrs)
        if not ra:
            return [f"{new} = cute.math.max({prev}, cutlass.full_like({prev}, {_compute_literal(op.compute_dtype, 0)}), propagate_nan=True)"], new
        slope, lower, upper = ra.get("negative_slope"), ra.get("lower_clip"), ra.get("upper_clip")
        rl = [f"_r{idx} = {prev}"]
        rv = f"_r{idx}"
        if slope is not None:
            # cuDNN RELU_FWD: x < lower ? lower + slope * (x - lower) : x, with lower defaulting to 0.
            # lower_clip is where the slope starts, not a clamp applied after it.
            lo = 0.0 if lower is None else lower
            shifted = rv if lo == 0.0 else f"({rv} - {_fl(rv, lo)})"
            rl.append(
                f"_rk{idx} = cute.math.max({rv}, {_fl(rv, lo)}, propagate_nan=True) + {_fl(rv, slope)} * cute.math.min({shifted}, {_fl(rv, 0.0)}, propagate_nan=True)"
            )
            cur = f"_rk{idx}"
        else:
            rl.append(f"_rl{idx} = cute.math.max({rv}, {_fl(rv, lower if lower is not None else 0.0)}, propagate_nan=True)")
            cur = f"_rl{idx}"
        if upper is not None:
            rl.append(f"_ru{idx} = cute.math.min({cur}, {_fl(rv, upper)}, propagate_nan=True)")
            cur = f"_ru{idx}"
        return rl + [f"{new} = {cur}"], new

    if op.op == "leaky_relu":
        slope = dict(op.attrs).get("negative_slope", 0.0)
        return [
            f"_lr{idx} = {prev}",
            f"{new} = cute.math.max(_lr{idx}, {_fl(f'_lr{idx}', 0.0)}, propagate_nan=True) + {_fl(f'_lr{idx}', slope)} * cute.math.min(_lr{idx}, {_fl(f'_lr{idx}', 0.0)}, propagate_nan=True)",
        ], new

    if op.op == "elu":
        return [
            f"_el{idx} = {prev}",
            f"{new} = cute.math.max(_el{idx}, {_fl(f'_el{idx}', 0.0)}, propagate_nan=True) + cute.math.exp2(cute.math.min(_el{idx}, {_fl(f'_el{idx}', 0.0)}, propagate_nan=True) * cutlass.full_like(_el{idx}, {_LOG2E}), fastmath=True) - {_fl(f'_el{idx}', 1.0)}",
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
        # An int32-compute index comes straight from the Int32 coordinate: Float32 rounds indices above 2**24.
        idx_t = DTYPE_TO_CUTLASS[op.compute_dtype]
        if axis == 1:
            return [f"{new} = cutlass.full_like({prev}, {idx_t}(row))"], new
        gl = [f"_gi{idx} = cute.make_rmem_tensor({vsize}, {idx_t})"]
        for k in range(vsize):
            _, col = _elem_coord(k)
            gl.append(f"_gi{idx}[{k}] = {idx_t}({col})")
        gl.append(f"{new} = _gi{idx}.load().to_vector()")
        return gl, new

    if op.op == "binary_select":
        assert other_in_chain is not None and third_in_chain is not None
        zero = _compute_literal(op.compute_dtype, 0)
        return [f"{new} = cutlass._mlir_helpers.vector.where({third_in_chain} != cutlass.full_like({third_in_chain}, {zero}), {prev}, {other_in_chain})"], new

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

        return [f"{new} = cute.math.exp2({prev} * cutlass.full_like({prev}, {_LOG2E}), fastmath=True)"], new

    if op.op == "abs":
        return [f"{new} = cute.math.abs({prev})"], new

    if op.op == "neg":
        if op.compute_dtype == "int32":
            return [f"{new} = cutlass.full_like({prev}, cutlass.Int32(0)) - {prev}"], new
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

        return [
            f"{new} = {prev} * cute.math.rcp(cutlass.full_like({prev}, cutlass.Float32(1.0)) + "
            f"cute.math.exp2(-{prev} * cutlass.full_like({prev}, {_LOG2E}), fastmath=True), approx=True, ftz=True)"
        ], new

    if op.op == "gelu_tanh":

        return [
            f"_g_x{idx} = {prev}",
            f"_g_inner{idx} = cutlass.full_like(_g_x{idx}, cutlass.Float32(0.7978845608028654)) * "
            f"(_g_x{idx} + cutlass.full_like(_g_x{idx}, cutlass.Float32(0.044715)) * _g_x{idx} * _g_x{idx} * _g_x{idx})",
            f"_g_tanh{idx} = {_tanh_expr(f'_g_inner{idx}')}",
            f"{new} = cutlass.full_like(_g_x{idx}, cutlass.Float32(0.5)) * _g_x{idx} * (cutlass.full_like(_g_x{idx}, cutlass.Float32(1.0)) + _g_tanh{idx})",
        ], new

    if op.op == "gelu":

        return [
            f"_e_x{idx} = {prev}",
            f"_e_erf{idx} = cute.math.erf(_e_x{idx} * cutlass.full_like(_e_x{idx}, {_compute_literal(op.compute_dtype, 0.7071067811865475)}))",
            f"{new} = cutlass.full_like(_e_x{idx}, {_compute_literal(op.compute_dtype, 0.5)}) * _e_x{idx} * "
            f"(cutlass.full_like(_e_x{idx}, {_compute_literal(op.compute_dtype, 1)}) + _e_erf{idx})",
        ], new

    if op.op in {"add", "mul", "sub", "div", "max", "min", "pow", "add_square"}:
        py_op = {"add": "+", "mul": "*", "sub": "-", "div": "/"}.get(op.op)
        if op.parent_idx_b is not None:

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


# int8 / uint8 values round to nearest even and saturate, NaN to 0 (cvt.rni.sat), as cuDNN's native engines store
# them. A bare .to() lowers to fptosi / trunci, which round toward zero and keep the low byte (2.75 -> 2, 300 -> 44).
_INT8_RANGES: dict[str, tuple[int, int]] = {"int8": (-128, 127), "uint8": (0, 255)}


def _emit_round(var: str, out_dtype: Dtype, tag: str, *, source_dtype: Dtype = "fp32") -> tuple[list[str], str]:
    """Round a register value to its declared dtype."""
    if out_dtype == "fp32" and source_dtype == "fp32":
        return [], var
    new = f"_r_{tag}"
    if out_dtype in _INT8_RANGES and source_dtype in ("fp32", "int32"):
        lo, hi = _INT8_RANGES[out_dtype]
        sat = f"_sat_{tag}"
        if source_dtype == "int32":
            lo_i, hi_i = _compute_literal("int32", lo), _compute_literal("int32", hi)
            lines = [f"{sat} = cute.math.min(cute.math.max({var}, cutlass.full_like({var}, {lo_i})), cutlass.full_like({var}, {hi_i}))"]
        else:
            # max(x, 0) + min(x, 0) is x, or 0 for NaN: max / min return the non-NaN operand.
            lines = [
                f"{sat} = cute.math.max({var}, {_fl(var, 0.0)}) + cute.math.min({var}, {_fl(var, 0.0)})",
                f"{sat} = cute.math.min(cute.math.max(cute.math.roundeven({sat}), {_fl(sat, lo)}), {_fl(sat, hi)})",
            ]
        return lines + [f"{new} = ({sat}).to({DTYPE_TO_CUTLASS[out_dtype]})"], new
    lines = [f"{new} = ({var}).to({DTYPE_TO_CUTLASS[out_dtype]})"]
    if out_dtype == "fp32" and source_dtype == "int32":
        lines.append(f"{new} = {new} + cutlass.full_like({new}, 0.0)")
    return lines, new


def _store_cast_expr(var: str, dtype: Dtype) -> str:
    """Format an output conversion expression."""
    if dtype == "uint8":

        return f"({var}).to(cutlass.Uint8).bitcast(cutlass.Int8)"
    return f"({var}).to({DTYPE_TO_CUTLASS[dtype]})"


def _dense_store_offset(i: int, is_fp4: bool, batch: int) -> str:
    """Generate a dense output store offset."""
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
    """Generate scalar stores through output strides."""
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
    """Choose the vector width for a dense output."""
    if dtype == "fp4_e2m1":
        return vsize
    dim, stride = dense_output_layout(chain, dtype, dim, stride)
    return min(vsize, _allowed_vsize(chain, dtype, dim, stride))


def _tap_vec_bytes(chain: FusionChain, dtype: Dtype, dim, stride, vsize: int) -> int:
    """Byte width (== required alignment) of one dense tap vector store."""
    if dtype == "fp4_e2m1":
        return max(vsize // 2, 4)
    return _tap_store_elems(chain, dtype, dim, stride, vsize) * DTYPE_BYTES[dtype]


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
    """Generate vector stores for one N-major output."""
    tap_var = f"_tap_{tap_idx}"
    store_elems = _tap_store_elems(chain, tap_dtype, dim, stride, vsize)
    lines = [f"{tap_var} = {source_var if converted else _store_cast_expr(source_var, tap_dtype)}"]
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
    """Generate a reduction output offset."""
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
    """Fold one register vector before its output atomic."""
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


def _quant_output_max(dtype: Dtype) -> str:
    """Return the maximum quantized output magnitude."""
    if dtype == "fp8_e4m3":
        return "cutlass.Float32(448.0)"
    if dtype == "fp8_e5m2":
        return "cutlass.Float32(57344.0)"
    if dtype == "fp4_e2m1":
        return "cutlass.Float32(6.0)"
    raise ValueError(f"block quantize output dtype {dtype!r} is not supported by codegen")


def _scale_store_dtype(scale_dtype: Dtype) -> str:
    """Return the storage type of a quantized scale."""
    return "cutlass.Int8" if scale_dtype == "fp8_e5m3" else DTYPE_TO_CUTLASS[scale_dtype]


def _f8_128x4_row_scale_index_expr(row: str, scale_col: str, n_col_quads: str, *, atom_base: str | None = None) -> str:
    """Physical byte index for one logical row scale in an F8_128x4 blob."""
    prefix = f"{atom_base} + " if atom_base is not None else ""
    return f"{prefix}(({row} // 128) * {n_col_quads} + ({scale_col} // 4)) * 512 + " f"({row} % 32) * 16 + (({row} % 128) // 32) * 4 + ({scale_col} % 4)"


def _e8m0_f32(bits: str, *, reciprocal: bool = False) -> str:
    """Widen a scale or its exact reciprocal without flushing subnormals."""
    exponent = f"(254 - {bits})" if reciprocal else bits
    special = f"({bits} >= 254)" if reciprocal else f"(({bits} == 0) | ({bits} == 255))"
    return f"(({exponent} << 23) | ({special}.to(cutlass.Int32) << 22)).bitcast(cutlass.Float32)"


def _emit_scale_quantize(p: str, sfx: str, src: str, scale_var: str, back_var: str, quant: BlockQuantizeSpec) -> list[str]:
    """Quantize a scale and recover its stored value."""
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
    """:func:`_emit_scale_quantize` for two scales at once."""
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


_MAINLOOP_IDX_BASE = {"a": 900, "b": 800}


_MAINLOOP_IDENTITY_CAST_INTRINSICS: dict[tuple[Dtype, Dtype], str] = {
    ("int8", "bf16"): "cute.arch.cvt_i8_bf16_intrinsic",
}


def generate_mainloop(chain: FusionChain, operand: str = "a") -> str:
    """Generate one operand mainloop transform."""
    ops = chain.mainloop_a_ops if operand == "a" else chain.mainloop_b_ops
    if not ops:
        return "pass"

    src_dtype = chain.matmul.a_dtype if operand == "a" else chain.matmul.b_dtype
    ab_dtype = DTYPE_TO_CUTLASS[src_dtype]
    vec_var = f"ml_vec_{operand}"
    out_var = f"ml_out_{operand}"
    base = _MAINLOOP_IDX_BASE[operand]
    load_dtype = chain.mainloop_a_load_dtype if operand == "a" else chain.mainloop_b_load_dtype
    identity_cast_intrinsic = (
        _MAINLOOP_IDENTITY_CAST_INTRINSICS.get((load_dtype, src_dtype)) if len(ops) == 1 and ops[0].op == "identity" and load_dtype is not None else None
    )
    if identity_cast_intrinsic is not None:
        cvt = f"{identity_cast_intrinsic}({vec_var}.ir_value(), ml_vec_elems)"
        return f"{out_var} = cutlass.Vector({cvt}, dtype={ab_dtype})"
    lines: list[str] = []
    result_var: dict[int, str] = {-1: vec_var}
    for i, op in enumerate(ops):
        parent = op.resolved_parent_idx(i)
        tag = f"ml_{operand}_{i}"
        parent_dtype = (load_dtype or src_dtype) if parent < 0 else ops[parent].out_dtype
        cast_lines, parent_var = _compute_cast(result_var[parent], op.compute_dtype, tag, source_dtype=parent_dtype)
        lines.extend(cast_lines)
        aux_loads = {}
        if op.aux is not None:
            ptr = f"{op.aux}.iterator.raw_ptr()"
            ref = chain.aux_by_name(op.aux)
            if ref.bcast_mode == "per_k":
                dt = DTYPE_TO_CUTLASS[op.compute_dtype]
                # The renderer selects logical (unswizzled) vector accesses for
                # this path. The operand is K-major, so vector lanes advance K.
                lines.extend(
                    [
                        f"{tag}_scale = cute.make_rmem_tensor(ml_vec_elems, {dt})",
                        f"{tag}_k = k_tile_idx * cta_tile_mnk[2] + ml_L % cta_tile_mnk[2]",
                        f"for {tag}_lane in cutlass.range_constexpr(ml_vec_elems):",
                        f"    {tag}_gk = {tag}_k + {tag}_lane",
                        f"    {tag}_scale[{tag}_lane] = {dt}(0.0)",
                        f"    if {tag}_gk < k:",
                        f"        {tag}_scale[{tag}_lane] = ({ptr} + {tag}_gk).load().to({dt})",
                    ]
                )
                aux_loads[op.aux] = f"{tag}_scale.load().to_vector()"
            else:
                aux_loads[op.aux] = f"cutlass.full_like({parent_var}, ({ptr} + 0).load().to({DTYPE_TO_CUTLASS[op.compute_dtype]}))"
        emit_lines, cur = _emit_op(op, parent_var, base + i, aux_loads)
        lines.extend(emit_lines)
        if op.out_dtype in ("fp8_e4m3", "fp8_e5m2") and (op.compute_dtype == "int32" or (op.op == "identity" and parent_dtype in ("int8", "uint8", "int32"))):
            cur = f"(({cur}).to(cutlass.Float32) + cutlass.full_like(({cur}).to(cutlass.Float32), 0.0))"
        round_lines, cur = _emit_round(cur, op.out_dtype, tag, source_dtype=op.compute_dtype)
        lines.extend(round_lines)
        result_var[i] = cur
    terminal_var = result_var[len(ops) - 1]
    lines.append(f"{out_var} = ({terminal_var}).to({ab_dtype})")
    major = "a_is_m_major" if operand == "a" else "b_is_n_major"
    group = f"{operand}_tma_group_elems"
    lines.extend(
        [
            "if cutlass.const_expr(mainloop_koob_fix):",
            f"    if cutlass.const_expr({major}):",
            f"        ml_k_of_L = (ml_L % ({group} * cta_tile_mnk[2])) // {group}",
            "    else:",
            "        ml_k_of_L = ml_L % cta_tile_mnk[2]",
            f"    ml_valid = cutlass.full_like({out_var}.to(cutlass.Float32), cutlass.Float32(ml_k_of_L)) < cutlass.full_like({out_var}.to(cutlass.Float32), cutlass.Float32(k - k_tile_idx * cta_tile_mnk[2]))",
            f"    {out_var} = cutlass._mlir_helpers.vector.where(ml_valid, {out_var}, cutlass.full_like({out_var}, 0.0))",
        ]
    )
    return "\n".join(lines)


def _emit_splitk_partial_store(vsize: int) -> list[str]:
    """Generate stores for split-K partial accumulators."""
    chunk = min(vsize, 4)
    lines = [f"_split_k_off = (tile_l * M + row) * N + col_j"]
    for s in range(0, vsize, chunk):
        lines.append(f"(gSplitK_partials_ptr + _split_k_off + {s}).store(vec_f32[{s} : {s + chunk}], alignment={chunk * 4})")
    return lines


@dataclass(frozen=True)
class _AuxCodegen:
    wide_offsets: bool = False

    def index_term(self, var: str, stride: int) -> str:
        if stride == 1:
            return var
        return f"cutlass.Int64({var}) * {stride}" if self.wide_offsets else f"{var} * {stride}"

    def index_expr(self, aux: TensorRef, *, row_var: str = "row", col_var: str = "col_j") -> str:
        """Generate an auxiliary tensor offset."""
        if len(aux.dim) == 1:
            axes = ((aux.dim[0], aux.stride[0], col_var),)
        elif len(aux.dim) == 2:
            axes = ((aux.dim[0], aux.stride[0], row_var), (aux.dim[1], aux.stride[1], col_var))
        elif len(aux.dim) == 3:
            lead_var = "group_idx" if aux.grouped_by_moe else "tile_l"
            axes = ((aux.dim[0], aux.stride[0], lead_var), (aux.dim[1], aux.stride[1], row_var), (aux.dim[2], aux.stride[2], col_var))
        else:
            raise ValueError(f"unsupported aux rank {len(aux.dim)} for {aux.name!r}")
        terms = [self.index_term(var, stride) for dim, stride, var in axes if dim != 1]
        return " + ".join(terms) if terms else "0"

    def bounded_prelude(self, chain: FusionChain, row_bound: str, col_bound: str, on_tma_arm: bool) -> list[str]:
        """Generate bounded auxiliary loads."""
        lines: list[str] = []
        for aux in chain.aux_tensors:
            if aux.bcast_mode not in ("per_col", "per_elem") or (not on_tma_arm and aux.stride[-1] == 1):
                continue
            n, ptr = (aux.name, _aux_ptr_var(aux.name))
            lines.append(f"_auxt_{n} = cute.make_rmem_tensor(vsize, {DTYPE_TO_CUTLASS[aux.dtype]})")
            lines.append(f"_auxv_{n} = _auxt_{n}.load().to_vector()")
            cond = [f"col_j + vsize <= {col_bound}", str(aux.stride[-1] == 1)]
            if _aux_reads_row(aux):
                cond.append(f"row < {row_bound}")
            lines.append(f"if {' & '.join((f'({c})' for c in cond))}:")
            lines.append(f"    _auxv_{n} = ({ptr} + {self.index_expr(aux)}).load(count=vsize, alignment=ALIGN_AUX_{n})")
            lines.append("else:")
            row_var = f"_auxr_{n}"
            if _aux_reads_row(aux):
                lines.append(f"    {row_var} = cute.math.min(cutlass.Int32(row), cutlass.Int32({row_bound}) - 1)")
            else:
                row_var = "row"
            lines.append("    for _auxk in cutlass.range_constexpr(vsize):")
            lines.append(f"        _auxc_{n} = cute.math.min(cutlass.Int32(col_j) + _auxk, cutlass.Int32({col_bound}) - 1)")
            idx = self.index_expr(aux, row_var=row_var, col_var=f"_auxc_{n}")
            lines.append(f"        _auxt_{n}[_auxk] = ({ptr} + {idx}).load()")
            lines.append(f"    _auxv_{n} = _auxt_{n}.load().to_vector()")
        return lines

    def scatter_prelude(self, chain: FusionChain) -> list[str]:
        """Generate scatter auxiliary loads."""
        swapped = isinstance(chain.moe, MoeSwapAbSpec)
        row = "row" if swapped else "moe_scatter_row(token_index, token_ks, row, group_end, M, moe_top_k)"
        col = "moe_scatter_row(token_index, token_ks, col_j + _auxk, group_end, N, moe_top_k)" if swapped else "col_j + _auxk"
        lines = []
        for aux in chain.aux_tensors:
            if aux.bcast_mode == "per_row":
                idx = self.index_expr(aux, row_var=f"cute.math.min(cutlass.Int32({row}), cutlass.Int32(M) - 1)")
                lines.append(f"{_aux_prefetch_var(aux.name)} = ({_aux_ptr_var(aux.name)} + {idx}).load()")
            if aux.bcast_mode not in ("per_col", "per_elem"):
                continue
            n = aux.name
            idx = self.index_expr(
                aux, row_var=f"cute.math.min(cutlass.Int32({row}), cutlass.Int32(M) - 1)", col_var=f"cute.math.min(cutlass.Int32({col}), cutlass.Int32(N) - 1)"
            )
            lines += [
                f"_auxt_{n} = cute.make_rmem_tensor(vsize, {DTYPE_TO_CUTLASS[aux.dtype]})",
                "for _auxk in cutlass.range_constexpr(vsize):",
                f"    _auxt_{n}[_auxk] = ({_aux_ptr_var(n)} + {idx}).load()",
                f"_auxv_{n} = _auxt_{n}.load().to_vector()",
            ]
        return lines

    def load_expr(self, aux: TensorRef, compute_dtype: Dtype, like_var: str, *, bounded: bool = False) -> str:
        """Generate an auxiliary load expression."""
        idx = self.index_expr(aux)
        cast = DTYPE_TO_CUTLASS[compute_dtype]
        if aux.bcast_mode == "scalar":
            return f"cutlass.full_like({like_var}, {_aux_prefetch_var(aux.name)}.to({cast}))"
        if aux.bcast_mode == "per_row":
            return f"cutlass.full_like({like_var}, {_aux_prefetch_var(aux.name)}.to({cast}))"
        if aux.bcast_mode in ("per_col", "per_elem"):
            if bounded:
                return f"_auxv_{aux.name}.to({cast})"
            return f"({_aux_ptr_var(aux.name)} + {idx}).load(count=vsize, alignment=ALIGN_AUX_{aux.name}).to({cast})"
        raise AssertionError(f"unknown bcast_mode {aux.bcast_mode!r}")


def _emit_pointwise_dag(chain, *, vsize, bounded, aux_load_expr, gemm_out_dtype=None):
    """Generate per-GEMM rounding and the shared pointwise DAG."""
    scatter = chain.has_moe and chain.moe.mode == "scatter"
    body_lines = []
    result_var: dict[int, str] = {}
    gemm_var: dict[int, str] = {}
    for g in range(chain.num_gemms):
        src = "vec_f32" if g == 0 else f"vec_f32_{g}"
        tag = "mm" if g == 0 else f"mm{g}"
        out_dtype = gemm_out_dtype(g) if gemm_out_dtype is not None else chain.gemm_out_dtype(g)
        if chain.matmul.accum_dtype == "int32" and not chain.widen_integer_accumulators and out_dtype != "int32":
            widened = f"_f_{tag}"
            body_lines.append(f"{widened} = ({src}).to(cutlass.Float32)")
            body_lines.append(f"{widened} = {widened} + cutlass.full_like({widened}, 0.0)")
            src = widened
        round_lines, var = _emit_round(src, out_dtype, tag)
        body_lines.extend(round_lines)
        gemm_var[g] = var

    def _parent_value(ref: int) -> str:
        """Var name for an op input: a GEMM output (ref < 0) or a prior op."""
        if is_gemm_source(ref):
            return gemm_var[gemm_index(ref)]
        return result_var[ref]

    def _parent_dtype(ref: int) -> Dtype:
        return chain.gemm_out_dtype(gemm_index(ref)) if is_gemm_source(ref) else chain.ops[ref].out_dtype

    for i, op in enumerate(chain.ops):
        if op.op == "aux_load":
            aux_ref = chain.aux_by_name(op.aux)
            body_lines.append(f"_op_{i} = {aux_load_expr(aux_ref, op.compute_dtype, 'vec_f32', bounded=bounded or aux_ref.stride[-1] != 1)}")
            round_lines, cur = _emit_round(f"_op_{i}", op.out_dtype, str(i), source_dtype=op.compute_dtype)
            body_lines.extend(round_lines)
            result_var[i] = cur
            continue
        parent = op.resolved_parent_idx(i)
        parent_raw = _parent_value(parent)
        cast_lines, parent_var = _compute_cast(parent_raw, op.compute_dtype, f"{i}_a", source_dtype=_parent_dtype(parent))
        body_lines.extend(cast_lines)
        aux_loads = {aux.name: aux_load_expr(aux, op.compute_dtype, parent_var, bounded=bounded or aux.stride[-1] != 1) for aux in chain.aux_tensors}
        other_in_chain = _parent_value(op.parent_idx_b) if op.parent_idx_b is not None else None
        if other_in_chain is not None:
            cast_lines, other_in_chain = _compute_cast(other_in_chain, op.compute_dtype, f"{i}_b", source_dtype=_parent_dtype(op.parent_idx_b))
            body_lines.extend(cast_lines)
        third_in_chain = _parent_value(op.parent_idx_c) if op.parent_idx_c is not None else None
        if third_in_chain is not None:
            cast_lines, third_in_chain = _compute_cast(third_in_chain, op.compute_dtype, f"{i}_c", source_dtype=_parent_dtype(op.parent_idx_c))
            body_lines.extend(cast_lines)
        lines, cur = _emit_op(op, parent_var, i, aux_loads, other_in_chain, third_in_chain, vsize=vsize)
        if scatter and op.op == "gen_index":
            token_axis = 2 if isinstance(chain.moe, MoeSwapAbSpec) else 1
            if dict(op.attrs).get("axis") == token_axis:
                idx_t = DTYPE_TO_CUTLASS[op.compute_dtype]
                if token_axis == 1:
                    lines = [f"{cur} = cutlass.full_like({parent_var}, {idx_t}(moe_scatter_row(token_index, token_ks, row, group_end, M, moe_top_k)))"]
                else:
                    lines = [
                        f"_gi{i} = cute.make_rmem_tensor(vsize, {idx_t})",
                        "for _sk in cutlass.range_constexpr(vsize):",
                        f"    _gi{i}[_sk] = {idx_t}(moe_scatter_row(token_index, token_ks, col_j + _sk, group_end, N, moe_top_k))",
                        f"{cur} = _gi{i}.load().to_vector()",
                    ]
        body_lines.extend(lines)
        round_lines, cur = _emit_round(cur, op.out_dtype, str(i), source_dtype=op.compute_dtype)
        body_lines.extend(round_lines)
        result_var[i] = cur
    return (body_lines, _parent_value)


def _emit_block_quant_row(
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
    """Generate row block-quantized data and scales."""
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
                f'{p}_out[{base + e}], {p}_out[{base + e + 1}] = cute.arch.mul_packed_f32x2(({p}_src[{base + e}], {p}_src[{base + e + 1}]), ({p}_inv{k}, {p}_inv{k}), rnd="rn", ftz=False)'
            )
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
        lines.extend([f"{p}_local_row = row - group_begin", f"{p}_ncb = ((N // {bs}) + 3) // 4", f"{p}_base = start_sf_block_m * {p}_ncb * 512"])
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
                        f"{p}_sidx{k} = {batch_index_expr} * quant_scale_stride_l_{quant_idx} + ((row // 128) * {p}_ncb{k} + ({p}_scol{k} // 4)) * 512 + (row % 32) * 16 + ((row % 128) // 32) * 4 + ({p}_scol{k} % 4)",
                    ]
                )
        else:
            lines.append(
                f"{p}_sidx{k} = {batch_index_expr} * quant_scale_stride_l_{quant_idx} + row * quant_scale_stride_m_{quant_idx} + {p}_scol{k} * quant_scale_stride_n_{quant_idx}"
            )
        _st = f"(gC_tap_{scale_tap_idx}_ptr + {p}_sidx{k}).store({p}_scale{k}, alignment=1)"
        if row_pred is None:
            lines.append(_st)
        else:
            width = (k + 1) * bs if swapped else vsize
            lines.append(f"if ({row_pred}) & (col_j + {width} <= {col_bound}):")
            lines.append(f"    {_st}")
    return lines


def _emit_reduction_avg_inverse(red_idx: int, red: ReductionSpec, chain: FusionChain, *, token_extent: str | None = None) -> list[str]:
    """Generate AVG normalization from runtime extents.

    A MoE kernel's token extent is the routed capacity, but its groups cover only the rows below the
    final offset. ``token_extent`` names that used-row count for an ungrouped AVG over the token axis;
    ``None`` divides by the capacity."""
    extents = ["M", "N"]
    if chain.has_moe:
        token_axis = 1 if isinstance(chain.moe, MoeSwapAbSpec) else 0
        if red.grouped_by_moe:
            extents[token_axis] = "cutlass.Int32(group_end) - cutlass.Int32(group_begin)"
        elif token_extent is not None:
            extents[token_axis] = token_extent
    factors = [f"cutlass.Float32({extent})" for dim, extent in zip(red.dim[1:], extents) if dim == 1]
    if not chain.has_moe and red.dim[0] == 1 and chain.matmul.batch != 1:
        factors.append(f"cutlass.Float32({float(chain.matmul.batch)})")
    count = " * ".join(factors) or "cutlass.Float32(1.0)"
    return [f"_red_{red_idx}_inv = cutlass.Float32(1.0) / ({count})"]


_SHARED_EXPORTS = {
    "_emit_reduction_avg_inverse": _emit_reduction_avg_inverse,
    "_emit_block_quant_row": _emit_block_quant_row,
    "_AuxCodegen": _AuxCodegen,
    "_emit_pointwise_dag": _emit_pointwise_dag,
    "EpilogueSnippets": EpilogueSnippets,
    "_aux_ptr_var": _aux_ptr_var,
    "_aux_prefetch_var": _aux_prefetch_var,
    "_compute_cast": _compute_cast,
    "_compute_literal": _compute_literal,
    "_elem_coord": _elem_coord,
    "tma_out_value": tma_out_value,
    "tma_out_ready_marker": tma_out_ready_marker,
    "_aux_reads_row": _aux_reads_row,
    "_LOG2E": _LOG2E,
    "_TWO_LOG2E": _TWO_LOG2E,
    "_tanh_expr": _tanh_expr,
    "_fl": _fl,
    "_step01": _step01,
    "_nonzero01": _nonzero01,
    "_sigmoid_lines": _sigmoid_lines,
    "_CMP_OPS": _CMP_OPS,
    "_CMP_PY_OPS": _CMP_PY_OPS,
    "_LOGIC_OPS": _LOGIC_OPS,
    "_BACKWARD_OPS": _BACKWARD_OPS,
    "_emit_binary_ext": _emit_binary_ext,
    "_emit_op": _emit_op,
    "_emit_round": _emit_round,
    "_store_cast_expr": _store_cast_expr,
    "_dense_store_offset": _dense_store_offset,
    "_emit_mmajor_scatter": _emit_mmajor_scatter,
    "_tap_store_elems": _tap_store_elems,
    "_tap_vec_bytes": _tap_vec_bytes,
    "_emit_moe_scatter_store": _emit_moe_scatter_store,
    "_emit_tap_store": _emit_tap_store,
    "_reduction_output_offset_expr": _reduction_output_offset_expr,
    "_emit_reduction_local_combine": _emit_reduction_local_combine,
    "_quant_output_max": _quant_output_max,
    "_scale_store_dtype": _scale_store_dtype,
    "_f8_128x4_row_scale_index_expr": _f8_128x4_row_scale_index_expr,
    "_e8m0_f32": _e8m0_f32,
    "_emit_scale_quantize": _emit_scale_quantize,
    "_emit_scale_quantize_pair": _emit_scale_quantize_pair,
    "_MAINLOOP_IDX_BASE": _MAINLOOP_IDX_BASE,
    "_MAINLOOP_IDENTITY_CAST_INTRINSICS": _MAINLOOP_IDENTITY_CAST_INTRINSICS,
    "generate_mainloop": generate_mainloop,
    "_emit_splitk_partial_store": _emit_splitk_partial_store,
}

_impl = importlib.import_module(f"{__package__}.{active_family()}.epilogue_codegen")
_impl.__dict__.update(_SHARED_EXPORTS)
sys.modules[__name__] = _impl
