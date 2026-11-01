# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Generate kernel epilogue source from fusion IR."""

from __future__ import annotations

from math import gcd

from ..epilogue_codegen import (
    _emit_reduction_avg_inverse,
    _emit_block_quant_row,
    _AuxCodegen,
    _emit_pointwise_dag,
    EpilogueSnippets,
    _aux_ptr_var,
    _aux_prefetch_var,
    tma_out_value,
    tma_out_ready_marker,
    _store_cast_expr,
    _dense_store_offset,
    _emit_mmajor_scatter,
    _tap_vec_bytes,
    _emit_moe_scatter_store,
    _emit_tap_store,
    _reduction_output_offset_expr,
    _emit_reduction_local_combine,
    _quant_output_max,
    _scale_store_dtype,
    _e8m0_f32,
    _emit_scale_quantize_pair,
    generate_mainloop,
    _emit_splitk_partial_store,
)

from ..dtypes import DTYPE_BYTES, DTYPE_TO_CUTLASS, _aux_align_reqs, _compute_output_vec_bytes, _output_align_reqs
from ..fusion_ir import (
    BlockQuantizeSpec,
    Dtype,
    FusionChain,
    MoeSwapAbSpec,
    ReductionSpec,
    TensorRef,
    gemm_index,
    is_gemm_source,
)

_aux_codegen = _AuxCodegen(wide_offsets=False)
_aux_index_expr = _aux_codegen.index_expr
_bounded_aux_prelude = _aux_codegen.bounded_prelude
_scatter_aux_prelude = _aux_codegen.scatter_prelude
_aux_load_expr = _aux_codegen.load_expr


def _emit_moe_combine_store(tap_idx, i, src, spec, chain, vsize):
    """Generate weighted MoE COMBINE stores."""
    swapped = isinstance(chain.moe, MoeSwapAbSpec)
    if not swapped and spec.major == "n":
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
    """A reduction is an atomic RMW, so an out-of-extent element cannot be clamped onto a valid one -- it has to be SKIPPED."""
    body = _emit_reduction_atomic_body(tap_idx, red_idx, red, source_var, chain, vsize)
    if row_pred is None:
        return body
    width = chunk_elems or vsize
    lines = [f"if ({row_pred}) & (col_j + {vsize} <= {col_bound}):"] + [f"    {ln}" for ln in body]
    if width == vsize:
        return lines
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
    """Generate one SM100 reduction contribution."""
    src = f"_red_{red_idx}_src"
    lines = [f"{src} = ({source_var}).to({DTYPE_TO_CUTLASS[red.compute_dtype]})"]
    if red.mode == "avg":
        lines.extend(_emit_reduction_avg_inverse(red_idx, red, chain))
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
    """Emit M-axis (col) block quantize for one epilogue vector."""
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

    lines.append(f"for {p}_vi in cutlass.range_constexpr(0, {vsize}, 4):")
    for lane_in_batch in range(4):
        if quant.block_size == 16:
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
        _st = f"(gC_tap_{scale_tap_idx}_ptr + {p}_sidx{k}).store({p}_scale_mine[{k}], alignment=1)"
        if row_pred is None:
            lines.append(_st)
        else:
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
    """Generate block-quantized data and scale stores."""
    if quant.axis == 1:
        return _emit_block_quant_col(quant, quant_idx, source_var, output_dtype, out_var, scale_tap_idx, chain, batch_index_expr, vsize, row_pred)
    return _emit_block_quant_row(quant, quant_idx, source_var, output_dtype, out_var, scale_tap_idx, chain, batch_index_expr, vsize, row_pred)


def _tap_fake_shape(tap, chain: FusionChain | None = None) -> str:
    """Declare a materialized output's symbolic shape."""
    if chain is not None and chain.has_moe and chain.moe.mode == "combine":
        return "(sym_m, cute.sym_int(), 1)" if isinstance(chain.moe, MoeSwapAbSpec) else "(cute.sym_int(), sym_n, 1)"
    if tap.is_quant_scale:
        if chain is None or not chain.quants:
            raise AssertionError("quant scale tap requires FusionChain context")
        q = chain.quants[int(tap.source.rsplit("_", 1)[1])]
        swapped = isinstance(chain.moe, MoeSwapAbSpec)
        if q.grouped_by_moe and ((q.axis == 1) == swapped):
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


def generate(
    chain: FusionChain,
    *,
    vec_bytes_epi: int = 32,
    output_elem_bytes: int = 2,
    tma_slots: "frozenset[int]" = frozenset(),
    packed_lanes: bool = False,
    workspace_split_k: bool = False,
) -> EpilogueSnippets:
    """Produce the two hook-site snippets, the extra kernel param list, and all per-tap plumbing."""
    vsize = vec_bytes_epi // output_elem_bytes
    scatter = chain.has_moe and chain.moe.mode == "scatter"
    row_bound = "group_end" if chain.has_moe and not isinstance(chain.moe, MoeSwapAbSpec) else "M"
    col_bound = "group_end" if isinstance(chain.moe, MoeSwapAbSpec) else "N"
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

    on_tma_arm = bool(tma_slots)
    tma_vec_bindings = [f"vec_f32_{g} = c_rmem_vecs[{g}]" for g in range(1, chain.num_gemms)] if on_tma_arm else []
    store_row_pred = None
    bounded = scatter or on_tma_arm or (isinstance(chain.moe, MoeSwapAbSpec) and bool(chain.quants))
    if bounded:
        store_row_pred = f"row < {row_bound}"
        if packed_lanes:
            store_row_pred = f"row_active & ({store_row_pred})"
    _aux_pre = _scatter_aux_prelude(chain) if scatter else _bounded_aux_prelude(chain, row_bound, col_bound, bounded)
    body_lines: list[str] = tma_vec_bindings + _aux_pre

    # Per-op result var name lookup (handles `identity` pass-throughs).
    pointwise_lines, _parent_value = _emit_pointwise_dag(chain, vsize=vsize, bounded=bounded, aux_load_expr=_aux_load_expr)
    body_lines.extend(pointwise_lines)

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
    if workspace_split_k:
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
    _out_reqs = _output_align_reqs(chain, tma_slots, vec_bytes=vec_bytes_epi)
    for i, tap in enumerate(taps):
        _fp4_data_tap = not tap.is_reduction and not tap.is_quant_scale and tap.dtype == "fp4_e2m1"
        _fake_dt = "cutlass.Int8" if _fp4_data_tap else _scale_store_dtype(tap.dtype) if tap.is_quant_scale else DTYPE_TO_CUTLASS[tap.dtype]
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
    tap_constants = []
    for i, tap in enumerate(taps):
        vb = DTYPE_BYTES[tap.dtype] if (tap.is_reduction or tap.is_quant_scale) else _tap_vec_bytes(chain, tap.dtype, tap.dim, tap.stride, vsize)
        tap_constants.append(f"vec_bytes_tap_{i} = {vb}")
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
