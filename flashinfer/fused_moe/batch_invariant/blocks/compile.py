# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Compile wrapper for the grouped GEMM bodies, every operand format: one TVM-FFI object per (knobs, chain[, packing]) with
dynamic Tk, N, K and E, the chain naming its operand format (``bf16`` for the bf16 classes' :class:`GroupedChain`, ``e4m3``
for the fp8 per-tensor classes' :class:`Fp8Chain`, the NVFP4 instruction classes for :class:`W4A4Chain`, the MXFP8 ones
for :class:`.epilogue_mx.MxChain`, ``e4m3_bs`` for the block-scale fp8 classes' :class:`.epilogue_fp8bs.Fp8BsChain`,
which also carries the scale block and the weight interleave), the output format
and the parameters; cache module ``batch_invariant_moe`` in FlashInfer's CuTe-DSL disk cache (this file, the routing,
permutation, combine and scatter kernels are its roots; the bodies of every format are one source closure, so an edit to
any moves every object's key). The persistent grid is a dynamic ``Int32`` argument of the compiled kernel
(``GRID_PLACEHOLDER`` at compile time; the launch passes the device's co-resident cluster count,
``flashinfer.gemm.batch_invariant.api.persistent_clusters``), so one object serves every grid. The chain's declared operands
select the bodies' entry: the plain one (``kernel(a3, b3, out2, counts, grid)``), ``with_scales`` (``kernel(a3, b3, out2,
counts, alpha, sa, si, grid)``), ``block_scaled`` (``kernel(a3, b3, sfa, sfb, out2, counts, alpha, pts, grid)``),
``block_scaled_static`` (``kernel(a3, b3, sfa, sfb, out2, out_sf, counts, alpha, pts, aux, grid)``), or the MX chains'
``block_scaled_plain`` (``kernel(a3, b3, sfa, sfb, out2, counts, rows_sf, grid)``) and ``block_scaled_quant`` (``kernel(a3,
b3, sfa, sfb, out2, out_sf, counts, rows_sf, grid)``: the fused requantization's codes and scale bytes); the block-scale fp8
chains take the same two entries with their fp32 block scales (``rows_sf`` unread; ``out_sf`` the fp32 scales).
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Any, Callable, TypeGuard, Union

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32

from ....batch_invariant import jit
from ....gemm.batch_invariant.blocks.operands import ALIGN, IN_DT, OUT_DT
from ....gemm.batch_invariant.generator import TILES_N, Knobs
from ....gemm.batch_invariant.spec import (
    A_ELEM_BITS_OF,
    BLOCK_K,
    B_ELEM_BITS_OF,
    BLOCK_SCALED,
    SF_ATOM_KBLOCKS,
    SF_ATOM_ROWS,
    SF_VEC,
    k_granule,
)
from .epilogue import (
    GROUP,
    OUT_FMTS_W4A4,
    GroupedChain,
    GroupedEpilogueBlock,
    Mxfp4Chain,
    W4A4Chain,
)
from .epilogue_fp8bs import IN_FMT_BS, Fp8BsChain
from .epilogue_mx import MX_OUT_FMTS, MxChain
from .kernel import CHAINS, GROUP_ALIGN, GroupedBlockGemm
from .operands import PACKINGS
from .scheduler import MAX_E
from .swap import SwappedGroupedGemm, is_swapped_record, swapped_knobs  # noqa: F401

MODULE = "batch_invariant_moe"
GRID_PLACEHOLDER = 1  # the compile-time value of the grid argument: a dynamic Int32 of the kernel, never compiled in
RASTER_PLACEHOLDER = 1  # the grouped body's raster-order argument at compile time: a dynamic Int32 as well
RASTERS = ("m", "n")


@functools.lru_cache(maxsize=None)
def object_record(cfg: Knobs) -> Knobs:
    """The record a compiled grouped object stands for: the knobs with the launch arguments (the raster order, the grid)
    at their defaults, so records differing only in them share one object."""
    return dataclasses.replace(cfg, raster_along="m", grid_clusters=0)


def raster_arg(cfg: Knobs) -> int:
    """The grouped body's raster-order launch argument of a record (1: m-blocks fastest inside a group); a value outside the
    record space is a host error (the argument reaches the kernel unchecked)."""
    if cfg.raster_along not in RASTERS:
        raise ValueError(
            f"batch-invariant MoE: raster {cfg.raster_along!r} outside {RASTERS}"
        )
    return 1 if cfg.raster_along == "m" else 0


#: The output dtypes by format: the dense GEMM's, plus the e4m3 output stored as bytes (the chain's paired conversion fills
#: byte registers), the block-scaled requantisations' e4m3 codes (``mxfp8`` / ``e4m3bs``: the block scales are their
#: second output) and the static NVFP4 class's packed codes; the output alignment of every format is the dense GEMM's (16-byte rows), the
#: requantised outputs' a whole scale block.
OUT_DT_OF = dict(
    OUT_DT,
    e4m3=cutlass.Uint8,
    mxfp8=cutlass.Uint8,
    nvfp4=cutlass.Uint8,
    e4m3bs=cutlass.Uint8,
)
OUT_ALIGN_OF = dict(ALIGN, mxfp8=32, nvfp4=32, e4m3bs=16)
#: A scale-factor buffer is a whole number of 512-byte atoms.
SF_BUFFER_DIVISIBILITY = SF_ATOM_ROWS * SF_ATOM_KBLOCKS
#: The chain classes of the block-scaled formats (each names its instruction class through ``at``).
BLOCKSCALED_CHAINS = (W4A4Chain, MxChain, Mxfp4Chain)
#: The output formats the block-scaled bodies' chains declare: the W4A4 chains' and the MX chains'.
BLOCK_SCALED_OUT_FMTS = tuple(dict.fromkeys(OUT_FMTS_W4A4 + MX_OUT_FMTS))

ChainLike = Union[str, GroupedEpilogueBlock]


def as_chain(chain: ChainLike, out_fmt: str = "bf16") -> GroupedEpilogueBlock:
    """The chain object of a bf16 legacy name (``plain`` / ``swiglu`` / ``relu2`` with ``out_fmt``) or the chain itself."""
    if isinstance(chain, str):
        assert chain in CHAINS and (chain != "swiglu" or out_fmt == "bf16"), (
            chain,
            out_fmt,
        )
        return GroupedChain.legacy(chain, out_fmt)
    assert isinstance(chain, GroupedEpilogueBlock), chain
    return chain


def _legacy_ok(chain: ChainLike, out_fmt: str) -> bool:
    return not isinstance(chain, str) or (chain in CHAINS and out_fmt in OUT_DT)


def fmt_tag(in_fmt: str) -> str:
    """The operand format's tag in the object names: the instruction class without the underscore before its step
    (``nvfp4_k128`` -> ``nvfp4k128``), the dense formats as they are."""
    return in_fmt.replace("_k", "k")


def _is_block_scaled(chain: GroupedEpilogueBlock) -> bool:
    return chain.in_fmt in BLOCK_SCALED


def _is_promoted(chain: GroupedEpilogueBlock) -> TypeGuard[Fp8BsChain]:
    """The block-scale fp8 chains (:class:`.epilogue_fp8bs.Fp8BsChain`): the promoted operand block."""
    return chain.in_fmt == IN_FMT_BS


def blockscaled_k_align(in_fmt: str) -> int:
    """The K alignment of a block-scaled format's operands: whole scale blocks of whole 16-byte rows (two NVFP4 blocks of 16
    per 16 bytes, one MX block of 32 e4m3 elements) and whole K granules of the dense class (the sm_100 mixed class's e2m1
    weight is unpacked by the TMA in 128-element runs: ``spec.k_granule``)."""
    return max(ALIGN[in_fmt], SF_VEC[in_fmt], k_granule(in_fmt))


def kernel_name(
    cfg: Knobs, chain: ChainLike, out_fmt: str = "bf16", packing: str = "dense"
) -> str:
    """The object's name: the operand format (the instruction class), the block-scaled bodies' packing of the A rows, the
    chain (activation, output format, parameters) and the record's compile-time identity (``Knobs.describe_object``: the
    raster order and the grid are launch arguments, so records differing only in them share one object)."""
    chain = as_chain(chain, out_fmt)
    mode = "_densea" if _is_block_scaled(chain) and packing == "dense" else ""
    return jit.sanitize_symbol_name(
        f"grouped_{fmt_tag(chain.in_fmt)}{mode}_{chain.describe()}_{object_record(cfg).describe_object()}"
    )


def static_body_ok(knobs: Knobs) -> bool:
    """The records whose epilogue fragment holds whole 16-output blocks of one row (the static class's fused quantization):
    the 128-row CTA tiles (thread = TMEM lane = row; the 1-CTA 128-row and the 2-CTA 256-row instructions)."""
    return knobs.tile_m == (256 if knobs.use_2cta_instrs else 128)


def _operands_ok(chain: GroupedEpilogueBlock, N: int, K: int, E: int) -> bool:
    """The operand rules shared by both bodies: 16-byte-aligned K and N (multiples of the format's alignment), the output
    rows aligned to the output format, whole ``2 x GROUP`` column blocks for the gated chain, at most ``MAX_E`` experts.
    The block-scaled formats: K in whole scale blocks of whole 16-byte rows (:func:`blockscaled_k_align`), N a multiple of
    8, the chain's output format one of the block-scaled chains' (the requantised outputs on whole scale blocks)."""
    if chain.out_fmt not in OUT_DT_OF:
        return False
    if chain.pair and N % (2 * GROUP):
        return False  # the gate/up pairing needs whole 2 x GROUP column blocks
    al = ALIGN[chain.in_fmt]
    n_out = N // 2 if chain.pair else N
    if _is_block_scaled(chain):
        return (
            chain.out_fmt in BLOCK_SCALED_OUT_FMTS
            and K % blockscaled_k_align(chain.in_fmt) == 0
            and N % 8 == 0
            and n_out % OUT_ALIGN_OF[chain.out_fmt] == 0
            and 1 <= E <= MAX_E
        )
    return (
        K % al == 0
        and N % al == 0
        and n_out % OUT_ALIGN_OF[chain.out_fmt] == 0
        and 1 <= E <= MAX_E
        and chain.shape_ok(N, K, E)
    )


def can_implement_grouped(
    knobs: Knobs,
    N: int,
    K: int,
    E: int,
    chain: ChainLike = "plain",
    out_fmt: str = "bf16",
) -> bool:
    """Acceptance of the grouped body: well-formed knobs for the chain's operand format at swizzle 1 (the group scheduler
    has no swizzle), no in-cluster split, no operand swap, no in-CTA chunk fold, the dense tiles' N (the narrow decode tiles
    have no staged-epilogue layout; the block-scaled plans' own tiles), no B-reuse tile, the static class's fused quantization
    on the 128-row CTA tiles, the operand rules -- plus the static TMEM thread-value check done at trace time."""
    if not _legacy_ok(chain, out_fmt):
        return False
    chain = as_chain(chain, out_fmt)
    if (
        not knobs.well_formed(chain.in_fmt)
        or knobs.swizzle_size != 1
        or knobs.cluster_k > 1
        or knobs.swap_ab
        or knobs.cta_chunks
        or knobs.gather_a
    ):
        return False
    if _is_block_scaled(chain):
        if knobs.b_reuse:
            return False
        if chain.static_quant and not static_body_ok(knobs):
            return False
    elif knobs.tile_n not in TILES_N:
        return False
    if not chain.record_ok(knobs):
        return False  # the chain's own record rule (the block-scale fp8 chains' promotion tiles)
    return _operands_ok(chain, N, K, E)


def _fakes(chain: GroupedEpilogueBlock, packing: str = "dense"):
    """The fake operand tensors of a compile: the symbolic shapes of A / B / out / counts and the chain's declared operands
    (``alpha (E,)``, ``sa (1,)``, ``si (1,)`` of the fp8 chains; the block-scaled chains' flat scale-factor buffers, ``pts`` at
    the padded row positions and the static class's ``out_sf`` / ``aux``)."""
    in_dt, al = IN_DT[chain.in_fmt], ALIGN[chain.in_fmt]
    out_dt, oal = OUT_DT_OF[chain.out_fmt], OUT_ALIGN_OF[chain.out_fmt]
    if chain.out_fmt == "nvfp4":
        oal = 16  # the packed codes: 16-byte rows of bytes
    sym_e = cute.sym_int(32)
    sym_no = cute.sym_int(32, divisibility=oal)
    c_fake = cute.runtime.make_fake_tensor(Int32, (sym_e,), (1,), assumed_align=4)
    f = dict(counts=c_fake, scales=())
    if _is_promoted(chain):
        # the block-scale fp8 operands: e4m3 elements, K in whole smem stages, N in whole scale blocks; the fp32 block scales
        # of the rows (any strides) and of the experts (K-contiguous); the fused requantization's fp32 scale output
        sym_r, sym_m = cute.sym_int(32), cute.sym_int(32)
        sym_n = cute.sym_int(32, divisibility=chain.block)
        sym_k = cute.sym_int(32, divisibility=BLOCK_K)
        sym_kb, sym_nb, sym_ob = cute.sym_int(32), cute.sym_int(32), cute.sym_int(32)

        def rows3(sym_rows):
            return cute.runtime.make_fake_tensor(
                in_dt,
                (sym_rows, sym_k, 1),
                (cute.sym_int64(divisibility=al), 1, cute.sym_int64(divisibility=al)),
                assumed_align=16,
            )

        f["a"], f["rows"] = rows3(sym_m), rows3(sym_r)
        f["b"] = cute.runtime.make_fake_tensor(
            in_dt,
            (sym_n, sym_k, sym_e),
            (cute.sym_int64(divisibility=al), 1, cute.sym_int64(divisibility=al)),
            assumed_align=16,
        )
        f["sfa"] = cute.runtime.make_fake_tensor(
            Float32,
            (sym_m, sym_kb),
            (cute.sym_int64(), cute.sym_int64()),
            assumed_align=4,
        )
        f["sfb"] = cute.runtime.make_fake_tensor(
            Float32,
            (sym_e, sym_nb, sym_kb),
            (cute.sym_int64(), cute.sym_int64(), 1),
            assumed_align=4,
        )
        f["out"] = cute.runtime.make_fake_tensor(
            out_dt,
            (sym_m, sym_no),
            (cute.sym_int64(divisibility=oal), 1),
            assumed_align=16,
        )
        f["out_sf"] = cute.runtime.make_fake_tensor(
            Float32, (sym_ob, cute.sym_int(32)), (cute.sym_int64(), 1), assumed_align=4
        )
        # the gather form's unpermuted rows and their scales (any strides), the plan's row map
        f["x"] = cute.runtime.make_fake_tensor(
            in_dt,
            (sym_r, sym_k),
            (cute.sym_int64(divisibility=al), 1),
            assumed_align=16,
        )
        f["sfa_u"] = cute.runtime.make_fake_tensor(
            Float32,
            (sym_r, sym_kb),
            (cute.sym_int64(), cute.sym_int64()),
            assumed_align=4,
        )
        f["src"] = cute.runtime.make_fake_tensor(
            Int32, (cute.sym_int(32),), (1,), assumed_align=16
        )
        return f
    if _is_block_scaled(chain):
        dense_a = packing == "dense"
        sym_m = (
            cute.sym_int(32) if dense_a else cute.sym_int(32, divisibility=GROUP_ALIGN)
        )
        sym_mp = (
            cute.sym_int(32, divisibility=GROUP_ALIGN) if dense_a else sym_m
        )  # the padded rows (pts, scale factors)
        sym_r = cute.sym_int(32)
        sym_n = cute.sym_int(32, divisibility=8)
        k_al = blockscaled_k_align(chain.in_fmt)

        def operand_fake(bits: int):
            """The element type, the symbolic K extent and the stride divisibility of an operand of ``bits`` per element: the
            byte-packed 4-bit operands (two elements per byte, K / 2 bytes per row, K in whole 16-byte rows of both operands),
            the 8-bit formats as given (K in whole scale blocks and K granules, 16-byte rows)."""
            if bits == 4:
                return (
                    cutlass.Uint8,
                    cute.sym_int(32, divisibility=max(16, k_al // 2)),
                    max(16, k_al // 2),
                )
            return in_dt, cute.sym_int(32, divisibility=k_al), al

        # the token operand (A: the rows) and the weight operand (B) at their own element widths: the one-format classes
        # share one width and one symbolic K (the compiled object's assumption, as before the mixed classes); the mixed
        # classes' tokens are e4m3 and their weights byte-packed e2m1, each with its own symbol
        a_dt, sym_kd, a_div = operand_fake(A_ELEM_BITS_OF[chain.in_fmt])
        if A_ELEM_BITS_OF[chain.in_fmt] == B_ELEM_BITS_OF[chain.in_fmt]:
            b_dt, sym_kd_b, b_div = a_dt, sym_kd, a_div
        else:
            b_dt, sym_kd_b, b_div = operand_fake(B_ELEM_BITS_OF[chain.in_fmt])

        def flat(div):
            return cute.runtime.make_fake_tensor(
                cutlass.Uint8,
                (cute.sym_int(32, divisibility=div),),
                (1,),
                assumed_align=16,
            )

        def packed3(sym_rows):
            return cute.runtime.make_fake_tensor(
                a_dt,
                (sym_rows, sym_kd, 1),
                (
                    cute.sym_int64(divisibility=a_div),
                    1,
                    cute.sym_int64(divisibility=a_div),
                ),
                assumed_align=16,
            )

        f["a"] = packed3(sym_m)
        f["rows"] = packed3(sym_r)
        f["b"] = cute.runtime.make_fake_tensor(
            b_dt,
            (sym_n, sym_kd_b, sym_e),
            (cute.sym_int64(divisibility=b_div), 1, cute.sym_int64(divisibility=b_div)),
            assumed_align=16,
        )
        f["sfa"], f["sfb"] = flat(SF_BUFFER_DIVISIBILITY), flat(SF_BUFFER_DIVISIBILITY)
        f["out"] = cute.runtime.make_fake_tensor(
            out_dt,
            (sym_m, sym_no),
            (cute.sym_int64(divisibility=oal), 1),
            assumed_align=16,
        )
        f["alpha"] = cute.runtime.make_fake_tensor(
            Float32, (sym_e,), (1,), assumed_align=4
        )
        f["pts"] = cute.runtime.make_fake_tensor(
            Float32, (sym_mp,), (1,), assumed_align=4
        )
        f["out_sf"] = flat(SF_BUFFER_DIVISIBILITY)
        f["aux"] = cute.runtime.make_fake_tensor(
            Float32, (sym_e,), (1,), assumed_align=4
        )
        # the gather form: the token rows' scale buffer (the linear bytes: whole 4-byte groups), their per-token scales (any
        # count: one entry is the unit scale), the plan's map
        f["sfa_lin"] = flat(4)
        f["pts_any"] = cute.runtime.make_fake_tensor(
            Float32, (cute.sym_int(32),), (1,), assumed_align=4
        )
        f["src"] = cute.runtime.make_fake_tensor(
            Int32, (cute.sym_int(32),), (1,), assumed_align=16
        )
        # the in-kernel plan's ids (slot order) and its dest output
        f["ids"] = cute.runtime.make_fake_tensor(
            Int32, (cute.sym_int(32),), (1,), assumed_align=16
        )
        f["dest"] = cute.runtime.make_fake_tensor(
            Int32, (cute.sym_int(32),), (1,), assumed_align=16
        )
        return f
    sym_r, sym_m, sym_n, sym_k = (
        cute.sym_int(32),
        cute.sym_int(32),
        cute.sym_int(32, divisibility=al),
        cute.sym_int(32, divisibility=al),
    )

    def strides3():  # fresh stride symbols per operand (shared symbols would tie the operands' strides together)
        return (cute.sym_int64(divisibility=al), 1, cute.sym_int64(divisibility=al))

    f["a"] = cute.runtime.make_fake_tensor(
        in_dt, (sym_m, sym_k, 1), strides3(), assumed_align=16
    )
    f["rows"] = cute.runtime.make_fake_tensor(
        in_dt, (sym_r, sym_k, 1), strides3(), assumed_align=16
    )
    f["b"] = cute.runtime.make_fake_tensor(
        in_dt, (sym_n, sym_k, sym_e), strides3(), assumed_align=16
    )
    f["out"] = cute.runtime.make_fake_tensor(
        out_dt, (sym_m, sym_no), (cute.sym_int64(divisibility=oal), 1), assumed_align=16
    )
    if chain.scale_operands:
        f["scales"] = tuple(
            cute.runtime.make_fake_tensor(cutlass.Float32, shape, (1,), assumed_align=4)
            for shape in ((sym_e,), (1,), (1,))
        )
    f["x"] = cute.runtime.make_fake_tensor(
        in_dt, (sym_r, sym_k), (cute.sym_int64(divisibility=al), 1), assumed_align=16
    )
    f["src"] = cute.runtime.make_fake_tensor(
        Int32, (cute.sym_int(32),), (1,), assumed_align=16
    )
    return f


def compile_grouped_fn(
    cfg: Knobs, chain: GroupedEpilogueBlock, packing: str = "dense"
) -> Callable[[], Any]:
    """The zero-argument compile closure of one grouped specialisation."""

    def compile_fn(**compile_options):
        f = _fakes(chain, packing)
        stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        # dynamic arguments of the kernel: the launch passes the device's grid and the record's raster order
        grid, raster = GRID_PLACEHOLDER, RASTER_PLACEHOLDER
        body = GroupedBlockGemm(cfg, chain, packing=packing)
        if _is_block_scaled(chain) or _is_promoted(chain):
            if isinstance(chain, (MxChain, Mxfp4Chain)) or _is_promoted(chain):
                # the MX chains (MXFP8, MXFP4) and the block-scale fp8 chains: no operand beyond the output (the quantising chain
                # writes its scales as a second output); the W4A4 unscaled oracle keeps ``block_scaled`` (alpha / pts unread)
                if chain.quant:
                    return cute.compile(
                        body.block_scaled_quant,
                        f["a"],
                        f["b"],
                        f["sfa"],
                        f["sfb"],
                        f["out"],
                        f["out_sf"],
                        f["counts"],
                        Int32(GROUP_ALIGN),
                        grid,
                        raster,
                        stream_fake,
                        options="--enable-tvm-ffi",
                        **compile_options,
                    )
                return cute.compile(
                    body.block_scaled_plain,
                    f["a"],
                    f["b"],
                    f["sfa"],
                    f["sfb"],
                    f["out"],
                    f["counts"],
                    Int32(GROUP_ALIGN),
                    grid,
                    raster,
                    stream_fake,
                    options="--enable-tvm-ffi",
                    **compile_options,
                )
            if chain.needs_aux:
                return cute.compile(
                    body.block_scaled_static,
                    f["a"],
                    f["b"],
                    f["sfa"],
                    f["sfb"],
                    f["out"],
                    f["out_sf"],
                    f["counts"],
                    f["alpha"],
                    f["pts"],
                    f["aux"],
                    grid,
                    raster,
                    stream_fake,
                    options="--enable-tvm-ffi",
                    **compile_options,
                )
            return cute.compile(
                body.block_scaled,
                f["a"],
                f["b"],
                f["sfa"],
                f["sfb"],
                f["out"],
                f["counts"],
                f["alpha"],
                f["pts"],
                grid,
                raster,
                stream_fake,
                options="--enable-tvm-ffi",
                **compile_options,
            )
        entry = body.with_scales if chain.scale_operands else body
        return cute.compile(
            entry,
            f["a"],
            f["b"],
            f["out"],
            f["counts"],
            *f["scales"],
            grid,
            raster,
            stream_fake,
            options="--enable-tvm-ffi",
            **compile_options,
        )

    return compile_fn


def get_compiled_grouped(
    cfg: Knobs,
    chain: ChainLike = "plain",
    out_fmt: str = "bf16",
    packing: str = "dense",
) -> Any:
    """``kernel(a3, b3, out2, counts, grid, raster)``, or ``kernel(a3, b3, out2, counts, alpha, sa, si, grid, raster)`` for a chain with
    scale operands: A ``(Tk, K, 1)`` K-major, B ``(N, K, E)`` K-major (the chain's operand format), out ``(Tk, Nout)``
    row-major (``Nout = N``, or ``N / 2`` for the gated chains; uint8 bytes for the e4m3 output), counts ``(E,)`` int32,
    ``alpha (E,)`` the per-expert weight scales, ``sa (1,)`` the A operand's scale, ``si (1,)`` the intermediate multiplier,
    ``grid`` the persistent grid in clusters (the device's co-resident cluster count for the record's cluster size).
    The block-scaled chains: ``kernel(a3, b3, sfa, sfb, out2, counts, alpha, pts, grid)`` (``kernel(a3, b3, sfa, sfb, out2,
    out_sf, counts, alpha, pts, aux, grid)`` for the static class) -- A ``(rows, K/2, 1)`` packed fp4 as uint8 (``packing``
    ``"padded"``: ``rows`` a multiple of 128; ``"dense"``: any row count, the scale factors and ``pts`` at the padded
    positions), B ``(N, K/2, E)``, SFA / SFB flat uint8 128x4 scale-factor buffers (SFB: E consecutive buffers), ``alpha
    (E,)`` the experts' alphas, ``pts (rows_sf,)`` the per-row scales. The MX chains: ``kernel(a3, b3, sfa, sfb, out2,
    counts, rows_sf, grid)`` (``kernel(a3, b3, sfa, sfb, out2, out_sf, counts, rows_sf, grid)`` for the fused
    requantization) -- A ``(rows, K, 1)`` e4m3, ``rows_sf`` the rows of its scale-factor buffer.
    ``chain``: a :class:`GroupedEpilogueBlock` or one of the bf16 class's legacy names (``plain`` / ``swiglu`` /
    ``relu2``)."""
    chain = as_chain(chain, out_fmt)
    assert packing in PACKINGS and (packing == "dense" or _is_block_scaled(chain)), (
        packing,
        chain,
    )
    assert not isinstance(chain, Fp8BsChain) or packing == "dense", chain
    # one loaded kernel per object record: the raster twins share it (``raster`` is a launch argument)
    return _compiled_grouped(object_record(cfg), chain, packing)


@functools.cache
def _compiled_grouped(cfg: Knobs, chain: GroupedEpilogueBlock, packing: str) -> Any:
    return jit.build_and_load(
        MODULE,
        kernel_name(cfg, chain, packing=packing),
        compile_grouped_fn(cfg, chain, packing),
    )


# ----------------------------------------------------------------------------- the swapped body (the swapped records)


def _swap_epi_wgs(chain: GroupedEpilogueBlock, token_tile: int) -> int:
    from .swap import epi_warpgroups

    return epi_warpgroups(token_tile, chain)


def swapped_kernel_name(
    cfg: Knobs,
    chain: ChainLike,
    out_fmt: str = "bf16",
    gather: bool = False,
    k_block: int = 0,
    sf_src: str = "128x4",
    plan_in_kernel: bool = False,
) -> str:
    """``swapped_<format>_<chain>_<record>[_gather][_sflin][_kb<K>][_e<W>][_ikp]``: the swapped record (its weight and token tiles)
    names the object beside the chain; the bf16 gather form (the token box gathered from the unpermuted rows) appends ``_gather``
    (the block-scaled gather record carries its own suffix), the linear scale source ``_sflin``; a block-scaled body's stage of
    ``k_block`` elements of K (the MXFP8 classes' two-atom stage) appends ``_kb<k_block>``; an epilogue over several warpgroups
    (the chains on the shared swapped epilogue at the wider token tiles) ``_e<W>``; the in-kernel plan ``_ikp``. The scale
    source is the gather form's: the box form reads the 128x4 buffer alone, so ``sf_src="linear"`` with a box record is
    refused (two objects under one name otherwise)."""
    chain = as_chain(chain, out_fmt)
    if sf_src == "linear" and not (gather or cfg.gather_a):
        raise ValueError(
            "sf_src='linear' is the gather form's scale source (gather=True or a record with gather_a); the box form "
            "reads the 128x4 scale buffer"
        )
    knobs = dataclasses.replace(cfg, grid_clusters=0)
    return jit.sanitize_symbol_name(
        f"swapped_{fmt_tag(chain.in_fmt)}_{chain.describe()}_{knobs.describe()}"
        + ("_gather" if gather else "")
        + ("_sflin" if sf_src == "linear" else "")
        + (f"_kb{k_block}" if k_block else "")
        + ("_ikp" if plan_in_kernel else "")
        + (
            f"_e{_swap_epi_wgs(chain, knobs.tile_n)}"
            if _swap_epi_wgs(chain, knobs.tile_n) > 1
            else ""
        )
    )


def can_implement_swapped(
    knobs: Knobs,
    N: int,
    K: int,
    E: int,
    chain: ChainLike = "plain",
    out_fmt: str = "bf16",
) -> bool:
    """Acceptance of the swapped body (:mod:`.swap`): a swapped record of the chain's format (a 128- or 64-row weight tile x
    the format's narrow token tiles, 1-CTA; 128 rows for the block-scaled formats) under the grouped body's operand rules;
    every chain of the classes (the body reads the dense packing)."""
    if not _legacy_ok(chain, out_fmt):
        return False
    chain = as_chain(chain, out_fmt)
    if not is_swapped_record(knobs, chain.in_fmt):
        return False
    if _is_block_scaled(chain) and not knobs.well_formed(chain.in_fmt):
        return False
    if not chain.swapped_record_ok(knobs):
        return False
    if knobs.gather_a and not _is_block_scaled(chain):
        return False  # the gather record is the block-scaled classes' (the bf16 gather form is the API's choice)
    return _operands_ok(chain, N, K, E)


def can_implement(
    knobs: Knobs,
    N: int,
    K: int,
    E: int,
    chain: ChainLike = "plain",
    out_fmt: str = "bf16",
) -> bool:
    """A record either body takes: the grouped body (a dense record) or the swapped body (a swapped record); the two record
    spaces are disjoint (the swapped flag)."""
    return can_implement_grouped(
        knobs, N, K, E, chain, out_fmt
    ) or can_implement_swapped(knobs, N, K, E, chain, out_fmt)


def compile_swapped_fn(
    cfg: Knobs,
    chain: GroupedEpilogueBlock,
    gather: bool = False,
    k_block: int = 0,
    sf_src: str = "128x4",
    plan_in_kernel: bool = False,
) -> Callable[[], Any]:
    """The zero-argument compile closure of one swapped specialisation: ``kernel(rows3, w3, out2, counts, [alpha, sa, si,]
    grid)`` -- ``rows (R, K, 1)`` the dense permuted rows (``R >= Tk``), ``w (N, K, E)``, ``out (Tk, Nout)``; the gather form
    (the bf16 classes) ``kernel(x2, src, k_div, w3, out2, counts, grid)`` -- ``x (T, K)`` the unpermuted rows, ``src (>= Tk,)``
    int32 the plan's slot of every expanded row; the block-scaled chains ``kernel(rows3, sfa, w3, sfw, out2, counts, alpha,
    pts, grid)`` (``kernel(rows3, sfa, w3, sfw, out2, out_sf, counts, alpha, pts, aux, grid)`` for the static class); the MX
    chains ``kernel(rows3, sfa, w3, sfw, out2, [out_sf,] counts, grid)`` with the stage ``k_block`` elements of K wide when
    given."""

    def compile_fn(**compile_options):
        f = _fakes(chain)
        stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        grid = GRID_PLACEHOLDER
        body = SwappedGroupedGemm(
            cfg,
            chain,
            gather=gather,
            sf_src=sf_src,
            k_block=k_block,
            plan_in_kernel=plan_in_kernel,
        )
        if _is_block_scaled(chain) and cfg.gather_a:
            sfx = f["sfa_lin"] if sf_src == "linear" else f["sfa"]
            if isinstance(chain, (MxChain, Mxfp4Chain)):
                # the MX chains' gather entries: the fused requantization (the MXFP8 classes; with the in-kernel plan the
                # decode form) or the plain chain
                if chain.quant and plan_in_kernel:
                    return cute.compile(
                        body.block_scaled_quant_gathered_plan,
                        f["rows"],
                        sfx,
                        f["ids"],
                        Int32(1),
                        Int32(1),
                        f["b"],
                        f["sfb"],
                        f["out"],
                        f["out_sf"],
                        f["counts"],
                        f["dest"],
                        grid,
                        stream_fake,
                        options="--enable-tvm-ffi",
                        **compile_options,
                    )
                if chain.quant:
                    return cute.compile(
                        body.block_scaled_quant_gathered,
                        f["rows"],
                        sfx,
                        f["src"],
                        Int32(1),
                        f["b"],
                        f["sfb"],
                        f["out"],
                        f["out_sf"],
                        f["counts"],
                        grid,
                        stream_fake,
                        options="--enable-tvm-ffi",
                        **compile_options,
                    )
                return cute.compile(
                    body.block_scaled_plain_gathered,
                    f["rows"],
                    sfx,
                    f["src"],
                    Int32(1),
                    f["b"],
                    f["sfb"],
                    f["out"],
                    f["counts"],
                    grid,
                    stream_fake,
                    options="--enable-tvm-ffi",
                    **compile_options,
                )
            if chain.needs_aux:
                return cute.compile(
                    body.block_scaled_static_gathered,
                    f["rows"],
                    sfx,
                    f["src"],
                    Int32(1),
                    f["b"],
                    f["sfb"],
                    f["out"],
                    f["out_sf"],
                    f["counts"],
                    f["alpha"],
                    f["pts_any"],
                    f["aux"],
                    grid,
                    stream_fake,
                    options="--enable-tvm-ffi",
                    **compile_options,
                )
            return cute.compile(
                body.block_scaled_gathered,
                f["rows"],
                sfx,
                f["src"],
                Int32(1),
                f["b"],
                f["sfb"],
                f["out"],
                f["counts"],
                f["alpha"],
                f["pts_any"],
                grid,
                stream_fake,
                options="--enable-tvm-ffi",
                **compile_options,
            )
        if _is_promoted(chain) and gather:
            # the block-scale fp8 gather form: the unpermuted rows and scales through the plan's map (the bf16 / f32 chains)
            return cute.compile(
                body.promoted_gathered,
                f["x"],
                f["sfa_u"],
                f["src"],
                Int32(1),
                f["b"],
                f["sfb"],
                f["out"],
                f["counts"],
                grid,
                stream_fake,
                options="--enable-tvm-ffi",
                **compile_options,
            )
        if _is_block_scaled(chain) or _is_promoted(chain):
            if isinstance(chain, (MxChain, Mxfp4Chain)) or _is_promoted(chain):
                # the MX and the block-scale fp8 chains' entries; the W4A4 unscaled oracle keeps ``block_scaled`` (alpha / pts unread)
                if chain.quant:
                    return cute.compile(
                        body.block_scaled_quant,
                        f["rows"],
                        f["sfa"],
                        f["b"],
                        f["sfb"],
                        f["out"],
                        f["out_sf"],
                        f["counts"],
                        grid,
                        stream_fake,
                        options="--enable-tvm-ffi",
                        **compile_options,
                    )
                return cute.compile(
                    body.block_scaled_plain,
                    f["rows"],
                    f["sfa"],
                    f["b"],
                    f["sfb"],
                    f["out"],
                    f["counts"],
                    grid,
                    stream_fake,
                    options="--enable-tvm-ffi",
                    **compile_options,
                )
            if chain.needs_aux:
                return cute.compile(
                    body.block_scaled_static,
                    f["rows"],
                    f["sfa"],
                    f["b"],
                    f["sfb"],
                    f["out"],
                    f["out_sf"],
                    f["counts"],
                    f["alpha"],
                    f["pts"],
                    f["aux"],
                    grid,
                    stream_fake,
                    options="--enable-tvm-ffi",
                    **compile_options,
                )
            return cute.compile(
                body.block_scaled,
                f["rows"],
                f["sfa"],
                f["b"],
                f["sfb"],
                f["out"],
                f["counts"],
                f["alpha"],
                f["pts"],
                grid,
                stream_fake,
                options="--enable-tvm-ffi",
                **compile_options,
            )
        if gather:
            return cute.compile(
                body.gathered,
                f["x"],
                f["src"],
                Int32(1),
                f["b"],
                f["out"],
                f["counts"],
                grid,
                stream_fake,
                options="--enable-tvm-ffi",
                **compile_options,
            )
        entry = body.with_scales if chain.scale_operands else body
        return cute.compile(
            entry,
            f["rows"],
            f["b"],
            f["out"],
            f["counts"],
            *f["scales"],
            grid,
            stream_fake,
            options="--enable-tvm-ffi",
            **compile_options,
        )

    return compile_fn


@functools.cache
def get_compiled_swapped(
    cfg: Knobs,
    chain: ChainLike = "plain",
    out_fmt: str = "bf16",
    gather: bool = False,
    k_block: int = 0,
    sf_src: str = "128x4",
    plan_in_kernel: bool = False,
) -> Any:
    """``kernel(rows3, w3, out2, counts, grid)`` (``kernel(rows3, w3, out2, counts, alpha, sa, si, grid)`` for a chain with
    scale operands; ``kernel(rows3, sfa, w3, sfw, out2, [out_sf,] counts, alpha, pts, [aux,] grid)`` for the block-scaled
    chains with the W4A4 scales; ``kernel(rows3, sfa, w3, sfw, out2, [out_sf,] counts, grid)`` for the MX chains): the
    swapped body for a swapped record of the chain's format (module ``batch_invariant_moe``): rows ``(R, K, 1)`` K-major
    (``R >= Tk``), w ``(N, K, E)`` K-major, out ``(Tk, Nout)``, counts ``(E,)``, ``grid`` the persistent grid in CTAs; the
    gather form ``kernel(x2, src, k_div, w3, out2, counts, grid)`` (the bf16 classes); the block-scaled gather record
    (``cfg.gather_a``): ``kernel(x3, sfx, src, k_div, w3, sfw, out2, [out_sf,] counts, [alpha, pts, [aux,]] grid)`` -- ``x (T,
    K/2, 1)`` the unpermuted packed token rows, ``sfx`` their flat scale buffer (``sf_src``), ``pts (T,)`` or ``(1,)``.
    ``k_block``: a block-scaled body's K elements per stage (0: the format's atom), one object per value."""
    chain = as_chain(chain, out_fmt)
    assert is_swapped_record(cfg, chain.in_fmt), (cfg, chain.in_fmt)
    assert sf_src in ("128x4", "linear"), sf_src
    name = swapped_kernel_name(
        cfg,
        chain,
        gather=gather,
        k_block=k_block,
        sf_src=sf_src,
        plan_in_kernel=plan_in_kernel,
    )  # refuses the box form with the linear source
    return jit.build_and_load(
        MODULE,
        name,
        compile_swapped_fn(cfg, chain, gather, k_block, sf_src, plan_in_kernel),
    )


# ----------------------------------------------------------------------------- the block-scaled classes' entries


def can_implement_grouped_blockscaled(
    knobs: Knobs, N: int, K: int, E: int, chain: GroupedEpilogueBlock, in_fmt: str
) -> bool:
    """:func:`can_implement_grouped` at the chain's instruction class ``in_fmt`` (the block-scaled classes name the class per
    expert weight shape, or per part)."""
    return isinstance(chain, BLOCKSCALED_CHAINS) and can_implement_grouped(
        knobs, N, K, E, chain.at(in_fmt)
    )


def can_implement_swapped_blockscaled(
    knobs: Knobs, N: int, K: int, E: int, chain: GroupedEpilogueBlock, in_fmt: str
) -> bool:
    """:func:`can_implement_swapped` at the chain's instruction class."""
    return isinstance(chain, BLOCKSCALED_CHAINS) and can_implement_swapped(
        knobs, N, K, E, chain.at(in_fmt)
    )
