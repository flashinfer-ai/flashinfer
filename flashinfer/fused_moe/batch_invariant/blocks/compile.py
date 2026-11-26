# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Compile wrapper for the grouped GEMM bodies, every operand format: one TVM-FFI object per (knobs, chain[, packing]) with
dynamic Tk, N, K and E, the chain naming its operand format (``bf16`` for the bf16 classes' :class:`GroupedChain`, ``e4m3``
for the fp8 per-tensor classes' :class:`Fp8Chain`, the NVFP4 instruction classes for :class:`W4A4Chain`), the output format
and the parameters; cache module ``batch_invariant_moe`` in FlashInfer's CuTe-DSL disk cache (this file, the routing,
permutation, combine and scatter kernels are its roots; the bodies of every format are one source closure, so an edit to
any moves every object's key). The persistent grid is a dynamic ``Int32`` argument of the compiled kernel
(``GRID_PLACEHOLDER`` at compile time; the launch passes the device's co-resident cluster count,
``flashinfer.gemm.batch_invariant.api.persistent_clusters``), so one object serves every grid. The chain's declared operands
select the bodies' entry: the plain one (``kernel(a3, b3, out2, counts, grid)``), ``with_scales`` (``kernel(a3, b3, out2,
counts, alpha, sa, si, grid)``), ``block_scaled`` (``kernel(a3, b3, sfa, sfb, out2, counts, alpha, pts, grid)``) or
``block_scaled_static`` (``kernel(a3, b3, sfa, sfb, out2, out_sf, counts, alpha, pts, aux, grid)``).
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Any, Callable, Union

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32

from ....batch_invariant import jit
from ....gemm.batch_invariant.blocks.operands import ALIGN, IN_DT, OUT_DT
from ....gemm.batch_invariant.generator import TILES_N, Knobs
from ....gemm.batch_invariant.spec import BLOCK_SCALED, SF_ATOM_KBLOCKS, SF_ATOM_ROWS
from .epilogue import (
    GROUP,
    OUT_FMTS_W4A4,
    GroupedChain,
    GroupedEpilogueBlock,
    Mxfp4Chain,
    W4A4Chain,
)
from .kernel import CHAINS, GROUP_ALIGN, GroupedBlockGemm
from .operands import PACKINGS
from .scheduler import MAX_E
from .swap import SwappedGroupedGemm, is_swapped_record, swapped_knobs  # noqa: F401

MODULE = "batch_invariant_moe"
GRID_PLACEHOLDER = 1  # the compile-time value of the grid argument: a dynamic Int32 of the kernel, never compiled in
#: The output dtypes by format: the dense GEMM's, plus the e4m3 output stored as bytes (the chain's paired conversion fills
#: byte registers) and the static NVFP4 class's packed codes; the output alignment of every format is the dense GEMM's
#: (16-byte rows), the packed codes' row a 16-byte row of codes and whole scale blocks.
OUT_DT_OF = dict(OUT_DT, e4m3=cutlass.Uint8, nvfp4=cutlass.Uint8)
OUT_ALIGN_OF = dict(ALIGN, nvfp4=32)
#: A scale-factor buffer is a whole number of 512-byte atoms.
SF_BUFFER_DIVISIBILITY = SF_ATOM_ROWS * SF_ATOM_KBLOCKS

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


def kernel_name(
    cfg: Knobs, chain: ChainLike, out_fmt: str = "bf16", packing: str = "dense"
) -> str:
    """The object's name: the operand format (the instruction class), the block-scaled bodies' packing of the A rows, the
    chain (activation, output format, parameters) and the record without its ``grid_clusters`` (the grid is a launch
    argument: records differing only in it share one object)."""
    chain = as_chain(chain, out_fmt)
    knobs = dataclasses.replace(cfg, grid_clusters=0)
    mode = "_densea" if _is_block_scaled(chain) and packing == "dense" else ""
    return jit.sanitize_symbol_name(
        f"grouped_{fmt_tag(chain.in_fmt)}{mode}_{chain.describe()}_{knobs.describe()}"
    )


def static_body_ok(knobs: Knobs) -> bool:
    """The records whose epilogue fragment holds whole 16-output blocks of one row (the static class's fused quantization):
    the 128-row CTA tiles (thread = TMEM lane = row; the 1-CTA 128-row and the 2-CTA 256-row instructions)."""
    return knobs.tile_m == (256 if knobs.use_2cta_instrs else 128)


def _operands_ok(chain: GroupedEpilogueBlock, N: int, K: int, E: int) -> bool:
    """The operand rules shared by both bodies: 16-byte-aligned K and N (multiples of the format's alignment), the output
    rows aligned to the output format, whole ``2 x GROUP`` column blocks for the gated chain, at most ``MAX_E`` experts.
    The block-scaled formats: a whole scale block per 16 bytes of K (the format's alignment), N a multiple of 8, the
    packed-code output of the static class on whole 16-byte rows of whole scale blocks."""
    if chain.out_fmt not in OUT_DT_OF:
        return False
    if chain.pair and N % (2 * GROUP):
        return False  # the gate/up pairing needs whole 2 x GROUP column blocks
    al = ALIGN[chain.in_fmt]
    n_out = N // 2 if chain.pair else N
    if _is_block_scaled(chain):
        return (
            chain.out_fmt in OUT_FMTS_W4A4
            and K % al == 0
            and N % 8 == 0
            and n_out % OUT_ALIGN_OF[chain.out_fmt] == 0
            and 1 <= E <= MAX_E
        )
    return (
        K % al == 0
        and N % al == 0
        and n_out % OUT_ALIGN_OF[chain.out_fmt] == 0
        and 1 <= E <= MAX_E
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
    if _is_block_scaled(chain):
        # the byte-packed 4-bit operands (two elements per byte, K / 2 bytes per row; the 8-bit element formats one per byte)
        dense_a = packing == "dense"
        sym_m = (
            cute.sym_int(32) if dense_a else cute.sym_int(32, divisibility=GROUP_ALIGN)
        )
        sym_mp = (
            cute.sym_int(32, divisibility=GROUP_ALIGN) if dense_a else sym_m
        )  # the padded rows (pts, scale factors)
        sym_r = cute.sym_int(32)
        sym_n, sym_kb = (
            cute.sym_int(32, divisibility=8),
            cute.sym_int(32, divisibility=16),
        )

        def flat(div):
            return cute.runtime.make_fake_tensor(
                cutlass.Uint8,
                (cute.sym_int(32, divisibility=div),),
                (1,),
                assumed_align=16,
            )

        def packed3(sym_rows):
            return cute.runtime.make_fake_tensor(
                cutlass.Uint8,
                (sym_rows, sym_kb, 1),
                (cute.sym_int64(divisibility=16), 1, cute.sym_int64(divisibility=16)),
                assumed_align=16,
            )

        f["a"] = packed3(sym_m)
        f["rows"] = packed3(sym_r)
        f["b"] = cute.runtime.make_fake_tensor(
            cutlass.Uint8,
            (sym_n, sym_kb, sym_e),
            (cute.sym_int64(divisibility=16), 1, cute.sym_int64(divisibility=16)),
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

    def compile_fn():
        f = _fakes(chain, packing)
        stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        grid = GRID_PLACEHOLDER  # a dynamic argument of the kernel: the launch passes the device's value
        body = GroupedBlockGemm(cfg, chain, packing=packing)
        if _is_block_scaled(chain):
            # the plain entry (no operand beyond the output) is the MXFP4 chains'; the W4A4 unscaled oracle keeps the scaled
            # entry and leaves its alpha / pts unread
            if isinstance(chain, Mxfp4Chain):
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
                    stream_fake,
                    options="--enable-tvm-ffi",
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
                    stream_fake,
                    options="--enable-tvm-ffi",
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
                stream_fake,
                options="--enable-tvm-ffi",
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
            stream_fake,
            options="--enable-tvm-ffi",
        )

    return compile_fn


@functools.cache
def get_compiled_grouped(
    cfg: Knobs,
    chain: ChainLike = "plain",
    out_fmt: str = "bf16",
    packing: str = "dense",
) -> Any:
    """``kernel(a3, b3, out2, counts, grid)``, or ``kernel(a3, b3, out2, counts, alpha, sa, si, grid)`` for a chain with
    scale operands: A ``(Tk, K, 1)`` K-major, B ``(N, K, E)`` K-major (the chain's operand format), out ``(Tk, Nout)``
    row-major (``Nout = N``, or ``N / 2`` for the gated chains; uint8 bytes for the e4m3 output), counts ``(E,)`` int32,
    ``alpha (E,)`` the per-expert weight scales, ``sa (1,)`` the A operand's scale, ``si (1,)`` the intermediate multiplier,
    ``grid`` the persistent grid in clusters (the device's co-resident cluster count for the record's cluster size).
    The block-scaled chains: ``kernel(a3, b3, sfa, sfb, out2, counts, alpha, pts, grid)`` (``kernel(a3, b3, sfa, sfb, out2,
    out_sf, counts, alpha, pts, aux, grid)`` for the static class) -- A ``(rows, K/2, 1)`` packed fp4 as uint8 (``packing``
    ``"padded"``: ``rows`` a multiple of 128; ``"dense"``: any row count, the scale factors and ``pts`` at the padded
    positions), B ``(N, K/2, E)``, SFA / SFB flat uint8 128x4 scale-factor buffers (SFB: E consecutive buffers), ``alpha
    (E,)`` the experts' alphas, ``pts (rows_sf,)`` the per-row scales.
    ``chain``: a :class:`GroupedEpilogueBlock` or one of the bf16 class's legacy names (``plain`` / ``swiglu`` /
    ``relu2``)."""
    chain = as_chain(chain, out_fmt)
    assert packing in PACKINGS and (packing == "dense" or _is_block_scaled(chain)), (
        packing,
        chain,
    )
    return jit.build_and_load(
        MODULE,
        kernel_name(cfg, chain, packing=packing),
        compile_grouped_fn(cfg, chain, packing),
    )


# ----------------------------------------------------------------------------- the swapped body (the swapped records)


def _swap_epi_wgs(token_tile: int) -> int:
    from .swap import epi_warpgroups

    return epi_warpgroups(token_tile)


def swapped_kernel_name(
    cfg: Knobs,
    chain: ChainLike,
    out_fmt: str = "bf16",
    gather: bool = False,
    sf_src: str = "128x4",
) -> str:
    """``swapped_<format>_<chain>_<record>[_gather][_sflin]``: the swapped record (its weight and token tiles) names the object
    beside the chain; the bf16 gather form (the token box gathered from the unpermuted rows) appends ``_gather`` (the
    block-scaled gather record carries its own suffix), the linear scale source ``_sflin``. The scale source is the gather
    form's: the box form reads the 128x4 buffer alone, so ``sf_src="linear"`` with a box record is refused (two objects
    under one name otherwise)."""
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
        + (
            f"_e{_swap_epi_wgs(knobs.tile_n)}"
            if _swap_epi_wgs(knobs.tile_n) > 1
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
    cfg: Knobs, chain: GroupedEpilogueBlock, gather: bool = False, sf_src: str = "128x4"
) -> Callable[[], Any]:
    """The zero-argument compile closure of one swapped specialisation: ``kernel(rows3, w3, out2, counts, [alpha, sa, si,]
    grid)`` -- ``rows (R, K, 1)`` the dense permuted rows (``R >= Tk``), ``w (N, K, E)``, ``out (Tk, Nout)``; the gather form
    (the bf16 classes) ``kernel(x2, src, k_div, w3, out2, counts, grid)`` -- ``x (T, K)`` the unpermuted rows, ``src (>= Tk,)``
    int32 the plan's slot of every expanded row; the block-scaled chains ``kernel(rows3, sfa, w3, sfw, out2, counts, alpha,
    pts, grid)`` (``kernel(rows3, sfa, w3, sfw, out2, out_sf, counts, alpha, pts, aux, grid)`` for the static class)."""

    def compile_fn():
        f = _fakes(chain)
        stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        grid = GRID_PLACEHOLDER
        body = SwappedGroupedGemm(cfg, chain, gather=gather, sf_src=sf_src)
        if _is_block_scaled(chain) and cfg.gather_a:
            sfx = f["sfa_lin"] if sf_src == "linear" else f["sfa"]
            if isinstance(chain, Mxfp4Chain):
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
            )
        if _is_block_scaled(chain):
            # the plain entry (no operand beyond the output) is the MXFP4 chains'; the W4A4 unscaled oracle keeps the scaled
            # entry and leaves its alpha / pts unread
            if isinstance(chain, Mxfp4Chain):
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
        )

    return compile_fn


@functools.cache
def get_compiled_swapped(
    cfg: Knobs,
    chain: ChainLike = "plain",
    out_fmt: str = "bf16",
    gather: bool = False,
    sf_src: str = "128x4",
) -> Any:
    """``kernel(rows3, w3, out2, counts, grid)`` (``kernel(rows3, w3, out2, counts, alpha, sa, si, grid)`` for a chain with
    scale operands; ``kernel(rows3, sfa, w3, sfw, out2, [out_sf,] counts, alpha, pts, [aux,] grid)`` for the block-scaled
    chains): the swapped body for a swapped record of the chain's format (module ``batch_invariant_moe``): rows ``(R, K, 1)``
    K-major (``R >= Tk``), w ``(N, K, E)`` K-major, out ``(Tk, Nout)``, counts ``(E,)``, ``grid`` the persistent grid in CTAs;
    the gather form ``kernel(x2, src, k_div, w3, out2, counts, grid)`` (the bf16 classes); the block-scaled gather record
    (``cfg.gather_a``): ``kernel(x3, sfx, src, k_div, w3, sfw, out2, [out_sf,] counts, [alpha, pts, [aux,]] grid)`` -- ``x (T,
    K/2, 1)`` the unpermuted packed token rows, ``sfx`` their flat scale buffer (``sf_src``), ``pts (T,)`` or ``(1,)``."""
    chain = as_chain(chain, out_fmt)
    assert is_swapped_record(cfg, chain.in_fmt), (cfg, chain.in_fmt)
    assert sf_src in ("128x4", "linear"), sf_src
    name = swapped_kernel_name(
        cfg, chain, gather=gather, sf_src=sf_src
    )  # refuses the box form with the linear source
    return jit.build_and_load(
        MODULE, name, compile_swapped_fn(cfg, chain, gather, sf_src)
    )


# ----------------------------------------------------------------------------- the block-scaled classes' entries


BLOCKSCALED_CHAINS = (W4A4Chain, Mxfp4Chain)


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
