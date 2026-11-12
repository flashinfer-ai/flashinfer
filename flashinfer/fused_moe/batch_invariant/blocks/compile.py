# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Compile wrapper for the grouped GEMM bodies on the dense blocks, both operand formats: one TVM-FFI object per (knobs,
chain) with dynamic Tk, N, K and E, the chain naming its operand format (``bf16`` for the bf16 classes' :class:`GroupedChain`,
``e4m3`` for the fp8 per-tensor classes' :class:`Fp8Chain`), the output format and the parameters; cache module
``batch_invariant_moe`` in FlashInfer's CuTe-DSL disk cache (this file, the routing, permutation and combine kernels are its
roots; the bodies of both formats are one source closure, so an edit to either moves every object's key). The persistent
grid is a dynamic ``Int32`` argument of the compiled kernel (``GRID_PLACEHOLDER`` at compile time; the launch passes the
device's co-resident cluster count, ``flashinfer.gemm.batch_invariant.api.persistent_clusters``), so one object serves every
grid. A chain with scale operands compiles the bodies' ``with_scales`` entry (``kernel(a3, b3, out2, counts, alpha, sa, si,
grid)``), the others the plain one (``kernel(a3, b3, out2, counts, grid)``).
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Any, Callable, Union

import cutlass
import cutlass.cute as cute
from cutlass import Int32

from ....batch_invariant import jit
from ....gemm.batch_invariant.blocks.operands import ALIGN, IN_DT, OUT_DT
from ....gemm.batch_invariant.generator import TILES_N, Knobs
from .epilogue import GROUP, GroupedChain, GroupedEpilogueBlock
from .kernel import CHAINS, GroupedBlockGemm
from .scheduler import MAX_E
from .swap import SwappedGroupedGemm, is_swapped_record, swapped_knobs  # noqa: F401

MODULE = "batch_invariant_moe"
GRID_PLACEHOLDER = 1  # the compile-time value of the grid argument: a dynamic Int32 of the kernel, never compiled in
#: The output dtypes by format: the dense GEMM's, plus the e4m3 output stored as bytes (the chain's paired conversion fills
#: byte registers); the output alignment of every format is the dense GEMM's (16-byte rows).
OUT_DT_OF = dict(OUT_DT, e4m3=cutlass.Uint8)
OUT_ALIGN_OF = ALIGN

ChainLike = Union[str, GroupedEpilogueBlock]


def as_chain(chain: ChainLike, out_fmt: str = "bf16") -> GroupedEpilogueBlock:
    """The chain object of a bf16 legacy name (``plain`` / ``swiglu`` / ``relu2`` with ``out_fmt``) or the chain itself."""
    if isinstance(chain, str):
        assert chain in CHAINS and (chain == "plain" or out_fmt == "bf16"), (
            chain,
            out_fmt,
        )
        return GroupedChain.legacy(chain, out_fmt)
    assert isinstance(chain, GroupedEpilogueBlock), chain
    return chain


def _legacy_ok(chain: ChainLike, out_fmt: str) -> bool:
    return not isinstance(chain, str) or (chain in CHAINS and out_fmt in OUT_DT)


def kernel_name(cfg: Knobs, chain: ChainLike, out_fmt: str = "bf16") -> str:
    """The object's name: the operand format, the chain (activation, output format, parameters) and the record without its
    ``grid_clusters`` (the grid is a launch argument: records differing only in it share one object)."""
    chain = as_chain(chain, out_fmt)
    knobs = dataclasses.replace(cfg, grid_clusters=0)
    return jit.sanitize_symbol_name(
        f"grouped_{chain.in_fmt}_{chain.describe()}_{knobs.describe()}"
    )


def _operands_ok(chain: GroupedEpilogueBlock, N: int, K: int, E: int) -> bool:
    """The operand rules shared by both bodies: 16-byte-aligned K and N (multiples of the format's alignment), the output
    rows aligned to the output format, whole ``2 x GROUP`` column blocks for the gated chain, at most ``MAX_E`` experts."""
    if chain.out_fmt not in OUT_DT_OF:
        return False
    if chain.pair and N % (2 * GROUP):
        return False  # the gate/up pairing needs whole 2 x GROUP column blocks
    al = ALIGN[chain.in_fmt]
    n_out = N // 2 if chain.pair else N
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
    """Acceptance of the grouped body on the dense blocks: well-formed knobs for the chain's operand format at swizzle 1
    (the group scheduler has no swizzle), no in-cluster split, no in-CTA chunk fold, the dense tiles' N (the narrow decode
    tiles have no staged-epilogue layout), the operand rules -- plus the static TMEM thread-value check done at trace
    time."""
    if not _legacy_ok(chain, out_fmt):
        return False
    chain = as_chain(chain, out_fmt)
    if (
        not knobs.well_formed(chain.in_fmt)
        or knobs.swizzle_size != 1
        or knobs.cluster_k > 1
        or knobs.swap_ab
        or knobs.cta_chunks
    ):
        return False
    if knobs.tile_n not in TILES_N:
        return False
    return _operands_ok(chain, N, K, E)


def _fakes(chain: GroupedEpilogueBlock):
    """The fake operand tensors of a compile: the symbolic shapes of A / B / out / counts and, for a chain with scale
    operands, of ``alpha (E,)``, ``sa (1,)`` and ``si (1,)``."""
    in_dt, al = IN_DT[chain.in_fmt], ALIGN[chain.in_fmt]
    out_dt, oal = OUT_DT_OF[chain.out_fmt], OUT_ALIGN_OF[chain.out_fmt]
    sym_r, sym_m, sym_n, sym_k, sym_e = (
        cute.sym_int(32),
        cute.sym_int(32),
        cute.sym_int(32, divisibility=al),
        cute.sym_int(32, divisibility=al),
        cute.sym_int(32),
    )
    sym_no = cute.sym_int(32, divisibility=oal)

    def strides3():  # fresh stride symbols per operand (shared symbols would tie the operands' strides together)
        return (cute.sym_int64(divisibility=al), 1, cute.sym_int64(divisibility=al))

    a_fake = cute.runtime.make_fake_tensor(
        in_dt, (sym_m, sym_k, 1), strides3(), assumed_align=16
    )
    rows_fake = cute.runtime.make_fake_tensor(
        in_dt, (sym_r, sym_k, 1), strides3(), assumed_align=16
    )
    b_fake = cute.runtime.make_fake_tensor(
        in_dt, (sym_n, sym_k, sym_e), strides3(), assumed_align=16
    )
    o_fake = cute.runtime.make_fake_tensor(
        out_dt, (sym_m, sym_no), (cute.sym_int64(divisibility=oal), 1), assumed_align=16
    )
    c_fake = cute.runtime.make_fake_tensor(Int32, (sym_e,), (1,), assumed_align=4)
    scales = ()
    if chain.scale_operands:
        scales = tuple(
            cute.runtime.make_fake_tensor(cutlass.Float32, shape, (1,), assumed_align=4)
            for shape in ((sym_e,), (1,), (1,))
        )
    x_fake = cute.runtime.make_fake_tensor(
        in_dt, (sym_r, sym_k), (cute.sym_int64(divisibility=al), 1), assumed_align=16
    )
    src_fake = cute.runtime.make_fake_tensor(
        Int32, (cute.sym_int(32),), (1,), assumed_align=16
    )
    return dict(
        a=a_fake,
        rows=rows_fake,
        b=b_fake,
        out=o_fake,
        counts=c_fake,
        scales=scales,
        x=x_fake,
        src=src_fake,
    )


def compile_grouped_fn(cfg: Knobs, chain: GroupedEpilogueBlock) -> Callable[[], Any]:
    """The zero-argument compile closure of one grouped specialisation on the dense blocks."""

    def compile_fn():
        f = _fakes(chain)
        stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        grid = GRID_PLACEHOLDER  # a dynamic argument of the kernel: the launch passes the device's value
        body = GroupedBlockGemm(cfg, chain)
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
    cfg: Knobs, chain: ChainLike = "plain", out_fmt: str = "bf16"
) -> Any:
    """``kernel(a3, b3, out2, counts, grid)``, or ``kernel(a3, b3, out2, counts, alpha, sa, si, grid)`` for a chain with
    scale operands: A ``(Tk, K, 1)`` K-major, B ``(N, K, E)`` K-major (the chain's operand format), out ``(Tk, Nout)``
    row-major (``Nout = N``, or ``N / 2`` for the gated chains; uint8 bytes for the e4m3 output), counts ``(E,)`` int32,
    ``alpha (E,)`` the per-expert weight scales, ``sa (1,)`` the A operand's scale, ``si (1,)`` the intermediate multiplier,
    ``grid`` the persistent grid in clusters (the device's co-resident cluster count for the record's cluster size).
    ``chain``: a :class:`GroupedEpilogueBlock` or one of the bf16 class's legacy names (``plain`` / ``swiglu`` /
    ``relu2``)."""
    chain = as_chain(chain, out_fmt)
    return jit.build_and_load(
        MODULE, kernel_name(cfg, chain), compile_grouped_fn(cfg, chain)
    )


# ----------------------------------------------------------------------------- the swapped body (the swapped records)


def swapped_kernel_name(
    cfg: Knobs,
    chain: ChainLike,
    out_fmt: str = "bf16",
    gather: bool = False,
) -> str:
    """``swapped_<format>_<chain>_<record>[_gather]``: the swapped record (its weight and token tiles) names the object beside
    the chain; the gather form (the token box gathered from the unpermuted rows) appends ``_gather``."""
    chain = as_chain(chain, out_fmt)
    knobs = dataclasses.replace(cfg, grid_clusters=0)
    return jit.sanitize_symbol_name(
        f"swapped_{chain.in_fmt}_{chain.describe()}_{knobs.describe()}"
        + ("_gather" if gather else "")
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
    the format's narrow token tiles, 1-CTA) under the grouped body's operand rules; every chain of the classes."""
    if not _legacy_ok(chain, out_fmt):
        return False
    chain = as_chain(chain, out_fmt)
    if not is_swapped_record(knobs, chain.in_fmt):
        return False
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
    cfg: Knobs, chain: GroupedEpilogueBlock, gather: bool = False
) -> Callable[[], Any]:
    """The zero-argument compile closure of one swapped specialisation: ``kernel(rows3, w3, out2, counts, [alpha, sa, si,]
    grid)`` -- ``rows (R, K, 1)`` the dense permuted rows (``R >= Tk``), ``w (N, K, E)``, ``out (Tk, Nout)``; the gather form
    (the bf16 classes) ``kernel(x2, src, k_div, w3, out2, counts, grid)`` -- ``x (T, K)`` the unpermuted rows, ``src (>= Tk,)``
    int32 the plan's slot of every expanded row (row ``p`` reads ``x[src[p] // k_div]``)."""

    def compile_fn():
        f = _fakes(chain)
        stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        grid = GRID_PLACEHOLDER
        body = SwappedGroupedGemm(cfg, chain, gather=gather)
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
) -> Any:
    """``kernel(rows3, w3, out2, counts, grid)`` (``kernel(rows3, w3, out2, counts, alpha, sa, si, grid)`` for a chain with
    scale operands): the swapped body for a swapped record of the chain's format (module ``batch_invariant_moe``): rows
    ``(R, K, 1)`` K-major (``R >= Tk``), w ``(N, K, E)`` K-major, out ``(Tk, Nout)``, counts ``(E,)``, ``grid`` the persistent
    grid in CTAs; the gather form ``kernel(x2, src, k_div, w3, out2, counts, grid)`` (the bf16 classes)."""
    chain = as_chain(chain, out_fmt)
    assert is_swapped_record(cfg, chain.in_fmt), (cfg, chain.in_fmt)
    return jit.build_and_load(
        MODULE,
        swapped_kernel_name(cfg, chain, gather=gather),
        compile_swapped_fn(cfg, chain, gather),
    )
