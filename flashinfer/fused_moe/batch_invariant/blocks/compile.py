# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Compile wrapper for the grouped GEMM on the dense blocks: one TVM-FFI object per (knobs, chain, output format) with
dynamic Tk, N, K and E; cache module ``batch_invariant_moe`` in FlashInfer's CuTe-DSL disk cache (this file, the
routing, permutation and combine kernels are its roots). The persistent grid is a dynamic ``Int32`` argument of the
compiled kernel (``GRID_PLACEHOLDER`` at compile time; the launch passes the device's co-resident cluster count,
``flashinfer.gemm.batch_invariant.api.persistent_clusters``), so one object serves every grid.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Any, Callable, Union

import cutlass.cute as cute
from cutlass import Int32

from ....batch_invariant import jit
from ....gemm.batch_invariant.blocks.operands import ALIGN, IN_DT, OUT_DT
from ....gemm.batch_invariant.generator import TILES_N, Knobs
from .epilogue import GROUP, GroupedChain
from .kernel import (  # noqa: F401  (A_PRODUCERS re-exported)
    A_PRODUCERS,
    CHAINS,
    GatherGroupedBlockGemm,
    GroupedBlockGemm,
)
from .scheduler import MAX_E

MODULE = "batch_invariant_moe"
GRID_PLACEHOLDER = 1  # the compile-time value of the grid argument: a dynamic Int32 of the kernel, never compiled in


def gather_suffix(gather: bool, a_producer: str = "gather4") -> str:
    """The gather variant's name suffix: ``_gather`` (the TMA gather4 producer) or ``_gather_ldgsts``."""
    assert a_producer in A_PRODUCERS, a_producer
    if not gather:
        return ""
    return "_gather" + ("" if a_producer == "gather4" else f"_{a_producer}")


def kernel_name(
    cfg: Knobs,
    chain: Union[str, GroupedChain],
    out_fmt: str = "bf16",
    gather: bool = False,
    a_producer: str = "gather4",
) -> str:
    """The object's name: the chain (activation, output format, parameters) and the record without its ``grid_clusters``
    (the grid is a launch argument: records differing only in it share one object); the gather variants (the A rows
    gathered by the kernel) append ``_gather`` / ``_gather_ldgsts``."""

    if isinstance(chain, str):
        chain = GroupedChain.legacy(chain, out_fmt)
    knobs = dataclasses.replace(cfg, grid_clusters=0)
    return jit.sanitize_symbol_name(
        f"grouped_bf16_{chain.describe()}_{knobs.describe()}"
        + gather_suffix(gather, a_producer)
    )


def can_implement_grouped(
    knobs: Knobs,
    N: int,
    K: int,
    E: int,
    chain: Union[str, GroupedChain] = "plain",
    out_fmt: str = "bf16",
) -> bool:
    """Acceptance of the grouped body on the dense blocks: well-formed knobs at swizzle 1 (the group scheduler has no
    swizzle), the dense tiles' N (the narrow decode tiles have no staged-epilogue layout), 16-byte-aligned operands (K
    and N multiples of the format's alignment), at most ``MAX_E`` experts, the gated chain on whole ``2 x GROUP`` column
    blocks -- plus the static TMEM thread-value check done at trace time."""
    if isinstance(chain, str):
        if chain not in CHAINS or out_fmt not in OUT_DT:
            return False
        chain = GroupedChain.legacy(chain, out_fmt)
    if (
        not knobs.well_formed()
        or knobs.swizzle_size != 1
        or knobs.cluster_k > 1
        or knobs.swap_ab
    ):
        return False
    if knobs.tile_n not in TILES_N:
        return False
    if chain.out_fmt not in OUT_DT:
        return False
    if chain.pair and N % (2 * GROUP):
        return False  # the gate/up pairing needs whole 2 x GROUP column blocks
    al = ALIGN["bf16"]
    n_out = N // 2 if chain.pair else N
    return (
        K % al == 0
        and N % al == 0
        and n_out % ALIGN[chain.out_fmt] == 0
        and 1 <= E <= MAX_E
    )


def compile_grouped_fn(
    cfg: Knobs, chain: GroupedChain, gather: bool = False, a_producer: str = "gather4"
) -> Callable[[], Any]:
    """The zero-argument compile closure of one grouped specialisation on the dense blocks. ``gather``: the variant whose
    A rows the kernel gathers (``kernel(x2, src, k_div, b3, out2, counts, grid)``: ``x (T, K)`` the unpermuted rows,
    ``src (Tk,)`` int32 the slot of every expanded row, row ``p`` reading ``x[src[p] // k_div]``) in place of the permuted
    ``a3``."""
    in_dt, al = IN_DT["bf16"], ALIGN["bf16"]
    out_dt, oal = OUT_DT[chain.out_fmt], ALIGN[chain.out_fmt]

    def compile_fn():
        sym_m, sym_n, sym_k, sym_e = (
            cute.sym_int(32),
            cute.sym_int(32, divisibility=al),
            cute.sym_int(32, divisibility=al),
            cute.sym_int(32),
        )
        sym_no = cute.sym_int(32, divisibility=oal)
        if gather:
            x_fake = cute.runtime.make_fake_tensor(
                in_dt,
                (cute.sym_int(32), sym_k),
                (cute.sym_int64(divisibility=al), 1),
                assumed_align=16,
            )
            src_fake = cute.runtime.make_fake_tensor(
                Int32, (cute.sym_int(32),), (1,), assumed_align=16
            )
            a_args = (x_fake, src_fake, Int32(1))

            def cls(cfg_, ch_):
                return GatherGroupedBlockGemm(cfg_, ch_, a_producer=a_producer)
        else:
            a_fake = cute.runtime.make_fake_tensor(
                in_dt,
                (sym_m, sym_k, 1),
                (cute.sym_int64(divisibility=al), 1, cute.sym_int64(divisibility=al)),
                assumed_align=16,
            )
            a_args = (a_fake,)
            cls = GroupedBlockGemm
        b_fake = cute.runtime.make_fake_tensor(
            in_dt,
            (sym_n, sym_k, sym_e),
            (cute.sym_int64(divisibility=al), 1, cute.sym_int64(divisibility=al)),
            assumed_align=16,
        )
        o_fake = cute.runtime.make_fake_tensor(
            out_dt,
            (sym_m, sym_no),
            (cute.sym_int64(divisibility=oal), 1),
            assumed_align=16,
        )
        c_fake = cute.runtime.make_fake_tensor(Int32, (sym_e,), (1,), assumed_align=4)
        stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        grid = GRID_PLACEHOLDER  # a dynamic argument of the kernel: the launch passes the device's value
        return cute.compile(
            cls(cfg, chain),
            *a_args,
            b_fake,
            o_fake,
            c_fake,
            grid,
            stream_fake,
            options="--enable-tvm-ffi",
        )

    return compile_fn


@functools.cache
def get_compiled_grouped(
    cfg: Knobs,
    chain: Union[str, GroupedChain] = "plain",
    out_fmt: str = "bf16",
    gather: bool = False,
    a_producer: str = "gather4",
) -> Any:
    """``kernel(a3, b3, out2, counts, grid)``: A ``(Tk, K, 1)`` K-major bf16, B ``(N, K, E)`` K-major bf16, out
    ``(Tk, Nout)`` row-major (``Nout = N``, or ``N / 2`` for the gated chains), counts ``(E,)`` int32, ``grid`` the
    persistent grid in clusters (the device's co-resident cluster count for the record's cluster size). ``chain``: a
    :class:`GroupedChain` or one of the bf16 class's legacy names (``plain`` / ``swiglu`` / ``relu2``). ``gather``: the
    variant ``kernel(x2, src, k_div, b3, out2, counts, grid)`` gathering row ``p`` as ``x[src[p] // k_div]``."""
    if isinstance(chain, str):
        assert chain in CHAINS and (chain == "plain" or out_fmt == "bf16"), (
            chain,
            out_fmt,
        )
        chain = GroupedChain.legacy(chain, out_fmt)
    return jit.build_and_load(
        MODULE,
        kernel_name(cfg, chain, gather=gather, a_producer=a_producer),
        compile_grouped_fn(cfg, chain, gather, a_producer),
    )
