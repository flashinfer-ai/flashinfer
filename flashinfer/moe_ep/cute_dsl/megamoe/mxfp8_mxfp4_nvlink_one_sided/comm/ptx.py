# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""PTX primitives for the Sentinel (data-as-flag) MegaMoE protocol.

Landing rows are armed with a sentinel the payload never takes; receivers poll the
data itself. The dispatch payload (E4M3 data and UE8M0 scales) uses the byte
sentinel ``0xFF``: senders rewrite any ``0xFF`` byte (a NaN) to ``0x7F`` (data) or
``0xFE`` (scales) before pushing, so a row has arrived exactly when none of its bytes
is ``0xFF``. A byte is written atomically by any writer, so the test is tear-proof
for LSU and bulk pushes alike. BF16 combine partials use the word sentinel
``0xFFFFFFFF``: writers rewrite every ``0xFFFF`` lane to ``0x7FFF`` and push with
32-bit-element vector stores, whose elements are single-copy atomic.

Remote pushes and polls use ``.volatile`` accesses (``.relaxed.sys`` semantics), so a
poll racing with a push is a defined race and the loaded words are used directly;
landing data is never re-read with a weak load after a poll.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, vector
from cutlass.cutlass_dsl import T, dsl_user_op

BYTE_SENTINEL_WORD = 0xFFFFFFFF
WORD_SENTINEL = 0xFFFFFFFF
# Polls give up after this many sweeps (seconds of waiting) and trap with a message,
# so a protocol bug fails loudly instead of hanging the job. A poll that can start
# before every peer has launched counts its budget only once the peers have sent.
POLL_SWEEP_LIMIT = 1 << 24


def _asm(
    result_type, operands, text, constraints, *, side_effects=True, loc=None, ip=None
):
    return llvm.inline_asm(
        result_type,
        operands,
        text,
        constraints,
        has_side_effects=side_effects,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _pack4(loaded, loc=None, ip=None):
    packed = vector.from_elements(
        ir.VectorType.get([4], T.i32(), loc=loc),
        [
            llvm.extractvalue(T.i32(), loaded, [index], loc=loc, ip=ip)
            for index in range(4)
        ],
        loc=loc,
        ip=ip,
    )
    return cute.TensorSSA(packed, 4, Uint32)


# ----------------------------------------------------------------------------
# Data path: volatile vector stores and polls
# ----------------------------------------------------------------------------


@dsl_user_op
def st_volatile_v4(address: Int64, packed, *, loc=None, ip=None) -> None:
    """Store four words with ``st.volatile.global.v4.u32`` (local or peer address)."""
    words = [packed[index].ir_value(loc=loc, ip=ip) for index in range(4)]
    _asm(
        None,
        [Int64(address).ir_value(loc=loc, ip=ip), *words],
        "st.volatile.global.v4.u32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def st_volatile_v4_words(
    address: Int64, w0: Uint32, w1: Uint32, w2: Uint32, w3: Uint32, *, loc=None, ip=None
) -> None:
    _asm(
        None,
        [
            Int64(address).ir_value(loc=loc, ip=ip),
            Uint32(w0).ir_value(loc=loc, ip=ip),
            Uint32(w1).ir_value(loc=loc, ip=ip),
            Uint32(w2).ir_value(loc=loc, ip=ip),
            Uint32(w3).ir_value(loc=loc, ip=ip),
        ],
        "st.volatile.global.v4.u32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ld_global_v4(address: Int64, *, loc=None, ip=None):
    """Load four words with a weak ``ld.global.v4.u32`` the compiler may schedule freely."""
    loaded = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [Int64(address).ir_value(loc=loc, ip=ip)],
        "ld.global.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        side_effects=False,
        loc=loc,
        ip=ip,
    )
    return _pack4(loaded, loc, ip)


@dsl_user_op
def ld_global_v4_no_allocate(address: Int64, *, loc=None, ip=None):
    """Load four words with ``ld.global.L1::no_allocate.v4.u32``: a weak streaming read that leaves L1 alone.

    It keeps its place among the other side-effecting accesses, so loads issued
    before a store of the same warp leave before it.
    """
    loaded = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [Int64(address).ir_value(loc=loc, ip=ip)],
        "ld.global.L1::no_allocate.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        loc=loc,
        ip=ip,
    )
    return _pack4(loaded, loc, ip)


@dsl_user_op
def ld_global_u32_no_allocate(address: Int64, *, loc=None, ip=None) -> Uint32:
    """Load one word with ``ld.global.L1::no_allocate.u32``; see ``ld_global_v4_no_allocate``."""
    return Uint32(
        _asm(
            T.i32(),
            [Int64(address).ir_value(loc=loc, ip=ip)],
            "ld.global.L1::no_allocate.u32 $0, [$1];",
            "=r,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def st_global_v4_words(
    address: Int64, w0: Uint32, w1: Uint32, w2: Uint32, w3: Uint32, *, loc=None, ip=None
) -> None:
    """Store four words with a weak ``st.global.v4.u32``."""
    _asm(
        None,
        [
            Int64(address).ir_value(loc=loc, ip=ip),
            Uint32(w0).ir_value(loc=loc, ip=ip),
            Uint32(w1).ir_value(loc=loc, ip=ip),
            Uint32(w2).ir_value(loc=loc, ip=ip),
            Uint32(w3).ir_value(loc=loc, ip=ip),
        ],
        "st.global.v4.u32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def red_add_sys_v4_bf16x2(
    address: Int64, w0: Uint32, w1: Uint32, w2: Uint32, w3: Uint32, *, loc=None, ip=None
) -> None:
    """Add eight BF16 values (four packed words) to memory with ``red.relaxed.sys.global.add.noftz.v4.bf16x2``.

    Each BF16 lane adds atomically with system scope, so several ranks may add into
    the same word in any order; the sum rounds to BF16 at every add.
    """
    _asm(
        None,
        [
            Int64(address).ir_value(loc=loc, ip=ip),
            Uint32(w0).ir_value(loc=loc, ip=ip),
            Uint32(w1).ir_value(loc=loc, ip=ip),
            Uint32(w2).ir_value(loc=loc, ip=ip),
            Uint32(w3).ir_value(loc=loc, ip=ip),
        ],
        "red.relaxed.sys.global.add.noftz.v4.bf16x2 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def st_global_u32(address: Int64, value: Uint32, *, loc=None, ip=None) -> None:
    """Store one word with a weak ``st.global.u32``."""
    _asm(
        None,
        [
            Int64(address).ir_value(loc=loc, ip=ip),
            Uint32(value).ir_value(loc=loc, ip=ip),
        ],
        "st.global.u32 [$0], $1;",
        "l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ld_volatile_v4(address: Int64, *, loc=None, ip=None):
    """Load four words with ``ld.volatile.global.v4.u32``."""
    loaded = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [Int64(address).ir_value(loc=loc, ip=ip)],
        "ld.volatile.global.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        loc=loc,
        ip=ip,
    )
    return _pack4(loaded, loc, ip)


@dsl_user_op
def ld_shared_v4(pointer: cute.Pointer, *, loc=None, ip=None):
    """Load four words from shared memory; ordered by its side effects against named barriers."""
    loaded = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [Int32(pointer.toint(loc=loc, ip=ip)).ir_value(loc=loc, ip=ip)],
        "ld.shared.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        loc=loc,
        ip=ip,
    )
    return _pack4(loaded, loc, ip)


@dsl_user_op
def ld_volatile_u32(address: Int64, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        _asm(
            T.i32(),
            [Int64(address).ir_value(loc=loc, ip=ip)],
            "ld.volatile.global.u32 $0, [$1];",
            "=r,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def st_volatile_u32(address: Int64, value: Uint32, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [
            Int64(address).ir_value(loc=loc, ip=ip),
            Uint32(value).ir_value(loc=loc, ip=ip),
        ],
        "st.volatile.global.u32 [$0], $1;",
        "l,r",
        loc=loc,
        ip=ip,
    )


# ----------------------------------------------------------------------------
# Ordering: release / acquire, proxy fences
# ----------------------------------------------------------------------------


@dsl_user_op
def st_release_sys_u32(address: Int64, value: Uint32, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [
            Int64(address).ir_value(loc=loc, ip=ip),
            Uint32(value).ir_value(loc=loc, ip=ip),
        ],
        "st.release.sys.global.u32 [$0], $1;",
        "l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ld_acquire_sys_u32(address: Int64, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        _asm(
            T.i32(),
            [Int64(address).ir_value(loc=loc, ip=ip)],
            "ld.acquire.sys.global.u32 $0, [$1];",
            "=r,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ld_acquire_gpu_u32(address: Int64, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        _asm(
            T.i32(),
            [Int64(address).ir_value(loc=loc, ip=ip)],
            "ld.acquire.gpu.global.u32 $0, [$1];",
            "=r,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def red_release_gpu_add_u32(
    address: Int64, value: Uint32, *, loc=None, ip=None
) -> None:
    _asm(
        None,
        [
            Int64(address).ir_value(loc=loc, ip=ip),
            Uint32(value).ir_value(loc=loc, ip=ip),
        ],
        "red.release.gpu.global.add.u32 [$0], $1;",
        "l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def atom_max_relaxed_gpu_u32(
    address: Int64, value: Uint32, *, loc=None, ip=None
) -> Uint32:
    """``atom.relaxed.gpu.global.max.u32``: raise a word to at least ``value`` and return what it held."""
    return Uint32(
        _asm(
            T.i32(),
            [
                Int64(address).ir_value(loc=loc, ip=ip),
                Uint32(value).ir_value(loc=loc, ip=ip),
            ],
            "atom.relaxed.gpu.global.max.u32 $0, [$1], $2;",
            "=r,l,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def red_release_gpu_max_u32(
    address: Int64, value: Uint32, *, loc=None, ip=None
) -> None:
    """``red.release.gpu.global.max.u32``: raise a word to at least ``value`` after this thread's earlier accesses."""
    _asm(
        None,
        [
            Int64(address).ir_value(loc=loc, ip=ip),
            Uint32(value).ir_value(loc=loc, ip=ip),
        ],
        "red.release.gpu.global.max.u32 [$0], $1;",
        "l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def atom_add_acq_rel_gpu_u32(
    address: Int64, value: Uint32, *, loc=None, ip=None
) -> Uint32:
    return Uint32(
        _asm(
            T.i32(),
            [
                Int64(address).ir_value(loc=loc, ip=ip),
                Uint32(value).ir_value(loc=loc, ip=ip),
            ],
            "atom.acq_rel.gpu.global.add.u32 $0, [$1], $2;",
            "=r,l,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def fence_proxy_async_global(*, loc=None, ip=None) -> None:
    """Order this thread's generic-proxy view of global memory before its later TMA reads."""
    _asm(None, [], "fence.proxy.async.global;", "", loc=loc, ip=ip)


@dsl_user_op
def fence_acq_rel_sys(*, loc=None, ip=None) -> None:
    _asm(None, [], "fence.acq_rel.sys;", "", loc=loc, ip=ip)


@dsl_user_op
def fence_acq_rel_gpu(*, loc=None, ip=None) -> None:
    _asm(None, [], "fence.acq_rel.gpu;", "", loc=loc, ip=ip)


# ----------------------------------------------------------------------------
# Streaming hints and timing
# ----------------------------------------------------------------------------


@dsl_user_op
def prefetch_l2_bulk(address: Int64, size_bytes: Int32, *, loc=None, ip=None) -> None:
    """``cp.async.bulk.prefetch.L2.global``: a cache hint, ``size_bytes`` a multiple of 16."""
    _asm(
        None,
        [
            Int64(address).ir_value(loc=loc, ip=ip),
            Int32(size_bytes).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.bulk.prefetch.L2.global [$0], $1;",
        "l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def discard_global_l2(address: Int64, *, loc=None, ip=None) -> None:
    """``discard.global.L2``: drop the 128-byte line at ``address`` (128-byte aligned) from L2 without writing it back.

    The line's contents become indeterminate, so only data that no later read
    depends on may be discarded.
    """
    _asm(
        None,
        [Int64(address).ir_value(loc=loc, ip=ip)],
        "discard.global.L2 [$0], 128;",
        "l",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def globaltimer_ns(*, loc=None, ip=None) -> Int64:
    return Int64(_asm(T.i64(), [], "mov.u64 $0, %globaltimer;", "=l", loc=loc, ip=ip))


@dsl_user_op
def smid(*, loc=None, ip=None) -> Int32:
    """``%smid``: the SM this thread runs on (constant for the CTA's lifetime)."""
    return Int32(_asm(T.i32(), [], "mov.u32 $0, %smid;", "=r", loc=loc, ip=ip))


@dsl_user_op
def red_max_relaxed_gpu_u64(address: Int64, value: Int64, *, loc=None, ip=None) -> None:
    """``red.relaxed.gpu.global.max.u64``: keep the latest of several timestamps."""
    _asm(
        None,
        [
            Int64(address).ir_value(loc=loc, ip=ip),
            Int64(value).ir_value(loc=loc, ip=ip),
        ],
        "red.relaxed.gpu.global.max.u64 [$0], $1;",
        "l,l",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def trap(*, loc=None, ip=None) -> None:
    _asm(None, [], "trap;", "", loc=loc, ip=ip)


@dsl_user_op
def nanosleep(ns: Int32, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [Int32(ns).ir_value(loc=loc, ip=ip)],
        "nanosleep.u32 $0;",
        "r",
        loc=loc,
        ip=ip,
    )


# ----------------------------------------------------------------------------
# Sentinel tests and canonicalization (pure integer math)
# ----------------------------------------------------------------------------


@cute.jit
def ff_byte_mask(word: Uint32) -> Uint32:
    """0x80 in every byte lane of ``word`` that equals 0xFF, else 0."""
    inverted = ~word
    # Exact zero-byte test of ``inverted``: no carries cross lanes because each lane is
    # first reduced to 7 bits.
    low7 = (inverted & Uint32(0x7F7F7F7F)) + Uint32(0x7F7F7F7F)
    return ~(low7 | inverted) & Uint32(0x80808080)


@cute.jit
def has_ff_byte(word: Uint32) -> cutlass.Boolean:
    return ff_byte_mask(word) != Uint32(0)


@cute.jit
def canonicalize_e4m3_word(word: Uint32) -> Uint32:
    """Rewrite every 0xFF byte (negative NaN) to 0x7F (NaN)."""
    return word ^ ff_byte_mask(word)


@cute.jit
def canonicalize_ue8m0_word(word: Uint32) -> Uint32:
    """Rewrite every 0xFF byte (NaN scale) to 0xFE."""
    return word ^ (ff_byte_mask(word) >> Uint32(7))


@cute.jit
def canonicalize_bf16_pair(word: Uint32) -> Uint32:
    """Rewrite every 0xFFFF lane (a NaN) to 0x7FFF so no word equals the sentinel."""
    inverted = ~word
    low15 = (inverted & Uint32(0x7FFF7FFF)) + Uint32(0x7FFF7FFF)
    return word ^ (~(low15 | inverted) & Uint32(0x80008000))
