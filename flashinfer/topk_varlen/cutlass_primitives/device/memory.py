"""Vector loads and stores: 16-byte global loads, 16-byte and 8-byte shared-memory accesses.

Selection kernels are limited by instruction issue, not bandwidth, so every element that can
move in a 128-bit instruction should.  These wrappers pin one PTX instruction each; the DSL's
copy atoms are avoided because NVVM rewrites adjacent 128-bit fp32 loads into 64-bit register
pairs, and the pair alignment fragments allocation at the 64-register wall (measured as spills
in the streaming kernel).

Addresses are byte addresses: ``Int64`` for global memory (``tensor.iterator.toint()``),
``Int32`` for shared memory (``pointer.toint()``).  Every 16-byte access needs a 16-byte
aligned address; the hardware faults otherwise.  Loads return raw ``Uint32`` words; callers
reinterpret (``.to(Int32)``, ``bitcast``) as their data demands.

Which global load to use:

* ``load_global_16``: coherent, cached in L1.  Data another CTA may have written this kernel.
* ``load_global_readonly_16``: the read-only data path (``LDG.E.128.CONSTANT``).  The row is
  never written during the kernel, and this path sustains far more loads in flight: the
  filter pass moved 1.6x more bytes per microsecond after switching to it.
* ``load_global_l2_16``: bypass L1, read straight from L2 (``.cg``).  For slabs written by
  other CTAs and merged by a last arriver, where L1 could hold a stale line.
"""

import cutlass
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

__all__ = [
    "load_global_16",
    "load_global_readonly_16",
    "load_global_l2_16",
    "load_global_l2_8",
    "load_global_l2_i32",
    "prefetch_l2",
    "load_shared_16",
    "store_shared_16",
    "clear_shared_16",
    "store_shared_8",
]

_FOUR_WORDS = "!llvm.struct<(i32, i32, i32, i32)>"
_TWO_WORDS = "!llvm.struct<(i32, i32)>"


def _load_words(asm, address_ir, address_constraint, count, loc, ip):
    """Issue one load returning ``count`` (2 or 4) 32-bit words; unpack to Uint32."""
    struct_type = ir.Type.parse(_FOUR_WORDS if count == 4 else _TWO_WORDS)
    outs = ",".join(["=r"] * count)
    packed = llvm.inline_asm(
        struct_type,
        [address_ir],
        asm,
        f"{outs},{address_constraint}",
        has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        cutlass.Uint32(llvm.extractvalue(T.i32(), packed, [i])) for i in range(count)
    )


@dsl_user_op
def load_global_16(address: cutlass.Int64, *, loc=None, ip=None):
    """Four Uint32 words from a 16-byte aligned global byte address, coherent (``ld.global.v4``).

    One instruction.  Use for data written earlier in this kernel by another CTA.
    """
    return _load_words(
        "ld.global.v4.b32 {$0, $1, $2, $3}, [$4];",
        cutlass.Int64(address).ir_value(loc=loc, ip=ip),
        "l",
        4,
        loc,
        ip,
    )


@dsl_user_op
def load_global_readonly_16(address: cutlass.Int64, *, loc=None, ip=None):
    """Four Uint32 words through the read-only path (``ld.global.nc.v4``).

    Precondition: no thread of this grid writes the line during the kernel.  One instruction;
    the preferred load for input rows.
    """
    return _load_words(
        "ld.global.nc.v4.b32 {$0, $1, $2, $3}, [$4];",
        cutlass.Int64(address).ir_value(loc=loc, ip=ip),
        "l",
        4,
        loc,
        ip,
    )


@dsl_user_op
def load_global_l2_16(address: cutlass.Int64, *, loc=None, ip=None):
    """Four Uint32 words bypassing L1 (``ld.global.cg.v4``), for lines other CTAs wrote.

    One instruction.  Pair with a release/acquire handshake (see ``atomics``) so the writes
    are visible before the read.
    """
    return _load_words(
        "ld.global.cg.v4.b32 {$0, $1, $2, $3}, [$4];",
        cutlass.Int64(address).ir_value(loc=loc, ip=ip),
        "l",
        4,
        loc,
        ip,
    )


@dsl_user_op
def load_global_l2_8(address: cutlass.Int64, *, loc=None, ip=None):
    """Two Uint32 words bypassing L1 (``ld.global.cg.v2``); the 8-byte (key, index) slab word.

    Address must be 8-byte aligned.  One instruction.
    """
    return _load_words(
        "ld.global.cg.v2.b32 {$0, $1}, [$2];",
        cutlass.Int64(address).ir_value(loc=loc, ip=ip),
        "l",
        2,
        loc,
        ip,
    )


@dsl_user_op
def load_global_l2_i32(address: cutlass.Int64, *, loc=None, ip=None):
    """One Int32 bypassing L1 (``ld.global.cg.b32``), for a word another CTA wrote.

    L1 is not coherent: a line this SM cached in an earlier launch would be returned stale by
    a plain load.  One instruction.
    """
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [cutlass.Int64(address).ir_value(loc=loc, ip=ip)],
            "ld.global.cg.b32 $0, [$1];",
            "=r,l",
            has_side_effects=True,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def prefetch_l2(address: cutlass.Int64, *, loc=None, ip=None) -> None:
    """Ask L2 to fetch the line holding ``address`` (``prefetch.global.L2``); no result.

    Measured: prefetching the next 64 KB tile during the filter pass made the streaming kernel
    slower (the pass is issue-bound, and the prefetch instructions compete for issue).  Kept
    for the sample phase, where a few prefetches ahead of a strided gather did pay.
    """
    llvm.inline_asm(
        None,
        [cutlass.Int64(address).ir_value(loc=loc, ip=ip)],
        "prefetch.global.L2 [$0];",
        "l",
        has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def load_shared_16(address: cutlass.Int32, *, loc=None, ip=None):
    """Four Uint32 words from a 16-byte aligned shared byte address (``ld.shared.v4``).

    One instruction; four histogram bins per thread per load when scanning a 256-bin
    histogram with 64 threads.
    """
    return _load_words(
        "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];",
        cutlass.Int32(address).ir_value(loc=loc, ip=ip),
        "r",
        4,
        loc,
        ip,
    )


@dsl_user_op
def store_shared_16(
    address: cutlass.Int32,
    w0: cutlass.Uint32,
    w1: cutlass.Uint32,
    w2: cutlass.Uint32,
    w3: cutlass.Uint32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Store four Uint32 words to a 16-byte aligned shared byte address (``st.shared.v4``).

    One instruction.
    """
    llvm.inline_asm(
        None,
        [
            cutlass.Int32(address).ir_value(loc=loc, ip=ip),
            cutlass.Uint32(w0).ir_value(loc=loc, ip=ip),
            cutlass.Uint32(w1).ir_value(loc=loc, ip=ip),
            cutlass.Uint32(w2).ir_value(loc=loc, ip=ip),
            cutlass.Uint32(w3).ir_value(loc=loc, ip=ip),
        ],
        "st.shared.v4.b32 [$0], {$1, $2, $3, $4};",
        "r,r,r,r,r",
        has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def clear_shared_16(address: cutlass.Int32, *, loc=None, ip=None) -> None:
    """Zero 16 bytes of shared memory at a 16-byte aligned byte address.

    One instruction with immediate zeros (no registers).  A 1024-thread block clears a
    16 KB stage in one instruction per thread.
    """
    llvm.inline_asm(
        None,
        [cutlass.Int32(address).ir_value(loc=loc, ip=ip)],
        "st.shared.v4.b32 [$0], {0, 0, 0, 0};",
        "r",
        has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def load_shared_8(address: cutlass.Int32, *, loc=None, ip=None):
    """Two Uint32 words from an 8-byte aligned shared byte address (``ld.shared.v2.b32``):
    ``(low, high)``, the pair ``store_shared_8`` wrote.  One instruction."""
    return _load_words(
        "ld.shared.v2.b32 {$0, $1}, [$2];",
        cutlass.Int32(address).ir_value(loc=loc, ip=ip),
        "r",
        2,
        loc,
        ip,
    )


@dsl_user_op
def store_shared_8(
    address: cutlass.Int32,
    low: cutlass.Uint32,
    high: cutlass.Uint32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Store two Uint32 words as one 8-byte shared access (``st.shared.v2.u32``).

    Address must be 8-byte aligned; ``low`` lands at the lower address.  Byte-identical to
    storing the Uint64 ``(high << 32) | low``, but the two words stay in independent
    registers, so a (key, index) pair carried through a loop goes straight into the store
    without a pack instruction.  One instruction.
    """
    llvm.inline_asm(
        None,
        [
            cutlass.Int32(address).ir_value(loc=loc, ip=ip),
            cutlass.Uint32(low).ir_value(loc=loc, ip=ip),
            cutlass.Uint32(high).ir_value(loc=loc, ip=ip),
        ],
        "st.shared.v2.u32 [$0], {$1, $2};",
        "r,r,r",
        has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
