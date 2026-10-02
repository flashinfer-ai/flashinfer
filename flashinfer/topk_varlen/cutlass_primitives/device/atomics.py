"""Atomic adds on shared and global memory, and the release/acquire pair for cross-CTA handoff.

Scope is the whole story here.  An atomic's scope names how far its effect must be ordered:
``cta`` (this block), ``gpu`` (this device), ``sys`` (host too).  A wider scope costs more.
Shared-memory counters need only ``cta``; the DSL's convenience ``atomicAdd`` is ``sys``
scope even on shared memory and measured slower for histogram bins.  Global counters that
other CTAs read need ``gpu``.

Two spellings of "add":

* ``*_add`` returns the old value (``atom``).  Needed for tickets: "give me my slot".
* ``*_count`` / ``*_add_noreturn`` returns nothing (``red``).  For histogram bins and
  arrival counts where nobody reads the result; ptxas emits the cheaper reduction form.

Cross-CTA handoff (a last arriver merging what the others wrote) is a release/acquire
protocol.  The writer publishes with ``fence_release_gpu`` followed by ``global_add``; the
one that sees ``old == others`` is last and reads the data with ``global_load_acquire`` or
an L2 load.  ``global_store_release`` resets the counter for the next launch.
"""

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

__all__ = [
    "shared_add",
    "shared_count",
    "shared_add_noreturn",
    "global_add",
    "global_add_acq_rel",
    "global_add_noreturn",
    "fence_release_gpu",
    "global_store_release",
    "global_load_acquire",
]


@cute.jit
def shared_add(ptr, val):
    """Add ``val`` (Int32) at shared pointer ``ptr``; return the old value.  CTA scope.

    Lowers to ``atom.relaxed.cta.shared.add``.  One instruction; same-address contention
    inside a block is cheaper than the shuffle-scan alternatives (measured in the emit).
    """
    return cute.arch.atomic_add(ptr, cutlass.Int32(val), sem="relaxed", scope="cta")


@dsl_user_op
def shared_count(address: cutlass.Int32, *, loc=None, ip=None) -> None:
    """Increment the Int32 at shared byte ``address`` by one; no result.  CTA scope.

    ``red.relaxed.cta.shared.add.u32 [address], 1``: the histogram-bin increment.  Takes
    the final byte address so ptxas fuses the bin shift and add into one address
    calculation against a pinned base (see ``compiler.pin_shared_address``).
    """
    llvm.inline_asm(
        None,
        [cutlass.Int32(address).ir_value(loc=loc, ip=ip)],
        "red.relaxed.cta.shared.add.u32 [$0], 1;",
        "r",
        has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cute.jit
def shared_add_noreturn(ptr, val):
    """Add ``val`` (Int32) at shared pointer ``ptr``; no result.  CTA scope, one instruction."""
    cute.arch.red(
        ptr, cutlass.Int32(val), op="add", dtype="s32", sem="relaxed", scope="cta"
    )


@cute.jit
def global_add(ptr, val):
    """Add ``val`` (Int32) at global pointer ``ptr``; return the old value.  GPU scope.

    ``atom.relaxed.gpu.global.add``.  Relaxed: pair with ``fence_release_gpu`` when the add
    publishes earlier writes.
    """
    return cute.arch.atomic_add(ptr, cutlass.Int32(val), sem="relaxed", scope="gpu")


@cute.jit
def global_add_acq_rel(ptr, val):
    """Add ``val`` (Int32) at global pointer ``ptr`` with acquire-release semantics, GPU scope;
    return the old value.

    ``atom.acq_rel.gpu.global.add``: the arrival handshake in one instruction.  Release orders
    this thread's earlier writes (the published slab) before the add; acquire makes every
    earlier arriver's published writes visible to the thread that sees the final count.
    Replaces a ``fence_release_gpu`` + relaxed add + ``fence.acq_rel`` triple (two memory-
    pipeline drains, one of them paid by every CTA; measured 0.2 us per row at 1M b=8).
    """
    return cute.arch.atomic_add(ptr, cutlass.Int32(val), sem="acq_rel", scope="gpu")


@cute.jit
def global_add_noreturn(ptr, val):
    """Add ``val`` (Int32) at global pointer ``ptr``; no result.  GPU scope, one instruction."""
    cute.arch.red(
        ptr, cutlass.Int32(val), op="add", dtype="s32", sem="relaxed", scope="gpu"
    )


@cute.jit
def fence_release_gpu():
    """Order this thread's earlier global writes before its later ones, device wide.

    ``fence.acq_rel.gpu``.  Issue after the slab writes and before the arrival add, so the
    last arriver's acquire load sees the slab.  Costs a memory-pipeline drain; once per CTA.
    """
    cute.arch.fence_acq_rel_gpu()


@dsl_user_op
def global_store_release(
    address: cutlass.Int64, val: cutlass.Int32, *, loc=None, ip=None
) -> None:
    """Store ``val`` at global byte ``address`` with release semantics, device wide.

    ``fence.acq_rel.gpu`` then ``st.release.gpu.global.b32``.  Resets an arrival counter so
    the next launch starts from zero and cannot observe this launch's slab.
    """
    llvm.inline_asm(
        None,
        [
            cutlass.Int64(address).ir_value(loc=loc, ip=ip),
            cutlass.Int32(val).ir_value(loc=loc, ip=ip),
        ],
        "fence.acq_rel.gpu;\nst.release.gpu.global.b32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def global_load_acquire(address: cutlass.Int64, *, loc=None, ip=None):
    """Int32 at global byte ``address`` with acquire semantics (``ld.global.acquire.gpu``).

    Everything the releasing thread wrote before its release is visible after this load.  A
    plain load, not an atomic: many CTAs can poll it without serializing on the line.
    """
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [cutlass.Int64(address).ir_value(loc=loc, ip=ip)],
            "ld.global.acquire.gpu.b32 $0, [$1];",
            "=r,l",
            has_side_effects=True,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )
