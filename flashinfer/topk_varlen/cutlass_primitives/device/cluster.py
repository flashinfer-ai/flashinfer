"""Thread-block clusters: rank, distributed shared memory (DSMEM) access, cluster barrier.

On SM90 and newer, up to 8 (portably) CTAs launched as a cluster run at the same time on one
GPC and can read and write each other's shared memory.  That is how a row split across a
cluster merges its histograms and candidates without a trip through global memory: each CTA
adds its 256 bin counts into rank 0's histogram over DSMEM, one cluster barrier, and rank 0
holds the total.

Addressing: a local shared byte address (``Int32``, from ``pointer.toint()``) is mapped once
per peer with ``peer_shared_address``; the result is a byte address in the peer's window and
ordinary offsets can be added to it.  All ``peer_*`` operations take such a mapped address.

Barrier: ``cluster_sync`` is arrive + wait with release/acquire semantics, which is what makes
DSMEM writes before it visible after it.  The DSL's ``cluster_arrive_relaxed`` has no release
and races the remote stores; it is not exposed here.

Availability: all of this needs SM90 or newer and a launch with ``cluster=(size, 1, 1)``.
The dispatcher checks the capability and the driver's co-resident cluster count; nothing
below this layer does.
"""

import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, nvvm
from cutlass.cutlass_dsl import T, dsl_user_op

__all__ = [
    "cluster_rank",
    "cluster_size",
    "peer_shared_address",
    "peer_load_i32",
    "peer_load_f32",
    "peer_load_16",
    "peer_store_i32",
    "peer_store_f32",
    "peer_store_u64",
    "peer_add_i32",
    "peer_min_u32",
    "peer_max_u32",
    "peer_add_release_i32",
    "shared_load_acquire_cluster",
    "cluster_arrive",
    "cluster_wait",
    "cluster_sync",
]


@cute.jit
def cluster_rank():
    """This CTA's index within its cluster, 0 .. size-1 (``%cluster_ctarank``)."""
    return cute.arch.block_idx_in_cluster()


@dsl_user_op
def cluster_size(*, loc=None, ip=None):
    """Number of CTAs in the cluster along x (``%cluster_nctaid.x``), the shape this library
    launches.  Not the DSL's ``cluster_dim``, which is the grid size counted in clusters."""
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [],
            "mov.u32 $0, %cluster_nctaid.x;",
            "=r",
            has_side_effects=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def peer_shared_address(
    local_address: cutlass.Int32, peer: cutlass.Int32, *, loc=None, ip=None
):
    """Map a local shared byte address to the same offset in CTA ``peer``'s shared memory.

    ``mapa.shared::cluster``.  Pure; one instruction.  Map a region base once and add offsets.
    """
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [
                cutlass.Int32(local_address).ir_value(loc=loc, ip=ip),
                cutlass.Int32(peer).ir_value(loc=loc, ip=ip),
            ],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


def _peer_load(asm, out_type, wrap, mapped, constraint_out, loc, ip):
    return wrap(
        llvm.inline_asm(
            out_type,
            [cutlass.Int32(mapped).ir_value(loc=loc, ip=ip)],
            asm,
            f"{constraint_out},r",
            has_side_effects=True,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def peer_load_i32(mapped_address: cutlass.Int32, *, loc=None, ip=None):
    """Int32 at a mapped peer address (``ld.shared::cluster.u32``).  One instruction; the
    latency is a few times a local shared load."""
    return _peer_load(
        "ld.shared::cluster.u32 $0, [$1];",
        T.i32(),
        cutlass.Int32,
        mapped_address,
        "=r",
        loc,
        ip,
    )


@dsl_user_op
def peer_load_f32(mapped_address: cutlass.Int32, *, loc=None, ip=None):
    """Float32 at a mapped peer address (``ld.shared::cluster.f32``).  One instruction."""
    return _peer_load(
        "ld.shared::cluster.f32 $0, [$1];",
        T.f32(),
        cutlass.Float32,
        mapped_address,
        "=f",
        loc,
        ip,
    )


@dsl_user_op
def peer_load_16(mapped_address: cutlass.Int32, *, loc=None, ip=None):
    """Four Uint32 words from a 16-byte aligned mapped peer address (``ld.shared::cluster.v4``).

    One instruction; the way to pull a peer's histogram (64 loads for 256 bins).
    """
    packed = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(i32, i32, i32, i32)>"),
        [cutlass.Int32(mapped_address).ir_value(loc=loc, ip=ip)],
        "ld.shared::cluster.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        cutlass.Uint32(llvm.extractvalue(T.i32(), packed, [i])) for i in range(4)
    )


def _peer_store(asm, constraint_val, mapped, val_ir, loc, ip):
    llvm.inline_asm(
        None,
        [cutlass.Int32(mapped).ir_value(loc=loc, ip=ip), val_ir],
        asm,
        f"r,{constraint_val}",
        has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def peer_store_i32(
    mapped_address: cutlass.Int32, val: cutlass.Int32, *, loc=None, ip=None
) -> None:
    """Store an Int32 at a mapped peer address (``st.shared::cluster.u32``).  One instruction.
    Visible to the peer after the next ``cluster_sync``."""
    _peer_store(
        "st.shared::cluster.u32 [$0], $1;",
        "r",
        mapped_address,
        cutlass.Int32(val).ir_value(loc=loc, ip=ip),
        loc,
        ip,
    )


@dsl_user_op
def peer_store_f32(
    mapped_address: cutlass.Int32, val: cutlass.Float32, *, loc=None, ip=None
) -> None:
    """Store a Float32 at a mapped peer address (``st.shared::cluster.f32``).  One instruction."""
    _peer_store(
        "st.shared::cluster.f32 [$0], $1;",
        "f",
        mapped_address,
        cutlass.Float32(val).ir_value(loc=loc, ip=ip),
        loc,
        ip,
    )


@dsl_user_op
def peer_store_u64(
    mapped_address: cutlass.Int32, val: cutlass.Uint64, *, loc=None, ip=None
) -> None:
    """Store a Uint64 at an 8-byte aligned mapped peer address (``st.shared::cluster.u64``).

    One instruction.  A (key, index) candidate is pushed as ``(key << 32) | index`` in one
    store so a reader never sees a key without its index.
    """
    _peer_store(
        "st.shared::cluster.u64 [$0], $1;",
        "l",
        mapped_address,
        cutlass.Uint64(val).ir_value(loc=loc, ip=ip),
        loc,
        ip,
    )


@dsl_user_op
def peer_add_i32(
    mapped_address: cutlass.Int32, val: cutlass.Int32, *, loc=None, ip=None
):
    """Add ``val`` to the Int32 at a mapped peer address; return the old value.  Cluster scope.

    ``atom.relaxed.cluster.shared::cluster.add``.  One instruction; the DSMEM histogram merge
    is 256 of these per CTA into rank 0's bins, or a cursor reservation on a peer's counter.
    """
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [
                cutlass.Int32(mapped_address).ir_value(loc=loc, ip=ip),
                cutlass.Int32(val).ir_value(loc=loc, ip=ip),
            ],
            "atom.relaxed.cluster.shared::cluster.add.u32 $0, [$1], $2;",
            "=r,r,r",
            has_side_effects=True,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


def _peer_reduce(op, mapped_address, val, loc, ip):
    llvm.inline_asm(
        None,
        [
            cutlass.Int32(mapped_address).ir_value(loc=loc, ip=ip),
            cutlass.Uint32(val).ir_value(loc=loc, ip=ip),
        ],
        f"red.relaxed.cluster.shared::cluster.{op}.u32 [$0], $1;",
        "r,r",
        has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def peer_min_u32(
    mapped_address: cutlass.Int32, val: cutlass.Uint32, *, loc=None, ip=None
) -> None:
    """Unsigned minimum of ``val`` into the Uint32 at a mapped peer address; no result.
    Cluster scope, one instruction.  The way peers fold a key range into rank 0 before the
    barrier after which they may exit (rank 0 must not read a peer's memory after that)."""
    _peer_reduce("min", mapped_address, val, loc, ip)


@dsl_user_op
def peer_max_u32(
    mapped_address: cutlass.Int32, val: cutlass.Uint32, *, loc=None, ip=None
) -> None:
    """Unsigned maximum of ``val`` into the Uint32 at a mapped peer address; no result.
    Cluster scope, one instruction."""
    _peer_reduce("max", mapped_address, val, loc, ip)


@dsl_user_op
def peer_add_release_i32(
    mapped_address: cutlass.Int32, val: cutlass.Int32, *, loc=None, ip=None
) -> None:
    """Add ``val`` to the Int32 at a mapped peer address with release semantics at cluster
    scope, no result (``red.release.cluster.shared::cluster.add``).

    The one-sided arrival: after a CTA's block barrier (so every thread's DSMEM stores into the
    peer precede this in the CTA's order), one thread adds 1 to the peer's arrival word; the
    release makes those stores visible to a thread of the peer that observes the count with an
    acquire load.  Lets a CTA leave a cluster phase without a full cluster barrier (v0.1.28,
    the clustered register kernel's tie hand-over).
    """
    llvm.inline_asm(
        None,
        [
            cutlass.Int32(mapped_address).ir_value(loc=loc, ip=ip),
            cutlass.Int32(val).ir_value(loc=loc, ip=ip),
        ],
        "fence.acq_rel.cluster;\nred.release.cluster.shared::cluster.add.u32 [$0], $1;",
        "r,r",
        has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def shared_load_acquire_cluster(address: cutlass.Int32, *, loc=None, ip=None):
    """Int32 at a local shared-memory address with acquire semantics at cluster scope
    (``ld.acquire.cluster.shared::cta``): the poll side of ``peer_add_release_i32``.  Every
    DSMEM store a peer released before its add is visible after the load that sees the add."""
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [cutlass.Int32(address).ir_value(loc=loc, ip=ip)],
            "ld.acquire.cluster.shared::cta.b32 $0, [$1];",
            "=r,r",
            has_side_effects=True,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def cluster_arrive(*, loc=None, ip=None) -> None:
    """Signal arrival at the cluster barrier with release semantics (``barrier.cluster.arrive
    .aligned``).  All threads of the CTA must execute it (aligned form)."""
    nvvm.cluster_arrive(aligned=True, loc=loc, ip=ip)


@dsl_user_op
def cluster_wait(*, loc=None, ip=None) -> None:
    """Wait until every CTA of the cluster has arrived, with acquire semantics
    (``barrier.cluster.wait.aligned``)."""
    nvvm.cluster_wait(aligned=True, loc=loc, ip=ip)


@cute.jit
def cluster_sync():
    """Full cluster barrier: arrive then wait.  DSMEM writes before it are visible after it.

    Cost: comparable to a block barrier plus the cross-SM round trip, about a microsecond on
    B200 for a cluster of 4.  Split into ``cluster_arrive`` / ``cluster_wait`` to overlap
    independent work with the wait.
    """
    cluster_arrive()
    cluster_wait()
