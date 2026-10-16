# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT


import enum
from dataclasses import dataclass, field, replace
from typing import Callable, NamedTuple, Union

from cutlass.cute.arch.nvvm_wrappers import inline_ptx
from cutlass.experimental import primitives as nvvm
import cutlass
import cutlass.cute as cute

WAIT_TIMEOUT = 1


class PipelineState(NamedTuple):
    idx: object
    phase: object

    @classmethod
    def start(cls, phase: int = 0):
        return cls(idx=cutlass.Int32(0), phase=cutlass.Int32(phase))


def advance(state, stages):
    if stages < 1:
        raise ValueError(f"PipelineState.advance requires stages >= 1, got {stages}")
    incr = state.idx + cutlass.Int32(1)
    stages_i = cutlass.Int32(stages)
    new_idx = incr % stages_i
    flip = incr // stages_i
    new_phase = state.phase ^ flip
    return PipelineState(idx=new_idx, phase=new_phase)


@cute.jit
def wait_try(mb, phase):
    """Untimed ``mbarrier.try_wait.parity`` loop through the DSL wrapper (no suspend hint, no inline PTX)."""
    while not nvvm.mbarrier_wait_parity(mb, phase, nvvm.MBarrierWait.TRY):
        pass


@cute.jit
def wait(mb, phase, spin: cutlass.Constexpr[bool] = False):
    """Spin on ``mb`` until its phase parity differs from ``phase``.

    ``spin=False`` (the default; an untouched call site compiles to the same cubin as before this kwarg existed) is the DSL
    wrapper's ``mbarrier.try_wait.parity`` with the ``time_limit`` suspend hint.  ``spin=True`` is the C++ reference's
    hint-less ``MBarrier::wait`` as inline PTX (the public wrapper always emits the 4-operand ``.timelimit`` form, so the
    hint-less one has to be spelled by hand).  Both arms keep ``.acquire.cta`` ordering, the same parity test and the same
    termination condition (spin until the phase flips; neither form has a real time limit, so ``timeout`` stays the hang
    detector).  Only the retry mechanics differ, and the difference is a PER-CALL-SITE choice, not a library default.

    MEASURED (host sm_107a trace-compiles at S=8K dense; perf on the Rubin node locked at 2376 MHz, A/B/A x3 with a control
    pair, CUPTI medians, TFLOPS ratios against develop): on sm_107a the default lowers per lane, ``SYNCS.PHASECHK.TRANS64.TRYWAIT
    P0 / @!P0 NANOSLEEP.SYNCS 0x1 / @!P0 SYNCS.PHASECHK / @!P0 BRA`` plus a ``BSSY`` reconverge (the warp is parked in hardware
    and woken by the phase event), while ``spin=True`` lowers to the uniform ``USYNCS.PHASECHK.TRANS64.TRYWAIT UP0 /
    BRA.U !UP0`` pair (two uniform instructions, no sleep; with every wait spun the d512 fp8 prefill goes 110 -> 8 divergent
    SYNCS.PHASECHK, 57 -> 6 NANOSLEEP, 57 -> 108 USYNCS.PHASECHK, 0 spills either way).  Spinning the per-KV-iteration RING
    waits gains +4.9 % on d128 bf16 @2K, +3.5 % on d512 fp8 @8K, +9.4 % on d192x128 mxfp8 @32K and +1.1..+4.3 % on d128
    mxfp8 / d192x128 f16 / d192x128 fp8 / d512 f16 (those four measured together with the predicated credit arrive), but
    LOSES -5.8 % @2K and -2.5 % @32K on d128 per-tensor fp8 and -2.2 % @32K on d512 mxfp8 (the loss follows the hint-less
    first check on the ring waits: keeping the sleeping retry after a hint-less first check, or spinning only the whole-tile
    idle waits, does not remove it), and spinning every wait is a loss on the linear-attention forward kernels (KDA -2.2 %
    @32K, GDP -5.0 % @8K).  On sm_100a ptxas lowers the hint-less form to ONE divergent ``SYNCS.PHASECHK`` plus a branch and
    emits no ``USYNCS.PHASECHK`` at all, so none of the sm_107a reasoning transfers and each sm100 kernel is its own
    measurement: the sm100 d128 f16 prefill opted its ring waits in (2026-10-05, +3..5 % dense / +1..3 % causal on both
    cc 10.0 and cc 10.3, S=2K..32K, cuDNN 9.28 control; its d64 flavor read mixed and stays sleeping), the other sm100
    kernels stay on the default.
    Hence the prefill kernels that gain opt their ring waits in through one module constant (``SPIN_RING_WAITS``);
    the waits a warp parks in for a whole tile (scheduler credits and CLC responses, the TMA-STG's O-ready wait,
    ``tmem_dealloc``, the end-of-kernel drains) and every other consumer keep the default.

    The inline-PTX labels are scoped to the ``{ }`` block, so the fixed names are legal at every instantiation.  No
    ``predicate=`` is passed (the DSL can lower one onto the last operand's VALUE, see ``arrive_expect_tx``); if this loop
    is ever predicated, branch around it instead.
    """
    if cutlass.const_expr(spin):
        nvvm.inline_ptx(
            "{\n\t.reg .pred P1;\n\tLAB_WAIT:\n\t"
            "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [{$r0}], {$r1};\n\t"
            "@P1 bra.uni DONE;\n\tbra.uni LAB_WAIT;\n\tDONE:\n\t}",
            read_only_args=[mb, cutlass.Int32(phase)],
        )
    else:
        while not nvvm.mbarrier_try_wait_parity(mb, phase, time_limit=WAIT_TIMEOUT):
            pass


# wait_poll shape: POLL_TIGHT_ITERS back-to-back test_wait's (~1 us), then a plain TIMER ``nanosleep.u32 POLL_SLEEP_NS``
# between tests.  The sleep is the timer form (SASS NANOSLEEP), NOT the event-sleep of a parked try_wait (NANOSLEEP.SYNCS) --
# the warp still never hands itself to the barrier unit.  MEASURED (d512 2x2 backward, B200 2026-10-01): a TIGHT test_wait
# loop on the MMA / TMA-LDG warp starves the compute warps that share its SMSP -- stage 2 ran 2.2x SLOWER (94575 vs 39166 us
# dense 8K); the two-phase form removes that while keeping the no-park property.
POLL_TIGHT_ITERS: int = 32
POLL_SLEEP_NS: int = 128


def poll_ptx(tight_iters: int, sleep_ns: int) -> str:
    """The inline-PTX body of :func:`wait_poll` for one (tight_iters, sleep_ns) shape -- plain Python so a host test can pin
    the rendering.  ``sleep_ns == 0`` renders the pure tight loop (no ``nanosleep`` instruction); ``tight_iters`` must be >= 1.
    Plain ``bra`` (not ``bra.uni``): the test's predicate is per lane and may flip between lanes within one iteration."""
    tight_iters, sleep_ns = int(tight_iters), int(sleep_ns)
    if tight_iters < 1:
        raise ValueError(f"wait_poll: tight_iters must be >= 1, got {tight_iters}")
    if sleep_ns < 0:
        raise ValueError(f"wait_poll: sleep_ns must be >= 0, got {sleep_ns}")
    test = "mbarrier.test_wait.parity.acquire.cta.shared::cta.b64 P1, [{$r0}], {$r1};"
    if sleep_ns == 0:
        return "{\n\t.reg .pred P1;\n\tLAB_POLL:\n\t" + test + "\n\t@!P1 bra LAB_POLL;\n\t}"
    return (
        "{\n\t.reg .pred P1;\n\t.reg .u32 n;\n\tmov.u32 n, 0;\n\tLAB_TIGHT:\n\t"
        + test
        + "\n\t@P1 bra DONE;\n\tadd.u32 n, n, 1;\n\t"
        + f"setp.lt.u32 P1, n, {tight_iters};\n\t@P1 bra LAB_TIGHT;\n\tLAB_SLEEP:\n\tnanosleep.u32 {sleep_ns};\n\t"
        + test
        + "\n\t@!P1 bra LAB_SLEEP;\n\tDONE:\n\t}"
    )


@cute.jit
def wait_poll(mb, phase, tight_iters: cutlass.Constexpr[int] = POLL_TIGHT_ITERS, sleep_ns: cutlass.Constexpr[int] = POLL_SLEEP_NS):
    """NON-BLOCKING poll: ``mbarrier.test_wait.parity.acquire.cta`` in an inline-PTX loop until the phase parity differs from
    ``phase``.  Unlike both :func:`wait` arms (``try_wait`` with the ``time_limit`` hint -> ``NANOSLEEP.SYNCS`` parked on the
    barrier's wake event; the hint-less ``try_wait`` spin -> ``SYNCS.PHASECHK.TRANS64.TRYWAIT``, which also suspends), the
    waiting warp never hands itself to the barrier unit, so it cannot miss a wake-up.

    WHEN IT IS REQUIRED (measured, B200, 2026-10-01): a barrier whose phase is completed by an operation issued from a CTA
    OUTSIDE the waiter's cta_group::2 pair -- the other pair leader's ``tcgen05.commit ... multicast::cluster`` (a K/V ring
    shared by two pairs), a twin CTA's remote ``mbarrier.arrive`` -- loses the wake-up of a parked waiter under GPU
    time-slicing (another process launching concurrently): the d512 2x2 BACKWARD hung within 2-74 launches with every
    parking form (default hint, 10 ms hint, hint-less spin) while this ``test_wait`` poll ran 200/200 and the pair-local
    4x1 chain 30000/30000 (lane_d512_bprop/fix/HANDOFF2.md: heartbeat dump = the other three CTAs' copies of the same
    barrier had advanced 5 chunks, the parked follower never woke).  ``MBarrier(poll=True)`` opts a barrier in; the SDPA
    2x2 forward's detector is ``test_sdpa_fwd_d512_2x2_sm100.py::test_two_by_two_cross_pair_waits_poll`` (+ its contention
    hygiene run).  Pair-local barriers keep :func:`wait`.  The DSL's ``nvvm.mbarrier_test_wait`` wrapper raises a TypeError on
    4.7.0, hence the inline PTX; labels are block-scoped, so the fixed names are legal at every instantiation.

    SHAPE (``tight_iters`` back-to-back tests, then a plain TIMER ``nanosleep.u32 sleep_ns`` between tests; ``sleep_ns == 0`` =
    the pure tight loop): the defaults are the module constants POLL_TIGHT_ITERS / POLL_SLEEP_NS (32 / 128), the kernels pass
    their own measured point.  MEASURED (d512 2x2 BACKWARD stage 2, B200, dense 8K): the tight loop on the MMA / TMA-LDG warp
    starved the compute warps sharing its SMSP -- 94575 us (2.2x loss); 32 tight / 128 ns: twin 40531 vs 4x1 43186 us (+5.9 %);
    4 tight / 64 ns: 41426 vs 41696 (+0.6 %, the sleep latency dominates).  The d512 2x2 FORWARD lost only ~0.3 % to the tight
    loop (its MMA / TMA-LDG waits are short) and keeps 32 / 128.  The timer sleep is SASS NANOSLEEP, never the event-sleep
    NANOSLEEP.SYNCS, so the warp still never parks on the barrier."""
    nvvm.inline_ptx(poll_ptx(tight_iters, sleep_ns), read_only_args=[mb, cutlass.Int32(phase)])


@cute.jit
def arrive(mb):
    nvvm.mbarrier_arrive(mb)


@cute.jit
def arrive_expect_tx(mb, n_bytes, pred=None):
    if cutlass.const_expr(pred is None):
        nvvm.mbarrier_arrive_expect_tx(mb, n_bytes)
    else:
        # A branch round the native op, NOT nvvm.inline_ptx(predicate=...): the
        # DSL can lower that predicate to the last operand's *value* rather than
        # a predicate register, emitting PTX like
        #     @512 mbarrier.arrive.expect_tx.release.cta.shared.b64 %rd271, [%r202], 512;
        # which ptxas rejects as a syntax error, surfaced only as the generic
        # "NVVM backend compilation failed".  The two forms are equivalent here --
        # a thread whose predicate is false does not arrive either way.
        if pred:
            nvvm.mbarrier_arrive_expect_tx(mb, n_bytes)


@cute.jit
def cga_arrive():
    nvvm.barrier_cluster_arrive_relaxed_aligned()


@cute.jit
def cga_wait():
    nvvm.barrier_cluster_wait_aligned()


@cute.jit
def named_barrier_fence(barrier_id: cutlass.Constexpr[int], thread_count: cutlass.Constexpr[int]):
    """``bar.sync id, n`` as inline PTX, used as a SCHEDULING fence for a publish the issuing warps do not consume.

    Why inline PTX and not ``nvvm.barrier_cta_sync``: ptxas list-schedules a basic block by readiness and sinks a
    ``tcgen05.st`` + ``mbarrier.arrive`` pair that has no consumer in this warp below every ready MUFU / FMA of the
    block (the sm100 d128 f16 softmax published alpha at 84 % and its first P chunk at 97 % of the body, so the
    correction and the MMA waited on them for nothing).  A barrier is a memory-ordering point ptxas keeps such
    memory ops ahead of; the intrinsic form drifts with the surrounding arithmetic, the asm form stays put.  It
    costs the ``n`` threads one barrier (they are the warps that just published, so they are already converged).
    Pure data-dependency pins measured worse: a ``mov`` asm is copy-propagated away by ptxas, and routing the
    arrive's mbarrier state token into the consumers stalls them for the arrive round trip.  Named-barrier ids are
    per kernel; keep them disjoint from the kernel's synchronisation barriers.
    """
    inline_ptx(f"bar.sync {barrier_id}, {thread_count};", write_only_types=[], read_only_args=[])


@cute.jit
def wait_on_dependent_grids():
    inline_ptx(
        "griddepcontrol.wait;",
        write_only_types=[],
        read_only_args=[],
    )


@cute.jit
def launch_dependent_grids():
    inline_ptx(
        "griddepcontrol.launch_dependents;",
        write_only_types=[],
        read_only_args=[],
    )


@cute.jit
def mbar_arrive_on_peer(mb, peer_cta_id, pred=None):
    peer_mb = nvvm.mapa(mb, peer_cta_id)
    if cutlass.const_expr(pred is None):
        nvvm.mbarrier_arrive(peer_mb, scope=nvvm.MemScope.CLUSTER, relaxed=True)
    else:
        nvvm.inline_ptx(
            "mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _, [{$r0}];",
            read_only_args=[peer_mb.ir_value()],
            predicate=pred,
        )


@cute.jit
def arrive_on_leader(mb, leader_cta_id, cta_group: int):
    """RELAXED cluster-scope arrive on the leader's mbar: for data the arrive does NOT have to order -- an async-proxy TMEM
    write (``tcgen05_st``) already completed by ``tcgen05_wait(STORE)``, or a count-only credit.  The relaxed form is what
    keeps ptxas from draining (``MEMBAR.ALL.GPU`` + ``CGAERRBAR`` before every arrive).  For a
    lane-written SMEM operand the peer reads, use :func:`arrive_on_leader_release`."""
    if cutlass.const_expr(cta_group == 1):
        nvvm.mbarrier_arrive(mb)
    else:
        peer_mb = nvvm.mapa(mb, leader_cta_id)
        nvvm.mbarrier_arrive(peer_mb, scope=nvvm.MemScope.CLUSTER, relaxed=True)


@cute.jit
def arrive_on_leader_release(mb, leader_cta_id, cta_group: int):
    """RELEASE arrive on the leader's mbar -- ``mbarrier.arrive.release.cta.shared::cluster.b64`` on the mapa'd leader
    barrier (cga1: the plain local arrive, so the call site needs no ``CTA_MMA`` guard).

    WHAT IT ORDERS: this thread's GENERIC SMEM stores (a ``store_swizzled`` / ``st.shared`` of an MMA operand into the
    follower's own slab) that a ``fence_proxy("async.shared", space="cta")`` has made visible to the async proxy, before
    the leader's ``mbarrier.try_wait.parity.acquire`` returns and its ``cta_group::2`` MMA reads that slab in place.  The
    relaxed :func:`arrive_on_leader` cannot publish such stores -- the peer-issued MMA may read the slab stale (a
    load-dependent first-launch race); the SM100 dkdv MXFP8 chain ships exactly this pattern as its P ``producer_commit``
    (``cute.arch.mbarrier_arrive(mb, dst_rank)`` = the DSL's default remote arrive, ``.release`` at CTA scope), and the
    shared scheduler credit uses the same ``.release.cta.shared::cluster`` form at 0 ``CGAERRBAR``.

    WHY NOT ``.release.cluster``: a cluster-scope release on a per-iteration path makes ptxas emit ``MEMBAR.ALL.GPU`` +
    ``CGAERRBAR`` ahead of the arrive -- a kernel-wide drain per arrive site (removing that drain took a cga2 kernel
    from 47 % to 92 % of SOL).  The CTA-scope release across CTAs is the DSL's own "historical" default for a remote arrive
    -- formally WEAKER than cluster scope.  PINNED today (``test_tile_dsl_release_arrive.py``): the PTX form,
    ``CGAERRBAR == MEMBAR.ALL.GPU == 0`` in SASS, and a single-launch 2-CTA publish whose leader reads the follower's
    slab through a generic ``ld.shared::cluster``.  The async-proxy consumer this helper exists for -- a peer-issued
    ``cta_group::2`` MMA over a lane-written slab under load -- is PINNED by a fresh-process sweep on Rubin (cc 10.7,
    2026-09-29): 72 fresh processes x {relaxed, release.cta, nofence} x {0, 2000 ns follower nanosleep}, every cell
    12 / 0 / 0 exit 0 / 124 / other, 0 stale reads in 5120 fenced publishes, the arrive-BEFORE-store positive control
    374 / 384 stale.  Two facts from it: on sm_107a the relaxed and the ``.release.cta`` remote arrives lower to the SAME
    SASS (``USYNCS.ARRIVE.TRANS64.RED.ACT0``), so this form is FREE, and the ordering instruction on the path is the
    caller's ``fence_proxy`` (``MEMBAR.ALL.CTA`` + ``FENCE.VIEW.ASYNC.S``) -- keep the fence; 0 stale reads without it is
    margin on that shape, not a licence (PTX memory model).  First consumer:
    ``sdpa/bwd/kernels/sm107/bprop_d256_mxfp8.py`` (``mb_p_ready``).  Lane ledger: same as :func:`arrive_on_leader` (one
    arrive per calling lane -- nothing here elects)."""
    if cutlass.const_expr(cta_group == 1):
        nvvm.mbarrier_arrive(mb)
    else:
        peer_mb = nvvm.mapa(mb, leader_cta_id)
        nvvm.mbarrier_arrive(peer_mb, scope=nvvm.MemScope.CTA, relaxed=False)


@cute.jit
def commit_mma(mb, mcast_mask, cta_group: int, pred=None):
    if cutlass.const_expr(pred is None):
        if cutlass.const_expr(cta_group == 1):
            nvvm.tcgen05_commit(mb, group=nvvm.CTAGroup.CTA_1)
        else:
            nvvm.tcgen05_commit(
                mb,
                multicast_mask=mcast_mask,
                group=nvvm.CTAGroup.CTA_2,
            )
    else:
        # Branch round the native ops rather than nvvm.inline_ptx(predicate=...);
        # see arrive_expect_tx above.  The multicast form has a second reason: a
        # constant-folded mask reaches the asm operand as an immediate, and the
        # emitted "tcgen05.commit...multicast::cluster.b64 [%r661], 3;" is
        # rejected by ptxas ("Arguments mismatch") because ctaMask must be a
        # register.  That is the integer twin of the float hazard opaque_f32_zero
        # documents.  The native op takes the mask as a value and keeps it in a
        # register, so both problems go away.
        if pred:
            if cutlass.const_expr(cta_group == 1):
                nvvm.tcgen05_commit(mb, group=nvvm.CTAGroup.CTA_1)
            else:
                nvvm.tcgen05_commit(mb, multicast_mask=mcast_mask, group=nvvm.CTAGroup.CTA_2)


class Producer(enum.IntEnum):
    THREAD = 0
    TMA_LOAD = 1
    MMA_COMMIT = 2
    LEADER = 3
    # A LEADER-waited mbar whose arrive PUBLISHES the arriving lanes' generic SMEM stores (a lane-written MMA operand the
    # leader's MMA reads across the pair): the .release.cta cross-CTA arrive of arrive_on_leader_release.  LEADER stays the
    # relaxed form for tcgen05_st / TMEM data ordered by tcgen05_wait(STORE).  Appended (append-only enum).
    LEADER_RELEASE = 4


class Scope(enum.IntEnum):
    LOCAL = 0
    LEADER = 1


_IntOrCountFn = Union[int, Callable[[int], int]]


@dataclass(frozen=True)
class MBarrier:
    base_ptr: object
    stages: cutlass.Constexpr[int]
    init_count: cutlass.Constexpr[object]
    producer: cutlass.Constexpr[int] = int(Producer.THREAD)
    scope: cutlass.Constexpr[int] = int(Scope.LOCAL)
    try_wait: cutlass.Constexpr[bool] = False
    spin: cutlass.Constexpr[bool] = False
    # poll=True: every wait on this barrier is the non-blocking test_wait poll (wait_poll) -- REQUIRED for a barrier whose
    # phase an operation from OUTSIDE the waiter's cta_group::2 pair completes (lost wake-up under time-slicing).  poll_tight /
    # poll_sleep_ns = the poll shape handed to wait_poll (append-only; the defaults are the module constants, so every existing
    # construction renders as before).
    poll: cutlass.Constexpr[bool] = False
    poll_tight: cutlass.Constexpr[int] = POLL_TIGHT_ITERS
    poll_sleep_ns: cutlass.Constexpr[int] = POLL_SLEEP_NS
    stage_idx: object = 0

    def __getitem__(self, i):
        return replace(self, stage_idx=i)

    @property
    def smem_ptr(self):
        if isinstance(self.stage_idx, int) and self.stage_idx == 0:
            return self.base_ptr
        return self.base_ptr.subview(self.stage_idx)

    def init(self, override_count=None):
        if override_count is not None:
            count = override_count
        elif isinstance(self.init_count, (tuple, list)):
            if not isinstance(self.stage_idx, int):
                raise TypeError("MBarrier with tuple init_count requires a Python-int " f"stage_idx (via [py_int]); got " f"{type(self.stage_idx).__name__}.")
            count = int(self.init_count[self.stage_idx])
        else:
            count = int(self.init_count)
        nvvm.mbarrier_init(self.smem_ptr, count)

    def wait(self, phase, spin: bool = False):
        # spin=True opts THIS call site into the hint-less uniform spin (see wait()); a barrier declared with spin=True opts
        # every wait on it in; the default is the sleeping form.  A barrier declared with try_wait=True waits through the
        # DSL wrapper's untimed try_wait loop instead (the linear attention kernels: one SYNCS.PHASECHK + branch on sm100,
        # no inline PTX, so their cubins are unchanged).
        if cutlass.const_expr(self.poll):
            wait_poll(self.smem_ptr, phase, tight_iters=self.poll_tight, sleep_ns=self.poll_sleep_ns)
        elif cutlass.const_expr(self.try_wait):
            wait_try(self.smem_ptr, phase)
        else:
            wait(self.smem_ptr, phase, spin=spin or self.spin)

    def arrive(
        self,
        *,
        n_bytes=None,
        mcast_mask=None,
        cta_group=None,
        leader_cta_id=None,
        pred=None,
    ):
        if cutlass.const_expr(self.producer == int(Producer.THREAD)):
            if cutlass.const_expr(pred is not None):
                raise TypeError(
                    "MBarrier(producer=THREAD).arrive() does NOT support pred= "
                    "(silently over-arrives). Keep the `if nvvm.elect_sync():` "
                    "branch around the plain arrive; only TMA_LOAD (expect_tx) "
                    "and MMA_COMMIT (commit) arrives have a predicated path."
                )
            arrive(self.smem_ptr)

        elif cutlass.const_expr(self.producer == int(Producer.TMA_LOAD)):
            if n_bytes is None:
                raise TypeError("MBarrier(producer=TMA_LOAD).arrive() requires n_bytes=")
            arrive_expect_tx(self.smem_ptr, n_bytes, pred=pred)

        elif cutlass.const_expr(self.producer == int(Producer.MMA_COMMIT)):
            if cta_group is None:
                raise TypeError("MBarrier(producer=MMA_COMMIT).arrive() requires " "cta_group= (Python compile-time int).")
            commit_mma(self.smem_ptr, mcast_mask, cta_group, pred=pred)

        elif cutlass.const_expr(self.producer == int(Producer.LEADER_RELEASE)):
            if cta_group is None or leader_cta_id is None:
                raise TypeError("MBarrier(producer=LEADER_RELEASE).arrive() requires " "cta_group= AND leader_cta_id=.")
            if cutlass.const_expr(pred is not None):
                raise TypeError(
                    "MBarrier(producer=LEADER_RELEASE).arrive() does NOT support pred=. "
                    "Use arrive_on_peer(pred=) for a predicated cross-CTA arrive, "
                    "or keep the `if elect_sync():` branch."
                )
            arrive_on_leader_release(self.smem_ptr, leader_cta_id, cta_group)

        else:
            if cta_group is None or leader_cta_id is None:
                raise TypeError("MBarrier(producer=LEADER).arrive() requires " "cta_group= AND leader_cta_id=.")
            if cutlass.const_expr(pred is not None):
                raise TypeError(
                    "MBarrier(producer=LEADER).arrive() does NOT support pred=. "
                    "Use arrive_on_peer(pred=) for a predicated cross-CTA arrive, "
                    "or keep the `if elect_sync():` branch."
                )
            arrive_on_leader(self.smem_ptr, leader_cta_id, cta_group)

    def arrive_on_peer(self, peer_cta_id, pred=None):
        mbar_arrive_on_peer(self.smem_ptr, peer_cta_id, pred=pred)
