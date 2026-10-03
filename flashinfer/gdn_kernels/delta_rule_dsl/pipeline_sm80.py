"""Stage bookkeeping for the sm_80 delta-rule kernel.

CuTe-DSL exposes its async pipelines from sm_90 up: ``PipelineCpAsync`` reads
like the thing this file replaces, but it builds on ``mbarrier``, and the DSL
will not emit a usable one for an sm_80 target. Measured against the compiler,
one operation at a time:

    mbarrier_init                     compiles and runs
    mbarrier_arrive                   NVVM backend rejects it
    mbarrier_test_wait / wait /
    try_wait / cp_async_mbarrier_
    arrive_noinc                      CONFIG_UNSUPPORTED_ARCH: 'sm_80'

This is a property of the DSL, not of the architecture. The PTX ISA has
``mbarrier.init``, ``mbarrier.arrive``, ``mbarrier.test_wait`` and
``cp.async.mbarrier.arrive`` from sm_80 on; only ``try_wait`` starts at sm_90.
Reaching them from here would mean writing the barrier protocol in inline PTX,
which is a larger piece of work than this file, and is the shape a later
warp-specialized variant would take.

What is left without it is the group counter -- ``cp.async.commit_group`` and
``cp.async.wait_group`` -- plus ordinary barriers, and that decides the
kernel's shape. An mbarrier tracks each stage separately, so one warp can fill
stage 3 while another drains stage 1; that is what lets the sm_90 kernel
dedicate a warp group to loading. The group counter is per-thread and retires
in issue order, so a warp cannot wait on a stage some other warp filled -- a
thread that issued no copies has nothing to wait for and walks straight
through. Through this interface a load warp and a math warp cannot hand work to
each other.

So the kernel drops the producer/consumer split: every thread issues its share
of a tile, every thread waits for it, and a block barrier is what makes the
tile visible to all of them. That also removes the hazard the split brought
with it -- a block-wide barrier reached by only some of the block never
completes, and with the roles gone there is no path that reaches one alone.

The two math warp groups still order themselves with named barriers, which
sm_80 has, so the class keeps the interface the call sites already use:

    producer_acquire(state)   wait until the stage is free to overwrite
    producer_commit(state)    close the async group that filled it
    consumer_wait(state)      wait until the stage holds this block's data
    consumer_release(state)   mark it reusable

``producer_get_barrier`` is deliberately absent: it hands out an mbarrier for a
TMA copy to signal, and there is no such thing here.
"""

from dataclasses import dataclass

import cutlass.cute as cute


@dataclass(frozen=True)
class PipelineCpAsyncSm80:
    """Waits on cp.async groups, with block barriers for ordering.

    ``num_stages`` is carried so a caller reading ``.num_stages`` behaves the
    same as with the mbarrier pipelines, but the waiting is not per stage.
    ``prefetch`` is how many blocks the loads run ahead of the math, and it is
    what ``consumer_wait`` leaves outstanding: with none, the wait drains
    everything and the current block's tiles have landed.

    A prefetch deeper than the smallest stage count would overwrite a buffer
    still being read, so the caller sets it from that minimum, not per pipeline.
    """

    num_stages: int
    prefetch: int = 0

    @cute.jit
    def producer_acquire(self, state):
        """Wait until this stage's previous contents are done being read.

        ``consumer_release`` already ended with a block barrier, and every
        thread that loads is a thread that read, so there is nothing further to
        wait on. Kept for interface parity with the mbarrier pipelines.
        """
        pass

    @cute.jit
    def producer_commit(self, state):
        """Close nothing here.

        ``load_qkv_cpasync`` commits a group per tensor as it issues, so by the
        time a pipeline would commit, its own copies are already in a group.
        Committing again would only add an empty one.

        Those three groups per block are why ``prefetch`` is a block count and
        not a group count: a look-ahead schedule would have to make the mapping
        exact -- one group per block, or a wait that counts groups per block --
        before it could leave a block's worth outstanding.
        """
        pass

    @cute.jit
    def consumer_wait(self, state):
        """Wait until this block's tiles have landed, then publish them.

        ``wait_group`` retires groups in issue order, so leaving ``prefetch``
        of them outstanding waits for exactly the block this state points at.
        The barrier is what carries one thread's copies to the rest.
        """
        cute.arch.cp_async_wait_group(self.prefetch)
        cute.arch.barrier()

    @cute.jit
    def consumer_release(self, state):
        """Mark the stage reusable.

        The barrier is what tells every thread that the readers are done, so
        the next load into this stage cannot run ahead of them.
        """
        cute.arch.barrier()


__all__ = ["PipelineCpAsyncSm80"]
