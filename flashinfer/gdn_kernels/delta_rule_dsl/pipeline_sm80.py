"""cp.async stage bookkeeping for the sm_80 delta-rule kernel.

The DSL emits no usable mbarrier for sm_80, so every thread waits on its own
cp.async groups and a block barrier publishes the tile; the interface matches
the mbarrier pipelines the call sites already use.
"""

from dataclasses import dataclass

import cutlass.cute as cute


@dataclass(frozen=True)
class PipelineCpAsyncSm80:
    """Waits on cp.async groups, with block barriers for ordering.

    ``prefetch`` is how many blocks the loads run ahead of the math; it must not
    exceed the smallest stage count.
    """

    num_stages: int
    prefetch: int = 0

    @cute.jit
    def producer_acquire(self, state):
        """No-op: ``consumer_release`` already ended with a block barrier."""
        pass

    @cute.jit
    def producer_commit(self, state):
        """No-op: ``load_qkv_cpasync`` commits a group per tensor as it issues."""
        pass

    @cute.jit
    def consumer_wait(self, state):
        """Wait for this block's tiles, leaving ``prefetch`` blocks in flight."""
        cute.arch.cp_async_wait_group(self.prefetch)
        cute.arch.barrier()

    @cute.jit
    def consumer_release(self, state):
        """Mark the stage reusable."""
        cute.arch.barrier()


__all__ = ["PipelineCpAsyncSm80"]
