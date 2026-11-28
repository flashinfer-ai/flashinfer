"""Route placement for the per-token MoE: ``ScanCounts`` (long batches), and the L2 prefetch helper of the grouped
GEMMs.

``ScanCounts``: one CTA (4 warps) per expert turns ``Partial[chunk, expert]`` route counts into
exclusive offsets in place and appends the expert's total at ``Partial[chunks, expert]``. Thread t
takes chunks 4 t .. 4 t + 3 of each 512-chunk step: thread-local inclusive scan, warp scan with
``shfl.sync.up``, warp totals through shared memory, plus the running total of earlier steps.
Integer results do not depend on the reduction order. No inline PTX.

Short batches have no placement kernel: ``SelectExperts`` ranks each route within its expert, and the ``Scatter``
(block_m 64) or the gate/up GEMM itself (block_m <= 32, ``GroupedGemm(place_tiles)``) turns the ranks into rows.
"""

import cutlass
import cutlass.cute as cute
import cutlass.memory
from cutlass import Int32, Int64
from cutlass._mlir.dialects import llvm, nvvm

GMEM = cute.AddressSpace.gmem


def l2_prefetch(addr, size: int):
    """``cp.async.bulk.prefetch.L2.global [addr], size`` (NVVM op): ``size`` bytes (a multiple of 16) from global
    memory into L2. Goes through the SM's TMA unit (queued with its TMA loads) and holds the kernel's completion
    until done."""
    ptr = llvm.inttoptr(llvm.PointerType.get(1), Int64(addr).ir_value())
    nvvm.cp_async_bulk_prefetch(ptr, Int32(size).ir_value())


def _gtensor(ptr, n=1 << 30):
    return cute.make_tensor(
        cute.make_ptr(Int32, ptr.toint(), GMEM, assumed_align=4), cute.make_layout(n)
    )


@cute.jit
def _block_scan(v, lane, warp, wsum, nwarps: cutlass.Constexpr):
    """Inclusive scan of one value per thread over the CTA in thread order, and the CTA total.
    ``wsum``: shared [nwarps] scratch; callers separate consecutive uses with a barrier."""
    for d in cutlass.range_constexpr(5):
        u = cute.arch.shuffle_sync_up(v, offset=1 << d)
        v = v + Int32(cutlass.select_(lane >= (1 << d), u, Int32(0)))
    if lane == 31:
        wsum[warp] = v
    cute.arch.barrier()
    before = Int32(0)
    total = Int32(0)
    for w in cutlass.range_constexpr(nwarps):
        x = wsum[w]
        before = before + Int32(cutlass.select_(w < warp, x, Int32(0)))
        total = total + x
    return v + before, total


class ScanCounts:
    """Per-expert exclusive offsets of the per-chunk route counts."""

    def __init__(self, num_experts):
        self.E, self.warps = num_experts, 4  # the long batches: no PDL

    @cute.jit
    def __call__(self, partial_ptr: cute.Pointer, chunks: Int32, stream):
        @cute.struct
        class Storage:
            wsum: cute.struct.MemRange[Int32, self.warps]

        self.storage = Storage
        self.kernel(partial_ptr, chunks).launch(
            grid=[self.E, 1, 1], block=[32 * self.warps, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(self, partial_ptr: cute.Pointer, chunks: Int32):
        E, NW = self.E, self.warps
        STEP = 128 * NW
        expert, _, _ = cute.arch.block_idx()
        t, _, _ = cute.arch.thread_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = t % 32
        st = cutlass.memory.SmemAllocator().allocate(self.storage)
        wsum = st.wsum.get_tensor(cute.make_layout(NW))
        part = _gtensor(partial_ptr)
        running = Int32(0)
        start = Int32(0)
        while start < chunks:
            c0 = start + 4 * t
            p = []
            for j in cutlass.range_constexpr(4):
                v = Int32(0)
                if c0 + j < chunks:
                    v = part[(c0 + j) * E + expert]
                p.append(v)
            incl = [p[0]]
            for j in cutlass.range_constexpr(1, 4):
                incl.append(incl[j - 1] + p[j])
            cute.arch.barrier()  # wsum of the previous step has been read
            scan, total = _block_scan(incl[3], lane, warp, wsum, NW)
            before = running + scan - incl[3]
            for j in cutlass.range_constexpr(4):
                if c0 + j < chunks:
                    part[(c0 + j) * E + expert] = before + incl[j] - p[j]
            running = running + total
            start = start + STEP
        if t == 0:
            part[chunks * E + expert] = running
