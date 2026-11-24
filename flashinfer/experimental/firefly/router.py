"""Routing for the per-token MoE: the caller's pre-routed expert ids and weights (FlashInfer's precomputed routing).

The ids [M, topk] Int32 and weights [M, topk] (FP32, BF16 or FP16: ``win_dtype``, converted to FP32 on load) are the
caller's (at ``ids_ptr`` / ``win_ptr``) and are only read. Ids outside [0, E) are skipped, as FlashInfer's TRT-LLM
routing skips ids outside the rank's experts (CUDA-graph padding uses -1): not counted, not ranked, ``inverse`` = -1,
not listed in eroute. The caller's weights are copied to ``w_ptr`` with the skipped routes' set to 0 (they may hold
anything, e.g. NaN), which the later kernels read. Block 0 also resets the grouped row count.

Short batches rank the routes (``ranks``; one token per CTA, one warp): lane s < topk adds 1 to
``ecount[expert of slot s]`` with a relaxed gpu-scope atomic and stores the returned rank (the route's
position among its expert's routes, in atomic arrival order) to ``inverse[route]``; ``scatter.Scatter``
turns ranks into rows. Without a Scatter (``eroute``, block_m <= 32) the lane also lists the route at
``eroute[expert * S + rank]`` (S: a static stride >= M): the gate/up places the tiles from ``ecount`` and gathers the
rows through it. With ``zero_out`` = H (the fused finalize without a Scatter: the gate/up's gather warps zero the
output row of each token's slot-0 route) a token whose slot 0 is skipped has its output row zeroed here. With PDL,
``griddepcontrol.wait`` then ``launch_dependents`` first.

Long batches (``partial``; 16 tokens / 4 warps per CTA): the CTA's routes are counted per expert into
``Partial[chunk, :]`` (shared-memory atomics of 1 per route), which ``placement.ScanCounts`` turns into row offsets.
No inline PTX.
"""

import cutlass
import cutlass.cute as cute
import cutlass.memory
from cutlass import BFloat16, Float16, Float32, Int32, Int64

from .gemm import use_pdl
from .scatter import zero_row

GMEM = cute.AddressSpace.gmem
NEG_INF = float("-inf")
INT_MAX = 0x7FFFFFFF


ROUTE_WEIGHT_DTYPES = (
    Float32,
    BFloat16,
    Float16,
)  # the caller's topk_weights (FlashInfer's cute_dsl takes all three)


@cute.jit
def load_route_weight(win_ptr: cute.Pointer, i: Int64, dtype: cutlass.Constexpr):
    """The caller's routing weight ``i`` (FP32, BF16 or FP16 at win_ptr) as FP32 (exact)."""
    if cutlass.const_expr(dtype is Float32):
        return cute.arch.load(
            cute.make_ptr(Float32, win_ptr.toint() + i * 4, GMEM, assumed_align=4),
            Float32,
        )
    else:  # (cute.arch.load's nvvm.load_ext takes no 16-bit floats: a one-element tensor's load, then extf)
        w = cute.make_tensor(
            cute.make_ptr(dtype, win_ptr.toint() + i * 2, GMEM, assumed_align=2),
            cute.make_layout(1),
        )
        return Float32(w[0])


class SelectExperts:
    """The caller's routing of ``rows_per_cta`` tokens per CTA (ranks or per-chunk counts).

    Args:
        num_experts: The (local) experts E.
        topk: Experts per token.
        rows_per_cta: Tokens per CTA (1 short batches, 16 long batches).
        num_warps: Warps per CTA (1, 4); warp w handles rows w, w + num_warps, ...
        partial: Count each expert's routes of the CTA's chunk into ``Partial`` (long batches).
        launch_pdl: ``griddepcontrol.wait`` / ``launch_dependents`` first.
        ranks: Short batches (no placement kernel): rank each route within its expert with an atomic on
            ``ecount`` (passed as ``partial_ptr``, zero on entry) into ``inverse``.
        eroute: With ``ranks``, the stride S > 0 at which each route (token * topk + slot) is also listed at
            ``eroute[expert * S + rank]`` (short batches without Scatter; S >= the tokens; the gate/up's
            ``GroupedGemm(tokens=)``); 0: none.
        zero_out: With ranks, H: zero the BF16 output row [H] (y_ptr) of a token whose slot-0 id is
            skipped.
        base: The first expert of this rank (expert parallelism, FlashInfer's local_expert_offset): the
            caller's ids are global, the local expert is id - base, skipped outside [0, E) (E: the local experts).
        win_dtype: The caller's weights' type (``ROUTE_WEIGHT_DTYPES``), converted to FP32.
    """

    def __init__(
        self,
        num_experts,
        topk,
        rows_per_cta,
        num_warps,
        *,
        partial,
        launch_pdl,
        ranks=False,
        eroute=0,
        zero_out=0,
        base=0,
        win_dtype=Float32,
    ):
        # topk: the routes' slots (global: under expert parallelism a token may route to more experts than this rank
        # holds; ids outside the rank's experts are skipped), 1 <= topk <= min(all experts, 32) (moe.check_support)
        assert rows_per_cta % num_warps == 0 and 1 <= topk <= 32
        assert not ranks or (
            rows_per_cta == 1 and num_warps == 1 and topk <= 32 and not partial
        )
        self.E, self.topk, self.rows, self.warps = (
            num_experts,
            topk,
            rows_per_cta,
            num_warps,
        )
        self.partial, self.launch_pdl, self.ranks = partial, launch_pdl, ranks
        assert not eroute or ranks
        self.eroute = eroute
        assert not zero_out or ranks
        self.zero_out = zero_out
        self.base = base
        assert win_dtype in ROUTE_WEIGHT_DTYPES
        self.win_dtype = win_dtype

    @cute.jit
    def __call__(
        self,
        ids_ptr: cute.Pointer,
        w_ptr: cute.Pointer,
        count_ptr: cute.Pointer,
        partial_ptr: cute.Pointer,
        inv_ptr: cute.Pointer,
        eroute_ptr: cute.Pointer,
        y_ptr: cute.Pointer,
        win_ptr: cute.Pointer,
        tokens: Int32,
        stream,
    ):
        @cute.struct
        class Storage:
            hist: cute.struct.MemRange[Int32, self.E]

        self.storage = Storage
        # launched for the short batches (ranks, PDL) and the long batches (partial)
        assert self.ranks or self.partial, (
            "SelectExperts launches rank (short batches) or count chunks (long batches)"
        )
        grid = (tokens + self.rows - 1) // self.rows
        self.kernel(
            ids_ptr,
            w_ptr,
            count_ptr,
            partial_ptr,
            inv_ptr,
            eroute_ptr,
            y_ptr,
            win_ptr,
            tokens,
        ).launch(
            grid=[grid, 1, 1],
            block=[32 * self.warps, 1, 1],
            stream=stream,
            use_pdl=use_pdl(self),
        )

    @cute.kernel
    def kernel(
        self,
        ids_ptr: cute.Pointer,
        w_ptr: cute.Pointer,
        count_ptr: cute.Pointer,
        partial_ptr: cute.Pointer,
        inv_ptr: cute.Pointer,
        eroute_ptr: cute.Pointer,
        y_ptr: cute.Pointer,
        win_ptr: cute.Pointer,
        tokens: Int32,
    ):
        E, T, R = self.E, self.topk, self.rows
        cta, _, _ = cute.arch.block_idx()
        t, _, _ = cute.arch.thread_idx()
        lane = t % 32
        if cutlass.const_expr(self.launch_pdl):
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()
        if cta == 0:
            if t == 0:
                cute.make_tensor(
                    cute.make_ptr(Int32, count_ptr.toint(), GMEM, assumed_align=4),
                    cute.make_layout(1),
                )[0] = Int32(0)
        ids = cute.make_tensor(
            cute.make_ptr(Int32, ids_ptr.toint(), GMEM, assumed_align=4),
            cute.make_layout(tokens * T),
        )
        wts = cute.make_tensor(
            cute.make_ptr(Float32, w_ptr.toint(), GMEM, assumed_align=4),
            cute.make_layout(tokens * T),
        )
        if cutlass.const_expr(R == 1 and self.ranks):
            mine = Int32(0)
            mine = ids[
                cta * T + cutlass.min(lane, T - 1)
            ]  # lane s < topk: the caller's expert of slot s
            if cutlass.const_expr(self.base):
                mine = mine - self.base
            ok = (mine >= 0) & (mine < E)  # ids outside [0, E) are skipped
            if lane < T:
                wv = load_route_weight(win_ptr, Int64(cta * T + lane), self.win_dtype)
                wts[cta * T + lane] = Float32(cutlass.select_(ok, wv, Float32(0.0)))
                invp = cute.make_ptr(
                    Int32,
                    inv_ptr.toint() + Int64(cta * T + lane) * 4,
                    GMEM,
                    assumed_align=4,
                )
                if ok:
                    rank = cute.arch.atomic_add(
                        cute.make_ptr(
                            Int32,
                            partial_ptr.toint() + Int64(mine) * 4,
                            GMEM,
                            assumed_align=4,
                        ),
                        Int32(1),
                        sem="relaxed",
                        scope="gpu",
                    )
                    cute.arch.store(invp, rank)
                    if cutlass.const_expr(self.eroute):
                        cute.arch.store(
                            cute.make_ptr(
                                Int32,
                                eroute_ptr.toint()
                                + (Int64(mine) * self.eroute + rank) * 4,
                                GMEM,
                                assumed_align=4,
                            ),
                            Int32(cta * T + lane),
                        )
                else:
                    cute.arch.store(invp, Int32(-1))
            if cutlass.const_expr(self.zero_out):
                ok0 = cute.arch.shuffle_sync(Int32(ok), 0)
                if (
                    ok0 == 0
                ):  # slot 0 skipped: no gather warp zeroes this token's output row
                    zero_row(y_ptr, cta, lane, self.zero_out, 32)
        if cutlass.const_expr(self.partial):
            st = cutlass.memory.SmemAllocator().allocate(self.storage)
            hist = st.hist.get_tensor(cute.make_layout(E))
            NTH = (
                32 * self.warps
            )  # E = 128 and R T = 128 (4 warps, 16 tokens of 8 routes): one each per thread
            for i in cutlass.range_constexpr((E + NTH - 1) // NTH):
                if cutlass.const_expr(E % NTH == 0):  # noqa: SIM114 (compile-time branch; the elif is a runtime check)
                    hist[t + i * NTH] = Int32(0)
                elif t + i * NTH < E:
                    hist[t + i * NTH] = Int32(0)
            cute.arch.barrier()  # also orders this CTA's id stores before the reads below
            for i in cutlass.range_constexpr((R * T + NTH - 1) // NTH):
                route = cta * (R * T) + t + i * NTH
                inr = route < tokens * T
                if cutlass.const_expr((R * T) % NTH != 0):
                    inr = inr & (t + i * NTH < R * T)
                if inr:  # (names not used elsewhere in the kernel: a dynamic branch's locals)
                    e_r = ids[route]
                    if cutlass.const_expr(self.base):
                        e_r = e_r - self.base
                    w_r = load_route_weight(win_ptr, Int64(route), self.win_dtype)
                    wts[route] = Float32(
                        cutlass.select_((e_r >= 0) & (e_r < E), w_r, Float32(0.0))
                    )
                    if (e_r >= 0) & (e_r < E):
                        cute.arch.atomic_add(
                            hist.iterator + e_r,
                            Int32(1),
                            sem="relaxed",
                            scope="cta",
                        )
            cute.arch.barrier()
            pt = cute.make_tensor(
                cute.make_ptr(Int32, partial_ptr.toint(), GMEM, assumed_align=4),
                cute.make_layout(1 << 30),
            )
            for i in cutlass.range_constexpr((E + NTH - 1) // NTH):
                if cutlass.const_expr(E % NTH == 0):  # noqa: SIM114 (compile-time branch; the elif is a runtime check)
                    pt[cta * E + t + i * NTH] = hist[t + i * NTH]
                elif t + i * NTH < E:
                    pt[cta * E + t + i * NTH] = hist[t + i * NTH]
