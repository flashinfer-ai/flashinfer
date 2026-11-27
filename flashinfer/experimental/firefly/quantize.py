"""``RouteInput``: the direct path's (M <= 4) input copy and routing weights (pre-routed ids and weights). The direct
path's intermediate is quantized by its GEMMs (``direct.py``)."""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64

from .gemm import use_pdl
from .router import ROUTE_WEIGHT_DTYPES, load_route_weight
from .scatter import load_row, st_u32, st_v4, store_chunks, zero_row

GMEM = cute.AddressSpace.gmem


class RouteInput:
    """Direct path (M <= 4): one warp per token copies the token's pre-quantized row to the direct GEMMs' input
    buffers (FP4 row [M, K / 2], block scales packed per 1-row tile in the 128x4 layout, row scale) and copies the
    token's routing weights for the down GEMM and the reduction.

    PDL: the dependents are triggered after the wait (the direct gate/up GEMM reads the caller's ids before its own
    wait). ``fused`` (``CuteMoE(use_fused_finalize=True)``): also zero the token's output row (BF16 [count, K] at
    y_ptr; otherwise unused), into which the direct down GEMM adds the routes.
    The ids are the caller's (ids_ptr); the caller's weights (win_ptr; FP32, BF16 or FP16: ``win_dtype``) are copied
    to w_ptr as FP32 with the routes of ids outside [0, E) set to 0 (skipped, as FlashInfer's TRT-LLM routing skips
    ids outside the rank's experts). ``row_scales`` False
    (per-tensor input, ``moe.CuteMoE(per_token_scale=False)``): no row scale is copied (the direct gate/up's
    ``unit_a_gs``)."""

    def __init__(
        self,
        k,
        *,
        num_experts=128,
        topk=8,
        fused=False,
        base=0,
        win_dtype=Float32,
        row_scales=True,
    ):
        self.row_scales = row_scales
        self.base = base  # expert parallelism: local expert = id - base (num_experts: the local ones)
        assert k % 64 == 0
        assert win_dtype in ROUTE_WEIGHT_DTYPES
        self.win_dtype = win_dtype
        self.k, self.launch_pdl = k, True
        self.E, self.topk = num_experts, topk
        self.fused = fused

    @cute.jit
    def __call__(
        self,
        q_in: cute.Pointer,
        sf_in: cute.Pointer,
        gs_in: cute.Pointer,
        q_ptr: cute.Pointer,
        sf_ptr: cute.Pointer,
        gs_ptr: cute.Pointer,
        ids_ptr: cute.Pointer,
        w_ptr: cute.Pointer,
        y_ptr: cute.Pointer,
        win_ptr: cute.Pointer,
        count: Int32,
        stream,
    ):
        self.kernel(
            q_in,
            sf_in,
            gs_in,
            q_ptr,
            sf_ptr,
            gs_ptr,
            ids_ptr,
            w_ptr,
            y_ptr,
            win_ptr,
            count,
        ).launch(
            grid=[count, 1, 1], block=[32, 1, 1], stream=stream, use_pdl=use_pdl(self)
        )

    @cute.kernel
    def kernel(
        self,
        q_in: cute.Pointer,
        sf_in: cute.Pointer,
        gs_in: cute.Pointer,
        q_ptr: cute.Pointer,
        sf_ptr: cute.Pointer,
        gs_ptr: cute.Pointer,
        ids_ptr: cute.Pointer,
        w_ptr: cute.Pointer,
        y_ptr: cute.Pointer,
        win_ptr: cute.Pointer,
        count: Int32,
    ):
        K = self.k
        NQ, NS = K // 32, K // 64
        pid, _, _ = cute.arch.block_idx()
        lane, _, _ = cute.arch.thread_idx()
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        qv, sw, gs = load_row(
            q_in, sf_in, gs_in, pid, lane, K, 32, row_scale=self.row_scales
        )
        store_chunks(st_v4, q_ptr.toint() + Int64(pid) * (K // 2), qv, lane, 32, NQ, 16)
        # 1-row tiles: word w (blocks 4 w .. 4 w + 3) at row * (K / 64 * 512) + w * 512
        store_chunks(
            st_u32, sf_ptr.toint() + Int64(pid) * (NS * 512), sw, lane, 32, NS, 512
        )
        if cutlass.const_expr(self.row_scales):
            cute.arch.store(
                cute.make_ptr(
                    Float32, gs_ptr.toint() + Int64(pid) * 4, GMEM, assumed_align=4
                ),
                gs,
            )
        if cutlass.const_expr(self.fused):
            zero_row(y_ptr, pid, lane, K, 32)
        ids = cute.make_tensor(
            cute.make_ptr(Int32, ids_ptr.toint(), GMEM, assumed_align=4),
            cute.make_layout(count * self.topk),
        )
        wts = cute.make_tensor(
            cute.make_ptr(Float32, w_ptr.toint(), GMEM, assumed_align=4),
            cute.make_layout(count * self.topk),
        )
        T = self.topk
        if lane < T:
            e = ids[pid * T + lane]
            if cutlass.const_expr(self.base):
                e = e - self.base
            wv = load_route_weight(win_ptr, Int64(pid * T + lane), self.win_dtype)
            wts[pid * T + lane] = Float32(
                cutlass.select_((e >= 0) & (e < self.E), wv, Float32(0.0))
            )
