"""Kimi-local route staging; retain ID width until after bounds checks."""

import cuda.bindings.driver as cuda
import cutlass.cute as cute
from cutlass import Float32, Int32


class StageRoutes:
    def __init__(self, experts, topk):
        self.experts, self.topk = experts, topk

    @cute.jit
    def __call__(
        self,
        ids: cute.Tensor,
        scores: cute.Tensor,
        ids_out: cute.Tensor,
        scores_out: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(ids, scores, ids_out, scores_out).launch(
            grid=((ids_out.shape[0] * self.topk + 255) // 256, 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        ids: cute.Tensor,
        scores: cute.Tensor,
        ids_out: cute.Tensor,
        scores_out: cute.Tensor,
    ):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        index = bid * 256 + tid
        if index < ids_out.shape[0] * self.topk:
            row, col = index // self.topk, index % self.topk
            expert, score = Int32(-1), Float32(0)
            if row < ids.shape[0]:
                raw = ids[row, col]
                if raw >= 0 and raw < self.experts:
                    expert = Int32(raw)
                    score = Float32(scores[row, col])
            ids_out[row, col] = expert
            scores_out[row, col] = score
