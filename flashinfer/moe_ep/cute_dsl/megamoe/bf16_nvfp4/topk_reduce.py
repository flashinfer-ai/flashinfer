# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Ordered BF16/FP32 top-k reduction inside the SM100 W4A16 MegaMoE kernel."""

from typing import Optional

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Float32, Int32


class Bf16TopkReduce:
    """Reduce eight hidden elements in FP32, optionally applying routing scores."""

    def __init__(self, hidden: int, num_topk: int) -> None:
        self.hidden = int(hidden)
        self.num_topk = int(num_topk)
        self.hidden_per_thread = 8
        self.hidden_tiles = self.hidden // self.hidden_per_thread
        self.prefetch = self.num_topk <= 16

    @cute.jit
    def _mark_alignment(self, tensor: cute.Tensor, align_bytes: int) -> cute.Tensor:
        p = tensor.iterator
        return cute.make_tensor(
            cute.make_ptr(p.dtype, p.toint(), p.memspace, assumed_align=align_bytes),
            tensor.layout,
        )

    @cute.jit
    def _prepare_bf16_worker(
        self,
        topk_score: Optional[cute.Tensor],
        score_reg: Optional[cute.Tensor],
        worker_idx: Int32,
    ):
        token_idx = worker_idx // self.hidden_tiles
        hidden_tile_idx = worker_idx % self.hidden_tiles
        if cutlass.const_expr(topk_score is not None and self.prefetch):
            cute.autovec_copy(topk_score[token_idx, None], score_reg)
        return token_idx, hidden_tile_idx

    @cute.jit
    def _reduce_bf16_worker(
        self,
        combine_output: cute.Tensor,
        topk_score: Optional[cute.Tensor],
        reduced_output: cute.Tensor,
        token_idx: Int32,
        hidden_tile_idx: Int32,
        score_reg: Optional[cute.Tensor],
    ):
        hidden_per_thread = self.hidden_per_thread
        num_topk: cutlass.Constexpr[int] = self.num_topk
        prefetch = self.prefetch
        out_dtype = reduced_output.element_type

        # The caller bounds each worker by rows * hidden_tiles before reduction.
        # (token, topk, hidden) -> (topk, hidden_per_thread)
        terms = cute.zipped_divide(
            combine_output[token_idx, None, None],
            (num_topk, hidden_per_thread),
        )[(None, None), (0, hidden_tile_idx)]
        # (token, hidden) -> (hidden_per_thread)
        dst = cute.zipped_divide(
            reduced_output[token_idx, None],
            (hidden_per_thread,),
        )[(None,), (hidden_tile_idx,)]

        load_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            cutlass.BFloat16,
            num_bits_per_copy=128,
        )
        acc = cute.make_rmem_tensor((hidden_per_thread,), cutlass.Float32)

        for k in cutlass.range_constexpr(0, num_topk, 1):
            term = cute.make_rmem_tensor(
                (hidden_per_thread,),
                cutlass.BFloat16,
            )
            cute.copy(load_atom, terms[k, None], term)
            if cutlass.const_expr(topk_score is not None):
                if cutlass.const_expr(not prefetch):
                    score_reg[k] = topk_score[token_idx, Int32(k)]
                score = Float32(score_reg[k])

            for i in cutlass.range_constexpr(0, hidden_per_thread, 2):
                value_pair = (Float32(term[i]), Float32(term[i + 1]))
                if cutlass.const_expr(topk_score is None):
                    if cutlass.const_expr(k == 0):
                        acc[i], acc[i + 1] = value_pair
                    else:
                        acc[i], acc[i + 1] = cute.arch.add_packed_f32x2(
                            (acc[i], acc[i + 1]), value_pair
                        )
                else:
                    if cutlass.const_expr(k != 0):
                        acc[i], acc[i + 1] = cute.arch.fma_packed_f32x2(
                            value_pair,
                            (score, score),
                            (acc[i], acc[i + 1]),
                        )
                    else:
                        acc[i], acc[i + 1] = cute.arch.mul_packed_f32x2(
                            value_pair,
                            (score, score),
                        )

        out = cute.make_rmem_tensor((hidden_per_thread,), out_dtype)
        out.store(acc.load().to(out_dtype))
        cute.copy(
            cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), out_dtype, num_bits_per_copy=128
            ),
            out,
            self._mark_alignment(dst, hidden_per_thread * out_dtype.width // 8),
        )
