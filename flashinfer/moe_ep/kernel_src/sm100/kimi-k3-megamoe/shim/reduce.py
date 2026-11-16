# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Mask-aware Kimi top-16 combine with ordered FP32 accumulation."""

from typing import Optional

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Float32, Int32

from ..src.kernel_src.blackwell.inference.mega.topk_reduce import TopkReduce


class KimiTopkReducer(TopkReduce):
    """Skip zero-score slots before reading their data or scales."""

    def __init__(
        self,
        hidden: int,
        num_topk: int,
        combine_format,
        quant_kind=None,
        max_tokens=None,
    ) -> None:
        if hidden != 3584 or num_topk != 16:
            raise ValueError("Kimi combine requires hidden=3584 and num_topk=16")
        if combine_format.name not in ("bf16", "32e4m3xe8m0"):
            raise ValueError("Kimi combine supports BF16 or MXFP8")
        self.hidden = hidden
        self.num_topk = num_topk
        self.combine_format = combine_format
        self.hidden_per_thread = 16 if combine_format.is_quantized else 8
        self.hidden_tiles = hidden // self.hidden_per_thread
        self.s18 = True
        self.s18_tpb = 32
        self.s18_ld = 2
        self.s18_st = 0
        self.s18_wpt = 1

    def s18_tag(self) -> str:
        return "_kimi_masked_s18_v1"

    @cute.kernel
    def _reduce_v2(
        self,
        combine_quant: cute.Tensor,
        combine_sf: Optional[cute.Tensor],
        topk_score: cute.Tensor,
        reduced_output: cute.Tensor,
        total_workers: Int32,
    ):
        hpt = self.hidden_per_thread
        num_topk: cutlass.Constexpr[int] = self.num_topk
        quantized = self.combine_format.is_quantized
        act_dtype = self.combine_format.act_dtype
        out_dtype = reduced_output.element_type
        worker = cute.arch.block_idx()[0] * Int32(32) + cute.arch.thread_idx()[0]

        if worker < total_workers:
            token = worker // self.hidden_tiles
            tile = worker % self.hidden_tiles
            codes = cute.zipped_divide(
                combine_quant[token, None, None], (num_topk, hpt)
            )[(None, None), (0, tile)]
            dst = cute.zipped_divide(reduced_output[token, None], (hpt,))[
                (None,), (tile,)
            ]
            scores = cute.make_rmem_tensor((num_topk,), cutlass.Float32)
            cute.autovec_copy(topk_score[token, None], scores)
            scales = cute.make_rmem_tensor((num_topk,), cutlass.Float8E8M0FNU)
            sf = None
            if cutlass.const_expr(quantized):
                sf = cute.zipped_divide(combine_sf[token, None, None], (num_topk, hpt))[
                    (None, None), (0, tile)
                ]

            # Keep the ordered 16-byte slot loads ahead of the reduction.
            terms = []
            for k in cutlass.range_constexpr(num_topk):
                term = cute.make_rmem_tensor((hpt,), act_dtype)
                if scores[k] != Float32(0):
                    self._s18_load16(codes[k, None], term)
                    if cutlass.const_expr(quantized):
                        scales[k] = sf[k, 0]
                terms.append(term)

            acc = cute.make_rmem_tensor((hpt,), cutlass.Float32)
            acc.fill(Float32(0.0))
            for k in cutlass.range_constexpr(num_topk):
                if scores[k] != Float32(0):
                    values = cute.make_rmem_tensor((hpt,), cutlass.Float32)
                    values.store(terms[k].load().to(cutlass.Float32))
                    score_pair = (scores[k], scores[k])
                    for i in cutlass.range_constexpr(0, hpt, 2):
                        pair = (values[i], values[i + 1])
                        if cutlass.const_expr(quantized):
                            scale = Float32(scales[k])
                            pair = cute.arch.mul_packed_f32x2(pair, (scale, scale))
                        acc[i], acc[i + 1] = cute.arch.fma_packed_f32x2(
                            pair, score_pair, (acc[i], acc[i + 1])
                        )

            out = cute.make_rmem_tensor((hpt,), out_dtype)
            out.store(acc.load().to(out_dtype))
            self._s18_store(out, dst)
