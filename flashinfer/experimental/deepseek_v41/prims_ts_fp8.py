# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
"""Shared-unit E4M3 SWA/main attention for the V4.1 adapter.

Original metadata is checked inside one finalizing native attention. Both sources
share online softmax, FP32 PV, sink normalization and final BF16 publication.
"""

import cutlass
import cutlass.cute as cute


class Fp8SwaDecodeKernel:
    def __init__(self, attention):
        self.attention = attention

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        swa_cache: cute.Tensor,
        main_cache: cute.Tensor,
        indices: cute.Tensor,
        topk_lens: cute.Tensor,
        seq_lens: cute.Tensor,
        sinks: object,
        out: cute.Tensor,
        lse: cute.Tensor,
        bmm1_scale: cutlass.Float32,
        bmm2_scale: cutlass.Float32,
        stream: object,
    ):
        final_lse = cute.make_tensor(
            lse.iterator,
            cute.make_layout((64, 1, indices.shape[0]), stride=(1, 64, 64)),
        )
        # Preserve the native seven-slot ABI. The producer-owned binder consumes
        # the original combined table/live lengths instead of prepared lists.
        direct = (indices, None, seq_lens, topk_lens, None, None, sinks)
        self.attention(
            q,
            q,
            main_cache,
            swa_cache,
            None,
            out,
            final_lse,
            None,
            cutlass.Int32(1),
            seq_lens,
            None,
            direct,
            bmm1_scale,
            bmm2_scale,
            stream,
        )
