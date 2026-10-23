# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
"""Native attention configuration and ABI adapter for V4.1 Prims-TS decode.

Original metadata is checked inside one finalizing native attention. Both sources
share online softmax, FP32 PV, sink normalization and final BF16 publication.
"""

import cutlass
import cutlass.cute as cute


def _make_fp8_attention(compressed_k, sm_count, native_fp4):
    from ....attention.prims_ts.kernels.mla_decode.sparse_policy import SparseMlaProfile
    from ....attention.prims_ts.kernels.mla_decode.throughput_latency_1cta.kernel import (
        ThroughputLatencyMlaDecodeTs,
    )
    from ....attention.prims_ts.kernels.mla_decode.throughput_latency_1cta.config import (
        MlaProfile,
    )
    from .resources import MixedFp8KvProducer

    # A fixed SWA HEAD precedes ceil(K/128) main LOOP tiles, including when
    # every SWA entry is invalid. One online softmax/PV state spans both pools.
    capacity = max(256, 128 + ((compressed_k + 127) // 128) * 128)
    attention = ThroughputLatencyMlaDecodeTs(
        batch_size=1,
        num_heads=64,
        seq_len_q=1,
        seq_len_k=capacity,
        latent_dim=512,
        rope_dim=0,
        page_size=1,
        max_active_clusters=sm_count,
        qkv_dtype="e4m3",
        out_dtype="bf16",
        profile=MlaProfile(
            name="dsv41_shared_pertensor_joint",
            kernel_variant="keeps_mma_ab",
            tile_size_q=64,
            use_persistent_scheduler=0,
        ),
        logical_num_heads=64,
        logical_seq_len_q=1,
        tile_size_q=64,
        explicit_split_kv=1,
        explicit_persistent=False,
        reduction_mode="gmem_separate",
        mask_type="dense",
        sparse_profile=SparseMlaProfile(
            family="keep",
            tile_size_q=64,
            split_kv=1,
            scheduler="nonpersistent",
            head_dim_ctas=1,
            gather_issue_warps=4,
            reuse_kv=True,
            reuse_kv_stages=8,
            kv_tile_size=128,
            page_pipeline_stages=1,
        ),
        finalize_output=True,
        kv_producer=MixedFp8KvProducer(native_fp4=native_fp4),
    )
    # Static here means host scalars, not device scales or value-specialized
    # binaries; both BMM factors remain runtime Float32.
    attention.direct_sparse = True
    attention.direct_static_scales = True
    attention.direct_split_softmax_scale = True
    attention.direct_sparse_capacities = (128, compressed_k)
    return attention


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
