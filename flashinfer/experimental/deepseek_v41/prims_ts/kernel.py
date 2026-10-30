# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
"""Native attention configuration and ABI adapter for V4.1 Prims-TS decode.

Original metadata is checked inside one finalizing native attention. Both sources
share online softmax, FP32 PV, sink normalization and final BF16 publication.
"""

import cutlass
import cutlass.cute as cute


def _make_fp8_attention(
    compressed_k, sm_count, native_fp4, head_dim_ctas=1, kv_splits=1
):
    from ....attention.prims_ts.kernels.mla_decode.sparse_policy import SparseMlaProfile
    from ....attention.prims_ts.kernels.mla_decode.throughput_latency_1cta.kernel import (
        ThroughputLatencyMlaDecodeTs,
    )
    from ....attention.prims_ts.kernels.mla_decode.throughput_latency_1cta.config import (
        MlaProfile,
    )
    from .resources import MixedFp8KvProducer

    # A fixed SWA tile precedes ceil(K/128) main tiles, including when every
    # SWA entry is invalid. One online softmax/PV state spans both pools per KV
    # split; only split 0 starts with the SWA tile.
    capacity = 128 + ((compressed_k + 127) // 128) * 128
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
            # V owners each read their D128 slices of the retained K tile, so
            # a split divides PV, correction and output work, not the loads.
            num_ctas_per_head_dim=head_dim_ctas,
            # KV splits divide the tile loads and FP4 conversion; the separate
            # reducer merges the BF16 partials and applies sinks once.
            num_ctas_per_seq_kv=kv_splits,
            use_multi_ctas_kv=int(kv_splits > 1),
            use_persistent_scheduler=0,
        ),
        logical_num_heads=64,
        logical_seq_len_q=1,
        tile_size_q=64,
        explicit_split_kv=kv_splits,
        explicit_persistent=False,
        reduction_mode="gmem_separate",
        mask_type="dense",
        sparse_profile=SparseMlaProfile(
            family="keep",
            tile_size_q=64,
            split_kv=kv_splits,
            scheduler="nonpersistent",
            head_dim_ctas=head_dim_ctas,
            # Eight loader warps halve each warp's FP4 conversion and SWA gather.
            gather_issue_warps=8,
            reuse_kv=True,
            reuse_kv_stages=8,
            kv_tile_size=128,
            # Two index stages let the next tile's row IDs load while the
            # loaders still stage the current tile.
            page_pipeline_stages=2,
        ),
        finalize_output=True,
        kv_producer=MixedFp8KvProducer(native_fp4=native_fp4),
        # Compile each main width's static SWA + main capacity; live lengths
        # still arrive with the metadata on every call.
        fixed_seqlen=True,
    )
    # Both BMM factors are runtime Float32 host scalars, not value-specialized
    # binaries.
    attention.direct_sparse = True
    attention.direct_sparse_capacities = (128, compressed_k)
    # Stage final BF16 rows through SMEM so global stores write whole lines.
    attention.coalesce_keeps_output = True
    # Overlap launch and prologue with the preceding grid (PDL).
    attention.use_pdl = True
    # B200 A/B: an entry release saves 2-3% for B=1 grids (4-20 CTAs) and
    # costs 2-6% once the grid covers more SMs (e.g. B16/K512: 12.50 vs 11.74 us).
    attention.pdl_entry_trigger_max_ctas = sm_count // 4
    return attention


class Fp8SwaDecodeKernel:
    def __init__(self, attention, kv_splits=1):
        self.attention = attention
        self.kv_splits = kv_splits

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
        partials: object,
        bmm1_scale: cutlass.Float32,
        bmm2_scale: cutlass.Float32,
        stream: object,
    ):
        final_lse = cute.make_tensor(
            lse.iterator,
            cute.make_layout((64, 1, indices.shape[0]), stride=(1, 64, 64)),
        )
        # Preserve the native six-slot ABI. The producer-owned binder consumes
        # the original combined table/live lengths instead of prepared lists.
        direct = (indices, None, seq_lens, topk_lens, sinks, None)
        self.attention(
            q,
            q,
            main_cache,
            swa_cache,
            None,
            out,
            final_lse,
            partials,
            cutlass.Int32(self.kv_splits),
            seq_lens,
            None,
            direct,
            bmm1_scale,
            bmm2_scale,
            stream,
        )
