# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FROST kernels of the gated attention block -- a FLAT package, written for Rubin (``sm_107a``).

The block was built for one arch, so there is no arch sub-package here: arch is
the ONLY axis that earns a directory (engine contract § 8) and a single arch
earns none.  Every other axis -- stage, dtype family -- stays in the filename.
Since 2026-10-29 the block's dense bf16 / f16 inference forms also run on cc
10.0 (``api.DENSE_SERVING_CCS``) through the kernels here that carry no arch pin
(``proj_gemm`` over the sm100 GEMM family, the two norm + RoPE kernels,
``cache_write``, ``elementwise``) and the shared SM100 SDPA kernels; the fused
projection forks (``--gpu-arch sm_107a``), the quantize / indexer / sparse
kernels stay Rubin-only and are declined by name there.  By stage of the
pipeline in ``api.py``::

    proj_gemm.py                stages (1) and (6): plan records + runners over the
                                RENDERED FROST GEMM template (writes no kernel)
    proj_gemm_norm_rope.py      stage (1) with (2)+(3) fused into its epilogue -- a
                                bf16 FORK of the rendered GEMM template
    proj_gemm_norm_rope_fp8.py  its FP8 twin: (1)+(2)+(3)+quantize, compact e4m3
                                q8 / k8 / v8 + bf16 gate16 out -- also a fork
    qk_norm_rope.py             stages (2)+(3), per-lane LDG pipeline
    qk_norm_rope_tma.py         stages (2)+(3), TMA-staged A/B of the above
    quantize.py                 (3q) / (5q): per-tensor e4m3 quantize pass (unfused FP8); the quantized BACKWARD's
                                amax / amax-partials passes, gradient casts and scalar-block init
    elementwise.py              stage (5) sigmoid gate, and (3b) V compaction
    cache_write.py              (4w) serving write-through: post-RoPE K / V (and the raw indexer key) into paged pools
                                at a slot mapping -- two operands per launch, HND / NHD / unified-row pools by strides
    qsa_prepare.py              (2+3+4w) under the block knob ``fuse_prepare``: the Q / K norm + RoPE in place and the paged
                                write-through of the post-RoPE K, the V and the raw indexer key as ONE launch; it also carries
                                the indexer's decode-step arms (the queries' norm + RoPE, the raw-key ring, the incremental compress;
                                the e4m3 store arm of the query and compress outputs under the e4m3 indexer cache)
    qsa_compress.py             (4i) the in-block indexer's key side: the raw indexer keys mean-pooled per 4-token block, RMSNormed
                                and rotated at the block start (the prefill form), the same body at pool 1 for the indexer
                                queries, and the step form (the raw-key ring + the incremental compress of the compressed-key cache)
                                with its ring read-only CHUNK form (``ring_write=False``: a prefill chunk behind a cached prefix
                                compresses the blocks it completes, the straddling one from the ring, and writes no ring row);
                                the e4m3 store arm (the e4m3 cast of the bf16 row) on the pool-1 and the step outputs; the
                                ``rounding`` attribute ("single" / "hf") of every compress chain
    qsa_kbar_gather.py          (4i, the extend step) the dense kbar gather: the paged compressed-key cache through the main block
                                table into a contiguous ``[B, n_blocks_max, 128]`` scorer operand (a 16-byte-granule bit copy, bf16 or
                                e4m3; zero rows for a dead / short sequence and for an unassigned page), plus the extend step's device
                                metadata -- the per-piece clamped first position ``q_pos0`` (the memory guard of the compact-logits
                                scratch), the per-row visible counts, the per-sequence report; no SMEM, no mbarrier
    qsa_canonical_list.py       (4i, the extend step; the prefill form too) the canonical row order of a selected block list in one
                                launch: one warp per row, the row's ids sorted DESCENDING by an in-register bitonic network over
                                ``shfl.sync.bfly``, ``-1`` past the position rule's count (a dead row all ``-1``), the per-row list
                                length; in place legal; no SMEM, no mbarrier
    sigmoid_gate_bwd.py         BACKWARD of stage (5): dO, dG into the GATE band, optional O_gated
    qk_norm_rope_bwd.py         BACKWARD of stages (2)+(3): exact RoPE adjoint + RMSNorm backward into
                                the Q / K bands, dW_norm partials + fixed-order reduce, the V band copy
    fp8_bwd_fused.py            the quantized BACKWARD's two fused small-kernel launches -- PROLOGUE (scalar init,
                                dY amax partials, the Q / K rebuild with its e4m3 epilogue, v8) and EPILOGUE (dW_norm
                                reduce, dqkvg quantize): block-range dispatch over the standalone kernels' bodies
    quantize_mxfp8.py           (3q, MXFP8): the block quantize pass -- rowwise / columnwise, the SDPA and the GEMM-canonical
                                SF layouts, and the DUAL-AXIS arm (both quantizations from one read)
    mxfp8_bwd_fused.py          the MXFP8 BACKWARD's two fused launches -- PROLOGUE (scalar init, dY amax partials, the Q / K
                                rebuild with its MX epilogue: q8 / q_T8 / k8 / k_T8 from a 32-token tile, v8) and EPILOGUE
                                (dW_norm reduce, the dual-axis canonical dqkvg cast): block-range dispatch over the bodies
    qsa_score_paged.py          the decode indexer's SCORER over the PAGED compressed-key cache (one compressed row per 4 tokens
                                on the main block table): fp32 ReLU-summed head scores per visible block + the per-row visible
                                counts -- the input contract of the decode top-k; one warp-level mma.sync per 8 keys x 16 D over
                                the (s, h) query rows padded to 16, the query tile staged once per CTA (one CTA barrier), the keys
                                streamed by cp.async through a per-warp SMEM ring, any CuTe DSL device from sm_80 (bf16 compressed rows;
                                the e4m3 READ arm -- e4m3 rows and queries on the fp8 mma.sync.m16n8k32, sm_89+); its standalone selector
                                ``qsa_select_decode_paged_forward`` (scorer + radix top-k) is the DSL-free host entry ``../../QSA/selection_paged.py``,
                                which also owns the kernel's host-side facts and gates the DSL floor BEFORE importing this module (Rule 7)
    qsa_topk_decode.py          the block-sparse indexer's DECODE top-k: one thread block per query row, the row's fp32 logits
                                in registers, one 4096-bin histogram pass plus an exact rank of the k-th bin's candidates,
                                the list written in canonical order (ids descending, -1 last); the DSL-free host entry is
                                ``gated_attention_block/qsa_topk.py`` -- standalone until the decode indexer binds it

**The SDPA stage owns no file here.**  It drives the shipped forward adapter
``cudnn.sdpa.fwd.api_dsl.SdpaFwdDslSm100`` in EVERY configuration; the
sigmoid-gate epilogue behind the block's ``fuse_gate`` is a production feature
of ``sdpa/fwd/kernels/sm107/prefill_d256_{f16,fp8}.py`` behind
``TemplateParams.epilogue_gate`` (engine rows: ``epilogue_gate_d_shapes``),
reached through ``SdpaFwdDslSm100(sample_gate=...)`` / ``execute(gate=...)``.

``proj_gemm_norm_rope*.py`` are forks of the rendered GEMM template BY DECISION
(their epilogues are the block's own norm / RoPE / quantize math on the GEMM
accumulator, not a feature the GEMM engine serves).  The two SDPA forks that
preceded the production ``epilogue_gate`` were deleted when it landed.
"""
