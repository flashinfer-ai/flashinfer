# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.

"""Experimental DeepSeek V4.1 index scoring, Frost decode and cache preparation."""

from .api_logging import flashinfer_experimental_api


@flashinfer_experimental_api
def deepseek_v41_decode(
    q, swa_cache, global_cache, swa_indices, global_indices, sink, *, plan=None
):
    """Frost mixed MXFP8/FP4 decode using CuTe DSL primitives on SM100.

    Q is contiguous BF16[B,1,64,512]. Caches are opaque uint8 page pools
    [pages,64,1,528] (SWA) and [pages,64,1,288] (global). SWA indices are
    int32[B,1,128]; global indices int32[B,1,K], K a positive multiple of64
    up to512. Slots -1 and out-of-capacity slots are masked. The caller owns
    causal visibility and initialization of every valid cache slot. Sink is
    FP32[64], with finite logits or -inf (disabled). Inputs must be finite
    after dequantization; no checkpoint or backward parity is claimed.

    Returns (out,lse,plan): BF16 output shaped like Q and FP32 natural-log
    LSE [B,64,1] excluding the sink. Empty rows give zero output and -inf LSE.
    Multiplications use BF16 Q/K/V/P; accumulation and softmax use FP32.
    Prepare once before capture, then reuse plan with identical declarations
    on its device. Replay overwrites plan-owned output/LSE/workspace; the
    captured graph keeps fixed input addresses: update values in place, or
    recapture/update the graph when addresses change. Ordinary plan calls
    may use new addresses with matching declarations. The same plan must
    not execute concurrently on different streams. Single
    token decode only; MTP and SM103 are not part of this initial scope.
    """
    from .experimental.deepseek_v41.decode import decode

    return decode(
        q, swa_cache, global_cache, swa_indices, global_indices, sink, plan=plan
    )


@flashinfer_experimental_api
def deepseek_v41_window_decode(q, cache, indices, sink, *, plan=None):
    """Window-only variant of deepseek_v41_decode, using the same Frost kernel.

    Accepts BF16[B,1,64,512] Q, uint8[pages,64,1,528] cache and
    int32[B,1,128] physical slot IDs. See deepseek_v41_decode for the plan,
    arithmetic, validity, output and sink-exclusive LSE contracts.
    """
    from .experimental.deepseek_v41.decode import decode

    return decode(q, cache, None, indices, None, sink, plan=plan)


@flashinfer_experimental_api
def deepseek_v41_quantize_cache(x, *, format, page_size=64, out=None, slots=None):
    """Quantize and write the two cache formats consumed by Frost DS4.1 decode.

    Optional Triton preparation helper; decode accepts compatible caches from
    any producer. Contiguous finite BF16/FP32 x[N,512] must already include
    any model-required RoPE. SM100 and page_size=64 only.

    ``main_kv_fp4`` uses E2M1 data, group16/E4M3 scales, no global scale;
    ``swa_mxfp8`` uses E4M3 data and group32/E8M0 scales. Main-cache scales
    must be representable in E4M3. Opaque uint8 output has shape
    [pages,64,1,288] or [pages,64,1,528]. Each page holds all data rows,
    followed by all scale rows; these are not interleaved token records.

    Without slots, input row i writes physical slot i. For incremental
    updates, provide caller-owned out and CUDA int32 slots[N]. Negative or
    out-of-capacity slots skip publication. Valid slots must be unique;
    uniqueness is a caller precondition and is not checked. Unwritten slots
    remain untouched and must not be attended to before initialization.
    Inputs and output must not overlap. Reusing out permits allocation-free
    CUDA Graph replay. No implicit QAT derivative.
    """
    from .experimental.deepseek_v41.cache import quantize_cache

    return quantize_cache(x, format=format, page_size=page_size, out=out, slots=slots)


@flashinfer_experimental_api
def deepseek_v41_index_scores_fp32(
    q_data,
    q_scales,
    kv_cache,
    weights,
    visible,
    block_table,
    *,
    max_context_len,
    candidates=None,
    out=None,
    backend="triton",
    workspace=None,
):
    """Tiled full/candidate MXFP4 index scores with FP32 dot/head accumulation.

    Q bytes[B,32,64], E8M0 scales[B,32,4], D128 paged index cache, BF16
    weights[B,32], int32 visible[B] and block_table[B,pages]. Logical context
    must fit the page table and int32. Optional int32 candidates[B,1..2048]
    contains block8 IDs; -1 or unreachable positions yield -inf scores.
    Caller guarantees initialized referenced pages (including tail slots),
    0<=visible<=max_context_len,
    finite BF16-representable decoded Q/K, finite weights and finite FP32
    dot products, weighted terms and head sums.
    Returns BF16[B,max_context_len or C*8], with 1024-byte aligned rows for
    selection. Reuse out for graph replay. Native Blackwell
    MXFP4 tensor-core products with FP32 accumulation and an ordered FP32
    head reduction; no checkpoint-ID or QAT claim.

    backend="triton" supports full and candidate scoring without scratch.
    backend="cute_dsl" reuses FlashInfer's native FP4 scorer (Dhiraj Reddy's
    TensorRT-LLM/DeepGEMM port), extended for H32 block8 candidates. Requires
    CUTLASS DSL>=4.7 and 16-byte-aligned Q, scales, weights and KV storage.
    It is intended for large candidate batches; benchmark both backends for
    your shape. Candidate order is preserved, including duplicates.

    The optional CuTe workspace is a contiguous 16-byte-aligned uint8 CUDA
    tensor. For B=batch and C=candidate count, allocate at least
    align16(4*B*C) + align16(B*ceil(C/4)*4) bytes. It stores encoded
    candidate IDs and packed validity; each scorer gathers its own KV scales.
    Use a separate workspace for concurrently executing calls. Reusing out
    and workspace avoids allocation; warm the call before CUDA Graph capture.
    The workspace and output must not overlap any input or each other.
    CuTe requires physical_pages*(page_size/8) and B*out.stride(0)
    to be below 2**31; the Triton backend supports larger address spans.
    KV data within a page is packed token-major first, followed by all four
    scale bytes per token, with page stride padded to a multiple of 512 bytes.
    """
    from .experimental.deepseek_v41.indexer_fp32 import index_scores_fp32

    return index_scores_fp32(
        q_data,
        q_scales,
        kv_cache,
        weights,
        visible,
        block_table,
        max_context_len=max_context_len,
        candidates=candidates,
        out=out,
        backend=backend,
        workspace=workspace,
    )


@flashinfer_experimental_api
def prepare_deepseek_v41_candidate_metadata(
    visible,
    block_table,
    candidates,
    *,
    page_size,
    num_physical_pages,
    max_context_len,
    out=None,
):
    """Publish an owned snapshot of H32 block8 candidate metadata on CUDA.

    visible is int32[B], block_table is int32[B,pages], and candidates is
    int32[B,C], with 1<=C<=2048. All tensors must be contiguous on the same
    SM100/SM103 device. page_size is 32, 64 or 128; max_context_len is a
    positive int32 context covered by the table. Caller guarantees
    0<=visible<=max_context_len. Invalid candidate or physical page IDs
    produce masked scores; duplicates and candidate order are preserved.

    The returned opaque metadata owns encoded IDs, validity and visibility.
    Later changes to these input tensors do not change the snapshot. Publish
    again when candidates, visibility or page mapping change. Pass an existing
    metadata object as out to reuse its storage; batch, candidate count, page
    size, physical page count, context and device must match. Warm preparation
    and consumption before CUDA Graph capture, then reuse metadata and scores.

    Publication runs on the current stream. Order consumers after publication
    using stream order or events, including replay of a captured publication.
    Do not republish a snapshot while any consumer reads it. Once publication
    completes, different layers or streams may read it with separate outputs.
    No metadata reuse across layers is assumed by the API.
    """
    from .experimental.deepseek_v41.candidate_metadata import prepare_candidate_metadata

    return prepare_candidate_metadata(
        visible,
        block_table,
        candidates,
        page_size=page_size,
        num_physical_pages=num_physical_pages,
        max_context_len=max_context_len,
        out=out,
    )


@flashinfer_experimental_api
def deepseek_v41_candidate_scores_fp32(
    q_data, q_scales, kv_cache, weights, metadata, *, out=None
):
    """Score one layer using explicitly prepared block8 candidate metadata.

    metadata must be returned by prepare_deepseek_v41_candidate_metadata.
    Q, scales, cache, weights, output and finite-value requirements match
    deepseek_v41_index_scores_fp32 with backend="cute_dsl". The same credited
    native CuTe scorer and ordered FP32 arithmetic are used; CUTLASS DSL>=4.7
    is required. Returns BF16[B,C*8] with the same candidate ordering.

    Layer tensors must match the snapshot's batch, page size, physical page
    count and device. Cache addresses and padded strides may differ between
    layers; address width is selected from each complete strided cache pool.
    Output and layer tensors must not overlap metadata storage, and output
    must not overlap inputs. This function only consumes the snapshot;
    publish it again explicitly when candidates, mapping or visibility change.
    Reuse out to avoid output allocation during graph replay.
    """
    from .experimental.deepseek_v41.candidate_metadata import candidate_scores_fp32

    return candidate_scores_fp32(q_data, q_scales, kv_cache, weights, metadata, out=out)
