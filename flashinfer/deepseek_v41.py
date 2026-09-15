# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.

"""Experimental DeepSeek V4.1 MXFP4 index scoring on Blackwell."""

from .api_logging import flashinfer_experimental_api


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
    align16(4*B*C) + align16(8*B*C) + 4*B*ceil(C*8/128)*128 bytes.
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
