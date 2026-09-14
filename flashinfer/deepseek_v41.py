# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.

"""Experimental Frost kernels for DeepSeek V4.1 on Blackwell."""

from .api_logging import flashinfer_experimental_api


@flashinfer_experimental_api
def deepseek_v41_small_index_scores(
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
):
    """Compute short-context MXFP4 index scores with FP32 dot/head reductions.

    Q uses byte data[B,32,64] and E8M0 scales[B,32,4]; K uses the D128
    physical cache layout from quantize_index_cache. BF16 weights[B,32],
    int32 visible[B] and block_table[B,pages] share Q's CUDA device.
    max_context_len is1..128; caller guarantees0<=visible<=max_context_len,
    finite Q/K/weights and live nonnegative physical pages. Optional int32
    candidates[B,C] gives1..16 block8 IDs; -1/unreachable columns produce-inf.
    Output is BF16[B,max_context_len or C*8], with1024-byte row stride alignment
    for DeepSelect. Reuse out for allocation-free graph calls. This explicit
    experimental kernel has FP32 accumulation, unlike BF16 partial-sum
    provider variants; it does not assert checkpoint selection-ID parity.
    """
    from .experimental.deepseek_v41.indexer import small_index_scores

    return small_index_scores(
        q_data,
        q_scales,
        kv_cache,
        weights,
        visible,
        block_table,
        max_context_len=max_context_len,
        candidates=candidates,
        out=out,
    )


@flashinfer_experimental_api
def deepseek_v41_paged_indices(indices, block_table, *, page_size=64, out=None):
    """Map logical int32 [B,Sq,K] token IDs into physical paged cache slots.

    The int32 [B,pages] table stores physical page numbers; page_size is
    32/64/128. Negative or out-of-capacity IDs and negative table entries
    produce -1. Caller guarantees nonnegative physical pages refer to live
    cache storage and page*page_size+offset fits int32. This does not apply
    causality or initialize cache pages. Reuse out for allocation-free calls
    and graph replay with changed IDs/table. Output may not alias inputs.
    """
    from .experimental.deepseek_v41.paging import paged_indices

    return paged_indices(indices, block_table, page_size=page_size, out=out)


@flashinfer_experimental_api
def deepseek_v41_token_workspace(scores, *, topk_tokens=512):
    """Allocate reusable aligned logits, visible lengths and token ID outputs."""
    from .experimental.deepseek_v41.selection import token_workspace

    return token_workspace(scores, topk_tokens=topk_tokens)


@flashinfer_experimental_api
def deepseek_v41_token_topk(
    scores, visible, *, candidates=None, topk_tokens=512, workspace=None
):
    """Select sorted causal token IDs from dense or candidate-block BF16 scores.

    With candidates=None, score column is the logical token ID and caller
    guarantees 0 <= visible <= score_width. Otherwise candidates are int32
    [queries,score_width/8] sorted unique logical block IDs, followed by -1
    padding; score column c corresponds to candidates[c//8]*8+c%8. Visible is
    the logical context length. Valid candidates may extend beyond visibility.
    Their ascending order makes reachable score columns a contiguous prefix.
    Visible scores must be finite; unreachable output slots are -1. Boundary
    ties use DeepSelect semantics. No QAT or selection derivative is attached.
    """
    from .experimental.deepseek_v41.selection import token_topk

    return token_topk(
        scores,
        visible,
        candidates=candidates,
        topk_tokens=topk_tokens,
        workspace=workspace,
    )


@flashinfer_experimental_api
def deepseek_v41_candidate_workspace(scores, *, topk_blocks=2048):
    """Allocate reusable candidate block logits, lengths and index output."""
    from .experimental.deepseek_v41.selection import candidate_workspace

    return candidate_workspace(scores, topk_blocks=topk_blocks)


@flashinfer_experimental_api
def deepseek_v41_candidate_blocks(scores, visible, *, topk_blocks=2048, workspace=None):
    """Select causal block8 candidates and pin the newest visible block.

    BF16 scores [queries,KV] and int32 visible [queries] are device tensors.
    Caller guarantees 0 <= visible <= KV and finite visible scores. The explicit
    DeepSelect provider selects up to topk_blocks, sorts valid block IDs, and
    pads unreachable slots with -1. Ties follow provider selection semantics;
    exact Torch index identity is not promised. Reuse workspace for capture.
    """
    from .experimental.deepseek_v41.selection import candidate_blocks

    return candidate_blocks(
        scores, visible, topk_blocks=topk_blocks, workspace=workspace
    )


@flashinfer_experimental_api
def deepseek_v41_quantize_index_cache(x, *, page_size=64, out=None, slots=None):
    """Fuse MXFP4 D128 quantization into DeepGEMM indexer cache pages.

    Each page contains all 64-byte data rows followed by all 4-byte scale rows.
    The returned opaque uint8 view has shape [pages,page_size,1,68], with page
    stride padded to 512 bytes for sparse paged indexing. It is NOT a tensor
    of interleaved data/scale token rows. Slots follow pack_cache's ownership
    contract. Caller-owned output permits allocation-free graph replay.
    """
    from .experimental.deepseek_v41.quantization import quantize_index_cache

    return quantize_index_cache(x, page_size=page_size, out=out, slots=slots)


@flashinfer_experimental_api
def deepseek_v41_quantize_cache(x, *, format, page_size=64, out=None, slots=None):
    """Fuse D512 quantization and paged cache/scatter update in one GPU launch.

    Formats, page layout and slot ownership match quantize/pack_cache below.
    ``out`` enables allocation-free replay; unwritten slots remain unchanged.
    """
    from .experimental.deepseek_v41.quantization import quantize_cache

    return quantize_cache(x, format=format, page_size=page_size, out=out, slots=slots)


@flashinfer_experimental_api
def deepseek_v41_quantize(x, *, format, data=None, scales=None):
    """Quantize contiguous BF16/FP32 rows into explicit V4.1 data/scale bytes.

    ``format`` is ``mxfp4`` (group32/E8M0), ``main_kv_fp4``
    (group16/E4M3, no global scale) or ``swa_mxfp8`` (group32/E8M0).
    Caller-provided outputs avoid allocation. This API defines no QAT derivative.
    """
    from .experimental.deepseek_v41.quantization import quantize

    return quantize(x, format=format, data=data, scales=scales)


@flashinfer_experimental_api
def deepseek_v41_pack_cache(data, scales, *, page_size=64, out=None, slots=None):
    """Pack data rows followed by scale rows within each physical cache page.

    Inputs are uint8 rows for D512 SWA FP8 (512+16 bytes) or global FP4
    (256+32 bytes). Optional CUDA int32 slots map input rows to physical slots;
    -1 skips a row, and the caller guarantees every other slot is in range and
    unique. Unwritten slots remain untouched and must not be attended to.
    """
    from .experimental.deepseek_v41.quantization import pack_cache

    return pack_cache(data, scales, page_size=page_size, out=out, slots=slots)


@flashinfer_experimental_api
def deepseek_v41_rope_quantize_cache(
    x, freqs, positions, *, format, out, slots, page_size=64
):
    """Rotate the last64 channels, round to BF16 and publish quantized cache rows.

    Contiguous BF16 x[rows,D], FP32 freqs[sequence,32,2] containing real/imaginary
    parts of precomputed complex rotations, and int32 positions/slots[rows].
    All tensors reside on one CUDA device. Formats: index_mxfp4(D128, group32/
    E8M0), main_kv_fp4(D512, group16/E4M3, no global scale), swa_mxfp8(D512,
    group32/E8M0). Caller owns the paged uint8 output and unique live slots.
    A -1 position or slot skips publication. Other positions/slots must be in
    range; active input values and frequencies must be finite, with main-KV
    scales representable in E4M3. Positions address the supplied frequency
    table directly: CSA2 completed groups use their FIRST raw-token position,
    twice the compressed ID. Input rows remain unchanged so index projection
    can consume pre-RoPE latents. Page32/64/128, with the component APIs' cache
    layouts and padded index-page stride. No allocation, QAT or host sync.
    """
    from .experimental.deepseek_v41.rope_cache import rope_quantize_cache

    return rope_quantize_cache(
        x, freqs, positions, format=format, out=out, slots=slots, page_size=page_size
    )


@flashinfer_experimental_api
def deepseek_v41_quantize_gemm(x, *, data=None, scales=None):
    """Quantize activation rows to E4M3 with packed group32/E8M0 GEMM scales.

    Contiguous CUDA BF16/FP32 x[M,K], K128..32768 divisible by128, finite
    values. Returns contiguous E4M3 data[M,K] and int32 scales[M,K/128],
    strides(1,round_up(M,4)). Each int32 packs four adjacent E8M0 bytes with
    the first group in the low byte. This is DeepGEMM's SM100 recipe(1,32)
    layout, distinct from the cuDNN FE six-dimensional scale layout. Unused
    scale-stride padding is untouched. Reuse both outputs for allocation-free
    graph replay. No tensor-global scale or implicit QAT derivative.
    """
    from .experimental.deepseek_v41.gemm_quantization import quantize_gemm

    return quantize_gemm(x, data=data, scales=scales)


@flashinfer_experimental_api
def deepseek_v41_rope(x, freqs, positions, *, inverse=False, out=None):
    """Rotate the last64 BF16 channels of query or attention-output heads.

    CUDA BF16 x[tokens,heads,D], D128/512, with dense heads and optional gaps
    between tokens; output is contiguous. FP32 complex frequency
    parts freqs[sequence,32,2]; device int32 positions[tokens]. Adjacent pairs
    rotate with the published FP32 complex FMA order and round to BF16.
    inverse=True conjugates the rotation for the attention output. Finite
    active inputs/frequencies and positions in range are required; -1 tokens
    preserve the output and must not be consumed. The non-RoPE channels are
    copied unchanged. Reuse out for allocation-free graph replay; out=x is
    supported for contiguous inputs. Other overlaps are rejected.
    Inference-only, no quantization.
    """
    from .experimental.deepseek_v41.rope import rope

    return rope(x, freqs, positions, inverse=inverse, out=out)


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
):
    """Tiled full/candidate MXFP4 index scores with FP32 dot/head accumulation.

    Q bytes[B,32,64], E8M0 scales[B,32,4], D128 paged index cache, BF16
    weights[B,32], int32 visible[B] and block_table[B,pages]. Logical context
    must fit the page table and int32. Optional int32 candidates[B,1..2048]
    contains block8 IDs; -1 or unreachable positions yield -inf scores.
    Caller guarantees initialized live pages, 0<=visible<=max_context_len,
    finite BF16-representable decoded Q/K and finite FP32 logits/weights.
    Returns BF16[B,max_context_len or C*8], with 1024-byte aligned rows for
    DeepSelect. Reuse out for allocation-free graph replay. Native Blackwell
    MXFP4 tensor-core products with FP32 accumulation and an ordered FP32
    head reduction; no checkpoint-ID or QAT claim.
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
    )
