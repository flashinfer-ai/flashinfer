# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.

"""Experimental Frost kernels for DeepSeek V4.1 on Blackwell."""

from .api_logging import flashinfer_experimental_api


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
def deepseek_v41_decode(
    q,
    swa_cache,
    global_cache,
    swa_indices,
    global_indices,
    sink,
    *,
    plan=None,
    backend="flashmla",
    arithmetic=None,
):
    """H64/D512 BF16 Q over SWA MXFP8 plus global group16/E4M3 FP4.

    Select ``backend="frost"`` for the native Blackwell implementation, or
    ``backend="flashmla"`` for the existing provider (default). Frost accepts
    ``arithmetic="bf16x3"`` (default) or ``"tf32x3"``; FlashMLA requires None.
    Returns ``(out, lse, plan)``. Q is
    ``[batch,query_tokens,64,512]`` and index tensors are physical slot IDs.
    Returned natural-log LSE excludes the sink. Prepare once before graph
    capture, then reuse the plan only with identical tensor declarations.
    Causal visibility is the caller's index contract. No backward is attached.
    Frost owns reusable output/LSE/workspace through its plan. Replays overwrite
    these same buffers; copy outputs explicitly if retaining previous steps.
    Plans are specific to the selected backend and arithmetic recipe.
    """
    from .experimental.deepseek_v41.decode import dispatch_decode

    return dispatch_decode(
        q,
        swa_cache,
        global_cache,
        swa_indices,
        global_indices,
        sink,
        plan=plan,
        backend=backend,
        arithmetic=arithmetic,
    )


@flashinfer_experimental_api
def deepseek_v41_window_decode(
    q,
    cache,
    indices,
    sink,
    *,
    plan=None,
    backend="flashmla",
    arithmetic=None,
):
    """H64/D512 BF16 queries over one V4.1 group32/E8M0 FP8 cache.

    ``backend="frost"`` selects native Blackwell decode; ``"flashmla"`` remains
    the default provider. Frost arithmetic is ``"bf16x3"`` (default) or
    ``"tf32x3"``; FlashMLA requires None. Cache is
    uint8[pages,page_size,1,528], using quantize_cache's paged layout. Q is
    BF16[B,Sq,64,512], indices int32[B,Sq,K], sink FP32[64]. Caller owns slot
    liveness/visibility. Returns(out,lse,plan); natural-log LSE excludes sink.
    Warm up once and reuse the identical-declaration plan for graph calls.
    Frost reuses plan-owned output/LSE/workspace; FlashMLA allocates through
    its provider. Both support CUDA Graph capture after preparation.
    No backward, projections, RoPE or speculative scheduler is attached.
    """
    from .experimental.deepseek_v41.decode import dispatch_decode

    return dispatch_decode(
        q,
        cache,
        None,
        indices,
        None,
        sink,
        plan=plan,
        backend=backend,
        arithmetic=arithmetic,
    )


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
def deepseek_v41_decode_fp32(
    q,
    swa_cache,
    global_cache,
    swa_indices,
    global_indices,
    sink,
    *,
    workspace=None,
    out=None,
    lse=None,
):
    """Mixed-cache decode with FP32 probabilities and an explicit TF32x3 PV.

    BF16 Q[B,Sq,64,512], existing packed SWA MXFP8/global group16-E4M3 FP4
    pages, int32 physical indices[B,Sq,K] and FP32 sink[64]. Index widths
    are padded multiples of64, at most192 SWA and512 global; -1 is invalid.
    Global cache/indices may both be None. Caller guarantees live slots,
    finite decoded values/FP32 logits/sinks and causal selected indices.
    Returns BF16 output, FP32 natural-log LSE[B,64,Sq] excluding the sink,
    and reusable workspace dict(partial,max,sum). Reuse all three outputs
    for allocation-free graph replay. Empty selections give zero output/-inf
    LSE. This is an explicit inference arithmetic recipe, distinct from the
    FlashMLA backend; it makes no checkpoint parity or backward claim.
    """
    from .experimental.deepseek_v41.decode_fp32 import decode_fp32

    return decode_fp32(
        q,
        swa_cache,
        global_cache,
        swa_indices,
        global_indices,
        sink,
        workspace=workspace,
        out=out,
        lse=lse,
    )


@flashinfer_experimental_api
def deepseek_v41_decode_bf16x3(
    q,
    swa_cache,
    global_cache,
    swa_indices,
    global_indices,
    sink,
    *,
    workspace=None,
    out=None,
    lse=None,
):
    """Mixed-cache decode with FP32 softmax and three BF16 PV terms.

    Same input/cache/output envelope as deepseek_v41_decode_fp32, with a
    separate arithmetic recipe: transposed BF16 QK and three-term BF16
    probability decomposition for TCGen5 PV. This is an explicit inference
    opt-in, not checkpoint parity or a backward implementation. Workspace
    uses64-key partials and is not interchangeable with the32-key TF32x3
    recipe. Allocate once, then reuse it with out/LSE for graph replay.
    """
    from .experimental.deepseek_v41.decode_fp32 import decode_fp32

    return decode_fp32(
        q,
        swa_cache,
        global_cache,
        swa_indices,
        global_indices,
        sink,
        workspace=workspace,
        out=out,
        lse=lse,
        recipe="bf16x3_tcgen",
        head_tile=None,
    )
