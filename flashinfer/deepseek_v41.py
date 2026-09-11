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
def deepseek_v41_compressor_decode(
    kv,
    score,
    kv_state,
    score_state,
    norm_weight,
    starts,
    *,
    eps=1e-20,
    out=None,
    positions=None,
):
    """Update CSA2 decode state, pool completed pairs and normalize pre-RoPE KV.

    FP32 projections kv/score[B,512], mutable FP32 states[B,2,512], contiguous
    BF16/FP32 norm_weight[512] and int32 starts[B], all on one CUDA device.
    Caller owns each request state and supplies sequential nonnegative token
    positions; on odd positions the previous even slot must be initialized.
    Projections, scores and norm weights must be finite. Returns BF16 out[B,512]
    and int32 compressed positions[B]: start//2 for odd tokens, otherwise-1.
    Incomplete rows leave out unchanged and must not be consumed. Reuse both
    outputs for allocation-free graph replay. Pooling rounds to BF16 before
    FP32 RMS normalization, as in the published compressor. This is inference
    only, without projection GEMMs, RoPE, quantization or a cache write.
    """
    from .experimental.deepseek_v41.compressor import compressor_decode

    return compressor_decode(
        kv,
        score,
        kv_state,
        score_state,
        norm_weight,
        starts,
        eps=eps,
        out=out,
        positions=positions,
    )


@flashinfer_experimental_api
def deepseek_v41_compressor_projection(
    x, wkv, wgate, *, workspace=None, kv=None, score=None
):
    """Compute both small-batch CSA2 FP32 projections from one BF16 input.

    Contiguous CUDA x[B,5120] BF16, wkv/wgate[512,5120] FP32, B in1..16.
    Uses eight deterministic split-K partials with TF32x3 products and FP32
    reduction; this is an explicit numerical recipe, not bitwise cuBLAS order.
    Inputs must be finite. Returns FP32 kv/score[B,512]. Reuse both outputs and
    workspace FP32[8,2,B,512] for allocation-free graph replay. No weight
    quantization, state update, normalization or training backward is attached.
    """
    from .experimental.deepseek_v41.compressor_projection import compressor_projection

    return compressor_projection(x, wkv, wgate, workspace=workspace, kv=kv, score=score)


@flashinfer_experimental_api
def deepseek_v41_compressor_step(
    x,
    wkv,
    wgate,
    kv_state,
    score_state,
    norm_weight,
    starts,
    *,
    eps=1e-20,
    workspace=None,
    out=None,
    positions=None,
):
    """Run native CSA2 dual projection and decode state/pool/norm in two launches.

    Contracts match compressor_projection plus compressor_decode: BF16
    x[B,5120], FP32 wkv/wgate[512,5120], FP32 states[B,2,512], BF16/FP32
    norm_weight[512], int32 sequential nonnegative starts[B], B in1..16.
    The TF32x3 partial reduction is fused into state/pool/norm. Prior even
    slots must be initialized before odd positions. Returns BF16 out[B,512]
    and int32 compressed positions[B]; -1/incomplete preserves the output row.
    Reuse FP32 workspace[8,2,B,512] and outputs for allocation-free graph replay.
    Input finiteness and state ownership follow the component APIs. No RoPE,
    index projection, quantized cache, attention or training backward is attached.
    """
    from .experimental.deepseek_v41.compressor_step import compressor_step

    return compressor_step(
        x,
        wkv,
        wgate,
        kv_state,
        score_state,
        norm_weight,
        starts,
        eps=eps,
        workspace=workspace,
        out=out,
        positions=positions,
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
def deepseek_v41_index_key(x, weight, norm_weight, *, eps=1e-20, out=None):
    """Project pre-RoPE compressed latents to normalized BF16 index keys.

    Contiguous CUDA BF16 x[B,512], BF16 weight[128,512], BF16/FP32
    norm_weight[128], B1..16. One native BF16 tensor-core projection followed
    by a BF16 rounding boundary, FP32 RMSNorm and final BF16 output[B,128].
    Finite inputs required; this is an explicit accumulation recipe without
    a bitwise cuBLAS-order guarantee. Reuse out for allocation-free graph
    replay. Inference only: no RoPE, quantization, cache write or backward.
    """
    from .experimental.deepseek_v41.index_key import index_key

    return index_key(x, weight, norm_weight, eps=eps, out=out)


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
