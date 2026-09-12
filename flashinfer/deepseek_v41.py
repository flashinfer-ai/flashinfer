# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.

"""Experimental Frost DS4.1 decode and optional cache preparation on SM100."""

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
