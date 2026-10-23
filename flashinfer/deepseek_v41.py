# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.

"""Experimental DS4.1 decode backends and optional cache preparation."""

from .api_logging import flashinfer_experimental_api


@flashinfer_experimental_api
def deepseek_v41_decode(*args, backend="cute-dsl", **kwargs):
    """Dispatch explicitly to the original CuTe DSL or native Prims-TS API.

    The default ``backend="cute-dsl"`` keeps the original Frost signature::

        deepseek_v41_decode(
            q, swa_cache, global_cache, swa_indices, global_indices, sink,
            plan=None,
        )

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
    not execute concurrently on different streams. Frost supports SM100 and
    single-token decode only; MTP and other GPU architectures are out of scope.

    Explicit ``backend="prims-ts"`` uses a separate caller-workspace API::

        deepseek_v41_decode(
            query, swa_kv_cache, workspace_buffer, sparse_indices=None,
            compressed_kv_cache=None, sparse_topk_lens=None, seq_lens=None,
            out=None, bmm1_scale=1.0, bmm2_scale=1.0, sinks=None,
            kv_layout="HND", backend="prims-ts",
        )

    Prims-TS requires E4M3 Q ``[B,1,64,512]``, typed E4M3 SWA width512,
    and optional packed uint8 FP4 main width288. BF16 Q is rejected without
    conversion or fallback. Only BF16 output is returned; supplied ``out``
    is returned by identity. Workspace is caller-owned contiguous int8/uint8,
    32-byte aligned, with capacity from :func:`deepseek_v41_decode_workspace_size`.
    No native public plan or LSE is returned.

    Native ``sparse_indices`` is required CUDA int32 ``[B,128+K]``; its
    first128 columns address SWA slots and the remaining K address main,
    where K=0/64/.../512. K>0 requires ``compressed_kv_cache``. Required
    int32 ``sparse_topk_lens[B]`` includes fixed128 SWA positions; main
    visibility is ``clamp(topk-128,0,K)``. Required int32 ``seq_lens[B]``
    gives SWA visibility ``[0,clamp(seq_lens,0,128))``. Invalid and reserved
    IDs mask safely; holes and duplicate multiplicity are preserved. Live
    metadata refreshes every call/replay. Callers own causal visibility and
    initialization of every valid slot.

    Native default HND caches are contiguous ``[pages,1,64,width]``;
    explicit NHD uses ``[pages,64,1,width]``. Normalization is zero-copy.
    Packed pages retain separate payload/scale planes. The shared cache
    writer defaults to NHD; request HND explicitly to match native defaults.

    Native main/SWA caches must use common cache units. ``bmm1_scale``
    is positive finite FP32 and ``bmm2_scale`` is finite FP32, both Python
    scalars. For Q descale q_s, common KV descale s>0 and attention factor
    alpha, pass ``alpha*q_s*s`` and ``s``. Defaults are1.0/1.0; explicitly
    pass ``bmm1_scale=512**-0.5`` for that attention factor with unit descales.
    Main staging directly rounds/saturates E2M1*group16_E4M3 to E4M3,
    including intentional +/-448 saturation, without request amax or division.
    Optional ``sinks`` is FP32[64], finite or -inf, and contributes once to
    output normalization. Empty rows produce exact zero output. Effective
    Q/K/V must be finite; this is inference-only.

    Prims-TS supports SM100, SM103 and SM107, with CuTe-DSL>=4.8 and its
    bundled CUDA13 compiler supporting the device's compile target. Its fixed
    Keep-AB/M64/BK128/retained-KV8/split1/nonpersistent schedule jointly
    attends SWA/main in one finalizing kernel with inline metadata and sinks.
    Workspace contains private LSE and a K0 dummy, not sanitized metadata.
    Capture requires caller out/workspace and eager warmup; keep addresses
    alive/fixed and update values in place. Changing Python scales requires
    recapture. Concurrent calls require separate writable buffers or caller
    synchronization. Workspace/out cannot overlap inputs or each other;
    read-only inputs may alias.

    Both backends are explicit, JIT-only H64/Sq1/D512 paths, with no automatic
    selector; Frost remains SM100-only. No full-model equivalence or performance
    is claimed.
    See ``examples/deepseek_v41/README.md`` for limitations and usage.
    """
    if backend == "cute-dsl":
        from .experimental.deepseek_v41.decode import decode
    elif backend == "prims-ts":
        from .experimental.deepseek_v41.prims_ts import decode_mixed as decode
    else:
        raise ValueError(f"backend must be 'cute-dsl' or 'prims-ts', got {backend!r}")
    return decode(*args, **kwargs)


@flashinfer_experimental_api
def deepseek_v41_window_decode(*args, backend="cute-dsl", **kwargs):
    """Window-only decode, preserving each backend's argument/result contract.

    The default uses the original ``(q, cache, indices, sink, *, plan=None)``
    signature and returns ``(out, lse, plan)``. Explicit ``backend="prims-ts"``
    accepts ``(query, swa_kv_cache, workspace_buffer, sparse_indices=None,
    sparse_topk_lens=None, seq_lens=None, out=None, bmm1_scale=1.0,
    bmm2_scale=1.0, sinks=None, kv_layout="HND")`` and returns only output.
    Native indices are int32[B,128] and topk lengths are int32[B] with128.
    See :func:`deepseek_v41_decode` for ownership, layout and scaling.
    """
    if backend == "cute-dsl":

        def decode_window(q, cache, indices, sink, *, plan=None):
            from .experimental.deepseek_v41.decode import decode

            return decode(q, cache, None, indices, None, sink, plan=plan)

    elif backend == "prims-ts":
        from .experimental.deepseek_v41.prims_ts import decode_window
    else:
        raise ValueError(f"backend must be 'cute-dsl' or 'prims-ts', got {backend!r}")
    return decode_window(*args, **kwargs)


@flashinfer_experimental_api
def deepseek_v41_decode_workspace_size(
    batch_size, compressed_topk=0, *, backend="prims-ts"
):
    """Return native Prims-TS workspace bytes, without querying CUDA.

    Only ``backend="prims-ts"`` is supported. ``batch_size`` is in[1,65535];
    ``compressed_topk`` is0 or a multiple of64 through512. The workspace
    must be32-byte aligned. The original CuTe DSL API owns scratch via its
    plan and does not consume this caller workspace.
    """
    from .experimental.deepseek_v41._workspace import workspace_size

    return workspace_size(batch_size, compressed_topk, backend=backend)


@flashinfer_experimental_api
def deepseek_v41_quantize_cache(
    x, *, format, page_size=64, out=None, slots=None, kv_layout="NHD"
):
    """Quantize/write DS4.1 cache bytes, preserving the original NHD default.

    Optional Triton helper for finite contiguous BF16/FP32 x[N,512], after
    model-required RoPE. SM100/SM103/SM107 and page64 only. For native shared descales,
    normalize writer inputs to common cache units upstream; this writer
    does not calculate a shared tensor descale. Triton's selected PTX assembler
    must support the device target (``sm_107a`` on SM107), independently of
    CuTe's bundled compiler.

    ``main_kv_fp4`` uses E2M1 data and group16/E4M3 scales, width288;
    ``swa_mxfp8`` uses E4M3 data and group32/E8M0 scales, width528. Opaque
    uint8 pages retain payload rows followed by scale rows, not interleaved
    token records. Main scales must be representable in E4M3.
    Native ``swa_fp8`` clamps to[-448,448], rounds to nearest-even E4M3,
    preserves signed zero/subnormals and returns typed E4M3 width512 with
    no scale plane. It requires E4M3 Q and explicit Prims-TS decode.

    Default NHD output is ``[pages,64,1,width]``; explicit HND gives
    ``[pages,1,64,width]`` with identical bytes. Request HND for both writers
    to match native decode's default. Supplied ``out`` is returned by identity.
    Without slots, row i writes slot i. Incremental updates require caller out
    and CUDA int32 slots[N]; valid slots must be unique (unchecked). Invalid
    slots skip publication. Unwritten slots remain untouched and must not be
    attended before initialization. Inputs/out cannot overlap. Reusing out
    permits allocation-free graph replay. No implicit QAT derivative.
    """
    from .experimental.deepseek_v41.cache import quantize_cache

    return quantize_cache(
        x,
        format=format,
        page_size=page_size,
        out=out,
        slots=slots,
        kv_layout=kv_layout,
    )
