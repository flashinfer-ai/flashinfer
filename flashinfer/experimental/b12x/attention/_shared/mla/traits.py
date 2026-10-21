"""Per-model traits for the unified SM120 sparse-MLA CuTeDSL backend.

Pure Python (no `cute` import): this module is consumed both by the launcher
and by `smem.py`/`launch.py`, and its enums double as `cutlass.const_expr`
specialization keys (int-valued) and as `KernelCompileSpec` `KeyField` entries.

All per-model constants are transcribed VERBATIM from
`.sm120port/verified_traits.md` (DSV4 and GLM_NSA columns). DSV3.2 / POW2_FP32
are DROPPED per `.sm120port/scope_decisions.md`.
"""

from __future__ import annotations

from dataclasses import dataclass
import os


KV_FP8_ROPE_ENV = "KV_FP8_ROPE"
KV_FP8_ROPE_ENABLED = os.environ.get(KV_FP8_ROPE_ENV, "0") == "1"


def kv_fp8_rope_enabled() -> bool:
    """Return the strict runtime gate for the GLM NVFP4 RoPE sub-record."""
    return KV_FP8_ROPE_ENABLED


# ---------------------------------------------------------------------------
# const_expr specialization keys (int-valued so they can key cutlass.const_expr
# branches AND KernelCompileSpec KeyField entries). DSV3_2 / POW2_FP32 dropped.
# ---------------------------------------------------------------------------
class ModelType:
    DSV4 = 0
    GLM_NSA = 1
    # GLM-5.3 Flash absorbed MLA: the full 512-wide query is NoPE and the
    # latent cache record carries no decoupled RoPE payload.
    GLM_NEXT = 2
    DSV41 = 3  # Heterogeneous full post-RoPE SWA FP8 / main FP4 records.
    # KVarN dual-rate MLA records (arXiv:2606.03458): Hadamard+VarN+RTN,
    # per-token k_codes/v_codes planes + per-page VarN/RTN tail scales.
    KVARN_MLA = 4
    # Precision-tail decode: GLM fp8 math (ARBITRARY_FP32) over records that
    # live in a 536B-stride kvarn pool (528B fp8 content + 8B pad). The 536B
    # gmem row stride is not 16B-aligned, so the IO arm stages scalar.
    KVARN_TAIL_FP8 = 5


class ComputeMode:
    FP8 = 0
    BF16 = 1


class ScaleFormat:
    UE8M0_BYTE = 0  # DSV4: power-of-2 exponent bytes in an 8B footer.
    ARBITRARY_FP32 = 1  # GLM: arbitrary FP32 inline scales (reference.py).
    NVFP4_E4M3 = 2  # GLM/DS MLA latent: E2M1 data + E4M3 group-16 scales.
    KVARN_DUAL = 3  # KVarN: dual-bit-width RTN codes + page-tail VarN/RTN scales.


# KVarN record width -> (k_bits, v_bits). Unambiguous vs the fp8 (528B) and
# nvfp4 (304/432B) GLM records; the shared reverse lookup for launchers that
# see only the cache tensor (decode _preparation, MG prefill launcher).
KVARN_RECORD_VARIANTS = {
    536: (4, 4),
    648: (5, 5),
    600: (5, 4),
    400: (4, 2),
    264: (4, 0),  # single-plane shared-code variant (kvarn_mla_k4_g256)
}


@dataclass(frozen=True)
class UnifiedMLATraits:
    """Frozen, hashable trait bundle for one (model, compute, scale) tuple.

    Mirrors FlashInfer's ``KVCacheTraits<MT>`` + ``ComputeTraits<MT,CM>`` so the
    traced kernel can constant-fold every model-divergent point. Hashable so it
    is usable in ``functools.lru_cache`` / ``KernelCompileSpec`` keys.
    """

    model_type: int
    compute_mode: int
    scale_format: int
    d_nope: int
    d_rope: int
    d_v: int
    quant_tile: int
    num_scales: int
    n_v_chunks: int
    nt_per_warp_xv: int
    kv_gmem_stride: int
    kv_smem_stride: int
    q_nope_stride: int
    bi: int
    hpb: int
    block_threads: int
    math_threads: int
    bulk_tx_bytes: int
    v_has_rope: bool
    has_extra_cache: bool
    fp8_rope: bool
    rope_gmem_offset: int
    rope_payload_bytes: int
    rope_scale_offset: int
    # NVFP4-only per-token second-level fp32 latent scale (record bytes
    # [292, 296) of the 368-byte fp8-rope record). False keeps every existing
    # specialization (and its smem layout / PTX) byte-identical.
    latent_scale_per_token: bool = False
    # KVarN dual-rate record geometry (ScaleFormat.KVARN_DUAL only).  These are
    # int-valued so they double as const_expr keys AND KernelCompileSpec
    # KeyField entries; 0/0/0 keeps every non-KVarN specialization byte-identical.
    k_bits: int = 0
    v_bits: int = 0
    group: int = 0
    # Decode-local canonical FP8 stage for V4.1's heterogeneous source rows.
    # This is not a cache ABI change: it exists only after the gathered row is
    # owned by the math CTA.
    fp8_internal: bool = False


def make_unified_traits(
    model_type: int,
    compute_mode: int,
    scale_format: int,
    fp8_rope: bool | None = None,
    latent_scale_per_token: bool = False,
    k_bits: int = 0,
    v_bits: int = 0,
    group: int = 0,
    kvarn_row_stride: int = 0,
) -> UnifiedMLATraits:
    """Build the trait bundle for one specialization tuple.

    Constants come straight from `.sm120port/verified_traits.md`. Raises
    ``ValueError`` for the dropped DSV3.2 / POW2_FP32 combinations and for any
    (model, scale_format) mismatch.
    """
    # Resolve the process-wide cache ABI once per traits construction.  The gate
    # is deliberately orthogonal to ScaleFormat.NVFP4_E4M3 so the latent's E2M1
    # data, E4M3 group scales, and outer-scale reconstruction remain unchanged.
    fp8_rope_requested = kv_fp8_rope_enabled() if fp8_rope is None else bool(fp8_rope)

    # BF16 compute_mode is deferred (both decode targets are FP8) but the enum
    # value is accepted so the const_expr branch can exist; FP8 is the only
    # validated path today.
    if compute_mode not in (ComputeMode.FP8, ComputeMode.BF16):
        raise ValueError(f"unsupported compute_mode {compute_mode!r}")

    # FAIL-CLOSED: the per-token fp32 latent scale lives at bytes [292, 296) of
    # the NVFP4 368-byte fp8-rope record ONLY. Any other (model, scale, rope)
    # tuple has no such field, so the mode must never silently build there.
    if latent_scale_per_token and scale_format != ScaleFormat.NVFP4_E4M3:
        raise ValueError(
            "latent_scale_per_token requires ScaleFormat.NVFP4_E4M3; "
            f"got scale_format={scale_format!r}"
        )

    if model_type == ModelType.DSV41:
        if scale_format != ScaleFormat.NVFP4_E4M3 or latent_scale_per_token:
            raise ValueError("DSV41 requires native mixed FP8/FP4 without outer scales")
        if fp8_rope is not None and fp8_rope:
            raise ValueError("DSV41 quantizes all 512 post-RoPE coordinates")
        return UnifiedMLATraits(
            model_type=ModelType.DSV41,
            compute_mode=ComputeMode.BF16,
            scale_format=ScaleFormat.NVFP4_E4M3,
            d_nope=512,
            d_rope=0,
            d_v=512,
            quant_tile=64,
            num_scales=8,
            n_v_chunks=8,
            nt_per_warp_xv=1,
            kv_gmem_stride=528,
            # Both source records are staged verbatim. The private u32 at
            # +528 distinguishes SWA from main for inline BF16 dequantization.
            kv_smem_stride=544,
            q_nope_stride=520,
            bi=64,
            hpb=16,
            block_threads=288,
            math_threads=256,
            bulk_tx_bytes=64 * 528,
            v_has_rope=False,
            has_extra_cache=True,
            fp8_rope=False,
            rope_gmem_offset=0,
            rope_payload_bytes=0,
            rope_scale_offset=-1,
        )

    if model_type == ModelType.DSV4:
        if scale_format != ScaleFormat.UE8M0_BYTE:
            raise ValueError(
                "DSV4 requires ScaleFormat.UE8M0_BYTE (footer); "
                f"got scale_format={scale_format!r}"
            )
        # DSV4 column of verified_traits.md (UE8M0_BYTE, V_HAS_ROPE=true).
        return UnifiedMLATraits(
            model_type=ModelType.DSV4,
            compute_mode=compute_mode,
            scale_format=ScaleFormat.UE8M0_BYTE,
            d_nope=448,
            d_rope=64,
            d_v=512,
            quant_tile=64,
            num_scales=7,  # 448/64
            n_v_chunks=7,
            nt_per_warp_xv=1,  # 64/8/8
            kv_gmem_stride=584,  # 448 + 128 + 8
            kv_smem_stride=464,  # 448 + 16
            q_nope_stride=464,
            bi=64,  # cands/chunk
            hpb=16,  # heads/CTA
            block_threads=288,  # 9 warps
            math_threads=256,  # 8 warps
            bulk_tx_bytes=36864,  # 64*(448+128); footer excluded (16-align caveat)
            v_has_rope=True,
            has_extra_cache=True,  # DSV4 dual-cache only
            fp8_rope=False,
            rope_gmem_offset=448,
            rope_payload_bytes=128,
            rope_scale_offset=-1,
        )

    if model_type == ModelType.GLM_NSA:
        if scale_format == ScaleFormat.NVFP4_E4M3:
            # NVFP4 MLA latent cache: 256B packed E2M1 NoPE + 32B E4M3
            # group-16 scales + 16B pad + 128B BF16 RoPE. Decode stages Q-NoPE
            # as BF16 and dequants FP4 K/V in-register for BF16 QK/PV MMAs.
            use_fp8_rope = fp8_rope_requested
            if latent_scale_per_token and not use_fp8_rope:
                raise ValueError(
                    "latent_scale_per_token requires the NVFP4 fp8-rope "
                    "368-byte record (fp8_rope=True); got the 432-byte record"
                )
            return UnifiedMLATraits(
                model_type=ModelType.GLM_NSA,
                compute_mode=ComputeMode.BF16,
                scale_format=ScaleFormat.NVFP4_E4M3,
                d_nope=512,
                d_rope=64,
                d_v=512,
                quant_tile=64,
                num_scales=8,  # logical FP4 steps; storage has 32 group-16 scales
                n_v_chunks=8,
                nt_per_warp_xv=1,
                kv_gmem_stride=368 if use_fp8_rope else 432,
                kv_smem_stride=288,
                q_nope_stride=520,  # BF16 Q-NoPE smem stride: D_NOPE + 8 elems.
                bi=64,
                hpb=16,
                block_threads=288,
                math_threads=256,
                # Decode stages the unchanged 288-byte latent plus either the
                # 128-byte BF16 rope or the aligned 80-byte scale/pad/FP8 tail.
                bulk_tx_bytes=23552 if use_fp8_rope else 26624,
                v_has_rope=False,
                has_extra_cache=False,
                fp8_rope=use_fp8_rope,
                rope_gmem_offset=304,
                rope_payload_bytes=64 if use_fp8_rope else 128,
                rope_scale_offset=288 if use_fp8_rope else -1,
                latent_scale_per_token=bool(latent_scale_per_token),
            )
        if scale_format != ScaleFormat.ARBITRARY_FP32:
            raise ValueError(
                "GLM_NSA requires ScaleFormat.ARBITRARY_FP32 (inline) or "
                "ScaleFormat.NVFP4_E4M3; "
                f"got scale_format={scale_format!r}"
            )
        # GLM_NSA column of verified_traits.md (ARBITRARY_FP32, V_HAS_ROPE=false).
        return UnifiedMLATraits(
            model_type=ModelType.GLM_NSA,
            compute_mode=compute_mode,
            scale_format=ScaleFormat.ARBITRARY_FP32,
            d_nope=512,
            d_rope=64,
            d_v=512,
            quant_tile=128,
            num_scales=4,  # 512/128
            n_v_chunks=4,
            nt_per_warp_xv=2,  # 128/8/8
            kv_gmem_stride=656,
            kv_smem_stride=528,  # 512 + 4*4
            q_nope_stride=528,
            bi=64,  # cands/chunk
            hpb=16,  # heads/CTA
            block_threads=288,
            math_threads=256,
            bulk_tx_bytes=41984,  # 64*(528+128)
            v_has_rope=False,
            has_extra_cache=False,
            fp8_rope=False,
            rope_gmem_offset=528,
            rope_payload_bytes=128,
            rope_scale_offset=-1,
        )

    if model_type == ModelType.GLM_NEXT:
        if scale_format == ScaleFormat.NVFP4_E4M3:
            if fp8_rope is not None and bool(fp8_rope):
                raise ValueError("GLM_NEXT has no RoPE cache payload")
            if not latent_scale_per_token:
                raise ValueError(
                    "GLM_NEXT NVFP4 requires an inline per-token latent scale"
                )
            return UnifiedMLATraits(
                model_type=ModelType.GLM_NEXT,
                compute_mode=ComputeMode.BF16,
                scale_format=ScaleFormat.NVFP4_E4M3,
                d_nope=512,
                d_rope=0,
                d_v=512,
                quant_tile=64,
                num_scales=8,
                n_v_chunks=8,
                nt_per_warp_xv=1,
                kv_gmem_stride=304,
                kv_smem_stride=288,
                q_nope_stride=520,
                bi=64,
                hpb=16,
                block_threads=288,
                math_threads=256,
                bulk_tx_bytes=64 * 288,
                v_has_rope=False,
                has_extra_cache=False,
                fp8_rope=False,
                rope_gmem_offset=304,
                rope_payload_bytes=0,
                rope_scale_offset=-1,
                latent_scale_per_token=True,
            )
        if scale_format != ScaleFormat.ARBITRARY_FP32:
            raise ValueError(
                "GLM_NEXT requires ScaleFormat.ARBITRARY_FP32 (inline) or "
                "ScaleFormat.NVFP4_E4M3; "
                f"got scale_format={scale_format!r}"
            )
        if compute_mode != ComputeMode.FP8:
            raise ValueError(
                "GLM_NEXT currently requires ComputeMode.FP8; "
                f"got compute_mode={compute_mode!r}"
            )
        if fp8_rope is not None and bool(fp8_rope):
            raise ValueError("GLM_NEXT has no RoPE cache payload")
        # GLM-5.3 Flash keeps the 512-dimensional absorbed latent used by the
        # GLM sparse-MLA math, but qk_rope_head_dim is zero. Its cache record is
        # therefore exactly the 512 E4M3 latent bytes plus four inline fp32
        # group scales. No RoPE suffix is present.
        return UnifiedMLATraits(
            model_type=ModelType.GLM_NEXT,
            compute_mode=compute_mode,
            scale_format=ScaleFormat.ARBITRARY_FP32,
            d_nope=512,
            d_rope=0,
            d_v=512,
            quant_tile=128,
            num_scales=4,
            n_v_chunks=4,
            nt_per_warp_xv=2,
            kv_gmem_stride=528,
            kv_smem_stride=528,
            q_nope_stride=528,
            bi=64,
            hpb=16,
            block_threads=288,
            math_threads=256,
            bulk_tx_bytes=33792,  # 64 * 528; there is no RoPE transaction.
            v_has_rope=False,
            has_extra_cache=False,
            fp8_rope=False,
            rope_gmem_offset=528,
            rope_payload_bytes=0,
            rope_scale_offset=-1,
        )

    if model_type == ModelType.KVARN_MLA:
        # KVarN dual-rate MLA record (arXiv:2606.03458): per-token k_codes /
        # v_codes planes at independent bit widths + a per-page (group-token)
        # tail carrying VarN channel scale + per-channel RTN scale/zero-point.
        # The record is staged verbatim into the kv_fp8 row; decode math
        # unpacks the bit planes and dequants to BF16 in-register (mirroring
        # the GLM_NSA NVFP4 arm), reading the page-tail scales from gmem.
        import math as _math

        k_bits = int(k_bits)
        v_bits = int(v_bits)
        group = int(group)
        if k_bits not in (2, 3, 4, 5, 6, 7, 8) or (
            v_bits != 0 and v_bits not in (2, 3, 4, 5, 6, 7, 8)
        ):
            raise ValueError(
                f"KVARN_MLA bit widths must be in [2, 8] (v=0 selects the "
                f"single-plane variant); got k={k_bits} v={v_bits}"
            )
        if group <= 0:
            raise ValueError(f"KVARN_MLA group must be positive; got {group!r}")
        latent_dim = 512
        k_code_bytes = _math.ceil(latent_dim * k_bits / 8)
        v_code_bytes = _math.ceil(latent_dim * v_bits / 8)
        # Mod-33 rule (mirrors vllm KvarnMlaLayout / _kvarn_writer): dual
        # variants keep the C4 alignment gap >= the 3072B per-page tile-scale
        # tail; single-plane variants (v_bits == 0) skip the bump -- records
        # end exactly on the C4 boundary (264 = 8 C4 pages for k4) and the
        # FP8 index tail starts flush.
        content = k_code_bytes + v_code_bytes + 6
        record_bytes = (content + 7) // 8 * 8
        if v_bits != 0:
            while record_bytes % 33 == 0 or record_bytes % 33 > 21:
                record_bytes += 8
        # The KV stage is double-buffered at bi x record_bytes x 2 bytes.
        # Wide records (k5v5: 648 B -> 107,392 B total at bi=64) exceed the
        # 99 KB SM120_SMEM_CARVEOUT_BYTES design budget, which the kvarn
        # assert (smem.py) replaces with the 108 KB KVARN cap. bi stays 64:
        # the kernel has a hardcoded bi=64 assumption (bi=32 reproduce is an
        # illegal __shared__ write; see kvarn-mla-design.md section 9).
        bi = 64
        return UnifiedMLATraits(
            model_type=ModelType.KVARN_MLA,
            compute_mode=ComputeMode.BF16,
            scale_format=ScaleFormat.KVARN_DUAL,
            d_nope=latent_dim,
            d_rope=0,
            d_v=latent_dim,
            quant_tile=64,
            num_scales=8,  # 512 / 64 logical BF16 K steps
            n_v_chunks=8,
            nt_per_warp_xv=1,
            kv_gmem_stride=kvarn_row_stride or record_bytes,
            kv_smem_stride=record_bytes,
            q_nope_stride=520,  # BF16 Q-NoPE smem stride: D_NOPE + 8 elems.
            bi=bi,
            hpb=16,
            block_threads=288,
            math_threads=256,
            # The record width is not 16B-aligned, so the IO arm scalar-loads
            # it (no cp.async.bulk transaction); bulk_tx_bytes stays 0.
            bulk_tx_bytes=0,
            v_has_rope=False,
            has_extra_cache=False,
            fp8_rope=False,
            rope_gmem_offset=record_bytes,
            rope_payload_bytes=0,
            rope_scale_offset=-1,
            k_bits=k_bits,
            v_bits=v_bits,
            group=group,
        )

    if model_type == ModelType.KVARN_TAIL_FP8:
        # Precision-tail decode arm: the GLM fp8 MATH (scale_format=
        # ARBITRARY_FP32, d_rope=0) over fp8 records stored in-place in
        # kvarn pool slots. Record content [0,528) is the stock GLM fp8
        # record (512 E4M3 latent + 4 inline fp32 group scales); the rest
        # of the slot is pad. The gmem row stride is the POOL's slot
        # width for this dtype -- kvarn_record_bytes(k_bits, v_bits):
        # 536 for k4v4, 600 for k5v4 -- NOT a hardcoded k4v4 536. The
        # row stride alternates 0/8 mod 16, so the IO arm stages it
        # SCALAR (bulk_tx_bytes=0) into the 528B smem kv_fp8 row the GLM
        # fp8 math expects (kv_smem_stride stays 528).
        import math as _math

        if not int(k_bits) or not int(v_bits):
            raise ValueError(
                "KVARN_TAIL_FP8 requires kvarn_k_bits/kvarn_v_bits on the "
                "Caps: the gmem row stride is the kvarn pool's slot width "
                "for the cache dtype (536 for k4v4, 600 for k5v4)"
            )
        tail_record = (
            (
                _math.ceil(512 * int(k_bits) / 8)
                + _math.ceil(512 * int(v_bits) / 8)
                + 6
                + 7
            )
            // 8
            * 8
        )
        while tail_record % 33 == 0 or tail_record % 33 > 21:
            tail_record += 8
        if tail_record < 528:
            raise ValueError(
                "KVARN_TAIL_FP8 needs pool slots of >= 528B to hold the "
                f"fp8 record; k={k_bits} v={v_bits} gives {tail_record}B "
                "(k4v2=400B cannot store the tail in place)"
            )
        if scale_format != ScaleFormat.ARBITRARY_FP32:
            raise ValueError(
                "KVARN_TAIL_FP8 requires ScaleFormat.ARBITRARY_FP32; "
                f"got scale_format={scale_format!r}"
            )
        if compute_mode != ComputeMode.FP8:
            raise ValueError(
                "KVARN_TAIL_FP8 requires ComputeMode.FP8; "
                f"got compute_mode={compute_mode!r}"
            )
        return UnifiedMLATraits(
            model_type=ModelType.KVARN_TAIL_FP8,
            compute_mode=compute_mode,
            scale_format=ScaleFormat.ARBITRARY_FP32,
            d_nope=512,
            d_rope=0,
            d_v=512,
            quant_tile=128,
            num_scales=4,
            n_v_chunks=4,
            nt_per_warp_xv=2,
            kv_gmem_stride=tail_record,
            kv_smem_stride=528,
            q_nope_stride=528,
            bi=64,
            hpb=16,
            block_threads=288,
            math_threads=256,
            bulk_tx_bytes=0,  # scalar staging (536B row stride not 16B-aligned)
            k_bits=int(k_bits),
            v_bits=int(v_bits),
            group=int(group),
            v_has_rope=False,
            has_extra_cache=False,
            fp8_rope=False,
            rope_gmem_offset=528,
            rope_payload_bytes=0,
            rope_scale_offset=-1,
        )

    raise ValueError(
        f"unsupported model_type {model_type!r} (DSV3_2 is dropped; "
        "valid: ModelType.DSV4, ModelType.DSV41, ModelType.GLM_NSA, "
        "ModelType.GLM_NEXT, ModelType.KVARN_MLA)"
    )


def infer_model_type(
    q_head_dim: int,
    kv_dtype,
    *,
    model_type: int | None = None,
) -> tuple[int, int, int]:
    """Map (q_head_dim, kv_dtype) -> (model_type, compute_mode, scale_format).

    ``q_head_dim`` is ``d_nope + d_rope``:
      - DSV4:  448 + 64 = 512 -> (DSV4, FP8, UE8M0_BYTE)
      - GLM:   512 + 64 = 576 -> (GLM_NSA, FP8, ARBITRARY_FP32)
      - GLM_NEXT: 512 + 0 = 512 -> (GLM_NEXT, FP8, ARBITRARY_FP32)
      - DSV41: 512 post-RoPE -> (DSV41, BF16, NVFP4_E4M3 mixed-source math)

    The 512-wide contracts are ambiguous by shape. Existing callers retain
    DSV4 as the compatibility default; GLM_NEXT callers must pass its explicit
    ``model_type`` identity.

    Both decode targets are FP8 today; ``kv_dtype`` is accepted for the future
    BF16 const_expr branch but does not currently change the result.
    """
    if model_type is not None:
        model_type = int(model_type)
        expected_q_head_dim = {
            ModelType.DSV4: 512,
            ModelType.GLM_NSA: 576,
            ModelType.GLM_NEXT: 512,
            ModelType.DSV41: 512,
            ModelType.KVARN_MLA: 512,
            ModelType.KVARN_TAIL_FP8: 512,
        }.get(model_type)
        if expected_q_head_dim is None:
            raise ValueError(f"unsupported explicit model_type={model_type!r}")
        if q_head_dim != expected_q_head_dim:
            raise ValueError(
                f"model_type={model_type} requires q_head_dim={expected_q_head_dim}; "
                f"got {q_head_dim}"
            )
        if model_type == ModelType.DSV41:
            return (ModelType.DSV41, ComputeMode.BF16, ScaleFormat.NVFP4_E4M3)
        if model_type == ModelType.DSV4:
            return (ModelType.DSV4, ComputeMode.FP8, ScaleFormat.UE8M0_BYTE)
        if model_type == ModelType.KVARN_MLA:
            return (ModelType.KVARN_MLA, ComputeMode.BF16, ScaleFormat.KVARN_DUAL)
        if model_type == ModelType.KVARN_TAIL_FP8:
            return (
                ModelType.KVARN_TAIL_FP8,
                ComputeMode.FP8,
                ScaleFormat.ARBITRARY_FP32,
            )
        return (model_type, ComputeMode.FP8, ScaleFormat.ARBITRARY_FP32)

    if q_head_dim == 512:
        return (ModelType.DSV4, ComputeMode.FP8, ScaleFormat.UE8M0_BYTE)
    if q_head_dim == 576:
        return (ModelType.GLM_NSA, ComputeMode.FP8, ScaleFormat.ARBITRARY_FP32)
    raise ValueError(
        f"unsupported q_head_dim={q_head_dim!r}; expected 512 (DSV4 or explicit "
        "GLM_NEXT) or 576 (GLM_NSA)"
    )


def is_glm_model_type(model_type: int) -> bool:
    """Return whether ``model_type`` uses the GLM latent-cache family."""
    return int(model_type) in (
        ModelType.GLM_NSA,
        ModelType.GLM_NEXT,
        ModelType.KVARN_MLA,
        ModelType.KVARN_TAIL_FP8,
    )


def resolve_unplanned_traits(
    q_head_dim: int,
    kv_dtype,
    record_bytes: int,
    *,
    model_type: int | None = None,
    scale_format: int | None = None,
    fp8_rope: bool | None = None,
    latent_scale_per_token: bool = False,
    k_bits: int = 0,
    v_bits: int = 0,
    group: int = 0,
) -> UnifiedMLATraits:
    """Resolve the compatibility-only direct-launch cache recipe.

    Serving integrations must provide the immutable traits stored by their
    sparse-MLA plan. This resolver exists for low-level direct-launch callers
    that have no plan artifact; their concrete record width is therefore the
    only available source for the legacy GLM_NSA RoPE-format choice.
    """
    model_type, compute_mode, inferred_scale_format = infer_model_type(
        int(q_head_dim), kv_dtype, model_type=model_type
    )
    scale_format = (
        int(inferred_scale_format) if scale_format is None else int(scale_format)
    )
    if is_glm_model_type(model_type) and scale_format == ScaleFormat.NVFP4_E4M3:
        compute_mode = ComputeMode.BF16
        if model_type == ModelType.GLM_NEXT:
            if fp8_rope not in (None, False):
                raise ValueError("GLM_NEXT has no RoPE cache payload")
            fp8_rope = False
            latent_scale_per_token = True
        elif fp8_rope is None:
            if int(record_bytes) not in (368, 432):
                raise ValueError(
                    "NVFP4 cache record must be 368 or 432 bytes, got "
                    f"{int(record_bytes)}"
                )
            fp8_rope = int(record_bytes) == 368
    if model_type in (ModelType.KVARN_MLA, ModelType.KVARN_TAIL_FP8):
        # Derive the dual-rate geometry from the concrete record width when the
        # caller did not spell it out (536->k4v4, 648->k5v5, 600->k5v4,
        # 400->k4v2, 272->k4 single-plane; the mod-33 padded widths match
        # KvarnMlaLayout / _kvarn_writer). The tail arm stores fp8 content in
        # the SAME pool slots, so its slot width follows the same map. v_bits=0
        # is a legitimate KVARN_MLA spelling (single-plane), so only k_bits==0
        # triggers derivation; a KVARN_TAIL_FP8 caller with v_bits=0 is
        # rejected later by the recipe validation.
        if int(k_bits) == 0:
            import math as _math

            latent_dim = 512
            for kb, vb in ((4, 4), (5, 5), (5, 4), (4, 2), (4, 0)):
                rc = (
                    _math.ceil(latent_dim * kb / 8)
                    + _math.ceil(latent_dim * vb / 8)
                    + 6
                )
                rb = (rc + 7) // 8 * 8
                if vb != 0:
                    while rb % 33 == 0 or rb % 33 > 21:
                        rb += 8
                if rb == int(record_bytes):
                    k_bits = kb
                    v_bits = vb
                    break
            else:
                raise ValueError(
                    f"KVARN_MLA record width {int(record_bytes)} matches no "
                    "launched variant (536/648/600/400/272)"
                )
        if int(group) == 0:
            group = 256
    traits = make_unified_traits(
        model_type,
        compute_mode,
        scale_format,
        fp8_rope=fp8_rope,
        latent_scale_per_token=bool(latent_scale_per_token),
        k_bits=int(k_bits),
        v_bits=int(v_bits),
        group=int(group),
    )
    if (
        model_type == ModelType.GLM_NEXT
        or model_type == ModelType.KVARN_MLA
        or model_type == ModelType.KVARN_TAIL_FP8
        or (scale_format == ScaleFormat.NVFP4_E4M3 and model_type != ModelType.DSV41)
    ) and int(record_bytes) != int(traits.kv_gmem_stride):
        if model_type not in (ModelType.GLM_NEXT, ModelType.KVARN_TAIL_FP8):
            raise ValueError(
                "NVFP4 cache record width disagrees with fp8_rope_override: "
                f"got {int(record_bytes)} bytes, expected "
                f"{int(traits.kv_gmem_stride)}"
            )
        raise ValueError(
            "sparse MLA cache record width does not match its recipe: "
            f"got {int(record_bytes)}, expected {int(traits.kv_gmem_stride)}"
        )
    return traits
