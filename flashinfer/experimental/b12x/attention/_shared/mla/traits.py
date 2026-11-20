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


class ComputeMode:
    FP8 = 0
    BF16 = 1


class ScaleFormat:
    UE8M0_BYTE = 0  # DSV4: power-of-2 exponent bytes in an 8B footer.
    ARBITRARY_FP32 = 1  # GLM: arbitrary FP32 inline scales (reference.py).
    NVFP4_E4M3 = 2  # GLM/DS MLA latent: E2M1 data + E4M3 group-16 scales.


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

    raise ValueError(
        f"unsupported model_type {model_type!r} (DSV3_2 is dropped; "
        "valid: ModelType.DSV4, ModelType.DSV41, ModelType.GLM_NSA, ModelType.GLM_NEXT)"
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
    return int(model_type) in (ModelType.GLM_NSA, ModelType.GLM_NEXT)


def resolve_unplanned_traits(
    q_head_dim: int,
    kv_dtype,
    record_bytes: int,
    *,
    model_type: int | None = None,
    scale_format: int | None = None,
    fp8_rope: bool | None = None,
    latent_scale_per_token: bool = False,
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
    traits = make_unified_traits(
        model_type,
        compute_mode,
        scale_format,
        fp8_rope=fp8_rope,
        latent_scale_per_token=bool(latent_scale_per_token),
    )
    if (
        model_type == ModelType.GLM_NEXT
        or (scale_format == ScaleFormat.NVFP4_E4M3 and model_type != ModelType.DSV41)
    ) and int(record_bytes) != int(traits.kv_gmem_stride):
        if model_type != ModelType.GLM_NEXT:
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
