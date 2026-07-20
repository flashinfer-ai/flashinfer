# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
# Ported from b12x b12x/attention/mla/traits.py @ e130f195 (2026-07-17) -- one-time curated port.
# Upstream b12x is a research sandbox; this tree is the canonical home.
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


def make_unified_traits(
    model_type: int,
    compute_mode: int,
    scale_format: int,
    fp8_rope: bool | None = None,
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

    raise ValueError(
        f"unsupported model_type {model_type!r} (DSV3_2 is dropped; "
        "valid: ModelType.DSV4, ModelType.GLM_NSA)"
    )


def infer_model_type(q_head_dim: int, kv_dtype) -> tuple[int, int, int]:
    """Map (q_head_dim, kv_dtype) -> (model_type, compute_mode, scale_format).

    ``q_head_dim`` is ``d_nope + d_rope``:
      - DSV4:  448 + 64 = 512 -> (DSV4, FP8, UE8M0_BYTE)
      - GLM:   512 + 64 = 576 -> (GLM_NSA, FP8, ARBITRARY_FP32)

    Both decode targets are FP8 today; ``kv_dtype`` is accepted for the future
    BF16 const_expr branch but does not currently change the result.
    """
    if q_head_dim == 512:
        return (ModelType.DSV4, ComputeMode.FP8, ScaleFormat.UE8M0_BYTE)
    if q_head_dim == 576:
        return (ModelType.GLM_NSA, ComputeMode.FP8, ScaleFormat.ARBITRARY_FP32)
    raise ValueError(
        f"unsupported q_head_dim={q_head_dim!r}; expected 512 (DSV4) or 576 (GLM_NSA)"
    )
