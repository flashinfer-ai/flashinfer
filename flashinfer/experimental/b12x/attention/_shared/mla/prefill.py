"""Unified SM120 sparse-MLA *prefill* DISPATCHER.

``run_unified_prefill`` is the prefill front door: it infers the model/compute/
scale traits, normalizes inputs, and routes to the FlashInfer-shaped multi-group
(MG) prefill kernel in ``prefill_mg.py`` (``run_unified_prefill_mg``). There is NO
decode-reuse fallback -- an unsupported shape HARD-FAILS (raise-not-fallback,
matching upstream).

Supported (MG) shapes:
  * DSV4/GLM single-cache: heads % 8 == 0
  * DSV4 single-cache: topk in {512, 1024, 2048} (FP8-QK) or 128 (BF16-QK)
    with 8-aligned heads split into a paired-head MG prefix plus optional
    single-group tails.
  * DSV4 dual-cache (extra/indexed tokens): topk in {128, 512} (BF16-QK) or
    {1024, 2048} (FP8-QK), heads % 8 == 0, using the same head
    partitioning, and pbs_extra in {2, 64}.
  * GLM_NSA: topk in {512, 1024, 2048}
  * GLM_NEXT: topk in {512, 1024, 2048, 2051, 2112}; 2112 is an
    alignment-only container whose per-row ``topk_length`` remains 2051.
Anything else (other topk, unsupported heads, GLM dual, etc.) raises ValueError.
DSV4 + GLM DECODE kernels are untouched and stay byte-identical.
"""

from __future__ import annotations

import os
from dataclasses import replace

import torch
from b12x._lib.compile_plan import compile_only_launches_enabled

from .traits import (
    ComputeMode,
    ModelType,
    ScaleFormat,
    UnifiedMLATraits,
    is_glm_model_type,
    resolve_unplanned_traits,
)

# DSV4 compressed contract head dim (q_nope 448 + q_rope 64).
_DSV4_HEAD_DIM = 512
# GLM_NSA uncompressed contract head dim (q_nope 512 + q_rope 64).
_GLM_HEAD_DIM = 576
# GLM per-token packed cache record (reference.pack_mla_kv_cache_reference).
_GLM_KV_GMEM_STRIDE = 656


def _cache_block_stride_bytes(
    cache: torch.Tensor,
    *,
    page_size: int,
    model_type: ModelType,
    record_bytes: int | None = None,
) -> int:
    from b12x.attention._shared.mla.compressed_reference import (
        COMPRESSED_SPARSE_MLA_BYTES_PER_TOKEN,
    )

    if is_glm_model_type(model_type) or model_type == ModelType.DSV41:
        # GLM-family per-token contiguous record: 656B (ARBITRARY_FP32) or
        # 432B (NVFP4_E4M3). ``record_bytes`` comes from traits.kv_gmem_stride.
        rec = int(record_bytes) if record_bytes is not None else _GLM_KV_GMEM_STRIDE
        expected = int(page_size) * rec
    else:
        expected = int(page_size) * COMPRESSED_SPARSE_MLA_BYTES_PER_TOKEN
    if is_glm_model_type(model_type) and cache.is_contiguous():
        return expected
    # The runtime page stride is part of the packed/padded cache contract.
    if cache.ndim >= 2:
        stride = int(cache.stride(0)) * int(cache.element_size())
        if stride < expected:
            raise ValueError(
                f"SM120 sparse MLA cache block stride {stride} is smaller than "
                f"page payload {expected}"
            )
        return stride
    return expected


def _mg_head_partitions(heads: int, hpb: int = 16) -> tuple[tuple[int, int, int], ...]:
    """Return ``(mg_n_hg, active_heads, head_offset)`` partitions for 8-aligned heads."""
    heads = int(heads)
    hpb = int(hpb)
    if heads <= 0 or heads % (hpb // 2) != 0:
        return ()

    parts: list[tuple[int, int, int]] = []
    paired_width = 2 * hpb
    paired_heads = (heads // paired_width) * paired_width
    if paired_heads:
        parts.append((2, paired_heads, 0))

    offset = paired_heads
    rem = heads - paired_heads
    if rem >= hpb:
        parts.append((1, hpb, offset))
        offset += hpb
        rem -= hpb
    if rem:
        parts.append((1, rem, offset))
    return tuple(parts)


_TOPK_CONTAINERS = (512, 1024, 2048)
_TOPK_EXACT = frozenset({128, 512, 1024, 2048, 2051, 2112})


def _topk_container(model_type: int, topk: int) -> int:
    """Return the DSV4 MG container width for a runtime top-k row.

    GLM sparse-MLA plans require the caller-owned index row to match the planned
    width. Live valid counts are runtime metadata, so the dispatcher must not
    replace a bound GLM row with an allocating padded tensor. Unsupported GLM
    widths remain unchanged here and fail the model-specific dispatch gate.
    DSV4 accepts the padded fixed-container representation and is the model
    family that requires sub-container widening.
    """
    if model_type != ModelType.DSV4:
        return topk
    if topk in _TOPK_EXACT:
        return topk
    for container in _TOPK_CONTAINERS:
        if topk < container:
            return container
    return topk


def run_unified_prefill(
    *,
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    topk_indices: torch.Tensor,
    sm_scale: float,
    page_block_size: int,
    topk_length: torch.Tensor | None = None,
    attn_sink: torch.Tensor | None = None,
    output: torch.Tensor | None = None,
    lse_out: torch.Tensor | None = None,
    stride_kv_block: int | None = None,
    extra_kv_cache: torch.Tensor | None = None,
    extra_indices: torch.Tensor | None = None,
    extra_topk_length: torch.Tensor | None = None,
    extra_page_block_size: int | None = None,
    stride_extra_kv_block: int | None = None,
    workspace=None,
    scale_format: int | None = None,
    model_type: int | None = None,
    latent_scale: float = 1.0,
    fp8_rope: bool | None = None,
    latent_scale_per_token: bool = False,
    traits_override: UnifiedMLATraits | None = None,
    prepared: object | None = None,
    mg_enabled: bool | None = None,
):
    """Unified SM120 sparse-MLA single-pass prefill -> BF16 O + base-2 LSE.

    Pure DISPATCHER: validates the contract, infers traits, and routes to the
    FlashInfer-shaped multi-group (MG) prefill kernel (``run_unified_prefill_mg``).
    Unsupported shapes HARD-FAIL (raise-not-fallback); there is no decode-reuse path.

    Routes DSV4 (q_head_dim==512, UE8M0 footer, V_HAS_ROPE), GLM_NSA
    (q_head_dim==576, ARBITRARY_FP32 inline scales, V==nope), and explicit
    GLM_NEXT (q_head_dim==512, ARBITRARY_FP32 inline scales, no RoPE) through the SAME kernel
    via the traits const_expr branches (model_type/scale_format/v_has_rope/
    nt_per_warp_xv), exactly like the decode launcher. DSV4 additionally supports a
    DUAL-CACHE union (extra_kv_cache / extra_indices / extra_topk_length /
    extra_page_block_size): the CTA attends over the UNION of the MAIN topk cache and
    the EXTRA cache in ONE online softmax (num_main_tiles main chunks then the extra
    chunks). The extra cache is DSV4-only (GLM has no extra section -> RAISE).

    Args:
      q:            (T, heads, D_QK) bf16. D_QK 512 (DSV4/explicit GLM_NEXT)
                    or 576 (GLM_NSA).
      kv_cache:     flat uint8 MAIN KV cache (reshaped to 1-D).
      topk_indices: (T, topk) int32 flat slot ids (-1 = invalid sentinel).
      sm_scale:     caller-owned physical softmax scale. GLM_NEXT uses
                    256**-0.5 even though its absorbed query width is 512.
      page_block_size: tokens per MAIN KV block (64 for DSV4/GLM).
      topk_length:  optional (T,) int32 per-token MAIN valid length; entries past it
                    are masked. Defaults to full ``topk`` for every token.
      attn_sink:    optional (heads,) fp32 per-head natural-log sink, folded into
                    the normalizer + base-2 LSE (FlashMLA V4).
      output:       optional pre-allocated (T, heads, D_V) bf16 output (else made).
      lse_out:      optional pre-allocated (T, heads) f32 base-2 LSE (else made).
      stride_kv_block: per-block gmem byte stride for the MAIN cache. Derived from
                    page_block_size + model_type when omitted.
      extra_kv_cache / extra_indices / extra_topk_length / extra_page_block_size:
                    DSV4 dual-cache EXTRA pool (all-or-none; partial trio RAISEs).
      stride_extra_kv_block: EXTRA per-block byte stride (derived when omitted).
      latent_scale: per-layer outer scale for the reconstructed NVFP4 latent.
      workspace:    unused (prefill is single-pass, no split/merge workspace);
                    accepted for launcher-signature symmetry.

    Returns (O[T, heads, D_V=512] bf16, lse[T, heads] f32 base-2).
    """
    del workspace  # prefill is single-pass; no split/merge workspace needed.

    q_head_dim = int(q.shape[-1])
    if q_head_dim not in (_DSV4_HEAD_DIM, _GLM_HEAD_DIM):
        # Genuinely-unsupported contract -> error like upstream (infer_model_type
        # ICHECKs d_qk in {512, 576}). NOT a legacy fallback.
        raise ValueError(
            "SM120 sparse MLA prefill supports q_head_dim=512 (DSV4 or "
            "explicit GLM_NEXT) or q_head_dim=576 (GLM_NSA); "
            f"got q_head_dim={q_head_dim}"
        )

    num_tokens, heads, _ = q.shape
    hpb = 16
    if heads % (hpb // 2) != 0:
        # 8-head tails use MG n_hg==1 + valid_hpb=8: one M=16 tile with 8 real
        # Q rows and 8 zero-padded rows. Other sub-8 head counts remain
        # unsupported -> RAISE (not legacy).
        raise ValueError(
            f"SM120 sparse MLA prefill requires heads divisible by {hpb // 2}, got {heads}"
        )

    if traits_override is not None:
        traits = traits_override
        model_type = int(traits.model_type)
        compute_mode = int(traits.compute_mode)
        scale_format = int(traits.scale_format)
    else:
        traits = resolve_unplanned_traits(
            q_head_dim,
            kv_cache.dtype,
            int(kv_cache.shape[-1]),
            model_type=model_type,
            scale_format=scale_format,
            fp8_rope=fp8_rope,
            latent_scale_per_token=bool(latent_scale_per_token),
        )
        model_type = int(traits.model_type)
        compute_mode = int(traits.compute_mode)
        scale_format = int(traits.scale_format)
    d_v = int(traits.d_v)

    # ── DSV4 dual-cache: validate the extra trio (all-or-none) and that it is DSV4. ──
    has_extra = (
        extra_kv_cache is not None
        or extra_indices is not None
        or extra_topk_length is not None
    )
    if has_extra:
        if (
            extra_kv_cache is None
            or extra_indices is None
            or extra_page_block_size is None
        ):
            raise ValueError(
                "SM120 sparse MLA prefill dual-cache requires extra_kv_cache, "
                "extra_indices, and extra_page_block_size together (partial extra "
                "trio is unsupported, matching upstream sparse_mla_sm120.cu:171-174)"
            )
        if model_type not in (ModelType.DSV4, ModelType.DSV41):
            raise ValueError(
                "SM120 sparse MLA prefill dual-cache (extra tokens) is DSV4-only "
                "(q_head_dim==512); GLM/DSV3.2 has no extra cache"
            )

    topk = int(topk_indices.shape[1])

    device = q.device
    if topk_length is None:
        topk_length = torch.full((num_tokens,), topk, dtype=torch.int32, device=device)
    else:
        topk_length = topk_length.to(device=device, dtype=torch.int32).contiguous()

    container = _topk_container(model_type, topk)
    if container != topk:
        # DSV4 short sequences can clamp runtime top-k below index_topk (for
        # example, 192 for a 192-token prefill). Its MG kernels take fixed
        # containers, so widen the row with the invalid sentinel while retaining
        # the per-token valid length.
        topk_indices = torch.nn.functional.pad(
            topk_indices, (0, container - topk), value=-1
        )
        topk = container

    if stride_kv_block is None:
        stride_kv_block = _cache_block_stride_bytes(
            kv_cache,
            page_size=int(page_block_size),
            model_type=model_type,
            record_bytes=int(traits.kv_gmem_stride),
        )

    q = q.contiguous()
    topk_indices = topk_indices.contiguous()
    if output is None:
        output = torch.empty(
            (num_tokens, heads, d_v), dtype=torch.bfloat16, device=device
        )
    if lse_out is None:
        lse_out = torch.empty((num_tokens, heads), dtype=torch.float32, device=device)

    def _run_partitioned_mg(
        *,
        compute_mode: int,
        model_type: int,
        scale_format: int,
        extra_kv_cache=None,
        extra_indices=None,
        extra_topk_length=None,
        extra_page_block_size=None,
        stride_extra_kv_block=None,
    ):
        from .prefill_mg import run_unified_prefill_mg

        partitions = _mg_head_partitions(heads, hpb)
        if model_type == ModelType.DSV41:
            # DSV4.1's 544-byte records permit one 16-head group per CTA.
            # One launch covers the full 16-aligned prefix; only an 8-head
            # remainder needs a separate valid-HPB launch.
            prefix_heads = heads - heads % hpb
            partitions = ((1, prefix_heads, 0),) if prefix_heads else ()
            if prefix_heads != heads:
                partitions += ((1, heads - prefix_heads, prefix_heads),)
        if not partitions:
            raise ValueError(
                f"SM120 sparse MLA prefill requires heads divisible by {hpb // 2}, got {heads}"
            )
        launches = []
        for partition_index, (mg_n_hg, active_heads, head_offset) in enumerate(
            partitions
        ):
            kwargs = dict(
                q=q,
                kv_cache=kv_cache,
                topk_indices=topk_indices,
                sm_scale=sm_scale,
                latent_scale=latent_scale,
                page_block_size=page_block_size,
                topk_length=topk_length,
                attn_sink=attn_sink,
                output=output,
                lse_out=lse_out,
                stride_kv_block=stride_kv_block,
                compute_mode=compute_mode,
                mg_n_hg=mg_n_hg,
                model_type=model_type,
                scale_format=scale_format,
                fp8_rope=bool(traits.fp8_rope),
                latent_scale_per_token=bool(traits.latent_scale_per_token),
                traits_override=replace(traits, compute_mode=compute_mode),
            )
            if prepared is not None:
                kwargs["prepared"] = (
                    prepared[partition_index]
                    if isinstance(prepared, tuple) else prepared
                )
            if extra_kv_cache is not None:
                kwargs.update(
                    extra_kv_cache=extra_kv_cache,
                    extra_indices=extra_indices,
                    extra_topk_length=extra_topk_length,
                    extra_page_block_size=extra_page_block_size,
                    stride_extra_kv_block=stride_extra_kv_block,
                )
            if len(partitions) > 1 or active_heads != heads or head_offset != 0:
                kwargs.update(active_heads=active_heads, head_offset=head_offset)
            launch = run_unified_prefill_mg(**kwargs)
            if launch is not None:
                launches.append(launch)
        if compile_only_launches_enabled():
            return tuple(launches)
        return output, lse_out

    if model_type == ModelType.DSV41:
        return _run_partitioned_mg(
            compute_mode=ComputeMode.BF16,
            model_type=model_type,
            scale_format=ScaleFormat.NVFP4_E4M3,
            extra_kv_cache=extra_kv_cache,
            extra_indices=extra_indices,
            extra_topk_length=extra_topk_length,
            extra_page_block_size=extra_page_block_size,
            stride_extra_kv_block=stride_extra_kv_block,
        )

    # ── MG (multi-head-group) gate ────────────────────────────────────────────
    # DSV4 main-cache. The MG kernel is parameterized by the head-group count
    # ``mg_n_hg`` (a const_expr; one HPB=16 head group per group):
    #   * paired 32-head prefix -> MG_N_HG=2
    #   * optional 16-head tail -> MG_N_HG=1
    #   * optional 8-head tail  -> MG_N_HG=1 + VALID_HPB=8
    # The dispatcher partitions every 8-aligned head count into that sequence.
    #
    # Within each group count, two FlashInfer-shaped QK specializations route:
    #   * topk in {512, 1024, 2048}  -> FP8-QK MG  (block-scaled E4M3 QK).
    #   * topk == 128                -> BF16-QK MG (S0 skips the FP8 Q-quant
    #     prologue; S1 dequants K e4m3->bf16 inline and runs a bf16 m16n8k16 QK;
    #     XV stays FP8). FlashInfer routes topk==128 to this BF16-QK kernel (the
    #     small K-loop where the Q-quant prologue would dominate); it lands a
    #     TIGHTER numeric (no Q-quant loss) than FP8.
    _mg_enabled = (
        os.environ.get("B12X_MLA_SM120_PREFILL_MG", "1") not in (
            "0",
            "false",
            "False",
            "off",
        )
        if mg_enabled is None else bool(mg_enabled)
    )
    # ── GLM (ARBITRARY_FP32, q=576, v_has_rope=False) MG gate ──────────────────
    # GLM has the SAME FlashInfer MG head-group structure as DSV4 (one CTA fuses
    # MG_N_HG HPB head groups, sharing the KV gather), differing only in the math
    # arms (post-MMA fp32 QK scale, raw-V + 2-pass-W XV, no XV-rope) and the
    # 656/528 KV geometry -- all const_expr-selected in the SAME MG kernel. Route
    # GLM prefill uses MG for every supported shard. The TP8 / eight-local-head
    # case keeps VALID_HPB=8; its PV path packs HIGH and LOW residual rows into
    # the two halves of the otherwise half-empty m16 tile:
    #   8 heads               -> MG_N_HG=1 + VALID_HPB=8
    #   paired 32-head prefix -> MG_N_HG=2
    #   optional 16-head tail -> MG_N_HG=1
    #   optional 8-head tail  -> MG_N_HG=1 + VALID_HPB=8
    # topk in {512,1024,2048}.
    _mg_glm = (
        _mg_enabled
        and not has_extra
        and is_glm_model_type(model_type)
        and scale_format == ScaleFormat.ARBITRARY_FP32
    )
    glm_topk_supported = topk in (512, 1024, 2048) or (
        model_type == ModelType.GLM_NEXT and topk in (2051, 2112)
    )
    if _mg_glm and glm_topk_supported:
        return _run_partitioned_mg(
            compute_mode=ComputeMode.FP8,
            model_type=model_type,
            scale_format=ScaleFormat.ARBITRARY_FP32,
        )
    # ── NVFP4 (E2M1 + E4M3 group-16, GLM-family) MG gate ───────────────────────
    # Same MG head-group structure as GLM; the math arms are the BF16-QK path
    # with native in-register E2M1/E4M3 dequant (the same math as the validated
    # NVFP4 decode). topk==128 additionally routes here (BF16-QK shape).
    _mg_nvfp4 = (
        _mg_enabled
        and not has_extra
        and is_glm_model_type(model_type)
        and scale_format == ScaleFormat.NVFP4_E4M3
    )
    nvfp4_topk_supported = topk in (128, 512, 1024, 2048) or (
        model_type == ModelType.GLM_NEXT and topk in (2051, 2112)
    )
    if _mg_nvfp4 and nvfp4_topk_supported:
        return _run_partitioned_mg(
            compute_mode=ComputeMode.BF16,
            model_type=model_type,
            scale_format=ScaleFormat.NVFP4_E4M3,
        )
    if (
        _mg_enabled
        and model_type == ModelType.DSV41
        and scale_format == ScaleFormat.NVFP4_E4M3
        and topk in (128, 512, 1024, 2048)
    ):
        return _run_partitioned_mg(
            compute_mode=ComputeMode.BF16,
            model_type=model_type,
            scale_format=ScaleFormat.NVFP4_E4M3,
        )
    _mg_base = (
        _mg_enabled
        and not has_extra
        and model_type == ModelType.DSV4
        and compute_mode == ComputeMode.FP8  # infer_model_type always FP8 for DSV4
        and scale_format == 0
    )
    if _mg_base and topk in (512, 1024, 2048):
        return _run_partitioned_mg(
            compute_mode=ComputeMode.FP8,
            model_type=ModelType.DSV4,
            scale_format=ScaleFormat.UE8M0_BYTE,
        )
    if _mg_base and topk == 128:
        return _run_partitioned_mg(
            compute_mode=ComputeMode.BF16,
            model_type=ModelType.DSV4,
            scale_format=ScaleFormat.UE8M0_BYTE,
        )

    # DSV4 dual-cache uses BF16-QK for the 128-token text window and the
    # 512-token image window, and FP8-QK for 1024/2048-wide main sections.
    # Both paths retain the main/extra union in one online softmax.
    if has_extra:
        if model_type == ModelType.DSV4 and int(topk) in (128, 512):
            return _run_partitioned_mg(
                compute_mode=ComputeMode.BF16,
                model_type=ModelType.DSV4,
                scale_format=ScaleFormat.UE8M0_BYTE,
                extra_kv_cache=extra_kv_cache,
                extra_indices=extra_indices,
                extra_topk_length=extra_topk_length,
                extra_page_block_size=extra_page_block_size,
                stride_extra_kv_block=stride_extra_kv_block,
            )
        if (
            _mg_enabled
            and model_type == ModelType.DSV4
            and compute_mode == ComputeMode.FP8
            and scale_format == ScaleFormat.UE8M0_BYTE
            and topk in (1024, 2048)
        ):
            return _run_partitioned_mg(
                compute_mode=ComputeMode.FP8,
                model_type=ModelType.DSV4,
                scale_format=ScaleFormat.UE8M0_BYTE,
                extra_kv_cache=extra_kv_cache,
                extra_indices=extra_indices,
                extra_topk_length=extra_topk_length,
                extra_page_block_size=extra_page_block_size,
                stride_extra_kv_block=stride_extra_kv_block,
            )
        raise ValueError(
            f"DSV4 dual-cache prefill (heads={heads}, topk={topk}, "
            f"pbs_extra={int(extra_page_block_size)}) requires MG dispatch; only "
            "DSV4 topk in {128,512} (BF16-QK) or topk in {1024,2048} (FP8-QK) "
            "with heads divisible by 8 is supported. "
            "No decode-reuse fallback."
        )

    # No MG gate matched. There is NO decode-reuse fallback: an unsupported
    # prefill shape HARD-FAILS (matching upstream's raise-not-fallback contract).
    raise ValueError(
        "SM120 sparse MLA prefill: unsupported shape "
        f"(model_type={int(model_type)}, heads={heads}, topk={topk}, "
        f"compute_mode={int(compute_mode)}, scale_format={int(scale_format)}, "
        f"has_extra={has_extra}, B12X_MLA_SM120_PREFILL_MG={'0' if not _mg_enabled else '1'}). "
        "Supported (MG) shapes: single-cache heads%8==0; "
        "DSV4 single-cache topk in {512, 1024, 2048} (FP8) or 128 "
        "(BF16-QK, heads%8==0); "
        "DSV4 dual-cache topk in {128, 512} (BF16-QK) or topk in {1024, 2048} "
        "(FP8-QK), heads%8==0, pbs_extra in {2, 64}; "
        "GLM_NSA topk in {512, 1024, 2048}; GLM_NEXT topk in "
        "{512, 1024, 2048, 2051, 2112}; "
        "NVFP4 (GLM-family, scale_format=2) topk in {128, 512, 1024, 2048}; "
        "GLM_NEXT NVFP4 additionally topk in {2051, 2112}. "
        "No decode-reuse fallback."
    )
