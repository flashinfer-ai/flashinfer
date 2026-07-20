# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
# Ported from b12x b12x/attention/mla/smem_mg.py @ 6627d342 (2026-07-19) -- one-time curated port.
# Upstream b12x is a research sandbox; this tree is the canonical home.
"""FlashInfer-shaped MG shared-memory layout for DSV4 SM120 prefill.

This layout mirrors the DSV4 ``SmemLayoutMG`` contract used by FlashInfer prefill:
two HPB head groups per CTA, one double-buffered NoPE KV stage, separate UE8M0
scale buffers, and no DSV4 RoPE KV staging. RoPE is loaded from global/L2 by the
math path.
"""

from __future__ import annotations

from dataclasses import dataclass

import cutlass
import cutlass.cute as cute

from .smem import SM120_SMEM_CARVEOUT_BYTES
from .traits import ComputeMode, UnifiedMLATraits


_MG_N_HG = 2
_KV_BUF_COUNT = 2
_W_FP8_BUF_COUNT = 2
_MATH_WARPS = 8
_W_FP8_PAD = 16
_MBAR_BYTES = _KV_BUF_COUNT * 8


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class SmemLayoutMG:
    mg_n_hg: int
    heads_per_cta: int

    # BF16-QK (ComputeMode.BF16) layout divergences. For the FP8 path these are
    # all 0/False and the layout is byte-identical to the validated FP8 layout.
    bf16_qk: bool
    q_nope_bf16_off: int
    q_nope_bf16_group_bytes: int
    q_nope_bf16_bytes: int
    q_nope_bf16_stride: int

    q_rope_off: int
    q_rope_group_bytes: int
    q_rope_bytes: int
    q_rope_stride: int

    q_fp8_off: int
    q_fp8_group_bytes: int
    q_fp8_bytes: int
    q_nope_stride: int

    q_sc_off: int
    q_sc_group_bytes: int
    q_sc_bytes: int
    q_sc_stride: int

    kv_fp8_off: int
    kv_fp8_buf_bytes: int
    kv_smem_stride: int

    kv_sc_off: int
    kv_sc_buf_bytes: int
    kv_sc_stride: int

    # kv_rope: GLM-only V/K rope bf16 staging (linear entry*D_ROPE + d),
    # double-buffered. DSV4 MG reads rope from global/L2, so kv_rope_buf_bytes==0
    # and the field is never indexed for DSV4 (const_expr-elided).
    kv_rope_off: int
    kv_rope_buf_bytes: int
    kv_rope_stride: int

    kv_bufs: int

    mbar_off: int
    mbar_bytes: int

    reduce_off: int
    reduce_group_bytes: int
    reduce_warp_max_group_off: int
    reduce_warp_sum_group_off: int
    reduce_bytes: int

    w_head_sc_off: int
    w_head_sc_group_bytes: int
    w_head_sc_bytes: int
    w_head_sc_stride: int

    w_fp8_off: int
    w_fp8_group_bytes: int
    w_fp8_parity_bytes: int
    w_fp8_stride: int
    w_fp8_bufs: int

    sm_p_full_off: int
    sm_p_full_group_bytes: int
    sm_p_full_bytes: int
    sm_p_full_stride: int

    total_bytes: int


def make_smem_layout_mg(
    traits: UnifiedMLATraits, mg_n_hg: int = _MG_N_HG
) -> SmemLayoutMG:
    # DSV4 (has_extra_cache) staged the 448B NoPE only (rope from global, UE8M0
    # footer in kv_sc). GLM (ARBITRARY_FP32, no extra cache) stages the 528B
    # NoPE+inline-fp32-scales row (scales inline, NO footer) PLUS a kv_rope buffer
    # (GLM has no global-rope path in MG). Both share the rest of the MG layout.
    is_glm = not traits.has_extra_cache
    if mg_n_hg not in (1, 2):
        raise ValueError(
            f"MG prefill layout supports mg_n_hg in {{1, 2}}, got {mg_n_hg}"
        )

    bi = traits.bi
    hpb = traits.hpb
    d_rope = traits.d_rope
    d_nope = traits.d_nope
    num_scales = traits.num_scales
    n_v_chunks = traits.n_v_chunks
    q_nope_stride = traits.q_nope_stride
    bufs = _KV_BUF_COUNT
    # MG head-group count: 2 (heads % 32 == 0, the validated path) or 1 (heads ==
    # 16, the small-TP shard). All *_bytes (group buffers) below are mg_n_hg *
    # *_group_bytes, so mg_n_hg==1 allocates exactly one head group's smem (half
    # the reduce / W / Q / sm_p region). The per-group *_group_bytes are identical
    # across both, so the group-0 byte offsets the kernel computes are unchanged.
    heads_per_cta = mg_n_hg * hpb

    bf16_qk = traits.compute_mode == ComputeMode.BF16
    # BF16-QK and GLM FP8 use Q-rope scratch aliased onto W_FP8. BF16 reloads
    # that small scratch cooperatively between tiles instead of keeping the
    # ldmatrix fragments live (and spilled) across the full kernel.
    alias_qrope = bf16_qk or is_glm
    # Keep the model-native padded KV row in both compute modes.  DSV4's 464-B
    # stride (448-B payload + 16-B pad) is part of the shared-memory bank layout,
    # not merely copy padding; FlashInfer retains it for BF16 QK as well.  The
    # XV-RoPE weights alias W_FP8 below, so the padded double buffer still fits
    # the SM120 carveout.
    kv_smem_stride = traits.kv_smem_stride

    off = 0

    if bf16_qk:
        # Q-NoPE stored as BF16 (no FP8 quant), HPB x (D_NOPE+8) per group. The
        # +8 bf16 tail keeps each head row 16 B-aligned for ldmatrix.x4. The FP8
        # q_fp8 / q_sc regions do not exist; their fields are zeroed below.
        q_nope_bf16_off = off
        q_nope_bf16_stride = d_nope + 8
        q_nope_bf16_group_bytes = hpb * q_nope_bf16_stride * 2
        q_nope_bf16_bytes = mg_n_hg * q_nope_bf16_group_bytes
        off = q_nope_bf16_off + q_nope_bf16_bytes
        # Assigned after W_FP8 is laid out below.
        q_fp8_off = 0
        q_fp8_group_bytes = 0
        q_fp8_bytes = 0
        q_sc_off = 0
        q_sc_stride = 0
        q_sc_group_bytes = 0
        q_sc_bytes = 0
    else:
        q_nope_bf16_off = 0
        q_nope_bf16_stride = 0
        q_nope_bf16_group_bytes = 0
        q_nope_bf16_bytes = 0

        # GLM consumes the aliased Q-RoPE tile through ldmatrix.x4.  A 64-bf16
        # row aliases every row onto one bank group; the 8-element pad rotates
        # rows by one 16-B bank group.  DSV4 FP8 keeps its established layout.
        q_rope_stride = d_rope + (8 if is_glm else 0)
        q_rope_group_bytes = hpb * q_rope_stride * 2
        q_rope_bytes = mg_n_hg * q_rope_group_bytes
        if alias_qrope:
            # GLM FP8: Q-rope is preloaded to registers once (S0) then its smem
            # scratch ALIASES W_FP8 (live only in S6). Assigned after W_FP8 off is
            # known, below. Reserve no inline q_rope region here.
            q_rope_off = 0
        else:
            q_rope_off = off
            off = q_rope_off + q_rope_bytes

        q_fp8_off = off
        q_fp8_group_bytes = hpb * q_nope_stride
        q_fp8_bytes = mg_n_hg * q_fp8_group_bytes
        off = q_fp8_off + q_fp8_bytes

        q_sc_off = off
        q_sc_stride = num_scales
        q_sc_group_bytes = hpb * num_scales * 4
        q_sc_bytes = mg_n_hg * q_sc_group_bytes
        off = q_sc_off + q_sc_bytes

    off = _align_up(off, 128)
    kv_fp8_off = off
    kv_fp8_buf_bytes = bi * kv_smem_stride
    off = kv_fp8_off + kv_fp8_buf_bytes * bufs

    kv_sc_off = off
    if is_glm:
        # GLM: scales are INLINE in the kv_fp8 KV_SMEM_STRIDE row (no footer buf).
        kv_sc_stride = 0
        kv_sc_buf_bytes = 0
    else:
        kv_sc_stride = 8
        kv_sc_buf_bytes = bi * kv_sc_stride
        off = kv_sc_off + kv_sc_buf_bytes * bufs

    # MG reads KV-rope from global/L2 for BOTH models (no smem staging), so the
    # kv_rope buffer is always empty. Kept as zeroed fields for layout symmetry.
    kv_rope_off = off
    kv_rope_stride = d_rope
    kv_rope_buf_bytes = 0

    off = _align_up(off, 16)
    mbar_off = off
    mbar_bytes = _MBAR_BYTES
    off = mbar_off + mbar_bytes

    reduce_off = off
    reduce_group_bytes = 2 * _MATH_WARPS * hpb * 4
    reduce_warp_max_group_off = 0
    reduce_warp_sum_group_off = _MATH_WARPS * hpb * 4
    reduce_bytes = mg_n_hg * reduce_group_bytes
    off = reduce_off + reduce_bytes

    w_head_sc_off = off
    w_head_sc_stride = hpb
    w_head_sc_group_bytes = n_v_chunks * hpb * 4
    w_head_sc_bytes = mg_n_hg * w_head_sc_group_bytes
    off = w_head_sc_off + w_head_sc_bytes

    w_fp8_off = off
    w_fp8_stride = bi + _W_FP8_PAD
    w_fp8_group_bytes = hpb * w_fp8_stride
    w_fp8_parity_bytes = mg_n_hg * w_fp8_group_bytes
    w_fp8_bufs = _W_FP8_BUF_COUNT
    off = w_fp8_off + w_fp8_parity_bytes * w_fp8_bufs

    if alias_qrope:
        # Q-rope smem scratch ALIASES the W_FP8 region (see above). Only live in
        # S0 (written by the cooperative load, immediately read to registers).
        # Shared by the BF16-QK (DSV4) and FP8 GLM paths.
        q_rope_off = w_fp8_off
        q_rope_stride = d_rope + 8
        q_rope_group_bytes = hpb * q_rope_stride * 2
        q_rope_bytes = mg_n_hg * q_rope_group_bytes
        assert q_rope_bytes <= w_fp8_parity_bytes * w_fp8_bufs, (
            "Q-rope scratch does not fit in the aliased W_FP8 region"
        )

    # XV-RoPE runs after XV-NoPE has consumed W_FP8.  Reuse that dead region for
    # the BF16 softmax-weight tile exactly as the native MG kernel does instead
    # of reserving a separate 4-KiB sm_p_full allocation.
    sm_p_full_off = w_fp8_off
    # A BI=64 bf16 row is exactly 128 B, so all sixteen rows presented to an
    # ldmatrix.x4 start in the same shared-memory bank group.  Add one 16-B
    # bank-group step per row: successive rows then cover all eight groups
    # before repeating, while the two padded head-group tiles still fit in the
    # dead double-buffered W_FP8 allocation (2 * 16 * 72 * 2 = 4608 B <= 5120 B).
    sm_p_full_stride = bi + 8
    sm_p_full_group_bytes = hpb * sm_p_full_stride * 2
    sm_p_full_bytes = mg_n_hg * sm_p_full_group_bytes
    assert sm_p_full_bytes <= w_fp8_parity_bytes * w_fp8_bufs, (
        "XV-RoPE weight tile does not fit in the aliased W_FP8 region"
    )

    total_bytes = _align_up(off, 128)

    return SmemLayoutMG(
        mg_n_hg=mg_n_hg,
        heads_per_cta=heads_per_cta,
        bf16_qk=bf16_qk,
        q_nope_bf16_off=q_nope_bf16_off,
        q_nope_bf16_group_bytes=q_nope_bf16_group_bytes,
        q_nope_bf16_bytes=q_nope_bf16_bytes,
        q_nope_bf16_stride=q_nope_bf16_stride,
        q_rope_off=q_rope_off,
        q_rope_group_bytes=q_rope_group_bytes,
        q_rope_bytes=q_rope_bytes,
        q_rope_stride=q_rope_stride,
        q_fp8_off=q_fp8_off,
        q_fp8_group_bytes=q_fp8_group_bytes,
        q_fp8_bytes=q_fp8_bytes,
        q_nope_stride=q_nope_stride,
        q_sc_off=q_sc_off,
        q_sc_group_bytes=q_sc_group_bytes,
        q_sc_bytes=q_sc_bytes,
        q_sc_stride=q_sc_stride,
        kv_fp8_off=kv_fp8_off,
        kv_fp8_buf_bytes=kv_fp8_buf_bytes,
        kv_smem_stride=kv_smem_stride,
        kv_sc_off=kv_sc_off,
        kv_sc_buf_bytes=kv_sc_buf_bytes,
        kv_sc_stride=kv_sc_stride,
        kv_rope_off=kv_rope_off,
        kv_rope_buf_bytes=kv_rope_buf_bytes,
        kv_rope_stride=kv_rope_stride,
        kv_bufs=bufs,
        mbar_off=mbar_off,
        mbar_bytes=mbar_bytes,
        reduce_off=reduce_off,
        reduce_group_bytes=reduce_group_bytes,
        reduce_warp_max_group_off=reduce_warp_max_group_off,
        reduce_warp_sum_group_off=reduce_warp_sum_group_off,
        reduce_bytes=reduce_bytes,
        w_head_sc_off=w_head_sc_off,
        w_head_sc_group_bytes=w_head_sc_group_bytes,
        w_head_sc_bytes=w_head_sc_bytes,
        w_head_sc_stride=w_head_sc_stride,
        w_fp8_off=w_fp8_off,
        w_fp8_group_bytes=w_fp8_group_bytes,
        w_fp8_parity_bytes=w_fp8_parity_bytes,
        w_fp8_stride=w_fp8_stride,
        w_fp8_bufs=w_fp8_bufs,
        sm_p_full_off=sm_p_full_off,
        sm_p_full_group_bytes=sm_p_full_group_bytes,
        sm_p_full_bytes=sm_p_full_bytes,
        sm_p_full_stride=sm_p_full_stride,
        total_bytes=total_bytes,
    )


def get_prefill_mg_shared_storage_cls(
    traits: UnifiedMLATraits, mg_n_hg: int = _MG_N_HG
):
    layout = make_smem_layout_mg(traits, mg_n_hg)

    class SharedStorageMG:
        pass

    is_glm = not traits.has_extra_cache

    # Tail (kv_fp8 .. w_fp8) is identical across DSV4 compute modes. DSV4
    # stages a separate UE8M0 footer ``kv_sc`` (rope from global/L2). GLM has NO
    # footer (the 4 inline fp32 scales travel in the kv_fp8 528B row) and also
    # reads rope from global/L2, so it allocates NEITHER kv_sc nor kv_rope. Each
    # model is its own compiled specialization, so the GLM struct cannot perturb
    # the DSV4 struct.
    if is_glm:
        kv_scale_field = {}
    else:
        kv_scale_field = {
            "kv_sc": cute.struct.MemRange[
                cutlass.Uint8, int(layout.kv_sc_buf_bytes * layout.kv_bufs)
            ],
        }

    # Only the Q prologue differs across compute modes: DSV4 FP8 stages {q_rope,
    # q_fp8, q_sc}; BF16 + GLM FP8 stage Q-rope into a scratch aliased onto the
    # W_FP8 region (no separate q_rope field).
    tail = {
        "kv_fp8": cute.struct.Align[
            cute.struct.MemRange[
                cutlass.Uint8, int(layout.kv_fp8_buf_bytes * layout.kv_bufs)
            ],
            128,
        ],
        **kv_scale_field,
        "mbar": cute.struct.Align[
            cute.struct.MemRange[cutlass.Uint64, int(layout.mbar_bytes // 8)],
            16,
        ],
        "reduce": cute.struct.MemRange[cutlass.Float32, int(layout.reduce_bytes // 4)],
        "w_head_sc": cute.struct.MemRange[
            cutlass.Float32, int(layout.w_head_sc_bytes // 4)
        ],
        "w_fp8": cute.struct.MemRange[
            cutlass.Uint8, int(layout.w_fp8_parity_bytes * layout.w_fp8_bufs)
        ],
    }

    if layout.bf16_qk:
        SharedStorageMG.__annotations__ = {
            "q_nope_bf16": cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.BFloat16, int(layout.q_nope_bf16_bytes // 2)
                ],
                128,
            ],
            **tail,
        }
    elif is_glm:
        # GLM FP8: {q_fp8, q_sc} only; Q-rope scratch aliases W_FP8 (no field).
        SharedStorageMG.__annotations__ = {
            "q_fp8": cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint8, int(layout.q_fp8_bytes)], 128
            ],
            "q_sc": cute.struct.MemRange[cutlass.Float32, int(layout.q_sc_bytes // 4)],
            **tail,
        }
    else:
        SharedStorageMG.__annotations__ = {
            "q_rope": cute.struct.Align[
                cute.struct.MemRange[cutlass.BFloat16, int(layout.q_rope_bytes // 2)],
                128,
            ],
            "q_fp8": cute.struct.MemRange[cutlass.Uint8, int(layout.q_fp8_bytes)],
            "q_sc": cute.struct.MemRange[cutlass.Float32, int(layout.q_sc_bytes // 4)],
            **tail,
        }

    storage_cls = cute.struct(SharedStorageMG)
    if layout.bf16_qk:
        expected_offsets = {"q_nope_bf16": layout.q_nope_bf16_off}
    elif is_glm:
        expected_offsets = {
            "q_fp8": layout.q_fp8_off,
            "q_sc": layout.q_sc_off,
        }
    else:
        expected_offsets = {
            "q_rope": layout.q_rope_off,
            "q_fp8": layout.q_fp8_off,
            "q_sc": layout.q_sc_off,
        }
    expected_offsets["kv_fp8"] = layout.kv_fp8_off
    if not is_glm:
        expected_offsets["kv_sc"] = layout.kv_sc_off
    expected_offsets.update(
        {
            "mbar": layout.mbar_off,
            "reduce": layout.reduce_off,
            "w_head_sc": layout.w_head_sc_off,
            "w_fp8": layout.w_fp8_off,
        }
    )

    actual_offsets = dict(storage_cls._offsets)
    if actual_offsets != expected_offsets:
        raise ValueError(
            "typed MLA MG SharedStorage offsets diverge from SmemLayoutMG: "
            f"typed={actual_offsets}, logical={expected_offsets}"
        )
    typed_bytes = int(storage_cls.size_in_bytes())
    if typed_bytes != layout.total_bytes:
        raise ValueError(
            "typed MLA MG SharedStorage size diverges from SmemLayoutMG: "
            f"typed={typed_bytes}, logical={layout.total_bytes}"
        )
    return storage_cls


def _run_module_asserts() -> None:
    from .traits import ComputeMode, ModelType, ScaleFormat, make_unified_traits

    traits = make_unified_traits(
        ModelType.DSV4, ComputeMode.FP8, ScaleFormat.UE8M0_BYTE
    )
    layout = make_smem_layout_mg(traits)
    assert not layout.bf16_qk
    assert layout.kv_smem_stride == 464
    assert layout.kv_fp8_buf_bytes == 64 * 464
    assert layout.kv_sc_buf_bytes == 64 * 8
    assert layout.q_rope_off == 0
    assert layout.total_bytes < SM120_SMEM_CARVEOUT_BYTES, (
        f"MG prefill smem {layout.total_bytes}B exceeds SM120 carveout "
        f"{SM120_SMEM_CARVEOUT_BYTES}B"
    )

    bf16 = make_unified_traits(ModelType.DSV4, ComputeMode.BF16, ScaleFormat.UE8M0_BYTE)
    bl = make_smem_layout_mg(bf16)
    assert bl.bf16_qk
    assert bl.kv_smem_stride == 464
    assert bl.q_nope_bf16_stride == 448 + 8
    assert bl.q_rope_off == bl.w_fp8_off
    assert bl.total_bytes < SM120_SMEM_CARVEOUT_BYTES, (
        f"BF16 MG prefill smem {bl.total_bytes}B exceeds SM120 carveout "
        f"{SM120_SMEM_CARVEOUT_BYTES}B"
    )

    # mg_n_hg==1 (heads==16): one head group. Every group buffer is exactly half
    # the mg_n_hg==2 size; the per-group *_group_bytes (and so the group-0 byte
    # offsets the kernel computes) are byte-identical to the 2-group layout, only
    # the region totals (and downstream offs) shrink. Fits trivially.
    fp8_1 = make_smem_layout_mg(traits, mg_n_hg=1)
    assert fp8_1.mg_n_hg == 1 and fp8_1.heads_per_cta == 16
    assert fp8_1.reduce_group_bytes == layout.reduce_group_bytes
    assert fp8_1.reduce_bytes == layout.reduce_bytes // 2
    assert fp8_1.w_head_sc_group_bytes == layout.w_head_sc_group_bytes
    assert fp8_1.w_head_sc_bytes == layout.w_head_sc_bytes // 2
    assert fp8_1.w_fp8_group_bytes == layout.w_fp8_group_bytes
    assert fp8_1.w_fp8_parity_bytes == layout.w_fp8_parity_bytes // 2
    assert fp8_1.q_fp8_group_bytes == layout.q_fp8_group_bytes
    assert fp8_1.q_fp8_bytes == layout.q_fp8_bytes // 2
    assert fp8_1.sm_p_full_group_bytes == layout.sm_p_full_group_bytes
    assert fp8_1.sm_p_full_bytes == layout.sm_p_full_bytes // 2
    assert fp8_1.total_bytes <= layout.total_bytes
    assert fp8_1.total_bytes < SM120_SMEM_CARVEOUT_BYTES
    get_prefill_mg_shared_storage_cls(traits, mg_n_hg=1)
    get_prefill_mg_shared_storage_cls(traits, mg_n_hg=2)

    bf16_1 = make_smem_layout_mg(bf16, mg_n_hg=1)
    assert bf16_1.mg_n_hg == 1 and bf16_1.bf16_qk
    assert bf16_1.q_nope_bf16_group_bytes == bl.q_nope_bf16_group_bytes
    assert bf16_1.q_nope_bf16_bytes == bl.q_nope_bf16_bytes // 2
    assert bf16_1.q_rope_off == bf16_1.w_fp8_off
    assert bf16_1.total_bytes < SM120_SMEM_CARVEOUT_BYTES
    get_prefill_mg_shared_storage_cls(bf16, mg_n_hg=1)
    get_prefill_mg_shared_storage_cls(bf16, mg_n_hg=2)

    # GLM (ARBITRARY_FP32, no extra cache): kv_smem_stride 528 (512 nope + 16
    # inline fp32 scales), NO kv_sc footer (inline scales), NO kv_rope (rope read
    # from global/L2), Q-rope aliased onto W_FP8 -- so mg_n_hg==2 fits the carveout.
    glm = make_unified_traits(
        ModelType.GLM_NSA, ComputeMode.FP8, ScaleFormat.ARBITRARY_FP32
    )
    for nhg in (1, 2):
        gl = make_smem_layout_mg(glm, mg_n_hg=nhg)
        assert not gl.bf16_qk
        assert gl.heads_per_cta == nhg * 16
        assert gl.kv_smem_stride == 528
        assert gl.kv_fp8_buf_bytes == 64 * 528
        assert gl.kv_sc_buf_bytes == 0  # inline scales, no footer.
        assert gl.kv_rope_buf_bytes == 0  # rope from global/L2 (DSV4 MG design).
        assert gl.q_rope_off == gl.w_fp8_off  # Q-rope scratch aliases W_FP8.
        assert gl.total_bytes < SM120_SMEM_CARVEOUT_BYTES, (
            f"GLM MG prefill smem {gl.total_bytes}B exceeds SM120 carveout "
            f"{SM120_SMEM_CARVEOUT_BYTES}B (mg_n_hg={nhg})"
        )
        get_prefill_mg_shared_storage_cls(glm, mg_n_hg=nhg)


_run_module_asserts()
