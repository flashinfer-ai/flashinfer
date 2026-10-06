"""GLM_NSA (uncompressed, ARBITRARY_FP32) sparse-MLA decode numeric reference.

This is the *ground-truth numeric oracle* for the P7b SM120 GLM_NSA decode port.
Unlike DSV4 (which mirrors a FlashInfer kernel), GLM_NSA is rs-1's OWN model: there
is no FlashInfer GLM reference PTX, so the correctness bar is purely NUMERICAL vs
``b12x.attention._shared.mla.reference.sparse_mla_reference`` (the same oracle the existing
GLM kernel / api.py path asserts against).

This module deliberately has NO dependency on the CuTeDSL kernel, on b12x._lib, or
on the real GLM-5.1 model weights. It REUSES the trusted pure-PyTorch primitives in
``b12x/attention/mla/reference.py`` rather than reimplementing them:

  - ``pack_mla_kv_cache_reference``   : bf16 (k_nope, k_rope) -> 656B/token GLM cache
  - ``unpack_mla_kv_cache_reference`` : inverse (dequant + dequant scales + rope)
  - ``sparse_mla_reference``          : dense SDPA over the gathered topk KV rows
  - ``_MLA_NOPE_DIM`` / ``_MLA_ROPE_DIM`` / ``_MLA_GROUP_SIZE`` / ``_MLA_PACKED_DIM``

GLM CACHE BYTE LAYOUT (authoritative = pack_mla_kv_cache_reference, NOT a footer):
  Per-token packed record is _MLA_PACKED_DIM = 656 contiguous uint8 bytes:
      [0   : 512)   nope   : 512 e4m3 bytes = 4 groups x 128 elems
      [512 : 528)   scales : 4 x INLINE fp32 (little-endian), one per 128-elem group
                            byte g*4 .. g*4+3  is  fp32 scale of group g (g in 0..3)
                            scale[g] = amax(|nope_group_g|) / FP8_E4M3_MAX  (ARBITRARY,
                            NOT a power-of-2 / UE8M0 exponent byte)
      [528 : 656)   rope   : 64 bf16 = 128 raw bytes (no quantization)
  i.e. nope_off=0, scales_off=512 (16B = 4 fp32), rope_off=528 (128B). The scales are
  stored INLINE per token (immediately after that token's nope), in contrast to DSV4's
  grouped UE8M0 byte FOOTER. dequant K = fp8_e4m3 * fp32_scale (kernel_onepass.py:544
  bfloat2_mul), and V == the nope-only first d_v=512 dims (V_HAS_ROPE=false): the rope
  tail is part of the QK score path only, never the PV output.

GLM_NSA traits (verified_traits.md / make_unified_traits(GLM_NSA, FP8, ARBITRARY_FP32)):
  d_nope=512  d_rope=64  d_v=512  q_head_dim=576 (=d_nope+d_rope)
  quant_tile=128  num_scales=4  n_v_chunks=4  nt_per_warp_xv=2
  kv_gmem_stride=656 (512 fp8 + 16 inline-fp32 + 128 rope-bf16)
  kv_smem_stride=q_nope_stride=528 (512 fp8 + 16 inline-fp32; rope staged separately)
  page_block_size (decode) = 64   v_has_rope=False   has_extra_cache=False

Public entry points (the build/verification agents call these):
  - make_glm_decode_case(num_heads=128, topk=64, *, num_blocks, page_block_size=64,
        invalidate_half=True, seed=0, device="cuda") -> dict{q[1,128,576],
        kv_cache (num_blocks*64, 1, 656) uint8 GLM layout, topk_indices[1,topk] i32,
        sm_scale float, expected_O[1,128,512] bf16, ...}
  - glm_decode_reference(q, kv_cache, topk_indices, sm_scale, *, v_head_dim=512,
        active_token_counts=None, return_lse=False)
        -> expected_O[num_tokens, num_heads, 512]  (or (O, lse_base2) if return_lse)

Run ``python -m tests.glm_ref`` to execute the internal self-tests.
"""

from __future__ import annotations

import math

import torch

# Reuse the trusted GLM packer / sparse oracle / constants (do NOT reimplement).
from b12x.attention._shared.mla.reference import (
    _FP8_E4M3_MAX,
    _MLA_GROUP_SIZE,
    _MLA_NOPE_DIM,
    _MLA_PACKED_DIM,
    _MLA_ROPE_DIM,
    pack_mla_kv_cache_reference,
    sparse_mla_reference,
    unpack_mla_kv_cache_reference,
)

# ── GLM_NSA traits (mirror make_unified_traits(GLM_NSA, FP8, ARBITRARY_FP32)) ──
GLM_D_NOPE = _MLA_NOPE_DIM  # 512
GLM_D_ROPE = _MLA_ROPE_DIM  # 64
GLM_D_V = _MLA_NOPE_DIM  # 512  (V_HAS_ROPE=false -> V == first d_v nope dims)
GLM_Q_HEAD_DIM = _MLA_NOPE_DIM + _MLA_ROPE_DIM  # 576
GLM_QUANT_TILE = _MLA_GROUP_SIZE  # 128
GLM_NUM_SCALES = _MLA_NOPE_DIM // _MLA_GROUP_SIZE  # 4
GLM_KV_GMEM_STRIDE = _MLA_PACKED_DIM  # 656

# Authoritative GLM cache byte offsets within one 656B/token record.
GLM_NOPE_OFF = 0
GLM_NOPE_BYTES = _MLA_NOPE_DIM  # 512 e4m3
GLM_SCALES_OFF = _MLA_NOPE_DIM  # 512
GLM_SCALES_BYTES = GLM_NUM_SCALES * 4  # 16  (4 inline fp32)
GLM_ROPE_OFF = _MLA_NOPE_DIM + GLM_SCALES_BYTES  # 528
GLM_ROPE_BYTES = _MLA_ROPE_DIM * 2  # 128  (64 bf16)
assert GLM_ROPE_OFF + GLM_ROPE_BYTES == GLM_KV_GMEM_STRIDE == 656

GLM_DECODE_PAGE_BLOCK_SIZE = 64
GLM_FP8_E4M3_MAX = float(_FP8_E4M3_MAX)


# ── Sparse-MLA decode reference (thin wrapper over sparse_mla_reference) ───────
def glm_decode_reference(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    topk_indices: torch.Tensor,
    sm_scale: float,
    *,
    v_head_dim: int = GLM_D_V,
    active_token_counts: torch.Tensor | None = None,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """GLM_NSA decode oracle: dense SDPA over the sparse-gathered + dequantized KV.

    Thin wrapper over :func:`sparse_mla_reference` (reference.py) — the topk slot ids
    ARE the ``page_table_1`` selection, the packed cache is dequantized internally
    (nope = e4m3*fp32_scale, rope = raw bf16), and V is the first ``v_head_dim`` dims
    of the union row (V_HAS_ROPE=false, so the rope tail contributes to scores only).

    Args:
      q:            [num_tokens, num_heads, 576] bf16/float (nope 512 + rope 64).
      kv_cache:     (num_slots, 1, 656) uint8 GLM packed cache (or (num_slots, 656);
                    a rank-2 cache is unsqueezed to rank-3 to match the reference).
      topk_indices: [num_tokens, topk] int32 flat slot ids (page_table_1). -1 = pad.
      sm_scale:     softmax scale (the unified GLM path uses 576**-0.5).
      v_head_dim:   512 for GLM (V == nope-only first d_v dims).
      active_token_counts: optional [num_tokens] int32 per-row valid prefix length.
      return_lse:   also return base-2 LSE [num_tokens, num_heads].

    Returns:
      expected_O [num_tokens, num_heads, v_head_dim] in q.dtype, or (O, lse_base2).
    """
    if kv_cache.ndim == 2:
        kv_cache = kv_cache.unsqueeze(1)
    if topk_indices.dtype != torch.int32:
        topk_indices = topk_indices.to(torch.int32)
    return sparse_mla_reference(
        q_all=q,
        kv_cache=kv_cache,
        page_table_1=topk_indices,
        active_token_counts=active_token_counts,
        sm_scale=float(sm_scale),
        v_head_dim=v_head_dim,
        return_lse=return_lse,
    )


# ── Test-case factory ─────────────────────────────────────────────────────────
def make_glm_decode_case(
    num_heads: int = 128,
    topk: int = 64,
    *,
    num_blocks: int,
    page_block_size: int = GLM_DECODE_PAGE_BLOCK_SIZE,
    invalidate_half: bool = True,
    seed: int = 0,
    num_tokens: int = 1,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> dict:
    """Build one self-consistent GLM_NSA decode test case on ``device``.

    Synthesizes random bf16 (k_nope, k_rope) directly (no real GLM weights needed),
    packs them into the 656B/token GLM cache via the trusted reference packer, builds
    a 576-dim q, and computes the oracle O via :func:`glm_decode_reference`.

    Returns a dict of cuda torch tensors:
      q                 [num_tokens, num_heads, 576] bf16
      kv_cache          (num_blocks*page_block_size, 1, 656) uint8 (GLM packed)
      kv_dequant        (num_slots, 1, 576) bf16 (nope 512 deq + rope 64) reference
      topk_indices      [num_tokens, topk] int32 (-1 in back half if invalidate_half)
      sm_scale          float  (= 576**-0.5, matching the unified GLM path)
      page_block_size   int
      num_blocks        int
      v_head_dim        int (512)
      expected_O        [num_tokens, num_heads, 512] bf16  (oracle output)
      expected_lse      [num_tokens, num_heads] fp32       (oracle base-2 LSE)
    """
    if num_blocks < 1:
        raise ValueError(f"num_blocks must be >= 1, got {num_blocks}")
    device = torch.device(device)
    gen = torch.Generator(device=device).manual_seed(seed)
    s_kv = num_blocks * page_block_size

    # Random KV in the same regime as the DSV4 harness (randn/10 clamp(-1,1)).
    k_nope = (
        torch.randn(
            s_kv, 1, GLM_D_NOPE, device=device, dtype=torch.bfloat16, generator=gen
        )
        / 10.0
    ).clamp(-1, 1)
    k_rope = (
        torch.randn(
            s_kv, 1, GLM_D_ROPE, device=device, dtype=torch.bfloat16, generator=gen
        )
        / 10.0
    ).clamp(-1, 1)

    # Pack -> 656B/token GLM cache (num_slots, 1, 656) uint8, then dequant for ref.
    kv_cache = pack_mla_kv_cache_reference(k_nope, k_rope)
    assert kv_cache.shape == (s_kv, 1, GLM_KV_GMEM_STRIDE), kv_cache.shape
    kv_dequant = unpack_mla_kv_cache_reference(kv_cache)  # (num_slots, 1, 576) bf16

    q = (
        torch.randn(
            num_tokens, num_heads, GLM_Q_HEAD_DIM, device=device, dtype=dtype, generator=gen
        )
        / 10.0
    ).clamp(-1, 1)

    indices = torch.randint(
        0, s_kv, (num_tokens, topk), device=device, dtype=torch.int32, generator=gen
    )
    if invalidate_half and topk >= 2:
        indices[:, topk // 2 :] = -1

    sm_scale = GLM_Q_HEAD_DIM**-0.5

    expected_O, expected_lse = glm_decode_reference(
        q,
        kv_cache,
        indices,
        sm_scale,
        v_head_dim=GLM_D_V,
        return_lse=True,
    )

    return {
        "q": q,
        "kv_cache": kv_cache,
        "kv_dequant": kv_dequant,
        "topk_indices": indices,
        "sm_scale": sm_scale,
        "page_block_size": page_block_size,
        "num_blocks": num_blocks,
        "v_head_dim": GLM_D_V,
        "expected_O": expected_O,
        "expected_lse": expected_lse,
    }


# ── Internal self-tests ─────────────────────────────────────────────────────--
def _self_test(device: str | torch.device = "cuda") -> None:
    device = torch.device(device)
    torch.manual_seed(0)

    # (1) GLM cache byte layout is exactly 512 nope + 16 inline-fp32 + 128 rope = 656,
    #     and the inline scales decode as ARBITRARY fp32 amax/FP8_MAX (NOT pow2). We
    #     read the bytes directly and confirm the offsets/values match the packer.
    s_kv = 64
    gen = torch.Generator(device=device).manual_seed(7)
    k_nope = (
        torch.randn(s_kv, 1, GLM_D_NOPE, device=device, dtype=torch.bfloat16, generator=gen)
        / 10.0
    ).clamp(-1, 1)
    k_rope = (
        torch.randn(s_kv, 1, GLM_D_ROPE, device=device, dtype=torch.bfloat16, generator=gen)
        / 10.0
    ).clamp(-1, 1)
    packed = pack_mla_kv_cache_reference(k_nope, k_rope)
    assert packed.shape == (s_kv, 1, GLM_KV_GMEM_STRIDE), packed.shape
    assert packed.dtype == torch.uint8
    pb = packed.view(s_kv, GLM_KV_GMEM_STRIDE)

    # Inline scales at [512:528) decode as 4 fp32 = amax(group)/FP8_MAX (arbitrary).
    scale_bytes = pb[:, GLM_SCALES_OFF : GLM_SCALES_OFF + GLM_SCALES_BYTES].contiguous()
    scales = scale_bytes.view(torch.float32).reshape(s_kv, GLM_NUM_SCALES)
    nope_f = k_nope.squeeze(1).float()
    for g in range(GLM_NUM_SCALES):
        block = nope_f[:, g * GLM_QUANT_TILE : (g + 1) * GLM_QUANT_TILE]
        expect = (block.abs().amax(dim=1) / GLM_FP8_E4M3_MAX).clamp_min(0)
        expect = torch.where(expect > 0, expect, torch.ones_like(expect))
        torch.testing.assert_close(scales[:, g], expect, atol=0, rtol=0)
    # These are NOT power-of-two (no UE8M0): mantissa bits are generally set.
    sbits = scales.reshape(-1).view(torch.int32)
    assert (sbits & 0x007FFFFF).any(), "GLM scales should be arbitrary fp32, not pow2"

    # (2) dequant(pack(x)) round-trips within e4m3 tolerance; rope is bf16-exact.
    deq = unpack_mla_kv_cache_reference(packed).squeeze(1).float()
    orig_nope = k_nope.squeeze(1).float()
    orig_rope = k_rope.squeeze(1).float()
    rope_err = (deq[:, GLM_D_NOPE:] - orig_rope).abs().max().item()
    assert rope_err < 1e-2, f"rope round-trip error too large: {rope_err}"
    nope_rel = (
        (deq[:, :GLM_D_NOPE] - orig_nope).abs() / orig_nope.abs().clamp(min=1e-3)
    )
    assert nope_rel.median().item() < 0.1, (
        f"nope round-trip median rel error too large: {nope_rel.median().item()}"
    )

    # (3) glm_decode_reference matches an INDEPENDENT brute-force dense attention on a
    #     tiny case (plain float matmul over the dequantized KV, no online softmax).
    case = make_glm_decode_case(
        num_heads=4, topk=8, num_blocks=2, invalidate_half=False, seed=3, device=device
    )
    q = case["q"].float()
    deq_pool = case["kv_dequant"].squeeze(1).float()  # (num_slots, 576)
    idx = case["topk_indices"].long()
    sm_scale = case["sm_scale"]
    nt, nh, _ = q.shape
    brute_O = torch.zeros(nt, nh, GLM_D_V, device=device)
    brute_lse = torch.zeros(nt, nh, device=device)
    for t in range(nt):
        rows = deq_pool.index_select(0, idx[t])  # [topk, 576]
        for h in range(nh):
            logits = (q[t, h] @ rows.t()) * sm_scale  # [topk]
            m = logits.max()
            w = torch.exp(logits - m)
            denom = w.sum()
            brute_O[t, h] = (w @ rows[:, :GLM_D_V]) / denom  # V = first 512 dims only
            brute_lse[t, h] = (m + torch.log(denom)) / math.log(2.0)
    torch.testing.assert_close(case["expected_O"].float(), brute_O, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(case["expected_lse"], brute_lse, atol=1e-3, rtol=1e-3)

    # (4) Default decode case: shapes/dtypes/sm_scale match the unified GLM contract.
    big = make_glm_decode_case(num_heads=128, topk=64, num_blocks=16, device=device)
    assert big["q"].shape == (1, 128, GLM_Q_HEAD_DIM)
    assert big["q"].dtype == torch.bfloat16
    assert big["kv_cache"].shape == (16 * 64, 1, GLM_KV_GMEM_STRIDE)
    assert big["kv_cache"].dtype == torch.uint8
    assert big["topk_indices"].shape == (1, 64)
    assert big["topk_indices"].dtype == torch.int32
    assert (big["topk_indices"][:, 32:] == -1).all(), "back half must be -1 (pad)"
    assert big["expected_O"].shape == (1, 128, GLM_D_V)
    assert big["expected_O"].dtype == torch.bfloat16
    assert big["expected_lse"].shape == (1, 128)
    assert abs(big["sm_scale"] - GLM_Q_HEAD_DIM**-0.5) < 1e-12
    assert torch.isfinite(big["expected_O"].float()).all()
    assert torch.isfinite(big["expected_lse"]).all()  # half valid -> finite lse

    # (5) all-invalid token: lse must be -inf, output 0 (matches kernel empty epilogue).
    case_e = make_glm_decode_case(
        num_heads=4, topk=4, num_blocks=2, invalidate_half=False, seed=2, device=device
    )
    case_e["topk_indices"][:] = -1
    O_e, lse_e = glm_decode_reference(
        case_e["q"], case_e["kv_cache"], case_e["topk_indices"], case_e["sm_scale"],
        return_lse=True,
    )
    assert torch.all(lse_e == float("-inf")), "all-invalid LSE must be -inf"
    assert torch.all(O_e.float() == 0.0), "all-invalid output must be 0"

    # (6) topk sweep {64, 128, 512}: finite, correctly-shaped outputs.
    for tk in (64, 128, 512):
        nblk = max(1, (tk + GLM_DECODE_PAGE_BLOCK_SIZE - 1) // GLM_DECODE_PAGE_BLOCK_SIZE)
        c = make_glm_decode_case(num_heads=128, topk=tk, num_blocks=nblk, device=device)
        assert c["q"].shape == (1, 128, GLM_Q_HEAD_DIM)
        assert c["expected_O"].shape == (1, 128, GLM_D_V)
        assert c["expected_lse"].shape == (1, 128)
        assert c["kv_cache"].shape[-1] == GLM_KV_GMEM_STRIDE
        assert torch.isfinite(c["expected_O"].float()).all()
        assert torch.isfinite(c["expected_lse"]).all()

    print(
        "glm_ref self-tests PASSED "
        "(layout/inline-fp32/round-trip/brute-force/default/empty/topk-sweep)"
    )


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    _self_test(dev)
