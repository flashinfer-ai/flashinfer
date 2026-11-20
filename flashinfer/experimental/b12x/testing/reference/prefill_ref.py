"""DSV4 sparse-MLA *prefill* numeric reference + test-case harness (pure PyTorch).

This module is the ground-truth numeric reference for the P8 SM120 DSV4 *prefill*
port. Like ``dsv4_ref.py`` (the decode reference it reuses), it has NO dependency
on the CuTeDSL kernel, on ``b12x._lib``, or on the FlashInfer runtime — it is
pure PyTorch so build/verification agents can compare kernel state and the final
BF16 O / log2 LSE against a trusted oracle.

PREFILL vs DECODE — the *numerics are identical*, only the kernel topology
differs (see ``.sm120port/prefill_map.md`` §differences_from_decode):

  * DECODE   = split-K: each CTA handles one KV chunk, emits partial_O + partial
               LSE, then a separate merge kernel combines them.
  * PREFILL  = single-pass: ONE 384-thread CTA per token consumes ALL topk tiles,
               with the ``acc_o`` accumulator + online-softmax m/l carried across
               every tile (per-tile exp-rescale), then emits the FINAL full BF16
               ``O[T, H, 512]`` and base-2 ``LSE[T, H]`` directly (no merge).

Because the single-pass online softmax is mathematically exact, the prefill
result for a token equals dense SDPA over that token's gathered+dequantized topk
rows — which is EXACTLY what ``dsv4_ref.dsv4_decode_reference`` already computes
(it is FlashInfer's ``_ref_sparse_attn``, already vectorized over T). So this
module's reference is a thin, explicit wrapper over that decode oracle. The value
this file adds over calling the decode oracle directly is the PREFILL HARNESS:

  * many query tokens (T > 64, the prefill regime),
  * PER-TOKEN variable ``topk_length`` that is deliberately NOT a multiple of
    BI=64 (exercises the kernel's last-partial-tile masking at lines 300-310 of
    prefill_kernel.cuh),
  * the attn_sink LSE-fold cross-checked against a direct softmax-with-sink.

DSV4 traits (identical to decode; from verified_traits.md):
  d_nope=448  d_rope=64  d_v=512  d_qk=512  quant_tile=64  num_scales=7
  n_v_chunks=7  kv_gmem_stride=584  page_block_size=64  V_HAS_ROPE=true  BI=64

attn_sink LSE-fold formula (prefill_kernel.cuh:548-560, base-2 / log2 domain):
    lse = softmax_lse(m, l) = (l > 0) ? log2(l) + m : -1e30      # base-2 LSE
    if attn_sink != null:
        sink_log2 = attn_sink[h] * LOG2E                          # natural→log2
        if lse != -1e30:  lse += log2(1 + exp2(sink_log2 - lse))  # log-sum-exp fold
        else:             lse  = sink_log2                        # empty-row → sink
And the OUTPUT is scaled by the sink sigmoid, folded into the normalizer
(prefill_kernel.cuh:485-506):
    O[t,h,:] *= sigmoid(lse_e[t,h] - sink[h])                     # natural-log domain
i.e. il = 1 / (l + exp2(sink_log2 - m)) instead of 1 / l. The torch reference
applies the equivalent ``out *= sigmoid(lse_e - sink)`` directly. These two
formulations are algebraically equal (verified by self-test 4 below).

Public entry points (build/verification agents call these):
  - dsv4_prefill_reference(q, packed_kv_cache, topk_indices, sm_scale, *,
        page_block_size=64, d_v=512, attn_sink=None, topk_length=None,
        kv_dequant=None)
        -> (O[T, num_heads, d_v] bf16, lse_log2[T, num_heads] fp32)
  - make_dsv4_prefill_case(num_tokens, num_heads=128, topk=512, *, num_blocks,
        page_block_size=64, with_sink=False, invalidate_half=True, seed=0,
        device="cuda", dtype=bf16) -> dict

Run ``python -m tests.prefill_ref`` to execute the internal self-tests.
"""

from __future__ import annotations

import math

import torch

# Reuse the DSV4 decode reference verbatim — prefill numerics ARE decode numerics.
from b12x.testing.reference.dsv4_ref import (  # noqa: F401  (re-exported traits used by callers/tests)
    DSV4_D_NOPE,
    DSV4_D_QK,
    DSV4_D_ROPE,
    DSV4_D_V,
    DSV4_KV_GMEM_STRIDE,
    DSV4_NUM_SCALES,
    DSV4_QUANT_TILE,
    FP8_MAX,
    dequantize_kv_dsv4,
    dsv4_decode_reference,
    quantize_kv_dsv4,
)

# Prefill tiling constant (BLOCK_SIZE_M = BI). The kernel processes topk in tiles
# of BI tokens; actual_ni = ceil(topk_len / BI). Used here only to construct
# per-token topk_length values that are deliberately non-multiples of BI so the
# harness exercises the kernel's last-partial-tile masking.
DSV4_PREFILL_BI = 64


# ── Prefill reference ─────────────────────────────────────────────────────────
def dsv4_prefill_reference(
    q: torch.Tensor,
    packed_kv_cache: torch.Tensor,
    topk_indices: torch.Tensor,
    sm_scale: float,
    *,
    page_block_size: int = 64,
    d_v: int = DSV4_D_V,
    attn_sink: torch.Tensor | None = None,
    topk_length: torch.Tensor | None = None,
    kv_dequant: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """DSV4 *prefill* oracle: full BF16 O + base-2 LSE for MANY query tokens.

    Each of the ``T`` query tokens attends to its OWN topk-selected, dequantized
    KV rows in a single pass. Mathematically this is dense SDPA per token, which
    the prefill kernel computes via cross-tile online softmax (acc_o + m/l carried
    over all topk tiles). The result therefore equals the decode oracle applied to
    all T tokens at once — this function delegates to
    :func:`dsv4_ref.dsv4_decode_reference` (already vectorized over T) so the two
    references can never diverge.

    Args:
      q:               [T, num_heads, d_qk=512] bf16 (or float). T is unrestricted
                       (the prefill regime is T > 64).
      packed_kv_cache: (num_blocks, page_block_size, 1, 584) uint8 from
                       :func:`quantize_kv_dsv4`.
      topk_indices:    [T, topk] int32, flat slot ids (slot = block*pbs + local).
                       -1 = invalid sentinel.
      sm_scale:        softmax scale (typically d_qk**-0.5).
      page_block_size: tokens per KV block (64 for DSV4).
      d_v:             value width carried through PV (512; full union row).
      attn_sink:       optional [num_heads] fp32 per-head sink (FlashMLA V4). Folded
                       into BOTH the output (×sigmoid(lse_e - sink)) and the base-2
                       LSE (+log2(1 + exp2(sink_log2 - lse))).
      topk_length:     optional [T] int32 per-token valid length; entries at
                       position >= topk_length[t] are masked (exercises the kernel's
                       per-token actual_ni = ceil(topk_length / BI) partial tile).
      kv_dequant:      optional precomputed bf16 dequant of the cache (skips dequant;
                       the harness passes it to avoid recomputation).

    Returns:
      (O[T, num_heads, d_v] bf16, lse_log2[T, num_heads] fp32).
    """
    return dsv4_decode_reference(
        q,
        packed_kv_cache,
        topk_indices,
        sm_scale,
        page_block_size=page_block_size,
        d_v=d_v,
        attn_sink=attn_sink,
        topk_length=topk_length,
        kv_dequant=kv_dequant,
    )


# ── Test-case factory ─────────────────────────────────────────────────────────
def make_dsv4_prefill_case(
    num_tokens: int,
    num_heads: int = 128,
    topk: int = 512,
    *,
    num_blocks: int,
    page_block_size: int = 64,
    with_sink: bool = False,
    invalidate_half: bool = True,
    seed: int = 0,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> dict:
    """Build one self-consistent DSV4 *prefill* test case on ``device``.

    Mirrors test_sparse_mla_sm120.py::test_sparse_mla_sm120_prefill_dsv4 inputs
    (randn/10 clamp(-1,1) Q & KV, sm_scale = d_qk**-0.5, half-invalid topk) but
    in the prefill regime (T query tokens, typically > 64) and additionally emits
    a PER-TOKEN ``topk_length`` that is intentionally NOT a multiple of BI=64, so
    the harness exercises the kernel's variable actual_ni / last-partial-tile
    masking. ``topk_length`` is None only as a contract option; here it is always
    populated (and capped at ``topk``).

    Args:
      num_tokens:      T query tokens (prefill regime: > 64). Required.
      num_heads:       attention heads (DSV4 prefill envelope: 16/32/64/128).
      topk:            candidate count per token (128/512/1024/...). The valid
                       ``topk_indices`` columns; topk_length[t] <= topk.
      num_blocks:      KV pool blocks. Required (s_kv = num_blocks * pbs slots).
      page_block_size: tokens per KV block (64 for DSV4).
      with_sink:       emit an [num_heads] fp32 attn_sink and fold it.
      invalidate_half: set the back half of each token's topk_indices to -1
                       (the -1 sentinel convention the kernel masks at line 292).
      seed:            RNG seed.
      device / dtype:  placement / Q,KV dtype.

    Returns a dict of torch tensors:
      q              [T, num_heads, 512] bf16
      kv_cache       (num_blocks, page_block_size, 1, 584) uint8 (packed)
      kv_dequant     (num_blocks, page_block_size, 1, 512) bf16
      topk_indices   [T, topk] int32  (back half = -1 if invalidate_half)
      topk_lengths   [T] int32  per-token valid length (NON-multiple-of-64;
                     also surfaced as "topk_length" for symmetry with the kernel arg)
      sm_scale       float
      attn_sink      [num_heads] fp32 or None
      page_block_size int
      expected_O     [T, num_heads, 512] bf16   (reference output)
      expected_lse   [T, num_heads] fp32        (reference log2 LSE)
    """
    device = torch.device(device)
    gen = torch.Generator(device=device).manual_seed(seed)
    d_qk, d_v = DSV4_D_QK, DSV4_D_V
    s_kv = num_blocks * page_block_size

    kv_bf16 = (
        torch.randn(
            num_blocks,
            page_block_size,
            1,
            d_qk,
            device=device,
            dtype=torch.bfloat16,
            generator=gen,
        )
        / 10.0
    ).clamp(-1, 1)
    kv_packed = quantize_kv_dsv4(kv_bf16)
    kv_dequant = dequantize_kv_dsv4(kv_packed)

    q = (
        torch.randn(
            num_tokens,
            num_heads,
            d_qk,
            device=device,
            dtype=dtype,
            generator=gen,
        )
        / 10.0
    ).clamp(-1, 1)

    indices = torch.randint(
        0, s_kv, (num_tokens, topk), device=device, dtype=torch.int32, generator=gen
    )
    if invalidate_half and topk >= 2:
        indices[:, topk // 2 :] = -1

    # Per-token variable valid length, deliberately NON-multiple of BI=64 to
    # exercise the kernel's last-partial-tile masking (actual_ni = ceil(len/BI),
    # last tile has len % BI valid entries). We pick lengths spread across the
    # full [BI//2, topk] range and force each off a 64-boundary. Capped at topk.
    bi = DSV4_PREFILL_BI
    if topk >= bi:
        lo = bi // 2  # 32: guarantees the first tile is itself partial sometimes
        span = topk - lo
        # Spread deterministically across tokens, then perturb so none lands on a
        # multiple of 64 (offsets 13/27/41/55 cycle through the 4 lane-pair cases).
        base = lo + (
            torch.arange(num_tokens, device=device, dtype=torch.int64) * 37
        ) % max(span, 1)
        perturb = torch.tensor([13, 27, 41, 55], device=device, dtype=torch.int64)
        lengths = base + perturb[torch.arange(num_tokens, device=device) % 4]
        lengths = lengths.clamp(max=topk, min=1)
        # Defensively push any accidental multiple-of-64 off the boundary.
        on_boundary = (lengths % bi) == 0
        lengths = torch.where(on_boundary & (lengths < topk), lengths + 7, lengths)
        lengths = torch.where((lengths % bi) == 0, lengths - 7, lengths)
        lengths = lengths.clamp(max=topk, min=1)
    else:
        # topk < BI: a single partial tile; vary length within [1, topk].
        lengths = (
            1 + (torch.arange(num_tokens, device=device, dtype=torch.int64) * 5) % topk
        ).clamp(max=topk, min=1)
    topk_lengths = lengths.to(torch.int32)

    attn_sink = (
        torch.randn(num_heads, device=device, dtype=torch.float32, generator=gen) * 2.0
        if with_sink
        else None
    )

    sm_scale = d_qk**-0.5

    expected_O, expected_lse = dsv4_prefill_reference(
        q,
        kv_packed,
        indices,
        sm_scale,
        page_block_size=page_block_size,
        d_v=d_v,
        attn_sink=attn_sink,
        topk_length=topk_lengths,
        kv_dequant=kv_dequant,
    )

    return {
        "q": q,
        "kv_cache": kv_packed,
        "kv_dequant": kv_dequant,
        "topk_indices": indices,
        "topk_lengths": topk_lengths,
        "topk_length": topk_lengths,  # alias matching the kernel arg name
        "sm_scale": sm_scale,
        "attn_sink": attn_sink,
        "page_block_size": page_block_size,
        "expected_O": expected_O,
        "expected_lse": expected_lse,
    }


# ── Internal self-tests ───────────────────────────────────────────────────────
def _self_test(device: str | torch.device = "cuda") -> None:
    device = torch.device(device)
    torch.manual_seed(0)

    # (1) For T=1 the prefill reference is byte-for-byte the decode reference
    #     (same call, same per-token math). Confirm on a non-trivial T=1 case
    #     WITHOUT topk_length (so we compare against the plain decode oracle),
    #     and again WITH a per-token length to confirm the shared masking path.
    nb = 8
    q = (torch.randn(1, 16, DSV4_D_QK, device=device) / 10.0).clamp(-1, 1).bfloat16()
    kv_bf16 = (
        torch.randn(nb, 64, 1, DSV4_D_QK, device=device, dtype=torch.bfloat16) / 10.0
    ).clamp(-1, 1)
    packed = quantize_kv_dsv4(kv_bf16)
    deq = dequantize_kv_dsv4(packed)
    idx = torch.randint(0, nb * 64, (1, 128), device=device, dtype=torch.int32)
    idx[:, 64:] = -1
    sm_scale = DSV4_D_QK**-0.5

    o_dec, l_dec = dsv4_decode_reference(q, packed, idx, sm_scale, kv_dequant=deq)
    o_pre, l_pre = dsv4_prefill_reference(q, packed, idx, sm_scale, kv_dequant=deq)
    assert torch.equal(o_dec, o_pre), "T=1 prefill O != decode O (no length)"
    assert torch.equal(l_dec, l_pre), "T=1 prefill LSE != decode LSE (no length)"

    tl = torch.tensor([93], device=device, dtype=torch.int32)  # non-mult-of-64
    o_dec2, l_dec2 = dsv4_decode_reference(
        q, packed, idx, sm_scale, topk_length=tl, kv_dequant=deq
    )
    o_pre2, l_pre2 = dsv4_prefill_reference(
        q, packed, idx, sm_scale, topk_length=tl, kv_dequant=deq
    )
    assert torch.equal(o_dec2, o_pre2), "T=1 prefill O != decode O (with length)"
    assert torch.equal(l_dec2, l_pre2), "T=1 prefill LSE != decode LSE (with length)"

    # (2) BRUTE-FORCE dense cross-check on a tiny MULTI-token case with PER-TOKEN
    #     variable, non-multiple-of-64 topk_length. Independent plain-float
    #     softmax over each token's masked valid rows (no online softmax, no
    #     vectorized einsum) must match the reference.
    case = make_dsv4_prefill_case(
        num_tokens=5,
        num_heads=3,
        topk=128,
        num_blocks=3,
        invalidate_half=True,
        with_sink=False,
        device=device,
        seed=7,
    )
    qf = case["q"].float()
    deqf = case["kv_dequant"].view(-1, DSV4_D_QK).float()
    idx = case["topk_indices"]
    lengths = case["topk_lengths"]
    sms = case["sm_scale"]
    T, H, _ = qf.shape
    topk = idx.shape[-1]
    # Confirm the harness really produced non-multiple-of-64 lengths.
    assert torch.any((lengths % DSV4_PREFILL_BI) != 0), (
        "harness failed to produce non-multiple-of-64 topk_lengths"
    )
    bruteO = torch.zeros(T, H, DSV4_D_V, device=device)
    bruteLSE = torch.zeros(T, H, device=device)
    for t in range(T):
        L = int(lengths[t].item())
        cols = idx[t]
        valid_cols = [k for k in range(topk) if k < L and int(cols[k].item()) >= 0]
        if not valid_cols:
            bruteLSE[t] = float("-inf")
            continue
        rows = deqf.index_select(
            0, cols[torch.tensor(valid_cols, device=device)].long()
        )  # [nvalid, d_qk]
        for h in range(H):
            logits = (qf[t, h] @ rows.t()) * sms  # [nvalid]
            m = logits.max()
            w = torch.exp(logits - m)
            denom = w.sum()
            bruteO[t, h] = (w @ rows[:, :DSV4_D_V]) / denom
            bruteLSE[t, h] = (m + torch.log(denom)) / math.log(2.0)
    torch.testing.assert_close(case["expected_O"].float(), bruteO, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(case["expected_lse"], bruteLSE, atol=1e-3, rtol=1e-3)

    # (3) PREFILL regime sanity: T > 64 tokens, full head count, finite outputs,
    #     correct shapes, and the harness emits per-token non-mult-of-64 lengths.
    big = make_dsv4_prefill_case(
        num_tokens=128,
        num_heads=128,
        topk=512,
        num_blocks=64,
        with_sink=False,
        device=device,
        seed=0,
    )
    assert big["q"].shape == (128, 128, DSV4_D_QK)
    assert big["expected_O"].shape == (128, 128, DSV4_D_V)
    assert big["expected_lse"].shape == (128, 128)
    assert big["kv_cache"].shape[-1] == DSV4_KV_GMEM_STRIDE
    assert big["topk_lengths"].shape == (128,)
    assert torch.all(big["topk_lengths"] <= 512) and torch.all(big["topk_lengths"] >= 1)
    assert torch.any((big["topk_lengths"] % DSV4_PREFILL_BI) != 0)
    assert torch.isfinite(big["expected_O"].float()).all()
    # Every token has >=1 valid candidate (back half -1, lengths>=1) → finite LSE.
    assert torch.isfinite(big["expected_lse"]).all()

    # (4) ATTN_SINK fold matches a DIRECT softmax-with-sink computation, computed
    #     two independent ways:
    #       (a) the reference's "out *= sigmoid(lse_e - sink)" + log2 LSE fold,
    #       (b) the KERNEL's normalizer form: il = 1 / (l + exp2(sink_log2 - m)),
    #           and lse = (l>0) ? log2(l)+m : -1e30, then +log2(1+exp2(sink-lse)).
    #     (a) and (b) are algebraically equal; verifying both pins the formula.
    case_s = make_dsv4_prefill_case(
        num_tokens=6,
        num_heads=4,
        topk=128,
        num_blocks=4,
        invalidate_half=True,
        with_sink=True,
        device=device,
        seed=3,
    )
    qf = case_s["q"].float()
    deqf = case_s["kv_dequant"].view(-1, DSV4_D_QK).float()
    idx = case_s["topk_indices"]
    lengths = case_s["topk_lengths"]
    sms = case_s["sm_scale"]
    sink = case_s["attn_sink"].float()  # [H], natural-log domain
    T, H, _ = qf.shape
    topk = idx.shape[-1]
    LOG2E = 1.0 / math.log(2.0)
    ln2 = math.log(2.0)

    directO = torch.zeros(T, H, DSV4_D_V, device=device)
    directLSE = torch.zeros(T, H, device=device)
    for t in range(T):
        Lk = int(lengths[t].item())
        cols = idx[t]
        valid_cols = [k for k in range(topk) if k < Lk and int(cols[k].item()) >= 0]
        for h in range(H):
            sink_h = float(sink[h].item())
            sink_log2 = sink_h * LOG2E
            if not valid_cols:
                # empty row: kernel sets lse = sink_log2, output 0.
                directLSE[t, h] = sink_log2
                continue
            rows = deqf.index_select(
                0, cols[torch.tensor(valid_cols, device=device)].long()
            )
            logits = (qf[t, h] @ rows.t()) * sms  # natural-log domain, scaled
            m_nat = logits.max()
            w = torch.exp(logits - m_nat)
            l_nat = w.sum()  # exp-domain row sum (natural, rel. to m_nat)
            lse_e = m_nat + torch.log(l_nat)  # natural-log LSE
            # (a) output sink scaling: out = (w@V / l) * sigmoid(lse_e - sink)
            base_out = (w @ rows[:, :DSV4_D_V]) / l_nat
            factor = torch.sigmoid(lse_e - sink_h)
            directO[t, h] = base_out * factor
            # (b) kernel normalizer form, base-2 domain: m2 = m_nat*LOG2E,
            #     l2 = sum exp2(s2 - m2) where s2 = logits*LOG2E.
            m2 = float((m_nat * LOG2E).item())
            s2 = logits * LOG2E
            l2 = torch.exp2(s2 - m2).sum()
            il_kernel = 1.0 / (l2 + math.pow(2.0, sink_log2 - m2))
            out_kernel = (torch.exp2(s2 - m2) @ rows[:, :DSV4_D_V]) * il_kernel
            torch.testing.assert_close(
                base_out * factor, out_kernel, atol=1e-4, rtol=1e-4
            )
            # base-2 LSE with fold
            lse2 = float((torch.log2(l2) + m2).item())
            directLSE[t, h] = lse2 + math.log2(1.0 + math.pow(2.0, sink_log2 - lse2))
    torch.testing.assert_close(
        case_s["expected_O"].float(), directO, atol=2e-2, rtol=2e-2
    )
    torch.testing.assert_close(case_s["expected_lse"], directLSE, atol=1e-2, rtol=1e-2)

    # (5) all-invalid token: with NO sink, lse = -inf and output 0; WITH sink,
    #     lse = sink_log2 (per the kernel's empty-row branch) and output 0.
    case_e = make_dsv4_prefill_case(
        num_tokens=3,
        num_heads=4,
        topk=128,
        num_blocks=4,
        invalidate_half=True,
        with_sink=False,
        device=device,
        seed=2,
    )
    case_e["topk_indices"][:] = -1
    O_e, lse_e = dsv4_prefill_reference(
        case_e["q"],
        case_e["kv_cache"],
        case_e["topk_indices"],
        case_e["sm_scale"],
        topk_length=case_e["topk_lengths"],
        kv_dequant=case_e["kv_dequant"],
    )
    assert torch.all(lse_e == float("-inf")), "all-invalid LSE must be -inf (no sink)"
    assert torch.all(O_e.float() == 0.0), "all-invalid output must be 0"

    sink_vec = torch.randn(4, device=device) * 2.0
    O_es, lse_es = dsv4_prefill_reference(
        case_e["q"],
        case_e["kv_cache"],
        case_e["topk_indices"],
        case_e["sm_scale"],
        attn_sink=sink_vec,
        topk_length=case_e["topk_lengths"],
        kv_dequant=case_e["kv_dequant"],
    )
    assert torch.all(O_es.float() == 0.0), "all-invalid sink output must be 0"
    exp_sink_lse = (sink_vec.float() / ln2).unsqueeze(0).expand(3, 4)
    torch.testing.assert_close(lse_es, exp_sink_lse, atol=1e-4, rtol=1e-4)

    # (6) topk / head / token sweep produces finite, correctly-shaped outputs and
    #     non-mult-of-64 lengths across the DSV4 prefill envelope.
    for nh, tk in ((16, 128), (32, 512), (64, 1024)):
        for T in (128, 256):
            c = make_dsv4_prefill_case(
                num_tokens=T,
                num_heads=nh,
                topk=tk,
                num_blocks=64,
                with_sink=(T == 256),
                device=device,
                seed=1,
            )
            assert c["q"].shape == (T, nh, DSV4_D_QK)
            assert c["expected_O"].shape == (T, nh, DSV4_D_V)
            assert c["expected_lse"].shape == (T, nh)
            assert c["topk_lengths"].shape == (T,)
            assert torch.all(c["topk_lengths"] <= tk)
            assert torch.any((c["topk_lengths"] % DSV4_PREFILL_BI) != 0)
            assert torch.isfinite(c["expected_O"].float()).all()
            assert torch.isfinite(c["expected_lse"]).all()

    print(
        "prefill_ref self-tests PASSED "
        "(T=1==decode / brute-force-variable-length / prefill-regime / "
        "attn_sink-fold / all-invalid / sweep)"
    )


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    _self_test(dev)
