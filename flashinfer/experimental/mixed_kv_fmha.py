# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
import triton
import triton.language as tl

# Triton lowers a 16-row tile to FP16 MMA on Blackwell, even for FP8 inputs.
TILE_M, TILE_N = 64, 64


@triton.jit
def _expand_v(v16, scale, N: tl.constexpr, D: tl.constexpr):
    # One uint16 contains four consecutive E2M1 channels, low nibble first.
    word = tl.inline_asm_elementwise(
        asm="mul.rn.satfinite.e4m3x4.e2m1x4.e4m3x4 $0, $1, $2;",
        constraints="=r,h,r",
        args=[v16, scale.to(tl.uint32) * 0x01010101],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )
    even = tl.join(word.to(tl.uint8), (word >> 16).to(tl.uint8))
    odd = tl.join((word >> 8).to(tl.uint8), (word >> 24).to(tl.uint8))
    return tl.reshape(tl.join(even, odd), (N, D)).to(tl.float8e4nv, bitcast=True)


@triton.jit
def _paged_mixed_qmul_fmha(
    Q,
    K,
    V16,
    SF,
    TABLE,
    LENS,
    INDPTR,
    OUT,
    q_str: tl.constexpr,
    k_str: tl.constexpr,
    v_str: tl.constexpr,
    sf_str: tl.constexpr,
    table_str: tl.constexpr,
    o_str: tl.constexpr,
    qk_scale,
    v_scale,
    GROUP: tl.constexpr,
    PAGE: tl.constexpr,
    D: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
):
    tile, kv_head, req = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    start = tl.load(INDPTR + req)
    qlen = tl.load(INDPTR + req + 1) - start
    if tile * M >= qlen * GROUP:
        return
    seqlen = tl.load(LENS + req)
    row = tile * M + tl.arange(0, M)
    t = row // GROUP
    h = kv_head * GROUP + row % GROUP
    d = tl.arange(0, D)
    qtoken = (start + t).to(tl.int64)
    q = tl.load(
        Q + (qtoken * q_str[0] + h * q_str[1])[:, None] + d[None, :],
        t[:, None] < qlen,
        0.0,
    )
    acc = tl.full((M, D), 0, tl.float32)
    denom = tl.full((M,), 0, tl.float32)
    m = tl.full((M,), -float("inf"), tl.float32)
    pos = seqlen - qlen + t
    limit = tl.minimum(seqlen, seqlen - qlen + ((tile + 1) * M - 1) // GROUP + 1)
    for block in range(tl.cdiv(limit, N)):
        n = block * N + tl.arange(0, N)
        page = tl.load(TABLE + req * table_str + n // PAGE, n < limit, 0).to(tl.int64)
        token = n % PAGE
        kb = page * k_str[0] + kv_head * k_str[1] + token * k_str[2]
        k = tl.load(K + kb[:, None] + d[None, :], n[:, None] < limit, 0.0)
        logits = tl.dot(q, tl.trans(k)) * qk_scale
        valid = (n[None, :] < limit) & (n[None, :] <= pos[:, None])
        logits = tl.where(valid, logits, -float("inf"))
        new_m = tl.maximum(m, tl.max(logits, 1))
        safe_m = tl.where(new_m == -float("inf"), 0.0, new_m)
        alpha = tl.exp(m - safe_m)
        p = tl.exp(logits - safe_m[:, None])
        denom = denom * alpha + tl.sum(p, 1)

        quartet = tl.arange(0, D // 4)
        vb = page * v_str[0] + kv_head * v_str[1] + token * v_str[2]
        v16 = tl.load(V16 + vb[:, None] + quartet[None, :], n[:, None] < limit, 0)
        # The packed writer stores [token//4, scale_hi, scale_lo, token%4].
        # Preserve the physical token stride for NHD as well as HND pages.
        group = quartet // 4
        tile_t = (token // 4 * 4)[:, None] + (group // (D // 64))[None, :]
        tile_s = (group % (D // 64) * 4)[None, :] + (token % 4)[:, None]
        sb = page * sf_str[0] + kv_head * sf_str[1]
        sf = tl.load(
            SF + sb[:, None] + tile_t * sf_str[2] + tile_s, n[:, None] < limit, 0
        )
        v = _expand_v(v16, sf, N, D)
        pv = tl.dot((p * 448.0).to(tl.float8e4nv), v)
        acc = acc * alpha[:, None] + pv * (1.0 / 448.0)
        m = new_m

    result = acc * (v_scale / tl.where(denom > 0, denom, 1.0))[:, None]
    tl.store(
        OUT + (qtoken * o_str[0] + h * o_str[1])[:, None] + d[None, :],
        result,
        t[:, None] < qlen,
    )


def run(q, k, v, sf, table, seq_lens, qo_indptr, max_q_len, out, qk_scale, v_scale):
    import torch

    if not q.is_cuda or torch.cuda.get_device_capability(q.device) not in (
        (10, 0),
        (10, 3),
    ):
        raise NotImplementedError(
            "packed QMUL requires SM100a or SM103a and ptxas >= 13.4"
        )
    if any(x.device != q.device for x in (k, v, sf, table, seq_lens, qo_indptr, out)):
        raise ValueError("all mixed FMHA tensors must be on the query device")
    fp8 = torch.float8_e4m3fn
    if (q.dtype, k.dtype, v.dtype, sf.dtype, out.dtype) != (
        fp8,
        fp8,
        torch.uint8,
        fp8,
        torch.bfloat16,
    ):
        raise ValueError(
            "expected E4M3 Q/K, uint8 packed V, E4M3 V scales, BF16 output"
        )
    if q.ndim != 3 or k.ndim != 4 or q.shape != out.shape:
        raise ValueError("expected Q/output [T,Hq,D] and cache [pages,Hkv,page_size,D]")
    pages, hkv, page_size, d = k.shape
    if (
        min(pages, hkv, page_size, q.shape[1]) < 1
        or page_size % 4
        or d not in (128, 256)
        or q.shape[2] != d
        or q.shape[1] % hkv
        or v.shape != (*k.shape[:3], d // 2)
        or sf.shape != (*k.shape[:3], d // 16)
    ):
        raise ValueError("incompatible logical cache dimensions or GQA head counts")
    if any(x.stride(-1) != 1 for x in (q, k, v, sf, out, table, seq_lens, qo_indptr)):
        raise ValueError("mixed FMHA requires unit innermost strides")
    if any(x.dtype != torch.int32 for x in (table, seq_lens, qo_indptr)):
        raise ValueError(
            "block table, sequence lengths and query offsets must be int32"
        )
    if (
        table.ndim != 2
        or seq_lens.shape != (table.shape[0],)
        or qo_indptr.shape != (table.shape[0] + 1,)
        or max_q_len < 0
    ):
        raise ValueError("incompatible ragged batch metadata")
    if not q.shape[0]:
        return out
    if max_q_len == 0 or not table.numel():
        raise ValueError(
            "nonempty queries require a positive launch bound and page table"
        )
    # Split decode across the KV dimension rather than serializing a long
    # request in one CTA. Limit dispatch to the measured GQA geometry.
    if (
        max_q_len == 1
        and (
            (table.shape[0] <= 4 and table.shape[1] <= 512)
            or (table.shape[0] <= 64 and table.shape[1] <= 160)
        )
        and hkv == 2
        and q.shape[1] == 8
        and d == 128
        and page_size == 64
    ):
        from .mixed_kv_decode import run_decode

        return run_decode(
            q, k, v, sf, table, seq_lens, qo_indptr, out, qk_scale, v_scale
        )
    # The dtype views preserve storage offsets and physical page/row strides.
    v16 = v.view(torch.uint16)
    group = q.shape[1] // hkv
    with torch.cuda.device(q.device):
        _paged_mixed_qmul_fmha[
            (triton.cdiv(max_q_len * group, TILE_M), hkv, table.shape[0])
        ](
            q,
            k,
            v16,
            sf.view(torch.uint8),
            table,
            seq_lens,
            qo_indptr,
            out,
            q.stride(),
            k.stride(),
            v16.stride(),
            sf.stride(),
            table.stride(0),
            out.stride(),
            qk_scale,
            v_scale,
            group,
            page_size,
            d,
            TILE_M,
            TILE_N,
            num_warps=4,
            num_stages=1,
            ptx_version=94,
        )
    return out
