"""Isolated Q block quantization and fused MLA K concat + K/V quantization."""

import torch
import triton
import triton.language as tl


@triton.jit
def _quant_q(
    X,
    CU,
    Y,
    S,
    H: tl.constexpr,
    D: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    T,
    BN: tl.constexpr,
):
    block, head, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    begin, end = tl.load(CU + batch), tl.load(CU + batch + 1)
    row = begin + block * BN + tl.arange(0, BN)
    col = tl.arange(0, D)
    x = tl.load(
        X + row[:, None] * S0 + head * S1 + col[None, :], row[:, None] < end, 0
    ).to(tl.float32)
    scale = tl.maximum(tl.max(tl.max(tl.abs(x), 1), 0) / 448.0, 1.0e-12)
    tl.store(
        Y + row[:, None] * H * D + head * D + col[None, :],
        (x / scale).to(tl.float8e4nv),
        row[:, None] < end,
    )
    tl.store(S + (batch * H + head) * T + block, scale)


@triton.jit
def _concat_quant_kv(
    KN,
    KR,
    V,
    CU,
    K8,
    V8,
    SK,
    SV,
    H: tl.constexpr,
    D: tl.constexpr,
    DN: tl.constexpr,
    NS0: tl.constexpr,
    NS1: tl.constexpr,
    RS0: tl.constexpr,
    VS0: tl.constexpr,
    VS1: tl.constexpr,
    T,
    BN: tl.constexpr,
):
    block, head, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    begin, end = tl.load(CU + batch), tl.load(CU + batch + 1)
    row = begin + block * BN + tl.arange(0, BN)
    col = tl.arange(0, D)
    kn = tl.load(
        KN + row[:, None] * NS0 + head * NS1 + col[None, :],
        (row[:, None] < end) & (col[None, :] < DN),
        0,
    ).to(tl.float32)
    kr = tl.load(
        KR + row[:, None] * RS0 + col[None, :] - DN,
        (row[:, None] < end) & (col[None, :] >= DN),
        0,
    ).to(tl.float32)
    k = kn + kr
    sk = tl.maximum(tl.max(tl.max(tl.abs(k), 1), 0) / 448.0, 1.0e-12)
    tl.store(
        K8 + row[:, None] * H * D + head * D + col[None, :],
        (k / sk).to(tl.float8e4nv),
        row[:, None] < end,
    )
    tl.store(SK + (batch * H + head) * T + block, sk)
    v = tl.load(
        V + row[:, None] * VS0 + head * VS1 + col[None, :], row[:, None] < end, 0
    ).to(tl.float32)
    sv = tl.maximum(tl.max(tl.max(tl.abs(v), 1), 0) / 448.0, 1.0e-12)
    tl.store(
        V8 + row[:, None] * H * D + head * D + col[None, :],
        (v / sv).to(tl.float8e4nv),
        row[:, None] < end,
    )
    tl.store(SV + (batch * H + head) * T + block, sv)


def prepare(q, k_nope, k_rope, v, cu_q, cu_k, max_q, max_k, tile_m=128, tile_n=64):
    assert q.ndim == 3 and q.shape[2] == v.shape[2] == 256
    assert k_nope.shape == (v.shape[0], q.shape[1], 192)
    assert k_rope.shape == (v.shape[0], 1, 64)
    assert tile_m in (64, 128) and tile_n in (64, 128)
    assert cu_q.numel() == cu_k.numel()
    b, h = cu_q.numel() - 1, q.shape[1]
    tq, tk = max(1, triton.cdiv(max_q, tile_m)), max(1, triton.cdiv(max_k, tile_n))
    q8 = torch.empty(q.shape, device=q.device, dtype=torch.float8_e4m3fn)
    k8 = torch.empty(v.shape, device=v.device, dtype=torch.float8_e4m3fn)
    v8 = torch.empty_like(k8)
    sq = torch.empty((b, h, tq), device=q.device, dtype=torch.float32)
    sk = torch.empty((b, h, tk), device=q.device, dtype=torch.float32)
    sv = torch.empty_like(sk)
    _quant_q[(tq, h, b)](
        q, cu_q, q8, sq, h, 256, q.stride(0), q.stride(1), tq, tile_m, num_warps=8
    )
    _concat_quant_kv[(tk, h, b)](
        k_nope,
        k_rope,
        v,
        cu_k,
        k8,
        v8,
        sk,
        sv,
        h,
        256,
        192,
        k_nope.stride(0),
        k_nope.stride(1),
        k_rope.stride(0),
        v.stride(0),
        v.stride(1),
        tk,
        tile_n,
        num_warps=8,
    )
    return q8, k8, v8, [sq, sk, sv]
