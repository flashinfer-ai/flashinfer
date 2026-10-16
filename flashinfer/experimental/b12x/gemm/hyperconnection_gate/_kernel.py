"""Tensor-core projection with BF16 gate rounding and stream reduction."""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["rows"])
def projection_gate(
    source,
    weight,
    normalized,
    out,
    rows,
    H: tl.constexpr,
    S: tl.constexpr,
    R: tl.constexpr,
    BM: tl.constexpr,
    BH: tl.constexpr,
    BK: tl.constexpr,
):
    m = (tl.program_id(0) * BM + tl.arange(0, BM)).to(tl.int64)
    lanes = tl.arange(0, S * BH)
    h = tl.program_id(1) * BH + lanes % BH
    n = (lanes // BH * H + h).to(tl.int64)
    kk = tl.arange(0, BK)
    acc = tl.full((BM, S * BH), 0, tl.float32)
    for step in range(tl.cdiv(R, BK)):
        k = step * BK + kk
        x = tl.load(
            source + m[:, None] * R + k[None, :],
            (m[:, None] < rows) & (k[None, :] < R),
            0,
        )
        w = tl.load(
            weight + n[None, :] * R + k[:, None], (h[None, :] < H) & (k[:, None] < R), 0
        )
        acc = tl.dot(x, w, acc)
    # Each stream's projection, sigmoid and product rounds before summation.
    logits = acc.to(tl.bfloat16).to(tl.float32)
    gate = tl.sigmoid(logits).to(tl.bfloat16)
    values = tl.load(
        normalized + m[:, None] * (S * H) + n[None, :],
        (m[:, None] < rows) & (h[None, :] < H),
        0,
    )
    products = (gate.to(tl.float32) * values.to(tl.float32)).to(tl.bfloat16)
    grouped = products.to(tl.float32).reshape((BM, S, BH))
    # Preserve the gate-reduction kernel's left-to-right FP32 additions.
    total = tl.full((BM, BH), 0, tl.float32)
    for stream in tl.static_range(S):
        total += tl.sum(
            tl.where(tl.arange(0, S)[None, :, None] == stream, grouped, 0), 1
        )
    output_h = tl.program_id(1) * BH + tl.arange(0, BH)
    tl.store(
        out + m[:, None] * H + output_h[None, :],
        total / S,
        (m[:, None] < rows) & (output_h[None, :] < H),
    )
