"""Load-time lossless GPU encoding of native FP4 block-scale bytes.

Each row keeps the lowest base covering the most of its scale bytes in a
sixteen-value (NVFP4) or two-value (MXFP4) window. Bytes outside that window
are stored verbatim as exceptions. The output is the slab checkpoint encoding,
so weight preparation consumes it exactly like compressed checkpoint scales.
No floating-point conversion is performed. Sizing the exception stream takes
one host synchronization; this is a load-time operation, not graph replay.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_BITS = {"nvfp4": 4, "mxfp4": 1}


@triton.jit
def _choose_bases(
    S,
    Bases,
    Counts,
    C: tl.constexpr,
    W: tl.constexpr,
    B: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, B)
    values = tl.load(S + row * C + col, col < C, 0).to(tl.int32)
    histogram = tl.histogram(values, 256, mask=col < C)
    ends = tl.cumsum(histogram, 0)
    base = tl.arange(0, 256)
    right = tl.gather(ends, tl.minimum(base + W - 1, 255), 0)
    left = tl.where(base > 0, tl.gather(ends, tl.maximum(base - 1, 0), 0), 0)
    covered = tl.where(base <= 256 - W, right - left, -1)
    best = tl.max(covered, 0)
    chosen = tl.min(tl.where(covered == best, base, 256), 0)
    tl.store(Bases + row, chosen.to(tl.uint8))
    tl.store(Counts + row, (C - best).to(tl.int64))


@triton.jit
def _pack_rows(
    S,
    Bases,
    Prefix,
    Fixed,
    Exceptions,
    R: tl.constexpr,
    C: tl.constexpr,
    BITS: tl.constexpr,
    B: tl.constexpr,
):
    # One program per row of the concatenated [expert, row] grid. A 16-row slab
    # holds sixteen bases, then each row's codes in column order, low bits first.
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, B)
    valid = col < C
    values = tl.load(S + row * C + col, valid, 0).to(tl.int32)
    base = tl.load(Bases + row).to(tl.int32)
    delta = values - base
    outside = valid & ((delta < 0) | (delta >= (1 << BITS)))
    rank = tl.cumsum(outside.to(tl.int32), 0) - 1
    start = tl.load(Prefix + row)
    record = ((row % R) * C + col).to(tl.uint32) | (values.to(tl.uint32) << 24)
    tl.store(Exceptions + start + rank, record, outside)
    width: tl.constexpr = (C * BITS + 7) // 8
    per: tl.constexpr = 8 // BITS
    slab = row // 16 * (16 * (1 + width))
    tl.store(Fixed + slab + row % 16, base.to(tl.uint8))
    codes = tl.where(valid & ~outside, delta, 0)
    shift = (tl.arange(0, per) * BITS)[None, :]
    packed = tl.sum(tl.reshape(codes, (B // per, per)) << shift, 1)
    byte = tl.arange(0, B // per)
    tl.store(
        Fixed + slab + 16 + row % 16 * width + byte, packed.to(tl.uint8), byte < width
    )


def encode_csf_source(scales: torch.Tensor, *, quantization: str):
    """Encode logical ``[expert, row, column]`` scale bytes as a slab ``CsfSource``.

    Each expert is one segment. Source bytes are never modified. Dense
    exceptions remain exact but can make the result larger than its input.
    """
    from b12x.loader._csf_source import CsfSource

    bits = _BITS.get(quantization)
    if bits is None:
        raise ValueError("CSF quantization must be nvfp4 or mxfp4")
    if (
        scales.dtype != torch.uint8
        or scales.ndim != 3
        or scales.device.type != "cuda"
        or not scales.is_contiguous()
    ):
        raise ValueError("CSF encoding requires contiguous CUDA uint8 [E,R,C] scales")
    e, r, c = scales.shape
    if e <= 0 or r <= 0 or r % 32 or c <= 0 or r * c > 1 << 24:
        raise ValueError("CSF encoding requires 32-row blocks and 24-bit positions")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("CSF encoding is a load-time operation, not graph replay")
    rows = e * r
    block = max(triton.next_power_of_2(c), 8 // bits)
    with torch.cuda.device(scales.device):
        bases = torch.empty(rows, dtype=torch.uint8, device=scales.device)
        counts = torch.empty(rows, dtype=torch.int64, device=scales.device)
        _choose_bases[(rows,)](scales, bases, counts, C=c, W=1 << bits, B=block)
        prefix = torch.zeros(rows + 1, dtype=torch.int64, device=scales.device)
        torch.cumsum(counts, 0, out=prefix[1:])
        exceptions = torch.empty(
            int(prefix[-1].item()), dtype=torch.uint32, device=scales.device
        )
        fixed = torch.empty(
            (e, r // 16, 16 * (1 + (c * bits + 7) // 8)),
            dtype=torch.uint8,
            device=scales.device,
        )
        _pack_rows[(rows,)](
            scales, bases, prefix, fixed, exceptions, R=r, C=c, BITS=bits, B=block
        )
    source = CsfSource(
        (fixed, exceptions), prefix[::r].contiguous(), r, c, quantization, e, "csf"
    )
    source.validate()
    return source
