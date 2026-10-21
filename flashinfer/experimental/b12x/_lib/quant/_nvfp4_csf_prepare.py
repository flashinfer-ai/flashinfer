"""CUDA checkpoint slicing, joining and compressed scale layout preparation."""

import torch
import triton
import triton.language as tl


@triton.jit
def _fixed(
    Src,
    Dst,
    Error,
    SR: tl.constexpr,
    SC: tl.constexpr,
    R0: tl.constexpr,
    C0: tl.constexpr,
    LR: tl.constexpr,
    LC: tl.constexpr,
    JOIN: tl.constexpr,
    R: tl.constexpr,
    C: tl.constexpr,
    LAYOUT: tl.constexpr,
    ROTATE: tl.constexpr,
    B: tl.constexpr,
):
    expert = tl.program_id(1)
    i = tl.program_id(0) * B + tl.arange(0, B)
    slab_size: tl.constexpr = 128 * (1 + C // 2)
    slab, offset = i // slab_size, i % slab_size
    code = offset - 128
    if LAYOUT == 0:
        row = tl.where(
            offset < 128,
            offset % 4 * 32 + offset // 4,
            code // 2 % 4 * 32 + code // 8 % 32,
        )
        column = (code // 256 * 2 + code % 2) * 2
        row2 = row
    else:
        row = tl.where(offset < 128, offset, code * 2 % 128)
        row2 = row + 1
        swapped = row // 4 * 4 + row % 2 * 2 + row % 4 // 2
        swapped2 = row2 // 4 * 4 + row2 % 2 * 2 + row2 % 4 // 2
        row = swapped // 64 * 64 + swapped % 8 * 8 + swapped % 64 // 8
        row2 = swapped2 // 64 * 64 + swapped2 % 8 * 8 + swapped2 % 64 // 8
        column = code * 2 // 128
    row += slab * 128
    row2 += slab * 128
    logical = (row + ROTATE) % (LR * JOIN)
    logical2 = (row2 + ROTATE) % (LR * JOIN)
    source_row = logical % LR + R0
    source_row2 = logical2 % LR + R0
    source_segment = expert * JOIN + logical // LR
    source_segment2 = expert * JOIN + logical2 // LR
    stride: tl.constexpr = 16 * (1 + SC // 2)
    base = source_segment * (SR // 16 * stride) + source_row // 16 * stride
    base2 = source_segment2 * (SR // 16 * stride) + source_row2 // 16 * stride
    is_base = offset < 128
    source = base + tl.where(
        is_base, source_row % 16, 16 + source_row % 16 * (SC // 2) + (column + C0) // 2
    )
    valid = (i < R * (1 + C // 2)) & (row < LR * JOIN)
    value = tl.load(Src + source, valid & (is_base | (column < LC)), 0)
    if LAYOUT == 1:
        source2 = base2 + 16 + source_row2 % 16 * (SC // 2) + (column + C0) // 2
        value2 = tl.load(
            Src + source2,
            (i < R * (1 + C // 2)) & ~is_base & (row2 < LR * JOIN) & (column < LC),
            0,
        )
        shift = (column + C0) % 2 * 4
        value = tl.where(
            is_base, value, ((value >> shift) & 15) | (((value2 >> shift) & 15) << 4)
        )
    if tl.sum((valid & is_base & (value > 240)).to(tl.int32), 0):
        tl.atomic_or(Error, 1)
    tl.store(Dst + expert * (R * (1 + C // 2)) + i, value, i < R * (1 + C // 2))


@triton.jit
def _exceptions(
    Src,
    Starts,
    Words,
    Keys,
    Error,
    N: tl.constexpr,
    SEGMENTS: tl.constexpr,
    SR: tl.constexpr,
    SC: tl.constexpr,
    R0: tl.constexpr,
    R1: tl.constexpr,
    C0: tl.constexpr,
    C1: tl.constexpr,
    JOIN: tl.constexpr,
    R: tl.constexpr,
    C: tl.constexpr,
    ROTATE: tl.constexpr,
    CODEC: tl.constexpr,
    B: tl.constexpr,
):
    i = tl.program_id(0) * B + tl.arange(0, B)
    valid = i < N
    lo = tl.full((B,), 0, tl.int32)
    hi = tl.full((B,), SEGMENTS, tl.int32)
    while tl.sum((lo < hi).to(tl.int32), 0) > 0:
        mid = (lo + hi + 1) // 2
        start = tl.load(Starts + mid)
        right = start <= i
        hi = tl.where((lo < hi) & ~right, mid - 1, hi)
        lo = tl.where((lo < hi) & right, mid, lo)
    segment = tl.minimum(lo, SEGMENTS - 1)
    start = tl.load(Starts + segment)
    width: tl.constexpr = 3 if CODEC == 2 else 4
    bits: tl.constexpr = 19 if CODEC == 2 else 24
    word = tl.full((B,), 0, tl.uint32)
    previous = tl.full((B,), 0, tl.uint32)
    for byte in tl.static_range(width):
        word |= tl.load(Src + i * width + byte, valid, 0).to(tl.uint32) << (8 * byte)
        previous |= tl.load(Src + (i - 1) * width + byte, valid & (i > start), 0).to(
            tl.uint32
        ) << (8 * byte)
    position = word & ((1 << bits) - 1)
    prev_position = previous & ((1 << bits) - 1)
    invalid = valid & (
        (position >= SR * SC) | ((i > start) & (prev_position >= position))
    )
    if tl.sum(invalid.to(tl.int32), 0):
        tl.atomic_or(Error, 2)
    row, column = position // SC, position % SC
    keep = (row >= R0) & (row < R1) & (column >= C0) & (column < C1)
    logical_rows: tl.constexpr = (R1 - R0) * JOIN
    row = (row - R0 + segment % JOIN * (R1 - R0) + logical_rows - ROTATE) % logical_rows
    position = row * C + column - C0
    word = (word & ~tl.full((), (1 << bits) - 1, tl.uint32)) | position.to(tl.uint32)
    key = (segment // JOIN).to(tl.int64) * (R * C) + position.to(tl.int64)
    tl.store(Words + i, word.to(tl.int64), valid)
    tl.store(Keys + i, tl.where(keep, key, -1), valid)


def convert_nvfp4_csf_storage(
    fixed,
    exceptions,
    starts,
    *,
    rows,
    columns,
    join=1,
    row_range=None,
    column_range=None,
    codec=0,
    layout=0,
    row_rotation=0,
    value_lut=None,
):
    """Convert contiguous CUDA checkpoint segments to an expert-major CSF batch.

    ``starts`` contains exception-record offsets for each segment plus its end.
    Consecutive ``join`` segments become one expert, in row order. Slicing is
    applied within each segment before joining and optional row rotation.
    Offset and payload validation synchronize; this is a loading operation.
    """
    from .nvfp4_csf import Nvfp4CsfBatch

    if codec not in (0, 1, 2) or layout not in (0, 1) or (codec and layout):
        raise ValueError("Unsupported NVFP4-CSF codec or layout")
    if rows <= 0 or rows % 16 or columns <= 0 or columns % (2 if codec == 0 else 8):
        raise ValueError("Unsupported NVFP4-CSF geometry")
    if codec and (
        rows % 128 or join != 1 or row_range is not None or column_range is not None
    ):
        raise ValueError("Scale slicing and joining require the byte-window codec")
    ranges = []
    for selected, extent, alignment in (
        (row_range, rows, 16),
        (column_range, columns, 2),
    ):
        first, last = (0, extent) if selected is None else selected
        if not 0 <= first < last <= extent or first % alignment or last % alignment:
            raise ValueError("CSF slices must be aligned, nonempty and in range")
        ranges.append((first, last))
    (r0, r1), (c0, c1) = ranges
    if (
        join not in (1, 2)
        or fixed.ndim != 3
        or not fixed.shape[0]
        or fixed.shape[0] % join
    ):
        raise ValueError("CSF segment count must be a nonzero multiple of join")
    segments = fixed.shape[0]
    experts = segments // join
    logical_rows, logical_columns = (r1 - r0) * join, c1 - c0
    r, c = triton.cdiv(logical_rows, 128) * 128, triton.cdiv(logical_columns, 4) * 4
    if r * c > 1 << (19 if codec == 2 else 24):
        raise ValueError("NVFP4-CSF position field is too small for the matrix")
    if row_rotation and (codec or not 0 < row_rotation < logical_rows):
        raise ValueError(
            "Row rotation requires the byte-window codec and a valid split"
        )
    width = 3 if codec == 2 else 4
    if (
        fixed.dtype != torch.uint8
        or fixed.shape[1:] != (rows // 16, 16 * (1 + columns // 2))
        or exceptions.dtype != torch.uint8
        or exceptions.ndim != 1
        or exceptions.numel() % width
        or starts.dtype != torch.int64
        or starts.shape != (segments + 1,)
        or fixed.device.type != "cuda"
        or any(
            t.device != fixed.device or not t.is_contiguous()
            for t in (fixed, exceptions, starts)
        )
    ):
        raise ValueError("Invalid contiguous CUDA CSF storage")
    n = exceptions.numel() // width
    invalid_offsets = (
        (starts[0] != 0) | (starts[-1] != n) | (starts[1:] < starts[:-1]).any()
    )
    # Validate offsets before any kernel uses them as addresses.
    if bool(invalid_offsets):
        raise ValueError("CSF exception offsets must be monotonic and cover the stream")
    error = torch.zeros((), dtype=torch.int32, device=fixed.device)
    output = (
        torch.empty(
            (experts, r // 128, 128 * (1 + c // 2)),
            dtype=torch.uint8,
            device=fixed.device,
        )
        if codec == 0
        else fixed
    )
    keys = torch.empty(n, dtype=torch.int64, device=fixed.device)
    words = torch.empty_like(keys)
    with torch.cuda.device(fixed.device):
        if codec == 0:
            _fixed[(triton.cdiv(r * (1 + c // 2), 1024), experts)](
                fixed,
                output,
                error,
                rows,
                columns,
                r0,
                c0,
                r1 - r0,
                c1 - c0,
                join,
                r,
                c,
                layout,
                row_rotation,
                1024,
            )
        if n:
            _exceptions[(triton.cdiv(n, 256),)](
                exceptions,
                starts,
                words,
                keys,
                error,
                n,
                segments,
                rows,
                columns,
                r0,
                r1,
                c0,
                c1,
                join,
                r,
                c,
                row_rotation,
                codec,
                256,
            )
        failed = error.item()
        if failed & 1:
            raise ValueError("NVFP4-CSF row bases must not exceed 240")
        if failed & 2:
            raise ValueError("NVFP4-CSF exceptions must be unique, sorted and in range")
        if (r0, r1, c0, c1) != (0, rows, 0, columns):
            keep = keys >= 0
            keys, words = keys[keep], words[keep]
        if row_rotation:
            order = torch.argsort(keys)
            keys, words = keys[order], words[order]
        bounds = torch.arange(experts, device=fixed.device)[:, None] * (
            r * c
        ) + torch.arange(r // 128 + 1, device=fixed.device)[None, :] * (128 * c)
        offsets = torch.searchsorted(keys, bounds)
        if codec == 2:
            encoded = (
                torch.stack(
                    (words & 255, (words >> 8) & 255, (words >> 16) & 255), dim=1
                )
                .to(torch.uint8)
                .flatten()
            )
        else:
            encoded = words.to(torch.uint32).view(torch.uint8)
    result = Nvfp4CsfBatch(
        output,
        encoded,
        offsets,
        r,
        c,
        codec,
        layout,
        value_lut,
        logical_rows if r != logical_rows or c != logical_columns else None,
        logical_columns if r != logical_rows or c != logical_columns else None,
    )
    result.validate()
    return result


def upload_and_convert(
    fixed_planes,
    exception_planes,
    *,
    rows,
    columns,
    device,
    codec=0,
    layout=0,
    row_rotation=0,
    value_lut=None,
):
    if len(fixed_planes) == 0 or len(fixed_planes) != len(exception_planes):
        raise ValueError("NVFP4-CSF component lists must be nonempty and equally sized")
    width = 3 if codec == 2 else 4
    fixed = torch.stack(
        [
            torch.as_tensor(f, dtype=torch.uint8).reshape(
                rows // 16, 16 * (1 + columns // 2)
            )
            for f in fixed_planes
        ]
    ).to(device)
    streams = [
        torch.as_tensor(e).contiguous().view(torch.uint8).flatten()
        for e in exception_planes
    ]
    starts = [0]
    for stream in streams:
        if stream.numel() % width:
            raise ValueError("Invalid NVFP4-CSF exception record length")
        starts.append(starts[-1] + stream.numel() // width)
    exceptions = torch.cat(streams).to(device)
    return convert_nvfp4_csf_storage(
        fixed,
        exceptions,
        torch.tensor(starts, device=device, dtype=torch.int64),
        rows=rows,
        columns=columns,
        codec=codec,
        layout=layout,
        row_rotation=row_rotation,
        value_lut=value_lut,
    )
