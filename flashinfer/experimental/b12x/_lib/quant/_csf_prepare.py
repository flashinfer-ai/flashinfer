"""GPU gathering and native packing of compressed checkpoint scales."""

import torch
import triton
import triton.language as tl

from .csf_blocks import BlockStorageLayout, CsfBlockPayload, _align16


@triton.jit
def _validate_source(
    P0,
    P1,
    P2,
    P3,
    P4,
    Starts,
    Error,
    SR: tl.constexpr,
    SC: tl.constexpr,
    BITS: tl.constexpr,
    LEGACY: tl.constexpr,
    N: tl.constexpr,
    SEGMENTS: tl.constexpr,
    B: tl.constexpr,
):
    segment = tl.program_id(1)
    i = tl.program_id(0) * B + tl.arange(0, B)
    start, end = tl.load(Starts + segment), tl.load(Starts + segment + 1)
    bad = (start < 0) | (end < start) | (end > N)
    bad |= ((segment == 0) & (start != 0)) | ((segment == SEGMENTS - 1) & (end != N))
    if LEGACY:
        stride: tl.constexpr = 16 * (1 + triton.cdiv(SC * BITS, 8))
        base = tl.load(
            P0 + segment * SR // 16 * stride + i // 16 * stride + i % 16, i < SR, 0
        )
    else:
        base = tl.load(P0 + segment * SR + i, i < SR, 0)
        cb: tl.constexpr = triton.cdiv(SC, 2)
        cg: tl.constexpr = triton.cdiv(SC, 4)
        blocks: tl.constexpr = SR // 32 * cb
        pages: tl.constexpr = triton.cdiv(triton.cdiv(SR, 128) * cg, 32)
        rg = i // (4 * cb)
        nrb = tl.minimum(4, SR // 32 - rg * 4)
        group = rg * cg + (i % (4 * cb)) // (nrb * 2)
        index = segment * (blocks + pages) + i + group // 32
        page_origin = tl.load(P2 + segment * pages + group // 32, i < blocks, 0).to(
            tl.int64
        )
        first = tl.load(P3 + index, i < blocks, 0).to(tl.int64)
        last = tl.load(P3 + index + 1, i < blocks, 0).to(tl.int64)
        col = i % (4 * cb) // nrb * 2
        count = last - first
        invalid = (
            (page_origin % 2 != 0)
            | (count < 0)
            | (count > 32 * tl.minimum(2, SC - col))
        )
        invalid |= page_origin // 2 + last > end - start
        invalid |= (i == 0) & ((page_origin != 0) | (first != 0))
        j = i + 1
        nrg = j // (4 * cb)
        nnrb = tl.minimum(4, SR // 32 - nrg * 4)
        ngroup = nrg * cg + j % (4 * cb) // tl.maximum(nnrb * 2, 1)
        npage = ngroup // 32
        nbase = tl.load(P2 + segment * pages + npage, j < blocks, 0).to(tl.int64)
        nfirst = tl.load(P3 + segment * (blocks + pages) + j + npage, j < blocks, 0).to(
            tl.int64
        )
        invalid |= (j < blocks) & (page_origin // 2 + last != nbase // 2 + nfirst)
        invalid |= (j < blocks) & (npage != group // 32) & (nfirst != 0)
        invalid |= (j == blocks) & (page_origin // 2 + last != end - start)
        bad |= tl.sum(((i < blocks) & invalid).to(tl.int32), 0) > 0
    bad |= tl.sum(((i < SR) & (base > 256 - (1 << BITS))).to(tl.int32), 0) > 0
    if bad:
        tl.atomic_or(Error, 1)


@triton.jit
def _validate_records(
    Pages,
    Offsets,
    Records,
    Starts,
    Error,
    SR: tl.constexpr,
    SC: tl.constexpr,
    LEGACY: tl.constexpr,
    B: tl.constexpr,
):
    segment, part = tl.program_id(1), tl.program_id(0)
    i = part * B + tl.arange(0, B)
    start, end = tl.load(Starts + segment), tl.load(Starts + segment + 1)
    if LEGACY:
        record = tl.load(Records + start + i, start + i < end, 0).to(tl.uint32)
        previous = tl.load(Records + start + i - 1, (i > 0) & (start + i < end), 0).to(
            tl.uint32
        )
        position = record & 0xFFFFFF
        invalid = (position >= SR * SC) | (
            (i > 0) & (position <= (previous & 0xFFFFFF))
        )
        invalid &= start + i < end
    else:
        cb: tl.constexpr = triton.cdiv(SC, 2)
        block, slot = i // 64, i % 64
        rg = block // (4 * cb)
        nrb = tl.minimum(4, SR // 32 - rg * 4)
        row = rg * 128 + block % (4 * cb) % tl.maximum(nrb, 1) * 32
        col = block % (4 * cb) // tl.maximum(nrb, 1) * 2
        valid = block < SR // 32 * cb
        lo, hi = _bounds(Pages, Offsets, Starts, segment, row, col, valid, SR, SC)
        valid &= lo + slot < hi
        record = tl.load(Records + lo + slot, valid, 0).to(tl.uint32)
        previous = tl.load(Records + lo + slot - 1, valid & (slot > 0), 0).to(tl.uint32)
        position = record & 255
        invalid = (position >= 64) | (col + position % 2 >= SC)
        invalid |= (slot > 0) & (position <= (previous & 255))
        invalid &= valid
    if tl.sum(invalid.to(tl.int32), 0):
        tl.atomic_or(Error, 2)


def validate_source(source):
    """Check GPU payload bounds before using checkpoint indexes as addresses."""
    args, geometry = _arguments(source)
    p0, p1, p2, p3, records, starts = args
    rows, columns = source.source_rows, source.source_columns
    legacy = source.encoding == "csf"
    error = torch.zeros((), dtype=torch.int32, device=source.device)
    blocks = rows // 32 * triton.cdiv(columns, 2)
    with torch.cuda.device(source.device):
        _validate_source[(triton.cdiv(max(rows, blocks), 256), source.segments)](
            *args,
            error,
            SR=rows,
            SC=columns,
            BITS=geometry["BITS"],
            LEGACY=legacy,
            N=records.numel(),
            SEGMENTS=source.segments,
            B=256,
        )
        if error.item():
            raise ValueError("Invalid CSF row bases or exception index bounds")
        if legacy:
            extent = int((starts[1:] - starts[:-1]).max().item())
        else:
            extent = blocks * 64
        if extent:
            _validate_records[(triton.cdiv(extent, 1024), source.segments)](
                p2,
                p3,
                records,
                starts,
                error,
                SR=rows,
                SC=columns,
                LEGACY=legacy,
                B=1024,
            )
        if error.item():
            raise ValueError(
                "CSF exception positions must be sorted, unique and in range"
            )


@triton.jit
def _location(
    expert,
    row,
    column,
    SR: tl.constexpr,
    SC: tl.constexpr,
    R0: tl.constexpr,
    R1: tl.constexpr,
    C0: tl.constexpr,
    C1: tl.constexpr,
    LR: tl.constexpr,
    JOIN: tl.constexpr,
    ROT: tl.constexpr,
):
    row = (row + ROT) % (LR * JOIN)
    segment = expert * JOIN + row // LR
    valid = (row % LR < R1 - R0) & (column < C1 - C0)
    return segment, row % LR + R0, column + C0, valid


@triton.jit
def _bounds(
    Pages,
    Offsets,
    Starts,
    segment,
    row,
    column,
    valid,
    SR: tl.constexpr,
    SC: tl.constexpr,
):
    cg: tl.constexpr = triton.cdiv(SC, 4)
    cb: tl.constexpr = triton.cdiv(SC, 2)
    pages: tl.constexpr = triton.cdiv(triton.cdiv(SR, 128) * cg, 32)
    nr = tl.minimum(128, SR - row // 128 * 128)
    group = row // 128 * cg + column // 4
    block = row // 128 * 4 * cb + column // 4 * 2 * (nr // 32)
    block += column % 4 // 2 * (nr // 32) + row % 128 // 32
    prefix = segment * (SR // 32 * cb + pages) + block + group // 32
    origin = tl.load(Starts + segment)
    origin += tl.load(Pages + segment * pages + group // 32, valid, 0).to(tl.int64) // 2
    lo = origin + tl.load(Offsets + prefix, valid, 0).to(tl.int64)
    hi = origin + tl.load(Offsets + prefix + 1, valid, 0).to(tl.int64)
    return lo, hi


@triton.jit
def _read(
    P0,
    P1,
    P2,
    P3,
    P4,
    Starts,
    expert,
    row,
    column,
    valid,
    SR: tl.constexpr,
    SC: tl.constexpr,
    BITS: tl.constexpr,
    LEGACY: tl.constexpr,
    R0: tl.constexpr,
    R1: tl.constexpr,
    C0: tl.constexpr,
    C1: tl.constexpr,
    LR: tl.constexpr,
    JOIN: tl.constexpr,
    ROT: tl.constexpr,
    PAD: tl.constexpr,
    EXCEPTIONS: tl.constexpr,
):
    segment, sr, sc, inside = _location(
        expert, row, column, SR, SC, R0, R1, C0, C1, LR, JOIN, ROT
    )
    valid &= inside
    if LEGACY:
        stride: tl.constexpr = 16 * (1 + triton.cdiv(SC * BITS, 8))
        slab = segment * (SR // 16 * stride) + sr // 16 * stride
        base = tl.load(P0 + slab + sr % 16, valid, PAD).to(tl.int32)
        field = sr % 16 * triton.cdiv(SC * BITS, 8) * 8 + sc * BITS
        packed = tl.load(P0 + slab + 16 + field // 8, valid, 0).to(tl.int32)
    else:
        base = tl.load(P0 + segment * SR + sr, valid, PAD).to(tl.int32)
        nr = tl.minimum(128, SR - sr // 128 * 128)
        width = tl.minimum(2, SC - sc // 2 * 2)
        field = sr // 128 * 128 * SC + sc // 4 * 4 * nr
        field += sc % 4 // 2 * 2 * nr + sr % 128 * width + sc % 2
        field *= BITS
        packed = tl.load(
            P1 + segment * (SR * SC * BITS // 8) + field // 8, valid, 0
        ).to(tl.int32)
    code = (packed >> (field % 8)) & ((1 << BITS) - 1)
    value = base + code
    found = tl.full(value.shape, False, tl.int1)
    if EXCEPTIONS:
        if LEGACY:
            lo = tl.load(Starts + segment, valid, 0)
            end = tl.load(Starts + segment + 1, valid, 0)
            target = sr * SC + sc
            mask: tl.constexpr = 0xFFFFFF
            shift: tl.constexpr = 24
        else:
            lo, end = _bounds(P2, P3, Starts, segment, sr, sc, valid, SR, SC)
            target = sr % 32 * 2 + sc % 2
            mask: tl.constexpr = 255
            shift: tl.constexpr = 8
        hi = end
        while tl.sum((lo < hi).to(tl.int32), 0) > 0:
            mid = (lo + hi) // 2
            word = tl.load(P4 + mid, valid & (lo < hi), 0).to(tl.uint32)
            right = (word & mask) < target
            lo = tl.where((lo < hi) & right, mid + 1, lo)
            hi = tl.where((lo < hi) & ~right, mid, hi)
        word = tl.load(P4 + lo, valid & (lo < end), 0).to(tl.uint32)
        found = valid & (lo < end) & ((word & mask) == target)
        value = tl.where(found, word >> shift, value)
    return base, code, value, found


@triton.jit
def _packed_row(row, C: tl.constexpr):
    swapped = row // 4 * 4 + row % 2 * 2 + row % 4 // 2
    if C == 1:
        return (
            swapped // 32 * 32
            + swapped % 8 // 2 * 8
            + swapped % 2
            + swapped % 32 // 8 * 2
        )
    return swapped // 64 * 64 + swapped % 8 * 8 + swapped % 64 // 8


@triton.jit
def _inverse_packed_row(row, C: tl.constexpr):
    if C == 1:
        transposed = row // 32 * 32 + row % 8 // 2 * 8 + row % 2 + row % 32 // 8 * 2
    else:
        transposed = row // 64 * 64 + row % 8 * 8 + row % 64 // 8
    return transposed // 4 * 4 + transposed % 2 * 2 + transposed % 4 // 2


@triton.jit
def _pack_codes(
    P0,
    P1,
    P2,
    P3,
    P4,
    Starts,
    Table,
    Dst,
    SR: tl.constexpr,
    SC: tl.constexpr,
    BITS: tl.constexpr,
    LEGACY: tl.constexpr,
    R0: tl.constexpr,
    R1: tl.constexpr,
    C0: tl.constexpr,
    C1: tl.constexpr,
    LR: tl.constexpr,
    JOIN: tl.constexpr,
    ROT: tl.constexpr,
    PAD: tl.constexpr,
    R: tl.constexpr,
    C: tl.constexpr,
    SECTIONS: tl.constexpr,
    DEST: tl.constexpr,
    LUT: tl.constexpr,
    BASES: tl.constexpr,
    CODES: tl.constexpr,
):
    group, expert = tl.program_id(0), tl.program_id(1)
    cg: tl.constexpr = triton.cdiv(C, 4)
    rr, cc = group // cg * 128, group % cg * 4
    nr, nc = tl.minimum(128, R - rr), tl.minimum(4, C - cc)
    i = tl.arange(0, 512)
    if DEST == 0:
        row, col = i // nc % (nr // 32) * 32 + i // (nc * (nr // 32)), i % nc
    elif DEST == 1:
        row, col = _packed_row(i % nr, C), i // nr
    else:
        row, col = i // nc, i % nc
    valid = i < nr * nc
    logical_row = expert % SECTIONS * R + rr + row
    base, code, value, _ = _read(
        P0,
        P1,
        P2,
        P3,
        P4,
        Starts,
        expert // SECTIONS,
        logical_row,
        cc + col,
        valid,
        SR,
        SC,
        BITS,
        LEGACY,
        R0,
        R1,
        C0,
        C1,
        LR,
        JOIN,
        ROT,
        PAD,
        LUT,
    )
    if LUT:
        base = tl.minimum(tl.load(Table + base), 256 - (1 << BITS)).to(tl.int32)
        value = tl.load(Table + value).to(tl.int32)
        code = value - base
        code = tl.where((code >= 0) & (code < (1 << BITS)), code, 0)
    width: tl.constexpr = 8 // BITS
    packed = tl.sum(
        tl.reshape(code, (512 // width, width))
        << (tl.arange(0, width)[None, :] * BITS),
        1,
    )
    byte = tl.arange(0, 512 // width)
    offset = expert * (R * C * BITS // 8) + (rr * C + cc * nr) * BITS // 8
    tl.store(Dst + CODES + offset + byte, packed, byte < nr * nc // width)
    # Each row's base is independent of the column group.
    row_index = tl.arange(0, 128)
    if DEST == 0:
        base_row = row_index % (nr // 32) * 32 + row_index // (nr // 32)
    elif DEST == 1:
        base_row = _packed_row(row_index, C)
    else:
        base_row = row_index
    base, _, _, _ = _read(
        P0,
        P1,
        P2,
        P3,
        P4,
        Starts,
        expert // SECTIONS,
        expert % SECTIONS * R + rr + base_row,
        tl.full((128,), 0, tl.int32),
        row_index < nr,
        SR,
        SC,
        BITS,
        LEGACY,
        R0,
        R1,
        C0,
        C1,
        LR,
        JOIN,
        ROT,
        PAD,
        False,
    )
    if LUT:
        base = tl.minimum(tl.load(Table + base), 256 - (1 << BITS))
    tl.store(
        Dst + BASES + expert * R + rr + row_index,
        base,
        (group % cg == 0) & (row_index < nr),
    )


@triton.jit
def _pack_records(
    P0,
    P1,
    P2,
    P3,
    P4,
    Starts,
    Table,
    Counts,
    Prefix,
    Dst,
    SR: tl.constexpr,
    SC: tl.constexpr,
    BITS: tl.constexpr,
    LEGACY: tl.constexpr,
    R0: tl.constexpr,
    R1: tl.constexpr,
    C0: tl.constexpr,
    C1: tl.constexpr,
    LR: tl.constexpr,
    JOIN: tl.constexpr,
    ROT: tl.constexpr,
    PAD: tl.constexpr,
    R: tl.constexpr,
    C: tl.constexpr,
    SECTIONS: tl.constexpr,
    DEST: tl.constexpr,
    LUT: tl.constexpr,
    WRITE: tl.constexpr,
    RECORDS: tl.constexpr,
    VALUES: tl.constexpr,
    PAGES: tl.constexpr,
    OFFSETS: tl.constexpr,
):
    group, expert = tl.program_id(0), tl.program_id(1)
    cg: tl.constexpr = triton.cdiv(C, 4)
    groups: tl.constexpr = triton.cdiv(R, 128) * cg
    pages: tl.constexpr = triton.cdiv(groups, 32)
    rr, cc = group // cg * 128, group % cg * 4
    nr, nc = tl.minimum(128, R - rr), tl.minimum(4, C - cc)
    if LUT or LEGACY:
        i = tl.arange(0, 512)
        row, col = i // 4, i % 4
        valid = (row < nr) & (col < nc)
        base, _, value, found = _read(
            P0,
            P1,
            P2,
            P3,
            P4,
            Starts,
            expert // SECTIONS,
            expert % SECTIONS * R + rr + row,
            cc + col,
            valid,
            SR,
            SC,
            BITS,
            LEGACY,
            R0,
            R1,
            C0,
            C1,
            LR,
            JOIN,
            ROT,
            PAD,
            True,
        )
        if LUT:
            base = tl.minimum(tl.load(Table + base), 256 - (1 << BITS)).to(tl.int32)
            value = tl.load(Table + value).to(tl.int32)
            found = (value < base) | (value >= base + (1 << BITS))
        keep = valid & found
    else:
        i = tl.arange(0, 1024)
        rb, pair, slot = i // 64 % 4, i // 256, i % 64
        logical_row = expert % SECTIONS * R + rr + rb * 32
        segment, sr, sc, inside = _location(
            expert // SECTIONS,
            logical_row,
            tl.full((1024,), 0, tl.int32),
            SR,
            SC,
            R0,
            R1,
            C0,
            C1,
            LR,
            JOIN,
            ROT,
        )
        sc = (C0 + cc) // 2 * 2 + pair * 2
        valid = (
            (rb * 32 < nr)
            & inside
            & (pair < triton.cdiv(C0 % 2 + 4, 2))
            & (sc < C1)
            & (sc < C0 + cc + nc)
        )
        lo, hi = _bounds(P2, P3, Starts, segment, sr, sc, valid, SR, SC)
        word = tl.load(P4 + lo + slot, valid & (lo + slot < hi), 0).to(tl.uint32)
        local = word & 255
        row, col = rb * 32 + local // 2, sc + local % 2 - C0 - cc
        value = word >> 8
        keep = valid & (lo + slot < hi) & (col >= 0) & (col < nc) & (cc + col < C1 - C0)
    index = expert * groups + group
    if not WRITE:
        tl.store(Counts + index, tl.sum(keep.to(tl.int32), 0))
    else:
        first = tl.load(Prefix + index)
        record = first + tl.cumsum(keep.to(tl.int32), 0) - 1
        if DEST == 1:
            position = col * nr + _inverse_packed_row(row, C)
        elif DEST == 3:
            position = row * nc + col
        else:
            position = row * 4 + col
        tl.store(
            (Dst + RECORDS).to(tl.pointer_type(tl.uint16)) + record, position, keep
        )
        tl.store(Dst + VALUES + record, value, keep)
        page_first = tl.load(Prefix + expert * groups + group // 32 * 32)
        prefix_index = expert * (groups + pages) + group + group // 32
        tl.store(
            (Dst + OFFSETS).to(tl.pointer_type(tl.uint16)) + prefix_index,
            first - page_first,
        )
        if group % 32 == 0:
            tl.store(
                (Dst + PAGES).to(tl.pointer_type(tl.uint32))
                + expert * pages
                + group // 32,
                page_first,
            )
        if group % 32 == 31 or group == groups - 1:
            end = tl.load(Prefix + index + 1)
            tl.store(
                (Dst + OFFSETS).to(tl.pointer_type(tl.uint16)) + prefix_index + 1,
                end - page_first,
            )


def _arguments(source):
    source.validate()
    if source.device.type != "cuda":
        raise ValueError("CSF payload preparation requires CUDA checkpoint components")
    parts = source.components
    if source.encoding == "csf":
        parts = (parts[0], parts[0], parts[0], parts[0], parts[1])
    r0, r1 = source.row_range or (0, source.source_rows)
    c0, c1 = source.column_range or (0, source.source_columns)
    return (*parts, source.starts), dict(
        SR=source.source_rows,
        SC=source.source_columns,
        BITS=4 if source.quantization == "nvfp4" else 1,
        LEGACY=source.encoding == "csf",
        R0=r0,
        R1=r1,
        C0=c0,
        C1=c1,
        LR=source.rows // source.join,
        JOIN=source.join,
        PAD=source.padding_base,
    )


def prepare_csf_payload(
    source, *, destination=0, rotation=0, group_rows=None, value_table=None
):
    """Gather native codes and compact exceptions without expanding a scale matrix."""
    args, geometry = _arguments(source)
    rows = source.rows if group_rows is None else group_rows
    if destination not in (0, 1, 2, 3) or rows <= 0 or rows % 32 or source.rows % rows:
        raise ValueError("CSF destination and row sections must match native geometry")
    if destination == 1 and rows % 64:
        raise ValueError("Packed MMA scales require whole 64-row groups")
    if rotation < 0 or rotation >= source.rows or rotation % 32:
        raise ValueError("CSF row rotation requires a complete in-range block")
    sections = source.rows // rows
    layout = BlockStorageLayout(
        rows, source.columns, source.num_experts * sections, source.quantization, True
    )
    table = source.components[0] if value_table is None else value_table
    if value_table is not None and (
        table.dtype != torch.uint8
        or table.shape != (256,)
        or table.device != source.device
    ):
        raise ValueError(
            "CSF scale conversion requires a CUDA uint8 table with 256 entries"
        )
    counts = torch.empty(
        layout.experts * layout.groups, dtype=torch.int32, device=source.device
    )
    prefix = torch.empty(counts.numel() + 1, dtype=torch.int64, device=source.device)
    kwargs = dict(
        **geometry,
        ROT=rotation,
        R=rows,
        C=source.columns,
        SECTIONS=sections,
        DEST=destination,
        LUT=value_table is not None,
    )
    grid = (layout.groups, layout.experts)
    with torch.cuda.device(source.device):
        _pack_records[grid](
            *args,
            table,
            counts,
            prefix,
            counts,
            **kwargs,
            WRITE=False,
            RECORDS=0,
            VALUES=0,
            PAGES=0,
            OFFSETS=0,
        )
        prefix[0].zero_()
        torch.cumsum(counts, 0, out=prefix[1:])
        total = int(prefix[-1].item())
        if total * 3 >= 1 << 32:
            raise ValueError("CSF exception arena must be smaller than 4 GiB")
        storage = torch.zeros(
            _align16(layout.source_records + 3 * total),
            dtype=torch.uint8,
            device=source.device,
        )
        values = layout.source_records + 2 * total
        storage[:8].view(torch.int64).fill_(values)
        _pack_codes[grid](
            *args, table, storage, **kwargs, BASES=layout.bases, CODES=layout.codes
        )
        _pack_records[grid](
            *args,
            table,
            counts,
            prefix,
            storage,
            **kwargs,
            WRITE=True,
            RECORDS=layout.source_records,
            VALUES=values,
            PAGES=layout.page_bases,
            OFFSETS=layout.offsets,
        )
    return CsfBlockPayload(
        storage,
        rows,
        source.columns,
        layout.experts,
        source.quantization,
        True,
        3,
        destination,
        sections,
    )


@triton.jit
def _expand_source(
    P0,
    P1,
    P2,
    P3,
    P4,
    Starts,
    Output,
    SR: tl.constexpr,
    SC: tl.constexpr,
    BITS: tl.constexpr,
    LEGACY: tl.constexpr,
    R0: tl.constexpr,
    R1: tl.constexpr,
    C0: tl.constexpr,
    C1: tl.constexpr,
    LR: tl.constexpr,
    JOIN: tl.constexpr,
    PAD: tl.constexpr,
    R: tl.constexpr,
    C: tl.constexpr,
    PR: tl.constexpr,
    PC: tl.constexpr,
    SWIZZLED: tl.constexpr,
):
    i, expert = tl.program_id(0) * 256 + tl.arange(0, 256), tl.program_id(1)
    row, col = i // PC, i % PC
    valid = (row < R) & (col < C)
    _, _, value, _ = _read(
        P0,
        P1,
        P2,
        P3,
        P4,
        Starts,
        expert,
        row,
        col,
        valid,
        SR,
        SC,
        BITS,
        LEGACY,
        R0,
        R1,
        C0,
        C1,
        LR,
        JOIN,
        0,
        PAD,
        True,
    )
    if SWIZZLED:
        offset = (
            row // 128 * (128 * PC)
            + col // 4 * 512
            + row % 32 * 16
            + row % 128 // 32 * 4
            + col % 4
        )
    else:
        offset = i
    tl.store(Output + expert * PR * PC + offset, tl.where(valid, value, 0), i < PR * PC)


def expand_source_scales(source, output, *, swizzled):
    """Supply native weight preparation with source scales using caller-owned GPU scratch."""
    args, kwargs = _arguments(source)
    pr, pc = output.shape[1:]
    with torch.cuda.device(source.device):
        _expand_source[(triton.cdiv(pr * pc, 256), source.num_experts)](
            *args,
            output.view(torch.uint8),
            **kwargs,
            R=source.rows,
            C=source.columns,
            PR=pr,
            PC=pc,
            SWIZZLED=swizzled,
        )
