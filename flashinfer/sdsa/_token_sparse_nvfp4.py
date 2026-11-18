# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 by FlashInfer team.
"""Convert selected NVFP4 MQA K/V tokens into FP8 attention workspace."""

import triton
import triton.language as tl


# Triton requires explicit constexpr objects for globals; host code reads .value.
# Storage format: two E2M1 values per byte and one E4M3 scale per block.
FP4_VALUES_PER_BYTE = tl.constexpr(2)
E2M1_MAGNITUDES = tl.constexpr((0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0))
NVFP4_SCALE_BLOCK_SIZE = tl.constexpr(16)

# TRT-LLM HND V scales transpose the token and scale-group axes within a tile.
V_SCALE_SWIZZLE_TOKENS = tl.constexpr(4)
V_SCALE_SWIZZLE_GROUPS = tl.constexpr(4)
HEAD_DIM_ALIGNMENT = tl.constexpr(
    NVFP4_SCALE_BLOCK_SIZE.value * V_SCALE_SWIZZLE_GROUPS.value
)
KV_COMPONENTS = tl.constexpr(2)

# E2M1's largest magnitude expands the E4M3 block-scale range by this factor.
# Divide before casting to FP8; callers restore it in the attention BMM scales.
FP8_NORMALIZATION_FACTOR = tl.constexpr(E2M1_MAGNITUDES.value[-1])

# Compact conversion: each group of rows (one request's query tokens) gets its
# selected tokens converted once, into its own region of the FP8 workspace, and
# its rows' K/V index lists point into that region. Selected keys are numbered
# with a bitmap of key positions (a bit per key, BITMAP_WORD_BITS keys per int32
# word); word WORDS of a group's bitmap row holds its key end (last live key
# position + 1). Dense groups skip the bitmap: their region holds every key below
# the key end, at its position.
BITMAP_WORD_BITS = tl.constexpr(32)
BITMAP_WORD_SHIFT = tl.constexpr(5)
# Bitmap words a numbering program scans at a time.
NUMBER_WORDS_PER_STEP = tl.constexpr(1024)
# Default tokens per conversion program and its warps (launch tuning only).
COMPACT_TOKENS_PER_TILE = 32
COMPACT_NUM_WARPS = 4


@triton.jit
def _popcount(words):
    """Set bits of each int32 word (SWAR; runs in the Triton interpreter too)."""
    x = words.to(tl.uint32, bitcast=True)
    x = x - ((x >> 1) & 0x55555555)
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F
    return ((x * 0x01010101) >> 24).to(tl.int32)


@triton.jit
def _e2m1x2_value(packed):
    """Both E2M1 codes of each byte (int32 holding 0..255; the low nibble first)
    as float32, decoded by Blackwell's ``cvt.rn.f16x2.e2m1x2``. Every E2M1 value
    is exact in float16, so these are the codes' exact values (-0.0 for code 8
    included)."""
    return tl.inline_asm_elementwise(
        asm="""{
        .reg .b8 code;
        .reg .b32 pair;
        .reg .b16 h0, h1;
        cvt.u8.u32 code, $2;
        cvt.rn.f16x2.e2m1x2 pair, code;
        mov.b32 {h0, h1}, pair;
        cvt.f32.f16 $0, h0;
        cvt.f32.f16 $1, h1;
        }""",
        constraints="=f,=f,r",
        args=[packed],
        dtype=(tl.float32, tl.float32),
        is_pure=True,
        pack=1,
    )


@triton.jit
def _e2m1x2_to_fp8(
    packed, sf, TOKENS: tl.constexpr, DIM: tl.constexpr, out_dtype: tl.constexpr
):
    """Packed E2M1 pairs ([TOKENS, DIM // 2] int32; byte b holds dims 2b and
    2b + 1) times their E4M3 block scales (``sf``, [TOKENS, DIM] float32),
    divided by six and rounded to ``out_dtype``. Every conversion path uses this
    helper, so they write the same FP8 bits for the same cache token."""
    low, high = _e2m1x2_value(packed)
    value = tl.reshape(tl.join(low, high), (TOKENS, DIM))
    return (value * sf / FP8_NORMALIZATION_FACTOR).to(out_dtype)


@triton.jit
def _row_keys(
    Positions, POS_ROW, Lengths, row, TOP_K: tl.constexpr, HAS_LENGTHS: tl.constexpr
):
    """A row's key positions and which are live: nonnegative, within the row's
    valid prefix. The prefix is Lengths[row] if given
    (entries past it are not read), else the row's count of nonnegative keys."""
    cols = tl.arange(0, TOP_K)
    pos = tl.load(Positions + row * POS_ROW + cols)
    valid = pos >= 0
    if HAS_LENGTHS:
        count = tl.load(Lengths + row)
    else:
        count = tl.sum(valid.to(tl.int32), 0)
    return cols, pos, valid & (cols < count), count


# Launches differ in GROUP_BASE and GROUPS; one compiled kernel serves them all.
@triton.jit(do_not_specialize=["GROUP_BASE", "GROUPS", "MAX_KEYS"])
def compact_mark_kernel(
    Positions,
    POS_ROW,
    Lengths,
    RowGroups,
    GROUP_BASE,
    GROUPS,
    GroupDense,
    Bitmap,
    WORDS,
    MAX_KEYS,
    TOP_K: tl.constexpr,
    HAS_DENSE: tl.constexpr,
    HAS_LENGTHS: tl.constexpr,
):
    """Set a bit per live key of the row in its group's bitmap row, and raise
    the group's key end. Keys at or past MAX_KEYS are not live; rows of groups
    outside [0, GROUPS) after subtracting GROUP_BASE are skipped."""
    row = tl.program_id(0).to(tl.int64)
    group = (tl.load(RowGroups + row) - GROUP_BASE).to(tl.int64)
    if (group >= 0) & (group < GROUPS):
        cols, pos, live, count = _row_keys(
            Positions, POS_ROW, Lengths, row, TOP_K, HAS_LENGTHS
        )
        live = live & (pos < MAX_KEYS)
        bitmap = Bitmap + group * (WORDS + 1)
        dense = False
        if HAS_DENSE:
            dense = tl.load(GroupDense + group) != 0
        if not dense:
            bit = tl.full((TOP_K,), 1, tl.uint32) << (pos & (BITMAP_WORD_BITS - 1)).to(
                tl.uint32
            )
            tl.atomic_or(
                bitmap + (tl.maximum(pos, 0) >> BITMAP_WORD_SHIFT),
                bit.to(tl.int32, bitcast=True),
                mask=live,
                sem="relaxed",
            )
        end = tl.max(tl.where(live, pos + 1, 0), 0)
        tl.atomic_max(bitmap + WORDS, end, sem="relaxed")


@triton.jit
def compact_number_kernel(
    Bitmap,
    Prefix,
    Sizes,
    Ends,
    GroupDense,
    WORDS,
    HAS_DENSE: tl.constexpr,
):
    """A group's region size, and for a union group each bitmap word's count of
    set bits in the words below it (the region column of its first key). Moves
    the key end from the bitmap row to Ends (the bitmap row's last word is left
    zero; compact_convert_kernel clears the rest)."""
    group = tl.program_id(0).to(tl.int64)
    bitmap = Bitmap + group * (WORDS + 1)
    end = tl.load(bitmap + WORDS)
    tl.store(bitmap + WORDS, end * 0)
    tl.store(Ends + group, end)
    dense = False
    if HAS_DENSE:
        dense = tl.load(GroupDense + group) != 0
    if dense:
        tl.store(Sizes + group, end)
    else:
        words = (end + BITMAP_WORD_BITS - 1) // BITMAP_WORD_BITS
        total = end * 0
        for first in range(0, words, NUMBER_WORDS_PER_STEP):
            word = first + tl.arange(0, NUMBER_WORDS_PER_STEP)
            inside = word < words
            counts = _popcount(tl.load(bitmap + word, mask=inside, other=0))
            below = total + tl.cumsum(counts, 0) - counts
            tl.store(Prefix + group * WORDS + word, below, mask=inside)
            total += tl.sum(counts, 0)
        tl.store(Sizes + group, total)


# Launches differ in GROUP_BASE and GROUPS; one compiled kernel serves them all.
@triton.jit(do_not_specialize=["GROUP_BASE", "GROUPS", "MAX_KEYS", "ZERO_ROW"])
def compact_remap_kernel(
    Positions,
    POS_ROW,
    Lengths,
    RowGroups,
    GROUP_BASE,
    GROUPS,
    GroupDense,
    GroupOffsets,
    Bitmap,
    Prefix,
    WORDS,
    MAX_KEYS,
    Tables,
    Counts,
    UnionKeys,
    ZERO_ROW,
    TOP_K: tl.constexpr,
    REPEAT: tl.constexpr,
    HAS_DENSE: tl.constexpr,
    HAS_LENGTHS: tl.constexpr,
):
    """Point each live key of the row at its row of the group's region, and
    other entries at the zero row; write the row's count. Union groups also
    record each region row's key position (duplicates write the same value).
    Each row's lists and count are written REPEAT times (one per attention
    request of the row), in consecutive rows of Tables and Counts. Padding
    rows (a negative RowGroups value) point every entry at the zero row; rows
    of other launches' groups (outside [0, GROUPS) after subtracting
    GROUP_BASE) are skipped: their lists and counts keep what that launch
    wrote."""
    row = tl.program_id(0).to(tl.int64)
    raw = tl.load(RowGroups + row)
    group = (raw - GROUP_BASE).to(tl.int64)
    padding = raw < 0
    if padding | ((group >= 0) & (group < GROUPS)):
        cols, pos, live, count = _row_keys(
            Positions, POS_ROW, Lengths, row, TOP_K, HAS_LENGTHS
        )
        live = live & (pos < MAX_KEYS) & (raw >= 0)
        # Padding rows read no group metadata (a launch may have no groups).
        own = raw >= 0
        group = tl.maximum(group, 0)
        offset = tl.load(GroupOffsets + group, mask=own, other=0).to(tl.int64)
        dense = False
        if HAS_DENSE:
            dense = tl.load(GroupDense + group, mask=own, other=0) != 0
        if dense:
            column = pos
        else:
            word_index = tl.maximum(pos, 0) >> BITMAP_WORD_SHIFT
            word = tl.load(
                Bitmap + group * (WORDS + 1) + word_index, mask=live, other=0
            )
            below = tl.load(Prefix + group * WORDS + word_index, mask=live, other=0)
            lower = (
                tl.full((TOP_K,), 1, tl.uint32)
                << (pos & (BITMAP_WORD_BITS - 1)).to(tl.uint32)
            ) - 1
            column = below + _popcount(word & lower.to(tl.int32, bitcast=True))
            tl.store(UnionKeys + offset + column, pos, mask=live)
        index = tl.where(live, offset + column, ZERO_ROW).to(tl.int32)
        for request in tl.static_range(REPEAT):
            out_row = row * REPEAT + request
            for side in tl.static_range(KV_COMPONENTS):
                tl.store(
                    Tables + (out_row * KV_COMPONENTS + side) * TOP_K + cols, index
                )
            tl.store(Counts + out_row, count)


@triton.jit
def _convert_tokens(
    K,
    V,
    KS,
    VS,
    PAGE_STRIDE,
    SF_PAGE_STRIDE,
    page,
    token,
    live,
    out_rows,
    DIM: tl.constexpr,
    TOKENS: tl.constexpr,
    V_SF_SWIZZLED: tl.constexpr,
    MASK_STORE: tl.constexpr,
):
    """Convert TOKENS cache tokens (int64 page, token in page) to normalized
    FP8 K then V at out_rows (pointers to [K | V] rows). Tokens that are not
    live read as zero, so their rows are zero unless MASK_STORE skips them.
    Whole tokens are loaded as contiguous bytes, and scales once per block."""
    half: tl.constexpr = DIM // FP4_VALUES_PER_BYTE
    scales_per_token: tl.constexpr = DIM // NVFP4_SCALE_BLOCK_SIZE
    scales_per_group: tl.constexpr = scales_per_token // V_SCALE_SWIZZLE_GROUPS
    byte = tl.arange(0, half)
    sf_dim = tl.arange(0, scales_per_token)
    dim = tl.arange(0, DIM)
    for side in tl.static_range(KV_COMPONENTS):
        data_ptr = K if side == 0 else V
        sf_ptr = KS if side == 0 else VS
        packed = tl.load(
            data_ptr + page[:, None] * PAGE_STRIDE + token[:, None] * half + byte,
            mask=live[:, None],
            other=0,
        ).to(tl.int32)
        if side == 0 or not V_SF_SWIZZLED:
            sf_offset = token[:, None] * scales_per_token + sf_dim[None, :]
        else:
            sf_offset = (
                (
                    token[:, None] // V_SCALE_SWIZZLE_TOKENS * V_SCALE_SWIZZLE_GROUPS
                    + sf_dim[None, :] // scales_per_group
                )
                * scales_per_token
                + sf_dim[None, :] % scales_per_group * V_SCALE_SWIZZLE_TOKENS
                + token[:, None] % V_SCALE_SWIZZLE_TOKENS
            )
        sf = tl.load(
            sf_ptr + page[:, None] * SF_PAGE_STRIDE + sf_offset,
            mask=live[:, None],
            other=0.0,
        ).to(tl.float32)
        # Scale s covers dims 16 s .. 16 s + 15.
        sf = tl.reshape(
            tl.broadcast_to(
                sf[:, :, None], (TOKENS, scales_per_token, NVFP4_SCALE_BLOCK_SIZE)
            ),
            (TOKENS, DIM),
        )
        value = _e2m1x2_to_fp8(packed, sf, TOKENS, DIM, out_rows.dtype.element_ty)
        if MASK_STORE:
            tl.store(out_rows + side * DIM + dim[None, :], value, mask=live[:, None])
        else:
            tl.store(out_rows + side * DIM + dim[None, :], value)


@triton.jit
def _convert_region_tile(
    K,
    V,
    KS,
    VS,
    PAGE_STRIDE,
    SF_PAGE_STRIDE,
    BlockTable,
    TABLE_ROW,
    UnionKeys,
    Out,
    offset,
    dense,
    request,
    tile,
    size,
    PAGE: tl.constexpr,
    DIM: tl.constexpr,
    TOKENS: tl.constexpr,
    V_SF_SWIZZLED: tl.constexpr,
):
    """Region rows [tile * TOKENS, tile * TOKENS + TOKENS) of a group (first
    row ``offset`` of Out, ``size`` region rows, ``dense`` or union, cache row
    ``request`` of BlockTable) to normalized FP8 K then V."""
    column = tile * TOKENS + tl.arange(0, TOKENS)
    live = column < size
    if dense:
        key = column
    else:
        key = tl.load(UnionKeys + offset + column, mask=live, other=0)
    key = tl.where(live, key, 0)
    page = tl.load(
        BlockTable + request * TABLE_ROW + key // PAGE, mask=live, other=0
    ).to(tl.int64)
    out_rows = Out + (offset + column)[:, None] * (KV_COMPONENTS * DIM)
    _convert_tokens(
        K,
        V,
        KS,
        VS,
        PAGE_STRIDE,
        SF_PAGE_STRIDE,
        page,
        key % PAGE,
        live,
        out_rows,
        DIM,
        TOKENS,
        V_SF_SWIZZLED,
        True,
    )


@triton.jit
def compact_convert_kernel(
    K,
    V,
    KS,
    VS,
    PAGE_STRIDE,
    SF_PAGE_STRIDE,
    BlockTable,
    TABLE_ROW,
    GroupRequests,
    GroupOffsets,
    GroupDense,
    Sizes,
    UnionKeys,
    Bitmap,
    WORDS,
    Ends,
    Out,
    ZeroOut,
    PAGE: tl.constexpr,
    DIM: tl.constexpr,
    TOKENS: tl.constexpr,
    MAX_CAPACITY,
    V_SF_SWIZZLED: tl.constexpr,
    HAS_DENSE: tl.constexpr,
    LOOP: tl.constexpr = False,
):
    """Convert a group's region to normalized FP8 K then V, TOKENS region rows
    per tile: program (t, g) converts tile t of group g, or with LOOP tiles t,
    t + grid width, ... of the group's own region size (the grid then need not
    span the largest region for every group). Region row c holds key
    UnionKeys[offset + c] (union group) or key c (dense group) of the group's
    request. A union group's programs also clear the bitmap words it used
    (every later launch finds the bitmap zero; mark sets no bits for dense
    groups), and program (0, 0) zeroes the zero row. Tiles are the grid's first
    axis (no 65535 limit), groups its second; no group converts region rows
    at or past MAX_CAPACITY (max_capacity), also within a partial tile."""
    tile0, group = tl.program_id(0), tl.program_id(1).to(tl.int64)
    union = True
    dense = False
    if HAS_DENSE:
        union = tl.load(GroupDense + group) == 0
        dense = tl.load(GroupDense + group) != 0
    if union:
        words = (tl.load(Ends + group) + BITMAP_WORD_BITS - 1) // BITMAP_WORD_BITS
        clear = tl.arange(0, NUMBER_WORDS_PER_STEP)
        step = tl.num_programs(0) * NUMBER_WORDS_PER_STEP
        for first in range(tile0 * NUMBER_WORDS_PER_STEP, words, step):
            tl.store(
                Bitmap + group * (WORDS + 1) + first + clear,
                clear * 0,
                mask=first + clear < words,
            )
    if (group == 0) & (tile0 == 0):
        zero = tl.zeros((KV_COMPONENTS * DIM,), tl.float32)
        tl.store(
            ZeroOut + tl.arange(0, KV_COMPONENTS * DIM), zero.to(Out.dtype.element_ty)
        )
    size = tl.minimum(tl.load(Sizes + group), MAX_CAPACITY)
    offset = tl.load(GroupOffsets + group).to(tl.int64)
    request = tl.load(GroupRequests + group).to(tl.int64)
    if LOOP:
        tiles = (size + TOKENS - 1) // TOKENS
        for tile in range(tile0, tiles, tl.num_programs(0)):
            _convert_region_tile(
                K,
                V,
                KS,
                VS,
                PAGE_STRIDE,
                SF_PAGE_STRIDE,
                BlockTable,
                TABLE_ROW,
                UnionKeys,
                Out,
                offset,
                dense,
                request,
                tile,
                size,
                PAGE,
                DIM,
                TOKENS,
                V_SF_SWIZZLED,
            )
    elif tile0.to(tl.int64) * TOKENS < size:
        _convert_region_tile(
            K,
            V,
            KS,
            VS,
            PAGE_STRIDE,
            SF_PAGE_STRIDE,
            BlockTable,
            TABLE_ROW,
            UnionKeys,
            Out,
            offset,
            dense,
            request,
            tile0,
            size,
            PAGE,
            DIM,
            TOKENS,
            V_SF_SWIZZLED,
        )


@triton.jit
def position_gather_kernel(
    K,
    V,
    KS,
    VS,
    PAGE_STRIDE,
    SF_PAGE_STRIDE,
    Positions,
    POS_ROW,
    Lengths,
    Requests,
    BlockTable,
    TABLE_ROW,
    Out,
    Tables,
    Counts,
    TOP_K: tl.constexpr,
    PAGE: tl.constexpr,
    DIM: tl.constexpr,
    TOKENS: tl.constexpr,
    V_SF_SWIZZLED: tl.constexpr,
    REPEAT: tl.constexpr,
    HAS_LENGTHS: tl.constexpr,
):
    """FP8 rows from key positions: row r's key j (cache slot
    block_table[request, key // PAGE] * PAGE + key % PAGE) becomes FP8 row
    r * TOP_K + j, zero for a negative key; both of the row's index lists
    (REPEAT copies, one per attention request) hold r * TOP_K + j, and its
    count is its number of nonnegative keys, as token_sparse_kv_indices
    (Lengths[r] if given; keys past it then give zero rows)."""
    row, tile = tl.program_id(0).to(tl.int64), tl.program_id(1)
    column = tile * TOKENS + tl.arange(0, TOKENS)
    key = tl.load(Positions + row * POS_ROW + column)
    live = key >= 0
    if HAS_LENGTHS:
        live = live & (column < tl.load(Lengths + row))
    key = tl.maximum(key, 0)
    request = tl.load(Requests + row).to(tl.int64)
    page = tl.load(
        BlockTable + request * TABLE_ROW + key // PAGE, mask=live, other=0
    ).to(tl.int64)
    local = row * TOP_K + column
    _convert_tokens(
        K,
        V,
        KS,
        VS,
        PAGE_STRIDE,
        SF_PAGE_STRIDE,
        page,
        key % PAGE,
        live,
        Out + local[:, None] * (KV_COMPONENTS * DIM),
        DIM,
        TOKENS,
        V_SF_SWIZZLED,
        False,
    )
    for request_copy in tl.static_range(REPEAT):
        out_row = row * REPEAT + request_copy
        for side in tl.static_range(KV_COMPONENTS):
            tl.store(
                Tables + (out_row * KV_COMPONENTS + side) * TOP_K + column,
                local.to(tl.int32),
            )
    if tile == 0:
        if HAS_LENGTHS:
            count = tl.load(Lengths + row)
        else:
            keys = tl.load(Positions + row * POS_ROW + tl.arange(0, TOP_K))
            count = tl.sum((keys >= 0).to(tl.int32), 0)
        for request_copy in tl.static_range(REPEAT):
            tl.store(Counts + row * REPEAT + request_copy, count)
