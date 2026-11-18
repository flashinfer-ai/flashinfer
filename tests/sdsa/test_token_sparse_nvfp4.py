"""NVFP4 KV cache -> FP8 attention workspace from key positions:
gather_nvfp4_kv_cache_by_position (per row, one launch) and compact_nvfp4_kv_cache
(each selected key once per group of rows), against a torch decode reference."""

from functools import partial

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] not in (10, 11, 12),
    reason="NVFP4 to FP8 conversion requires a Blackwell GPU",
)
same = partial(torch.testing.assert_close, rtol=0, atol=0, equal_nan=True)

_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_PAGE = 64


def _nvfp4_pages(pages, page_size, head_dim, swizzled, data_pad=64, sf_pad=16):
    """Random packed K/V pages and random scales with distinct page strides."""
    scale_dim, data_dim = head_dim // 16, head_dim // 2
    data_buf = torch.randint(
        256,
        (pages, 2, page_size * data_dim + data_pad),
        dtype=torch.uint8,
        device="cuda",
    )
    data = data_buf[:, :, : page_size * data_dim].view(pages, 2, 1, page_size, data_dim)
    # Log-uniform positive scales across E4M3's normal and subnormal range.
    sf_linear = torch.exp2(
        torch.empty(pages, 2, page_size, scale_dim, device="cuda").uniform_(-9, 8.8)
    ).to(torch.float8_e4m3fn)
    sf_buf = torch.zeros(
        pages, 2, page_size * scale_dim + sf_pad, dtype=torch.uint8, device="cuda"
    )
    v_layout = sf_linear[:, 1]
    if swizzled:
        v_layout = (
            v_layout.reshape(pages, page_size // 4, 4, 4, scale_dim // 4)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )
    sf_buf[:, 0, : page_size * scale_dim].copy_(
        sf_linear[:, 0].contiguous().view(torch.uint8).reshape(pages, -1)
    )
    sf_buf[:, 1, : page_size * scale_dim].copy_(
        v_layout.contiguous().view(torch.uint8).reshape(pages, -1)
    )
    sf = (
        sf_buf[:, :, : page_size * scale_dim]
        .view(torch.float8_e4m3fn)
        .view(pages, 2, 1, page_size, scale_dim)
    )
    packed = data.reshape(pages, 2, page_size, data_dim)
    return data, sf, packed, sf_linear


def _decode_nvfp4(packed, sf_linear):
    """Reference: E2M1 value times its E4M3 block scale, divided by six, in FP8."""
    lut = torch.tensor(_E2M1 + tuple(-v for v in _E2M1), device=packed.device)
    codes = torch.stack((packed & 15, packed >> 4), -1).flatten(-2).long()
    values = lut[codes] * sf_linear.float().repeat_interleave(16, dim=-1)
    return (values / 6).to(torch.float8_e4m3fn)


def _case(requests, context, head_dim, swizzled, seed, page=_PAGE):
    """NVFP4 pages for ``requests`` requests of ``context`` keys each, behind a
    shuffled block table of ``page``-key pages, and their decoded reference."""
    torch.manual_seed(seed)
    per_request = -(-context // page)
    pages = requests * per_request + 3
    data, sf, packed, sf_linear = _nvfp4_pages(pages, page, head_dim, swizzled)
    order = torch.randperm(pages, device="cuda")[: requests * per_request]
    block_table = order.view(requests, per_request).to(torch.int32).contiguous()
    decoded = _decode_nvfp4(packed, sf_linear).view(pages, 2, page, head_dim)
    return data, sf, block_table, decoded


def _positions(counts, top_k, context, base=None, swap=0, stale=False):
    """int32 [rows, top_k] key positions: each row's ``counts[r]`` distinct keys
    first, then -1 (or stale nonnegative keys when ``stale``). With ``base``,
    rows take its first keys and replace up to ``swap`` of them, so rows of a
    group overlap like MTP verify tokens."""
    rows = []
    for count in counts:
        if base is None:
            keys = torch.randperm(context)[:count]
        else:
            keys = base[:count].clone()
            if swap and count:
                fresh = torch.randperm(context)
                fresh = fresh[~torch.isin(fresh, keys)][: min(swap, count)]
                where = torch.randperm(count)[: fresh.numel()]
                keys[where] = fresh
        row = torch.full((top_k,), -1, dtype=torch.int32)
        row[:count] = keys.to(torch.int32)
        if stale and count < top_k:
            row[count:] = torch.randint(context, (top_k - count,), dtype=torch.int32)
        rows.append(row)
    return torch.stack(rows).cuda()


def _reference(
    decoded, positions, requests, block_table, counts, page_size=_PAGE, max_keys=None
):
    """Expected K/V values [rows, top_k, 2, head_dim] (zero for dead entries)
    and the live mask (keys at or past ``max_keys`` are dead)."""
    rows, top_k = positions.shape
    keys = positions.long()
    live = (torch.arange(top_k, device="cuda")[None] < counts[:, None]) & (keys >= 0)
    if max_keys is not None:
        live &= keys < max_keys
    pages = block_table[requests.long()].long()
    k = keys.clamp(min=0)
    page = pages.gather(1, (k // page_size).clamp(max=pages.shape[1] - 1))
    token = (k % page_size).long()
    expected = decoded[page, :, token].float()  # [rows, top_k, 2, head_dim]
    return expected.masked_fill(~live[..., None, None], 0), live


def _check(out, tables, seq_lens, expected, live, counts, requests_per_row, zero_row):
    """Every live entry of every attention request addresses the expected FP8
    values (K and V separately); dead entries address zeros (or ``zero_row``)."""
    rows, top_k, _, head_dim = expected.shape
    flat = out.view(-1, 2, head_dim)
    for i in range(requests_per_row):
        lists = tables[i : rows * requests_per_row : requests_per_row].long()
        same(seq_lens[i : rows * requests_per_row : requests_per_row], counts)
        for side in (0, 1):
            got = flat[lists[:, side].clamp(min=0), side].float()
            # Bit patterns, so -0.0 and +0.0 differ.
            same(got.view(torch.int32), expected[:, :, side].view(torch.int32))
            if zero_row is not None:
                assert bool((lists[:, side][~live] == zero_row).all())


def test_every_e2m1_code_and_scale():
    """Every byte (two E2M1 codes) under every finite E4M3 block scale, both
    signs, converts to the reference's FP8 bits (-0.0 included)."""
    from flashinfer.sdsa.token_sparse import gather_nvfp4_kv_cache_by_position

    head_dim, page, top_k = 128, 64, 512
    data_dim, scale_dim = head_dim // 2, head_dim // 16
    codes = torch.arange(256, device="cuda")
    scales = codes[(codes & 0x7F) != 0x7F].to(torch.uint8)  # E4M3 without NaN
    # Scale block b (8 bytes) takes scale b // 32 and bytes 8 (b % 32) .. + 7:
    # 32 blocks per scale cover every byte once.
    tokens = -(-scales.numel() * 32 // scale_dim)
    pages = -(-tokens // page)
    block = torch.arange(pages * page * scale_dim, device="cuda")
    sf_codes = scales[(block // 32).clamp(max=scales.numel() - 1)]
    byte_codes = (block % 32 * 8)[:, None] + torch.arange(8, device="cuda")
    packed = byte_codes.to(torch.uint8).view(pages, page, data_dim)
    sf_linear = sf_codes.view(torch.float8_e4m3fn).view(pages, page, scale_dim)
    data = packed.view(pages, 1, page, data_dim)
    sf = sf_linear.view(pages, 1, page, scale_dim)
    rows = pages * page // top_k
    positions = torch.arange(rows * top_k, dtype=torch.int32, device="cuda")
    out = torch.empty(
        rows * top_k, 1, 1, 2 * head_dim, device="cuda", dtype=torch.float8_e4m3fn
    )
    gather_nvfp4_kv_cache_by_position(
        (data, data.clone()),
        (sf, sf.clone()),
        positions.view(rows, top_k),
        torch.zeros(rows, dtype=torch.int32, device="cuda"),
        torch.arange(pages, dtype=torch.int32, device="cuda")[None],
        out=out,
        block_tables=torch.empty(rows, 2, top_k, dtype=torch.int32, device="cuda"),
        seq_lens=torch.empty(rows, dtype=torch.int32, device="cuda"),
        v_sf_swizzled=False,
    )
    expected = _decode_nvfp4(packed, sf_linear).view(-1, head_dim).view(torch.uint8)
    got = out.view(-1, 2, head_dim).view(torch.uint8)
    assert torch.equal(got[:, 0], expected)
    assert torch.equal(got[:, 1], expected)


@pytest.mark.parametrize("swizzled", [True, False])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("top_k", [512, 2048])
@pytest.mark.parametrize("requests_per_row", [1, 2])
@pytest.mark.parametrize("stale", [False, True])
@pytest.mark.parametrize("page", [64, 16])
def test_gather_nvfp4_by_position(
    swizzled, head_dim, top_k, requests_per_row, stale, page
):
    """Positions -> FP8 rows and index lists in one launch, also after a
    CUDA-graph replay with other positions."""
    from flashinfer.sdsa.token_sparse import gather_nvfp4_kv_cache_by_position

    context = 3000
    data, sf, block_table, decoded = _case(
        3, context, head_dim, swizzled, 7 + top_k, page
    )
    requests = torch.tensor([0, 0, 1, 2, 2], dtype=torch.int32, device="cuda")
    counts = torch.tensor(
        [min(top_k, context), 17, 0, 300, top_k - 1], dtype=torch.int32, device="cuda"
    )
    positions = _positions(counts.tolist(), top_k, context, stale=stale)
    rows = positions.shape[0]
    out = torch.empty(
        rows * top_k, 1, 1, 2 * head_dim, device="cuda", dtype=torch.float8_e4m3fn
    )
    tables = torch.empty(
        rows * requests_per_row, 2, top_k, dtype=torch.int32, device="cuda"
    )
    seq_lens = torch.empty(rows * requests_per_row, dtype=torch.int32, device="cuda")
    lengths = counts.clone() if stale else None

    def run():
        gather_nvfp4_kv_cache_by_position(
            tuple(data.unbind(1)),
            tuple(sf.unbind(1)),
            positions,
            requests,
            block_table,
            out=out,
            block_tables=tables,
            seq_lens=seq_lens,
            lengths=lengths,
            requests_per_row=requests_per_row,
            v_sf_swizzled=swizzled,
        )

    def check():
        expected, live = _reference(
            decoded, positions, requests, block_table, counts, page
        )
        _check(out, tables, seq_lens, expected, live, counts, requests_per_row, None)

    run()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    positions.copy_(_positions(counts.tolist(), top_k, context, stale=stale))
    out.view(torch.uint8).fill_(0x38)  # stale 1.0 values must be overwritten
    graph.replay()
    check()


@pytest.mark.parametrize("swizzled", [True, False])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("dense", ["none", "mixed"])
@pytest.mark.parametrize("requests_per_row", [1, 2])
@pytest.mark.parametrize("page", [64, 16])
@pytest.mark.parametrize(
    "variant", ["plain", "stale", "split", "loop1", "loop3", "pad"]
)
def test_compact_nvfp4(
    swizzled, head_dim, dense, requests_per_row, page, variant, monkeypatch
):
    """Each group's selected keys converted once; every row's lists address the
    expected values for its entries, bit for bit the bytes
    gather_nvfp4_kv_cache_by_position writes; dead entries address the zero
    row; the bitmap is left zero, also across a CUDA-graph replay.

    stale: row lengths with stale keys past them, and max_keys below the
    context (not a multiple of 32): keys at or past it are dead but counted.
    split: two launches over different group ranges (group_base) share the
    positions and the output tables; each skips the other's rows.
    loop1 / loop3: the conversion grid capped to one / three programs per
    group, each looping over its group's tiles (stride 1 / 3).
    pad: an extra padding row (row group -1) points every entry at the zero
    row."""
    from flashinfer.sdsa.token_sparse import (
        compact_nvfp4_bitmap_numel,
        compact_nvfp4_kv_cache,
        compact_nvfp4_scratch_numel,
        gather_nvfp4_kv_cache_by_position,
    )

    if variant.startswith("loop"):
        from flashinfer.sdsa import token_sparse

        programs = 0 if variant == "loop1" else 7  # 3 groups: 1 / 3 per group
        monkeypatch.setattr(token_sparse, "_convert_programs", lambda _: programs)
    top_k, context = 512, 1500
    stale = variant == "stale"
    data, sf, block_table, decoded = _case(
        3, context, head_dim, swizzled, 11 + head_dim, page
    )
    # Group 0: six MTP-like rows of request 0 (overlapping selections);
    # group 1: three rows of request 1 (dense in "mixed"); group 2: two rows.
    group_rows = [6, 3, 2]
    counts_per_group = [[512, 512, 500, 512, 7, 0], [512, 300, 512], [100, 512]]

    def make_positions():
        return torch.cat(
            [
                _positions(
                    cnts,
                    top_k,
                    context,
                    base=torch.randperm(context),
                    swap=40,
                    stale=stale,
                )
                for cnts in counts_per_group
            ]
        )

    pad = variant == "pad"
    if pad:  # a padding row after the groups' rows
        inner = make_positions

        def make_positions():
            return torch.cat([inner(), _positions([300], top_k, context)])

    positions = make_positions()
    rows = positions.shape[0]
    counts = torch.tensor(
        sum(counts_per_group, []) + ([300] if pad else []),
        dtype=torch.int32,
        device="cuda",
    )
    lengths = counts.clone() if stale else None
    max_keys = context - 37 if stale else context
    row_groups = torch.repeat_interleave(
        torch.arange(3, dtype=torch.int32), torch.tensor(group_rows)
    )
    if pad:
        row_groups = torch.cat([row_groups, torch.tensor([-1], dtype=torch.int32)])
    row_groups = row_groups.cuda()
    group_requests = torch.arange(3, dtype=torch.int32, device="cuda")
    group_dense = torch.tensor(
        [0, 1 if dense == "mixed" else 0, 0], dtype=torch.int32, device="cuda"
    )
    caps = [
        context if bool(group_dense[g]) else min(n * top_k, context)
        for g, n in enumerate(group_rows)
    ]
    offsets = [0]
    for cap in caps[:-1]:
        offsets.append(offsets[-1] + cap)
    zero_row = offsets[-1] + caps[-1]
    group_offsets = torch.tensor(offsets, dtype=torch.int64, device="cuda")
    out = torch.empty(
        zero_row + 1, 1, 1, 2 * head_dim, device="cuda", dtype=torch.float8_e4m3fn
    )
    out.view(torch.uint8).fill_(0x38)
    tables = torch.empty(
        rows * requests_per_row, 2, top_k, dtype=torch.int32, device="cuda"
    )
    seq_lens = torch.empty(rows * requests_per_row, dtype=torch.int32, device="cuda")
    scratch = torch.empty(
        compact_nvfp4_scratch_numel(3, max_keys, zero_row + 1),
        dtype=torch.int32,
        device="cuda",
    )
    bitmap = torch.zeros(
        compact_nvfp4_bitmap_numel(3, max_keys), dtype=torch.int32, device="cuda"
    )
    row_requests = group_requests[row_groups.long().clamp(min=0)]
    # (group_base, groups) per launch
    launches = [(0, 2), (2, 1)] if variant == "split" else [(0, 3)]

    def run():
        for base, groups in launches:
            part = slice(base, base + groups)
            compact_nvfp4_kv_cache(
                tuple(data.unbind(1)),
                tuple(sf.unbind(1)),
                positions,
                row_groups,
                group_requests[part],
                block_table,
                group_offsets[part],
                out=out,
                block_tables=tables,
                seq_lens=seq_lens,
                scratch=scratch,
                bitmap=bitmap,
                max_keys=max_keys,
                max_capacity=max(caps[part]),
                zero_row=zero_row,
                group_base=base,
                group_dense=group_dense[part],
                lengths=lengths,
                requests_per_row=requests_per_row,
                v_sf_swizzled=swizzled,
            )

    def check():
        expected, live = _reference(
            decoded, positions, row_requests, block_table, counts, page, max_keys
        )
        if pad:  # the padding row: every entry at the zero row, its count kept
            expected[-1] = 0
            live[-1] = False
        _check(
            out, tables, seq_lens, expected, live, counts, requests_per_row, zero_row
        )
        assert not bitmap.any()
        # The same bytes as the per-row path for every live entry.
        rows_out = torch.empty(
            rows * top_k, 1, 1, 2 * head_dim, device="cuda", dtype=torch.float8_e4m3fn
        )
        gather_nvfp4_kv_cache_by_position(
            tuple(data.unbind(1)),
            tuple(sf.unbind(1)),
            positions,
            row_requests,
            block_table,
            out=rows_out,
            block_tables=torch.empty(rows, 2, top_k, dtype=torch.int32, device="cuda"),
            seq_lens=torch.empty(rows, dtype=torch.int32, device="cuda"),
            lengths=lengths,
            v_sf_swizzled=swizzled,
        )
        per_row = rows_out.view(torch.uint8).view(rows, top_k, 2 * head_dim)
        compact = out.view(torch.uint8).view(-1, 2 * head_dim)
        lists = tables[::requests_per_row].long()
        for side in (0, 1):
            half = slice(side * head_dim, (side + 1) * head_dim)
            got = compact[lists[:, side].clamp(min=0)][..., half]
            assert torch.equal(got[live], per_row[..., half][live])

    run()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    positions.copy_(make_positions())
    graph.replay()
    check()


def test_nvfp4_conversion_rejects_bad_inputs():
    """Malformed inputs raise ValueError before any launch."""
    from flashinfer.sdsa.token_sparse import (
        compact_nvfp4_bitmap_numel,
        compact_nvfp4_kv_cache,
        compact_nvfp4_scratch_numel,
        gather_nvfp4_kv_cache_by_position,
    )

    head_dim, top_k, context = 128, 512, 600
    data, sf, block_table, _ = _case(2, context, head_dim, True, 3)
    positions = _positions([100, 512], top_k, context)
    i32 = partial(torch.zeros, dtype=torch.int32, device="cuda")
    out = torch.empty(
        4 * top_k, 1, 1, 2 * head_dim, device="cuda", dtype=torch.float8_e4m3fn
    )
    gather_args = dict(
        paged_kv_cache=tuple(data.unbind(1)),
        kv_cache_sf=tuple(sf.unbind(1)),
        positions=positions,
        requests=torch.arange(2, dtype=torch.int32, device="cuda"),
        block_table=block_table,
        out=out,
        block_tables=i32(2, 2, top_k),
        seq_lens=i32(2),
    )
    gather_nvfp4_kv_cache_by_position(**gather_args)  # valid
    # head_dim 192: a multiple of 64, but not a power of two.
    dim192 = dict(
        paged_kv_cache=tuple(
            torch.zeros(4, 1, _PAGE, 96, dtype=torch.uint8, device="cuda")
            for _ in range(2)
        ),
        kv_cache_sf=tuple(
            torch.zeros(4, 1, _PAGE, 12, dtype=torch.uint8, device="cuda").view(
                torch.float8_e4m3fn
            )
            for _ in range(2)
        ),
        out=torch.empty(
            4 * top_k, 1, 1, 2 * 192, device="cuda", dtype=torch.float8_e4m3fn
        ),
    )
    bad_gather = [
        dim192,
        dict(seq_lens=i32(2, 0)),  # rank 2: no elements despite shape[0]
        dict(requests=torch.arange(2, dtype=torch.int32)),  # CPU
        dict(requests=torch.arange(4, dtype=torch.int32, device="cuda")[::2]),
        dict(requests=torch.arange(2, dtype=torch.int64, device="cuda")),
        dict(requests=torch.zeros(1, dtype=torch.int32, device="cuda")),  # short
        dict(block_table=block_table[:, :0]),  # no pages
        dict(lengths=torch.zeros(2, dtype=torch.int32)),  # CPU
        dict(requests_per_row=0),
        dict(tokens_per_program=3),
        dict(num_warps=3),
        dict(kv_cache_sf=(sf.unbind(1)[0].cpu(), sf.unbind(1)[1])),  # scales on CPU
    ]
    for change in bad_gather:
        with pytest.raises(ValueError):
            gather_nvfp4_kv_cache_by_position(**{**gather_args, **change})

    max_keys = context
    compact_args = dict(
        paged_kv_cache=tuple(data.unbind(1)),
        kv_cache_sf=tuple(sf.unbind(1)),
        positions=positions,
        row_groups=torch.arange(2, dtype=torch.int32, device="cuda"),
        group_requests=torch.arange(2, dtype=torch.int32, device="cuda"),
        block_table=block_table,
        group_offsets=torch.tensor([0, top_k], dtype=torch.int64, device="cuda"),
        out=out,
        block_tables=i32(2, 2, top_k),
        seq_lens=i32(2),
        scratch=i32(compact_nvfp4_scratch_numel(2, max_keys, out.shape[0])),
        bitmap=i32(compact_nvfp4_bitmap_numel(2, max_keys)),
        max_keys=max_keys,
        max_capacity=top_k,
        zero_row=2 * top_k,
    )
    compact_nvfp4_kv_cache(**compact_args)  # valid
    bad_compact = [
        dim192,
        dict(seq_lens=i32(2, 0)),
        dict(row_groups=torch.arange(4, dtype=torch.int32, device="cuda")[::2]),
        dict(row_groups=torch.arange(2, dtype=torch.int64, device="cuda")),
        dict(group_requests=torch.zeros(1, dtype=torch.int32, device="cuda")),
        dict(group_requests=torch.arange(2, dtype=torch.int32)),  # CPU
        dict(group_offsets=torch.tensor([0, top_k], dtype=torch.int32, device="cuda")),
        dict(group_dense=torch.zeros(1, dtype=torch.int32, device="cuda")),
        dict(block_table=block_table[:, :0]),
        dict(scratch=i32(2 * compact_args["scratch"].numel())[::2]),
        dict(bitmap=torch.zeros_like(compact_args["bitmap"], dtype=torch.int64)),
        dict(zero_row=out.shape[0]),
        dict(requests_per_row=0),
        dict(tokens_per_program=24),
        dict(paged_kv_cache=(data.unbind(1)[0], data.unbind(1)[1].cpu())),  # V on CPU
    ]
    for change in bad_compact:
        with pytest.raises(ValueError):
            compact_nvfp4_kv_cache(**{**compact_args, **change})


@pytest.mark.parametrize("capacity", [256, 257])
@pytest.mark.parametrize("loop", [False, True])
def test_compact_nvfp4_stops_at_max_capacity(loop, capacity, monkeypatch):
    """Region rows at or past max_capacity are not converted, with or without
    the looping grid: rows of out past a group's first max_capacity rows keep
    their contents, also within a partial last tile (capacity 257)."""
    from flashinfer.sdsa import token_sparse
    from flashinfer.sdsa.token_sparse import (
        compact_nvfp4_bitmap_numel,
        compact_nvfp4_kv_cache,
        compact_nvfp4_scratch_numel,
    )

    if loop:
        monkeypatch.setattr(token_sparse, "_convert_programs", lambda _: 0)
    head_dim, top_k, context = 128, 512, 1500
    data, sf, block_table, _ = _case(1, context, head_dim, True, 5)
    positions = _positions([512, 512], top_k, context)  # union > 512 keys
    region = 2 * top_k
    out = torch.empty(
        region + 1, 1, 1, 2 * head_dim, device="cuda", dtype=torch.float8_e4m3fn
    )
    out.view(torch.uint8).fill_(0x38)
    compact_nvfp4_kv_cache(
        tuple(data.unbind(1)),
        tuple(sf.unbind(1)),
        positions,
        torch.zeros(2, dtype=torch.int32, device="cuda"),
        torch.zeros(1, dtype=torch.int32, device="cuda"),
        block_table,
        torch.zeros(1, dtype=torch.int64, device="cuda"),
        out=out,
        block_tables=torch.empty(2, 2, top_k, dtype=torch.int32, device="cuda"),
        seq_lens=torch.empty(2, dtype=torch.int32, device="cuda"),
        scratch=torch.empty(
            compact_nvfp4_scratch_numel(1, context, region + 1),
            dtype=torch.int32,
            device="cuda",
        ),
        bitmap=torch.zeros(
            compact_nvfp4_bitmap_numel(1, context), dtype=torch.int32, device="cuda"
        ),
        max_keys=context,
        max_capacity=capacity,
        zero_row=region,
    )
    rows = out.view(torch.uint8).view(-1, 2 * head_dim)
    assert bool((rows[capacity:region] == 0x38).all())
    assert not bool((rows[:capacity] == 0x38).all(-1).any())


def test_compact_nvfp4_without_groups():
    """A launch with no groups still points its padding rows at the zero row
    and zeroes it, reads no group metadata, and leaves rows of other launches'
    groups as they are."""
    from flashinfer.sdsa.token_sparse import (
        compact_nvfp4_bitmap_numel,
        compact_nvfp4_kv_cache,
        compact_nvfp4_scratch_numel,
    )

    head_dim, top_k, context = 128, 512, 600
    data, sf, block_table, _ = _case(1, context, head_dim, True, 9)
    positions = _positions([100, 300], top_k, context)
    zero_row = 2 * top_k
    out = torch.empty(
        zero_row + 1, 1, 1, 2 * head_dim, device="cuda", dtype=torch.float8_e4m3fn
    )
    out.view(torch.uint8).fill_(0x38)
    tables = torch.full((2, 2, top_k), 7, dtype=torch.int32, device="cuda")
    seq_lens = torch.full((2,), 7, dtype=torch.int32, device="cuda")
    compact_nvfp4_kv_cache(
        tuple(data.unbind(1)),
        tuple(sf.unbind(1)),
        positions,
        # a padding row, and a row of another launch's group
        torch.tensor([-1, 3], dtype=torch.int32, device="cuda"),
        torch.empty(0, dtype=torch.int32, device="cuda"),
        block_table,
        torch.empty(0, dtype=torch.int64, device="cuda"),
        out=out,
        block_tables=tables,
        seq_lens=seq_lens,
        scratch=torch.empty(
            compact_nvfp4_scratch_numel(0, context, out.shape[0]),
            dtype=torch.int32,
            device="cuda",
        ),
        bitmap=torch.zeros(
            compact_nvfp4_bitmap_numel(1, context), dtype=torch.int32, device="cuda"
        ),
        max_keys=context,
        max_capacity=top_k,
        zero_row=zero_row,
    )
    rows = out.view(torch.uint8).view(-1, 2 * head_dim)
    assert bool((tables[0] == zero_row).all())
    assert int(seq_lens[0]) == 100
    assert bool((tables[1] == 7).all()) and int(seq_lens[1]) == 7
    assert not bool(rows[zero_row].any())
    assert bool((rows[:zero_row] == 0x38).all())
