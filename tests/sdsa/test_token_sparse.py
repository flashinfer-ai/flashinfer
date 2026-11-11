"""SDSA (Simplified DSA) token-sparse attention inputs from key positions:
token_sparse_kv_indices's K/V index lists, group_union_masks's unions and masks."""

import doctest
from functools import partial

import pytest
import torch

import flashinfer.sdsa.token_sparse
from flashinfer.sdsa.token_sparse import (
    group_union_masks,
    token_sparse_kv_indices,
    union_key_limit,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="token-sparse attention requires SM10x",
)
same = partial(torch.testing.assert_close, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("top_k", [512, 2048])
@pytest.mark.parametrize(
    ("tokens", "requests_per_token"), [(1, 1), (3, 2), (77, 1), (5, 4)]
)
@torch.inference_mode()
def test_token_sparse_kv_indices_match_reference_and_replay(
    tokens: int, requests_per_token: int, top_k: int
) -> None:
    used = tokens * requests_per_token
    # Each token's Top-K: `counts` keys of its request, then -1s.
    table = torch.randperm(4 * 2**14, device="cuda").int().view(4, 2**14)
    requests = torch.randint(0, 4, (tokens,), dtype=torch.int32, device="cuda")
    counts = torch.randint(0, top_k + 1, (tokens,), dtype=torch.int32).cuda()
    keys = torch.randint(0, 2**20, (tokens, top_k), dtype=torch.int32).cuda()
    keys[torch.arange(top_k, device="cuda") >= counts[:, None]] = -1
    slots = table[requests.long()[:, None], keys.clamp(min=0) // 64] * 64 + keys % 64
    tables = torch.full((used + 3, 2, top_k), -7, dtype=torch.int32, device="cuda")
    lengths = tables[:, 0, 0].clone()
    expected_tables, expected_lengths = tables.clone(), lengths.clone()
    expected = slots.masked_fill(keys < 0, -1).repeat_interleave(requests_per_token, 0)
    expected_tables[:used] = expected[:, None]
    expected_lengths[:used] = counts.repeat_interleave(requests_per_token)
    run = partial(
        token_sparse_kv_indices,
        keys,
        requests,
        table,
        requests_per_token=requests_per_token,
    )
    run(block_tables=tables, seq_lens=lengths)  # warm-up compiles
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(block_tables=tables, seq_lens=lengths)
    for _ in range(2):  # eager, then replayed
        same(tables, expected_tables)
        same(lengths, expected_lengths)
        tables.fill_(-7)
        lengths.fill_(-7)
        graph.replay()


@pytest.mark.parametrize("at_limit", [False, True])
@pytest.mark.parametrize("top_k", [512, 2048])
@pytest.mark.parametrize("group", [1, 4, 16])
@torch.inference_mode()
def test_group_union_masks_match_reference_and_replay(
    group: int, top_k: int, at_limit: bool
) -> None:
    """Two requests of 128 tokens each: request 0 has 5000 keys; request 1 has
    600000, past the 2**19 keys of a narrower block table, or union_key_limit,
    past 2**20. Short rows end in -1s; padding rows (mask row -1) end each
    request's tokens, and the last group is all padding (union row -1). Mask and
    union rows run in reverse order. Nothing is written past a union or a mask
    row's words. Row 0 has a key at num_keys, which counts as -1."""
    torch.manual_seed(group)
    rows = 256
    long = union_key_limit(group, top_k) if at_limit else 600000
    contexts, width = [5000, long], long // 64 + 1
    tables = torch.randperm(2 * width, device="cuda").int().view(2, width)
    request = torch.arange(rows, device="cuda") // 128
    lattice = torch.tensor(contexts, device="cuda")[request, None] // 4096
    picked = torch.rand(rows, 4096, device="cuda").argsort(1)[:, :top_k]
    picked = picked * lattice + torch.randint_like(picked, 64) % lattice
    counts = torch.full((rows,), top_k, device="cuda")
    counts[1::3] = torch.randint(1, top_k, (len(counts[1::3]),), device="cuda")
    counts[100:128] = counts[-max(group, 3) :] = 0  # padding rows
    keys = picked.masked_fill(torch.arange(top_k, device="cuda") >= counts[:, None], -1)
    num_keys = contexts[1]
    keys[0, 0] = num_keys  # past the keys: counts as -1
    positions = keys.int()
    live = counts > 0
    query_rows = torch.full((rows,), -1, dtype=torch.int32, device="cuda")
    query_rows[live] = torch.arange(int(live.sum()), device="cuda").flip(0).int()
    used = live.view(-1, group).any(1)
    unions = int(used.sum())
    group_unions = torch.full_like(used, -1, dtype=torch.int32)
    group_unions[used] = torch.arange(unions, device="cuda").flip(0).int()
    requests = (torch.arange(rows // group, device="cuda") * group // 128).int()
    columns = min(group * top_k, 128 * -(-contexts[1] // 128))
    columns = 256 * -(-columns // 256)
    union = torch.full((unions, columns), -7, dtype=torch.int32, device="cuda")
    sizes = torch.full((unions,), -7, dtype=torch.int32, device="cuda")
    mask = torch.full((int(live.sum()), columns // 32), -7, dtype=torch.int32).cuda()
    outputs = dict(union=union, union_sizes=sizes, mask=mask, num_keys=num_keys)
    inputs = positions, query_rows, group_unions, requests, tables

    def run() -> None:
        for buffer in union, sizes, mask:
            buffer.fill_(-7)
        group_union_masks(*inputs, **outputs)

    def check() -> None:
        for g in range(rows // group):
            mine = keys[g * group : (g + 1) * group]
            mine = mine.masked_fill((mine < 0) | (mine >= num_keys), -1)
            members = mine[mine >= 0].unique()
            size, table = len(members), tables[requests[g]].long()
            if not size:  # an all-padding group
                continue
            row = int(group_unions[g])
            assert sizes[row] == size
            same(union[row, :size], (table[members // 64] * 64 + members % 64).int())
            assert bool((union[row, size:] == -7).all())
            words = 4 * -(-size // 128)
            bits = torch.zeros(group, 32 * words, dtype=torch.long, device="cuda")
            column = torch.searchsorted(members, mine.clamp(min=0))
            bits.scatter_reduce_(1, column, (mine >= 0).long(), "amax")
            weights = 1 << torch.arange(32, device="cuda")
            packed = (bits.view(group, words, 32) * weights).sum(-1).int()
            queries = query_rows[g * group : (g + 1) * group]
            same(mask[queries[queries >= 0], :words], packed[queries >= 0])
            assert bool((mask[queries[queries >= 0], words:] == -7).all())

    run()  # warm-up compiles
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    context = torch.tensor(contexts, device="cuda")[request, None]
    keys = torch.where(keys >= 0, context - 1 - keys, -1)  # new keys
    keys[0, 0] = num_keys
    positions.copy_(keys)
    graph.replay()
    check()


def test_union_key_limit_examples() -> None:
    if torch.cuda.get_device_properties().shared_memory_per_block_optin != 227 * 1024:
        pytest.skip("The examples are for 227 KiB of shared memory")
    results = doctest.testmod(flashinfer.sdsa.token_sparse)
    assert results.attempted and not results.failed


_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


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
    lut = torch.tensor(_E2M1 + tuple(-v for v in _E2M1), device=packed.device)
    codes = torch.stack((packed & 15, packed >> 4), -1).flatten(-2).long()
    values = lut[codes] * sf_linear.float().repeat_interleave(16, dim=-1)
    return (values / 6).to(torch.float8_e4m3fn)


def _expected_gather(decoded, slots, lengths, head_dim):
    rows, _, top_k = slots.shape
    expected = torch.zeros(rows, top_k, 2, head_dim, device="cuda")
    for row, length in enumerate(lengths.tolist()):
        for side in (0, 1):
            chosen = slots[row, side, :length].long()
            expected[row, :length, side] = (
                decoded[:, side]
                .reshape(-1, head_dim)[chosen.clamp(min=0)]
                .float()
                .masked_fill(chosen[:, None] < 0, 0)
            )
    return expected


@pytest.mark.parametrize(
    "page_size,swizzled", [(4, True), (64, True), (4, False), (64, False), (6, False)]
)
@pytest.mark.parametrize("head_dim", [64, 128, 192, 256, 320])
def test_gather_nvfp4_selected_tokens(page_size, swizzled, head_dim):
    """Packed pages gather to FP8 without reading invalid or padded slots."""
    from flashinfer.sdsa.token_sparse import gather_nvfp4_kv_cache

    torch.manual_seed(1234 + page_size + head_dim)
    pages, rows, top_k = 7, 3, 512
    data, sf, packed, sf_linear = _nvfp4_pages(pages, page_size, head_dim, swizzled)
    assert data.stride(0) != sf.stride(0)
    slots = torch.randint(
        pages * page_size, (rows, 2, top_k), device="cuda", dtype=torch.int32
    )
    lengths = torch.tensor([0, 17, top_k], device="cuda", dtype=torch.int32)
    slots[0].fill_(2**30)  # ignored because the row is empty
    slots[2, 0, 0] = -1  # a negative slot inside the prefix reads as zero
    # Row 1 keeps valid slots past its length; they must still produce zeros.
    out = torch.empty(
        rows * top_k, 1, 1, 2 * head_dim, device="cuda", dtype=torch.float8_e4m3fn
    )
    tables = torch.empty_like(slots)
    decoded = _decode_nvfp4(packed, sf_linear)

    def run():
        gather_nvfp4_kv_cache(
            tuple(data.unbind(1)),
            tuple(sf.unbind(1)),
            slots,
            lengths,
            out=out,
            block_tables=tables,
            v_sf_swizzled=swizzled,
        )

    def check():
        torch.testing.assert_close(
            out.view(rows, top_k, 2, head_dim).float(),
            _expected_gather(decoded, slots, lengths, head_dim),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            tables,
            torch.arange(rows * top_k, device="cuda", dtype=torch.int32)
            .reshape(rows, 1, top_k)
            .expand(-1, 2, -1),
        )

    run()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    # Replay with a changed slot and changed lengths.
    slots[2, 1, 0] = (slots[2, 1, 0] + 1) % (pages * page_size)
    lengths.copy_(torch.tensor([3, 40, top_k - 5], device="cuda", dtype=torch.int32))
    slots[0].copy_(slots[2])
    out.view(torch.uint8).fill_(0x38)  # stale 1.0 values must be overwritten
    graph.replay()
    check()


def test_gather_nvfp4_input_offsets_beyond_int32():
    """Pages past 2 GiB of packed data are addressed without wrapping."""
    from flashinfer.sdsa.token_sparse import gather_nvfp4_kv_cache

    torch.manual_seed(7)
    page_size, head_dim, top_k = 64, 128, 8
    pages = 2**31 // (2 * page_size * head_dim // 2) + 2
    data_dim, scale_dim = head_dim // 2, head_dim // 16
    data = torch.zeros(
        pages, 2, 1, page_size, data_dim, dtype=torch.uint8, device="cuda"
    )
    assert data[-1].data_ptr() - data.data_ptr() > 2**31
    data[-1].random_(0, 256)
    sf = torch.ones(
        pages, 2, 1, page_size, scale_dim, dtype=torch.float8_e4m3fn, device="cuda"
    )
    sf[-1] = torch.exp2(
        torch.empty(2, 1, page_size, scale_dim, device="cuda").uniform_(-6, 8)
    ).to(torch.float8_e4m3fn)
    slots = (pages - 1) * page_size + torch.randint(
        page_size, (1, 2, top_k), device="cuda", dtype=torch.int32
    )
    lengths = torch.tensor([top_k], device="cuda", dtype=torch.int32)
    out = torch.empty(1 * top_k, 1, 1, 2 * head_dim, device="cuda").to(
        torch.float8_e4m3fn
    )
    gather_nvfp4_kv_cache(
        tuple(data.unbind(1)),
        tuple(sf.unbind(1)),
        slots,
        lengths,
        out=out,
        block_tables=torch.empty_like(slots),
        v_sf_swizzled=False,
    )
    last = _decode_nvfp4(
        data[-1].reshape(1, 2, page_size, data_dim),
        sf[-1].reshape(1, 2, page_size, scale_dim),
    )
    local = slots - (pages - 1) * page_size
    torch.testing.assert_close(
        out.view(1, top_k, 2, head_dim).float(),
        _expected_gather(last, local, lengths, head_dim),
        rtol=0,
        atol=0,
    )


def _gather_inputs(page_size=4, head_dim=64, rows=1, top_k=4, device="cuda"):
    data = tuple(
        torch.zeros(2, 1, page_size, head_dim // 2, dtype=torch.uint8, device=device)
        for _ in range(2)
    )
    scales = tuple(
        torch.ones(
            2, 1, page_size, head_dim // 16, dtype=torch.float8_e4m3fn, device=device
        )
        for _ in range(2)
    )
    slots = torch.zeros(rows, 2, top_k, dtype=torch.int32, device=device)
    return dict(
        paged_kv_cache=data,
        kv_cache_sf=scales,
        slots=slots,
        seq_lens=torch.full((rows,), top_k, dtype=torch.int32, device=device),
        out=torch.empty(
            rows * top_k, 1, 1, 2 * head_dim, dtype=torch.float8_e4m3fn, device=device
        ),
        block_tables=torch.empty_like(slots),
    )


@pytest.mark.parametrize(
    "invalid,match",
    [
        ("slots_rank", "slot lists shaped"),
        ("data_dtype", "uint8"),
        ("page_size", "four-token"),
        ("head_dim", "four-token"),
        ("device", "same CUDA device"),
    ],
)
def test_gather_nvfp4_rejects_invalid_inputs(invalid, match):
    """Invalid shapes, dtypes and devices fail with a clear message."""
    from flashinfer.sdsa.token_sparse import gather_nvfp4_kv_cache

    kwargs = _gather_inputs(
        page_size=6 if invalid == "page_size" else 4,
        head_dim=96 if invalid == "head_dim" else 64,
    )
    if invalid == "slots_rank":
        kwargs["slots"] = kwargs["slots"][0]
    elif invalid == "data_dtype":
        kwargs["paged_kv_cache"] = tuple(
            t.view(torch.int8) for t in kwargs["paged_kv_cache"]
        )
    elif invalid == "device":
        kwargs["seq_lens"] = kwargs["seq_lens"].cpu()
    with pytest.raises(ValueError, match=match):
        gather_nvfp4_kv_cache(**kwargs)


def test_gather_nvfp4_runs_on_the_tensors_device():
    """The launch follows the tensors' device, not the current device."""
    from flashinfer.sdsa.token_sparse import gather_nvfp4_kv_cache

    if torch.cuda.device_count() < 2:
        pytest.skip("Requires two GPUs")
    torch.manual_seed(11)
    packed = torch.randint(256, (2, 2, 1, 4, 32), dtype=torch.uint8)
    results = []
    for device in ("cuda:0", "cuda:1"):
        kwargs = _gather_inputs(page_size=4, head_dim=64, top_k=4, device=device)
        kwargs["paged_kv_cache"][0].copy_(packed[:, 0])
        kwargs["paged_kv_cache"][1].copy_(packed[:, 1])
        kwargs["slots"].copy_(torch.tensor([[[0, 1, 2, 3], [4, 5, 6, 7]]]))
        with torch.cuda.device(0):
            gather_nvfp4_kv_cache(**kwargs)
        torch.cuda.synchronize(device)
        results.append(kwargs["out"].float().cpu())
    torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)


@pytest.mark.parametrize("cache_index", [None, 0, 1, 2, 3])
def test_gather_nvfp4_rejects_missing_page_dimensions(cache_index):
    """Malformed cache ranks are rejected even when there is no gather work."""
    from flashinfer.sdsa.token_sparse import gather_nvfp4_kv_cache

    caches = [
        torch.empty(2, 1, 4, width, dtype=dtype, device="cuda")
        for width, dtype in [(32, torch.uint8)] * 2 + [(4, torch.float8_e4m3fn)] * 2
    ]
    if cache_index is None:
        # These matching rank-three shapes formerly passed every layout check.
        caches = [t[:, :, 0].contiguous() for t in caches]
    else:
        caches[cache_index] = caches[cache_index].flatten(1, 2)
    slots = torch.empty(0, 2, 4, dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="rank-four HND"):
        gather_nvfp4_kv_cache(
            tuple(caches[:2]),
            tuple(caches[2:]),
            slots,
            torch.empty(0, dtype=torch.int32, device="cuda"),
            out=torch.empty(0, 1, 1, 128, dtype=torch.float8_e4m3fn, device="cuda"),
            block_tables=torch.empty_like(slots),
        )


@pytest.mark.parametrize("page_size,head_dim", [(0, 64), (4, 0)])
def test_gather_nvfp4_rejects_empty_page_or_head(page_size, head_dim):
    """The kernel's page division and scale swizzle require positive extents."""
    from flashinfer.sdsa.token_sparse import gather_nvfp4_kv_cache

    data = tuple(
        torch.empty(2, 1, page_size, head_dim // 2, dtype=torch.uint8, device="cuda")
        for _ in range(2)
    )
    scales = tuple(
        torch.empty(
            2, 1, page_size, head_dim // 16, dtype=torch.float8_e4m3fn, device="cuda"
        )
        for _ in range(2)
    )
    slots = torch.empty(0, 2, 4, dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="positive page_size and head_dim"):
        gather_nvfp4_kv_cache(
            data,
            scales,
            slots,
            torch.empty(0, dtype=torch.int32, device="cuda"),
            out=torch.empty(
                0, 1, 1, 2 * head_dim, dtype=torch.float8_e4m3fn, device="cuda"
            ),
            block_tables=torch.empty_like(slots),
        )


def test_gather_nvfp4_output_offsets_beyond_int32():
    """Large FP8 workspaces must not wrap their byte offsets at 2 GiB."""
    from flashinfer.sdsa.token_sparse import gather_nvfp4_kv_cache

    rows, top_k, head_dim = 10923, 512, 192
    data = tuple(
        torch.full((1, 1, 4, head_dim // 2), 0x22, dtype=torch.uint8, device="cuda")
        for _ in range(2)
    )
    scales = tuple(
        torch.full(
            (1, 1, 4, head_dim // 16), 6.0, dtype=torch.float8_e4m3fn, device="cuda"
        )
        for _ in range(2)
    )
    slots = torch.zeros(rows, 2, top_k, dtype=torch.int32, device="cuda")
    out = torch.full(
        (rows * top_k, 1, 1, 2 * head_dim),
        float("nan"),
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    tables = torch.empty_like(slots)
    gather_nvfp4_kv_cache(
        data,
        scales,
        slots,
        torch.full((rows,), top_k, dtype=torch.int32, device="cuda"),
        out=out,
        block_tables=tables,
    )
    # Each nibble is E2M1 1.0 and the scale cancels the workspace divisor.
    edges = torch.stack((out[0], out[-1])).float()
    torch.testing.assert_close(edges, torch.ones_like(edges), atol=0, rtol=0)
    expected = torch.arange(top_k, device="cuda", dtype=torch.int32)
    expected += (rows - 1) * top_k
    torch.testing.assert_close(tables[-1], expected.expand(2, -1))
