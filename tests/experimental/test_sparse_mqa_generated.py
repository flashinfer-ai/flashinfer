"""Sparse prepared API: exact analytical values at several sparse capacities,
metadata decoding at runtime page sizes / block sizes / split widths,
non-default-stream dependencies, changed-input graph replay, invalid-slot
retention, and the optional native metadata consumer.

Analytical fixture derived from DeepGEMM schedule semantics.
Copyright (c) 2025 DeepSeek; upstream-derived portions are MIT licensed.
"""

import pytest
import torch

from flashinfer.experimental.deepgemm_sparse_mqa import sparse_mqa as _runtime
from flashinfer.sparse_mqa import prepare_sparse_mqa_logits, prepare_sparse_mqa_metadata

HEADS = 32


def _skip_unless_exported():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    try:
        _runtime.device_facts(torch.cuda.current_device())
    except RuntimeError as error:
        pytest.skip(str(error))


def analytical_case(fmt, paged, capacity=2048, page=64, block=8, queries=9):
    row = 64 if fmt == "mxfp4" else 128
    keys = 640 if fmt == "mxfp4" else 512
    positive, negative = (0x22, 0xAA) if fmt == "mxfp4" else (0x38, 0xB8)
    pages = keys // page
    max_blocks = min(capacity, keys // block)
    counts = [max_blocks - 2 * ((qi // 2) % 4) - qi % 2 for qi in range(queries)]
    requests = [qi // 3 for qi in range(queries)]
    table = [[(p + request) % pages for p in range(pages)] for request in requests]

    def tensor(value, dtype):
        return torch.tensor(value, dtype=dtype, device="cuda")

    def packed(value, shape, dtype=torch.uint8):
        return torch.frombuffer(value, dtype=dtype).reshape(shape).to("cuda")

    qbytes, qscale = bytearray(), bytearray()
    for qi in range(queries):
        qbytes.extend(bytes([positive]) * (16 * row))
        qbytes.extend(bytes([negative]) * (16 * row))
        qscale.extend(bytes([127 + qi % 2]) * (HEADS * 4))
    q = packed(qbytes, (queries, HEADS, row))
    sf_q = packed(qscale, (queries, HEADS), torch.int32)
    weights = tensor(
        [[1 / 32] * 16 + [1 / 64] * 16 for _ in range(queries)], torch.bfloat16
    )
    kvbytes, kvscale = bytearray(), bytearray()
    if paged:
        stride = (page * (row + 4) + 511) // 512 * 512
        for pi in range(pages):
            for token in range(page):
                kvbytes.extend(bytes([negative if token % 2 else positive]) * row)
            kvbytes.extend(
                bytes([127 + pi % 2, 128 + pi % 2, 126 + pi % 2, 127 + pi % 2]) * page
            )
            kvbytes.extend(bytes(stride - page * (row + 4)))
        kv, sf_kv = packed(kvbytes, (pages, stride)), None
    else:
        for token in range(keys):
            pi = token // page
            kvbytes.extend(bytes([negative if token % 2 else positive]) * row)
            kvscale.extend(
                bytes([127 + pi % 2, 128 + pi % 2, 126 + pi % 2, 127 + pi % 2])
            )
        kv = packed(kvbytes, (keys, row))
        sf_kv = packed(kvscale, (keys,), torch.int32)
    sparse = tensor(
        [list(range(n)) + [n - 1] * (capacity - n) for n in counts], torch.int32
    )
    ends = tensor([n * block for n in counts], torch.int32)
    kwargs = dict(fmt=fmt, sparse_block_kv=block, page_kv=page)
    if paged:
        kwargs.update(
            context_lens=ends,
            block_table=tensor(table, torch.int32),
            request_indices=tensor(requests, torch.int32),
        )
    else:
        kwargs.update(starts=torch.zeros_like(ends), ends=ends, num_kv_tokens=keys)
    return dict(
        q=q,
        sf_q=sf_q,
        kv=kv,
        sf_kv=sf_kv,
        weights=weights,
        sparse=sparse,
        ends=ends,
        kwargs=kwargs,
        counts=counts,
        table=table,
        fmt=fmt,
        paged=paged,
        page=page,
        block=block,
        capacity=capacity,
        row=row,
        queries=queries,
    )


def analytical_expected(case, changed):
    rows = []
    for qi, count in enumerate(case["counts"]):
        valid = count - int(changed)
        values = []
        for token in range(valid * case["block"]):
            pi = (
                case["table"][qi][token // case["page"]]
                if case["paged"]
                else token // case["page"]
            )
            dot = 144 * (1 << (qi % 2 + pi % 2))
            values.append((dot // (4 if token % 2 else 2)) * (2 if changed else 1))
        rows.append(values + [-1] * (case["capacity"] * case["block"] - len(values)))
    return torch.tensor(rows, device="cuda", dtype=torch.bfloat16)


def _run_pipeline(case):
    metadata = prepare_sparse_mqa_metadata(case["sparse"], **case["kwargs"])
    output = torch.full(
        (case["queries"], case["capacity"] * case["block"]),
        -1,
        device="cuda",
        dtype=torch.bfloat16,
    )
    plan = prepare_sparse_mqa_logits(
        case["q"],
        case["sf_q"],
        case["kv"],
        case["sf_kv"],
        case["weights"],
        metadata,
        output=output,
    )
    plan.run()
    torch.cuda.synchronize()
    return metadata, plan, output


@pytest.mark.parametrize("capacity", [128, 1024, 2048])
@pytest.mark.parametrize("fmt", ["mxfp4", "mxfp8"])
@pytest.mark.parametrize("paged", [False, True])
def test_sparse_logits_analytical(fmt, paged, capacity):
    """Exact values at sparse capacities the programs receive at runtime; invalid slots keep their contents."""
    _skip_unless_exported()
    case = analytical_case(fmt, paged, capacity=capacity)
    metadata, plan, output = _run_pipeline(case)
    assert torch.equal(output, analytical_expected(case, False))
    assert not bool(torch.count_nonzero(metadata.workspace[[0, 32, 64]]))
    key = _runtime.metadata_route_key(
        paged=paged,
        capacity=capacity,
        sparse_block_kv=8,
        page_kv=64,
        num_sms=metadata.num_sms,
    )
    # The production geometry (capacity 2048, 8-token blocks, 64-token pages) on an exported SM count runs the
    # exact-geometry program, every other capacity the runtime program of the layout.
    assert metadata.num_sms in _runtime.LIMITS["exact_num_sms"]
    assert (":exact:" in key) == (capacity == _runtime.LIMITS["exact_capacity"])
    assert plan.programs["metadata"] == _runtime.ROUTES[key]


@pytest.mark.parametrize("fmt", ["mxfp4", "mxfp8"])
@pytest.mark.parametrize("paged", [False, True])
def test_sparse_metadata_stream_and_replay(fmt, paged):
    _skip_unless_exported()
    case = analytical_case(fmt, paged)
    metadata = prepare_sparse_mqa_metadata(case["sparse"], **case["kwargs"])
    output = torch.full(
        (9, case["capacity"] * case["block"]), -1, device="cuda", dtype=torch.bfloat16
    )
    plan = prepare_sparse_mqa_logits(
        case["q"],
        case["sf_q"],
        case["kv"],
        case["sf_kv"],
        case["weights"],
        metadata,
        output=output,
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        plan.run()
    stream.synchronize()
    assert torch.equal(output, analytical_expected(case, False))
    with torch.cuda.stream(stream):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            plan.run()
    for changed in (False, True):
        with torch.cuda.stream(stream):
            if changed:
                case["weights"].mul_(2)
                case["ends"].sub_(case["block"])
            output.fill_(-1)
            graph.replay()
        stream.synchronize()
        assert torch.equal(output, analytical_expected(case, changed))
        assert not bool(torch.count_nonzero(metadata.workspace[[0, 32, 64]]))


def decode_metadata(words, *, blocks_per_split, num_sms):
    """Decode the packed metadata into split headers, records and schedule entries."""
    total, waves, unaligned = int(words[0]), int(words[1]), int(words[2])
    split_words = 4 + blocks_per_split * 2
    splits = []
    for s in range(total):
        base = 4 + s * split_words
        header = [int(v) for v in words[base : base + 4]]
        records = words[base + 4 : base + split_words].reshape(blocks_per_split, 2)
        num = header[1] & 0x7FFFFFFF
        splits.append(
            dict(
                qbase=header[0],
                num=num,
                contiguous=bool(header[1] >> 31),
                b0=header[2],
                b1=header[3],
                physical=[int(v) for v in records[:num, 0]],
                offsets=[(int(v) & 0xFFFF, int(v) >> 16) for v in records[:num, 1]],
            )
        )
    sched = 4 + total * split_words
    entries = words[sched : sched + waves * num_sms * 4].reshape(waves * num_sms, 4)
    entries = [
        [int(v) for v in entry] for entry in entries if int(entry[0]) != int(entry[1])
    ]
    return dict(
        total=total, waves=waves, unaligned=unaligned, splits=splits, entries=entries
    )


def check_metadata(plan, *, sparse, counts, paged, block, page, block_table=None):
    """Every (query, slot) below the query's block count appears exactly once with its physical block;
    the schedule covers every split exactly once with the split's query block."""
    words = plan.metadata.view(torch.int32).cpu().to(torch.int64) & 0xFFFFFFFF
    nb = _runtime.blocks_per_split(plan.config["fmt"], block)
    decoded = decode_metadata(words, blocks_per_split=nb, num_sms=plan.num_sms)
    rows = sparse.cpu().tolist()
    table = block_table.cpu().tolist() if paged else None
    bpp = page // block
    seen = set()
    for split in decoded["splits"]:
        qbase, nq_slots = (
            split["qbase"],
            [split["b0"]] + ([split["b1"]] if split["b1"] != 0xFFFFFFFF else []),
        )
        for physical, offsets in zip(split["physical"], split["offsets"], strict=False):
            logical = None
            for j, base in enumerate(nq_slots):
                offset = offsets[j]
                if offset == 0xFFFF:
                    continue
                slot = base + offset
                q = qbase + j
                assert slot < counts[q], (q, slot, counts[q])
                value = rows[q][slot]
                assert logical in (None, value), (q, slot, value, logical)
                logical = value
                assert (q, slot) not in seen, (q, slot)
                seen.add((q, slot))
            assert logical is not None
            expected = (
                table[qbase][logical // bpp] * bpp + logical % bpp
                if paged
                else logical * block
            )
            assert physical == expected, (qbase, logical, physical, expected)
        if split["contiguous"]:
            assert split["num"] == nb
    assert seen == {
        (q, slot) for q, count in enumerate(counts) for slot in range(count)
    }
    covered = sorted(
        (begin, end, qbase) for begin, end, qbase, _nq in decoded["entries"]
    )
    assert sum(end - begin for begin, end, _ in covered) == decoded["total"]
    cursor = 0
    for begin, end, qbase in covered:
        assert begin == cursor, (begin, cursor)
        assert all(decoded["splits"][s]["qbase"] == qbase for s in range(begin, end))
        cursor = end
    assert cursor == decoded["total"]


def geometry_case(fmt, paged, *, capacity, block, page, queries=37, seed=7):
    """Random sorted sparse rows with varied block counts inside each query's window."""
    g = torch.Generator().manual_seed(seed)
    requests = [qi // 4 for qi in range(queries)]
    window_blocks = 3 * capacity // 2 if capacity < 1024 else capacity // 2 + 64
    counts = [
        int(torch.randint(1, min(capacity, window_blocks) + 1, (1,), generator=g))
        for _ in range(queries)
    ]
    rows = []
    for count in counts:
        chosen = (
            torch.randperm(window_blocks, generator=g)[:count].sort().values.tolist()
        )
        rows.append(chosen + [chosen[-1]] * (capacity - count))
    sparse = torch.tensor(rows, dtype=torch.int32, device="cuda")
    tokens = window_blocks * block
    # Each query's window holds exactly its block count (the top-k contract: the
    # row carries min(capacity, window blocks) selected blocks, then padding).
    ends = torch.tensor(
        [count * block for count in counts], dtype=torch.int32, device="cuda"
    )
    kwargs = dict(fmt=fmt, sparse_block_kv=block, page_kv=page)
    if paged:
        pages = (tokens + page - 1) // page
        table = torch.stack(
            [torch.randperm(pages, generator=g) for _ in range(max(requests) + 1)]
        ).to(torch.int32)
        table = table[torch.tensor(requests)].contiguous().cuda()
        kwargs.update(
            context_lens=ends,
            block_table=table,
            request_indices=torch.tensor(requests, dtype=torch.int32, device="cuda"),
        )
    else:
        table = None
        kwargs.update(starts=torch.zeros_like(ends), ends=ends, num_kv_tokens=tokens)
    return dict(sparse=sparse, kwargs=kwargs, counts=counts, block_table=table)


@pytest.mark.parametrize(
    "fmt,paged,capacity,block,page",
    [
        ("mxfp4", True, 256, 8, 32),
        ("mxfp8", True, 512, 16, 128),
        ("mxfp4", True, 1024, 8, 128),
        ("mxfp4", False, 256, 16, 64),
        ("mxfp8", False, 1024, 8, 64),
        ("mxfp8", False, 2048, 16, 64),
        ("mxfp4", True, 2048, 8, 64),
        ("mxfp8", False, 2048, 8, 64),
    ],
)
def test_sparse_metadata_runtime_geometry(fmt, paged, capacity, block, page):
    """The runtime metadata kernel takes the split width, capacity, block size and page size at runtime; the
    production geometry (capacity 2048, 8-token blocks, 64-token pages) runs the exact-geometry program."""
    _skip_unless_exported()
    case = geometry_case(fmt, paged, capacity=capacity, block=block, page=page)
    plan = prepare_sparse_mqa_metadata(case["sparse"], **case["kwargs"])
    key = _runtime.metadata_route_key(
        paged=paged,
        capacity=capacity,
        sparse_block_kv=block,
        page_kv=page,
        num_sms=plan.num_sms,
    )
    exact = (capacity, block) == (_runtime.LIMITS["exact_capacity"], 8) and (
        not paged or page == 64
    )
    assert (":exact:" in key) == exact
    # A device SM count the exact program was not compiled for falls back to the runtime program.
    foreign_sms = max(_runtime.LIMITS["exact_num_sms"]) + 1
    assert not _runtime.metadata_route_key(
        paged=paged,
        capacity=capacity,
        sparse_block_kv=block,
        page_kv=page,
        num_sms=foreign_sms,
    ).count(":exact:")
    assert plan.program == _runtime.ROUTES[key]
    plan.run()
    torch.cuda.synchronize()
    check_metadata(
        plan,
        sparse=case["sparse"],
        counts=case["counts"],
        paged=paged,
        block=block,
        page=page,
        block_table=case["block_table"],
    )
    assert not bool(torch.count_nonzero(plan.workspace[[0, 32, 64]]))


def test_sparse_metadata_rejects_non_power_of_two_page():
    """Pages hold a power-of-two number of blocks: the kernel splits pages with shifts."""
    _skip_unless_exported()
    case = geometry_case("mxfp4", True, capacity=256, block=8, page=64)
    with pytest.raises(ValueError, match="power of two"):
        prepare_sparse_mqa_metadata(case["sparse"], **dict(case["kwargs"], page_kv=24))


def test_sparse_logits_rejects_unexported_geometry():
    _skip_unless_exported()
    case = analytical_case("mxfp4", True, capacity=128, page=32)
    metadata = prepare_sparse_mqa_metadata(case["sparse"], **case["kwargs"])
    with pytest.raises(NotImplementedError):
        prepare_sparse_mqa_logits(
            case["q"],
            case["sf_q"],
            case["kv"],
            case["sf_kv"],
            case["weights"],
            metadata,
        )


@pytest.mark.parametrize("fmt", ["mxfp4", "mxfp8"])
@pytest.mark.parametrize("paged", [False, True])
def test_sparse_metadata_native_consumer(fmt, paged):
    """The standalone helper retains the cross-library byte ABI of the native consumer."""
    _skip_unless_exported()
    deep_gemm = pytest.importorskip("deep_gemm")
    native_logits = (
        "fp8_fp4_paged_sparse_mqa_logits" if paged else "fp8_fp4_sparse_mqa_logits"
    )
    if not hasattr(deep_gemm, native_logits):
        pytest.skip(
            f"installed deep_gemm has no {native_logits} (sparse MQA indexer not built)"
        )
    case = analytical_case(fmt, paged)
    metadata, _plan, output = _run_pipeline(case)
    expected = analytical_expected(case, False)
    previous = deep_gemm.get_num_sms()
    deep_gemm.set_num_sms(metadata.num_sms)
    try:
        dtype = torch.int8 if fmt == "mxfp4" else torch.float8_e4m3fn
        q = case["q"].view(dtype)
        common = dict(
            weights=case["weights"],
            metadata=metadata.metadata,
            num_max_sparse_blocks=case["capacity"],
            sparse_block_kv=case["block"],
        )
        if paged:
            kv = case["kv"]
            width = case["row"] + 4
            cache = kv.as_strided(
                (kv.shape[0], case["page"], 1, width), (kv.stride(0), width, width, 1)
            )
            native = deep_gemm.fp8_fp4_paged_sparse_mqa_logits(
                q=(q[:, None], case["sf_q"][:, None]), kv_cache=cache, **common
            )
        else:
            native = deep_gemm.fp8_fp4_sparse_mqa_logits(
                q=(q, case["sf_q"]),
                kv=(case["kv"].view(dtype), case["sf_kv"]),
                **common,
            )
    finally:
        deep_gemm.set_num_sms(previous)
    valid = expected != -1
    assert torch.equal(native[valid], expected[valid])
    assert torch.equal(output, expected)
