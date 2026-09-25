"""Sparse prepared API: exact analytical values, native metadata consumption,
non-default-stream dependencies, changed-input graph replay, invalid-slot retention.

Analytical fixture derived from DeepGEMM schedule semantics.
Copyright (c) 2025 DeepSeek; upstream-derived portions are MIT licensed.
"""

import pytest
import torch

from flashinfer.experimental.deepgemm_sparse_mqa import sparse_mqa as _runtime
from flashinfer.sparse_mqa import prepare_sparse_mqa_logits, prepare_sparse_mqa_metadata


def _skip_unless_exported():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    try:
        arch = _runtime.device_arch(torch.device("cuda"))
    except RuntimeError as error:
        pytest.skip(str(error))
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    if sms not in _runtime.supported_num_sms(arch):
        pytest.skip(
            f"The exported {arch} schedules cover {_runtime.supported_num_sms(arch)} SMs, "
            f"this device has {sms}"
        )


def analytical_case(fmt, paged):
    queries, capacity, block, page = 9, 2048, 8, 64
    row = 64 if fmt == "mxfp4" else 128
    keys = 640 if fmt == "mxfp4" else 512
    positive, negative = (0x22, 0xAA) if fmt == "mxfp4" else (0x38, 0xB8)
    pages = keys // page
    counts = [keys // block - 2 * ((qi // 2) % 4) - qi % 2 for qi in range(queries)]
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
        qscale.extend(bytes([127 + qi % 2]) * (32 * 4))
    q = packed(qbytes, (queries, 32, row))
    sf_q = packed(qscale, (queries, 32), torch.int32)
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


def consume_native(case, metadata):
    import deep_gemm

    dtype = torch.int8 if case["fmt"] == "mxfp4" else torch.float8_e4m3fn
    q = case["q"].view(dtype)
    common = dict(
        weights=case["weights"],
        metadata=metadata,
        num_max_sparse_blocks=case["capacity"],
        sparse_block_kv=case["block"],
    )
    if case["paged"]:
        kv = case["kv"]
        width = case["row"] + 4
        cache = kv.as_strided(
            (kv.shape[0], case["page"], 1, width), (kv.stride(0), width, width, 1)
        )
        return deep_gemm.fp8_fp4_paged_sparse_mqa_logits(
            q=(q[:, None], case["sf_q"][:, None]), kv_cache=cache, **common
        )
    return deep_gemm.fp8_fp4_sparse_mqa_logits(
        q=(q, case["sf_q"]), kv=(case["kv"].view(dtype), case["sf_kv"]), **common
    )


@pytest.mark.parametrize("fmt", ["mxfp4", "mxfp8"])
@pytest.mark.parametrize("paged", [False, True])
def test_sparse_metadata_stream_and_replay(fmt, paged):
    _skip_unless_exported()
    deep_gemm = pytest.importorskip("deep_gemm")
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
    previous = deep_gemm.get_num_sms()
    deep_gemm.set_num_sms(metadata.num_sms)
    try:
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
            expected = analytical_expected(case, changed)
            assert torch.equal(output, expected)
            assert not bool(torch.count_nonzero(metadata.workspace[[0, 32, 64]]))
            # The standalone helper must retain the same cross-library byte ABI.
            metadata.run()
            native = consume_native(case, metadata.metadata)
            valid = expected != -1
            assert torch.equal(native[valid], expected[valid])
    finally:
        deep_gemm.set_num_sms(previous)
