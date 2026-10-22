"""SM89 FA2 MLA FP8-cache decode regressions, independent of experiment scripts.

Copyright (c) 2026 by FlashInfer team.
Licensed under the Apache License, Version 2.0.
"""

import math

import pytest
import torch

from flashinfer.mla import BatchMLAPagedAttentionWrapper, MLAPlanMetadata

pytestmark = pytest.mark.gpu_exclusive


@pytest.fixture(autouse=True, scope="module")
def require_sm89():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 9):
        pytest.skip("SM89 FP8 MLA adaptation requires compute capability 8.9")
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = old


# Both plan-time layouts, partial pages, high-head causal cluster boundaries,
# empty requests, and non-power-of-two page/head counts.
CASES = [
    pytest.param([0], 8, 1, False, id="empty"),
    pytest.param([1, 0, 17], 16, 1, True, id="page1-mixed"),
    pytest.param([17], 192, 32, True, id="causal-head192-tail"),
    pytest.param([33, 65], 257, 64, False, id="head257-tail"),
    pytest.param([7, 129, 513], 64, 32, True, id="variable"),
    pytest.param([8193], 128, 64, False, id="long-split"),
    pytest.param([8193, 8193], 16, 32, True, id="large-layout"),
    pytest.param([1025] * 17, 8, 64, False, id="batch17"),
]


def make_inputs(lengths, heads, page, strided=False):
    torch.manual_seed(731)
    batch = len(lengths)
    counts = [(n + page - 1) // page for n in lengths]
    pool = max(3, sum(counts) + 7)
    ids = torch.randperm(pool).tolist()[: sum(counts)]
    offsets = [0]
    for count in counts:
        offsets.append(offsets[-1] + count)
    # Padding remains NaN, including the invalid tail of each allocated page.
    caches = []
    for dim in (512, 64):
        backing = torch.full(
            (pool, page, dim * (2 if strided else 1)), float("nan"), device="cuda"
        )
        for i, n in enumerate(lengths):
            for j in range(counts[i]):
                valid = min(page, n - j * page)
                backing[ids[offsets[i] + j], :valid, :dim] = (
                    torch.randn(valid, dim, device="cuda") * 0.25
                )
        caches.append(backing.to(torch.float8_e4m3fn)[..., :dim])
    queries = []
    for dim in (512, 64):
        queries.append(
            torch.randn(
                batch,
                heads,
                dim * (2 if strided else 1),
                device="cuda",
                dtype=torch.bfloat16,
            )[..., :dim]
            * 0.25
        )
    # Preserve query stride (arithmetic above may create contiguous output).
    if strided:
        # Last dimension stays contiguous; rows have padding.
        queries = [
            torch.cat((q, torch.zeros_like(q)), dim=-1)[..., : q.shape[-1]]
            for q in queries
        ]
    table = torch.full((batch, max(1, max(counts))), -1, dtype=torch.int32)
    for i, count in enumerate(counts):
        table[i, :count] = torch.tensor(
            ids[offsets[i] : offsets[i + 1]], dtype=torch.int32
        )
    meta = MLAPlanMetadata.csr(
        torch.arange(batch + 1, dtype=torch.int32),
        torch.tensor(offsets, dtype=torch.int32),
        torch.tensor(ids, dtype=torch.int32, device="cuda"),
        torch.tensor(lengths, dtype=torch.int32),
    )
    dense = MLAPlanMetadata.dense(meta.qo_indptr, table, meta.kv_len_arr, 1)
    return queries, caches, meta, dense, ids, offsets


def oracle(queries, caches, lengths, ids, offsets, page, scales, basee):
    # The API dequantizes into BF16: round scale, multiply, then round the
    # product. Attention itself is independently evaluated in FP32 here.
    decoded = [
        (
            x.to(torch.bfloat16) * torch.tensor(s, device="cuda", dtype=torch.bfloat16)
        ).float()
        for x, s in zip(caches, scales, strict=True)
    ]
    out = torch.zeros_like(queries[0])
    lse = torch.full(queries[0].shape[:2], -float("inf"), device="cuda")
    for i, n in enumerate(lengths):
        if n == 0:
            continue
        page_ids = torch.tensor(ids[offsets[i] : offsets[i + 1]], device="cuda")
        c, p = [x[page_ids].reshape(-1, x.shape[-1])[:n] for x in decoded]
        scores = (
            queries[0][i].float() @ c.T + queries[1][i].float() @ p.T
        ) / math.sqrt(192)
        out[i] = (scores.softmax(-1) @ c).to(torch.bfloat16)
        lse[i] = scores.logsumexp(-1) * (1 if basee else math.log2(math.e))
    return out, lse


def plan(wrapper, metadata, heads, page, causal=False, lse_mode="base2", **overrides):
    kwargs = dict(
        metadata=metadata,
        num_heads=heads,
        head_dim_ckv=512,
        head_dim_kpe=64,
        page_size=page,
        causal=causal,
        sm_scale=1 / math.sqrt(192),
        q_data_type=torch.bfloat16,
        kv_data_type=torch.float8_e4m3fn,
        output_dtype=torch.bfloat16,
        query_layout="split",
        kv_cache_layout="split",
        lse_mode=lse_mode,
        scale_mode="kv-per-tensor",
    )
    kwargs.update(overrides)
    wrapper.plan(**kwargs)


def new_wrapper(batch=1, graph=False):
    kwargs = {}
    if graph:
        kwargs = dict(
            qo_indptr=torch.empty(batch + 1, dtype=torch.int32, device="cuda"),
            kv_indptr=torch.empty(batch + 1, dtype=torch.int32, device="cuda"),
            kv_indices=torch.empty(65536, dtype=torch.int32, device="cuda"),
            kv_len_arr=torch.empty(batch, dtype=torch.int32, device="cuda"),
        )
    return BatchMLAPagedAttentionWrapper(
        torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda"),
        backend="fa2",
        use_cuda_graph=graph,
        **kwargs,
    )


def assert_reference(out, lse, ref, ref_lse):
    # Strict existing MLA FP32-reference tolerance, tighter than the upstream
    # FP8 BF16-backend comparison's 1.5e-2 absolute threshold.
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(lse, ref_lse, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("lengths,heads,page,strided", CASES)
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("basee", [False, True])
def test_sm89_fp8_decode_reference(lengths, heads, page, strided, causal, basee):
    q, kv, csr, dense, ids, offsets = make_inputs(lengths, heads, page, strided)
    scales = (0.3, 1.3) if strided else (0.7, 0.125)
    w = new_wrapper()
    plan(
        w, dense if strided else csr, heads, page, causal, "basee" if basee else "base2"
    )
    out = torch.empty_like(q[0])
    lse = torch.empty(q[0].shape[:2], device="cuda")
    actual, actual_lse = w.run(
        query=tuple(q),
        kv_cache=tuple(kv),
        out=out,
        lse=lse,
        return_lse=True,
        return_lse_base_on_e=basee,
        ckv_scale=scales[0],
        kpe_scale=scales[1],
    )
    assert actual is out and actual_lse is lse
    ref, ref_lse = oracle(q, kv, lengths, ids, offsets, page, scales, basee)
    assert_reference(out, lse, ref, ref_lse)
    # Default return_lse=False must still honor caller-provided LSE storage.
    actual = w.run(
        query=tuple(q),
        kv_cache=tuple(kv),
        out=out,
        lse=lse,
        return_lse_base_on_e=basee,
        ckv_scale=scales[0],
        kpe_scale=scales[1],
    )
    assert actual is out
    assert_reference(out, lse, ref, ref_lse)


@pytest.mark.parametrize("heads", [16, 128])
def test_sm89_fp8_graph_replan(heads):
    q, kv, csr, dense, ids, offsets = make_inputs([33, 65], heads, 32, True)
    w = new_wrapper(2, graph=True)
    plan(w, csr, heads, 32)
    out = torch.empty_like(q[0])
    lse = torch.empty(q[0].shape[:2], device="cuda")

    def run():
        return w.run(
            query=tuple(q),
            kv_cache=tuple(kv),
            out=out,
            lse=lse,
            return_lse=True,
            ckv_scale=0.3,
            kpe_scale=1.3,
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        run()
    for lengths in ([33, 65], [17, 0], [33, 65]):
        # Replan both logical page counts and lengths; physical pool addresses
        # and graph-owned metadata buffers remain stable.
        current_ids, current_offsets = [], [0]
        for i, n in enumerate(lengths):
            current_ids.extend(ids[offsets[i] : offsets[i] + (n + 31) // 32])
            current_offsets.append(len(current_ids))
        metadata = MLAPlanMetadata.csr(
            csr.qo_indptr,
            torch.tensor(current_offsets, dtype=torch.int32),
            torch.tensor(current_ids, device="cuda", dtype=torch.int32),
            torch.tensor(lengths, dtype=torch.int32),
        )
        plan(w, metadata, heads, 32)
        out.fill_(float("nan"))
        lse.fill_(float("nan"))
        g.replay()
        ref, ref_lse = oracle(
            q, kv, lengths, current_ids, current_offsets, 32, (0.3, 1.3), False
        )
        assert_reference(out, lse, ref, ref_lse)


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"q_data_type": torch.float16}, "bfloat16"),
        ({"output_dtype": torch.float16}, "bfloat16"),
        ({"kv_data_type": torch.float8_e5m2}, "float8|FP8|supported"),
        ({"head_dim_ckv": 256}, "SM89.*512.*64"),
        ({"head_dim_kpe": 128}, "SM89.*512.*64"),
    ],
)
def test_sm89_fp8_reject_dtype(overrides, match):
    _, _, csr, _, _, _ = make_inputs([17], 16, 32)
    with pytest.raises((ValueError, RuntimeError), match=match):
        plan(new_wrapper(), csr, 16, 32, **overrides)


def test_sm89_fp8_reject_prefill():
    _, _, csr, _, _, _ = make_inputs([17], 16, 32)
    csr.qo_indptr[1] = 2
    with pytest.raises((ValueError, RuntimeError), match="single-token decode"):
        plan(new_wrapper(), csr, 16, 32)


@pytest.mark.parametrize(
    "scales,match",
    [
        ({}, "required"),
        ({"ckv_scale": 1.0}, "kpe_scale is required"),
        ({"ckv_scale": 0.0, "kpe_scale": 1.0}, "finite positive"),
        ({"ckv_scale": 1.0, "kpe_scale": float("nan")}, "finite positive"),
    ],
)
def test_sm89_fp8_reject_invalid_scales(scales, match):
    q, kv, csr, _, _, _ = make_inputs([17], 16, 32)
    w = new_wrapper()
    plan(w, csr, 16, 32, lse_mode="none")
    with pytest.raises(ValueError, match=match):
        w.run(query=tuple(q), kv_cache=tuple(kv), **scales)


def test_sm89_fp8_default_output_without_lse():
    q, kv, csr, _, ids, offsets = make_inputs([17], 16, 32)
    w = new_wrapper()
    plan(w, csr, 16, 32, lse_mode="none")
    out = w.run(query=tuple(q), kv_cache=tuple(kv), ckv_scale=0.3, kpe_scale=1.3)
    ref, _ = oracle(q, kv, [17], ids, offsets, 32, (0.3, 1.3), False)
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("lengths", [[17], [8193, 8193]])
@pytest.mark.parametrize("causal", [False, True])
def test_sm89_existing_bf16_decode(lengths, causal):
    q, kv, csr, _, ids, offsets = make_inputs(lengths, 16, 32)
    # Existing BF16 FA2 path must remain numerically unchanged.
    kv = [x.to(torch.bfloat16) for x in kv]
    w = new_wrapper()
    plan(w, csr, 16, 32, causal, kv_data_type=torch.bfloat16, scale_mode="default")
    out, lse = w.run(query=tuple(q), kv_cache=tuple(kv), return_lse=True)
    ref, ref_lse = oracle(q, kv, lengths, ids, offsets, 32, (1.0, 1.0), False)
    assert_reference(out, lse, ref, ref_lse)


@pytest.mark.parametrize("first_length", [4097, 32769])
def test_sm89_fp8_graph_replan_across_layout_threshold(monkeypatch, first_length):
    import flashinfer.mla._batch_mla._backends._fa_common as fa_common

    q, kv, csr, _, ids, offsets = make_inputs([32769], 128, 64)
    w = new_wrapper(1, graph=True)

    def set_length(n):
        count = (n + 63) // 64
        metadata = MLAPlanMetadata.csr(
            csr.qo_indptr,
            torch.tensor([0, count], dtype=torch.int32),
            csr.kv_indices[:count],
            torch.tensor([n], dtype=torch.int32),
        )
        plan(w, metadata, 128, 64)

    set_length(first_length)
    frozen_schedule = w._planned_backend._sm89_schedule
    assert frozen_schedule[0] == (32 if first_length == 4097 else 64)
    frozen_module = w._planned_backend._cached_module
    out = torch.empty_like(q[0])
    lse = torch.empty(q[0].shape[:2], device="cuda")

    def run():
        return w.run(
            query=tuple(q),
            kv_cache=tuple(kv),
            out=out,
            lse=lse,
            return_lse=True,
            ckv_scale=0.3,
            kpe_scale=1.3,
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        run()
    other_length = 32769 if first_length == 4097 else 4097
    # A graph already encodes a kernel specialization and launch geometry.
    # Neither changed lengths nor a changed selector recommendation may alter
    # its plan. Mock only the recommendation, not the schedule-freeze logic.
    monkeypatch.setattr(
        fa_common,
        "_sm89_mla_q_tile",
        lambda *args: 64 if frozen_schedule[0] == 32 else 32,
    )
    for n in (first_length, other_length, first_length):
        set_length(n)
        assert w._planned_backend._sm89_schedule == frozen_schedule
        assert w._planned_backend._cached_module is frozen_module
        previous_backend = w._planned_backend
        count = (n + 63) // 64
        invalid = MLAPlanMetadata.csr(
            torch.tensor([0, 2], dtype=torch.int32),
            torch.tensor([0, count], dtype=torch.int32),
            csr.kv_indices[:count],
            torch.tensor([n], dtype=torch.int32),
        )
        with pytest.raises((ValueError, RuntimeError), match="single-token decode"):
            plan(w, invalid, 128, 64)
        assert w._planned_backend is previous_backend
        assert w._planned_backend._sm89_schedule == frozen_schedule
        if n == other_length:
            valid_metadata = MLAPlanMetadata.csr(
                csr.qo_indptr,
                torch.tensor([0, count], dtype=torch.int32),
                csr.kv_indices[:count],
                torch.tensor([n], dtype=torch.int32),
            )
            plan_info = tuple(w._plan_info)
            buffers = {
                name: getattr(w, name).clone()
                for name in (
                    "_qo_indptr_buf",
                    "_kv_indptr_buf",
                    "_kv_indices_buf",
                    "_kv_len_arr_buf",
                )
            }
            # Scalars are arguments baked into a captured kernel too. Reject
            # structural changes before touching any published metadata/state.
            for field, value in (
                ("num_heads", 64),
                ("causal", True),
                ("page_size", 32),
                ("sm_scale", 0.125),
                ("q_data_type", torch.float16),
                ("kv_data_type", torch.bfloat16),
                ("output_dtype", torch.float16),
                ("use_profiler", True),
            ):
                with pytest.raises(ValueError, match="changed:.*" + field):
                    plan(w, valid_metadata, 128, 64, **{field: value})
                assert w._planned_backend is previous_backend
                assert tuple(w._plan_info) == plan_info
                for name, saved in buffers.items():
                    torch.testing.assert_close(getattr(w, name), saved, rtol=0, atol=0)
        out.fill_(float("nan"))
        lse.fill_(float("nan"))
        g.replay()
        ref, ref_lse = oracle(q, kv, [n], ids, offsets, 64, (0.3, 1.3), False)
        assert_reference(out, lse, ref, ref_lse)


@pytest.mark.parametrize("lengths,heads", [([17, 65], 16), ([8193, 8193], 128)])
def test_sm89_fp8_physical_page_group_scales(lengths, heads):
    q, kv, csr, _, ids, offsets = make_inputs(lengths, heads, 32, True)
    pool, page = kv[0].shape[:2]
    # Scales are indexed by PHYSICAL page/token, then 128-wide CKV group.
    row_scales = torch.linspace(0.5, 1.5, pool * page, device="cuda").reshape(
        pool, page, 1
    )
    group_scales = torch.tensor([0.125, 0.3, 0.7, 1.3], device="cuda")
    scales = (row_scales * group_scales).contiguous()
    w = new_wrapper()
    plan(w, csr, heads, page)
    out, lse = w.run(
        query=tuple(q),
        kv_cache=tuple(kv),
        return_lse=True,
        ckv_scale_arr=scales,
        kpe_scale=0.3,
    )
    # Group scales retain FP32 precision until the product is rounded to BF16;
    # scalar scales instead are themselves rounded to BF16 before multiplying.
    decoded_c = (
        (kv[0].float().reshape(pool, page, 4, 128) * scales[..., None])
        .reshape(pool, page, 512)
        .to(torch.bfloat16)
    )
    decoded_p = kv[1].to(torch.bfloat16) * torch.tensor(
        0.3, device="cuda", dtype=torch.bfloat16
    )
    ref, ref_lse = oracle(
        q, [decoded_c, decoded_p], lengths, ids, offsets, page, (1.0, 1.0), False
    )
    assert_reference(out, lse, ref, ref_lse)


@pytest.mark.parametrize("length,tile", [(4097, 32), (8193, 64)])
@pytest.mark.parametrize("offset", [0, 1])
def test_sm89_fp8_workspace_boundary(length, tile, offset):
    """Execute at exact capacity with canaries, reject a one-byte deficit."""
    q, kv, csr, _, ids, offsets = make_inputs([length], 128, 32)
    # Native MLA allocates two rows per launched CTA for partial O and LSE.
    # H128 uses two CTAs per cluster for both layouts.
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    rows = 2 * (sms // 2) * 2 * tile
    payload = rows * (512 * 2 + 4)
    backing = torch.full((payload + 128,), 0xA5, dtype=torch.uint8, device="cuda")
    start = 32 + offset
    padding = (-(backing.data_ptr() + start)) % 16
    size = payload + padding
    workspace = backing[start : start + size]
    short = BatchMLAPagedAttentionWrapper(workspace[:-1], backend="fa2")
    with pytest.raises((ValueError, RuntimeError), match="float workspace bytes"):
        plan(short, csr, 128, 32)
    assert bool((backing == 0xA5).all())

    w = BatchMLAPagedAttentionWrapper(workspace, backend="fa2")
    plan(w, csr, 128, 32)
    assert w._planned_backend._sm89_schedule[0] == tile
    out, lse = w.run(
        query=tuple(q),
        kv_cache=tuple(kv),
        return_lse=True,
        ckv_scale=0.3,
        kpe_scale=1.3,
    )
    ref, ref_lse = oracle(q, kv, [length], ids, offsets, 32, (0.3, 1.3), False)
    assert_reference(out, lse, ref, ref_lse)
    assert bool((backing[:start] == 0xA5).all())
    assert bool((backing[start + size :] == 0xA5).all())

    if tile == 32:
        # A larger valid request selects Q64 and needs twice the partial storage.
        # Rejection must leave the successful Q32 plan usable.
        _, _, larger, _, _, _ = make_inputs([8193], 128, 32)
        previous = w._planned_backend
        with pytest.raises((ValueError, RuntimeError), match="float workspace bytes"):
            plan(w, larger, 128, 32)
        assert w._planned_backend is previous
        out, lse = w.run(
            query=tuple(q),
            kv_cache=tuple(kv),
            return_lse=True,
            ckv_scale=0.3,
            kpe_scale=1.3,
        )
        assert_reference(out, lse, ref, ref_lse)
        assert bool((backing[:start] == 0xA5).all())
        assert bool((backing[start + size :] == 0xA5).all())


@pytest.mark.parametrize("length", [4097, 8193])
def test_sm89_fp8_auto_backend(length):
    q, kv, csr, _, ids, offsets = make_inputs([length], 128, 32)
    w = BatchMLAPagedAttentionWrapper(
        torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda"),
        backend="auto",
    )
    plan(w, csr, 128, 32)
    assert w._planned_backend._backend == "fa2"
    out, lse = w.run(
        query=tuple(q),
        kv_cache=tuple(kv),
        return_lse=True,
        ckv_scale=0.3,
        kpe_scale=1.3,
    )
    ref, ref_lse = oracle(q, kv, [length], ids, offsets, 32, (0.3, 1.3), False)
    assert_reference(out, lse, ref, ref_lse)


@pytest.mark.parametrize("heads,length,tile", [(16, 8193, 32), (128, 8193, 64)])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("boundary", ["last", "tile", "split"])
def test_sm89_fp8_dominant_boundary_tokens(heads, length, tile, causal, boundary):
    """A missed key must change O visibly, even with a strict random-test tolerance."""
    page = 64
    q, caches, csr, _, ids, offsets = make_inputs([length], heads, page)
    w = new_wrapper()
    plan(w, csr, heads, page, causal)
    assert w._planned_backend._sm89_schedule[0] == tile
    if boundary == "last":
        positions = (length - 2, length - 1)
    elif boundary == "tile":
        kv_tile = 32 if tile == 32 else 16
        positions = (kv_tile - 1, kv_tile)
    else:
        # Read actual split starts, rather than assume a scheduler heuristic.
        # MLAPlanInfo stores byte offsets; work_indptr has one entry per cluster.
        info = w._planned_backend._plan_info
        words = w._planned_backend._int_workspace_buffer.view(torch.int32).cpu()
        num_clusters = info[1]
        work_indptr_offset = info[15] // 4
        kv_start_offset = info[13] // 4
        num_work = int(words[work_indptr_offset + num_clusters])
        starts = words[kv_start_offset : kv_start_offset + num_work]
        interior = starts[(starts > 0) & (starts < length)]
        assert interior.numel() > 0, "test must exercise a split-KV boundary"
        split = int(interior.min())
        positions = (split - 1, split)

    # Use positional keys to choose two different tokens for alternating heads.
    # CKV carries distinct values in each output band and within each band.
    # All values are exactly representable in E4M3 and BF16.
    c = torch.zeros_like(caches[0], dtype=torch.float32)
    p = torch.zeros_like(caches[1], dtype=torch.float32)
    values = 1 + (torch.arange(512, device="cuda").float() // 64) * 0.25
    values += (torch.arange(512, device="cuda") % 4).float() * 0.25
    for group, token in enumerate(positions):
        physical = ids[token // page]
        c[physical, token % page] = values * (1 if group == 0 else -1)
        p[physical, token % page, group] = 64
    # Invalid pages and the final page tail remain poisoned.
    for index, original in enumerate(caches):
        dest = c if index == 0 else p
        dest.masked_fill_(torch.isnan(original.float()), float("nan"))
    caches = [c.to(torch.float8_e4m3fn), p.to(torch.float8_e4m3fn)]
    q[0].zero_()
    q[1].zero_()
    q[1][
        0, torch.arange(heads, device="cuda"), torch.arange(heads, device="cuda") % 2
    ] = 8
    out, lse = w.run(
        query=tuple(q),
        kv_cache=tuple(caches),
        return_lse=True,
        ckv_scale=1.0,
        kpe_scale=1.0,
    )
    expected, expected_lse = oracle(
        q, caches, [length], ids, offsets, page, (1.0, 1.0), False
    )
    assert_reference(out, lse, expected, expected_lse)
    signs = torch.where(torch.arange(heads, device="cuda") % 2 == 0, 1, -1)
    torch.testing.assert_close(
        out[0].float(), signs[:, None] * values, rtol=1e-3, atol=1e-3
    )
    # Demonstrate sensitivity: removing each selected token changes its heads'
    # answer by orders of magnitude more than the production tolerance.
    for group, token in enumerate(positions):
        modified = p.clone()
        modified[ids[token // page], token % page, group] = 0
        missing, _ = oracle(
            q,
            [caches[0], modified.to(torch.float8_e4m3fn)],
            [length],
            ids,
            offsets,
            page,
            (1.0, 1.0),
            False,
        )
        assert (
            missing[:, group::2].float() - expected[:, group::2].float()
        ).abs().max() > 0.5


@pytest.mark.parametrize("length,tile", [(4097, 32), (8193, 64)])
def test_sm89_fp8_causal_partial_head_cluster_last_key(length, tile):
    # H192 spans multiple CTA clusters. floor(cluster_end / H) would drop
    # the last visible key for the first cluster, whereas ceil keeps it.
    test_sm89_fp8_dominant_boundary_tokens(192, length, tile, True, "last")


def test_sm89_eager_plan_selects_schedule_once(monkeypatch):
    from flashinfer.mla._batch_mla._backends import _fa_common

    calls = []
    select = _fa_common._sm89_mla_q_tile

    def counted(*args, **kwargs):
        calls.append(1)
        return select(*args, **kwargs)

    monkeypatch.setattr(_fa_common, "_sm89_mla_q_tile", counted)
    _, _, csr, _, _, _ = make_inputs([257], 64, 32)
    plan(new_wrapper(), csr, 64, 32)
    assert len(calls) == 1
