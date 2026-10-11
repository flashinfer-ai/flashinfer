"""Dense prepared API: PyTorch-oracle numerics, exact schedule metadata, the -inf
mask, non-default-stream dependencies and changed-input graph replay on
SM100a/SM103a.

The reference is the DeepGEMM lightning-indexer specification evaluated in
PyTorch (dequantized operands, weighted ReLU over heads, per-row windows) and a
scalar specification of the DeepGEMM schedule metadata. DeepGEMM itself is an
additional oracle when its build exposes the dense MQA logits API; production
imports and launches never use it.
"""

import bisect

import pytest
import torch

from flashinfer.dense_mqa import prepare_dense_mqa_logits
from flashinfer.experimental.deepgemm_dense_mqa import dense_mqa as _runtime

_FP4_LUT = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)

# The twenty scheduled model rows plus rows outside the old route table: KV
# lengths never catalogued (2048, 8192, 65536, 1048576 for one query) and query
# counts that are not 1, 16 or 128 (partial four-query blocks, 33 and 132
# queries, and 2052/2053 queries on the unbounded metadata tier).
CASES = [
    (precision, queries, keys)
    for precision in ("fp4", "fp8")
    for queries, keys in (
        (1, 4096),
        (1, 32768),
        (1, 131072),
        (16, 4096),
        (16, 32768),
        (16, 131072),
        (128, 4096),
        (128, 32768),
        (128, 131072),
        (16, 1048576),
    )
] + [
    ("fp4", 7, 2048),
    ("fp4", 33, 8192),
    ("fp4", 130, 65536),
    ("fp8", 3, 8192),
    ("fp8", 16, 2048),
    ("fp8", 128, 2048),
    ("fp8", 1, 1048576),
    ("fp8", 130, 8192),
    ("fp4", 2052, 2048),
    ("fp8", 2052, 2048),
    ("fp8", 2053, 2048),
    ("fp8", 132, 2048),
    ("fp8", 33, 8192),
]


def _skip_unless_exported():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    try:
        _runtime.device_arch(torch.device("cuda"))
    except RuntimeError as error:
        pytest.skip(str(error))


def _inputs(precision, queries, keys):
    torch.manual_seed(101)
    if precision == "fp4":
        q = torch.randint(0, 256, (queries, 32, 64), device="cuda", dtype=torch.uint8)
        kv = torch.randint(0, 256, (keys, 64), device="cuda", dtype=torch.uint8)
        qs = torch.randint(124, 131, (queries, 32, 4), device="cuda", dtype=torch.uint8)
        ks = torch.randint(124, 131, (keys, 4), device="cuda", dtype=torch.uint8)
        rows = queries
    else:
        rows = max(4, queries)
        q = torch.randn(rows, 32, 128, device="cuda").to(torch.float8_e4m3fn)
        kv = torch.randn(keys, 128, device="cuda").to(torch.float8_e4m3fn)
        qs = None
        ks = torch.rand(keys, device="cuda") + 0.5
    weights = torch.randn(rows, 32, device="cuda")
    starts = torch.zeros(queries, device="cuda", dtype=torch.int32)
    ends = torch.full_like(starts, keys)
    ends[1::3] = 17
    ends[2::3] = 0
    if queries > 3:
        starts[3::4] = 6
    # Window contract: 0 <= start <= end <= K.
    starts = torch.minimum(starts, ends)
    return q, kv, qs, ks, weights, starts, ends


def _dequantize_fp4(packed, scales):
    """Packed E2M1 bytes [rows, 64] and UE8M0 scales [rows, 4] -> FP32 [rows, 128]."""
    lut = torch.tensor(_FP4_LUT, dtype=torch.float32, device=packed.device)
    values = torch.empty(
        packed.shape[0], 128, dtype=torch.float32, device=packed.device
    )
    values[:, 0::2] = lut[(packed & 0xF).long()]
    values[:, 1::2] = lut[((packed >> 4) & 0xF).long()]
    exponent = torch.pow(2.0, scales.float() - 127.0).repeat_interleave(32, dim=1)
    return values * exponent


def _reference(precision, q, kv, qs, ks, weights, starts, ends, queries, keys):
    """logits[q, k] = sum_h relu(Q[q,h] . KV[k]) * w[q,h] (* kv_scale[k] for FP8) in the window, -inf outside."""
    if precision == "fp4":
        q_f32 = _dequantize_fp4(q.reshape(-1, 64), qs.reshape(-1, 4)).view(
            queries, 32, 128
        )
        kv_f32 = _dequantize_fp4(kv, ks)
        kv_scale = None
    else:
        q_f32 = q[:queries].float()
        kv_f32 = kv.float()
        kv_scale = ks
    logits = torch.zeros(queries, keys, dtype=torch.float32, device=q.device)
    for head in range(32):
        logits += (
            torch.relu(q_f32[:, head] @ kv_f32.T) * weights[:queries, head : head + 1]
        )
    if kv_scale is not None:
        logits *= kv_scale[None, :]
    position = torch.arange(keys, device=q.device)[None, :]
    inside = (position >= starts[:, None]) & (position < ends[:, None])
    return logits.masked_fill(~inside, float("-inf"))


def _metadata_reference(starts, ends, kv, sms):
    """Scalar specification of the DeepGEMM cost-partition metadata."""
    spans, work_prefix, cost_prefix = [], [], []
    work = cost = 0
    for q in range(0, len(starts), 4):
        base = min(min(v, kv) for v in starts[q : q + 4]) // 4 * 4
        end = max(min(v, kv) for v in ends[q : q + 4])
        splits = (end - base + 255) // 256
        spans.extend((base, splits))
        work += splits
        cost += splits + int(splits > 0)
        work_prefix.append(work)
        cost_prefix.append(cost)

    def locate(sm):
        target = sm * (cost // sms) + min(sm, cost % sms)
        if target == cost:
            return len(work_prefix), 0, work
        block = bisect.bisect_right(cost_prefix, target)
        before_work = work_prefix[block - 1] if block else 0
        before_cost = cost_prefix[block - 1] if block else 0
        split = min(
            max(target - before_cost, 1) - 1, work_prefix[block] - before_work - 1
        )
        return block, split, before_work + split

    boundaries = [locate(sm) for sm in range(sms + 1)]
    header = [v for q, split, _ in boundaries[:-1] for v in (q, split)]
    header += [boundaries[sm + 1][2] - boundaries[sm][2] for sm in range(sms)]
    return header, spans


def _check(plan, precision, expected, starts, ends, keys, sms):
    actual = plan.logical_output
    finite = torch.isfinite(expected)
    assert torch.equal(torch.isfinite(actual), finite)
    assert bool(torch.isneginf(actual[~finite]).all())
    assert bool(torch.isneginf(plan.output[:, keys:]).all())
    torch.testing.assert_close(
        actual[finite],
        expected[finite],
        atol=1.0 if precision == "fp4" else 0.1,
        rtol=0.1,
    )
    if precision == "fp4":
        left, right = actual[finite].double(), expected[finite].double()
        denominator = (left.square() + right.square()).sum()
        distance = (
            0.0
            if denominator == 0
            else float(1 - 2 * (left * right).sum() / denominator)
        )
        assert distance < 5e-6
    header, spans = _metadata_reference(starts.tolist(), ends.tolist(), keys, sms)
    span_offset = (3 * sms + 1) // 2 * 2
    metadata = plan.metadata.tolist()
    assert metadata[: 3 * sms] == header
    assert metadata[span_offset:] == spans


def _deep_gemm_check(
    plan, precision, q, kv, qs, ks, weights, starts, ends, queries, keys, sms
):
    try:
        import deep_gemm
    except ImportError:
        return
    # DeepGEMM builds differ in the dense MQA API they expose; the oracle
    # runs only where the dense metadata and logits entry points exist.
    if not all(
        hasattr(deep_gemm, name)
        for name in (
            "get_num_sms",
            "set_num_sms",
            "get_mqa_logits_metadata",
            "fp8_fp4_mqa_logits",
        )
    ):
        return
    if precision == "fp4":
        native_q = (q.view(torch.int8), qs.view(torch.int32).view(queries, 32))
        native_kv = (kv.view(torch.int8), ks.view(torch.int32).view(keys))
    else:
        native_q, native_kv = (q[:queries], None), (kv, ks)
    old_sms = deep_gemm.get_num_sms()
    deep_gemm.set_num_sms(sms)
    try:
        meta = deep_gemm.get_mqa_logits_metadata(starts, ends, keys, 32)
        expected = deep_gemm.fp8_fp4_mqa_logits(
            q=native_q,
            kv=native_kv,
            weights=weights[:queries],
            cu_seq_len_k_start=starts,
            cu_seq_len_k_end=ends,
            clean_logits=True,
            max_seqlen_k=0,
            logits_dtype=torch.float32,
            schedule_meta=meta,
        )
    finally:
        deep_gemm.set_num_sms(old_sms)
    torch.cuda.synchronize()
    finite = torch.isfinite(expected)
    actual = plan.logical_output
    assert torch.equal(torch.isfinite(actual), finite)
    torch.testing.assert_close(
        actual[finite],
        expected[finite],
        atol=1.0 if precision == "fp4" else 0.1,
        rtol=0.1,
    )
    span = (3 * sms + 1) // 2 * 2
    assert torch.equal(plan.metadata[: 3 * sms], meta[: 3 * sms])
    assert torch.equal(plan.metadata[span:], meta[span:])


@pytest.mark.parametrize("precision,queries,keys", CASES)
def test_dense_mqa_logits(precision, queries, keys):
    _skip_unless_exported()
    q, kv, qs, ks, weights, starts, ends = _inputs(precision, queries, keys)
    plan = prepare_dense_mqa_logits(
        precision, q, kv, weights, starts, ends, q_scales=qs, kv_scales=ks
    )
    assert plan.route_name == _runtime.route_name(precision, queries, keys)
    sms = plan.num_sms
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        # A device-side input dependency must reach the current-stream launch.
        weights.mul_(0.5)
        plan.run()
    stream.synchronize()
    first = plan.output.clone()
    expected = _reference(
        precision, q, kv, qs, ks, weights, starts, ends, queries, keys
    )
    _check(plan, precision, expected, starts, ends, keys, sms)
    _deep_gemm_check(
        plan, precision, q, kv, qs, ks, weights, starts, ends, queries, keys, sms
    )
    with torch.cuda.stream(stream):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            plan.run()
    for changed in (False, True):
        with torch.cuda.stream(stream):
            if changed:
                weights.neg_()
                ends.copy_(keys - ends)
                starts.copy_(torch.minimum(starts, ends))
            plan.output.fill_(float("nan"))
            plan.metadata.fill_(0x55555555)
            graph.replay()
        stream.synchronize()
        if changed:
            expected = _reference(
                precision, q, kv, qs, ks, weights, starts, ends, queries, keys
            )
        _check(plan, precision, expected, starts, ends, keys, sms)
        if not changed:
            assert torch.equal(plan.output, first)


@pytest.mark.parametrize(
    "precision,queries,keys,route",
    [
        ("fp4", 1, 4096, "fp4:q1"),
        ("fp4", 16, 1048576, "fp4:le16"),
        ("fp4", 7, 2048, "fp4:le16"),
        ("fp4", 17, 2048, "fp4:le128"),
        ("fp4", 130, 65536, "fp4:le2048"),
        ("fp4", 2049, 4096, "fp4:any"),
        ("fp8", 1, 4096, "fp8:q1:short"),
        ("fp8", 1, 131072, "fp8:q1:short"),
        ("fp8", 1, 131328, "fp8:q1"),
        ("fp8", 128, 4096, "fp8:q128:short"),
        ("fp8", 128, 4352, "fp8:full:le128"),
        ("fp8", 16, 2048, "fp8:full:le16"),
        ("fp8", 3, 8192, "fp8:partial:le16"),
        ("fp8", 130, 8192, "fp8:partial:le2048"),
    ],
)
def test_dense_mqa_route_selection(precision, queries, keys, route):
    assert _runtime.route_name(precision, queries, keys) == route


@pytest.mark.parametrize("sm_count", [132, 148, 152])
def test_dense_mqa_sm_count_definition(sm_count):
    """The FP8 indexer programs are built per launch grid: a CTA-budget override
    compiles the fused single-query program with that SM count defined on the
    compile line (its own JIT module) and the result is exact against the
    reference and the metadata rule -- including a count the device does not
    have (above its SM count the extra CTAs run as a second wave)."""
    _skip_unless_exported()
    arch, _device_sms = _runtime.device_facts(torch.cuda.current_device())
    q, kv, qs, ks, weights, starts, ends = _inputs("fp8", 1, 4096)
    plan = _runtime.DenseMqaPlan(
        "fp8", q, kv, weights, starts, ends, kv_scales=ks, sm_count=sm_count
    )
    assert plan.route_name == "fp8:q1:short"
    assert plan.num_sms == sm_count
    program = dict(plan.route["stages"])["logits"]
    assert plan.program_names == [program]
    record = _runtime._catalog()["programs"][program]
    assert record["definitions"] == ["SM_COUNT"]
    assert _runtime.program_definitions(record, sm_count) == {"SM_COUNT": sm_count}
    assert _runtime.program_spec(arch, program, sm_count).name.endswith(
        f"_sm_count{sm_count}"
    )
    plan.run()
    torch.cuda.synchronize()
    expected = _reference("fp8", q, kv, qs, ks, weights, starts, ends, 1, 4096)
    _check(plan, "fp8", expected, starts, ends, 4096, sm_count)


def test_dense_mqa_rejects_unsupported_shapes():
    _skip_unless_exported()
    q, kv, qs, ks, weights, starts, ends = _inputs("fp8", 4, 4096)
    with pytest.raises(ValueError):
        prepare_dense_mqa_logits(
            "fp8", q, kv[:4000], weights, starts, ends, kv_scales=ks[:4000]
        )
    with pytest.raises(ValueError):
        prepare_dense_mqa_logits("fp8", q, kv, weights, starts, ends[:3], kv_scales=ks)
    with pytest.raises(ValueError):
        prepare_dense_mqa_logits("bf16", q, kv, weights, starts, ends, kv_scales=ks)


# ---------------------------------------------------------------------------
# DeepGEMM-signature one-shot entry (fp8_mqa_logits) and the head-count table
# ---------------------------------------------------------------------------


def test_dense_route_table_is_catalog_driven():
    """Host-only: the 32-head names are unchanged, other head counts carry the
    ``h<H>`` infix with tiers named by their token ceiling, and availability
    (including the per-head KV alignment) is the catalog's word."""
    assert _runtime.route_name("fp8", 16, 4096) == "fp8:full:le16"
    assert _runtime.route_name("fp8", 1, 4096) == "fp8:q1:short"
    assert _runtime.route_name("fp8", 130, 8192) == "fp8:partial:le2048"
    assert _runtime.dense_route_available(32, 16, 4096)
    assert not _runtime.dense_route_available(32, 16, 4100)  # 32 heads: K % 256
    record = _runtime.route_record("fp8", 16, 4096)
    assert record["num_heads"] == 32 and record["block_q"] == 4
    assert record["clean_logits"] == "fused" and record["kv_alignment"] == 256
    if 64 in _runtime.heads():
        assert _runtime.block_q(64) == 2
        assert _runtime.route_name("fp8", 16, 4096, 64) == "fp8:h64:full:le64"
        assert _runtime.route_name("fp8", 3, 4100, 64) == "fp8:h64:partial:le8"
        assert _runtime.route_name("fp8", 1, 300, 64) == "fp8:h64:q1"
        assert _runtime.route_name("fp8", 1024, 5124, 64) == "fp8:h64:full:le1024"
        assert _runtime.route_name("fp8", 37, 37, 64) == "fp8:h64:partial:short"
        assert _runtime.route_name("fp8", 16, 16, 64) == "fp8:h64:full:short"
        assert _runtime.route_name("fp8", 2, 2, 64) == "fp8:h64:full:le8"
        assert _runtime.route_name("fp8", 3, 257, 64) == "fp8:h64:partial:le8"
        # W-dense contract: single-stage gridDim-strided programs, no metadata stage, no query bound.
        assert not _runtime.schedules_metadata(64) and _runtime.max_queries(64) is None
        # Availability of a 64-head point is decided per architecture (policy.dense_admission).
        for arch in sorted(_runtime._catalog()["arches"]):
            admitted = set(_runtime.h64_admission(arch)["admitted_routes"])
            assert _runtime.dense_route_available(64, 37, 4137, arch=arch) == (
                "fp8:h64:partial:le64" in admitted
            )
            assert _runtime.dense_route_available(64, 100_000, 7, arch=arch) == (
                "fp8:h64:full:any" in admitted
            )
        h64_route = next(
            r for r in _runtime._catalog()["routes"] if r.startswith("fp8:h64:")
        )
        record = _runtime._catalog()["routes"][h64_route]
        assert record["clean_logits"] == "raw" and record["kv_alignment"] == 1
        assert [stage for stage, _p in record["stages"]] == ["logits"]
        # Only tiers admitted on at least one architecture have a record (and programs); a route withheld
        # everywhere is absent from the table. Every catalogued 64-head route is a single logits stage whose
        # program is a catalogued program (program ids are the generated-unit names, not template names).
        programs = set(_runtime._catalog()["programs"])
        for r in _runtime._catalog()["routes"]:
            if r.startswith("fp8:h64:"):
                names = _runtime.program_names(r)
                assert len(names) == 1 and set(names) <= programs, (r, names)
    else:
        assert not _runtime.dense_route_available(64, 16, 4096)
    assert (
        _runtime.schedules_metadata(32)
        and _runtime.max_queries(32)
        == 4 * _runtime._catalog()["policy"]["max_q_blocks"]
    )
    assert not _runtime.dense_route_available(16, 16, 4096)
    assert not _runtime.dense_route_available(32, 0, 4096)


@pytest.mark.parametrize(
    "queries,keys,heads",
    [
        (16, 4096, 32),
        (3, 2048, 32),
        (1, 4096, 32),
        (128, 4096, 32),
        (37, 4100, 64),
        (1, 8192, 64),
    ],
)
def test_fp8_mqa_logits_one_shot(queries, keys, heads):
    """``fp8_mqa_logits(q, (kv, scales), weights, ks, ke)``: the plan's result on
    an exact ``[Q]`` slice (rows below the query block padded by the entry), the
    same bits whether ``clean_logits`` is passed or not, and ``max_seqlen_k``
    rejected."""
    _skip_unless_exported()
    arch = _runtime.device_arch(torch.device("cuda"))
    if not _runtime.dense_route_available(heads, queries, keys, arch=arch):
        pytest.skip(
            f"catalog has no admitted route for H={heads}, Q={queries}, K={keys} on {arch}"
        )
    from flashinfer.dense_mqa import fp8_mqa_logits

    torch.manual_seed(3)
    q = torch.randn(queries, heads, 128, device="cuda").to(torch.float8_e4m3fn)
    kv = torch.randn(keys, 128, device="cuda").to(torch.float8_e4m3fn)
    scales = torch.rand(keys, device="cuda") + 0.5
    weights = torch.randn(queries, heads, device="cuda")
    ks = torch.zeros(queries, device="cuda", dtype=torch.int32)
    ke = torch.full_like(ks, keys)
    ke[1::3] = 17
    ks = torch.minimum(ks, ke)
    record = _runtime.route_record("fp8", queries, keys, heads)
    out = fp8_mqa_logits(q, (kv, scales), weights, ks, ke, clean_logits=False)
    if record["clean_logits"] == "fused":
        out_clean = fp8_mqa_logits(q, (kv, scales), weights, ks, ke, clean_logits=True)
    else:
        with pytest.raises(ValueError):
            fp8_mqa_logits(q, (kv, scales), weights, ks, ke, clean_logits=True)
        out_clean = out
    torch.cuda.synchronize()
    assert tuple(out.shape) == (queries, keys)
    assert out.dtype == torch.float32 and out.stride(1) == 1 and out.stride(0) % 4 == 0
    assert torch.equal(out, out_clean)
    if heads == 32:
        expected = _reference(
            "fp8", q, kv, None, scales, weights, ks, ke, queries, keys
        )
    else:
        q_f32, kv_f32 = q.float(), kv.float()
        expected = torch.zeros(queries, keys, device="cuda")
        for head in range(heads):
            expected += (
                torch.relu(q_f32[:, head] @ kv_f32.T) * weights[:, head : head + 1]
            )
        expected *= scales[None, :]
        position = torch.arange(keys, device="cuda")[None, :]
        expected = expected.masked_fill(
            ~((position >= ks[:, None]) & (position < ke[:, None])), float("-inf")
        )
    finite = torch.isfinite(expected)
    if record["clean_logits"] == "fused":
        assert torch.equal(torch.isfinite(out), finite)
    else:
        assert torch.isfinite(
            out[finite]
        ).all()  # raw stores: cells outside the windows are unspecified
    torch.testing.assert_close(out[finite], expected[finite], atol=0.1, rtol=0.1)
    with pytest.raises(ValueError):
        fp8_mqa_logits(q, (kv, scales), weights, ks, ke, max_seqlen_k=keys)
    with pytest.raises(ValueError):
        fp8_mqa_logits(q, kv, weights, ks, ke)  # kv must be the (values, scales) pair
    try:
        import deep_gemm
    except ImportError:
        return
    if not hasattr(deep_gemm, "fp8_mqa_logits") or heads != 32:
        return
    native = deep_gemm.fp8_mqa_logits(
        q, (kv, scales), weights, ks, ke, clean_logits=False
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(out[finite], native[finite], atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("queries,keys,heads", [(16, 4096, 32), (37, 4100, 64)])
def test_fp8_mqa_logits_graph_replay(queries, keys, heads):
    """The one-shot entry captured in a CUDA graph launches on the capture
    stream: a replay after the captured output is poisoned reproduces the eager
    bits (an empty capture leaves the poison and fails), and the logits are
    non-trivial."""
    _skip_unless_exported()
    arch = _runtime.device_arch(torch.device("cuda"))
    if not _runtime.dense_route_available(heads, queries, keys, arch=arch):
        pytest.skip(
            f"catalog has no admitted route for H={heads}, Q={queries}, K={keys} on {arch}"
        )
    from flashinfer.dense_mqa import fp8_mqa_logits

    torch.manual_seed(5)
    q = torch.randn(queries, heads, 128, device="cuda").to(torch.float8_e4m3fn)
    kv = torch.randn(keys, 128, device="cuda").to(torch.float8_e4m3fn)
    scales = torch.rand(keys, device="cuda") + 0.5
    weights = torch.randn(queries, heads, device="cuda")
    ks = torch.zeros(queries, device="cuda", dtype=torch.int32)
    ke = torch.full_like(ks, keys)
    eager = fp8_mqa_logits(
        q, (kv, scales), weights, ks, ke
    )  # eager warm-up builds the module
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = fp8_mqa_logits(q, (kv, scales), weights, ks, ke)
    with torch.cuda.stream(stream):
        captured.fill_(float("nan"))
        graph.replay()
    stream.synchronize()
    assert torch.equal(captured, eager), (
        "graph replay does not reproduce the eager logits"
    )
    assert torch.isfinite(captured).all() and bool((captured != 0).any())
    with torch.cuda.stream(stream):
        weights.neg_()
        captured.fill_(float("nan"))
        graph.replay()
        changed = fp8_mqa_logits(q, (kv, scales), weights, ks, ke)
    stream.synchronize()
    assert torch.equal(captured, changed) and not torch.equal(captured, eager)


def test_catalog_programs_are_delivered_and_distinct():
    """Host-only: every program a dense or paged route names is catalogued, its generated units exist
    under FlashInfer's ``csrc`` (a module: its own ``<name>_kernel.cu`` + ``<name>_binding.cu``; a
    sequence: >= 2 module kernels + its own binding), no two generated kernel units are byte-identical
    and no generated unit is unreferenced -- a route table pointing at a program hash whose units were
    regenerated under a new hash fails here, not at JIT build time."""
    import hashlib

    from flashinfer.jit import env

    csrc = env.FLASHINFER_CSRC_DIR
    catalog = _runtime._catalog()
    programs = catalog["programs"]
    generated = "csrc/experimental/deepgemm_dense_mqa/generated"
    for table in ("routes", "paged_routes"):
        for route, record in catalog.get(table, {}).items():
            for _stage, program in record["stages"]:
                assert program in programs, (table, route, program)
            assert not record.get("sequence") or record["sequence"] in programs, (
                table,
                route,
            )
    referenced, kernels = set(), {}
    for name, record in programs.items():
        for source in record["sources"]:
            assert source.startswith("csrc/"), (name, source)
            assert (csrc / source.removeprefix("csrc/")).is_file(), (name, source)
            referenced.add(source)
        if record.get(
            "standalone", True
        ):  # an omitted-binding module carries no closure digest
            for arch in record["arches"]:
                assert arch in (record.get("closure_sha256") or {}), (name, arch)
        if record["kind"] == "module":
            expected = {f"{generated}/{name}_kernel.cu"}
            if record["standalone"]:
                expected.add(f"{generated}/{name}_binding.cu")
            assert set(record["sources"]) == expected, (name, record["sources"])
            kernels[f"{generated}/{name}_kernel.cu"] = name
        else:
            assert record["kind"] == "sequence", name
            devices = [s for s in record["sources"] if s.endswith("_kernel.cu")]
            assert (
                len(devices) >= 2
                and f"{generated}/{name}_binding.cu" in record["sources"]
            ), name
            for device in devices:
                owner = device.rsplit("/", 1)[1].removesuffix("_kernel.cu")
                assert programs.get(owner, {}).get("kind") == "module", (name, device)
    digests = {}
    for source, name in sorted(kernels.items()):
        digest = hashlib.sha256(
            (csrc / source.removeprefix("csrc/")).read_bytes()
        ).hexdigest()
        assert digest not in digests, (
            f"{name} kernel unit is byte-identical to {digests[digest]}'s"
        )
        digests[digest] = name
    generated_dir = csrc / generated.removeprefix("csrc/")
    orphans = sorted(
        p.name
        for p in generated_dir.iterdir()
        if p.suffix == ".cu" and f"{generated}/{p.name}" not in referenced
    )
    assert not orphans, f"generated units not referenced by the catalog: {orphans}"


def test_h64_admission_matches_the_catalog_table():
    """Host-only: the shipped allow-list of the 64-head family is decided per architecture
    (``policy.dense_admission``): on every catalogued arch the admitted and withheld tier names are disjoint
    and together name every 64-head tier (none when the family is not exported), a reason accompanies a
    non-empty withheld list, every admitted route has a catalog record, and ``dense_route_available`` on that
    arch is True exactly for the query counts of admitted tiers. Without ``arch`` the architectures must
    agree, otherwise the call raises (no silent admit)."""
    catalog = _runtime._catalog()
    arches = sorted(catalog["arches"])
    tiers = {
        "fp8:h64:q1",
        *(
            f"fp8:h64:{kind}:{tier}"
            for kind in ("full", "partial")
            for tier in ("le8", "le64", "le1024", "any", "short")
        ),
    }
    verdicts, sample = {}, {}
    for arch in arches:
        admission = _runtime.h64_admission(arch)
        admitted, withheld = (
            set(admission["admitted_routes"]),
            set(admission["withheld_routes"]),
        )
        assert admitted <= tiers and withheld <= tiers and not admitted & withheld
        assert admitted <= set(catalog["routes"])
        assert bool(withheld) == (admission["reason"] is not None)
        if 64 not in _runtime.heads():
            assert not admitted and not withheld
            continue
        assert admitted | withheld == tiers
        for queries in (1, 2, 3, 8, 9, 16, 37, 64, 65, 128, 1024, 1025, 4096, 100_000):
            route = _runtime.route_name("fp8", queries, 300, 64)
            assert _runtime.dense_route_available(64, queries, 300, arch=arch) == (
                route in admitted
            ), (arch, queries, route)
            verdicts.setdefault(route, set()).add(route in admitted)
            sample.setdefault(route, queries)
        # The one-split ``:short`` routes (K <= dense_short.max_kv, >= dense_short.min_q_blocks query blocks):
        # served where admitted, otherwise by the tier route of the query count -- never the stock kernel, so
        # dense_route_available follows the served route and the fallback is a route the arch admits.
        short = catalog["policy"]["dense_short"]
        assert short["fallback"] == "tier" and int(short["min_q_blocks"]) == 2
        max_kv = int(short["max_kv"])
        for queries in (3, 4, 8, 16, 37, 64, 128, 129, 255, 256):
            route = _runtime.route_name("fp8", queries, max_kv, 64)
            assert route.endswith(":short"), (queries, route)
            served = _runtime.served_route("fp8", queries, max_kv, 64, arch=arch)
            kind = route.split(":")[-2]
            if route in admitted:
                assert served == route
            else:
                assert served == f"fp8:h64:{kind}:{_runtime.metadata_tier(queries, 64)}"
            assert _runtime.dense_route_available(64, queries, max_kv, arch=arch) == (
                served in admitted
            ), (arch, queries, served)
            record = _runtime.route_record("fp8", queries, max_kv, 64, arch=arch)
            assert record["stages"][0][0] == "logits"
        for queries in (1, 2):
            route = _runtime.route_name("fp8", queries, max_kv, 64)
            assert not route.endswith(":short")
        assert not _runtime.route_name("fp8", 3, max_kv + 1, 64).endswith(":short")
        assert not _runtime.route_name("fp8", 37, max_kv, 32).endswith(":short")
    for route, outcomes in verdicts.items():
        if len(outcomes) == 1:
            assert (
                _runtime.dense_route_available(64, sample[route], 300, arch=None)
                == outcomes.pop()
            )
        else:
            with pytest.raises(ValueError, match="pass arch"):
                _runtime.dense_route_available(64, sample[route], 300, arch=None)
    with pytest.raises(ValueError, match="arch must be one of"):
        _runtime.h64_admission("sm_999x")


def _route_point(route_name, record):
    """A legal ``(queries, keys)`` of one dense route: the KV length that selects the
    route's KV range (``:short`` fused routes below their ceiling, the non-fused
    ``fp8:q1`` above ``fused_q1_max_kv``, the 64-head routes any length) and the
    smallest query count the router maps to the route name."""
    policy = _runtime._catalog()["policy"]
    precision = route_name.split(":")[0]
    num_heads = int(record["num_heads"])
    if route_name.endswith(":short"):
        # 32 heads: the fused routes below their KV ceiling; 64 heads: the one-split routes at K <= dense_short.max_kv
        short_kv = int(policy["dense_short"]["max_kv"])
        keys = 512 if num_heads == _runtime.NUM_HEADS else short_kv
    elif route_name == "fp8:q1":
        keys = int(policy["fused_q1_max_kv"]) + 256
    else:
        keys = 512 if num_heads == _runtime.NUM_HEADS else 300
    for queries in range(1, 4200):
        if _runtime.route_name(precision, queries, keys, num_heads) == route_name:
            return queries, keys
    raise AssertionError(
        f"no query count in 1..4199 selects {route_name} at K = {keys}"
    )


def test_dense_bindings_cover_every_program_argument():
    """Host-only, every dense route (32- and 64-head, every tier): every operand of
    every program the route launches -- its prepared ``sequence`` (``stage.name``
    keys) or each of its ``stages`` -- is produced by ``stage_bindings`` under the
    name the catalog ``arg_plan`` uses, looked up exactly as ``_submission`` does.
    The contract-level check that fails before a prebuild or launch would (an earlier
    export left the paged logits programs' ``num_sms`` parameter unbound);
    ``stage_bindings`` only reshapes and views, so CPU tensors suffice."""
    catalog = _runtime._catalog()
    num_sms = 4
    checked = 0
    for route_name, record in catalog["routes"].items():
        precision = route_name.split(":")[0]
        num_heads, bq = int(record["num_heads"]), int(record["block_q"])
        queries, keys = _route_point(route_name, record)
        assert any(
            _runtime.dense_route_available(
                num_heads, queries, keys, precision, arch=arch
            )
            for arch in catalog["arches"]
        ), (route_name, queries, keys)
        starts = torch.zeros(queries, dtype=torch.int32)
        ends = torch.full((queries,), keys, dtype=torch.int32)
        if precision == "fp4":
            q = torch.zeros((queries, num_heads, 64), dtype=torch.uint8)
            kv = torch.zeros((keys, 64), dtype=torch.uint8)
            q_scales = torch.zeros((queries, num_heads, 4), dtype=torch.uint8)
            kv_scales = torch.zeros((keys, 4), dtype=torch.uint8)
            rows = queries
        else:
            rows = max(bq, queries) if num_heads == _runtime.NUM_HEADS else queries
            head_dim = _runtime.HEAD_DIM
            q = torch.zeros((rows, num_heads, head_dim), dtype=torch.float8_e4m3fn)
            kv = torch.zeros((keys, head_dim), dtype=torch.float8_e4m3fn)
            q_scales = None
            kv_scales = torch.zeros(_runtime.align4(keys), dtype=torch.float32)[:keys]
        weights = torch.zeros((rows, num_heads), dtype=torch.float32)
        stride = _runtime.logits_stride(keys)
        output = torch.zeros((queries, stride), dtype=torch.float32)
        metadata = None
        if _runtime.schedules_metadata(num_heads):
            words = _runtime.metadata_words(queries, num_sms, num_heads)
            metadata = torch.zeros(words, dtype=torch.int32)
        bindings = _runtime.stage_bindings(
            precision,
            q,
            kv,
            weights,
            starts,
            ends,
            output,
            metadata,
            q_scales=q_scales,
            kv_scales=kv_scales,
            num_sms=num_sms,
            num_heads=num_heads,
        )
        sequence = record.get("sequence")
        if sequence:
            selections = [(None, sequence)]
        else:
            selections = [tuple(stage) for stage in record["stages"]]
        for stage, program in selections:
            missing = []
            for kind, key in catalog["programs"][program]["arg_plan"]:
                assert kind != "workspace", (route_name, program, key)
                selected, name = key.split(".", 1) if stage is None else (stage, key)
                if name not in bindings.get(selected, {}):
                    missing.append(key)
            assert not missing, (
                f"{route_name} program {program} (stage {stage}): unbound arguments {missing}"
            )
            checked += 1
    assert checked >= len(catalog["routes"])
