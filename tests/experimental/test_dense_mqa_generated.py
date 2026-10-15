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
# queries, and 2052/2053 queries on the unbounded metadata tier), plus one row
# per two-launch route at the first KV length above its single-launch class's
# policy bound, two rows past the fused Q <= 2048 route's union bound (the
# le2048 tier pairs), and the two 32-head rows whose KV length is not a multiple
# of 256 (served where the served route record's ``kv_alignment`` is 1 for the
# device's architecture; skipped elsewhere).
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
    ("fp8", 1, 1048832),
    ("fp8", 16, 1048832),
    ("fp8", 3, 1048832),
    ("fp8", 128, 131328),
    ("fp8", 33, 1048832),
    ("fp8", 132, 8448),
    ("fp8", 130, 8448),
    ("fp8", 2052, 8448),
    ("fp8", 2053, 8448),
    ("fp4", 1, 131328),
    ("fp4", 16, 1048832),
    ("fp4", 33, 1048832),
    ("fp4", 130, 65792),
    ("fp4", 2052, 65792),
    ("fp8", 130, 65792),
    ("fp8", 132, 65792),
    ("fp8", 37, 4137),
    ("fp8", 1024, 33803),
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
        # the plan reads the FP32 scales through a [1, K] view with an align4(K) row stride
        ks = (torch.rand(_runtime.align4(keys), device="cuda") + 0.5)[:keys]
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
    arch = _runtime.device_arch(torch.device("cuda"))
    if not _runtime.dense_route_available(32, queries, keys, precision, arch=arch):
        pytest.skip(
            f"catalog has no admitted route for {precision}, Q={queries}, K={keys} on {arch}"
        )
    q, kv, qs, ks, weights, starts, ends = _inputs(precision, queries, keys)
    plan = prepare_dense_mqa_logits(
        precision, q, kv, weights, starts, ends, q_scales=qs, kv_scales=ks
    )
    assert plan.route_name == _runtime.route_name(precision, queries, keys)
    # One submission per stage, or one for a prepared sequence; a ``:short``
    # route is a single logits launch that writes the schedule itself.
    assert plan.launch_count == (
        1 if plan.route.get("sequence") else len(plan.route["stages"])
    )
    if plan.route_name.endswith(":short"):
        assert plan.launch_count == 1
        assert [stage for stage, _p in plan.route["stages"]] == ["logits"]
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
        # FP4: fused single-query chain, then the scan class of the query count
        # (smallest covering block ceiling, its own KV bound), then the pair.
        ("fp4", 1, 4096, "fp4:q1:short"),
        ("fp4", 1, 131072, "fp4:q1:short"),
        ("fp4", 1, 131328, "fp4:q1"),
        ("fp4", 7, 2048, "fp4:le32:short"),
        ("fp4", 16, 1048576, "fp4:le32:short"),
        ("fp4", 16, 1048832, "fp4:le16"),
        ("fp4", 17, 2048, "fp4:le32:short"),
        ("fp4", 128, 4096, "fp4:le32:short"),
        ("fp4", 33, 1048832, "fp4:le128"),
        ("fp4", 130, 65536, "fp4:le1024:short"),
        ("fp4", 130, 65792, "fp4:le2048"),
        ("fp4", 2049, 4096, "fp4:le1024:short"),
        ("fp4", 2052, 65792, "fp4:any"),
        ("fp4", 4100, 2048, "fp4:any"),  # 1025 query blocks: no scan class
        # FP8: fused Q1 chain, plain Q1 pair, fused Q128 text, the scan class
        # qualified by block kind (Q = 128 never takes a class), then the pair.
        ("fp8", 1, 4096, "fp8:q1:short"),
        ("fp8", 1, 1048576, "fp8:q1:short"),
        ("fp8", 1, 1048832, "fp8:q1"),
        ("fp8", 128, 4096, "fp8:q128:short"),
        ("fp8", 128, 131072, "fp8:q128:short"),
        ("fp8", 128, 131328, "fp8:full:le128"),
        (
            "fp8",
            128,
            4352,
            "fp8:q128:short",
        ),  # Q = 128 is the fused Q128 rule's up to its bound, whatever the tier's order
        ("fp8", 128, 4353, "fp8:q128:short"),
        (
            "fp8",
            128,
            131077,
            "fp8:full:le128",
        ),  # never the le32 class, never the fused Q <= 128 text (K > 4352)
        # On the overlap tiers (le128: 1 < Q <= 128, le2048: 128 < Q <= 2048) the published
        # ``single_launch_precedence`` of the tier decides between the v9 scan class and
        # the r3 fused tiered text: ``{first family: route}`` (published: ``qle`` first).
        (
            "fp8",
            16,
            2048,
            {"le_class": "fp8:full:le32:short", "qle": "fp8:qle128:short"},
        ),
        (
            "fp8",
            16,
            4096,
            {"le_class": "fp8:full:le32:short", "qle": "fp8:qle128:short"},
        ),
        (
            "fp8",
            16,
            4353,
            "fp8:full:le32:short",
        ),  # above the fused Q <= 128 bound: the class under either order
        ("fp8", 16, 32768, "fp8:full:le32:short"),
        ("fp8", 16, 1048576, "fp8:full:le32:short"),
        ("fp8", 16, 1048832, "fp8:full:le16"),
        ("fp8", 3, 8192, "fp8:partial:le32:short"),
        ("fp8", 3, 1048832, "fp8:partial:le16"),
        (
            "fp8",
            2,
            300,
            {"le_class": "fp8:partial:le32:short", "qle": "fp8:qle128:short"},
        ),
        ("fp8", 33, 8192, "fp8:partial:le32:short"),
        ("fp8", 33, 1048832, "fp8:partial:le128"),
        (
            "fp8",
            37,
            4137,
            {"le_class": "fp8:partial:le32:short", "qle": "fp8:qle128:short"},
        ),
        ("fp8", 37, 4353, "fp8:partial:le32:short"),
        (
            "fp8",
            124,
            4096,
            {"le_class": "fp8:full:le32:short", "qle": "fp8:qle128:short"},
        ),
        (
            "fp8",
            132,
            2048,
            {"le_class": "fp8:full:le768:short", "qle": "fp8:qle2048:short"},
        ),
        (
            "fp8",
            132,
            8192,
            {"le_class": "fp8:full:le768:short", "qle": "fp8:qle2048:short"},
        ),
        (
            "fp8",
            132,
            200,
            {"le_class": "fp8:full:le768:short", "qle": "fp8:qle2048:short"},
        ),
        (
            "fp8",
            130,
            8192,
            {"le_class": "fp8:partial:le768:short", "qle": "fp8:qle2048:short"},
        ),
        (
            "fp8",
            129,
            4137,
            {"le_class": "fp8:partial:le768:short", "qle": "fp8:qle2048:short"},
        ),
        # above the le768 class's bound (8192) and within the fused Q <= 2048 bound -- a per-arch
        # bound (sm_100a 34048, sm_103a 65536); the logical name applies the most permissive
        # one and served_route the device's -- the fused text under either order; the
        # two-launch tiers start above the union bound and above Q = 2048 (the le768 class
        # up to 8192, then the pair)
        ("fp8", 132, 8448, "fp8:qle2048:short"),
        ("fp8", 130, 8448, "fp8:qle2048:short"),
        ("fp8", 1024, 33803, "fp8:qle2048:short"),
        ("fp8", 2048, 34048, "fp8:qle2048:short"),
        ("fp8", 2048, 34049, "fp8:qle2048:short"),
        ("fp8", 1024, 65536, "fp8:qle2048:short"),
        ("fp8", 2048, 65537, "fp8:full:le2048"),
        ("fp8", 1024, 65792, "fp8:full:le2048"),
        ("fp8", 132, 65792, "fp8:full:le2048"),
        ("fp8", 130, 65792, "fp8:partial:le2048"),
        ("fp8", 129, 65537, "fp8:partial:le2048"),
        ("fp8", 2052, 2048, "fp8:full:le768:short"),
        ("fp8", 2052, 4096, "fp8:full:le768:short"),
        ("fp8", 2052, 8448, "fp8:full:any"),
        ("fp8", 2053, 2048, "fp8:partial:le768:short"),
        ("fp8", 2049, 4137, "fp8:partial:le768:short"),
        ("fp8", 2053, 8448, "fp8:partial:any"),
        ("fp8", 3076, 2048, "fp8:full:any"),  # 769 query blocks: no scan class
    ],
)
def test_dense_mqa_route_selection(precision, queries, keys, route):
    if isinstance(route, dict):
        # a point both single-launch families cover: the published order of its overlap tier decides (one order on every
        # architecture for a logical route; ``served_route`` per device otherwise)
        tier = _runtime._overlap_tier(queries)
        orders = {
            _runtime._single_launch_precedence(a, tier)
            for a in sorted(_runtime._catalog()["arches"])
        }
        assert len(orders) == 1 and next(iter(orders)), orders
        route = route[next(iter(orders))[0]]
    assert _runtime.route_name(precision, queries, keys) == route
    if route == "fp8:qle128:short":
        assert _runtime._catalog()["policy"]["fused_qle128_max_kv"] == 4352
    if route == "fp8:qle2048:short":
        bounds = _runtime._catalog()["policy"]["fused_qle2048_max_kv"]
        assert bounds == {"sm_100a": 34048, "sm_103a": 65536}
        # the served route applies the device's own bound: the two-launch tier of the query count past it
        tier = f"fp8:{'full' if queries % 4 == 0 else 'partial'}:le2048"
        for arch, bound in bounds.items():
            served = _runtime.served_route(precision, queries, keys, arch=arch)
            assert served == (route if keys <= int(bound) else tier), (arch, served)
        if any(keys > int(b) for b in bounds.values()):
            with pytest.raises(ValueError, match="pass arch"):
                _runtime.served_route(precision, queries, keys)
        else:
            assert _runtime.served_route(precision, queries, keys) == route


def test_dense_single_launch_policy_is_published():
    """Host-only: the v10 policy carries the single-launch classes the router
    reads -- the FP8 scan ceilings with one KV bound each, the fused tiered
    bounds (``fused_qle128_max_kv``; ``fused_qle2048_max_kv`` per architecture)
    and the ``fp4`` block (fused Q1 bound, scan ceilings, one KV bound each) --
    as positive multiples of the 256-token KV alignment, ceilings ascending.
    Every ``:short`` 32-head route is one ``logits`` stage of the catalog with a
    per-architecture K rule; the fused tiered routes are catalogued exactly where
    ``single_launch_precedence`` gives the ``qle`` family a domain (the export's
    rule: a family serves iff its own KV bound exceeds the bound of every family
    ahead of it in the tier's order)."""
    catalog = _runtime._catalog()
    arches = sorted(catalog["arches"])
    policy = catalog["policy"]
    tables = {"fp8": policy, "fp4": policy["fp4"]}
    for precision, table in tables.items():
        ceilings = [int(c) for c in table["fused_scan_ceilings"]]
        assert ceilings == sorted(ceilings) and ceilings[0] >= 1, precision
        for ceiling in ceilings:
            bound = int(table[f"fused_scan{ceiling}_max_kv"])
            assert bound >= 256 and bound % 256 == 0, (precision, ceiling)
        q1_bound = int(table["fused_q1_max_kv"])
        assert q1_bound >= 256 and q1_bound % 256 == 0, precision
    assert int(policy["fused_q128_max_kv"]) % 256 == 0
    qle128_bound = int(policy["fused_qle128_max_kv"])
    assert qle128_bound >= 256 and qle128_bound % 256 == 0
    qle2048_bounds = {a: int(v) for a, v in policy["fused_qle2048_max_kv"].items()}
    assert sorted(qle2048_bounds) == arches
    assert all(v >= 256 and v % 256 == 0 for v in qle2048_bounds.values())
    short = {
        route
        for route, record in catalog["routes"].items()
        if route.endswith(":short") and int(record["num_heads"]) == 32
    }
    expected = {"fp8:q1:short", "fp8:q128:short", "fp4:q1:short"}
    for ceiling in tables["fp8"]["fused_scan_ceilings"]:
        for kind in ("full", "partial"):
            expected.add(f"fp8:{kind}:le{int(ceiling)}:short")
    for ceiling in tables["fp4"]["fused_scan_ceilings"]:
        expected.add(f"fp4:le{int(ceiling)}:short")
    # the fused tiered routes: on each overlap tier the ``qle`` family's KV bound against the bound of the scan class that
    # serves the tier's query counts (the smallest ceiling covering the tier's query blocks); the route is catalogued iff
    # some architecture's order gives the family a domain (``["qle", "le_class"]`` on both tiers: both routes; the v9
    # order ``["le_class", "qle"]`` on le128 would drop ``fp8:qle128:short``, whose bound is below the le32 class's)
    ceilings = [int(c) for c in policy["fused_scan_ceilings"]]
    fused = set()
    for tier, max_q, route in (
        ("le128", _runtime.FUSED_QLE128_MAX_Q, "fp8:qle128:short"),
        ("le2048", _runtime.FUSED_QLE2048_MAX_Q, "fp8:qle2048:short"),
    ):
        blocks = (max_q + _runtime.BLOCK_Q - 1) // _runtime.BLOCK_Q
        ceiling = next(c for c in ceilings if blocks <= c)
        class_bound = int(policy[f"fused_scan{ceiling}_max_kv"])
        for arch in arches:
            bounds = {
                "qle": qle128_bound if tier == "le128" else qle2048_bounds[arch],
                "le_class": class_bound,
            }
            order = list(policy["single_launch_precedence"][arch][tier])
            if "qle" not in order:
                continue
            ahead = [bounds[f] for f in order[: order.index("qle")]]
            if bounds["qle"] > max(ahead, default=0):
                fused.add(route)
    assert short == expected | fused
    for route in short:
        record = catalog["routes"][route]
        assert [stage for stage, _p in record["stages"]] == ["logits"], route
        assert not record.get("sequence"), route
        assert record["clean_logits"] == "fused", route
        # the K rule is published per architecture: whole 256-row blocks for the scan classes and the fused Q1 / Q128
        # texts, any K >= 1 for the fused tiered texts (their KV boxes may extend past the tensor)
        divisor = 1 if route in fused else 256
        assert int(record["kv_min"]) == divisor, route
        assert sorted(record["kv_alignment"]) == arches, route
        assert all(int(v) == divisor for v in record["kv_alignment"].values()), route


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
    program = dict(plan.route["stages"])["logits"][arch]
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
    arch = _runtime.device_arch(torch.device("cuda"))
    q, kv, qs, ks, weights, starts, ends = _inputs("fp8", 4, 4096)
    with pytest.raises(ValueError):
        prepare_dense_mqa_logits("fp8", q, kv, weights, starts, ends[:3], kv_scales=ks)
    with pytest.raises(ValueError):
        prepare_dense_mqa_logits("bf16", q, kv, weights, starts, ends, kv_scales=ks)
    # FP4 keeps whole 256-row KV blocks (the route record's kv_alignment is 256 on every architecture)
    q4, kv4, qs4, ks4, w4, s4, e4 = _inputs("fp4", 4, 4096)
    with pytest.raises(ValueError):
        prepare_dense_mqa_logits(
            "fp4", q4, kv4[:4000], w4, s4, e4, q_scales=qs4, kv_scales=ks4[:4000]
        )
    # A 32-head route whose programs keep whole 256-row KV blocks (kv_min 256) is never served below one block.
    # Q = 2052 (513 query blocks) is past the fused Q <= 2048 route, whose boxes may extend past the tensor, so the
    # le768 scan class owns the call while K <= fused_scan768_max_kv; its record publishes kv_min 256 on every
    # architecture and the call is refused (the catalog's word, not a restated rule).
    q2, kv2, _qs, ks2, w2, s2, e2 = _inputs("fp8", 2052, 200)
    assert _runtime.route_name("fp8", 2052, 200) == "fp8:full:le768:short"
    assert _runtime.route_kv_min("fp8:full:le768:short") == 256
    assert not _runtime.dense_route_available(32, 2052, 200, arch=arch)
    with pytest.raises(ValueError):
        prepare_dense_mqa_logits("fp8", q2, kv2, w2, s2, e2, kv_scales=ks2)
    # A 32-head K that is not a multiple of 256 is served exactly where the route record publishes
    # kv_alignment 1 for this architecture (the catalog's word, not a restated rule): above the fused
    # Q <= 2048 bound the two-launch tier record decides.
    above = max(_runtime._fused_qle2048_bounds().values()) + 4000
    q3, kv3, _qs, ks3, w3, s3, e3 = _inputs("fp8", 132, above)
    assert _runtime.route_name("fp8", 132, above) == "fp8:full:le2048"
    served = _runtime.dense_route_available(32, 132, above, arch=arch)
    assert served == (_runtime.route_kv_alignment("fp8:full:le2048", arch=arch) == 1)
    if served:
        plan = prepare_dense_mqa_logits("fp8", q3, kv3, w3, s3, e3, kv_scales=ks3)
        assert plan.route_name == "fp8:full:le2048"
    else:
        with pytest.raises(ValueError, match="K % 256 == 0"):
            prepare_dense_mqa_logits("fp8", q3, kv3, w3, s3, e3, kv_scales=ks3)


# ---------------------------------------------------------------------------
# DeepGEMM-signature one-shot entry (fp8_mqa_logits) and the head-count table
# ---------------------------------------------------------------------------


def test_dense_route_table_is_catalog_driven():
    """Host-only: the 32-head names carry no head infix -- the shipped routes, the
    ``dense_mqa.v9`` single-launch classes by query-block kind and ceiling, the
    ``dense_mqa.v10`` single-launch fused tiered routes ``fp8:qle128:short``
    (1 < Q <= 128, K <= ``fused_qle128_max_kv``) and ``fp8:qle2048:short``
    (128 < Q <= 2048, K <= ``fused_qle2048_max_kv[arch]``, the per-arch bound)
    consulted in the published per-tier, per-arch order
    ``single_launch_precedence`` (published: the fused texts first), the pairs by
    metadata tier -- other head counts carry the ``h<H>`` infix with tiers named
    by their token ceiling, and availability (including the per-route, per-arch
    KV alignment and the per-arch fused Q <= 2048 bound) is the catalog's word."""
    catalog = _runtime._catalog()
    policy = catalog["policy"]
    routes = catalog["routes"]
    arches = sorted(catalog["arches"])
    fused_max_kv = int(policy["fused_qle128_max_kv"])
    assert 4096 <= fused_max_kv < 8192
    bounds = _runtime._fused_qle2048_bounds()
    assert set(bounds) == set(arches) and all(v % 256 == 0 for v in bounds.values())
    fused_2048_max_kv = max(
        bounds.values()
    )  # the most permissive bound: the logical route's
    assert fused_max_kv < min(bounds.values()) < fused_2048_max_kv
    scan32, scan768 = (
        int(policy["fused_scan32_max_kv"]),
        int(policy["fused_scan768_max_kv"]),
    )
    assert fused_max_kv < scan32 and scan768 < min(bounds.values())
    precedence = policy["single_launch_precedence"]
    assert set(precedence) == set(arches)
    for arch in arches:
        assert set(precedence[arch]) == {"le128", "le2048"}
        for tier, order in precedence[arch].items():
            assert set(order) <= {"le_class", "qle"} and len(set(order)) == len(
                order
            ), (arch, tier, order)
            assert tuple(order) == _runtime._single_launch_precedence(arch, tier)
    assert _runtime._single_launch_precedence(arches[0], None) == ("le_class",)
    with pytest.raises(ValueError, match="arch must be one of"):
        _runtime._single_launch_precedence("sm_999x", "le128")
    first = {
        tier: {precedence[a][tier][0] if precedence[a][tier] else None for a in arches}
        for tier in ("le128", "le2048")
    }
    # order-independent points
    assert _runtime.route_name("fp8", 1, 4096) == "fp8:q1:short"
    assert _runtime.route_name("fp8", 16, scan32 + 256) == "fp8:full:le16"
    assert (
        _runtime.route_name("fp8", 130, fused_2048_max_kv + 256) == "fp8:partial:le2048"
    )
    assert _runtime.route_name("fp8", 130, scan768 + 256) == "fp8:qle2048:short"
    assert _runtime.route_name("fp8", 2049, 8192) == "fp8:partial:le768:short"
    assert _runtime.route_name("fp8", 2049, scan768 + 256) == "fp8:partial:any"
    # the fused Q <= 128 route is catalogued iff some architecture consults it before the le32 class (its bound is below the
    # class's, so it has no domain behind the class); the fused Q <= 2048 route serves above the le768 class's bound either way
    assert ("fp8:qle128:short" in routes) == ("qle" in first["le128"])
    assert "fp8:qle2048:short" in routes
    if first["le128"] == {
        "le_class"
    }:  # the v9 routing of Cake main (the classes first)
        assert _runtime.route_name("fp8", 16, 4096) == "fp8:full:le32:short"
        assert _runtime.route_name("fp8", 37, 4096) == "fp8:partial:le32:short"
        assert _runtime.route_name("fp8", 16, scan32) == "fp8:full:le32:short"
    elif first["le128"] == {"qle"}:
        assert _runtime.route_name("fp8", 16, 4096) == "fp8:qle128:short"
        assert (
            _runtime.route_name("fp8", 16, fused_max_kv + 256) == "fp8:full:le32:short"
        )
    else:  # the architectures disagree on the tier's order: no logical route on it
        with pytest.raises(ValueError, match="pass arch"):
            _runtime.route_name("fp8", 16, 4096)
    if first["le2048"] == {"le_class"}:
        assert _runtime.route_name("fp8", 130, 8192) == "fp8:partial:le768:short"
        assert _runtime.route_name("fp8", 132, 200) == "fp8:full:le768:short"
    elif first["le2048"] == {"qle"}:
        assert _runtime.route_name("fp8", 130, 8192) == "fp8:qle2048:short"
        assert _runtime.route_name("fp8", 132, 200) == "fp8:qle2048:short"
    else:
        with pytest.raises(ValueError, match="pass arch"):
            _runtime.route_name("fp8", 130, 8192)
    assert _runtime.dense_route_available(32, 16, 4096)
    assert _runtime.dense_route_available(32, 16, 1048832)
    # 32 heads: the K rule is each route record's, per architecture (FP4 256; an FP8 route 256, or 1 on the
    # architectures where its programs take any K), with the record's kv_min (256 for the shipped two-stage
    # programs and the v9 scan classes, whose KV boxes are fitted to one block; 1 for the fused tiered programs).
    fp8_h32_alignment = _runtime.kv_alignment(32, "fp8")
    assert fp8_h32_alignment in (1, 256) and _runtime.kv_alignment(32, "fp4") == 256
    for route, record in routes.items():
        assert sorted(record["kv_alignment"]) == arches, route
        assert all(int(v) in (1, 256) for v in record["kv_alignment"].values()), route
        assert int(record["kv_min"]) in (1, 256), route
        if route.startswith("fp4"):
            assert all(int(v) == 256 for v in record["kv_alignment"].values()), route
        if route.startswith("fp8:h64"):
            assert record["kv_min"] == 1
            assert all(int(v) == 1 for v in record["kv_alignment"].values()), route
    for keys in (4096, 1048832):
        record = _runtime.route_record("fp8", 16, keys)
        assert record["num_heads"] == 32 and record["block_q"] == 4
        assert record["clean_logits"] == "fused"
        assert (
            record["kv_alignment"]
            == routes[_runtime.route_name("fp8", 16, keys)]["kv_alignment"]
        )
    for arch in arches:
        assert _runtime.route_kv_alignment("fp8:full:le16", arch=arch) in (1, 256)
        # A 32-head ``:short`` route is served everywhere (no per-arch admission); the K verdict is the served record's
        assert _runtime.served_route("fp8", 16, 4096, arch=arch) == _runtime.route_name(
            "fp8", 16, 4096
        )
        assert _runtime.served_route("fp4", 7, 2048, arch=arch) == "fp4:le32:short"
        assert _runtime.dense_route_available(32, 7, 2048, "fp4", arch=arch)
        for queries, keys in (
            (16, 4100),
            (16, 8196),
            (37, 4137),
            (1024, 33803),
            (1024, fused_2048_max_kv + 13),
            (132, 200),
            (3, 100),
        ):
            served = _runtime.served_route("fp8", queries, keys, arch=arch)
            record = routes[served]
            expected = (
                keys >= int(record["kv_min"])
                and keys % int(record["kv_alignment"][arch]) == 0
            )
            assert (
                _runtime.dense_route_available(32, queries, keys, arch=arch) == expected
            ), (arch, queries, keys, served)
        assert not _runtime.dense_route_available(
            32, 16, 4100, precision="fp4", arch=arch
        )
        # below the route's smallest KV length nothing is served, whatever the divisor (a two-stage
        # 32-head route: Q > 2048 above the le768 class's bound); the fused tiered routes' boxes may extend past the tensor (kv_min 1)
        assert not _runtime.dense_route_available(32, 2052, 200, arch=arch)
        # above the le768 class's bound a Q > 2048 call takes its two-stage tier route and the K verdict is that
        # record's, per architecture: ``fp8:full:any`` takes any K on sm_100a and whole blocks on sm_103a;
        # ``fp8:partial:any`` whole blocks everywhere, so the unaligned point is refused there on both
        unaligned = scan768 + 13
        tier_route = _runtime.served_route("fp8", 2052, unaligned, arch=arch)
        assert tier_route == "fp8:full:any", (arch, tier_route)
        whole = unaligned % int(routes[tier_route]["kv_alignment"][arch]) == 0
        assert whole == (arch == "sm_100a"), arch
        assert _runtime.dense_route_available(32, 2052, unaligned, arch=arch) == whole
        assert (
            _runtime.served_route("fp8", 2053, unaligned, arch=arch)
            == "fp8:partial:any"
        )
        assert not _runtime.dense_route_available(32, 2053, unaligned, arch=arch)
    # The single-launch fused Q <= 2048 route (and the fused Q <= 128 route where catalogued) takes any K (its KV boxes
    # extend past the tensor) on every architecture; the two-launch Q >= 129 full class is admitted for any K on the
    # architectures the producer measured a win on -- sm_100a -- and keeps whole 256-row blocks on sm_103a.
    assert routes["fp8:qle2048:short"]["kv_min"] == 1
    assert all(
        int(v) == 1 for v in routes["fp8:qle2048:short"]["kv_alignment"].values()
    )
    if "fp8:qle128:short" in routes:
        assert routes["fp8:qle128:short"]["kv_min"] == 1
        assert all(
            int(v) == 1 for v in routes["fp8:qle128:short"]["kv_alignment"].values()
        )
        assert _runtime.dense_route_available(32, 37, 4137, arch="sm_100a")
        assert _runtime.dense_route_available(32, 37, 4137, arch="sm_103a")
        assert _runtime.dense_route_available(32, 3, 100, arch="sm_103a")
        assert _runtime.dense_route_available(32, 37, 4137)
    else:  # the classes first: the q37 point is the le32 class's (whole blocks) on both architectures -> stock, no arch needed
        for arch in arches:
            assert not _runtime.dense_route_available(32, 37, 4137, arch=arch)
            assert (
                _runtime.served_route("fp8", 37, 4137, arch=arch)
                == "fp8:partial:le32:short"
            )
        assert not _runtime.dense_route_available(32, 37, 4137)
    assert _runtime.dense_route_available(32, 1024, 33803, arch="sm_100a")
    assert _runtime.dense_route_available(32, 1024, 33803, arch="sm_103a")
    assert _runtime.dense_route_available(32, 1024, 33792, arch="sm_103a")
    assert _runtime.dense_route_available(
        32, 1024, fused_2048_max_kv + 13, arch="sm_100a"
    )
    assert not _runtime.dense_route_available(
        32, 1024, fused_2048_max_kv + 13, arch="sm_103a"
    )
    # Without ``arch`` the K verdict must agree across the architectures (no silent admit).
    assert _runtime.dense_route_available(32, 1024, 33803)
    with pytest.raises(ValueError, match="pass arch"):
        _runtime.dense_route_available(32, 1024, fused_2048_max_kv + 13)
    # inside the larger per-arch bound only: served on both arches, by the fused route where the bound covers K and by the
    # two-launch full tier elsewhere (any K on sm_100a); no architecture-less verdict
    band = min(bounds.values()) + 256
    assert _runtime.route_name("fp8", 1024, band) == "fp8:qle2048:short"
    assert (
        _runtime.served_route("fp8", 1024, band, arch="sm_103a") == "fp8:qle2048:short"
    )
    assert _runtime.served_route("fp8", 1024, band, arch="sm_100a") == "fp8:full:le2048"
    assert _runtime.dense_route_available(32, 1024, band, arch="sm_103a")
    assert _runtime.dense_route_available(32, 1024, band, arch="sm_100a")
    assert (
        _runtime.route_record("fp8", 1024, band, arch="sm_103a")["stages"]
        == routes["fp8:qle2048:short"]["stages"]
    )
    assert (
        _runtime.route_record("fp8", 1024, band, arch="sm_100a")["stages"]
        == routes["fp8:full:le2048"]["stages"]
    )
    with pytest.raises(ValueError, match="pass arch"):
        _runtime.dense_route_available(32, 1024, band)
    with pytest.raises(ValueError, match="pass arch"):
        _runtime.route_record("fp8", 1024, band)
    with pytest.raises(ValueError, match="pass arch"):
        _runtime.route_kv_alignment("fp8:full:le2048")
    assert _runtime.route_kv_alignment("fp8:full:le2048", arch="sm_100a") == 1
    assert _runtime.route_kv_alignment("fp8:full:le2048", arch="sm_103a") == 256
    assert _runtime.route_kv_alignment("fp8:full:le128", arch="sm_100a") == 256
    assert _runtime.route_kv_alignment("fp8:partial:le2048", arch="sm_100a") == 256
    record = _runtime.route_record("fp8", 16, 32768)
    assert record["num_heads"] == 32 and record["block_q"] == 4
    assert (
        record["clean_logits"] == "fused"
        and record["kv_alignment"]
        == routes[_runtime.route_name("fp8", 16, 32768)]["kv_alignment"]
    )
    # an aligned le128-tier point is served (and has a record) under either order; its text is single-launch either way
    served = _runtime.served_route("fp8", 37, 4096, arch="sm_100a")
    single = _runtime.route_record("fp8", 37, 4096, arch="sm_100a")
    assert (
        single["clean_logits"] == "fused"
        and single["stages"] == routes[served]["stages"]
    )
    assert [stage for stage, _p in single["stages"]] == ["logits"]
    assert served in ("fp8:partial:le32:short", "fp8:qle128:short")
    fused_2048 = _runtime.route_record("fp8", 1024, 33803, arch="sm_103a")
    assert (
        fused_2048["clean_logits"] == "fused"
        and fused_2048["stages"] == routes["fp8:qle2048:short"]["stages"]
    )
    assert [stage for stage, _p in fused_2048["stages"]] == ["logits"]
    assert _runtime.route_record("fp4", 16, 4096)["kv_alignment"] == {
        a: 256 for a in arches
    }
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
        assert record["clean_logits"] == "raw" and record["kv_min"] == 1
        assert all(int(v) == 1 for v in record["kv_alignment"].values())
        assert [stage for stage, _p in record["stages"]] == ["logits"]
        # Only tiers admitted on at least one architecture have a record (and programs); a route withheld
        # everywhere is absent from the table. Every catalogued 64-head route is a single logits stage whose
        # program on each architecture is a catalogued program delivered for that architecture (program ids
        # are the generated-unit names, not template names; the 64-head production text is chosen per
        # architecture, so the union over the architectures may hold one program per architecture).
        programs = _runtime._catalog()["programs"]
        arches = sorted(_runtime._catalog()["arches"])
        for r in _runtime._catalog()["routes"]:
            if r.startswith("fp8:h64:"):
                for a in arches:
                    names = _runtime.program_names(r, arch=a)
                    assert len(names) == 1 and names[0] in programs, (r, a, names)
                    assert a in programs[names[0]]["arches"], (r, a, names)
                union = _runtime.program_names(r)
                assert 1 <= len(union) <= len(arches), (r, union)
                assert set(union) <= set(programs), (r, union)
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
        # 32-head unaligned K (served only when the catalog publishes kv_alignment.fp8.32 == 1; skipped otherwise)
        (37, 4137, 32),
        (5, 300, 32),
        (3, 100, 32),
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
    """Host-only: every program a dense or paged route names is catalogued on exactly the architectures
    the route is delivered on, its generated units exist under FlashInfer's ``csrc`` (a module: its own
    ``<name>_kernel.cu`` + ``<name>_binding.cu``; a sequence: >= 2 module kernels + its own binding),
    no two generated kernel units are byte-identical and no generated unit is unreferenced -- a route
    table pointing at a program hash whose units were regenerated under a new hash fails here, not at
    JIT build time."""
    import hashlib

    from flashinfer.jit import env

    csrc = env.FLASHINFER_CSRC_DIR
    catalog = _runtime._catalog()
    programs = catalog["programs"]
    generated = "csrc/experimental/deepgemm_dense_mqa/generated"
    # A window-selected text route is delivered only on the architectures whose window
    # table names it (``policy.paged.dynamic_scheduler.static_window_routes``); every
    # other route binds a program on every catalogued architecture.
    text_route_arches = {}
    for arch, table in catalog["policy"]["paged"]["dynamic_scheduler"][
        "static_window_routes"
    ].items():
        for pairs in table.values():
            for _window, named in pairs:
                text_route_arches.setdefault(named, set()).add(arch)
    for table in ("routes", "paged_routes"):
        for route, record in catalog.get(table, {}).items():
            expected = sorted(text_route_arches.get(route, catalog["arches"]))
            for _stage, by_arch in record["stages"]:
                where = (table, route, by_arch, expected)
                assert sorted(by_arch) == expected, where
                for arch, program in by_arch.items():
                    where = (table, route, arch, program)
                    assert program in programs, where
                    assert arch in programs[program]["arches"], where
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
        # 32 heads never take the 64-head one-split route: at this KV length a 32-head call takes its
        # single-launch route -- the v9 in-kernel scan class of its query count, or the r3 fused tiered
        # text where the published order of its tier puts that family first; above the le32 class's
        # policy bound the metadata + logits pair serves Q = 37; above Q = 3072 no scan class applies.
        order = _runtime._single_launch_precedence(arch, "le128")
        route32 = _runtime.served_route("fp8", 37, max_kv, 32, arch=arch)
        assert route32 == (
            "fp8:qle128:short"
            if order and order[0] == "qle"
            else "fp8:partial:le32:short"
        ), route32
        order = _runtime._single_launch_precedence(arch, "le2048")
        route32 = _runtime.served_route("fp8", 129, max_kv, 32, arch=arch)
        assert route32 == (
            "fp8:qle2048:short"
            if order and order[0] == "qle"
            else "fp8:partial:le768:short"
        ), route32
        assert _runtime.route_name("fp8", 2049, max_kv, 32) == "fp8:partial:le768:short"
        above32 = int(catalog["policy"]["fused_scan32_max_kv"]) + 256
        assert not _runtime.route_name("fp8", 37, above32, 32).endswith(":short")
        assert not _runtime.route_name("fp8", 3073, max_kv, 32).endswith(":short")
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
    route's KV range (``:short`` routes below their class's bound -- or, for the
    family consulted second on its overlap tier, the first KV length above the
    bound of the family ahead of it -- a 32-head two-launch route at the first KV
    length above every single-launch bound that would otherwise own the query
    count, the 64-head routes any length) and the smallest query count the router
    maps to the route name."""
    policy = _runtime._catalog()["policy"]
    precision = route_name.split(":")[0]
    num_heads = int(record["num_heads"])
    if num_heads == _runtime.NUM_HEADS:
        # every single-launch bound of the precision: a route is reached at 512 or 256 past one of them
        table = policy if precision == "fp8" else policy["fp4"]
        bounds = [int(table["fused_q1_max_kv"])]
        for ceiling in table["fused_scan_ceilings"]:
            bounds.append(int(table[f"fused_scan{int(ceiling)}_max_kv"]))
        if precision == "fp8":
            bounds.append(int(policy["fused_q128_max_kv"]))
            bounds.append(int(policy["fused_qle128_max_kv"]))
            bounds.extend(int(v) for v in policy["fused_qle2048_max_kv"].values())
        candidates = [512] + sorted({bound + 256 for bound in bounds})
    elif route_name.endswith(":short"):
        # the 64-head one-split routes at K <= dense_short.max_kv
        candidates = [int(policy["dense_short"]["max_kv"])]
    else:
        candidates = [300]
    for keys in candidates:
        for queries in range(1, 4200):
            try:
                found = (
                    _runtime.route_name(precision, queries, keys, num_heads)
                    == route_name
                )
            except ValueError:  # no logical route at this point (the architectures' orders differ on its tier)
                continue
            if found:
                return queries, keys
    raise AssertionError(
        f"no query count in 1..4199 selects {route_name} at K in {candidates}"
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
            # Every program a stage launches on any architecture binds the same way.
            selections = [
                (stage, program)
                for stage, by_arch in record["stages"]
                for program in sorted(set(by_arch.values()))
            ]
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
