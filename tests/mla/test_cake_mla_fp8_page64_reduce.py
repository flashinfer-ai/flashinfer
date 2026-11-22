"""The two split-reducer builds of Cake's FP8 page-64 decode (domain ``mla_fp8_page64_pdl``).

The FP8 page-64 producer writes one BF16 partial output and one ``(max, sum)``
pair per (query row, split, head); a reducer merges the splits in a fixed order.
The binding links two reducer builds that write the same O and LSE bits: the
eight-head-slice build, and the head build, which spreads a row's 128 heads over
up to 128 CTAs and stages every partial row of a CTA once in shared memory. The
binding launches the head build for at most ``CAKE_FP8_HEAD_REDUCE_MAX_ROWS``
query rows split at least ``CAKE_FP8_HEAD_REDUCE_MIN_SPLIT_PER_ROW`` ways per
row when the reducer CTA count the host passes (``plan_fp8_reduce_ctas``) lets a
CTA stage all of its partial rows; the slice build otherwise. These tests check:

- the host rule: the reducer CTA count of every launch outside the head build
  is the one it was before, a head launch satisfies the head build's bounds,
  and the host's prediction (``fp8_head_reduce``) is the binding's own test
  (no GPU needed);
- the binding's defaults and the modules' compile definitions equal the host's
  constants, and the head build's opt-in and launch use one dynamic
  shared-memory size (no GPU needed);
- on a Blackwell device, the head build's O and LSE equal the slice build's bit
  for bit at one and two query rows, KV lengths 384 to 131072, rows with empty
  splits, with and without LSE output, at the rule's CTA count and at forced
  ones, and at forced split counts from 2 to 37; the profiler shows the build
  the host predicts;
- run-to-run bitwise determinism of the head build, eager and under CUDA-graph
  replay, and equal bits in the default and the stateless mode.
"""

import math
import re
import warnings

import pytest
import torch

from flashinfer.mla import cake_trtllm_mla_blackwell as cake_module
from flashinfer.utils import get_compute_capability

NUM_HEADS, QK_DIM, LATENT, PAGE_SIZE = 128, 576, 512, 64
BMM1 = 1.0 / (192**0.5)
SLICE_REDUCER = "cake_trtllm_mla_blackwell_4de3d4bf2f8c5467bf29"
# SM counts of the rule check: B200 (148) and GB300 (152), and others around them;
# 120 and 240 hit the slice-build fallback at one query row (240 also at two).
_RULE_SMS = (100, 120, 132, 148, 152, 160, 188, 240)
_RULE_LENGTHS = (
    128,
    384,
    640,
    1024,
    2048,
    4096,
    16384,
    16512,
    28800,
    32768,
    131072,
    262144,
)


def _slice_ctas(rows, num_sms):
    """The reducer CTA count of every launch before the head build."""
    return min(64, max(1, (num_sms * 2) // max(1, rows * 2)))


def _head_bound(num_split, num_ctas):
    return (
        1 <= num_ctas <= 128
        and -(-NUM_HEADS // num_ctas) * num_split
        <= cake_module._FP8_HEAD_REDUCE_MAX_ITEMS
    )


def test_fp8_reduce_plan_rule():
    """``plan_fp8_reduce_ctas`` keeps every launch outside the head build at the
    CTA count it had, gives a head launch ``min(128, num_sms // rows)`` CTAs
    within the stage bound, and ``fp8_head_reduce`` of the planned count says
    which build the binding launches. No GPU needed."""
    assert cake_module._FP8_HEAD_REDUCE_MIN_SPLIT_PER_ROW == 8
    assert cake_module._FP8_HEAD_REDUCE_MAX_CTAS == 128
    assert cake_module._FP8_HEAD_REDUCE_MAX_ITEMS == 112
    max_rows = cake_module._FP8_HEAD_REDUCE_MAX_ROWS
    assert set(max_rows) == {"sm_100a", "sm_103a"}
    assert all(0 <= value <= 2 for value in max_rows.values())
    assert max_rows["sm_103a"] == 2
    plan_split = cake_module.plan_num_split
    plan = cake_module.plan_fp8_reduce_ctas
    head = cake_module.fp8_head_reduce
    # (num_sms, rows, max_seq_len) -> (splits, head CTAs, slice CTAs); no head CTAs
    # where the head build does not apply (fewer than eight splits per row, or more
    # rows than any architecture's maximum).
    expected = {
        (148, 1, 384): (2, None, 64),
        (148, 1, 1024): (4, None, 64),
        (148, 1, 4096): (16, 128, 64),
        (148, 1, 16384): (64, 128, 64),
        (148, 1, 32768): (74, 128, 64),
        (148, 2, 384): (2, None, 64),
        (148, 2, 16384): (37, 74, 64),
        (148, 2, 32768): (37, 74, 64),
        (148, 3, 32768): (24, None, 49),
        (152, 1, 384): (2, None, 64),
        (152, 1, 2048): (8, 128, 64),
        (152, 1, 4096): (16, 128, 64),
        (152, 1, 32768): (76, 128, 64),
        (152, 1, 262144): (76, 128, 64),
        (152, 2, 1024): (4, None, 64),
        (152, 2, 2048): (8, None, 64),
        (152, 2, 4096): (16, 76, 64),
        (152, 2, 32768): (38, 76, 64),
        (152, 3, 32768): (25, None, 50),
    }
    for key, (split, head_ctas, slice_ctas) in expected.items():
        num_sms, rows, max_seq_len = key
        assert plan_split(rows, max_seq_len, num_sms) == split
        for arch in ("sm_100a", "sm_103a"):
            chosen = head_ctas is not None and rows <= max_rows[arch]
            want = head_ctas if chosen else slice_ctas
            case = (arch, num_sms, rows, max_seq_len)
            assert plan(rows, split, num_sms, arch) == want, case
            assert head(rows, split, want, arch) == chosen, case
    heads = 0
    for arch in ("sm_100a", "sm_103a", "sm_90a"):
        arch_rows = max_rows.get(arch, 0)
        for num_sms in _RULE_SMS:
            for rows in range(1, 129):
                for max_seq_len in _RULE_LENGTHS:
                    split = plan_split(rows, max_seq_len, num_sms)
                    ctas = plan(rows, split, num_sms, arch)
                    case = (arch, num_sms, rows, max_seq_len, split, ctas)
                    head_ctas = min(128, max(1, num_sms // rows))
                    chosen = (
                        rows <= arch_rows
                        and split >= 8 * rows
                        and _head_bound(split, head_ctas)
                    )
                    assert ctas == (
                        head_ctas if chosen else _slice_ctas(rows, num_sms)
                    ), case
                    # The binding's test of the planned count is the host's choice.
                    assert head(rows, split, ctas, arch) == chosen, case
                    if chosen:
                        assert 1 <= ctas <= 128, case
                        assert math.ceil(128 / ctas) * split <= 112, case
                        heads += 1
    assert heads > 0
    # The fallback above the stage bound: 120 SMs at one row (120 CTAs stage two
    # heads of 60 splits each), 240 SMs at one row (120 splits) and two rows.
    for arch in ("sm_100a", "sm_103a"):
        for num_sms, rows in ((120, 1), (240, 1), (240, 2)):
            if rows > max_rows[arch]:
                continue
            split = plan_split(rows, 32768, num_sms)
            assert split >= 57, (arch, num_sms, rows, split)
            ctas = plan(rows, split, num_sms, arch)
            assert ctas == 64 and not head(rows, split, ctas, arch)


def _block(text, index):
    start = text.index(f"namespace CAKE_VARIANT_NS_{index} {{")
    return text[start : text.index(f"}}  // namespace CAKE_VARIANT_NS_{index}", start)]


def test_fp8_reduce_binding_matches_host():
    """The binding's head-build defaults equal the host's constants (its
    ``CAKE_FP8_HEAD_REDUCE_MAX_ROWS`` default is the sm_100a value), the sm_103a
    module sets its own value once and no module overrides the other three;
    both FP8 page-64 modules link the slice build at ``dependencies[1]`` and the
    head build at ``dependencies[2]``, whose ``Run`` blocks launch them; the head
    build's opt-in and launch use one dynamic shared-memory size above 48 KiB.
    No GPU needed."""
    text = cake_module._source_path(cake_module._FP8_PAGE64_BINDING).read_text()
    max_rows = cake_module._FP8_HEAD_REDUCE_MAX_ROWS
    defaults = {
        "CAKE_FP8_HEAD_REDUCE_MAX_ROWS": max_rows["sm_100a"],
        "CAKE_FP8_HEAD_REDUCE_MIN_SPLIT_PER_ROW": (
            cake_module._FP8_HEAD_REDUCE_MIN_SPLIT_PER_ROW
        ),
        "CAKE_FP8_HEAD_REDUCE_MAX_CTAS": cake_module._FP8_HEAD_REDUCE_MAX_CTAS,
        "CAKE_FP8_HEAD_REDUCE_MAX_ITEMS": cake_module._FP8_HEAD_REDUCE_MAX_ITEMS,
    }
    for macro, value in defaults.items():
        assert text.count(f"#define {macro} ") == 1, macro
        assert f"#define {macro} {value}\n" in text, macro
    assert (
        "if (num_queries <= CAKE_FP8_HEAD_REDUCE_MAX_ROWS &&\n"
        "        num_split >= (int64_t)CAKE_FP8_HEAD_REDUCE_MIN_SPLIT_PER_ROW"
        " * num_queries &&"
    ) in text
    assert (
        "((128 + (int64_t)num_reduce_ctas - 1) / num_reduce_ctas) * num_split "
        "<= CAKE_FP8_HEAD_REDUCE_MAX_ITEMS) {"
    ) in text
    head_block = _block(text, 2)
    opt_in = re.findall(
        r"CakeSetMaxDynamicSmem\(\s*reinterpret_cast<const void\*>\((kernel_\w+)\),"
        r"\s*(\d+)\)",
        head_block,
    )
    launch = re.findall(r"config\.dynamicSmemBytes = (\d+)u;", head_block)
    assert len(opt_in) == 1 and len(launch) == 1, (opt_in, launch)
    assert int(opt_in[0][1]) == int(launch[0]) > 48 * 1024
    assert "dim3 block(128u, 1u, 1u);" in head_block
    kernels = {
        index: set(
            re.findall(
                r"reinterpret_cast<const void\*>\((kernel_\w+)\)", _block(text, index)
            )
        )
        for index in (1, 2)
    }
    modules = {
        family["arch"]: (name, family)
        for name, family in cake_module.FAMILIES.items()
        if family["binding"] == cake_module._FP8_PAGE64_BINDING
    }
    assert set(modules) == {"sm_100a", "sm_103a"}
    head_programs = set()
    for arch, (name, family) in modules.items():
        dependencies = family["dependencies"]
        assert len(dependencies) == 3 and dependencies[1] == SLICE_REDUCER, name
        assert kernels[1] == {f"kernel_{dependencies[1]}"}, name
        assert kernels[2] == {f"kernel_{dependencies[2]}"}, name
        head_programs.add(dependencies[2])
        assert cake_module.MODULES[dependencies[2]]["arches"] == ["sm_100a", "sm_103a"]
        flags = family.get("binding_flags", [])
        for macro in (
            "CAKE_FP8_HEAD_REDUCE_MIN_SPLIT_PER_ROW",
            "CAKE_FP8_HEAD_REDUCE_MAX_CTAS",
            "CAKE_FP8_HEAD_REDUCE_MAX_ITEMS",
        ):
            assert not any(flag.startswith(f"-D{macro}") for flag in flags), name
        rows_flags = [
            flag for flag in flags if flag.startswith("-DCAKE_FP8_HEAD_REDUCE_MAX_ROWS")
        ]
        namespaces = [flag for flag in flags if flag.startswith("-DCAKE_VARIANT_NS_2=")]
        if arch == "sm_100a":
            assert not rows_flags and not namespaces, name
        else:
            assert rows_flags == [
                f"-DCAKE_FP8_HEAD_REDUCE_MAX_ROWS={max_rows[arch]}"
            ], name
            assert len(namespaces) == 1, name
            assert namespaces[0].startswith(
                f"-DCAKE_VARIANT_NS_2=variant_{dependencies[2]}_sm_103a_"
            ), name
    assert len(head_programs) == 1


# --- On a Blackwell device ---------------------------------------------------------


def _require_blackwell() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if get_compute_capability(torch.device("cuda")) not in ((10, 0), (10, 3)):
        pytest.skip("Cake Blackwell MLA decode requires SM100a or SM103a")


@pytest.fixture
def fp8_domains(monkeypatch):
    """The domains ``trtllm_mla_blackwell_decode`` selects, in call order."""
    seen: list = []
    select = cake_module.select_domain

    def spy(**kwargs):
        domain = select(**kwargs)
        seen.append(domain)
        return domain

    monkeypatch.setattr(cake_module, "select_domain", spy)
    return seen


def _inputs(batch, q_len, max_seq_len, lengths, seed=0):
    """FP8 query, page-64 KV cache, page table and KV lengths. ``lengths``:
    ``uniform`` (every request ``max_seq_len``) or ``varied`` (one short
    request, whose last splits are empty: the only request at a third of
    ``max_seq_len``; with two requests the second at an eighth)."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    width = -(-max_seq_len // PAGE_SIZE)
    pages = batch * width
    query = torch.randn(
        batch, q_len, NUM_HEADS, QK_DIM, device="cuda", generator=generator
    )
    kv_cache = torch.randn(pages, PAGE_SIZE, QK_DIM, device="cuda", generator=generator)
    block_tables = torch.randperm(pages, device="cuda", generator=generator)
    seq_lens = [max_seq_len] * batch
    if lengths == "varied":
        if batch == 1:
            seq_lens[0] = max(q_len, max_seq_len // 3)
        else:
            seq_lens[-1] = max(q_len, max_seq_len // 8)
    return (
        (query * 0.5).to(torch.float8_e4m3fn),
        (kv_cache * 0.5).to(torch.float8_e4m3fn),
        block_tables.to(torch.int32).view(batch, width),
        torch.tensor(seq_lens, dtype=torch.int32, device="cuda"),
    )


def _cake(query, kv_cache, block_tables, seq_lens, max_seq_len, **kwargs):
    """Explicit ``backend="cake"``: the Blackwell module's decode entry."""
    arguments = {
        "qk_nope_head_dim": 128,
        "kv_lora_rank": LATENT,
        "qk_rope_head_dim": QK_DIM - LATENT,
        "sparse_mla_top_k": 0,
        "out": None,
        "bmm1_scale": BMM1,
        "bmm2_scale": 1.0,
        "sinks": None,
        "skip_softmax_threshold_scale_factor": None,
        "enable_pdl": None,
        "uses_shared_paged_kv_idx": True,
        "lse": None,
        "return_lse": False,
        "cum_seq_lens_q": None,
        "max_q_len": None,
        "multi_ctas_kv_counter_buffer": None,
        "sparse_mla_top_k_lens": None,
        "enable_dcp": False,
        "backend": "cake",
    }
    arguments.update(kwargs)
    return cake_module.trtllm_mla_blackwell_decode(
        query, kv_cache, block_tables, seq_lens, max_seq_len, **arguments
    )


def _nan_out(query):
    return torch.full(
        (*query.shape[:-1], LATENT), float("nan"), dtype=torch.bfloat16, device="cuda"
    )


def _bits(result):
    """The bytes of the output (and the LSE) as integers: NaN-safe equality."""
    if isinstance(result, tuple):
        out, lse = result
        return out.view(torch.int16).clone(), lse.contiguous().view(torch.int32).clone()
    return (result.view(torch.int16).clone(),)


def _reducers(rows):
    """The kernel names of the slice and the head build that this device's FP8
    page-64 module launches for ``rows`` query rows."""
    arch, num_sms = cake_module._device_facts(torch.cuda.current_device())
    family = cake_module.FAMILIES[
        cake_module.launch_family(cake_module.DOMAIN_FP8_PAGE64, arch, num_sms, rows)
    ]
    assert family["binding"] == cake_module._FP8_PAGE64_BINDING
    return f"kernel_{family['dependencies'][1]}", f"kernel_{family['dependencies'][2]}"


def _launched_reducers(call, names):
    """The reducer kernels among ``names`` that ``call`` launches, from the
    profiler's device activities; ``None`` when the profiler records no device
    activity (a CUPTI client such as compute-sanitizer is attached), so the
    caller still compares the bits."""
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        result = call()
        torch.cuda.synchronize()
    launched = [
        event.name
        for event in prof.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
    ]
    if not launched:
        warnings.warn(
            "torch.profiler captured no device events: the launched reducer is "
            "not checked",
            stacklevel=2,
        )
        return result, None
    return result, sorted({name for name in launched if name in names})


@pytest.fixture
def forced_ctas(monkeypatch):
    """Force the reducer CTA count the launch takes (``None``: the rule's own);
    records ``(rows, splits, CTAs)`` of every planned launch."""
    used: list = []
    plan = cake_module.plan_fp8_reduce_ctas

    def force(value):
        def spy(work_items, num_split, num_sms, arch):
            ctas = (
                plan(work_items, num_split, num_sms, arch) if value is None else value
            )
            used.append((work_items, num_split, ctas))
            return ctas

        monkeypatch.setattr(cake_module, "plan_fp8_reduce_ctas", spy)
        return used

    return force


def _expected_head(rows, num_split):
    """Whether this device's rule takes the head build (``fp8_head_reduce`` at
    the head CTA count, restated)."""
    arch, num_sms = cake_module._device_facts(torch.cuda.current_device())
    return (
        rows <= cake_module._FP8_HEAD_REDUCE_MAX_ROWS.get(arch, 0)
        and num_split >= 8 * rows
        and _head_bound(num_split, min(128, num_sms // rows))
    )


def _check_runs(decode, rows, ctas_values, forced_ctas, names, label):
    """Run ``decode`` at the rule's CTA count and at each forced one; every run
    launches the build ``fp8_head_reduce`` predicts (``names``: the slice and
    the head build's kernels) and writes the first run's bits. Returns the
    predicted build of the rule's run."""
    arch, _ = cake_module._device_facts(torch.cuda.current_device())
    slice_name, head_name = names
    first = None
    natural_head = None
    for value in (None, *ctas_values):
        used = forced_ctas(value)
        used.clear()
        result, launched = _launched_reducers(decode, set(names))
        assert len(used) == 1, (label, value, used)
        work_items, num_split, ctas = used[0]
        assert work_items == rows and num_split >= 2, (label, used)
        is_head = cake_module.fp8_head_reduce(rows, num_split, ctas, arch)
        if launched is not None:
            assert launched == [head_name if is_head else slice_name], (
                label,
                value,
                ctas,
                launched,
            )
        bits = _bits(result)
        if first is None:
            first, natural_head = bits, is_head
            continue
        for index, (got, want) in enumerate(zip(bits, first, strict=True)):
            assert torch.equal(got, want), (label, value, ctas, index)
    return natural_head


# (batch, q_len): one query row, two query rows of one request, two requests.
_SHAPES = ((1, 1), (1, 2), (2, 1))
_LENGTHS = (384, 1024, 4096, 32768, 131072)


@pytest.mark.parametrize("return_lse", [False, True])
@pytest.mark.parametrize("lengths", ["uniform", "varied"])
@pytest.mark.parametrize("max_seq_len", _LENGTHS)
@pytest.mark.parametrize("batch,q_len", _SHAPES)
def test_fp8_head_reduce_matches_shipped_bits(
    fp8_domains, forced_ctas, batch, q_len, max_seq_len, lengths, return_lse
):
    """At the rule's reducer CTA count and at 64, 7 (where a CTA's partial rows
    fit its stage) and 1 (never the head build) CTAs, the output and the LSE are
    the same bits; each run launches the reducer build ``fp8_head_reduce``
    predicts, and the 1-CTA run the slice build."""
    _require_blackwell()
    rows = batch * q_len
    _, num_sms = cake_module._device_facts(torch.cuda.current_device())
    num_split = cake_module.plan_num_split(rows, max_seq_len, num_sms)
    assert num_split >= 2
    query, kv_cache, block_tables, seq_lens = _inputs(
        batch, q_len, max_seq_len, lengths
    )
    label = f"b{batch} q{q_len} kv{max_seq_len} {lengths} lse{int(return_lse)}"
    forced = [64]
    if _head_bound(num_split, 7):
        forced.append(7)
    forced.append(1)

    def decode():
        return _cake(
            query,
            kv_cache,
            block_tables,
            seq_lens,
            max_seq_len,
            out=_nan_out(query),
            return_lse=return_lse,
        )

    names = _reducers(rows)
    natural_head = _check_runs(decode, rows, forced, forced_ctas, names, label)
    assert natural_head == _expected_head(rows, num_split), label
    assert set(fp8_domains) == {cake_module.DOMAIN_FP8_PAGE64}


_FORCED_SPLITS = (2, 3, 5, 9, 17, 33, 37)


@pytest.mark.parametrize("num_split", _FORCED_SPLITS)
@pytest.mark.parametrize("batch,q_len", [(1, 1), (2, 1)])
def test_fp8_head_reduce_forced_split_counts_match_shipped_bits(
    monkeypatch, fp8_domains, forced_ctas, batch, q_len, num_split
):
    """KV 32768 at one and two query rows with the split count forced to 2 to
    37 (one producer wave at two rows): the rule's reducer CTA count and 1 CTA
    (the slice build) write the same output and LSE bits."""
    _require_blackwell()
    rows = batch * q_len
    max_seq_len = 32768
    monkeypatch.setattr(
        cake_module, "plan_num_split", lambda work_items, length, sms: num_split
    )
    query, kv_cache, block_tables, seq_lens = _inputs(
        batch, q_len, max_seq_len, "varied"
    )
    label = f"b{batch} q{q_len} split{num_split}"

    def decode():
        return _cake(
            query,
            kv_cache,
            block_tables,
            seq_lens,
            max_seq_len,
            out=_nan_out(query),
            return_lse=True,
        )

    names = _reducers(rows)
    natural_head = _check_runs(decode, rows, [1], forced_ctas, names, label)
    assert natural_head == _expected_head(rows, num_split), label
    assert set(fp8_domains) == {cake_module.DOMAIN_FP8_PAGE64}


# name: (batch, q_len, max_seq_len)
_DETERMINISM_CASES = {
    "rows1_kv32768": (1, 1, 32768),
    "rows2_q_len2_kv4096": (1, 2, 4096),
}


@pytest.mark.parametrize("case", sorted(_DETERMINISM_CASES))
def test_fp8_head_reduce_is_deterministic(fp8_domains, forced_ctas, case):
    """200 eager calls and 1000 replays of a captured graph give the first
    call's bits; the stateless mode gives them too. The first call, eager,
    launches the build the host predicts (the head build where the device's
    rule takes it) and opts it into its shared memory before the capture."""
    _require_blackwell()
    batch, q_len, max_seq_len = _DETERMINISM_CASES[case]
    rows = batch * q_len
    arch, num_sms = cake_module._device_facts(torch.cuda.current_device())
    num_split = cake_module.plan_num_split(rows, max_seq_len, num_sms)
    # Every split of the request is busy: the bits compared cover every split.
    query, kv_cache, block_tables, seq_lens = _inputs(
        batch, q_len, max_seq_len, "uniform"
    )

    def decode(output, **kwargs):
        return _cake(
            query, kv_cache, block_tables, seq_lens, max_seq_len, out=output, **kwargs
        )

    used = forced_ctas(None)
    slice_name, head_name = _reducers(rows)
    first, launched = _launched_reducers(
        lambda: decode(_nan_out(query)), {slice_name, head_name}
    )
    first = first.clone()
    is_head = cake_module.fp8_head_reduce(rows, num_split, used[-1][2], arch)
    assert is_head == _expected_head(rows, num_split), case
    if launched is not None:
        assert launched == [head_name if is_head else slice_name], (case, launched)
    assert bool(first.isfinite().all())
    for call in range(200):
        assert torch.equal(decode(_nan_out(query)), first), call
    assert torch.equal(decode(_nan_out(query), stateless=True), first)
    out = _nan_out(query)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        decode(out)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        decode(out)
    mismatches = torch.zeros((), dtype=torch.int64, device="cuda")
    for _ in range(1000):
        out.fill_(float("nan"))
        graph.replay()
        mismatches += (out != first).any()
    torch.cuda.synchronize()
    assert int(mismatches) == 0
    assert set(fp8_domains) == {cake_module.DOMAIN_FP8_PAGE64}
