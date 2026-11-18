"""The batch-invariant sparse MLA classes ``mla.sparse.<dtype>.d<576|512>`` (SM100 / SM103 / SM107).

Bitwise only. The quick tier runs by default; ``FLASHINFER_MLA_BI_NIGHTLY=1`` adds the larger batches, the
prefill-sized calls, more head counts and the remaining dtype / layout pairs. The device-free tests (registry,
fingerprint, list arithmetic, Lean cases) run everywhere.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import warnings

import pytest
import torch

import flashinfer.batch_invariant as bi
from flashinfer.batch_invariant.mla import cases, lean_cases, spec
from flashinfer.batch_invariant.mla.classes import OPERATORS

warnings.filterwarnings("ignore", message=".*experimental.*")

NIGHTLY = os.environ.get("FLASHINFER_MLA_BI_NIGHTLY", "0") == "1"
needs_device = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not bi.is_supported_device(torch.device("cuda"), spec.MLA_ARCHS),
    reason="needs a CUDA device of SM100 / SM103 / SM107 with the CuTe DSL",
)
LAKE = shutil.which("lake") or os.path.expanduser("~/.elan/bin/lake")
needs_lake = pytest.mark.skipif(
    not os.path.exists(LAKE), reason="Lean toolchain (elan / lake) not installed"
)

SCALE = cases.SOFTMAX_SCALE
TOPK = 2048
#: (dtype, head_dim, heads): GLM-5.3 (576 RoPE rows, 64 heads and the 8-head TP 8 rank) and GLM-5.3-Flash (512 NoPE
#: rows, 16 heads per TP 4 rank).
CONFIGS = [
    ("e4m3", 576, 64),
    ("e4m3", 576, 8),
    ("e4m3", 512, 16),
    ("bf16", 576, 8),
    ("bf16", 512, 16),
]
if NIGHTLY:
    CONFIGS += [
        ("bf16", 576, 64),
        ("e4m3", 512, 64),
        ("bf16", 512, 64),
        ("e4m3", 576, 1),
        ("e4m3", 576, 5),
        ("bf16", 512, 32),
        ("e4m3", 576, 128),
    ]
config = pytest.mark.parametrize(
    "cfg", CONFIGS, ids=lambda c: f"{c[0]}-d{c[1]}-h{c[2]}"
)
#: The requests of the shared pool: (seed, tokens). Request 0 is long, 1 is shorter than one tile, 2 ends inside the
#: selected width, 3 holds exactly top-k tokens plus the Flash tail.
REQUESTS = [(11, 6000), (12, 150), (13, 1500), (14, 2051), (15, 3000), (16, 40)]
#: Target rows: (request, visible tokens). Lists of 1, 2, 127, 128, 129 and 300 keys, full lists, and a row deep in
#: the prompt.
TARGETS = [
    (1, 1),
    (1, 2),
    (1, 127),
    (2, 128),
    (2, 129),
    (2, 300),
    (2, 1500),
    (3, 2047),
    (3, 2048),
    (0, 2049),
    (0, 6000),
    (4, 3000),
]


def same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    return (
        a.shape == b.shape
        and a.dtype == b.dtype
        and torch.equal(
            a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
        )
    )


def make_wrapper(cfg, rows, width=TOPK, **plan):
    from flashinfer.attention.prims_ts import BatchSparseMLADecodePagedTSWrapper

    dtype, head_dim, heads = cfg
    wrapper = BatchSparseMLADecodePagedTSWrapper()
    plan.setdefault("softmax_scale", SCALE)
    plan.setdefault("batch_invariant", True)
    plan.setdefault("return_lse", True)
    batch_size = plan.pop("batch_size", rows)
    wrapper.plan(
        torch.device("cuda"),
        batch_size,
        heads,
        max_topk=width,
        q_dtype=cases.torch_dtype(dtype),
        qk_rope_head_dim=head_dim - 512,
        **plan,
    )
    return wrapper


class Rows:
    """Query rows over one pool: ``q [n, H, D]``, ``indices [n, width]``, ``lengths [n]``."""

    def __init__(self, cfg, pool, targets, width=TOPK, topk=TOPK):
        dtype, head_dim, heads = cfg
        self.cfg, self.pool, self.width = cfg, pool, width
        rows = [pool.row(r, visible, topk, width) for r, visible in targets]
        self.q = torch.stack(
            [pool.query(r, visible - 1, heads) for r, visible in targets]
        )
        self.indices = torch.stack([row for row, _ in rows]).cuda()
        self.lengths = torch.tensor([n for _, n in rows], dtype=torch.int32).cuda()

    def __len__(self):
        return self.q.shape[0]

    def take(self, order):
        picked = object.__new__(Rows)
        picked.cfg, picked.pool, picked.width = self.cfg, self.pool, self.width
        order = torch.as_tensor(order, device="cuda")
        picked.q = self.q[order].contiguous()
        picked.indices = self.indices[order].contiguous()
        picked.lengths = self.lengths[order].contiguous()
        return picked


def run_batch(rows, wrapper=None, **run):
    """One fixed-query call with one query token per request: ``(out [n, H, 512], lse [n, H])``."""
    _, head_dim, heads = rows.cfg
    n = len(rows)
    wrapper = wrapper or make_wrapper(rows.cfg, n, rows.width)
    out, lse = wrapper.run(
        rows.q.view(n, 1, heads, head_dim),
        rows.pool.kv,
        rows.indices,
        rows.lengths,
        **run,
    )
    return out.view(n, heads, 512), lse.view(n, heads)


def run_alone(rows):
    """Every row as its own one-row decode call: the reference of every comparison."""
    wrapper = make_wrapper(rows.cfg, 1, rows.width)
    outs = [run_batch(rows.take([i]), wrapper) for i in range(len(rows))]
    return torch.cat([o for o, _ in outs]), torch.cat([l for _, l in outs])


_SHARED = {}


def shared(cfg):
    """The pool, the target rows and their one-row reference bits of a configuration (built once)."""
    if cfg not in _SHARED:
        dtype, head_dim, _ = cfg
        pool = cases.Pool(REQUESTS, head_dim, dtype, "cuda")
        targets = Rows(cfg, pool, TARGETS)
        _SHARED[cfg] = (pool, targets, run_alone(targets))
    return _SHARED[cfg]


def mates(cfg, pool, count, kind):
    """Neighbour rows: ``short`` lists, ``long`` (full) lists or a mix."""
    spans = {"short": [(5, 40), (1, 150)], "long": [(0, 6000), (4, 3000)]}
    spans["mixed"] = spans["short"] + spans["long"] + [(2, 700)]
    picks = [spans[kind][i % len(spans[kind])] for i in range(count)]
    return Rows(
        cfg,
        pool,
        [(r, max(1, visible - 3 * i)) for i, (r, visible) in enumerate(picks)],
    )


def join(*parts):
    joined = object.__new__(Rows)
    joined.cfg, joined.pool, joined.width = parts[0].cfg, parts[0].pool, parts[0].width
    joined.q = torch.cat([p.q for p in parts])
    joined.indices = torch.cat([p.indices for p in parts])
    joined.lengths = torch.cat([p.lengths for p in parts])
    return joined


# --------------------------------------------------------------------------------------------------------------------
# Device-free: registry, fingerprint, list arithmetic, Lean cases
# --------------------------------------------------------------------------------------------------------------------


def test_classes_are_registered():
    assert spec.class_names() == [
        "mla.sparse.e4m3.d576",
        "mla.sparse.e4m3.d512",
        "mla.sparse.bf16.d576",
        "mla.sparse.bf16.d512",
    ]
    prints = set()
    for op in OPERATORS:
        assert bi.class_name(op) == spec.FAMILY
        for dtype in (torch.float8_e4m3fn, torch.bfloat16):
            for head_dim in spec.HEAD_DIMS:
                fp = bi.fingerprint(op, dtype=dtype, head_dim=head_dim)
                name = spec.class_name(dtype, head_dim)
                assert fp.startswith(name.replace(".", "-") + "-")
                assert fp == spec.spec_fingerprint(spec.dtype_name(dtype), head_dim)
                prints.add(fp)
    assert len(prints) == 4
    with pytest.raises(TypeError):
        bi.fingerprint(OPERATORS[0], dtype="e4m3", head_dim=576, heads=64)
    with pytest.raises(ValueError):
        bi.fingerprint(OPERATORS[0], dtype="e4m3", head_dim=320)
    with pytest.raises(ValueError):
        bi.fingerprint(OPERATORS[0], dtype=torch.float16, head_dim=576)


def test_fingerprint_covers_the_kernel_sources():
    files = spec.term_source_files()
    for needed in (
        "mla_decode/sparse_policy.py",
        "mla_decode/throughput_2cta/kernel.py",
        "mla_decode/throughput_2cta/resources.py",
        "mla_decode/helpers/sparse_indices.py",
        "mla_decode/helpers/math.py",
    ):
        assert needed in files
    assert len(set(files)) == len(files)


def test_pinned_profile_is_the_policy_profile():
    policy = pytest.importorskip(
        "flashinfer.attention.prims_ts.kernels.mla_decode.sparse_policy"
    )
    default = policy.SparseMlaProfile(family="2cta")
    for dtype, key in (("e4m3", "fp8"), ("bf16", "bf16")):
        profile = policy.batch_invariant_sparse_mla_profile(dtype=key)
        changed = {
            f: getattr(profile, f)
            for f in profile.__dataclass_fields__
            if f == "family" or getattr(profile, f) != getattr(default, f)
        }
        expected = dict(spec.PROFILES[dtype])
        for field in ("split_kv", "head_dim_ctas"):
            assert getattr(profile, field) == expected.pop(field) == 1
        assert changed == expected
        assert not profile.defer_max_update
    with pytest.raises(ValueError):
        policy.batch_invariant_sparse_mla_profile(dtype="fp8", kv_cache_format="nvfp4")


def test_list_arithmetic():
    assert spec.tile_bounds(0) == [] and spec.key_tiles(1) == 1
    for length in (1, 127, 128, 129, 2048, 2051):
        bounds = spec.tile_bounds(length)
        assert bounds[0][0] == 0 and bounds[-1][0] < length <= bounds[-1][1]
        assert all(b - a == spec.KEY_TILE for a, b in bounds)
        assert [a for a, _ in bounds] == sorted(a for a, _ in bounds)
    assert spec.list_is_packed([3, 9, 4, -1, -1], 3)
    assert not spec.list_is_packed([3, -1, 4, -1], 3)
    assert spec.list_is_packed([-1, -1], 0)


def test_case_inputs_are_exact_and_reproducible():
    v = cases.values(4096, 5)
    assert torch.equal(v, cases.values(4096, 5)) and not torch.equal(
        v, cases.values(4096, 6)
    )
    assert torch.equal(v * 8, (v * 8).round()) and float(v.abs().max()) <= 2.0
    assert torch.equal(v.to(torch.bfloat16).float(), v)
    assert hashlib.sha256(v.numpy().tobytes()).hexdigest()[:16] == "f88fc2160fcbff43"
    picked = cases.selection(7, 6000, 2048)
    assert len(picked) == 2048 and picked == sorted(set(picked)) and picked[-1] < 6000
    assert cases.selection(7, 5, 2048) == [0, 1, 2, 3, 4]
    assert len(cases.DIGESTS) == len(cases.DIGEST_CASES) * len(cases.DIGEST_ROWS)


def test_generated_lean_cases_are_current():
    with open(lean_cases.CASES_PATH) as f:
        committed = f.read()
    assert committed == lean_cases.render(), (
        "regenerate with: python -m flashinfer.batch_invariant.mla.lean_cases"
    )
    assert "sorry" not in committed and "native_decide" not in committed


def test_placement_mirror_matches_the_lean_statements():
    """The Python mirror of ``MlaSparse.lean`` (``call_nth``, ``chunked_eq_call``, ``grouped_eq_call``,
    ``padded_prefix``) on a symbolic row term: a check of the mirror, not of the kernel."""
    term = lambda q, keys: ("row", q, tuple(keys))
    rows = [(f"q{i}", [f"k{i}.{j}" for j in range(i % 4)]) for i in range(9)]
    alone = [lean_cases.place_decode(term, row) for row in rows]
    assert lean_cases.place_batch(term, rows) == alone
    assert lean_cases.place_batch(term, rows[::-1]) == alone[::-1]
    assert lean_cases.place_prefill(term, rows) == alone
    for cut in range(len(rows) + 1):
        assert lean_cases.place_chunked(term, rows, [cut, len(rows) - cut]) == alone
    assert lean_cases.place_mtp(term, [rows[:2], rows[2:3], rows[3:]]) == [
        alone[:2],
        alone[2:3],
        alone[3:],
    ]
    padded = rows[:3] + [("pad", [])] * 5
    assert lean_cases.place_batch(term, padded)[:3] == alone[:3]


def test_unsupported_device_raises_before_any_kernel():
    from flashinfer.batch_invariant.mla import api

    with pytest.raises(bi.BatchInvariantUnsupportedError):
        api.check_plan(
            torch.device("cpu"),
            num_qo_heads=8,
            q_dtype=torch.bfloat16,
            qk_rope_head_dim=64,
            softmax_scale=SCALE,
            refused={},
        )


@needs_lake
def test_lean_model_matches_python_spec():
    env = dict(
        os.environ,
        ELAN_NO_OVERRIDE_NOTICE="1",
        PATH=os.path.dirname(LAKE) + os.pathsep + os.environ["PATH"],
    )

    def lake(*args, timeout):
        return subprocess.run(
            [LAKE, *args],
            cwd=lean_cases.LEAN_DIR,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    r = lake("build", "BIKernels.MlaSparse", "BIKernels.MlaSparseCases", timeout=3600)
    assert r.returncode == 0, "Lean proofs failed:\n" + (r.stdout + r.stderr)[-4000:]
    r = lake("build", "mlasparseconformance", timeout=1800)
    assert r.returncode == 0, (r.stdout + r.stderr)[-4000:]
    r = lake("exe", "mlasparseconformance", timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    lean = [line for line in r.stdout.splitlines() if line.startswith("mla")]
    assert lean == lean_cases.conformance_lines()


# --------------------------------------------------------------------------------------------------------------------
# Device: the flag selects the pinned schedule and leaves the default path alone
# --------------------------------------------------------------------------------------------------------------------


@needs_device
def test_flag_selects_one_schedule_for_every_plan():
    from flashinfer.attention.prims_ts import mla_decode
    from flashinfer.attention.prims_ts.kernels.mla_decode import sparse_policy

    for dtype, key in ((torch.float8_e4m3fn, "fp8"), (torch.bfloat16, "bf16")):
        pinned = sparse_policy.batch_invariant_sparse_mla_profile(dtype=key)
        seen_default = set()
        for rope in (64, 0):
            for heads in (8, 16, 64, 128):
                for rows, q_len in (
                    (1, 1),
                    (8, 1),
                    (64, 1),
                    (512, 1),
                    (1, 4096),
                    (4, 8),
                ):
                    kw = dict(
                        max_topk=TOPK,
                        max_q_len=q_len,
                        q_dtype=dtype,
                        qk_rope_head_dim=rope,
                    )
                    plan = mla_decode._resolve_sparse_mla_plan(
                        "cuda",
                        rows,
                        heads,
                        softmax_scale=SCALE,
                        batch_invariant=True,
                        **kw,
                    )
                    assert plan[1] == pinned and plan[2].split_kv == 1
                    seen_default.add(
                        mla_decode._resolve_sparse_mla_plan("cuda", rows, heads, **kw)[
                            1
                        ]
                    )
        assert len(seen_default) > 1, "the default path still plans by extent"


@needs_device
def test_planned_kernel_state_is_the_recorded_state():
    for cfg in (("e4m3", 576, 64), ("bf16", 512, 16)):
        kernel = make_wrapper(cfg, 3)._state["compile_spec"].kernel
        got = {name: getattr(kernel, name) for name in spec.KERNEL_STATE}
        assert got == spec.KERNEL_STATE, cfg
        assert kernel.assume_valid_prefix is False
        assert (
            make_wrapper(cfg, 3, assume_valid_prefix=True)
            ._state["compile_spec"]
            .kernel.assume_valid_prefix
        )


@needs_device
def test_default_schedule_is_not_batch_invariant():
    """Negative control: without the flag the same comparison fails, so the invariance tests can fail."""
    cfg = ("e4m3", 576, 64)
    pool, targets, _ = shared(cfg)
    full = targets.take([10, 11])
    alone = [
        run_batch(full.take([i]), make_wrapper(cfg, 1, batch_invariant=False))[0]
        for i in range(2)
    ]
    batch = join(full, mates(cfg, pool, 62, "long"))
    out = run_batch(batch, make_wrapper(cfg, 64, batch_invariant=False))[0]
    assert not same_bits(out[:2], torch.cat(alone))


@needs_device
@config
def test_matches_float64_reference(cfg):
    """Sanity of the recorded rows (not the acceptance criterion): close to an fp64 softmax over the same list."""
    dtype, head_dim, heads = cfg
    pool, targets, (out, lse) = shared(cfg)
    flat = pool.kv.view(-1, head_dim).float().double()
    for i in range(len(targets)):
        n = int(targets.lengths[i])
        keys = flat[targets.indices[i, :n].long()]
        logits = targets.q[i].float().double() @ keys.T * SCALE
        ref = torch.softmax(logits, -1) @ keys[:, :512]
        ref_lse = torch.logsumexp(logits, -1)
        tol = 0.05 if dtype == "e4m3" else 0.01
        assert float((out[i].float().double() - ref).abs().max()) < tol
        assert float((lse[i].double() - ref_lse).abs().max()) < 1e-3 * max(
            1.0, float(ref_lse.abs().max())
        )


# --------------------------------------------------------------------------------------------------------------------
# R1: determinism (reruns, re-plan, fresh processes, CUDA-graph replay)
# --------------------------------------------------------------------------------------------------------------------


@needs_device
@config
def test_reruns_and_replans_repeat(cfg):
    pool, targets, (ref_out, ref_lse) = shared(cfg)
    batch = join(targets, mates(cfg, pool, 5, "mixed"))
    wrapper = make_wrapper(cfg, len(batch))
    first = run_batch(batch, wrapper)
    for _ in range(3):
        again = run_batch(batch, wrapper)
        assert same_bits(again[0], first[0]) and same_bits(again[1], first[1])
    replanned = run_batch(batch)
    assert same_bits(replanned[0], first[0]) and same_bits(replanned[1], first[1])
    assert same_bits(first[0][: len(targets)], ref_out)
    assert same_bits(first[1][: len(targets)], ref_lse)


def _digest_cases():
    return cases.run_digest_cases("cuda")


@needs_device
def test_recorded_digests():
    """One table for SM100, SM103 and SM107: the bits do not depend on the architecture."""
    got = _digest_cases()
    wrong = sorted(k for k in cases.DIGESTS if got.get(k) != cases.DIGESTS[k])
    assert not wrong and set(got) == set(cases.DIGESTS), wrong


@needs_device
def test_outputs_repeat_across_fresh_processes():
    runs = [
        subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--digest"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for _ in range(2)
    ]
    first, second = (json.loads(r.strip().splitlines()[-1]) for r in runs)
    assert first == second == cases.DIGESTS


@needs_device
@config
def test_graph_replay_equals_eager(cfg):
    _, head_dim, heads = cfg
    pool, targets, (ref_out, ref_lse) = shared(cfg)
    n = 8
    batch = targets.take(range(n))
    wrapper = make_wrapper(cfg, n)
    q = batch.q.view(n, 1, heads, head_dim).clone()
    indices, lengths = batch.indices.clone(), batch.lengths.clone()
    out = torch.empty(n, 1, heads, 512, dtype=torch.bfloat16, device="cuda")
    lse = torch.empty(n, 1, heads, dtype=torch.float32, device="cuda")
    call = lambda: wrapper.run(
        q, pool.kv, indices, lengths, out=out, lse=lse, validate=False
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            call()
    torch.cuda.synchronize()
    for order in (range(n), range(len(targets) - n, len(targets)), [0] * n):
        order = list(order)
        nxt = targets.take(order)
        q.copy_(nxt.q.view(n, 1, heads, head_dim))
        indices.copy_(nxt.indices)
        lengths.copy_(nxt.lengths)
        out.zero_()
        lse.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert same_bits(out.view(n, heads, 512), ref_out[order])
        assert same_bits(lse.view(n, heads), ref_lse[order])
    lengths[n // 2 :] = 0
    graph.replay()
    torch.cuda.synchronize()
    assert same_bits(out.view(n, heads, 512)[: n // 2], ref_out[order][: n // 2])
    assert not out.view(n, heads, 512)[n // 2 :].float().any()


# --------------------------------------------------------------------------------------------------------------------
# R2: batch size, position, neighbours, zero-length padding rows
# --------------------------------------------------------------------------------------------------------------------


@needs_device
@config
def test_batch_size_position_and_neighbours_do_not_change_bits(cfg):
    pool, targets, (ref_out, ref_lse) = shared(cfg)
    n = len(targets)
    sizes = [2, 7, 64] + ([256, 512] if NIGHTLY else [])
    for size in sizes:
        for kind in ("short", "long", "mixed"):
            for place in ("first", "last", "spread"):
                if size < n:
                    picked = [(size * 5 + i * 3) % n for i in range(size - 1)]
                    part, others = targets.take(picked), mates(cfg, pool, 1, kind)
                else:
                    picked = list(range(n))
                    part, others = targets, mates(cfg, pool, size - n, kind)
                k, m = len(part), len(others)
                if place == "first":
                    batch, where = join(part, others), list(range(k))
                elif place == "last":
                    batch, where = join(others, part), list(range(m, m + k))
                else:
                    step = max(1, (k + m) // k)
                    where = [min(i * step, m + i) for i in range(k)]
                    order = [None] * (k + m)
                    for i, w in enumerate(where):
                        order[w] = i
                    free = iter(range(k, k + m))
                    order = [o if o is not None else next(free) for o in order]
                    batch = join(part, others).take(order)
                out, lse = run_batch(batch)
                assert same_bits(out[where], ref_out[picked]), (size, kind, place)
                assert same_bits(lse[where], ref_lse[picked]), (size, kind, place)


@needs_device
@config
def test_permutation_and_zero_length_padding_rows(cfg):
    pool, targets, (ref_out, ref_lse) = shared(cfg)
    n = len(targets)
    order = [(i * 5 + 3) % n for i in range(n)]
    out, lse = run_batch(targets.take(order))
    assert same_bits(out, ref_out[order]) and same_bits(lse, ref_lse[order])
    for bucket in (16, 64):
        pad = mates(cfg, pool, bucket - n, "long")
        pad.lengths.zero_()
        out, lse = run_batch(join(targets, pad))
        assert same_bits(out[:n], ref_out) and same_bits(lse[:n], ref_lse)
        assert not out[n:].float().any()


# --------------------------------------------------------------------------------------------------------------------
# R4: prefill-sized call == chunked == one-row decode == multi-token rows; the list as an input
# --------------------------------------------------------------------------------------------------------------------


def prefill_rows(cfg, pool, request, count):
    """The first ``count`` token rows of a request: token ``p`` sees ``[0, p]`` (short rows up to top-k)."""
    return Rows(cfg, pool, [(request, p + 1) for p in range(count)])


@needs_device
@config
def test_prefill_chunked_decode_and_multi_token_rows_agree(cfg):
    _, head_dim, heads = cfg
    pool, _, _ = shared(cfg)
    count = 2304 if NIGHTLY else 160
    rows = prefill_rows(cfg, pool, 0, count)
    if NIGHTLY:
        ref_out, ref_lse = run_batch(rows)
        probe = rows.take([0, 1, 127, 128, 129, 2047, 2048, count - 1])
        alone = run_alone(probe)
        where = [0, 1, 127, 128, 129, 2047, 2048, count - 1]
        assert same_bits(ref_out[where], alone[0]) and same_bits(
            ref_lse[where], alone[1]
        )
    else:
        ref_out, ref_lse = run_alone(rows)
    kv = pool.kv

    fixed = make_wrapper(cfg, count, batch_size=1, max_q_len=count)
    out, lse = fixed.run(
        rows.q.view(1, count, heads, head_dim), kv, rows.indices, rows.lengths
    )
    assert same_bits(out.view(count, heads, 512), ref_out)
    assert same_bits(lse.view(count, heads), ref_lse)

    packed = make_wrapper(cfg, count, batch_size=1, max_q_len=count, packed_query=True)
    ptr = torch.tensor([0, count], dtype=torch.int32, device="cuda")
    out, lse = packed.run(rows.q, kv, rows.indices, rows.lengths, qo_indptr=ptr)
    assert same_bits(out, ref_out) and same_bits(lse, ref_lse)

    start = 0
    for piece in (count // 3, 1, count - count // 3 - 1):
        chunk = rows.take(range(start, start + piece))
        ptr = torch.tensor([0, piece], dtype=torch.int32, device="cuda")
        out, lse = packed.run(chunk.q, kv, chunk.indices, chunk.lengths, qo_indptr=ptr)
        assert same_bits(out, ref_out[start : start + piece])
        assert same_bits(lse, ref_lse[start : start + piece])
        start += piece

    for tokens in (2, 4, 8):
        requests = count // tokens
        used = requests * tokens
        mtp = make_wrapper(cfg, used, batch_size=requests, max_q_len=tokens)
        out, lse = mtp.run(
            rows.q[:used].view(requests, tokens, heads, head_dim),
            kv,
            rows.indices[:used],
            rows.lengths[:used],
        )
        assert same_bits(out.view(used, heads, 512), ref_out[:used])
        assert same_bits(lse.view(used, heads), ref_lse[:used])

    ragged = make_wrapper(cfg, 12, batch_size=3, max_q_len=4, packed_query=True)
    ptr = torch.tensor([0, 1, 5, 8], dtype=torch.int32, device="cuda")
    part = rows.take(range(8))
    out, lse = ragged.run(part.q, kv, part.indices, part.lengths, qo_indptr=ptr)
    assert same_bits(out, ref_out[:8]) and same_bits(lse, ref_lse[:8])


@needs_device
@config
def test_list_width_and_tail_padding_do_not_change_bits(cfg):
    """Widths 2,048 to 2,052 for the same live list; a ``-1`` tail declared live; the Flash rows of 2,049 to 2,051."""
    pool, _, _ = shared(cfg)
    spans = [(1, 150), (2, 1500), (0, 6000), (3, 2047)]
    ref = run_alone(Rows(cfg, pool, spans))
    for width in (2049, 2050, 2051, 2052, 4096):
        rows = Rows(cfg, pool, spans, width=width)
        got = run_batch(rows)
        assert same_bits(got[0], ref[0]) and same_bits(got[1], ref[1]), width
        rows.lengths.fill_(width)
        got = run_batch(rows, validate=False)
        assert same_bits(got[0], ref[0]) and same_bits(got[1], ref[1]), width
    rows = Rows(cfg, pool, spans)
    promised = run_batch(rows, make_wrapper(cfg, len(rows), assume_valid_prefix=True))
    assert same_bits(promised[0], ref[0]) and same_bits(promised[1], ref[1])
    tails = [(3, 2049), (3, 2050), (3, 2051), (0, 6000)]
    wide = {w: Rows(cfg, pool, tails, width=w, topk=2051) for w in (2051, 2052)}
    ref = run_alone(wide[2051])
    assert [int(n) for n in wide[2051].lengths] == [2049, 2050, 2051, 2051]
    for width, rows in wide.items():
        batch = join(rows, mates_wide(cfg, pool, 5, width))
        got = run_batch(batch)
        assert same_bits(got[0][:4], ref[0]) and same_bits(got[1][:4], ref[1]), width


def mates_wide(cfg, pool, count, width):
    return Rows(
        cfg, pool, [(4, 3000 - 7 * i) for i in range(count)], width=width, topk=2051
    )


@needs_device
@config
def test_the_list_is_an_ordered_input(cfg):
    """Negative controls: another order or another key changes the row (the tests above can fail)."""
    pool, _, _ = shared(cfg)
    rows = Rows(cfg, pool, [(0, 6000)])
    ref = run_alone(rows)
    flipped = rows.take([0])
    flipped.indices[0, :TOPK] = rows.indices[0, :TOPK].flip(0)
    assert not same_bits(run_alone(flipped)[0], ref[0])
    other = Rows(cfg, pool, [(0, 5999)])
    other.q = rows.q
    assert not same_bits(run_alone(other)[0], ref[0])


# --------------------------------------------------------------------------------------------------------------------
# Adversarial rows
# --------------------------------------------------------------------------------------------------------------------


def special_values(dtype):
    if dtype == "bf16":
        raw = [
            float("nan"),
            float("inf"),
            float("-inf"),
            -0.0,
            9.1835e-41,
            -9.1835e-41,
            3.3e38,
        ]
        return torch.tensor(raw, dtype=torch.float32).to(torch.bfloat16)
    codes = [0x7F, 0xFF, 0x80, 0x01, 0x81, 0x7E, 0xFE]
    return torch.tensor(codes, dtype=torch.uint8).view(torch.float8_e4m3fn)


@needs_device
@config
def test_special_values_in_one_row_do_not_reach_the_others(cfg):
    """NaN, infinities, signed zero, subnormals and the largest magnitudes in a neighbour's query and in the cache
    rows only the neighbour selects leave every other row's bits unchanged, and the neighbour itself repeats."""
    dtype, head_dim, heads = cfg
    pool, targets, (ref_out, ref_lse) = shared(cfg)
    n = len(targets)
    specials = special_values(dtype).cuda()
    poison_pool = cases.Pool(REQUESTS + [(99, 400)], head_dim, dtype, "cuda")
    assert same_bits(
        poison_pool.kv.view(-1, head_dim)[poison_pool.slots[0].long().cuda()],
        pool.kv.view(-1, head_dim)[pool.slots[0].long().cuda()],
    )
    victims = Rows(cfg, poison_pool, TARGETS)
    bad = Rows(cfg, poison_pool, [(6, 400), (6, 200), (6, 1)])
    own = poison_pool.slots[6].long().cuda()
    flat = poison_pool.kv.view(-1, head_dim)
    for i in range(specials.numel()):
        flat[own[i :: specials.numel()], (i * 37) % head_dim] = specials[i]
        flat[own[i :: 3 * specials.numel()], 512 - 1 - i] = specials[i]
    for i in range(specials.numel()):
        bad.q[i % 2, i % heads, (i * 53) % head_dim] = specials[i]
    for order in ([0, 1, 2], [2, 0, 1]):
        batch = join(victims, bad.take(order))
        first = run_batch(batch)
        assert same_bits(first[0][:n], ref_out) and same_bits(first[1][:n], ref_lse)
        again = run_batch(join(bad.take(order), victims))
        assert same_bits(again[0][3:], ref_out) and same_bits(again[1][3:], ref_lse)
        assert same_bits(again[0][:3], first[0][n:]) and same_bits(
            again[1][:3], first[1][n:]
        )
    alone = run_alone(bad)
    assert same_bits(alone[0], first[0][n:][[1, 2, 0]]) and same_bits(
        alone[1], first[1][n:][[1, 2, 0]]
    )


# --------------------------------------------------------------------------------------------------------------------
# The stable function
# --------------------------------------------------------------------------------------------------------------------


def stable(rows, **kw):
    import flashinfer

    _, head_dim, heads = rows.cfg
    n = len(rows)
    args = dict(
        query=rows.q.view(n, 1, heads, head_dim),
        kv_cache=rows.pool.kv.unsqueeze(1),
        workspace_buffer=torch.zeros(1024, dtype=torch.int8, device="cuda"),
        qk_nope_head_dim=192,
        kv_lora_rank=512,
        qk_rope_head_dim=head_dim - 512,
        block_tables=rows.indices.view(n, 1, rows.width),
        seq_lens=torch.full((n,), 1, dtype=torch.int32, device="cuda"),
        max_seq_len=1,
        sparse_mla_top_k=rows.width,
        bmm1_scale=SCALE,
        bmm2_scale=1.0,
        batch_invariant=True,
    )
    args.update(kw)
    return flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(**args)


@needs_device
@config
def test_stable_function_runs_the_class(cfg):
    _, head_dim, heads = cfg
    pool, targets, (ref_out, ref_lse) = shared(cfg)
    n = len(targets)
    out = stable(targets, sparse_mla_top_k_lens=targets.lengths)
    assert same_bits(out.view(n, heads, 512), ref_out)
    out = stable(targets)
    assert same_bits(out.view(n, heads, 512), ref_out)
    out, lse = stable(
        targets,
        sparse_mla_top_k_lens=targets.lengths,
        return_lse=True,
        return_lse_base="basee",
    )
    assert same_bits(out.view(n, heads, 512), ref_out) and same_bits(
        lse.view(n, heads), ref_lse
    )
    given = torch.empty(n, heads, 512, dtype=torch.bfloat16, device="cuda")
    ptr = torch.arange(n + 1, dtype=torch.int32, device="cuda")
    got = stable(
        targets,
        query=targets.q,
        block_tables=targets.indices,
        cum_seq_lens_q=ptr,
        max_q_len=1,
        out=given,
        sparse_mla_top_k_lens=targets.lengths,
    )
    assert got is given and same_bits(given, ref_out)
    for width in (2051, 2052):
        wide = Rows(cfg, pool, TARGETS, width=width)
        assert same_bits(
            stable(wide, sparse_mla_top_k_lens=wide.lengths).view(n, heads, 512),
            ref_out,
        )
    scale = torch.tensor([SCALE], dtype=torch.float32, device="cuda")
    assert same_bits(stable(targets, bmm1_scale=scale).view(n, heads, 512), ref_out)
    mtp = targets.take(range(8))
    out = stable(
        mtp,
        query=mtp.q.view(2, 4, heads, head_dim),
        block_tables=mtp.indices.view(2, 4, TOPK),
        sparse_mla_top_k_lens=mtp.lengths,
    )
    assert same_bits(out.view(8, heads, 512), ref_out[:8])


@needs_device
def test_stable_function_refuses_inputs_outside_the_class(monkeypatch):
    import flashinfer

    cfg = CONFIGS[1]
    _, targets, (ref_out, _) = shared(cfg)
    rows = targets.take(range(2))
    sink = torch.zeros(cfg[2], dtype=torch.float32, device="cuda")
    counter = torch.zeros(64, dtype=torch.uint8, device="cuda")
    refused = [
        (dict(backend="trtllm-gen"), 'backend="trtllm-gen"'),
        (dict(backend="cake"), 'backend="cake"'),
        (dict(backend="cute-dsl"), 'backend="cute-dsl"'),
        (dict(sparse_mla_top_k=0), "sparse_mla_top_k=0"),
        (dict(sinks=[sink]), "sinks="),
        (dict(enable_dcp=True), "enable_dcp=True"),
        (dict(cp_world=2), "cp_world=2"),
        (dict(cp_rank=1), "cp_rank=1"),
        (dict(return_lse=True), "return_lse_base=None"),
        (dict(return_lse=True, return_lse_base="base2"), "return_lse_base='base2'"),
        (dict(skip_softmax_threshold_scale_factor=1.0), "skip_softmax"),
        (dict(use_fp16_softmax=True), "use_fp16_softmax"),
        (dict(uses_shared_paged_kv_idx=False), "uses_shared_paged_kv_idx"),
        (dict(multi_ctas_kv_counter_buffer=counter), "multi_ctas_kv_counter_buffer"),
        (dict(cute_dsl_impl="modular"), "cute_dsl_impl"),
        (dict(kv_scale_format="arbitrary_fp32"), "kv_scale_format"),
        (dict(kv_cache=rows.pool.kv.view(torch.uint8).unsqueeze(1)), "KV cache"),
        (dict(kv_lora_rank=256), "kv_lora_rank=256"),
        (dict(bmm1_scale=0.0), "bmm1_scale=0.0"),
        (dict(bmm1_scale=None), "bmm1_scale=None"),
        (
            dict(
                out=torch.empty(2, 1, cfg[2], 512, dtype=torch.float32, device="cuda")
            ),
            "out of shape",
        ),
        (dict(block_tables=rows.indices.view(2, 1, TOPK)[:, :, :1024]), "block_tables"),
    ]
    for kw, fragment in refused:
        with pytest.raises(bi.BatchInvariantUnsupportedError, match=fragment):
            stable(rows, **kw)
    with pytest.raises(TypeError):
        stable(rows, batch_invariant=1)

    holes = targets.take([5, 6])
    holes.indices[1, 7] = -1
    assert stable(holes, sparse_mla_top_k_lens=holes.lengths).shape[0] == 2
    monkeypatch.setenv("FLASHINFER_VALIDATE_INPUTS", "1")
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="hole"):
        stable(holes, sparse_mla_top_k_lens=holes.lengths)
    monkeypatch.delenv("FLASHINFER_VALIDATE_INPUTS")

    _, head_dim, heads = cfg
    out = flashinfer.prefill.trtllm_prefill_with_kv_cache_mla(
        query=rows.q.view(1, 2, heads, head_dim),
        kv_cache=rows.pool.kv.unsqueeze(1),
        workspace_buffer=torch.zeros(1024, dtype=torch.int8, device="cuda"),
        qk_nope_head_dim=192,
        kv_lora_rank=512,
        qk_rope_head_dim=head_dim - 512,
        block_tables=rows.indices.view(1, 2, TOPK),
        seq_lens=torch.ones(1, dtype=torch.int32, device="cuda"),
        max_seq_len=1,
        sparse_mla_top_k=TOPK,
        bmm1_scale=SCALE,
        bmm2_scale=1.0,
        sparse_mla_top_k_lens=rows.lengths,
        batch_invariant=True,
    )
    assert same_bits(out.view(2, heads, 512), ref_out[:2])


@needs_device
def test_stable_function_scales_and_graph_replay():
    """A call's scales override the plan's; a captured graph survives any number of later call shapes."""
    cfg = CONFIGS[2]
    _, head_dim, heads = cfg
    pool, targets, (ref_out, _) = shared(cfg)
    n = 4
    rows = targets.take(range(n))
    assert same_bits(stable(rows).view(n, heads, 512), ref_out[:n])
    for softmax_scale, output_scale in ((0.03125, 1.0), (SCALE, 0.5), (0.03125, 0.25)):
        want = run_batch(
            rows,
            make_wrapper(
                cfg, n, softmax_scale=softmax_scale, output_scale=output_scale
            ),
        )[0]
        got = stable(rows, bmm1_scale=softmax_scale, bmm2_scale=output_scale)
        assert same_bits(got.view(n, heads, 512), want)
    planned = torch.tensor([SCALE], dtype=torch.float32, device="cuda")
    other = torch.tensor([0.03125], dtype=torch.float32, device="cuda")
    want = run_batch(rows, make_wrapper(cfg, n, softmax_scale=0.03125))[0]
    assert same_bits(stable(rows, bmm1_scale=planned).view(n, heads, 512), ref_out[:n])
    assert same_bits(stable(rows, bmm1_scale=other).view(n, heads, 512), want)
    planned.fill_(0.03125)
    assert same_bits(stable(rows, bmm1_scale=planned).view(n, heads, 512), want)
    assert same_bits(stable(rows).view(n, heads, 512), ref_out[:n])

    q = rows.q.view(n, 1, heads, head_dim).clone()
    tables = rows.indices.view(n, 1, TOPK).clone()
    lengths = rows.lengths.clone()
    out = torch.empty(n, 1, heads, 512, dtype=torch.bfloat16, device="cuda")
    call = lambda: stable(
        rows, query=q, block_tables=tables, sparse_mla_top_k_lens=lengths, out=out
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            call()
    torch.cuda.synchronize()
    for extra in range(5, 5 + 80):
        stable(mates(cfg, pool, extra, "short"))
    junk = [torch.ones(257, device="cuda") for _ in range(64)]
    order = [7, 2, 9, 0]
    nxt = targets.take(order)
    q.copy_(nxt.q.view(n, 1, heads, head_dim))
    tables.copy_(nxt.indices.view(n, 1, TOPK))
    lengths.copy_(nxt.lengths)
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert same_bits(out.view(n, heads, 512), ref_out[order]) and len(junk) == 64


# --------------------------------------------------------------------------------------------------------------------
# Refusals of the wrapper
# --------------------------------------------------------------------------------------------------------------------


@needs_device
def test_wrapper_refuses_inputs_outside_the_class():
    from flashinfer.attention.prims_ts import (
        MLAAttentionTuning,
        batch_sparse_mla_decode_with_paged_kv_cache,
    )

    cfg = CONFIGS[2]
    refused = [
        dict(softmax_scale=None),
        dict(use_attention_sinks=True),
        dict(max_extra_topk=128),
        dict(output_quant_block_size=32, num_output_groups=1),
        dict(tuning=MLAAttentionTuning(softmax_rescale_threshold_log2=0.0)),
        dict(kv_cache_format="nvfp4"),
        dict(kv_cache_format="fp8"),
        dict(kv_cache_scale=0.5),
        dict(extra_kv_cache_format="nvfp4"),
    ]
    for kw in refused:
        with pytest.raises(bi.BatchInvariantUnsupportedError):
            make_wrapper(cfg, 2, **kw)
    with pytest.raises(bi.BatchInvariantUnsupportedError):
        make_wrapper(("e4m3", 576, 8), 2, kv_cache_format="nvfp4.fp8rope")
    with pytest.raises(TypeError):
        make_wrapper(cfg, 2, batch_invariant=1)

    pool, targets, (ref_out, _) = shared(cfg)
    rows = targets.take([5, 6])
    holes = rows.take([0, 1])
    holes.indices[1, 7] = -1
    wrapper = make_wrapper(cfg, 2)
    with pytest.raises(bi.BatchInvariantUnsupportedError):
        run_batch(holes, wrapper)
    assert same_bits(run_batch(rows, wrapper)[0], ref_out[[5, 6]])
    _, head_dim, heads = cfg
    with pytest.raises(bi.BatchInvariantUnsupportedError):
        batch_sparse_mla_decode_with_paged_kv_cache(
            rows.q.view(2, 1, heads, head_dim),
            pool.kv,
            rows.indices,
            rows.lengths,
            batch_invariant=True,
        )
    out = batch_sparse_mla_decode_with_paged_kv_cache(
        rows.q.view(2, 1, heads, head_dim),
        pool.kv,
        rows.indices,
        rows.lengths,
        softmax_scale=SCALE,
        batch_invariant=True,
    )
    assert same_bits(out.view(2, heads, 512), ref_out[[5, 6]])


@needs_device
@pytest.mark.skipif(not NIGHTLY, reason="nightly tier")
@config
def test_scales_are_row_inputs(cfg):
    """Other softmax / output scales, host and tensor: still one result per row."""
    pool, targets, _ = shared(cfg)
    _, head_dim, heads = cfg
    batch = join(targets, mates(cfg, pool, 20, "mixed"))
    n = len(targets)
    for softmax_scale, output_scale in ((0.03125, 1.0), (0.0625, 0.5)):
        one = make_wrapper(
            cfg, 1, softmax_scale=softmax_scale, output_scale=output_scale
        )
        ref = [run_batch(targets.take([i]), one) for i in range(n)]
        many = make_wrapper(
            cfg, len(batch), softmax_scale=softmax_scale, output_scale=output_scale
        )
        out, lse = run_batch(batch, many)
        assert same_bits(out[:n], torch.cat([o for o, _ in ref]))
        assert same_bits(lse[:n], torch.cat([l for _, l in ref]))
        tensors = dict(
            softmax_scale=torch.tensor(softmax_scale, device="cuda"),
            output_scale=torch.tensor(output_scale, device="cuda"),
        )
        out2, _ = run_batch(batch, make_wrapper(cfg, len(batch), **tensors))
        assert same_bits(out2, out)


if __name__ == "__main__" and "--digest" in sys.argv:
    print(json.dumps(_digest_cases(), sort_keys=True))
