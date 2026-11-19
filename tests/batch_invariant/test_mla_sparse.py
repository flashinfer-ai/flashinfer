"""The batch-invariant sparse MLA classes ``mla.sparse.<dtype>.d<576|512>`` (SM100 / SM103 / SM107).

Bitwise only. The default tier covers every guarantee on the GLM-5.3 and GLM-5.3-Flash shapes;
``FLASHINFER_MLA_BI_NIGHTLY=1`` (set by hand, no CI lane sets it) repeats it at batches of 256 and 512, a 2,304-row
prefill, more head counts and the remaining dtype / layout pairs. The device-free tests (registry, fingerprint, list
arithmetic, Lean cases) run everywhere.
"""

import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import warnings

import numpy
import pytest
import torch

import flashinfer.batch_invariant as bi
from flashinfer.batch_invariant.mla import cases, lean_cases, spec
from flashinfer.batch_invariant.mla.classes import OPERATORS

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
#: (dtype, head_dim, heads): GLM-5.3 RoPE rows at 64 and 8 heads, GLM-5.3-Flash NoPE rows at 16 heads.
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
#: Requests of the shared pool, (seed, tokens): long, under one tile, inside the width, top-k plus the Flash tail.
REQUESTS = [(11, 6000), (12, 150), (13, 1500), (14, 2051), (15, 3000), (16, 40)]
#: Target rows, (request, visible tokens): lists of 1 to 300 keys around the tile edge, full lists, a deep row.
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


class Planned:
    """A plan of the class behind the argument form of these tests: any query shape ``[..., H, D]``."""

    def __init__(self, cfg, **plan):
        from flashinfer.batch_invariant import mla

        dtype, head_dim, heads = cfg
        plan.setdefault("softmax_scale", SCALE)
        self.cfg = cfg
        self.plan = mla.plan(
            torch.device("cuda"),
            heads,
            q_dtype=cases.torch_dtype(dtype),
            qk_rope_head_dim=head_dim - 512,
            **plan,
        )

    def run(self, query, kv, indices, lengths, **run):
        _, head_dim, heads = self.cfg
        prefix = query.shape[:-2]
        out, lse = run.pop("out", None), run.pop("lse", None)
        got = self.plan.run(
            query.reshape(-1, heads, head_dim),
            kv.reshape(-1, head_dim),
            indices,
            lengths,
            out=None if out is None else out.view(-1, heads, 512),
            lse=None if lse is None else lse.view(-1, heads),
            return_lse=True,
            **run,
        )
        return got[0].view(*prefix, heads, 512), got[1].view(*prefix, heads)


def make_wrapper(cfg, **plan):
    return Planned(cfg, **plan)


def default_wrapper(cfg, rows, width=TOPK):
    """The default (not batch-invariant) ``prims_ts`` schedule, for the negative control."""
    from flashinfer.attention.prims_ts import BatchSparseMLADecodePagedTSWrapper

    dtype, head_dim, heads = cfg
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wrapper = BatchSparseMLADecodePagedTSWrapper()
        wrapper.plan(
            torch.device("cuda"),
            rows,
            heads,
            max_topk=width,
            q_dtype=cases.torch_dtype(dtype),
            qk_rope_head_dim=head_dim - 512,
            softmax_scale=SCALE,
            return_lse=True,
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
    wrapper = wrapper or make_wrapper(rows.cfg)
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
    wrapper = make_wrapper(rows.cfg)
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
        "mla_decode/throughput_2cta/kernel.py",
        "mla_decode/throughput_2cta/resources.py",
        "mla_decode/throughput_2cta/tasks.py",
        "mla_decode/helpers/sparse_indices.py",
        "mla_decode/helpers/math.py",
    ):
        assert needed in files
    assert len(set(files)) == len(files)
    assert not any(
        "sparse_policy" in f or "throughput_latency_1cta" in f for f in files
    )
    root = os.path.dirname(spec.__file__)
    assert all(os.path.isfile(os.path.join(root, "kernel", f)) for f in files)
    assert spec.source_digest() == spec.KERNEL_DIGEST, (
        "flashinfer/batch_invariant/mla/kernel changed: reproduce the recorded digests, then update KERNEL_DIGEST"
    )


def test_kernel_copy_is_self_contained():
    """The class owns its kernel sources: no file of the package names the ``prims_ts`` attention tree in code."""
    import ast

    root = os.path.dirname(spec.__file__)
    for base, _, names in os.walk(root):
        for name in names:
            if not name.endswith(".py"):
                continue
            path = os.path.join(base, name)
            with open(path) as f:
                tree = ast.parse(f.read())
            for node in ast.walk(tree):
                modules = []
                if isinstance(node, ast.ImportFrom):
                    modules.append(node.module or "")
                elif isinstance(node, ast.Import):
                    modules += [a.name for a in node.names]
                for module in modules:
                    assert "attention" not in module.split("."), (path, module)
                    assert not module.startswith("flashinfer.prims_ts"), (path, module)


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
    assert len(cases.DIGESTS) == len(cases.DIGEST_CASES) * (
        len(cases.DIGEST_ROWS) + len(cases.DENSE_ROWS)
    )
    for bits, dtype in ((4, torch.float8_e4m3fn), (8, torch.bfloat16)):
        d = cases.dense_values(4096, 5, bits=bits)
        assert torch.equal(d, cases.dense_values(4096, 5, bits=bits))
        assert torch.equal(d.to(dtype).float(), d)
        assert float(d.abs().min()) == 2.0**-6 and float(d.abs().max()) < 16.0
        exact = (d[:576].double() * d[576:1152].double()).sum()
        assert float(exact.float()) != float(exact), "the score must not fit fp32"


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
    from flashinfer.batch_invariant import mla

    with pytest.raises(bi.BatchInvariantUnsupportedError):
        mla.plan(
            torch.device("cpu"),
            8,
            q_dtype=torch.bfloat16,
            qk_rope_head_dim=64,
            softmax_scale=SCALE,
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


_ISOLATION = """
import sys, warnings, torch
import flashinfer
from flashinfer.batch_invariant.mla import cases, spec
dev = torch.device("cuda")
seen = warnings.catch_warnings(record=True).__enter__()
warnings.simplefilter("always")
for name, dtype, head_dim, heads in cases.DIGEST_CASES[:2] + cases.DIGEST_CASES[-2:]:
    dt = cases.torch_dtype(dtype)
    q = torch.zeros(3, 2, heads, head_dim, dtype=dt, device=dev)
    kv = torch.zeros(64, 16, head_dim, dtype=dt, device=dev)
    tables = torch.arange(6 * 300, dtype=torch.int32, device=dev).view(3, 2, 300) % 1024
    flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
        query=q, kv_cache=kv, workspace_buffer=torch.zeros(8, dtype=torch.int8, device=dev), qk_nope_head_dim=192,
        kv_lora_rank=512, qk_rope_head_dim=head_dim - 512, block_tables=tables,
        seq_lens=torch.ones(3, dtype=torch.int32, device=dev), max_seq_len=1, sparse_mla_top_k=300,
        bmm1_scale=0.0625, bmm2_scale=1.0, batch_invariant=True,
    )
torch.cuda.synchronize()
loaded = sorted(m for m in sys.modules if m.startswith("flashinfer.attention.prims_ts"))
print("LOADED", loaded, [str(w.category.__name__) for w in seen if "xperimental" in w.category.__name__])
"""


@needs_device
def test_flagged_path_does_not_load_the_default_kernel_or_policy():
    """A fresh process that plans, compiles and runs the classes through the stable function never imports
    ``flashinfer.attention.prims_ts`` (neither its kernels nor its schedule policy can reach the classes) and emits no
    experimental-API warning."""
    env = dict(os.environ, FLASHINFER_CUTE_DSL_DISABLE_CACHE="1")
    run = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", _ISOLATION],
        capture_output=True,
        text=True,
        env=env,
    )
    assert run.returncode == 0, run.stderr[-2000:]
    assert run.stdout.strip().splitlines()[-1] == "LOADED [] []", run.stdout[-2000:]


@needs_device
def test_planned_kernel_state_is_the_recorded_state():
    from flashinfer.batch_invariant.mla import engine

    state = {}
    for dtype in spec.DTYPES:
        for rope in (64, 0):
            for heads in (1, 8, 16, 64, 128):
                kernel = engine.build_kernel(torch.device("cuda"), heads, dtype, rope)
                got = {name: getattr(kernel, name) for name in spec.KERNEL_STATE}
                assert got == spec.KERNEL_STATE, (dtype, rope, heads)
                profile = spec.PROFILES[dtype]
                assert kernel.sparse_gather_warps == profile["gather_issue_warps"]
                assert kernel.sparse_kv_stages == profile.get("kv_pipeline_stages", 0)
                state[dtype, rope, heads] = kernel.compile_signature()
    cfg = ("e4m3", 576, 64)
    planned = make_wrapper(cfg).plan
    assert planned.kernel.compile_signature() == state["e4m3", 64, 64]
    assert make_wrapper(cfg).plan is planned


@needs_device
def test_launch_takes_any_contiguous_leading_shape_and_no_rows():
    cfg = CONFIGS[0]
    _, head_dim, heads = cfg
    pool, targets, (ref_out, ref_lse) = shared(cfg)
    rows = targets.take(range(6))
    plan = make_wrapper(cfg).plan
    out = torch.zeros(2, 3, heads, 512, dtype=torch.bfloat16, device="cuda")
    lse = torch.zeros(6, 1, heads, dtype=torch.float32, device="cuda")
    plan.launch(
        rows.q.view(2, 3, heads, head_dim),
        pool.kv,
        rows.indices,
        rows.lengths,
        out,
        lse,
    )
    torch.cuda.synchronize()
    assert same_bits(out.view(6, heads, 512), ref_out[:6])
    assert same_bits(lse.view(6, heads), ref_lse[:6])
    none = torch.empty(0, dtype=torch.int32, device="cuda")
    empty = plan.run(
        rows.q[:0], pool.kv, rows.indices[:0], none, return_lse=True, validate=True
    )
    assert empty[0].shape == (0, heads, 512) and empty[1].shape == (0, heads)


@needs_device
def test_one_plan_serves_every_row_count_and_width():
    """The row count and the list width are launch-time extents: no schedule is chosen from them."""
    from flashinfer.batch_invariant.mla import engine

    cfg = CONFIGS[1]
    pool, targets, (ref_out, _) = shared(cfg)
    before = len(engine._plans)
    for n in (1, 3, 12):
        assert same_bits(run_batch(targets.take(range(n)))[0], ref_out[:n])
    wide = Rows(cfg, pool, TARGETS, width=2051)
    assert same_bits(run_batch(wide)[0], ref_out)
    assert len(engine._plans) == before


@needs_device
def test_default_schedule_is_not_batch_invariant():
    """Negative control: on today's default ``prims_ts`` schedule the same comparison fails, so the invariance tests
    can fail. The classes do not depend on that schedule: if it stops differing, this control has nothing to show."""
    pytest.importorskip("flashinfer.attention.prims_ts")
    cfg = ("e4m3", 576, 64)
    _, head_dim, heads = cfg
    pool, targets, _ = shared(cfg)
    full = targets.take([10, 11])

    def default(rows):
        n = len(rows)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out, _ = default_wrapper(cfg, n).run(
                rows.q.view(n, 1, heads, head_dim), pool.kv, rows.indices, rows.lengths
            )
        return out.view(n, heads, 512)

    alone = [default(full.take([i])) for i in range(2)]
    batch = join(full, mates(cfg, pool, 62, "long"))
    assert not same_bits(default(batch)[:2], torch.cat(alone)), (
        "the default prims_ts schedule gave batch-independent bits: the control needs another shape"
    )
    flagged = run_batch(batch)[0][:2]
    assert same_bits(
        flagged, torch.cat([run_batch(full.take([i]))[0] for i in range(2)])
    )


@needs_device
def test_inputs_at_storage_offsets_and_strided_lists():
    cfg = CONFIGS[1]
    _, head_dim, heads = cfg
    pool, targets, (ref_out, ref_lse) = shared(cfg)
    rows = targets.take(range(4))
    plan = make_wrapper(cfg).plan
    flat = pool.kv.view(-1, head_dim)
    q = torch.zeros(6, heads, head_dim, dtype=rows.q.dtype, device="cuda")
    q[2:] = rows.q
    wide = torch.full((4, TOPK + 8), -1, dtype=torch.int32, device="cuda")
    wide[:, 4 : 4 + TOPK] = rows.indices
    out = torch.full((9, heads, 512), 7.0, dtype=torch.bfloat16, device="cuda")
    lse = torch.full((10, heads), 7.0, dtype=torch.float32, device="cuda")
    got = plan.run(
        q[2:],
        flat,
        wide[:, 4 : 4 + TOPK],
        rows.lengths,
        out=out[3:7],
        lse=lse[4:8],
        return_lse=True,
    )
    assert same_bits(got[0], ref_out[:4]) and same_bits(got[1], ref_lse[:4])
    assert bool((out[:3] == 7).all() and (out[7:] == 7).all())
    assert bool((lse[:4] == 7).all() and (lse[8:] == 7).all())
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="lse"):
        plan.run(
            rows.q,
            flat,
            rows.indices,
            rows.lengths,
            lse=torch.zeros(heads, 4, dtype=torch.float32, device="cuda"),
        )


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
    wrapper = make_wrapper(cfg)
    first = run_batch(batch, wrapper)
    for _ in range(3):
        again = run_batch(batch, wrapper)
        assert same_bits(again[0], first[0]) and same_bits(again[1], first[1])
    from flashinfer.batch_invariant.mla import engine

    dtype, head_dim, _ = cfg
    old = wrapper.plan
    rebuilt = engine.SparseMLAPlan(
        old.device, old.num_qo_heads, old.q_dtype, head_dim - 512, SCALE, 1.0
    )
    assert rebuilt._compiled is not old._compiled
    wrapper.plan = rebuilt
    replanned = run_batch(batch, wrapper)
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
    assert not wrong and set(got) == set(cases.DIGESTS), (
        f"{wrong}: the table was recorded with {cases.RECORDED_WITH}; this run has nvidia-cutlass-dsl "
        f"{spec._dsl_version()} on CUDA {torch.version.cuda}. The approximate exp2 / log2 / reciprocal and two "
        "fused multiply-adds are the compiler's choice, so a new toolchain can move the bits."
    )


@needs_device
@pytest.mark.parametrize(
    "case", cases.DIGEST_CASES, ids=[c[0] for c in cases.DIGEST_CASES]
)
def test_full_significand_rows_do_not_depend_on_the_batch(case):
    """Inputs whose Q.K fold rounds in fp32: a row alone, in a batch and at another position gives the same bits."""
    from flashinfer.batch_invariant import mla

    _, dtype, head_dim, heads = case
    pool = cases.Pool([cases.DIGEST_REQUEST], head_dim, dtype, "cuda", dense=True)
    plan = mla.plan(
        "cuda",
        heads,
        q_dtype=pool.dtype,
        qk_rope_head_dim=head_dim - 512,
        softmax_scale=SCALE,
    )
    visibles = [1, 127, 129, 300, 2048, 4096, 3000]
    built = [pool.row(0, v, 2048, 2052) for v in visibles]
    indices = torch.stack([r for r, _ in built]).cuda()
    lengths = torch.tensor([n for _, n in built], dtype=torch.int32, device="cuda")
    query = torch.stack([pool.query(0, v - 1, heads) for v in visibles])
    kv = pool.kv.view(-1, head_dim)
    out, lse = plan.run(query, kv, indices, lengths, return_lse=True)
    order = [6, 2, 5, 0, 4, 1, 3]
    out2, lse2 = plan.run(
        query[order].contiguous(),
        kv,
        indices[order].contiguous(),
        lengths[order].contiguous(),
        return_lse=True,
    )
    assert same_bits(out2, out[order]) and same_bits(lse2, lse[order])
    for i in range(len(visibles)):
        one, one_lse = plan.run(
            query[i : i + 1].contiguous(),
            kv,
            indices[i : i + 1, :2048].contiguous(),
            lengths[i : i + 1].contiguous(),
            return_lse=True,
        )
        assert same_bits(one, out[i : i + 1]) and same_bits(one_lse, lse[i : i + 1])


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
    wrapper = make_wrapper(cfg)
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

    plan = make_wrapper(cfg)
    out, lse = plan.run(rows.q, kv, rows.indices, rows.lengths)
    assert same_bits(out, ref_out) and same_bits(lse, ref_lse)

    start = 0
    for piece in (count // 3, 1, count - count // 3 - 1):
        chunk = rows.take(range(start, start + piece))
        out, lse = plan.run(chunk.q, kv, chunk.indices, chunk.lengths)
        assert same_bits(out, ref_out[start : start + piece])
        assert same_bits(lse, ref_lse[start : start + piece])
        start += piece

    lse_kw = dict(return_lse=True, return_lse_base="basee")
    out, lse = stable(
        rows,
        query=rows.q.view(1, count, heads, head_dim),
        block_tables=rows.indices.view(1, count, TOPK),
        sparse_mla_top_k_lens=rows.lengths,
        **lse_kw,
    )
    assert same_bits(out.view(count, heads, 512), ref_out)
    assert same_bits(lse.view(count, heads), ref_lse)
    for tokens in (2, 4, 8):
        requests = count // tokens
        used = requests * tokens
        out, lse = stable(
            rows,
            query=rows.q[:used].view(requests, tokens, heads, head_dim),
            block_tables=rows.indices[:used].view(requests, tokens, TOPK),
            sparse_mla_top_k_lens=rows.lengths[:used],
            **lse_kw,
        )
        assert same_bits(out.view(used, heads, 512), ref_out[:used])
        assert same_bits(lse.view(used, heads), ref_lse[:used])
    part = rows.take(range(8))
    ptr = torch.tensor([0, 1, 5, 8], dtype=torch.int32, device="cuda")
    out, lse = stable(
        part,
        query=part.q,
        block_tables=part.indices,
        cum_seq_lens_q=ptr,
        max_q_len=4,
        sparse_mla_top_k_lens=part.lengths,
        **lse_kw,
    )
    assert same_bits(out, ref_out[:8]) and same_bits(lse.view(8, heads), ref_lse[:8])


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
    host = numpy.float32(SCALE)
    assert same_bits(stable(targets, bmm1_scale=host).view(n, heads, 512), ref_out)
    odd = torch.empty(n * heads + 1, dtype=torch.float32, device="cuda")[1:]
    stable(
        targets,
        sparse_mla_top_k_lens=targets.lengths,
        lse=odd.view(n, heads),
        return_lse_base="basee",
    )
    assert odd.data_ptr() % 16 and same_bits(odd.view(n, heads), ref_lse)
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
        (dict(return_lse_base="bogus"), "return_lse_base"),
        (dict(cp_world=torch.tensor([1, 2], device="cuda")), "cp_world"),
        (dict(bmm1_scale=10**400), "bmm1_scale"),
        (
            dict(lse=torch.zeros(cfg[2], 2, device="cuda"), return_lse_base="basee"),
            "lse",
        ),
    ]
    for kw, fragment in refused:
        with pytest.raises(bi.BatchInvariantUnsupportedError, match=fragment):
            stable(rows, **kw)
    for not_bool in (1, 0, None, "yes"):
        with pytest.raises(TypeError):
            stable(rows, batch_invariant=not_bool)

    holes = targets.take([5, 6])
    holes.indices[1, 7] = -1
    assert stable(holes, sparse_mla_top_k_lens=holes.lengths).shape[0] == 2
    monkeypatch.setenv("FLASHINFER_VALIDATE_INPUTS", "1")
    tail = targets.take([5, 6])
    assert same_bits(stable(tail), stable(tail, sparse_mla_top_k_lens=tail.lengths))
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="hole"):
        stable(holes, sparse_mla_top_k_lens=holes.lengths)
    far = targets.take([5, 6])
    far.indices[0, 3] = rows.pool.kv.shape[0] * rows.pool.kv.shape[1]
    with pytest.raises(ValueError, match="outside the cache"):
        stable(far, sparse_mla_top_k_lens=far.lengths)
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
            make_wrapper(cfg, softmax_scale=softmax_scale, output_scale=output_scale),
        )[0]
        got = stable(rows, bmm1_scale=softmax_scale, bmm2_scale=output_scale)
        assert same_bits(got.view(n, heads, 512), want)
    planned = torch.tensor([SCALE], dtype=torch.float32, device="cuda")
    other = torch.tensor([0.03125], dtype=torch.float32, device="cuda")
    want = run_batch(rows, make_wrapper(cfg, softmax_scale=0.03125))[0]
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
    assert same_bits(out.view(n, heads, 512), ref_out[order])
    del junk


# --------------------------------------------------------------------------------------------------------------------
# Refusals of the plan
# --------------------------------------------------------------------------------------------------------------------


@needs_device
def test_plan_refuses_inputs_outside_the_class():
    from flashinfer.batch_invariant import mla

    cfg = CONFIGS[2]
    dtype, head_dim, heads = cfg
    good = dict(
        q_dtype=cases.torch_dtype(dtype),
        qk_rope_head_dim=head_dim - 512,
        softmax_scale=SCALE,
    )
    refused = [
        (heads, dict(softmax_scale=None), "softmax_scale"),
        (heads, dict(softmax_scale=-1.0), "softmax_scale"),
        (heads, dict(softmax_scale=float("inf")), "softmax_scale"),
        (heads, dict(output_scale=0.0), "output_scale"),
        (heads, dict(q_dtype=torch.float16), "q_dtype"),
        (heads, dict(qk_rope_head_dim=32), "qk_rope_head_dim"),
        (129, {}, "heads"),
        (0, {}, "heads"),
    ]
    for count, kw, fragment in refused:
        with pytest.raises(bi.BatchInvariantUnsupportedError, match=fragment):
            mla.plan("cuda", count, **{**good, **kw})

    pool, targets, (ref_out, _) = shared(cfg)
    rows = targets.take([5, 6])
    plan = make_wrapper(cfg)
    holes = rows.take([0, 1])
    holes.indices[1, 7] = -1
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="hole"):
        run_batch(holes, plan)
    far = rows.take([0, 1])
    far.indices[0, 3] = pool.kv.shape[0] * pool.kv.shape[1]
    with pytest.raises(ValueError, match="outside the cache"):
        run_batch(far, plan)
    long = rows.take([0, 1])
    long.lengths[1] = rows.width + 1
    with pytest.raises(ValueError, match="length"):
        run_batch(long, plan)
    flat = pool.kv.view(-1, head_dim)
    layouts = [
        dict(query=rows.q[:, :, :-1]),
        dict(query=rows.q.to(torch.float16)),
        dict(kv=flat[:, :-1]),
        dict(kv=flat[::2]),
        dict(kv=pool.kv.to(torch.float16)),
        dict(indices=rows.indices.long()),
        dict(indices=rows.indices[:1]),
        dict(lengths=rows.lengths[:1]),
        dict(lengths=rows.lengths.long()),
        dict(out=torch.empty(2, heads, 256, dtype=torch.bfloat16, device="cuda")),
        dict(out=torch.empty(2, heads, 512, dtype=torch.float16, device="cuda")),
        dict(lse=torch.empty(2, heads + 1, dtype=torch.float32, device="cuda")),
    ]
    for kw in layouts:
        args = dict(query=rows.q, kv=flat, indices=rows.indices, lengths=rows.lengths)
        args.update(kw)
        with pytest.raises(bi.BatchInvariantUnsupportedError):
            plan.plan.run(
                args["query"],
                args["kv"],
                args["indices"],
                args["lengths"],
                out=args.get("out"),
                lse=args.get("lse"),
                validate=False,
            )
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="kind"):
        plan.plan.run(
            rows.q,
            flat,
            rows.indices,
            rows.lengths,
            softmax_scale=torch.tensor([SCALE], device="cuda"),
        )
    assert same_bits(run_batch(rows, plan)[0], ref_out[[5, 6]])


@needs_device
@config
def test_scales_are_row_inputs(cfg):
    """Other softmax / output scales, host and tensor: still one result per row."""
    pool, targets, _ = shared(cfg)
    _, head_dim, heads = cfg
    batch = join(targets, mates(cfg, pool, 20, "mixed"))
    n = len(targets)
    for softmax_scale, output_scale in ((0.03125, 1.0), (0.0625, 0.5)):
        one = make_wrapper(cfg, softmax_scale=softmax_scale, output_scale=output_scale)
        ref = [run_batch(targets.take([i]), one) for i in range(n)]
        many = make_wrapper(cfg, softmax_scale=softmax_scale, output_scale=output_scale)
        out, lse = run_batch(batch, many)
        assert same_bits(out[:n], torch.cat([o for o, _ in ref]))
        assert same_bits(lse[:n], torch.cat([l for _, l in ref]))
        tensors = dict(
            softmax_scale=torch.tensor(softmax_scale, device="cuda"),
            output_scale=torch.tensor(output_scale, device="cuda"),
        )
        out2, _ = run_batch(batch, make_wrapper(cfg, **tensors))
        assert same_bits(out2, out)


def _carve(arena, offset, shape, dtype):
    count = int(numpy.prod(shape)) * torch.empty((), dtype=dtype).element_size()
    return arena[offset : offset + count].view(dtype).view(shape)


@needs_device
@pytest.mark.parametrize("cfg", [CONFIGS[1], CONFIGS[4]], ids=["e4m3", "bf16"])
def test_outputs_sharing_memory_with_an_argument_are_refused(cfg):
    """``out`` or ``lse`` over the query, the cache, the list, the lengths, a tensor scale or each other: refused
    before launch (on the device such a call returns wrong rows or faults)."""
    dtype, head_dim, heads = cfg
    pool, targets, (ref_out, ref_lse) = shared(cfg)
    n = 4
    rows = targets.take(range(n))
    scale = torch.zeros(4, dtype=torch.float32, device="cuda") + SCALE
    plan = make_wrapper(cfg, softmax_scale=scale[:1]).plan
    size = rows.q.element_size()
    sizes = [n * heads * head_dim * size, 4096 * head_dim * size, n * TOPK * 4, n * 4]
    step = (max(sizes) + n * heads * 1024 + 255) // 64 * 64
    arena = torch.zeros(6 * step, dtype=torch.uint8, device="cuda")
    q = _carve(arena, 0, (n, heads, head_dim), rows.q.dtype)
    kv = _carve(arena, step, (4096, head_dim), rows.q.dtype)
    indices = _carve(arena, 2 * step, (n, TOPK), torch.int32)
    lengths = _carve(arena, 3 * step, (n,), torch.int32)
    q.copy_(rows.q), kv.copy_(pool.kv.view(-1, head_dim)[:4096])
    indices.copy_(rows.indices.clamp(max=4095)), lengths.copy_(rows.lengths)

    def run(out_at, lse_at, **kw):
        out = _carve(arena, out_at, (n, heads, 512), torch.bfloat16)
        lse = _carve(arena, lse_at, (n, heads), torch.float32)
        return plan.run(
            q, kv, indices, lengths, out=out, lse=lse, return_lse=True, **kw
        )

    clean = run(4 * step, 5 * step)
    clean = (clean[0].clone(), clean[1].clone())
    for at in (0, 64, step, step + 4096, 2 * step, 3 * step):
        with pytest.raises(bi.BatchInvariantUnsupportedError, match="sharing memory"):
            run(at, 5 * step)
        with pytest.raises(bi.BatchInvariantUnsupportedError, match="sharing memory"):
            run(4 * step, at)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="sharing memory"):
        run(4 * step, 4 * step + 128)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="sharing memory"):
        plan.run(
            q,
            kv,
            indices,
            lengths,
            out=_carve(arena, 4 * step, (n, heads, 512), torch.bfloat16),
            softmax_scale=_carve(arena, 4 * step + 64, (1,), torch.float32),
        )
    again = run(4 * step, 5 * step)
    assert same_bits(again[0], clean[0]) and same_bits(again[1], clean[1])
    # adjacent buffers are not overlap
    out = _carve(
        arena,
        n * heads * head_dim * size // 64 * 64 + 64,
        (n, heads, 512),
        torch.bfloat16,
    )
    assert same_bits(plan.run(q, kv, indices, lengths, out=out), clean[0])


@needs_device
def test_calls_past_the_row_limit_and_empty_first_calls():
    """The bf16 classes take at most 65,535 rows per call (typed refusal, not a launch error); an empty call with
    default lengths works as the first call of its width."""
    from flashinfer.batch_invariant import mla
    from flashinfer.batch_invariant.mla import engine

    plan = mla.plan(
        torch.device("cuda"),
        1,
        q_dtype=torch.bfloat16,
        qk_rope_head_dim=64,
        softmax_scale=SCALE,
    )
    rows = 65536
    q = torch.zeros(rows, 1, 576, dtype=torch.bfloat16, device="cuda")
    kv = torch.zeros(8, 576, dtype=torch.bfloat16, device="cuda")
    indices = torch.zeros(rows, 1, dtype=torch.int32, device="cuda")
    lengths = torch.ones(rows, dtype=torch.int32, device="cuda")
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="65535 rows"):
        plan.run(q, kv, indices, lengths, validate=False)
    out = plan.run(q[:-1], kv, indices[:-1], lengths[:-1], validate=False)
    assert same_bits(out[:1], out[-1:])
    device = torch.device("cuda", torch.cuda.current_device())
    assert engine.full_lengths(device, 0, 1777).shape == (0,)
    assert int(engine.full_lengths(device, 3, 1777).sum()) == 3 * 1777


@needs_device
def test_list_validation_during_capture_and_arrays_in_refused_slots():
    cfg = CONFIGS[1]
    pool, targets, (ref_out, _) = shared(cfg)
    rows = targets.take(range(3))
    plan = make_wrapper(cfg).plan
    out = torch.empty(3, cfg[2], 512, dtype=torch.bfloat16, device="cuda")
    flat = pool.kv.view(-1, cfg[1])
    plan.run(rows.q, flat, rows.indices, rows.lengths, out=out)
    graph = torch.cuda.CUDAGraph()
    with (
        pytest.raises(RuntimeError, match="cannot run during CUDA graph capture"),
        torch.cuda.graph(graph),
    ):
        plan.run(rows.q, flat, rows.indices, rows.lengths, out=out, validate=True)
    torch.cuda.synchronize()
    assert same_bits(plan.run(rows.q, flat, rows.indices, rows.lengths), ref_out[:3])
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="sinks"):
        stable(rows, sinks=numpy.zeros(4, dtype=numpy.float32))
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="qk_rope_head_dim"):
        stable(rows, qk_rope_head_dim=False)


@needs_device
def test_kept_buffers_are_filled_before_another_stream_reads_them():
    """A row count first used on a busy stream: its kept buffers are complete when a second stream uses them."""
    from flashinfer.batch_invariant.mla import engine

    device = torch.device("cuda", torch.cuda.current_device())
    junk = [
        torch.full((1 << k,), 0x7F7F7F7F, dtype=torch.int32, device="cuda")
        for k in (8, 10, 12, 14)
        for _ in range(64)
    ]
    torch.cuda.synchronize()
    del junk
    busy, other = torch.cuda.Stream(), torch.cuda.Stream()
    load = torch.randn(4000, 4000, device="cuda")
    with torch.cuda.stream(busy):
        for _ in range(30):
            load = (load @ load).clamp_(-1, 1)
        engine.full_lengths(device, 911, 1999)
    with torch.cuda.stream(other):
        seen = engine.full_lengths(device, 911, 1999).clone()
        other.synchronize()
        assert bool((seen == 1999).all())
    torch.cuda.synchronize()


@needs_device
def test_buffers_do_not_grow_with_the_number_of_row_counts():
    """Every row count of a prefill reuses a few kept buffers; ``clear_plans`` forgets plans and buffers, and a graph
    captured before it still replays. Kept last: it empties the plan cache."""
    from flashinfer.batch_invariant import mla
    from flashinfer.batch_invariant.mla import engine

    cfg = CONFIGS[2]
    pool, targets, (ref_out, _) = shared(cfg)
    one = targets.take([3])
    mla.clear_plans()
    stable(one)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    for n in range(1, 61):
        rows = targets.take([3] * n)
        out = stable(rows)
        assert same_bits(out.view(n, cfg[2], 512)[-1:], ref_out[3:4])
        del out, rows
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() - before < 64 * 1024
    plan = make_wrapper(cfg).plan
    assert len(plan._rows.backing) == 1 and plan._rows.capacity == 64
    kept = torch.empty(1, 1, cfg[2], 512, dtype=torch.bfloat16, device="cuda")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        stable(one, out=kept)
    del plan
    mla.clear_plans()
    assert not engine._plans and not engine._full and not engine._full_views
    gc.collect()
    torch.cuda.empty_cache()
    junk = [
        torch.full((1 << k,), 0x7F7F7F7F, dtype=torch.int32, device="cuda")
        for k in (6, 8, 10, 12)
        for _ in range(100)
    ]
    kept.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert same_bits(kept.view(1, cfg[2], 512), ref_out[3:4])
    del junk
    assert same_bits(stable(one).view(1, cfg[2], 512), ref_out[3:4])


if __name__ == "__main__" and "--digest" in sys.argv:
    print(json.dumps(_digest_cases(), sort_keys=True))
