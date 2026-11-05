"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Run-to-run bitwise determinism of the ragged BF16 MoE grouped GEMM routes.
#
# Every case fixes its inputs (seeded from its label), prepares one launch and
# reuses it, so the plan, the caller-owned buffers and the current stream's
# scheduler / wave workspaces stay the same over the repeats.  The output is
# filled with zero bytes before each of the first ``REPEATS`` launches and with
# ``FILL_BYTE`` before each of the next ``REPEATS``: no route accumulates into
# ``out`` (each one overwrites it), so the bits must not depend on its prior
# contents either.  Every launch's raw bits (an integer view) must equal the
# first launch's.
#
# Arms: ``planned`` launches the prepared launch as planned (for a data
# gradient whose plan probes the queue split, after its probe has completed, so
# every compared launch passes the probed split); ``even`` (data gradient only)
# launches the same prepared launch with the probed split disabled, so it passes
# the plan's even split, and its bits must also equal the planned arm's.
# ``test_route_coverage`` checks that the cases reach every weight-gradient
# program (k tile 256 / 512, bf16 / fp32 output, with the tail reduce) and the
# probed data gradient on the device, and ``test_fresh_processes_give_the_same_bits``
# compares output digests of every case across two fresh processes and this one.

import dataclasses
import hashlib
import json
import math
import subprocess
import sys
import zlib
from pathlib import Path

import pytest
import torch

from tests.test_helpers.cake_package import cake_package

PACKAGE = "cake_moe_grouped_det_gemm"
cb = cake_package(PACKAGE, "cake_backend")
cake_jit = cake_package(PACKAGE, "cake_jit")
HOST_PLAN_CONSTANTS = cake_jit.HOST_PLAN_CONSTANTS

REPEATS = 10  # launches per output pre-state (zero bytes, then FILL_BYTE)
RACE_REPEATS = 100  # the flashinfer-ai/flashinfer#6570 shape
FILL_BYTE = 0x5A
SPOT_ROWS = 64
SPOT_COLS = 256
WORKER_TIMEOUT_S = 3600

# The 58 MoE projection rows of benchmarks/bench_cake_moe_grouped_det_gemm.py:
# (N, K) of w ``[E, N, K]``; E=4 balanced / skewed / one-empty group sizes at two
# token counts for every op, the fp32 weight gradient of the balanced E=4 rows,
# and E=8/16/32 balanced over 131072 rows.
BENCH_PROJ = {"gate_up": (4096, 6144), "down": (6144, 2048)}
E4_GROUPS = {
    112576: {
        "balanced": [28128, 28128, 28128, 28192],
        "skewed": [45024, 33760, 22528, 11264],
        "one_empty": [56288, 33760, 22528, 0],
    },
    136064: {
        "balanced": [34016, 34016, 34016, 34016],
        "skewed": [54432, 40832, 27200, 13600],
        "one_empty": [68032, 40832, 27200, 0],
    },
}
EXPERT_COUNTS = (8, 16, 32)
EXPERT_COUNT_SUM_M = 131072

# Edge cases, each run for fwd, dgrad and the bf16 / fp32 weight gradient.
# (N, K): the two MoE projections; ODD has K % 512 != 0 (the 256-wide k tile and
# the 256-column fwd / dgrad tiles); RACE is the #6570 reproduction shape.
ODD_PROJ = (1280, 1792)
RACE_PROJ = (1024, 6144)
EDGE_SIZES = {
    "one_group": [4000],
    "empty_groups": [0, 3000, 0, 0, 517, 4096, 0, 1],
    "one_nonempty": [0, 0, 5000, 0],
    "nonmult_m": [1, 255, 257, 1023, 4097, 63, 65],
    "all_empty": [0, 0, 0, 0],
    # Fewer rows per group than k512_min_rows_per_group: the 256-wide k tile.
    "e64_m2048": [2048] * 64,
    "e128_m512": [512] * 128,
}
ODD_SIZES = ("empty_groups", "nonmult_m")
RACE_SIZES = [32] * 256
# Weight gradient only: the 256-wide k tile (K % 512 != 0) over two long groups,
# with 84 / 112 / 224 output tiles, so that on every architecture's cluster count
# (74, 76, 106, 108) some case leaves a short last wave that is split over the
# clusters and summed by the tail reduce.
K256_TAIL_K = 1792
K256_TAIL_N_BLOCKS = (6, 8, 16)
K256_TAIL_SIZES = [8192, 8191]

OPS = ("fwd", "dgrad", "wgrad")
INT_VIEW = {torch.bfloat16: torch.int16, torch.float32: torch.int32}


@dataclasses.dataclass(frozen=True)
class Case:
    label: str
    op: str
    n: int
    k: int
    sizes: tuple[int, ...]
    out_dtype: torch.dtype
    repeats: int = REPEATS


def _case(label, op, nk, sizes, out_dtype=torch.bfloat16, repeats=REPEATS):
    n, k = nk
    tag = "f32" if out_dtype == torch.float32 else "bf16"
    return Case(
        label=f"{op}_{label}_{tag}",
        op=op,
        n=n,
        k=k,
        sizes=tuple(sizes),
        out_dtype=out_dtype,
        repeats=repeats,
    )


def bench_cases():
    cases = []
    for proj, nk in BENCH_PROJ.items():
        for op in OPS:
            for dists in E4_GROUPS.values():
                for dist, sizes in dists.items():
                    cases.append(
                        _case(f"{proj}_e4_m{sum(sizes)}_{dist}", op, nk, sizes)
                    )
    for proj, nk in BENCH_PROJ.items():
        for dists in E4_GROUPS.values():
            sizes = dists["balanced"]
            cases.append(
                _case(
                    f"{proj}_e4_m{sum(sizes)}_balanced",
                    "wgrad",
                    nk,
                    sizes,
                    out_dtype=torch.float32,
                )
            )
    for e in EXPERT_COUNTS:
        for proj, nk in BENCH_PROJ.items():
            for op in OPS:
                sizes = [EXPERT_COUNT_SUM_M // e] * e
                cases.append(
                    _case(f"{proj}_e{e}_m{EXPERT_COUNT_SUM_M}_balanced", op, nk, sizes)
                )
    assert len(cases) == 58
    return cases


def _edge_variants(label, nk, sizes, repeats=REPEATS):
    cases = [_case(label, "fwd", nk, sizes), _case(label, "dgrad", nk, sizes)]
    # The weight gradient addresses dW [E, N, K] with 32-bit element offsets.
    if len(sizes) * nk[0] * nk[1] <= int(HOST_PLAN_CONSTANTS["int32_elems"]):
        cases += [
            _case(label, "wgrad", nk, sizes, repeats=repeats),
            _case(label, "wgrad", nk, sizes, out_dtype=torch.float32, repeats=repeats),
        ]
    return cases


def edge_cases():
    cases = []
    for name, sizes in EDGE_SIZES.items():
        for proj, nk in BENCH_PROJ.items():
            cases += _edge_variants(f"edge_{name}_{proj}", nk, sizes)
    for name in ODD_SIZES:
        cases += _edge_variants(f"edge_{name}_odd", ODD_PROJ, EDGE_SIZES[name])
    cases += _edge_variants("race6570_e256_m32", RACE_PROJ, RACE_SIZES, RACE_REPEATS)
    for blocks in K256_TAIL_N_BLOCKS:
        nk = (256 * blocks, K256_TAIL_K)
        label = f"k256tail_n{nk[0]}"
        cases.append(_case(label, "wgrad", nk, K256_TAIL_SIZES))
        cases.append(
            _case(label, "wgrad", nk, K256_TAIL_SIZES, out_dtype=torch.float32)
        )
    return cases


CASES = bench_cases() + edge_cases()
CASE_BY_LABEL = {case.label: case for case in CASES}
assert len(CASE_BY_LABEL) == len(CASES)
ARMS = [
    (case.label, arm)
    for case in CASES
    for arm in (("planned", "even") if case.op == "dgrad" else ("planned",))
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _device():
    if not torch.cuda.is_available():
        pytest.skip("the ragged grouped GEMM programs require an SM100/SM103/SM107 GPU")
    device = torch.device("cuda", 0)
    if cb.arch_for(device) is None:
        pytest.skip("the ragged grouped GEMM programs require an SM100/SM103/SM107 GPU")
    return device


def _require_program(case, device):
    out_dtype = case.out_dtype if case.op == "wgrad" else None
    if not cb.generated_program_available(device, op=case.op, out_dtype=out_dtype):
        pytest.skip(
            f"no generated program for {case.op} registered for {cb.arch_for(device)}"
        )


def _inputs(case, device):
    """Fixed inputs of ``case``: seeded from its label, so every process builds the same bits."""
    gen = torch.Generator(device=device)
    gen.manual_seed(zlib.crc32(case.label.encode()))
    sum_m, num_groups = sum(case.sizes), len(case.sizes)
    scale = 1.0 / math.sqrt(case.k)

    def randn(*shape, scale=1.0):
        t = torch.randn(*shape, generator=gen, device=device, dtype=torch.float32)
        return t.mul_(scale).to(torch.bfloat16)

    offs = torch.tensor(case.sizes, dtype=torch.int32, device=device).cumsum(
        0, dtype=torch.int32
    )
    if case.op == "fwd":
        a = randn(sum_m, case.k, scale=scale)
        b = randn(num_groups, case.n, case.k, scale=scale)
        out = torch.empty(sum_m, case.n, dtype=torch.bfloat16, device=device)
    elif case.op == "dgrad":
        a = randn(sum_m, case.n)
        b = randn(num_groups, case.n, case.k, scale=scale)
        out = torch.empty(sum_m, case.k, dtype=torch.bfloat16, device=device)
    else:
        a = randn(sum_m, case.n)
        b = randn(sum_m, case.k, scale=scale)
        out = torch.empty(
            num_groups, case.n, case.k, dtype=case.out_dtype, device=device
        )
    return a, b, offs, out


def _prepare(case, a, b, offs, out):
    if case.op == "fwd":
        return cb.prepare_grouped_det_gemm_fwd(x=a, w=b, offs=offs, out=out)
    if case.op == "dgrad":
        return cb.prepare_grouped_det_gemm_dgrad(g=a, w=b, offs=offs, out=out)
    return cb.prepare_grouped_det_gemm_wgrad(
        g=a, x=b, offs=offs, out=out, out_dtype=case.out_dtype, num_groups=None
    )


def _even_split(launch):
    """``launch`` with the probed queue split disabled: it passes the plan's even ``c0`` / ``c1``."""
    return dataclasses.replace(
        launch,
        launches=tuple(
            dataclasses.replace(item, split_slots=()) for item in launch.launches
        ),
    )


def _settle_probe(launch, device):
    """Launch once and wait, so the probe a probed launch starts has completed; return
    the split the next launches pass (None for an even plan or where the probe gave none)."""
    launch.launch()
    torch.cuda.synchronize(device)
    if launch.plan.get("queue_split") != "probed":
        return None
    return cb.stream_queue_split(device)


def _bits(out):
    return out.view(INT_VIEW[out.dtype])


def _plan_record(launch):
    keys = (
        "grid",
        "clusters",
        "c0",
        "c1",
        "queue_split",
        "tile_n",
        "tile_k",
        "tail_tile_k",
        "num_tail",
        "tail_splits",
        "raster_rows",
        "wave_mode",
        "launches",
    )
    return {key: launch.plan[key] for key in keys if key in launch.plan}


def _run_repeats(launch, out, repeats):
    """Launch ``2 * repeats`` times (zero bytes, then FILL_BYTE in ``out`` before each);
    return (reference bits of the first launch, launches, mismatching launches, worst
    count of differing elements)."""
    ref, launches, mismatches, worst = None, 0, 0, 0
    for fill in (0, FILL_BYTE):
        for _ in range(repeats):
            out.view(torch.uint8).fill_(fill)
            launch.launch()
            bits = _bits(out)
            launches += 1
            if ref is None:
                ref = bits.clone()
            elif not torch.equal(bits, ref):
                mismatches += 1
                worst = max(worst, int((bits != ref).sum()))
    return ref, launches, mismatches, worst


def _spot_check(case, a, b, out):
    """Guard against a vacuous pass: the last non-empty group's first block matches an fp64
    reference, and an empty group's weight gradient is exact zeros."""
    ends = [sum(case.sizes[: i + 1]) for i in range(len(case.sizes))]
    bounds = list(zip([0] + ends[:-1], ends, strict=True))
    nonempty = [e for e, (s, t) in enumerate(bounds) if t > s]
    if not nonempty:
        return
    e = nonempty[-1]
    s, t = bounds[e]
    if case.op == "fwd":
        r, c = min(t, s + SPOT_ROWS), min(case.n, SPOT_COLS)
        ref = a[s:r].double() @ b[e, :c].double().T
        got = out[s:r, :c]
    elif case.op == "dgrad":
        r, c = min(t, s + SPOT_ROWS), min(case.k, SPOT_COLS)
        ref = a[s:r].double() @ b[e][:, :c].double()
        got = out[s:r, :c]
    else:
        rn, ck = min(case.n, SPOT_COLS // 2), min(case.k, SPOT_COLS // 2)
        ref = a[s:t, :rn].double().T @ b[s:t, :ck].double()
        got = out[e, :rn, :ck]
        for empty in (i for i, (s0, t0) in enumerate(bounds) if t0 == s0):
            assert int(_bits(out[empty]).count_nonzero()) == 0, (
                f"group {empty} is empty but its weight gradient is not exact zeros"
            )
    torch.testing.assert_close(got.double(), ref, atol=3e-2, rtol=2e-2)


# ---------------------------------------------------------------------------
# Repeat-run determinism
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("label", "arm"), ARMS, ids=[f"{x}-{y}" for x, y in ARMS])
def test_repeat_launches_give_identical_bits(label, arm, record_property):
    case = CASE_BY_LABEL[label]
    device = _device()
    _require_program(case, device)
    a, b, offs, out = _inputs(case, device)
    launch = _prepare(case, a, b, offs, out)
    probed = launch.plan.get("queue_split") == "probed"
    split = _settle_probe(launch, device)
    if probed and split is None and sum(case.sizes) > 0:
        pytest.fail(
            "the plan probes the queue split but the probe gave none on this device "
            "(no CUDA compiler, or it did not see every SM and both domains)"
        )
    planned_bits = None
    if arm == "even":
        # The planned launch after the settled probe passes the probed split.
        out.view(torch.uint8).fill_(0)
        launch.launch()
        planned_bits = _bits(out).clone()
        launch = _even_split(launch)
    ref, launches, mismatches, worst = _run_repeats(launch, out, case.repeats)
    torch.cuda.synchronize(device)
    record = dict(
        case=label,
        op=case.op,
        arm=arm,
        arch=cb.arch_for(device),
        sm_count=cb.sm_count(device),
        record_name=launch.record_name,
        stages=list(launch.stages),
        plan=_plan_record(launch),
        probed_split=list(split) if split is not None else None,
        sizes_e=len(case.sizes),
        sum_m=sum(case.sizes),
        n=case.n,
        k=case.k,
        out_dtype=str(case.out_dtype).replace("torch.", ""),
        prestates=["zero", f"0x{FILL_BYTE:02x}"],
        launches=launches,
        mismatches=mismatches,
        worst_diff_elems=worst,
    )
    if arm == "even" and probed:
        record["equal_to_probed"] = bool(torch.equal(ref, planned_bits))
    record_property("det", json.dumps(record, sort_keys=True))
    print("DET " + json.dumps(record, sort_keys=True))
    assert mismatches == 0, (
        f"{mismatches} of {launches - 1} repeat launches differ from the first "
        f"(worst {worst} elements)"
    )
    if arm == "even" and probed:
        assert record["equal_to_probed"], (
            "the even split gives other bits than the probed split"
        )
    elif arm == "even":
        assert torch.equal(ref, planned_bits)
    _spot_check(case, a, b, out)


# ---------------------------------------------------------------------------
# Coverage of the routes by the cases
# ---------------------------------------------------------------------------


def test_route_coverage(record_property):
    """The cases reach, on this device, both weight-gradient k tiles for both output
    dtypes with the tail reduce, and (where the plan probes it) the probed queue split
    of the data gradient."""
    device = _device()
    arch = cb.arch_for(device)
    seen = {}
    for case in CASES:
        out_dtype = case.out_dtype if case.op == "wgrad" else None
        if not cb.generated_program_available(device, op=case.op, out_dtype=out_dtype):
            continue
        a, b, offs, out = _inputs(case, device)
        launch = _prepare(case, a, b, offs, out)
        plan = launch.plan
        key = (
            case.op,
            str(case.out_dtype).replace("torch.", ""),
            plan.get("tile_k"),
            "+".join(launch.stages) or "none",
            plan.get("queue_split"),
        )
        seen.setdefault(key, []).append(case.label)
        del a, b, offs, out, launch
    rows = sorted([*key, len(labels)] for key, labels in seen.items())
    record_property("coverage", json.dumps(dict(arch=arch, rows=rows)))
    print("COVERAGE " + json.dumps(dict(arch=arch, rows=rows)))
    for dtype in ("bfloat16", "float32"):
        for tile_k in (256, 512):
            assert any(
                key[0] == "wgrad"
                and key[1] == dtype
                and key[2] == tile_k
                and key[3].endswith("+tail_reduce")
                for key in seen
            ), (
                f"no case reaches the {dtype} wgrad k{tile_k} program with the tail reduce"
            )
    assert any(key[0] == "fwd" for key in seen)
    assert any(key[0] == "dgrad" for key in seen)
    if arch in HOST_PLAN_CONSTANTS["domain_split_archs"]:
        assert any(key[0] == "dgrad" and key[4] == "probed" for key in seen)
        assert any(key[0] == "dgrad" and key[4] == "even" for key in seen)


# ---------------------------------------------------------------------------
# Across fresh processes
# ---------------------------------------------------------------------------


def _digest_cases(labels):
    """sha256 of every case's output after a settled planned launch, with its plan and the
    split that launch passed."""
    device = torch.device("cuda", 0)
    digests = {}
    for label in labels:
        case = CASE_BY_LABEL[label]
        out_dtype = case.out_dtype if case.op == "wgrad" else None
        if not cb.generated_program_available(device, op=case.op, out_dtype=out_dtype):
            continue
        a, b, offs, out = _inputs(case, device)
        launch = _prepare(case, a, b, offs, out)
        split = _settle_probe(launch, device)
        out.view(torch.uint8).fill_(FILL_BYTE)
        launch.launch()
        torch.cuda.synchronize(device)
        data = out.contiguous().view(torch.uint8).reshape(-1).cpu().numpy()
        plan = _plan_record(launch)
        digests[label] = dict(
            sha256=hashlib.sha256(data).hexdigest(),
            plan={k: v for k, v in plan.items() if k not in ("c0", "c1")},
            stages=list(launch.stages),
            split=list(split) if split is not None else None,
        )
        del a, b, offs, out, launch, data
    return digests


def _worker_main(labels):
    print("DIGESTS_JSON " + json.dumps(_digest_cases(labels), sort_keys=True))


def _fresh_process_digests(labels):
    """Digests of ``labels`` from a new interpreter that imports this module by name."""
    here = Path(__file__).resolve()
    root = here.parents[2]
    module = ".".join(here.relative_to(root).with_suffix("").parts)
    code = (
        f"import sys; sys.path.insert(0, {str(root)!r}); import importlib; "
        f"importlib.import_module({module!r})._worker_main(sys.argv[1:])"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code, *labels],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=WORKER_TIMEOUT_S,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    lines = [x for x in proc.stdout.splitlines() if x.startswith("DIGESTS_JSON ")]
    assert len(lines) == 1, proc.stdout[-2000:]
    return json.loads(lines[0][len("DIGESTS_JSON ") :])


def test_fresh_processes_give_the_same_bits(record_property):
    """Two fresh processes (each with its own JIT load, stream workspaces and queue probe)
    and this one give the same output bits for every case."""
    device = _device()
    labels = list(CASE_BY_LABEL)
    here = _digest_cases(labels)
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    first = _fresh_process_digests(labels)
    second = _fresh_process_digests(labels)
    assert here, "no case has a generated program on this device"
    differ = []
    for label, d in here.items():
        for name, other in (("first", first), ("second", second)):
            o = other.get(label)
            if o is None or o["sha256"] != d["sha256"] or o["plan"] != d["plan"]:
                differ.append((label, name))
    splits = sorted(
        {
            json.dumps([d["split"] for d in (here[x], first[x], second[x])])
            for x in here
            if here[x]["split"] is not None
        }
    )
    summary = dict(
        arch=cb.arch_for(device),
        cases=len(here),
        processes=3,
        differ=differ,
        probed_splits_here_first_second=splits,
    )
    record_property("fresh_processes", json.dumps(summary, sort_keys=True))
    print("XPROC " + json.dumps(summary, sort_keys=True))
    assert not differ, (
        f"{len(differ)} case digests differ across processes: {differ[:8]}"
    )
