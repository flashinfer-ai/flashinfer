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

Reproducibility of the non-deterministic Cake MoE grouped GEMM against the
deterministic package.

Over the 58 MoE projection rows of the benchmark and the edge cases of the
deterministic package's determinism suite (seeded fixed inputs per case):

* the module ``flashinfer.cake_moe_grouped_gemm_nondet_train`` gives the
  deterministic package's bits for ``fwd`` and ``dgrad``;
* ``fwd`` and ``dgrad`` of the non-deterministic package give the deterministic
  package's bits, the data gradient's own sm_100a L2 evict-first build included
  (it runs where the deterministic package launches its wide tile);
* the non-deterministic weight gradient, launched ``REPEATS`` times into an
  output filled with zero bytes and ``REPEATS`` times into one filled with
  ``FILL_BYTE`` (the split-tail slots poisoned with ``FILL_BYTE`` before every
  launch), gives the bits of its own first launch every time and stays within
  the deterministic package's budget of the deterministic weight gradient; a
  case without a split one-launch tail gives the deterministic bits (``REPRO``
  lines report the comparison);
* with the tail split switched off in both packages, the 512-wide k tile's
  ordered one-launch tail builds give the deterministic builds' bits (every
  whole tile is reduced in the deterministic order);
* the cases reach the ordered split tail on this device for both output dtypes;
* on the one-group E1 rows (``E1_SUM_M``), where the deterministic package's own
  fp32 weight gradient exceeds ``FP32_REL_L2``, the non-deterministic fp32 weight
  gradient's rel-L2 and worst / max |ref| against the fp64 reference are each at
  most ``DET_ERROR_RATIO`` times the deterministic package's at the same inputs;
* row-padded ``g`` and ``x`` on the shape of the 256-wide k tile's ``main_wave``
  weight gradient give the deterministic package's bits.

On the CPU, the registry mirrors the deterministic package's routes and stages,
and the host plan constants are the deterministic package's but for those of
``OWN_CONSTANTS``.
"""

import dataclasses
import json
import math
import zlib

import pytest
import torch

pytest.importorskip("tvm_ffi")

import flashinfer.cake_moe_grouped_gemm_nondet_train as facade
from tests.cake.test_helpers.cake_package import cake_package

cb = cake_package("cake_moe_grouped_gemm_nondet_train", "cake_backend")
cake_jit = cake_package("cake_moe_grouped_gemm_nondet_train", "cake_jit")
det_cb = cake_package("cake_moe_grouped_gemm_det_train", "cake_backend")
det_jit = cake_package("cake_moe_grouped_gemm_det_train", "cake_jit")
HOST_PLAN_CONSTANTS = cake_jit.HOST_PLAN_CONSTANTS

REPEATS = 10  # launches per output pre-state (zero bytes, then FILL_BYTE)
RACE_REPEATS = 50  # the flashinfer-ai/flashinfer#6570 shape
FILL_BYTE = 0x5A
ATOL = RTOL = 1e-2  # bf16 outputs
FP32_REL_L2 = 1e-4  # fp32 weight gradient
# The #5678 E1 customer rows reduce 112,576 / 136,064 rows in one group.  There the
# deterministic package's own fp32 weight gradient has rel-L2 1.17e-4 / 1.41e-4
# against the fp64 reference on GB300, above FP32_REL_L2, so these rows use the
# deterministic package's error at the same inputs as the bar: per output, rel-L2 and
# worst / max |ref| each at most DET_ERROR_RATIO times the deterministic package's.
E1_SUM_M = (112576, 136064)
DET_ERROR_RATIO = 1.1

# The 58 MoE projection rows of benchmarks/bench_cake_moe_grouped_gemm_nondet_train.py.
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
ODD_PROJ = (1280, 1792)
RACE_PROJ = (1024, 6144)
EDGE_SIZES = {
    "one_group": [4000],
    "empty_groups": [0, 3000, 0, 0, 517, 4096, 0, 1],
    "one_nonempty": [0, 0, 5000, 0],
    "nonmult_m": [1, 255, 257, 1023, 4097, 63, 65],
    "all_empty": [0, 0, 0, 0],
    "e64_m2048": [2048] * 64,
    "e128_m512": [512] * 128,
    # Long groups: the 512-wide k tile.
    "e2_long": [16400, 16666],
    "e3_long_empty": [16400, 0, 16666],
}
ODD_SIZES = ("empty_groups", "nonmult_m")
RACE_SIZES = [32] * 256
# Weight gradient only: the 256-wide k tile (K % 512 != 0) over two long groups,
# with 84 / 112 / 224 output tiles, so that on every architecture's cluster count
# some case splits its last wave and sums it with the ordered tail reduce.
K256_TAIL_K = 1792
K256_TAIL_N_BLOCKS = (6, 8, 16)
K256_TAIL_SIZES = [8192, 8191]

# The 256-wide k tile's wave-synchronised weight gradient (``main_wave`` on the
# architectures of ``wave_sync_archs[256]``, else ``main``), on row-padded operands.
K256_WAVE_NK = (4096, 4096)
K256_WAVE_SIZES = [2048] * 8
ROW_PAD = 256

OPS = ("fwd", "dgrad", "wgrad")
INT_VIEW = {torch.bfloat16: torch.int16, torch.float32: torch.int32}

# The host plan constants of this package alone: the keys of the one-launch split
# tail and of the queue walk.  Every other constant is the deterministic package's.
OWN_CONSTANTS = frozenset(
    {
        "unordered_tail_archs",
        "ordered_tail_archs",
        "atomic_tail_plan",
        "queue_walk",
        "queue_sched_words",
    }
)
# The queue walk's stages (``cake_backend.wgrad_queue_walk``): this package's alone.
QUEUE_STAGES = frozenset({cb.QUEUE_STAGE, cb.QUEUE_STAGE + cb.ACC_SUFFIX})
# Data-gradient cases of the evict-first rule (at least every architecture's
# ``dgrad_evict_first_min_rows`` rows per group on average,
# ``K >= dgrad_evict_first_min_k``): (label, sizes, N, K),
# over group sizes off every tile multiple, with a one-row group (the second and
# third) and an empty group (the third).
EF_CASES = (
    ("e2_long", (16390, 16500), 64, 4096),
    ("one_row_off_tile", (1, 16383, 16385, 32767), 1088, 4608),
    ("empty_one_row", (0, 49151, 1, 16384), 4096, 6144),
)


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


def _skewed_sizes(e, sum_m=EXPERT_COUNT_SUM_M, align=32):
    """The skewed family of ``E4_GROUPS`` over ``e`` groups: weights ``e, e - 1, ..., 1``,
    each group ``sum_m * w / sum(w)`` rows floored to ``align``, the ``align``-row chunks
    left to the largest remainders (ties: the lower group).  ``e`` = 4 at 112,576 /
    136,064 rows gives ``E4_GROUPS``' skewed sizes."""
    weights = list(range(e, 0, -1))
    total = sum(weights)
    sizes = [(sum_m * w // total) // align * align for w in weights]
    rest = [sum_m * w / total - s for w, s in zip(weights, sizes, strict=True)]
    for i in sorted(range(e), key=lambda i: (-rest[i], i))[
        : (sum_m - sum(sizes)) // align
    ]:
        sizes[i] += align
    assert sum(sizes) == sum_m, sizes
    return sizes


# The queue walk's uneven rows (``cake_backend.wgrad_queue_walk``): E8 / E16 / E32
# over ``EXPERT_COUNT_SUM_M`` rows, skewed and one-empty (the skewed sizes with the
# last group's rows moved to the first).
QUEUE_SIZES = {}
for _e in EXPERT_COUNTS:
    _sizes = _skewed_sizes(_e)
    QUEUE_SIZES[f"e{_e}_m{EXPERT_COUNT_SUM_M}_skewed"] = _sizes
    QUEUE_SIZES[f"e{_e}_m{EXPERT_COUNT_SUM_M}_one_empty"] = (
        [_sizes[0] + _sizes[-1]] + _sizes[1:-1] + [0]
    )
assert _skewed_sizes(4, 112576) == E4_GROUPS[112576]["skewed"]
assert _skewed_sizes(4, 136064) == E4_GROUPS[136064]["skewed"]


def queue_cases():
    cases = []
    for proj, nk in BENCH_PROJ.items():
        for label, sizes in QUEUE_SIZES.items():
            for out_dtype in (torch.bfloat16, torch.float32):
                cases.append(
                    _case(f"{proj}_{label}", "wgrad", nk, sizes, out_dtype=out_dtype)
                )
    return cases


def _edge_variants(label, nk, sizes, repeats=REPEATS):
    cases = [_case(label, "fwd", nk, sizes), _case(label, "dgrad", nk, sizes)]
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


CASES = bench_cases() + edge_cases() + queue_cases()
CASE_BY_LABEL = {case.label: case for case in CASES}
assert len(CASE_BY_LABEL) == len(CASES)
K512_LABELS = [
    case.label
    for case in CASES
    if case.op == "wgrad" and sum(case.sizes) and case.k % 512 == 0
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


def _require_programs(case, device):
    out_dtype = case.out_dtype if case.op == "wgrad" else None
    for backend in (cb, det_cb):
        if not backend.generated_program_available(
            device, op=case.op, out_dtype=out_dtype
        ):
            pytest.skip(
                f"no generated program for {case.op} registered for "
                f"{backend.arch_for(device)} in {backend.__name__}"
            )


def _inputs(case, device):
    """Fixed inputs of ``case``: seeded from its label."""
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
    elif case.op == "dgrad":
        a = randn(sum_m, case.n)
        b = randn(num_groups, case.n, case.k, scale=scale)
    else:
        a = randn(sum_m, case.n)
        b = randn(sum_m, case.k, scale=scale)
    return a, b, offs


def _out(case, device):
    sum_m, num_groups = sum(case.sizes), len(case.sizes)
    if case.op == "fwd":
        return torch.empty(sum_m, case.n, dtype=torch.bfloat16, device=device)
    if case.op == "dgrad":
        return torch.empty(sum_m, case.k, dtype=torch.bfloat16, device=device)
    return torch.empty(num_groups, case.n, case.k, dtype=case.out_dtype, device=device)


def _prepare(backend, case, a, b, offs, out):
    """A prepared launch of ``case`` from the deterministic (``det_cb``) or the
    non-deterministic (``cb``) package."""
    if backend is det_cb:
        if case.op == "fwd":
            return det_cb.cake_moe_grouped_gemm_det_train_prepare_forward(
                a, b, offs, out=out
            )
        if case.op == "dgrad":
            return det_cb.cake_moe_grouped_gemm_det_train_prepare_dgrad(
                a, b, offs, out=out
            )
        return det_cb.cake_moe_grouped_gemm_det_train_prepare_wgrad(
            a, b, offs, out=out, out_dtype=case.out_dtype
        )
    if case.op == "fwd":
        return cb.cake_moe_grouped_gemm_nondet_train_prepare_forward(
            a, b, offs, out=out
        )
    if case.op == "dgrad":
        return cb.cake_moe_grouped_gemm_nondet_train_prepare_dgrad(a, b, offs, out=out)
    return cb.cake_moe_grouped_gemm_nondet_train_prepare_wgrad(
        a, b, offs, out=out, out_dtype=case.out_dtype
    )


def _facade(case, a, b, offs, out):
    """The facade's call of a ``fwd`` / ``dgrad`` case (non-deterministic only)."""
    if case.op == "fwd":
        return facade.cake_moe_grouped_gemm_nondet_train_forward(a, b, offs, out=out)
    return facade.cake_moe_grouped_gemm_nondet_train_dgrad(a, b, offs, out=out)


def _bits(out):
    return out.view(INT_VIEW[out.dtype])


def _poison_tail_sums(device):
    """Fill the split-tail slots kept for the current stream with ``FILL_BYTE`` bytes (they
    have no rest state: the ordered builds overwrite every slot they read)."""
    buffer = cb.stream_workspace("tail_sums", device)
    if buffer is not None:
        buffer.view(torch.uint8).fill_(FILL_BYTE)


def _budget(out, ref, out_dtype):
    """Error of ``out`` against ``ref`` in the budget's unit: the worst
    ``|out - ref| / (atol + rtol |ref|)`` for bf16 (<= 1 passes), the rel-L2 over
    ``FP32_REL_L2`` for fp32 (<= 1 passes)."""
    out, ref = out.float(), ref.float()
    if out_dtype == torch.bfloat16:
        return float(((out - ref).abs() / (ATOL + RTOL * ref.abs())).max())
    norm = float(ref.norm())
    if norm == 0.0:
        return 0.0 if float(out.norm()) == 0.0 else math.inf
    return float((out - ref).norm()) / norm / FP32_REL_L2


def _plan_record(launch):
    keys = (
        "grid",
        "tile_n",
        "tile_k",
        "tail_tile_k",
        "num_tail",
        "tail_splits",
        "unordered_tail",
        "ordered_tail",
        "raster_rows",
        "wave_mode",
        "launches",
    )
    return {key: launch.plan[key] for key in keys if key in launch.plan}


# ---------------------------------------------------------------------------
# Registry against the deterministic package (CPU)
# ---------------------------------------------------------------------------


def _one_launch_plan(arg_plan):
    """The deterministic argument plan of a k512 program with its fp32 partials replaced
    by the one-launch split tail's slots and tickets."""
    out = []
    for item in arg_plan:
        if tuple(item) == ("buffer", "partials"):
            out += [("buffer", "tail_sums"), ("buffer", "tail_tickets")]
        else:
            out.append(tuple(item))
    return out


def test_routes_mirror_the_deterministic_package():
    """The routes are the deterministic package's with ``det`` removed from their names,
    over the same architectures and, per architecture, the same stages bound to
    programs with the same launch shape.  The fwd, dgrad and 256-wide k tile programs
    take the deterministic argument plans; on the architectures of
    ``ordered_tail_archs`` (every architecture of the k512 routes) the 512-wide k
    tile's take the one-launch split tail's ``tail_sums`` slots and ``tail_tickets`` in
    place of the partials, and those routes serve no ``tail_reduce`` stage there.  A program may also take the row bases of ``A`` / ``B``
    (``cake_backend.ROW_BASES``: the same tensors again), which a deterministic build
    of an older kernel does not.  A wave-synchronised stage may be missing on an
    architecture whose host plan never selects it (not in ``wave_sync_archs`` for the
    route's k tile).  The accumulate stages (``<stage>_acc``) are left out of the
    mirror on both sides; their twins are mirrored, and ``test_cake_moe_grouped_gemm_nondet_train_acc.py`` checks them against
    their twins.  The queue walk's stages (``QUEUE_STAGES``) are this package's alone
    and left out (``test_queue_stages_take_the_half_release_arguments``).  The L2 evict-first twins of the wide tile (``main_wide_ef``) are
    copies like every other fwd / dgrad stage.  (A CPU check; it lives here because
    the backend suite imports no package but its own.)"""
    det_routes, det_programs = det_jit.ROUTES, det_jit.PROGRAMS
    routes, programs = cake_jit.ROUTES, cake_jit.PROGRAMS
    assert set(routes) == {
        name.replace("cake_moe_grouped_det_gemm", "cake_moe_grouped_gemm")
        for name in det_routes
    }
    for det_name, det_route in det_routes.items():
        name = det_name.replace("cake_moe_grouped_det_gemm", "cake_moe_grouped_gemm")
        route = routes[name]
        assert route["op"] == det_route["op"]
        assert route["arches"] == det_route["arches"]
        for arch in route["arches"]:
            tile_k = 512 if name.endswith("_k512") else 256
            one_launch = cb.wgrad_ordered_tail(
                tile_k, arch, HOST_PLAN_CONSTANTS
            ) or cb.wgrad_unordered_tail(tile_k, arch, HOST_PLAN_CONSTANTS)
            if route["op"] == "wgrad":
                assert one_launch == (tile_k == 512), (name, arch)
            det_stages = {
                s["name"]
                for s in det_route["stages"]
                if arch in det_programs[s["program"]]["arches"]
                and not s["name"].endswith(cb.ACC_SUFFIX)
            }
            stages = {
                s["name"]
                for s in route["stages"]
                if arch in programs[s["program"]]["arches"]
                and not s["name"].endswith(cb.ACC_SUFFIX)
                and s["name"] not in QUEUE_STAGES
            }
            expected_stages = det_stages - {"tail_reduce"} if one_launch else det_stages
            unselected = set()
            if route["op"] == "wgrad":
                tile_k = 512 if name.endswith("_k512") else 256
                if arch not in HOST_PLAN_CONSTANTS["wave_sync_archs"][tile_k]:
                    unselected = {"main_wave", "main_wave_halves"}
            assert stages <= expected_stages, (name, arch, stages - expected_stages)
            assert expected_stages - stages <= unselected, (
                name,
                arch,
                expected_stages - stages,
            )
            for stage in stages:
                det_binding = det_jit.stage_binding(det_name, stage, arch)
                binding = cake_jit.stage_binding(name, stage, arch)
                assert binding["specializations"] == det_binding["specializations"]
                det_program = det_programs[det_binding["program"]]
                program = programs[binding["program"]]
                assert program["launch"] == det_program["launch"], (name, stage, arch)
                assert program["ffi_entry"] == det_program["ffi_entry"]
                row_bases = {("buffer", rows) for rows in cb.ROW_BASES}
                expected = [
                    i
                    for i in (
                        _one_launch_plan(det_program["arg_plan"])
                        if one_launch
                        else [tuple(i) for i in det_program["arg_plan"]]
                    )
                    if i not in row_bases
                ]
                plan = [tuple(i) for i in program["arg_plan"]]
                assert [i for i in plan if i not in row_bases] == expected, (
                    name,
                    stage,
                    arch,
                )
                assert program["library"].startswith("cake_moe_grouped_gemm_")
                assert binding["program"].startswith("cake_moe_grouped_gemm_")


def test_queue_stages_take_the_half_release_arguments():
    """The queue walk's stages: served by both 512-wide k tile routes (the accumulate
    stage by the fp32 route alone) on the architectures of the ``queue_walk`` rules and
    nowhere else, each program with the launch shape and FFI entry of the route's
    ``main_halves`` program on that architecture and its argument plan plus the
    ``sched`` buffer, which the weight-gradient binding binds to the ``queue_sched``
    stream workspace.  Every rule names a stage the queue walk may stand in for, an
    output kind and a group range that the routes serve.  (A CPU check.)"""
    assert cb.BINDING_WORKSPACE_KINDS["wgrad"]["sched"] == cb.QUEUE_SCHED
    assert cb.QUEUE_SCHED in cb.WORKSPACE_KINDS
    assert cb.QUEUE_SCHED not in cb.SIZED_STREAM_WORKSPACES
    rules = HOST_PLAN_CONSTANTS["queue_walk"]
    rule_arches = {rule["arch"] for rule in rules}
    for rule in rules:
        assert rule["stage"] in cb.QUEUE_SELECTED_STAGES, rule
        assert set(rule["outputs"]) <= {"bfloat16", "float32", "accumulate"}, rule
        low, high = rule["groups"]
        assert 1 <= low <= high, rule
        assert rule["sm_counts"] and all(int(v) > 0 for v in rule["sm_counts"]), rule
    for out_dtype in ("bfloat16", "float32"):
        name = cake_jit.record_name("wgrad", out_dtype=out_dtype, tile_k=512)
        for arch in cake_jit.ROUTES[name]["arches"]:
            served = cake_jit.stage_served(name, cb.QUEUE_STAGE, arch)
            assert served == (arch in rule_arches), (name, arch)
            acc_served = cake_jit.stage_served(
                name, cb.QUEUE_STAGE + cb.ACC_SUFFIX, arch
            )
            assert acc_served == (served and out_dtype == "float32"), (name, arch)
            if not served:
                continue
            for stage, twin in (
                (cb.QUEUE_STAGE, "main_halves"),
                (cb.QUEUE_STAGE + cb.ACC_SUFFIX, "main_halves" + cb.ACC_SUFFIX),
            ):
                if not cake_jit.stage_served(name, stage, arch):
                    continue
                binding = cake_jit.stage_binding(name, stage, arch)
                twin_binding = cake_jit.stage_binding(name, twin, arch)
                assert binding["specializations"] == twin_binding["specializations"]
                program = cake_jit.PROGRAMS[binding["program"]]
                halves = cake_jit.PROGRAMS[twin_binding["program"]]
                assert program["launch"] == halves["launch"], (name, stage, arch)
                assert program["ffi_entry"] == halves["ffi_entry"], (name, stage, arch)
                plan = [tuple(i) for i in program["arg_plan"]]
                assert plan.count(("buffer", "sched")) == 1, (name, stage, arch)
                assert [i for i in plan if i != ("buffer", "sched")] == [
                    tuple(i) for i in halves["arg_plan"]
                ], (name, stage, arch)
            if out_dtype == "float32":
                for rule in rules:
                    if rule["arch"] == arch and "accumulate" in rule["outputs"]:
                        assert acc_served, (name, arch, rule)


def test_host_plan_constants_mirror_the_deterministic_package():
    """Every host plan constant but those of ``OWN_CONSTANTS`` is the deterministic
    package's, so this package's host takes the deterministic stage rules at the
    deterministic thresholds (the evict-first twins' included).  (A CPU check.)"""
    det_c, c = det_jit.HOST_PLAN_CONSTANTS, HOST_PLAN_CONSTANTS
    assert set(det_c) - OWN_CONSTANTS == set(c) - OWN_CONSTANTS
    for key in sorted(set(det_c) - OWN_CONSTANTS):
        assert c[key] == det_c[key], key


# ---------------------------------------------------------------------------
# Against the deterministic package
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("label", list(CASE_BY_LABEL))
def test_against_the_deterministic_package(label, record_property):
    case = CASE_BY_LABEL[label]
    device = _device()
    _require_programs(case, device)
    a, b, offs = _inputs(case, device)
    det_out = _out(case, device)
    det = _prepare(det_cb, case, a, b, offs, det_out)
    det_out.view(torch.uint8).fill_(0)
    det.launch()
    if case.op != "wgrad":
        # The facade runs this package; its fwd / dgrad give the deterministic bits.
        facade_out = _out(case, device)
        facade_out.view(torch.uint8).fill_(FILL_BYTE)
        _facade(case, a, b, offs, facade_out)
        torch.cuda.synchronize(device)
        assert torch.equal(_bits(facade_out), _bits(det_out)), "facade fwd / dgrad"
        del facade_out
    out = _out(case, device)
    launch = _prepare(cb, case, a, b, offs, out)
    record = dict(
        case=label,
        op=case.op,
        arch=cb.arch_for(device),
        sm_count=cb.sm_count(device),
        record_name=launch.record_name,
        stages=list(launch.stages),
        det_stages=list(det.stages),
        plan=_plan_record(launch),
        sizes_e=len(case.sizes),
        sum_m=sum(case.sizes),
        n=case.n,
        k=case.k,
        out_dtype=str(case.out_dtype).replace("torch.", ""),
    )
    if case.op != "wgrad":
        # fwd / dgrad: the deterministic package's stages and programs' bits.
        assert list(launch.stages) == list(det.stages), (launch.stages, det.stages)
        for fill in (0, FILL_BYTE):
            out.view(torch.uint8).fill_(fill)
            launch.launch()
            torch.cuda.synchronize(device)
            assert torch.equal(_bits(out), _bits(det_out)), f"fill 0x{fill:02x}"
        record.update(equal_to_det=True)
        print("REPRO " + json.dumps(record, sort_keys=True))
        record_property("repro", json.dumps(record, sort_keys=True))
        return
    first = None
    launches = mismatches = differ_from_det = 0
    worst_vs_det = worst_vs_first = 0.0
    worst_elems = 0
    for fill in (0, FILL_BYTE):
        for _ in range(case.repeats):
            _poison_tail_sums(device)
            out.view(torch.uint8).fill_(fill)
            launch.launch()
            launches += 1
            worst_vs_det = max(worst_vs_det, _budget(out, det_out, case.out_dtype))
            if not torch.equal(_bits(out), _bits(det_out)):
                differ_from_det += 1
            if first is None:
                first = out.clone()
                continue
            if not torch.equal(_bits(out), _bits(first)):
                mismatches += 1
                worst_elems = max(worst_elems, int((_bits(out) != _bits(first)).sum()))
                worst_vs_first = max(
                    worst_vs_first, _budget(out, first, case.out_dtype)
                )
    torch.cuda.synchronize(device)
    # Empty groups are exact zeros in every launch's output (checked on the last).
    ends = [sum(case.sizes[: i + 1]) for i in range(len(case.sizes))]
    for e, (s, t) in enumerate(zip([0] + ends[:-1], ends, strict=True)):
        if t == s:
            assert int(_bits(out[e]).count_nonzero()) == 0, f"empty group {e}"
    # Where the two packages' ``k512_min_rows_per_group`` differ, the deterministic
    # package may take the 512-wide k tile where this package takes the 256-wide one.
    tile, det_tile = int(launch.plan["tile_k"]), int(det.plan["tile_k"])
    same_tile = tile == det_tile
    if not same_tile:
        arch = cb.arch_for(device)
        avg = sum(case.sizes) // max(len(case.sizes), 1)
        det_min = int(det_jit.HOST_PLAN_CONSTANTS["k512_min_rows_per_group"][arch])
        own_min = int(HOST_PLAN_CONSTANTS["k512_min_rows_per_group"][arch])
        assert (det_tile, tile) == (512, 256), (det_tile, tile)
        assert det_min <= avg < own_min, (det_min, avg, own_min)
    record.update(
        same_tile_as_det=same_tile,
        launches=launches,
        run_to_run_mismatches=mismatches,
        worst_diff_elems=worst_elems,
        worst_vs_first_budget=worst_vs_first,
        launches_differing_from_det=differ_from_det,
        worst_vs_det_budget=worst_vs_det,
        exercised=bool(
            (launch.plan["ordered_tail"] or launch.plan["unordered_tail"])
            and launch.plan["tail_splits"] > 1
        ),
        order_varying=launch.order_varying,
    )
    print("REPRO " + json.dumps(record, sort_keys=True))
    record_property("repro", json.dumps(record, sort_keys=True))
    assert not launch.order_varying, record
    # Every launch gives the bits of the first, whatever the output and the slots held.
    assert mismatches == 0 and worst_vs_first == 0.0, record
    if not record["exercised"] and same_tile:
        # No split one-launch tail: the deterministic programs' order, the same bits.
        assert differ_from_det == 0, record
    # A split tail's bits may differ from the deterministic package's (another tail
    # plan and group order), within its budget.
    assert worst_vs_det <= 1.0, record


@pytest.mark.parametrize("label,sizes,n,k", EF_CASES, ids=[c[0] for c in EF_CASES])
def test_dgrad_evict_first_gives_the_deterministic_bits(
    label, sizes, n, k, monkeypatch
):
    """Where ``dgrad_evict_first`` holds, the data gradient launches the L2 evict-first
    twin of its wide tile (``main_wide_ef``), as the deterministic package does.  The
    cache policy changes no arithmetic: over a one-row group, an empty group and group sizes
    off every tile multiple, dX has the bits of the deterministic package's data
    gradient, of this package's wide tile (``main_wide``) and of its 256-column tile
    (``main``), whatever bytes the output held."""
    device = _device()
    case = Case(f"dgrad_ef_{label}", "dgrad", n, k, tuple(sizes), torch.bfloat16)
    _require_programs(case, device)
    c = HOST_PLAN_CONSTANTS
    arch = cb.arch_for(device)
    sum_m, groups = sum(sizes), len(sizes)
    # Every case is over the rule's thresholds: the rule holds on its architectures.
    assert sum_m // groups >= max(
        int(v) for v in c["dgrad_evict_first_min_rows"].values()
    ), label
    assert k >= int(c["dgrad_evict_first_min_k"]), label
    assert k % int(c["wide_block_n"]) == 0, label
    assert any(size % 128 for size in sizes), label
    a, b, offs = _inputs(case, device)
    det_out = _out(case, device)
    det = _prepare(det_cb, case, a, b, offs, det_out)
    det_out.view(torch.uint8).fill_(0)
    det.launch()
    out = _out(case, device)
    launch = _prepare(cb, case, a, b, offs, out)
    name = launch.record_name
    wide = arch in c["wide_dgrad_archs"] and cake_jit.stage_served(
        name, "main_wide", arch
    )
    ef = (
        wide
        and cb.dgrad_evict_first(k, arch, sum_m, groups, c)
        and cake_jit.stage_served(name, "main_wide_ef", arch)
    )
    ts33 = (
        wide
        and not ef
        and cb.dgrad_wide_ts33(k, arch, sum_m, groups, c)
        and cake_jit.stage_served(name, "main_wide_ts33", arch)
    )
    assert ef == (arch in c["dgrad_evict_first_archs"]), arch
    assert launch.stages[0] == (
        "main_wide_ef"
        if ef
        else "main_wide_ts33"
        if ts33
        else "main_wide"
        if wide
        else "main"
    )
    assert list(det.stages) == list(launch.stages), (det.stages, launch.stages)
    for fill in (0, FILL_BYTE):
        out.view(torch.uint8).fill_(fill)
        launch.launch()
        torch.cuda.synchronize(device)
        assert torch.equal(_bits(out), _bits(det_out)), f"fill 0x{fill:02x}"
    ends = [sum(sizes[: i + 1]) for i in range(groups)]
    for s, t in zip([0] + ends[:-1], ends, strict=True):
        if t == s + 1:
            assert int(_bits(out[s:t]).count_nonzero()) > 0, label
    if not ef:
        pytest.skip(f"no evict-first data-gradient tile on {arch}")
    monkeypatch.setitem(
        c,
        "dgrad_evict_first_min_rows",
        dict.fromkeys(c["dgrad_evict_first_min_rows"], 1 << 30),
    )
    plain_out = _out(case, device)
    plain = _prepare(cb, case, a, b, offs, plain_out)
    assert plain.stages[0] == "main_wide", plain.stages
    assert plain.plan["tile_n"] == launch.plan["tile_n"]
    plain_out.view(torch.uint8).fill_(FILL_BYTE)
    plain.launch()
    torch.cuda.synchronize(device)
    assert torch.equal(_bits(plain_out), _bits(out)), "main_wide"
    del plain_out
    monkeypatch.setitem(c, "wide_dgrad_archs", [])
    narrow_out = _out(case, device)
    narrow = _prepare(cb, case, a, b, offs, narrow_out)
    assert narrow.stages[0] == "main", narrow.stages
    narrow_out.view(torch.uint8).fill_(FILL_BYTE)
    narrow.launch()
    torch.cuda.synchronize(device)
    assert torch.equal(_bits(narrow_out), _bits(out)), "main"


def _queue_as_selected(launch, det, case, device):
    """``launch.stages`` with a bound queue-walk stage replaced by the main stage it stands
    in for (the deterministic launch's first stage, which both packages select alike),
    after checking that a ``queue_walk`` rule names that stage, the output and the group
    count on this device."""
    stages = tuple(launch.stages)
    if not stages or stages[0] not in QUEUE_STAGES:
        return stages
    selected = det.stages[0].removesuffix(cb.ACC_SUFFIX)
    fused = stages[0].endswith(cb.ACC_SUFFIX)
    output = (
        "accumulate"
        if fused
        else {torch.bfloat16: "bfloat16", torch.float32: "float32"}[case.out_dtype]
    )
    assert cb.wgrad_queue_walk(
        selected,
        output,
        cb.arch_for(device),
        cb.sm_count(device),
        len(case.sizes),
    ), (stages, det.stages)
    return (selected + (cb.ACC_SUFFIX if fused else ""),) + stages[1:]


@pytest.mark.parametrize("label", K512_LABELS)
def test_unsplit_k512_builds_give_the_deterministic_bits(label, monkeypatch):
    """With the tail split switched off in both packages (``max_splits`` 1), a weight
    gradient on the 512-wide k tile runs every tile whole in both, and the ordered
    one-launch tail build gives the deterministic build's bits for every form the case
    selects, the queue walk included where a ``queue_walk`` rule binds it (it keeps
    every tile's k-step order)."""
    case = CASE_BY_LABEL[label]
    device = _device()
    _require_programs(case, device)
    monkeypatch.setitem(HOST_PLAN_CONSTANTS["atomic_tail_plan"], "max_splits", 1)
    monkeypatch.setitem(det_jit.HOST_PLAN_CONSTANTS["tail_plan"], "max_splits", 1)
    a, b, offs = _inputs(case, device)
    det_out, out = _out(case, device), _out(case, device)
    det = _prepare(det_cb, case, a, b, offs, det_out)
    launch = _prepare(cb, case, a, b, offs, out)
    if launch.plan["tile_k"] != 512:
        pytest.skip("the case takes the 256-wide k tile on this device")
    assert _queue_as_selected(launch, det, case, device) == det.stages
    assert launch.plan["tail_splits"] == det.plan["tail_splits"] == 1
    for fill in (0, FILL_BYTE):
        out.view(torch.uint8).fill_(fill)
        det_out.view(torch.uint8).fill_(fill ^ 0xFF)
        launch.launch()
        det.launch()
        torch.cuda.synchronize(device)
        assert torch.equal(_bits(out), _bits(det_out)), (label, launch.stages)


def _row_padded(t):
    """``t`` as the leading columns of a wider tensor whose other columns hold NaN."""
    cols = t.size(-1)
    wide = torch.full(
        (*t.shape[:-1], cols + ROW_PAD), float("nan"), dtype=t.dtype, device=t.device
    )
    wide[..., :cols] = t
    return wide[..., :cols]


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
def test_row_padded_wgrad_gives_the_deterministic_bits(out_dtype):
    """Row-padded ``g`` and ``x`` (the leading columns of wider NaN-filled tensors) on the
    shape of the 256-wide k tile's ``main_wave`` weight gradient (sm_100a / sm_103a;
    ``main`` on sm_107a): both packages take the same stages, and the non-deterministic
    package gives the deterministic package's bits."""
    case = _case(
        "k256_wave_row_padded", "wgrad", K256_WAVE_NK, K256_WAVE_SIZES, out_dtype
    )
    device = _device()
    _require_programs(case, device)
    a, b, offs = _inputs(case, device)
    a, b = _row_padded(a), _row_padded(b)
    out, det_out = _out(case, device), _out(case, device)
    launch = _prepare(cb, case, a, b, offs, out)
    det = _prepare(det_cb, case, a, b, offs, det_out)
    wave = cb.arch_for(device) in HOST_PLAN_CONSTANTS["wave_sync_archs"][256]
    assert launch.plan["tile_k"] == 256, launch.plan
    assert launch.stages[0] == ("main_wave" if wave else "main"), launch.stages
    assert launch.stages == det.stages
    out.view(torch.uint8).fill_(FILL_BYTE)
    det_out.view(torch.uint8).fill_(FILL_BYTE ^ 0xFF)
    launch.launch()
    det.launch()
    torch.cuda.synchronize(device)
    assert torch.equal(_bits(out), _bits(det_out))


@pytest.mark.parametrize("sum_m", E1_SUM_M)
@pytest.mark.parametrize("proj", list(BENCH_PROJ))
def test_e1_fp32_wgrad_error_within_det(proj, sum_m, record_property):
    """The fp32 weight gradient of an E1 customer row (one group of ``sum_m`` rows):
    against the fp64 reference, its rel-L2 and worst / max |ref| are each at most
    ``DET_ERROR_RATIO`` times the deterministic package's at the same inputs, and a
    repeat (output refilled, split-tail slots poisoned) gives the first launch's bits.
    On a supported GPU a missing program of either package fails instead of skipping,
    and a deterministic output that is not finite fails the comparison."""
    device = _device()
    case = _case(
        f"{proj}_e1_m{sum_m}_balanced",
        "wgrad",
        BENCH_PROJ[proj],
        [sum_m],
        out_dtype=torch.float32,
    )
    for backend in (cb, det_cb):
        assert backend.generated_program_available(
            device, op="wgrad", out_dtype=torch.float32
        ), f"no fp32 wgrad program for {backend.arch_for(device)} in {backend.__name__}"
    a, b, offs = _inputs(case, device)
    # The per-group fp64 reference, rounded to fp32 (one group).
    ref = (a.double().T @ b.double()).float().unsqueeze(0)
    det_out = _out(case, device)
    det = _prepare(det_cb, case, a, b, offs, det_out)
    det_out.fill_(float("nan"))
    det.launch()
    out = _out(case, device)
    launch = _prepare(cb, case, a, b, offs, out)
    first = None
    for fill in (0, FILL_BYTE):
        _poison_tail_sums(device)
        out.view(torch.uint8).fill_(fill)
        launch.launch()
        if first is None:
            first = out.clone()
        else:
            assert torch.equal(_bits(out), _bits(first)), f"fill 0x{fill:02x}"
    torch.cuda.synchronize(device)
    ref64 = ref.double()
    ref_max = float(ref64.abs().max())

    def errors(t):
        diff = t.double() - ref64
        return float(diff.norm() / ref64.norm()), float(diff.abs().max()) / ref_max

    rel_l2, worst = errors(first)
    det_rel_l2, det_worst = errors(det_out)
    record = dict(
        case=case.label,
        arch=cb.arch_for(device),
        sm_count=cb.sm_count(device),
        stages=list(launch.stages),
        det_stages=list(det.stages),
        plan=_plan_record(launch),
        rel_l2=rel_l2,
        det_rel_l2=det_rel_l2,
        worst=worst,
        det_worst=det_worst,
        equal_to_det=bool(torch.equal(_bits(first), _bits(det_out))),
    )
    print("E1VSDET " + json.dumps(record, sort_keys=True))
    record_property("e1_vs_det", json.dumps(record, sort_keys=True))
    assert math.isfinite(det_rel_l2) and math.isfinite(det_worst), record
    assert det_rel_l2 > 0.0 and det_worst > 0.0, record
    assert rel_l2 <= DET_ERROR_RATIO * det_rel_l2, record
    assert worst <= DET_ERROR_RATIO * det_worst, record


# The accuracy bar of the queue walk's uneven rows against the deterministic package
# at the same inputs (``test_queue_rows_error_within_det``): rel-L2 and worst / max |ref|
# against the fp64 reference at most these multiples of the deterministic package's.
QUEUE_REL_L2_RATIO = 1.1
QUEUE_WORST_RATIO = 1.5
# The tiny-scale variant: g and x drawn same-sign and scaled by 2^-TINY_LOG2, so the
# products and the fp32 partial sums of the split tail fall near and below the
# smallest normal fp32 value.
TINY_LOG2 = 62
QUEUE_OUTPUTS = ("bfloat16", "float32", "accumulate")


@pytest.mark.parametrize("scale", ["normal", "tiny"])
@pytest.mark.parametrize("output", QUEUE_OUTPUTS)
@pytest.mark.parametrize("label", list(QUEUE_SIZES))
def test_queue_rows_error_within_det(label, output, scale, record_property):
    """The down projection's weight gradient over the queue walk's uneven rows (skewed
    and one-empty, E8 / E16 / E32), as a bf16 or fp32 overwrite or a fused fp32
    accumulate into a random fp32 ``main_grad``, with normal or tiny-scale inputs:
    against the fp64 reference, its rel-L2 and worst / max |ref| are at most
    ``QUEUE_REL_L2_RATIO`` / ``QUEUE_WORST_RATIO`` times the deterministic package's at
    the same inputs, and a repeat gives the first launch's bits.  On a device whose
    ``queue_walk`` rules name the row, the launch binds the queue walk; elsewhere the
    row checks the stage it selects."""
    device = _device()
    sizes = QUEUE_SIZES[label]
    n, k = BENCH_PROJ["down"]
    out_dtype = torch.bfloat16 if output == "bfloat16" else torch.float32
    accumulate = output == "accumulate"
    case = _case(f"down_{label}_{scale}", "wgrad", (n, k), sizes, out_dtype=out_dtype)
    for backend in (cb, det_cb):
        assert backend.generated_program_available(
            device, op="wgrad", out_dtype=out_dtype
        ), (
            f"no {output} wgrad program for {backend.arch_for(device)} in {backend.__name__}"
        )
    a, b, offs = _inputs(case, device)
    if scale == "tiny":
        a = (a.float().abs() * 2.0**-TINY_LOG2).to(torch.bfloat16)
        b = (b.float().abs() * 2.0**-TINY_LOG2).to(torch.bfloat16)
    ref = torch.zeros(len(sizes), n, k, dtype=torch.float64, device=device)
    start = 0
    for e, rows in enumerate(sizes):
        if rows:
            ref[e] = (
                a[start : start + rows].double().T @ b[start : start + rows].double()
            )
        start += rows
    base = None
    if accumulate:
        gen = torch.Generator(device=device).manual_seed(zlib.crc32(label.encode()))
        scale_mg = float(ref.abs().max()) or 1.0
        base = torch.randn(
            len(sizes), n, k, generator=gen, device=device, dtype=torch.float32
        ).mul_(scale_mg)
        ref += base.double()

    def run(backend):
        out = (
            base.clone()
            if accumulate
            else torch.full(
                (len(sizes), n, k), float("nan"), dtype=out_dtype, device=device
            )
        )
        if backend is det_cb:
            launch = det_cb.cake_moe_grouped_gemm_det_train_prepare_wgrad(
                a, b, offs, out=out, out_dtype=out_dtype, accumulate=accumulate
            )
        else:
            launch = cb.cake_moe_grouped_gemm_nondet_train_prepare_wgrad(
                a, b, offs, out=out, out_dtype=out_dtype, accumulate=accumulate
            )
        launch.launch()
        return launch, out

    det, det_out = run(det_cb)
    launch, out = run(cb)
    first = out.clone()
    # A repeat: the output refilled (main_grad restored), the split-tail slots poisoned.
    _poison_tail_sums(device)
    if accumulate:
        out.copy_(base)
    else:
        out.view(torch.uint8).fill_(FILL_BYTE)
    launch.launch()
    torch.cuda.synchronize(device)
    assert torch.equal(_bits(out), _bits(first)), "repeat"
    ref_max = float(ref.abs().max())
    ref_norm = float(ref.norm())

    def errors(t):
        diff = t.double() - ref
        return float(diff.norm()) / ref_norm, float(diff.abs().max()) / ref_max

    rel_l2, worst = errors(first)
    det_rel_l2, det_worst = errors(det_out)
    arch, sms = cb.arch_for(device), cb.sm_count(device)
    queue = any(stage.startswith(cb.QUEUE_STAGE) for stage in launch.stages)
    record = dict(
        case=case.label,
        output=output,
        arch=arch,
        sm_count=sms,
        stages=list(launch.stages),
        det_stages=list(det.stages),
        queue_walk=queue,
        plan=_plan_record(launch),
        rel_l2=rel_l2,
        det_rel_l2=det_rel_l2,
        worst=worst,
        det_worst=det_worst,
        equal_to_det=bool(torch.equal(_bits(first), _bits(det_out))),
    )
    print("QUEUEVSDET " + json.dumps(record, sort_keys=True))
    record_property("queue_vs_det", json.dumps(record, sort_keys=True))
    tile_k = int(launch.plan["tile_k"])
    if tile_k == 512:
        selected, _wave = cb.select_wgrad_stage(
            launch.record_name,
            k,
            tile_k,
            arch,
            sum_m=sum(sizes),
            num_groups=len(sizes),
        )
        fused = launch.plan["accumulate"] == "fused"
        kind = "accumulate" if fused else ("float32" if accumulate else output)
        stage = cb.QUEUE_STAGE + (cb.ACC_SUFFIX if fused else "")
        want = cb.wgrad_queue_walk(
            selected, kind, arch, sms, len(sizes)
        ) and cake_jit.stage_served(launch.record_name, stage, arch)
        assert queue == want, record
    assert math.isfinite(det_rel_l2) and math.isfinite(det_worst), record
    assert det_rel_l2 > 0.0 and det_worst > 0.0, record
    assert rel_l2 <= QUEUE_REL_L2_RATIO * det_rel_l2, record
    assert worst <= QUEUE_WORST_RATIO * det_worst, record


def test_cases_exercise_the_ordered_tail(record_property):
    """The cases reach, on this device, the 512-wide k tile's ordered one-launch split
    tail (``ordered_tail`` and ``tail_splits > 1``) for both output dtypes, and the
    256-wide k tile with its two-launch tail reduce."""
    device = _device()
    arch = cb.arch_for(device)
    seen = {}
    for case in CASES:
        if case.op != "wgrad" or not sum(case.sizes):
            continue
        if not cb.generated_program_available(
            device, op="wgrad", out_dtype=case.out_dtype
        ):
            continue
        a, b, offs = _inputs(case, device)
        launch = _prepare(cb, case, a, b, offs, _out(case, device))
        plan = launch.plan
        key = (
            str(case.out_dtype).replace("torch.", ""),
            plan["tile_k"],
            "+".join(launch.stages),
            bool(plan["ordered_tail"] and plan["tail_splits"] > 1),
        )
        assert not plan["unordered_tail"], (case.label, plan)
        seen.setdefault(key, []).append(case.label)
        del a, b, offs, launch
    rows = sorted([*key, len(labels)] for key, labels in seen.items())
    record_property("coverage", json.dumps(dict(arch=arch, rows=rows)))
    print("COVERAGE " + json.dumps(dict(arch=arch, rows=rows)))
    assert cb.wgrad_ordered_tail(512, arch), arch
    for dtype in ("bfloat16", "float32"):
        assert any(key[0] == dtype and key[1] == 512 and key[3] for key in seen), (
            f"no case splits the ordered {dtype} tail on {arch}"
        )
        assert any(
            key[0] == dtype and key[1] == 256 and key[2].endswith("+tail_reduce")
            for key in seen
        ), f"no case reaches the {dtype} 256-wide k tile with its tail reduce"
