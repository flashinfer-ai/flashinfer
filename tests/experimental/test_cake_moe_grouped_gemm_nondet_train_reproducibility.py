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
* row-padded ``g`` and ``x`` on the shape of the 256-wide k tile's ``main_wave``
  weight gradient give the deterministic package's bits.

On the CPU, the registry mirrors the deterministic package's routes and stages
(but for the data gradient's own sm_100a ``main_wide_ef`` build, held to the
deterministic wide tile's launch shape and argument plan).
"""

import dataclasses
import json
import math
import zlib

import pytest
import torch

pytest.importorskip("tvm_ffi")

import flashinfer.cake_moe_grouped_gemm_nondet_train as facade
from tests.test_helpers.cake_package import cake_package

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

# This package's own builds beside the copies: (route, stage) -> the deterministic
# package's stage of the same kernel and launch.  The data gradient's L2 evict-first
# twin of the wide tile on the architectures this package's
# ``dgrad_evict_first_archs`` adds (sm_100a): the deterministic wide
# domain-scheduled kernel with evict-first G loads and dX stores.
OWN_STAGES = {("cake_moe_grouped_gemm_dgrad", "main_wide_ef"): "main_wide"}
# Data-gradient cases of the evict-first rule (at least ``dgrad_evict_first_min_rows``
# rows per group on average, ``K >= dgrad_evict_first_min_k``): (label, sizes, N, K),
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


CASES = bench_cases() + edge_cases()
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


def _own_stage(case, name, arch):
    """The stage of ``OWN_STAGES`` this package's host takes for a fwd / dgrad
    ``case`` on ``arch`` where the deterministic package takes its twin, else None:
    the data gradient's ``main_wide_ef`` where this package's ``dgrad_evict_first``
    holds on an architecture the deterministic package's constants do not list."""
    sum_m, groups = sum(case.sizes), len(case.sizes)
    if (
        case.op == "dgrad"
        and cb.dgrad_evict_first(case.k, arch, sum_m, groups, HOST_PLAN_CONSTANTS)
        and arch not in det_jit.HOST_PLAN_CONSTANTS["dgrad_evict_first_archs"]
        and cake_jit.stage_served(name, "main_wide_ef", arch)
        and cb.fwd_dgrad_tile_n("dgrad", case.n, case.k, arch, HOST_PLAN_CONSTANTS)
        == int(HOST_PLAN_CONSTANTS["wide_block_n"])
    ):
        return "main_wide_ef"
    return None


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
    route's k tile).  The accumulate stages of the 512-wide k tile (``<stage>_acc``,
    which the deterministic package does not have) are left out of the mirror; their
    twins are mirrored, and ``test_cake_moe_grouped_gemm_nondet_train_acc.py`` checks them against
    their twins.  So is a stage of ``OWN_STAGES`` on an architecture where the
    deterministic route does not serve it (the data gradient's sm_100a
    ``main_wide_ef``, this package's own evict-first build): it is held to the
    deterministic stage of the same kernel (``main_wide``), with that stage's launch
    shape, FFI entry and argument plan, under a program name of its own, and only on
    the architectures this package's ``dgrad_evict_first_archs`` adds.  (A CPU check;
    it lives here because the backend suite imports no package but its own.)"""
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
            }
            stages = {
                s["name"]
                for s in route["stages"]
                if arch in programs[s["program"]]["arches"]
                and not s["name"].endswith(cb.ACC_SUFFIX)
            }
            own = {
                stage
                for stage in stages
                if (name, stage) in OWN_STAGES and stage not in det_stages
            }
            stages -= own
            for stage in own:
                twin = OWN_STAGES[(name, stage)]
                assert stage == "main_wide_ef" and route["op"] == "dgrad", (name, stage)
                assert arch in HOST_PLAN_CONSTANTS["dgrad_evict_first_archs"], arch
                det_ef_archs = det_jit.HOST_PLAN_CONSTANTS["dgrad_evict_first_archs"]
                assert arch not in det_ef_archs, arch
                det_binding = det_jit.stage_binding(det_name, twin, arch)
                binding = cake_jit.stage_binding(name, stage, arch)
                assert binding["specializations"] == det_binding["specializations"]
                det_program = det_programs[det_binding["program"]]
                program = programs[binding["program"]]
                assert program["launch"] == det_program["launch"], (name, stage, arch)
                assert program["ffi_entry"] == det_program["ffi_entry"]
                row_bases = {("buffer", rows) for rows in cb.ROW_BASES}
                plan = [
                    tuple(i) for i in program["arg_plan"] if tuple(i) not in row_bases
                ]
                det_plan = [tuple(i) for i in det_program["arg_plan"]]
                assert plan == det_plan, (name, stage)
                # Not a copy: a program of its own, beside the copy of the twin.
                copy = cake_jit.stage_binding(name, twin, arch)["program"]
                assert binding["program"] != copy
                assert binding["program"].startswith("cake_moe_grouped_gemm_")
                assert (
                    binding["program"].replace(
                        "cake_moe_grouped_gemm", "cake_moe_grouped_det_gemm"
                    )
                    not in det_programs
                )
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
                expected = (
                    _one_launch_plan(det_program["arg_plan"])
                    if one_launch
                    else [tuple(i) for i in det_program["arg_plan"]]
                )
                row_bases = {("buffer", rows) for rows in cb.ROW_BASES}
                plan = [tuple(i) for i in program["arg_plan"]]
                assert [i for i in plan if i not in row_bases] == expected, (
                    name,
                    stage,
                    arch,
                )
                assert program["library"].startswith("cake_moe_grouped_gemm_")
                assert binding["program"].startswith("cake_moe_grouped_gemm_")


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
        # fwd / dgrad: the deterministic programs' bits.  The stages are the
        # deterministic package's, but where this package's host takes its own
        # evict-first build of the data gradient's wide tile (``OWN_STAGES``).
        own = _own_stage(case, launch.record_name, cb.arch_for(device))
        twin = OWN_STAGES.get((launch.record_name, own))
        expected = [own if stage == twin else stage for stage in det.stages]
        assert list(launch.stages) == expected, (launch.stages, det.stages)
        record.update(own_stage=own)
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
    record.update(
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
    if not record["exercised"]:
        # No split one-launch tail: the deterministic programs' order, the same bits.
        assert differ_from_det == 0, record
    # A split tail's bits may differ from the deterministic package's (another tail
    # plan and group order), within its budget.
    assert worst_vs_det <= 1.0, record


@pytest.mark.parametrize("label,sizes,n,k", EF_CASES, ids=[c[0] for c in EF_CASES])
def test_dgrad_evict_first_gives_the_deterministic_bits(
    label, sizes, n, k, monkeypatch
):
    """Where this package's ``dgrad_evict_first`` holds, the data gradient launches the
    L2 evict-first twin of its wide tile (``main_wide_ef``; on sm_100a this package's
    own build, where the deterministic package launches ``main_wide``).  The cache
    policy changes no arithmetic: over a one-row group, an empty group and group sizes
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
    assert sum_m // groups >= int(c["dgrad_evict_first_min_rows"]), label
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
    if ef and arch not in det_jit.HOST_PLAN_CONSTANTS["dgrad_evict_first_archs"]:
        assert det.stages[0] == "main_wide", det.stages
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
    monkeypatch.setitem(c, "dgrad_evict_first_min_rows", 1 << 30)
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


@pytest.mark.parametrize("label", K512_LABELS)
def test_unsplit_k512_builds_give_the_deterministic_bits(label, monkeypatch):
    """With the tail split switched off in both packages (``max_splits`` 1), a weight
    gradient on the 512-wide k tile runs every tile whole in both, and the ordered
    one-launch tail build gives the deterministic build's bits for every form the case
    selects."""
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
    assert launch.stages == det.stages
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
