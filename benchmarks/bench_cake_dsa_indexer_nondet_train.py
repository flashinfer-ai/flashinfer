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

"""Benchmark the non-deterministic Cake DSA indexer top-k against the deterministic one (SM100 / SM103 / SM107).

Times the *complete operator* of both packages on the rows of
``benchmarks/bench_cake_dsa_indexer_det_train.py`` (loaded by path: its row
definitions, input builders, CUPTI timer and deterministic arm are reused
unchanged; ``--rows`` takes its row and group names, ``iid`` / ``sink`` /
``pool`` / ``all`` / ``fp8``, the last measured in the BF16 form here):

* ``det``: ``flashinfer.experimental.cake.cake_dsa_indexer_det_train`` -- the
  scan, the split merge where the dispatch splits the key ranges, and the
  ordering pass that writes every row in ascending id order;
* ``nondet``: ``flashinfer.experimental.cake.cake_dsa_indexer_nondet_train``
  -- the same scan programs and the plain split merge, without an ordering
  pass: every row holds the same set of (id, score bits) pairs in no
  specified slot order (flashinfer-ai/flashinfer#8946, point 4).

Both arms are prepared runners over caller-owned outputs and workspace;
``det_aa`` / ``nondet_aa`` prepare the same arm a second time (the in-process
A/A band) and ``det_nosample`` / ``nondet_nosample`` run with the sampled
first threshold off (``sample_tiles_max=0``).  The arms are compared as sets
(``same_sets``: per row the valid count and the sorted (id, score bits)
pairs), never slot by slot.  Every arm's plan is recorded;
``same_scan_program`` says whether ``det`` and ``nondet`` dispatched the same
scan key and split.  A row whose arms select different sets, whose output
breaks the reference's structure checks (rows put in ascending id order) or,
with ``--verify``, that fails the reference is flagged ``FAILED``: it gets no
``det / nondet`` ratio, and the script exits non-zero after writing its JSON.

Timing: ``flashinfer.testing.bench_gpu_time`` with CUPTI activity tracing and a
cold L2 between iterations (per-iteration GPU span of the whole operator);
``--cupti-required`` turns the CUDA-event fallback into a failure.  Every row
is measured in three paired phases -- the arms in the given order (``AB``), in
reverse order (``BA``) and interleaved one call at a time (``IL``) -- and the
medians of each phase and of all samples are reported with the wall time, the
peak extra device memory and the workspace bytes.  Per row the script prints
``det`` ms, ``nondet`` ms and ``det / nondet`` (above 1: the non-deterministic
package is faster), and a summary table at the end.  ``--verify`` judges every
arm, its rows put in ascending id order first, against the FP64 / FP32
reference of ``tests/experimental/test_helpers/cake_dsa_indexer_reference.py`` (small rows
only).

Usage::

    python benchmarks/bench_cake_dsa_indexer_nondet_train.py [--rows P1 S2 ...] [--arms det,nondet]
        [--reps 20] [--strided-k] [--verify] [--cupti-required] [--json out.json]
    python benchmarks/bench_cake_dsa_indexer_nondet_train.py --rows sink --arms det,nondet,det_aa,nondet_aa
        --cupti-required --json out.json
"""

import argparse
import importlib.util
import itertools
import json
import math
import statistics
import sys
import warnings
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import flashinfer  # noqa: E402
from flashinfer.api_logging import ExperimentalWarning  # noqa: E402
from flashinfer.experimental.cake.cake_dsa_indexer_nondet_train import (  # noqa: E402
    cake_backend as nondet_backend,
)
from tests.experimental.test_helpers import cake_dsa_indexer_reference as ref  # noqa: E402


def _load_det_bench():
    """The deterministic package's benchmark module, loaded by path (it is a script, not a package module)."""
    path = Path(__file__).resolve().with_name("bench_cake_dsa_indexer_det_train.py")
    spec = importlib.util.spec_from_file_location(
        "bench_cake_dsa_indexer_det_train", path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


det_bench = _load_det_bench()
det_backend = det_bench.cake_backend


# ---------------------------------------------------------------------------
# Rows as sets
# ---------------------------------------------------------------------------


def row_pairs(out) -> torch.Tensor:
    """int64 ``[T, K]``: every slot as ``id << 32 | score bits``, each row sorted (the row's multiset of pairs)."""
    ids, scores = out
    bits = scores.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    return ((ids.to(torch.int64) << 32) | bits).sort(dim=1).values


def valid_counts(indices) -> torch.Tensor:
    return (indices >= 0).sum(dim=1)


def same_sets(a, b) -> bool:
    """Per row the same valid count and the same (id, score bits) pairs; slot order and padding positions free."""
    if tuple(a[0].shape) != tuple(b[0].shape) or tuple(a[1].shape) != tuple(b[1].shape):
        return False
    return bool(torch.equal(valid_counts(a[0]), valid_counts(b[0]))) and bool(
        torch.equal(row_pairs(a), row_pairs(b))
    )


def canonical(out):
    """``out`` with every row in ascending id order (valid slots, then the padding): the reference's slot order."""
    ids, scores = out
    key = torch.where(ids >= 0, ids.to(torch.int64), 1 << 32)
    order = key.sort(dim=1, stable=True).indices
    return ids.gather(1, order), scores.gather(1, order)


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------


def nondet_plan_record(inputs, knobs: dict) -> dict:
    """The host plan of the non-deterministic package for ``inputs`` under ``knobs``."""
    plan = nondet_backend.plan_dsa_indexer_nondet_topk(
        inputs.num_queries,
        inputs.num_keys,
        inputs.num_segments,
        top_k=inputs.top_k,
        ratio=inputs.ratio,
        device=inputs.device,
        **det_bench.row_extra(inputs),
        **knobs,
    )
    return dict(
        scan_key=plan.scan_key,
        merge_key=plan.merge_key,
        kind=plan.kind,
        block_q=int(plan.block_q),
        n_split=int(plan.n_split),
        grid_ctas=int(plan.grid_ctas),
        cand_cap=int(plan.cand_cap),
        check_period=int(plan.check_period),
        sample_tiles_max=int(plan.sample_tiles_max),
        sample_shift_permille=int(plan.sample_shift_permille),
        far_sample=int(plan.far_sample),
        sample_fit=bool(plan.sample_fit),
    )


class NondetArm(det_bench.Arm):
    """A prepared runner of the non-deterministic package over caller-owned outputs and workspace."""

    name = "nondet"
    package = "nondet"

    def __init__(self, inputs, *, knobs=None, name=None):
        super().__init__(inputs)
        self.knobs = dict(knobs or {})
        if name is not None:
            self.name = name
        dev = inputs.device
        self.workspace_bytes = (
            nondet_backend.cake_dsa_indexer_nondet_train_topk_workspace_size(
                inputs.num_queries,
                inputs.num_keys,
                inputs.num_segments,
                top_k=inputs.top_k,
                ratio=inputs.ratio,
                device=dev,
            )
        )
        self.workspace = torch.empty(
            self.workspace_bytes, dtype=torch.uint8, device=dev
        )
        self.indices = torch.empty(
            (inputs.num_queries, inputs.top_k), dtype=torch.int32, device=dev
        )
        self.scores = torch.empty(
            (inputs.num_queries, inputs.top_k), dtype=torch.float32, device=dev
        )
        self.plan = nondet_plan_record(inputs, self.knobs)
        self.runner = nondet_backend.prepare_dsa_indexer_nondet_topk(
            inputs.q,
            inputs.k,
            inputs.w,
            inputs.cu_seqlens_q,
            inputs.cu_seqlens_k,
            **inputs.kwargs(),
            **det_bench.row_extra(inputs),
            **self.knobs,
            workspace_buffer=self.workspace,
            indices=self.indices,
            scores=self.scores,
        )

    def __call__(self):
        return self.runner()

    def outputs(self, out):
        """None: the runner writes into ``indices`` / ``scores`` allocated before the measured call."""
        return ()


def _det(name, knobs=None):
    def make(inputs):
        arm = det_bench.CakeArm(inputs, knobs=knobs, name=name)
        arm.package = "det"
        return arm

    return make


def _nondet(name, knobs=None):
    def make(inputs):
        return NondetArm(inputs, knobs=knobs, name=name)

    return make


# name -> factory(inputs) -> Arm.  ``det_aa`` / ``nondet_aa`` prepare the same arm a second time (the in-process A/A
# band); the ``_nosample`` arms run the same programs with the sampled first threshold off: the same sets.
ARMS = {
    "det": _det("det"),
    "nondet": _nondet("nondet"),
    "det_aa": _det("det_aa"),
    "nondet_aa": _nondet("nondet_aa"),
    "det_nosample": _det("det_nosample", knobs=det_bench.NOSAMPLE),
    "nondet_nosample": _nondet("nondet_nosample", knobs=det_bench.NOSAMPLE),
}
DEFAULT_ARMS = "det,nondet"
TIMING_BOUNDARY = (
    "det arms: the complete operator of the deterministic package (scan, merge, ordering pass) of a prepared runner "
    "into caller-owned outputs and workspace; nondet arms: the complete operator of the non-deterministic package "
    "(scan, and the plain merge on a split call) of a prepared runner into caller-owned outputs and workspace; "
    "CUPTI GPU span of every launched kernel per iteration, cold L2"
)


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def measure_row(name, inputs, arms, args):
    row = det_bench.ROWS[name]
    reps = args.reps if args.reps is not None else row["reps"]
    prepared = {}
    for arm_name in arms:
        try:
            prepared[arm_name] = ARMS[arm_name](inputs)
        except Exception as exc:
            # a comparison without one of its arms is not the A/B the row exists for
            raise SystemExit(
                f"[{name}] {arm_name}: {type(exc).__name__}: {exc}"
            ) from exc
    result = dict(
        row=name,
        T=inputs.num_queries,
        Tkv=inputs.num_keys,
        segments=inputs.num_segments,
        top_k=inputs.top_k,
        ratio=inputs.ratio,
        reps=reps,
        arms={},
    )
    samples = {a: {"AB": [], "BA": [], "IL": []} for a in prepared}
    order = list(prepared)
    for arm_name in order:
        samples[arm_name]["AB"] = det_bench.gpu_times(
            prepared[arm_name], reps, args.warmup
        )
    for arm_name in reversed(order):
        samples[arm_name]["BA"] = det_bench.gpu_times(
            prepared[arm_name], reps, args.warmup
        )
    for _ in range(reps):
        for arm_name in order:
            samples[arm_name]["IL"].extend(
                det_bench.gpu_times(prepared[arm_name], 1, 0)
            )
    outputs = {}
    for arm_name, arm in prepared.items():
        out = arm()
        torch.cuda.synchronize()
        outputs[arm_name] = (out[0].clone(), out[1].clone())
        every = (
            samples[arm_name]["AB"] + samples[arm_name]["BA"] + samples[arm_name]["IL"]
        )
        record = dict(
            package=arm.package,
            gpu_ms={
                phase: statistics.median(v) for phase, v in samples[arm_name].items()
            },
            wall_ms=det_bench.wall_ms(arm, min(reps, 5)),
            peak_extra_bytes=det_bench.peak_extra_bytes(arm),
            workspace_bytes=arm.workspace_bytes,
            plan=arm.plan,
            knobs=arm.knobs,
            # the structure checks of the reference on the rows put in ascending id order
            structure_failures=ref.structure_failures(
                inputs, *canonical(outputs[arm_name])
            )[:3],
        )
        record["gpu_ms"]["all"] = statistics.median(every)
        result["arms"][arm_name] = record
    # set identity of every pair of arms: the contract says every arm selects the same set per row
    result["same_sets"] = {
        f"{a}|{b}": same_sets(outputs[a], outputs[b])
        for a, b in itertools.combinations(outputs, 2)
    }
    result["all_arms_same_sets"] = (
        all(result["same_sets"].values()) if result["same_sets"] else None
    )
    if args.verify:
        reference = ref.select_reference(inputs)
        for arm_name, out in outputs.items():
            verdict = ref.judge(canonical(out), reference)
            result["arms"][arm_name]["verdict"] = str(verdict).splitlines()[0]
            result["arms"][arm_name]["verdict_passed"] = bool(verdict.passed)
    # a row is correct when every pair of arms selects the same sets, no arm breaks the output structure and, with
    # --verify, every arm passes the reference; no speedup is published for a row that is not
    failures = [
        f"same_sets {key}" for key, same in result["same_sets"].items() if not same
    ]
    for arm_name, rec in result["arms"].items():
        if rec["structure_failures"]:
            failures.append(f"{arm_name}: structure {rec['structure_failures'][0]}")
        if rec.get("verdict_passed") is False:
            failures.append(f"{arm_name}: {rec['verdict']}")
    result["failures"] = failures
    if "det" in prepared and "nondet" in prepared:
        det_ms = result["arms"]["det"]["gpu_ms"]["all"]
        nondet_ms = result["arms"]["nondet"]["gpu_ms"]["all"]
        result["det_ms"] = det_ms
        result["nondet_ms"] = nondet_ms
        result["det_over_nondet"] = None if failures else det_ms / nondet_ms
        det_plan, nondet_plan = prepared["det"].plan, prepared["nondet"].plan
        result["same_scan_program"] = (
            det_plan["scan_key"] == nondet_plan["scan_key"]
            and det_plan["n_split"] == nondet_plan["n_split"]
        )
    return result


def print_row(result):
    for arm_name, rec in result["arms"].items():
        gpu = rec["gpu_ms"]
        bad = (
            f" STRUCTURE {rec['structure_failures'][0]}"
            if rec["structure_failures"]
            else ""
        )
        verdict = f" {rec['verdict']}" if "verdict" in rec else ""
        print(
            f"  [{result['row']}] {arm_name:<16} gpu {gpu['all']:10.3f} ms (AB {gpu['AB']:.3f} / BA {gpu['BA']:.3f} / IL {gpu['IL']:.3f})"
            f" wall {rec['wall_ms']:.3f} ms extra {rec['peak_extra_bytes'] / 2**20:.1f} MiB"
            f" workspace {rec['workspace_bytes'] / 2**20:.1f} MiB {rec['plan']['scan_key']} n_split {rec['plan']['n_split']}"
            f"{bad}{verdict}",
            flush=True,
        )
    if result.get("det_over_nondet") is not None:
        print(
            f"  [{result['row']}] det {result['det_ms']:.3f} ms  nondet {result['nondet_ms']:.3f} ms  "
            f"det/nondet {result['det_over_nondet']:.3f}  same scan program {result['same_scan_program']}",
            flush=True,
        )
    if result["failures"]:
        print(
            f"  [{result['row']}] FAILED (no speedup published): {result['failures']}",
            flush=True,
        )
    differing = [key for key, same in result["same_sets"].items() if not same]
    tail = f"; DIFFER {differing}" if differing else ""
    print(
        f"  [{result['row']}] same_sets: {len(result['same_sets']) - len(differing)}/{len(result['same_sets'])} pairs{tail}",
        flush=True,
    )


def print_summary(results):
    """The det / nondet table; a row with a correctness failure is flagged FAILED and has no ratio."""
    rows = [r for r in results if "det_ms" in r]
    if not rows:
        return
    print("\n| row | T | Tkv | top_k | det ms | nondet ms | det / nondet | same sets |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for r in rows:
        ratio = "FAILED" if r["failures"] else f"{r['det_over_nondet']:.3f}"
        print(
            f"| {r['row']} | {r['T']} | {r['Tkv']} | {r['top_k']} | {r['det_ms']:.3f} | {r['nondet_ms']:.3f} | "
            f"{ratio} | {r['all_arms_same_sets']} |"
        )
    ratios = [r["det_over_nondet"] for r in rows if not r["failures"]]
    if ratios:
        geomean = math.exp(statistics.fmean(math.log(v) for v in ratios))
        print(
            f"\ndet / nondet over {len(ratios)} correct rows: geometric mean {geomean:.3f}, "
            f"min {min(ratios):.3f}, max {max(ratios):.3f}",
            flush=True,
        )
    failed = [r["row"] for r in rows if r["failures"]]
    if failed:
        print(f"FAILED rows (excluded): {failed}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--rows",
        nargs="*",
        default=None,
        help="rows of the deterministic benchmark to measure (default: its default rows; the group names iid / sink / "
        "pool / all / fp8 expand)",
    )
    parser.add_argument(
        "--arms",
        default=DEFAULT_ARMS,
        help=f"comma-separated arms in AB order (default: {DEFAULT_ARMS}; known: {', '.join(ARMS)})",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=None,
        help="measured iterations per phase (default: the row's count)",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--strided-k",
        action="store_true",
        help="hand k as the [:, :128] view of a [Tkv, 704] packed tensor",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="judge every arm against the FP64 / FP32 reference (small rows)",
    )
    parser.add_argument(
        "--cupti-required",
        action="store_true",
        help="fail when flashinfer.testing.bench_gpu_time falls back from CUPTI to CUDA events",
    )
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    warnings.simplefilter("ignore", ExperimentalWarning)
    if torch.are_deterministic_algorithms_enabled():
        raise SystemExit(
            "torch.use_deterministic_algorithms(True) is set: every run of the nondet arms would alert"
        )
    det_bench.TIMER["required"] = bool(args.cupti_required)
    device = torch.device("cuda", torch.cuda.current_device())
    props = torch.cuda.get_device_properties(device)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arms {unknown}; known: {sorted(ARMS)}")
    if len(set(arms)) != len(arms):
        raise SystemExit(
            f"an arm may be named once; use det_aa / nondet_aa for an A/A pair: {arms}"
        )
    rows = det_bench.expand_rows(
        det_bench.DEFAULT_ROWS if args.rows is None else args.rows
    )
    unknown_rows = [r for r in rows if r not in det_bench.ROWS]
    if unknown_rows:
        raise SystemExit(
            f"unknown rows {unknown_rows}; known: {sorted(det_bench.ROWS)} and the groups {sorted(det_bench.ROW_GROUPS)}"
        )
    for package, backend in (("det", det_backend), ("nondet", nondet_backend)):
        used = any(a.split("_")[0] == package for a in arms)
        if used and not backend.generated_program_available(device):
            raise SystemExit(
                f"generated {package} programs not registered for this device"
            )
    print(
        f"{props.name} ({props.multi_processor_count} SMs, cc {props.major}.{props.minor}, CUDA {torch.version.cuda}), "
        f"rows {rows}, arms {arms}"
    )
    print(f"timing boundary: {TIMING_BOUNDARY}", flush=True)
    results = []
    for name in rows:
        inputs = det_bench.build_inputs(name, device, args.strided_k)
        print(
            f"[{name}] T={inputs.num_queries} Tkv={inputs.num_keys} segments={inputs.num_segments} "
            f"top_k={inputs.top_k} ratio={inputs.ratio}: {det_bench.ROWS[name]['note']}",
            flush=True,
        )
        result = measure_row(name, inputs, arms, args)
        print_row(result)
        results.append(result)
        del inputs
        torch.cuda.empty_cache()
    print_summary(results)
    print(f"timer: {det_bench.timer_record()}", flush=True)
    if args.json:
        Path(args.json).write_text(
            json.dumps(
                dict(
                    device=props.name,
                    sms=props.multi_processor_count,
                    compute_capability=f"{props.major}.{props.minor}",
                    cuda=torch.version.cuda,
                    torch=torch.__version__,
                    flashinfer=str(Path(flashinfer.__file__).resolve().parent),
                    packages=dict(
                        det=str(Path(det_backend.__file__).resolve().parent),
                        nondet=str(Path(nondet_backend.__file__).resolve().parent),
                    ),
                    results=results,
                    timer=det_bench.timer_record(),
                    arms=arms,
                    warmup=args.warmup,
                    reps=args.reps,
                    strided_k=args.strided_k,
                    timing_boundary=TIMING_BOUNDARY,
                ),
                indent=1,
            )
        )
        print(f"wrote {args.json}")
    failed = {r["row"]: r["failures"] for r in results if r["failures"]}
    if failed:
        raise SystemExit(f"correctness failures (no speedup published): {failed}")


if __name__ == "__main__":
    main()
