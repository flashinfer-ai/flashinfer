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

"""Benchmark the experimental VibeCUDA Kimi-K3 AttnRes backend (unfused).

Rows: all 22 original unfused Kimi-K3 serving/dispatch rows, followed by
the 70 no-delta rows of the CAKE attention-residual workload grid. The groups
retain their distinct seeds, bank capacities and weight distributions, even
when dimensions coincide. Arms: ``vibecuda`` (PDL on, the production
launch mode), ``vibecuda_nopdl``, ``cake`` (the fused-CAKE generated-program
backend; only registered on the K=4 unfused cells, reported as unsupported
elsewhere) and ``torch`` (the eager FP32 reference; informational).

The ``sglang`` arm runs the optimized standalone TMA implementation from
SGLang PR32890, commit fb207b72b02a2adfab3a4a18f5e3b5fa57bb1049.
Pass its clean checkout as ``--sglang-root``; install that checkout's runtime
dependencies in the benchmark environment. No residual addition or snapshot
write is included in this unfused comparison. Every workload is retained even
when CAKE does not register it.

Timing: ``flashinfer.testing.bench_gpu_time`` (CUPTI activity tracing, cold L2
between iterations, per-iteration GPU span; medians over ``--iters``).
CUPTI must be available; alternate timing methods are rejected. The unfused
operator mutates no input; pristine state is still restored between arms.
``--accuracy`` enforces the public CAKE test tolerances and reports the maximum
absolute output error of every arm against the FP32 reference. It also checks
that every input is unchanged and includes the original M=4/K=2 correctness
row, separately from the timed matrix and its performance means.

Usage::

    python benchmarks/bench_vibecuda_kimi_k3_attn_res.py [--rows m16_k4 ...]
        --sglang-root /path/to/sglang
        [--arms vibecuda,vibecuda_nopdl,cake,sglang,torch] [--iters 100]
        [--accuracy] [--json out.json]

Pinned-revision baselines: this script also runs from a checkout of the pinned
CAKE revision (where the VibeCUDA backend does not exist); the ``vibecuda``
arms are then rejected with a clear error and the baseline arms (``cake``,
``torch``) time the pinned code. Run it in two clean processes — the PR
checkout for the candidate (+ latest-head CAKE diagnostic) and the pinned
checkout for the configured baseline — then merge the two JSONs row by row
with ``benchmarks/merge_vibecuda_kimi_k3_attn_res.py``
(``--candidate vc_cand.json --baseline vc_pinned.json``; both processes must
use identical rows, seeds, and --iters).

In ``--accuracy`` mode the output tensor is prefilled with NaN before every
arm invocation, so a stale, empty, or partially written output tensor fails
loudly instead of passing by accident.
"""

import argparse
import importlib
from importlib.machinery import PathFinder
import json
from pathlib import Path
import statistics
import subprocess
import sys
import warnings

import torch

from flashinfer.experimental.cake_kimi_k3_attn_res import cake_backend as cb
from flashinfer.kimi_k3_attn_res import prepare_kimi_k3_attn_res
from flashinfer.testing import bench_gpu_time

try:
    from flashinfer.experimental.vibecuda_kimi_k3_attn_res.vibecuda_backend import (
        vibecuda_supported_device,
    )
except ImportError:  # baseline-only checkouts (e.g. the pinned CAKE revision)
    vibecuda_supported_device = None

H = cb.HIDDEN_SIZE
EPS = 1e-6
# Match the public CAKE repository tests, for every measured implementation.
ATOL = 8e-2
RTOL = 3e-2
SGLANG_COMMIT = "fb207b72b02a2adfab3a4a18f5e3b5fa57bb1049"


class _PinnedSGLangFinder:
    def __init__(self, package):
        self.package = package

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "sglang" and not fullname.startswith("sglang."):
            return None
        search = [str(self.package.parent)] if fullname == "sglang" else path
        if search is None or (
            fullname != "sglang"
            and any(
                not Path(entry).resolve().is_relative_to(self.package)
                for entry in search
            )
        ):
            raise ImportError(
                f"SGLang package path escaped pinned checkout: {fullname}"
            )
        spec = PathFinder.find_spec(fullname, search, target)
        if spec is None:
            # An editable install must not fill holes in the selected tree.
            # Missing optional modules follow the pinned source's own handling.
            raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
        if spec.origin is not None and not Path(spec.origin).resolve().is_relative_to(
            self.package
        ):
            raise ImportError(f"SGLang import escaped pinned checkout: {fullname}")
        return spec


def load_sglang_tma(root):
    if root is None:
        raise ValueError("the sglang arm requires --sglang-root")
    root = Path(root).resolve(strict=True)
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != SGLANG_COMMIT:
        raise ValueError(f"SGLang checkout must be at {SGLANG_COMMIT}, got {revision}")
    subprocess.run(
        ["git", "-C", str(root), "diff", "--quiet", "HEAD", "--", "python/sglang"],
        check=True,
    )
    package = root / "python/sglang"
    previous = sys.path[:]
    finder = _PinnedSGLangFinder(package)
    sys.meta_path.insert(0, finder)
    try:
        sys.path.insert(0, str(package.parent))
        module = importlib.import_module("sglang.kernels.ops.kimi_k3.attn_res")
        for name, loaded in tuple(sys.modules.items()):
            if name == "sglang" or name.startswith("sglang."):
                origin = getattr(loaded, "__file__", None)
                if origin and not Path(origin).resolve().is_relative_to(package):
                    raise RuntimeError(f"SGLang import escaped pinned checkout: {name}")
        return module.attn_res_fused_tma
    except BaseException:
        sys.meta_path.remove(finder)
        raise
    finally:
        sys.path[:] = previous


def check_accuracy(actual, expected):
    torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)
    if not bool(torch.isfinite(actual).all()):
        raise AssertionError("benchmark output contains non-finite values")


def time_arm(fn, iters):
    # The upstream helper warns before falling back to event timing. A
    # CUPTI-labelled result must fail instead of silently changing its clock.
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        times = bench_gpu_time(
            fn,
            dry_run_iters=5,
            repeat_iters=iters,
            enable_cupti=True,
            use_cuda_graph=False,
            cold_l2_cache=True,
        )
    return float(statistics.median(times))


# Timed rows of the original evaluation plus production-ladder band edges.
_ROWS = [
    (1, 1),
    (8, 2),
    (16, 4),
    (32, 4),
    (64, 8),
    (128, 8),
    (256, 8),
    (512, 8),
    (1024, 8),
    (2048, 8),
    (4096, 8),
    (8192, 8),
    (9364, 8),
    (3, 3),
    (32, 3),
    (96, 5),
    (4, 6),
    (1, 7),
    (300, 7),
    (289, 2),
    (512, 2),
    (289, 4),
]


def rows():
    return {**{f"m{M}_k{K}": (M, K) for M, K in _ROWS}, **CAKE_UNFUSED_ROWS}


# Public CAKE no-delta grid; keep its input distribution separate from the
# original serving inputs rather than replacing rows with matching dimensions.
CAKE_UNFUSED_ROWS = {
    f"cake_m{M}_k{K}": (M, K)
    for M, K in (
        [
            (M, K)
            for M in (
                1,
                2,
                4,
                8,
                16,
                32,
                64,
                128,
                256,
                512,
                1024,
                2048,
                4096,
                8192,
                16384,
            )
            for K in (0, 1, 4, 8)
        ]
        + [(M, K) for M in (1, 4096) for K in (2, 3, 5, 6, 7)]
    )
}


# Original PR32890 task correctness row; keep it outside performance means.
CORRECTNESS_ROWS = {"m4_k2": (4, 2)}


def make_inputs(M, K, seed, family="original"):
    if family == "cake_unfused":
        g = torch.Generator(device="cuda").manual_seed(31000)
        prefix = torch.empty((M, H), device="cuda", dtype=torch.bfloat16).uniform_(
            -1, 1, generator=g
        )
        blocks = torch.empty((M, 8, H), device="cuda", dtype=torch.bfloat16).uniform_(
            -1, 1, generator=g
        )
        ones = torch.empty(H, device="cuda", dtype=torch.bfloat16).normal_(
            1, 0.05, generator=g
        )
        ones.fill_(1)
        norm_weight = torch.empty_like(ones).normal_(0, 0.02, generator=g)
        output_norm_weight = torch.empty_like(ones).normal_(1, 0.05, generator=g)
        return dict(
            prefix=prefix,
            blocks=blocks,
            norm_weight=norm_weight,
            output_norm_weight=output_norm_weight,
            out=torch.full_like(prefix, float("nan")),
            ones=ones,
        )
    if family != "original":
        raise ValueError(f"unknown input family: {family}")
    g = torch.Generator(device="cuda").manual_seed(seed + M * 131 + K)
    prefix = torch.empty((M, H), device="cuda", dtype=torch.bfloat16).uniform_(
        -1, 1, generator=g
    )
    blocks = torch.empty((M, K, H), device="cuda", dtype=torch.bfloat16).uniform_(
        -1, 1, generator=g
    )
    norm_weight = torch.empty(H, device="cuda", dtype=torch.bfloat16).normal_(
        1.0, 0.05, generator=g
    )
    output_norm_weight = torch.empty(H, device="cuda", dtype=torch.bfloat16).normal_(
        1.0, 0.05, generator=g
    )
    out = torch.empty((M, H), device="cuda", dtype=torch.bfloat16)
    ones = torch.ones(H, device="cuda", dtype=torch.bfloat16)
    return dict(
        prefix=prefix,
        blocks=blocks,
        norm_weight=norm_weight,
        output_norm_weight=output_norm_weight,
        out=out,
        ones=ones,
    )


def build_arms(names, inputs, K, sglang_tma=None):
    arms = {}
    common = dict(
        prefix=inputs["prefix"],
        blocks=inputs["blocks"],
        norm_weight=inputs["norm_weight"],
        output_norm_weight=inputs["output_norm_weight"],
        out=inputs["out"],
    )
    if "vibecuda" in names:
        arms["vibecuda"] = prepare_kimi_k3_attn_res(
            **common,
            delta=None,
            qk_weight=None,
            num_blocks=K,
            eps=EPS,
            output_norm_eps=EPS,
            enable_pdl=True,
            backend="vibecuda",
        ).launch
    if "vibecuda_nopdl" in names:
        arms["vibecuda_nopdl"] = prepare_kimi_k3_attn_res(
            **common,
            delta=None,
            qk_weight=None,
            num_blocks=K,
            eps=EPS,
            output_norm_eps=EPS,
            enable_pdl=False,
            backend="vibecuda",
        ).launch
    if "cake" in names:
        try:
            arms["cake"] = prepare_kimi_k3_attn_res(
                **common,
                delta=None,
                qk_weight=inputs["ones"],
                num_blocks=K,
                eps=EPS,
                output_norm_eps=EPS,
                enable_pdl=True,
                backend="cake",
            ).launch
        except (NotImplementedError, ValueError) as e:
            arms["cake"] = e
    if "sglang" in names:
        if sglang_tma is None:
            raise ValueError("the sglang arm has no qualified implementation")

        def _sglang():
            # qk_weight is exactly one for the unfused operator, so cw is the
            # immutable norm weight itself; no input-dependent work is hoisted.
            sglang_tma(
                inputs["prefix"],
                inputs["blocks"],
                inputs["norm_weight"],
                inputs["output_norm_weight"],
                inputs["out"],
                K,
                EPS,
                write_prefix=False,
            )

        arms["sglang"] = (
            NotImplementedError("SGLang PR32890 requires num_blocks in [1, 8]")
            if K == 0
            else _sglang
        )
    if "torch" in names:

        def _torch_ref():
            cb.reference_kimi_k3_attn_res(
                inputs["prefix"],
                None,
                inputs["blocks"],
                inputs["norm_weight"],
                inputs["ones"],
                inputs["output_norm_weight"],
                inputs["out"],
                num_blocks=K,
                block_write_idx=-1,
                eps=EPS,
                output_norm_eps=EPS,
            )

        arms["torch"] = _torch_ref
    return arms


def check_arm_accuracy(fn, inputs, pristine, K, label):
    for name, value in pristine.items():
        inputs[name].copy_(value)
    inputs["out"].fill_(float("nan"))
    fn()
    got = inputs["out"].float().clone()
    if not bool(torch.isfinite(got).all()):
        raise RuntimeError(f"{label}: output not fully written")
    for name, value in pristine.items():
        if name != "out":
            torch.testing.assert_close(
                inputs[name],
                value,
                atol=0,
                rtol=0,
                msg=f"{label}: mutated read-only input {name}",
            )
    ref = {name: value.clone() for name, value in pristine.items()}
    cb.reference_kimi_k3_attn_res(
        ref["prefix"],
        None,
        ref["blocks"],
        ref["norm_weight"],
        ref["ones"],
        ref["output_norm_weight"],
        ref["out"],
        num_blocks=K,
        block_write_idx=-1,
        eps=EPS,
        output_norm_eps=EPS,
    )
    check_accuracy(got, ref["out"].float())
    return float((got - ref["out"].float()).abs().max())


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--rows", nargs="*", default=None)
    parser.add_argument("--arms", default="vibecuda,vibecuda_nopdl,cake,sglang")
    parser.add_argument("--sglang-root", type=Path)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--accuracy", action="store_true")
    parser.add_argument("--json", default=None)
    parser.add_argument("--seed", type=int, default=40910000)
    args = parser.parse_args(argv)
    device = torch.device("cuda")
    names = [a for a in args.arms.split(",") if a]
    sglang_tma = load_sglang_tma(args.sglang_root) if "sglang" in names else None
    if vibecuda_supported_device is not None:
        if not vibecuda_supported_device(device):
            raise RuntimeError(
                "the VibeCUDA Kimi-K3 AttnRes backend only supports SM100 / SM103"
            )
    elif any(n.startswith("vibecuda") for n in names):
        raise RuntimeError(
            "this checkout has no VibeCUDA Kimi-K3 AttnRes backend; "
            "run baseline arms only (e.g. --arms cake,torch)"
        )
    table = rows()
    selected = args.rows or list(table)
    results = []
    ratios = {"cake": [], "sglang": []}
    for label in selected:
        M, K = table[label]
        family = "cake_unfused" if label in CAKE_UNFUSED_ROWS else "original"
        inputs = make_inputs(M, K, args.seed, family)
        pristine = {k: v.clone() for k, v in inputs.items()}
        arms = build_arms(names, inputs, K, sglang_tma)
        row = dict(label=label, M=M, K=K, input_family=family)
        for name, fn in arms.items():
            if isinstance(fn, Exception):
                row[f"{name}_ms"] = None
                row[f"{name}_status"] = f"unsupported: {type(fn).__name__}"
                continue
            for k, v in pristine.items():
                inputs[k].copy_(v)
            row[f"{name}_ms"] = time_arm(fn, args.iters)
            if args.accuracy:
                row[f"{name}_max_abs_err"] = check_arm_accuracy(
                    fn, inputs, pristine, K, f"{label}/{name}"
                )
        for baseline in ("cake", "sglang", "torch"):
            base = row.get(f"{baseline}_ms")
            if base and row.get("vibecuda_ms"):
                ratio = base / row["vibecuda_ms"]
                row[f"speedup_vibecuda_vs_{baseline}"] = ratio
                if baseline in ratios:
                    ratios[baseline].append(ratio)
        results.append(row)
        print(json.dumps(row))
    correctness_results = []
    if args.accuracy:
        for label, (M, K) in CORRECTNESS_ROWS.items():
            inputs = make_inputs(M, K, args.seed)
            pristine = {name: value.clone() for name, value in inputs.items()}
            arms = build_arms(names, inputs, K, sglang_tma)
            row = dict(label=label, M=M, K=K, kind="correctness")
            for name, fn in arms.items():
                if isinstance(fn, Exception):
                    row[f"{name}_status"] = f"unsupported: {type(fn).__name__}"
                else:
                    row[f"{name}_max_abs_err"] = check_arm_accuracy(
                        fn, inputs, pristine, K, f"{label}/{name}"
                    )
            correctness_results.append(row)
            print(json.dumps(row))
    for baseline, values in ratios.items():
        if not values:
            continue
        geo = 1.0
        for r in values:
            geo *= r
        geo **= 1.0 / len(values)
        print(
            json.dumps(
                dict(
                    baseline=baseline,
                    arithmetic_mean=sum(values) / len(values),
                    geometric_mean=geo,
                    supported_rows=len(values),
                    requested_rows=len(selected),
                )
            )
        )
    if args.json:
        with open(args.json, "w") as f:
            json.dump(
                dict(
                    device=torch.cuda.get_device_name(device),
                    capability=list(torch.cuda.get_device_capability(device)),
                    torch=torch.__version__,
                    timing=dict(
                        method="cupti",
                        cold_l2_cache=True,
                        use_cuda_graph=False,
                        dry_run_iters=5,
                        repeat_iters=args.iters,
                    ),
                    accuracy_policy=dict(atol=ATOL, rtol=RTOL),
                    sglang_commit=SGLANG_COMMIT if "sglang" in names else None,
                    rows=results,
                    correctness_rows=correctness_results,
                ),
                f,
                indent=1,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
