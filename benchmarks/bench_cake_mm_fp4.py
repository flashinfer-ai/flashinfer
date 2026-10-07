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

Benchmark of the Cake per-token NVFP4 path (``nvfp4_quantize(..., backend="cake")``
+ ``mm_fp4(..., backend="cake")``) on SM100 / SM103.

Per shape the prepared quantize + GEMM chain is timed with CUPTI under CUDA-graph
replay (cold L2; the span from the first kernel's start to the last kernel's end,
the deployment mode of these decode-shape chains and free of host launch cadence),
which is the GPU acceptance ratio of a chain row.  Diagnostic columns: the eager
span of the same callable (host-bound on small shapes: launch-arrival jitter of
several percent), every kernel on its own, the gap between the two kernels, and
the host microseconds of the eager entries and of each prepared FFI call.
Single-launch shapes (quantizer alone) use the plain CUPTI kernel time.  With ``--base <checkout>``
the same shapes run against the family of another FlashInfer checkout in the
same process (its host module is loaded by path and builds through this
checkout's JIT), alternating the arm order every round, and the report carries
the candidate / base ratio per shape and the geometric mean per kind.

    python benchmarks/bench_cake_mm_fp4.py --rows quick
    python benchmarks/bench_cake_mm_fp4.py --rows diag   # the kernel breakdown of 8 shapes
    python benchmarks/bench_cake_mm_fp4.py --base /path/to/flashinfer-at-base --rounds 5 \\
        --out ab.md --json ab.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import statistics
import sys
import time
import types
from pathlib import Path

import torch

GLOBAL_SCALE_INV = 1.0 / (448.0 * 6.0)
FAMILIES = ((7168, 2112), (8192, 8192), (7168, 18432), (8192, 28672), (16384, 7168))
QUICK_M = (1, 9, 16, 32, 128, 130, 512, 2048)
FULL_M = (1, 8, 9, 12, 16, 17, 32, 128, 130, 257, 512, 2048, 8192)
# (K, N) outside the measured families.
OFF_MATRIX = ((4096, 3200), (16, 300))
QUANT_ONLY = ((7168, 2688, 4096), (1, 130, 2048))
DIAG = (
    ((16384, 7168), (1, 130, 512)),
    ((8192, 28672), (32,)),
    ((7168, 2112), (128,)),
    ((8192, 8192), (128,)),
)


def rows(kind: str) -> list[dict]:
    if kind == "diag":
        out = [
            dict(kind="chain", k=k, n=n, m=m, fold=True)
            for (k, n), ms in DIAG
            for m in ms
        ]
        return out + [
            dict(kind="quant", k=7168, n=0, m=m, fold=False) for m in (1, 130)
        ]
    if kind == "quant-large":
        # the quantizer alone on large row sets (3.5-55 waves of one-row CTAs): every row width the
        # shipped checkout serves plus the two off-matrix widths (candidate-only there)
        return [
            dict(kind="quant", k=k, n=0, m=m, fold=False)
            for k in (2688, 4096, 7168, 8192, 16384, 18432, 28672)
            for m in (512, 2048, 8192)
        ]
    ms = QUICK_M if kind == "quick" else FULL_M
    out = [
        dict(kind="chain", k=k, n=n, m=m, fold=True) for k, n in FAMILIES for m in ms
    ]
    (k, n), ms_off = OFF_MATRIX
    out += [dict(kind="chain", k=k, n=n, m=m, fold=True) for m in ms_off]
    ks, ms_q = QUANT_ONLY
    out += [dict(kind="quant", k=k, n=0, m=m, fold=False) for k in ks for m in ms_q]
    return out


def load_base_family(base_root: Path) -> types.ModuleType:
    """Load ``flashinfer/experimental/cake_nvfp4_per_token`` of another checkout.

    The package is registered under a private name whose ``utils`` and ``jit``
    sub-modules are this checkout's (same API), so the base host module's relative
    imports resolve and its programs build through the same JIT infrastructure;
    its generated sources and registry come from ``base_root``.
    """
    import flashinfer.jit
    import flashinfer.jit.core
    import flashinfer.jit.env
    import flashinfer.utils

    root = "cake_ab_base_flashinfer"
    pkg = types.ModuleType(root)
    pkg.__path__ = []
    sys.modules[root] = pkg
    sys.modules[f"{root}.utils"] = flashinfer.utils
    sys.modules[f"{root}.jit"] = flashinfer.jit
    sys.modules[f"{root}.jit.core"] = flashinfer.jit.core
    sys.modules[f"{root}.jit.env"] = flashinfer.jit.env
    exp = types.ModuleType(f"{root}.experimental")
    exp.__path__ = []
    sys.modules[exp.__name__] = exp
    fam = types.ModuleType(f"{root}.experimental.cake_nvfp4_per_token")
    fam.__path__ = [str(base_root / "flashinfer/experimental/cake_nvfp4_per_token")]
    sys.modules[fam.__name__] = fam
    return importlib.import_module(f"{fam.__name__}.cake_backend")


def weights(n: int, k: int, device: torch.device, seed: int):
    from flashinfer import SfLayout, nvfp4_quantize

    g = torch.Generator(device=device).manual_seed(seed)
    w = torch.randn(n, k, device=device, dtype=torch.bfloat16, generator=g)
    w_global_sf = (448.0 * 6.0) / w.float().abs().max()
    w_fp4, w_sf = nvfp4_quantize(
        w, w_global_sf, sfLayout=SfLayout.layout_128x4, do_shuffle=False
    )
    return w_fp4.view(torch.uint8), w_sf, (1.0 / w_global_sf).reshape(1).float()


class Arm:
    """One checkout's family host bound to one shape."""

    def __init__(self, cb, row: dict, x: torch.Tensor, gs_inv: torch.Tensor, w):
        self.cb = cb
        self.row = row
        m, k = x.shape
        device = x.device
        self.x, self.gs_inv = x, gs_inv
        self.ws = cb.allocate_nvfp4_per_token_quantize_outputs(m, k, device)
        if row["kind"] == "chain":
            w_fp4, w_sf, w_scale = w
            self.w_fp4, self.w_sf = w_fp4, w_sf
            self.w_scale = w_scale if row["fold"] else None
            self.out = torch.empty((m, row["n"]), dtype=torch.bfloat16, device=device)
            self.runner = cb.prepare_nvfp4_per_token_chain(
                x, gs_inv, w_fp4, w_sf, self.out, self.ws, out_scale=self.w_scale
            )
            # One kernel per launch (split-K is a cluster split inside the GEMM).
            self.kernels = {"quant": self.runner.quantize, "gemm": self.runner.gemm}
            # The two launches as one callable: its CUPTI span (first kernel start
            # to last kernel end) also holds the gap between the two kernels.
            self.span = self.runner
        else:
            self.runner = cb.prepare_nvfp4_per_token_quantize(x, gs_inv, self.ws)
            self.kernels = {"quant": self.runner}

    def eager(self) -> None:
        cb = self.cb
        if self.row["kind"] == "chain":
            a_fp4, a_sf, alpha = cb.nvfp4_quantize_per_token(
                self.x, GLOBAL_SCALE_INV, out_scale=self.w_scale
            )
            cb.mm_fp4_per_token(a_fp4, self.w_fp4.T, a_sf, self.w_sf.T, alpha, self.out)
        else:
            cb.nvfp4_quantize_per_token(self.x, GLOBAL_SCALE_INV)

    def result(self) -> torch.Tensor:
        return self.out if self.row["kind"] == "chain" else self.ws.fp4


def gpu_ms(fn, iters: int, graph: bool = False) -> float:
    from flashinfer.testing import bench_gpu_time_with_cupti

    times = bench_gpu_time_with_cupti(
        fn,
        dry_run_iters=10,
        repeat_iters=iters,
        cold_l2_cache=True,
        use_cuda_graph=graph,
    )
    return float(statistics.median(times))


def host_us(fn, n: int = 200) -> float:
    fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    t1 = time.perf_counter()
    torch.cuda.synchronize()
    return (t1 - t0) / n * 1e6


def host_dist(fn, n: int = 400) -> dict[str, float]:
    """Median and p90 host microseconds per call of ``fn`` (launches stay queued)."""
    fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e6)
    torch.cuda.synchronize()
    times.sort()
    return {"p50": times[n // 2], "p90": times[(n * 9) // 10]}


def geomean(values: list[float]) -> float:
    return (
        math.exp(sum(math.log(v) for v in values) / len(values))
        if values
        else float("nan")
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", type=Path, help="FlashInfer checkout of the base arm")
    ap.add_argument(
        "--rows", choices=("quick", "full", "diag", "quant-large"), default="quick"
    )
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--iters", type=int, default=40, help="CUPTI samples per round")
    ap.add_argument("--out", type=Path, help="markdown report")
    ap.add_argument("--json", type=Path, help="raw per-round samples")
    ap.add_argument("--min-row", type=float, default=0.98)
    ap.add_argument("--min-geomean", type=float, default=1.00)
    args = ap.parse_args()

    import flashinfer
    from flashinfer.experimental.cake_nvfp4_per_token import cake_backend as cand

    device = torch.device("cuda", 0)
    props = torch.cuda.get_device_properties(device)
    arms = {"cand": cand}
    if args.base is not None:
        arms["base"] = load_base_family(args.base.resolve())
    print(
        f"device {props.name} ({props.multi_processor_count} SMs) | candidate "
        f"{Path(flashinfer.__file__).parent} | base {args.base or '-'}"
    )
    results = []
    for index, row in enumerate(rows(args.rows)):
        m, k, n = row["m"], row["k"], row["n"]
        g = torch.Generator(device=device).manual_seed(900 + index)
        x = torch.randn(m, k, device=device, dtype=torch.bfloat16, generator=g) / (
            448 * 6
        )
        gs_inv = torch.tensor([GLOBAL_SCALE_INV], dtype=torch.float32, device=device)
        w = weights(n, k, device, 1000 + index) if row["kind"] == "chain" else None
        label = f"{row['kind']}_k{k}" + (f"_n{n}" if n else "") + f"_m{m}"
        bound = {}
        unsupported = {}
        for name, cb in arms.items():
            try:
                bound[name] = Arm(cb, row, x, gs_inv, w)
                bound[name].runner()
                torch.cuda.synchronize()
            except NotImplementedError as error:
                # The checkout's own dispatcher names a program it does not register
                # for this shape (e.g. token counts between its tiles or a row length
                # outside its matrix): that arm refuses the shape.  The base arm plans
                # with the base checkout's dispatcher, so a base refusal means the
                # shipped code does not serve the shape and the row is candidate-only.
                unsupported[name] = str(error).splitlines()[0][:120]
        if "cand" not in bound:
            results.append(dict(row, label=label, cand_unsupported=unsupported["cand"]))
            print(f"{label:<28} cand n/a ({unsupported['cand']})", flush=True)
            continue
        if "base" in bound:
            same = torch.equal(bound["cand"].result(), bound["base"].result())
        else:
            same = None
        samples = {
            name: {part: [] for part in arm.kernels} for name, arm in bound.items()
        }
        span_samples = {name: [] for name in bound}
        graph_samples = {name: [] for name in bound}
        host = {name: [] for name in bound}
        prepared_host = {name: [] for name in bound}
        part_host = {
            name: {part: [] for part in arm.kernels} for name, arm in bound.items()
        }
        for round_index in range(args.rounds):
            order = list(bound) if round_index % 2 == 0 else list(reversed(bound))
            for name in order:
                for part, fn in bound[name].kernels.items():
                    samples[name][part].append(gpu_ms(fn, args.iters))
                if row["kind"] == "chain":
                    span_samples[name].append(gpu_ms(bound[name].span, args.iters))
                    graph_samples[name].append(
                        gpu_ms(bound[name].span, args.iters, graph=True)
                    )
                host[name].append(host_us(bound[name].eager))
                prepared_host[name].append(host_us(bound[name].runner))
                for part, fn in bound[name].kernels.items():
                    part_host[name][part].append(host_dist(fn))
        kernel_ms = {
            name: {part: statistics.median(v) for part, v in parts.items()}
            for name, parts in samples.items()
        }
        record = dict(
            row,
            label=label,
            kernel_ms=kernel_ms,
            gpu_ms={name: sum(parts.values()) for name, parts in kernel_ms.items()},
            host_us={name: statistics.median(v) for name, v in host.items()},
            samples=samples,
            host_samples=host,
            prepared_host_us={
                name: statistics.median(v) for name, v in prepared_host.items()
            },
            # Host microseconds of each prepared FFI call: median of the per-round
            # medians and of the per-round p90s.
            part_host_us={
                name: {
                    part: {
                        q: statistics.median(d[q] for d in dists)
                        for q in ("p50", "p90")
                    }
                    for part, dists in parts.items()
                }
                for name, parts in part_host.items()
            },
            bitwise=same,
            base_unsupported=unsupported.get("base"),
        )
        if row["kind"] == "chain":
            record["span_ms"] = {
                name: statistics.median(v) for name, v in span_samples.items()
            }
            record["span_samples"] = span_samples
            record["graph_ms"] = {
                name: statistics.median(v) for name, v in graph_samples.items()
            }
            record["graph_samples"] = graph_samples
            # GPU-visible idle time between the two kernels of the chain.
            record["gap_us"] = {
                name: (record["span_ms"][name] - record["gpu_ms"][name]) * 1e3
                for name in bound
            }
        if "base" in bound:
            # Acceptance ratio: graph-replay span for chain rows, kernel time otherwise.
            gate = record["graph_ms"] if row["kind"] == "chain" else record["gpu_ms"]
            record["ratio"] = gate["base"] / gate["cand"]
            record["kernel_sum_ratio"] = (
                record["gpu_ms"]["base"] / record["gpu_ms"]["cand"]
            )
            record["kernel_ratio"] = {
                part: kernel_ms["base"][part] / kernel_ms["cand"][part]
                for part in kernel_ms["cand"]
            }
            # Rounds in which the candidate kernel was faster / equal / slower.
            record["tally"] = {
                part: [
                    sum(
                        c < b
                        for c, b in zip(
                            samples["cand"][part], samples["base"][part], strict=True
                        )
                    ),
                    sum(
                        c == b
                        for c, b in zip(
                            samples["cand"][part], samples["base"][part], strict=True
                        )
                    ),
                    sum(
                        c > b
                        for c, b in zip(
                            samples["cand"][part], samples["base"][part], strict=True
                        )
                    ),
                ]
                for part in kernel_ms["cand"]
            }
            record["host_ratio"] = record["host_us"]["base"] / record["host_us"]["cand"]
            if row["kind"] == "chain":
                record["span_ratio"] = (
                    record["span_ms"]["base"] / record["span_ms"]["cand"]
                )
        results.append(record)
        parts_cand = " ".join(
            f"{part} {ms * 1e3:.2f}" for part, ms in kernel_ms["cand"].items()
        )
        if row["kind"] == "chain":
            line = f"{label:<28} cand graph {record['graph_ms']['cand'] * 1e3:8.2f} us"
            line += f" kernels {record['gpu_ms']['cand'] * 1e3:8.2f} us ({parts_cand})"
        else:
            line = f"{label:<28} cand {record['gpu_ms']['cand'] * 1e3:8.2f} us ({parts_cand})"
        if "base" in unsupported:
            line += f"  base refuses, candidate-only ({unsupported['base']})"
        elif "base" in bound:
            parts_base = " ".join(
                f"{part} {ms * 1e3:.2f}" for part, ms in kernel_ms["base"].items()
            )
            parts_ratio = " ".join(
                f"{part} {ratio:.4f}" for part, ratio in record["kernel_ratio"].items()
            )
            tally = " ".join(
                f"{part} {w}/{t}/{l}" for part, (w, t, l) in record["tally"].items()
            )
            if row["kind"] == "chain":
                line += f"  base graph {record['graph_ms']['base'] * 1e3:8.2f} us"
            line += (
                f"  base kernels {record['gpu_ms']['base'] * 1e3:8.2f} us ({parts_base})"
                f"  ratio {record['ratio']:.4f} (kernels {record['kernel_sum_ratio']:.4f}: "
                f"{parts_ratio})  tally<,=,> {tally}"
                f"  host {record['host_us']['cand']:.1f}/{record['host_us']['base']:.1f} us"
                f"  prepared {record['prepared_host_us']['cand']:.1f}/"
                f"{record['prepared_host_us']['base']:.1f} us"
            )
            calls = " ".join(
                f"{part} {record['part_host_us']['cand'][part]['p50']:.1f}"
                f"({record['part_host_us']['cand'][part]['p90']:.1f})/"
                f"{record['part_host_us']['base'][part]['p50']:.1f}"
                f"({record['part_host_us']['base'][part]['p90']:.1f})"
                for part in record["part_host_us"]["cand"]
            )
            line += f"  call p50(p90) us {calls}"
            if row["kind"] == "chain":
                line += (
                    f"  eager span {record['span_ms']['cand'] * 1e3:.2f}/"
                    f"{record['span_ms']['base'] * 1e3:.2f} us (ratio {record['span_ratio']:.4f})"
                    f"  gap {record['gap_us']['cand']:.2f}/{record['gap_us']['base']:.2f} us"
                )
            line += f"  bitwise {same}"
        else:
            line += f"  host {record['host_us']['cand']:.1f} us"
        print(line, flush=True)

    summary = {
        "device": props.name,
        "sm_count": props.multi_processor_count,
        "rows": results,
    }
    passed = True
    summary["cand_unsupported"] = [
        r["label"] for r in results if r.get("cand_unsupported")
    ]
    if "base" in arms:
        paired = [r for r in results if "ratio" in r]
        summary["base_unsupported"] = [
            r["label"] for r in results if r.get("base_unsupported")
        ]
        for kind in ("chain", "quant"):
            summary[f"geomean_{kind}"] = geomean(
                [r["ratio"] for r in paired if r["kind"] == kind]
            )
        all_ratios = [r["ratio"] for r in paired]
        summary["geomean"] = geomean(all_ratios)
        summary["min_ratio"] = min(all_ratios)
        summary["host_geomean"] = geomean([r["host_ratio"] for r in paired])
        summary["span_geomean"] = geomean(
            [r["span_ratio"] for r in paired if "span_ratio" in r]
        )
        for part in ("quant", "gemm"):
            summary[f"geomean_kernel_{part}"] = geomean(
                [r["kernel_ratio"][part] for r in paired if part in r["kernel_ratio"]]
            )
        summary["worst_rows"] = [
            r["label"] for r in sorted(paired, key=lambda r: r["ratio"])[:5]
        ]
        passed = (
            summary["min_ratio"] >= args.min_row
            and summary["geomean"] >= args.min_geomean
        )
        summary["passed"] = passed
        print(
            f"geomean {summary['geomean']:.4f} (chain {summary['geomean_chain']:.4f}, "
            f"quant {summary['geomean_quant']:.4f}) min {summary['min_ratio']:.4f} "
            f"kernels quant {summary['geomean_kernel_quant']:.4f} gemm "
            f"{summary['geomean_kernel_gemm']:.4f}; chain span geomean "
            f"{summary['span_geomean']:.4f}; host geomean "
            f"{summary['host_geomean']:.3f}; base refuses (candidate-only) "
            f"{len(summary['base_unsupported'])} rows, cand unsupported "
            f"{len(summary['cand_unsupported'])} rows -> {'PASS' if passed else 'FAIL'}"
        )
    if args.json:
        args.json.write_text(json.dumps(summary, indent=1))
    if args.out:
        lines = [
            f"# Cake per-token NVFP4 A/B on {props.name} ({props.multi_processor_count} SMs)",
            "",
        ]
        if "base" in arms:
            lines += [
                "| shape | gate base us | gate cand us | ratio (gate) | kernels base us (quant + gemm) "
                "| kernels cand us (quant + gemm) | quant ratio | gemm ratio | eager span ratio "
                "| host base us | host cand us | bitwise |",
                "|---|---|---|---|---|---|---|---|---|---|---|---|",
                "",
                "Gate: chain rows = CUPTI span of the chain under CUDA-graph replay; "
                "quantizer rows = CUPTI kernel time. Other columns are diagnostic.",
                "",
            ]

            def parts(r, name):
                return " + ".join(
                    f"{ms * 1e3:.2f}" for ms in r["kernel_ms"][name].values()
                )

            for r in results:
                if r.get("cand_unsupported"):
                    lines.append(
                        f"| {r['label']} | - | n/a (cand: {r['cand_unsupported']}) | - | - | - | - | - | - | - | - | - |"
                    )
                    continue
                gate_key = "graph_ms" if r["kind"] == "chain" else "gpu_ms"
                if "ratio" not in r:
                    lines.append(
                        f"| {r['label']} | shipped refuses ({r['base_unsupported']}) | "
                        f"{r[gate_key]['cand'] * 1e3:.2f} | - | - | "
                        f"{r['gpu_ms']['cand'] * 1e3:.2f} ({parts(r, 'cand')}) | - | - | - | - | "
                        f"{r['host_us']['cand']:.1f} | - |"
                    )
                else:
                    kr = r["kernel_ratio"]
                    lines.append(
                        f"| {r['label']} | {r[gate_key]['base'] * 1e3:.2f} | {r[gate_key]['cand'] * 1e3:.2f} | "
                        f"{r['ratio']:.4f} | {r['gpu_ms']['base'] * 1e3:.2f} ({parts(r, 'base')}) | "
                        f"{r['gpu_ms']['cand'] * 1e3:.2f} ({parts(r, 'cand')}) | "
                        f"{kr.get('quant', float('nan')):.4f} | {kr.get('gemm', float('nan')):.4f} | "
                        f"{r.get('span_ratio', float('nan')):.4f} | "
                        f"{r['host_us']['base']:.1f} | {r['host_us']['cand']:.1f} | {r['bitwise']} |"
                    )
            lines += [
                "",
                f"geomean {summary['geomean']:.4f} (chain {summary['geomean_chain']:.4f}, quant "
                f"{summary['geomean_quant']:.4f}; kernels quant {summary['geomean_kernel_quant']:.4f}, "
                f"gemm {summary['geomean_kernel_gemm']:.4f}); min ratio {summary['min_ratio']:.4f}; "
                f"host-time geomean {summary['host_geomean']:.3f}; {'PASS' if passed else 'FAIL'} "
                f"(row >= {args.min_row}, geomean >= {args.min_geomean})",
            ]
        else:
            lines += ["| shape | us (quant + gemm) | host us |", "|---|---|---|"]
            lines += [
                f"| {r['label']} | {r['gpu_ms']['cand'] * 1e3:.2f} ({' + '.join(f'{ms * 1e3:.2f}' for ms in r['kernel_ms']['cand'].values())}) | {r['host_us']['cand']:.1f} |"
                if not r.get("cand_unsupported")
                else f"| {r['label']} | n/a ({r['cand_unsupported']}) | - |"
                for r in results
            ]
        args.out.write_text("\n".join(lines) + "\n")
    return 0 if passed else 3


if __name__ == "__main__":
    sys.exit(main())
