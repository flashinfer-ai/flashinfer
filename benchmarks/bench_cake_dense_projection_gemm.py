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

"""Benchmark the experimental Cake dense projection GEMMs (SM100 / SM103 / SM107).

Times the GLM-5.2 training GEMMs of ``flashinfer.experimental.dense_projection_gemm``
paired against ``torch.matmul`` on the same strided views:

* the 11 projection rows (``o_proj``, ``q_a``, ``q_b``, ``kv_a``, ``shared_gate_up``,
  ``shared_down``, ``dense_gate_up``, ``dense_down``, ``indexer_q``, ``indexer_k``,
  ``indexer_hw``) x {forward, input gradient, weight gradient} x T in {16231, 16172}
  (BF16 outputs; ``--f32-grads`` adds the FP32-output gradient rows);
* the two batched MLA head-projection rows (``qabs``: 64 heads, 192 -> 512 inside a
  256-wide slot; ``vproj``: 64 heads, 512 -> 256, out_in weight) x the three operations;
* the FP32 router rows (``X[T, 6144] @ W[256, 6144].T`` and its gradients) through the
  split-BF16x3 emulation, paired against cuBLAS SGEMM (TF32 disabled).

Timing: ``flashinfer.testing.bench_gpu_time`` with CUPTI activity tracing and a cold L2
between iterations (per-iteration GPU span); medians over ``--steps`` iterations, the
Cake arm re-timed after the baseline so both figures come from the same clock regime.
``--accuracy`` adds the error against an FP64 reference of the same views.

Usage::

    python benchmarks/bench_cake_dense_projection_gemm.py [--rows kv_a_fwd_bf16_t16231 ...]
        [--ops fwd,dgrad,wgrad] [--T 16231,16172] [--f32-grads] [--no-mla] [--no-router]
        [--steps 20] [--accuracy] [--json out.json]
"""

import argparse
import json
import statistics
import sys
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flashinfer.experimental.dense_projection_gemm import cake_backend  # noqa: E402
from flashinfer.testing import bench_gpu_time  # noqa: E402

# GLM-5.2 projection rows: label -> (K = in features, N = out features).
PROJECTION_ROWS = {
    "o_proj": (16384, 6144),
    "q_a": (6144, 2048),
    "q_b": (2048, 16384),
    "kv_a": (6144, 576),
    "shared_gate_up": (6144, 2048),
    "shared_down": (2048, 6144),
    "dense_gate_up": (6144, 12288),
    "dense_down": (12288, 6144),
    "indexer_q": (2048, 4096),
    "indexer_k": (6144, 128),
    "indexer_hw": (6144, 32),
}
MLA_ROWS = {
    "qabs": dict(
        H=64, D_in=192, D_out=512, act_stride_head=256, weight_layout="in_out"
    ),
    "vproj": dict(
        H=64, D_in=512, D_out=256, act_stride_head=512, weight_layout="out_in"
    ),
}
ROUTER_K, ROUTER_N = 6144, 256
OPS = ("fwd", "dgrad", "wgrad")
CANONICAL_T = (16231, 16172)
SEED = 758


def row_table(T_values, ops, f32_grads, mla, router):
    """label -> row spec, in the order the rows are reported."""
    rows = {}
    for T in T_values:
        for label in PROJECTION_ROWS:
            for op in ops:
                dtypes = ("bf16",) if op == "fwd" or not f32_grads else ("bf16", "f32")
                for dt in dtypes:
                    rows[f"{label}_{op}_{dt}_t{T}"] = dict(
                        family="proj", row=label, op=op, out_dtype=dt, T=T
                    )
        if mla:
            for label in MLA_ROWS:
                for op in ops:
                    rows[f"mla_{label}_{op}_bf16_t{T}"] = dict(
                        family="mla", row=label, op=op, out_dtype="bf16", T=T
                    )
        if router:
            for op in ops:
                rows[f"router_{op}_t{T}"] = dict(
                    family="router", row="router", op=op, out_dtype="f32", T=T
                )
    return rows


# ---------------------------------------------------------------------------
# Inputs (the CAKE-758 eval harness distributions)
# ---------------------------------------------------------------------------


def make_inputs(spec, seed, device):
    torch.manual_seed(seed)
    g = torch.Generator(device=device).manual_seed(seed)
    fam, op, T = spec["family"], spec["op"], int(spec["T"])
    out_dt = {"bf16": torch.bfloat16, "f32": torch.float32}[spec["out_dtype"]]

    def rnd(*shape, scale=1.0):
        return (torch.randn(*shape, device=device, generator=g) * scale).to(
            torch.bfloat16
        )

    if fam == "proj":
        K, N = PROJECTION_ROWS[spec["row"]]
        X, W, G = rnd(T, K), rnd(N, K, scale=K**-0.5), rnd(T, N)
        if op == "fwd":
            A, B, out = X, W.t(), torch.empty(T, N, device=device, dtype=out_dt)
        elif op == "dgrad":
            A, B, out = G, W, torch.empty(T, K, device=device, dtype=out_dt)
        else:
            A, B, out = G.t(), X, torch.empty(N, K, device=device, dtype=out_dt)
        return dict(A=A, B=B, out=out, X=X, W=W, G=G, flops=2.0 * T * K * N)
    if fam == "mla":
        r = MLA_ROWS[spec["row"]]
        H, Din, Dout, slot = r["H"], r["D_in"], r["D_out"], r["act_stride_head"]
        act = rnd(T, H, slot)[..., :Din]
        if r["weight_layout"] == "in_out":
            weight = rnd(H, Din, Dout, scale=Din**-0.5)
            w_in_out = weight
        else:
            weight = rnd(H, Dout, Din, scale=Din**-0.5)
            w_in_out = weight.transpose(1, 2)
        d_out = rnd(T, H, Dout)
        if op == "fwd":
            A, B = act.permute(1, 0, 2), w_in_out
            out = torch.empty(T, H, Dout, device=device, dtype=out_dt).permute(1, 0, 2)
        elif op == "dgrad":
            A, B = d_out.permute(1, 0, 2), w_in_out.transpose(1, 2)
            out = torch.empty(T, H, slot, device=device, dtype=out_dt)[
                ..., :Din
            ].permute(1, 0, 2)
        elif r["weight_layout"] == "in_out":
            A, B, out = (
                act.permute(1, 2, 0),
                d_out.permute(1, 0, 2),
                torch.empty_like(weight, dtype=out_dt),
            )
        else:
            A, B, out = (
                d_out.permute(1, 2, 0),
                act.permute(1, 0, 2),
                torch.empty_like(weight, dtype=out_dt),
            )
        return dict(A=A, B=B, out=out, flops=2.0 * T * H * Din * Dout)
    K, N = ROUTER_K, ROUTER_N
    X = torch.randn(T, K, device=device, generator=g)
    W = torch.randn(N, K, device=device, generator=g) * 0.02
    G = torch.randn(T, N, device=device, generator=g) * 1e-3
    if op == "fwd":
        A, B, out = X, W.t(), torch.empty(T, N, device=device)
    elif op == "dgrad":
        A, B, out = G, W, torch.empty(T, K, device=device)
    else:
        A, B, out = G.t(), X, torch.empty(N, K, device=device)
    return dict(A=A, B=B, out=out, X=X, W=W, G=G, flops=2.0 * T * K * N)


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------


def cake_arm(spec, inp):
    """The prepared Cake launch of the row (allocation-free replays)."""
    if spec["family"] == "router":
        prepared = cake_backend.prepare_router_fp32_gemm(inp["A"], inp["B"], inp["out"])
    elif spec["family"] == "proj" and spec["op"] == "wgrad":
        prepared = cake_backend.prepare_projection_wgrad(inp["G"], inp["X"], inp["out"])
    else:
        prepared = cake_backend.prepare_dense_projection_gemm(
            inp["A"], inp["B"], inp["out"]
        )
    return prepared


def torch_arm(spec, inp):
    """``torch.matmul`` on the same views: cuBLAS for the BF16 rows (bf16 GEMM + cast for
    the fp32-output gradient rows, torch has no native bf16 -> fp32 route), SGEMM with TF32
    disabled for the router rows."""
    A, B, out = inp["A"], inp["B"], inp["out"]
    if spec["family"] == "router":
        torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
            torch.backends.cuda.matmul.fp32_precision = "ieee"
    direct = out.dtype == A.dtype and out.is_contiguous()

    def run():
        if direct:
            torch.matmul(A, B, out=out)
        else:
            out.copy_(torch.matmul(A, B))
        return out

    return run


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def median_ms(fn, steps):
    times = bench_gpu_time(
        fn, dry_run_iters=3, repeat_iters=steps, enable_cupti=True, cold_l2_cache=True
    )
    return float(statistics.median(times))


def accuracy(inp, out):
    expected = torch.matmul(inp["A"].double(), inp["B"].double())
    got = out.t() if out.shape != expected.shape else out
    diff = (got.double() - expected).abs()
    record = dict(
        finite=bool(torch.isfinite(got).all()),
        max_abs_err=float(diff.max()),
        rel_fro=float(diff.norm() / max(float(expected.norm()), 1e-300)),
        ref_absmax=float(expected.abs().max()),
    )
    if got.dtype == torch.bfloat16:
        record["bf16_violations"] = int((diff > 1e-2 + 1e-2 * expected.abs()).sum())
    return record


def measure_row(label, spec, args):
    inp = make_inputs(spec, SEED, args.device)
    entry = dict(row=label, spec=spec, flops=inp["flops"])
    try:
        prepared = cake_arm(spec, inp)
        entry.update(
            template=prepared.template,
            module=prepared.module_name,
            grid=list(prepared.grid),
        )
        if spec["family"] == "router":
            entry["splits"] = prepared.splits
        prepared.launch()  # initializes the descriptor workspace outside the timed region
        torch.cuda.synchronize()
        if args.accuracy:
            entry["cake_accuracy"] = accuracy(inp, inp["out"])
        baseline = torch_arm(spec, inp)
        baseline()
        torch.cuda.synchronize()
        if args.accuracy:
            entry["torch_accuracy"] = accuracy(inp, inp["out"])
        # Equal-sample comparison: both arms are timed twice in alternating order
        # (Cake / torch / Cake / torch) and reduced the same way, so neither arm
        # gets the "best of two medians" advantage.
        cake_a = median_ms(prepared.launch, args.steps)
        torch_a = median_ms(baseline, args.steps)
        cake_b = median_ms(prepared.launch, args.steps)
        torch_b = median_ms(baseline, args.steps)
        cake_ms = min(cake_a, cake_b)
        torch_ms = min(torch_a, torch_b)
        entry.update(
            cake_ms=cake_ms,
            cake_ms_pair=[cake_a, cake_b],
            torch_ms_pair=[torch_a, torch_b],
            torch_ms=torch_ms,
            speedup_vs_torch=torch_ms / cake_ms,
            cake_tflops=inp["flops"] / (cake_ms * 1e-3) / 1e12,
            torch_tflops=inp["flops"] / (torch_ms * 1e-3) / 1e12,
        )
        print(
            f"{label:34s} {entry['template']:36s} cake {cake_ms:8.4f} ms ({entry['cake_tflops']:7.1f} TFLOPS)"
            f"  torch {torch_ms:8.4f} ms  x{entry['speedup_vs_torch']:5.2f}",
            flush=True,
        )
    except (
        Exception
    ) as exc:  # a missing instance or a failing row is recorded, not hidden
        entry["error"] = f"{type(exc).__name__}: {exc}"
        entry["traceback"] = traceback.format_exc()
        print(f"{label:34s} failed: {entry['error']}", flush=True)
    del inp
    torch.cuda.empty_cache()
    return entry


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--rows", nargs="*", default=None, help="row labels (default: every row)"
    )
    parser.add_argument("--ops", default="fwd,dgrad,wgrad")
    parser.add_argument("--T", default=",".join(str(t) for t in CANONICAL_T))
    parser.add_argument(
        "--f32-grads", action="store_true", help="add the FP32-output gradient rows"
    )
    parser.add_argument("--no-mla", action="store_true")
    parser.add_argument("--no-router", action="store_true")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--accuracy", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    ops = tuple(o for o in args.ops.split(",") if o)
    unknown = sorted(set(ops) - set(OPS))
    if unknown:
        parser.error(f"unknown ops {unknown}; choose from {list(OPS)}")
    T_values = tuple(int(t) for t in args.T.split(",") if t)
    rows = row_table(T_values, ops, args.f32_grads, not args.no_mla, not args.no_router)
    if args.rows:
        missing = sorted(set(args.rows) - set(rows))
        if missing:
            parser.error(f"unknown rows {missing}")
        rows = {label: rows[label] for label in args.rows}
    device = torch.device(args.device)
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(device)
    args.device = device
    results = dict(
        device=torch.cuda.get_device_name(),
        capability=list(torch.cuda.get_device_capability()),
        sm_count=int(torch.cuda.get_device_properties(device).multi_processor_count),
        arch=cake_backend.arch_for(device),
        torch=torch.__version__,
        steps=args.steps,
        rows=[],
    )
    for label, spec in rows.items():
        results["rows"].append(measure_row(label, spec, args))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2, default=str) + "\n")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
