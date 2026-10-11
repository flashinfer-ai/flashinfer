# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
W8A16 FP8 GEMM benchmark on the Cosmos3-Super Linear shapes.

    out[M, N] = act[M, K] @ (fp8_weight[N, K] * weight_scale).T (+ bias)

The weight is FP8 e4m3 with ONE per-tensor fp32 scale (the layout TRT-LLM's
FP8QDQ loader produces); the activation is bf16 or fp16, the output is the
activation dtype.  TRT-LLM's ``W8A16FP8LinearMethod.apply`` today is a
cast-copy -- ``weight.to(dtype).mul_(scale)`` then ``F.linear`` -- and this
script benchmarks the FlashInfer fused kernel (``flashinfer.gemm.w8a16_fp8_gemm``,
weight dequantized on the fly, never materialised) against it.

Columns reported per (layer, TP, M) row:

    us         median GPU time (CUPTI kernel span, L2 flushed per iteration)
    TFLOP/s    2*M*N*K / t
    GB/s(W8)   N*K bytes of FP8 weight / t  (the memory-bound view of decode)
    vs cast    cast-copy time / kernel time   (>1 means the kernel is faster)
    vs cublas  cuBLAS-bf16-on-dequantised-weight time / kernel time (ceiling: ~1)
    max|err|   max abs error of the kernel vs an fp32 reference
    max rel    max |err| / (|ref| + 1) of the kernel vs the fp32 reference

Correctness gate: the kernel's max abs error against the fp32 reference must
not exceed ``--tol-mult`` times the cast-copy path's own error against that
reference (both only differ from fp32 by bf16/fp16 accumulation order), with a
small absolute floor.  A failing shape is marked FAIL and the exit status is 1.

Shapes
------
Cosmos3-Super ``transformer/config.json``: hidden 5120, intermediate 25600,
64 q heads, 8 kv heads, head_dim 128.  Static-FP8 Cosmos3 in TRT-LLM builds
the und self-attention with separate q/k/v, the gen cross-attention with a
fused qkv, the MLP with *split* gate/up (``split_gate_up=uses_static_fp8``).
Column-parallel layers shard N, row-parallel layers shard K.  Single GPU; the
TP degree only sets the shard sizes.

M (token rows through one Linear) comes from the cookbook workloads in
NVIDIA/cosmos#343 (1024x1024 images, 1280x720x189-frame videos, text padded
to ``max_sequence_length`` 4096).  The W8A16 path runs on the first/last
denoising steps of the gen tower and always on the und tower:

    1024    T2I gen tokens      (64x64 latent, patch 2)
    2048    T2I, cond+uncond batched (CFG on one GPU)
    4096    und text tower      (padded to max_sequence_length)
    8192    und text, cond+uncond
    44160   720p x 189f gen tokens  (48 x 23 x 40 latent patches)
    88320   720p x 189f, cond+uncond

Usage
-----
    python benchmarks/bench_w8a16_fp8_gemm.py                      # whole table
    python benchmarks/bench_w8a16_fp8_gemm.py --tp 4 --layer mlp.down --m 8192
    python benchmarks/bench_w8a16_fp8_gemm.py --shape 5120,25600 --m 2048  # N,K
    python benchmarks/bench_w8a16_fp8_gemm.py --impls kernel --m 44160 --csv out.csv
    python benchmarks/bench_w8a16_fp8_gemm.py --dtype fp16 --bias

``--impls`` picks which columns run; ``--shape N,K`` with ``--m`` pins one
problem so another pipeline (ncu, a regression bisect) can capture exactly it.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import os
import statistics
import sys
from dataclasses import dataclass
from functools import partial
from typing import Callable

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Cosmos3-Super geometry -> Linear (N, K) per TP degree
# ---------------------------------------------------------------------------

HIDDEN = 5120
INTERMEDIATE = 25600
NUM_Q_HEADS = 64
NUM_KV_HEADS = 8
HEAD_DIM = 128
Q_DIM = NUM_Q_HEADS * HEAD_DIM  # 8192
KV_DIM = NUM_KV_HEADS * HEAD_DIM  # 1024


@dataclass(frozen=True)
class LayerSpec:
    name: str
    n: int  # out_features at TP1
    k: int  # in_features at TP1
    shard: str  # "col" shards N, "row" shards K

    def at_tp(self, tp: int) -> tuple[int, int]:
        if self.shard == "col":
            assert self.n % tp == 0, (self.name, tp)
            return self.n // tp, self.k
        assert self.k % tp == 0, (self.name, tp)
        return self.n, self.k // tp


# Order matters only for the table.  Shapes that coincide (und.k == und.v,
# attn.out shared by und/gen, mlp.gate == mlp.up) are listed once.
LAYERS: list[LayerSpec] = [
    LayerSpec("und.q", Q_DIM, HIDDEN, "col"),
    LayerSpec("und.kv", KV_DIM, HIDDEN, "col"),
    LayerSpec("gen.qkv", Q_DIM + 2 * KV_DIM, HIDDEN, "col"),
    LayerSpec("attn.out", HIDDEN, Q_DIM, "row"),
    LayerSpec("mlp.gate_up", INTERMEDIATE, HIDDEN, "col"),
    LayerSpec("mlp.down", HIDDEN, INTERMEDIATE, "row"),
]
LAYER_BY_NAME = {layer.name: layer for layer in LAYERS}

DEFAULT_TPS = [2, 4, 8]
DEFAULT_MS = [1024, 2048, 4096, 8192, 44160, 88320]

DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16}

# ---------------------------------------------------------------------------
# Timing (pattern from bench_checkpointing_ssu.py::_time_kernel_cupti)
# ---------------------------------------------------------------------------

_L2_FLUSH_SIZE = 32 * 1024 * 1024  # float32 elements -> 128 MB
_l2_flush: torch.Tensor | None = None


def _init_l2_flush() -> None:
    global _l2_flush
    if _l2_flush is None:
        _l2_flush = torch.empty(_L2_FLUSH_SIZE, dtype=torch.float32, device="cuda")


def _compute_stats(latencies_us: list[float]) -> tuple[float, float, float]:
    median_us = statistics.median(latencies_us)
    s = sorted(latencies_us)
    p95_us = s[min(int(0.95 * len(s)), len(s) - 1)]
    p99_us = s[min(int(0.99 * len(s)), len(s) - 1)]
    return median_us, p95_us, p99_us


def finalize_cupti() -> None:
    """cuptiFinalize() is once-per-process teardown; never per timing."""
    try:
        from cupti import cupti

        cupti.finalize()
    except Exception:
        pass


def _capture_graphs(args, run_fn: Callable[[], None]):
    """Eager warmup, then two graphs: (L2 flush) and (the measured op)."""
    run_fn()
    torch.cuda.synchronize()

    g_reset = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_reset):
        if args.l2_flush:
            _l2_flush.fill_(0.0)
    torch.cuda.synchronize()

    g_run = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_run):
        run_fn()
    torch.cuda.synchronize()

    for _ in range(args.warmup):
        g_reset.replay()
        g_run.replay()
    torch.cuda.synchronize()
    return g_reset, g_run


def _time_kernel_cupti(args, run_fn: Callable[[], None], tag: str):
    from cupti import cupti

    def _buf_requested():
        return 8 * 1024 * 1024, 0

    def _buf_completed(launches, kernels, activities):
        for a in activities:
            if a.kind in (
                cupti.ActivityKind.CONCURRENT_KERNEL,
                cupti.ActivityKind.MEMCPY,
                cupti.ActivityKind.MEMSET,
            ):
                name = (
                    a.name if a.kind == cupti.ActivityKind.CONCURRENT_KERNEL else None
                )
                kernels.append((a.start, a.end, a.correlation_id, name))
            elif a.kind in (cupti.ActivityKind.RUNTIME, cupti.ActivityKind.DRIVER):
                launches.append((a.start, a.end, a.correlation_id))

    g_reset, g_run = _capture_graphs(args, run_fn)

    launches: list[tuple[int, int, int]] = []
    kernels: list[tuple[int, int, int, str | None]] = []
    iter_timestamps: list[tuple[int, int]] = []

    kinds = [
        cupti.ActivityKind.RUNTIME,
        cupti.ActivityKind.CONCURRENT_KERNEL,
        cupti.ActivityKind.DRIVER,
        cupti.ActivityKind.MEMCPY,
        cupti.ActivityKind.MEMSET,
    ]
    for kind in kinds:
        cupti.activity_enable(kind)
    cupti.activity_register_callbacks(
        _buf_requested, partial(_buf_completed, launches, kernels)
    )

    torch.cuda.nvtx.range_push(tag)
    for _ in range(args.iters):
        g_reset.replay()
        torch.cuda.synchronize()
        start_cpu = cupti.get_timestamp()
        g_run.replay()
        torch.cuda.synchronize()
        end_cpu = cupti.get_timestamp()
        iter_timestamps.append((start_cpu, end_cpu))
    torch.cuda.nvtx.range_pop()

    cupti.activity_flush_all(0)
    for kind in kinds:
        cupti.activity_disable(kind)
    # No cupti.finalize() here: once per process, in main().

    corr_to_kernels: dict[int, list] = {}
    for k in kernels:
        corr_to_kernels.setdefault(k[2], []).append(k)
    sorted_launches = sorted(launches, key=lambda launch: launch[0])
    launch_starts = [launch[0] for launch in sorted_launches]

    latencies_us: list[float] = []
    for idx, (t0, t1) in enumerate(iter_timestamps):
        li = bisect.bisect_left(launch_starts, t0)
        ri = bisect.bisect_right(launch_starts, t1)
        corr_ids = {sorted_launches[i][2] for i in range(li, ri)}
        iter_kernels = []
        for cid in corr_ids:
            iter_kernels.extend(corr_to_kernels.get(cid, []))
        if not iter_kernels:
            raise ValueError(f"No kernel activities recorded for iteration {idx}")
        min_start = min(k[0] for k in iter_kernels)
        max_end = max(k[1] for k in iter_kernels)
        latencies_us.append((max_end - min_start) / 1e3)
        if os.environ.get("W8A16_CUPTI_DEBUG") and idx == 0:
            print(
                f"\n[CUPTI-DEBUG] tag={tag} n_kernels={len(iter_kernels)}",
                file=sys.stderr,
            )
            for k in sorted(iter_kernels, key=lambda r: r[0]):
                nm = k[3] if k[3] is not None else "<memcpy/memset>"
                print(
                    f"  start={(k[0] - min_start) / 1e3:8.3f}  "
                    f"end={(k[1] - min_start) / 1e3:8.3f} us  {nm}",
                    file=sys.stderr,
                )
    return _compute_stats(latencies_us)


def _time_kernel_cuda_graph(args, run_fn: Callable[[], None], tag: str):
    """CUDA-event fallback (includes graph launch overhead; CUPTI preferred)."""
    g_reset, g_run = _capture_graphs(args, run_fn)
    latencies_us: list[float] = []
    for _ in range(args.iters):
        g_reset.replay()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        g_run.replay()
        end.record()
        torch.cuda.synchronize()
        latencies_us.append(start.elapsed_time(end) * 1e3)
    return _compute_stats(latencies_us)


_CUPTI_UNAVAILABLE_WARNED = False


def _time_kernel(args, run_fn: Callable[[], None], tag: str):
    if args.l2_flush:
        _init_l2_flush()
    if args.cupti:
        try:
            import cupti  # noqa: F401
        except ImportError:
            global _CUPTI_UNAVAILABLE_WARNED
            if not _CUPTI_UNAVAILABLE_WARNED:
                print(
                    "# WARNING: --cupti requested but cupti-python not installed; "
                    "falling back to CUDA-graph timing.  pip install -U cupti-python",
                    file=sys.stderr,
                )
                _CUPTI_UNAVAILABLE_WARNED = True
        else:
            return _time_kernel_cupti(args, run_fn, tag)
    return _time_kernel_cuda_graph(args, run_fn, tag)


# ---------------------------------------------------------------------------
# Implementations under test
# ---------------------------------------------------------------------------


def make_cast_copy(
    act: torch.Tensor, w_fp8: torch.Tensor, scale: torch.Tensor, bias
) -> Callable[[], torch.Tensor]:
    """TRT-LLM W8A16FP8LinearMethod.apply, verbatim semantics."""
    dtype = act.dtype

    def run() -> torch.Tensor:
        weight = w_fp8.to(dtype).mul_(scale.to(dtype))
        return F.linear(act, weight, bias)

    return run


def make_cublas(
    act: torch.Tensor, w_deq: torch.Tensor, bias
) -> Callable[[], torch.Tensor]:
    """cuBLAS 16-bit GEMM on a pre-dequantised weight: the ceiling."""

    def run() -> torch.Tensor:
        return F.linear(act, w_deq, bias)

    return run


def make_kernel(
    act: torch.Tensor,
    w_fp8: torch.Tensor,
    scale: torch.Tensor,
    bias,
    prepared: bool = False,
) -> Callable[[], torch.Tensor] | None:
    """FlashInfer kernel; ``prepared`` uses the opt-in prepared weight layout
    (the one-time ``prepare_w8a16_fp8_weight`` copy is done here, untimed)."""
    try:
        from flashinfer.gemm import prepare_w8a16_fp8_weight, w8a16_fp8_gemm
    except ImportError:
        return None
    out = torch.empty(act.shape[0], w_fp8.shape[0], dtype=act.dtype, device=act.device)
    w_use = prepare_w8a16_fp8_weight(w_fp8) if prepared else w_fp8

    def run() -> torch.Tensor:
        return w8a16_fp8_gemm(
            act, w_use, scale, bias=bias, out=out, weight_prepared=prepared
        )

    return run


# ---------------------------------------------------------------------------
# Per-shape benchmark
# ---------------------------------------------------------------------------


@dataclass
class Row:
    layer: str
    tp: int
    m: int
    n: int
    k: int
    us: dict  # impl -> median us
    max_abs: dict  # impl -> max abs err vs fp32 ref
    max_rel: dict  # impl -> max rel err vs fp32 ref
    passed: bool | None  # None when the kernel was not run


def _fp32_reference(act: torch.Tensor, w_deq_f32: torch.Tensor, bias) -> torch.Tensor:
    ref = act.float() @ w_deq_f32.T
    if bias is not None:
        ref += bias.float()
    return ref


def _errors(out: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    diff = (out.float() - ref).abs()
    max_abs = diff.max().item()
    max_rel = (diff / (ref.abs() + 1.0)).max().item()
    return max_abs, max_rel


def bench_shape(args, layer: str, tp: int, m: int, n: int, k: int) -> Row:
    dtype = DTYPE_MAP[args.dtype]
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(args.seed)

    act = torch.randn(m, k, dtype=torch.float32, device=dev, generator=gen).to(dtype)
    # Weight in a realistic range, then quantised to fp8 with one tensor scale
    # the way the FP8QDQ checkpoint stores it (scale = amax / 448).
    w32 = torch.randn(n, k, dtype=torch.float32, device=dev, generator=gen) * 0.02
    scale = (w32.abs().amax() / 448.0).reshape(()).to(torch.float32)
    w_fp8 = (w32 / scale).to(torch.float8_e4m3fn)
    del w32
    bias = (
        torch.randn(n, dtype=torch.float32, device=dev, generator=gen).to(dtype)
        if args.bias
        else None
    )
    w_deq_f32 = w_fp8.float() * scale
    w_deq = w_deq_f32.to(dtype)
    ref = _fp32_reference(act, w_deq_f32, bias)

    impls: dict[str, Callable[[], torch.Tensor]] = {}
    if "cast" in args.impls:
        impls["cast"] = make_cast_copy(act, w_fp8, scale, bias)
    if "cublas" in args.impls:
        impls["cublas"] = make_cublas(act, w_deq, bias)
    for name, prepared in (("kernel", False), ("kernel_prep", True)):
        if name not in args.impls:
            continue
        fn = make_kernel(act, w_fp8, scale, bias, prepared=prepared)
        if fn is None:
            print(
                "# WARNING: flashinfer.gemm.w8a16_fp8_gemm not available; "
                f"{name} column skipped",
                file=sys.stderr,
            )
        else:
            impls[name] = fn

    us: dict[str, float] = {}
    max_abs: dict[str, float] = {}
    max_rel: dict[str, float] = {}
    for name, fn in impls.items():
        out = fn()
        torch.cuda.synchronize()
        max_abs[name], max_rel[name] = _errors(out, ref)
        del out
        tag = f"w8a16[{layer} tp{tp} M={m} N={n} K={k}] {name}"
        us[name], _, _ = _time_kernel(args, lambda: fn(), tag)

    passed: bool | None = None
    kernel_cols = [c for c in ("kernel", "kernel_prep") if c in impls]
    if kernel_cols:
        # Both the kernel and the cast-copy differ from fp32 only by 16-bit
        # accumulation order; demand the kernel be no worse than that path
        # (with a floor so an exactly-fp32-accumulating reference isn't
        # penalised for a tiny cast-copy error).
        cast_err = max_abs.get("cast")
        if cast_err is None:
            # cast column not requested: compute it once for the gate only.
            out = make_cast_copy(act, w_fp8, scale, bias)()
            torch.cuda.synchronize()
            cast_err, _ = _errors(out, ref)
            del out
        floor = args.tol_floor * ref.abs().max().item()
        passed = all(
            max_abs[c] <= max(args.tol_mult * cast_err, floor)
            and torch.isfinite(torch.tensor(max_abs[c]))
            for c in kernel_cols
        )

    del act, w_fp8, w_deq, w_deq_f32, ref
    torch.cuda.empty_cache()
    return Row(layer, tp, m, n, k, us, max_abs, max_rel, passed)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt_us(v: float | None) -> str:
    return f"{v:13.1f}" if v is not None else f"{'-':>13}"


def _ratio(num: float | None, den: float | None) -> str:
    if num is None or den is None or den == 0:
        return f"{'-':>7}"
    r = num / den
    return f"{r:7.3f}" if r < 0.1 else f"{r:7.2f}"


def _impl_cols(args) -> list[str]:
    return [i for i in ("cast", "cublas", "kernel", "kernel_prep") if i in args.impls]


def table_header(args) -> str:
    hdr = f"{'layer':<12}{'tp':>3}{'M':>7}{'N':>7}{'K':>7}"
    for i in _impl_cols(args):
        hdr += f"{i + ' us':>15}"
    hdr += f"{'TFLOP/s':>9}{'GB/s(W8)':>10}{'vs cast':>8}{'vs cublas':>10}"
    hdr += f"{'max|err|':>10}{'max rel':>10}{'cast|err|':>10}{'gate':>6}"
    return hdr + "\n" + "-" * len(hdr)


def format_row(r: Row, args) -> str:
    # Throughput columns describe the kernel when it ran, else the cast path.
    primary = next(
        (c for c in ("kernel_prep", "kernel", "cublas", "cast") if c in r.us), "cast"
    )
    t = r.us.get(primary)
    flops = 2.0 * r.m * r.n * r.k
    tflops = flops / (t * 1e-6) / 1e12 if t else float("nan")
    gbs = (r.n * r.k) / (t * 1e-6) / 1e9 if t else float("nan")
    line = f"{r.layer:<12}{r.tp:>3}{r.m:>7}{r.n:>7}{r.k:>7}"
    for i in _impl_cols(args):
        line += f"  {_fmt_us(r.us.get(i))}"
    line += f"{tflops:9.1f}{gbs:10.1f}"
    line += f" {_ratio(r.us.get('cast'), r.us.get(primary))}"
    line += f"   {_ratio(r.us.get('cublas'), r.us.get(primary))}"
    line += f"{r.max_abs.get(primary, float('nan')):10.3e}"
    line += f"{r.max_rel.get(primary, float('nan')):10.3e}"
    line += f"{r.max_abs.get('cast', float('nan')):10.3e}"
    gate = "-" if r.passed is None else ("ok" if r.passed else "FAIL")
    return line + f"{gate:>6}"


def print_table(rows: list[Row], args) -> None:
    print(table_header(args))
    for r in rows:
        print(format_row(r, args))


def write_csv(rows: list[Row], path: str, args) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "layer",
                "tp",
                "M",
                "N",
                "K",
                "dtype",
                "bias",
                "cast_us",
                "cublas_us",
                "kernel_us",
                "kernel_prep_us",
                "kernel_max_abs",
                "kernel_max_rel",
                "kernel_prep_max_abs",
                "cast_max_abs",
                "cast_max_rel",
                "gate",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.layer,
                    r.tp,
                    r.m,
                    r.n,
                    r.k,
                    args.dtype,
                    int(args.bias),
                    r.us.get("cast", ""),
                    r.us.get("cublas", ""),
                    r.us.get("kernel", ""),
                    r.us.get("kernel_prep", ""),
                    r.max_abs.get("kernel", ""),
                    r.max_rel.get("kernel", ""),
                    r.max_abs.get("kernel_prep", ""),
                    r.max_abs.get("cast", ""),
                    r.max_rel.get("cast", ""),
                    "" if r.passed is None else ("ok" if r.passed else "FAIL"),
                ]
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--tp",
        type=int,
        nargs="+",
        default=DEFAULT_TPS,
        help="TP degrees (shard sizes only; single GPU)",
    )
    p.add_argument(
        "--layer",
        nargs="+",
        default=[l.name for l in LAYERS],
        choices=[l.name for l in LAYERS],
        help="Linear layers to run",
    )
    p.add_argument("--m", type=int, nargs="+", default=DEFAULT_MS, help="token rows")
    p.add_argument(
        "--shape",
        type=str,
        default=None,
        help="fixed 'N,K' instead of the layer table (combine with --m)",
    )
    p.add_argument("--dtype", choices=list(DTYPE_MAP), default="bf16")
    p.add_argument("--bias", action="store_true", help="add a per-N bias")
    p.add_argument(
        "--impls",
        nargs="+",
        default=["cast", "cublas", "kernel", "kernel_prep"],
        choices=["cast", "cublas", "kernel", "kernel_prep"],
    )
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-l2-flush", dest="l2_flush", action="store_false")
    p.add_argument("--no-cupti", dest="cupti", action="store_false")
    p.add_argument(
        "--tol-mult",
        type=float,
        default=2.0,
        help="kernel max|err| may be at most this times the cast-copy's",
    )
    p.add_argument(
        "--tol-floor",
        type=float,
        default=2e-3,
        help="absolute floor on the gate as a fraction of max|ref|",
    )
    p.add_argument("--csv", type=str, default=None)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    props = torch.cuda.get_device_properties(0)
    print(
        f"# device: {props.name}  SMs={props.multi_processor_count}  "
        f"dtype={args.dtype} bias={args.bias} iters={args.iters} "
        f"l2_flush={args.l2_flush} timer={'cupti' if args.cupti else 'cuda-graph'}"
    )

    problems: list[tuple[str, int, int, int]] = []  # (layer, tp, n, k)
    if args.shape:
        n, k = (int(v) for v in args.shape.split(","))
        problems.append((f"{n}x{k}", 1, n, k))
    else:
        for tp in args.tp:
            for name in args.layer:
                n, k = LAYER_BY_NAME[name].at_tp(tp)
                problems.append((name, tp, n, k))

    rows: list[Row] = []
    # Rows stream out as they finish (the naive rungs take a long time on the
    # large shapes); the CSV is written once at the end.
    print(table_header(args), flush=True)
    try:
        for layer, tp, n, k in problems:
            for m in args.m:
                rows.append(bench_shape(args, layer, tp, m, n, k))
                print(format_row(rows[-1], args), flush=True)
    finally:
        finalize_cupti()

    if args.csv:
        write_csv(rows, args.csv, args)
        print(f"# wrote {args.csv}")
    failed = [r for r in rows if r.passed is False]
    if failed:
        print(f"# CORRECTNESS GATE FAILED on {len(failed)} shape(s)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
