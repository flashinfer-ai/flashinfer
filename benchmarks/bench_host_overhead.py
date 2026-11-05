"""Per-call HOST (CPU) overhead of the GEMM dispatch path.

Host cost only matters when it exceeds the kernel, which is the eager
LLM-serving regime: small M (decode), or prefill without CUDA graphs on a GEMM
short enough that dispatch is comparable to the kernel.  This measures enqueue
rate -- calls with no per-call sync, clock stopped before draining -- so
wall/call is the host cost.  ``--kernel`` adds GPU time per call (torch
profiler, all device activity) and flags rows where host > GPU, i.e. where an
eager caller starves the GPU.

    python benchmarks/bench_host_overhead.py [--api bf16 fp4 mxfp8 fp8 quant]
        [--shape M N K] [--kernel] [--json out.json] [--compare base.json]

Use --compare to print a before/after table when changing the dispatch path.
"""

import argparse
import json
import statistics
import time

import torch

import flashinfer

DEFAULT_SHAPE = (8, 2048, 2048)  # M, N, K -- kernel ~2 us, dispatch dominates


def bench(fn, reps=300, warmup=50, chunk=25):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    # Short chunks keep the launch queue from filling when the kernel is long
    # (prefill); a full queue blocks the host and inflates the reading.
    chunk = min(chunk, reps)
    samples = []
    for _ in range(max(1, reps // chunk)):
        t0 = time.perf_counter()
        for _ in range(chunk):
            fn()
        # Stop the clock before draining: GPU completion is not host cost.
        samples.append((time.perf_counter() - t0) / chunk * 1e6)
        torch.cuda.synchronize()
    return statistics.median(samples)


def kernel_us(fn, iters=10):
    """GPU time per call (summed device activity) and device ops per call."""
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile

    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    total_ns, n = 0, 0
    for e in prof.profiler.kineto_results.events():
        if e.device_type() == DeviceType.CUDA:
            total_ns += e.duration_ns()
            n += 1
    return total_ns / iters / 1e3, n / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--shape",
        type=int,
        nargs=3,
        default=list(DEFAULT_SHAPE),
        metavar=("M", "N", "K"),
    )
    ap.add_argument(
        "--api",
        nargs="+",
        default=["bf16", "fp4"],
        choices=["bf16", "fp4", "mxfp8", "fp8", "quant"],
    )
    ap.add_argument(
        "--only", nargs="+", default=[], help="keep rows whose name contains one"
    )
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--kernel", action="store_true", help="also report GPU us/call")
    ap.add_argument("--json", type=str, default="")
    ap.add_argument("--compare", type=str, default="")
    args = ap.parse_args()

    dev = torch.device("cuda:0")
    m, n, k = args.shape
    a = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
    b = torch.randn(k, n, device=dev, dtype=torch.bfloat16).t().contiguous().t()
    w = b.t().contiguous()  # (n, k) weight
    out = torch.empty(m, n, device=dev, dtype=torch.bfloat16)

    props = torch.cuda.get_device_properties(0)
    print(
        f"{props.name} sm{props.major}{props.minor} | M={m} N={n} K={k} "
        f"| torch {torch.__version__} | flashinfer {flashinfer.__version__}"
    )

    results = {}
    kernels = {}

    def wanted(*names):
        return not args.only or any(x in n for n in names for x in args.only)

    def add(name, fn):
        if name != "torch.mm" and not wanted(name):
            return
        # A backend can pass its first call and still fail later (warmup, re-plan);
        # skip it rather than abort the whole table.
        try:
            fn()
            results[name] = bench(fn, args.reps)
            if args.kernel:
                kernels[name] = kernel_us(fn)
        except Exception as exc:
            msg = str(exc).strip().splitlines()
            print(f"  skip {name}: {msg[0][:90] if msg else type(exc).__name__}")

    add("torch.mm", lambda: torch.mm(a, b, out=out))

    if "bf16" in args.api:
        for backend in ("auto", "cudnn", "cublaslt", "cutlass", "tgv", "tinygemm"):
            add(
                f"mm_bf16({backend})",
                lambda x=backend: flashinfer.mm_bf16(a, b, out=out, backend=x),
            )

    blackwell = props.major in (10, 12)
    lowp = ("auto", "cudnn", "cutlass", "cute-dsl", "cutedsl_low_latency")
    # Prepare inputs only for rows --only keeps, so a failing unselected
    # preparation cannot abort the selected ones.
    fp4_rows = [f"mm_fp4({x})" for x in lowp + ("trtllm",)]
    want_fp4 = "fp4" in args.api and wanted(*fp4_rows)
    want_nvfp4_quant = "quant" in args.api and wanted(
        "nvfp4_quantize(a)", "fp4_quantize(a)"
    )
    if blackwell and (want_fp4 or want_nvfp4_quant):
        layout = flashinfer.SfLayout.layout_128x4
        ga = (448 * 6) / a.float().abs().max()
        gw = (448 * 6) / w.float().abs().max()
        a4, a_sf = flashinfer.nvfp4_quantize(a, ga, sfLayout=layout, do_shuffle=False)
        w4, w_sf = flashinfer.nvfp4_quantize(w, gw, sfLayout=layout, do_shuffle=False)
        alpha = (1.0 / (ga * gw)).reshape(1).float()

    if blackwell and want_fp4:
        for backend in lowp:
            add(
                f"mm_fp4({backend})",
                lambda x=backend: flashinfer.mm_fp4(
                    a4,
                    w4.T,
                    a_sf,
                    w_sf.T,
                    alpha,
                    torch.bfloat16,
                    out=out,
                    block_size=16,
                    use_8x4_sf_layout=False,
                    backend=x,
                ),
            )
    if blackwell and want_fp4 and wanted("mm_fp4(trtllm)") and n % 128 == 0:
        # trtllm swaps A and B, so its weight is shuffled (N % 128) at load time.
        w4s, w_sfs = flashinfer.nvfp4_quantize(w, gw, sfLayout=layout, do_shuffle=True)
        add(
            "mm_fp4(trtllm)",
            lambda: flashinfer.mm_fp4(
                a4,
                w4s.T,
                a_sf,
                w_sfs.T,
                alpha,
                torch.bfloat16,
                out=out,
                block_size=16,
                use_8x4_sf_layout=False,
                backend="trtllm",
            ),
        )

    if blackwell and "mxfp8" in args.api and wanted(*(f"mm_mxfp8({x})" for x in lowp)):
        a8, a8_sf = flashinfer.mxfp8_quantize(a, is_sf_swizzled_layout=True)
        w8, w8_sf = flashinfer.mxfp8_quantize(w, is_sf_swizzled_layout=True)
        for backend in lowp:
            add(
                f"mm_mxfp8({backend})",
                lambda x=backend: flashinfer.mm_mxfp8(
                    a8, w8.t(), a8_sf, w8_sf, out=out, backend=x
                ),
            )
    if (
        blackwell
        and "mxfp8" in args.api
        and wanted("mm_mxfp8(trtllm)")
        and n % 128 == 0
        and k % 256 == 0
    ):
        w8l, w8l_sf = flashinfer.mxfp8_quantize(w, is_sf_swizzled_layout=False)
        w8t = flashinfer.shuffle_matrix_a(w8l, 128).reshape(n, k)
        w8t_sf = flashinfer.shuffle_matrix_sf_a(
            w8l_sf.reshape(n, k // 32), 128, num_elts_per_sf=32
        ).reshape(-1)
        a8x, a8x_sf = flashinfer.mxfp8_quantize(
            a, sf_swizzle_layout=flashinfer.SfLayout.layout_8x4
        )
        add(
            "mm_mxfp8(trtllm)",
            lambda: flashinfer.mm_mxfp8(
                a8x,
                w8t.t(),
                a8x_sf,
                w8t_sf,
                out=out,
                backend="trtllm",
                use_8x4_sf_layout=True,
            ),
        )

    if "fp8" in args.api and props.major >= 9:
        a_f8 = a.to(torch.float8_e4m3fn)
        w_f8 = w.to(torch.float8_e4m3fn)
        scale = torch.ones(1, device=dev, dtype=torch.float32)
        for backend in ("auto", "cudnn", "cublas", "cutlass", "cutedsl_low_latency"):
            add(
                f"mm_fp8({backend})",
                lambda x=backend: flashinfer.mm_fp8(
                    a_f8, w_f8, scale, out=out, backend=x
                ),
            )
        a3, b3, o3 = a_f8.unsqueeze(0), w_f8.t().unsqueeze(0), out.unsqueeze(0)
        for backend in ("auto", "cudnn", "cublas", "cutlass"):
            add(
                f"bmm_fp8({backend})",
                lambda x=backend: flashinfer.bmm_fp8(
                    a3, b3, scale, scale, torch.bfloat16, o3, backend=x
                ),
            )

    if blackwell and "quant" in args.api:
        # Per-call activation quantization: part of the launch cost a serving
        # framework pays in front of every W4A4 / W8A8 GEMM.
        add(
            "mxfp8_quantize(a)",
            lambda: flashinfer.mxfp8_quantize(a, is_sf_swizzled_layout=True),
        )
        add(
            "mxfp8_quantize(a, cute-dsl)",
            lambda: flashinfer.mxfp8_quantize(
                a, is_sf_swizzled_layout=True, backend="cute-dsl"
            ),
        )
        add(
            "nvfp4_quantize(a)",
            lambda: flashinfer.nvfp4_quantize(a, ga, sfLayout=layout, do_shuffle=False),
        )
        add("fp4_quantize(a)", lambda: flashinfer.fp4_quantize(a, ga))

    base = None
    if args.compare:
        with open(args.compare) as f:
            saved = json.load(f)
        if saved.get("shape") != [m, n, k] or saved.get("gpu") != props.name:
            raise SystemExit(
                f"--compare baseline was measured on {saved.get('gpu')} shape "
                f"{saved.get('shape')}; this run is {props.name} shape {[m, n, k]}"
            )
        base = saved["results"]
    width = max(len(x) for x in results)
    print(
        f"\n{'':<{width}}  {'us/call':>9s}"
        + (f"{'gpu us':>9s}{'ops':>6s}" if kernels else "")
        + (f"{'baseline':>10s}{'delta':>10s}" if base else "")
    )
    for name, us in results.items():
        line = f"{name:<{width}}  {us:9.1f}"
        if kernels:
            gpu, ops = kernels.get(name, (float("nan"), 0.0))
            line += f"{gpu:9.1f}{ops:6.1f}"
        if base and name in base:
            d = us - base[name]
            line += f"{base[name]:10.1f}{d:+10.1f}"
        if kernels and us > kernels.get(name, (float("inf"),))[0]:
            line += "  host>gpu"
        print(line)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(
                {
                    "shape": [m, n, k],
                    "gpu": props.name,
                    "results": results,
                    "kernels": kernels,
                },
                f,
                indent=1,
            )
        print(f"\n-> {args.json}")


if __name__ == "__main__":
    main()
