"""Compare native M=1 blockscaled GEMM with a reusable M=128 padding workaround.

Run on SM120/121 from a FlashInfer checkout:
    python benchmarks/bench_fp8_blockscaled_single_row.py --n 1024 --k 2048
    python benchmarks/bench_fp8_blockscaled_single_row.py --compare-groupwise

For aligned-M regression measurements, run the same script against each checkout
using PYTHONPATH, with --m 128 (or another positive multiple of 128).

Inputs are ready FP8 bytes/scales; both paths write caller-owned contiguous out.
Padding buffers are allocated once. Each padded call includes the A input copy,
GEMM and output copy. Graph timing includes those device activities but excludes
host dispatch; eager timing includes the completed call and final synchronization.
Compilation, quantization and one-time buffer allocation are outside timing.
"""

import argparse
from importlib import import_module
from importlib.metadata import version
import json
import math
import random
import statistics
import time

import torch

import flashinfer
from flashinfer.gemm import gemm_fp8_nt_blockscaled, gemm_fp8_nt_groupwise
from flashinfer.testing import bench_gpu_time
from flashinfer.utils import is_sm12x_supported


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=1)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--scale-major-mode", choices=["K", "MN"], default="K")
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--compare-groupwise", action="store_true")
    args = parser.parse_args()
    if args.m != 1 and (args.m <= 0 or args.m % 128):
        parser.error("M must be 1 or a positive multiple of 128")
    if args.compare_groupwise and args.m != 1:
        parser.error("--compare-groupwise requires M=1 to reuse the original scales")
    if args.n <= 0 or args.k <= 0 or args.n % 128 or args.k % 128:
        parser.error("N and K must be positive multiples of 128")
    if args.samples < 1 or args.rounds < 1:
        parser.error("samples and rounds must be positive")
    # The graph fallback cannot rotate tensors captured only through closures.
    # Require CUPTI so cold_graph always includes an actual L2 flush.
    try:
        import_module("cupti.cupti")
        cupti_version = version("cupti-python")
        if int(cupti_version.split(".")[0]) < 13:
            raise RuntimeError(f"found cupti-python {cupti_version}")
    except Exception as exc:
        parser.error(f"CUPTI >= 13 is required for warm/cold measurements: {exc}")
    if not is_sm12x_supported(torch.device("cuda")):
        parser.error("SM120/121 required")
    torch.manual_seed(9556)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    m, n, k, mode = args.m, args.n, args.k, args.scale_major_mode
    a = (torch.randn((m, k), device="cuda") / 4).to(torch.float8_e4m3fn)
    b = (torch.randn((n, k), device="cuda") / math.sqrt(k)).to(torch.float8_e4m3fn)
    sa = torch.rand((math.ceil(m / 128), k // 128), device="cuda") + 0.25
    sb = torch.rand((n // 128, k // 128), device="cuda") + 0.25
    ad = a.float() * sa.repeat_interleave(128, 0)[:m].repeat_interleave(128, 1)
    bd = b.float() * sb.repeat_interleave(128, 0).repeat_interleave(128, 1)
    ref = (ad * bd).sum(1).unsqueeze(0) if m == 1 else ad @ bd.T
    del ad, bd
    if mode == "MN":
        sa, sb = sa.T.contiguous(), sb.T.contiguous()
    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    if m == 1:
        ap = torch.zeros((128, k), device="cuda", dtype=torch.float8_e4m3fn)
        op = torch.empty((128, n), device="cuda", dtype=out.dtype)

    def padded():
        ap[:1].copy_(a)
        gemm_fp8_nt_blockscaled(ap, b, sa, sb, scale_major_mode=mode, out=op)
        out.copy_(op[:1])

    def native():
        gemm_fp8_nt_blockscaled(a, b, sa, sb, scale_major_mode=mode, out=out)

    def groupwise():
        gemm_fp8_nt_groupwise(
            a,
            b,
            sa,
            sb,
            scale_major_mode=mode,
            scale_granularity_mnk=(1, 128, 128),
            out=out,
            backend="cutlass",
        )

    funcs = (
        {"pad_reuse": padded, "native_m1": native}
        if m == 1
        else {"blockscaled": native}
    )
    if args.compare_groupwise:
        funcs["groupwise"] = groupwise
    for fn in funcs.values():
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        torch.testing.assert_close(out.float(), ref, atol=1e-3, rtol=1e-2)
    raw = {
        name: {key: [] for key in ("eager_complete", "graph", "cold_graph")}
        for name in funcs
    }
    rng = random.Random(9556)
    for _ in range(args.rounds):
        order = list(funcs)
        rng.shuffle(order)
        for name in order:
            fn = funcs[name]
            for _ in range(args.samples):
                torch.cuda.synchronize()
                start = time.perf_counter_ns()
                fn()
                torch.cuda.synchronize()
                raw[name]["eager_complete"].append(
                    (time.perf_counter_ns() - start) / 1000
                )
            for cold, key in ((False, "graph"), (True, "cold_graph")):
                times = bench_gpu_time(
                    fn,
                    enable_cupti=True,
                    use_cuda_graph=True,
                    dry_run_iters=5,
                    repeat_iters=args.samples,
                    cold_l2_cache=cold,
                    l2_flush_size_mb=128 if cold else None,
                )
                raw[name][key].extend(t * 1000 for t in times)
    result = {
        "gpu": torch.cuda.get_device_name(),
        "cuda": torch.version.cuda,
        "cupti_python": cupti_version,
        "torch": torch.__version__,
        "flashinfer": flashinfer.__version__,
        "module": flashinfer.__file__,
        "shape_mnk": [m, n, k],
        "layout": mode,
        "contract": "FP8 inputs ready; BF16 out; full-call GPU span; warm/cold L2; eager includes sync",
        "median_us": {
            name: {key: statistics.median(vals) for key, vals in metrics.items()}
            for name, metrics in raw.items()
        },
        "raw_us": raw,
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
