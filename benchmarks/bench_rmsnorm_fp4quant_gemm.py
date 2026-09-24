# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the FlashInfer project
"""Compare real RMSNorm/NVFP4 -> cuBLASLt chains with a pinned CuTe baseline.

This is a synthetic single-layer caller, not a model or serving integration.
The installed CuTe implementation is the candidate. Supply the unmodified
rmsnorm_fp4quant.py as --baseline-source and the built host bridge as --library.
Both producers share inputs, weights, a selected GEMM algorithm, workspace and
output buffers. Weight quantization and reference checks are outside timing.
Each event interval times actual producer-plus-GEMM CUDA Graph execution.
Use --allocate-outputs for a separate public-allocation control.
"""

import argparse
import importlib.util
import sys
import ctypes as C
import json
import statistics
import hashlib
from pathlib import Path
import torch
from flashinfer.cute_dsl.rmsnorm_fp4quant import rmsnorm_fp4quant
from flashinfer.quantization.fp4_quantization import fp4_quantize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument(
        "--allocate-outputs",
        action="store_true",
        help="Control: allocate producer outputs instead of sharing fixed buffers",
    )
    parser.add_argument("--shape", type=int, nargs=3, action="append")
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--scale", type=float, default=32.0)
    args = parser.parse_args()
    if args.baseline_source:
        spec = importlib.util.spec_from_file_location(
            "flashinfer.cute_dsl._chain_baseline", args.baseline_source
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        baseline_producer = module.rmsnorm_fp4quant

    lib = C.CDLL(str(args.library.resolve()))
    P, I = C.c_void_p, C.c_int
    for name, argtypes, ret in [
        ("gemm_create", [I, I, I, P, P], P),
        ("gemm_status", [P], I),
        ("gemm_count", [P], I),
        ("gemm_select", [P, I], None),
        ("gemm_algo_id", [P], I),
        ("gemm_run", [P, P, P, P, P, P, C.c_float, C.c_size_t], I),
        ("gemm_destroy", [P], None),
    ]:
        fn = getattr(lib, name)
        fn.argtypes = argtypes
        fn.restype = ret
    lookup = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6], device="cuda"
    )

    def decode(q, sf, m, k):
        raw = q.view(torch.uint8).reshape(m, k // 2)
        values = torch.stack(
            (lookup[(raw & 15).long()], lookup[(raw >> 4).long()]), -1
        ).reshape(m, k)
        mt, kt = (m + 127) // 128, (k // 16 + 3) // 4
        scales = (
            sf.view(torch.uint8)
            .flatten()[: mt * kt * 512]
            .view(torch.float8_e4m3fn)
            .float()
        )
        scales = (
            scales.reshape(mt, kt, 32, 4, 4)
            .permute(0, 3, 2, 1, 4)
            .reshape(mt * 128, kt * 4)[:m, : k // 16]
        )
        return (values.reshape(m, k // 16, 16) * scales[..., None]).reshape(m, k)

    torch.manual_seed(73)
    shapes = [
        (1, 4096, 4096),
        (24, 4096, 4096),
        (128, 4096, 4096),
        (1024, 4096, 4096),
        (16, 2048, 7168),
    ]
    if args.full:
        shapes = [
            (1, 4096, 4096),
            (2, 4096, 4096),
            (3, 4096, 4096),
            (7, 4096, 4096),
            (8, 4096, 4096),
            (24, 4096, 4096),
            (128, 4096, 4096),
            (512, 4096, 4096),
            (1, 2048, 7168),
            (4, 2048, 7168),
            (16, 2048, 7168),
            (256, 2048, 7168),
            (4096, 2048, 7168),
            (1024, 4096, 4096),
            (16384, 4096, 4096),
        ]
    if args.shape:
        shapes = args.shape
    if args.reverse:
        shapes.reverse()
    print(
        json.dumps(
            dict(
                candidate_sha256=hashlib.sha256(
                    Path(sys.modules[rmsnorm_fp4quant.__module__].__file__).read_bytes()
                ).hexdigest(),
                baseline_sha256=hashlib.sha256(
                    args.baseline_source.read_bytes()
                ).hexdigest(),
                bridge_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
                torch=torch.__version__,
                gpu=torch.cuda.get_device_name(),
                scope="full matrix" if args.full else "development complete chain",
                seed=args.seed,
                global_scale=args.scale,
                reverse=args.reverse,
                unroll=50,
                replays=100,
                rounds=6,
                preallocated=(not args.allocate_outputs),
            )
        ),
        flush=True,
    )
    for m, n, k in shapes:
        torch.manual_seed(args.seed + m + n + k)
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        gamma = (0.8 + 0.4 * torch.rand(k, device="cuda")).bfloat16()
        w = (torch.randn(n, k, device="cuda") / k**0.5).bfloat16()
        ga = torch.tensor([args.scale], device="cuda", dtype=torch.float32)
        gw = torch.tensor([32.0], device="cuda", dtype=torch.float32)
        wq, ws = fp4_quantize(
            w,
            global_scale=gw,
            sf_vec_size=16,
            sf_use_ue8m0=False,
            is_sf_swizzled_layout=True,
            backend="cuda",
        )
        aq, asc = rmsnorm_fp4quant(
            x, gamma, global_scale=ga, is_sf_swizzled_layout=True
        )
        fixed_buffers = (
            dict(y_fp4=aq, block_scale=asc) if (not args.allocate_outputs) else {}
        )
        y = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
        ctx = lib.gemm_create(m, n, k, asc.data_ptr(), ws.data_ptr())
        try:
            assert lib.gemm_status(ctx) == 0 and lib.gemm_count(ctx) > 0
            lib.gemm_select(ctx, 0)

            def chain():
                q, s = rmsnorm_fp4quant(
                    x,
                    gamma,
                    global_scale=ga,
                    is_sf_swizzled_layout=True,
                    **fixed_buffers,
                )
                status = lib.gemm_run(
                    ctx,
                    q.data_ptr(),
                    wq.data_ptr(),
                    s.data_ptr(),
                    ws.data_ptr(),
                    y.data_ptr(),
                    1 / (args.scale * 32),
                    torch.cuda.current_stream().cuda_stream,
                )
                assert status == 0, status
                return q, s

            aq, asc = chain()
            torch.cuda.synchronize()
            # Independent FP64 reference from the actual packed operands.
            ri = torch.randint(m, (256,), device="cuda")
            ci = torch.randint(n, (256,), device="cuda")
            xd = decode(aq, asc, m, k) / args.scale
            wd = decode(wq, ws, n, k) / 32
            expected = (xd[ri].double() * wd[ci].double()).sum(-1)
            error = ((y[ri, ci].double() - expected).norm() / expected.norm()).item()
            assert error < 0.01, error
            graph = torch.cuda.CUDAGraph()
            for _ in range(3):
                chain()
            torch.cuda.synchronize()
            with torch.cuda.graph(graph):
                chain()
            graph.replay()
            torch.cuda.synchronize()
            candidate_y = y.clone()
            bq, bs = baseline_producer(
                x, gamma, global_scale=ga, is_sf_swizzled_layout=True
            )
            quant_error = (
                (decode(aq, asc, m, k) - decode(bq, bs, m, k)).norm()
                / decode(bq, bs, m, k).norm()
            ).item()

            def baseline_chain():
                q, s = baseline_producer(
                    x,
                    gamma,
                    global_scale=ga,
                    is_sf_swizzled_layout=True,
                    **fixed_buffers,
                )
                status = lib.gemm_run(
                    ctx,
                    q.data_ptr(),
                    wq.data_ptr(),
                    s.data_ptr(),
                    ws.data_ptr(),
                    y.data_ptr(),
                    1 / (args.scale * 32),
                    torch.cuda.current_stream().cuda_stream,
                )
                assert status == 0, status

            for _ in range(3):
                baseline_chain()
            torch.cuda.synchronize()
            chain_error = (
                (y.float() - candidate_y.float()).norm() / y.float().norm()
            ).item()
            assert quant_error < 0.005 and chain_error < 0.005, (
                quant_error,
                chain_error,
            )
            # Capture repeated actual chains to amortize host replay overhead.
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(50):
                    chain()
            bg = torch.cuda.CUDAGraph()
            with torch.cuda.graph(bg):
                for _ in range(50):
                    baseline_chain()

            def time_graph(g):
                a = torch.cuda.Event(enable_timing=True)
                b = torch.cuda.Event(enable_timing=True)
                a.record()
                for _ in range(100):
                    g.replay()
                b.record()
                b.synchronize()
                return a.elapsed_time(b) * 10 / 50

            samples = {"baseline": [], "candidate": []}
            for round_id in range(6):
                order = [("baseline", bg), ("candidate", graph)]
                if round_id % 2:
                    order.reverse()
                for name, g in order:
                    samples[name].append(time_graph(g))
            base = statistics.median(samples["baseline"])
            new = statistics.median(samples["candidate"])
            print(
                json.dumps(
                    dict(
                        shape=[m, n, k],
                        algorithm=lib.gemm_algo_id(ctx),
                        relative_rms=error,
                        quant_relative_rms_vs_base=quant_error,
                        chain_relative_rms_vs_base=chain_error,
                        baseline_us=base,
                        candidate_us=new,
                        speedup=base / new,
                        samples_us=samples,
                    )
                ),
                flush=True,
            )
        finally:
            lib.gemm_destroy(ctx)
    print("FULL_CHAIN_COMPARISON_PASS", flush=True)


if __name__ == "__main__":
    main()
