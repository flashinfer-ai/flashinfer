# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Warm-L2 CUDA-graph experiment, not an end-to-end serving benchmark.

Optionally include the existing TRT-LLM padding operation. No timings include
JIT compilation. Inputs/outputs are preallocated except that existing TRT op's
outputs, which CUDA graph capture allocates once. Router baseline is torch.addmm,
not TRT-LLM's shape-dependent router backend; serving speedups cannot be inferred.
"""

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch

from flashinfer.cute_dsl import add_rmsnorm_fp4quant
from flashinfer.cute_dsl.add_rmsnorm_router_fp4quant import (
    _experimental_add_rmsnorm_router_fp4quant,
)
from flashinfer.testing import bench_gpu_time


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 7, 32, 228, 7700])
    parser.add_argument("--hidden", type=int, default=2880)
    parser.add_argument("--physical", type=int, default=3072)
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--trtllm-padding", action="store_true")
    parser.add_argument("--profile-dir", type=Path)
    args = parser.parse_args()
    if args.trtllm_padding:
        import tensorrt_llm  # noqa: F401 - registers the padding op
    print(
        f"# GPU={torch.cuda.get_device_name()}, torch={torch.__version__}, graph, warm-L2",
        flush=True,
    )
    print(
        "# Zero input keeps the in-place random residual stationary across replay; no data-dependent shortcut."
    )
    writer = csv.writer(sys.stdout)
    writer.writerow(["tokens", "variant", "p50_us", "p90_us"])
    rng = random.Random(42)
    for m in args.tokens:
        torch.manual_seed(42)
        h, k, e = args.hidden, args.physical, args.experts
        residual = torch.randn((m, h), dtype=torch.bfloat16, device="cuda")
        x = torch.zeros_like(residual)
        weight = torch.randn_like(residual[0])
        router = torch.randn((e, h), dtype=x.dtype, device=x.device) / h**0.5
        bias = torch.randn((e,), dtype=x.dtype, device=x.device)
        scale = torch.tensor([3.7], device=x.device)
        normalized = torch.empty_like(x)
        fp4 = torch.empty((m, k // 2), dtype=torch.float4_e2m1fn_x2, device=x.device)
        sf = torch.empty((m, k // 16), dtype=torch.float8_e4m3fn, device=x.device)
        compact_fp4 = torch.empty((m, h // 2), dtype=fp4.dtype, device=x.device)
        compact_sf = torch.empty((m, h // 16), dtype=sf.dtype, device=x.device)
        logits = torch.empty((m, e), dtype=x.dtype, device=x.device)

        def direct_pad():
            add_rmsnorm_fp4quant(
                x,
                residual,
                weight,
                global_scale=scale,
                y_out=normalized,
                output_hidden_size=k,
                y_fp4=fp4,
                block_scale=sf,
                enable_pdl=False,
            )
            torch.addmm(bias, normalized, router.T, out=logits)
            return fp4, sf, logits

        def all_fused():
            return _experimental_add_rmsnorm_router_fp4quant(
                x,
                residual,
                weight,
                router,
                router_bias=bias,
                global_scale=scale,
                output_hidden_size=k,
                y_fp4=fp4,
                block_scale=sf,
                router_logits=logits,
                enable_pdl=False,
            )

        def separate_pad():
            add_rmsnorm_fp4quant(
                x,
                residual,
                weight,
                global_scale=scale,
                y_out=normalized,
                y_fp4=compact_fp4,
                block_scale=compact_sf,
                enable_pdl=False,
            )
            padded, scales = torch.ops.trtllm.pad_nvfp4_activation(
                compact_fp4,
                compact_sf,
                k,
            )
            torch.addmm(bias, normalized, router.T, out=logits)
            return padded, scales, logits

        variants = {
            "direct_pad+router_gemm": direct_pad,
            "experimental_all_fused": all_fused,
        }
        if args.trtllm_padding:
            variants["separate_pad+router_gemm"] = separate_pad
        reference = [t.clone() for t in direct_pad()]
        for name, fn in variants.items():
            actual = fn()
            for i in (0, 1):
                torch.testing.assert_close(
                    actual[i].view(torch.uint8),
                    reference[i].view(torch.uint8),
                    rtol=0,
                    atol=0,
                )
            # Smoke gate only. Detailed FP64 forward-error tests live in tests/norm.
            torch.testing.assert_close(actual[2], reference[2], rtol=0.01, atol=0.01)
            if args.profile_dir:
                args.profile_dir.mkdir(parents=True, exist_ok=True)
                torch.cuda.synchronize()
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                ) as prof:
                    fn()
                    torch.cuda.synchronize()
                prof.export_chrome_trace(str(args.profile_dir / f"{name}-m{m}.json"))
                events = [
                    event
                    for event in prof.events()
                    if event.device_type == torch.autograd.DeviceType.CUDA
                ]
                print(
                    f"# {name} M={m} device_events={len(events)}: {[event.name for event in events]}",
                    flush=True,
                )
                if name == "experimental_all_fused":
                    assert len(events) == 1, "Expected exactly one device kernel"
        times = {name: [] for name in variants}
        for _ in range(args.trials):
            order = list(variants)
            rng.shuffle(order)
            for name in order:
                times[name].extend(
                    bench_gpu_time(
                        variants[name],
                        use_cuda_graph=True,
                        cold_l2_cache=False,
                        dry_run_iters=30,
                        repeat_iters=args.repeats,
                        num_iters_within_graph=10,
                    )
                )
        for name, samples in times.items():
            writer.writerow(
                [
                    m,
                    name,
                    f"{np.median(samples) * 1000:.3f}",
                    f"{np.percentile(samples, 90) * 1000:.3f}",
                ]
            )
        sys.stdout.flush()


if __name__ == "__main__":
    main()
