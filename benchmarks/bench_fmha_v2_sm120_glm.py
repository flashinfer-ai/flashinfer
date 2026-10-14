"""Measure GLM FP8 FMHA-v2 without quantization or allocation in the timed region."""

import argparse
import json
import statistics
from pathlib import Path

import torch
import tvm_ffi

import flashinfer
from flashinfer.jit import env
from flashinfer.prefill import get_trtllm_fmha_v2_sm120_module
from flashinfer.testing import bench_gpu_time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[1024, 2048, 5093])
    parser.add_argument("--heads", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--baseline-library", type=Path)
    parser.add_argument(
        "--jit-only", action="store_true", help="Ignore prebuilt AOT kernels"
    )
    args = parser.parse_args()
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("This benchmark targets SM120")
    if args.jit_only:
        env.FLASHINFER_AOT_DIR = Path("/nonexistent-fmha-v2-aot")
        env.FLASHINFER_AOT_PROVIDERS = ()
    modules = {"current": get_trtllm_fmha_v2_sm120_module()}
    if args.baseline_library:
        modules["baseline"] = tvm_ffi.load_module(str(args.baseline_library.resolve()))
    print(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(),
                "timer": "CUDA graph/events",
                "flashinfer_version": flashinfer.__version__,
                "flashinfer_path": flashinfer.__file__,
            }
        )
    )
    torch.manual_seed(42)
    for seq_len in args.seq_lens:
        shape = (args.batch_size, seq_len, args.heads, 256)
        operands = []
        scales = []
        for _ in range(3):
            x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
            scale = x.abs().amax().float() / 448
            operands.append((x.float() / scale).to(torch.float8_e4m3fn))
            scales.append(scale.reshape(1))
        scale_qk, scale_v = scales[0] * scales[1] / 16, scales[2]
        outputs = {
            name: torch.empty(shape, device="cuda", dtype=torch.bfloat16)
            for name in modules
        }

        def run(name):
            modules[name].run(
                *operands,
                outputs[name],
                None,
                scale_qk,
                scale_v,
                args.heads,
                256,
                seq_len,
                1.0,
                1.0,
                1.0,
                True,
                True,
                True,
            )

        for name in modules:
            run(name)
        if "baseline" in modules:
            torch.testing.assert_close(
                outputs["current"], outputs["baseline"], rtol=0, atol=0
            )
        samples = {name: [] for name in modules}
        for repeat in range(4):
            names = list(modules)[:: 1 if repeat % 2 == 0 else -1]
            for name in names:
                times = bench_gpu_time(
                    lambda name=name: run(name),
                    enable_cupti=False,
                    use_cuda_graph=True,
                    cold_l2_cache=False,
                    dry_run_iters=5,
                    repeat_iters=30,
                    num_iters_within_graph=20,
                )
                samples[name].append(statistics.median(times))
        print(
            json.dumps(
                {
                    "shape": shape,
                    "median_ms": {
                        name: statistics.median(times)
                        for name, times in samples.items()
                    },
                    "samples_ms": samples,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
