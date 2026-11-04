"""Reproduce public FP8/native-NVFP4 prefill timings with explicit fusion boundaries.

Run from the FlashInfer repository with PYTHONPATH=. and CUTE_PREFILL_ARTIFACT_DIR
set to an existing output directory. --baseline-module accepts an archived
FlashInfer sparse-MLA module with the same execution-plan ABI.
"""

import argparse
import json
import math

import torch

from benchmarks.sparse_mla_sm120_cute_prefill.evaluate import ROOT, paired
from tests.attention.dspark_mla_utils import dspark_fixture, numerical_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=("v41_fp8", "nvfp4"), default="v41_fp8")
    parser.add_argument("--baseline-module")
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=24)
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    f = dspark_fixture(
        1,
        "prefill",
        args.format,
        heads=args.heads,
        prefill_tokens=args.tokens,
        prefill_cp_size=4,
        history_tokens=0,
    )
    if args.format == "nvfp4":
        from tests.attention.test_sparse_mla_sm120_nvfp4_prefill_mega import (
            prefill_calls,
        )

        mega, unfused, _ = prefill_calls(f)
    else:

        def mega():
            return f["wrapper"].run_mega(
                f["q"],
                f["caches"][0],
                f["indices"][0],
                f["out"],
                512**-0.5,
                positions=f["positions"],
                cos_sin=f["cs"],
                output_scales=f["scales"],
                topk_length=f["lengths"][0],
                attn_sink=f["sink"],
                extra_kv_cache=f["caches"][1],
                extra_indices=f["indices"][1],
                extra_topk_length=f["lengths"][1],
                return_lse=True,
                lse_scale=math.log(2),
            )

        unfused = f["baseline"]
    calls = {"mega": mega, "unfused": unfused}
    if args.baseline_module:
        import tvm_ffi

        old = tvm_ffi.load_module(args.baseline_module)
        if args.format == "nvfp4":
            _, calls["original_unfused"], _ = prefill_calls(f, old)
        else:
            from benchmarks.bench_sparse_mla_sm120_mega import use_baseline_module

            use_baseline_module(f, old)
            calls["original_unfused"] = f["baseline"]
    lse = mega()
    rows = torch.linspace(0, f["q"].shape[0] - 1, 16, device="cuda").long()
    metrics = numerical_metrics(f, lse, rows)
    limit = 0.20 if args.format == "nvfp4" else 0.07
    lse_limit = 0.20 if args.format == "nvfp4" else 0.06
    assert metrics["finite_output"] and metrics["relative_l2"] < limit, metrics
    assert metrics["lse_max_abs"] < lse_limit, metrics
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as trace:
        mega()
        unfused()
    kernels = [
        event.name
        for event in trace.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
    ]
    result = {
        "gpu": torch.cuda.get_device_name(),
        "format": args.format,
        "q": list(f["q"].shape),
        "kv": [list(t.shape) for t in f["caches"]],
        "metrics": metrics,
        "kernels": kernels,
        "ms": paired(calls, args.repeats),
    }
    (ROOT / "integrated.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
