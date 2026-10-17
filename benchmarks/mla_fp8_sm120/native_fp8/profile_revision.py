"""Two MLA launches for Nsight Compute: warm up once, then profile once.

Use --launch-skip 1 --launch-count 1 and filter BatchMLAPagedAttentionFP8SM120.
Keep --clock-control none --cache-control none on a shared device.
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from bench_fp8_kv import wait_idle
from kv_bench_kernels import quantize_rows
from native_fp8.bench_revision import BASELINE_COMMIT, baseline_module
from native_fp8.wrapper import NativeMLA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous", action="store_true")
    parser.add_argument("--groups", type=int, choices=(2, 4), default=4)
    parser.add_argument("--size", type=int, default=2048)
    args = parser.parse_args()
    wait_idle(1300 * 1024**2)
    torch.manual_seed(47)
    size = args.size
    workspace = torch.empty(128 * 1024**2, device="cuda", dtype=torch.uint8)
    q = torch.randn(size, 20, 576, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn((size + 15) // 16, 16, 576, device="cuda", dtype=torch.bfloat16)
    q8, qs = quantize_rows(q)
    kv8, ks = quantize_rows(kv)
    cls = baseline_module(BASELINE_COMMIT).NativeMLA if args.previous else NativeMLA
    config = {} if args.previous else dict(share_p=True)
    runner = cls(
        workspace,
        torch.tensor([0, size], dtype=torch.int32),
        torch.tensor([0, kv.shape[0]], dtype=torch.int32),
        torch.arange(kv.shape[0], dtype=torch.int32),
        torch.tensor([size], dtype=torch.int32),
        heads=20,
        page_size=16,
        causal=True,
        bm=64,
        bn=64,
        groups=args.groups,
        workers=110,
        fused=False,
        **config,
    )
    runner.run_prequantized(q8, kv8, qs, ks)
    torch.cuda.synchronize()
    wait_idle()
    runner.run_prequantized(q8, kv8, qs, ks)
    torch.cuda.synchronize()
    print("PROFILE DONE", runner.config, runner.attributes)


if __name__ == "__main__":
    main()
