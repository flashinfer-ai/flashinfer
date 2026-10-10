"""Runnable native FP8 decode/prefill example; preserve existing GPU processes."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from bench_fp8_kv import wait_idle
from kv_bench_kernels import quantize_rows
from native_fp8.wrapper import NativeMLA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prefill", action="store_true", help="Run 2048-token causal prefill."
    )
    args = parser.parse_args()
    wait_idle(700 * 1024**2)
    qlen, klen = (2048, 2048) if args.prefill else (1, 1024)
    config = (
        dict(bm=64, bn=64, groups=4, workers=110, fused=False, share_p=True)
        if args.prefill
        else {}
    )
    workspace = torch.empty(128 * 1024**2, device="cuda", dtype=torch.uint8)
    q = torch.randn(qlen, 20, 576, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(klen // 16, 16, 576, device="cuda", dtype=torch.bfloat16)
    kv8, scales = quantize_rows(kv)
    runner = NativeMLA(
        workspace,
        torch.tensor([0, qlen], dtype=torch.int32),
        torch.tensor([0, klen // 16], dtype=torch.int32),
        torch.arange(klen // 16, dtype=torch.int32),
        torch.tensor([klen], dtype=torch.int32),
        heads=20,
        page_size=16,
        causal=args.prefill,
        **config,
    )
    output = runner.run(q, kv8, scales)
    torch.cuda.synchronize()
    print(
        "Output:", output.shape, output.dtype, "finite:", output.isfinite().all().item()
    )


if __name__ == "__main__":
    main()
