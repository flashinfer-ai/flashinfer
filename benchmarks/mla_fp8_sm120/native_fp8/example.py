"""Runnable native FP8 decode example; preserve existing GPU processes."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from bench_fp8_kv import wait_idle
from kv_bench_kernels import quantize_rows
from native_fp8.wrapper import NativeMLA


def main():
    wait_idle(700 * 1024**2)
    workspace = torch.empty(128 * 1024**2, device="cuda", dtype=torch.uint8)
    q = torch.randn(1, 20, 576, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(64, 16, 576, device="cuda", dtype=torch.bfloat16)
    kv8, scales = quantize_rows(kv)
    runner = NativeMLA(
        workspace,
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([0, 64], dtype=torch.int32),
        torch.arange(64, dtype=torch.int32),
        torch.tensor([1024], dtype=torch.int32),
        heads=20,
        page_size=16,
    )
    output = runner.run(q, kv8, scales)
    torch.cuda.synchronize()
    print(
        "Output:", output.shape, output.dtype, "finite:", output.isfinite().all().item()
    )


if __name__ == "__main__":
    main()
