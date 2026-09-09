"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

TRT-LLM Gen MoE activation-gather benchmark.

Compares the two ways of bringing the permuted post-activation FC1 output back
into expanded (token, slot) order:

  torch:  the eager gather bgmv_moe_gemm2_lora_delta used to run --
          zeros([P, I]) + index_select + masked scatter
  fused:  trtllm_gen_moe_gather_activation

over decode-shaped (T 1..256) and prefill-shaped (T 1024..8192) batches.

Usage:
    FLASHINFER_DISABLE_VERSION_CHECK=1 python benchmarks/bench_trtllm_gen_moe_gather_activation.py
"""

import os

os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")

import argparse

import numpy as np
import torch

from flashinfer.fused_moe import trtllm_gen_moe_gather_activation
from flashinfer.testing.utils import bench_gpu_time


def make_case(num_tokens, top_k, intermediate_size, device):
    num_slots = num_tokens * top_k
    # Routing pads the permuted buffer past the expanded slot count.
    num_padded = num_slots + 128
    activation_output = (
        torch.randn(num_padded, intermediate_size, device=device).to(torch.bfloat16)
        * 0.1
    )
    e2p = torch.randperm(num_padded, dtype=torch.int32, device=device)[:num_slots]
    # ~1/8 of the slots are routed outside the local expert shard.
    e2p = torch.where(torch.rand(num_slots, device=device) < 0.125, -1, e2p)
    out = torch.empty(
        num_tokens, top_k, intermediate_size, dtype=torch.bfloat16, device=device
    )
    return activation_output, e2p, out


def median_us(fn):
    return float(np.median(bench_gpu_time(fn))) * 1e3


def run_case(num_tokens, top_k, intermediate_size, device):
    activation_output, e2p, out = make_case(
        num_tokens, top_k, intermediate_size, device
    )
    num_slots = num_tokens * top_k

    def torch_gather():
        perm = e2p.reshape(-1).to(torch.int64)
        valid = perm >= 0
        gathered = torch.zeros(
            num_slots, intermediate_size, dtype=torch.bfloat16, device=device
        )
        gathered[valid] = activation_output[perm[valid]]
        return gathered

    def fused_gather():
        trtllm_gen_moe_gather_activation(activation_output, e2p, top_k, out=out)

    t_torch = median_us(torch_gather)
    t_fused = median_us(fused_gather)

    # Read the gathered rows once, write the output once.
    bytes_moved = 2 * 2 * num_slots * intermediate_size
    gbps = bytes_moved / (t_fused * 1e-6) / 1e9

    print(
        f"| {num_tokens:>6} | {intermediate_size:>6} | {top_k:>2} "
        f"| {t_torch:9.1f} | {t_fused:9.1f} | {t_torch / t_fused:7.2f}x | {gbps:7.0f} |"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-tokens",
        type=int,
        nargs="+",
        default=[1, 8, 32, 64, 256, 1024, 4096, 8192],
    )
    parser.add_argument(
        "--intermediate-size", type=int, nargs="+", default=[512, 1024, 2048]
    )
    parser.add_argument("--top-k", type=int, nargs="+", default=[8])
    args = parser.parse_args()

    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print("| tokens |  inter |  k | torch (us) | fused (us) | speedup | GB/s    |")
    print("|--------|--------|----|------------|------------|---------|---------|")
    for intermediate_size in args.intermediate_size:
        for top_k in args.top_k:
            for num_tokens in args.num_tokens:
                run_case(num_tokens, top_k, intermediate_size, device)


if __name__ == "__main__":
    main()
