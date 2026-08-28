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

TRT-LLM Gen MoE finalize benchmark.

Compares the MoE-LoRA FC2 combine flows downstream of the delta production
(the bgmv delta GEMMs are identical in both flows and excluded):

  separate: finalize (the fused launchers' do_finalize=True equivalent)
            followed by a full-output add of the [T, H] LoRA delta
  fused:    trtllm_gen_moe_finalize with the delta folded into the combine

plus the fused per-slot [T, top_k, H] delta variant that a decomposed
(combine=False) bgmv expand would produce.

Usage:
    FLASHINFER_DISABLE_VERSION_CHECK=1 python benchmarks/bench_trtllm_gen_moe_finalize.py
"""

import os

os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")

import argparse

import numpy as np
import torch

from flashinfer.fused_moe import trtllm_gen_moe_finalize
from flashinfer.testing.utils import bench_gpu_time


def make_case(num_tokens, top_k, hidden_size, device):
    num_expanded = num_tokens * top_k
    gemm2_output = (
        torch.randn(num_expanded, hidden_size, device=device).to(torch.bfloat16) * 0.1
    )
    expert_weights = torch.rand(num_tokens, top_k, dtype=torch.bfloat16, device=device)
    e2p = torch.randperm(num_expanded, dtype=torch.int32, device=device)
    delta_2d = (
        torch.randn(num_tokens, hidden_size, device=device).to(torch.bfloat16) * 0.01
    )
    delta_3d = (
        torch.randn(num_tokens, top_k, hidden_size, device=device).to(torch.bfloat16)
        * 0.01
    )
    out = torch.empty(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
    return gemm2_output, expert_weights, e2p, delta_2d, delta_3d, out


def median_us(fn):
    return float(np.median(bench_gpu_time(fn))) * 1e3


def run_case(num_tokens, top_k, hidden_size, device):
    gemm2_output, expert_weights, e2p, delta_2d, delta_3d, out = make_case(
        num_tokens, top_k, hidden_size, device
    )

    def finalize_only():
        trtllm_gen_moe_finalize(gemm2_output, expert_weights, e2p, out=out)

    def separate():  # current flow: finalize, then add the combined delta
        trtllm_gen_moe_finalize(gemm2_output, expert_weights, e2p, out=out)
        out.add_(delta_2d)

    def fused_2d():
        trtllm_gen_moe_finalize(
            gemm2_output, expert_weights, e2p, lora_delta=delta_2d, out=out
        )

    def fused_3d():
        trtllm_gen_moe_finalize(
            gemm2_output,
            expert_weights,
            e2p,
            lora_delta=delta_3d,
            lora_apply_expert_weights=True,
            out=out,
        )

    t_base = median_us(finalize_only)
    t_sep = median_us(separate)
    t_f2 = median_us(fused_2d)
    t_f3 = median_us(fused_3d)

    # Effective bandwidth of the fused 2D flow: read K permuted rows + delta,
    # write the output once.
    bytes_moved = 2 * (
        num_tokens * top_k * hidden_size  # gathered gemm2 rows
        + num_tokens * hidden_size  # delta_2d read
        + num_tokens * hidden_size  # output write
    )
    gbps = bytes_moved / (t_f2 * 1e-6) / 1e9

    print(
        f"| {num_tokens:>6} | {hidden_size:>6} | {top_k:>2} "
        f"| {t_base:9.1f} | {t_sep:12.1f} | {t_f2:8.1f} | {t_f3:8.1f} "
        f"| {t_sep / t_f2:7.2f}x | {gbps:7.0f} |"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-tokens",
        type=int,
        nargs="+",
        default=[1, 16, 64, 256, 1024, 4096, 8192],
    )
    parser.add_argument(
        "--hidden-size", type=int, nargs="+", default=[2880, 4096, 7168]
    )
    parser.add_argument("--top-k", type=int, nargs="+", default=[8])
    args = parser.parse_args()

    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(
        "| tokens | hidden |  k | base (us) | sep add (us) | f2d (us) | f3d (us) "
        "| speedup | GB/s    |"
    )
    print(
        "|--------|--------|----|-----------|--------------|----------|----------"
        "|---------|---------|"
    )
    for hidden_size in args.hidden_size:
        for top_k in args.top_k:
            for num_tokens in args.num_tokens:
                run_case(num_tokens, top_k, hidden_size, device)


if __name__ == "__main__":
    main()
