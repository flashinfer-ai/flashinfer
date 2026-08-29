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
"""

"""Cold-L2 CUPTI benchmark for the prepared SM100 BGMV MoE backend."""

import argparse
import json
import math
import statistics
import warnings

import torch

from flashinfer.fused_moe.bgmv_moe import (
    bgmv_moe_expand,
    bgmv_moe_shrink,
    fill_w_ptr,
    prepare_bgmv_moe,
)
from flashinfer.testing.utils import bench_gpu_time_with_cupti


SHAPES = [
    (hidden_size, num_tokens)
    for hidden_size in (3072, 2688)
    for num_tokens in (1, 4, 8, 32, 256, 512, 1024)
]


def _make_inputs(hidden_size: int, num_tokens: int):
    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16
    rank = 32
    num_experts = 128
    num_loras = 8
    top_k = 2
    num_pairs = num_tokens * top_k
    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device) * 0.1
    lora_a = (
        torch.randn(
            num_loras,
            num_experts,
            rank,
            hidden_size,
            dtype=dtype,
            device=device,
        )
        * 0.01
    )
    lora_b = (
        torch.randn(
            num_loras,
            num_experts,
            hidden_size,
            rank,
            dtype=dtype,
            device=device,
        )
        * 0.01
    )
    sorted_token_ids = torch.arange(
        num_tokens, dtype=torch.int64, device=device
    ).repeat_interleave(top_k)
    expert_ids = torch.randint(
        0, num_experts, (num_pairs,), dtype=torch.int64, device=device
    )
    lora_indices = torch.randint(
        0, num_loras, (num_tokens,), dtype=torch.int64, device=device
    )
    topk_weights = torch.softmax(
        torch.randn(num_tokens, top_k, dtype=torch.float32, device=device), dim=-1
    ).reshape(-1)
    return (
        x,
        [lora_a],
        [lora_b],
        sorted_token_ids,
        expert_ids,
        lora_indices,
        topk_weights,
        num_experts,
    )


def _cupti_median_us(fn, repeat_time_ms: int) -> float:
    # The testing helper warns before silently falling back to CUDA Events.
    # Treat that warning as an error so reportable rows remain CUPTI-only.
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        samples_ms = bench_gpu_time_with_cupti(
            fn,
            dry_run_time_ms=100,
            repeat_time_ms=repeat_time_ms,
            cold_l2_cache=True,
            use_cuda_graph=False,
        )
    return 1000.0 * float(statistics.median(samples_ms))


def _run_shape(hidden_size: int, num_tokens: int, repeat_time_ms: int):
    inputs = _make_inputs(hidden_size, num_tokens)
    (
        x,
        lora_a_weights,
        lora_b_weights,
        sorted_token_ids,
        expert_ids,
        lora_indices,
        topk_weights,
        num_experts,
    ) = inputs
    num_pairs = int(sorted_token_ids.numel())

    baseline_shrink = torch.empty(1, num_pairs, 32, dtype=x.dtype, device=x.device)
    baseline_output = torch.empty(
        num_tokens, hidden_size, dtype=torch.float32, device=x.device
    )
    w_ptr_a = torch.empty(1, num_experts, dtype=torch.int64, device=x.device)
    w_ptr_b = torch.empty(1, num_experts, dtype=torch.int64, device=x.device)
    lora_stride_a = fill_w_ptr(w_ptr_a, lora_a_weights[0], num_experts, 0)
    lora_stride_b = fill_w_ptr(w_ptr_b, lora_b_weights[0], num_experts, 0)
    slice_start_loc = torch.zeros(1, dtype=torch.int64, device=x.device)

    def baseline():
        baseline_shrink.zero_()
        baseline_output.zero_()
        bgmv_moe_shrink(
            baseline_shrink,
            x,
            w_ptr_a,
            sorted_token_ids,
            expert_ids,
            lora_indices,
            lora_stride_a,
        )
        bgmv_moe_expand(
            baseline_output,
            baseline_shrink,
            w_ptr_b,
            sorted_token_ids,
            expert_ids,
            topk_weights,
            lora_indices,
            slice_start_loc,
            [hidden_size],
            lora_stride_b,
        )

    plan = prepare_bgmv_moe(*inputs, backend="blackwell")
    candidate_output = plan.run()
    baseline()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        candidate_output,
        baseline_output,
        atol=1e-2,
        rtol=1e-2,
    )

    baseline_us = _cupti_median_us(baseline, repeat_time_ms)
    candidate_us = _cupti_median_us(plan.run, repeat_time_ms)
    return {
        "hidden_size": hidden_size,
        "num_tokens": num_tokens,
        "rank": 32,
        "num_experts": 128,
        "top_k": 2,
        "num_loras": 8,
        "dtype": "bfloat16",
        "baseline_us": baseline_us,
        "candidate_us": candidate_us,
        "speedup": baseline_us / candidate_us,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat-time-ms", type=int, default=1000)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError("this benchmark requires an exact SM100 CUDA device")

    rows = [
        _run_shape(hidden_size, num_tokens, args.repeat_time_ms)
        for hidden_size, num_tokens in SHAPES
    ]
    baseline_geomean_us = math.exp(
        sum(math.log(row["baseline_us"]) for row in rows) / len(rows)
    )
    candidate_geomean_us = math.exp(
        sum(math.log(row["candidate_us"]) for row in rows) / len(rows)
    )
    report = {
        "gpu": torch.cuda.get_device_name(),
        "timing": "CUPTI GPU activity span, cold L2",
        "scope": "zero + shrink + expand",
        "rows": rows,
        "baseline_geomean_us": baseline_geomean_us,
        "candidate_geomean_us": candidate_geomean_us,
        "geomean_speedup": baseline_geomean_us / candidate_geomean_us,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
