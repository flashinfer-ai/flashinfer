"""
Copyright (c) 2024 by FlashInfer team.

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

import argparse
import pprint

import torch
from torch.nn import functional as F

import flashinfer.fused_moe as fused_moe
from flashinfer import fp4_quantize
from flashinfer.autotuner import AutoTuner, autotune, get_config_path
from flashinfer.testing.utils import bench_gpu_time_with_cudagraph

FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max


test_configs = [
    {
        "hidden_size": 7168,
        "num_experts": 256,
        "top_k": 8,
        "intermediate_size": 256,
    },
    {
        "hidden_size": 7168,
        "num_experts": 32,
        "top_k": 8,
        "intermediate_size": 2048,
    },
]


def compute_routing(
    router_logits: torch.Tensor, top_k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute routing weights and selected experts from router logits.

    Args:
        router_logits (torch.Tensor): Router logits of shape [batch_size, num_experts]
        top_k (int): Number of experts to route to per token

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - routing_weights: Expert weights of shape [batch_size, top_k]
            - selected_experts: Expert indices of shape [batch_size, top_k]
    """
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.float()
    return routing_weights, selected_experts


def bench_cutlass_fused_moe(
    batch_size,
    hidden_size,
    num_experts,
    top_k,
    intermediate_size,
    skip_autotune,
):
    torch.manual_seed(42)
    quant_blocksize = 16
    round_up = lambda x, y: (x + y - 1) // y * y
    e = num_experts
    m = batch_size
    n = intermediate_size
    k = hidden_size
    otype = torch.bfloat16
    wtype = torch.float8_e4m3fn
    w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=otype) / 10
    w1_cutlass = torch.cat((w1[:, n:, :], w1[:, :n, :]), dim=1).contiguous()

    sf_w1_2n = round_up(2 * n, 128)
    sf_w1_k = round_up(k // quant_blocksize, 4)
    w1_blockscale = torch.empty(
        (e, sf_w1_2n, sf_w1_k), device="cuda", dtype=torch.float8_e4m3fn
    )
    w1_blockscale_cutlass = torch.empty(
        (e, sf_w1_2n, sf_w1_k), device="cuda", dtype=torch.float8_e4m3fn
    )

    w2 = torch.randn((e, k, n), device="cuda", dtype=otype) / 10
    sf_w2_k = round_up(k, 128)
    sf_w2_n = round_up(n // quant_blocksize, 4)
    w2_blockscale = torch.empty(
        (e, sf_w2_k, sf_w2_n), device="cuda", dtype=torch.float8_e4m3fn
    )
    w1_q = torch.empty((e, 2 * n, k // 2), device="cuda", dtype=torch.uint8)
    w1_q_cutlass = torch.empty((e, 2 * n, k // 2), device="cuda", dtype=torch.uint8)
    w2_q = torch.empty((e, k, n // 2), device="cuda", dtype=torch.uint8)
    w1_gs = torch.empty((e,), device="cuda", dtype=torch.float32)
    w2_gs = torch.empty((e,), device="cuda", dtype=torch.float32)

    for expert in range(e):
        w1_amax = torch.abs(w1).max().to(torch.float32)
        w2_amax = torch.abs(w2).max().to(torch.float32)
        w1_gs[expert] = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w1_amax
        w2_gs[expert] = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w2_amax

        w1_q[expert], w1_blockscale[expert] = fp4_quantize(w1[expert], w1_gs[expert])

        w1_q_cutlass[expert], w1_blockscale_cutlass[expert] = fp4_quantize(
            w1_cutlass[expert], w1_gs[expert]
        )

        w2_q[expert], w2_blockscale[expert] = fp4_quantize(w2[expert], w2_gs[expert])

    x = torch.randn(m, k, dtype=otype).cuda()
    a1_gs = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / torch.abs(x).max().to(
        torch.float32
    ).cuda()
    a1_gs = torch.tensor(1.0, device="cuda", dtype=torch.float32)
    a2_gs = torch.tensor(1.0, device="cuda", dtype=torch.float32)
    router_logits = torch.randn(m, e, dtype=otype).cuda()
    routing_weights, selected_experts = compute_routing(router_logits, top_k)

    flash_output = torch.zeros_like(x)

    quant_scales = [
        a1_gs,
        w1_blockscale.view(torch.int32),
        1.0 / (a1_gs * w1_gs),
        a2_gs,
        w2_blockscale.view(torch.int32),
        1.0 / (a2_gs * w2_gs),
    ]
    hidden_states = x
    hidden_states, input_sf = fp4_quantize(x, a1_gs)

    # Warmup
    for _ in range(3):
        _ = fused_moe.cutlass_fused_moe(
            hidden_states,
            selected_experts.to(torch.int),
            routing_weights,
            w1_q.contiguous().view(torch.long),
            w2_q.contiguous().view(torch.long),
            otype,
            quant_scales=quant_scales,
            input_sf=input_sf,
            output=flash_output,
            tune_max_num_tokens=16384,
        )

    if not skip_autotune:
        with torch.inference_mode(), autotune(True):
            _ = fused_moe.cutlass_fused_moe(
                hidden_states,
                selected_experts.to(torch.int),
                routing_weights,
                w1_q.contiguous().view(torch.long),
                w2_q.contiguous().view(torch.long),
                otype,
                quant_scales=quant_scales,
                input_sf=input_sf,
                output=flash_output,
                tune_max_num_tokens=16384,
            )
    ms_list = bench_gpu_time_with_cudagraph(
        lambda: fused_moe.cutlass_fused_moe(
            hidden_states,
            selected_experts.to(torch.int),
            routing_weights,
            w1_q.contiguous().view(torch.long),
            w2_q.contiguous().view(torch.long),
            otype,
            quant_scales=quant_scales,
            input_sf=input_sf,
            output=flash_output,
        )
    )
    avg_ms = sum(ms_list) / len(ms_list)
    print(f"{'input':<15} {'weight1':<20} {'weight2':<20} {'time(ms)'}")
    print(
        f"{str(tuple(hidden_states.shape)):<15} {str(tuple(w1.shape)):<20} {str(tuple(w2.shape)):<20} {avg_ms:.3f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--update-config",
        action="store_true",
        help="Update the config file with the new profiling results",
    )
    parser.add_argument(
        "--num-tokens", type=int, default=32, help="Number of tokens to profile"
    )
    parser.add_argument("--skip-autotune", action="store_true", help="Skip autotuning")
    args = parser.parse_args()
    AutoTuner.get().clear_cache()

    for config in test_configs:
        bench_cutlass_fused_moe(
            args.num_tokens,
            config["hidden_size"],
            config["num_experts"],
            config["top_k"],
            config["intermediate_size"],
            args.skip_autotune,
        )

    configs = AutoTuner.get().profiling_cache
    if args.update_config and configs:
        # The original key contains a runner's hash in k[2] which might be different across machines.
        # So, we remove it for now. v[0] and v[1] are the runner id and the tactic.
        converted = {str((k[0], k[1], k[3])): (v[0], v[1]) for k, v in configs.items()}
        config_path = get_config_path(is_module=False)
        with open(config_path, "w") as f:
            f.write("best_configs = ")
            pprint.pprint(converted, stream=f)
        print(f"Saved the cache to {config_path}")
