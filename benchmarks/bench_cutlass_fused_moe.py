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
import json
import os
import sys
import time

import torch
from torch.nn import functional as F
from triton.testing import do_bench

import flashinfer
import flashinfer.fused_moe as fused_moe
from flashinfer import fp4_quantize

# ------------------------------------------------------------------------------------------------

class empty_suppress:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


class suppress_stdout_stderr:
    def __enter__(self):
        self.outnull_file = open(os.devnull, 'w')
        self.errnull_file = open(os.devnull, 'w')

        self.old_stdout_fileno_undup = sys.stdout.fileno()
        self.old_stderr_fileno_undup = sys.stderr.fileno()

        self.old_stdout_fileno = os.dup(sys.stdout.fileno())
        self.old_stderr_fileno = os.dup(sys.stderr.fileno())

        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr

        os.dup2(self.outnull_file.fileno(), self.old_stdout_fileno_undup)
        os.dup2(self.errnull_file.fileno(), self.old_stderr_fileno_undup)

        sys.stdout = self.outnull_file
        sys.stderr = self.errnull_file
        return self

    def __exit__(self, *_):
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

        os.dup2(self.old_stdout_fileno, self.old_stdout_fileno_undup)
        os.dup2(self.old_stderr_fileno, self.old_stderr_fileno_undup)

        os.close(self.old_stdout_fileno)
        os.close(self.old_stderr_fileno)

        self.outnull_file.close()
        self.errnull_file.close()


def bench_kineto(fn, kernel_names, num_tests: int = 30,
                 suppress_kineto_output: bool = False,
                 trace_path: str = None, flush_l2: bool = True,
                 with_multiple_kernels: bool = False):
    # Conflict with Nsight Systems
    using_nsys = int(os.environ.get('DG_NSYS_PROFILING', 0))

    # By default, flush L2 with an excessive 8GB memset to give the GPU some (literal) chill time without full idle
    flush_l2_size = int(8e9 // 4)

    # For some auto-tuning kernels with prints
    fn()

    # Profile
    suppress = suppress_stdout_stderr if suppress_kineto_output and not using_nsys else empty_suppress
    with suppress():
        schedule = torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1) if not using_nsys else None
        profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA], schedule=schedule) if not using_nsys else empty_suppress()
        with profiler:
            for i in range(2):
                for _ in range(num_tests):
                    if flush_l2:
                        torch.empty(flush_l2_size, dtype=torch.int, device='cuda').zero_()
                    fn()

                if not using_nsys:
                    profiler.step()

    # Return 1 if using Nsight Systems
    if using_nsys:
        return 1

    # Parse the profiling table
    assert isinstance(kernel_names, str) or isinstance(kernel_names, tuple)
    is_tuple = isinstance(kernel_names, tuple)
    prof_lines = profiler.key_averages().table(sort_by='cuda_time_total', max_name_column_width=100).split('\n')
    print(f"prof_lines=\n" + "\n".join(prof_lines))
    kernel_names = (kernel_names, ) if isinstance(kernel_names, str) else kernel_names
    assert all([isinstance(name, str) for name in kernel_names])
    if not with_multiple_kernels:
        for name in kernel_names:
            assert sum([name in line for line in prof_lines]) == 1, f'Errors of the kernel {name} in the profiling table'

    # Save chrome traces
    if trace_path is not None:
        print(f"export_chrome_trace to {trace_path=}")
        profiler.export_chrome_trace(trace_path)

    # Return average kernel times
    units = {'ms': 1e3, 'us': 1e6}
    kernel_times = []
    for name in kernel_names:
        total_time = 0
        total_num = 0
        for line in prof_lines:
            if name in line:
                time_str = line.split()[-2]
                num_str = line.split()[-1]
                for unit, scale in units.items():
                    if unit in time_str:
                        total_time += float(time_str.replace(unit, '')) / scale * int(num_str)
                        total_num += int(num_str)
                        break
        kernel_times.append(total_time / total_num)

    return tuple(kernel_times) if is_tuple else kernel_times[0]


# ------------------------------------------------------------------------------------------------

BATCH_SIZES = [
    # TODO more
    # 1,
    # 2,
    # 4,
    # 8,
    # 16,
    # 24,
    # 32,
    # 48,
    # 64,
    # 96,
    # 128,
    # 256,
    # 384, # NOTE ADD
    # 512,
    768, # NOTE ADD
    # 1024,
    # 1536,
    # 2048,
    # 3072,
    # 4096,
]

configs = []
hidden_size = 7168
num_experts = [32, 256]
top_k = [8]
intermediate_size = [256, 2048]
FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max
FP8_DTYPE = torch.float8_e4m3fn

test_configs = [
    # NOTE MODIFIED ADD
    *[
        {
            "hidden_size": 7168,
            "num_experts": num_experts,
            "top_k": 8,
            "intermediate_size": 2048,
        }
        for num_experts in [
            # TODO more
            # 288 // 1,
            # 288 // 2,
            # 288 // 4,
            # 288 // 8,
            # 288 // 16,
            288 // 32,
            # 288 // 48,
            # 288 // 72,
        ]
    ],

    # --- old ---
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
    repeats = 3
    from flashinfer.autotuner import AutoTuner, autotune

    AutoTuner.get().clear_cache()
    with torch.inference_mode(), autotune():
        for _ in range(2):
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
            )
    # NOTE MODIFIED
    # ms = do_bench(
    #     lambda: fused_moe.cutlass_fused_moe(
    #         hidden_states,
    #         selected_experts.to(torch.int),
    #         routing_weights,
    #         w1_q.contiguous().view(torch.long),
    #         w2_q.contiguous().view(torch.long),
    #         otype,
    #         quant_scales=quant_scales,
    #         input_sf=input_sf,
    #         output=flash_output,
    #     )
    # )
    trace_dir = os.environ.get("BENCH_KINETO_TRACE_DIR")
    bench_kineto(
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
        ),
        kernel_names="what",
        trace_path=f"{trace_dir}/{time.time()}.json.gz" if trace_dir else None,
    )

    # NOTE MODIFIED
    print(f"MAIN_OUTPUT=" + json.dumps(dict(
        batch_size=batch_size,
        num_experts=num_experts,
        top_k=top_k,
        intermediate_size=intermediate_size,
        execution_time_us=ms * 1000,
    )))
    # print(
    #     f"batch_size={batch_size}, num_experts={num_experts}, top_k={top_k}, intermediate_size={intermediate_size}"
    # )
    # print(f"execution time: {ms}ms")


if __name__ == "__main__":
    for config in test_configs:
        hidden_size = config["hidden_size"]
        num_experts = config["num_experts"]
        top_k = config["top_k"]
        intermediate_size = config["intermediate_size"]
        for batch_size in BATCH_SIZES:
            bench_cutlass_fused_moe(
                batch_size,
                hidden_size,
                num_experts,
                top_k,
                intermediate_size,
            )
