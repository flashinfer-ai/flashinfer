# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Benchmark the contributed FuseMoE Blackwell kernel against
flashinfer's existing TRT-LLM FP8 block-scale MoE.

Both kernels are run on the DeepSeek-V3 EP=8 shape that the contributed
kernel was tuned for:
    hidden=7168, intermediate=2048, num_experts=256, num_local_experts=32,
    top_k=8, n_group=8, topk_group=4, routed_scaling_factor=2.5.
"""

import argparse
import statistics
import subprocess

import torch

import flashinfer
from flashinfer.testing.utils import bench_gpu_time


def _check_gpu_idle(strict: bool) -> None:
    """Warn (or error in --strict mode) if the selected GPU has other tenants.

    Co-tenanted GPUs deliver flaky benchmark numbers — most often the trtllm
    reference ends up with high tail latency from kernel-queue contention,
    which biases the speedup ratio. Run on an idle GPU for stable numbers.
    """
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={torch.cuda.current_device()}",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return  # nvidia-smi unavailable; skip the check
    util_str, mem_str = (s.strip() for s in out.split(",", 1))
    util = int(util_str)
    mem_mib = int(mem_str)
    # Allow up to ~5GB resident (a sibling python process holding context)
    # and up to 10% util. Anything above that is co-tenant noise.
    if util > 10 or mem_mib > 5000:
        msg = (
            f"GPU {torch.cuda.current_device()} not idle: util={util}%, "
            f"mem_used={mem_mib} MiB. Benchmark numbers will be unreliable."
        )
        if strict:
            raise RuntimeError(
                msg + " Re-run with a clean GPU or pass --no-idle-check."
            )
        print(f"[WARNING] {msg}")


HIDDEN = 7168
INTERMEDIATE = 2048
NUM_EXPERTS = 256
NUM_LOCAL_EXPERTS = 32
TOP_K = 8
N_GROUP = 8
TOPK_GROUP = 4
ROUTED_SCALING = 2.5
BLOCK = 128

DEFAULT_TOKEN_COUNTS = [1, 8, 32, 128, 512, 1024, 2048, 4096]


def make_inputs(num_tokens: int, device="cuda", seed: int = 0):
    g = torch.Generator(device=device).manual_seed(seed)

    routing_logits = torch.randn(
        num_tokens, NUM_EXPERTS, device=device, dtype=torch.float32, generator=g
    )
    routing_bias = (
        torch.randn(NUM_EXPERTS, device=device, dtype=torch.bfloat16, generator=g)
        * 0.01
    )

    # Hidden states: produce zero FP8 with unit scales — this gives a
    # numerically valid input regardless of any saturation behaviour and
    # makes the two kernels comparable for throughput.
    hidden_states = torch.zeros(
        num_tokens, HIDDEN, device=device, dtype=torch.float8_e4m3fn
    )
    hidden_states_scale = torch.ones(
        HIDDEN // BLOCK, num_tokens, device=device, dtype=torch.float32
    )

    gemm1_weights = torch.zeros(
        NUM_LOCAL_EXPERTS,
        2 * INTERMEDIATE,
        HIDDEN,
        device=device,
        dtype=torch.float8_e4m3fn,
    )
    gemm1_weights_scale = torch.ones(
        NUM_LOCAL_EXPERTS,
        (2 * INTERMEDIATE) // BLOCK,
        HIDDEN // BLOCK,
        device=device,
        dtype=torch.float32,
    )
    gemm2_weights = torch.zeros(
        NUM_LOCAL_EXPERTS,
        HIDDEN,
        INTERMEDIATE,
        device=device,
        dtype=torch.float8_e4m3fn,
    )
    gemm2_weights_scale = torch.ones(
        NUM_LOCAL_EXPERTS,
        HIDDEN // BLOCK,
        INTERMEDIATE // BLOCK,
        device=device,
        dtype=torch.float32,
    )
    return dict(
        routing_logits=routing_logits,
        routing_bias=routing_bias,
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        gemm1_weights=gemm1_weights,
        gemm1_weights_scale=gemm1_weights_scale,
        gemm2_weights=gemm2_weights,
        gemm2_weights_scale=gemm2_weights_scale,
    )


def run_trtllm(inputs):
    return flashinfer.trtllm_fp8_block_scale_moe(
        routing_logits=inputs["routing_logits"],
        routing_bias=inputs["routing_bias"],
        hidden_states=inputs["hidden_states"],
        hidden_states_scale=inputs["hidden_states_scale"],
        gemm1_weights=inputs["gemm1_weights"],
        gemm1_weights_scale=inputs["gemm1_weights_scale"],
        gemm2_weights=inputs["gemm2_weights"],
        gemm2_weights_scale=inputs["gemm2_weights_scale"],
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        n_group=N_GROUP,
        topk_group=TOPK_GROUP,
        intermediate_size=INTERMEDIATE,
        local_expert_offset=0,
        local_num_experts=NUM_LOCAL_EXPERTS,
        routed_scaling_factor=ROUTED_SCALING,
        routing_method_type=2,  # DeepSeekV3
    )


def run_contributed(inputs):
    return flashinfer.fusemoe_blackwell_fp8_dsv3(
        routing_logits=inputs["routing_logits"],
        routing_bias=inputs["routing_bias"],
        hidden_states=inputs["hidden_states"],
        hidden_states_scale=inputs["hidden_states_scale"],
        gemm1_weights=inputs["gemm1_weights"],
        gemm1_weights_scale=inputs["gemm1_weights_scale"],
        gemm2_weights=inputs["gemm2_weights"],
        gemm2_weights_scale=inputs["gemm2_weights_scale"],
        local_expert_offset=0,
        routed_scaling_factor=ROUTED_SCALING,
    )


def time_one(fn, inputs, dry_run_iters: int, num_iters: int):
    # Single warmup outside of the timer so JIT load / first-call setup
    # (CUTLASS .so build, FP8 LUT init, …) is not measured.
    try:
        fn(inputs)
    except Exception as exc:
        return None, None, str(exc)
    torch.cuda.synchronize()
    times_ms = bench_gpu_time(
        fn,
        input_args=(inputs,),
        dry_run_iters=dry_run_iters,
        repeat_iters=num_iters,
        enable_cupti=True,
    )
    median_ms = statistics.median(times_ms)
    std_ms = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0
    return median_ms, std_ms, None


def fmt_us(ms):
    return f"{ms * 1000:8.2f}" if ms is not None else "    n/a "


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--num-tokens",
        type=str,
        default=None,
        help="comma-separated list, defaults to a sweep",
    )
    p.add_argument("--num-iters", type=int, default=50)
    p.add_argument("--dry-run-iters", type=int, default=10)
    p.add_argument(
        "--no-idle-check",
        action="store_true",
        help="Skip the idle-GPU sanity check (useful in CI on dedicated runners).",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Abort if the GPU is not idle instead of warning.",
    )
    args = p.parse_args()

    if not args.no_idle_check:
        _check_gpu_idle(strict=args.strict)

    tokens_list = (
        [int(x) for x in args.num_tokens.split(",")]
        if args.num_tokens is not None
        else DEFAULT_TOKEN_COUNTS
    )

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"Shape: H={HIDDEN}, I={INTERMEDIATE}, E={NUM_EXPERTS}, "
        f"E_loc={NUM_LOCAL_EXPERTS}, top_k={TOP_K}, "
        f"n_group={N_GROUP}, topk_group={TOPK_GROUP}\n"
    )
    print(
        f"{'tokens':>7} | {'trtllm (us)':>12} | {'contrib (us)':>13} | {'speedup':>8}"
    )
    print("-" * 52)

    for n in tokens_list:
        inp = make_inputs(n)

        t_trt, s_trt, err_trt = time_one(
            run_trtllm, inp, args.dry_run_iters, args.num_iters
        )
        t_con, s_con, err_con = time_one(
            run_contributed, inp, args.dry_run_iters, args.num_iters
        )

        if err_trt:
            print(f"{n:>7} | trtllm error: {err_trt}")
            continue
        if err_con:
            print(f"{n:>7} | contributed error: {err_con}")
            continue

        speedup = t_trt / t_con if (t_con and t_con > 0) else float("nan")
        print(
            f"{n:>7} | {fmt_us(t_trt)} ±{fmt_us(s_trt).strip():>6} | "
            f"{fmt_us(t_con)} ±{fmt_us(s_con).strip():>6} | "
            f"{speedup:>7.2f}x"
        )


if __name__ == "__main__":
    main()
