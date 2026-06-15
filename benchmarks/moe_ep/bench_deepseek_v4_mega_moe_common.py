"""Shared fixtures for DeepSeek V4 MegaMoE vLLM vs moe_ep_v2 benchmarks.

Provides CLI defaults, deterministic bf16 expert weights, routing tensors,
and timing helpers. Each backend script loads/forwards through its own APIs.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import torch

# -------------------- default problem size (matches dummy_fp8_fp4_mega_moe.py) ----
DEFAULT_HIDDEN = 4096
DEFAULT_INTERMEDIATE = 2048
DEFAULT_NUM_TOKENS = 8192
DEFAULT_NUM_MAX_TOKENS = 8192
DEFAULT_NUM_LOCAL_EXPERTS = 64
DEFAULT_NUM_TOPK = 6
DEFAULT_ACTIVATION_CLAMP = 10.0
DEFAULT_FAST_MATH = True
DEFAULT_RENORMALIZE = True
DEFAULT_ROUTED_SCALING_FACTOR = 1.0
DEFAULT_HASH_VOCAB_SIZE = 129280

WEIGHT_SEED_BASE = 0
ROUTING_SEED_BASE = 1

# Printed by each backend benchmark (steady-state / cold-start semantics differ).
VLLM_TIMING_MODE_NOTE = (
    "symm_buffer + finalized weights cached; deep_gemm mega kernel warm"
)
VLLM_COLD_START_NOTE = "first forward: symm buffer alloc"

MOE_EP_V2_TIMING_MODE_NOTE = (
    "symm_buffer + preprocessed weights cached; deep_gemm fp8_fp4_mega_moe warm"
)
MOE_EP_V2_COLD_START_NOTE = "first forward: symm buffer alloc"


def parse_benchmark_args(*, description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    parser.add_argument("--intermediate", type=int, default=DEFAULT_INTERMEDIATE)
    parser.add_argument("--num-tokens", type=int, default=DEFAULT_NUM_TOKENS)
    parser.add_argument("--num-max-tokens", type=int, default=DEFAULT_NUM_MAX_TOKENS)
    parser.add_argument(
        "--num-local-experts",
        type=int,
        default=DEFAULT_NUM_LOCAL_EXPERTS,
        help="Experts owned by each EP rank.",
    )
    parser.add_argument("--topk", type=int, default=DEFAULT_NUM_TOPK)
    parser.add_argument(
        "--activation-clamp",
        type=float,
        default=DEFAULT_ACTIVATION_CLAMP,
        help="Swiglu clamp (e2e: float(config.swiglu_limit) or None).",
    )
    parser.add_argument(
        "--no-activation-clamp",
        action="store_true",
        help="Pass activation_clamp=None like swiglu_limit=None in e2e.",
    )
    parser.add_argument(
        "--fast-math",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_FAST_MATH,
    )
    parser.add_argument(
        "--renormalize",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_RENORMALIZE,
        help="norm_topk_prob passed to fused_topk_bias (vLLM e2e routing only).",
    )
    parser.add_argument(
        "--routed-scaling-factor",
        type=float,
        default=DEFAULT_ROUTED_SCALING_FACTOR,
    )
    parser.add_argument(
        "--random-routing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Shared deterministic topk fixtures from make_benchmark_routing_inputs "
            "(default; required for apples-to-apples vLLM vs moe_ep_v2 runs)."
        ),
    )
    parser.add_argument(
        "--hash-moe",
        action="store_true",
        help=(
            "Hash-MoE routing via vLLM fused_topk_bias "
            "(vLLM benchmark only; requires --no-random-routing)."
        ),
    )
    parser.add_argument(
        "--hash-vocab-size",
        type=int,
        default=DEFAULT_HASH_VOCAB_SIZE,
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument(
        "--cold-start",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Time the first forward before warmup (includes symm buffer alloc; "
            "default on)."
        ),
    )
    return parser.parse_args()


@dataclass(frozen=True)
class BenchTiming:
    """Forward latency breakdown."""

    steady_avg_ms: float
    steady_first_ms: float
    steady_min_ms: float
    steady_max_ms: float
    cold_start_ms: float | None


@dataclass(frozen=True)
class BenchmarkWeights:
    """Per-rank bf16 expert tensors consumed by each backend loader."""

    w13: torch.Tensor
    w2: torch.Tensor


@dataclass(frozen=True)
class BenchmarkInputs:
    """Activation + routing tensors passed into each backend forward."""

    hidden_states: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor


def require_env_rank() -> tuple[int, int]:
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    return local_rank, world_size


def check_problem(
    hidden: int,
    intermediate: int,
    num_tokens: int,
    num_max_tokens: int,
    num_experts: int,
    world_size: int,
) -> None:
    if hidden % 128 != 0 or intermediate % 128 != 0:
        raise ValueError("hidden and intermediate must be multiples of 128")
    if num_tokens > num_max_tokens:
        raise ValueError("num_tokens must be <= num_max_tokens")
    if num_experts % world_size != 0:
        raise ValueError("num_experts must be divisible by world_size")


def warn_kernel_token_sizing(
    *,
    rank: int,
    num_tokens: int,
    num_max_tokens: int,
) -> None:
    """``fp8_fp4_mega_moe`` sizes work from ``num_max_tokens_per_rank``, not batch len."""
    if rank != 0 or num_tokens == num_max_tokens:
        return
    print(
        "WARNING: deep_gemm.fp8_fp4_mega_moe uses num_max_tokens_per_rank="
        f"{num_max_tokens} for the symm buffer and kernel grid, not "
        f"num_tokens={num_tokens}. To scale kernel cost with batch size, pass "
        f"matching --num-max-tokens (e.g. --num-tokens {num_tokens} "
        f"--num-max-tokens {num_tokens}).",
        flush=True,
    )


def activation_clamp_from_args(args: argparse.Namespace) -> float | None:
    return None if args.no_activation_clamp else args.activation_clamp


def routing_mode_from_args(args: argparse.Namespace) -> str:
    if args.random_routing:
        return "random"
    if args.hash_moe:
        return "hash-moe"
    return "gate+sqrtsoftplus"


def require_shared_routing_benchmark(args: argparse.Namespace) -> None:
    """moe_ep_v2 benchmarks the expert kernel only; routing must match vLLM fixtures."""
    if args.hash_moe:
        raise SystemExit(
            "ERROR: --hash-moe is vLLM-only. Use "
            "bench_deepseek_v4_mega_moe_experts.py with "
            "--no-random-routing --hash-moe."
        )
    if not args.random_routing:
        raise SystemExit(
            "ERROR: moe_ep_v2 has no gate routing API. Use shared routing fixtures "
            "(--random-routing, default) or run the vLLM benchmark with "
            "--no-random-routing for gate/hash routing."
        )


def make_shared_benchmark_inputs(
    rank: int,
    args: argparse.Namespace,
    *,
    num_experts: int,
) -> BenchmarkInputs:
    """Identical activation/routing tensors for both backend benchmarks."""
    return make_benchmark_routing_inputs(
        rank,
        num_tokens=args.num_tokens,
        hidden=args.hidden,
        num_experts=num_experts,
        topk=args.topk,
    )


def make_benchmark_weights(
    rank: int,
    *,
    num_local_experts: int,
    hidden: int,
    intermediate: int,
) -> BenchmarkWeights:
    """Per-rank bf16 w13/w2; same RNG stream on every rank/backend."""
    torch.manual_seed(WEIGHT_SEED_BASE + rank)
    w13_rows: list[torch.Tensor] = []
    w2_rows: list[torch.Tensor] = []
    for _ in range(num_local_experts):
        w1 = torch.randn(intermediate, hidden, device="cuda", dtype=torch.bfloat16)
        w3 = torch.randn(intermediate, hidden, device="cuda", dtype=torch.bfloat16)
        w2 = torch.randn(hidden, intermediate, device="cuda", dtype=torch.bfloat16)
        w13_rows.append(torch.cat([w1, w3], dim=0))
        w2_rows.append(w2)
    return BenchmarkWeights(w13=torch.stack(w13_rows), w2=torch.stack(w2_rows))


def make_benchmark_routing_inputs(
    rank: int,
    *,
    num_tokens: int,
    hidden: int,
    num_experts: int,
    topk: int,
) -> BenchmarkInputs:
    """Random topk routing fixtures shared by both backends."""
    torch.manual_seed(ROUTING_SEED_BASE + rank)
    hidden_states = torch.randn(
        num_tokens, hidden, device="cuda", dtype=torch.bfloat16
    )
    scores = torch.randn(
        num_tokens, num_experts, device="cuda", dtype=torch.float32
    )
    topk_weights, topk_ids = torch.topk(
        scores, topk, dim=-1, largest=True, sorted=False
    )
    return BenchmarkInputs(
        hidden_states=hidden_states,
        topk_weights=topk_weights.to(torch.float32),
        topk_ids=topk_ids.to(torch.int64),
    )


def bench_forward_ms(
    run_once,
    *,
    warmup: int,
    repeat: int,
    cold_start: bool,
) -> BenchTiming:
    def time_once() -> float:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        run_once()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end)

    cold_start_ms = time_once() if cold_start else None

    for _ in range(warmup):
        run_once()
    torch.cuda.synchronize()

    elapsed_ms = [time_once() for _ in range(repeat)]

    return BenchTiming(
        steady_avg_ms=sum(elapsed_ms) / len(elapsed_ms),
        steady_first_ms=elapsed_ms[0],
        steady_min_ms=min(elapsed_ms),
        steady_max_ms=max(elapsed_ms),
        cold_start_ms=cold_start_ms,
    )


def print_benchmark_header(
    *,
    title: str,
    rank: int,
    world_size: int,
    num_local_experts: int,
    args: argparse.Namespace,
    num_experts: int,
    activation_clamp: float | None,
    routing_mode: str,
    extra_lines: tuple[str, ...] = (),
    timing_mode_note: str = "",
) -> None:
    if rank != 0:
        return
    print(title)
    print(f"  world_size={world_size} ep_local_experts={num_local_experts}")
    print(
        f"  hidden={args.hidden} intermediate={args.intermediate} "
        f"num_experts={num_experts} topk={args.topk}"
    )
    print(
        f"  num_tokens={args.num_tokens} num_max_tokens={args.num_max_tokens} "
        f"kernel_tokens={args.num_max_tokens} "
        f"clamp={activation_clamp} fast_math={args.fast_math}"
    )
    print(
        f"  routing={routing_mode} renormalize={args.renormalize} "
        f"routed_scaling_factor={args.routed_scaling_factor}"
    )
    for line in extra_lines:
        print(line)
    print(
        f"  warmup={args.warmup} repeat={args.repeat} "
        f"cold_start={'on' if args.cold_start else 'off'}"
    )
    if timing_mode_note:
        print(f"  timing_mode=steady-state ({timing_mode_note})")


def print_benchmark_timing(
    timing: BenchTiming,
    *,
    rank: int,
    cold_start_note: str = "first forward: symm buffer alloc",
) -> None:
    if rank != 0:
        return
    print(
        f"  steady_avg_ms={timing.steady_avg_ms:.3f} "
        f"first={timing.steady_first_ms:.3f} "
        f"min={timing.steady_min_ms:.3f} max={timing.steady_max_ms:.3f}"
    )
    if timing.cold_start_ms is not None:
        print(
            f"  cold_start_ms={timing.cold_start_ms:.3f} ({cold_start_note})"
        )


# Back-compat aliases for any out-of-tree callers.
generate_bf16_expert_weights = make_benchmark_weights
make_random_routing_inputs = make_benchmark_routing_inputs
