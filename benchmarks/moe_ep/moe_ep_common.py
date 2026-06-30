"""Shared fixtures for MoE-EP-only DeepSeek V4 MegaMoE benchmarks.

Provides CLI defaults, deterministic bf16 expert weights, routing tensors,
backend identifiers, and timing helpers (CUDA events + CUDA graphs).
"""

from __future__ import annotations

import argparse
import os
import statistics
from dataclasses import dataclass
from typing import Callable, Literal

import torch

from flashinfer.testing.utils import bench_gpu_time_with_cudagraph

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

TimingMode = Literal["cudagraph", "cuda_event", "cupti"]
BenchScope = Literal["e2e", "prestaged"]

PRESTAGED_BENCH_BACKEND_IDS = frozenset({"fi_nvfp4", "fi_mxfp8"})

BACKEND_IDS: tuple[str, ...] = (
    "vllm_deepgemm",
    "fi_deep_gemm",
    "fi_nvfp4",
    "fi_mxfp8",
)

FI_MEGAKERNEL_BY_BACKEND: dict[str, str] = {
    "fi_deep_gemm": "deep_gemm_mega",
    "fi_nvfp4": "nvfp4_cutedsl",
    "fi_mxfp8": "mxfp8_cutedsl",
}

# Only deep_gemm mega_moe compute is CUDA-graph safe today (no in-kernel sync).
FI_CUDAGRAPH_BACKEND_IDS = frozenset({"fi_deep_gemm"})

BACKEND_LABELS: dict[str, str] = {
    "vllm_deepgemm": "vLLM DeepseekV4MegaMoEExperts (deep_gemm_mega_moe)",
    "fi_deep_gemm": "flashinfer MoEEpMegaLayer + DeepGemmMegaMoeConfig",
    "fi_nvfp4": "flashinfer MoEEpMegaLayer + Nvfp4CutedslMegaMoeConfig",
    "fi_mxfp8": "flashinfer MoEEpMegaLayer + Mxfp8CutedslMegaMoeConfig",
}

VLLM_TIMING_MODE_NOTE = (
    "symm_buffer + finalized weights cached; bench hot path stage + mega_moe "
    "(no torch.ops / set_forward_context)"
)
VLLM_COLD_START_NOTE = "first forward: symm buffer alloc"

FI_TIMING_MODE_NOTES: dict[str, str] = {
    "fi_deep_gemm": (
        "symm_buffer + preprocessed weights cached; bench hot path stage + mega_moe"
    ),
    "fi_nvfp4": (
        "symm_buffer + preprocessed weights cached; nvfp4_cutedsl mega kernel warm"
    ),
    "fi_mxfp8": (
        "symm_buffer + preprocessed weights cached; mxfp8_cutedsl mega kernel warm"
    ),
}

FI_COLD_START_NOTE = "first forward: symm buffer alloc"

CUDAGRAPH_TIMING_MODE_NOTE = (
    "eager input staging + CUDA graph replay of mega_moe compute "
    "(staging not capture-safe; both are in the timed window for vLLM/fi parity)"
)


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
            "(default; required for apples-to-apples vLLM vs moe_ep runs)."
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
            "default on). Ignored when --timing=cudagraph."
        ),
    )
    return parser.parse_args()


def parse_bench_args(*, description: str) -> argparse.Namespace:
    """Benchmark CLI: extends :func:`parse_benchmark_args` with backend/timing."""
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
    )
    parser.add_argument("--no-activation-clamp", action="store_true")
    parser.add_argument(
        "--fast-math",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_FAST_MATH,
    )
    parser.add_argument(
        "--renormalize",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_RENORMALIZE,
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
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument(
        "--cold-start",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Time first forward before warmup (cuda_event/cupti only).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="fi_deep_gemm",
        choices=(*BACKEND_IDS, "all"),
        help="MegaMoE backend to benchmark, or 'all' to run each sequentially.",
    )
    parser.add_argument(
        "--timing",
        type=str,
        default="cudagraph",
        choices=("cudagraph", "cuda_event", "cupti"),
        help="Timing backend (default: CUDA graph replay for serving-like steady state).",
    )
    parser.add_argument(
        "--no-cuda-graph",
        action="store_true",
        help="Alias for --timing=cuda_event.",
    )
    parser.add_argument(
        "--cold-l2",
        action="store_true",
        help="Use cold-L2 rotating buffers in CUDA graph timing (default: warm L2).",
    )
    parser.add_argument(
        "--bench-scope",
        type=str,
        default="e2e",
        choices=("e2e", "prestaged"),
        help=(
            "e2e: bf16 activations + on-the-fly quant in stage_inputs (default). "
            "prestaged: pre-quantize nvfp4/mxfp8 activations once, then time "
            "forward with quantize_input=False (memcpy staging + kernel)."
        ),
    )
    return parser.parse_args()


def resolve_timing_mode(args: argparse.Namespace) -> TimingMode:
    if getattr(args, "no_cuda_graph", False):
        return "cuda_event"
    return args.timing


def flashinfer_backend_ids() -> tuple[str, ...]:
    return tuple(k for k in BACKEND_IDS if k.startswith("fi_"))


# vLLM first, then deep_gemm, then quant backends (matches archive sweep order).
_BENCHMARK_BACKEND_PRIORITY: dict[str, int] = {
    "vllm_deepgemm": 0,
    "fi_deep_gemm": 1,
    "fi_mxfp8": 2,
    "fi_nvfp4": 3,
}

# Post-check timing in verify: only the comparable deep_gemm pair. Quant backends
# init NVSHMEM and skew deep_gemm timings if run in the same process first.
VERIFY_TIMING_BACKEND_IDS: tuple[str, ...] = ("vllm_deepgemm", "fi_deep_gemm")


def benchmark_backend_order(backend_ids: list[str]) -> list[str]:
    return sorted(backend_ids, key=lambda b: _BENCHMARK_BACKEND_PRIORITY.get(b, 99))


def fi_backend_supports_cudagraph(backend_id: str) -> bool:
    return backend_id in FI_CUDAGRAPH_BACKEND_IDS


def resolve_fi_timing_mode(backend_id: str, timing_mode: TimingMode) -> TimingMode:
    """Quant cutedsl megakernels call ``cuda.synchronize()``; use cuda events."""
    if timing_mode == "cudagraph" and not fi_backend_supports_cudagraph(backend_id):
        return "cuda_event"
    return timing_mode


def fi_timing_fallback_note(
    backend_id: str,
    requested: TimingMode,
    effective: TimingMode,
) -> str | None:
    if requested == effective:
        return None
    return (
        f"{backend_id}: requested timing={requested} unsupported "
        f"(cutedsl mega kernels are not CUDA-graph safe); using {effective}"
    )


def resolve_bench_scope(backend_id: str, bench_scope: BenchScope) -> BenchScope:
    if bench_scope == "prestaged" and backend_id not in PRESTAGED_BENCH_BACKEND_IDS:
        return "e2e"
    return bench_scope


def bench_scope_fallback_note(
    backend_id: str,
    requested: BenchScope,
    effective: BenchScope,
) -> str | None:
    if requested == effective:
        return None
    return (
        f"{backend_id}: bench_scope={requested} only applies to "
        f"{sorted(PRESTAGED_BENCH_BACKEND_IDS)}; using {effective}"
    )


def bench_scope_note_for_scope(bench_scope: BenchScope) -> str:
    if bench_scope == "prestaged":
        return (
            "activations pre-quantized once outside timed loop; "
            "quantize_input=False (memcpy stage + kernel)"
        )
    return "quantize_input=True preprocess_weights=True; bf16→quant in stage_inputs"


@dataclass(frozen=True)
class BenchTiming:
    """Forward latency breakdown."""

    steady_avg_ms: float
    steady_first_ms: float
    steady_min_ms: float
    steady_max_ms: float
    cold_start_ms: float | None
    timing_mode: TimingMode = "cuda_event"


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


def require_sm100(local_rank: int) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("ERROR: CUDA is required")
    capability = torch.cuda.get_device_capability(local_rank)
    if capability[0] != 10:
        raise SystemExit(
            f"ERROR: rank {local_rank} needs SM100 (Blackwell), "
            f"got sm_{capability[0]}{capability[1]}"
        )


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
    if getattr(args, "hash_moe", False):
        return "hash-moe"
    return "gate+sqrtsoftplus"


def require_shared_routing_benchmark(args: argparse.Namespace) -> None:
    """moe_ep benchmarks the expert kernel only; routing must match vLLM fixtures."""
    if getattr(args, "hash_moe", False):
        raise SystemExit(
            "ERROR: --hash-moe is vLLM-only. Use the vLLM benchmark with "
            "--no-random-routing --hash-moe."
        )
    if not args.random_routing:
        raise SystemExit(
            "ERROR: moe_ep has no gate routing API. Use shared routing fixtures "
            "(--random-routing, default)."
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
    run_once: Callable[[], object],
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
        timing_mode="cuda_event",
    )


def bench_deep_gemm_mega_cudagraph_ms(
    stage_inputs: Callable[[], object],
    run_compute: Callable[[], object],
    *,
    warmup: int,
    repeat: int,
) -> BenchTiming:
    """Time eager staging + graphed ``fp8_fp4_mega_moe`` compute (deep_gemm backends).

    Staging is not CUDA-graph captured (bf16→fp8 quant is not capture-safe) but is
    included in the timed window so vLLM and flashinfer deep_gemm paths match.
    """
    for _ in range(warmup):
        stage_inputs()
        run_compute()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_compute()
    torch.cuda.synchronize()

    elapsed_ms: list[float] = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        stage_inputs()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        elapsed_ms.append(start.elapsed_time(end))

    return BenchTiming(
        steady_avg_ms=sum(elapsed_ms) / len(elapsed_ms),
        steady_first_ms=elapsed_ms[0],
        steady_min_ms=min(elapsed_ms),
        steady_max_ms=max(elapsed_ms),
        cold_start_ms=None,
        timing_mode="cudagraph",
    )


def bench_forward_cudagraph_ms(
    run_once: Callable[[], object],
    *,
    warmup: int,
    repeat: int,
    cold_l2_cache: bool = False,
) -> BenchTiming:
    """CUDA graph replay of a single callable (full capture; not for deep_gemm parity)."""
    for _ in range(warmup):
        run_once()
    torch.cuda.synchronize()

    measurements = bench_gpu_time_with_cudagraph(
        run_once,
        dry_run_iters=0,
        repeat_iters=repeat,
        cold_l2_cache=cold_l2_cache,
        num_iters_within_graph=1,
    )
    return BenchTiming(
        steady_avg_ms=statistics.mean(measurements),
        steady_first_ms=measurements[0],
        steady_min_ms=min(measurements),
        steady_max_ms=max(measurements),
        cold_start_ms=None,
        timing_mode="cudagraph",
    )


def bench_forward(
    run_once: Callable[[], object],
    *,
    timing_mode: TimingMode,
    warmup: int,
    repeat: int,
    cold_start: bool,
    cold_l2_cache: bool = False,
) -> BenchTiming:
    if timing_mode == "cudagraph":
        return bench_forward_cudagraph_ms(
            run_once,
            warmup=warmup,
            repeat=repeat,
            cold_l2_cache=cold_l2_cache,
        )
    if timing_mode == "cupti":
        from flashinfer.testing.utils import bench_gpu_time_with_cupti

        for _ in range(warmup):
            run_once()
        torch.cuda.synchronize()
        measurements = bench_gpu_time_with_cupti(
            run_once,
            dry_run_iters=0,
            repeat_iters=repeat,
            use_cuda_graph=False,
            cold_l2_cache=cold_l2_cache,
        )
        return BenchTiming(
            steady_avg_ms=statistics.mean(measurements),
            steady_first_ms=measurements[0],
            steady_min_ms=min(measurements),
            steady_max_ms=max(measurements),
            cold_start_ms=None,
            timing_mode="cupti",
        )
    return bench_forward_ms(
        run_once,
        warmup=warmup,
        repeat=repeat,
        cold_start=cold_start,
    )


def timing_mode_note_for_backend(
    backend_id: str,
    *,
    timing_mode: TimingMode,
) -> str:
    if timing_mode == "cudagraph":
        return CUDAGRAPH_TIMING_MODE_NOTE
    if backend_id == "vllm_deepgemm":
        return VLLM_TIMING_MODE_NOTE
    return FI_TIMING_MODE_NOTES.get(backend_id, MOE_EP_V2_TIMING_MODE_NOTE)


def cold_start_note_for_backend(backend_id: str) -> str:
    if backend_id == "vllm_deepgemm":
        return VLLM_COLD_START_NOTE
    return FI_COLD_START_NOTE


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
    backend_id: str | None = None,
    timing_mode: TimingMode | None = None,
    extra_lines: tuple[str, ...] = (),
    timing_mode_note: str = "",
) -> None:
    if rank != 0:
        return
    print(title)
    if backend_id is not None:
        print(f"  backend={backend_id} ({BACKEND_LABELS[backend_id]})")
    if timing_mode is not None:
        print(f"  timing={timing_mode}")
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
    print(f"  warmup={args.warmup} repeat={args.repeat}")
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
MOE_EP_V2_TIMING_MODE_NOTE = FI_TIMING_MODE_NOTES["fi_deep_gemm"]
MOE_EP_V2_COLD_START_NOTE = FI_COLD_START_NOTE
