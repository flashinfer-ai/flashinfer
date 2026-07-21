"""SM90 push FP8 MegaMoE benchmark and correctness/performance gate."""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sm90_push_megamoe_reference import reference_moe_fp8_weights_streaming


CONFIGS = {
    "TINY": dict(tokens=256, hidden_size=1024, intermediate=1024, experts=8, top_k=2),
    "SMALL": dict(
        tokens=2048, hidden_size=4096, intermediate=2048, experts=32, top_k=6
    ),
    "DSV3": dict(
        tokens=2048, hidden_size=7168, intermediate=2048, experts=256, top_k=8
    ),
}


@dataclass
class Result:
    mode: str
    ep: int
    rank: int
    config: dict = field(default_factory=dict)
    payload_dtype: str = ""
    combine_dtype: str = ""
    routing: str = ""
    graph: bool = False
    dedup: bool = False
    grouped_combine: bool = False
    fuse_fc1: bool = False
    run_id: str = ""
    case_id: str = ""
    warmup: int = 0
    iters: int = 0
    git_commit: str = ""
    git_dirty: bool = False
    source_hash: str = ""
    torch_version: str = ""
    cuda_version: str = ""
    gpu_name: str = ""
    cos: float = float("nan")
    err_ratio: float = float("nan")
    growth: float = float("nan")
    p50_ms: float = float("nan")
    p99_ms: float = float("nan")
    baseline_p50_ms: float = float("nan")
    speedup_p50: float = float("nan")


def log(rank: int, message: str) -> None:
    print(f"[rank {rank}] {message}", flush=True)


def collect_provenance(device: torch.device) -> dict:
    """Collect the repository and kernel sources that determine this result."""
    root = Path(__file__).resolve().parents[1]

    def git_output(*args: str) -> str:
        try:
            result = subprocess.run(
                ["git", "-C", str(root), *args],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.stdout.strip() if result.returncode == 0 else ""
        except Exception:
            return ""

    source_paths = (
        "benchmarks/bench_sm90_push_megamoe.py",
        "benchmarks/sm90_push_megamoe_reference.py",
        "benchmarks/sm90_push_megamoe_baseline.py",
        "flashinfer/moe_ep/__init__.py",
        "flashinfer/moe_ep/modes/mega_layer.py",
        "flashinfer/moe_ep/backends/mega/kernel/__init__.py",
        "flashinfer/moe_ep/backends/mega/kernel/sm90_push_fp8/__init__.py",
        "flashinfer/moe_ep/backends/mega/kernel/sm90_push_fp8/backend.py",
        "flashinfer/moe_ep/backends/mega/kernel/sm90_push_fp8/config.py",
        "flashinfer/moe_ep/backends/mega/kernel/sm90_push_fp8/staging.py",
        "flashinfer/moe_ep/backends/mega/kernel/sm90_push_fp8/weights.py",
        "flashinfer/moe_ep/kernel_src/sm90_push_megamoe/__init__.py",
        "flashinfer/moe_ep/kernel_src/sm90_push_megamoe/shim/gemm.py",
        "flashinfer/moe_ep/kernel_src/sm90_push_megamoe/shim/jit.py",
        "flashinfer/moe_ep/kernel_src/sm90_push_megamoe/shim/protocol.py",
        "flashinfer/moe_ep/kernel_src/sm90_push_megamoe/shim/runner.py",
        "flashinfer/moe_ep/kernel_src/sm90_push_megamoe/shim/weights.py",
        "flashinfer/moe_ep/kernel_src/sm90_push_megamoe/src/a2a/sm90_push_a2a_ops.cu",
        "flashinfer/moe_ep/kernel_src/sm90_push_megamoe/src/a2a/sm90_push_a2a.cuh",
        "flashinfer/moe_ep/kernel_src/sm90_push_megamoe/src/fp8_gemm/fp8_moe_binding.cu",
        "flashinfer/moe_ep/kernel_src/sm90_push_megamoe/src/fp8_gemm/fp8_moe_fc1_fused.cuh",
        "flashinfer/moe_ep/kernel_src/sm90_push_megamoe/src/fp8_gemm/fp8_moe_jit.cuh",
        "flashinfer/moe_ep/kernel_src/sm90_push_megamoe/src/fp8_gemm/fp8_moe_launcher.cuh",
        "flashinfer/moe_ep/kernel_src/sm90_push_megamoe/src/fp8_gemm/fp8_moe_scheduler.cuh",
        "csrc/nv_internal/tensorrt_llm/deep_gemm/scheduler.cuh",
        "csrc/nv_internal/tensorrt_llm/deep_gemm/fp8_gemm_impl.cuh",
        "csrc/nv_internal/tensorrt_llm/deep_gemm/fp8_gemm.cuh",
        "csrc/nv_internal/tensorrt_llm/deep_gemm/jit_utils.cuh",
        "csrc/nv_internal/tensorrt_llm/deep_gemm/compiler.cuh",
        "csrc/nv_internal/tensorrt_llm/deep_gemm/runtime.cuh",
        "csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/"
        "fp8_blockscale_gemm/fp8_blockscale_gemm_kernel.cuh",
    )
    digest = hashlib.sha256()
    for relative in source_paths:
        path = root / relative
        if path.exists():
            digest.update(relative.encode())
            digest.update(path.read_bytes())
    return {
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_dirty": bool(git_output("status", "--porcelain", "-uno")),
        "source_hash": digest.hexdigest()[:12],
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "",
        "gpu_name": (
            torch.cuda.get_device_name(device) if torch.cuda.is_available() else ""
        ),
    }


def json_record(record: dict) -> str:
    """Encode one strict JSONL record, mapping non-finite values to null."""

    def clean(value):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value

    return json.dumps(clean(record), allow_nan=False)


def make_weights(
    num_experts: int,
    intermediate_size: int,
    hidden_size: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    w13 = (
        torch.randn(
            num_experts,
            2 * intermediate_size,
            hidden_size,
            generator=generator,
        )
        * hidden_size**-0.5
    ).to(device=device, dtype=torch.bfloat16)
    w2 = (
        torch.randn(
            num_experts,
            hidden_size,
            intermediate_size,
            generator=generator,
        )
        * intermediate_size**-0.5
    ).to(device=device, dtype=torch.bfloat16)
    return w13, w2


def make_routing(
    num_tokens: int,
    num_experts: int,
    top_k: int,
    routing: str,
    world_size: int,
    rank: int,
    num_local_experts: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if routing == "hot":
        ids = torch.zeros(num_tokens, top_k, dtype=torch.int32)
    elif routing == "hot1":
        ids = (
            torch.randn(num_tokens, num_experts, generator=generator)
            .topk(top_k, dim=1)
            .indices.to(torch.int32)
        )
        ids[:, 0] = 0
    elif routing == "all_remote":
        logits = torch.randn(num_tokens, num_experts, generator=generator)
        if world_size > 1:
            start = rank * num_local_experts
            logits[:, start : start + num_local_experts] = float("-inf")
        ids = logits.topk(top_k, dim=1).indices.to(torch.int32)
    else:
        ids = (
            torch.randn(num_tokens, num_experts, generator=generator)
            .topk(top_k, dim=1)
            .indices.to(torch.int32)
        )
    weights = torch.rand(num_tokens, top_k, generator=generator) + 0.1
    weights = weights / weights.sum(dim=1, keepdim=True)
    return (
        ids.to(device),
        weights.to(device=device, dtype=torch.float32),
    )


def time_rounds(
    fn, warmup: int, iters: int, world_size: int = 1
) -> tuple[float, float]:
    """Return barrier-aligned p50 and p99 GPU times in milliseconds."""
    dist = None
    if world_size > 1:
        import torch.distributed as dist

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    elapsed = []
    for _ in range(iters):
        if dist is not None:
            dist.barrier()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        elapsed.append(start.elapsed_time(end))
    values = torch.tensor(elapsed, dtype=torch.float64)
    if dist is not None:
        dist.all_reduce(values, op=dist.ReduceOp.MAX)
    values = values.sort().values
    return (
        float(values[len(values) // 2]),
        float(values[min(len(values) - 1, int(len(values) * 0.99))]),
    )


def _destroy_layers(layers) -> None:
    for layer in reversed(layers):
        layer.destroy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=list(CONFIGS), default="SMALL")
    parser.add_argument("--tokens", type=int, help="override the config token count")
    parser.add_argument(
        "--token-capacity", type=int, help="maximum tokens per rank (default: tokens)"
    )
    parser.add_argument("--payload-dtype", choices=["fp8", "bf16"], default="fp8")
    parser.add_argument("--combine-dtype", choices=["fp8", "bf16"], default="fp8")
    parser.add_argument(
        "--dedup",
        action="store_true",
        help="store one dispatch payload per token and destination rank",
    )
    parser.add_argument(
        "--grouped-combine",
        action="store_true",
        help="pre-reduce FP8 combine rows by source rank; requires FP8 combine",
    )
    parser.add_argument(
        "--fuse-fc1",
        action="store_true",
        help="use the fused SwiGLU and activation-quantization FC1 epilogue",
    )
    parser.add_argument(
        "--routing",
        choices=["random", "hot", "hot1", "all_remote"],
        default="random",
    )
    parser.add_argument("--graph", action="store_true", help="time CUDA graph replays")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument(
        "--torch-profile",
        action="store_true",
        help="profile the E2E public layer call and print rank-zero GPU kernel times",
    )
    parser.add_argument(
        "--nsys-capture",
        action="store_true",
        help="bracket timed E2E rounds with the CUDA profiler API",
    )
    parser.add_argument(
        "--case-id",
        default="",
        help="comparison identifier shared by all legs of one benchmark case",
    )
    parser.add_argument("--json", type=Path, help="append strict JSONL result records")
    parser.add_argument("--assert-cos-min", type=float)
    parser.add_argument("--assert-growth-max", type=float)
    parser.add_argument("--assert-speedup-min", type=float)
    parser.add_argument("--assert-p99-jitter-max", type=float)
    args = parser.parse_args()

    if args.grouped_combine and args.combine_dtype != "fp8":
        raise SystemExit("--grouped-combine requires --combine-dtype fp8")
    if args.iters <= 0 or args.warmup < 0:
        raise SystemExit("--iters must be positive and --warmup must be non-negative")
    if not torch.cuda.is_available():
        raise SystemExit("SM90 push benchmark requires CUDA")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    process_group = None
    if world_size > 1:
        import torch.distributed as dist

        try:
            dist.init_process_group(backend="cpu:gloo,cuda:nccl")
        except (ValueError, RuntimeError):
            dist.init_process_group(backend="gloo")
        process_group = dist.group.WORLD
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", str(rank))))
    else:
        torch.cuda.set_device(0)

    identifiers = [
        time.strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}" if rank == 0 else None,
        (args.case_id or uuid.uuid4().hex[:12]) if rank == 0 else None,
    ]
    if world_size > 1:
        import torch.distributed as dist

        dist.broadcast_object_list(identifiers, src=0)
    run_id, case_id = identifiers

    try:
        from flashinfer.moe_ep import (
            BootstrapConfig,
            FleetParams,
            MegaConfig,
            MoEEpLayer,
            MoEEpTensors,
            MoEWeightPack,
            Sm90PushFp8MegaMoeConfig,
        )
    except ImportError as error:
        raise SystemExit(
            "SM90 push FP8 backend registration is unavailable; expected "
            "flashinfer.moe_ep.Sm90PushFp8MegaMoeConfig"
        ) from error

    config = dict(CONFIGS[args.config])
    if args.tokens is not None:
        config["tokens"] = args.tokens
    num_tokens = config["tokens"]
    hidden_size = config["hidden_size"]
    intermediate_size = config["intermediate"]
    num_experts = config["experts"]
    top_k = config["top_k"]
    token_capacity = args.token_capacity or num_tokens
    if num_experts % world_size:
        raise SystemExit(
            f"experts {num_experts} not divisible by world size {world_size}"
        )
    if token_capacity < num_tokens:
        raise SystemExit(
            f"token capacity {token_capacity} is smaller than tokens {num_tokens}"
        )

    num_local_experts = num_experts // world_size
    device = torch.device("cuda", torch.cuda.current_device())
    w13, w2 = make_weights(
        num_experts, intermediate_size, hidden_size, seed=7, device=device
    )
    local_start = rank * num_local_experts
    local_end = local_start + num_local_experts

    bootstrap = BootstrapConfig(
        world_size=world_size,
        rank=rank,
        process_group=process_group,
    )
    fleet_params = FleetParams(
        num_experts=num_experts,
        max_tokens_per_rank=token_capacity,
        token_hidden_size=hidden_size,
    )

    def build_layer(combine_dtype: str):
        kernel_config = Sm90PushFp8MegaMoeConfig(
            intermediate_size=intermediate_size,
            top_k=top_k,
            payload_dtype=args.payload_dtype,
            combine_dtype=combine_dtype,
            dedup_dispatch=args.dedup,
            grouped_combine=args.grouped_combine and combine_dtype == "fp8",
            fuse_fc1_epilogue=args.fuse_fc1,
        )
        weights = MoEWeightPack(
            w13=w13[local_start:local_end].contiguous(),
            w2=w2[local_start:local_end].contiguous(),
        )
        return MoEEpLayer(
            bootstrap=bootstrap,
            fleet_params=fleet_params,
            weights=weights,
            backend=MegaConfig(
                megakernel=kernel_config,
                quantize_input=True,
                preprocess_weights=True,
            ),
        )

    hidden_states = torch.randn(
        num_tokens,
        hidden_size,
        generator=torch.Generator(device="cpu").manual_seed(11 + rank),
    ).to(device=device, dtype=torch.bfloat16)
    topk_ids, topk_weights = make_routing(
        num_tokens,
        num_experts,
        top_k,
        args.routing,
        world_size,
        rank,
        num_local_experts,
        seed=13 + rank,
        device=device,
    )
    inputs = MoEEpTensors(
        hidden_states=hidden_states,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
    )

    metadata = {
        "dedup": args.dedup,
        "grouped_combine": args.grouped_combine,
        "fuse_fc1": args.fuse_fc1,
        "run_id": run_id,
        "case_id": case_id,
        "warmup": args.warmup,
        "iters": args.iters,
        **collect_provenance(device),
    }
    result = Result(
        mode="e2e",
        ep=world_size,
        rank=rank,
        config=config,
        payload_dtype=args.payload_dtype,
        combine_dtype=args.combine_dtype,
        routing=args.routing,
        graph=args.graph,
        **metadata,
    )

    layers = []
    layer = build_layer(args.combine_dtype)
    layers.append(layer)
    output = layer(inputs).clone()
    torch.cuda.synchronize()
    if not bool(torch.isfinite(output).all()):
        _destroy_layers(layers)
        raise SystemExit(f"[rank {rank}] FATAL: NaN/Inf in output")

    reference = reference_moe_fp8_weights_streaming(
        hidden_states, w13, w2, topk_ids, topk_weights
    )
    reference_rms = reference.float().pow(2).mean().sqrt().clamp_min(1e-6)
    result.err_ratio = float(
        (output.float() - reference).pow(2).mean().sqrt() / reference_rms
    )
    result.cos = float(
        torch.nn.functional.cosine_similarity(
            output.float().flatten(), reference.float().flatten(), dim=0
        )
    )

    anchor_layer = None
    if args.combine_dtype == "fp8":
        anchor_layer = build_layer("bf16")
        layers.append(anchor_layer)
        anchor_output = anchor_layer(inputs).clone()
        torch.cuda.synchronize()
        anchor_error = float(
            (anchor_output.float() - reference).pow(2).mean().sqrt() / reference_rms
        )
        result.growth = result.err_ratio / max(anchor_error, 1e-12)
    log(
        rank,
        f"correctness: cos={result.cos:.5f} err={result.err_ratio:.4f} "
        f"growth={result.growth:.3f}",
    )

    if args.graph:
        static_hidden = hidden_states.clone()
        static_ids = topk_ids.clone()
        static_weights = topk_weights.clone()
        static_inputs = MoEEpTensors(
            hidden_states=static_hidden,
            topk_ids=static_ids,
            topk_weights=static_weights,
        )
        for _ in range(2):
            layer(static_inputs)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_output = layer(static_inputs)

        replay_hidden = torch.randn(
            num_tokens,
            hidden_size,
            generator=torch.Generator(device="cpu").manual_seed(1011 + rank),
        ).to(device=device, dtype=torch.bfloat16)
        replay_ids, replay_weights = make_routing(
            num_tokens,
            num_experts,
            top_k,
            args.routing,
            world_size,
            rank,
            num_local_experts,
            seed=1013 + rank,
            device=device,
        )
        static_hidden.copy_(replay_hidden)
        static_ids.copy_(replay_ids)
        static_weights.copy_(replay_weights)
        graph.replay()
        torch.cuda.synchronize()
        graph_output = static_output.clone()
        graph_reference = reference_moe_fp8_weights_streaming(
            replay_hidden, w13, w2, replay_ids, replay_weights
        )
        graph_rms = graph_reference.float().pow(2).mean().sqrt().clamp_min(1e-6)
        graph_error = float(
            (graph_output.float() - graph_reference).pow(2).mean().sqrt() / graph_rms
        )
        graph_cos = float(
            torch.nn.functional.cosine_similarity(
                graph_output.float().flatten(),
                graph_reference.float().flatten(),
                dim=0,
            )
        )
        result.err_ratio = max(result.err_ratio, graph_error)
        result.cos = min(result.cos, graph_cos)
        graph_growth = float("nan")
        if anchor_layer is not None:
            replay_inputs = MoEEpTensors(
                hidden_states=replay_hidden,
                topk_ids=replay_ids,
                topk_weights=replay_weights,
            )
            anchor_replay = anchor_layer(replay_inputs).clone()
            torch.cuda.synchronize()
            anchor_replay_error = float(
                (anchor_replay.float() - graph_reference).pow(2).mean().sqrt()
                / graph_rms
            )
            graph_growth = graph_error / max(anchor_replay_error, 1e-12)
            result.growth = max(result.growth, graph_growth)
        if not bool(torch.isfinite(graph_output).all()):
            _destroy_layers(layers)
            raise SystemExit(f"[rank {rank}] FATAL: NaN/Inf in graph output")
        log(
            rank,
            f"graph correctness: cos={graph_cos:.5f} err={graph_error:.4f} "
            f"growth={graph_growth:.3f}",
        )
        forward = graph.replay
    else:
        forward = lambda: layer(inputs)

    if args.torch_profile:
        from torch.profiler import ProfilerActivity, profile

        for _ in range(args.warmup):
            forward()
        torch.cuda.synchronize()
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
        ) as profiler:
            for _ in range(args.iters):
                if world_size > 1:
                    import torch.distributed as dist

                    dist.barrier()
                forward()
            torch.cuda.synchronize()
        if rank == 0:
            print(
                profiler.key_averages().table(
                    sort_by="cuda_time_total",
                    row_limit=60,
                    max_name_column_width=60,
                ),
                flush=True,
            )
        if world_size > 1:
            import torch.distributed as dist

            dist.barrier()
        _destroy_layers(layers)
        return

    if args.nsys_capture:
        for _ in range(args.warmup):
            forward()
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
        result.p50_ms, result.p99_ms = time_rounds(forward, 0, args.iters, world_size)
        torch.cuda.profiler.stop()
    else:
        result.p50_ms, result.p99_ms = time_rounds(
            forward, args.warmup, args.iters, world_size
        )
    log(
        rank,
        f"e2e: p50={result.p50_ms:.3f} ms p99={result.p99_ms:.3f} ms",
    )

    if world_size == 1 and not args.skip_baseline:
        from types import SimpleNamespace

        from sm90_push_megamoe_baseline import sm90_moe_baseline_local

        baseline_inputs = SimpleNamespace(
            hidden_states=hidden_states,
            w13=w13,
            w2=w2,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )
        baseline_output = sm90_moe_baseline_local(baseline_inputs)
        baseline_cos = float(
            torch.nn.functional.cosine_similarity(
                baseline_output.float().flatten(),
                reference.float().flatten(),
                dim=0,
            )
        )
        result.baseline_p50_ms, _ = time_rounds(
            lambda: sm90_moe_baseline_local(baseline_inputs),
            max(args.warmup // 2, 2),
            max(args.iters // 2, 10),
        )
        result.speedup_p50 = result.baseline_p50_ms / result.p50_ms
        log(
            rank,
            f"baseline: p50={result.baseline_p50_ms:.3f} ms "
            f"cos={baseline_cos:.5f} speedup={result.speedup_p50:.3f}x",
        )

    records = [asdict(result)]
    if world_size > 1:
        import torch.distributed as dist

        gathered = [None] * world_size
        dist.all_gather_object(gathered, records)
        records = [record for rank_records in gathered for record in rank_records]

    failures = []
    if rank == 0:
        worst_cos = min(record["cos"] for record in records)
        growth_values = [
            record["growth"]
            for record in records
            if record["growth"] == record["growth"]
        ]
        worst_growth = max(growth_values) if growth_values else None
        worst_jitter = max(record["p99_ms"] / record["p50_ms"] for record in records)
        if args.assert_cos_min is not None and worst_cos < args.assert_cos_min:
            failures.append(f"cos {worst_cos:.5f} < {args.assert_cos_min}")
        if (
            args.assert_growth_max is not None
            and worst_growth is not None
            and worst_growth > args.assert_growth_max
        ):
            failures.append(f"growth {worst_growth:.3f} > {args.assert_growth_max}")
        if args.assert_speedup_min is not None:
            speedups = [
                record["speedup_p50"]
                for record in records
                if record["speedup_p50"] == record["speedup_p50"]
            ]
            if speedups and min(speedups) < args.assert_speedup_min:
                failures.append(
                    f"speedup {min(speedups):.3f} < {args.assert_speedup_min}"
                )
        if (
            args.assert_p99_jitter_max is not None
            and worst_jitter > args.assert_p99_jitter_max
        ):
            failures.append(
                f"p99/p50 jitter {worst_jitter:.3f} > {args.assert_p99_jitter_max}"
            )
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            with args.json.open("a", encoding="utf-8") as output_file:
                for record in records:
                    output_file.write(json_record(record) + "\n")
        for failure in failures:
            print(f"GATE FAIL: {failure}", flush=True)
        if not failures:
            print("ALL GATES PASS", flush=True)

    failure_count = [len(failures)]
    if world_size > 1:
        import torch.distributed as dist

        dist.broadcast_object_list(failure_count, src=0)
        dist.barrier()
    _destroy_layers(layers)
    if failure_count[0]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
