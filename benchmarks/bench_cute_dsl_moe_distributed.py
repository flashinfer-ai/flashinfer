#!/usr/bin/env python3
"""Distributed CuTe DSL DeepSeek-V3 MoE benchmark.

Compares two activation contracts over the same routed-MoE workload:

- W4A4 quantizes BF16 activations to NVFP4 before the MoE, using per-tensor
  scaling by default or per-token scaling with ``--use-per-token-activation``.
- W4A16 keeps activations in BF16 and decodes NVFP4 weights online.

Use torchrun to benchmark both real expert- and tensor-parallel communication:

    torchrun --standalone --nproc-per-node=8 \\
        benchmarks/bench_cute_dsl_moe_distributed.py

Run Nsight Systems mode directly to capture and report per-kernel breakdowns
for all four topology/activation combinations:

    python3 benchmarks/bench_cute_dsl_moe_distributed.py \
        --mode profile_nsys

Nsight Compute mode uses kernel replay to capture every local compute kernel
with the full metric set and embeds correlated sources found recursively under
the FlashInfer repository. It simulates the exact post-communication EP/TP
shapes in one process; use Nsight Systems mode for real communication:

    python3 benchmarks/bench_cute_dsl_moe_distributed.py \
        --mode profile_ncu --profile-iters 1

The workload stages remain available as NVTX ranges. Kernel attribution uses
those ranges rather than matching kernel names.
"""

import argparse
import csv
import gc
import itertools
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from bench_moe_deepseek import (
    BASE_INTERMEDIATE_SIZE,
    CFG,
    is_sm100_family,
)

DISTRIBUTED_TOKEN_COUNTS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
_PROFILE_CASE_ENV = "FLASHINFER_CUTE_DSL_MOE_PROFILE_CASE"
_PROFILE_MODES = ("profile_nsys", "profile_ncu")
_NCU_PROFILE_STAGES = ("routing", "activation prep/quant", "local MoE")
# Nsight global thread IDs reserve the low 24 bits for the thread ID.
_NSYS_GLOBAL_ID_PROCESS_MASK = -(1 << 24)


@dataclass(frozen=True)
class BenchVariant:
    name: str
    use_nvfp4_activations: bool


BENCH_VARIANTS = (
    BenchVariant("w4a4", use_nvfp4_activations=True),
    BenchVariant("w4a16", use_nvfp4_activations=False),
)


@dataclass
class DistributedBenchResult:
    num_tokens: int
    e2e_ms: float


@dataclass(frozen=True)
class KernelProfileRow:
    stage: str
    stage_launch: int
    name: str
    time_percent: float
    total_ms: float
    instances: int
    average_us: float


def _verbose_print(args, *values):
    if args.verbose:
        print(*values, flush=True)


def _parse_profile_case(profile_case):
    mode, variant_name = profile_case.split("::", maxsplit=1)
    variants = {variant.name: variant for variant in BENCH_VARIANTS}
    if mode not in ("ep", "tp") or variant_name not in variants:
        raise ValueError(f"Invalid {_PROFILE_CASE_ENV}: {profile_case}")
    return mode, variants[variant_name]


def _profile_worker_arguments(args, num_tokens):
    arguments = [
        "--num-tokens",
        str(num_tokens),
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--num-gpus",
        str(args.num_gpus),
        "--allreduce-backend",
        args.allreduce_backend,
        "--mode",
        args.mode,
        "--profile-iters",
        str(args.profile_iters),
    ]
    if args.use_per_token_activation:
        arguments.append("--use-per-token-activation")
    if not args.use_fused_finalize:
        arguments.append("--no-fused-finalize")
    if not args.enable_pdl:
        arguments.append("--no-pdl")
    if args.verbose:
        arguments.append("--verbose")
    return arguments


def _load_nsys_kernel_rows(nsys, report, verbose):
    sqlite_report = report.with_suffix(".sqlite")
    command = [
        nsys,
        "export",
        "--type=sqlite",
        "--force-overwrite=true",
        "--output",
        str(sqlite_report),
        str(report),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    if verbose:
        print(completed.stdout, end="")
        print(completed.stderr, end="", file=sys.stderr)

    with sqlite3.connect(sqlite_report) as connection:
        raw_rows = connection.execute(
            """
            WITH stage_ranges AS (
                SELECT
                    nvtx.start,
                    nvtx.end,
                    nvtx.globalTid,
                    substr(COALESCE(nvtx.text, range_name.value), 8) AS stage
                FROM NVTX_EVENTS AS nvtx
                LEFT JOIN StringIds AS range_name ON range_name.id = nvtx.textId
                WHERE
                    nvtx.end IS NOT NULL
                    AND COALESCE(nvtx.text, range_name.value) LIKE 'stage::%'
            ),
            stage_kernel_launches AS (
                SELECT
                    stage_ranges.stage,
                    kernel.shortName,
                    kernel.start AS kernel_start,
                    kernel.end AS kernel_end,
                    ROW_NUMBER() OVER (
                        PARTITION BY
                            stage_ranges.start,
                            stage_ranges.end,
                            stage_ranges.globalTid
                        ORDER BY kernel.start, kernel.gridId
                    ) AS stage_launch
                FROM stage_ranges
                JOIN CUPTI_ACTIVITY_KIND_RUNTIME AS runtime ON
                    runtime.start >= stage_ranges.start
                    AND runtime.start <= stage_ranges.end
                    AND (runtime.globalTid & :process_mask) =
                        (stage_ranges.globalTid & :process_mask)
                JOIN CUPTI_ACTIVITY_KIND_KERNEL AS kernel ON
                    kernel.correlationId = runtime.correlationId
                    AND kernel.globalPid =
                        (runtime.globalTid & :process_mask)
            )
            SELECT
                stage_kernel_launches.stage,
                stage_kernel_launches.stage_launch,
                kernel_name.value,
                SUM(
                    stage_kernel_launches.kernel_end -
                        stage_kernel_launches.kernel_start
                ) AS total_ns,
                COUNT(*) AS instances,
                AVG(
                    stage_kernel_launches.kernel_end -
                        stage_kernel_launches.kernel_start
                ) AS average_ns
            FROM stage_kernel_launches
            JOIN StringIds AS kernel_name ON
                kernel_name.id = stage_kernel_launches.shortName
            GROUP BY
                stage_kernel_launches.stage,
                stage_kernel_launches.stage_launch,
                stage_kernel_launches.shortName
            ORDER BY
                MIN(stage_kernel_launches.kernel_start),
                stage_kernel_launches.stage,
                stage_kernel_launches.stage_launch,
                stage_kernel_launches.shortName
            """,
            {"process_mask": _NSYS_GLOBAL_ID_PROCESS_MASK},
        ).fetchall()

    total_kernel_ns = sum(row[3] for row in raw_rows)
    if not total_kernel_ns:
        raise RuntimeError(f"Nsight Systems reported no staged kernels in {report}")
    return [
        KernelProfileRow(
            stage=stage,
            stage_launch=stage_launch,
            name=name,
            time_percent=total_ns / total_kernel_ns * 100,
            total_ms=total_ns / 1e6,
            instances=instances,
            average_us=average_ns / 1e3,
        )
        for stage, stage_launch, name, total_ns, instances, average_ns in raw_rows
    ]


def _print_nsys_kernel_breakdown(mode, variant, num_tokens, num_gpus, rows, report):
    print(
        f"\nNsight Systems kernel breakdown: {mode.upper()}{num_gpus} "
        f"{variant.upper()}, {num_tokens} global tokens"
    )
    print(
        "stage                     | launch | GPU time | total (ms) | "
        "instances | avg (us) | kernel"
    )
    csv_report = report.with_suffix(".kernels.csv")
    with csv_report.open("w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            (
                "mode",
                "variant",
                "num_tokens",
                "num_gpus",
                "stage",
                "stage_launch",
                "gpu_time_percent",
                "total_ms",
                "instances",
                "average_us",
                "kernel",
            )
        )
        for row in rows:
            display_name = row.name if len(row.name) <= 96 else f"{row.name[:93]}..."
            print(
                f"{row.stage:<25} | {row.stage_launch:>6} | "
                f"{row.time_percent:>7.2f}% | "
                f"{row.total_ms:>10.3f} | {row.instances:>9} | "
                f"{row.average_us:>8.3f} | {display_name}"
            )
            csv_writer.writerow(
                (
                    mode,
                    variant,
                    num_tokens,
                    num_gpus,
                    row.stage,
                    row.stage_launch,
                    f"{row.time_percent:.6f}",
                    f"{row.total_ms:.6f}",
                    row.instances,
                    f"{row.average_us:.6f}",
                    row.name,
                )
            )
    print("GPU time is aggregated across all ranks and captured iterations.")
    print(f"Nsight Systems report: {report}")
    print(f"Kernel CSV: {csv_report}")


def _run_nsys_profiles(args, token_counts):
    nsys = shutil.which("nsys")
    torchrun = shutil.which("torchrun")
    if nsys is None:
        raise RuntimeError("--mode profile_nsys requires Nsight Systems (nsys)")
    if torchrun is None:
        raise RuntimeError("--mode profile_nsys requires torchrun")

    if args.nsys_output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="flashinfer-cute-dsl-moe-nsys-"))
    else:
        output_dir = Path(args.nsys_output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

    script = Path(__file__).resolve()
    for num_tokens, mode, variant in itertools.product(
        token_counts, ("ep", "tp"), BENCH_VARIANTS
    ):
        profile_label = f"{mode}::{variant.name}"
        output = output_dir / f"{mode}_{variant.name}_t{num_tokens}"
        env = os.environ.copy()
        env[_PROFILE_CASE_ENV] = profile_label
        command = [
            nsys,
            "profile",
            "--force-overwrite=true",
            "--sample=none",
            "--cpuctxsw=none",
            "--trace=cuda,nvtx,nccl",
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=stop",
            f"--show-output={'true' if args.verbose else 'false'}",
            f"--output={output}",
            torchrun,
            "--standalone",
            f"--nproc-per-node={args.num_gpus}",
            str(script),
            *_profile_worker_arguments(args, num_tokens),
        ]
        print(
            f"\nCapturing {mode.upper()}{args.num_gpus} "
            f"{variant.name.upper()} at {num_tokens} tokens "
            "with Nsight Systems...",
            flush=True,
        )
        _verbose_print(args, "Command:", " ".join(command))
        subprocess.run(command, check=True, env=env)

        report = output.with_suffix(".nsys-rep")
        rows = _load_nsys_kernel_rows(nsys, report, args.verbose)
        _print_nsys_kernel_breakdown(
            mode,
            variant.name,
            num_tokens,
            args.num_gpus,
            rows,
            report,
        )

    print(f"\nNsight Systems reports: {output_dir}")
    return 0


def _run_ncu_profiles(args, token_counts):
    ncu = shutil.which("ncu")
    if ncu is None:
        raise RuntimeError("--mode profile_ncu requires Nsight Compute (ncu)")

    if args.ncu_output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="flashinfer-cute-dsl-moe-ncu-"))
    else:
        output_dir = Path(args.ncu_output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()
    source_root = script.parents[1]
    lineinfo_cache = output_dir / "cute_dsl_lineinfo_cache"
    lineinfo_cache.mkdir(exist_ok=True)
    for num_tokens, mode, variant in itertools.product(
        token_counts, ("ep", "tp"), BENCH_VARIANTS
    ):
        profile_label = f"{mode}::{variant.name}"
        output = output_dir / f"{mode}_{variant.name}_t{num_tokens}"
        log_file = output.with_suffix(".ncu.log")
        env = os.environ.copy()
        env[_PROFILE_CASE_ENV] = profile_label
        env["CUTE_DSL_CACHE_DIR"] = str(lineinfo_cache)
        env["CUTE_DSL_LINEINFO"] = "1"
        command = [
            ncu,
            "--set=full",
            "--target-processes=all",
            "--replay-mode=kernel",
            "--profile-from-start=off",
            "--nvtx",
            *(
                argument
                for stage in _NCU_PROFILE_STAGES
                for argument in ("--nvtx-include", f"stage::{stage}/")
            ),
            "--import-source=yes",
            f"--source-folders={source_root}",
            "--print-summary=per-kernel",
            "--force-overwrite",
            f"--log-file={log_file}",
            f"--export={output}",
            sys.executable,
            str(script),
            *_profile_worker_arguments(args, num_tokens),
        ]
        print(
            f"\nCapturing non-communication kernels for "
            f"{mode.upper()}{args.num_gpus} {variant.name.upper()} "
            f"at {num_tokens} tokens with Nsight Compute...",
            flush=True,
        )
        _verbose_print(args, "Command:", " ".join(command))
        subprocess.run(command, check=True, env=env)

        reports = sorted(output_dir.glob(f"{output.name}*.ncu-rep"))
        if not reports:
            raise RuntimeError(f"Nsight Compute did not create a report for {output}")
        for report in reports:
            print(f"Nsight Compute report: {report}")
        print(f"Nsight Compute log: {log_file}")

    print(f"\nNsight Compute reports: {output_dir}")
    return 0


def _distributed_moe_config(
    args,
    variant,
    intermediate_size,
    num_local_experts,
    local_expert_offset,
    tune_max_num_tokens,
):
    from flashinfer.fused_moe import (
        BackendOptions,
        CuteDslConfig,
        ExecutionConfig,
        ExpertConfig,
        MoEConfig,
        QuantConfig,
        QuantVariant,
        RoutingConfig,
    )

    return MoEConfig(
        routing=RoutingConfig(num_experts=CFG.num_experts, top_k=CFG.top_k),
        quant=QuantConfig(
            variant=(
                QuantVariant.NVFP4
                if variant.use_nvfp4_activations
                else QuantVariant.W4A16
            ),
            per_token_scale=(
                args.use_per_token_activation and variant.use_nvfp4_activations
            ),
        ),
        experts=ExpertConfig(
            intermediate_size=intermediate_size,
            local_expert_offset=local_expert_offset,
            local_num_experts=num_local_experts,
        ),
        backend=BackendOptions(candidates=(CuteDslConfig(),)),
        execution=ExecutionConfig(
            enable_pdl=args.enable_pdl,
            tune_max_num_tokens=tune_max_num_tokens,
            use_fused_finalize=args.use_fused_finalize,
        ),
    )


def _create_distributed_weights(num_local_experts, intermediate_size, rank, device):
    generator = torch.Generator(device=device).manual_seed(1000 + rank)
    w13 = (
        torch.randn(
            num_local_experts,
            2 * intermediate_size,
            CFG.hidden_size,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        / 10
    )
    w2 = (
        torch.randn(
            num_local_experts,
            CFG.hidden_size,
            intermediate_size,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        / 10
    )
    return w13, w2


def _token_partition(num_tokens, rank, world_size):
    tokens_per_rank, remainder = divmod(num_tokens, world_size)
    local_num_tokens = tokens_per_rank + int(rank < remainder)
    token_offset = rank * tokens_per_rank + min(rank, remainder)
    return local_num_tokens, token_offset


def _create_distributed_inputs(num_tokens, rank, world_size, device):
    local_num_tokens, token_offset = _token_partition(num_tokens, rank, world_size)
    input_generator = torch.Generator(device=device).manual_seed(42 + rank)
    hidden_states = (
        torch.randn(
            local_num_tokens,
            CFG.hidden_size,
            dtype=torch.bfloat16,
            device=device,
            generator=input_generator,
        )
        / 10
    )
    routing_generator = torch.Generator(device=device).manual_seed(137)
    global_router_logits = torch.randn(
        num_tokens,
        CFG.num_experts,
        dtype=torch.float32,
        device=device,
        generator=routing_generator,
    )
    router_logits = global_router_logits.narrow(0, token_offset, local_num_tokens)
    bias_generator = torch.Generator(device=device).manual_seed(911)
    routing_bias = (
        torch.randn(
            CFG.num_experts,
            dtype=torch.bfloat16,
            device=device,
            generator=bias_generator,
        )
        * 0.01
    ).float()
    return hidden_states, router_logits, global_router_logits, routing_bias


def _route_tokens(router_logits, routing_bias, topk_values, topk_indices):
    from flashinfer.fused_moe import fused_topk_deepseek

    if router_logits.shape[0] == 0:
        return
    fused_topk_deepseek(
        scores=router_logits,
        bias=routing_bias,
        n_group=CFG.n_group,
        topk_group=CFG.topk_group,
        topk=CFG.top_k,
        routed_scaling_factor=CFG.routed_scaling_factor,
        topk_values=topk_values,
        topk_indices=topk_indices,
    )


def _max_rank_sample(value, dist, device):
    sample = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(sample, op=dist.ReduceOp.MAX)
    return sample.item()


def _run_profile_iteration(stage_calls):
    for name, stage_call in stage_calls:
        torch.cuda.nvtx.range_push(f"stage::{name}")
        stage_call()
        torch.cuda.nvtx.range_pop()


def _run_distributed_iterations(
    args,
    num_tokens,
    run_once,
    profile_once,
    l2_flush,
    dist,
    device,
    profile_label,
):
    for _ in range(args.warmup):
        run_once()
    torch.cuda.synchronize()
    dist.barrier()

    if args.mode in _PROFILE_MODES:
        torch.cuda.cudart().cudaProfilerStart()
        torch.cuda.nvtx.range_push(profile_label)
        for _ in range(args.profile_iters):
            l2_flush.zero_()
            torch.cuda.synchronize()
            dist.barrier()
            profile_once()
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.nvtx.range_pop()
        torch.cuda.cudart().cudaProfilerStop()
        return None

    samples = []
    for _ in range(args.iters):
        l2_flush.zero_()
        torch.cuda.synchronize()
        dist.barrier()
        e2e_start = torch.cuda.Event(enable_timing=True)
        e2e_end = torch.cuda.Event(enable_timing=True)
        e2e_start.record()
        run_once()
        e2e_end.record()
        e2e_end.synchronize()
        samples.append(_max_rank_sample(e2e_start.elapsed_time(e2e_end), dist, device))
    return DistributedBenchResult(num_tokens, float(np.median(samples)))


def _benchmark_distributed_ep(
    args,
    variant,
    num_tokens,
    max_tokens_per_rank_budget,
    rank,
    world_size,
    device,
):
    import torch.distributed as dist

    from flashinfer.autotuner import autotune
    from flashinfer.comm import MoeAlltoAll
    from flashinfer.comm.mapping import Mapping
    from flashinfer.comm.mnnvl import MnnvlConfig, TorchDistBackend
    from flashinfer.quantization.nvfp4_quantization_utils import (
        current_nvfp4_4over6_config,
        make_nvfp4_global_scale,
    )
    from flashinfer.testing.utils import get_l2_cache_size

    def run_setup_phase(name, fn):
        prefix = (
            f"[rank {rank}] EP setup variant={variant.name} tokens={num_tokens} {name}"
        )
        _verbose_print(args, f"{prefix}: start")
        result = fn()
        torch.cuda.synchronize()
        _verbose_print(args, f"{prefix}: complete")
        dist.barrier()
        return result

    local_num_tokens, _ = _token_partition(num_tokens, rank, world_size)
    runtime_max_tokens_per_rank = (num_tokens + world_size - 1) // world_size
    num_local_experts = CFG.num_experts // world_size
    local_expert_offset = rank * num_local_experts
    layer, weight_pack = run_setup_phase(
        "local MoE construction",
        lambda: _create_distributed_moe_layer(
            args,
            variant,
            CFG.intermediate_size,
            num_local_experts,
            local_expert_offset,
            max_tokens_per_rank_budget * world_size,
            rank,
            device,
        ),
    )
    mapping = Mapping(
        rank=rank,
        tp_size=world_size,
        moe_ep_size=world_size,
        world_size=world_size,
        gpus_per_node=world_size,
        pp_size=1,
        cp_size=1,
    )
    workspace_size_per_rank = MoeAlltoAll.get_moe_workspace_size_per_rank(
        world_size,
        CFG.top_k,
        max_tokens_per_rank_budget,
        CFG.hidden_size,
    )
    moe_a2a = run_setup_phase(
        "MoeAlltoAll construction",
        lambda: MoeAlltoAll(
            mapping=mapping,
            max_num_tokens=max_tokens_per_rank_budget,
            top_k=CFG.top_k,
            num_experts=CFG.num_experts,
            workspace_size_per_rank=workspace_size_per_rank,
            mnnvl_config=MnnvlConfig(comm_backend=TorchDistBackend(dist.group.WORLD)),
        ),
    )
    hidden_states, router_logits, _, routing_bias = _create_distributed_inputs(
        num_tokens, rank, world_size, device
    )
    topk_values = torch.empty(
        local_num_tokens, CFG.top_k, dtype=torch.float32, device=device
    )
    topk_indices = torch.empty(
        local_num_tokens, CFG.top_k, dtype=torch.int32, device=device
    )
    global_scale = None
    if variant.use_nvfp4_activations:
        global_scale = make_nvfp4_global_scale(
            hidden_states,
            per_token_activation=args.use_per_token_activation,
            global_scale=1.0,
            nvfp4_4over6_config=current_nvfp4_4over6_config(),
        )
    l2_flush = torch.empty(2 * get_l2_cache_size(), dtype=torch.int8, device=device)

    def dispatch():
        # Match SGLang's FlashInfer dispatcher: send BF16 activations with the
        # routing payloads, then sanitize routes this rank does not own.
        recv_hidden_states, recv_topk_indices, recv_topk_values = moe_a2a.dispatch(
            topk_indices,
            [hidden_states, topk_indices, topk_values],
            runtime_max_tokens_per_rank,
            invalid_token_expert_id=CFG.num_experts,
            expert_id_payload_index=1,
        )
        num_received_tokens = world_size * runtime_max_tokens_per_rank
        return (
            recv_hidden_states.view(num_received_tokens, CFG.hidden_size),
            recv_topk_indices.view(num_received_tokens, CFG.top_k),
            recv_topk_values.view(num_received_tokens, CFG.top_k),
        )

    def compute(recv_hidden_states, recv_topk_indices, recv_topk_values):
        activation_pack = _make_distributed_activation_pack(
            args,
            variant,
            recv_hidden_states,
            recv_topk_indices,
            recv_topk_values,
            global_scale,
        )
        return layer(activation_pack, weight_pack)

    def combine(local_output):
        # SGLang's CuTe DSL runner does not write into the A2A workspace, so
        # combine stages the local output before returning it to source ranks.
        return moe_a2a.combine(
            local_output.view(world_size, runtime_max_tokens_per_rank, CFG.hidden_size),
            runtime_max_tokens_per_rank,
        )

    def run_once():
        _route_tokens(router_logits, routing_bias, topk_values, topk_indices)
        combine(compute(*dispatch()))

    profile_state = {}

    def profile_dispatch():
        profile_state["dispatched"] = dispatch()

    def profile_activation_prep():
        profile_state["activation_pack"] = _make_distributed_activation_pack(
            args,
            variant,
            *profile_state["dispatched"],
            global_scale,
        )

    def profile_local_moe():
        profile_state["local_output"] = layer(
            profile_state["activation_pack"], weight_pack
        )

    def profile_once():
        _run_profile_iteration(
            (
                (
                    "routing",
                    lambda: _route_tokens(
                        router_logits, routing_bias, topk_values, topk_indices
                    ),
                ),
                ("dispatch", profile_dispatch),
                ("activation prep/quant", profile_activation_prep),
                ("local MoE", profile_local_moe),
                ("combine", lambda: combine(profile_state["local_output"])),
            )
        )

    # Finish the setup collective before CuTe DSL selects a tactic. Inference
    # warmup likewise tunes the local runner outside the steady-state A2A phase.
    run_setup_phase(
        "routing",
        lambda: _route_tokens(router_logits, routing_bias, topk_values, topk_indices),
    )
    dispatched_inputs = run_setup_phase("dispatch", dispatch)
    tuning_inputs = run_setup_phase(
        "preserve dispatched inputs",
        lambda: tuple(tensor.clone() for tensor in dispatched_inputs),
    )
    run_setup_phase(
        "combine",
        lambda: combine(
            torch.zeros(
                world_size * runtime_max_tokens_per_rank,
                CFG.hidden_size,
                dtype=torch.bfloat16,
                device=device,
            )
        ),
    )

    def tune_local_moe():
        with autotune(True):
            return compute(*tuning_inputs)

    run_setup_phase("CuTe DSL tactic selection", tune_local_moe)

    return _run_distributed_iterations(
        args,
        num_tokens,
        run_once,
        profile_once,
        l2_flush,
        dist,
        device,
        f"ep::{variant.name}",
    )


def _make_distributed_activation_pack(
    args,
    variant,
    hidden_states,
    topk_indices,
    topk_values,
    global_scale,
):
    from flashinfer.fused_moe import MoEActivationPack

    if not variant.use_nvfp4_activations:
        return MoEActivationPack(
            hidden_states_q=hidden_states,
            hidden_states_scale=None,
            topk_ids=topk_indices,
            topk_weights=topk_values,
        )

    from flashinfer import SfLayout, nvfp4_quantize
    from flashinfer.fp4_quantization import fp4_quantize

    if args.use_per_token_activation:
        hidden_states_q, hidden_states_scale, per_token_scale = nvfp4_quantize(
            hidden_states,
            global_scale,
            sfLayout=SfLayout.layout_linear,
            per_token_activation=True,
            backend="cute-dsl",
        )
    else:
        hidden_states_q, hidden_states_scale = fp4_quantize(
            hidden_states,
            global_scale,
            16,
            False,
            False,
            backend="cute-dsl",
        )
        per_token_scale = None
    if hidden_states_scale.dim() > 2:
        hidden_states_scale = hidden_states_scale.squeeze(-1)
    return MoEActivationPack(
        hidden_states_q=hidden_states_q,
        hidden_states_scale=hidden_states_scale,
        topk_ids=topk_indices,
        topk_weights=topk_values,
        per_token_scale=per_token_scale,
    )


def _create_distributed_moe_layer(
    args,
    variant,
    intermediate_size,
    num_local_experts,
    local_expert_offset,
    tune_max_num_tokens,
    rank,
    device,
):
    from flashinfer.fused_moe import CuteDslConfig, MoELayer, MoEWeightPack

    moe_config = _distributed_moe_config(
        args,
        variant,
        intermediate_size,
        num_local_experts,
        local_expert_offset,
        tune_max_num_tokens,
    )
    w13, w2 = _create_distributed_weights(
        num_local_experts, intermediate_size, rank, device
    )
    weight_pack = MoEWeightPack()
    weight_pack.prepare_for(
        "cute_dsl_nvfp4",
        CuteDslConfig.prepare_weights(
            w13,
            w2,
            num_local_experts=num_local_experts,
            hidden_size=CFG.hidden_size,
            intermediate_size=intermediate_size,
            device=device,
        ),
    )
    return MoELayer(moe_config, device=device), weight_pack


def _run_ncu_compute_profile(args, num_tokens, mode, variant):
    from flashinfer.quantization.nvfp4_quantization_utils import (
        current_nvfp4_4over6_config,
        make_nvfp4_global_scale,
    )
    from flashinfer.testing.utils import get_l2_cache_size

    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    if mode == "ep":
        rows = args.num_gpus * ((num_tokens + args.num_gpus - 1) // args.num_gpus)
        intermediate_size = CFG.intermediate_size
        num_local_experts = CFG.num_experts // args.num_gpus
    else:
        rows = num_tokens
        intermediate_size = BASE_INTERMEDIATE_SIZE // args.num_gpus
        num_local_experts = CFG.num_experts

    layer, weight_pack = _create_distributed_moe_layer(
        args,
        variant,
        intermediate_size,
        num_local_experts,
        0,
        rows,
        0,
        device,
    )
    hidden_states, _, router_logits, routing_bias = _create_distributed_inputs(
        rows, 0, 1, device
    )
    topk_values = torch.empty(rows, CFG.top_k, dtype=torch.float32, device=device)
    topk_indices = torch.empty(rows, CFG.top_k, dtype=torch.int32, device=device)
    global_scale = None
    if variant.use_nvfp4_activations:
        global_scale = make_nvfp4_global_scale(
            hidden_states,
            per_token_activation=args.use_per_token_activation,
            global_scale=1.0,
            nvfp4_4over6_config=current_nvfp4_4over6_config(),
        )
    l2_flush = torch.empty(2 * get_l2_cache_size(), dtype=torch.int8, device=device)
    profile_state = {}

    def route():
        _route_tokens(router_logits, routing_bias, topk_values, topk_indices)

    def prepare_activation():
        profile_state["activation_pack"] = _make_distributed_activation_pack(
            args,
            variant,
            hidden_states,
            topk_indices,
            topk_values,
            global_scale,
        )

    def run_local_moe():
        profile_state["local_output"] = layer(
            profile_state["activation_pack"], weight_pack
        )

    stage_calls = (
        ("routing", route),
        ("activation prep/quant", prepare_activation),
        ("local MoE", run_local_moe),
    )
    for _ in range(args.warmup):
        for _, stage_call in stage_calls:
            stage_call()
    torch.cuda.synchronize()

    print(
        f"Profiling simulated post-communication {mode.upper()}{args.num_gpus} "
        f"{variant.name.upper()}: {num_tokens} global tokens, {rows} local rows",
        flush=True,
    )
    torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.nvtx.range_push(f"{mode}::{variant.name}")
    for _ in range(args.profile_iters):
        l2_flush.zero_()
        torch.cuda.synchronize()
        _run_profile_iteration(stage_calls)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()
    return 0


def _benchmark_distributed_tp(
    args, variant, num_tokens, rank, world_size, device, report_backend=False
):
    import torch.distributed as dist

    from flashinfer.autotuner import autotune
    from flashinfer.comm import (
        AllReduceFusionPattern,
        allreduce_fusion,
        create_allreduce_fusion_workspace,
    )
    from flashinfer.comm.mnnvl import TorchDistBackend
    from flashinfer.quantization.nvfp4_quantization_utils import (
        current_nvfp4_4over6_config,
        make_nvfp4_global_scale,
    )
    from flashinfer.testing.utils import get_l2_cache_size

    intermediate_size = BASE_INTERMEDIATE_SIZE // world_size
    layer, weight_pack = _create_distributed_moe_layer(
        args,
        variant,
        intermediate_size,
        CFG.num_experts,
        0,
        num_tokens,
        rank,
        device,
    )
    local_hidden_states, _, router_logits, routing_bias = _create_distributed_inputs(
        num_tokens, rank, world_size, device
    )
    topk_values = torch.empty(num_tokens, CFG.top_k, dtype=torch.float32, device=device)
    topk_indices = torch.empty(num_tokens, CFG.top_k, dtype=torch.int32, device=device)
    hidden_states = torch.empty(
        num_tokens,
        CFG.hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )
    gathered_hidden_states = list(
        hidden_states.split(
            [_token_partition(num_tokens, r, world_size)[0] for r in range(world_size)]
        )
    )
    dist.all_gather(gathered_hidden_states, local_hidden_states)
    global_scale = None
    if variant.use_nvfp4_activations:
        global_scale = make_nvfp4_global_scale(
            hidden_states,
            per_token_activation=args.use_per_token_activation,
            nvfp4_4over6_config=current_nvfp4_4over6_config(),
        )
    allreduce_output = torch.empty_like(hidden_states)
    workspace = create_allreduce_fusion_workspace(
        backend=args.allreduce_backend,
        world_size=world_size,
        rank=rank,
        max_token_num=num_tokens,
        hidden_dim=CFG.hidden_size,
        dtype=torch.bfloat16,
        gpus_per_node=world_size,
        comm_backend=TorchDistBackend(),
    )
    if rank == 0 and report_backend and args.verbose:
        print(f"FlashInfer all-reduce backend: {workspace.backend}")
    l2_flush = torch.empty(2 * get_l2_cache_size(), dtype=torch.int8, device=device)

    def run_once():
        dist.all_gather(gathered_hidden_states, local_hidden_states)
        _route_tokens(router_logits, routing_bias, topk_values, topk_indices)
        activation_pack = _make_distributed_activation_pack(
            args,
            variant,
            hidden_states,
            topk_indices,
            topk_values,
            global_scale,
        )
        local_output = layer(activation_pack, weight_pack)
        allreduce_fusion(
            input=local_output,
            workspace=workspace,
            pattern=AllReduceFusionPattern.kAllReduce,
            launch_with_pdl=args.enable_pdl,
            output=allreduce_output,
        )

    profile_state = {}

    def profile_activation_prep():
        profile_state["activation_pack"] = _make_distributed_activation_pack(
            args,
            variant,
            hidden_states,
            topk_indices,
            topk_values,
            global_scale,
        )

    def profile_local_moe():
        profile_state["local_output"] = layer(
            profile_state["activation_pack"], weight_pack
        )

    def profile_all_reduce():
        allreduce_fusion(
            input=profile_state["local_output"],
            workspace=workspace,
            pattern=AllReduceFusionPattern.kAllReduce,
            launch_with_pdl=args.enable_pdl,
            output=allreduce_output,
        )

    def profile_once():
        _run_profile_iteration(
            (
                (
                    "all-gather",
                    lambda: dist.all_gather(
                        gathered_hidden_states, local_hidden_states
                    ),
                ),
                (
                    "routing",
                    lambda: _route_tokens(
                        router_logits, routing_bias, topk_values, topk_indices
                    ),
                ),
                ("activation prep/quant", profile_activation_prep),
                ("local MoE", profile_local_moe),
                ("all-reduce", profile_all_reduce),
            )
        )

    _route_tokens(router_logits, routing_bias, topk_values, topk_indices)
    tuning_activation_pack = _make_distributed_activation_pack(
        args,
        variant,
        hidden_states,
        topk_indices,
        topk_values,
        global_scale,
    )
    with autotune(True):
        layer(tuning_activation_pack, weight_pack)
    torch.cuda.synchronize()
    dist.barrier()

    try:
        return _run_distributed_iterations(
            args,
            num_tokens,
            run_once,
            profile_once,
            l2_flush,
            dist,
            device,
            f"tp::{variant.name}",
        )
    finally:
        workspace.destroy()


def _run_parallel_mode(
    args,
    token_counts,
    mode,
    rank,
    world_size,
    device,
    dist,
    selected_variant=None,
):
    if rank == 0:
        if args.mode in _PROFILE_MODES:
            print(
                f"\nMode: real {mode.upper()}{world_size}, "
                "profiling=NVTX stages, cache=cold L2"
            )
        else:
            print(
                f"\nMode: real {mode.upper()}{world_size}, "
                "timing=max rank CUDA events, cache=cold L2"
            )
            if selected_variant is None:
                print(
                    "global tokens | W4A4 + activation quant (ms) | "
                    "W4A16 (ms) | W4A16 / W4A4"
                )
            else:
                print(f"global tokens | {selected_variant.upper()} (ms)")

    max_tokens_per_rank_budget = None
    if mode == "ep":
        # SGLang reserves at least 4096 dispatch tokens per rank. The live
        # collective still uses each case's runtime_max_tokens_per_rank.
        max_tokens_per_rank_budget = max(
            4096,
            max(
                (num_tokens + world_size - 1) // world_size
                for num_tokens in token_counts
            ),
        )

    reported_backend = False
    for num_tokens in token_counts:
        row = {}
        variants = (
            BENCH_VARIANTS
            if selected_variant is None
            else tuple(
                variant
                for variant in BENCH_VARIANTS
                if variant.name == selected_variant
            )
        )
        for variant in variants:
            if mode == "ep":
                result = _benchmark_distributed_ep(
                    args,
                    variant,
                    num_tokens,
                    max_tokens_per_rank_budget,
                    rank,
                    world_size,
                    device,
                )
            else:
                result = _benchmark_distributed_tp(
                    args,
                    variant,
                    num_tokens,
                    rank,
                    world_size,
                    device,
                    report_backend=not reported_backend,
                )
                reported_backend = True

            if rank == 0 and result is not None:
                row[variant.name] = result.e2e_ms
                print(
                    "DISTRIBUTED_CSV,"
                    f"{mode},{variant.name},{num_tokens},{world_size},"
                    f"{result.e2e_ms:.6f}"
                )
            dist.barrier()
            gc.collect()
            torch.cuda.empty_cache()

        if rank == 0 and len(row) == 2:
            ratio = row["w4a16"] / row["w4a4"]
            print(
                f"{num_tokens:>13} | {row['w4a4']:>28.3f} | "
                f"{row['w4a16']:>10.3f} | {ratio:>13.3f}x"
            )
        elif rank == 0 and row:
            value = row[selected_variant]
            print(f"{num_tokens:>13} | {value:>11.3f}")


def _run_distributed_benchmark(args, token_counts):
    import torch.distributed as dist

    local_rank = int(os.environ["LOCAL_RANK"])
    _verbose_print(args, f"[local rank {local_rank}] distributed setup: start")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    _verbose_print(
        args,
        f"[local rank {local_rank}] NCCL process-group initialization: start",
    )
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    _verbose_print(
        args,
        f"[rank {rank}] NCCL process-group initialization: complete "
        f"(world_size={world_size})",
    )

    try:
        if world_size != args.num_gpus:
            raise ValueError(
                f"torchrun world size {world_size} does not match --num-gpus "
                f"{args.num_gpus}"
            )
        if rank == 0:
            print("\nDeepSeek-V3 distributed CuTe DSL MoE benchmark")

        selected_mode = None
        selected_variant_name = None
        profile_case = os.environ.get(_PROFILE_CASE_ENV)
        if profile_case is not None:
            selected_mode, selected_variant = _parse_profile_case(profile_case)
            selected_variant_name = selected_variant.name

        modes = ("ep", "tp") if selected_mode is None else (selected_mode,)
        for mode in modes:
            _run_parallel_mode(
                args,
                token_counts,
                mode,
                rank,
                world_size,
                device,
                dist,
                selected_variant_name,
            )
            dist.barrier()
        return 0
    finally:
        dist.destroy_process_group()


def main():
    warnings.filterwarnings(
        "ignore",
        message="cold_l2_cache=True but no GPU tensors found.*",
        module=r"flashinfer\.testing\.utils",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-tokens",
        type=str,
        default=None,
        help=(
            "Comma-separated global token counts (default: powers of two from 1 to "
            "4096 for benchmark, 32 and 4096 for profiling)."
        ),
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=8,
        help=(
            "Number of torchrun processes, up to eight GPUs on one node. In "
            "Nsight Compute mode, this is the logical EP/TP degree."
        ),
    )
    parser.add_argument(
        "--allreduce-backend",
        choices=["auto", "trtllm", "mnnvl"],
        default="auto",
        help="FlashInfer all-reduce backend for TP.",
    )
    parser.add_argument(
        "--use-per-token-activation",
        action="store_true",
        help="Use per-token NVFP4 activation scaling for the W4A4 case.",
    )
    parser.add_argument(
        "--no-fused-finalize",
        action="store_false",
        dest="use_fused_finalize",
        help="Use deterministic two-stage finalize.",
    )
    parser.add_argument(
        "--no-pdl",
        action="store_false",
        dest="enable_pdl",
        help="Disable Programmatic Dependent Launch.",
    )
    parser.add_argument(
        "--mode",
        choices=["benchmark", *_PROFILE_MODES],
        default="benchmark",
        help="Benchmark, capture Nsight Systems timelines, or capture Nsight Compute metrics.",
    )
    parser.add_argument(
        "--profile-iters",
        type=int,
        default=10,
        help="Number of iterations captured in each profiler run.",
    )
    parser.add_argument(
        "--nsys-output-dir",
        type=str,
        default=None,
        help="Directory for Nsight Systems reports (default: a temporary directory).",
    )
    parser.add_argument(
        "--ncu-output-dir",
        type=str,
        default=None,
        help="Directory for Nsight Compute reports (default: a temporary directory).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print distributed setup progress and profiler worker output.",
    )
    args = parser.parse_args()

    if not 1 <= args.num_gpus <= 8:
        parser.error("--num-gpus must be between 1 and 8")
    if CFG.num_experts % args.num_gpus != 0:
        parser.error(f"--num-gpus must divide the expert count ({CFG.num_experts})")
    if BASE_INTERMEDIATE_SIZE % args.num_gpus != 0:
        parser.error(
            "--num-gpus must divide the expert intermediate size "
            f"({BASE_INTERMEDIATE_SIZE})"
        )
    if args.profile_iters < 1:
        parser.error("--profile-iters must be positive")
    if not is_sm100_family():
        print("ERROR: Requires SM100 family GPU (Blackwell: SM100, SM103)")
        return 1

    if args.num_tokens:
        tokens = [int(value) for value in args.num_tokens.split(",")]
    elif args.mode in _PROFILE_MODES:
        tokens = [32, 4096]
    else:
        tokens = DISTRIBUTED_TOKEN_COUNTS

    profile_case = os.environ.get(_PROFILE_CASE_ENV)
    if args.mode == "profile_ncu" and profile_case is not None:
        if "LOCAL_RANK" in os.environ:
            parser.error("profile_ncu workers must run without torchrun")
        if len(tokens) != 1:
            parser.error("profile_ncu workers require exactly one token count")
        mode, variant = _parse_profile_case(profile_case)
        return _run_ncu_compute_profile(args, tokens[0], mode, variant)
    if args.mode in _PROFILE_MODES and profile_case is None:
        if "LOCAL_RANK" in os.environ:
            parser.error(f"run {args.mode} mode directly, without torchrun")
        if args.mode == "profile_nsys":
            return _run_nsys_profiles(args, tokens)
        return _run_ncu_profiles(args, tokens)
    if "LOCAL_RANK" not in os.environ:
        parser.error(f"{args.mode} mode must be launched with torchrun")

    return _run_distributed_benchmark(args, tokens)


if __name__ == "__main__":
    raise SystemExit(main())
