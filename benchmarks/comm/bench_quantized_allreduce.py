# Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All rights reserved.
# Copyright (c) 2025 by FlashInfer team.
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

"""Quantized AllReduce benchmark: latency + error analysis.

Produces two key results for the PR:
  1. Latency comparison across 3 variants:
     - NCCL dist.all_reduce (universal baseline)
     - PyTorch symm_mem multimem_all_reduce_ (native symmetric memory)
     - FP8 Quantized AllReduce (this PR)
  2. Error analysis: max/mean error across data distributions and sizes

Usage:
    mpirun -np 8 python benchmarks/comm/bench_quantized_allreduce.py

Options:
    --latency-only    Skip error analysis
    --error-only      Skip latency benchmark
    --json FILE       Write results to JSON
    --warmup N        Warmup iterations (default: 100)
    --iters N         Benchmark iterations (default: 50)
"""

import argparse
import gc
import json
import os
import statistics
from datetime import datetime, timezone

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from mpi4py import MPI

from flashinfer.testing.utils import bench_gpu_time

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

BENCH_SIZES_BYTES = [
    2048,  # 2KB
    8192,  # 8KB
    32768,  # 32KB
    131072,  # 128KB
    524288,  # 512KB
    2097152,  # 2MB
    8388608,  # 8MB
    33554432,  # 32MB
    134217728,  # 128MB
    536870912,  # 512MB
    2147483648,  # 2GB
]


def size_label(n_bytes: int) -> str:
    if n_bytes < 1024:
        return f"{n_bytes}B"
    elif n_bytes < 1024**2:
        return f"{n_bytes // 1024}KB"
    elif n_bytes < 1024**3:
        return f"{n_bytes // 1024**2}MB"
    else:
        return f"{n_bytes / 1024**3:.1f}GB"


def setup():
    """Initialize distributed via mpi4py (consistent with FlashInfer benchmarks)."""
    mpi_comm = MPI.COMM_WORLD
    rank = mpi_comm.Get_rank()
    world_size = mpi_comm.Get_size()
    local_rank = mpi_comm.Split_type(MPI.COMM_TYPE_SHARED).Get_rank()
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29501")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl", init_method="env://", rank=rank, world_size=world_size
    )
    return rank, world_size, local_rank


def make_tensor(numel: int, rank: int, distribution: str = "normal") -> torch.Tensor:
    """Create a BF16 tensor with specified distribution.

    Each (rank, numel, distribution) combination gets an independent seed
    so that different sizes/distributions produce statistically independent data.
    """
    gen = torch.Generator(device="cuda")
    dist_hash = hash(distribution) & 0xFFFF
    seed = 42 + rank * 1000003 + numel * 7 + dist_hash
    gen.manual_seed(seed)
    if distribution == "normal":
        return torch.randn(numel, dtype=torch.bfloat16, device="cuda", generator=gen)
    elif distribution == "uniform":
        return (
            torch.rand(numel, dtype=torch.bfloat16, device="cuda", generator=gen) * 2
            - 1
        )
    elif distribution == "sparse":
        t = torch.zeros(numel, dtype=torch.bfloat16, device="cuda")
        mask = torch.rand(numel, device="cuda", generator=gen) < 0.1
        t[mask] = torch.randn(
            mask.sum(), dtype=torch.bfloat16, device="cuda", generator=gen
        )
        return t
    elif distribution == "heavy_tail":
        t = torch.randn(numel, dtype=torch.bfloat16, device="cuda", generator=gen)
        spikes = torch.rand(numel, device="cuda", generator=gen) < 0.01
        t[spikes] *= 100
        return t
    elif distribution == "activations":
        t = torch.randn(numel, dtype=torch.float32, device="cuda", generator=gen)
        t = torch.nn.functional.gelu(t)
        return t.to(torch.bfloat16)
    elif distribution == "swiglu":
        x = torch.randn(numel, dtype=torch.float32, device="cuda", generator=gen)
        y = torch.randn(numel, dtype=torch.float32, device="cuda", generator=gen)
        t = torch.nn.functional.silu(x) * y
        return t.to(torch.bfloat16)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")


# ---------------------------------------------------------------------------
# Backend 1: NCCL AllReduce (universal baseline)
# ---------------------------------------------------------------------------


_nccl_out: torch.Tensor | None = None


def nccl_allreduce(tensor: torch.Tensor) -> torch.Tensor:
    global _nccl_out
    if _nccl_out is None or _nccl_out.shape != tensor.shape:
        _nccl_out = torch.empty_like(tensor)
    _nccl_out.copy_(tensor)
    dist.all_reduce(_nccl_out)
    return _nccl_out


# ---------------------------------------------------------------------------
# Backend 2: PyTorch Symmetric Memory multimem_all_reduce_
# ---------------------------------------------------------------------------

_symm_cache: dict = {}
_symm_bufs: list = []


def init_symm_mem():
    symm_mem.enable_symm_mem_for_group(dist.group.WORLD.group_name)


_symm_out: torch.Tensor | None = None


def symm_mem_allreduce(tensor: torch.Tensor) -> torch.Tensor:
    global _symm_out
    numel = tensor.numel()
    key = (numel, tensor.dtype, tensor.device)
    if key not in _symm_cache:
        buf = symm_mem.empty((numel,), dtype=tensor.dtype, device=tensor.device)
        _symm_bufs.append(buf)
        grp = dist.group.WORLD.group_name
        symm_mem.rendezvous(buf, grp)
        _symm_cache[key] = (buf, grp)
    buf, grp = _symm_cache[key]
    buf[:numel].copy_(tensor.view(-1))
    torch.ops.symm_mem.multimem_all_reduce_(buf[:numel], "sum", grp)
    if _symm_out is None or _symm_out.shape != tensor.shape:
        _symm_out = torch.empty_like(tensor)
    _symm_out.copy_(buf[:numel].view(tensor.shape))
    return _symm_out


# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Backend 4: FP8 Quantized AllReduce (this PR)
# ---------------------------------------------------------------------------

_fi_quant_available = False


def try_init_quantized_ar() -> bool:
    global _fi_quant_available
    try:
        from flashinfer.comm import quantized_all_reduce  # noqa: F401

        _fi_quant_available = True
        return True
    except Exception:
        return False


def quantized_allreduce(tensor: torch.Tensor) -> torch.Tensor:
    from flashinfer.comm import quantized_all_reduce

    return quantized_all_reduce(tensor, dist.group.WORLD)


# ---------------------------------------------------------------------------
# Ground truth for error measurement
# ---------------------------------------------------------------------------


def compute_ground_truth_bf16(tensor: torch.Tensor) -> torch.Tensor:
    """BF16 NCCL allreduce — the baseline users would get without quantization."""
    out = tensor.clone()
    dist.all_reduce(out)
    return out


# ---------------------------------------------------------------------------
# Result 1: Latency benchmark
# ---------------------------------------------------------------------------


def run_latency_benchmark(rank, ws, warmup, iters):
    """Compare latency using FlashInfer's bench_gpu_time with CUDA graph timing.

    Methodology matches flashinfer/benchmarks/routines/allreduce_comm.py:
    - CUDA graph capture (amortizes launch overhead)
    - MPI barrier before each measurement
    - Reports median time across iterations
    """
    mpi_comm = MPI.COMM_WORLD
    has_fi_quant = try_init_quantized_ar()

    backends = [
        ("nccl", nccl_allreduce, True),
        ("symm_mem", symm_mem_allreduce, True),
        ("fp8_quantized_ar", quantized_allreduce, has_fi_quant),
    ]
    active = [(n, f) for n, f, e in backends if e]

    results = []

    if rank == 0:
        print(f"\n{'=' * 90}")
        print(f"  LATENCY BENCHMARK — {ws} GPUs ({torch.cuda.get_device_name(0)})")
        print(f"  Backends: {', '.join(n for n, _ in active)}")
        print(
            "  Method: bench_gpu_time(use_cuda_graph=True, num_iters_within_graph=10)"
        )
        print(f"  Warmup={warmup}, Iters={iters}")
        print(f"{'=' * 90}\n")

        col_w = 16
        header = f"{'Size':>8} |"
        for name, _ in active:
            header += f" {name:>{col_w}} |"
        print(header)
        print("-" * len(header))

    for size_bytes in BENCH_SIZES_BYTES:
        numel = size_bytes // 2  # BF16
        numel = (numel // (ws * 8)) * (ws * 8) or ws * 8
        tensor = make_tensor(numel, rank, "normal")

        row = {
            "size_bytes": size_bytes,
            "size_label": size_label(size_bytes),
            "numel": numel,
        }

        if rank == 0:
            line = f"{size_label(size_bytes):>8} |"

        for name, fn in active:
            # All ranks must agree on skip/run to avoid deadlocks in collectives
            mpi_comm.Barrier()

            try:
                # Verify kernel works
                _ = fn(tensor)
                torch.cuda.synchronize()
                can_run = True
            except Exception as e:
                can_run = False
                err_msg = str(e)[:30]

            # All ranks vote -- skip if ANY rank failed
            all_can_run = mpi_comm.allreduce(int(can_run), op=MPI.MIN)

            if not all_can_run:
                row[f"{name}_us"] = None
                row[f"{name}_err"] = err_msg if not can_run else "other rank failed"
                if rank == 0:
                    line += f" {'SKIP':>{col_w}} |"
                continue

            try:
                # Synchronize all ranks before timing
                mpi_comm.Barrier()
                torch.cuda.synchronize()

                # Multi-rank collectives (NCCL, symm_mem, etc.) switch to
                # multi-kernel algorithms at ~1MB that cannot be captured in
                # a CUDA graph.  Use CUDA-event timing for all backends to
                # keep the comparison fair and avoid deadlocks.
                # cold_l2_cache=False: allreduce is NVLink-bound, not L2-bound,
                # and rotating buffers deadlock with multi-rank collectives.
                times_ms = bench_gpu_time(
                    fn=fn,
                    input_args=(tensor,),
                    dry_run_iters=warmup,
                    repeat_iters=iters,
                    use_cuda_graph=True,
                    num_iters_within_graph=10,
                    cold_l2_cache=False,
                )
                med_ms = statistics.median(times_ms)
                med_us = med_ms * 1000.0
                bw_gbps = (size_bytes / 1e9) / (med_ms / 1e3) if med_ms > 0 else 0
                row[f"{name}_us"] = med_us
                row[f"{name}_bw_gbps"] = bw_gbps
                if rank == 0:
                    line += f" {med_us:>7.1f}us {bw_gbps:>4.1f}G |"
            except Exception as e:
                row[f"{name}_us"] = None
                row[f"{name}_err"] = str(e)[:30]
                if rank == 0:
                    line += f" {'SKIP':>{col_w}} |"

        results.append(row)
        if rank == 0:
            print(line, flush=True)

    # Speedup summary
    if rank == 0 and has_fi_quant:
        print("\n  Speedup (fp8_quantized_ar vs others):")
        print(f"  {'Size':>8} | {'vs NCCL':>8} | {'vs symm_mem':>11}")
        print(f"  {'-' * 36}")
        for row in results:
            fp8 = row.get("fp8_quantized_ar_us")
            nccl = row.get("nccl_us")
            smm = row.get("symm_mem_us")
            if fp8:
                vs_nccl = f"{nccl / fp8:.2f}x" if nccl else "N/A"
                vs_smm = f"{smm / fp8:.2f}x" if smm else "N/A"
                print(f"  {row['size_label']:>8} | {vs_nccl:>8} | {vs_smm:>11}")

    # Bandwidth summary (GB/s)
    if rank == 0:
        print("\n  Bandwidth (GB/s):")
        header = f"  {'Size':>8} |"
        for name, _ in active:
            header += f" {name:>12} |"
        print(header)
        print(f"  {'-' * (len(header) - 2)}")
        for row in results:
            line = f"  {row['size_label']:>8} |"
            for name, _ in active:
                bw = row.get(f"{name}_bw_gbps")
                if bw is not None:
                    line += f" {bw:>10.1f}G |"
                else:
                    line += f" {'—':>12} |"
            print(line)

    return results


# ---------------------------------------------------------------------------
# Result 2: Error analysis
# ---------------------------------------------------------------------------

DISTRIBUTIONS = ["normal", "uniform", "sparse", "heavy_tail", "activations", "swiglu"]
ERROR_SIZES_BYTES = [32768, 524288, 8388608, 134217728]
NUM_TRIALS = 5


def run_error_analysis(rank, ws):
    """Measure FP8 quantization error across distributions and sizes.

    Runs NUM_TRIALS independent draws per (distribution, size) and reports
    mean ± std of each error metric.
    """
    has_fi_quant = try_init_quantized_ar()
    if not has_fi_quant:
        if rank == 0:
            print("ERROR: quantized_all_reduce not available, skipping error analysis")
        return []

    results = []

    if rank == 0:
        print(f"\n{'=' * 90}")
        print(f"  ERROR ANALYSIS — {ws} GPUs, {NUM_TRIALS} trials per config")
        print(f"  Distributions: {', '.join(DISTRIBUTIONS)}")
        print(f"  Sizes: {', '.join(size_label(s) for s in ERROR_SIZES_BYTES)}")
        print(f"{'=' * 90}\n")

        print(
            f"{'Distribution':>12} | {'Size':>8} | {'mean_rel':>18} | {'max_abs':>18}"
        )
        print("-" * 68)

    for distribution in DISTRIBUTIONS:
        for size_bytes in ERROR_SIZES_BYTES:
            numel = size_bytes // 2
            numel = (numel // (ws * 8)) * (ws * 8) or ws * 8

            trial_mean_rels = []
            trial_max_abs = []

            for trial in range(NUM_TRIALS):
                gen = torch.Generator(device="cuda")
                dist_hash = hash(distribution) & 0xFFFF
                seed = 42 + rank * 1000003 + numel * 7 + dist_hash + trial * 31
                gen.manual_seed(seed)

                if distribution == "normal":
                    tensor = torch.randn(
                        numel, dtype=torch.bfloat16, device="cuda", generator=gen
                    )
                elif distribution == "uniform":
                    tensor = (
                        torch.rand(
                            numel, dtype=torch.bfloat16, device="cuda", generator=gen
                        )
                        * 2
                        - 1
                    )
                elif distribution == "sparse":
                    tensor = torch.zeros(numel, dtype=torch.bfloat16, device="cuda")
                    mask = torch.rand(numel, device="cuda", generator=gen) < 0.1
                    tensor[mask] = torch.randn(
                        mask.sum(), dtype=torch.bfloat16, device="cuda", generator=gen
                    )
                elif distribution == "heavy_tail":
                    tensor = torch.randn(
                        numel, dtype=torch.bfloat16, device="cuda", generator=gen
                    )
                    spikes = torch.rand(numel, device="cuda", generator=gen) < 0.01
                    tensor[spikes] *= 100
                elif distribution == "activations":
                    t = torch.randn(
                        numel, dtype=torch.float32, device="cuda", generator=gen
                    )
                    tensor = torch.nn.functional.gelu(t).to(torch.bfloat16)
                elif distribution == "swiglu":
                    x = torch.randn(
                        numel, dtype=torch.float32, device="cuda", generator=gen
                    )
                    y = torch.randn(
                        numel, dtype=torch.float32, device="cuda", generator=gen
                    )
                    tensor = (torch.nn.functional.silu(x) * y).to(torch.bfloat16)
                else:
                    raise ValueError(f"Unknown distribution: {distribution}")

                dist.barrier()
                ref_bf16 = compute_ground_truth_bf16(tensor)
                torch.cuda.synchronize()
                out_fp8 = quantized_allreduce(tensor)

                abs_err = (out_fp8.float() - ref_bf16.float()).abs()
                ref_f32 = ref_bf16.float()
                nonzero_mask = ref_f32.abs() > 0.01
                if nonzero_mask.any():
                    rel_err = abs_err[nonzero_mask] / ref_f32[nonzero_mask].abs()
                    trial_mean_rels.append(rel_err.mean().item())
                else:
                    trial_mean_rels.append(0.0)
                trial_max_abs.append(abs_err.max().item())

            mean_rel_avg = statistics.mean(trial_mean_rels)
            mean_rel_std = statistics.stdev(trial_mean_rels) if NUM_TRIALS > 1 else 0.0
            max_abs_avg = statistics.mean(trial_max_abs)
            max_abs_std = statistics.stdev(trial_max_abs) if NUM_TRIALS > 1 else 0.0

            row = {
                "distribution": distribution,
                "size_bytes": size_bytes,
                "size_label": size_label(size_bytes),
                "numel": numel,
                "mean_rel_avg": mean_rel_avg,
                "mean_rel_std": mean_rel_std,
                "max_abs_avg": max_abs_avg,
                "max_abs_std": max_abs_std,
            }
            results.append(row)

            if rank == 0:
                print(
                    f"{distribution:>12} | {size_label(size_bytes):>8} | "
                    f"{mean_rel_avg:>7.4f} ± {mean_rel_std:.4f} | "
                    f"{max_abs_avg:>7.4f} ± {max_abs_std:.4f}"
                )

    # Summary
    if rank == 0:
        print("\n  Summary by distribution (mean ± std across sizes and trials):")
        print(f"  {'Distribution':>12} | {'mean_rel':>18} | {'max_abs':>18}")
        print(f"  {'-' * 56}")
        for d in DISTRIBUTIONS:
            d_rows = [r for r in results if r["distribution"] == d]
            avg_rel = statistics.mean(r["mean_rel_avg"] for r in d_rows)
            std_rel = statistics.mean(r["mean_rel_std"] for r in d_rows)
            avg_max = statistics.mean(r["max_abs_avg"] for r in d_rows)
            std_max = statistics.mean(r["max_abs_std"] for r in d_rows)
            print(
                f"  {d:>12} | {avg_rel:>7.4f} ± {std_rel:.4f} | "
                f"{avg_max:>7.4f} ± {std_max:.4f}"
            )

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Quantized AllReduce PR benchmark")
    parser.add_argument("--latency-only", action="store_true")
    parser.add_argument("--error-only", action="store_true")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()

    rank, ws, _ = setup()
    init_symm_mem()
    gc.disable()

    all_results = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "world_size": ws,
            "gpu": torch.cuda.get_device_name(0),
            "pytorch": torch.__version__,
        }
    }

    if not args.error_only:
        latency_results = run_latency_benchmark(rank, ws, args.warmup, args.iters)
        all_results["latency"] = latency_results

    if not args.latency_only:
        error_results = run_error_analysis(rank, ws)
        all_results["error"] = error_results

    if rank == 0 and args.json:
        with open(args.json, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nResults saved to {args.json}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
