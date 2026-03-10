# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
AllReduce Communication Benchmark Routine

This module provides benchmarking for AllReduce fusion operations using
FlashInfer's unified AllReduce API (create_allreduce_fusion_workspace +
allreduce_fusion). Designed to run with mpirun for multi-GPU benchmarking.

Supports backends: trtllm, mnnvl, auto
Supports patterns: allreduce, ar_residual_rmsnorm

Launch examples:
    # Basic allreduce with auto backend
    mpirun -np 8 python benchmarks/flashinfer_benchmark.py \
        --routine allreduce_fusion \
        --num_tokens 64 --hidden_size 4096

    # With specific backend
    mpirun -np 8 python benchmarks/flashinfer_benchmark.py \
        --routine allreduce_fusion \
        --num_tokens 64 --hidden_size 4096 \
        --ar_backend mnnvl

    # AllReduce + Residual + RMSNorm fusion
    mpirun -np 8 python benchmarks/flashinfer_benchmark.py \
        --routine allreduce_fusion \
        --num_tokens 64 --hidden_size 4096 \
        --pattern ar_residual_rmsnorm

    # With validation (recommended for first run)
    mpirun -np 8 python benchmarks/flashinfer_benchmark.py \
        --routine allreduce_fusion \
        --num_tokens 64 --hidden_size 4096 \
        --validate

Options:
    --ar_backend auto|trtllm|mnnvl : Backend selection (default: auto)
    --pattern allreduce|ar_residual_rmsnorm : Fusion pattern (default: allreduce)
    --validate                     : Run correctness validation before benchmarking

Note: Both oneshot and twoshot strategies are always benchmarked and reported.
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
from mpi4py import MPI

from flashinfer.comm import (
    AllReduceFusionPattern,
    AllReduceFusionWorkspace,
    allreduce_fusion,
    create_allreduce_fusion_workspace,
)
from flashinfer.comm.mnnvl import MnnvlMemory, TorchDistBackend
from flashinfer.norm import rmsnorm
from flashinfer.testing.utils import bench_gpu_time

try:
    from .flashinfer_benchmark_utils import (
        dtype_str_to_torch_dtype,
        print_perf_metrics,
    )
except ImportError:
    from flashinfer_benchmark_utils import (
        dtype_str_to_torch_dtype,
        print_perf_metrics,
    )

PATTERN_NAME_TO_CODE = {
    "allreduce": AllReduceFusionPattern.kAllReduce,
    "ar_residual_rmsnorm": AllReduceFusionPattern.kARResidualRMSNorm,
}

PATTERN_CODE_TO_NAME = {v: k for k, v in PATTERN_NAME_TO_CODE.items()}


def _setup_mpi_and_device() -> Tuple[MPI.Comm, int, int, int]:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
    local_rank = node_comm.Get_rank()
    torch.cuda.set_device(local_rank)
    return comm, rank, world_size, local_rank


def _init_torch_distributed(rank: int, world_size: int):
    """Initialize torch.distributed for TRTLLM backend (uses NCCL for IPC)."""
    import os

    import torch.distributed as dist

    if dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def _cleanup_torch_distributed():
    import torch.distributed as dist

    if dist.is_initialized():
        dist.destroy_process_group()


def _calculate_allreduce_busbw(
    message_size_bytes: int,
    world_size: int,
    time_ms: float,
) -> float:
    """Calculate bus bandwidth for allreduce in TB/sec.

    Uses the standard NCCL bus bandwidth formula:
    busbw = message_size * 2 * (N-1) / N / time
    The 2*(N-1)/N factor accounts for the ring allreduce algorithm
    (reduce-scatter + allgather phases).
    """
    if time_ms <= 0:
        return 0.0
    correction_factor = 2.0 * (world_size - 1) / world_size
    return message_size_bytes * correction_factor / (time_ms * 1e-3) / 1e12


def _calculate_allreduce_algbw(
    message_size_bytes: int,
    time_ms: float,
) -> float:
    """Calculate algorithmic bandwidth (message_size / time) in TB/sec."""
    if time_ms <= 0:
        return 0.0
    return message_size_bytes / (time_ms * 1e-3) / 1e12


def _validate_allreduce(
    workspace: AllReduceFusionWorkspace,
    num_tokens: int,
    hidden_size: int,
    input_dtype: torch.dtype,
    pattern_code: int,
    use_oneshot: Optional[bool],
    world_size: int,
    rank: int,
    comm: MPI.Comm,
    verbose: int = 0,
) -> bool:
    """Validate allreduce correctness via comparison with torch sum reference."""
    torch.manual_seed(42)

    # Create identical data across ranks for reference computation
    x_full = torch.randn(
        (world_size, num_tokens, hidden_size), dtype=input_dtype, device="cuda"
    )
    x_local = x_full[rank].clone()

    allreduce_ref = x_full.sum(dim=0)

    if pattern_code == AllReduceFusionPattern.kAllReduce:
        output = torch.empty_like(x_local)
        allreduce_fusion(
            input=x_local,
            workspace=workspace,
            pattern=pattern_code,
            launch_with_pdl=False,
            output=output,
            use_oneshot=use_oneshot,
        )
        torch.cuda.synchronize()

        try:
            torch.testing.assert_close(output, allreduce_ref, atol=0.05, rtol=0.05)
            passed = True
        except AssertionError as e:
            passed = False
            if rank == 0:
                print(f"[VALIDATE] AllReduce mismatch: {e}")

    elif pattern_code == AllReduceFusionPattern.kARResidualRMSNorm:
        residual = torch.randn(
            (num_tokens, hidden_size), dtype=input_dtype, device="cuda"
        )
        norm_weight = torch.randn((hidden_size,), dtype=input_dtype, device="cuda")
        eps = 1e-5

        # Broadcast residual and norm_weight so all ranks have the same data
        residual_cpu = comm.bcast(residual.cpu().numpy(), root=0)
        norm_weight_cpu = comm.bcast(norm_weight.cpu().numpy(), root=0)
        residual = torch.from_numpy(residual_cpu).to(device="cuda", dtype=input_dtype)
        norm_weight = torch.from_numpy(norm_weight_cpu).to(
            device="cuda", dtype=input_dtype
        )

        norm_out = torch.empty_like(x_local)
        residual_out = torch.empty_like(x_local)

        allreduce_fusion(
            input=x_local,
            workspace=workspace,
            pattern=pattern_code,
            launch_with_pdl=False,
            residual_out=residual_out,
            norm_out=norm_out,
            residual_in=residual,
            rms_gamma=norm_weight,
            rms_eps=eps,
            use_oneshot=use_oneshot,
        )
        torch.cuda.synchronize()

        # Reference: allreduce + residual + rmsnorm
        ref_residual_out = allreduce_ref + residual
        ref_norm_out = rmsnorm(ref_residual_out, norm_weight, eps, enable_pdl=False)

        try:
            torch.testing.assert_close(norm_out, ref_norm_out, atol=0.15, rtol=0.05)
            torch.testing.assert_close(
                residual_out, ref_residual_out, atol=0.05, rtol=0.05
            )
            passed = True
        except AssertionError as e:
            passed = False
            if rank == 0:
                print(f"[VALIDATE] Fused allreduce mismatch: {e}")
    else:
        if rank == 0:
            print(f"[VALIDATE] Skipping validation for pattern {pattern_code}")
        return True

    all_passed = comm.allreduce(passed, op=MPI.LAND)
    if rank == 0:
        if all_passed:
            print("[VALIDATE] PASSED: All ranks validated successfully")
        else:
            print("[VALIDATE] FAILED: Validation errors detected")
    return all_passed


def _benchmark_single_config(
    workspace: AllReduceFusionWorkspace,
    num_tokens: int,
    hidden_size: int,
    input_dtype: torch.dtype,
    pattern_code: int,
    use_oneshot: Optional[bool],
    args,
    comm: MPI.Comm,
    rank: int,
    world_size: int,
) -> Optional[dict]:
    """Benchmark a single (shape, pattern, backend) configuration.

    Returns result dict on rank 0, None on other ranks.
    """
    device = torch.device("cuda")

    # Check workspace capacity
    if not workspace.is_buffer_size_sufficient(
        world_size, num_tokens, hidden_size, input_dtype, use_oneshot
    ):
        if rank == 0 and args.verbose >= 1:
            print(
                f"[SKIP] Workspace insufficient for num_tokens={num_tokens}, "
                f"hidden_size={hidden_size}, use_oneshot={use_oneshot}"
            )
        return None

    # Create input tensors
    x = torch.ones((num_tokens, hidden_size), dtype=input_dtype, device=device)

    if pattern_code == AllReduceFusionPattern.kAllReduce:
        output = torch.empty_like(x)

        def run_allreduce(inp):
            allreduce_fusion(
                input=inp,
                workspace=workspace,
                pattern=pattern_code,
                launch_with_pdl=True,
                output=output,
                use_oneshot=use_oneshot,
            )
            return output

    elif pattern_code == AllReduceFusionPattern.kARResidualRMSNorm:
        residual = torch.randn_like(x)
        norm_weight = torch.randn((hidden_size,), dtype=input_dtype, device=device)
        norm_out = torch.empty_like(x)
        residual_out = torch.empty_like(x)

        def run_allreduce(inp):
            allreduce_fusion(
                input=inp,
                workspace=workspace,
                pattern=pattern_code,
                launch_with_pdl=True,
                residual_out=residual_out,
                norm_out=norm_out,
                residual_in=residual,
                rms_gamma=norm_weight,
                rms_eps=1e-5,
                use_oneshot=use_oneshot,
            )
            return norm_out

    else:
        if rank == 0:
            print(f"[SKIP] Unsupported pattern code {pattern_code}")
        return None

    # Synchronize before benchmarking
    comm.Barrier()
    torch.cuda.synchronize()

    try:
        times = bench_gpu_time(
            fn=run_allreduce,
            input_args=(x,),
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            sleep_after_run=False,
            enable_cupti=args.use_cupti,
            use_cuda_graph=not args.no_cuda_graph,
            cold_l2_cache=True,
        )
    except RuntimeError as e:
        # Kernel may fail for very large message sizes or unsupported configs
        if rank == 0:
            elem_size = torch.tensor([], dtype=input_dtype).element_size()
            msg_size_mb = num_tokens * hidden_size * elem_size / (1024 * 1024)
            print(
                f"[ERROR] Kernel failed for shape=({num_tokens}, {hidden_size}) "
                f"msg_size={msg_size_mb:.1f} MiB: {e}"
            )
        return None

    num_measure_iters = len(times)

    # Gather times from all ranks and use max (communication is synchronous)
    all_times = comm.allgather(times)

    if rank == 0:
        per_iter_max = [max(t[i] for t in all_times) for i in range(num_measure_iters)]
        median_time = float(np.median(per_iter_max))
        std_time = float(np.std(per_iter_max))

        elem_size = torch.tensor([], dtype=input_dtype).element_size()
        message_size_bytes = num_tokens * hidden_size * elem_size

        busbw = _calculate_allreduce_busbw(message_size_bytes, world_size, median_time)
        algbw = _calculate_allreduce_algbw(message_size_bytes, median_time)

        backend_name = workspace.backend
        pattern_name = PATTERN_CODE_TO_NAME.get(pattern_code, str(pattern_code))
        oneshot_str = ""
        if use_oneshot is True:
            oneshot_str = "_oneshot"
        elif use_oneshot is False:
            oneshot_str = "_twoshot"

        label = f"{backend_name}_{pattern_name}{oneshot_str}"

        print_perf_metrics(label, median_time, std_time, torch.nan, busbw)

        if args.verbose >= 1:
            print(
                f"  algbw={algbw:.4f} TB/s, "
                f"msg_size={message_size_bytes / 1024:.1f} KiB, "
                f"time={median_time * 1000:.1f} us"
            )

        return {
            "routine": args.routine,
            "median_time": median_time,
            "std_time": std_time,
            "tflops": "N/A",
            "tb_per_sec": busbw,
            "backend": backend_name,
            "resolved_backend": label,
            "num_tokens": num_tokens,
            "hidden_size": hidden_size,
            "input_dtype": str(input_dtype),
        }

    return None


def run_allreduce_comm_test(args):
    """Entry point from flashinfer_benchmark.py."""
    if args.routine == "allreduce_fusion":
        return test_allreduce_fusion(args)
    else:
        raise ValueError(f"Unsupported routine: {args.routine}")


def parse_allreduce_comm_args(line, parser):
    """Parse allreduce-specific command line arguments."""
    parser.add_argument(
        "--num_tokens",
        type=int,
        default=64,
        help="Number of tokens (rows) in the input tensor.",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=4096,
        help="Hidden dimension size.",
    )
    parser.add_argument(
        "--input_dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16"],
        help="Data type for input tensors.",
    )
    parser.add_argument(
        "--ar_backend",
        type=str,
        default="auto",
        choices=["auto", "trtllm", "mnnvl"],
        help="AllReduce backend. 'auto' uses heuristic.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="allreduce",
        choices=list(PATTERN_NAME_TO_CODE.keys()),
        help="Fusion pattern.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        default=False,
        help="Run correctness validation before benchmarking.",
    )

    args = parser.parse_args(line)
    return args


def test_allreduce_fusion(args):
    """Benchmark allreduce fusion across shapes, backends, and patterns."""
    comm, rank, world_size, local_rank = _setup_mpi_and_device()
    gpus_per_node = torch.cuda.device_count()

    if world_size < 2:
        if rank == 0:
            print("[ERROR] AllReduce benchmark requires at least 2 ranks")
        return []

    if rank == 0:
        print(
            f"[INFO] AllReduce benchmark: world_size={world_size}, "
            f"gpus_per_node={gpus_per_node}"
        )

    input_dtype = dtype_str_to_torch_dtype(args.input_dtype)
    res = []

    num_tokens = args.num_tokens
    hidden_size = args.hidden_size
    backend_list = [args.ar_backend]
    pattern_list = [args.pattern]
    oneshot_list: List[Optional[bool]] = [True, False]

    # Initialize backends
    torch_dist_initialized = False

    needs_mnnvl = any(b in ("mnnvl", "auto") for b in backend_list)
    needs_trtllm = any(b in ("trtllm", "auto") for b in backend_list)

    if needs_mnnvl:
        try:
            MnnvlMemory.initialize()
        except Exception as e:
            if rank == 0:
                print(f"[WARNING] MNNVL initialization failed: {e}")
            backend_list = [b for b in backend_list if b != "mnnvl"]
            if not backend_list:
                if rank == 0:
                    print("[ERROR] No backends available after MNNVL init failure")
                return []

    if needs_trtllm:
        try:
            _init_torch_distributed(rank, world_size)
            torch_dist_initialized = True
        except Exception as e:
            if rank == 0:
                print(f"[WARNING] torch.distributed initialization failed: {e}")
            backend_list = [b for b in backend_list if b != "trtllm"]
            if not backend_list:
                if rank == 0:
                    print(
                        "[ERROR] No backends available after torch.distributed init failure"
                    )
                return []

    try:
        for backend in backend_list:
            # Create workspace sized for the largest shape
            comm_backend = TorchDistBackend() if torch_dist_initialized else None

            try:
                workspace = create_allreduce_fusion_workspace(
                    backend=backend,
                    world_size=world_size,
                    rank=rank,
                    max_token_num=num_tokens,
                    hidden_dim=hidden_size,
                    dtype=input_dtype,
                    gpus_per_node=gpus_per_node,
                    comm_backend=comm_backend,
                    force_oneshot_support=True,
                )
            except Exception as e:
                if rank == 0:
                    print(
                        f"[ERROR] Failed to create workspace for backend={backend}: {e}"
                    )
                continue

            if rank == 0:
                print(
                    f"[INFO] Created workspace: backend={workspace.backend}, "
                    f"shape=({num_tokens}, {hidden_size})"
                )

            try:
                for pattern_name in pattern_list:
                    pattern_code = PATTERN_NAME_TO_CODE[pattern_name]

                    # MNNVL only supports patterns 0 and 1
                    if workspace.backend == "mnnvl" and pattern_code > 1:
                        if rank == 0 and args.verbose >= 1:
                            print(
                                f"[SKIP] MNNVL does not support pattern {pattern_name}"
                            )
                        continue

                    for use_oneshot in oneshot_list:
                        if rank == 0 and args.verbose >= 1:
                            print(
                                f"[INFO] Benchmarking: backend={workspace.backend}, "
                                f"pattern={pattern_name}, "
                                f"shape=({num_tokens}, {hidden_size}), "
                                f"use_oneshot={use_oneshot}"
                            )

                        # Validate if requested
                        if args.validate:
                            valid = _validate_allreduce(
                                workspace=workspace,
                                num_tokens=num_tokens,
                                hidden_size=hidden_size,
                                input_dtype=input_dtype,
                                pattern_code=pattern_code,
                                use_oneshot=use_oneshot,
                                world_size=world_size,
                                rank=rank,
                                comm=comm,
                                verbose=args.verbose,
                            )
                            if not valid:
                                if rank == 0:
                                    print(
                                        "[ERROR] Validation failed, skipping benchmark"
                                    )
                                continue

                        result = _benchmark_single_config(
                            workspace=workspace,
                            num_tokens=num_tokens,
                            hidden_size=hidden_size,
                            input_dtype=input_dtype,
                            pattern_code=pattern_code,
                            use_oneshot=use_oneshot,
                            args=args,
                            comm=comm,
                            rank=rank,
                            world_size=world_size,
                        )

                        if result is not None:
                            res.append(result)

            finally:
                workspace.destroy()

    finally:
        if torch_dist_initialized:
            _cleanup_torch_distributed()

    return res
