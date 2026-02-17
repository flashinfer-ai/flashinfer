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

# ==============================================================================
# Triton reference implementation for selective_state_update.
# Imported from tests/mamba/selective_state_update_triton.py to avoid code
# duplication. See that file for the canonical Triton kernel source.
# ==============================================================================

import importlib
import os
from collections import defaultdict

import numpy as np
import torch

import flashinfer
from flashinfer.testing.utils import bench_gpu_time

from .flashinfer_benchmark_utils import (
    dtype_str_to_torch_dtype,
    get_device,
    is_close_stats,
    print_perf_metrics,
    filter_backends_by_compute_capability,
)

# ---- Import Triton reference kernel from tests/mamba/ ----
# The canonical Triton selective_state_update lives in tests/mamba/selective_state_update_triton.py.
# We import it here rather than duplicating ~400 lines of kernel code.


def _import_triton_reference():
    """Import selective_state_update_triton from tests/mamba/.

    Uses importlib to load the module directly by file path, avoiding sys.path
    pollution and fragile relative path assumptions.
    """
    # Resolve path: benchmarks/routines/mamba.py -> ../../tests/mamba/selective_state_update_triton.py
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    _repo_root = os.path.normpath(os.path.join(_this_dir, "..", ".."))
    _triton_ref_path = os.path.join(
        _repo_root, "tests", "mamba", "selective_state_update_triton.py"
    )

    if not os.path.isfile(_triton_ref_path):
        raise ImportError(
            f"Cannot find Triton reference kernel at: {_triton_ref_path}\n"
            f"Expected location: <repo>/tests/mamba/selective_state_update_triton.py\n"
            f"Make sure you are running from within the FlashInfer repository."
        )

    spec = importlib.util.spec_from_file_location(
        "selective_state_update_triton", _triton_ref_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.selective_state_update_triton


selective_state_update_triton_reference = _import_triton_reference()


# ==============================================================================
# Benchmark infrastructure
# ==============================================================================


def run_mamba_test(args):
    """
    Run a mamba test.

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    if args.routine == "selective_state_update":
        return testSelectiveStateUpdate(args)
    else:
        raise ValueError(f"Unsupported routine: {args.routine}")


def parse_mamba_args(line, parser):
    """
    Parse command line arguments for mamba test configuration.

    Args:
        line: Command line arguments
        parser: ArgumentParser object already populated with shared arguments

    Returns:
        Parsed argument namespace
    """
    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,
        help="Batch size (number of sequences).",
    )
    parser.add_argument(
        "--nheads",
        type=int,
        required=True,
        help="Number of SSM heads.",
    )
    parser.add_argument(
        "--dim",
        type=int,
        required=True,
        help="Head dimension (headdim).",
    )
    parser.add_argument(
        "--dstate",
        type=int,
        required=True,
        help="SSM state size.",
    )
    parser.add_argument(
        "--ngroups",
        type=int,
        required=False,
        default=8,
        help="Number of groups for B and C matrices. nheads must be divisible by ngroups.",
    )
    parser.add_argument(
        "--cache_steps",
        type=int,
        required=False,
        default=0,
        help="Number of steps/tokens for multi-token prediction. 0 = single-token prediction.",
    )
    parser.add_argument(
        "--input_dtype",
        type=str,
        required=False,
        default="bfloat16",
        choices=["bfloat16"],
        help="Data type for input tensors (x, B, C, z). Only bfloat16 is supported.",
    )
    parser.add_argument(
        "--state_dtype",
        type=str,
        required=False,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Data type for the SSM state cache.",
    )
    parser.add_argument(
        "--weight_dtype",
        type=str,
        required=False,
        default="float32",
        choices=["bfloat16", "float32"],
        help="Data type for weight tensors (dt, D, dt_bias).",
    )
    parser.add_argument(
        "--has_z",
        action="store_true",
        default=False,
        help="Include z tensor for gating (z * sigmoid(z) applied to output).",
    )
    parser.add_argument(
        "--dt_softplus",
        action="store_true",
        default=False,
        help="Apply softplus to dt before use.",
    )
    parser.add_argument(
        "--backends",
        type=str,
        required=False,
        nargs="+",
        default=["flashinfer"],
        choices=["flashinfer", "triton"],
        help="Kernel backends to benchmark. Default: flashinfer",
    )

    args = parser.parse_args(line)

    # Validate nheads divisibility
    if args.nheads % args.ngroups != 0:
        raise ValueError(
            f"nheads ({args.nheads}) must be divisible by ngroups ({args.ngroups})."
        )

    # Validate dim is supported by the CUDA kernel dispatch
    supported_dims = [64, 128, 256]
    if "flashinfer" in args.backends and args.dim not in supported_dims:
        raise ValueError(
            f"dim ({args.dim}) is not supported by the FlashInfer kernel. "
            f"Supported dim values: {supported_dims}. "
            f"Use --backends triton for unsupported dim values."
        )

    # Validate dstate is supported by the CUDA kernel dispatch
    supported_dstates = [64, 128, 256]
    if "flashinfer" in args.backends and args.dstate not in supported_dstates:
        raise ValueError(
            f"dstate ({args.dstate}) is not supported by the FlashInfer kernel. "
            f"Supported dstate values: {supported_dstates}. "
            f"Use --backends triton for unsupported dstate values."
        )

    # Validate nheads/ngroups ratio is supported by the CUDA kernel
    supported_ratios = [1, 8, 16]
    ratio = args.nheads // args.ngroups
    if ratio not in supported_ratios:
        raise ValueError(
            f"nheads/ngroups ratio ({ratio} = {args.nheads}/{args.ngroups}) is not supported by the FlashInfer kernel. "
            f"Supported ratios: {supported_ratios}."
        )

    if args.verbose >= 1:
        print(f"[INFO] {args = }")
    return args


def testSelectiveStateUpdate(args):
    """
    Test selective_state_update API for Mamba layers.

    This test:
    1. Generates random input tensors for SSM state update
    2. Runs selective_state_update with the requested backend(s)
       - 'flashinfer': FlashInfer CUDA kernel (architecture-specific: base/SM90/SM100+)
       - 'triton': Triton reference implementation
    3. Optionally runs reference check (compares against Triton reference)
    4. Measures performance metrics (memory bandwidth)

    Supports both single-token prediction (STP, cache_steps=0) and
    multi-token prediction (MTP, cache_steps>=1) modes.

    Note: selective_state_update is memory-bandwidth bound, so TB/sec is the
    primary performance metric.

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    if args.verbose >= 1:
        print("[INFO] Running testSelectiveStateUpdate")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    ## Parse input arguments
    backends = args.backends[:]  # Make a copy to avoid modifying the original
    batch_size = args.batch_size
    nheads = args.nheads
    dim = args.dim
    dstate = args.dstate
    ngroups = args.ngroups
    cache_steps = args.cache_steps
    has_z = args.has_z
    dt_softplus = args.dt_softplus
    is_cuda_graph_compatible = not args.no_cuda_graph
    run_refcheck = args.refcheck
    res = []

    backends = filter_backends_by_compute_capability(backends, args.routine, device)
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

    input_dtype = dtype_str_to_torch_dtype(args.input_dtype)
    state_dtype = dtype_str_to_torch_dtype(args.state_dtype)
    weight_dtype = dtype_str_to_torch_dtype(args.weight_dtype)
    ## Done parsing input arguments

    ## Determine STP vs MTP mode
    is_mtp = cache_steps >= 1
    T = cache_steps if is_mtp else None

    ## Prepare input tensors (mirrors tests/mamba/utils.py::create_test_inputs)
    ssm_state_cache_size = max(384, batch_size * 10)

    # State cache: (total_entries, nheads, dim, dstate) - contiguous
    state_cache = torch.randn(
        ssm_state_cache_size, nheads, dim, dstate, dtype=state_dtype, device=device
    )

    # Input x: (batch_size, [T,] nheads, dim)
    if T is not None:
        x = torch.randn(batch_size, T, nheads, dim, dtype=input_dtype, device=device)
    else:
        x = torch.randn(batch_size, nheads, dim, dtype=input_dtype, device=device)

    # dt: broadcasting across dim (one value per head)
    if T is not None:
        dt_base = torch.randn(batch_size, T, nheads, dtype=weight_dtype, device=device)
        dt = dt_base.as_strided(
            (batch_size, T, nheads, dim), (T * nheads, nheads, 1, 0)
        )
    else:
        dt_base = torch.randn(batch_size, nheads, dtype=weight_dtype, device=device)
        dt = dt_base.as_strided((batch_size, nheads, dim), (nheads, 1, 0))

    # A: (nheads, dim, dstate) - negative values, broadcasting (one value per head)
    A_base = -torch.rand(nheads, dtype=torch.float32, device=device) - 1.0
    A = A_base.as_strided((nheads, dim, dstate), (1, 0, 0))

    # B, C: (batch_size, [T,] ngroups, dstate)
    if T is not None:
        B = torch.randn(
            batch_size, T, ngroups, dstate, dtype=input_dtype, device=device
        )
        C = torch.randn(
            batch_size, T, ngroups, dstate, dtype=input_dtype, device=device
        )
    else:
        B = torch.randn(batch_size, ngroups, dstate, dtype=input_dtype, device=device)
        C = torch.randn(batch_size, ngroups, dstate, dtype=input_dtype, device=device)

    # D: (nheads, dim) - broadcasting (one value per head)
    D_base = torch.randn(nheads, dtype=weight_dtype, device=device)
    D = D_base.as_strided((nheads, dim), (1, 0))

    # dt_bias: (nheads, dim) - broadcasting (one value per head)
    dt_bias_base = torch.rand(nheads, dtype=weight_dtype, device=device) - 4.0
    dt_bias = dt_bias_base.as_strided((nheads, dim), (1, 0))

    # Slot indices for state batching
    slot_idx = torch.randperm(ssm_state_cache_size, dtype=torch.int64, device=device)[
        :batch_size
    ]

    # Optional z tensor for gating
    z = None
    if has_z:
        if T is not None:
            z = torch.randn(
                batch_size, T, nheads, dim, dtype=input_dtype, device=device
            )
        else:
            z = torch.randn(batch_size, nheads, dim, dtype=input_dtype, device=device)

    if args.verbose >= 2:
        print(f"[VVERBOSE] Mode: {'MTP' if is_mtp else 'STP'}")
        print(f"[VVERBOSE] {state_cache.shape = }, {state_cache.dtype = }")
        print(f"[VVERBOSE] {x.shape = }, {x.dtype = }")
        print(f"[VVERBOSE] {dt.shape = }, {dt.dtype = }")
        print(f"[VVERBOSE] {A.shape = }, {A.dtype = }")
        print(f"[VVERBOSE] {B.shape = }, {B.dtype = }")
        print(f"[VVERBOSE] {C.shape = }, {C.dtype = }")
        print(f"[VVERBOSE] {D.shape = }, {D.dtype = }")
        print(f"[VVERBOSE] {dt_bias.shape = }, {dt_bias.dtype = }")
        print(f"[VVERBOSE] {slot_idx.shape = }")
        print(f"[VVERBOSE] {has_z = }, {dt_softplus = }")
        if z is not None:
            print(f"[VVERBOSE] {z.shape = }, {z.dtype = }")

    # Cache steps for Triton reference (None for STP, integer for MTP)
    triton_cache_steps = cache_steps if cache_steps > 0 else None

    def run_backend(backend, state, x, dt, A, B, C, D):
        if backend == "flashinfer":
            return flashinfer.mamba.selective_state_update(
                state,
                x,
                dt,
                A,
                B,
                C,
                D,
                z=z,
                dt_bias=dt_bias,
                dt_softplus=dt_softplus,
                state_batch_indices=slot_idx,
                cache_steps=cache_steps,
            )
        elif backend == "triton":
            return selective_state_update_triton_reference(
                state,
                x,
                dt,
                A,
                B,
                C,
                D,
                z=z,
                dt_bias=dt_bias,
                dt_softplus=dt_softplus,
                state_batch_indices=slot_idx,
                cache_steps=triton_cache_steps,
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    # Reference check: use Triton as golden reference
    # Save a clean snapshot of state_cache before any benchmarking, because
    # bench_gpu_time mutates state_cache in-place across many iterations.
    # All refcheck clones must come from this clean snapshot.
    has_reference_output = False
    clean_state_snapshot = state_cache.clone() if run_refcheck else None
    if run_refcheck:
        ref_state = clean_state_snapshot.clone()
        reference_output = (
            selective_state_update_triton_reference(
                ref_state,
                x,
                dt,
                A,
                B,
                C,
                D,
                z=z,
                dt_bias=dt_bias,
                dt_softplus=dt_softplus,
                state_batch_indices=slot_idx,
                cache_steps=triton_cache_steps,
            )
            .detach()
            .clone()
        )
        has_reference_output = True

    # Storage for timing results and outputs
    backend_times = {backend: [] for backend in backends}
    outputs = {}
    for cur_backend in backends:
        if run_refcheck and cur_backend != "triton":
            # Always clone from the clean snapshot, not from state_cache
            # (which may have been mutated by previous backend's bench_gpu_time)
            fresh_state = clean_state_snapshot.clone()
            outputs[cur_backend] = (
                run_backend(cur_backend, fresh_state, x, dt, A, B, C, D)
                .detach()
                .clone()
            )
        backend_times[cur_backend] = bench_gpu_time(
            fn=run_backend,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            enable_cupti=args.use_cupti,
            use_cuda_graph=is_cuda_graph_compatible,
            input_args=(cur_backend, state_cache, x, dt, A, B, C, D),
        )

    # Compare outputs against Triton reference
    tested_backends = list(outputs.keys())
    tested_outputs = list(outputs.values())
    if len(tested_backends) > 0:
        if run_refcheck and has_reference_output:
            for i in range(len(tested_backends)):
                (
                    num_different_elements,
                    num_elements,
                    num_different_elements_percentage,
                ) = is_close_stats(
                    reference_output.float(),
                    tested_outputs[i].float(),
                    rtol=1e-2,
                    atol=1e-3,
                )
                # Allow up to 0.01% of elements to differ (floating-point edge cases)
                mismatch_threshold_pct = 0.01
                if num_different_elements_percentage > mismatch_threshold_pct:
                    print(
                        f"[ERROR] Output tensor mismatch from backend {tested_backends[i]}: "
                        f"{num_different_elements}/{num_elements} ({num_different_elements_percentage:.4f}%) elements differ "
                        f"(threshold: {mismatch_threshold_pct}%)"
                    )
                    if not args.allow_output_mismatch:
                        raise AssertionError(
                            f"[ERROR] Backend {tested_backends[i]} output mismatch with {num_different_elements} elements"
                        )
                elif num_different_elements > 0:
                    if args.verbose >= 1:
                        print(
                            f"[REFCHECK] Backend {tested_backends[i]}: PASSED "
                            f"({num_different_elements}/{num_elements} elements differ "
                            f"({num_different_elements_percentage:.4f}%), within {mismatch_threshold_pct}% threshold)"
                        )
                else:
                    if args.verbose >= 1:
                        print(
                            f"[REFCHECK] Backend {tested_backends[i]}: PASSED (all {num_elements} elements match)"
                        )

    # Compute and report performance metrics
    T_val = cache_steps if cache_steps > 0 else 1

    for backend in backends:
        if len(backend_times[backend]) > 0:
            median_time = np.median(backend_times[backend])
            std_time = np.std(backend_times[backend])

            # Memory bandwidth calculation (physical bytes accessed)
            # Read:
            #   state: batch_size * nheads * dim * dstate (via slot_idx indirection)
            #   x: batch_size * T * nheads * dim
            #   dt: batch_size * T * nheads (broadcast across dim)
            #   A: nheads (broadcast across dim and dstate)
            #   B: batch_size * T * ngroups * dstate
            #   C: batch_size * T * ngroups * dstate
            #   D: nheads (broadcast across dim)
            #   dt_bias: nheads (broadcast across dim)
            #   z (optional): batch_size * T * nheads * dim
            # Write:
            #   state: batch_size * nheads * dim * dstate
            #   output: batch_size * T * nheads * dim
            read_bytes = (
                batch_size * nheads * dim * dstate * state_dtype.itemsize  # state
                + batch_size * T_val * nheads * dim * input_dtype.itemsize  # x
                + batch_size * T_val * nheads * weight_dtype.itemsize  # dt (broadcast)
                + nheads * 4  # A (float32, broadcast)
                + batch_size * T_val * ngroups * dstate * input_dtype.itemsize  # B
                + batch_size * T_val * ngroups * dstate * input_dtype.itemsize  # C
                + nheads * weight_dtype.itemsize  # D (broadcast)
                + nheads * weight_dtype.itemsize  # dt_bias (broadcast)
            )
            if has_z:
                read_bytes += batch_size * T_val * nheads * dim * input_dtype.itemsize

            write_bytes = (
                batch_size * nheads * dim * dstate * state_dtype.itemsize  # state
                + batch_size * T_val * nheads * dim * input_dtype.itemsize  # output
            )

            problem_bytes = read_bytes + write_bytes

            # FLOPs estimate (TIE_HDIM case, where dt/A/D/dt_bias broadcast across dim):
            # Per (dim, dstate) element per (batch, T, head):
            #   state * dA: 1 mul, dB * x[:, None]: 1 mul, state + ...: 1 add,
            #   state * C[None, :]: 1 mul, sum reduction: ~1 add => 5 FLOPs/element
            problem_flops = batch_size * T_val * nheads * dim * dstate * 5
            tflops = problem_flops / (10**9 * median_time)  # TFLOPs/sec
            tb_per_sec = problem_bytes / (10**9 * median_time)  # TB/sec

            print_perf_metrics(backend, median_time, std_time, tflops, tb_per_sec)

            if args.output_path is not None:
                cur_res = defaultdict(str)
                cur_res["routine"] = args.routine
                cur_res["median_time"] = median_time
                cur_res["std_time"] = std_time
                cur_res["tflops"] = tflops
                cur_res["tb_per_sec"] = tb_per_sec
                cur_res["backend"] = backend
                # Mamba-specific columns
                cur_res["batch_size"] = batch_size
                cur_res["nheads"] = nheads
                cur_res["dim"] = dim
                cur_res["dstate"] = dstate
                cur_res["ngroups"] = ngroups
                cur_res["cache_steps"] = cache_steps
                cur_res["input_dtype"] = str(input_dtype)
                cur_res["state_dtype"] = str(state_dtype)
                cur_res["weight_dtype"] = str(weight_dtype)
                cur_res["has_z"] = has_z
                cur_res["dt_softplus"] = dt_softplus
                cur_res["case_tag"] = args.case_tag
                res.append(cur_res)
    return res
