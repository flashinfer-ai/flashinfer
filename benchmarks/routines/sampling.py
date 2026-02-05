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


def run_sampling_test(args):
    """
    Run a sampling test.

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    if args.routine == "softmax":
        return testSoftmax(args)
    elif args.routine == "sampling_from_probs":
        return testSamplingFromProbs(args)
    elif args.routine == "sampling_from_logits":
        return testSamplingFromLogits(args)
    elif args.routine == "top_k_sampling_from_probs":
        return testTopKSamplingFromProbs(args)
    elif args.routine == "top_p_sampling_from_probs":
        return testTopPSamplingFromProbs(args)
    elif args.routine == "top_k_top_p_sampling_from_probs":
        return testTopKTopPSamplingFromProbs(args)
    elif args.routine == "top_k_top_p_sampling_from_logits":
        return testTopKTopPSamplingFromLogits(args)
    elif args.routine == "min_p_sampling_from_probs":
        return testMinPSamplingFromProbs(args)
    elif args.routine == "top_k_renorm_probs":
        return testTopKRenormProbs(args)
    elif args.routine == "top_p_renorm_probs":
        return testTopPRenormProbs(args)
    elif args.routine == "top_k_mask_logits":
        return testTopKMaskLogits(args)
    elif args.routine == "chain_speculative_sampling":
        return testChainSpeculativeSampling(args)
    elif args.routine == "top_k":
        return testTopK(args)
    elif args.routine == "top_k_page_table_transform":
        return testTopKPageTableTransform(args)
    elif args.routine == "top_k_ragged_transform":
        return testTopKRaggedTransform(args)
    else:
        raise ValueError(f"Unsupported routine: {args.routine}")


def parse_sampling_args(line, parser):
    """
    Parse command line arguments for sampling test configuration.

    Args:
        line: Command line arguments
        parser: ArgumentParser object already populated with shared arguments

    Returns:
        Parsed argument namespace
    """
    # Routines that don't use vocab_size (they use max_len instead)
    routines_without_vocab_size = [
        "top_k_page_table_transform",
        "top_k_ragged_transform",
    ]

    # Pre-parse to check routine for conditional requirements
    pre_parser = parser
    pre_args, _ = pre_parser.parse_known_args(line[:])

    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,
        help="Batch size.",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        required=(pre_args.routine not in routines_without_vocab_size),
        default=None,
        help="Vocabulary size.",
    )
    parser.add_argument(
        "--input_dtype",
        type=str,
        required=False,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Data type of the input tensor.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        required=False,
        default=50,
        help="Top-K value for top-k sampling.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        required=False,
        default=0.9,
        help="Top-P threshold for top-p sampling.",
    )
    parser.add_argument(
        "--min_p",
        type=float,
        required=False,
        default=0.1,
        help="Min-P threshold for min-p sampling.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        required=False,
        default=1.0,
        help="Temperature for softmax.",
    )
    parser.add_argument(
        "--filter_apply_order",
        type=str,
        required=False,
        default="top_k_first",
        choices=["top_k_first", "joint"],
        help="Order of applying top-k and top-p filters.",
    )
    parser.add_argument(
        "--num_speculate_tokens",
        type=int,
        required=False,
        default=5,
        help="Number of speculative tokens for chain speculative sampling.",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        required=False,
        default=4096,
        help="Max sequence length for top_k_page_table_transform and top_k_ragged_transform.",
    )
    parser.add_argument(
        "--num_rows",
        type=int,
        required=False,
        default=None,
        help="Number of rows for top_k_page_table_transform and top_k_ragged_transform. Defaults to batch_size.",
    )
    parser.add_argument(
        "--backends",
        type=str,
        required=False,
        nargs="+",
        default=["cuda"],
        choices=["cuda"],
        help="Kernel backends to test. Default: cuda",
    )

    args = parser.parse_args(line)

    # Default num_rows to batch_size if not specified
    if args.num_rows is None:
        args.num_rows = args.batch_size

    if args.verbose >= 1:
        print(f"[INFO] {args = }")
    return args


def testSoftmax(args):
    """
    Test softmax API.

    This test:
    1. Generates random input logits
    2. Runs flashinfer.sampling.softmax
    3. Measures performance metrics (TB/sec)

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    if args.verbose >= 1:
        print("[INFO] Running testSoftmax")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    ## Parse input arguments
    backends = args.backends[:]
    batch_size = args.batch_size
    vocab_size = args.vocab_size
    temperature = args.temperature
    is_cuda_graph_compatible = not args.no_cuda_graph
    run_refcheck = args.refcheck
    res = []

    backends = filter_backends_by_compute_capability(backends, args.routine, device)
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

    input_dtype = dtype_str_to_torch_dtype(args.input_dtype)

    ## Prepare input tensors
    logits = torch.randn(batch_size, vocab_size, dtype=input_dtype, device=device)

    if args.verbose >= 2:
        print(f"[VVERBOSE] {logits.shape = }")
        print(f"[VVERBOSE] {logits.dtype = }")

    def run_backend(backend, logits):
        if backend == "cuda":
            return flashinfer.sampling.softmax(logits, temperature=temperature)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    # Storage for timing results and outputs for refcheck
    backend_times = {backend: [] for backend in backends}
    outputs = {}
    for cur_backend in backends:
        if run_refcheck:
            outputs[cur_backend] = run_backend(cur_backend, logits).detach()
        backend_times[cur_backend] = bench_gpu_time(
            fn=run_backend,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            enable_cupti=args.use_cupti,
            use_cuda_graph=is_cuda_graph_compatible,
            input_args=(cur_backend, logits),
        )

    # Reference check: compare against PyTorch softmax
    if run_refcheck and outputs:
        reference_output = torch.softmax(logits.float() / temperature, dim=-1)
        for backend, output in outputs.items():
            num_diff, num_total, pct_diff = is_close_stats(
                reference_output, output.float(), rtol=1e-3, atol=1e-5
            )
            if num_diff > 0:
                print(
                    f"[REFCHECK] Backend {backend}: {num_diff}/{num_total} "
                    f"({pct_diff:.2f}%) elements differ from PyTorch reference"
                )
                if not args.allow_output_mismatch:
                    raise AssertionError(
                        f"[ERROR] Backend {backend} output mismatch with {num_diff} elements"
                    )

    for backend in backends:
        if len(backend_times[backend]) > 0:
            median_time = np.median(backend_times[backend])
            std_time = np.std(backend_times[backend])

            # Memory bandwidth calculation for softmax
            # Read: logits (input)
            # Write: probs (float32 output)
            problem_bytes = (
                batch_size * vocab_size * input_dtype.itemsize  # input read
                + batch_size * vocab_size * 4  # output write (float32)
            )
            tflops = 0  # Memory-bound operation
            tb_per_sec = problem_bytes / (10**9 * median_time)  # in TB/sec

            print_perf_metrics(backend, median_time, std_time, tflops, tb_per_sec)

            if args.output_path is not None:
                cur_res = defaultdict(str)
                cur_res["routine"] = args.routine
                cur_res["median_time"] = median_time
                cur_res["std_time"] = std_time
                cur_res["tflops"] = tflops
                cur_res["tb_per_sec"] = tb_per_sec
                cur_res["vocab_size"] = vocab_size
                cur_res["temperature"] = temperature
                cur_res["backend"] = backend
                cur_res["case_tag"] = args.case_tag
                res.append(cur_res)
    return res


def testSamplingFromProbs(args):
    """
    Test sampling_from_probs API.

    This test:
    1. Generates random input probabilities
    2. Runs flashinfer.sampling.sampling_from_probs
    3. Measures performance metrics (TB/sec)

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    if args.verbose >= 1:
        print("[INFO] Running testSamplingFromProbs")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    ## Parse input arguments
    backends = args.backends[:]
    batch_size = args.batch_size
    vocab_size = args.vocab_size
    # Use explicit seed/offset to enable CUDA graph compatibility
    seed = args.random_seed
    offset = 0
    is_cuda_graph_compatible = not args.no_cuda_graph
    res = []

    backends = filter_backends_by_compute_capability(backends, args.routine, device)
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

    ## Prepare input tensors (probs are always float32)
    pre_norm_prob = torch.rand(batch_size, vocab_size, device=device)
    probs = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)

    if args.verbose >= 2:
        print(f"[VVERBOSE] {probs.shape = }")
        print(f"[VVERBOSE] {probs.dtype = }")

    def run_backend(backend, probs):
        if backend == "cuda":
            return flashinfer.sampling.sampling_from_probs(
                probs, seed=seed, offset=offset
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    # Storage for timing results
    backend_times = {backend: [] for backend in backends}
    for cur_backend in backends:
        backend_times[cur_backend] = bench_gpu_time(
            fn=run_backend,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            enable_cupti=args.use_cupti,
            use_cuda_graph=is_cuda_graph_compatible,
            input_args=(cur_backend, probs),
        )

    for backend in backends:
        if len(backend_times[backend]) > 0:
            median_time = np.median(backend_times[backend])
            std_time = np.std(backend_times[backend])

            # Memory bandwidth calculation
            # Read: probs (float32)
            # Write: samples (int32)
            problem_bytes = (
                batch_size * vocab_size * 4  # probs read (float32)
                + batch_size * 4  # samples write (int32)
            )
            tflops = 0  # Memory-bound operation
            tb_per_sec = problem_bytes / (10**9 * median_time)  # in TB/sec

            print_perf_metrics(backend, median_time, std_time, tflops, tb_per_sec)

            if args.output_path is not None:
                cur_res = defaultdict(str)
                cur_res["routine"] = args.routine
                cur_res["median_time"] = median_time
                cur_res["std_time"] = std_time
                cur_res["tflops"] = tflops
                cur_res["tb_per_sec"] = tb_per_sec
                cur_res["vocab_size"] = vocab_size
                cur_res["backend"] = backend
                cur_res["case_tag"] = args.case_tag
                res.append(cur_res)
    return res


def testSamplingFromLogits(args):
    """
    Test sampling_from_logits API.

    This test:
    1. Generates random input logits
    2. Runs flashinfer.sampling.sampling_from_logits
    3. Measures performance metrics (TB/sec)

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    if args.verbose >= 1:
        print("[INFO] Running testSamplingFromLogits")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    ## Parse input arguments
    backends = args.backends[:]
    batch_size = args.batch_size
    vocab_size = args.vocab_size
    # Use explicit seed/offset to enable CUDA graph compatibility
    seed = args.random_seed
    offset = 0
    is_cuda_graph_compatible = not args.no_cuda_graph
    res = []

    backends = filter_backends_by_compute_capability(backends, args.routine, device)
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

    input_dtype = dtype_str_to_torch_dtype(args.input_dtype)

    ## Prepare input tensors
    logits = torch.randn(batch_size, vocab_size, dtype=input_dtype, device=device)

    if args.verbose >= 2:
        print(f"[VVERBOSE] {logits.shape = }")
        print(f"[VVERBOSE] {logits.dtype = }")

    def run_backend(backend, logits):
        if backend == "cuda":
            return flashinfer.sampling.sampling_from_logits(
                logits, seed=seed, offset=offset
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    # Storage for timing results
    backend_times = {backend: [] for backend in backends}
    for cur_backend in backends:
        backend_times[cur_backend] = bench_gpu_time(
            fn=run_backend,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            enable_cupti=args.use_cupti,
            use_cuda_graph=is_cuda_graph_compatible,
            input_args=(cur_backend, logits),
        )

    for backend in backends:
        if len(backend_times[backend]) > 0:
            median_time = np.median(backend_times[backend])
            std_time = np.std(backend_times[backend])

            # Memory bandwidth calculation
            # Read: logits (input_dtype)
            # Write: samples (int32)
            problem_bytes = (
                batch_size * vocab_size * input_dtype.itemsize  # logits read
                + batch_size * 4  # samples write (int32)
            )
            tflops = 0  # Memory-bound operation
            tb_per_sec = problem_bytes / (10**9 * median_time)  # in TB/sec

            print_perf_metrics(backend, median_time, std_time, tflops, tb_per_sec)

            if args.output_path is not None:
                cur_res = defaultdict(str)
                cur_res["routine"] = args.routine
                cur_res["median_time"] = median_time
                cur_res["std_time"] = std_time
                cur_res["tflops"] = tflops
                cur_res["tb_per_sec"] = tb_per_sec
                cur_res["vocab_size"] = vocab_size
                cur_res["backend"] = backend
                cur_res["case_tag"] = args.case_tag
                res.append(cur_res)
    return res


def testTopKSamplingFromProbs(args):
    """
    Test top_k_sampling_from_probs API.

    This test:
    1. Generates random input probabilities
    2. Runs flashinfer.sampling.top_k_sampling_from_probs
    3. Measures performance metrics (TB/sec)

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    if args.verbose >= 1:
        print("[INFO] Running testTopKSamplingFromProbs")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    ## Parse input arguments
    backends = args.backends[:]
    batch_size = args.batch_size
    vocab_size = args.vocab_size
    top_k = args.top_k
    # Use explicit seed/offset to enable CUDA graph compatibility
    seed = args.random_seed
    offset = 0
    is_cuda_graph_compatible = not args.no_cuda_graph
    res = []

    backends = filter_backends_by_compute_capability(backends, args.routine, device)
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

    ## Prepare input tensors
    pre_norm_prob = torch.rand(batch_size, vocab_size, device=device)
    probs = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)

    if args.verbose >= 2:
        print(f"[VVERBOSE] {probs.shape = }")
        print(f"[VVERBOSE] {probs.dtype = }")
        print(f"[VVERBOSE] {top_k = }")

    def run_backend(backend, probs):
        if backend == "cuda":
            return flashinfer.sampling.top_k_sampling_from_probs(
                probs, top_k, seed=seed, offset=offset
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    # Storage for timing results
    backend_times = {backend: [] for backend in backends}
    for cur_backend in backends:
        backend_times[cur_backend] = bench_gpu_time(
            fn=run_backend,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            enable_cupti=args.use_cupti,
            use_cuda_graph=is_cuda_graph_compatible,
            input_args=(cur_backend, probs),
        )

    for backend in backends:
        if len(backend_times[backend]) > 0:
            median_time = np.median(backend_times[backend])
            std_time = np.std(backend_times[backend])

            # Memory bandwidth calculation
            # Read: probs (float32)
            # Write: samples (int32)
            problem_bytes = (
                batch_size * vocab_size * 4  # probs read (float32)
                + batch_size * 4  # samples write (int32)
            )
            tflops = 0  # Memory-bound operation
            tb_per_sec = problem_bytes / (10**9 * median_time)  # in TB/sec

            print_perf_metrics(backend, median_time, std_time, tflops, tb_per_sec)

            if args.output_path is not None:
                cur_res = defaultdict(str)
                cur_res["routine"] = args.routine
                cur_res["median_time"] = median_time
                cur_res["std_time"] = std_time
                cur_res["tflops"] = tflops
                cur_res["tb_per_sec"] = tb_per_sec
                cur_res["vocab_size"] = vocab_size
                cur_res["top_k"] = top_k
                cur_res["backend"] = backend
                cur_res["case_tag"] = args.case_tag
                res.append(cur_res)
    return res


def testTopPSamplingFromProbs(args):
    """
    Test top_p_sampling_from_probs API.

    This test:
    1. Generates random input probabilities
    2. Runs flashinfer.sampling.top_p_sampling_from_probs
    3. Measures performance metrics (TB/sec)

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    if args.verbose >= 1:
        print("[INFO] Running testTopPSamplingFromProbs")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    ## Parse input arguments
    backends = args.backends[:]
    batch_size = args.batch_size
    vocab_size = args.vocab_size
    top_p = args.top_p
    # Use explicit seed/offset to enable CUDA graph compatibility
    seed = args.random_seed
    offset = 0
    is_cuda_graph_compatible = not args.no_cuda_graph
    res = []

    backends = filter_backends_by_compute_capability(backends, args.routine, device)
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

    ## Prepare input tensors
    pre_norm_prob = torch.rand(batch_size, vocab_size, device=device)
    probs = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)

    if args.verbose >= 2:
        print(f"[VVERBOSE] {probs.shape = }")
        print(f"[VVERBOSE] {probs.dtype = }")
        print(f"[VVERBOSE] {top_p = }")

    def run_backend(backend, probs):
        if backend == "cuda":
            return flashinfer.sampling.top_p_sampling_from_probs(
                probs, top_p, seed=seed, offset=offset
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    # Storage for timing results
    backend_times = {backend: [] for backend in backends}
    for cur_backend in backends:
        backend_times[cur_backend] = bench_gpu_time(
            fn=run_backend,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            enable_cupti=args.use_cupti,
            use_cuda_graph=is_cuda_graph_compatible,
            input_args=(cur_backend, probs),
        )

    for backend in backends:
        if len(backend_times[backend]) > 0:
            median_time = np.median(backend_times[backend])
            std_time = np.std(backend_times[backend])

            # Memory bandwidth calculation
            # Read: probs (float32)
            # Write: samples (int32)
            problem_bytes = (
                batch_size * vocab_size * 4  # probs read (float32)
                + batch_size * 4  # samples write (int32)
            )
            tflops = 0  # Memory-bound operation
            tb_per_sec = problem_bytes / (10**9 * median_time)  # in TB/sec

            print_perf_metrics(backend, median_time, std_time, tflops, tb_per_sec)

            if args.output_path is not None:
                cur_res = defaultdict(str)
                cur_res["routine"] = args.routine
                cur_res["median_time"] = median_time
                cur_res["std_time"] = std_time
                cur_res["tflops"] = tflops
                cur_res["tb_per_sec"] = tb_per_sec
                cur_res["vocab_size"] = vocab_size
                cur_res["top_p"] = top_p
                cur_res["backend"] = backend
                cur_res["case_tag"] = args.case_tag
                res.append(cur_res)
    return res


def testTopKTopPSamplingFromProbs(args):
    """
    Test top_k_top_p_sampling_from_probs API.

    This test:
    1. Generates random input probabilities
    2. Runs flashinfer.sampling.top_k_top_p_sampling_from_probs
    3. Measures performance metrics (TB/sec)

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    if args.verbose >= 1:
        print("[INFO] Running testTopKTopPSamplingFromProbs")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    ## Parse input arguments
    backends = args.backends[:]
    batch_size = args.batch_size
    vocab_size = args.vocab_size
    top_k = args.top_k
    top_p = args.top_p
    filter_apply_order = args.filter_apply_order
    # Use explicit seed/offset to enable CUDA graph compatibility
    seed = args.random_seed
    offset = 0
    is_cuda_graph_compatible = not args.no_cuda_graph
    res = []

    backends = filter_backends_by_compute_capability(backends, args.routine, device)
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

    ## Prepare input tensors
    pre_norm_prob = torch.rand(batch_size, vocab_size, device=device)
    probs = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)

    if args.verbose >= 2:
        print(f"[VVERBOSE] {probs.shape = }")
        print(f"[VVERBOSE] {probs.dtype = }")
        print(f"[VVERBOSE] {top_k = }")
        print(f"[VVERBOSE] {top_p = }")
        print(f"[VVERBOSE] {filter_apply_order = }")

    def run_backend(backend, probs):
        if backend == "cuda":
            return flashinfer.sampling.top_k_top_p_sampling_from_probs(
                probs,
                top_k,
                top_p,
                filter_apply_order=filter_apply_order,
                seed=seed,
                offset=offset,
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    # Storage for timing results
    backend_times = {backend: [] for backend in backends}
    for cur_backend in backends:
        backend_times[cur_backend] = bench_gpu_time(
            fn=run_backend,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            enable_cupti=args.use_cupti,
            use_cuda_graph=is_cuda_graph_compatible,
            input_args=(cur_backend, probs),
        )

    for backend in backends:
        if len(backend_times[backend]) > 0:
            median_time = np.median(backend_times[backend])
            std_time = np.std(backend_times[backend])

            # Memory bandwidth calculation
            # Read: probs (float32)
            # Write: samples (int32)
            problem_bytes = (
                batch_size * vocab_size * 4  # probs read (float32)
                + batch_size * 4  # samples write (int32)
            )
            tflops = 0  # Memory-bound operation
            tb_per_sec = problem_bytes / (10**9 * median_time)  # in TB/sec

            print_perf_metrics(backend, median_time, std_time, tflops, tb_per_sec)

            if args.output_path is not None:
                cur_res = defaultdict(str)
                cur_res["routine"] = args.routine
                cur_res["median_time"] = median_time
                cur_res["std_time"] = std_time
                cur_res["tflops"] = tflops
                cur_res["tb_per_sec"] = tb_per_sec
                cur_res["vocab_size"] = vocab_size
                cur_res["top_k"] = top_k
                cur_res["top_p"] = top_p
                cur_res["filter_apply_order"] = filter_apply_order
                cur_res["backend"] = backend
                cur_res["case_tag"] = args.case_tag
                res.append(cur_res)
    return res


def testTopKTopPSamplingFromLogits(args):
    """
    Test top_k_top_p_sampling_from_logits API.

    This test:
    1. Generates random input logits
    2. Runs flashinfer.sampling.top_k_top_p_sampling_from_logits
    3. Measures performance metrics (TB/sec)

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    if args.verbose >= 1:
        print("[INFO] Running testTopKTopPSamplingFromLogits")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    ## Parse input arguments
    backends = args.backends[:]
    batch_size = args.batch_size
    vocab_size = args.vocab_size
    top_k = args.top_k
    top_p = args.top_p
    filter_apply_order = args.filter_apply_order
    # Use explicit seed/offset to enable CUDA graph compatibility
    seed = args.random_seed
    offset = 0
    is_cuda_graph_compatible = not args.no_cuda_graph
    res = []

    backends = filter_backends_by_compute_capability(backends, args.routine, device)
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

    input_dtype = dtype_str_to_torch_dtype(args.input_dtype)

    ## Prepare input tensors
    logits = torch.randn(batch_size, vocab_size, dtype=input_dtype, device=device)

    if args.verbose >= 2:
        print(f"[VVERBOSE] {logits.shape = }")
        print(f"[VVERBOSE] {logits.dtype = }")
        print(f"[VVERBOSE] {top_k = }")
        print(f"[VVERBOSE] {top_p = }")
        print(f"[VVERBOSE] {filter_apply_order = }")

    def run_backend(backend, logits):
        if backend == "cuda":
            return flashinfer.sampling.top_k_top_p_sampling_from_logits(
                logits,
                top_k,
                top_p,
                filter_apply_order=filter_apply_order,
                seed=seed,
                offset=offset,
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    # Storage for timing results
    backend_times = {backend: [] for backend in backends}
    for cur_backend in backends:
        backend_times[cur_backend] = bench_gpu_time(
            fn=run_backend,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            enable_cupti=args.use_cupti,
            use_cuda_graph=is_cuda_graph_compatible,
            input_args=(cur_backend, logits),
        )

    for backend in backends:
        if len(backend_times[backend]) > 0:
            median_time = np.median(backend_times[backend])
            std_time = np.std(backend_times[backend])

            # Memory bandwidth calculation
            # Read: logits (input_dtype)
            # Write: samples (int32)
            problem_bytes = (
                batch_size * vocab_size * input_dtype.itemsize  # logits read
                + batch_size * 4  # samples write (int32)
            )
            tflops = 0  # Memory-bound operation
            tb_per_sec = problem_bytes / (10**9 * median_time)  # in TB/sec

            print_perf_metrics(backend, median_time, std_time, tflops, tb_per_sec)

            if args.output_path is not None:
                cur_res = defaultdict(str)
                cur_res["routine"] = args.routine
                cur_res["median_time"] = median_time
                cur_res["std_time"] = std_time
                cur_res["tflops"] = tflops
                cur_res["tb_per_sec"] = tb_per_sec
                cur_res["vocab_size"] = vocab_size
                cur_res["top_k"] = top_k
                cur_res["top_p"] = top_p
                cur_res["filter_apply_order"] = filter_apply_order
                cur_res["backend"] = backend
                cur_res["case_tag"] = args.case_tag
                res.append(cur_res)
    return res


def testMinPSamplingFromProbs(args):
    """
    Test min_p_sampling_from_probs API.

    This test:
    1. Generates random input probabilities
    2. Runs flashinfer.sampling.min_p_sampling_from_probs
    3. Measures performance metrics (TB/sec)

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    if args.verbose >= 1:
        print("[INFO] Running testMinPSamplingFromProbs")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    ## Parse input arguments
    backends = args.backends[:]
    batch_size = args.batch_size
    vocab_size = args.vocab_size
    min_p = args.min_p
    # Use explicit seed/offset to enable CUDA graph compatibility
    seed = args.random_seed
    offset = 0
    is_cuda_graph_compatible = not args.no_cuda_graph
    res = []

    backends = filter_backends_by_compute_capability(backends, args.routine, device)
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

    ## Prepare input tensors
    pre_norm_prob = torch.rand(batch_size, vocab_size, device=device)
    probs = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)

    if args.verbose >= 2:
        print(f"[VVERBOSE] {probs.shape = }")
        print(f"[VVERBOSE] {probs.dtype = }")
        print(f"[VVERBOSE] {min_p = }")

    def run_backend(backend, probs):
        if backend == "cuda":
            return flashinfer.sampling.min_p_sampling_from_probs(
                probs, min_p, seed=seed, offset=offset
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    # Storage for timing results
    backend_times = {backend: [] for backend in backends}
    for cur_backend in backends:
        backend_times[cur_backend] = bench_gpu_time(
            fn=run_backend,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            enable_cupti=args.use_cupti,
            use_cuda_graph=is_cuda_graph_compatible,
            input_args=(cur_backend, probs),
        )

    for backend in backends:
        if len(backend_times[backend]) > 0:
            median_time = np.median(backend_times[backend])
            std_time = np.std(backend_times[backend])

            # Memory bandwidth calculation
            # Read: probs (float32)
            # Write: samples (int32)
            problem_bytes = (
                batch_size * vocab_size * 4  # probs read (float32)
                + batch_size * 4  # samples write (int32)
            )
            tflops = 0  # Memory-bound operation
            tb_per_sec = problem_bytes / (10**9 * median_time)  # in TB/sec

            print_perf_metrics(backend, median_time, std_time, tflops, tb_per_sec)

            if args.output_path is not None:
                cur_res = defaultdict(str)
                cur_res["routine"] = args.routine
                cur_res["median_time"] = median_time
                cur_res["std_time"] = std_time
                cur_res["tflops"] = tflops
                cur_res["tb_per_sec"] = tb_per_sec
                cur_res["vocab_size"] = vocab_size
                cur_res["min_p"] = min_p
                cur_res["backend"] = backend
                cur_res["case_tag"] = args.case_tag
                res.append(cur_res)
    return res


def testTopKRenormProbs(args):
    """
    Test top_k_renorm_probs API.

    This test:
    1. Generates random input probabilities
    2. Runs flashinfer.sampling.top_k_renorm_probs
    3. Measures performance metrics (TB/sec)

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    if args.verbose >= 1:
        print("[INFO] Running testTopKRenormProbs")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    ## Parse input arguments
    backends = args.backends[:]
    batch_size = args.batch_size
    vocab_size = args.vocab_size
    top_k = args.top_k
    is_cuda_graph_compatible = not args.no_cuda_graph
    run_refcheck = args.refcheck
    res = []

    backends = filter_backends_by_compute_capability(backends, args.routine, device)
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

    input_dtype = dtype_str_to_torch_dtype(args.input_dtype)

    ## Prepare input tensors
    pre_norm_prob = torch.rand(batch_size, vocab_size, dtype=input_dtype, device=device)
    probs = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)

    if args.verbose >= 2:
        print(f"[VVERBOSE] {probs.shape = }")
        print(f"[VVERBOSE] {probs.dtype = }")
        print(f"[VVERBOSE] {top_k = }")

    def run_backend(backend, probs):
        if backend == "cuda":
            return flashinfer.sampling.top_k_renorm_probs(probs, top_k)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    # Storage for timing results and outputs for refcheck
    backend_times = {backend: [] for backend in backends}
    outputs = {}
    for cur_backend in backends:
        if run_refcheck:
            outputs[cur_backend] = run_backend(cur_backend, probs).detach()
        backend_times[cur_backend] = bench_gpu_time(
            fn=run_backend,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            enable_cupti=args.use_cupti,
            use_cuda_graph=is_cuda_graph_compatible,
            input_args=(cur_backend, probs),
        )

    # Reference check: PyTorch implementation of top-k renormalization
    # Keep top-k values, set rest to 0, then renormalize
    # Note: Small mismatches can occur due to tie-breaking at the k-th boundary
    if run_refcheck and outputs:
        topk_vals, topk_indices = torch.topk(probs.float(), k=top_k, dim=-1)
        reference_output = torch.zeros_like(probs, dtype=torch.float32)
        reference_output.scatter_(-1, topk_indices, topk_vals)
        reference_output = reference_output / reference_output.sum(dim=-1, keepdim=True)
        reference_output = reference_output.to(input_dtype)

        for backend, output in outputs.items():
            num_diff, num_total, pct_diff = is_close_stats(
                reference_output.float(), output.float(), rtol=1e-2, atol=1e-2
            )
            # Allow tiny mismatch percentage (<0.01%) due to tie-breaking at k-th boundary
            if pct_diff > 0.01:
                print(
                    f"[REFCHECK] Backend {backend}: {num_diff}/{num_total} "
                    f"({pct_diff:.2f}%) elements differ from PyTorch reference"
                )
                if not args.allow_output_mismatch:
                    raise AssertionError(
                        f"[ERROR] Backend {backend} output mismatch with {num_diff} elements"
                    )
            elif num_diff > 0 and args.verbose >= 1:
                print(
                    f"[REFCHECK] Backend {backend}: {num_diff}/{num_total} "
                    f"({pct_diff:.4f}%) elements differ (within acceptable threshold)"
                )

    for backend in backends:
        if len(backend_times[backend]) > 0:
            median_time = np.median(backend_times[backend])
            std_time = np.std(backend_times[backend])

            # Memory bandwidth calculation
            # Read: probs (input_dtype)
            # Write: renorm_probs (input_dtype)
            problem_bytes = (
                batch_size * vocab_size * input_dtype.itemsize  # probs read
                + batch_size * vocab_size * input_dtype.itemsize  # renorm_probs write
            )
            tflops = 0  # Memory-bound operation
            tb_per_sec = problem_bytes / (10**9 * median_time)  # in TB/sec

            print_perf_metrics(backend, median_time, std_time, tflops, tb_per_sec)

            if args.output_path is not None:
                cur_res = defaultdict(str)
                cur_res["routine"] = args.routine
                cur_res["median_time"] = median_time
                cur_res["std_time"] = std_time
                cur_res["tflops"] = tflops
                cur_res["tb_per_sec"] = tb_per_sec
                cur_res["vocab_size"] = vocab_size
                cur_res["top_k"] = top_k
                cur_res["backend"] = backend
                cur_res["case_tag"] = args.case_tag
                res.append(cur_res)
    return res


def testTopPRenormProbs(args):
    """
    Test top_p_renorm_probs API.

    This test:
    1. Generates random input probabilities
    2. Runs flashinfer.sampling.top_p_renorm_probs
    3. Measures performance metrics (TB/sec)

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    if args.verbose >= 1:
        print("[INFO] Running testTopPRenormProbs")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    ## Parse input arguments
    backends = args.backends[:]
    batch_size = args.batch_size
    vocab_size = args.vocab_size
    top_p = args.top_p
    is_cuda_graph_compatible = not args.no_cuda_graph
    run_refcheck = args.refcheck
    res = []

    backends = filter_backends_by_compute_capability(backends, args.routine, device)
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

    ## Prepare input tensors (top_p_renorm_probs uses float32)
    pre_norm_prob = torch.rand(batch_size, vocab_size, device=device)
    probs = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)

    if args.verbose >= 2:
        print(f"[VVERBOSE] {probs.shape = }")
        print(f"[VVERBOSE] {probs.dtype = }")
        print(f"[VVERBOSE] {top_p = }")

    def run_backend(backend, probs):
        if backend == "cuda":
            return flashinfer.sampling.top_p_renorm_probs(probs, top_p)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    # Storage for timing results and outputs for refcheck
    backend_times = {backend: [] for backend in backends}
    outputs = {}
    for cur_backend in backends:
        if run_refcheck:
            outputs[cur_backend] = run_backend(cur_backend, probs).detach()
        backend_times[cur_backend] = bench_gpu_time(
            fn=run_backend,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            enable_cupti=args.use_cupti,
            use_cuda_graph=is_cuda_graph_compatible,
            input_args=(cur_backend, probs),
        )

    # Reference check: PyTorch implementation of top-p renormalization
    # Sort probs descending, compute cumsum, threshold at top_p, renormalize
    if run_refcheck and outputs:
        sorted_probs, sorted_indices = torch.sort(
            probs.float(), dim=-1, descending=True
        )
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
        # Create mask: keep probs where cumsum <= top_p (shift by 1 to include the boundary element)
        cumsum_shifted = torch.cat(
            [torch.zeros_like(cumsum_probs[:, :1]), cumsum_probs[:, :-1]], dim=-1
        )
        mask = cumsum_shifted < top_p
        # Keep at least one element per row
        mask[:, 0] = True
        # Zero out elements beyond top-p threshold
        sorted_probs_masked = sorted_probs * mask.float()
        # Scatter back to original positions
        reference_output = torch.zeros_like(probs)
        reference_output.scatter_(-1, sorted_indices, sorted_probs_masked)
        # Renormalize
        reference_output = reference_output / reference_output.sum(dim=-1, keepdim=True)

        for backend, output in outputs.items():
            num_diff, num_total, pct_diff = is_close_stats(
                reference_output, output.float(), rtol=1e-2, atol=1e-3
            )
            if num_diff > 0:
                print(
                    f"[REFCHECK] Backend {backend}: {num_diff}/{num_total} "
                    f"({pct_diff:.2f}%) elements differ from PyTorch reference"
                )
                if not args.allow_output_mismatch:
                    raise AssertionError(
                        f"[ERROR] Backend {backend} output mismatch with {num_diff} elements"
                    )

    for backend in backends:
        if len(backend_times[backend]) > 0:
            median_time = np.median(backend_times[backend])
            std_time = np.std(backend_times[backend])

            # Memory bandwidth calculation
            # Read: probs (float32)
            # Write: renorm_probs (float32)
            problem_bytes = (
                batch_size * vocab_size * 4  # probs read (float32)
                + batch_size * vocab_size * 4  # renorm_probs write (float32)
            )
            tflops = 0  # Memory-bound operation
            tb_per_sec = problem_bytes / (10**9 * median_time)  # in TB/sec

            print_perf_metrics(backend, median_time, std_time, tflops, tb_per_sec)

            if args.output_path is not None:
                cur_res = defaultdict(str)
                cur_res["routine"] = args.routine
                cur_res["median_time"] = median_time
                cur_res["std_time"] = std_time
                cur_res["tflops"] = tflops
                cur_res["tb_per_sec"] = tb_per_sec
                cur_res["vocab_size"] = vocab_size
                cur_res["top_p"] = top_p
                cur_res["backend"] = backend
                cur_res["case_tag"] = args.case_tag
                res.append(cur_res)
    return res


def testTopKMaskLogits(args):
    """
    Test top_k_mask_logits API.

    This test:
    1. Generates random input logits
    2. Runs flashinfer.sampling.top_k_mask_logits
    3. Measures performance metrics (TB/sec)

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    if args.verbose >= 1:
        print("[INFO] Running testTopKMaskLogits")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    ## Parse input arguments
    backends = args.backends[:]
    batch_size = args.batch_size
    vocab_size = args.vocab_size
    top_k = args.top_k
    is_cuda_graph_compatible = not args.no_cuda_graph
    run_refcheck = args.refcheck
    res = []

    backends = filter_backends_by_compute_capability(backends, args.routine, device)
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

    input_dtype = dtype_str_to_torch_dtype(args.input_dtype)

    ## Prepare input tensors
    logits = torch.randn(batch_size, vocab_size, dtype=input_dtype, device=device)

    if args.verbose >= 2:
        print(f"[VVERBOSE] {logits.shape = }")
        print(f"[VVERBOSE] {logits.dtype = }")
        print(f"[VVERBOSE] {top_k = }")

    def run_backend(backend, logits):
        if backend == "cuda":
            return flashinfer.sampling.top_k_mask_logits(logits, top_k)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    # Storage for timing results and outputs for refcheck
    backend_times = {backend: [] for backend in backends}
    outputs = {}
    for cur_backend in backends:
        if run_refcheck:
            outputs[cur_backend] = run_backend(cur_backend, logits).detach()
        backend_times[cur_backend] = bench_gpu_time(
            fn=run_backend,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            enable_cupti=args.use_cupti,
            use_cuda_graph=is_cuda_graph_compatible,
            input_args=(cur_backend, logits),
        )

    # Reference check: PyTorch implementation of top-k masking
    # Keep top-k logits, set rest to -inf
    if run_refcheck and outputs:
        topk_vals, topk_indices = torch.topk(logits.float(), k=top_k, dim=-1)
        reference_output = torch.full_like(logits, float("-inf"), dtype=torch.float32)
        reference_output.scatter_(-1, topk_indices, topk_vals)
        reference_output = reference_output.to(input_dtype)

        for backend, output in outputs.items():
            out = output.float()
            ref = reference_output.float()

            # Check that the same positions are masked (-inf)
            out_masked = torch.isinf(out) & (out < 0)
            ref_masked = torch.isinf(ref) & (ref < 0)
            mask_match = (out_masked == ref_masked).all()

            if not mask_match:
                num_mask_diff = (out_masked != ref_masked).sum().item()
                print(
                    f"[REFCHECK] Backend {backend}: Mask mismatch - "
                    f"{num_mask_diff} positions have different masking"
                )
                if not args.allow_output_mismatch:
                    raise AssertionError(f"[ERROR] Backend {backend} mask mismatch")
            else:
                # Check that unmasked values match the reference
                unmasked_positions = ~out_masked
                if unmasked_positions.any():
                    num_diff, num_total, pct_diff = is_close_stats(
                        ref[unmasked_positions],
                        out[unmasked_positions],
                        rtol=1e-3,
                        atol=1e-5,
                    )
                    if num_diff > 0:
                        print(
                            f"[REFCHECK] Backend {backend}: {num_diff}/{num_total} "
                            f"({pct_diff:.2f}%) unmasked elements differ"
                        )
                        if not args.allow_output_mismatch:
                            raise AssertionError(
                                f"[ERROR] Backend {backend} value mismatch"
                            )

    for backend in backends:
        if len(backend_times[backend]) > 0:
            median_time = np.median(backend_times[backend])
            std_time = np.std(backend_times[backend])

            # Memory bandwidth calculation
            # Read: logits (input_dtype)
            # Write: masked_logits (input_dtype)
            problem_bytes = (
                batch_size * vocab_size * input_dtype.itemsize  # logits read
                + batch_size * vocab_size * input_dtype.itemsize  # masked_logits write
            )
            tflops = 0  # Memory-bound operation
            tb_per_sec = problem_bytes / (10**9 * median_time)  # in TB/sec

            print_perf_metrics(backend, median_time, std_time, tflops, tb_per_sec)

            if args.output_path is not None:
                cur_res = defaultdict(str)
                cur_res["routine"] = args.routine
                cur_res["median_time"] = median_time
                cur_res["std_time"] = std_time
                cur_res["tflops"] = tflops
                cur_res["tb_per_sec"] = tb_per_sec
                cur_res["vocab_size"] = vocab_size
                cur_res["top_k"] = top_k
                cur_res["backend"] = backend
                cur_res["case_tag"] = args.case_tag
                res.append(cur_res)
    return res


def testChainSpeculativeSampling(args):
    """
    Test chain_speculative_sampling API.

    This test:
    1. Generates random draft and target probabilities
    2. Runs flashinfer.sampling.chain_speculative_sampling
    3. Measures performance metrics (TB/sec)

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    if args.verbose >= 1:
        print("[INFO] Running testChainSpeculativeSampling")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    ## Parse input arguments
    backends = args.backends[:]
    batch_size = args.batch_size
    vocab_size = args.vocab_size
    num_speculate_tokens = args.num_speculate_tokens
    # Use explicit seed/offset to enable CUDA graph compatibility
    seed = args.random_seed
    offset = 0
    is_cuda_graph_compatible = not args.no_cuda_graph
    res = []

    backends = filter_backends_by_compute_capability(backends, args.routine, device)
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

    ## Prepare input tensors
    # Draft probs: (batch_size, num_speculate_tokens, vocab_size)
    pre_norm_draft_prob = torch.rand(
        batch_size, num_speculate_tokens, vocab_size, device=device
    )
    draft_probs = pre_norm_draft_prob / pre_norm_draft_prob.sum(dim=-1, keepdim=True)

    # Draft token IDs: (batch_size, num_speculate_tokens)
    draft_token_ids = torch.randint(
        vocab_size, (batch_size, num_speculate_tokens), device=device, dtype=torch.int32
    )

    # Target probs: (batch_size, num_speculate_tokens + 1, vocab_size)
    pre_norm_target_prob = torch.rand(
        batch_size, num_speculate_tokens + 1, vocab_size, device=device
    )
    target_probs = pre_norm_target_prob / pre_norm_target_prob.sum(dim=-1, keepdim=True)

    if args.verbose >= 2:
        print(f"[VVERBOSE] {draft_probs.shape = }")
        print(f"[VVERBOSE] {draft_token_ids.shape = }")
        print(f"[VVERBOSE] {target_probs.shape = }")
        print(f"[VVERBOSE] {num_speculate_tokens = }")

    def run_backend(backend, draft_probs, draft_token_ids, target_probs):
        if backend == "cuda":
            return flashinfer.sampling.chain_speculative_sampling(
                draft_probs, draft_token_ids, target_probs, seed=seed, offset=offset
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    # Storage for timing results
    backend_times = {backend: [] for backend in backends}
    for cur_backend in backends:
        backend_times[cur_backend] = bench_gpu_time(
            fn=run_backend,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            enable_cupti=args.use_cupti,
            use_cuda_graph=is_cuda_graph_compatible,
            input_args=(cur_backend, draft_probs, draft_token_ids, target_probs),
        )

    for backend in backends:
        if len(backend_times[backend]) > 0:
            median_time = np.median(backend_times[backend])
            std_time = np.std(backend_times[backend])

            n = num_speculate_tokens
            # Memory bandwidth calculation
            # Read: draft_probs + draft_token_ids + target_probs
            # Write: output_token_ids + accepted_num + emitted_num
            problem_bytes = (
                batch_size * n * vocab_size * 4  # draft_probs read (float32)
                + batch_size * n * 4  # draft_token_ids read (int32)
                + batch_size * (n + 1) * vocab_size * 4  # target_probs read (float32)
                + batch_size * (n + 1) * 4  # output_token_ids write (int32)
                + batch_size * 4  # accepted_num write (int32)
                + batch_size * 4  # emitted_num write (int32)
            )
            tflops = 0  # Memory-bound operation
            tb_per_sec = problem_bytes / (10**9 * median_time)  # in TB/sec

            print_perf_metrics(backend, median_time, std_time, tflops, tb_per_sec)

            if args.output_path is not None:
                cur_res = defaultdict(str)
                cur_res["routine"] = args.routine
                cur_res["median_time"] = median_time
                cur_res["std_time"] = std_time
                cur_res["tflops"] = tflops
                cur_res["tb_per_sec"] = tb_per_sec
                cur_res["vocab_size"] = vocab_size
                cur_res["num_speculate_tokens"] = num_speculate_tokens
                cur_res["backend"] = backend
                cur_res["case_tag"] = args.case_tag
                res.append(cur_res)
    return res


def testTopK(args):
    """
    Test top_k API (radix-based top-k selection).

    This test:
    1. Generates random input tensor
    2. Runs flashinfer.top_k
    3. Measures performance metrics (TB/sec)

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    if args.verbose >= 1:
        print("[INFO] Running testTopK")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    ## Parse input arguments
    backends = args.backends[:]
    batch_size = args.batch_size
    vocab_size = args.vocab_size
    top_k = args.top_k
    is_cuda_graph_compatible = not args.no_cuda_graph
    run_refcheck = args.refcheck
    res = []

    backends = filter_backends_by_compute_capability(backends, args.routine, device)
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

    input_dtype = dtype_str_to_torch_dtype(args.input_dtype)

    ## Prepare input tensors
    input_tensor = torch.randn(batch_size, vocab_size, dtype=input_dtype, device=device)

    if args.verbose >= 2:
        print(f"[VVERBOSE] {input_tensor.shape = }")
        print(f"[VVERBOSE] {input_tensor.dtype = }")
        print(f"[VVERBOSE] {top_k = }")

    def run_backend(backend, input_tensor):
        if backend == "cuda":
            return flashinfer.top_k(input_tensor, top_k)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    # Storage for timing results and outputs for refcheck
    backend_times = {backend: [] for backend in backends}
    outputs = {}
    for cur_backend in backends:
        if run_refcheck:
            outputs[cur_backend] = run_backend(cur_backend, input_tensor)
            # top_k returns (values, indices) tuple - detach both
            outputs[cur_backend] = (
                outputs[cur_backend][0].detach(),
                outputs[cur_backend][1].detach(),
            )
        backend_times[cur_backend] = bench_gpu_time(
            fn=run_backend,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            enable_cupti=args.use_cupti,
            use_cuda_graph=is_cuda_graph_compatible,
            input_args=(cur_backend, input_tensor),
        )

    # Reference check: compare against PyTorch torch.topk
    # Note: FlashInfer top_k returns UNSORTED results by default, so we compare
    # sorted values to verify the same elements are selected
    if run_refcheck and outputs:
        ref_values, ref_indices = torch.topk(input_tensor.float(), k=top_k, dim=-1)

        for backend, (out_values, out_indices) in outputs.items():
            # Sort both outputs to compare (FlashInfer returns unsorted by default)
            ref_sorted, _ = torch.sort(ref_values, dim=-1, descending=True)
            out_sorted, _ = torch.sort(out_values.float(), dim=-1, descending=True)

            # Check sorted values match
            num_diff_vals, num_total_vals, pct_diff_vals = is_close_stats(
                ref_sorted, out_sorted, rtol=1e-3, atol=1e-5
            )

            # Verify indices point to correct values in original tensor
            gathered_vals = torch.gather(
                input_tensor.float(), dim=-1, index=out_indices
            )
            idx_vals_match = torch.allclose(
                gathered_vals, out_values.float(), rtol=1e-3, atol=1e-5
            )

            if num_diff_vals > 0 or not idx_vals_match:
                if num_diff_vals > 0:
                    print(
                        f"[REFCHECK] Backend {backend}: {num_diff_vals}/{num_total_vals} "
                        f"({pct_diff_vals:.2f}%) sorted values differ from PyTorch reference"
                    )
                if not idx_vals_match:
                    print(
                        f"[REFCHECK] Backend {backend}: indices don't match their values"
                    )
                if not args.allow_output_mismatch:
                    raise AssertionError(f"[ERROR] Backend {backend} output mismatch")

    for backend in backends:
        if len(backend_times[backend]) > 0:
            median_time = np.median(backend_times[backend])
            std_time = np.std(backend_times[backend])

            # Memory bandwidth calculation
            # Read: input (input_dtype)
            # Write: values (input_dtype) + indices (int64)
            problem_bytes = (
                batch_size * vocab_size * input_dtype.itemsize  # input read
                + batch_size * top_k * input_dtype.itemsize  # values write
                + batch_size * top_k * 8  # indices write (int64)
            )
            tflops = 0  # Memory-bound operation
            tb_per_sec = problem_bytes / (10**9 * median_time)  # in TB/sec

            print_perf_metrics(backend, median_time, std_time, tflops, tb_per_sec)

            if args.output_path is not None:
                cur_res = defaultdict(str)
                cur_res["routine"] = args.routine
                cur_res["median_time"] = median_time
                cur_res["std_time"] = std_time
                cur_res["tflops"] = tflops
                cur_res["tb_per_sec"] = tb_per_sec
                cur_res["vocab_size"] = vocab_size
                cur_res["top_k"] = top_k
                cur_res["backend"] = backend
                cur_res["case_tag"] = args.case_tag
                res.append(cur_res)
    return res


def testTopKPageTableTransform(args):
    """
    Test top_k_page_table_transform API.

    This test:
    1. Generates random input scores and page table
    2. Runs flashinfer.top_k_page_table_transform
    3. Measures performance metrics (TB/sec)

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    if args.verbose >= 1:
        print("[INFO] Running testTopKPageTableTransform")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    ## Parse input arguments
    backends = args.backends[:]
    batch_size = args.batch_size
    num_rows = args.num_rows
    max_len = args.max_len
    top_k = args.top_k
    is_cuda_graph_compatible = not args.no_cuda_graph
    run_refcheck = args.refcheck
    res = []

    backends = filter_backends_by_compute_capability(backends, args.routine, device)
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

    input_dtype = dtype_str_to_torch_dtype(args.input_dtype)

    ## Prepare input tensors
    # Input scores: (num_rows, max_len)
    input_scores = torch.randn(num_rows, max_len, dtype=input_dtype, device=device)

    # Source page table: (batch_size, max_len)
    src_page_table = torch.randint(
        0, 1000, (batch_size, max_len), dtype=torch.int32, device=device
    )

    # Lengths: (num_rows,)
    lengths = torch.full((num_rows,), max_len, dtype=torch.int32, device=device)

    if args.verbose >= 2:
        print(f"[VVERBOSE] {input_scores.shape = }")
        print(f"[VVERBOSE] {input_scores.dtype = }")
        print(f"[VVERBOSE] {src_page_table.shape = }")
        print(f"[VVERBOSE] {lengths.shape = }")
        print(f"[VVERBOSE] {top_k = }")

    def run_backend(backend, input_scores, src_page_table, lengths):
        if backend == "cuda":
            return flashinfer.top_k_page_table_transform(
                input_scores, src_page_table, lengths, top_k
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    # Storage for timing results and outputs for refcheck
    backend_times = {backend: [] for backend in backends}
    outputs = {}
    for cur_backend in backends:
        if run_refcheck:
            outputs[cur_backend] = run_backend(
                cur_backend, input_scores, src_page_table, lengths
            ).detach()
        backend_times[cur_backend] = bench_gpu_time(
            fn=run_backend,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            enable_cupti=args.use_cupti,
            use_cuda_graph=is_cuda_graph_compatible,
            input_args=(cur_backend, input_scores, src_page_table, lengths),
        )

    # Reference check: PyTorch implementation of top-k + page table transform
    # For each row i: output[i, j] = src_page_table[i, topk_indices[i, j]]
    # Note: FlashInfer uses unsorted top-k internally, so we compare sorted sets per row
    if run_refcheck and outputs:
        # Get top-k indices
        _, topk_indices = torch.topk(input_scores.float(), k=top_k, dim=-1)
        # Gather from page table - row index maps to batch index (1:1 when row_to_batch is None)
        reference_output = torch.gather(
            src_page_table[:num_rows], dim=-1, index=topk_indices.int()
        )

        for backend, output in outputs.items():
            # Sort both outputs per row to compare sets (order may differ due to unsorted top-k)
            ref_sorted, _ = torch.sort(reference_output, dim=-1)
            out_sorted, _ = torch.sort(output, dim=-1)

            matches = (ref_sorted == out_sorted).all()
            if not matches:
                num_diff = (ref_sorted != out_sorted).sum().item()
                num_total = reference_output.numel()
                pct_diff = num_diff / num_total * 100.0
                print(
                    f"[REFCHECK] Backend {backend}: {num_diff}/{num_total} "
                    f"({pct_diff:.2f}%) sorted page table entries differ from PyTorch reference"
                )
                if not args.allow_output_mismatch:
                    raise AssertionError(
                        f"[ERROR] Backend {backend} output mismatch with {num_diff} elements"
                    )

    for backend in backends:
        if len(backend_times[backend]) > 0:
            median_time = np.median(backend_times[backend])
            std_time = np.std(backend_times[backend])

            # Memory bandwidth calculation
            # Read: input_scores + src_page_table
            # Write: output_page_table
            problem_bytes = (
                num_rows * max_len * input_dtype.itemsize  # input_scores read
                + batch_size * max_len * 4  # src_page_table read (int32)
                + num_rows * top_k * 4  # output_page_table write (int32)
            )
            tflops = 0  # Memory-bound operation
            tb_per_sec = problem_bytes / (10**9 * median_time)  # in TB/sec

            print_perf_metrics(backend, median_time, std_time, tflops, tb_per_sec)

            if args.output_path is not None:
                cur_res = defaultdict(str)
                cur_res["routine"] = args.routine
                cur_res["median_time"] = median_time
                cur_res["std_time"] = std_time
                cur_res["tflops"] = tflops
                cur_res["tb_per_sec"] = tb_per_sec
                cur_res["max_len"] = max_len
                cur_res["num_rows"] = num_rows
                cur_res["top_k"] = top_k
                cur_res["backend"] = backend
                cur_res["case_tag"] = args.case_tag
                res.append(cur_res)
    return res


def testTopKRaggedTransform(args):
    """
    Test top_k_ragged_transform API.

    This test:
    1. Generates random input scores
    2. Runs flashinfer.top_k_ragged_transform
    3. Measures performance metrics (TB/sec)

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    if args.verbose >= 1:
        print("[INFO] Running testTopKRaggedTransform")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    ## Parse input arguments
    backends = args.backends[:]
    num_rows = args.num_rows
    max_len = args.max_len
    top_k = args.top_k
    is_cuda_graph_compatible = not args.no_cuda_graph
    run_refcheck = args.refcheck
    res = []

    backends = filter_backends_by_compute_capability(backends, args.routine, device)
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

    input_dtype = dtype_str_to_torch_dtype(args.input_dtype)

    ## Prepare input tensors
    # Input scores: (num_rows, max_len)
    input_scores = torch.randn(num_rows, max_len, dtype=input_dtype, device=device)

    # Offsets: (num_rows,)
    offsets = torch.arange(
        0, num_rows * max_len, max_len, dtype=torch.int32, device=device
    )

    # Lengths: (num_rows,)
    lengths = torch.full((num_rows,), max_len, dtype=torch.int32, device=device)

    if args.verbose >= 2:
        print(f"[VVERBOSE] {input_scores.shape = }")
        print(f"[VVERBOSE] {input_scores.dtype = }")
        print(f"[VVERBOSE] {offsets.shape = }")
        print(f"[VVERBOSE] {lengths.shape = }")
        print(f"[VVERBOSE] {top_k = }")

    def run_backend(backend, input_scores, offsets, lengths):
        if backend == "cuda":
            return flashinfer.top_k_ragged_transform(
                input_scores, offsets, lengths, top_k
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    # Storage for timing results and outputs for refcheck
    backend_times = {backend: [] for backend in backends}
    outputs = {}
    for cur_backend in backends:
        if run_refcheck:
            outputs[cur_backend] = run_backend(
                cur_backend, input_scores, offsets, lengths
            ).detach()
        backend_times[cur_backend] = bench_gpu_time(
            fn=run_backend,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            enable_cupti=args.use_cupti,
            use_cuda_graph=is_cuda_graph_compatible,
            input_args=(cur_backend, input_scores, offsets, lengths),
        )

    # Reference check: PyTorch implementation of top-k + ragged index transform
    # For each row i: output_indices[i, j] = topk_indices[i, j] + offsets[i]
    # Note: FlashInfer uses unsorted top-k internally, so we compare sorted sets per row
    if run_refcheck and outputs:
        # Get top-k indices
        _, topk_indices = torch.topk(input_scores.float(), k=top_k, dim=-1)
        # Add offsets to each row's indices
        reference_output = topk_indices.int() + offsets.unsqueeze(-1)

        for backend, output in outputs.items():
            # Sort both outputs per row to compare sets (order may differ due to unsorted top-k)
            ref_sorted, _ = torch.sort(reference_output, dim=-1)
            out_sorted, _ = torch.sort(output, dim=-1)

            matches = (ref_sorted == out_sorted).all()
            if not matches:
                num_diff = (ref_sorted != out_sorted).sum().item()
                num_total = reference_output.numel()
                pct_diff = num_diff / num_total * 100.0
                print(
                    f"[REFCHECK] Backend {backend}: {num_diff}/{num_total} "
                    f"({pct_diff:.2f}%) sorted ragged indices differ from PyTorch reference"
                )
                if not args.allow_output_mismatch:
                    raise AssertionError(
                        f"[ERROR] Backend {backend} output mismatch with {num_diff} elements"
                    )

    for backend in backends:
        if len(backend_times[backend]) > 0:
            median_time = np.median(backend_times[backend])
            std_time = np.std(backend_times[backend])

            # Memory bandwidth calculation
            # Read: input_scores
            # Write: output_indices
            problem_bytes = (
                num_rows * max_len * input_dtype.itemsize  # input_scores read
                + num_rows * top_k * 4  # output_indices write (int32)
            )
            tflops = 0  # Memory-bound operation
            tb_per_sec = problem_bytes / (10**9 * median_time)  # in TB/sec

            print_perf_metrics(backend, median_time, std_time, tflops, tb_per_sec)

            if args.output_path is not None:
                cur_res = defaultdict(str)
                cur_res["routine"] = args.routine
                cur_res["median_time"] = median_time
                cur_res["std_time"] = std_time
                cur_res["tflops"] = tflops
                cur_res["tb_per_sec"] = tb_per_sec
                cur_res["max_len"] = max_len
                cur_res["num_rows"] = num_rows
                cur_res["top_k"] = top_k
                cur_res["backend"] = backend
                cur_res["case_tag"] = args.case_tag
                res.append(cur_res)
    return res
