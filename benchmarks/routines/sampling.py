"""Copyright (c) 2025 by FlashInfer team.

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
    filter_backends_by_compute_capability,
    get_device,
    is_close_stats,
    print_perf_metrics,
)


def run_sampling_test(args):
    """Run a sampling test. We expose all sampling API in this benchmark.
    TopK is under sampling_topk.py- please see it for topk benchmark.

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results

    """
    if args.routine == "sampling_from_probs":
        return testSamplingFromProbs(args)
    if args.routine == "top_p_sampling_from_probs":
        return testTopPSamplingFromProbs(args)
    if args.routine == "top_k_sampling_from_probs":
        return testTopKSamplingFromProbs(args)
    if args.routine == "top_k_top_p_sampling_from_probs":
        return testTopKTopPSamplingFromProbs(args)
    if args.routine == "top_k_renorm_probs":
        return testTopKRenormProbs(args)
    if args.routine == "top_p_renorm_probs":
        return testTopPRenormProbs(args)
    if args.routine == "top_k_mask_logits":
        return testTopKMaskLogits(args)
    raise ValueError(f"Unsupported routine: {args.routine}")


def parse_sampling_args(line, parser):
    """Parse command line arguments for sampling test configuration.

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
        help="Batch size (number of sequences to sample from).",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        required=False,
        default=128256,
        help="Vocabulary size. Default: 128256 (Llama 3 vocab size).",
    )
    parser.add_argument(
        "--input_dtype",
        type=str,
        required=False,
        default="float32",
        help="Data type of the input tensor. Default: float32.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        required=False,
        default=0.9,
        help="Top-p threshold for nucleus sampling. Default: 0.9",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        required=False,
        default=50,
        help="Top-k threshold for top-k sampling. Default: 50",
    )
    parser.add_argument(
        "--no_deterministic",
        action="store_true",
        default=False,
        help="Disable deterministic sampling. Default: deterministic is enabled.",
    )
    parser.add_argument(
        "--backends",
        type=str,
        required=False,
        nargs="+",
        default=["cuda"],
        choices=["cuda"],
        help="Backend to test. Default: cuda.",
    )

    args = parser.parse_args(line)
    if args.verbose >= 1:
        print(f"[INFO] {args = }")
    return args


def testSamplingFromProbs(args):
    """Test sampling_from_probs API.

    This test:
    Sampling rng is not compatible with CUDA Graphs
    in the specific implementation in flashinfer,
    so we disable them for test.
    1. Generates random probability distributions and normalize them in FP32
    2. Runs sampling_from_probs
    3. Fetch performance numbers.

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
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}",
        )

    ## Parse input arguments
    backends = args.backends[:]
    batch_size = args.batch_size
    vocab_size = args.vocab_size
    deterministic = not args.no_deterministic
    res = []

    backends = filter_backends_by_compute_capability(backends, args.routine, device)
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

    input_dtype = dtype_str_to_torch_dtype(args.input_dtype)
    if input_dtype not in [torch.float32, torch.float16, torch.bfloat16]:
        raise ValueError(
            f"Unsupported input dtype: {args.input_dtype}. Supported dtypes are float32, float16, bfloat16.",
        )

    # Sampling uses RNG which is incompatible with CUDA graph capture
    is_cuda_graph_compatible = False

    ## Prepare input tensors
    input_shape = (batch_size, vocab_size)
    # Generate random probabilities and normalize in float32 for numerical stability
    pre_norm_probs = torch.rand(input_shape, dtype=torch.float32, device=device)
    probs = pre_norm_probs / pre_norm_probs.sum(dim=-1, keepdim=True)
    probs = probs.to(input_dtype)

    if args.verbose >= 2:
        print(f"[VVERBOSE] {probs.shape = }")
        print(f"[VVERBOSE] {probs.dtype = }")

    def run_backend(backend, probs):
        if backend == "cuda":
            return flashinfer.sampling.sampling_from_probs(
                probs,
                deterministic=deterministic,
            )
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

            # Memory bandwidth calculation for sampling
            # Read: probs tensor
            # Write: samples tensor (int32)
            num_elements = np.prod(input_shape)
            problem_bytes = (
                num_elements * input_dtype.itemsize  # probs read
                + batch_size * 4  # samples write (int32)
            )
            # Sampling is memory-bound
            problem_flops = num_elements  # rough estimate
            tflops = problem_flops / (10**9 * median_time)
            tb_per_sec = problem_bytes / (10**9 * median_time)

            print_perf_metrics(backend, median_time, std_time, tflops, tb_per_sec)

            if args.output_path is not None:
                cur_res = defaultdict(str)
                cur_res["routine"] = args.routine
                cur_res["median_time"] = median_time
                cur_res["std_time"] = str(std_time)
                cur_res["tflops"] = tflops
                cur_res["tb_per_sec"] = tb_per_sec
                cur_res["input_dtype"] = str(input_dtype)
                cur_res["vocab_size"] = vocab_size
                cur_res["deterministic"] = str(deterministic)
                cur_res["backend"] = backend
                cur_res["case_tag"] = args.case_tag
                res.append(cur_res)
    return res


def testTopPSamplingFromProbs(args):
    """Test top_p_sampling_from_probs API.

    This test:
    1. Generates random probability distributions and normalize them in FP32
    2. Runs top_p_sampling_from_probs (nucleus sampling)
    3. Fetch performance numbers.

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
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}",
        )

    ## Parse input arguments
    backends = args.backends[:]
    batch_size = args.batch_size
    vocab_size = args.vocab_size
    top_p = args.top_p
    deterministic = not args.no_deterministic
    res = []

    backends = filter_backends_by_compute_capability(backends, args.routine, device)
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

    input_dtype = dtype_str_to_torch_dtype(args.input_dtype)
    if input_dtype not in [torch.float32, torch.float16, torch.bfloat16]:
        raise ValueError(
            f"Unsupported input dtype: {args.input_dtype}. Supported dtypes are float32, float16, bfloat16.",
        )

    # Sampling uses RNG which is incompatible with CUDA graph capture
    is_cuda_graph_compatible = False

    ## Prepare input tensors
    input_shape = (batch_size, vocab_size)
    # Generate random probabilities and normalize in float32 for numerical stability
    pre_norm_probs = torch.rand(input_shape, dtype=torch.float32, device=device)
    probs = pre_norm_probs / pre_norm_probs.sum(dim=-1, keepdim=True)
    probs = probs.to(input_dtype)

    if args.verbose >= 2:
        print(f"[VVERBOSE] {probs.shape = }")
        print(f"[VVERBOSE] {probs.dtype = }")
        print(f"[VVERBOSE] {top_p = }")

    def run_backend(backend, probs):
        if backend == "cuda":
            return flashinfer.sampling.top_p_sampling_from_probs(
                probs,
                top_p=top_p,
                deterministic=deterministic,
            )
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

            num_elements = np.prod(input_shape)
            problem_bytes = (
                num_elements * input_dtype.itemsize  # probs read
                + batch_size * 4  # samples write (int32)
            )
            problem_flops = num_elements * 2  # sorting/filtering ops
            tflops = problem_flops / (10**9 * median_time)
            tb_per_sec = problem_bytes / (10**9 * median_time)

            print_perf_metrics(backend, median_time, std_time, tflops, tb_per_sec)

            if args.output_path is not None:
                cur_res = defaultdict(str)
                cur_res["routine"] = args.routine
                cur_res["median_time"] = median_time
                cur_res["std_time"] = str(std_time)
                cur_res["tflops"] = tflops
                cur_res["tb_per_sec"] = tb_per_sec
                cur_res["input_dtype"] = str(input_dtype)
                cur_res["vocab_size"] = vocab_size
                cur_res["top_p"] = top_p
                cur_res["deterministic"] = str(deterministic)
                cur_res["backend"] = backend
                cur_res["case_tag"] = args.case_tag
                res.append(cur_res)
    return res


def testTopKSamplingFromProbs(args):
    """Test top_k_sampling_from_probs API.

    This test:
    1. Generates random probability distributions
    2. Runs top_k_sampling_from_probs
    3. Measures performance metrics

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
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}",
        )

    ## Parse input arguments
    backends = args.backends[:]
    batch_size = args.batch_size
    vocab_size = args.vocab_size
    top_k = args.top_k
    deterministic = not args.no_deterministic
    res = []

    backends = filter_backends_by_compute_capability(backends, args.routine, device)
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

    input_dtype = dtype_str_to_torch_dtype(args.input_dtype)
    if input_dtype not in [torch.float32, torch.float16, torch.bfloat16]:
        raise ValueError(
            f"Unsupported input dtype: {args.input_dtype}. Supported dtypes are float32, float16, bfloat16.",
        )

    # Sampling uses RNG which is incompatible with CUDA graph capture
    is_cuda_graph_compatible = False

    ## Prepare input tensors
    input_shape = (batch_size, vocab_size)
    # Generate random probabilities and normalize in float32 for numerical stability
    pre_norm_probs = torch.rand(input_shape, dtype=torch.float32, device=device)
    probs = pre_norm_probs / pre_norm_probs.sum(dim=-1, keepdim=True)
    probs = probs.to(input_dtype)

    if args.verbose >= 2:
        print(f"[VVERBOSE] {probs.shape = }")
        print(f"[VVERBOSE] {probs.dtype = }")
        print(f"[VVERBOSE] {top_k = }")

    def run_backend(backend, probs):
        if backend == "cuda":
            return flashinfer.sampling.top_k_sampling_from_probs(
                probs,
                top_k=top_k,
                deterministic=deterministic,
            )
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

            num_elements = np.prod(input_shape)
            problem_bytes = (
                num_elements * input_dtype.itemsize  # probs read
                + batch_size * 4  # samples write (int32)
            )
            problem_flops = num_elements * 2  # sorting/filtering ops
            tflops = problem_flops / (10**9 * median_time)
            tb_per_sec = problem_bytes / (10**9 * median_time)

            print_perf_metrics(backend, median_time, std_time, tflops, tb_per_sec)

            if args.output_path is not None:
                cur_res = defaultdict(str)
                cur_res["routine"] = args.routine
                cur_res["median_time"] = median_time
                cur_res["std_time"] = str(std_time)
                cur_res["tflops"] = tflops
                cur_res["tb_per_sec"] = tb_per_sec
                cur_res["input_dtype"] = str(input_dtype)
                cur_res["vocab_size"] = vocab_size
                cur_res["top_k"] = top_k
                cur_res["deterministic"] = str(deterministic)
                cur_res["backend"] = backend
                cur_res["case_tag"] = args.case_tag
                res.append(cur_res)
    return res


def testTopKTopPSamplingFromProbs(args):
    """Test top_k_top_p_sampling_from_probs API.

    This test:
    1. Generates random probability distributions and normalize them in FP32
    2. Runs top_k_top_p_sampling_from_probs (combined top-k and top-p)
    3. Fetch performance numbers.

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
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}",
        )

    ## Parse input arguments
    backends = args.backends[:]
    batch_size = args.batch_size
    vocab_size = args.vocab_size
    top_k = args.top_k
    top_p = args.top_p
    deterministic = not args.no_deterministic
    res = []

    backends = filter_backends_by_compute_capability(backends, args.routine, device)
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

    input_dtype = dtype_str_to_torch_dtype(args.input_dtype)
    if input_dtype not in [torch.float32, torch.float16, torch.bfloat16]:
        raise ValueError(
            f"Unsupported input dtype: {args.input_dtype}. Supported dtypes are float32, float16, bfloat16.",
        )

    # Sampling uses RNG which is incompatible with CUDA graph capture
    is_cuda_graph_compatible = False

    ## Prepare input tensors
    input_shape = (batch_size, vocab_size)
    # Generate random probabilities and normalize in float32 for numerical stability
    pre_norm_probs = torch.rand(input_shape, dtype=torch.float32, device=device)
    probs = pre_norm_probs / pre_norm_probs.sum(dim=-1, keepdim=True)
    probs = probs.to(input_dtype)

    if args.verbose >= 2:
        print(f"[VVERBOSE] {probs.shape = }")
        print(f"[VVERBOSE] {probs.dtype = }")
        print(f"[VVERBOSE] {top_k = }")
        print(f"[VVERBOSE] {top_p = }")

    def run_backend(backend, probs):
        if backend == "cuda":
            return flashinfer.sampling.top_k_top_p_sampling_from_probs(
                probs,
                top_k=top_k,
                top_p=top_p,
                deterministic=deterministic,
            )
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

            num_elements = np.prod(input_shape)
            problem_bytes = (
                num_elements * input_dtype.itemsize  # probs read
                + batch_size * 4  # samples write (int32)
            )
            problem_flops = num_elements * 3  # more ops for combined filtering
            tflops = problem_flops / (10**9 * median_time)
            tb_per_sec = problem_bytes / (10**9 * median_time)

            print_perf_metrics(backend, median_time, std_time, tflops, tb_per_sec)

            if args.output_path is not None:
                cur_res = defaultdict(str)
                cur_res["routine"] = args.routine
                cur_res["median_time"] = median_time
                cur_res["std_time"] = str(std_time)
                cur_res["tflops"] = tflops
                cur_res["tb_per_sec"] = tb_per_sec
                cur_res["input_dtype"] = str(input_dtype)
                cur_res["vocab_size"] = vocab_size
                cur_res["top_k"] = top_k
                cur_res["top_p"] = top_p
                cur_res["deterministic"] = str(deterministic)
                cur_res["backend"] = backend
                cur_res["case_tag"] = args.case_tag
                res.append(cur_res)
    return res


def testTopKRenormProbs(args):
    """Test top_k_renorm_probs API.

    This test:
    1. Generates random probability distributions and normalize them in FP32
    2. Runs top_k_renorm_probs (renormalize by top-k thresholding)
    3. Fetch performance numbers.

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
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}",
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
    if input_dtype not in [torch.float32, torch.float16, torch.bfloat16]:
        raise ValueError(
            f"Unsupported input dtype: {args.input_dtype}. Supported dtypes are float32, float16, bfloat16.",
        )

    ## Prepare input tensors
    input_shape = (batch_size, vocab_size)
    # Generate random probabilities and normalize in float32 for numerical stability
    pre_norm_probs = torch.rand(input_shape, dtype=torch.float32, device=device)
    probs = pre_norm_probs / pre_norm_probs.sum(dim=-1, keepdim=True)
    probs = probs.to(input_dtype)

    if args.verbose >= 2:
        print(f"[VVERBOSE] {probs.shape = }")
        print(f"[VVERBOSE] {probs.dtype = }")
        print(f"[VVERBOSE] {top_k = }")

    def run_backend(backend, probs):
        if backend == "cuda":
            return flashinfer.sampling.top_k_renorm_probs(probs, top_k=top_k)
        raise ValueError(f"Unsupported backend: {backend}")

    # Reference implementation for refcheck
    has_reference_output = False
    if run_refcheck:
        # PyTorch reference: keep top-k, set rest to 0, renormalize
        topk_vals, topk_indices = torch.topk(probs.float(), k=top_k, dim=-1)
        reference_output = torch.zeros_like(probs)
        # NOTE: dont explicitly specify dtype here
        # keep it the same as input.
        reference_output.scatter_(-1, topk_indices, topk_vals)
        reference_output = reference_output / reference_output.sum(dim=-1, keepdim=True)
        reference_output = reference_output.to(input_dtype)
        has_reference_output = True

    # Storage for timing results and outputs
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
                    atol=1e-2,
                )
                if num_different_elements > 0:
                    print(
                        f"[ERROR] Output tensor mismatch from backend {tested_backends[i]}: "
                        f"{num_different_elements}/{num_elements} ({num_different_elements_percentage:.2f}%) elements differ",
                    )
                    if not args.allow_output_mismatch:
                        raise AssertionError(
                            f"[ERROR] Backend {tested_backends[i]} output mismatch with {num_different_elements} elements",
                        )

    for backend in backends:
        if len(backend_times[backend]) > 0:
            median_time = np.median(backend_times[backend])
            std_time = np.std(backend_times[backend])

            num_elements = np.prod(input_shape)
            problem_bytes = (
                num_elements * input_dtype.itemsize  # probs read
                + num_elements * input_dtype.itemsize  # renorm_probs write
            )
            problem_flops = num_elements * 2
            tflops = problem_flops / (10**9 * median_time)
            tb_per_sec = problem_bytes / (10**9 * median_time)

            print_perf_metrics(backend, median_time, std_time, tflops, tb_per_sec)

            if args.output_path is not None:
                cur_res = defaultdict(str)
                cur_res["routine"] = args.routine
                cur_res["median_time"] = median_time
                cur_res["std_time"] = str(std_time)
                cur_res["tflops"] = tflops
                cur_res["tb_per_sec"] = tb_per_sec
                cur_res["input_dtype"] = str(input_dtype)
                cur_res["vocab_size"] = vocab_size
                cur_res["top_k"] = top_k
                cur_res["backend"] = backend
                cur_res["case_tag"] = args.case_tag
                res.append(cur_res)
    return res


def testTopPRenormProbs(args):
    """Test top_p_renorm_probs API.

    This test:
    1. Generates random probability distributions
    2. Runs top_p_renorm_probs (renormalize by top-p thresholding)
    3. Measures performance metrics

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
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}",
        )

    ## Parse input arguments
    backends = args.backends[:]
    batch_size = args.batch_size
    vocab_size = args.vocab_size
    top_p = args.top_p
    is_cuda_graph_compatible = not args.no_cuda_graph
    res = []

    backends = filter_backends_by_compute_capability(backends, args.routine, device)
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

    input_dtype = dtype_str_to_torch_dtype(args.input_dtype)
    if input_dtype not in [torch.float32, torch.float16, torch.bfloat16]:
        raise ValueError(
            f"Unsupported input dtype: {args.input_dtype}. Supported dtypes are float32, float16, bfloat16.",
        )

    ## Prepare input tensors
    input_shape = (batch_size, vocab_size)
    # Generate random probabilities and normalize in float32 for numerical stability
    pre_norm_probs = torch.rand(input_shape, dtype=torch.float32, device=device)
    probs = pre_norm_probs / pre_norm_probs.sum(dim=-1, keepdim=True)
    probs = probs.to(input_dtype)

    if args.verbose >= 2:
        print(f"[VVERBOSE] {probs.shape = }")
        print(f"[VVERBOSE] {probs.dtype = }")
        print(f"[VVERBOSE] {top_p = }")

    def run_backend(backend, probs):
        if backend == "cuda":
            return flashinfer.sampling.top_p_renorm_probs(probs, top_p=top_p)
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

            num_elements = np.prod(input_shape)
            problem_bytes = (
                num_elements * input_dtype.itemsize  # probs read
                + num_elements
                * input_dtype.itemsize  # renorm_probs write (same dtype as input)
            )
            problem_flops = num_elements * 2
            tflops = problem_flops / (10**9 * median_time)
            tb_per_sec = problem_bytes / (10**9 * median_time)

            print_perf_metrics(backend, median_time, std_time, tflops, tb_per_sec)

            if args.output_path is not None:
                cur_res = defaultdict(str)
                cur_res["routine"] = args.routine
                cur_res["median_time"] = median_time
                cur_res["std_time"] = str(std_time)
                cur_res["tflops"] = tflops
                cur_res["tb_per_sec"] = tb_per_sec
                cur_res["input_dtype"] = str(input_dtype)
                cur_res["vocab_size"] = vocab_size
                cur_res["top_p"] = top_p
                cur_res["backend"] = backend
                cur_res["case_tag"] = args.case_tag
                res.append(cur_res)
    return res


def testTopKMaskLogits(args):
    """Test top_k_mask_logits API.

    This test:
    1. Generates random logits
    2. Runs top_k_mask_logits (mask logits by top-k thresholding)
    3. Measures performance metrics

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
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}",
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
    if input_dtype not in [torch.float32, torch.float16, torch.bfloat16]:
        raise ValueError(
            f"Unsupported input dtype: {args.input_dtype}. Supported dtypes are float32, float16, bfloat16.",
        )

    ## Prepare input tensors
    input_shape = (batch_size, vocab_size)
    logits = torch.randn(input_shape, dtype=input_dtype, device=device)

    if args.verbose >= 2:
        print(f"[VVERBOSE] {logits.shape = }")
        print(f"[VVERBOSE] {logits.dtype = }")
        print(f"[VVERBOSE] {top_k = }")

    def run_backend(backend, logits):
        if backend == "cuda":
            return flashinfer.sampling.top_k_mask_logits(logits, top_k=top_k)
        raise ValueError(f"Unsupported backend: {backend}")

    # Reference implementation for refcheck
    has_reference_output = False
    if run_refcheck:
        # PyTorch reference: keep top-k logits, set rest to -inf
        topk_vals, topk_indices = torch.topk(logits.float(), k=top_k, dim=-1)
        reference_output = torch.full_like(logits, float("-inf"))
        # NOTE: dont explicitly specify dtype here
        # keep it the same as input.
        reference_output.scatter_(-1, topk_indices, topk_vals)
        reference_output = reference_output.to(input_dtype)
        has_reference_output = True

    # Storage for timing results and outputs
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

    tested_backends = list(outputs.keys())
    tested_outputs = list(outputs.values())
    if len(tested_backends) > 0:
        if run_refcheck and has_reference_output:
            for i in range(len(tested_backends)):
                # For masked logits, check:
                # 1. Same positions are masked (-inf)
                # 2. Unmasked values match the original logits
                out = tested_outputs[i].float()
                ref = reference_output.float()

                # Check that the same positions are masked
                out_masked = torch.isinf(out) & (out < 0)
                ref_masked = torch.isinf(ref) & (ref < 0)
                mask_match = (out_masked == ref_masked).all()
                if not mask_match:
                    print(f"[ERROR] Mask mismatch from backend {tested_backends[i]}")
                    if not args.allow_output_mismatch:
                        raise AssertionError(
                            f"[ERROR] Backend {tested_backends[i]} mask mismatch",
                        )

                # Check that unmasked values match the reference (original top-k logits)
                unmasked_positions = ~out_masked
                if unmasked_positions.any():
                    out_unmasked = out[unmasked_positions]
                    ref_unmasked = ref[unmasked_positions]
                    (
                        num_different_elements,
                        num_elements,
                        num_different_elements_percentage,
                    ) = is_close_stats(ref_unmasked, out_unmasked, rtol=1e-3, atol=1e-3)
                    if num_different_elements > 0:
                        print(
                            f"[ERROR] Unmasked values mismatch from backend {tested_backends[i]}: "
                            f"{num_different_elements}/{num_elements} ({num_different_elements_percentage:.2f}%) elements differ",
                        )
                        if not args.allow_output_mismatch:
                            raise AssertionError(
                                f"[ERROR] Backend {tested_backends[i]} unmasked values mismatch",
                            )

    for backend in backends:
        if len(backend_times[backend]) > 0:
            median_time = np.median(backend_times[backend])
            std_time = np.std(backend_times[backend])

            num_elements = np.prod(input_shape)
            problem_bytes = (
                num_elements * input_dtype.itemsize  # logits read
                + num_elements * input_dtype.itemsize  # masked_logits write
            )
            problem_flops = num_elements * 2
            tflops = problem_flops / (10**9 * median_time)
            tb_per_sec = problem_bytes / (10**9 * median_time)

            print_perf_metrics(backend, median_time, std_time, tflops, tb_per_sec)

            if args.output_path is not None:
                cur_res = defaultdict(str)
                cur_res["routine"] = args.routine
                cur_res["median_time"] = median_time
                cur_res["std_time"] = str(std_time)
                cur_res["tflops"] = tflops
                cur_res["tb_per_sec"] = tb_per_sec
                cur_res["input_dtype"] = str(input_dtype)
                cur_res["vocab_size"] = vocab_size
                cur_res["top_k"] = top_k
                cur_res["backend"] = backend
                cur_res["case_tag"] = args.case_tag
                res.append(cur_res)
    return res
