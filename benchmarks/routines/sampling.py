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


# ============================================================
# Shared helpers to reduce boilerplate across sampling benchmarks
# ============================================================


def _setup_sampling_benchmark(args, routine_name):
    """Common setup: logging, device, backend filtering, dtype validation.

    Returns:
        tuple: (device, backends, input_dtype). backends is empty list if none
               are available for the current compute capability.

    """
    if args.verbose >= 1:
        print(f"[INFO] Running {routine_name}")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}",
        )

    backends = args.backends[:]
    backends = filter_backends_by_compute_capability(backends, args.routine, device)
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return device, [], None

    input_dtype = dtype_str_to_torch_dtype(args.input_dtype)
    if input_dtype not in [torch.float32, torch.float16, torch.bfloat16]:
        raise ValueError(
            f"Unsupported input dtype: {args.input_dtype}. "
            f"Supported dtypes are float32, float16, bfloat16.",
        )

    return device, backends, input_dtype


def _create_normalized_probs(batch_size, vocab_size, input_dtype, device):
    """Generate random probability distributions, normalized in FP32 for stability.

    Returns:
        tuple: (probs tensor, input_shape tuple)

    """
    input_shape = (batch_size, vocab_size)
    pre_norm_probs = torch.rand(input_shape, dtype=torch.float32, device=device)
    probs = pre_norm_probs / pre_norm_probs.sum(dim=-1, keepdim=True)
    return probs.to(input_dtype), input_shape


def _bench_sampling(
    args,
    backends,
    run_backend,
    input_tensor,
    is_cuda_graph_compatible,
    run_refcheck=False,
):
    """Run timing across backends, optionally collecting outputs for refcheck.

    Returns:
        tuple: (backend_times dict, outputs dict). outputs is empty if
               run_refcheck is False.

    """
    backend_times = {}
    outputs = {}
    for cur_backend in backends:
        if run_refcheck:
            outputs[cur_backend] = run_backend(cur_backend, input_tensor).detach()
        backend_times[cur_backend] = bench_gpu_time(
            fn=run_backend,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            enable_cupti=args.use_cupti,
            use_cuda_graph=is_cuda_graph_compatible,
            input_args=(cur_backend, input_tensor),
        )
    return backend_times, outputs


def _collect_results(
    args,
    backends,
    backend_times,
    input_dtype,
    problem_bytes,
    problem_flops,
    extra_result_fields,
):
    """Calculate perf metrics and build result dicts.

    Returns:
        list: Result dicts for CSV output.

    """
    res = []
    for backend in backends:
        if len(backend_times[backend]) > 0:
            median_time = np.median(backend_times[backend])
            std_time = np.std(backend_times[backend])
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
                cur_res["backend"] = backend
                cur_res["case_tag"] = args.case_tag
                cur_res.update(extra_result_fields)
                res.append(cur_res)
    return res


def _check_is_close(outputs, reference_output, args, rtol=1e-2, atol=1e-2):
    """Compare backend outputs to reference using is_close_stats."""
    for backend, output in outputs.items():
        (
            num_different_elements,
            num_elements,
            num_different_elements_percentage,
        ) = is_close_stats(
            reference_output.float(),
            output.float(),
            rtol=rtol,
            atol=atol,
        )
        if num_different_elements > 0:
            print(
                f"[ERROR] Output tensor mismatch from backend {backend}: "
                f"{num_different_elements}/{num_elements} "
                f"({num_different_elements_percentage:.2f}%) elements differ",
            )
            if not args.allow_output_mismatch:
                raise AssertionError(
                    f"[ERROR] Backend {backend} output mismatch "
                    f"with {num_different_elements} elements",
                )


# ============================================================
# Public API
# ============================================================


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


# ============================================================
# Individual benchmark functions
# ============================================================


def testSamplingFromProbs(args):
    """Test sampling_from_probs API.

    Sampling RNG is not compatible with CUDA Graphs in the specific
    implementation in flashinfer, so CUDA graph capture is disabled.

    """
    device, backends, input_dtype = _setup_sampling_benchmark(
        args,
        "testSamplingFromProbs",
    )
    if not backends:
        return []

    batch_size = args.batch_size
    deterministic = not args.no_deterministic
    probs, input_shape = _create_normalized_probs(
        batch_size,
        args.vocab_size,
        input_dtype,
        device,
    )

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

    backend_times, _ = _bench_sampling(
        args,
        backends,
        run_backend,
        probs,
        is_cuda_graph_compatible=False,
    )

    num_elements = np.prod(input_shape)
    return _collect_results(
        args,
        backends,
        backend_times,
        input_dtype,
        problem_bytes=num_elements * input_dtype.itemsize + batch_size * 4,
        problem_flops=num_elements,
        extra_result_fields={
            "vocab_size": args.vocab_size,
            "deterministic": str(deterministic),
        },
    )


def testTopPSamplingFromProbs(args):
    """Test top_p_sampling_from_probs API (nucleus sampling)."""
    device, backends, input_dtype = _setup_sampling_benchmark(
        args,
        "testTopPSamplingFromProbs",
    )
    if not backends:
        return []

    batch_size = args.batch_size
    top_p = args.top_p
    deterministic = not args.no_deterministic
    probs, input_shape = _create_normalized_probs(
        batch_size,
        args.vocab_size,
        input_dtype,
        device,
    )

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

    backend_times, _ = _bench_sampling(
        args,
        backends,
        run_backend,
        probs,
        is_cuda_graph_compatible=False,
    )

    num_elements = np.prod(input_shape)
    return _collect_results(
        args,
        backends,
        backend_times,
        input_dtype,
        problem_bytes=num_elements * input_dtype.itemsize + batch_size * 4,
        problem_flops=num_elements * 2,
        extra_result_fields={
            "vocab_size": args.vocab_size,
            "top_p": top_p,
            "deterministic": str(deterministic),
        },
    )


def testTopKSamplingFromProbs(args):
    """Test top_k_sampling_from_probs API."""
    device, backends, input_dtype = _setup_sampling_benchmark(
        args,
        "testTopKSamplingFromProbs",
    )
    if not backends:
        return []

    batch_size = args.batch_size
    top_k = args.top_k
    deterministic = not args.no_deterministic
    probs, input_shape = _create_normalized_probs(
        batch_size,
        args.vocab_size,
        input_dtype,
        device,
    )

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

    backend_times, _ = _bench_sampling(
        args,
        backends,
        run_backend,
        probs,
        is_cuda_graph_compatible=False,
    )

    num_elements = np.prod(input_shape)
    return _collect_results(
        args,
        backends,
        backend_times,
        input_dtype,
        problem_bytes=num_elements * input_dtype.itemsize + batch_size * 4,
        problem_flops=num_elements * 2,
        extra_result_fields={
            "vocab_size": args.vocab_size,
            "top_k": top_k,
            "deterministic": str(deterministic),
        },
    )


def testTopKTopPSamplingFromProbs(args):
    """Test top_k_top_p_sampling_from_probs API (combined top-k and top-p)."""
    device, backends, input_dtype = _setup_sampling_benchmark(
        args,
        "testTopKTopPSamplingFromProbs",
    )
    if not backends:
        return []

    batch_size = args.batch_size
    top_k = args.top_k
    top_p = args.top_p
    deterministic = not args.no_deterministic
    probs, input_shape = _create_normalized_probs(
        batch_size,
        args.vocab_size,
        input_dtype,
        device,
    )

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

    backend_times, _ = _bench_sampling(
        args,
        backends,
        run_backend,
        probs,
        is_cuda_graph_compatible=False,
    )

    num_elements = np.prod(input_shape)
    return _collect_results(
        args,
        backends,
        backend_times,
        input_dtype,
        problem_bytes=num_elements * input_dtype.itemsize + batch_size * 4,
        problem_flops=num_elements * 3,
        extra_result_fields={
            "vocab_size": args.vocab_size,
            "top_k": top_k,
            "top_p": top_p,
            "deterministic": str(deterministic),
        },
    )


def testTopKRenormProbs(args):
    """Test top_k_renorm_probs API (renormalize by top-k thresholding)."""
    device, backends, input_dtype = _setup_sampling_benchmark(
        args,
        "testTopKRenormProbs",
    )
    if not backends:
        return []

    top_k = args.top_k
    is_cuda_graph_compatible = not args.no_cuda_graph
    run_refcheck = args.refcheck
    probs, input_shape = _create_normalized_probs(
        args.batch_size,
        args.vocab_size,
        input_dtype,
        device,
    )

    if args.verbose >= 2:
        print(f"[VVERBOSE] {probs.shape = }")
        print(f"[VVERBOSE] {probs.dtype = }")
        print(f"[VVERBOSE] {top_k = }")

    def run_backend(backend, probs):
        if backend == "cuda":
            return flashinfer.sampling.top_k_renorm_probs(probs, top_k=top_k)
        raise ValueError(f"Unsupported backend: {backend}")

    backend_times, outputs = _bench_sampling(
        args,
        backends,
        run_backend,
        probs,
        is_cuda_graph_compatible,
        run_refcheck=run_refcheck,
    )

    if run_refcheck and outputs:
        # PyTorch reference: keep top-k, set rest to 0, renormalize
        topk_vals, topk_indices = torch.topk(probs.float(), k=top_k, dim=-1)
        reference_output = torch.zeros_like(probs)
        reference_output.scatter_(-1, topk_indices, topk_vals)
        reference_output = reference_output / reference_output.sum(
            dim=-1,
            keepdim=True,
        )
        reference_output = reference_output.to(input_dtype)
        _check_is_close(outputs, reference_output, args)

    num_elements = np.prod(input_shape)
    return _collect_results(
        args,
        backends,
        backend_times,
        input_dtype,
        problem_bytes=num_elements * input_dtype.itemsize * 2,
        problem_flops=num_elements * 2,
        extra_result_fields={
            "vocab_size": args.vocab_size,
            "top_k": top_k,
        },
    )


def testTopPRenormProbs(args):
    """Test top_p_renorm_probs API (renormalize by top-p thresholding)."""
    device, backends, input_dtype = _setup_sampling_benchmark(
        args,
        "testTopPRenormProbs",
    )
    if not backends:
        return []

    top_p = args.top_p
    is_cuda_graph_compatible = not args.no_cuda_graph
    probs, input_shape = _create_normalized_probs(
        args.batch_size,
        args.vocab_size,
        input_dtype,
        device,
    )

    if args.verbose >= 2:
        print(f"[VVERBOSE] {probs.shape = }")
        print(f"[VVERBOSE] {probs.dtype = }")
        print(f"[VVERBOSE] {top_p = }")

    def run_backend(backend, probs):
        if backend == "cuda":
            return flashinfer.sampling.top_p_renorm_probs(probs, top_p=top_p)
        raise ValueError(f"Unsupported backend: {backend}")

    backend_times, _ = _bench_sampling(
        args,
        backends,
        run_backend,
        probs,
        is_cuda_graph_compatible,
    )

    num_elements = np.prod(input_shape)
    return _collect_results(
        args,
        backends,
        backend_times,
        input_dtype,
        problem_bytes=num_elements * input_dtype.itemsize * 2,
        problem_flops=num_elements * 2,
        extra_result_fields={
            "vocab_size": args.vocab_size,
            "top_p": top_p,
        },
    )


def testTopKMaskLogits(args):
    """Test top_k_mask_logits API (mask logits by top-k thresholding)."""
    device, backends, input_dtype = _setup_sampling_benchmark(
        args,
        "testTopKMaskLogits",
    )
    if not backends:
        return []

    top_k = args.top_k
    is_cuda_graph_compatible = not args.no_cuda_graph
    run_refcheck = args.refcheck

    # Unlike other sampling benchmarks, this operates on raw logits, not probs
    input_shape = (args.batch_size, args.vocab_size)
    logits = torch.randn(input_shape, dtype=input_dtype, device=device)

    if args.verbose >= 2:
        print(f"[VVERBOSE] {logits.shape = }")
        print(f"[VVERBOSE] {logits.dtype = }")
        print(f"[VVERBOSE] {top_k = }")

    def run_backend(backend, logits):
        if backend == "cuda":
            return flashinfer.sampling.top_k_mask_logits(logits, top_k=top_k)
        raise ValueError(f"Unsupported backend: {backend}")

    backend_times, outputs = _bench_sampling(
        args,
        backends,
        run_backend,
        logits,
        is_cuda_graph_compatible,
        run_refcheck=run_refcheck,
    )

    if run_refcheck and outputs:
        # PyTorch reference: keep top-k logits, set rest to -inf
        topk_vals, topk_indices = torch.topk(logits.float(), k=top_k, dim=-1)
        reference_output = torch.full_like(logits, float("-inf"))
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
                print(f"[ERROR] Mask mismatch from backend {backend}")
                if not args.allow_output_mismatch:
                    raise AssertionError(
                        f"[ERROR] Backend {backend} mask mismatch",
                    )

            # Check that unmasked values match the reference
            unmasked_positions = ~out_masked
            if unmasked_positions.any():
                (
                    num_different_elements,
                    num_elements,
                    num_different_elements_percentage,
                ) = is_close_stats(
                    ref[unmasked_positions],
                    out[unmasked_positions],
                    rtol=1e-3,
                    atol=1e-3,
                )
                if num_different_elements > 0:
                    print(
                        f"[ERROR] Unmasked values mismatch from backend {backend}: "
                        f"{num_different_elements}/{num_elements} "
                        f"({num_different_elements_percentage:.2f}%) elements differ",
                    )
                    if not args.allow_output_mismatch:
                        raise AssertionError(
                            f"[ERROR] Backend {backend} unmasked values mismatch",
                        )

    num_elements = np.prod(input_shape)
    return _collect_results(
        args,
        backends,
        backend_times,
        input_dtype,
        problem_bytes=num_elements * input_dtype.itemsize * 2,
        problem_flops=num_elements * 2,
        extra_result_fields={
            "vocab_size": args.vocab_size,
            "top_k": top_k,
        },
    )
