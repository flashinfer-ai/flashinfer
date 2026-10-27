"""
Copyright (c) 2025 by FlashInfer team.

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
    blank_unused_mode_columns,
    dtype_str_to_torch_dtype,
    get_device,
    print_perf_metrics,
    filter_backends_by_compute_capability,
    fp8_code_distance,
    warn_if_pdl_unsupported,
)


def _rope_fn(name, inplace):
    """Return flashinfer.rope.<name> or its <name>_inplace twin (same signature)."""
    return getattr(flashinfer.rope, f"{name}_inplace" if inplace else name)


def run_rope_test(args):
    """Run one rope testlist line; see _dispatch_rope_test for the routine table."""
    res = _dispatch_rope_test(args)
    # Mode flags added in 2026-10 stay blank on rows that do not use them, so
    # pre-existing lines keep their CSV columns (and dashboard keys).
    return blank_unused_mode_columns(
        args,
        res,
        defaults={"inplace": {"False"}, "mla_kv": {"False"}},
    )


def _dispatch_rope_test(args):
    """
    Run a RoPE test.

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    if args.routine == "apply_rope":
        return testApplyRope(args)
    elif args.routine == "apply_rope_pos_ids":
        return testApplyRopePosIds(args)
    elif args.routine == "apply_llama31_rope":
        return testApplyLlama31Rope(args)
    elif args.routine == "apply_llama31_rope_pos_ids":
        return testApplyLlama31RopePosIds(args)
    elif args.routine == "apply_rope_with_cos_sin_cache":
        return testApplyRopeWithCosSinCache(args)
    elif args.routine == "mla_rope_quantize_fp8":
        return testMlaRopeQuantizeFp8(args)
    elif args.routine == "rope_quantize_fp8":
        return testRopeQuantizeFp8(args)
    elif args.routine == "rope_quantize_fp8_append_paged_kv_cache":
        return testRopeQuantizeFp8AppendPagedKvCache(args)
    else:
        raise ValueError(f"Unsupported routine: {args.routine}")


def parse_rope_args(line, parser):
    """
    Parse command line arguments for RoPE test configuration.

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
        help="Batch size.",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        required=True,
        help="Sequence length (qkv_len or kv_len).",
    )
    parser.add_argument(
        "--num_qo_heads",
        type=int,
        required=True,
        help="Number of query/output heads.",
    )
    parser.add_argument(
        "--num_kv_heads",
        type=int,
        required=True,
        help="Number of key/value heads.",
    )
    parser.add_argument(
        "--head_dim",
        type=int,
        required=True,
        help="Head dimension.",
    )
    parser.add_argument(
        "--rotary_dim",
        type=int,
        required=False,
        default=None,
        help="Rotary dimension (defaults to head_dim if not specified).",
    )
    parser.add_argument(
        "--no_rope_dim",
        type=int,
        required=False,
        default=0,
        help="Number of dimensions without RoPE (for MLA). Default: 0.",
    )
    parser.add_argument(
        "--input_dtype",
        type=str,
        required=False,
        default="float16",
        choices=["float16", "bfloat16"],
        help="Data type of the input tensor.",
    )
    parser.add_argument(
        "--quant_dtype",
        type=str,
        required=False,
        default="fp8_e4m3",
        choices=["fp8_e4m3", "fp8_e5m2"],
        help="Quantized data type for FP8 routines.",
    )
    parser.add_argument(
        "--rope_scale",
        type=float,
        required=False,
        default=1.0,
        help="RoPE scaling factor.",
    )
    parser.add_argument(
        "--rope_theta",
        type=float,
        required=False,
        default=10000.0,
        help="RoPE theta base frequency.",
    )
    parser.add_argument(
        "--interleave",
        action="store_true",
        help="Use interleaved rotary embedding (GPT-J style).",
    )
    parser.add_argument(
        "--page_size",
        type=int,
        required=False,
        default=16,
        help="Page size for paged KV cache.",
    )
    parser.add_argument(
        "--kv_layout",
        type=str,
        required=False,
        default="NHD",
        choices=["NHD", "HND"],
        help="KV cache layout.",
    )
    parser.add_argument(
        "--low_freq_factor",
        type=float,
        required=False,
        default=1.0,
        help="Low frequency factor for Llama 3.1 RoPE.",
    )
    parser.add_argument(
        "--high_freq_factor",
        type=float,
        required=False,
        default=4.0,
        help="High frequency factor for Llama 3.1 RoPE.",
    )
    parser.add_argument(
        "--old_context_len",
        type=int,
        required=False,
        default=8192,
        help="Old context length for Llama 3.1 RoPE.",
    )
    parser.add_argument(
        "--backends",
        type=str,
        required=False,
        nargs="+",
        default=["cuda"],
        choices=["cuda", "cutile"],
        help="Kernel backends to test. Default: cuda. cutile: rope_quantize_fp8 "
        "with --mla_kv --interleave only.",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        default=False,
        help="Call the *_inplace variant (apply_rope*, apply_llama31_rope*, "
        "apply_rope_with_cos_sin_cache), which overwrites Q/K.",
    )
    parser.add_argument(
        "--mla_kv",
        action="store_true",
        default=False,
        help="MLA layout for rope_quantize_fp8 and "
        "rope_quantize_fp8_append_paged_kv_cache: one shared 2D K latent "
        "(--num_kv_heads ignored), (ckv, kpe) caches for the append routine.",
    )

    args = parser.parse_args(line)

    # Default rotary_dim to head_dim if not specified
    if args.rotary_dim is None:
        args.rotary_dim = args.head_dim

    # Flags that a routine would otherwise silently ignore
    inplace_routines = (
        "apply_rope",
        "apply_rope_pos_ids",
        "apply_llama31_rope",
        "apply_llama31_rope_pos_ids",
        "apply_rope_with_cos_sin_cache",
    )
    mla_kv_routines = ("rope_quantize_fp8", "rope_quantize_fp8_append_paged_kv_cache")
    if args.inplace and args.routine not in inplace_routines:
        parser.error(f"--inplace has no effect on {args.routine} (no _inplace variant)")
    if args.mla_kv and args.routine not in mla_kv_routines:
        parser.error(
            f"--mla_kv applies to {', '.join(mla_kv_routines)} only "
            "(mla_rope_quantize_fp8 is always MLA)"
        )
    if args.mla_kv and args.kv_layout != "NHD":
        parser.error("--mla_kv uses 3D (ckv, kpe) caches; --kv_layout does not apply")

    if args.verbose >= 1:
        print(f"[INFO] {args = }")
    return args


def testApplyRope(args):
    """
    Test apply_rope API (with indptr/offsets).

    This test:
    1. Generates random Q and K tensors
    2. Runs flashinfer.rope.apply_rope with indptr/offsets
    3. Measures performance metrics (TB/sec)

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    warn_if_pdl_unsupported(args, args.routine)
    if args.verbose >= 1:
        print("[INFO] Running testApplyRope")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    ## Parse input arguments
    backends = args.backends[:]
    batch_size = args.batch_size
    seq_len = args.seq_len
    num_qo_heads = args.num_qo_heads
    num_kv_heads = args.num_kv_heads
    head_dim = args.head_dim
    rotary_dim = args.rotary_dim
    rope_scale = args.rope_scale
    rope_theta = args.rope_theta
    interleave = args.interleave
    is_cuda_graph_compatible = not args.no_cuda_graph
    res = []

    backends = filter_backends_by_compute_capability(backends, args.routine, device)
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

    input_dtype = dtype_str_to_torch_dtype(args.input_dtype)

    ## Prepare input tensors
    # Shape: (batch_size * seq_len, num_heads, head_dim)
    q = torch.randn(
        batch_size * seq_len, num_qo_heads, head_dim, dtype=input_dtype, device=device
    )
    k = torch.randn(
        batch_size * seq_len, num_kv_heads, head_dim, dtype=input_dtype, device=device
    )

    # indptr for ragged tensor
    indptr = torch.arange(
        0, (batch_size + 1) * seq_len, seq_len, dtype=torch.int32, device=device
    )

    # offsets (per-request position offset)
    offsets = torch.zeros(batch_size, dtype=torch.int32, device=device)

    if args.verbose >= 2:
        print(f"[VVERBOSE] {q.shape = }")
        print(f"[VVERBOSE] {k.shape = }")
        print(f"[VVERBOSE] {indptr.shape = }")
        print(f"[VVERBOSE] {offsets.shape = }")
        print(f"[VVERBOSE] {rotary_dim = }")
        print(f"[VVERBOSE] {rope_scale = }")
        print(f"[VVERBOSE] {rope_theta = }")
        print(f"[VVERBOSE] {interleave = }")

    def run_backend(backend, q, k, indptr, offsets):
        if backend == "cuda":
            return _rope_fn("apply_rope", args.inplace)(
                q,
                k,
                indptr=indptr,
                offsets=offsets,
                rotary_dim=rotary_dim,
                interleave=interleave,
                rope_scale=rope_scale,
                rope_theta=rope_theta,
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
            input_args=(cur_backend, q, k, indptr, offsets),
        )

    for backend in backends:
        if len(backend_times[backend]) > 0:
            median_time = np.median(backend_times[backend])
            std_time = np.std(backend_times[backend])

            total_tokens = batch_size * seq_len
            # Memory bandwidth calculation
            # Read: q + k
            # Write: q_rope + k_rope
            problem_bytes = (
                total_tokens * num_qo_heads * head_dim * input_dtype.itemsize  # q read
                + total_tokens
                * num_kv_heads
                * head_dim
                * input_dtype.itemsize  # k read
                + total_tokens
                * num_qo_heads
                * head_dim
                * input_dtype.itemsize  # q_rope write
                + total_tokens
                * num_kv_heads
                * head_dim
                * input_dtype.itemsize  # k_rope write
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
                cur_res["seq_len"] = seq_len
                cur_res["num_qo_heads"] = num_qo_heads
                cur_res["num_kv_heads"] = num_kv_heads
                cur_res["head_dim"] = head_dim
                cur_res["rotary_dim"] = rotary_dim
                cur_res["rope_theta"] = rope_theta
                cur_res["rope_scale"] = rope_scale
                cur_res["interleave"] = interleave
                cur_res["backend"] = backend
                cur_res["case_tag"] = args.case_tag
                res.append(cur_res)
    return res


def testApplyRopePosIds(args):
    """
    Test apply_rope API with pos_ids.

    This test:
    1. Generates random Q and K tensors
    2. Runs flashinfer.rope.apply_rope with pos_ids
    3. Measures performance metrics (TB/sec)

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    warn_if_pdl_unsupported(args, args.routine)
    if args.verbose >= 1:
        print("[INFO] Running testApplyRopePosIds")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    ## Parse input arguments
    backends = args.backends[:]
    batch_size = args.batch_size
    seq_len = args.seq_len
    num_qo_heads = args.num_qo_heads
    num_kv_heads = args.num_kv_heads
    head_dim = args.head_dim
    rotary_dim = args.rotary_dim
    rope_scale = args.rope_scale
    rope_theta = args.rope_theta
    interleave = args.interleave
    is_cuda_graph_compatible = not args.no_cuda_graph
    res = []

    backends = filter_backends_by_compute_capability(backends, args.routine, device)
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

    input_dtype = dtype_str_to_torch_dtype(args.input_dtype)

    ## Prepare input tensors
    # Shape: (batch_size * seq_len, num_heads, head_dim)
    q = torch.randn(
        batch_size * seq_len, num_qo_heads, head_dim, dtype=input_dtype, device=device
    )
    k = torch.randn(
        batch_size * seq_len, num_kv_heads, head_dim, dtype=input_dtype, device=device
    )

    # pos_ids: (batch_size * seq_len,)
    pos_ids = torch.arange(seq_len, dtype=torch.int32, device=device).repeat(batch_size)

    if args.verbose >= 2:
        print(f"[VVERBOSE] {q.shape = }")
        print(f"[VVERBOSE] {k.shape = }")
        print(f"[VVERBOSE] {pos_ids.shape = }")
        print(f"[VVERBOSE] {rotary_dim = }")
        print(f"[VVERBOSE] {rope_scale = }")
        print(f"[VVERBOSE] {rope_theta = }")
        print(f"[VVERBOSE] {interleave = }")

    def run_backend(backend, q, k, pos_ids):
        if backend == "cuda":
            return _rope_fn("apply_rope_pos_ids", args.inplace)(
                q,
                k,
                pos_ids,
                rotary_dim=rotary_dim,
                interleave=interleave,
                rope_scale=rope_scale,
                rope_theta=rope_theta,
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
            input_args=(cur_backend, q, k, pos_ids),
        )

    for backend in backends:
        if len(backend_times[backend]) > 0:
            median_time = np.median(backend_times[backend])
            std_time = np.std(backend_times[backend])

            total_tokens = batch_size * seq_len
            # Memory bandwidth calculation
            # Read: q + k + pos_ids
            # Write: q_rope + k_rope
            problem_bytes = (
                total_tokens * num_qo_heads * head_dim * input_dtype.itemsize  # q read
                + total_tokens
                * num_kv_heads
                * head_dim
                * input_dtype.itemsize  # k read
                + total_tokens * 4  # pos_ids read (int32)
                + total_tokens
                * num_qo_heads
                * head_dim
                * input_dtype.itemsize  # q_rope write
                + total_tokens
                * num_kv_heads
                * head_dim
                * input_dtype.itemsize  # k_rope write
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
                cur_res["seq_len"] = seq_len
                cur_res["num_qo_heads"] = num_qo_heads
                cur_res["num_kv_heads"] = num_kv_heads
                cur_res["head_dim"] = head_dim
                cur_res["rotary_dim"] = rotary_dim
                cur_res["rope_theta"] = rope_theta
                cur_res["rope_scale"] = rope_scale
                cur_res["interleave"] = interleave
                cur_res["backend"] = backend
                cur_res["case_tag"] = args.case_tag
                res.append(cur_res)
    return res


def testApplyLlama31Rope(args):
    """
    Test apply_llama31_rope API (with indptr/offsets).

    This test:
    1. Generates random Q and K tensors
    2. Runs flashinfer.rope.apply_llama31_rope with indptr/offsets
    3. Measures performance metrics (TB/sec)

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    warn_if_pdl_unsupported(args, args.routine)
    if args.verbose >= 1:
        print("[INFO] Running testApplyLlama31Rope")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    ## Parse input arguments
    backends = args.backends[:]
    batch_size = args.batch_size
    seq_len = args.seq_len
    num_qo_heads = args.num_qo_heads
    num_kv_heads = args.num_kv_heads
    head_dim = args.head_dim
    rotary_dim = args.rotary_dim
    rope_scale = args.rope_scale
    rope_theta = args.rope_theta
    interleave = args.interleave
    low_freq_factor = args.low_freq_factor
    high_freq_factor = args.high_freq_factor
    old_context_len = args.old_context_len
    is_cuda_graph_compatible = not args.no_cuda_graph
    res = []

    backends = filter_backends_by_compute_capability(backends, args.routine, device)
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

    input_dtype = dtype_str_to_torch_dtype(args.input_dtype)

    ## Prepare input tensors
    # Shape: (batch_size * seq_len, num_heads, head_dim)
    q = torch.randn(
        batch_size * seq_len, num_qo_heads, head_dim, dtype=input_dtype, device=device
    )
    k = torch.randn(
        batch_size * seq_len, num_kv_heads, head_dim, dtype=input_dtype, device=device
    )

    # indptr for ragged tensor
    indptr = torch.arange(
        0, (batch_size + 1) * seq_len, seq_len, dtype=torch.int32, device=device
    )

    # offsets (per-request position offset)
    offsets = torch.zeros(batch_size, dtype=torch.int32, device=device)

    if args.verbose >= 2:
        print(f"[VVERBOSE] {q.shape = }")
        print(f"[VVERBOSE] {k.shape = }")
        print(f"[VVERBOSE] {indptr.shape = }")
        print(f"[VVERBOSE] {offsets.shape = }")
        print(f"[VVERBOSE] {rotary_dim = }")
        print(f"[VVERBOSE] {rope_scale = }")
        print(f"[VVERBOSE] {rope_theta = }")
        print(f"[VVERBOSE] {interleave = }")
        print(f"[VVERBOSE] {low_freq_factor = }")
        print(f"[VVERBOSE] {high_freq_factor = }")
        print(f"[VVERBOSE] {old_context_len = }")

    def run_backend(backend, q, k, indptr, offsets):
        if backend == "cuda":
            return _rope_fn("apply_llama31_rope", args.inplace)(
                q,
                k,
                indptr=indptr,
                offsets=offsets,
                rotary_dim=rotary_dim,
                interleave=interleave,
                rope_scale=rope_scale,
                rope_theta=rope_theta,
                low_freq_factor=low_freq_factor,
                high_freq_factor=high_freq_factor,
                old_context_len=old_context_len,
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
            input_args=(cur_backend, q, k, indptr, offsets),
        )

    for backend in backends:
        if len(backend_times[backend]) > 0:
            median_time = np.median(backend_times[backend])
            std_time = np.std(backend_times[backend])

            total_tokens = batch_size * seq_len
            # Memory bandwidth calculation
            # Read: q + k
            # Write: q_rope + k_rope
            problem_bytes = (
                total_tokens * num_qo_heads * head_dim * input_dtype.itemsize  # q read
                + total_tokens
                * num_kv_heads
                * head_dim
                * input_dtype.itemsize  # k read
                + total_tokens
                * num_qo_heads
                * head_dim
                * input_dtype.itemsize  # q_rope write
                + total_tokens
                * num_kv_heads
                * head_dim
                * input_dtype.itemsize  # k_rope write
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
                cur_res["seq_len"] = seq_len
                cur_res["num_qo_heads"] = num_qo_heads
                cur_res["num_kv_heads"] = num_kv_heads
                cur_res["head_dim"] = head_dim
                cur_res["rotary_dim"] = rotary_dim
                cur_res["rope_theta"] = rope_theta
                cur_res["rope_scale"] = rope_scale
                cur_res["interleave"] = interleave
                cur_res["backend"] = backend
                cur_res["case_tag"] = args.case_tag
                res.append(cur_res)
    return res


def testApplyLlama31RopePosIds(args):
    """
    Test apply_llama31_rope API with pos_ids.

    This test:
    1. Generates random Q and K tensors
    2. Runs flashinfer.rope.apply_llama31_rope with pos_ids
    3. Measures performance metrics (TB/sec)

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    warn_if_pdl_unsupported(args, args.routine)
    if args.verbose >= 1:
        print("[INFO] Running testApplyLlama31RopePosIds")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    ## Parse input arguments
    backends = args.backends[:]
    batch_size = args.batch_size
    seq_len = args.seq_len
    num_qo_heads = args.num_qo_heads
    num_kv_heads = args.num_kv_heads
    head_dim = args.head_dim
    rotary_dim = args.rotary_dim
    rope_scale = args.rope_scale
    rope_theta = args.rope_theta
    interleave = args.interleave
    low_freq_factor = args.low_freq_factor
    high_freq_factor = args.high_freq_factor
    old_context_len = args.old_context_len
    is_cuda_graph_compatible = not args.no_cuda_graph
    res = []

    backends = filter_backends_by_compute_capability(backends, args.routine, device)
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

    input_dtype = dtype_str_to_torch_dtype(args.input_dtype)

    ## Prepare input tensors
    # Shape: (batch_size * seq_len, num_heads, head_dim)
    q = torch.randn(
        batch_size * seq_len, num_qo_heads, head_dim, dtype=input_dtype, device=device
    )
    k = torch.randn(
        batch_size * seq_len, num_kv_heads, head_dim, dtype=input_dtype, device=device
    )

    # pos_ids: (batch_size * seq_len,)
    pos_ids = torch.arange(seq_len, dtype=torch.int32, device=device).repeat(batch_size)

    if args.verbose >= 2:
        print(f"[VVERBOSE] {q.shape = }")
        print(f"[VVERBOSE] {k.shape = }")
        print(f"[VVERBOSE] {pos_ids.shape = }")
        print(f"[VVERBOSE] {rotary_dim = }")
        print(f"[VVERBOSE] {rope_scale = }")
        print(f"[VVERBOSE] {rope_theta = }")
        print(f"[VVERBOSE] {interleave = }")
        print(f"[VVERBOSE] {low_freq_factor = }")
        print(f"[VVERBOSE] {high_freq_factor = }")
        print(f"[VVERBOSE] {old_context_len = }")

    def run_backend(backend, q, k, pos_ids):
        if backend == "cuda":
            return _rope_fn("apply_llama31_rope_pos_ids", args.inplace)(
                q,
                k,
                pos_ids,
                rotary_dim=rotary_dim,
                interleave=interleave,
                rope_scale=rope_scale,
                rope_theta=rope_theta,
                low_freq_factor=low_freq_factor,
                high_freq_factor=high_freq_factor,
                old_context_len=old_context_len,
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
            input_args=(cur_backend, q, k, pos_ids),
        )

    for backend in backends:
        if len(backend_times[backend]) > 0:
            median_time = np.median(backend_times[backend])
            std_time = np.std(backend_times[backend])

            total_tokens = batch_size * seq_len
            # Memory bandwidth calculation
            # Read: q + k + pos_ids
            # Write: q_rope + k_rope
            problem_bytes = (
                total_tokens * num_qo_heads * head_dim * input_dtype.itemsize  # q read
                + total_tokens
                * num_kv_heads
                * head_dim
                * input_dtype.itemsize  # k read
                + total_tokens * 4  # pos_ids read (int32)
                + total_tokens
                * num_qo_heads
                * head_dim
                * input_dtype.itemsize  # q_rope write
                + total_tokens
                * num_kv_heads
                * head_dim
                * input_dtype.itemsize  # k_rope write
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
                cur_res["seq_len"] = seq_len
                cur_res["num_qo_heads"] = num_qo_heads
                cur_res["num_kv_heads"] = num_kv_heads
                cur_res["head_dim"] = head_dim
                cur_res["rotary_dim"] = rotary_dim
                cur_res["rope_theta"] = rope_theta
                cur_res["rope_scale"] = rope_scale
                cur_res["interleave"] = interleave
                cur_res["backend"] = backend
                cur_res["case_tag"] = args.case_tag
                res.append(cur_res)
    return res


def testApplyRopeWithCosSinCache(args):
    """
    Test apply_rope_with_cos_sin_cache API.

    This test:
    1. Generates random Q and K tensors with precomputed cos/sin cache
    2. Runs flashinfer.rope.apply_rope_with_cos_sin_cache
    3. Measures performance metrics (TB/sec)

    Note: This API uses flattened Q/K tensors and a combined cos_sin_cache.

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    warn_if_pdl_unsupported(args, args.routine)
    if args.verbose >= 1:
        print("[INFO] Running testApplyRopeWithCosSinCache")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    ## Parse input arguments
    backends = args.backends[:]
    batch_size = args.batch_size
    seq_len = args.seq_len
    num_qo_heads = args.num_qo_heads
    num_kv_heads = args.num_kv_heads
    head_dim = args.head_dim
    rotary_dim = args.rotary_dim
    interleave = args.interleave
    is_cuda_graph_compatible = not args.no_cuda_graph
    res = []

    backends = filter_backends_by_compute_capability(backends, args.routine, device)
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

    input_dtype = dtype_str_to_torch_dtype(args.input_dtype)

    ## Prepare input tensors
    total_tokens = batch_size * seq_len
    # Shape: (total_tokens, num_heads * head_dim) - flattened for this API
    q = torch.randn(
        total_tokens, num_qo_heads * head_dim, dtype=input_dtype, device=device
    )
    k = torch.randn(
        total_tokens, num_kv_heads * head_dim, dtype=input_dtype, device=device
    )

    # Precomputed cos_sin_cache: (max_seq_len, rotary_dim)
    # First half is cos, second half is sin
    max_seq_len = seq_len
    # API requires FP32 cache regardless of Q/K dtype (#5025).
    cos_sin_cache = torch.randn(
        max_seq_len, rotary_dim, dtype=torch.float32, device=device
    )

    # positions: (total_tokens,)
    positions = torch.arange(seq_len, dtype=torch.long, device=device).repeat(
        batch_size
    )

    # is_neox is the inverse of interleave
    is_neox = not interleave

    if args.verbose >= 2:
        print(f"[VVERBOSE] {q.shape = }")
        print(f"[VVERBOSE] {k.shape = }")
        print(f"[VVERBOSE] {cos_sin_cache.shape = }")
        print(f"[VVERBOSE] {positions.shape = }")
        print(f"[VVERBOSE] {is_neox = }")

    def run_backend(backend, positions, q, k, cos_sin_cache):
        if backend == "cuda":
            return _rope_fn("apply_rope_with_cos_sin_cache", args.inplace)(
                positions,
                q,
                k,
                head_dim,
                cos_sin_cache,
                is_neox=is_neox,
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
            input_args=(cur_backend, positions, q, k, cos_sin_cache),
        )

    for backend in backends:
        if len(backend_times[backend]) > 0:
            median_time = np.median(backend_times[backend])
            std_time = np.std(backend_times[backend])

            # Memory bandwidth calculation
            # Read: q + k + cos_sin_cache + positions
            # Write: q_rope + k_rope
            problem_bytes = (
                total_tokens * num_qo_heads * head_dim * input_dtype.itemsize  # q read
                + total_tokens
                * num_kv_heads
                * head_dim
                * input_dtype.itemsize  # k read
                + max_seq_len * rotary_dim * 4  # cos_sin_cache read (float32)
                + total_tokens * 8  # positions read (int64)
                + total_tokens
                * num_qo_heads
                * head_dim
                * input_dtype.itemsize  # q_rope write
                + total_tokens
                * num_kv_heads
                * head_dim
                * input_dtype.itemsize  # k_rope write
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
                cur_res["seq_len"] = seq_len
                cur_res["num_qo_heads"] = num_qo_heads
                cur_res["num_kv_heads"] = num_kv_heads
                cur_res["head_dim"] = head_dim
                cur_res["rotary_dim"] = rotary_dim
                cur_res["interleave"] = interleave
                cur_res["backend"] = backend
                cur_res["case_tag"] = args.case_tag
                res.append(cur_res)
    return res


def testMlaRopeQuantizeFp8(args):
    """
    Test mla_rope_quantize_fp8 API (for MLA attention).

    This test:
    1. Generates random pre-split Q and K tensors (rotary and non-rotary parts)
    2. Creates precomputed cos_sin_cache
    3. Runs flashinfer.rope.mla_rope_quantize_fp8
    4. Measures performance metrics (TB/sec)

    Note: This API takes pre-split q_rope, k_rope, q_nope, k_nope tensors
    and a precomputed cos_sin_cache. It is the same as rope_quantize_fp8.

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    if args.verbose >= 1:
        print("[INFO] Running testMlaRopeQuantizeFp8")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    ## Parse input arguments
    backends = args.backends[:]
    batch_size = args.batch_size
    seq_len = args.seq_len
    num_qo_heads = args.num_qo_heads
    num_kv_heads = args.num_kv_heads
    head_dim = args.head_dim
    no_rope_dim = args.no_rope_dim
    rope_dim = head_dim - no_rope_dim
    interleave = args.interleave
    is_cuda_graph_compatible = not args.no_cuda_graph
    res = []

    backends = filter_backends_by_compute_capability(backends, args.routine, device)
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

    input_dtype = dtype_str_to_torch_dtype(args.input_dtype)
    quant_dtype = dtype_str_to_torch_dtype(args.quant_dtype)

    ## Prepare input tensors (pre-split for this API)
    total_tokens = batch_size * seq_len
    max_seq_len = seq_len

    # q_rope: (total_tokens, num_qo_heads, rope_dim)
    q_rope = torch.randn(
        total_tokens, num_qo_heads, rope_dim, dtype=input_dtype, device=device
    )
    # k_rope: (total_tokens, rope_dim) for MLA (no num_kv_heads dimension)
    k_rope = torch.randn(total_tokens, rope_dim, dtype=input_dtype, device=device)
    # q_nope: (total_tokens, num_qo_heads, no_rope_dim) or None
    q_nope = (
        torch.randn(
            total_tokens, num_qo_heads, no_rope_dim, dtype=input_dtype, device=device
        )
        if no_rope_dim > 0
        else None
    )
    # k_nope: (total_tokens, no_rope_dim) for MLA or None
    k_nope = (
        torch.randn(total_tokens, no_rope_dim, dtype=input_dtype, device=device)
        if no_rope_dim > 0
        else None
    )

    # Precomputed cos_sin_cache: (max_seq_len, rope_dim) in float32
    cos_sin_cache = torch.randn(
        max_seq_len, rope_dim, dtype=torch.float32, device=device
    )

    # pos_ids: (total_tokens,)
    pos_ids = torch.arange(seq_len, dtype=torch.int32, device=device).repeat(batch_size)

    # is_neox is the inverse of interleave
    is_neox = not interleave

    if args.verbose >= 2:
        print(f"[VVERBOSE] {q_rope.shape = }")
        print(f"[VVERBOSE] {k_rope.shape = }")
        print(
            f"[VVERBOSE] q_nope.shape = {q_nope.shape if q_nope is not None else None}"
        )
        print(
            f"[VVERBOSE] k_nope.shape = {k_nope.shape if k_nope is not None else None}"
        )
        print(f"[VVERBOSE] {cos_sin_cache.shape = }")
        print(f"[VVERBOSE] {pos_ids.shape = }")
        print(f"[VVERBOSE] {rope_dim = }")
        print(f"[VVERBOSE] {no_rope_dim = }")
        print(f"[VVERBOSE] {is_neox = }")

    def run_backend(backend, q_rope, k_rope, q_nope, k_nope, cos_sin_cache, pos_ids):
        if backend == "cuda":
            return flashinfer.rope.mla_rope_quantize_fp8(
                q_rope,
                k_rope,
                q_nope,
                k_nope,
                cos_sin_cache,
                pos_ids,
                is_neox=is_neox,
                quantize_dtype=quant_dtype,
                enable_pdl=args.enable_pdl,
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
            input_args=(
                cur_backend,
                q_rope,
                k_rope,
                q_nope,
                k_nope,
                cos_sin_cache,
                pos_ids,
            ),
        )

    for backend in backends:
        if len(backend_times[backend]) > 0:
            median_time = np.median(backend_times[backend])
            std_time = np.std(backend_times[backend])

            # Memory bandwidth calculation for MLA
            # Read: q_rope + k_rope + q_nope + k_nope + cos_sin_cache + pos_ids
            # Write: q_rope_out + k_rope_out + q_nope_out + k_nope_out
            nope_bytes = 0
            if no_rope_dim > 0:
                nope_bytes = (
                    total_tokens
                    * num_qo_heads
                    * no_rope_dim
                    * input_dtype.itemsize  # q_nope read
                    + total_tokens
                    * no_rope_dim
                    * input_dtype.itemsize  # k_nope read (MLA shape)
                    + total_tokens
                    * num_qo_heads
                    * no_rope_dim
                    * quant_dtype.itemsize  # q_nope_out write
                    + total_tokens
                    * no_rope_dim
                    * quant_dtype.itemsize  # k_nope_out write (MLA shape)
                )
            problem_bytes = (
                total_tokens
                * num_qo_heads
                * rope_dim
                * input_dtype.itemsize  # q_rope read
                + total_tokens
                * rope_dim
                * input_dtype.itemsize  # k_rope read (MLA shape)
                + nope_bytes
                + max_seq_len * rope_dim * 4  # cos_sin_cache read (float32)
                + total_tokens * 4  # pos_ids read (int32)
                + total_tokens
                * num_qo_heads
                * rope_dim
                * quant_dtype.itemsize  # q_rope_out write
                + total_tokens
                * rope_dim
                * quant_dtype.itemsize  # k_rope_out write (MLA shape)
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
                cur_res["seq_len"] = seq_len
                cur_res["num_qo_heads"] = num_qo_heads
                cur_res["num_kv_heads"] = num_kv_heads
                cur_res["head_dim"] = head_dim
                cur_res["no_rope_dim"] = no_rope_dim
                cur_res["interleave"] = interleave
                cur_res["quant_dtype"] = args.quant_dtype
                cur_res["backend"] = backend
                cur_res["case_tag"] = args.case_tag
                res.append(cur_res)
    return res


def testRopeQuantizeFp8(args):
    """
    Test rope_quantize_fp8 API.

    This test:
    1. Generates random pre-split Q and K tensors (rotary and non-rotary parts)
    2. Creates precomputed cos_sin_cache
    3. Runs flashinfer.rope.rope_quantize_fp8
    4. Measures performance metrics (TB/sec)

    Note: This API takes pre-split q_rope, k_rope, q_nope, k_nope tensors
    and a precomputed cos_sin_cache.

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    if args.verbose >= 1:
        print("[INFO] Running testRopeQuantizeFp8")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    ## Parse input arguments
    backends = args.backends[:]
    batch_size = args.batch_size
    seq_len = args.seq_len
    num_qo_heads = args.num_qo_heads
    num_kv_heads = args.num_kv_heads
    head_dim = args.head_dim
    rotary_dim = args.rotary_dim
    no_rope_dim = args.no_rope_dim
    interleave = args.interleave
    mla_kv = args.mla_kv
    is_cuda_graph_compatible = not args.no_cuda_graph
    run_refcheck = args.refcheck
    res = []

    backends = filter_backends_by_compute_capability(backends, args.routine, device)
    # The cuTile kernel supports only the interleaved (GPT-J) layout with a 2D
    # MLA key (flashinfer/rope.py rope_quantize_fp8).
    if "cutile" in backends and not (interleave and mla_kv):
        print(
            "[INFO] cutile backend requires --interleave and --mla_kv. Skipping cutile."
        )
        backends.remove("cutile")
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

    input_dtype = dtype_str_to_torch_dtype(args.input_dtype)
    quant_dtype = dtype_str_to_torch_dtype(args.quant_dtype)

    ## Prepare input tensors (pre-split for this API)
    total_tokens = batch_size * seq_len
    max_seq_len = seq_len
    # MLA: a single 2D K latent shared by all heads
    if mla_kv:
        num_kv_heads = 1
    k_head_shape = () if mla_kv else (num_kv_heads,)

    # q_rope: (total_tokens, num_qo_heads, rotary_dim)
    q_rope = torch.randn(
        total_tokens, num_qo_heads, rotary_dim, dtype=input_dtype, device=device
    )
    # k_rope: (total_tokens, num_kv_heads, rotary_dim), MLA (total_tokens, rotary_dim)
    k_rope = torch.randn(
        total_tokens, *k_head_shape, rotary_dim, dtype=input_dtype, device=device
    )
    # q_nope: (total_tokens, num_qo_heads, no_rope_dim) or None
    q_nope = (
        torch.randn(
            total_tokens, num_qo_heads, no_rope_dim, dtype=input_dtype, device=device
        )
        if no_rope_dim > 0
        else None
    )
    # k_nope: (total_tokens, num_kv_heads, no_rope_dim) or None
    k_nope = (
        torch.randn(
            total_tokens, *k_head_shape, no_rope_dim, dtype=input_dtype, device=device
        )
        if no_rope_dim > 0
        else None
    )

    # Precomputed cos_sin_cache: (max_seq_len, rotary_dim) in float32
    cos_sin_cache = torch.randn(
        max_seq_len, rotary_dim, dtype=torch.float32, device=device
    )

    # pos_ids: (total_tokens,)
    pos_ids = torch.arange(seq_len, dtype=torch.int32, device=device).repeat(batch_size)

    # is_neox is the inverse of interleave
    is_neox = not interleave

    if args.verbose >= 2:
        print(f"[VVERBOSE] {q_rope.shape = }")
        print(f"[VVERBOSE] {k_rope.shape = }")
        print(
            f"[VVERBOSE] q_nope.shape = {q_nope.shape if q_nope is not None else None}"
        )
        print(
            f"[VVERBOSE] k_nope.shape = {k_nope.shape if k_nope is not None else None}"
        )
        print(f"[VVERBOSE] {cos_sin_cache.shape = }")
        print(f"[VVERBOSE] {pos_ids.shape = }")
        print(f"[VVERBOSE] {rotary_dim = }")
        print(f"[VVERBOSE] {no_rope_dim = }")
        print(f"[VVERBOSE] {is_neox = }")

    def run_backend(backend, q_rope, k_rope, q_nope, k_nope, cos_sin_cache, pos_ids):
        if backend in ("cuda", "cutile"):
            return flashinfer.rope.rope_quantize_fp8(
                q_rope,
                k_rope,
                q_nope,
                k_nope,
                cos_sin_cache,
                pos_ids,
                is_neox=is_neox,
                quantize_dtype=quant_dtype,
                enable_pdl=args.enable_pdl,
                backend=backend,
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    # Cross-backend reference check: all four fp8 outputs within one fp8 step
    if run_refcheck and len(backends) < 2:
        print(
            "[INFO] refcheck skipped: single backend (rope_quantize_fp8 refcheck "
            "compares cuda against cutile)."
        )
    if run_refcheck and len(backends) > 1:
        outputs = {
            b: [
                t.detach().clone()
                for t in run_backend(
                    b, q_rope, k_rope, q_nope, k_nope, cos_sin_cache, pos_ids
                )
            ]
            for b in backends
        }
        ref_backend = backends[0]
        for cur_backend in backends[1:]:
            for name, cur, ref in zip(
                ("q_rope", "k_rope", "q_nope", "k_nope"),
                outputs[cur_backend],
                outputs[ref_backend],
                strict=False,
            ):
                if cur.numel() == 0:
                    continue
                num_bad = int((fp8_code_distance(cur, ref) > 1).sum().item())
                if num_bad > 0:
                    msg = (
                        f"[{args.routine}] {name}: {cur_backend} vs {ref_backend}: "
                        f"{num_bad}/{cur.numel()} elements differ by > 1 fp8 step"
                    )
                    if args.allow_output_mismatch:
                        print(f"[WARNING] {msg}")
                    else:
                        raise AssertionError(msg)

    # Storage for timing results
    backend_times = {backend: [] for backend in backends}
    for cur_backend in backends:
        backend_times[cur_backend] = bench_gpu_time(
            fn=run_backend,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            enable_cupti=args.use_cupti,
            use_cuda_graph=is_cuda_graph_compatible,
            input_args=(
                cur_backend,
                q_rope,
                k_rope,
                q_nope,
                k_nope,
                cos_sin_cache,
                pos_ids,
            ),
        )

    for backend in backends:
        if len(backend_times[backend]) > 0:
            median_time = np.median(backend_times[backend])
            std_time = np.std(backend_times[backend])

            # Memory bandwidth calculation
            # Read: q_rope + k_rope + q_nope + k_nope + cos_sin_cache + pos_ids
            # Write: q_rope_out + k_rope_out + q_nope_out + k_nope_out
            nope_bytes = 0
            if no_rope_dim > 0:
                nope_bytes = (
                    total_tokens
                    * num_qo_heads
                    * no_rope_dim
                    * input_dtype.itemsize  # q_nope read
                    + total_tokens
                    * num_kv_heads
                    * no_rope_dim
                    * input_dtype.itemsize  # k_nope read
                    + total_tokens
                    * num_qo_heads
                    * no_rope_dim
                    * quant_dtype.itemsize  # q_nope_out write
                    + total_tokens
                    * num_kv_heads
                    * no_rope_dim
                    * quant_dtype.itemsize  # k_nope_out write
                )
            problem_bytes = (
                total_tokens
                * num_qo_heads
                * rotary_dim
                * input_dtype.itemsize  # q_rope read
                + total_tokens
                * num_kv_heads
                * rotary_dim
                * input_dtype.itemsize  # k_rope read
                + nope_bytes
                + max_seq_len * rotary_dim * 4  # cos_sin_cache read (float32)
                + total_tokens * 4  # pos_ids read (int32)
                + total_tokens
                * num_qo_heads
                * rotary_dim
                * quant_dtype.itemsize  # q_rope_out write
                + total_tokens
                * num_kv_heads
                * rotary_dim
                * quant_dtype.itemsize  # k_rope_out write
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
                cur_res["seq_len"] = seq_len
                cur_res["num_qo_heads"] = num_qo_heads
                cur_res["num_kv_heads"] = num_kv_heads
                cur_res["head_dim"] = head_dim
                cur_res["rotary_dim"] = rotary_dim
                cur_res["no_rope_dim"] = no_rope_dim
                cur_res["interleave"] = interleave
                cur_res["quant_dtype"] = args.quant_dtype
                cur_res["backend"] = backend
                cur_res["case_tag"] = args.case_tag
                res.append(cur_res)
    return res


def testRopeQuantizeFp8AppendPagedKvCache(args):
    """
    Test rope_quantize_fp8_append_paged_kv_cache API.

    This test:
    1. Generates random pre-split Q, K, V tensors with precomputed cos_sin_cache
    2. Creates paged KV cache in FP8
    3. Runs flashinfer.rope.rope_quantize_fp8_append_paged_kv_cache
    4. Measures performance metrics (TB/sec)

    Note: This API takes pre-split tensors (q_rope, k_rope, q_nope, k_nope, v)
    and a precomputed cos_sin_cache. The paged KV cache is a tuple of
    (k_cache, v_cache) both in FP8.

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    if args.verbose >= 1:
        print("[INFO] Running testRopeQuantizeFp8AppendPagedKvCache")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    ## Parse input arguments
    backends = args.backends[:]
    batch_size = args.batch_size
    seq_len = args.seq_len
    num_qo_heads = args.num_qo_heads
    num_kv_heads = args.num_kv_heads
    head_dim = args.head_dim
    rotary_dim = args.rotary_dim
    no_rope_dim = head_dim - rotary_dim  # MLA: kv_lora_rank (ckv width)
    interleave = args.interleave
    page_size = args.page_size
    kv_layout = args.kv_layout
    mla_kv = args.mla_kv
    is_cuda_graph_compatible = not args.no_cuda_graph
    res = []
    # MLA: one shared 2D K latent, no V, (ckv_cache, kpe_cache) paged caches
    if mla_kv:
        num_kv_heads = 1

    backends = filter_backends_by_compute_capability(backends, args.routine, device)
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

    input_dtype = dtype_str_to_torch_dtype(args.input_dtype)
    quant_dtype = dtype_str_to_torch_dtype(args.quant_dtype)

    ## Prepare input tensors (pre-split for this API)
    total_tokens = batch_size * seq_len
    max_seq_len = seq_len

    # q_rope: (total_tokens, num_qo_heads, rotary_dim)
    q_rope = torch.randn(
        total_tokens, num_qo_heads, rotary_dim, dtype=input_dtype, device=device
    )
    k_head_shape = () if mla_kv else (num_kv_heads,)
    # k_rope: (total_tokens, num_kv_heads, rotary_dim), MLA (total_tokens, rotary_dim)
    k_rope = torch.randn(
        total_tokens, *k_head_shape, rotary_dim, dtype=input_dtype, device=device
    )
    # q_nope: (total_tokens, num_qo_heads, no_rope_dim) or None
    q_nope = (
        torch.randn(
            total_tokens, num_qo_heads, no_rope_dim, dtype=input_dtype, device=device
        )
        if no_rope_dim > 0
        else None
    )
    # k_nope: (total_tokens, num_kv_heads, no_rope_dim) or None
    k_nope = (
        torch.randn(
            total_tokens, *k_head_shape, no_rope_dim, dtype=input_dtype, device=device
        )
        if no_rope_dim > 0
        else None
    )
    # v: (total_tokens, num_kv_heads, head_dim); MLA has no V
    v = (
        None
        if mla_kv
        else torch.randn(
            total_tokens, num_kv_heads, head_dim, dtype=input_dtype, device=device
        )
    )

    # Precomputed cos_sin_cache: (max_seq_len, rotary_dim) in float32
    cos_sin_cache = torch.randn(
        max_seq_len, rotary_dim, dtype=torch.float32, device=device
    )

    # pos_ids: (total_tokens,)
    pos_ids = torch.arange(seq_len, dtype=torch.int32, device=device).repeat(batch_size)

    # Paged KV cache - separate k and v caches as a tuple
    # Note: FP8 tensors cannot be created with randn, use empty instead
    num_pages_per_request = (seq_len + page_size - 1) // page_size
    total_pages = batch_size * num_pages_per_request
    if mla_kv:
        # MLA caches are 3D (max_pages, page_size, dim): ckv, then kpe
        k_cache = torch.empty(
            total_pages, page_size, no_rope_dim, dtype=quant_dtype, device=device
        )
        v_cache = torch.empty(
            total_pages, page_size, rotary_dim, dtype=quant_dtype, device=device
        )
    elif kv_layout == "NHD":
        k_cache = torch.empty(
            total_pages,
            page_size,
            num_kv_heads,
            head_dim,
            dtype=quant_dtype,
            device=device,
        )
        v_cache = torch.empty(
            total_pages,
            page_size,
            num_kv_heads,
            head_dim,
            dtype=quant_dtype,
            device=device,
        )
    else:  # HND
        k_cache = torch.empty(
            total_pages,
            num_kv_heads,
            page_size,
            head_dim,
            dtype=quant_dtype,
            device=device,
        )
        v_cache = torch.empty(
            total_pages,
            num_kv_heads,
            page_size,
            head_dim,
            dtype=quant_dtype,
            device=device,
        )
    paged_kv_cache = (k_cache, v_cache)

    # KV indices: page indices for each request
    kv_indices = torch.arange(total_pages, dtype=torch.int32, device=device)

    # KV indptr: (batch_size + 1,)
    kv_indptr = torch.arange(
        0,
        (batch_size + 1) * num_pages_per_request,
        num_pages_per_request,
        dtype=torch.int32,
        device=device,
    )

    # Batch indices: which request each token belongs to
    batch_indices = torch.arange(
        batch_size, dtype=torch.int32, device=device
    ).repeat_interleave(seq_len)

    # Positions: position within each request's sequence for each token
    positions = torch.arange(seq_len, dtype=torch.int32, device=device).repeat(
        batch_size
    )

    # is_neox is the inverse of interleave
    is_neox = not interleave

    if args.verbose >= 2:
        print(f"[VVERBOSE] {q_rope.shape = }")
        print(f"[VVERBOSE] {k_rope.shape = }")
        print(
            f"[VVERBOSE] q_nope.shape = {q_nope.shape if q_nope is not None else None}"
        )
        print(
            f"[VVERBOSE] k_nope.shape = {k_nope.shape if k_nope is not None else None}"
        )
        print(f"[VVERBOSE] v.shape = {v.shape if v is not None else None}")
        print(f"[VVERBOSE] {cos_sin_cache.shape = }")
        print(f"[VVERBOSE] {pos_ids.shape = }")
        print(f"[VVERBOSE] k_cache.shape = {k_cache.shape}")
        print(f"[VVERBOSE] v_cache.shape = {v_cache.shape}")
        print(f"[VVERBOSE] {kv_indices.shape = }")
        print(f"[VVERBOSE] {kv_indptr.shape = }")
        print(f"[VVERBOSE] {batch_indices.shape = }")
        print(f"[VVERBOSE] {positions.shape = }")
        print(f"[VVERBOSE] {rotary_dim = }")
        print(f"[VVERBOSE] {no_rope_dim = }")
        print(f"[VVERBOSE] {is_neox = }")
        print(f"[VVERBOSE] {page_size = }")
        print(f"[VVERBOSE] {kv_layout = }")

    def run_backend(
        backend,
        q_rope,
        k_rope,
        q_nope,
        k_nope,
        v,
        cos_sin_cache,
        pos_ids,
        paged_kv_cache,
        kv_indices,
        kv_indptr,
        batch_indices,
        positions,
    ):
        if backend == "cuda":
            return flashinfer.rope.rope_quantize_fp8_append_paged_kv_cache(
                q_rope,
                k_rope,
                q_nope,
                k_nope,
                v,
                cos_sin_cache,
                pos_ids,
                paged_kv_cache,
                kv_indices,
                kv_indptr,
                batch_indices,
                positions,
                is_neox=is_neox,
                quantize_dtype=quant_dtype,
                page_size=page_size,
                kv_layout=kv_layout,
                enable_pdl=args.enable_pdl,
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
            input_args=(
                cur_backend,
                q_rope,
                k_rope,
                q_nope,
                k_nope,
                v,
                cos_sin_cache,
                pos_ids,
                paged_kv_cache,
                kv_indices,
                kv_indptr,
                batch_indices,
                positions,
            ),
        )

    for backend in backends:
        if len(backend_times[backend]) > 0:
            median_time = np.median(backend_times[backend])
            std_time = np.std(backend_times[backend])

            # Memory bandwidth calculation
            # Read: q_rope + k_rope + q_nope + k_nope + v + cos_sin_cache + pos_ids
            # Write: q_rope_out + q_nope_out + paged_kv_cache (k and v)
            nope_bytes = 0
            if no_rope_dim > 0:
                nope_bytes = (
                    total_tokens
                    * num_qo_heads
                    * no_rope_dim
                    * input_dtype.itemsize  # q_nope read
                    + total_tokens
                    * num_kv_heads
                    * no_rope_dim
                    * input_dtype.itemsize  # k_nope read
                    + total_tokens
                    * num_qo_heads
                    * no_rope_dim
                    * quant_dtype.itemsize  # q_nope_out write
                )
            problem_bytes = (
                total_tokens
                * num_qo_heads
                * rotary_dim
                * input_dtype.itemsize  # q_rope read
                + total_tokens
                * num_kv_heads
                * rotary_dim
                * input_dtype.itemsize  # k_rope read
                + (0 if mla_kv else 1)
                * total_tokens
                * num_kv_heads
                * head_dim
                * input_dtype.itemsize  # v read (none for MLA)
                + nope_bytes
                + max_seq_len * rotary_dim * 4  # cos_sin_cache read (float32)
                + total_tokens * 4  # pos_ids read (int32)
                + total_tokens
                * num_qo_heads
                * rotary_dim
                * quant_dtype.itemsize  # q_rope_out write
                + total_tokens
                * num_kv_heads
                * head_dim
                * quant_dtype.itemsize
                * (1 if mla_kv else 2)  # k, v (MLA: ckv + kpe) to paged cache
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
                cur_res["seq_len"] = seq_len
                cur_res["num_qo_heads"] = num_qo_heads
                cur_res["num_kv_heads"] = num_kv_heads
                cur_res["head_dim"] = head_dim
                cur_res["rotary_dim"] = rotary_dim
                cur_res["no_rope_dim"] = no_rope_dim
                cur_res["interleave"] = interleave
                cur_res["quant_dtype"] = args.quant_dtype
                cur_res["page_size"] = page_size
                cur_res["kv_layout"] = kv_layout
                cur_res["backend"] = backend
                cur_res["case_tag"] = args.case_tag
                res.append(cur_res)
    return res
