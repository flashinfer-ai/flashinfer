# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark routine for ``flashinfer.top_k_varlen``.

``top_k_varlen`` is a **sparse-attention KV-index selection** primitive — it
picks the top-K KV positions per request under variable per-request sequence
lengths — NOT a vocabulary-sampling op. It therefore lives in its own routine
module (mirroring ``flashinfer/top_k/topk_varlen.py``) rather than under
``routines/sampling.py``, and its backends are radix / gvr / radix_cutlass
(there is no generic "cuda" backend here).

Entry points (the unified-benchmark convention):
  * ``parse_topk_varlen_args(line, parser)`` — routine-specific CLI args.
  * ``run_topk_varlen_test(args)``           — dispatch for this category.
"""

from collections import defaultdict

import numpy as np
import torch

import flashinfer
from flashinfer.testing.utils import bench_gpu_time
from flashinfer.utils import get_compute_capability

from .flashinfer_benchmark_utils import (
    dtype_str_to_torch_dtype,
    get_device,
    print_perf_metrics,
)

# top_k_varlen's three runners. Per-CC support is NOT hard-coded here; it is
# resolved at runtime from top_k_varlen's @backend_requirement decorator via
# ``flashinfer.top_k_varlen.is_backend_supported(backend, cc)`` (the single
# source of truth), mirroring how the GEMM routines rely on their support
# checkers (e.g. mm_fp4 / bmm_fp8) instead of a compute-capability table.
_TOP_K_VARLEN_BACKENDS = ("radix", "gvr", "radix_cutlass")


def parse_topk_varlen_args(line, parser):
    """Parse command line arguments for the ``top_k_varlen`` routine."""
    parser.add_argument(
        "--backends",
        type=str,
        required=False,
        nargs="+",
        default=None,
        choices=list(_TOP_K_VARLEN_BACKENDS),
        help=(
            "Backends to benchmark. Default: every backend supported on the "
            "current GPU (resolved via top_k_varlen's @backend_requirement "
            "support checks). 'radix' = CuTe DSL multi-CTA (Blackwell), 'gvr' = "
            "GVR LB (Blackwell), 'radix_cutlass' = masked CUTLASS radix (any GPU)."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,
        help="Number of requests (rows).",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        required=True,
        help="Per-row width N (max sequence length scanned).",
    )
    parser.add_argument(
        "--input_dtype",
        type=str,
        required=False,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Data type of the input logits tensor.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        required=False,
        default=512,
        help="Number of top elements to select per row.",
    )

    args = parser.parse_args(line)

    if args.verbose >= 1:
        print(f"[INFO] {args = }")
    return args


def run_topk_varlen_test(args):
    """Route a ``top_k_varlen`` benchmark case to its test function."""
    if args.routine == "top_k_varlen":
        return testTopKVarlen(args)
    raise ValueError(f"Unsupported routine: {args.routine}")


def testTopKVarlen(args):
    """Benchmark top_k_varlen with three runners: 'radix', 'radix_cutlass', 'gvr'.

    Runners
    -------
    radix          — CuTe DSL single-pass multi-CTA radix (Blackwell sm_100+);
                     ``pre_idx=None``. Filtered out on older hardware.
    radix_cutlass  — masked CUTLASS radix fallback; ``pre_idx=None``. Runs on any GPU.
    gvr            — GVR LB kernel; passes a pre-computed ``pre_idx``. Blackwell
                     (sm_100/103) only. Per-CC support is resolved via
                     ``top_k_varlen.is_backend_supported`` (its
                     ``@backend_requirement`` decorator).

    Reference check compares the *set* of selected indices against ``torch.topk``
    applied to logits masked to ``seq_lens``.
    """
    if args.verbose >= 1:
        print("[INFO] Running testTopKVarlen")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)
    if args.generate_repro_command:
        print(f"[INFO] To reproduce this test case, run: {args.repro_command}")

    # Default (unset) --backends means "every backend this routine supports".
    backends = args.backends
    if backends is None:
        backends = list(_TOP_K_VARLEN_BACKENDS)
    batch_size = args.batch_size
    max_seq_len = args.max_seq_len
    top_k = args.top_k
    is_cuda_graph_compatible = False  # GVR LB uses dynamic counters; not graph-safe
    run_refcheck = args.refcheck
    res = []

    # Narrow to backends supported on this GPU using top_k_varlen's
    # @backend_requirement support checks (single source of truth) rather than a
    # hard-coded compute-capability table -- mirrors the GEMM routines' reliance
    # on mm_fp4 / bmm_fp8 support checkers.
    major, minor = get_compute_capability(device)
    cc = major * 10 + minor
    for backend in list(backends):
        if not flashinfer.top_k_varlen.is_backend_supported(backend, cc):
            backends.remove(backend)
            print(
                f"[WARNING] {backend} for routine {args.routine} is not supported "
                f"on compute capability {major}.{minor}. Skipping."
            )
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

    input_dtype = dtype_str_to_torch_dtype(args.input_dtype)

    logits = torch.randn(batch_size, max_seq_len, dtype=input_dtype, device=device)
    seq_lens = torch.full((batch_size,), max_seq_len, dtype=torch.int32, device=device)

    # pre_idx: argmax in col-0, sequential fill elsewhere (GVR convention)
    argmax_idx = logits.argmax(dim=-1).int()
    pre_idx = torch.zeros(batch_size, top_k, dtype=torch.int32, device=device)
    pre_idx[:, 0] = argmax_idx
    for j in range(1, top_k):
        pre_idx[:, j] = j

    if args.verbose >= 2:
        print(f"[VVERBOSE] {logits.shape = }, {seq_lens.shape = }, {top_k = }")

    def run_backend(backend, logits):
        if backend == "radix":
            # CuTe DSL multi-CTA radix (Blackwell); no pre_idx needed.
            return flashinfer.top_k_varlen(
                logits, seq_lens, top_k, pre_idx=None, backend="radix"
            )
        elif backend == "radix_cutlass":
            # Masked CUTLASS radix fallback (any GPU); no pre_idx needed.
            return flashinfer.top_k_varlen(
                logits, seq_lens, top_k, pre_idx=None, backend="radix_cutlass"
            )
        elif backend == "gvr":
            return flashinfer.top_k_varlen(
                logits, seq_lens, top_k, pre_idx=pre_idx, backend="gvr"
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    backend_times = {backend: [] for backend in backends}
    outputs = {}
    for cur_backend in backends:
        if run_refcheck:
            # top_k_varlen returns (indices, values_or_None); refcheck compares indices.
            outputs[cur_backend] = run_backend(cur_backend, logits)[0].detach()
        backend_times[cur_backend] = bench_gpu_time(
            fn=run_backend,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            enable_cupti=args.use_cupti,
            use_cuda_graph=is_cuda_graph_compatible,
            input_args=(cur_backend, logits),
        )

    if run_refcheck and outputs:
        col_idx = torch.arange(max_seq_len, device=device).unsqueeze(0)
        masked = logits.masked_fill(col_idx >= seq_lens.unsqueeze(1), float("-inf"))
        ref_indices = torch.topk(masked.float(), k=top_k, dim=-1).indices.int()

        for backend, out_indices in outputs.items():
            mismatches = sum(
                set(ref_indices[row].cpu().tolist())
                != set(out_indices[row].cpu().tolist())
                for row in range(batch_size)
            )
            pct = 100.0 * mismatches / batch_size
            if mismatches > 0:
                print(
                    f"[REFCHECK] Backend {backend}: {mismatches}/{batch_size} rows "
                    f"({pct:.1f}%) differ from torch.topk reference"
                )
                if not args.allow_output_mismatch:
                    raise AssertionError(f"[ERROR] Backend {backend} output mismatch")

    for backend in backends:
        if len(backend_times[backend]) > 0:
            median_time = np.median(backend_times[backend])
            std_time = np.std(backend_times[backend])

            problem_bytes = (
                batch_size * max_seq_len * input_dtype.itemsize  # logits read
                + batch_size * top_k * 4  # pre_idx read (int32)
                + batch_size * top_k * 4  # indices write (int32)
            )
            tb_per_sec = problem_bytes / (1e9 * median_time)

            print_perf_metrics(backend, median_time, std_time, 0, tb_per_sec)

            if args.output_path is not None:
                cur_res = defaultdict(str)
                cur_res["routine"] = args.routine
                cur_res["median_time"] = median_time
                cur_res["std_time"] = std_time
                cur_res["tflops"] = 0
                cur_res["tb_per_sec"] = tb_per_sec
                cur_res["batch_size"] = batch_size
                cur_res["max_seq_len"] = max_seq_len
                cur_res["top_k"] = top_k
                cur_res["backend"] = backend
                cur_res["case_tag"] = args.case_tag
                res.append(cur_res)
    return res
