# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark routine for ``flashinfer.top_k_varlen``.

``top_k_varlen`` is a **sparse-attention KV-index selection** primitive — it
picks the top-K KV positions per request under variable per-request sequence
lengths — NOT a vocabulary-sampling op. It therefore lives in its own routine
module (mirroring ``flashinfer/topk_varlen/topk_varlen.py``) rather than under
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
from flashinfer.utils import BackendSupportedError, get_compute_capability

from .flashinfer_benchmark_utils import (
    dtype_str_to_torch_dtype,
    get_device,
    print_perf_metrics,
)

# top_k_varlen's runners. Per-CC support is NOT hard-coded here; it is
# resolved at runtime from top_k_varlen's @backend_requirement decorator via
# ``flashinfer.top_k_varlen.is_backend_supported(backend, cc)`` (the single
# source of truth), mirroring how the GEMM routines rely on their support
# checkers (e.g. mm_fp4 / bmm_fp8) instead of a compute-capability table.
_TOP_K_VARLEN_BACKENDS = ("radix", "gvr", "gvr_2", "radix_cutlass")
# Opt-in only (not in the default set, so lines without --backends keep their
# meaning): radix_filter (large-N specialist) and auto (the library heuristic).
_EXTRA_BACKENDS = ("radix_filter", "auto")


def parse_topk_varlen_args(line, parser):
    """Parse command line arguments for the ``top_k_varlen`` routine."""
    parser.add_argument(
        "--backends",
        type=str,
        required=False,
        nargs="+",
        default=None,
        choices=list(_TOP_K_VARLEN_BACKENDS + _EXTRA_BACKENDS),
        help=(
            "Backends to benchmark. Default: radix gvr gvr_2 radix_cutlass, "
            "narrowed to those supported on the current GPU (resolved via "
            "top_k_varlen's @backend_requirement support checks). 'radix' = CuTe "
            "DSL multi-CTA (Blackwell), 'gvr' = GVR LB (Blackwell), 'radix_cutlass' "
            "= masked CUTLASS radix (any GPU), 'radix_filter' = filtered radix "
            "(10.0/10.3/10.7), 'auto' = the library's per-shape choice."
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
    parser.add_argument(
        "--next_n",
        type=int,
        required=False,
        default=1,
        help="Speculative-decode tokens per request: logits has batch_size * next_n rows.",
    )
    parser.add_argument(
        "--compress_ratio",
        type=int,
        required=False,
        default=1,
        choices=[1, 4],
        help=(
            "KV tokens per logit column (4 = DSv4 compressed indexer). The logits "
            "width is max_seq_len // compress_ratio; seq_lens stay in KV tokens."
        ),
    )
    parser.add_argument(
        "--seq_len_distribution",
        type=str,
        required=False,
        default="uniform",
        choices=["uniform", "random"],
        help=(
            "Per-request KV length: 'uniform' = every request is max_seq_len long; "
            "'random' = uniform in [top_k * compress_ratio + next_n, max_seq_len]."
        ),
    )
    parser.add_argument(
        "--no_load_balance",
        action="store_true",
        default=False,
        help="Pass load_balance=False (gvr's single-CTA-per-row path).",
    )
    parser.add_argument(
        "--return_values",
        action="store_true",
        default=False,
        help="Also return the selected logit values.",
    )
    parser.add_argument(
        "--topk_mode",
        type=str,
        required=False,
        default="decode",
        choices=["decode", "prefill"],
        help=(
            "'decode' (default): one row per request (x next_n), keys from column 0. "
            "'prefill': the DSA indexer's windowed mode (row_starts): --chunk_len new "
            "tokens per request score one packed key axis of batch_size * "
            "max_seq_len columns; served by gvr_2 / auto only."
        ),
    )
    parser.add_argument(
        "--chunk_len",
        type=int,
        required=False,
        default=0,
        help=(
            "--topk_mode prefill only: new tokens (rows) per request in the chunk; "
            "the remaining max_seq_len - chunk_len keys are a cached prefix. "
            "0 (default) = max_seq_len (no prefix)."
        ),
    )
    parser.add_argument(
        "--page_size",
        type=int,
        required=False,
        default=0,
        help=(
            "--topk_mode decode only: > 0 writes physical KV slots through a "
            "per-request page table of this page size (fused paged output, "
            "gvr_2 / auto only, indices only). 0 (default) = logical columns."
        ),
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
    """Benchmark top_k_varlen with its runners: 'radix', 'radix_cutlass', 'gvr',
    'gvr_2', 'radix_filter', 'auto'.

    Runners
    -------
    radix          — CuTe DSL single-pass multi-CTA radix (Blackwell sm_100+);
                     ``pre_idx=None``. Filtered out on older hardware.
    radix_cutlass  — masked CUTLASS radix fallback; ``pre_idx=None``. Runs on any GPU.
    gvr            — GVR LB kernel; passes a pre-computed ``pre_idx``. Blackwell
                     (sm_100/103/107) only.
    gvr_2          — self-sampling GVR V2 (TRT-LLM PR #17821 port); passes the
                     same ``pre_idx``. fp32 logits + top_k in {512, 1024, 2048}.
    radix_filter   — filtered radix (sm_100/103/107, cutlass-dsl >= 4.8); hint-free.
    auto           — whatever the library's shape heuristic picks for this call.

    Per-CC support is resolved via ``top_k_varlen.is_backend_supported``; a
    backend whose @backend_requirement check rejects this problem (dtype,
    top_k, compress_ratio, DSL version, ...) is skipped after one probe call.

    Reference check compares, per row, the multiset of selected logit values
    against ``torch.topk`` over that row's valid columns (tie-aware).
    """
    if args.verbose >= 1:
        print("[INFO] Running testTopKVarlen")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)
    if args.generate_repro_command:
        print(f"[INFO] To reproduce this test case, run: {args.repro_command}")

    # Default (unset) --backends means the four original backends.
    backends = args.backends
    if backends is None:
        backends = list(_TOP_K_VARLEN_BACKENDS)
    batch_size = args.batch_size
    max_seq_len = args.max_seq_len
    top_k = args.top_k
    next_n = args.next_n
    compress_ratio = args.compress_ratio
    load_balance = not args.no_load_balance
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
        if backend != "auto" and not flashinfer.top_k_varlen.is_backend_supported(
            backend, cc
        ):
            backends.remove(backend)
            print(
                f"[WARNING] {backend} for routine {args.routine} is not supported "
                f"on compute capability {major}.{minor}. Skipping."
            )
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

    prefill = args.topk_mode == "prefill"
    paged = args.page_size > 0
    if prefill and (next_n != 1 or compress_ratio != 1 or paged):
        raise ValueError(
            "--topk_mode prefill takes no --next_n, --compress_ratio or --page_size"
        )
    if paged and args.return_values:
        raise ValueError("--page_size output is indices only (no --return_values)")
    if prefill or paged:
        # Windowed (row_starts) and paged-output modes are served by gvr_2 only
        # (topk_varlen.py backend docs); auto resolves to it.
        for backend in list(backends):
            if backend not in ("gvr_2", "auto"):
                backends.remove(backend)
                print(
                    f"[WARNING] {backend} does not serve "
                    f"{'windowed (prefill)' if prefill else 'paged-output'} mode. "
                    "Skipping."
                )
        if len(backends) == 0:
            print("[ERROR] No backends to test. Exiting.")
            return res
    if prefill:
        return _test_topk_varlen_prefill(args, backends, device)

    input_dtype = dtype_str_to_torch_dtype(args.input_dtype)

    num_rows = batch_size * next_n
    width = max_seq_len // compress_ratio
    logits = torch.randn(num_rows, width, dtype=input_dtype, device=device)
    if args.seq_len_distribution == "random":
        lo = min(top_k * compress_ratio + next_n, max_seq_len)
        seq_lens = torch.randint(
            lo, max_seq_len + 1, (batch_size,), dtype=torch.int32, device=device
        )
    else:
        seq_lens = torch.full(
            (batch_size,), max_seq_len, dtype=torch.int32, device=device
        )

    # pre_idx: argmax in col-0, sequential fill elsewhere (GVR convention); one
    # hint row per request, taken from its first draft row.
    argmax_idx = logits[::next_n].argmax(dim=-1).int()
    pre_idx = torch.zeros(batch_size, top_k, dtype=torch.int32, device=device)
    pre_idx[:, 0] = argmax_idx
    for j in range(1, top_k):
        pre_idx[:, j] = j

    if args.verbose >= 2:
        print(f"[VVERBOSE] {logits.shape = }, {seq_lens.shape = }, {top_k = }")

    page_kwargs = {}
    if paged:
        # Each request owns a random permutation of distinct physical pages.
        pages_per_req = -(-width // args.page_size)
        page_table = (
            torch.randperm(batch_size * pages_per_req, device=device)
            .view(batch_size, pages_per_req)
            .int()
        )
        page_kwargs = dict(page_table=page_table, page_size=args.page_size)

    def run_backend(backend, logits):
        # radix / radix_cutlass / radix_filter are hint-free; gvr needs the hint,
        # gvr_2 and auto use it when given (paged output is hint-free).
        hint = pre_idx if backend in ("gvr", "gvr_2", "auto") and not paged else None
        return flashinfer.top_k_varlen(
            logits,
            seq_lens,
            top_k,
            pre_idx=hint,
            compress_ratio=compress_ratio,
            next_n=next_n,
            return_values=args.return_values,
            backend=backend,
            load_balance=load_balance,
            **page_kwargs,
        )

    # One probe call per backend: the library's @backend_requirement problem
    # checks reject unsupported (dtype, top_k, compress_ratio, DSL) combinations
    # before launch; such a backend is skipped instead of failing the line.
    outputs = {}
    for backend in list(backends):
        try:
            outputs[backend] = run_backend(backend, logits)[0].detach().clone()
        except BackendSupportedError as e:
            reason = str(e)
        except ValueError as e:
            if "Problem size is not supported" not in str(e):
                raise
            reason = str(e)
        else:
            continue
        backends.remove(backend)
        print(f"[WARNING] {backend} skipped: {reason}")
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

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

    # Row t of request i ranks its first (seq_len - next_n + t + 1) // cr columns.
    t = torch.arange(num_rows, device=device) % next_n
    lens = seq_lens.repeat_interleave(next_n)
    valid = ((lens - next_n + t + 1) // compress_ratio).clamp(0, width)
    # Logit columns charged to the bandwidth figure: uniform mode keeps the
    # historical full-width charge; random mode charges each row's real length.
    if args.seq_len_distribution == "random":
        logit_cols = int(valid.sum().item())
    else:
        logit_cols = num_rows * width

    if run_refcheck:
        col_idx = torch.arange(width, device=device).unsqueeze(0)
        masked = logits.float().masked_fill(
            col_idx >= valid.unsqueeze(1), float("-inf")
        )
        ref_vals = torch.topk(masked, k=top_k, dim=-1).values.sort(dim=-1).values

        if paged:
            # Physical slot -> logical column of the row's own request.
            ps = args.page_size
            local_page = torch.empty(
                page_table.numel(), dtype=torch.long, device=device
            )
            local_page[page_table.flatten().long()] = torch.arange(
                page_table.shape[1], device=device
            ).repeat(batch_size)
        for backend, out_indices in outputs.items():
            if paged:
                slots = out_indices.long().clamp_min(0)
                out_indices = torch.where(
                    out_indices >= 0,
                    local_page[slots // ps] * ps + slots % ps,
                    out_indices.long(),
                )
            got = torch.gather(masked, 1, out_indices.long().clamp_min(0))
            got = got.masked_fill(out_indices < 0, float("-inf")).sort(dim=-1).values
            row_bad = ~((got == ref_vals) | (got.isinf() & ref_vals.isinf())).all(-1)
            mismatches = int(row_bad.sum().item())
            if mismatches > 0:
                pct = 100.0 * mismatches / num_rows
                print(
                    f"[REFCHECK] Backend {backend}: {mismatches}/{num_rows} rows "
                    f"({pct:.1f}%) differ from torch.topk reference"
                )
                if not args.allow_output_mismatch:
                    raise AssertionError(f"[ERROR] Backend {backend} output mismatch")

    for backend in backends:
        if len(backend_times[backend]) > 0:
            median_time = np.median(backend_times[backend])
            std_time = np.std(backend_times[backend])

            problem_bytes = (
                logit_cols * input_dtype.itemsize  # logits read
                # pre_idx read (int32): charged to every backend, including the
                # hint-free ones (pre-existing accounting, kept for continuity)
                + batch_size * top_k * 4
                # page-table read for paged output
                + (page_table.numel() * 4 if paged else 0)
                + num_rows * top_k * 4  # indices write (int32)
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
                cur_res["next_n"] = next_n
                cur_res["compress_ratio"] = compress_ratio
                cur_res["seq_len_distribution"] = args.seq_len_distribution
                cur_res["load_balance"] = load_balance
                cur_res["return_values"] = args.return_values
                if paged:
                    cur_res["page_size"] = args.page_size
                    cur_res["topk_mode"] = "decode_paged"
                cur_res["backend"] = backend
                cur_res["case_tag"] = args.case_tag
                res.append(cur_res)
    return res


def _test_topk_varlen_prefill(args, backends, device):
    """Windowed (prefill) mode: the DSA indexer's chunked-prefill layout.

    Every request contributes --chunk_len new-token rows and owns max_seq_len
    consecutive columns of one packed key axis (its cached prefix, then the
    chunk). Row r of request q ranks columns [row_starts[r], row_starts[r] +
    seq_lens[r]) with seq_lens[r] = prefix + position + 1 (causal).
    """
    res = []
    batch_size, max_seq_len, top_k = args.batch_size, args.max_seq_len, args.top_k
    chunk = args.chunk_len or max_seq_len
    if not 0 < chunk <= max_seq_len:
        raise ValueError("--chunk_len must be in [1, max_seq_len]")
    prefix = max_seq_len - chunk
    input_dtype = dtype_str_to_torch_dtype(args.input_dtype)
    num_rows = batch_size * chunk
    num_cols = batch_size * max_seq_len
    logits = torch.randn(num_rows, num_cols, dtype=input_dtype, device=device)
    req = torch.arange(num_rows, device=device) // chunk
    pos = torch.arange(num_rows, device=device) % chunk
    row_starts = (req * max_seq_len).int().contiguous()
    seq_lens = (prefix + pos + 1).int().contiguous()

    def run_backend(backend, logits):
        return flashinfer.top_k_varlen(
            logits,
            seq_lens,
            top_k,
            row_starts=row_starts,
            max_window_len=max_seq_len,
            return_values=args.return_values,
            backend=backend,
        )

    outputs = {}
    for backend in list(backends):
        try:
            outputs[backend] = run_backend(backend, logits)[0].detach().clone()
        except (BackendSupportedError, ValueError) as e:
            backends.remove(backend)
            print(f"[WARNING] {backend} skipped: {e}")
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

    backend_times = {
        backend: bench_gpu_time(
            fn=run_backend,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            enable_cupti=args.use_cupti,
            use_cuda_graph=False,
            input_args=(backend, logits),
        )
        for backend in backends
    }

    if args.refcheck:
        col = torch.arange(num_cols, device=device).unsqueeze(0)
        start = row_starts.long().unsqueeze(1)
        in_window = (col >= start) & (col < start + seq_lens.long().unsqueeze(1))
        masked = logits.float().masked_fill(~in_window, float("-inf"))
        k = min(top_k, num_cols)
        ref_vals = torch.topk(masked, k=k, dim=-1).values.sort(dim=-1).values
        for backend, out_indices in outputs.items():
            # Window-local positions -> logits columns.
            cols = (out_indices.long() + start).clamp(0, num_cols - 1)
            got = torch.gather(masked, 1, cols)
            got = got.masked_fill(out_indices < 0, float("-inf"))
            got = got.sort(dim=-1).values[:, -k:]
            row_bad = ~((got == ref_vals) | (got.isinf() & ref_vals.isinf())).all(-1)
            mismatches = int(row_bad.sum().item())
            if mismatches > 0:
                print(
                    f"[REFCHECK] Backend {backend}: {mismatches}/{num_rows} rows "
                    "differ from torch.topk reference"
                )
                if not args.allow_output_mismatch:
                    raise AssertionError(f"[ERROR] Backend {backend} output mismatch")

    window_cols = int(seq_lens.sum().item())
    for backend in backends:
        median_time = np.median(backend_times[backend])
        std_time = np.std(backend_times[backend])
        problem_bytes = (
            window_cols * input_dtype.itemsize  # in-window logits read
            + num_rows * 8  # row_starts + seq_lens
            + num_rows * top_k * 4  # indices write
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
            cur_res["topk_mode"] = "prefill"
            cur_res["chunk_len"] = chunk
            cur_res["return_values"] = args.return_values
            cur_res["backend"] = backend
            cur_res["case_tag"] = args.case_tag
            res.append(cur_res)
    return res
