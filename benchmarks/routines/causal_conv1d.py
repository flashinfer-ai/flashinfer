# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the FlashInfer project
"""Inference convolution comparisons including state reads and writes."""

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from time import perf_counter

import numpy as np
import torch

from flashinfer.mamba import causal_conv1d_fn, causal_conv1d_update
from flashinfer.mamba.causal_conv1d import _prefill_workspace_size
from flashinfer.testing.utils import bench_gpu_time

from .flashinfer_benchmark_utils import (
    filter_backends_by_compute_capability,
    get_device,
    print_perf_metrics,
    warn_if_pdl_unsupported,
)


def parse_causal_conv1d_args(line, parser):
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument(
        "--dim", type=int, required=True, help="Convolution channel count."
    )
    parser.add_argument(
        "--input_row_stride",
        type=int,
        help="Input row stride in elements; defaults to dim.",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=1,
        help="Tokens per sequence, or maximum with --varlen.",
    )
    parser.add_argument(
        "--varlen", action="store_true", help="Unequal prefill lengths."
    )
    parser.add_argument(
        "--seq_lens",
        type=int,
        nargs="+",
        help="Explicit per-sequence prefill lengths, including empty sequences.",
    )
    parser.add_argument(
        "--speculative",
        action="store_true",
        help="Update with an extended verification-history pool.",
    )
    parser.add_argument(
        "--has_initial_state",
        action="store_true",
        help="Prefill reads existing history; update always reads it.",
    )
    parser.add_argument(
        "--profile_launches",
        action="store_true",
        help="Count launches in a separate process after timing each case.",
    )
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--state_layout", choices=("ds", "sd"), default="sd")
    parser.add_argument(
        "--weight_dtype", choices=("bfloat16", "float32"), default="float32"
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=("flashinfer", "triton"),
        default=["flashinfer", "triton"],
    )
    args = parser.parse_args(line)
    if min(args.batch_size, args.dim, args.seq_len) <= 0:
        parser.error("batch_size, dim and seq_len must be positive")
    if args.input_row_stride is None:
        args.input_row_stride = args.dim
    if args.input_row_stride < args.dim:
        parser.error("input_row_stride must be at least dim")
    if args.num_iters <= 0:
        parser.error("num_iters must be positive")
    if args.varlen and args.routine == "causal_conv1d_update":
        parser.error("the Triton update benchmark requires uniform sequence lengths")
    if args.seq_lens is not None:
        if (
            args.routine != "causal_conv1d_fn"
            or args.varlen
            or len(args.seq_lens) != args.batch_size
            or min(args.seq_lens) < 0
            or sum(args.seq_lens) == 0
        ):
            parser.error(
                "seq_lens requires a nonempty prefill batch, one nonnegative length per sequence, and no --varlen"
            )
        args.seq_len = max(args.seq_lens)
    if args.speculative and args.routine == "causal_conv1d_fn":
        parser.error("--speculative applies to causal_conv1d_update")
    return args


@torch.inference_mode()
def run_causal_conv1d_test(args, *, profile_only=False):
    # routines.mamba adds the existing reference directory to sys.path.
    from triton_reference.causal_conv1d_triton import (
        causal_conv1d_fn as triton_prefill,
        causal_conv1d_update as triton_update,
    )

    device = get_device(args)
    backends = filter_backends_by_compute_capability(
        args.backends[:], args.routine, device
    )
    if not backends:
        return []
    warn_if_pdl_unsupported(args, args.routine)
    torch.manual_seed(args.random_seed)
    n, c, t = args.batch_size, args.dim, args.seq_len
    prefill = args.routine == "causal_conv1d_fn"
    lengths = (
        args.seq_lens
        if args.seq_lens is not None
        else ([1 + (i * 17) % t for i in range(n)] if args.varlen else [t] * n)
    )
    rows = sum(lengths)
    backing = (
        torch.randn(rows, args.input_row_stride, device=device, dtype=torch.bfloat16)
        * 0.2
    )
    x = backing[:, :c]
    weight = (
        torch.randn(c, 4, device=device, dtype=getattr(torch, args.weight_dtype)) * 0.2
    )
    bias = torch.randn(c, device=device) * 0.1 if args.bias else None
    offsets = torch.tensor(
        [0] + torch.tensor(lengths).cumsum(0).tolist(),
        dtype=torch.int32,
        device=device,
    )
    slots = torch.randperm(n, device=device).to(torch.int32)
    initial = torch.full((n,), args.has_initial_state, device=device, dtype=torch.bool)
    accepted = (
        torch.ones(n, device=device, dtype=torch.int32) if args.speculative else None
    )
    history = t + 2 if args.speculative else 3
    state_shape = (n, history, c) if args.state_layout == "sd" else (n, c, history)
    clean = torch.randn(state_shape, device=device, dtype=torch.bfloat16) * 0.2
    if args.state_layout == "sd":
        clean = clean.transpose(1, 2)
    output = torch.empty_like(x)
    workspace_size = _prefill_workspace_size(rows, n) if prefill else 0
    workspace = (
        torch.empty(workspace_size, device=device, dtype=torch.int32)
        if workspace_size
        else None
    )

    def call(backend, state):
        if backend == "flashinfer":
            if prefill:
                return causal_conv1d_fn(
                    x,
                    weight,
                    state,
                    offsets,
                    slots,
                    initial,
                    bias=bias,
                    out=output,
                    workspace=workspace,
                )
            return causal_conv1d_update(
                x,
                weight,
                state,
                slots,
                cu_seqlens=offsets if t > 1 else None,
                num_accepted_tokens=accepted,
                bias=bias,
                out=output,
            )
        if prefill:
            return triton_prefill(
                x.T,
                weight,
                bias,
                state,
                offsets,
                lengths,
                cache_indices=slots,
                has_initial_state=initial,
                activation="silu",
            ).T
        value = triton_update(
            x.view(n, t, c).transpose(1, 2),
            state,
            weight,
            bias=bias,
            activation="silu",
            conv_state_indices=slots,
            num_accepted_tokens=accepted,
        )
        return value.transpose(1, 2).reshape(rows, c)

    if profile_only:
        counts = {}
        for backend in backends:
            state = clean.clone()
            for _ in range(max(1, args.dry_run_iters)):
                call(backend, state)
            state.copy_(clean)
            torch.cuda.synchronize(device)
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as profile:
                call(backend, state)
                torch.cuda.synchronize(device)
            counts[backend] = sum(
                event.device_type == torch.autograd.DeviceType.CUDA
                for event in profile.events()
            )
        return counts

    passed = graph_passed = None
    if args.refcheck and "flashinfer" in backends:
        reference_state = clean.clone()
        reference = call("triton", reference_state).clone()
        candidate_state = clean.clone()
        candidate = call("flashinfer", candidate_state).clone()
        torch.testing.assert_close(candidate, reference, atol=1e-3, rtol=1e-2)
        torch.testing.assert_close(
            candidate_state.view(torch.int16),
            reference_state.view(torch.int16),
            atol=0,
            rtol=0,
        )
        passed = True
        if not args.no_cuda_graph:
            for backend, expected, expected_state in (
                ("flashinfer", candidate, candidate_state),
                ("triton", reference, reference_state),
            ):
                graph_state = clean.clone()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = call(backend, graph_state)
                graph_state.copy_(clean)
                graph.replay()
                torch.testing.assert_close(
                    captured.view(torch.int16),
                    expected.view(torch.int16),
                    atol=0,
                    rtol=0,
                )
                torch.testing.assert_close(
                    graph_state.view(torch.int16),
                    expected_state.view(torch.int16),
                    atol=0,
                    rtol=0,
                )
                del graph, captured, graph_state
            graph_passed = True
    states = {backend: clean.clone() for backend in backends}
    times = {backend: [] for backend in backends}
    for backend in backends:
        call(backend, states[backend])
    # State resets and JIT are outside timing; both arms replay their own pools.
    for backend in backends:
        states[backend].copy_(clean)
        torch.cuda.synchronize(device)
        times[backend] = bench_gpu_time(
            lambda backend=backend: call(backend, states[backend]),
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            enable_cupti=args.use_cupti,
            use_cuda_graph=not args.no_cuda_graph,
            cold_l2_cache=True,
        )
    results = []
    for backend in backends:
        torch.cuda.synchronize(device)
        start = perf_counter()
        for _ in range(args.num_iters):
            call(backend, states[backend])
        host_us = (perf_counter() - start) * 1e6 / args.num_iters
        torch.cuda.synchronize(device)
        median = float(np.median(times[backend]))
        std = float(np.std(times[backend]))
        state_reads = n * c * 3 * 2 if not prefill or args.has_initial_state else 0
        state_writes = n * c * history * 2
        logical_bytes = (
            x.numel() * 4
            + state_reads
            + state_writes
            + weight.numel() * weight.element_size()
            + (bias.numel() * bias.element_size() if bias is not None else 0)
        )
        bandwidth = logical_bytes / (median * 1e9)
        tflops = rows * c * 8 / (median * 1e9)
        print_perf_metrics(backend, median, std, tflops, bandwidth)
        print(f"[INFO] {backend}: eager host submission {host_us:.3f} us/call")
        results.append(
            defaultdict(
                str,
                dict(
                    routine=args.routine,
                    backend=backend,
                    median_time=median,
                    std_time=std,
                    tflops=tflops,
                    tb_per_sec=bandwidth,
                    batch_size=n,
                    dim=c,
                    input_dtype="bfloat16",
                    weight_dtype=args.weight_dtype,
                    num_tokens=rows,
                    conv_tokens_per_sequence=t,
                    conv_bias=args.bias,
                    conv_speculative=args.speculative,
                    conv_initial_state=args.has_initial_state if prefill else True,
                    conv_layout=args.state_layout,
                    conv_input_row_stride=x.stride(0),
                    varlen=args.varlen or args.seq_lens is not None,
                    conv_seq_lens=" ".join(map(str, lengths)),
                    refcheck_passed=passed,
                    conv_graph_refcheck_passed=graph_passed,
                    conv_workspace_bytes=(
                        workspace_size * 4 if backend == "flashinfer" else 0
                    ),
                    conv_host_submit_us=host_us,
                    conv_cuda_ops=None,
                    case_tag=args.case_tag,
                ),
            )
        )
    if args.profile_launches:
        with ProcessPoolExecutor(
            max_workers=1, mp_context=get_context("spawn")
        ) as pool:
            counts = pool.submit(
                run_causal_conv1d_test, args, profile_only=True
            ).result()
        for result in results:
            result["conv_cuda_ops"] = counts[result["backend"]]
            print(
                f"[INFO] {result['backend']}: {result['conv_cuda_ops']} CUDA launches"
            )
    return results
