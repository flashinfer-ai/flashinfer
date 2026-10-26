"""Benchmark MoE expert-parallel dispatch/combine of any MoEEpCommunication backend.

This is a communication-only benchmark. Only ``MoEEpCommunication.dispatch()``
and ``MoEEpCommunication.combine()`` are timed; input quantization and the
simulated expert output happen outside the timed phases, and no expert GEMM
runs between them. To time dispatch, expert computation and combine together
as one MoE layer, use ``benchmarks/moe_ep/bench_moe_ep.py``.

dispatch_us and combine_us report CUPTI kernel spans (first kernel start to
last kernel end of each phase), using CUDA graph replay by default or eager
execution with --no_cuda_graph. Backends that cannot be captured run eagerly.
Kernel spans need cupti-python>=13.4; without it, timing falls back to CUDA
events and benchmark_metadata.warning records the reason. --kernel_breakdown
additionally reports per-kernel statistics, C++ kernel names demangled by the
cxxfilt package. benchmarks/moe_ep/requirements.txt lists both packages.

Launch on one node, for example:

    # Sweep local batch sizes 1..1024 (powers of 2) on EP8.
    torchrun --standalone --nproc-per-node=8 benchmarks/moe_ep/bench_moe_ep_comm.py \\
        --backend nvlink_one_sided --profile deepseek_v3 -b 1 -e 1024 -f 2

    # Balanced routing, per-kernel breakdown and per-iteration stats, JSON report.
    torchrun --standalone --nproc-per-node=8 benchmarks/moe_ep/bench_moe_ep_comm.py \\
        --backend nvlink_one_sided --profile deepseek_v4_pro --perfect_router \\
        --kernel_breakdown --iter_stats -b 1 -e 1024 -f 2 --output_file out.json

    # Pin nvlink_one_sided to its CFT counted-write transport ("off": no CFT, so
    # small steps take the Lamport transport unless --lamport off); "auto" leaves
    # the choice to the backend.
    torchrun --standalone --nproc-per-node=8 benchmarks/moe_ep/bench_moe_ep_comm.py \\
        --backend nvlink_one_sided --profile kimi_k3 --cft on -b 1 -e 1024 -f 2

    # Compare nvlink_one_sided with and without its Lamport transport, which takes
    # the small steps ahead of CFT ("auto" turns it off under --cft on).
    torchrun --standalone --nproc-per-node=8 benchmarks/moe_ep/bench_moe_ep_comm.py \\
        --backend nvlink_one_sided --profile deepseek_v3 --lamport off -b 1 -e 128 -f 2

Omit --backend to benchmark every backend available on all ranks. Multi-node
runs use the usual torchrun rendezvous flags instead of --standalone.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from moe_ep_bench_utils import (
    PROFILES,
    add_profile_args,
    add_sweep_args,
    agree_on_cupti,
    allgather,
    first_error,
    gather_kernel_breakdown,
    gather_per_rank,
    kernel_spans_cupti,
    l2_flush_buffer,
    local_batch_sizes,
    make_routing,
    print_rank0,
    start_cupti,
    sync,
    warn_if_kernel_names_mangled,
    warn_kernel_span_unavailable,
)

_TAG = "bench_moe_ep_comm"

FP8_E4M3_MAX = 448.0
FP4_E2M1_MAX = 6.0
# The flashinfer.fused_moe.QuantFormat each --quant choice dispatches in;
# bf16 dispatches the unquantized token row.
DISPATCH_FORMATS = {
    "bf16": None,
    "fp8": "FP8PerTensor",
    "fp8_block_scale": "DeepSeekFp8",
    "mxfp8": "MXFP8",
    "nvfp4": "NVFP4",
}
QUANT_FORMATS = tuple(DISPATCH_FORMATS)
# NVLinkOneSidedConfig.cft and .lamport for each --cft / --lamport choice.
_CFT_MODES = {"auto": None, "on": True, "off": False}
_LAMPORT_MODES = {"auto": None, "on": True, "off": False}


def _backend_configs(args: argparse.Namespace) -> Dict[str, Callable[[], Any]]:
    """Benchmarkable backends: name -> factory of its MoEEpCommunication config."""
    from flashinfer.moe_ep import (
        CakeAlltoAllConfig,
        NVLinkOneSidedConfig,
        NVLinkTwoSidedConfig,
    )

    low_precision = bool(args.use_low_precision_combine)
    return {
        "nvlink_one_sided": lambda: NVLinkOneSidedConfig(
            use_low_precision_combine=low_precision,
            cft=_CFT_MODES[args.cft],
            lamport=_LAMPORT_MODES[args.lamport],
        ),
        "cake": lambda: CakeAlltoAllConfig(use_low_precision_combine=low_precision),
        "nvlink_two_sided": NVLinkTwoSidedConfig,
    }


# ----------------------------------------------------------------------------
# Input quantization
#
# The payload only has to have the byte layout the MoE computation would
# consume, so these reference quantizers run in plain PyTorch outside the
# timed region. Scale factors use the linear (unswizzled) per-token layout the
# dispatch carries.
# ----------------------------------------------------------------------------


def quantize_fp8_per_tensor(x: torch.Tensor) -> Tuple[torch.Tensor, None]:
    """FP8 E4M3 with one per-tensor scale; the scalar scale is not dispatched."""
    amax = x.abs().amax().float().clamp(min=1e-12)
    return (x.float() * (FP8_E4M3_MAX / amax)).to(torch.float8_e4m3fn), None


def quantize_fp8_block_scale(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """DeepSeek FP8: E4M3 values with one FP32 scale per 128 elements."""
    num_tokens, hidden = x.shape
    if hidden % 128:
        raise ValueError(f"fp8_block_scale needs hidden_size % 128 == 0, got {hidden}")
    blocks = x.float().view(num_tokens, hidden // 128, 128)
    scales = blocks.abs().amax(-1, keepdim=True).clamp(min=1e-12) / FP8_E4M3_MAX
    values = (blocks / scales).to(torch.float8_e4m3fn)
    return values.view(num_tokens, hidden), scales.squeeze(-1)


def quantize_mxfp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """MXFP8: E4M3 values with one UE8M0 scale per 32 elements."""
    num_tokens, hidden = x.shape
    if hidden % 32:
        raise ValueError(f"mxfp8 needs hidden_size % 32 == 0, got {hidden}")
    blocks = x.float().view(num_tokens, hidden // 32, 32)
    amax = blocks.abs().amax(-1, keepdim=True).clamp(min=2.0**-126)
    exponent = torch.ceil(torch.log2(amax / FP8_E4M3_MAX)).clamp(-127, 127)
    values = (blocks / torch.exp2(exponent)).to(torch.float8_e4m3fn)
    scales = (exponent.squeeze(-1) + 127).to(torch.uint8)
    return values.view(num_tokens, hidden), scales


def _e2m1_codes(values: torch.Tensor) -> torch.Tensor:
    """Round to the nearest FP4 E2M1 value and return its 4-bit sign-magnitude code."""
    midpoints = torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
        device=values.device,
        dtype=values.dtype,
    )
    magnitude = torch.bucketize(values.abs().clamp(max=FP4_E2M1_MAX), midpoints)
    sign = (values < 0).to(torch.uint8) << 3
    return magnitude.to(torch.uint8) | sign


def quantize_nvfp4(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """NVFP4: packed E2M1 pairs with one E4M3 scale per 16 elements.

    The per-tensor global scale is folded into the block scales; like the
    per-tensor FP8 scale it is not part of the dispatched payload.
    """
    num_tokens, hidden = x.shape
    if hidden % 16:
        raise ValueError(f"nvfp4 needs hidden_size % 16 == 0, got {hidden}")
    xf = x.float()
    global_scale = FP8_E4M3_MAX * FP4_E2M1_MAX / xf.abs().amax().clamp(min=1e-12)
    blocks = xf.view(num_tokens, hidden // 16, 16)
    block_amax = blocks.abs().amax(-1, keepdim=True)
    scales = (block_amax / FP4_E2M1_MAX * global_scale).to(torch.float8_e4m3fn)
    dequant_scale = scales.float() / global_scale
    scaled = torch.where(
        dequant_scale > 0, blocks / dequant_scale, torch.zeros_like(blocks)
    )
    codes = _e2m1_codes(scaled).view(num_tokens, hidden)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return packed, scales.view(num_tokens, hidden // 16).view(torch.uint8)


def quantize_input(
    x: torch.Tensor, quant: str
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if quant == "bf16":
        return x, None
    if quant == "fp8":
        return quantize_fp8_per_tensor(x)
    if quant == "fp8_block_scale":
        return quantize_fp8_block_scale(x)
    if quant == "mxfp8":
        return quantize_mxfp8(x)
    if quant == "nvfp4":
        return quantize_nvfp4(x)
    raise ValueError(f"unknown quant format {quant!r}; expected one of {QUANT_FORMATS}")


def _make_inputs(
    local_num_tokens: int,
    hidden_size: int,
    top_k: int,
    num_experts: int,
    ep_size: int,
    quant: str,
    perfect_router: bool,
    device: torch.device,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Return (hidden_states, hidden_states_scale, topk_ids, topk_weights)."""
    hidden_states = torch.randn(
        local_num_tokens, hidden_size, dtype=torch.bfloat16, device=device
    )
    hidden_states, hidden_states_scale = quantize_input(hidden_states, quant)
    topk_ids, topk_weights = make_routing(
        local_num_tokens, top_k, num_experts, ep_size, perfect_router, device
    )
    return hidden_states, hidden_states_scale, topk_ids, topk_weights


# ----------------------------------------------------------------------------
# Timing
# ----------------------------------------------------------------------------


def _time_dispatch_and_combine(
    comm: Any,
    *,
    hidden_states: torch.Tensor,
    hidden_states_scale: Optional[torch.Tensor],
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    step_max_tokens: int,
    hidden_size: int,
    warmup: int,
    iters: int,
    use_cuda_graph: bool,
    cupti_ctx: Optional[Any],
) -> Tuple[List[float], List[float], Dict[str, Any]]:
    """Measure per-iteration dispatch/combine latency in microseconds.

    An eager round first discovers the receive shape. Warmup and timed
    iterations then run either inside one CUDA graph replay or eagerly. L2
    flushing and zeroing the simulated expert output sit outside the timed
    phases. Returns CUDA-event times plus CUPTI statistics (kernel spans and
    per-kernel times) when CUPTI is enabled.
    """
    device = hidden_states.device
    l2_buffer = l2_flush_buffer(device)

    def _dispatch() -> torch.Tensor:
        received = comm.dispatch(
            hidden_states,
            topk_ids,
            topk_weights,
            hidden_states_scale=hidden_states_scale,
            runtime_max_tokens_per_rank=step_max_tokens,
        )
        return received.hidden_states

    # ---- Discover the receive shape and the combine input buffer ----
    recv_hidden_states = _dispatch()
    workspace_output = comm.get_combine_input_buffer(torch.bfloat16)
    static_output = (
        None
        if workspace_output is not None
        else torch.zeros(
            recv_hidden_states.shape[0],
            hidden_size,
            dtype=torch.bfloat16,
            device=device,
        )
    )

    def _expert_output() -> torch.Tensor:
        # Backends with a workspace-backed combine input hand out a fresh view
        # after every dispatch; writing the expert output there skips a copy.
        buffer = comm.get_combine_input_buffer(torch.bfloat16)
        return static_output if buffer is None else buffer

    comm.combine(workspace_output if workspace_output is not None else static_output)
    torch.cuda.synchronize()

    # ---- Timing events ----
    # External events become event-record nodes when captured into a CUDA
    # graph, so they stay queryable with elapsed_time() after replay.
    d_starts, d_ends, c_starts, c_ends = (
        [torch.cuda.Event(enable_timing=True, external=True) for _ in range(iters)]
        for _ in range(4)
    )
    # A CUDA event handle is created lazily on its first record().
    for event in d_starts + d_ends + c_starts + c_ends:
        event.record()
    torch.cuda.synchronize()

    phase_event_ids = []
    if cupti_ctx is not None:
        cupti, cupti_kernels, cupti_events = cupti_ctx
        for i in range(iters):
            phase_event_ids.append(
                tuple(
                    cupti.get_cuda_event_id(event.cuda_event)
                    for event in (d_starts[i], d_ends[i], c_starts[i], c_ends[i])
                )
            )

    def _run_iterations() -> None:
        for _ in range(warmup):
            l2_buffer.zero_()
            _dispatch()
            output = _expert_output()
            output.zero_()
            comm.combine(output)
        for i in range(iters):
            l2_buffer.zero_()
            d_starts[i].record()
            _dispatch()
            d_ends[i].record()
            output = _expert_output()
            output.zero_()
            c_starts[i].record()
            comm.combine(output)
            c_ends[i].record()

    if use_cuda_graph:
        # One replay covers warmup and timed iterations, so no host gaps remain.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _run_iterations()

    if cupti_ctx is not None:
        cupti.activity_flush_all(0)
        cupti_kernels.clear()
        cupti_events.clear()
    sync()
    if use_cuda_graph:
        graph.replay()
    else:
        _run_iterations()
    sync()
    if cupti_ctx is not None:
        cupti.activity_flush_all(0)

    dispatch_us = [d_starts[i].elapsed_time(d_ends[i]) * 1e3 for i in range(iters)]
    combine_us = [c_starts[i].elapsed_time(c_ends[i]) * 1e3 for i in range(iters)]

    stats: Dict[str, Any] = {
        "dispatch_kernels": [],
        "combine_kernels": [],
        "other_kernels": [],
    }
    if cupti_ctx is not None:
        try:
            stats = kernel_spans_cupti(
                cupti_kernels,
                cupti_events,
                ("dispatch", "combine"),
                phase_event_ids,
            )
        except RuntimeError as exc:
            # Keep every rank on the same collective sequence even if one trace is incomplete.
            stats["cupti_error"] = str(exc)
    return dispatch_us, combine_us, stats


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--backend",
        type=str.lower,
        default=None,
        choices=[
            "nvlink_one_sided",
            "cake",
            "nvlink_two_sided",
        ],
        help="Backend to benchmark (default: every backend available on all ranks).",
    )
    add_profile_args(parser)
    parser.add_argument(
        "--quant",
        default=None,
        choices=QUANT_FORMATS,
        help="Override the profile's dispatched activation format.",
    )
    add_sweep_args(parser)
    parser.add_argument("--iters", type=int, default=200, help="Timed iterations.")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations.")
    parser.add_argument(
        "--max_num_tokens_per_rank",
        type=int,
        default=None,
        help="Workspace capacity per rank (default: the largest scanned local batch).",
    )
    parser.add_argument(
        "--kernel_breakdown",
        action="store_true",
        help="Report per-kernel statistics (C++ kernel names demangled when the "
        "cxxfilt package is installed).",
    )
    parser.add_argument(
        "--iter_stats",
        action="store_true",
        help="Report mean/median/stdev/min/max over iterations instead of the mean.",
    )
    parser.add_argument("--output_file", default=None, help="Write a JSON report here.")
    parser.add_argument(
        "--random_seed",
        type=int,
        default=1234,
        help="Input seed (rank r uses seed + r).",
    )
    parser.add_argument(
        "--perfect_router",
        action="store_true",
        help="Deterministic balanced routing, removing load imbalance across ranks.",
    )
    parser.add_argument(
        "--use_low_precision_combine",
        action="store_true",
        help="Send combine payloads as FP8 (NVLink one-sided backends).",
    )
    parser.add_argument(
        "--cft",
        choices=sorted(_CFT_MODES),
        default="auto",
        help="CFT counted writes of nvlink_one_sided: auto (where supported, up "
        "to its token-count limits), on (every supported step) or off.",
    )
    parser.add_argument(
        "--lamport",
        choices=sorted(_LAMPORT_MODES),
        default="auto",
        help="Lamport transport of nvlink_one_sided, which takes the small steps "
        "ahead of CFT: auto (where supported; off under --cft on), on (not with "
        "--cft on) or off.",
    )
    parser.add_argument(
        "--no_cuda_graph",
        action="store_true",
        help="Run eagerly instead of replaying one CUDA graph.",
    )
    args = parser.parse_args()
    if args.lamport == "on" and args.cft == "on":
        parser.error("--lamport on and --cft on both ask for every eligible step")
    return args


def main() -> None:
    args = parse_args()

    cupti_ctx, cupti_error = start_cupti()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", device_id=device)
    rank, ep_size = dist.get_rank(), dist.get_world_size()
    cupti_ctx, cupti_warning = agree_on_cupti(cupti_ctx, cupti_error, _TAG)
    if args.kernel_breakdown:
        warn_if_kernel_names_mangled(_TAG)

    from flashinfer.fused_moe import QuantFormat
    from flashinfer.moe_ep import (
        BootstrapConfig,
        MoEEpCommParams,
        available_communication_backends,
        create_communication,
    )

    profile = PROFILES[args.profile]
    hidden_size = int(args.hidden_size or profile.hidden_size)
    top_k = int(args.top_k or profile.top_k)
    num_experts = int(args.num_experts or profile.num_experts)
    quant = args.quant or profile.quant
    if num_experts % ep_size:
        raise ValueError(
            f"num_experts ({num_experts}) must be divisible by EP size ({ep_size})"
        )
    batch_sizes = local_batch_sizes(args)
    max_num_tokens_per_rank = int(args.max_num_tokens_per_rank or max(batch_sizes))
    if max_num_tokens_per_rank < max(batch_sizes):
        raise ValueError(
            "--max_num_tokens_per_rank must cover the largest scanned batch"
        )

    backend_configs = _backend_configs(args)
    # A backend runs only if every rank can run it; otherwise collectives would diverge.
    available = set(available_communication_backends())
    runnable = [
        name
        for name in backend_configs
        if all(allgather(backend_configs[name]().backend_name in available))
    ]
    backends = [name for name in runnable if args.backend in (None, name)]
    if args.backend is not None and not backends:
        print_rank0(
            f"[bench_moe_ep_comm] {args.backend} is not available on every rank; "
            f"available on all ranks: {runnable}"
        )

    metadata: Dict[str, Any] = {
        "bench": "bench_moe_ep_comm",
        "profile": args.profile,
        "backends": backends,
        "ep_size": ep_size,
        "hidden_size": hidden_size,
        "top_k": top_k,
        "num_experts": num_experts,
        "experts_per_rank": num_experts // ep_size,
        "quant": quant,
        "local_batch_size": batch_sizes,
        "max_num_tokens_per_rank": max_num_tokens_per_rank,
        "perfect_router": bool(args.perfect_router),
        "use_low_precision_combine": bool(args.use_low_precision_combine),
        "cft": args.cft,
        "lamport": args.lamport,
        "random_seed": int(args.random_seed),
        "device": torch.cuda.get_device_name(device),
        "cuda_graph": not args.no_cuda_graph,
        "cupti_enabled": cupti_ctx is not None,
        "warning": cupti_warning,
    }
    print_rank0(json.dumps(metadata, indent=2))

    torch.manual_seed(int(args.random_seed) + rank)
    torch.cuda.manual_seed_all(int(args.random_seed) + rank)
    bootstrap = BootstrapConfig(world_size=ep_size, rank=rank, device=local_rank)
    dispatch_format = DISPATCH_FORMATS[quant]
    params = MoEEpCommParams(
        num_experts=num_experts,
        top_k=top_k,
        max_tokens_per_rank=max_num_tokens_per_rank,
        hidden_size=hidden_size,
        dispatch_format=None
        if dispatch_format is None
        else QuantFormat[dispatch_format],
    )
    results: List[Dict[str, Any]] = []

    for backend_name in backends:
        comm, error = None, None
        try:
            comm = create_communication(
                bootstrap, params, backend_configs[backend_name]()
            )
        except (RuntimeError, ValueError, NotImplementedError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        error = first_error(error)
        if error is not None:
            print_rank0(f"[bench_moe_ep_comm] Skipping {backend_name}: {error}")
            if comm is not None:
                comm.destroy()
            continue
        use_cuda_graph = not args.no_cuda_graph and comm.supports_cuda_graph
        if not args.no_cuda_graph and not use_cuda_graph:
            print_rank0(
                f"[bench_moe_ep_comm] {backend_name} cannot be captured; timing it eagerly."
            )

        for local_num_tokens in batch_sizes:
            step_max_tokens = max(allgather(int(local_num_tokens)))
            if step_max_tokens > comm.params.max_tokens_per_rank:
                print_rank0(
                    f"[bench_moe_ep_comm] Skipping {backend_name} @ "
                    f"local_batch_size={local_num_tokens}: exceeds "
                    f"max_tokens_per_rank={comm.params.max_tokens_per_rank}."
                )
                continue
            hidden_states, hidden_states_scale, topk_ids, topk_weights = _make_inputs(
                local_num_tokens,
                hidden_size,
                top_k,
                num_experts,
                ep_size,
                quant,
                bool(args.perfect_router),
                device,
            )
            timings, error = None, None
            try:
                timings = _time_dispatch_and_combine(
                    comm,
                    hidden_states=hidden_states,
                    hidden_states_scale=hidden_states_scale,
                    topk_ids=topk_ids,
                    topk_weights=topk_weights,
                    step_max_tokens=step_max_tokens,
                    hidden_size=hidden_size,
                    warmup=int(args.warmup),
                    iters=int(args.iters),
                    use_cuda_graph=use_cuda_graph,
                    cupti_ctx=cupti_ctx,
                )
            except (NotImplementedError, ValueError) as exc:
                # Input-dependent rejections (e.g. an unsupported payload) are
                # raised before any collective, identically on every rank.
                error = f"{type(exc).__name__}: {exc}"
            error = first_error(error)
            if error is not None:
                print_rank0(
                    f"[bench_moe_ep_comm] Skipping {backend_name} @ "
                    f"local_batch_size={local_num_tokens}: {error}"
                )
                break
            dispatch_us, combine_us, stats = timings

            cupti_errors = first_error(stats.get("cupti_error"))
            if cupti_errors is not None:
                warning = warn_kernel_span_unavailable(
                    f"{backend_name} @ local_batch_size={local_num_tokens}: {cupti_errors}",
                    _TAG,
                )
                metadata["warning"] = (
                    f"{metadata['warning']}\n{warning}"
                    if metadata["warning"]
                    else warning
                )
            elif cupti_ctx is not None:
                # Every rank reports the same timing source for this measurement.
                dispatch_us = stats["dispatch_us_kernel_span"]
                combine_us = stats["combine_us_kernel_span"]

            output: Dict[str, Any] = {
                "backend": backend_name,
                "local_batch_size": int(local_num_tokens),
                "cuda_graph": use_cuda_graph,
                "dispatch_us": gather_per_rank(dispatch_us, bool(args.iter_stats)),
                "combine_us": gather_per_rank(combine_us, bool(args.iter_stats)),
            }
            if args.kernel_breakdown:
                output.update(
                    gather_kernel_breakdown(
                        stats,
                        ("dispatch_kernels", "combine_kernels", "other_kernels"),
                        bool(args.iter_stats),
                    )
                )
            if rank == 0:
                print(json.dumps(output, indent=2), flush=True)
                results.append(output)

        sync()
        comm.destroy()

    if rank == 0 and args.output_file and results:
        output_dir = os.path.dirname(args.output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output_file, "w") as f:
            json.dump({"benchmark_metadata": metadata, "results": results}, f, indent=2)
        print(f"Report written to {args.output_file}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
