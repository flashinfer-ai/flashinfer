"""Benchmark MoE expert-parallel dispatch/combine of any MoEEpCommunication backend.

Only the communication is timed: ``MoEEpCommunication.dispatch()`` and
``MoEEpCommunication.combine()``. Input quantization and the simulated expert
output happen outside the timed phases.

dispatch_us and combine_us report CUPTI kernel spans (first kernel start to
last kernel end of each phase), using CUDA graph replay by default or eager
execution with --no_cuda_graph. Backends that cannot be captured run eagerly.
If CUPTI is unavailable, timing falls back to CUDA events and
benchmark_metadata.warning records the reason. --kernel_breakdown additionally
reports per-kernel statistics.

Launch on one node, for example:

    # Sweep local batch sizes 1..1024 (powers of 2) on EP8.
    torchrun --standalone --nproc-per-node=8 benchmarks/comm/bench_moe_ep_comm.py \\
        --backend nvlink_one_sided --profile deepseek_v3 -b 1 -e 1024 -f 2

    # Balanced routing, per-kernel breakdown and per-iteration stats, JSON report.
    torchrun --standalone --nproc-per-node=8 benchmarks/comm/bench_moe_ep_comm.py \\
        --backend nvlink_one_sided --profile deepseek_v4_pro --perfect_router \\
        --kernel_breakdown --iter_stats -b 1 -e 1024 -f 2 --output_file out.json

Omit --backend to benchmark every backend available on all ranks. Multi-node
runs use the usual torchrun rendezvous flags instead of --standalone.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

FP8_E4M3_MAX = 448.0
FP4_E2M1_MAX = 6.0
QUANT_FORMATS = ("bf16", "fp8", "mxfp8", "nvfp4")


@dataclass(frozen=True)
class Profile:
    name: str
    hidden_size: int
    top_k: int
    num_experts: int
    # Format of the dispatched activations (see QUANT_FORMATS).
    quant: str


PROFILES: Dict[str, Profile] = {
    "gpt_oss": Profile(
        "gpt_oss", hidden_size=2880, top_k=4, num_experts=128, quant="mxfp8"
    ),
    # FP8 block-scale MoE dispatches BF16 activations and quantizes after
    # dispatch, so the communication payload is BF16.
    "deepseek_v3": Profile(
        "deepseek_v3", hidden_size=7168, top_k=8, num_experts=256, quant="bf16"
    ),
    "deepseek_v4_flash": Profile(
        "deepseek_v4_flash", hidden_size=4096, top_k=6, num_experts=256, quant="mxfp8"
    ),
    "deepseek_v4_pro": Profile(
        "deepseek_v4_pro", hidden_size=7168, top_k=6, num_experts=384, quant="mxfp8"
    ),
    # All-to-all exchanges latent MoE activations, not the model's 7168-wide states.
    "kimi_k3": Profile(
        "kimi_k3", hidden_size=3584, top_k=16, num_experts=896, quant="mxfp8"
    ),
    "qwen3p8_2p4t_a95b": Profile(
        "qwen3p8_2p4t_a95b", hidden_size=8192, top_k=10, num_experts=512, quant="bf16"
    ),
}


def _backend_configs(args: argparse.Namespace) -> Dict[str, Callable[[], Any]]:
    """Benchmarkable backends: name -> factory of its MoEEpCommunication config."""
    from flashinfer.moe_ep import (
        NCCLEPConfig,
        NVLinkOneSidedConfig,
        NVLinkTwoSidedConfig,
    )

    low_precision = bool(args.use_low_precision_combine)
    return {
        "nvlink_one_sided": lambda: NVLinkOneSidedConfig(
            kernel="trtllm", use_low_precision_combine=low_precision
        ),
        "nvlink_one_sided_cake": lambda: NVLinkOneSidedConfig(
            kernel="cake", use_low_precision_combine=low_precision
        ),
        "nvlink_two_sided": NVLinkTwoSidedConfig,
        "nccl_ep": NCCLEPConfig,
    }


# ----------------------------------------------------------------------------
# Distributed helpers
# ----------------------------------------------------------------------------


def _rank() -> int:
    return dist.get_rank()


def _allgather(obj: Any) -> List[Any]:
    out: List[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(out, obj)
    return out


def _print_rank0(msg: str) -> None:
    if _rank() == 0:
        print(msg, flush=True)


def _sync() -> None:
    torch.cuda.synchronize()
    dist.barrier()


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
    experts_per_rank = num_experts // ep_size
    if perfect_router:
        # Balance ranks and experts: walk the flattened (token, k) slots as
        # rank r expert 0, rank r+1 expert 0, ..., then rank r expert 1, ...
        slots = torch.arange(local_num_tokens * top_k, device=device) + _rank()
        target_rank = slots % ep_size
        local_expert = (slots // ep_size) % experts_per_rank
        topk_ids = (target_rank * experts_per_rank + local_expert).view(
            local_num_tokens, top_k
        )
    else:
        topk_ids = (
            torch.rand(local_num_tokens, num_experts, device=device)
            .topk(top_k, -1)
            .indices
        )
    topk_weights = torch.rand(
        local_num_tokens, top_k, dtype=torch.float32, device=device
    )
    return hidden_states, hidden_states_scale, topk_ids.to(torch.int32), topk_weights


# ----------------------------------------------------------------------------
# CUPTI kernel spans
# ----------------------------------------------------------------------------


def _demangle_names(names: List[str]) -> Dict[str, str]:
    try:
        import cxxfilt
    except ImportError:
        return {name: name for name in names}
    demangled = {}
    for name in names:
        try:
            demangled[name] = cxxfilt.demangle(name)
        except cxxfilt.InvalidName:
            demangled[name] = name
    return demangled


def _build_kernel_stats_cupti(
    cupti_kernels: List[Tuple[str, int, int]],
    cupti_events: List[Tuple[int, int]],
    phase_event_ids: List[Tuple[int, int, int, int]],
) -> Dict[str, Any]:
    """Attribute kernels to the dispatch/combine event windows of each iteration."""
    expected_ids = {event_id for iteration in phase_event_ids for event_id in iteration}
    if len(expected_ids) != 4 * len(phase_event_ids):
        raise RuntimeError(
            "Each benchmark timing event must have a distinct CUPTI event ID."
        )
    event_timestamps: Dict[int, int] = {}
    for event_id, timestamp in cupti_events:
        if event_id not in expected_ids:
            continue
        if timestamp <= 0 or event_id in event_timestamps:
            raise RuntimeError(
                f"CUPTI returned an invalid or duplicate timing event: {event_id}"
            )
        event_timestamps[event_id] = timestamp
    missing_ids = expected_ids - event_timestamps.keys()
    if missing_ids:
        raise RuntimeError(
            f"CUPTI is missing {len(missing_ids)} of {len(expected_ids)} timing events. "
            "CUDA_EVENT tracking must be enabled before CUDA context creation."
        )
    if not cupti_kernels:
        raise RuntimeError("CUPTI captured no kernels for the timed run.")

    phase_windows = []
    for ids in phase_event_ids:
        d_start, d_end, c_start, c_end = (
            event_timestamps[event_id] for event_id in ids
        )
        if not d_start <= d_end <= c_start <= c_end:
            raise RuntimeError(
                "CUPTI timing events are not in dispatch/combine execution order."
            )
        if phase_windows and d_start < phase_windows[-1][3]:
            raise RuntimeError("CUPTI iteration timing windows overlap.")
        phase_windows.append((d_start, d_end, c_start, c_end))

    phase_bounds: Dict[str, List[Optional[Tuple[int, int]]]] = {
        phase: [None] * len(phase_windows) for phase in ("dispatch", "combine")
    }
    cupti_kernels.sort(key=lambda kernel: kernel[1])
    demangled_names = _demangle_names(list({name for name, _, _ in cupti_kernels}))
    kernel_times: Dict[str, Dict[str, List[float]]] = {
        "dispatch": {},
        "combine": {},
        "other": {},
    }

    for name, kernel_start, kernel_end in cupti_kernels:
        if kernel_start <= 0 or kernel_end <= kernel_start:
            raise RuntimeError(f"CUPTI returned invalid kernel timestamps for {name}.")
        category = "other"
        for iteration, (d_start, d_end, c_start, c_end) in enumerate(phase_windows):
            for phase, start, end in (
                ("dispatch", d_start, d_end),
                ("combine", c_start, c_end),
            ):
                if kernel_start >= start and kernel_end <= end:
                    category = phase
                    bounds = phase_bounds[phase][iteration]
                    phase_bounds[phase][iteration] = (
                        (min(bounds[0], kernel_start), max(bounds[1], kernel_end))
                        if bounds is not None
                        else (kernel_start, kernel_end)
                    )
                    break
                if kernel_start < end and kernel_end > start:
                    raise RuntimeError(
                        f"CUPTI kernel {name} crosses a {phase} timing boundary."
                    )
            if category != "other":
                break
        kernel_times[category].setdefault(demangled_names.get(name, name), []).append(
            (kernel_end - kernel_start) / 1e3
        )

    def _build(category: str) -> List[Dict[str, Any]]:
        result = [
            {"name": name, "count": len(times), "_times": times}
            for name, times in kernel_times[category].items()
        ]
        result.sort(
            key=lambda kernel: sum(kernel["_times"]) / len(kernel["_times"]),
            reverse=True,
        )
        return result

    spans = {}
    for phase, bounds in phase_bounds.items():
        if any(bound is None for bound in bounds):
            raise RuntimeError(
                f"CUPTI captured no {phase} kernels in one or more timed iterations."
            )
        # A span keeps inter-kernel gaps but counts overlapping PDL kernels once.
        spans[f"{phase}_us_kernel_span"] = [
            (bound[1] - bound[0]) / 1e3 for bound in bounds if bound is not None
        ]
    return {
        **spans,
        "dispatch_kernels": _build("dispatch"),
        "combine_kernels": _build("combine"),
        "other_kernels": _build("other"),
    }


def _init_cupti() -> Tuple[Any, List[Tuple[str, int, int]], List[Tuple[int, int]]]:
    """Enable kernel and CUDA-event activity tracing; call before CUDA context creation."""
    from cupti import cupti

    cupti_kernels: List[Tuple[str, int, int]] = []
    cupti_events: List[Tuple[int, int]] = []

    def _buf_requested() -> Tuple[int, int]:
        return 8 * 1024 * 1024, 0

    def _buf_completed(activities) -> None:
        for activity in activities:
            if activity.kind == cupti.ActivityKind.CONCURRENT_KERNEL:
                cupti_kernels.append((activity.name, activity.start, activity.end))
            elif activity.kind == cupti.ActivityKind.CUDA_EVENT:
                cupti_events.append((activity.event_id, activity.device_timestamp))

    enabled = []
    try:
        cupti.activity_register_callbacks(_buf_requested, _buf_completed)
        for kind in (
            cupti.ActivityKind.CONCURRENT_KERNEL,
            cupti.ActivityKind.CUDA_EVENT,
        ):
            cupti.activity_enable(kind)
            enabled.append(kind)
        cupti.activity_enable_cuda_event_device_timestamps(1)
    except (cupti.cuptiError, AttributeError) as exc:
        cleanup_errors = []
        for kind in reversed(enabled):
            try:
                cupti.activity_disable(kind)
            except cupti.cuptiError as cleanup_exc:
                cleanup_errors.append(str(cleanup_exc))
        raise RuntimeError(
            f"Cannot enable CUPTI kernel/event tracing: {exc}; cleanup errors: {cleanup_errors}"
        ) from exc
    return cupti, cupti_kernels, cupti_events


_CUPTI_FALLBACK_WARNING_SHOWN = False


def _warn_kernel_span_unavailable(reason: str) -> str:
    global _CUPTI_FALLBACK_WARNING_SHOWN
    warning = (
        "CUPTI kernel-span timing is unavailable. Falling back to CUDA-event timing, "
        "which may include non-kernel bubbles before the first kernel and after the last "
        "kernel of each phase. Kernel-span timing excludes these boundary bubbles and is "
        "generally more representative of communication-kernel time in end-to-end "
        f"workloads. Reason: {reason}"
    )
    if not _CUPTI_FALLBACK_WARNING_SHOWN:
        _print_rank0(f"[bench_moe_ep_comm] WARNING: {warning}")
        _CUPTI_FALLBACK_WARNING_SHOWN = True
    return warning


def _agree_on_cupti(
    local_ctx: Optional[Any], local_error: Optional[str]
) -> Tuple[Optional[Any], Optional[str]]:
    """Use CUPTI only if it initialized on every rank."""
    errors = _allgather(local_error)
    if not any(errors):
        return local_ctx, None
    reasons = [f"rank{rank}: {error}" for rank, error in enumerate(errors) if error]
    if local_ctx is not None:
        cupti = local_ctx[0]
        for kind in (
            cupti.ActivityKind.CUDA_EVENT,
            cupti.ActivityKind.CONCURRENT_KERNEL,
        ):
            try:
                cupti.activity_disable(kind)
            except cupti.cuptiError as exc:
                reasons.append(f"local CUPTI cleanup failed: {exc}")
    return None, _warn_kernel_span_unavailable("; ".join(reasons))


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
    l2_size = torch.cuda.get_device_properties(device).L2_cache_size
    l2_buffer = torch.empty(l2_size // 2, dtype=torch.int32, device=device)

    def _dispatch() -> torch.Tensor:
        received = comm.dispatch(
            hidden_states,
            topk_ids,
            topk_weights,
            hidden_states_scale=hidden_states_scale,
            max_tokens_per_rank=step_max_tokens,
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
    if use_cuda_graph:
        # cudaEventRecordExternal (0x1) keeps events recorded inside a CUDA graph
        # queryable with elapsed_time() after replay.
        cudart = ctypes.CDLL("libcudart.so")
        cudart.cudaEventRecordWithFlags.restype = ctypes.c_int
        cudart.cudaEventRecordWithFlags.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint,
        ]

    def _record(event: torch.cuda.Event) -> None:
        if not use_cuda_graph:
            event.record()
            return
        stream = torch.cuda.current_stream().cuda_stream
        ret = cudart.cudaEventRecordWithFlags(event.cuda_event, stream, 0x1)
        if ret != 0:
            raise RuntimeError(f"cudaEventRecordWithFlags failed with code {ret}")

    d_starts, d_ends, c_starts, c_ends = (
        [torch.cuda.Event(enable_timing=True) for _ in range(iters)] for _ in range(4)
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
            _record(d_starts[i])
            _dispatch()
            _record(d_ends[i])
            output = _expert_output()
            output.zero_()
            _record(c_starts[i])
            comm.combine(output)
            _record(c_ends[i])

    if use_cuda_graph:
        # One replay covers warmup and timed iterations, so no host gaps remain.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _run_iterations()

    if cupti_ctx is not None:
        cupti.activity_flush_all(0)
        cupti_kernels.clear()
        cupti_events.clear()
    _sync()
    if use_cuda_graph:
        graph.replay()
    else:
        _run_iterations()
    _sync()
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
            stats = _build_kernel_stats_cupti(
                cupti_kernels, cupti_events, phase_event_ids
            )
        except RuntimeError as exc:
            # Keep every rank on the same collective sequence even if one trace is incomplete.
            stats["cupti_error"] = str(exc)
    return dispatch_us, combine_us, stats


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------


def _compute_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "stdev": 0.0, "min": 0.0, "max": 0.0}
    s = sorted(values)
    n = len(s)
    mean = sum(s) / n
    return {
        "mean": mean,
        "median": s[n // 2],
        "stdev": (sum((x - mean) ** 2 for x in s) / n) ** 0.5,
        "min": s[0],
        "max": s[-1],
    }


def _summarize(times: List[float], iter_stats: bool) -> Any:
    if iter_stats:
        return _compute_stats(times)
    return sum(times) / len(times) if times else 0.0


def _gather_per_rank(times_us: List[float], iter_stats: bool) -> Dict[str, Any]:
    return {
        f"rank{i}": _summarize(times, iter_stats)
        for i, times in enumerate(_allgather(times_us))
    }


def _gather_kernel_breakdown(stats: Dict[str, Any], iter_stats: bool) -> Dict[str, Any]:
    categories = ("dispatch_kernels", "combine_kernels", "other_kernels")
    # Gather once per rank so differing kernel lists cannot desynchronize collectives.
    local = {
        category: {
            kernel["name"]: kernel.get("_times", [])
            for kernel in stats.get(category, [])
        }
        for category in categories
    }
    all_ranks = _allgather(local)
    merged: Dict[str, Any] = {}
    for category in categories:
        names: List[str] = []
        for rank_payload in all_ranks:
            names.extend(name for name in rank_payload[category] if name not in names)
        merged[category] = []
        for name in names:
            per_rank_times = [
                rank_payload[category].get(name, []) for rank_payload in all_ranks
            ]
            merged[category].append(
                {
                    "name": name,
                    "count": max(len(times) for times in per_rank_times),
                    "per_rank": {
                        f"rank{i}": _summarize(times, iter_stats)
                        for i, times in enumerate(per_rank_times)
                    },
                }
            )
    return merged


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
            "nvlink_one_sided_cake",
            "nvlink_two_sided",
            "nccl_ep",
        ],
        help="Backend to benchmark (default: every backend available on all ranks).",
    )
    parser.add_argument(
        "--profile",
        default="deepseek_v3",
        choices=sorted(PROFILES),
        help="Model profile supplying hidden_size, top_k, num_experts and quant.",
    )
    parser.add_argument(
        "--hidden_size", type=int, default=None, help="Override the profile."
    )
    parser.add_argument("--top_k", type=int, default=None, help="Override the profile.")
    parser.add_argument(
        "--num_experts", type=int, default=None, help="Override the profile."
    )
    parser.add_argument(
        "--quant",
        default=None,
        choices=QUANT_FORMATS,
        help="Override the profile's dispatched activation format.",
    )
    # Size sweep, nccl-tests style with local batch sizes (tokens) instead of bytes.
    parser.add_argument(
        "-b", "--minbatch", type=int, default=640, help="Smallest local batch."
    )
    parser.add_argument(
        "-e", "--maxbatch", type=int, default=None, help="Largest local batch."
    )
    parser.add_argument(
        "-i", "--stepbatch", type=int, default=None, help="Additive step (128)."
    )
    parser.add_argument(
        "-f", "--stepfactor", type=float, default=None, help="Multiplicative step."
    )
    parser.add_argument("--iters", type=int, default=200, help="Timed iterations.")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations.")
    parser.add_argument(
        "--max_num_tokens_per_rank",
        type=int,
        default=None,
        help="Workspace capacity per rank (default: the largest scanned local batch).",
    )
    parser.add_argument(
        "--kernel_breakdown", action="store_true", help="Report per-kernel statistics."
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
        "--no_cuda_graph",
        action="store_true",
        help="Run eagerly instead of replaying one CUDA graph.",
    )
    return parser.parse_args()


def _local_batch_sizes(args: argparse.Namespace) -> List[int]:
    if args.stepbatch is not None and args.stepfactor is not None:
        raise ValueError("Use only one of -i/--stepbatch and -f/--stepfactor.")
    minb = int(args.minbatch if args.minbatch is not None else args.maxbatch)
    maxb = int(args.maxbatch if args.maxbatch is not None else minb)
    if minb <= 0 or maxb < minb:
        raise ValueError("Need 0 < --minbatch <= --maxbatch.")
    if args.stepfactor is not None:
        if args.stepfactor <= 1.0:
            raise ValueError("--stepfactor must be > 1.0")
        sizes: List[int] = []
        current = float(minb)
        while int(current) <= maxb:
            if not sizes or sizes[-1] != int(current):
                sizes.append(int(current))
            current *= args.stepfactor
        return sizes
    step = int(args.stepbatch or 128)
    if step <= 0:
        raise ValueError("--stepbatch must be > 0")
    return list(range(minb, maxb + 1, step))


def _first_error(local_error: Optional[str]) -> Optional[str]:
    errors = _allgather(local_error)
    reasons = [f"rank{rank}: {error}" for rank, error in enumerate(errors) if error]
    return "; ".join(reasons) or None


def main() -> None:
    args = parse_args()

    # CUDA_EVENT activity records need tracing enabled before the CUDA
    # context exists, i.e. before the device is bound.
    cupti_ctx, cupti_error = None, None
    try:
        cupti_ctx = _init_cupti()
    except (ImportError, OSError, RuntimeError) as exc:
        cupti_error = f"{type(exc).__name__}: {exc}"

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", device_id=device)
    rank, ep_size = dist.get_rank(), dist.get_world_size()
    cupti_ctx, cupti_warning = _agree_on_cupti(cupti_ctx, cupti_error)

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
    local_batch_sizes = _local_batch_sizes(args)
    max_num_tokens_per_rank = int(
        args.max_num_tokens_per_rank or max(local_batch_sizes)
    )
    if max_num_tokens_per_rank < max(local_batch_sizes):
        raise ValueError(
            "--max_num_tokens_per_rank must cover the largest scanned batch"
        )

    backend_configs = _backend_configs(args)
    # A backend runs only if every rank can run it; otherwise collectives would diverge.
    available = set(available_communication_backends())
    runnable = [
        name
        for name in backend_configs
        if all(_allgather(backend_configs[name]().backend_name in available))
    ]
    backends = [name for name in runnable if args.backend in (None, name)]
    if args.backend is not None and not backends:
        _print_rank0(
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
        "local_batch_size": local_batch_sizes,
        "max_num_tokens_per_rank": max_num_tokens_per_rank,
        "perfect_router": bool(args.perfect_router),
        "use_low_precision_combine": bool(args.use_low_precision_combine),
        "random_seed": int(args.random_seed),
        "device": torch.cuda.get_device_name(device),
        "cuda_graph": not args.no_cuda_graph,
        "cupti_enabled": cupti_ctx is not None,
        "warning": cupti_warning,
    }
    _print_rank0(json.dumps(metadata, indent=2))

    torch.manual_seed(int(args.random_seed) + rank)
    torch.cuda.manual_seed_all(int(args.random_seed) + rank)
    bootstrap = BootstrapConfig(world_size=ep_size, rank=rank, device=local_rank)
    params = MoEEpCommParams(
        num_experts=num_experts,
        top_k=top_k,
        max_tokens_per_rank=max_num_tokens_per_rank,
        hidden_size=hidden_size,
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
        error = _first_error(error)
        if error is not None:
            _print_rank0(f"[bench_moe_ep_comm] Skipping {backend_name}: {error}")
            if comm is not None:
                comm.destroy()
            continue
        use_cuda_graph = not args.no_cuda_graph and comm.supports_cuda_graph
        if not args.no_cuda_graph and not use_cuda_graph:
            _print_rank0(
                f"[bench_moe_ep_comm] {backend_name} cannot be captured; timing it eagerly."
            )

        for local_num_tokens in local_batch_sizes:
            step_max_tokens = max(_allgather(int(local_num_tokens)))
            if not comm.is_workload_feasible(step_max_tokens):
                _print_rank0(
                    f"[bench_moe_ep_comm] Skipping {backend_name} @ "
                    f"local_batch_size={local_num_tokens}: workload not feasible."
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
            error = _first_error(error)
            if error is not None:
                _print_rank0(
                    f"[bench_moe_ep_comm] Skipping {backend_name} @ "
                    f"local_batch_size={local_num_tokens}: {error}"
                )
                break
            dispatch_us, combine_us, stats = timings

            cupti_errors = _first_error(stats.get("cupti_error"))
            if cupti_errors is not None:
                warning = _warn_kernel_span_unavailable(
                    f"{backend_name} @ local_batch_size={local_num_tokens}: {cupti_errors}"
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
                "dispatch_us": _gather_per_rank(dispatch_us, bool(args.iter_stats)),
                "combine_us": _gather_per_rank(combine_us, bool(args.iter_stats)),
            }
            if args.kernel_breakdown:
                output.update(_gather_kernel_breakdown(stats, bool(args.iter_stats)))
            if rank == 0:
                print(json.dumps(output, indent=2), flush=True)
                results.append(output)

        _sync()
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
