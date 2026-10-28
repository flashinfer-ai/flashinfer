"""Shared helpers of the MoE expert-parallel benchmarks.

``bench_moe_ep.py`` (one whole MoE layer) and ``bench_moe_ep_comm.py``
(dispatch and combine only) share their model profiles, routing, batch-size
sweep, CUPTI kernel-span timing and per-rank reporting.

Timing model. Each timed iteration is split into named phases that run in a
fixed order, and the benchmark records a pair of CUDA events around every
phase. With CUPTI, a phase's time is its kernel span: from the first kernel
that starts inside the phase's event window to the last kernel that ends
inside it. The events only delimit the windows; kernels are never classified
by name.
"""

from __future__ import annotations

import argparse
import importlib.util
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class Profile:
    """Routed-expert shape of one model.

    ``quant`` is the format of the dispatched activations; ``weight_quant``
    and ``intermediate_size`` only matter when the experts run.
    """

    name: str
    hidden_size: int
    top_k: int
    num_experts: int
    quant: str
    intermediate_size: int
    weight_quant: str


PROFILES: Dict[str, Profile] = {
    "gpt_oss": Profile(
        "gpt_oss",
        hidden_size=2880,
        top_k=4,
        num_experts=128,
        quant="mxfp8",
        intermediate_size=2880,
        weight_quant="mxfp4",
    ),
    # FP8 block-scale MoE dispatches BF16 activations and quantizes after
    # dispatch, so the communication payload is BF16.
    "deepseek_v3": Profile(
        "deepseek_v3",
        hidden_size=7168,
        top_k=8,
        num_experts=256,
        quant="bf16",
        intermediate_size=2048,
        weight_quant="bf16",
    ),
    "deepseek_v4_flash": Profile(
        "deepseek_v4_flash",
        hidden_size=4096,
        top_k=6,
        num_experts=256,
        quant="mxfp8",
        intermediate_size=2048,
        weight_quant="mxfp4",
    ),
    "deepseek_v4_pro": Profile(
        "deepseek_v4_pro",
        hidden_size=7168,
        top_k=6,
        num_experts=384,
        quant="mxfp8",
        intermediate_size=3072,
        weight_quant="mxfp4",
    ),
    # All-to-all exchanges latent MoE activations, not the model's 7168-wide states.
    "kimi_k3": Profile(
        "kimi_k3",
        hidden_size=3584,
        top_k=16,
        num_experts=896,
        quant="mxfp8",
        intermediate_size=3072,
        weight_quant="mxfp4",
    ),
    "qwen3p8_2p4t_a95b": Profile(
        "qwen3p8_2p4t_a95b",
        hidden_size=8192,
        top_k=10,
        num_experts=512,
        quant="bf16",
        intermediate_size=2048,
        weight_quant="bf16",
    ),
}


def add_profile_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--profile",
        default="deepseek_v3",
        choices=sorted(PROFILES),
        help="Model profile supplying the expert shape and formats.",
    )
    parser.add_argument(
        "--hidden_size", type=int, default=None, help="Override the profile."
    )
    parser.add_argument("--top_k", type=int, default=None, help="Override the profile.")
    parser.add_argument(
        "--num_experts", type=int, default=None, help="Override the profile."
    )


# ----------------------------------------------------------------------------
# Distributed helpers
# ----------------------------------------------------------------------------


def rank() -> int:
    return dist.get_rank()


def allgather(obj: Any) -> List[Any]:
    out: List[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(out, obj)
    return out


def print_rank0(msg: str) -> None:
    if rank() == 0:
        print(msg, flush=True)


def sync() -> None:
    torch.cuda.synchronize()
    dist.barrier()


def first_error(local_error: Optional[str]) -> Optional[str]:
    """Every rank's error, joined; None when no rank failed."""
    errors = allgather(local_error)
    reasons = [f"rank{r}: {error}" for r, error in enumerate(errors) if error]
    return "; ".join(reasons) or None


# ----------------------------------------------------------------------------
# Workload
# ----------------------------------------------------------------------------


def add_sweep_args(parser: argparse.ArgumentParser) -> None:
    # nccl-tests style, with local batch sizes (tokens) instead of bytes.
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


def local_batch_sizes(args: argparse.Namespace) -> List[int]:
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


def make_routing(
    local_num_tokens: int,
    top_k: int,
    num_experts: int,
    ep_size: int,
    perfect_router: bool,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return int32 ``topk_ids`` and float32 ``topk_weights``, ``[tokens, top_k]``."""
    experts_per_rank = num_experts // ep_size
    if perfect_router:
        # Balance ranks and experts: walk the flattened (token, k) slots as
        # rank r expert 0, rank r+1 expert 0, ..., then rank r expert 1, ...
        slots = torch.arange(local_num_tokens * top_k, device=device) + rank()
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
    return topk_ids.to(torch.int32), topk_weights


def l2_flush_buffer(device: torch.device) -> torch.Tensor:
    """A buffer twice the L2 size; zeroing it evicts the previous iteration."""
    l2_size = torch.cuda.get_device_properties(device).L2_cache_size
    return torch.empty(l2_size // 2, dtype=torch.int32, device=device)


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


def warn_if_kernel_names_mangled(tag: str) -> None:
    """Warn that ``--kernel_breakdown`` cannot demangle C++ kernel names
    without the optional ``cxxfilt`` package."""
    if importlib.util.find_spec("cxxfilt") is None:
        print_rank0(
            f"[{tag}] WARNING: the cxxfilt package is not installed, so "
            "--kernel_breakdown reports C++ kernels by their mangled names; "
            "`pip install -r benchmarks/moe_ep/requirements.txt` to demangle them."
        )


def kernel_spans_cupti(
    cupti_kernels: List[Tuple[str, int, int]],
    cupti_events: List[Tuple[int, int]],
    phases: Sequence[str],
    window_event_ids: List[Tuple[int, ...]],
    optional_phases: Sequence[str] = (),
) -> Dict[str, Any]:
    """Attribute kernels to the phase windows of each timed iteration.

    ``window_event_ids`` holds, per iteration, the CUPTI ids of the start and
    end events of every phase in ``phases`` order. Returns per-iteration spans
    ``<phase>_us_kernel_span`` and ``total_us_kernel_span`` (first kernel of
    the first phase to last kernel of the last one), plus the per-kernel times
    of each phase (``<phase>_kernels``) and of kernels outside every window
    (``other_kernels``). A phase in ``optional_phases`` may launch no kernels;
    its span is then 0.
    """
    num_events = 2 * len(phases)
    expected_ids = {
        event_id for iteration in window_event_ids for event_id in iteration
    }
    if len(expected_ids) != num_events * len(window_event_ids):
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

    windows: List[List[Tuple[int, int]]] = []
    for ids in window_event_ids:
        stamps = [event_timestamps[event_id] for event_id in ids]
        if any(a > b for a, b in zip(stamps, stamps[1:], strict=False)):
            raise RuntimeError(
                f"CUPTI timing events are not in {'/'.join(phases)} execution order."
            )
        if windows and stamps[0] < windows[-1][-1][1]:
            raise RuntimeError("CUPTI iteration timing windows overlap.")
        windows.append(list(zip(stamps[0::2], stamps[1::2], strict=True)))

    bounds: Dict[str, List[Optional[Tuple[int, int]]]] = {
        phase: [None] * len(windows) for phase in phases
    }
    cupti_kernels.sort(key=lambda kernel: kernel[1])
    demangled_names = _demangle_names(list({name for name, _, _ in cupti_kernels}))
    kernel_times: Dict[str, Dict[str, List[float]]] = {
        category: {} for category in (*phases, "other")
    }

    for name, kernel_start, kernel_end in cupti_kernels:
        if kernel_start <= 0 or kernel_end <= kernel_start:
            raise RuntimeError(f"CUPTI returned invalid kernel timestamps for {name}.")
        category = "other"
        for iteration, iteration_windows in enumerate(windows):
            for phase, (start, end) in zip(phases, iteration_windows, strict=True):
                if kernel_start >= start and kernel_end <= end:
                    category = phase
                    bound = bounds[phase][iteration]
                    bounds[phase][iteration] = (
                        (min(bound[0], kernel_start), max(bound[1], kernel_end))
                        if bound is not None
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

    def _kernels(category: str) -> List[Dict[str, Any]]:
        result = [
            {"name": name, "count": len(times), "_times": times}
            for name, times in kernel_times[category].items()
        ]
        result.sort(
            key=lambda kernel: sum(kernel["_times"]) / len(kernel["_times"]),
            reverse=True,
        )
        return result

    stats: Dict[str, Any] = {}
    for phase in phases:
        if phase not in optional_phases and any(b is None for b in bounds[phase]):
            raise RuntimeError(
                f"CUPTI captured no {phase} kernels in one or more timed iterations."
            )
        # A span keeps inter-kernel gaps but counts overlapping PDL kernels once.
        stats[f"{phase}_us_kernel_span"] = [
            (b[1] - b[0]) / 1e3 if b is not None else 0.0 for b in bounds[phase]
        ]
        stats[f"{phase}_kernels"] = _kernels(phase)
    stats["other_kernels"] = _kernels("other")
    totals = []
    for iteration in range(len(windows)):
        present = [bounds[p][iteration] for p in phases if bounds[p][iteration]]
        totals.append((max(b[1] for b in present) - min(b[0] for b in present)) / 1e3)
    stats["total_us_kernel_span"] = totals
    return stats


def init_cupti() -> Tuple[Any, List[Tuple[str, int, int]], List[Tuple[int, int]]]:
    """Enable kernel and CUDA-event activity tracing; call before CUDA context creation."""
    from cupti import cupti

    # Phase attribution needs CUDA-event IDs, added in cupti-python 13.4.
    if not hasattr(cupti, "get_cuda_event_id"):
        raise RuntimeError(
            "cupti-python lacks get_cuda_event_id; install cupti-python>=13.4"
        )

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


def warn_kernel_span_unavailable(reason: str, tag: str) -> str:
    global _CUPTI_FALLBACK_WARNING_SHOWN
    warning = (
        "CUPTI kernel-span timing is unavailable. Falling back to CUDA-event timing, "
        "which may include non-kernel bubbles before the first kernel and after the last "
        "kernel of each phase. Kernel-span timing excludes these boundary bubbles and is "
        "generally more representative of kernel time in end-to-end workloads. "
        f"Reason: {reason}"
    )
    if not _CUPTI_FALLBACK_WARNING_SHOWN:
        print_rank0(f"[{tag}] WARNING: {warning}")
        _CUPTI_FALLBACK_WARNING_SHOWN = True
    return warning


def agree_on_cupti(
    local_ctx: Optional[Any], local_error: Optional[str], tag: str
) -> Tuple[Optional[Any], Optional[str]]:
    """Use CUPTI only if it initialized on every rank."""
    errors = allgather(local_error)
    if not any(errors):
        return local_ctx, None
    reasons = [f"rank{r}: {error}" for r, error in enumerate(errors) if error]
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
    return None, warn_kernel_span_unavailable("; ".join(reasons), tag)


def start_cupti() -> Tuple[Optional[Any], Optional[str]]:
    """Try to enable CUPTI tracing; returns ``(context, error)``.

    Must run before the CUDA context exists, i.e. before the device is bound:
    CUDA_EVENT activity records need tracing enabled first.
    """
    try:
        return init_cupti(), None
    except (ImportError, OSError, RuntimeError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------


def compute_stats(values: List[float]) -> Dict[str, float]:
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


def summarize(times: List[float], iter_stats: bool) -> Any:
    if iter_stats:
        return compute_stats(times)
    return sum(times) / len(times) if times else 0.0


def gather_per_rank(times_us: List[float], iter_stats: bool) -> Dict[str, Any]:
    return {
        f"rank{i}": summarize(times, iter_stats)
        for i, times in enumerate(allgather(times_us))
    }


def gather_kernel_breakdown(
    stats: Dict[str, Any], categories: Sequence[str], iter_stats: bool
) -> Dict[str, Any]:
    """Per-kernel times of every rank for each ``<category>`` in ``stats``."""
    # Gather once per rank so differing kernel lists cannot desynchronize collectives.
    local = {
        category: {
            kernel["name"]: kernel.get("_times", [])
            for kernel in stats.get(category, [])
        }
        for category in categories
    }
    all_ranks = allgather(local)
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
                        f"rank{i}": summarize(times, iter_stats)
                        for i, times in enumerate(per_rank_times)
                    },
                }
            )
    return merged


def rank_mean(per_rank: Dict[str, Any]) -> float:
    """Average over ranks of a ``gather_per_rank`` result."""
    values = [v["mean"] if isinstance(v, dict) else v for v in per_rank.values()]
    return sum(values) / len(values)
