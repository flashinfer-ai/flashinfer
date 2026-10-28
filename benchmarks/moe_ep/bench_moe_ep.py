"""Benchmark one MoE expert-parallel layer: dispatch, expert computation and combine.

Times ``MoEEpSplitLayer.forward()`` for one model profile, one communication
backend and one expert-compute backend over a sweep of local batch sizes. The
layer runs four stages (``MoEEpSplitLayer.STAGES``): ``quantize`` turns the
BF16 hidden states into the dispatch payload, ``dispatch`` and ``combine`` run
the all-to-all, and ``compute`` runs the routed experts.

What is reported has three levels of detail, each chosen on its own:

- ``--timing_scope`` picks the span of the forward. ``dispatch_compute_combine``
  (default) runs from the first dispatch kernel to the last combine kernel, the
  part a fused dispatch-experts-combine kernel replaces, so it leaves out the
  quantization before dispatch; ``forward`` runs from the first quantize kernel
  to the last combine kernel.
- ``--stage_breakdown`` reports the stages inside that span instead of the span:
  ``dispatch``, ``compute`` and ``combine``, and also ``quantize`` for the
  ``forward`` scope.
- ``--kernel_breakdown`` adds the time of every kernel inside each reported span
  or stage, C++ kernel names demangled by the ``cxxfilt`` package.

A time is a CUPTI kernel span: first kernel start to last kernel end inside a
window delimited by two CUDA events. The events, recorded by the benchmark
around the forward and by the layer's ``stage_hook`` at stage boundaries, only
delimit the windows; without cupti-python>=13.4 the times fall back to the
events themselves and benchmark_metadata.warning records the reason. Every
event node in a replayed graph delays the kernel after it by several
microseconds, so events sit only where the chosen level needs them: the
default scope has one boundary inside the forward, where dispatch starts, and
``--stage_breakdown`` puts one at every stage boundary. Those also stop a
kernel of one stage from overlapping the next through programmatic dependent
launch, so the stages add up to more than the span they split, which is why
the span itself is not reported with them. ``--kernel_breakdown`` reads the
same trace and changes no time.

To time the all-to-all alone, use ``bench_moe_ep_comm.py``, which reports the
same dispatch and combine spans. ``benchmarks/moe_ep/requirements.txt`` lists
the optional packages of both (``cupti-python``, ``cxxfilt``).

Warmup and timed iterations are captured in one CUDA graph and replayed once
(``--no_cuda_graph`` runs them eagerly); every iteration flushes L2 first. One
layer, whose workspace holds the largest batch (``--max_num_tokens_per_rank``),
serves the whole sweep: each step passes its largest local batch over the ranks
as ``MoEEpTensors.runtime_max_tokens_per_rank``, so the all-to-all and the
experts run with that token count, as ``bench_moe_ep_comm.py`` runs the
communication. The first, untimed forward of every batch size profiles the
expert kernels' tactics with ``flashinfer.autotune`` (``--no_autotune`` keeps
their fallback tactics; ``--autotune_cache`` reuses the profiles across runs).

Launch on one node, for example:

    # DeepSeek-V4-Pro experts (MXFP4 weights, MXFP8 activations) behind the
    # one-sided all-to-all on its fence transport, local batches 1..1024.
    torchrun --standalone --nproc-per-node=8 benchmarks/moe_ep/bench_moe_ep.py \\
        --comm nvlink_one_sided --cft off --compute trtllm \\
        --profile deepseek_v4_pro --perfect_router -b 1 -e 1024 -f 2 \\
        --kernel_breakdown --output_file out.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
    rank_mean,
    start_cupti,
    sync,
    warn_if_kernel_names_mangled,
    warn_kernel_span_unavailable,
)

_TAG = "bench_moe_ep"

# The flashinfer.fused_moe.QuantFormat of each --quant / --weight_quant choice.
QUANT_FORMATS = {
    "bf16": "BF16",
    "fp8_block_scale": "DeepSeekFp8",
    "mxfp8": "MXFP8",
    "mxfp4": "MXFP4",
    "nvfp4": "NVFP4",
}
ACTIVATION_FORMATS = ("bf16", "fp8_block_scale", "mxfp8", "nvfp4")
COMM_BACKENDS = ("nvlink_one_sided", "cake", "nvlink_two_sided")
COMPUTE_BACKENDS = ("trtllm", "cutedsl")
# NVLinkOneSidedConfig.cft for each --cft choice.
_CFT_MODES = {"auto": None, "on": True, "off": False}
TIMING_SCOPES = ("dispatch_compute_combine", "forward")
# The stage a fused dispatch-experts-combine kernel starts at.
_DISPATCH_STAGE = "dispatch"


def _comm_config(args: argparse.Namespace):
    from flashinfer.moe_ep import (
        CakeAlltoAllConfig,
        NVLinkOneSidedConfig,
        NVLinkTwoSidedConfig,
    )

    low_precision = bool(args.use_low_precision_combine)
    if args.comm == "nvlink_one_sided":
        return NVLinkOneSidedConfig(
            use_low_precision_combine=low_precision, cft=_CFT_MODES[args.cft]
        )
    if args.comm == "cake":
        return CakeAlltoAllConfig(use_low_precision_combine=low_precision)
    return NVLinkTwoSidedConfig()


def _compute_candidate(compute: str, weight_quant: str):
    """The fused_moe backend config of a --compute choice.

    The layer rejects a candidate that cannot serve the quantization; the
    benchmark reports that error instead of keeping its own support table.
    """
    from flashinfer.fused_moe.api import (
        CuteDslConfig,
        TrtllmBf16Config,
        TrtllmFp4Config,
        TrtllmFp8BlockConfig,
    )

    if compute == "cutedsl":
        return CuteDslConfig()
    if weight_quant == "bf16":
        return TrtllmBf16Config()
    if weight_quant in ("nvfp4", "mxfp4"):
        return TrtllmFp4Config()
    return TrtllmFp8BlockConfig()


class _Windows:
    """External CUDA events delimiting named windows of every timed iteration.

    ``iteration`` is the timed iteration being recorded; outside one (None)
    nothing is recorded, so warmup iterations carry no events.
    """

    def __init__(self, names: Sequence[str], iters: int) -> None:
        self.names = tuple(names)
        self._index = {name: k for k, name in enumerate(self.names)}
        # External events become event-record nodes when captured into a CUDA
        # graph, so they stay queryable with elapsed_time() after replay.
        self.events = [
            [
                (
                    torch.cuda.Event(enable_timing=True, external=True),
                    torch.cuda.Event(enable_timing=True, external=True),
                )
                for _ in self.names
            ]
            for _ in range(iters)
        ]
        # A CUDA event handle is created lazily on its first record().
        for pairs in self.events:
            for start, end in pairs:
                start.record()
                end.record()
        torch.cuda.synchronize()
        self.iteration: Optional[int] = None

    def mark(self, name: str, end: bool) -> None:
        """Record the start or end of window ``name`` in the current iteration."""
        if self.iteration is None:
            return
        if name not in self._index:
            raise RuntimeError(f"no timing window {name!r}; windows: {self.names}")
        self.events[self.iteration][self._index[name]][int(end)].record()

    @contextlib.contextmanager
    def window(self, name: str):
        self.mark(name, False)
        yield
        self.mark(name, True)

    def cupti_ids(self, cupti) -> List[Tuple[int, ...]]:
        return [
            tuple(
                cupti.get_cuda_event_id(event.cuda_event)
                for pair in pairs
                for event in pair
            )
            for pairs in self.events
        ]

    def elapsed_us(self) -> Dict[str, List[float]]:
        """Per-iteration CUDA-event time of every window."""
        return {
            name: [pairs[k][0].elapsed_time(pairs[k][1]) * 1e3 for pairs in self.events]
            for k, name in enumerate(self.names)
        }


def _reported(scope: str, stage_breakdown: bool, stages: Sequence[str]) -> List[str]:
    """The reported windows, each under its own name: the scope, or with
    ``stage_breakdown`` the stages inside it."""
    if not stage_breakdown:
        return [scope]
    first = stages.index(_DISPATCH_STAGE) if scope == "dispatch_compute_combine" else 0
    return list(stages[first:])


def _window_names(
    scope: str, stage_breakdown: bool, stages: Sequence[str]
) -> Tuple[str, ...]:
    """Every event window of one timed forward, in execution order."""
    if stage_breakdown:
        return tuple(stages)
    if scope == "dispatch_compute_combine":
        return ("before_dispatch", "dispatch_compute_combine")
    return ("forward",)


def _scope_windows(
    scope: str, stage_breakdown: bool, stages: Sequence[str], iters: int
):
    """Windows, layer stage hook and per-forward bracket of one timed forward.

    Returns the :class:`_Windows`, the ``stage_hook`` to install (or None) and
    a context manager factory entered around each timed forward.
    """
    windows = _Windows(_window_names(scope, stage_breakdown, stages), iters)
    if stage_breakdown:
        return windows, windows.window, contextlib.nullcontext
    if scope == "dispatch_compute_combine":

        @contextlib.contextmanager
        def boundary(stage: str):
            if stage == _DISPATCH_STAGE:
                windows.mark("before_dispatch", True)
                windows.mark("dispatch_compute_combine", False)
            yield

        @contextlib.contextmanager
        def around():
            windows.mark("before_dispatch", False)
            yield
            windows.mark("dispatch_compute_combine", True)

        return windows, boundary, around
    return windows, None, lambda: windows.window("forward")


def _merge_kernels(*kernel_lists: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per-kernel statistics of several windows as one list, by kernel name."""
    times: Dict[str, List[float]] = {}
    for kernels in kernel_lists:
        for kernel in kernels:
            times.setdefault(kernel["name"], []).extend(kernel["_times"])
    merged = [
        {"name": name, "count": len(values), "_times": values}
        for name, values in times.items()
    ]
    merged.sort(key=lambda k: sum(k["_times"]) / len(k["_times"]), reverse=True)
    return merged


def _time_scope(
    layer: Any,
    t: Any,
    forward_kwargs: Dict[str, Any],
    l2_buffer: torch.Tensor,
    *,
    scope: str,
    stage_breakdown: bool,
    stages: Sequence[str],
    warmup: int,
    iters: int,
    use_cuda_graph: bool,
    cupti_ctx: Optional[Any],
) -> Tuple[Dict[str, List[float]], Dict[str, Any]]:
    """Time the forward; returns the event times and CUPTI statistics of every
    window."""
    windows, hook, around = _scope_windows(scope, stage_breakdown, stages, iters)
    window_event_ids = windows.cupti_ids(cupti_ctx[0]) if cupti_ctx else []

    def _run_iterations() -> None:
        for _ in range(warmup):
            l2_buffer.zero_()
            layer.forward(t, **forward_kwargs)
        for i in range(iters):
            l2_buffer.zero_()
            windows.iteration = i
            with around():
                layer.forward(t, **forward_kwargs)
            windows.iteration = None

    graph = None
    layer.stage_hook = hook
    try:
        if use_cuda_graph:
            # One replay covers warmup and timed iterations, so no host gaps remain.
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                _run_iterations()
        if cupti_ctx is not None:
            cupti, cupti_kernels, cupti_events = cupti_ctx
            cupti.activity_flush_all(0)
            cupti_kernels.clear()
            cupti_events.clear()
        sync()
        if graph is not None:
            graph.replay()
        else:
            _run_iterations()
        sync()
    finally:
        layer.stage_hook = None
    stats: Dict[str, Any] = {}
    if cupti_ctx is not None:
        cupti.activity_flush_all(0)
        try:
            stats = kernel_spans_cupti(
                cupti_kernels,
                cupti_events,
                windows.names,
                window_event_ids,
                # The payload of an unquantized dispatch needs no kernel.
                optional_phases=("quantize", "before_dispatch"),
            )
        except RuntimeError as exc:
            # Keep every rank on the same collective sequence even if one trace is incomplete.
            stats["cupti_error"] = str(exc)
    del graph
    return windows.elapsed_us(), stats


def _time_layer(
    layer: Any,
    t: Any,
    *,
    scope: str,
    stage_breakdown: bool,
    stages: Sequence[str],
    warmup: int,
    iters: int,
    use_cuda_graph: bool,
    cupti_ctx: Optional[Any],
    tune: bool,
    autotune_cache: Optional[str],
    expert_rows: int,
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]], Dict[str, Any]]:
    """Per-iteration times (µs) of every reported window (see :func:`_reported`).

    One eager forward first builds the expert kernels, picks their tactics
    and warms the transport; it is never timed. With ``tune`` it runs inside
    ``flashinfer.autotune``, which profiles the tactics of the step's
    ``expert_rows`` (the token rows the experts receive) unless
    ``autotune_cache`` (a tactic file, read and then extended) covers them;
    without it the kernels take their cached or fallback tactics.
    Returns the CUDA-event times, the CUPTI kernel spans and the per-kernel
    statistics of the reported windows, every other kernel in
    ``other_kernels`` (with ``cupti_error`` when a trace was incomplete).
    """
    from flashinfer.autotuner import autotune

    l2_buffer = l2_flush_buffer(t.hidden_states.device)
    graph_state = layer.create_graph_state(t) if use_cuda_graph else None
    forward_kwargs = {"graph_state": graph_state} if use_cuda_graph else {}
    # Only the step's own row count is profiled, not every bucket up to the
    # layer's tuning ceiling.
    with autotune(tune, cache=autotune_cache, tuning_buckets=(expert_rows,)):
        layer.forward(t, **forward_kwargs)
    sync()

    times, stats = _time_scope(
        layer,
        t,
        forward_kwargs,
        l2_buffer,
        scope=scope,
        stage_breakdown=stage_breakdown,
        stages=stages,
        warmup=warmup,
        iters=iters,
        use_cuda_graph=use_cuda_graph,
        cupti_ctx=cupti_ctx,
    )
    names = _reported(scope, stage_breakdown, stages)
    event_times = {name: times[name] for name in names}
    span_times = {
        name: stats[f"{name}_us_kernel_span"]
        for name in names
        if f"{name}_us_kernel_span" in stats
    }
    kernels = {
        f"{name}_kernels": stats[f"{name}_kernels"]
        for name in names
        if f"{name}_kernels" in stats
    }
    if "other_kernels" in stats:
        unreported = [
            stats.get(f"{window}_kernels", [])
            for window in _window_names(scope, stage_breakdown, stages)
            if window not in names
        ]
        kernels["other_kernels"] = _merge_kernels(stats["other_kernels"], *unreported)
    if "cupti_error" in stats:
        kernels["cupti_error"] = stats["cupti_error"]
    if graph_state is not None:
        # A layer holds one live graph state; the next batch size needs its own.
        graph_state.destroy()
    return event_times, span_times, kernels


def _random_weights(
    num_local_experts: int, hidden_size: int, intermediate_size: int, device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """BF16 gate/up and down projections of this rank's experts."""
    w13 = torch.randn(
        num_local_experts,
        2 * intermediate_size,
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )
    w2 = torch.randn(
        num_local_experts,
        hidden_size,
        intermediate_size,
        dtype=torch.bfloat16,
        device=device,
    )
    return w13.mul_(hidden_size**-0.5), w2.mul_(intermediate_size**-0.5)


def _build_layer(
    args: argparse.Namespace,
    *,
    rank: int,
    ep_size: int,
    local_rank: int,
    shape: Dict[str, Any],
    max_tokens_per_rank: int,
    weights: Tuple[torch.Tensor, torch.Tensor],
):
    from flashinfer.fused_moe.api import (
        BackendOptions,
        ExecutionConfig,
        ExpertConfig,
        MoEConfig,
        QuantConfig,
        QuantFormat,
        RoutingConfig,
    )
    from flashinfer.moe_ep import (
        BootstrapConfig,
        EpAlgorithm,
        EpLayout,
        FleetParams,
        FusedMoeKernelConfig,
        MoEEpLayer,
        MoEWeightPack,
        SplitConfig,
    )

    num_local_experts = shape["num_experts"] // ep_size
    moe_config = MoEConfig(
        routing=RoutingConfig(num_experts=shape["num_experts"], top_k=shape["top_k"]),
        quant=QuantConfig(
            weight=QuantFormat[QUANT_FORMATS[shape["weight_quant"]]],
            activation=QuantFormat[QUANT_FORMATS[shape["quant"]]],
        ),
        experts=ExpertConfig(
            intermediate_size=shape["intermediate_size"],
            local_expert_offset=rank * num_local_experts,
            local_num_experts=num_local_experts,
        ),
        backend=BackendOptions(
            candidates=(_compute_candidate(args.compute, shape["weight_quant"]),)
        ),
        # The rank-major exchange hands the experts every rank's tokens.
        execution=ExecutionConfig(tune_max_num_tokens=max_tokens_per_rank * ep_size),
    )
    w13, w2 = weights
    return MoEEpLayer(
        BootstrapConfig(world_size=ep_size, rank=rank, device=local_rank),
        FleetParams(
            num_experts=shape["num_experts"],
            max_tokens_per_rank=max_tokens_per_rank,
            token_hidden_size=shape["hidden_size"],
            algorithm=EpAlgorithm.LOW_LATENCY,
            layout=EpLayout.RANK_MAJOR,
        ),
        MoEWeightPack(w13=w13, w2=w2),
        backend=SplitConfig(
            comm=_comm_config(args),
            kernel=FusedMoeKernelConfig(moe_config=moe_config),
        ),
    )


def _write_report(
    path: Optional[str], metadata: Dict[str, Any], results: List[Dict[str, Any]]
) -> None:
    if not path:
        return
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"benchmark_metadata": metadata, "results": results}, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--comm",
        type=str.lower,
        default="nvlink_one_sided",
        choices=COMM_BACKENDS,
        help="Communication backend of the layer.",
    )
    parser.add_argument(
        "--compute",
        type=str.lower,
        default="trtllm",
        choices=COMPUTE_BACKENDS,
        help="Expert computation: the TRT-LLM-Gen or the CuTe DSL fused MoE.",
    )
    add_profile_args(parser)
    parser.add_argument(
        "--intermediate_size", type=int, default=None, help="Override the profile."
    )
    parser.add_argument(
        "--quant",
        default=None,
        choices=ACTIVATION_FORMATS,
        help="Override the profile's activation (dispatch) format.",
    )
    parser.add_argument(
        "--weight_quant",
        default=None,
        choices=sorted(QUANT_FORMATS),
        help="Override the profile's expert weight format.",
    )
    add_sweep_args(parser)
    parser.add_argument(
        "--max_num_tokens_per_rank",
        type=int,
        default=None,
        help="Workspace capacity per rank (default: the largest scanned local batch).",
    )
    parser.add_argument("--iters", type=int, default=200, help="Timed iterations.")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations.")
    parser.add_argument(
        "--kernel_breakdown",
        action="store_true",
        help="Also report the time of every kernel inside each reported span "
        "(C++ kernel names demangled when the cxxfilt package is installed).",
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
        help="Input and weight seed (rank r uses seed + r).",
    )
    parser.add_argument(
        "--perfect_router",
        action="store_true",
        help="Deterministic balanced routing, removing load imbalance across ranks.",
    )
    parser.add_argument(
        "--use_low_precision_combine",
        action="store_true",
        help="Send combine payloads as FP8 (nvlink_one_sided and cake).",
    )
    parser.add_argument(
        "--cft",
        choices=sorted(_CFT_MODES),
        default="auto",
        help="CFT counted writes of nvlink_one_sided: auto (where supported, up "
        "to its token-count limits), on (every supported step) or off.",
    )
    parser.add_argument(
        "--no_cuda_graph",
        action="store_true",
        help="Run eagerly instead of replaying one CUDA graph.",
    )
    parser.add_argument(
        "--no_autotune",
        action="store_true",
        help="Run the expert kernels with their cached or fallback tactics "
        "instead of profiling them in the untimed first forward.",
    )
    parser.add_argument(
        "--autotune_cache",
        default=None,
        help="Tactic file prefix; rank r reads and extends <prefix>.rank<r>.json, "
        "so repeated runs skip the profiling.",
    )
    parser.add_argument(
        "--timing_scope",
        choices=TIMING_SCOPES,
        default="dispatch_compute_combine",
        help="Span of the forward to time: dispatch_compute_combine (first "
        "dispatch to last combine kernel) or forward.",
    )
    parser.add_argument(
        "--stage_breakdown",
        action="store_true",
        help="Report the stages inside the timing scope instead of its span.",
    )
    args = parser.parse_args()
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

    from flashinfer.moe_ep import MoEEpSplitLayer, MoEEpTensors

    profile = PROFILES[args.profile]
    shape = {
        "hidden_size": int(args.hidden_size or profile.hidden_size),
        "top_k": int(args.top_k or profile.top_k),
        "num_experts": int(args.num_experts or profile.num_experts),
        "intermediate_size": int(args.intermediate_size or profile.intermediate_size),
        "quant": args.quant or profile.quant,
        "weight_quant": args.weight_quant or profile.weight_quant,
    }
    if shape["num_experts"] % ep_size:
        raise ValueError(
            f"num_experts ({shape['num_experts']}) must be divisible by EP size ({ep_size})"
        )
    batch_sizes = local_batch_sizes(args)
    max_num_tokens_per_rank = int(args.max_num_tokens_per_rank or max(batch_sizes))
    if max_num_tokens_per_rank < max(batch_sizes):
        raise ValueError(
            "--max_num_tokens_per_rank must cover the largest scanned batch"
        )
    stages = MoEEpSplitLayer.STAGES
    names = _reported(args.timing_scope, args.stage_breakdown, stages)

    metadata: Dict[str, Any] = {
        "bench": _TAG,
        "profile": args.profile,
        "comm": args.comm,
        "compute": args.compute,
        "ep_size": ep_size,
        **shape,
        "experts_per_rank": shape["num_experts"] // ep_size,
        "local_batch_size": batch_sizes,
        "max_num_tokens_per_rank": max_num_tokens_per_rank,
        "perfect_router": bool(args.perfect_router),
        "use_low_precision_combine": bool(args.use_low_precision_combine),
        "cft": args.cft,
        "random_seed": int(args.random_seed),
        "device": torch.cuda.get_device_name(device),
        "cuda_graph": not args.no_cuda_graph,
        "autotune": not args.no_autotune,
        "autotune_cache": args.autotune_cache,
        "timing_scope": args.timing_scope,
        "stage_breakdown": bool(args.stage_breakdown),
        "stages": list(stages),
        "reported": names,
        "cupti_enabled": cupti_ctx is not None,
        "warning": cupti_warning,
    }
    print_rank0(json.dumps(metadata, indent=2))

    torch.manual_seed(int(args.random_seed) + rank)
    torch.cuda.manual_seed_all(int(args.random_seed) + rank)
    weights = _random_weights(
        shape["num_experts"] // ep_size,
        shape["hidden_size"],
        shape["intermediate_size"],
        device,
    )
    results: List[Dict[str, Any]] = []
    autotune_cache = (
        f"{args.autotune_cache}.rank{rank}.json" if args.autotune_cache else None
    )

    layer, error = None, None
    try:
        # Construction is local to the rank; the transport comes up on the
        # first forward.
        layer = _build_layer(
            args,
            rank=rank,
            ep_size=ep_size,
            local_rank=local_rank,
            shape=shape,
            max_tokens_per_rank=max_num_tokens_per_rank,
            weights=weights,
        )
    except (RuntimeError, ValueError, TypeError, NotImplementedError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    error = first_error(error)
    if error is not None:
        print_rank0(f"[{_TAG}] Cannot build {args.comm} + {args.compute}: {error}")
        batch_sizes = []
    use_cuda_graph = not args.no_cuda_graph

    for local_num_tokens in batch_sizes:
        label = f"{args.comm} + {args.compute} @ local_batch_size={local_num_tokens}"
        hidden_states = torch.randn(
            local_num_tokens, shape["hidden_size"], dtype=torch.bfloat16, device=device
        )
        topk_ids, topk_weights = make_routing(
            local_num_tokens,
            shape["top_k"],
            shape["num_experts"],
            ep_size,
            bool(args.perfect_router),
            device,
        )
        step_max_tokens = max(allgather(int(local_num_tokens)))
        t = MoEEpTensors(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            runtime_max_tokens_per_rank=step_max_tokens,
        )
        timings, error = None, None
        try:
            timings = _time_layer(
                layer,
                t,
                scope=args.timing_scope,
                stage_breakdown=bool(args.stage_breakdown),
                stages=stages,
                warmup=int(args.warmup),
                iters=int(args.iters),
                use_cuda_graph=use_cuda_graph,
                cupti_ctx=cupti_ctx,
                tune=not args.no_autotune,
                autotune_cache=autotune_cache,
                # The rank-major exchange hands the experts every rank's rows.
                expert_rows=step_max_tokens * ep_size,
            )
        except (RuntimeError, ValueError, NotImplementedError) as exc:
            # Kernel selection rejects an unsupported configuration identically
            # on every rank, before the transport's first collective returns.
            error = f"{type(exc).__name__}: {exc}"
        error = first_error(error)
        if error is not None:
            # A failed forward may leave the transport mid-round, so the sweep
            # stops here.
            print_rank0(f"[{_TAG}] Stopping at {label}: {error}")
            break
        times, span_times, stats = timings

        cupti_errors = first_error(stats.get("cupti_error"))
        if cupti_errors is not None:
            warning = warn_kernel_span_unavailable(f"{label}: {cupti_errors}", _TAG)
            metadata["warning"] = (
                f"{metadata['warning']}\n{warning}" if metadata["warning"] else warning
            )
        elif cupti_ctx is not None:
            # Every rank reports the same timing source for this measurement.
            times = span_times

        output: Dict[str, Any] = {
            "comm": args.comm,
            "compute": args.compute,
            "local_batch_size": int(local_num_tokens),
            "cuda_graph": use_cuda_graph,
            **{
                f"{name}_us": gather_per_rank(times[name], bool(args.iter_stats))
                for name in names
            },
        }
        if args.kernel_breakdown:
            output.update(
                gather_kernel_breakdown(
                    stats,
                    (*(f"{name}_kernels" for name in names), "other_kernels"),
                    bool(args.iter_stats),
                )
            )
        if rank == 0:
            summary = ", ".join(
                f"{name} {rank_mean(output[f'{name}_us']):.1f}" for name in names
            )
            print(
                f"[{_TAG}] local_batch_size={local_num_tokens}: {summary} us",
                flush=True,
            )
            results.append(output)
            # Rewritten after every batch size, so a later failure keeps the
            # points already measured.
            _write_report(args.output_file, metadata, results)

    # Every rank joins the barrier, including one whose layer failed to build.
    sync()
    if layer is not None:
        layer.destroy()

    if rank == 0 and results:
        _write_report(args.output_file, metadata, results)
        if args.output_file:
            print(f"Report written to {args.output_file}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
