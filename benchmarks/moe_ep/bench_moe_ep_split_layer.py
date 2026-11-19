"""Benchmark one MoE expert-parallel layer of separate kernels: dispatch, experts, combine.

Times ``MoEEpSplitLayer.forward()`` with one communication backend (``--comm``)
and one expert-compute backend (``--compute``) for a model profile over a sweep
of local batch sizes. The forward runs four stages (``MoEEpSplitLayer.STAGES``):
``quantize`` turns the BF16 hidden states into the dispatch payload,
``dispatch`` and ``combine`` run the all-to-all, and ``compute`` runs the routed
experts. One layer, whose workspace holds the largest batch, serves the sweep,
and each step runs at its own token count. The first, untimed forward of each
step autotunes the expert kernels (``--no_autotune``, ``--autotune_cache``).

A time is a CUPTI kernel span: first kernel start to last kernel end inside a
window delimited by two CUDA events, per rank and iteration. Warmup and timed
iterations run in one CUDA graph replay (``--no_cuda_graph``: eagerly), and
every iteration flushes L2 first. Rank 0 prints the mean over ranks; the JSON
report also holds every rank's time and the slowest rank's per iteration.
Without cupti-python>=13.4 (``requirements.txt``) the times fall back to the
CUDA events, and ``benchmark_metadata.warning`` says why.

Three options choose what is reported:

- ``--timing_scope``: ``dispatch_compute_combine`` (default) spans the first
  dispatch kernel to the last combine kernel, the part a fused MegaMoE layer
  replaces; ``forward`` also includes ``quantize``.
- ``--stage_breakdown``: every stage inside the scope instead of the scope. An
  event at each stage boundary delays the next kernel by a few microseconds and
  stops it from overlapping the previous stage, so the stages add up to more
  than the scope.
- ``--kernel_breakdown``: also every kernel's time inside each reported window;
  it changes no time.

Launch on one node, for example:

    # DeepSeek-V4-Pro experts (MXFP4 weights, MXFP8 activations) behind the
    # one-sided all-to-all, local batches 1..1024.
    torchrun --standalone --nproc-per-node=8 \\
        benchmarks/moe_ep/bench_moe_ep_split_layer.py \\
        --comm nvlink_one_sided --compute trtllm --profile deepseek_v4_pro \\
        --perfect_router -b 1 -e 1024 -f 2 --kernel_breakdown --output_file out.json

    # The same per stage, with the all-to-all on its fence protocol.
    torchrun --standalone --nproc-per-node=8 \\
        benchmarks/moe_ep/bench_moe_ep_split_layer.py \\
        --comm nvlink_one_sided --sync_protocol fence --compute trtllm \\
        --profile deepseek_v4_pro --perfect_router -b 1 -e 1024 -f 2 \\
        --stage_breakdown

``bench_moe_ep_comm.py`` times the all-to-all alone and
``bench_moe_ep_mega_layer.py`` a fused MegaMoE layer, both the same way.
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
    TimingWindows,
    add_layer_args,
    add_profile_args,
    add_sweep_args,
    agree_on_cupti,
    allgather,
    first_error,
    gather_window_results,
    l2_flush_buffer,
    layer_metadata,
    layer_shape,
    local_batch_sizes,
    make_routing,
    print_rank0,
    random_expert_weights,
    rank_mean,
    resolve_times,
    start_cupti,
    sync,
    time_windows,
    warn_if_kernel_names_mangled,
    write_report,
)

_TAG = "bench_moe_ep_split_layer"

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
COMPUTE_BACKENDS = ("trtllm", "cutedsl", "prims_ts")
# NVLinkOneSidedConfig.sync_protocol choices.
SYNC_PROTOCOLS = ("auto", "fence", "sentinel")
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
            use_low_precision_combine=low_precision,
            sync_protocol=args.sync_protocol,
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
        PrimsTsConfig,
        TrtllmBf16Config,
        TrtllmFp4Config,
        TrtllmFp8BlockConfig,
    )

    if compute == "cutedsl":
        return CuteDslConfig()
    if compute == "prims_ts":
        return PrimsTsConfig()
    if weight_quant == "bf16":
        return TrtllmBf16Config()
    if weight_quant in ("nvfp4", "mxfp4"):
        return TrtllmFp4Config()
    return TrtllmFp8BlockConfig()


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

    Returns the :class:`TimingWindows`, the ``stage_hook`` to install (or None)
    and a context manager factory entered around each forward.
    """
    windows = TimingWindows(_window_names(scope, stage_breakdown, stages), iters)
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
    Returns what :func:`moe_ep_bench_utils.time_windows` returns.
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

    windows, hook, around = _scope_windows(scope, stage_breakdown, stages, iters)

    def forward() -> None:
        with around():
            layer.forward(t, **forward_kwargs)

    layer.stage_hook = hook
    try:
        result = time_windows(
            forward,
            windows,
            _reported(scope, stage_breakdown, stages),
            l2_buffer=l2_buffer,
            warmup=warmup,
            use_cuda_graph=use_cuda_graph,
            cupti_ctx=cupti_ctx,
            # The payload of an unquantized dispatch needs no kernel.
            optional_windows=("quantize", "before_dispatch"),
        )
    finally:
        layer.stage_hook = None
    if graph_state is not None:
        # A layer holds one live graph state; the next batch size needs its own.
        graph_state.destroy()
    return result


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
        help="Expert computation: the TRT-LLM-Gen, CuTe DSL or Prims-TS fused MoE.",
    )
    add_profile_args(parser)
    add_layer_args(parser)
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
    parser.add_argument(
        "--use_low_precision_combine",
        action="store_true",
        help="Send combine payloads as FP8 (nvlink_one_sided and cake).",
    )
    parser.add_argument(
        "--sync_protocol",
        "--sync-protocol",
        choices=SYNC_PROTOCOLS,
        default="auto",
        help="Sync protocol of nvlink_one_sided: fence or sentinel for every "
        "step, or auto (default): the sentinel protocol for steps of at most 256 "
        "tokens per rank where every rank supports it, the fence protocol for "
        "the others.",
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
    return parser.parse_args()


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

    shape = layer_shape(args, ep_size)
    batch_sizes = local_batch_sizes(args)
    max_num_tokens_per_rank = int(args.max_num_tokens_per_rank or max(batch_sizes))
    if max_num_tokens_per_rank < max(batch_sizes):
        raise ValueError(
            "--max_num_tokens_per_rank must cover the largest scanned batch"
        )
    stages = MoEEpSplitLayer.STAGES
    names = _reported(args.timing_scope, args.stage_breakdown, stages)

    metadata = layer_metadata(
        _TAG,
        args,
        shape,
        ep_size=ep_size,
        batch_sizes=batch_sizes,
        device=device,
        cupti_enabled=cupti_ctx is not None,
        warning=cupti_warning,
    )
    metadata.update(
        {
            "comm": args.comm,
            "compute": args.compute,
            "max_num_tokens_per_rank": max_num_tokens_per_rank,
            "use_low_precision_combine": bool(args.use_low_precision_combine),
            "sync_protocol": args.sync_protocol,
            # nvlink_one_sided's largest step on the sentinel protocol, set once
            # the transport is up.
            "sentinel_max_tokens": None,
            "autotune": not args.no_autotune,
            "autotune_cache": args.autotune_cache,
            "timing_scope": args.timing_scope,
            "stage_breakdown": bool(args.stage_breakdown),
            "stages": list(stages),
            "reported": names,
        }
    )
    print_rank0(json.dumps(metadata, indent=2))

    torch.manual_seed(int(args.random_seed) + rank)
    torch.cuda.manual_seed_all(int(args.random_seed) + rank)
    weights = random_expert_weights(
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
        event_times, span_times, kernels = timings
        if metadata["sentinel_max_tokens"] is None:
            metadata["sentinel_max_tokens"] = getattr(
                layer._communication, "sentinel_max_tokens", None
            )
        times = resolve_times(event_times, span_times, kernels, metadata, label, _TAG)
        output: Dict[str, Any] = {
            "comm": args.comm,
            "compute": args.compute,
            "local_batch_size": int(local_num_tokens),
            "cuda_graph": use_cuda_graph,
            **gather_window_results(
                times,
                kernels,
                names,
                iter_stats=bool(args.iter_stats),
                kernel_breakdown=bool(args.kernel_breakdown),
            ),
        }
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
            write_report(args.output_file, metadata, results)

    # Every rank joins the barrier, including one whose layer failed to build.
    sync()
    if layer is not None:
        layer.destroy()

    if rank == 0 and results:
        write_report(args.output_file, metadata, results)
        if args.output_file:
            print(f"Report written to {args.output_file}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
