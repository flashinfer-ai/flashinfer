"""Benchmark one fused MoE expert-parallel layer: a MegaMoE backend of ``MoEEpMegaLayer``.

Times the MegaMoE call (dispatch, expert GEMMs and combine in the backend's
fused kernels) of the backend that ``--kernel_name`` selects, for a model
profile over a sweep of local batch sizes; every rank runs the same batch.
Each batch size builds its own layer whose capacity per rank is that batch, as
a deployment sized for it would, and warms it up untimed.

A time is a CUPTI kernel span: first kernel start to last kernel end inside a
window delimited by two CUDA events, per rank and iteration. Warmup and timed
iterations run in one CUDA graph replay (``--no_cuda_graph``: eagerly), and
every iteration flushes L2 first. Rank 0 prints the slowest rank's time per
iteration, median over iterations, with p10 and p90; the JSON report also
holds every rank's time. Without cupti-python>=13.4 (``requirements.txt``) the
times fall back to the CUDA events, and ``benchmark_metadata.warning`` says why.

``--timing_scope`` sets what the window holds:

- ``mega`` (default): the backend's compute call alone
  (``MoEEpMegaLayer.compute_staged``). Staging, which quantizes the BF16 hidden
  states and writes them and the routing into the workspace, runs after the L2
  flush and before the window. Its kernels stay out of the window, but where
  the call waits on its peers (a few tokens per rank), a longer staging still
  lengthens that wait by a few microseconds.
- ``forward``: ``MoEEpMegaLayer.forward``, staging included.

Neither times an output copy: the output is a workspace view where the backend
offers one. Other options:

- ``--list_kernels``: print the kernel names it runs and exit.
- ``--kernel_breakdown``: also every kernel's time inside the window.
- ``--backend_config``: further fields of the selected backend's kernel
  config, as a JSON object.
- ``--shared_expert``: one MXFP8 shared expert fused into the layer, on
  backends whose config has a ``shared_expert`` field (the NVLinkOneSided
  MegaMoE). FlashInfer's DeepGEMM backend (``sm100_fp8_fp4_bf16_deepgemm``)
  does not expose DeepGEMM's fused shared expert, so it rejects the option.

Profiles with MXFP4 weights give every backend the same packed MXFP4 weights;
the others give BF16 weights for the backend to quantize.

Launch on one node, for example (B300; ``sm_100a`` on B200):

    # DeepSeek-V4.1-Flash routed experts, local batches 1, 4, ..., 4096, on the
    # NVLinkOneSided MegaMoE and then on DeepGEMM's.
    CUTE_DSL_ARCH=sm_103a torchrun --standalone --nproc-per-node=8 \\
        benchmarks/moe_ep/bench_moe_ep_mega_layer.py \\
        --kernel_name sm103_mxfp8_mxfp4_bf16_nvlink_one_sided \\
        --profile deepseek_v4_1_flash --activation_clamp 10 --perfect_router \\
        -b 1 -e 4096 -f 4 --kernel_breakdown --output_file ours.json
    CUTE_DSL_ARCH=sm_103a torchrun --standalone --nproc-per-node=8 \\
        benchmarks/moe_ep/bench_moe_ep_mega_layer.py \\
        --kernel_name sm100_fp8_fp4_bf16_deepgemm \\
        --profile deepseek_v4_1_flash --activation_clamp 10 --perfect_router \\
        -b 1 -e 4096 -f 4 --output_file deepgemm.json
"""

from __future__ import annotations

import argparse
import dataclasses
import enum
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from moe_ep_bench_utils import (
    TimingWindows,
    add_layer_args,
    add_profile_args,
    add_sweep_args,
    agree_on_cupti,
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

_TAG = "bench_moe_ep_mega_layer"
TIMING_SCOPES = ("mega", "forward")
# The midpoints between consecutive E2M1 magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, 6.
_E2M1_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


def _kernel_config_types() -> Dict[str, type]:
    """The MegaMoE kernel configs ``flashinfer.moe_ep`` exports, by kernel name."""
    import flashinfer.moe_ep as moe_ep

    types: Dict[str, type] = {}
    for name in dir(moe_ep):
        cls = getattr(moe_ep, name)
        if not (
            name.endswith("MegaMoeConfig")
            and isinstance(cls, type)
            and dataclasses.is_dataclass(cls)
        ):
            continue
        fields = {field.name: field for field in dataclasses.fields(cls)}
        kernel_name = getattr(fields.get("kernel_name"), "default", None)
        if isinstance(kernel_name, str):
            types.setdefault(kernel_name, cls)
    return types


def _kernel_config(args: argparse.Namespace, shape: Dict[str, Any]):
    types = _kernel_config_types()
    if args.kernel_name not in types:
        raise ValueError(
            f"unknown --kernel_name {args.kernel_name!r}; known: {sorted(types)}"
        )
    config_type = types[args.kernel_name]
    fields = {field.name: field for field in dataclasses.fields(config_type)}
    kwargs: Dict[str, Any] = {
        "intermediate_size": shape["intermediate_size"],
        "top_k": shape["top_k"],
    }
    if args.activation_clamp is not None:
        kwargs["activation_clamp"] = args.activation_clamp
    if args.shared_expert:
        if "shared_expert" not in fields:
            raise ValueError(f"{args.kernel_name} has no fused shared expert")
        kwargs["shared_expert"] = "fp8_in_kernel"
        if args.activation_clamp is not None and "shared_gate_up_clamp" in fields:
            kwargs["shared_gate_up_clamp"] = args.activation_clamp
    extra = json.loads(args.backend_config)
    if not isinstance(extra, dict):
        raise ValueError("--backend_config must be a JSON object")
    # The fields set above also shape the generated inputs and weights, so they
    # come only from the benchmark's own options.
    overridden = sorted(set(extra) & set(kwargs))
    if overridden:
        raise ValueError(
            f"--backend_config cannot set {overridden}; set them through the "
            "benchmark's own options (e.g. --top_k, --intermediate_size), "
            "which also shape the inputs and weights"
        )
    kwargs.update(extra)
    unknown = sorted(set(kwargs) - set(fields))
    if unknown:
        raise ValueError(f"{config_type.__name__} has no fields {unknown}")
    # JSON cannot hold an enum member, so a string given for a field whose
    # default is an enum names a member, e.g. '{"finalize_mode": "NONE"}'.
    for name, value in kwargs.items():
        enum_type = type(fields[name].default)
        if isinstance(value, str) and issubclass(enum_type, enum.Enum):
            if value not in enum_type.__members__:
                raise ValueError(
                    f"{name} must name a member of {enum_type.__name__} "
                    f"{list(enum_type.__members__)}, got {value!r}"
                )
            kwargs[name] = enum_type[value]
    return config_type(**kwargs)


def _quantize_mx(x: torch.Tensor, fp4: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """MXFP4 (E2M1, two per byte, low nibble first) or MXFP8 (E4M3) along the
    last dim, with UE8M0 scale codes (uint8) per 32 values rounded up to powers
    of two."""
    blocks = x.float().unflatten(-1, (-1, 32))
    amax = blocks.abs().amax(-1) * (1.0 / (6.0 if fp4 else 448.0))
    bits = amax.clamp(min=2.0**-126).view(torch.int32)
    codes = (((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).int()).clamp(1, 254)
    scaled = (blocks / (codes << 23).view(torch.float32).unsqueeze(-1)).flatten(-2)
    if not fp4:
        return scaled.to(torch.float8_e4m3fn), codes.to(torch.uint8)
    # Nearest E2M1 magnitude; a tie goes to the even code, so the odd-numbered
    # midpoints round up.
    magnitude = scaled.abs()
    nibbles = torch.zeros_like(magnitude, dtype=torch.uint8)
    for j, midpoint in enumerate(_E2M1_MIDPOINTS):
        nibbles += (magnitude >= midpoint if j % 2 else magnitude > midpoint).to(
            torch.uint8
        )
    nibbles |= ((scaled < 0) & (nibbles != 0)).to(torch.uint8) << 3
    pairs = nibbles.unflatten(-1, (-1, 2))
    return pairs[..., 0] | (pairs[..., 1] << 4), codes.to(torch.uint8)


def _weights(args: argparse.Namespace, shape: Dict[str, Any], ep_size: int, device):
    """This rank's expert weights as the layer takes them."""
    from flashinfer.moe_ep import MoEWeightPack, PrequantizedMoEWeightsWithSharedExpert

    hidden, intermediate = shape["hidden_size"], shape["intermediate_size"]
    w13, w2 = random_expert_weights(
        shape["num_experts"] // ep_size, hidden, intermediate, device
    )
    if shape["weight_quant"] != "mxfp4":
        if args.shared_expert:
            raise ValueError("--shared_expert needs a profile with MXFP4 weights")
        return MoEWeightPack(w13=w13, w2=w2)
    # One expert at a time bounds the quantization's scratch memory.
    w13_q, w13_sf = zip(*(_quantize_mx(w, fp4=True) for w in w13), strict=True)
    w2_q, w2_sf = zip(*(_quantize_mx(w, fp4=True) for w in w2), strict=True)
    routed = [torch.stack(leg) for leg in (w13_q, w2_q, w13_sf, w2_sf)]
    del w13, w2
    if not args.shared_expert:
        return MoEWeightPack(*routed)
    # The shared expert is the same on every rank.
    generator = torch.Generator(device=device).manual_seed(int(args.random_seed))
    shared_w13, shared_w13_sf = _quantize_mx(
        torch.randn(2 * intermediate, hidden, device=device, generator=generator)
        * hidden**-0.5,
        fp4=False,
    )
    shared_w2, shared_w2_sf = _quantize_mx(
        torch.randn(hidden, intermediate, device=device, generator=generator)
        * intermediate**-0.5,
        fp4=False,
    )
    return PrequantizedMoEWeightsWithSharedExpert(
        *routed,
        shared_w13=shared_w13,
        shared_w2=shared_w2,
        shared_w13_scale=shared_w13_sf,
        shared_w2_scale=shared_w2_sf,
    )


def _build_layer(
    kernel_config: Any,
    weights: Any,
    *,
    rank: int,
    ep_size: int,
    local_rank: int,
    shape: Dict[str, Any],
    capacity: int,
):
    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpMegaLayer,
    )

    return MoEEpMegaLayer(
        BootstrapConfig(world_size=ep_size, rank=rank, device=local_rank),
        FleetParams(
            num_experts=shape["num_experts"],
            max_tokens_per_rank=capacity,
            token_hidden_size=shape["hidden_size"],
        ),
        weights,
        backend=MegaConfig(megakernel=kernel_config, quantize_input=True),
    )


def _time_layer(
    layer: Any,
    t: Any,
    *,
    scope: str,
    warmup: int,
    iters: int,
    use_cuda_graph: bool,
    cupti_ctx: Optional[Any],
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]], Dict[str, Any]]:
    """Per-iteration times (µs) of the ``scope`` window; returns what
    :func:`moe_ep_bench_utils.time_windows` returns.

    ``t.output`` is None where the backend returns a workspace view, else a
    preallocated output.
    """
    # Allocates the workspace, compiles and launches once, eagerly.
    layer.warmup(t)
    sync()
    windows = TimingWindows((scope,), iters)
    view = t.output is None

    def forward() -> None:
        if scope == "mega":
            layer.stage_inputs(t)
            with windows.window(scope):
                layer.compute_staged(output=t.output)
        else:
            with windows.window(scope):
                layer.forward(t, return_workspace_view=view)

    return time_windows(
        forward,
        windows,
        (scope,),
        l2_buffer=l2_flush_buffer(t.hidden_states.device),
        warmup=warmup,
        use_cuda_graph=use_cuda_graph,
        cupti_ctx=cupti_ctx,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--kernel_name",
        help="Registered MegaMoE kernel, e.g. sm100_fp8_fp4_bf16_deepgemm "
        "(--list_kernels prints them all).",
    )
    parser.add_argument(
        "--list_kernels",
        action="store_true",
        help="Print every kernel name this benchmark runs, with its config "
        "class, and exit; needs no GPU or torchrun.",
    )
    add_profile_args(parser)
    add_layer_args(parser)
    add_sweep_args(parser)
    parser.add_argument(
        "--activation_clamp",
        type=float,
        default=None,
        help="SwiGLU clamp, the kernel config's activation_clamp (DeepSeek-V4: 10).",
    )
    parser.add_argument(
        "--backend_config",
        default="{}",
        help="Further kernel-config fields as a JSON object.",
    )
    parser.add_argument(
        "--shared_expert",
        action="store_true",
        help="Fuse one MXFP8 shared expert of the routed experts' shape.",
    )
    parser.add_argument(
        "--timing_scope",
        choices=TIMING_SCOPES,
        default="mega",
        help="mega: the compute call, staging excluded; forward: the whole forward.",
    )
    args = parser.parse_args()
    if not args.list_kernels and not args.kernel_name:
        parser.error("--kernel_name is required (--list_kernels prints the names)")
    return args


def main() -> None:
    args = parse_args()
    if args.list_kernels:
        for name, config_type in sorted(_kernel_config_types().items()):
            print(f"{name}  ({config_type.__name__})")
        return

    cupti_ctx, cupti_error = start_cupti()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", device_id=device)
    rank, ep_size = dist.get_rank(), dist.get_world_size()
    cupti_ctx, cupti_warning = agree_on_cupti(cupti_ctx, cupti_error, _TAG)
    if args.kernel_breakdown:
        warn_if_kernel_names_mangled(_TAG)

    from flashinfer.moe_ep import MoEEpTensors

    shape = layer_shape(args, ep_size)
    batch_sizes = local_batch_sizes(args)
    scope = args.timing_scope
    kernel_config = _kernel_config(args, shape)

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
            "kernel_name": args.kernel_name,
            "kernel_config": json.loads(
                json.dumps(dataclasses.asdict(kernel_config), default=str)
            ),
            "shared_expert": bool(args.shared_expert),
            "timing_scope": scope,
            "reported": [scope],
        }
    )
    print_rank0(json.dumps(metadata, indent=2))

    torch.manual_seed(int(args.random_seed) + rank)
    torch.cuda.manual_seed_all(int(args.random_seed) + rank)
    weights = _weights(args, shape, ep_size, device)
    results: List[Dict[str, Any]] = []
    use_cuda_graph = not args.no_cuda_graph

    layer = None
    for local_num_tokens in batch_sizes:
        label = f"{args.kernel_name} @ local_batch_size={local_num_tokens}"
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
        timings, error = None, None
        try:
            layer = _build_layer(
                kernel_config,
                weights,
                rank=rank,
                ep_size=ep_size,
                local_rank=local_rank,
                shape=shape,
                capacity=local_num_tokens,
            )
            t = MoEEpTensors(
                hidden_states=hidden_states,
                topk_ids=topk_ids.long(),
                topk_weights=topk_weights,
                output=None
                if layer.supports_output_view
                else torch.empty_like(hidden_states),
            )
            timings = _time_layer(
                layer,
                t,
                scope=scope,
                warmup=int(args.warmup),
                iters=int(args.iters),
                use_cuda_graph=use_cuda_graph,
                cupti_ctx=cupti_ctx,
            )
        except (ValueError, TypeError, NotImplementedError) as exc:
            # A backend rejects a configuration the same way on every rank, so
            # every rank reaches the agreement below. Runtime errors (a failed
            # launch, an allocation) can stop one rank inside a collective the
            # others wait in; they propagate, and torchrun stops every rank.
            # The report already holds every batch size measured before.
            error = f"{type(exc).__name__}: {exc}"
        error = first_error(error)
        if error is not None:
            print_rank0(f"[{_TAG}] Stopping at {label}: {error}")
            break
        event_times, span_times, kernels = timings
        times = resolve_times(event_times, span_times, kernels, metadata, label, _TAG)
        output: Dict[str, Any] = {
            "kernel_name": args.kernel_name,
            "local_batch_size": int(local_num_tokens),
            "max_tokens_per_rank": int(local_num_tokens),
            "cuda_graph": use_cuda_graph,
            **gather_window_results(
                times,
                kernels,
                [scope],
                iter_stats=bool(args.iter_stats),
                kernel_breakdown=bool(args.kernel_breakdown),
            ),
        }
        # Workspace release is collective and drains every rank first.
        layer.destroy()
        layer = None
        if rank == 0:
            slowest = output[f"{scope}_max_rank_us"]
            print(
                f"[{_TAG}] local_batch_size={local_num_tokens}: {scope} slowest rank "
                f"{slowest['median']:.1f} us (p10 {slowest['p10']:.1f}, "
                f"p90 {slowest['p90']:.1f}), rank mean "
                f"{rank_mean(output[f'{scope}_us']):.1f} us",
                flush=True,
            )
            results.append(output)
            # Rewritten after every batch size, so a later failure keeps the
            # points already measured.
            write_report(args.output_file, metadata, results)

    # Every rank joins the barrier, including after a failed step.
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
