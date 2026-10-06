"""Qualify and time prepared IQ2_XS MoE using actual safetensors expert weights.

Run with ``python -m benchmarks.benchmark_iq2_xs_moe --help``. JSONL output
retains checkpoint identity, source hashes, correctness and graph samples.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
import hashlib
import gc
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import torch
import torch.nn.functional as F

from b12x.moe import fused_moe as moe
from b12x.preparation import PreparationSession, PreparedCall, require_prepared
from b12x.testing.iq2_xs_reference import moe_reference_iq2_xs
from .iq2_xs_checkpoint import IQ2XSLayer, load_iq2_xs_layer
from .moe_preparation import request_for_capacity


@dataclass
class Inputs:
    x: torch.Tensor
    ids: torch.Tensor
    probabilities: torch.Tensor
    output: torch.Tensor
    expert_map: torch.Tensor | None


@contextmanager
def capture_lifetime():
    """Keep deferred CUDA library destructors outside graph capture."""
    gc.collect()
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()


def make_inputs(layer: IQ2XSLayer, tokens: int, device, *, mapped: bool) -> Inputs:
    generator = torch.Generator(device=device).manual_seed(419 + tokens)
    x = torch.randn(
        (tokens, layer.hidden_size),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    ids = (
        torch.arange(tokens * layer.top_k, device=device, dtype=torch.int32).reshape(
            tokens, layer.top_k
        )
        % layer.route_num_experts
    )
    probabilities = torch.full(ids.shape, 1.0 / layer.top_k, device=device)
    return Inputs(
        x,
        ids,
        probabilities,
        torch.empty_like(x),
        layer.expert_map(device) if mapped else None,
    )


def set_routes(inputs: Inputs, pattern: str, *, experts: int, offset: int = 0) -> None:
    tokens, top_k = inputs.ids.shape
    slots = torch.arange(top_k, device=inputs.ids.device)
    rows = torch.arange(tokens, device=inputs.ids.device)[:, None]
    if pattern == "hot":
        ids = slots.expand(tokens, top_k)
    elif pattern == "imbalanced":
        ids = slots + torch.where(rows % 8 == 0, rows * top_k, 0)
    elif pattern == "balanced":
        ids = rows * top_k + slots
    else:
        raise ValueError(f"unknown route pattern {pattern!r}")
    inputs.ids.copy_((ids + offset) % experts)


def reference(layer: IQ2XSLayer, inputs: Inputs, *, activation="silu") -> torch.Tensor:
    """Independent FP32 oracle with BF16 weight, projection and activation boundaries."""
    return moe_reference_iq2_xs(
        inputs.x,
        layer.weights.w13,
        layer.weights.w2,
        inputs.ids,
        inputs.probabilities,
        activation=activation,
        expert_ids=layer.expert_ids,
    )


def check(actual, expected) -> dict[str, float]:
    assert torch.isfinite(actual).all(), "nonfinite MoE output"
    if torch.count_nonzero(expected) == 0:
        assert torch.count_nonzero(actual) == 0, "nonzero output for zero contribution"
        return {"cosine": 1.0, "relative_l2": 0.0}
    assert torch.count_nonzero(actual) > 0, "empty MoE output"
    a, b = actual.float().flatten(), expected.float().flatten()
    cosine = F.cosine_similarity(a, b, dim=0).item()
    relative_l2 = ((a - b).norm() / b.norm()).item()
    assert cosine >= 0.999 and relative_l2 <= 0.01, (cosine, relative_l2)
    return {"cosine": cosine, "relative_l2": relative_l2}


def prepare_experts(layer: IQ2XSLayer, device, *, activation="silu"):
    plan = moe.plan_weights(
        source=moe.PackedSource(format=layer.weights.codec, w13_layout="w31"),
        activation=moe.ActivationSpec(
            mode="a16", nonlinearity=activation, io_dtype=torch.bfloat16
        ),
        geometry=moe.MoEGeometry(
            num_experts=len(layer.expert_ids),
            hidden_size=layer.hidden_size,
            intermediate_size=layer.intermediate_size,
        ),
    )
    source = moe.BlockQuantWeights(
        layer.weights.w13.to(device),
        layer.weights.w2.to(device),
        codec=layer.weights.codec,
    )
    experts = moe.prepare_weights(plan=plan, weights=source)
    prepared = experts._impl.representation.value
    payload = sum(
        t.numel() * t.element_size()
        for t in (prepared.w13, prepared.w13_scale, prepared.w2, prepared.w2_scale)
    )
    assert payload == source.w13.numel() + source.w2.numel()
    return experts, payload


def qualify_capacity(
    layer,
    experts,
    session,
    *,
    capacity,
    counts,
    route_mode,
    deterministic=False,
    activation="silu",
    patterns=("balanced", "hot", "imbalanced"),
    repeats=20,
    launches=50,
    config=None,
    autotune=False,
):
    """Exercise several live counts with compilation and kernel resolution frozen."""
    from b12x._lib import compiler
    from b12x._lib.runtime_control import kernel_resolution_guard
    from b12x.moe._shared.kernels.w4a16 import kernel
    from b12x.moe.fused_moe import _preparation

    mapped = layer.expert_ids != tuple(range(layer.route_num_experts))
    declaration = moe.plan_execution(
        experts=experts,
        capacity=moe.ExecutionCapacity(
            max_tokens=capacity,
            top_k=layer.top_k,
            route_num_experts=layer.route_num_experts,
        ),
        routing=moe.RoutingSpec(deterministic_output=deterministic),
        override=None
        if autotune
        else config
        or moe.MoeDecodeConfig(
            backend="w4a16",
            route_planner="internal",
            max_active_clusters=None,
            w4a16_route_mode=route_mode,
        ),
    )
    warmup = make_inputs(layer, capacity, experts.device, mapped=mapped)
    expected_warmup = (
        reference(layer, warmup, activation=activation) if autotune else None
    )
    activation_source = warmup.x.clone() if autotune else None

    def factory(state):
        scratch = tuple(
            torch.empty(spec.shape, dtype=spec.dtype, device=experts.device)
            for spec in state.scratch.scratch_specs()
        )
        bound = state.bind(
            scratch=scratch,
            a=warmup.x,
            topk_ids=warmup.ids,
            topk_weights=warmup.probabilities,
            output=warmup.output,
            route_expert_map=warmup.expert_map,
        )
        if autotune:
            # Qualify every candidate on the real weights before the session
            # measures it. The selected plan also passes the frozen-resolution,
            # mutation, poison and graph-replay checks below.
            warmup.output.fill_(float("nan"))
            bound.run()
            check(warmup.output, expected_warmup)
        return PreparedCall(
            run=bound.run,
            output=warmup.output,
            produce=(lambda: warmup.x.copy_(activation_source)) if autotune else None,
            owners=(scratch, bound, warmup, activation_source),
        )

    request = request_for_capacity(
        declaration,
        name=f"iq2-{capacity}-{route_mode}-{deterministic}",
        calls={capacity: factory},
    )
    session.prepare((request,))
    state = require_prepared(declaration, "moe.decode")
    state = state.variants[capacity] if hasattr(state, "variants") else state
    scratch = tuple(
        torch.empty(spec.shape, dtype=spec.dtype, device=experts.device)
        for spec in state.scratch.scratch_specs()
    )
    scratch_pointers = tuple(t.data_ptr() for t in scratch)

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "prepared IQ2_XS bind/run reached compilation or kernel resolution"
        )

    results = []
    with ExitStack() as frozen:
        frozen.enter_context(kernel_resolution_guard("IQ2_XS capacity qualification"))
        frozen.enter_context(patch.object(compiler, "compile", forbidden))
        frozen.enter_context(patch.object(_preparation, "compile_fused_moe", forbidden))
        frozen.enter_context(patch.object(kernel, "compile_w4a16_fused_moe", forbidden))
        frozen.enter_context(patch.object(kernel, "compile_w4a16_topk_sum", forbidden))
        for tokens in counts:
            inputs = make_inputs(layer, tokens, experts.device, mapped=mapped)
            binding = moe.bind(
                declaration,
                scratch=scratch,
                a=inputs.x,
                topk_ids=inputs.ids,
                topk_weights=inputs.probabilities,
                output=inputs.output,
                route_expert_map=inputs.expert_map,
            )
            for pattern in patterns:
                set_routes(inputs, pattern, experts=layer.route_num_experts)
                expected = reference(layer, inputs, activation=activation)
                inputs.output.fill_(float("nan"))
                moe.run(binding=binding)
                metrics = check(inputs.output, expected)
                graph = torch.cuda.CUDAGraph()
                try:
                    with capture_lifetime(), session.capture(), torch.cuda.graph(graph):
                        moe.run(binding=binding)
                    inputs.x.add_(0.03125)
                    set_routes(
                        inputs, pattern, experts=layer.route_num_experts, offset=13
                    )
                    inputs.probabilities.mul_(0.75)
                    expected = reference(layer, inputs, activation=activation)
                    inputs.output.fill_(float("nan"))
                    for buffer in scratch:
                        if buffer.dtype in (
                            torch.float32,
                            torch.bfloat16,
                            torch.float16,
                        ):
                            buffer.fill_(float("nan"))
                    pointer = inputs.output.data_ptr()
                    allocated = torch.cuda.memory_stats(experts.device)[
                        "allocation.all.allocated"
                    ]
                    graph.replay()
                    torch.cuda.synchronize(experts.device)
                    assert (
                        torch.cuda.memory_stats(experts.device)[
                            "allocation.all.allocated"
                        ]
                        == allocated
                    )
                    assert inputs.output.data_ptr() == pointer
                    assert tuple(t.data_ptr() for t in scratch) == scratch_pointers
                    metrics.update(
                        {
                            f"replay_{key}": value
                            for key, value in check(inputs.output, expected).items()
                        }
                    )
                    samples = []
                    start, end = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    for _ in range(repeats):
                        start.record()
                        for _ in range(launches):
                            graph.replay()
                        end.record()
                        end.synchronize()
                        samples.append(start.elapsed_time(end) * 1000 / launches)
                    # Zero contribution must overwrite previous nonzero output on replay.
                    inputs.probabilities.zero_()
                    inputs.output.fill_(float("nan"))
                    graph.replay()
                    check(inputs.output, torch.zeros_like(inputs.output))
                    inputs.probabilities.fill_(1.0 / layer.top_k)
                    if inputs.expert_map is not None:
                        inputs.expert_map.fill_(-1)
                        inputs.output.fill_(float("nan"))
                        graph.replay()
                        check(inputs.output, torch.zeros_like(inputs.output))
                        inputs.expert_map.copy_(layer.expert_map(experts.device))
                    results.append(
                        {
                            "tokens": tokens,
                            "capacity": capacity,
                            "route_mode": route_mode,
                            "deterministic": deterministic,
                            "pattern": pattern,
                            "mapped": mapped,
                            "config": state.config.to_dict(),
                            "graph_us": samples,
                            **metrics,
                        }
                    )
                finally:
                    graph.reset()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument(
        "--device",
        required=True,
        help="Assigned CUDA device within the existing visibility mask",
    )
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 20, 39])
    parser.add_argument("--tp-sizes", type=int, nargs="+", default=[1, 2])
    parser.add_argument(
        "--counts", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 128, 512]
    )
    parser.add_argument("--expert-ids", type=int, nargs="+")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--launches", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Race production MoE candidates, then qualify the selected plans",
    )
    args = parser.parse_args()
    if min(*args.counts, args.repeats, args.launches) <= 0:
        parser.error("counts, repeats and launches must be positive")
    device = torch.device(args.device)
    if device.type != "cuda":
        parser.error("qualification requires an assigned CUDA device")
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device)
    if (props.major, props.minor) not in ((12, 0), (12, 1)):
        parser.error("IQ2_XS qualification targets SM120/SM121")
    torch.backends.cuda.matmul.allow_tf32 = False
    root = Path(__file__).resolve().parents[3]
    source_files = sorted(
        [
            *root.glob("flashinfer/experimental/b12x/moe/**/*.py"),
            root / "flashinfer/experimental/b12x/_lib/intrinsics.py",
            root / "flashinfer/experimental/b12x/_lib/quant/iq2_xs.py",
            root / "flashinfer/experimental/b12x/_lib/quant/block_codec.py",
            root / "flashinfer/experimental/b12x/testing/iq2_xxs_reference.py",
            root / "flashinfer/experimental/b12x/testing/iq2_xs_reference.py",
            root / "flashinfer/experimental/b12x/testing/q8_0_reference.py",
            root / "benchmarks/experimental/b12x/benchmark_iq2_xs_moe.py",
            root / "benchmarks/experimental/b12x/iq2_xs_checkpoint.py",
            root / "benchmarks/experimental/b12x/moe_preparation.py",
        ]
    )
    provenance = {
        "command": sys.argv,
        "worktree": str(root),
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "source_sha256": hashlib.sha256(
            b"".join(
                str(p.relative_to(root)).encode() + p.read_bytes() for p in source_files
            )
        ).hexdigest(),
        "torch": torch.__version__,
        "cutlass_dsl": importlib.metadata.version("nvidia-cutlass-dsl"),
        "gpu": str(props),
        "uuid": str(props.uuid),
        "snapshot": str(args.snapshot.resolve()),
        "metric": "microseconds per CUDA graph replay; lower is better",
    }
    with args.output.open("x") as output, torch.cuda.device(device):
        for layer in args.layers:
            for tp in args.tp_sizes:
                for rank in range(tp):
                    loaded = load_iq2_xs_layer(
                        args.snapshot,
                        layer=layer,
                        tp_size=tp,
                        tp_rank=rank,
                        expert_ids=None
                        if args.expert_ids is None
                        else tuple(args.expert_ids),
                    )
                    experts, payload = prepare_experts(
                        loaded, device, activation=loaded.activation
                    )
                    source_digest = hashlib.sha256(
                        loaded.weights.w13.numpy().tobytes()
                        + loaded.weights.w2.numpy().tobytes()
                    ).hexdigest()
                    with PreparationSession(
                        device=device, autotune=args.tune, compile_workers=1
                    ) as session:
                        cases = [
                            (max(args.counts), tuple(args.counts), "packed", False),
                            (max(args.counts), tuple(args.counts), "packed", True),
                        ]
                        cases += [
                            (m, (m,), "direct", False) for m in args.counts if m <= 8
                        ]
                        if args.tune:
                            cases = [
                                (
                                    m,
                                    tuple(n for n in args.counts if n <= m),
                                    "auto",
                                    False,
                                )
                                for m in sorted(
                                    {
                                        max(args.counts),
                                        *(m for m in args.counts if m <= 8),
                                    }
                                )
                            ]
                        for capacity, counts, route_mode, deterministic in cases:
                            results = qualify_capacity(
                                loaded,
                                experts,
                                session,
                                capacity=capacity,
                                counts=counts,
                                route_mode=route_mode,
                                deterministic=deterministic,
                                repeats=args.repeats,
                                launches=args.launches,
                                activation=loaded.activation,
                                autotune=args.tune,
                            )
                            for result in results:
                                row = {
                                    **provenance,
                                    "layer": layer,
                                    "tp_size": tp,
                                    "tp_rank": rank,
                                    "payload_bytes": payload,
                                    "weights_sha256": source_digest,
                                    "expert_ids": loaded.expert_ids,
                                    **result,
                                }
                                output.write(json.dumps(row) + "\n")
                                output.flush()
                                print(
                                    json.dumps(
                                        {
                                            key: row[key]
                                            for key in (
                                                "layer",
                                                "tp_size",
                                                "tp_rank",
                                                "tokens",
                                                "route_mode",
                                                "pattern",
                                                "cosine",
                                                "relative_l2",
                                                "graph_us",
                                                "config",
                                            )
                                        }
                                    ),
                                    flush=True,
                                )


if __name__ == "__main__":
    from b12x.testing.memory import absorb_small_page_fragments

    absorb_small_page_fragments()
    main()
