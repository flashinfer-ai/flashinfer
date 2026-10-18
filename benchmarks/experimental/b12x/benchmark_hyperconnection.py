#!/usr/bin/env python3
"""Benchmark the Qwen3.8 Flash Next HyperConnection runtime API.

The measured path contains only public ``b12x.norm.hyperconnection`` entry
points.  PyTorch expressions are correctness oracles and are never timed.
Projection GEMMs are outside the HyperConnection API, so the full-chain case
uses precomputed projection outputs while preserving the dependency from
grouped RMSNorm into gate reduction.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
import pathlib
import shlex
import statistics
import subprocess
import sys
from collections.abc import Callable, Iterable
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import torch

from b12x.norm import hyperconnection as hc
from benchmarks.experimental.b12x.common import (
    bench_cuda_graph,
    bench_gpu_ms,
    capture_cuda_graph,
    make_l2_flush_fn,
    nvidia_smi_gpu_mode_snapshot,
    require_sm120,
)


STREAMS = 4
HIDDEN_SIZE = 2560
LOWRANK = 320
DTYPE = torch.bfloat16
EPS = 1.0e-6

DEFAULT_PROFILE_TOKENS = (1, 4, 16, 128, 512, 2048)
OPERATORS = (
    "grouped_rmsnorm",
    "scaled_silu",
    "gate_mean",
    "combine",
    "combine_norm",
    "full_chain",
)
MODES = ("eager", "graph")

_OUTPUT_NAMES = {
    "grouped_rmsnorm": ("normalized",),
    "scaled_silu": ("bottleneck",),
    "gate_mean": ("block_input",),
    "combine": ("combined",),
    "combine_norm": ("combined", "next_normalized"),
    "full_chain": (
        "normalized",
        "bottleneck",
        "block_input",
        "combined",
        "next_normalized",
    ),
}


@dataclass(frozen=True)
class Profile:
    """Fixed Qwen3.8 Flash Next HyperConnection geometry."""

    tokens: int
    streams: int = STREAMS
    hidden_size: int = HIDDEN_SIZE
    lowrank: int = LOWRANK

    def __post_init__(self) -> None:
        if self.tokens <= 0:
            raise ValueError(f"tokens must be positive, got {self.tokens}")
        if (self.streams, self.hidden_size, self.lowrank) != (
            STREAMS,
            HIDDEN_SIZE,
            LOWRANK,
        ):
            raise ValueError(
                "the benchmark contract is fixed at "
                f"S={STREAMS}, H={HIDDEN_SIZE}, R={LOWRANK}"
            )

    @property
    def label(self) -> str:
        return f"t{self.tokens}_s{self.streams}_h{self.hidden_size}_r{self.lowrank}"


@dataclass
class _Case:
    profile: Profile
    plan: hc.Plan
    binding: hc.Binding
    state: torch.Tensor
    norm_weight: torch.Tensor
    projected_down: torch.Tensor
    normalized_for_gate: torch.Tensor
    gate_logits: torch.Tensor
    block_output: torch.Tensor
    injection_logits: torch.Tensor

    def launch(self, operator: str) -> tuple[torch.Tensor, ...]:
        if operator == "grouped_rmsnorm":
            return (
                hc.run_grouped_rmsnorm(
                    self.state,
                    self.norm_weight,
                    eps=EPS,
                    binding=self.binding,
                ),
            )
        if operator == "scaled_silu":
            return (hc.run_scaled_silu(self.projected_down, binding=self.binding),)
        if operator == "gate_mean":
            return (
                hc.run_gate_mean(
                    self.normalized_for_gate,
                    self.gate_logits,
                    binding=self.binding,
                ),
            )
        if operator == "combine":
            return (
                hc.run_combine(
                    self.state,
                    self.block_output,
                    self.injection_logits,
                    plan=self.plan,
                ),
            )
        if operator == "combine_norm":
            return hc.run_combine_norm(
                self.state,
                self.block_output,
                self.injection_logits,
                self.norm_weight,
                eps=EPS,
                plan=self.plan,
            )
        if operator == "full_chain":
            normalized = hc.run_grouped_rmsnorm(
                self.state,
                self.norm_weight,
                eps=EPS,
                binding=self.binding,
            )
            bottleneck = hc.run_scaled_silu(
                self.projected_down,
                binding=self.binding,
            )
            block_input = hc.run_gate_mean(
                normalized,
                self.gate_logits,
                binding=self.binding,
            )
            combined, next_normalized = hc.run_combine_norm(
                self.state,
                self.block_output,
                self.injection_logits,
                self.norm_weight,
                eps=EPS,
                plan=self.plan,
            )
            return (
                normalized,
                bottleneck,
                block_input,
                combined,
                next_normalized,
            )
        raise ValueError(f"unknown operator {operator!r}")

    def reference(self, operator: str) -> tuple[torch.Tensor, ...]:
        streams = self.profile.streams
        if operator == "grouped_rmsnorm":
            return (
                hc.reference.grouped_rmsnorm(
                    self.state,
                    self.norm_weight,
                    streams=streams,
                    eps=EPS,
                ),
            )
        if operator == "scaled_silu":
            return (
                hc.reference.scaled_silu(
                    self.projected_down,
                    streams=streams,
                ),
            )
        if operator == "gate_mean":
            return (
                hc.reference.gate_mean(
                    self.normalized_for_gate,
                    self.gate_logits,
                    streams=streams,
                ),
            )
        if operator == "combine":
            return (
                hc.reference.combine(
                    self.state,
                    self.block_output,
                    self.injection_logits,
                    streams=streams,
                ),
            )
        if operator == "combine_norm":
            return hc.reference.combine_norm(
                self.state,
                self.block_output,
                self.injection_logits,
                self.norm_weight,
                streams=streams,
                eps=EPS,
            )
        if operator == "full_chain":
            normalized = hc.reference.grouped_rmsnorm(
                self.state,
                self.norm_weight,
                streams=streams,
                eps=EPS,
            )
            bottleneck = hc.reference.scaled_silu(
                self.projected_down,
                streams=streams,
            )
            block_input = hc.reference.gate_mean(
                normalized,
                self.gate_logits,
                streams=streams,
            )
            combined, next_normalized = hc.reference.combine_norm(
                self.state,
                self.block_output,
                self.injection_logits,
                self.norm_weight,
                streams=streams,
                eps=EPS,
            )
            return (
                normalized,
                bottleneck,
                block_input,
                combined,
                next_normalized,
            )
        raise ValueError(f"unknown operator {operator!r}")


def build_plan_binding(
    *,
    device: torch.device | str,
    tokens: int,
    policy=None,
) -> tuple[hc.Plan, hc.Binding]:
    """Build the public Caps -> plan -> bind lifecycle for one profile."""
    profile = Profile(tokens=tokens)
    device = torch.device(device)
    width = profile.streams * profile.hidden_size
    plan = hc.plan(
        hc.Caps(
            device=device,
            max_tokens=profile.tokens,
            hidden_size=profile.hidden_size,
            streams=profile.streams,
            lowrank=profile.lowrank,
            dtype=DTYPE,
        ),
        policy=policy,
    )
    binding = hc.bind(
        plan,
        tokens=profile.tokens,
        normalized=torch.empty(
            (profile.tokens, width),
            dtype=DTYPE,
            device=device,
        ),
        bottleneck=torch.empty(
            (profile.tokens, profile.lowrank),
            dtype=DTYPE,
            device=device,
        ),
        block_input=torch.empty(
            (profile.tokens, profile.hidden_size),
            dtype=DTYPE,
            device=device,
        ),
    )
    return plan, binding


def parse_name_filter(
    value: str,
    *,
    choices: tuple[str, ...],
    label: str,
) -> tuple[str, ...]:
    """Parse an ordered comma-separated filter, accepting ``all``."""
    items = tuple(part.strip() for part in value.split(",") if part.strip())
    if not items:
        raise ValueError(f"{label} filter must not be empty")
    if "all" in items:
        if items != ("all",):
            raise ValueError(f"{label}=all cannot be combined with named values")
        return choices
    unknown = tuple(item for item in items if item not in choices)
    if unknown:
        raise ValueError(
            f"unknown {label}: {', '.join(unknown)}; choices are {', '.join(choices)}"
        )
    return tuple(dict.fromkeys(items))


def parse_token_filter(value: str) -> tuple[int, ...]:
    """Parse an ordered, positive comma-separated token profile filter."""
    try:
        tokens = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise ValueError("tokens must be a comma-separated list of integers") from error
    if not tokens:
        raise ValueError("tokens filter must not be empty")
    if any(token <= 0 for token in tokens):
        raise ValueError(f"token counts must be positive, got {tokens}")
    return tuple(dict.fromkeys(tokens))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokens",
        default=",".join(map(str, DEFAULT_PROFILE_TOKENS)),
        help="Comma-separated live-token profiles.",
    )
    parser.add_argument(
        "--operators",
        default="all",
        help=f"Comma-separated operators or all: {', '.join(OPERATORS)}.",
    )
    parser.add_argument(
        "--modes",
        default="all",
        help=f"Comma-separated timing modes or all: {', '.join(MODES)}.",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--l2-flush", action="store_true")
    parser.add_argument("--l2-flush-bytes", type=int, default=0)
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        help="Optional JSONL output path; every record is also printed.",
    )
    return parser


def _randn_bf16(
    shape: tuple[int, ...],
    *,
    generator: torch.Generator,
    device: torch.device,
    divisor: float,
) -> torch.Tensor:
    return (
        torch.randn(
            shape,
            generator=generator,
            dtype=torch.float32,
            device=device,
        )
        .div_(divisor)
        .to(DTYPE)
        .contiguous()
    )


def _make_case(
    profile: Profile,
    *,
    seed: int,
    device: torch.device,
    policy=None,
) -> _Case:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    width = profile.streams * profile.hidden_size
    plan, binding = build_plan_binding(
        device=device,
        tokens=profile.tokens,
        policy=policy,
    )
    return _Case(
        profile=profile,
        plan=plan,
        binding=binding,
        state=_randn_bf16(
            (profile.tokens, width),
            generator=generator,
            device=device,
            divisor=3.0,
        ),
        norm_weight=_randn_bf16(
            (width,),
            generator=generator,
            device=device,
            divisor=32.0,
        ),
        projected_down=_randn_bf16(
            (profile.tokens, profile.lowrank),
            generator=generator,
            device=device,
            divisor=2.0,
        ),
        normalized_for_gate=_randn_bf16(
            (profile.tokens, width),
            generator=generator,
            device=device,
            divisor=2.0,
        ),
        gate_logits=_randn_bf16(
            (profile.tokens, width),
            generator=generator,
            device=device,
            divisor=2.0,
        ),
        block_output=_randn_bf16(
            (profile.tokens, profile.hidden_size),
            generator=generator,
            device=device,
            divisor=4.0,
        ),
        injection_logits=_randn_bf16(
            (profile.tokens, profile.streams),
            generator=generator,
            device=device,
            divisor=2.0,
        ),
    )


def _validate_outputs(
    *,
    operator: str,
    actual: tuple[torch.Tensor, ...],
    expected: tuple[torch.Tensor, ...],
) -> dict[str, dict[str, float | int | bool]]:
    names = _OUTPUT_NAMES[operator]
    if len(actual) != len(names) or len(expected) != len(names):
        raise RuntimeError(f"{operator} returned an unexpected output arity")
    result: dict[str, dict[str, float | int | bool]] = {}
    for name, actual_tensor, expected_tensor in zip(
        names,
        actual,
        expected,
        strict=True,
    ):
        finite = bool(torch.isfinite(actual_tensor.float()).all().item())
        nonzero = int(torch.count_nonzero(actual_tensor).item())
        if not finite or nonzero == 0:
            raise AssertionError(
                f"{operator}.{name} failed finite/nonzero gate: "
                f"finite={finite}, nonzero={nonzero}"
            )
        atol = 2.0e-2 if "normalized" in name else 8.0e-3
        difference = actual_tensor.float() - expected_tensor.float()
        max_abs = float(difference.abs().max().item())
        torch.testing.assert_close(
            actual_tensor,
            expected_tensor,
            rtol=0,
            atol=atol,
            msg=lambda message, op=operator, output=name: (
                f"{op}.{output} failed the Torch reference gate: {message}"
            ),
        )
        result[name] = {
            "finite": finite,
            "nonzero": nonzero,
            "max_abs": max_abs,
            "rtol": 0.0,
            "atol": atol,
        }
    return result


def _correctness_gate(case: _Case, operator: str) -> dict[str, object]:
    actual = case.launch(operator)
    expected = case.reference(operator)
    torch.cuda.synchronize(case.state.device)
    return {
        "status": "passed",
        "oracle": "b12x.norm.hyperconnection.reference",
        "outputs": _validate_outputs(
            operator=operator,
            actual=actual,
            expected=expected,
        ),
    }


def _eager_samples_us(
    launch: Callable[[], object],
    *,
    warmup: int,
    samples: int,
    l2_flush: Callable[[], None] | None,
) -> list[float]:
    for _ in range(warmup):
        if l2_flush is not None:
            l2_flush()
        launch()
    torch.cuda.synchronize()
    return [
        bench_gpu_ms(launch, warmup=0, iters=1, l2_flush=l2_flush) * 1000.0
        for _ in range(samples)
    ]


def _graph_samples_us(
    case: _Case,
    operator: str,
    *,
    warmup: int,
    samples: int,
    l2_flush: Callable[[], None] | None,
) -> tuple[list[float], dict[str, object], dict[str, object]]:
    captured_outputs: list[torch.Tensor] = []

    def launch_and_retain() -> tuple[torch.Tensor, ...]:
        outputs = case.launch(operator)
        captured_outputs[:] = outputs
        return outputs

    graph = capture_cuda_graph(launch_and_retain, warmup=warmup)
    if not captured_outputs:
        raise RuntimeError(f"{operator} graph capture did not retain outputs")
    correctness = {
        "status": "passed",
        "oracle": "b12x.norm.hyperconnection.reference",
        "outputs": _validate_outputs(
            operator=operator,
            actual=tuple(captured_outputs),
            expected=case.reference(operator),
        ),
    }
    addresses = [tensor.data_ptr() for tensor in captured_outputs]
    for tensor in captured_outputs:
        tensor.fill_(float("nan"))
    torch.cuda.synchronize(case.state.device)
    allocated_before = torch.cuda.memory_allocated(case.state.device)
    graph.replay()
    torch.cuda.synchronize(case.state.device)
    allocated_after = torch.cuda.memory_allocated(case.state.device)
    replay_delta = allocated_after - allocated_before
    if replay_delta != 0:
        raise AssertionError(f"{operator} graph replay allocated {replay_delta} bytes")
    replay_addresses = [tensor.data_ptr() for tensor in captured_outputs]
    if replay_addresses != addresses:
        raise AssertionError(f"{operator} graph output addresses changed on replay")
    correctness = {
        "status": "passed",
        "oracle": "b12x.norm.hyperconnection.reference",
        "outputs": _validate_outputs(
            operator=operator,
            actual=tuple(captured_outputs),
            expected=case.reference(operator),
        ),
        "graph_replay_after_output_poison": True,
    }
    stats = bench_cuda_graph(graph, replays=samples, l2_flush=l2_flush)
    graph_contract = {
        "stable_output_addresses": True,
        "output_addresses": addresses,
        "replay_allocation_delta_bytes": replay_delta,
        "replay_after_output_poison": True,
    }
    return list(stats["replay_us"]), graph_contract, correctness


def _percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _timing_summary(samples_us: list[float]) -> dict[str, object]:
    if not samples_us:
        raise ValueError("timing sample list must not be empty")
    return {
        "unit": "us",
        "count": len(samples_us),
        "minimum": min(samples_us),
        "p10": _percentile(samples_us, 0.10),
        "median": statistics.median(samples_us),
        "p90": _percentile(samples_us, 0.90),
        "maximum": max(samples_us),
        "raw_samples_us": samples_us,
    }


def _git_provenance() -> dict[str, object]:
    repo = pathlib.Path(__file__).resolve().parents[3]

    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=repo,
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"

    return {
        "worktree": str(repo),
        "commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "dirty_paths": git("status", "--short").splitlines(),
    }


def _device_provenance(device: torch.device) -> dict[str, object]:
    properties = torch.cuda.get_device_properties(device)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    return {
        "logical_device": device.index,
        "cuda_visible_devices": visible,
        "name": properties.name,
        "uuid": str(getattr(properties, "uuid", "unknown")),
        "capability": list(torch.cuda.get_device_capability(device)),
        "total_memory_bytes": properties.total_memory,
    }


class _Emitter:
    def __init__(self, output: pathlib.Path | None) -> None:
        self._output = output
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8"):
                pass

    def emit(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, sort_keys=True)
        print(line, flush=True)
        if self._output is not None:
            with self._output.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")


def _validate_device(device: torch.device) -> None:
    if not hc.is_supported(device):
        raise SystemExit(
            "b12x.norm.hyperconnection reports that its runtime kernels are "
            f"unsupported on {device}"
        )


def _parse_args(
    parser: argparse.ArgumentParser,
    argv: Iterable[str] | None,
) -> tuple[argparse.Namespace, tuple[int, ...], tuple[str, ...], tuple[str, ...]]:
    args = parser.parse_args(argv)
    if args.warmup <= 0 or args.samples <= 0:
        parser.error("warmup and samples must be positive")
    try:
        tokens = parse_token_filter(args.tokens)
        operators = parse_name_filter(
            args.operators,
            choices=OPERATORS,
            label="operators",
        )
        modes = parse_name_filter(args.modes, choices=MODES, label="modes")
    except ValueError as error:
        parser.error(str(error))
    return args, tokens, operators, modes


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_parser()
    args, tokens, operators, modes = _parse_args(parser, argv)
    device = require_sm120()
    _validate_device(device)
    l2_flush = make_l2_flush_fn(args.l2_flush, args.l2_flush_bytes)
    emitter = _Emitter(args.output)
    command_argv = list(sys.argv[1:] if argv is None else argv)
    gpu_mode_before = nvidia_smi_gpu_mode_snapshot()
    emitter.emit(
        {
            "type": "provenance",
            "schema": "b12x-hyperconnection-benchmark-v1",
            "command": shlex.join(
                [sys.executable, str(pathlib.Path(__file__).resolve()), *command_argv]
            ),
            "git": _git_provenance(),
            "device": _device_provenance(device),
            "gpu_mode_before": gpu_mode_before,
            "torch_version": str(torch.__version__),
            "torch_cuda_version": torch.version.cuda,
            "runtime_requirements": list(hc.META.requires),
            "timed_path": "b12x.norm.hyperconnection public runtime API",
            "fallback_path": "none",
            "reference_timed": False,
            "profiles": [asdict(Profile(token)) for token in tokens],
            "operators": list(operators),
            "modes": list(modes),
            "warmup": args.warmup,
            "samples": args.samples,
            "seed": args.seed,
            "l2_flush": args.l2_flush,
            "l2_flush_bytes": args.l2_flush_bytes,
        }
    )

    with torch.inference_mode():
        for profile_index, token_count in enumerate(tokens):
            profile = Profile(token_count)
            case_seed = args.seed + profile_index * 1009
            case = _make_case(
                profile,
                seed=case_seed,
                device=device,
            )
            for operator in operators:
                eager_correctness = _correctness_gate(case, operator)
                for mode in modes:
                    graph_contract: dict[str, object] | None = None
                    if mode == "eager":
                        samples_us = _eager_samples_us(
                            lambda active_case=case, op=operator: active_case.launch(
                                op
                            ),
                            warmup=args.warmup,
                            samples=args.samples,
                            l2_flush=l2_flush,
                        )
                        correctness = eager_correctness
                    else:
                        samples_us, graph_contract, correctness = _graph_samples_us(
                            case,
                            operator,
                            warmup=args.warmup,
                            samples=args.samples,
                            l2_flush=l2_flush,
                        )
                    emitter.emit(
                        {
                            "type": "result",
                            "schema": "b12x-hyperconnection-benchmark-v1",
                            "profile": {"label": profile.label, **asdict(profile)},
                            "operator": operator,
                            "mode": mode,
                            "case_seed": case_seed,
                            "correctness": correctness,
                            "graph_contract": graph_contract,
                            "timing": _timing_summary(samples_us),
                        }
                    )
            del case

    emitter.emit(
        {
            "type": "completion",
            "schema": "b12x-hyperconnection-benchmark-v1",
            "status": "passed",
            "gpu_mode_before": gpu_mode_before,
            "gpu_mode_after": nvidia_smi_gpu_mode_snapshot(),
        }
    )


if __name__ == "__main__":
    from b12x.testing.memory import absorb_small_page_fragments

    absorb_small_page_fragments()
    main()
