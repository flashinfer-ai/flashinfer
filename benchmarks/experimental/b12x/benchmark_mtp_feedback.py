#!/usr/bin/env python3
"""Benchmark Qwen3.8 Flash Next MTP feedback through its public API.

The benchmark matrix fixes the production geometry at four residual streams and hidden
size 2560, then covers single-token decode, a four-token speculative step, and
four prefill sizes including a padded-row boundary. Every case validates the
seeded PyTorch oracle before recording eager-launch and CUDA-graph-replay
samples.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any

import torch

from b12x.sequence import mtp_feedback as mtp

try:
    from benchmarks.experimental.b12x.common import (
        bench_cuda_graph,
        bench_gpu_ms,
        capture_cuda_graph,
        make_l2_flush_fn,
        nvidia_smi_gpu_mode_snapshot,
        require_sm120,
    )
except ModuleNotFoundError:
    from common import (  # type: ignore[no-redef]
        bench_cuda_graph,
        bench_gpu_ms,
        capture_cuda_graph,
        make_l2_flush_fn,
        nvidia_smi_gpu_mode_snapshot,
        require_sm120,
    )


_HIDDEN_SIZE = 2_560
_STREAMS = 4
_DTYPE = torch.bfloat16
PLANNER_CAPACITY_TOKENS = 4_096
_RTOL = 2.0e-2
_ATOL = 4.0e-2
_RESULT_KIND = "qwen38_flash_next_mtp_feedback_benchmark_v1"


@dataclass(frozen=True)
class Profile:
    """One live-token count in the Qwen3.8 Flash Next MTP workload."""

    name: str
    phase: str
    tokens: int


PROFILES = (
    Profile(name="decode-t1", phase="decode", tokens=1),
    Profile(name="spec-t4", phase="spec", tokens=4),
    Profile(name="prefill-t17", phase="prefill", tokens=17),
    Profile(name="prefill-t128", phase="prefill", tokens=128),
    Profile(name="prefill-t512", phase="prefill", tokens=512),
    Profile(name="prefill-t4096", phase="prefill", tokens=4096),
)
_PROFILE_BY_NAME = {profile.name: profile for profile in PROFILES}
_PHASES = tuple(dict.fromkeys(profile.phase for profile in PROFILES))


def _parse_filter(value: str) -> tuple[str, ...] | None:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items:
        raise argparse.ArgumentTypeError("expected 'all' or a comma-separated list")
    if len(items) == 1 and items[0] == "all":
        return None
    if "all" in items:
        raise argparse.ArgumentTypeError("'all' cannot be combined with named filters")
    if len(set(items)) != len(items):
        raise argparse.ArgumentTypeError("filter entries must be unique")
    return items


def _select_profiles(
    profile_names: tuple[str, ...] | None,
    phases: tuple[str, ...] | None,
) -> tuple[Profile, ...]:
    unknown_profiles = set(profile_names or ()) - set(_PROFILE_BY_NAME)
    if unknown_profiles:
        choices = ", ".join(_PROFILE_BY_NAME)
        unknown = ", ".join(sorted(unknown_profiles))
        raise ValueError(f"unknown profiles: {unknown}; choices: {choices}")
    unknown_phases = set(phases or ()) - set(_PHASES)
    if unknown_phases:
        choices = ", ".join(_PHASES)
        unknown = ", ".join(sorted(unknown_phases))
        raise ValueError(f"unknown phases: {unknown}; choices: {choices}")

    selected_names = set(profile_names) if profile_names is not None else None
    selected_phases = set(phases) if phases is not None else None
    selected = tuple(
        profile
        for profile in PROFILES
        if (selected_names is None or profile.name in selected_names)
        and (selected_phases is None or profile.phase in selected_phases)
    )
    if not selected:
        raise ValueError("profile and phase filters select no benchmark cases")
    return selected


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--profiles",
        type=_parse_filter,
        default=None,
        help=(
            "comma-separated profile names or 'all'; choices are "
            + ", ".join(_PROFILE_BY_NAME)
        ),
    )
    parser.add_argument(
        "--phases",
        type=_parse_filter,
        default=None,
        help="comma-separated phases or 'all'; choices are " + ", ".join(_PHASES),
    )
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="print the profile set as JSON and exit without requiring CUDA",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument(
        "--capacity-tokens",
        type=int,
        default=PLANNER_CAPACITY_TOKENS,
        help=(
            "fixed planner/buffer capacity shared by every selected live-token "
            f"profile (default: {PLANNER_CAPACITY_TOKENS})"
        ),
    )
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--eps", type=float, default=1.0e-6)
    parser.add_argument(
        "--flush-l2",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="evict L2 before each timed eager launch and graph replay",
    )
    parser.add_argument("--l2-flush-bytes", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.warmup < 1:
        parser.error("--warmup must be positive")
    if args.samples < 1:
        parser.error("--samples must be positive")
    if not math.isfinite(args.eps) or args.eps <= 0.0:
        parser.error("--eps must be finite and positive")
    if args.l2_flush_bytes < 0:
        parser.error("--l2-flush-bytes must be non-negative")
    try:
        args.selected_profiles = _select_profiles(args.profiles, args.phases)
    except ValueError as error:
        parser.error(str(error))
    if args.capacity_tokens < 1:
        parser.error("--capacity-tokens must be positive")
    largest_live_tokens = max(profile.tokens for profile in args.selected_profiles)
    if args.capacity_tokens < largest_live_tokens:
        parser.error(
            "--capacity-tokens must cover every selected profile; "
            f"need at least {largest_live_tokens}"
        )
    return args


def _randn_bf16(
    shape: tuple[int, ...],
    *,
    scale: float,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    return (
        torch.randn(
            shape,
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        .mul_(scale)
        .to(_DTYPE)
        .contiguous()
    )


def _make_binding(
    profile: Profile,
    *,
    seed: int,
    device: torch.device,
    capacity_tokens: int,
    policy=None,
) -> mtp.Binding:
    """Allocate and bind one case exclusively through the public lifecycle."""

    caps = mtp.Caps(
        device=device,
        max_tokens=capacity_tokens,
        hidden_size=_HIDDEN_SIZE,
        streams=_STREAMS,
        dtype=_DTYPE,
    )
    planned = mtp.plan(caps, policy=policy)
    (scratch_spec,) = planned.scratch_specs()
    generator = torch.Generator(device=device).manual_seed(seed)
    scratch = torch.empty(
        scratch_spec.shape,
        dtype=scratch_spec.dtype,
        device=scratch_spec.device,
    )
    output = torch.full(
        planned.output_shape(),
        float("nan"),
        dtype=_DTYPE,
        device=device,
    )
    return mtp.bind(
        planned,
        scratch=scratch,
        token_embedding=_randn_bf16(
            (capacity_tokens, _HIDDEN_SIZE),
            scale=0.4,
            generator=generator,
            device=device,
        ),
        multi_state=_randn_bf16(
            (capacity_tokens, _STREAMS, _HIDDEN_SIZE),
            scale=0.4,
            generator=generator,
            device=device,
        ),
        token_norm_weight=torch.nn.Parameter(
            _randn_bf16(
                (_HIDDEN_SIZE,),
                scale=0.05,
                generator=generator,
                device=device,
            ),
            requires_grad=False,
        ),
        state_norm_weight=torch.nn.Parameter(
            _randn_bf16(
                (_STREAMS * _HIDDEN_SIZE,),
                scale=0.05,
                generator=generator,
                device=device,
            ),
            requires_grad=False,
        ),
        embedding_fc_weight=torch.nn.Parameter(
            _randn_bf16(
                (_HIDDEN_SIZE, _HIDDEN_SIZE),
                scale=_HIDDEN_SIZE**-0.5,
                generator=generator,
                device=device,
            ),
            requires_grad=False,
        ),
        hidden_fc_weight=torch.nn.Parameter(
            _randn_bf16(
                (_HIDDEN_SIZE, _HIDDEN_SIZE),
                scale=_HIDDEN_SIZE**-0.5,
                generator=generator,
                device=device,
            ),
            requires_grad=False,
        ),
        output=output,
        tokens=profile.tokens,
    )


def _comparison_metrics(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    rtol: float = _RTOL,
    atol: float = _ATOL,
) -> dict[str, float | bool]:
    actual_f32 = actual.float()
    expected_f32 = expected.float()
    difference = actual_f32 - expected_f32
    expected_norm = torch.linalg.vector_norm(expected_f32).clamp_min(1.0e-20)
    cosine = torch.nn.functional.cosine_similarity(
        actual_f32.reshape(1, -1),
        expected_f32.reshape(1, -1),
    )
    return {
        "finite": bool(torch.isfinite(actual).all().item()),
        "nonzero": bool(torch.count_nonzero(actual).item()),
        "close": bool(torch.allclose(actual, expected, rtol=rtol, atol=atol)),
        "max_abs": float(difference.abs().max().item()),
        "relative_l2": float(
            (torch.linalg.vector_norm(difference) / expected_norm).item()
        ),
        "cosine": float(cosine[0].item()),
    }


def _require_correct(label: str, metrics: dict[str, float | bool]) -> None:
    failed = [name for name in ("finite", "nonzero", "close") if not metrics[name]]
    if failed:
        raise RuntimeError(
            f"{label} failed {', '.join(failed)} correctness gates: {metrics}"
        )


def _summary(samples_us: list[float]) -> dict[str, float]:
    if not samples_us:
        raise ValueError("at least one timing sample is required")
    ordered = sorted(float(sample) for sample in samples_us)
    return {
        "median_us": statistics.median(ordered),
        "p10_us": ordered[int(0.10 * (len(ordered) - 1))],
        "p90_us": ordered[int(0.90 * (len(ordered) - 1))],
        "min_us": ordered[0],
        "max_us": ordered[-1],
    }


def _eager_samples_us(
    launch,
    *,
    warmup: int,
    samples: int,
    l2_flush,
) -> list[float]:
    timings = [
        bench_gpu_ms(
            launch,
            warmup=warmup,
            iters=1,
            l2_flush=l2_flush,
        )
        * 1_000.0
    ]
    for _ in range(samples - 1):
        timings.append(
            bench_gpu_ms(
                launch,
                warmup=0,
                iters=1,
                l2_flush=l2_flush,
            )
            * 1_000.0
        )
    return timings


def _benchmark_profile(
    profile: Profile,
    *,
    seed: int,
    device: torch.device,
    eps: float,
    warmup: int,
    samples: int,
    l2_flush,
    capacity_tokens: int,
    policy=None,
) -> dict[str, Any]:
    binding = _make_binding(
        profile,
        seed=seed,
        device=device,
        capacity_tokens=capacity_tokens,
        policy=policy,
    )
    reference = mtp.reference.feedback(
        binding.token_embedding,
        binding.multi_state,
        binding.token_norm_weight,
        binding.state_norm_weight,
        binding.embedding_fc_weight,
        binding.hidden_fc_weight,
        eps=eps,
    )
    reference_finite = bool(torch.isfinite(reference).all().item())
    reference_nonzero = bool(torch.count_nonzero(reference).item())
    if not reference_finite or not reference_nonzero:
        raise RuntimeError(
            f"{profile.name} reference failed finite/nonzero gates: "
            f"finite={reference_finite}, nonzero={reference_nonzero}"
        )

    output_address = binding.output.data_ptr()
    scratch_address = binding.scratch.data_ptr()

    def launch() -> torch.Tensor:
        return mtp.run(binding, eps=eps)

    eager_output = launch()
    torch.cuda.synchronize(device)
    if eager_output.data_ptr() != output_address:
        raise RuntimeError(f"{profile.name} eager run did not return caller output")
    eager_metrics = _comparison_metrics(eager_output, reference)
    _require_correct(f"{profile.name} eager", eager_metrics)
    eager_samples = _eager_samples_us(
        launch,
        warmup=warmup,
        samples=samples,
        l2_flush=l2_flush,
    )

    graph = capture_cuda_graph(launch, warmup=warmup)
    binding.output.fill_(float("nan"))
    binding.scratch.fill_(0xFF)
    torch.cuda.synchronize(device)
    allocated_before = torch.cuda.memory_allocated(device)
    graph.replay()
    torch.cuda.synchronize(device)
    allocated_after = torch.cuda.memory_allocated(device)
    graph_metrics = _comparison_metrics(binding.output, reference)
    _require_correct(f"{profile.name} graph replay", graph_metrics)
    if allocated_after != allocated_before:
        raise RuntimeError(
            f"{profile.name} graph replay allocated storage: "
            f"before={allocated_before}, after={allocated_after}"
        )
    if binding.output.data_ptr() != output_address:
        raise RuntimeError(f"{profile.name} graph replay changed output storage")
    if binding.scratch.data_ptr() != scratch_address:
        raise RuntimeError(f"{profile.name} graph replay changed scratch storage")

    graph_samples = bench_cuda_graph(
        graph,
        replays=samples,
        l2_flush=l2_flush,
    )
    graph_replay_samples = graph_samples["replay_us"]
    scratch_bytes = sum(spec.nbytes for spec in binding.plan.scratch_specs())
    eager_summary = _summary(eager_samples)
    graph_summary = _summary(graph_replay_samples)
    return {
        "profile": profile.name,
        "phase": profile.phase,
        "seed": seed,
        "shape": {
            "tokens": profile.tokens,
            "capacity_tokens": capacity_tokens,
            "streams": _STREAMS,
            "hidden_size": _HIDDEN_SIZE,
            "dtype": "bfloat16",
        },
        "storage": {
            "scratch_bytes": scratch_bytes,
            "caller_output": True,
            "stable_output_address": binding.output.data_ptr() == output_address,
            "stable_scratch_address": binding.scratch.data_ptr() == scratch_address,
            "graph_replay_allocation_delta_bytes": allocated_after - allocated_before,
        },
        "correctness": {
            "passed": True,
            "reference_finite": reference_finite,
            "reference_nonzero": reference_nonzero,
            "eager": eager_metrics,
            "graph_replay_after_output_poison": graph_metrics,
            "graph_replay_after_scratch_poison": True,
            "rtol": _RTOL,
            "atol": _ATOL,
        },
        "timings": {
            "unit": "microseconds",
            "direction": "lower_is_better",
            "eager": eager_summary,
            "cuda_graph_replay": graph_summary,
            "graph_over_eager": (
                graph_summary["median_us"] / eager_summary["median_us"]
            ),
        },
        "raw_samples_us": {
            "eager": eager_samples,
            "cuda_graph_metadata": graph_samples["metadata_us"],
            "cuda_graph_replay": graph_replay_samples,
            "cuda_graph_step": graph_samples["step_us"],
        },
    }


def _git_value(*args: str) -> str:
    root = Path(__file__).resolve().parents[3]
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    return completed.stdout.strip()


def _profile_listing(profiles: tuple[Profile, ...]) -> list[dict[str, object]]:
    return [
        {"name": profile.name, "phase": profile.phase, "tokens": profile.tokens}
        for profile in profiles
    ]


def _profile_seed(base_seed: int, profile: Profile) -> int:
    return int(base_seed) + 10_007 * PROFILES.index(profile)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    profiles = args.selected_profiles
    if args.list_profiles:
        print(json.dumps(_profile_listing(profiles), indent=2))
        return
    if args.output is not None and args.output.expanduser().exists():
        raise SystemExit(f"refusing to overwrite benchmark result: {args.output}")

    require_sm120()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise SystemExit(f"--device must select CUDA, got {device}")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(device)
    device = torch.device("cuda", torch.cuda.current_device())

    properties = torch.cuda.get_device_properties(device)
    mode_before = nvidia_smi_gpu_mode_snapshot()
    l2_flush = make_l2_flush_fn(
        args.flush_l2,
        bytes_hint=args.l2_flush_bytes,
    )
    cases = []
    for profile in profiles:
        capacity_tokens = args.capacity_tokens
        cases.append(
            _benchmark_profile(
                profile,
                seed=_profile_seed(args.seed, profile),
                device=device,
                eps=args.eps,
                warmup=args.warmup,
                samples=args.samples,
                l2_flush=l2_flush,
                capacity_tokens=capacity_tokens,
            )
        )

    root = Path(__file__).resolve().parents[3]
    result = {
        "kind": _RESULT_KIND,
        "provenance": {
            "command": [sys.executable, *sys.argv],
            "cwd": os.getcwd(),
            "worktree": str(root),
            "git_commit": _git_value("rev-parse", "HEAD"),
            "git_branch": _git_value("branch", "--show-current"),
            "git_dirty": bool(_git_value("status", "--porcelain")),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "contract": {
            "model": "Qwen3.8 Flash Next",
            "operator": "b12x.sequence.mtp_feedback",
            "api_lifecycle": ["Caps", "plan", "bind", "run"],
            "streams": _STREAMS,
            "hidden_size": _HIDDEN_SIZE,
            "dtype": "bfloat16",
            "reference": "b12x.sequence.mtp_feedback.reference.feedback",
            "projection_backend": "cutedsl",
            "projection_specialization": "fixed_capacity_runtime_live_rows",
            "triton_role": "normalization_and_reduction_auxiliaries",
            "eager_timed": True,
            "cuda_graph_replay_timed": False,
            "timing_method": "stream_gated_events_v1",
            "raw_samples_preserved": True,
        },
        "hardware": {
            "device": str(device),
            "name": properties.name,
            "uuid": str(getattr(properties, "uuid", "")),
            "compute_capability": [properties.major, properties.minor],
            "total_memory_bytes": properties.total_memory,
            "gpu_mode_before": mode_before,
            "gpu_mode_after": nvidia_smi_gpu_mode_snapshot(),
        },
        "parameters": {
            "profiles": _profile_listing(profiles),
            "capacity_tokens": args.capacity_tokens,
            "warmup": args.warmup,
            "samples": args.samples,
            "seed": args.seed,
            "eps": args.eps,
            "flush_l2": args.flush_l2,
            "l2_flush_bytes_hint": args.l2_flush_bytes,
        },
        "cases": cases,
    }
    encoded = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as handle:
            handle.write(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    from b12x.testing.memory import absorb_small_page_fragments

    absorb_small_page_fragments()
    main()
