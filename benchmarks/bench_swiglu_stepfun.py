# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""Compare native TRTLLM SwiGLU Step and OpenAI SwiGLU routed MoE.

Example on a B300::

    python benchmarks/bench_swiglu_stepfun.py --output-dir /tmp/swiglu_stepfun

The default shape is H=4096, I=1536, E=64, top-k=8, with 8/64/512/2048
tokens. Each precision uses identical native weights and prepared activations
for OA (alpha=1.702, beta=1, limit=7) and Step limits 7 and 16. Preparation,
quantization and independent tactic tuning precede timing. Timed GPU work
includes precomputed routing, fused FC1 activation, FC2 and finalization.

Alternating measurement order reduces drift. CUPTI reports the elapsed span
from the first GPU activity to the last activity in one CUDA graph replay,
including overlapping kernels and gaps. If CUPTI is unavailable, the recorded
method is CUDA graph timing with events. The cache policy applies to both
tactic tuning and timing: the default warm policy does not flush L2, while
the cold policy flushes before each sample. Tuning uses multiple graph replays
to select tactics for the same execution regime as timing. The report contains
all samples and paired bootstrap intervals across round medians.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
import warnings

import torch

import flashinfer
from flashinfer.artifacts import ArtifactPath, CheckSumHash
from flashinfer.autotuner import AutoTuner, autotune
from flashinfer.jit.env import FLASHINFER_CUBIN_DIR
from flashinfer.fused_moe import (
    BackendOptions,
    CakeStepFunConfig,
    ExecutionConfig,
    ExpertConfig,
    MoEActivationPack,
    MoEConfig,
    MoEWeightPack,
    QuantConfig,
    QuantFormat,
    RoutingConfig,
    RoutingInputMode,
    SwiGLU,
    SwiGLUStep,
    TrtllmBf16Config,
    TrtllmFp4Config,
    TrtllmFp8BlockConfig,
    TrtllmFp8PerTensorConfig,
)
from flashinfer.fused_moe.prepare import _activation_param_view
from flashinfer.fused_moe.runners import (
    CakeStepFunRunner,
    TrtllmBf16RoutedRunner,
    TrtllmFp4RoutedRunner,
    TrtllmFp8BlockRunner,
    TrtllmFp8PerTensorRunner,
)
from flashinfer.testing.utils import bench_gpu_time


MODES = {
    "bf16": (TrtllmBf16Config, TrtllmBf16RoutedRunner, QuantFormat.BF16),
    "mxfp8": (TrtllmFp8BlockConfig, TrtllmFp8BlockRunner, QuantFormat.MXFP8),
    "nvfp4": (TrtllmFp4Config, TrtllmFp4RoutedRunner, QuantFormat.NVFP4),
    "fp8": (
        TrtllmFp8PerTensorConfig,
        TrtllmFp8PerTensorRunner,
        QuantFormat.FP8PerTensor,
    ),
}
# The per-tensor FP8 mode is opt-in; the default precision set is unchanged.
DEFAULT_PRECISIONS = ("bf16", "mxfp8", "nvfp4")
# Alternative FC1 backends measured next to the native trtllm-gen kernels on the
# same weights, activations and autotune policy. Cake runs NVFP4 StepFun only.
BACKENDS = {
    "trtllm": None,
    "cake": (CakeStepFunConfig, CakeStepFunRunner),
}
# Per-tensor FP8 calibration as in the FlashInfer tests: 448 / amax(sample) for
# the activations and 448 / 256 for the FC1 output (StepFun is bounded by L^2).
FP8_CALIBRATION_TOKENS = 2048
FP8_INTERMEDIATE_SCALE_GLOBAL = 448.0 / 256.0


def baseline_variant(precision):
    # Per-tensor FP8 cannot represent the OpenAI SwiGLU scalars; its paired
    # baseline is the default SwiGLU.
    return "swiglu" if precision == "fp8" else "oai"


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--precisions",
        nargs="+",
        choices=tuple(MODES),
        default=list(DEFAULT_PRECISIONS),
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=tuple(BACKENDS),
        default=["trtllm"],
        help="FC1 backends measured on the same inputs; cake applies to nvfp4 only.",
    )
    parser.add_argument("--tokens", nargs="+", type=int, default=[8, 64, 512, 2048])
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=1536)
    parser.add_argument("--experts", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--step-limits", nargs="+", type=float, default=[7.0, 16.0])
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument(
        "--cache-policy",
        choices=("warm", "cold"),
        default="warm",
        help="Use the same L2 policy for tactic selection and measured samples.",
    )
    parser.add_argument("--tuning-graph-replays", type=int, default=20)
    parser.add_argument("--tuning-repeat", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1091)
    parser.add_argument("--pdl", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cupti", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--regression-threshold", type=float, default=0.05)
    parser.add_argument(
        "--check-performance",
        action="store_true",
        help="Exit nonzero unless every upper 95%% paired interval is within the threshold.",
    )
    args = parser.parse_args()
    if (
        min(
            *args.tokens,
            args.hidden,
            args.intermediate,
            args.experts,
            args.top_k,
            args.rounds,
            args.iterations,
            args.warmups,
            args.tuning_graph_replays,
            args.tuning_repeat,
        )
        <= 0
    ):
        parser.error("shapes and timing counts must be positive")
    if args.hidden % 128 or args.intermediate % 128:
        parser.error("hidden and intermediate must be divisible by 128")
    if args.top_k > args.experts or args.experts > 256:
        parser.error("require top-k <= experts <= 256")
    if any(not 0 < limit < float("inf") for limit in args.step_limits):
        parser.error("Step limits must be positive and finite")
    if args.rounds < 3 or args.regression_threshold < 0:
        parser.error("require rounds >= 3 and regression-threshold >= 0")
    if args.cache_policy == "cold" and not args.cupti:
        parser.error("cold graph timing requires CUPTI to flush L2 before each sample")
    if "cake" in args.backends and "nvfp4" not in args.precisions:
        parser.error("the cake backend runs NVFP4 only; include nvfp4 in --precisions")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output-dir must be empty or absent")
    return args


def command_output(*command):
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else result.stderr.strip()


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value):
        return repr(value)
    try:
        return list(value)
    except TypeError:
        return repr(value)


def tensor_digest(tensor):
    data = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def make_activations(args, tokens, backend, quant, hidden_states_scale_global=None):
    # Reset the seed so every precision also sees the same canonical tokens.
    torch.manual_seed(args.seed + tokens)
    hidden = torch.randn(tokens, args.hidden, device="cuda", dtype=torch.bfloat16)
    ids = (
        torch.arange(tokens * args.top_k, device="cuda", dtype=torch.int32).reshape(
            tokens, args.top_k
        )
        % args.experts
    )
    weights = torch.rand(tokens, args.top_k, device="cuda", dtype=torch.float32)
    weights /= weights.sum(-1, keepdim=True)
    if quant.weight is QuantFormat.BF16:
        prepared, scales = hidden, None
    elif quant.weight is QuantFormat.FP8PerTensor:
        prepared, scales = backend.prepare_activations(
            hidden, hidden_states_scale_global=hidden_states_scale_global
        )
    else:
        prepared, scales = backend.prepare_activations(hidden, quant=quant)
    pack = MoEActivationPack(
        hidden_states_q=prepared,
        hidden_states_scale=scales,
        topk_ids=ids,
        topk_weights=weights,
        routing_input_mode=RoutingInputMode.UnpackedPrecomputed,
    )
    hashes = {
        "canonical_activations": tensor_digest(hidden),
        "expert_ids": tensor_digest(ids),
        "expert_weights": tensor_digest(weights),
    }
    return pack, hashes


def make_candidate(
    args, tokens, runner_cls, backend, quant, activation, base_view, act, arm="trtllm"
):
    config = MoEConfig(
        routing=RoutingConfig(num_experts=args.experts, top_k=args.top_k),
        quant=quant,
        experts=ExpertConfig(
            intermediate_size=args.intermediate, local_num_experts=args.experts
        ),
        activation=activation,
        backend=BackendOptions(candidates=(backend(),)),
        execution=ExecutionConfig(enable_pdl=args.pdl, tune_max_num_tokens=tokens),
    )
    runner = runner_cls(config, torch.device("cuda", torch.cuda.current_device()))
    runner.check_support()
    runner.build()
    # Native weight tensors are shared by all candidates. Only activation ABI
    # scalars differ; this helper is the one used by public weight preparation.
    view = {
        key: value
        for key, value in base_view.items()
        if key not in ("gemm1_alpha", "gemm1_beta", "gemm1_clamp_limit")
    }
    view.update(
        _activation_param_view(activation, args.experts, act.hidden_states_q.device)
    )
    if arm == "cake":
        # The Cake FC1 kernels read an explicit per-expert limit in raw accumulator
        # units (no implicit default), as CakeStepFunConfig.prepare_weights stores it.
        view["gemm1_clamp_limit"] = (
            activation.limit / base_view["output1_scale_gate_scalar"]
        ).contiguous()
    elif quant.weight is QuantFormat.FP8PerTensor and isinstance(
        activation, SwiGLUStep
    ):
        # Per-tensor FP8 clamps raw accumulators; convert the logical limit as
        # TrtllmFp8PerTensorConfig.prepare_weights does.
        view["gemm1_clamp_limit"] = (
            activation.limit / base_view["output1_scales_gate_scalar"]
        ).contiguous()
    native = MoEWeightPack({runner.backend_key: view})
    packed = runner.pack_inputs(act, native)
    launch_kwargs = runner.launch_kwargs_for(packed)
    tuner = AutoTuner.get()
    # The production runner profiles cold L2 by default. This benchmark's
    # sustained graph timing needs tactics selected under the same cache
    # policy; change only the copied config, preserving its input/routing rules.
    tuning_config = dataclasses.replace(
        runner.tuning_config_for(packed),
        use_cuda_graph=True,
        use_cold_l2_cache=args.cache_policy == "cold",
        use_cold_l2_graph_replay=False,
        cuda_graph_profile_replays=args.tuning_graph_replays,
        profiling_repeat=args.tuning_repeat,
    )
    with autotune(tuning_buckets=(tokens,)):
        _, tactic = tuner.choose_one(
            custom_op=f"moe_swiglu_stepfun_{runner.backend_key}",
            runners=[runner],
            inputs=packed,
            tuning_config=tuning_config,
            **launch_kwargs,
        )
    # Initialize tactic workspaces before CUDA graph capture.
    runner.forward(packed, tactic=tactic, do_preparation=True, **launch_kwargs)
    for _ in range(args.warmups):
        runner.forward(packed, tactic=tactic, **launch_kwargs)
    torch.cuda.synchronize()
    output = runner.forward(packed, tactic=tactic, **launch_kwargs)
    torch.cuda.synchronize()
    if not torch.isfinite(output).all().item():
        raise RuntimeError(f"nonfinite output from {activation!r}")
    return runner, packed, tactic, launch_kwargs


def measure(args, candidate):
    runner, packed, tactic, kwargs = candidate
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        times = bench_gpu_time(
            lambda: runner.forward(packed, tactic=tactic, **kwargs),
            dry_run_iters=args.warmups,
            repeat_iters=args.iterations,
            enable_cupti=args.cupti,
            use_cuda_graph=True,
            cold_l2_cache=args.cache_policy == "cold",
        )
    messages = [str(w.message) for w in caught]
    fallback = any("Falling back" in message for message in messages)
    if args.cache_policy == "cold" and fallback:
        raise RuntimeError(
            "CUPTI fell back to graph events without an L2 flush; "
            "cannot measure the requested cold cache policy"
        )
    method = (
        "cupti_activity_span_cuda_graph"
        if args.cupti and not fallback
        else "cuda_events_cuda_graph"
    )
    samples = [float(value) * 1000 for value in times]
    if not samples or min(samples) <= 0:
        raise RuntimeError("GPU timer returned empty or nonpositive samples")
    return samples, method, messages


def paired_interval(oai, step, seed):
    rng = random.Random(seed)
    ratios = [b / a for a, b in zip(oai, step, strict=True)]
    bootstraps = sorted(
        statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(5000)
    )
    return statistics.median(ratios), bootstraps[125], bootstraps[4874]


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    imported_source = Path(flashinfer.__file__).resolve().parent.parent
    source = Path(
        os.environ.get("SWIGLU_FLASHINFER_SOURCE", imported_source)
    ).absolute()
    copy_manifest_path = os.environ.get("SWIGLU_SOURCE_MANIFEST_PATH")
    copy_manifest = None
    if copy_manifest_path and Path(copy_manifest_path).is_file():
        copy_manifest = json.loads(Path(copy_manifest_path).read_text())
        if Path(copy_manifest["git_source"]).absolute() != source:
            raise RuntimeError("Staged manifest names a different authoritative source")
        for name, expected in copy_manifest["files"].items():
            actual = hashlib.sha256((imported_source / name).read_bytes()).hexdigest()
            if actual != expected:
                raise RuntimeError(
                    f"Staged source differs from recorded snapshot: {name}"
                )
    git_commit = (
        copy_manifest["git_commit"]
        if copy_manifest and "git_commit" in copy_manifest
        else command_output("git", "-C", str(source), "rev-parse", "HEAD")
    )
    git_diff_sha256 = (
        copy_manifest["git_diff_sha256"]
        if copy_manifest and "git_diff_sha256" in copy_manifest
        else hashlib.sha256(
            command_output(
                "git",
                "-C",
                str(source),
                "-c",
                "diff.ignoreSubmodules=all",
                "diff",
                "HEAD",
            ).encode()
        ).hexdigest()
    )
    metadata = {
        "arguments": vars(args),
        "command": sys.argv,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": torch.cuda.get_device_capability(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "flashinfer": flashinfer.__version__,
        "flashinfer_source": str(source),
        "imported_flashinfer_source": str(imported_source),
        "copied_source_manifest": copy_manifest,
        "git_commit": git_commit,
        "git_diff_sha256": git_diff_sha256,
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "nvidia_smi": command_output("nvidia-smi"),
        "nvcc": command_output("nvcc", "--version"),
        "timing_scope": "native unified runner forward: routing, fused FC1 activation, FC2, finalize; preparation and input quantization excluded",
        "routing": "uniform round-robin expert ids; normalized FP32 weights; UnpackedPrecomputed",
        "cold_l2_cache": args.cache_policy == "cold",
        "autotune": "independent per activation, precision, backend, token count",
        "backends": list(args.backends),
        "tuning_policy": {
            "timer": "AutoTuner-selected v1 CUDA graph timer",
            "requested_timer_environment": os.environ.get("FLASHINFER_AUTOTUNE_TIMER"),
            "cold_l2_cache": args.cache_policy == "cold",
            "use_cold_l2_graph_replay": False,
            "cuda_graph_profile_replays": args.tuning_graph_replays,
            "profiling_repeat": args.tuning_repeat,
        },
        "measurement_policy": {
            "requested_timer": "cupti" if args.cupti else "cuda_events",
            "use_cuda_graph": True,
            "cold_l2_cache": args.cache_policy == "cold",
            "samples_per_round": args.iterations,
            "actual_methods": [],
        },
        "trtllm_gen_bmm_artifact_path": ArtifactPath.TRTLLM_GEN_BMM,
        "trtllm_gen_bmm_expected_checksum_sha256": CheckSumHash.TRTLLM_GEN_BMM,
        "flashinfer_cubin_dir": str(FLASHINFER_CUBIN_DIR),
        "environment": {
            key: value
            for key, value in os.environ.items()
            if key
            in (
                "FLASHINFER_CUBIN_DIR",
                "FLASHINFER_WORKSPACE_BASE",
                "FLASHINFER_CUDA_ARCH_LIST",
                "FLASHINFER_FORCE_MOE_KERNELS",
            )
        },
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=json_default) + "\n"
    )
    torch.manual_seed(args.seed)
    w1 = (
        torch.randn(
            args.experts,
            2 * args.intermediate,
            args.hidden,
            device="cuda",
            dtype=torch.bfloat16,
        )
        / args.hidden**0.5
    )
    w2 = (
        torch.randn(
            args.experts,
            args.hidden,
            args.intermediate,
            device="cuda",
            dtype=torch.bfloat16,
        )
        / args.intermediate**0.5
    )
    metadata["canonical_weights_sha256"] = {
        "w1": tensor_digest(w1),
        "w2": tensor_digest(w2),
    }
    checksum_path = FLASHINFER_CUBIN_DIR / ArtifactPath.TRTLLM_GEN_BMM / "checksums.txt"
    metadata["resolved_checksum_path"] = str(checksum_path.resolve())
    metadata["resolved_checksum_sha256"] = (
        hashlib.sha256(checksum_path.read_bytes()).hexdigest()
        if checksum_path.is_file()
        else None
    )
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=json_default) + "\n"
    )
    step_variants = {
        f"step_limit_{limit:g}": SwiGLUStep(limit=limit) for limit in args.step_limits
    }
    baselines = {"oai": SwiGLU(alpha=1.702, beta=1.0, limit=7.0), "swiglu": SwiGLU()}
    rows = []
    actual_methods = set()
    with (args.output_dir / "samples.jsonl").open("w") as raw:
        for precision in args.precisions:
            backend, runner_cls, fmt = MODES[precision]
            quant = QuantConfig(weight=fmt, activation=fmt)
            baseline = baseline_variant(precision)
            variants = {baseline: baselines[baseline], **step_variants}
            prepare_kwargs = dict(
                num_local_experts=args.experts,
                hidden_size=args.hidden,
                intermediate_size=args.intermediate,
            )
            hidden_states_scale_global = None
            if precision == "fp8":
                torch.manual_seed(args.seed)
                sample = torch.randn(
                    FP8_CALIBRATION_TOKENS,
                    args.hidden,
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                hidden_states_scale_global = 448.0 / sample.float().abs().max()
                prepare_kwargs["hidden_states_scale_global"] = hidden_states_scale_global
                prepare_kwargs["intermediate_scale_global"] = FP8_INTERMEDIATE_SCALE_GLOBAL
            elif precision != "bf16":
                prepare_kwargs["quant"] = quant
            base_view = backend.prepare_weights(w1, w2, **prepare_kwargs)
            arms = [
                arm
                for arm in args.backends
                if arm == "trtllm" or precision == "nvfp4"
            ]
            for tokens in args.tokens:
                act, hashes = make_activations(
                    args, tokens, backend, quant, hidden_states_scale_global
                )
                print(f"Preparing {precision} tokens={tokens}", flush=True)
                candidates = {
                    name: make_candidate(
                        args,
                        tokens,
                        runner_cls,
                        backend,
                        quant,
                        activation,
                        base_view,
                        act,
                    )
                    for name, activation in variants.items()
                }
                # Alternative backends share the view, activations and tuning policy;
                # each is paired with the native kernel on the same activation.
                for arm in arms:
                    if arm == "trtllm":
                        continue
                    arm_backend, arm_runner_cls = BACKENDS[arm]
                    candidates.update(
                        {
                            f"{arm}:{name}": make_candidate(
                                args,
                                tokens,
                                arm_runner_cls,
                                arm_backend,
                                quant,
                                activation,
                                base_view,
                                act,
                                arm=arm,
                            )
                            for name, activation in step_variants.items()
                        }
                    )
                variant_of = {
                    name: variants[name.split(":", 1)[-1]] for name in candidates
                }
                round_medians = {name: [] for name in candidates}
                methods = {name: set() for name in candidates}
                for round_idx in range(args.rounds):
                    order = list(candidates)
                    order = (
                        order[round_idx % len(order) :]
                        + order[: round_idx % len(order)]
                    )
                    if round_idx % 2:
                        order.reverse()
                    for position, name in enumerate(order):
                        samples, method, messages = measure(args, candidates[name])
                        med = statistics.median(samples)
                        round_medians[name].append(med)
                        methods[name].add(method)
                        actual_methods.add(method)
                        record = dict(
                            precision=precision,
                            backend=name.split(":", 1)[0] if ":" in name else "trtllm",
                            tokens=tokens,
                            variant=name,
                            activation=repr(variant_of[name]),
                            round=round_idx,
                            position=position,
                            method=method,
                            median_us=med,
                            samples_us=samples,
                            warnings=messages,
                            hashes=hashes,
                            tactic=candidates[name][2],
                            cache_policy=args.cache_policy,
                            tuning_graph_replays=args.tuning_graph_replays,
                            tuning_repeat=args.tuning_repeat,
                        )
                        raw.write(json.dumps(record, default=json_default) + "\n")
                        raw.flush()
                        print(
                            f"{precision} tokens={tokens} round={round_idx} {name}: {med:.3f} us ({method})",
                            flush=True,
                        )
                oai = round_medians[baseline]
                for name, step in round_medians.items():
                    if name == baseline:
                        continue
                    # Native step variants pair with the native baseline; another
                    # backend's step variant pairs with the native kernel on the
                    # same activation.
                    arm, _, variant_name = name.rpartition(":")
                    arm = arm or "trtllm"
                    reference_name = baseline if arm == "trtllm" else variant_name
                    reference = round_medians[reference_name]
                    ratio, low, high = paired_interval(reference, step, args.seed + tokens)
                    threshold = 1 + args.regression_threshold
                    status = (
                        "pass"
                        if high <= threshold
                        else "regression"
                        if low > threshold
                        else "inconclusive"
                    )
                    row = dict(
                        precision=precision,
                        backend=arm,
                        baseline=f"trtllm:{reference_name}",
                        candidate=f"{arm}:{variant_name}",
                        tokens=tokens,
                        hidden=args.hidden,
                        intermediate=args.intermediate,
                        experts=args.experts,
                        top_k=args.top_k,
                        step_limit=variant_of[name].limit,
                        pdl=args.pdl,
                        cache_policy=args.cache_policy,
                        tuning_graph_replays=args.tuning_graph_replays,
                        tuning_repeat=args.tuning_repeat,
                        oai_median_us=statistics.median(oai),
                        step_median_us=statistics.median(step),
                        baseline_median_us=statistics.median(reference),
                        candidate_median_us=statistics.median(step),
                        median_latency_ratio=statistics.median(step)
                        / statistics.median(reference),
                        paired_median_ratio=ratio,
                        ratio_ci95_low=low,
                        ratio_ci95_high=high,
                        strict_no_slowdown_status=(
                            "supported"
                            if high <= 1.0
                            else "slowdown"
                            if low > 1.0
                            else "inconclusive"
                        ),
                        regression_threshold=args.regression_threshold,
                        status=status,
                        timing_method=";".join(
                            sorted(methods[name] | methods[reference_name])
                        ),
                    )
                    rows.append(row)
                    print(json.dumps(row), flush=True)
                del candidates, act
            del base_view
            torch.cuda.empty_cache()
    with (args.output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    # Module build may have fetched the authenticated manifest after startup.
    metadata["resolved_checksum_sha256"] = (
        hashlib.sha256(checksum_path.read_bytes()).hexdigest()
        if checksum_path.is_file()
        else None
    )
    metadata["nvidia_smi_after"] = command_output("nvidia-smi")
    metadata["measurement_policy"]["actual_methods"] = sorted(actual_methods)
    metadata["tuning_policy"]["actual_timer"] = (
        "globaltimer_cuda_graph"
        if AutoTuner.get()._use_global_timer
        else "cuda_events_cuda_graph"
    )
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=json_default) + "\n"
    )
    AutoTuner.get().save_configs(str(args.output_dir / "autotune.json"))
    if args.check_performance and any(row["status"] != "pass" for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
