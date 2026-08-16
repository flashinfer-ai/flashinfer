#!/usr/bin/env python3
"""Benchmark pre-routed Kimi-K3-shaped CuTe DSL W4A16 SiTU MoE.

The fixed workload is E=896, topK=16, H=3584, I=3072, situ_beta=4, and
situ_linear_beta=25. This is a W4A16 derivative of the Kimi-K3 shape, not a
claim about Kimi-K3's native weight format. Reuse one --autotune-cache across
fresh-process A/B/A arms to hold tactics fixed while keeping JIT caches private.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

E, TOP_K, H, I, BETA, LINEAR_BETA = 896, 16, 3584, 3072, 4.0, 25.0
DEFAULT_TOKENS = (1, 8, 64, 512, 2048, 4096, 8192)
torch: Any = None  # Imported after argparse so --help works on the control host.


def _emit(record: dict[str, Any], records: list[dict[str, Any]]) -> None:
    records.append(record)
    print("JSON " + json.dumps(record, default=str, sort_keys=True), flush=True)


def _tokens(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item) for item in value.split(",") if item)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "--tokens must be comma-separated integers"
        ) from error
    if (
        not parsed
        or any(item <= 0 for item in parsed)
        or len(set(parsed)) != len(parsed)
    ):
        raise argparse.ArgumentTypeError("--tokens must be unique positive integers")
    return parsed


def _command(command: list[str], cwd: Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            command, cwd=cwd, capture_output=True, text=True, timeout=15
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _environment(args: argparse.Namespace, flashinfer: Any) -> dict[str, Any]:
    imported = Path(flashinfer.__file__).resolve()
    repo = next((path for path in imported.parents if (path / ".git").exists()), None)
    props = torch.cuda.get_device_properties(0)
    smi = _command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,power.limit,clocks.max.sm",
            "--format=csv,noheader,nounits",
        ]
    )
    return {
        "record": "environment",
        "arm": args.arm,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "flashinfer_version": getattr(flashinfer, "__version__", None),
        "flashinfer_file": str(imported),
        "revision": args.revision
        or (_command(["git", "rev-parse", "HEAD"], repo) if repo else None),
        "cutlass_dsl": _version("nvidia-cutlass-dsl"),
        "cupti_python": _version("cupti-python"),
        "gpu_name": props.name,
        "compute_capability": [props.major, props.minor],
        "gpu_total_memory_bytes": props.total_memory,
        "nvidia_smi": smi,
        "container_image": os.environ.get("CONTAINER_IMAGE")
        or os.environ.get("NVIDIA_PYTORCH_VERSION"),
        "workload": {
            "experts": E,
            "top_k": TOP_K,
            "hidden": H,
            "intermediate": I,
            "situ_beta": BETA,
            "situ_linear_beta": LINEAR_BETA,
        },
        "measurement": {
            "timing": args.timing,
            "cold_l2": True,
            "cuda_graph": False,
            "tokens": args.tokens,
            "warmup": args.warmup,
            "iters": args.iters,
            "repeats": args.repeats,
            "autotune_cache": str(args.autotune_cache),
            "autotune_cache_preexisting": args.autotune_cache.is_file(),
        },
    }


def _make_weight(
    experts: int, rows: int, columns: int, chunk_experts: int, seed: int
) -> tuple[Any, Any]:
    from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout
    from flashinfer.fp4_quantization import fp4_quantize

    generator = torch.Generator(device="cuda").manual_seed(seed)
    global_scale = torch.ones(1, dtype=torch.float32, device="cuda")
    packed = scales = None
    scales_per_expert = 0
    for start in range(0, experts, chunk_experts):
        count = min(chunk_experts, experts - start)
        source = torch.empty(
            (count * rows, columns), dtype=torch.bfloat16, device="cuda"
        )
        source.normal_(0.0, 0.1, generator=generator)
        packed_chunk, scale_chunk = fp4_quantize(
            source,
            global_scale=global_scale,
            sf_vec_size=16,
            is_sf_swizzled_layout=True,
        )
        if packed is None:
            packed = torch.empty(
                (experts, rows, columns // 2), dtype=packed_chunk.dtype, device="cuda"
            )
            scales_per_expert = scale_chunk.numel() // count
            scales = torch.empty(
                experts * scales_per_expert, dtype=scale_chunk.dtype, device="cuda"
            )
        packed[start : start + count].copy_(
            packed_chunk.reshape(count, rows, columns // 2)
        )
        scales[start * scales_per_expert : (start + count) * scales_per_expert].copy_(
            scale_chunk.reshape(-1)
        )
        del source, packed_chunk, scale_chunk
    assert packed is not None and scales is not None
    return packed, convert_sf_to_mma_layout(
        scales, m=rows, k=columns, num_groups=experts, sf_vec_size=16
    )


def _make_tensors(args: argparse.Namespace) -> tuple[Any, ...]:
    max_tokens = max(args.tokens)
    generator = torch.Generator(device="cuda").manual_seed(20260815)
    x = torch.empty((max_tokens, H), dtype=torch.bfloat16, device="cuda")
    x.normal_(0.0, 0.1, generator=generator)
    selected = torch.arange(max_tokens * TOP_K, device="cuda").reshape(
        max_tokens, TOP_K
    )
    selected = selected.remainder(E).to(torch.int32)
    scales = torch.rand(
        (max_tokens, TOP_K), dtype=torch.float32, device="cuda", generator=generator
    )
    scales /= scales.sum(dim=-1, keepdim=True)
    print("SETUP quantizing W1", flush=True)
    w1, w1_sf = _make_weight(E, 2 * I, H, args.weight_chunk_experts, 20260816)
    torch.cuda.empty_cache()
    print("SETUP quantizing W2", flush=True)
    w2, w2_sf = _make_weight(E, H, I, args.weight_chunk_experts, 20260817)
    alpha = torch.ones(E, dtype=torch.float32, device="cuda")
    torch.cuda.synchronize()
    return x, selected, scales, w1, w1_sf, alpha, w2, w2_sf, alpha.clone()


def _run(wrapper: Any, inputs: tuple[Any, ...], tactic: Any = None) -> Any:
    x, selected, scales, w1, w1_sf, a1, w2, w2_sf, a2 = inputs
    return wrapper.run(
        x=x,
        x_sf=None,
        token_selected_experts=selected,
        token_final_scales=scales,
        w1_weight=w1,
        w1_weight_sf=w1_sf,
        w1_alpha=a1,
        fc2_input_scale=None,
        w2_weight=w2,
        w2_weight_sf=w2_sf,
        w2_alpha=a2,
        tactic=tactic,
    )


def _selected_tactic(wrapper: Any, inputs: tuple[Any, ...]) -> Any:
    from flashinfer.autotuner import AutoTuner

    runner = wrapper._w4a16_runner
    if runner is None:
        raise RuntimeError("W4A16 runner was not initialized")
    output = torch.empty((inputs[0].shape[0], H), dtype=torch.bfloat16, device="cuda")
    tuner_inputs = [*inputs, output]
    hit, _, tactic, _ = AutoTuner.get().search_cache(
        "CuteDslMoEWrapper::run::W4A16::Situ",
        [runner],
        tuple(tuple(tensor.shape) for tensor in tuner_inputs),
        runner.tuning_config,
        inputs=tuner_inputs,
    )
    if not hit:
        raise RuntimeError("autotune completed without a cache hit")
    return tactic


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, help="fresh-process A/B/A label")
    parser.add_argument("--revision", required=True, help="source commit under test")
    parser.add_argument("--tokens", type=_tokens, default=DEFAULT_TOKENS)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timing", choices=("event", "cupti"), default="event")
    parser.add_argument("--autotune-cache", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--weight-chunk-experts", type=int, default=16)
    return parser


def main() -> int:
    global torch
    parser = _parser()
    args = parser.parse_args()
    if args.warmup < 0 or args.iters <= 0 or args.repeats <= 0:
        parser.error("warmup must be nonnegative; iters and repeats must be positive")
    if args.weight_chunk_experts <= 0:
        parser.error("weight chunk experts must be positive")
    if args.timing == "cupti" and (len(args.tokens) != 1 or args.repeats != 1):
        parser.error(
            "CUPTI requires exactly one token shape and --repeats 1 per process"
        )
    try:
        import torch as torch_module
    except ImportError as error:
        parser.error(f"PyTorch is required: {error}")
    torch = torch_module
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    torch.cuda.set_device(0)
    props = torch.cuda.get_device_properties(0)
    if props.major != 10:
        parser.error(f"SM100-family GPU required; got SM{props.major}{props.minor}")
    if args.timing == "cupti":
        try:
            from cupti import cupti as _cupti  # noqa: F401

            if int((_version("cupti-python") or "0").split(".", 1)[0]) < 13:
                raise RuntimeError("cupti-python >= 13 is required")
        except Exception as error:
            parser.error(f"CUPTI timing unavailable: {error}")

    import flashinfer
    from flashinfer import ActivationType, CuteDslMoEWrapper
    from flashinfer.autotuner import autotune
    from flashinfer.testing.utils import bench_gpu_time

    records: list[dict[str, Any]] = []
    args.autotune_cache.parent.mkdir(parents=True, exist_ok=True)
    tune_mode = not args.autotune_cache.is_file()
    _emit(_environment(args, flashinfer), records)
    tensors = _make_tensors(args)
    wrapper = CuteDslMoEWrapper(
        E,
        TOP_K,
        H,
        I,
        activation_type=ActivationType.Swiglu,
        situ_beta=BETA,
        situ_linear_beta=LINEAR_BETA,
        quant_mode="w4a16",
        use_cuda_graph=False,
    )

    all_finite = True
    for tokens in args.tokens:
        inputs = (
            tensors[0][:tokens],
            tensors[1][:tokens],
            tensors[2][:tokens],
            *tensors[3:],
        )
        with autotune(tune_mode, cache=str(args.autotune_cache)):
            _run(wrapper, inputs)
        torch.cuda.synchronize()
        tactic = _selected_tactic(wrapper, inputs)
        _emit(
            {
                "record": "tactic",
                "arm": args.arm,
                "tokens": tokens,
                "cache_hit": True,
                "tactic": tactic,
            },
            records,
        )
        medians = []
        for repeat in range(1, args.repeats + 1):
            samples = [
                float(value)
                for value in bench_gpu_time(
                    fn=lambda *call_inputs: _run(wrapper, call_inputs, tactic),
                    dry_run_iters=args.warmup,
                    repeat_iters=args.iters,
                    enable_cupti=args.timing == "cupti",
                    use_cuda_graph=False,
                    input_args=inputs,
                    cold_l2_cache=True,
                    sleep_after_run=False,
                )
            ]
            median = statistics.median(samples)
            medians.append(median)
            _emit(
                {
                    "record": "result",
                    "arm": args.arm,
                    "tokens": tokens,
                    "repeat": repeat,
                    "timing": args.timing,
                    "cold_l2": True,
                    "cuda_graph": False,
                    "median_ms": median,
                    "samples_ms": samples,
                },
                records,
            )
        output = _run(wrapper, inputs, tactic).float()
        torch.cuda.synchronize()
        finite = bool(torch.isfinite(output).all().item())
        all_finite &= finite
        _emit(
            {
                "record": "summary",
                "arm": args.arm,
                "tokens": tokens,
                "run_medians_ms": medians,
                "median_of_run_medians_ms": statistics.median(medians),
                "output_all_finite": finite,
            },
            records,
        )

    _emit(
        {"record": "complete", "arm": args.arm, "all_outputs_finite": all_finite},
        records,
    )
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(records, default=str, indent=2) + "\n")
    return 0 if all_finite else 2


if __name__ == "__main__":
    sys.exit(main())
