"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch


@dataclass(frozen=True)
class BenchConfig:
    batch_size: int
    num_qo_heads: int
    num_kv_heads: int
    qo_len: int
    kv_len: int
    head_dim: int
    causal: bool

    @property
    def label(self) -> str:
        mode = "causal" if self.causal else "noncausal"
        return (
            f"B{self.batch_size}-Hq{self.num_qo_heads}-Hkv{self.num_kv_heads}-"
            f"Sq{self.qo_len}-Sk{self.kv_len}-D{self.head_dim}-{mode}"
        )


CANONICAL_CONFIGS = (
    BenchConfig(4, 8, 8, 4096, 4096, 128, False),
    BenchConfig(2, 16, 16, 4096, 4096, 128, False),
    BenchConfig(1, 8, 8, 32768, 32768, 128, False),
    BenchConfig(4, 8, 8, 4096, 4096, 128, True),
)


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in ("1", "true", "yes", "causal"):
        return True
    if normalized in ("0", "false", "no", "noncausal"):
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def _parse_shape(value: str) -> BenchConfig:
    fields = value.split(",")
    if len(fields) != 7:
        raise argparse.ArgumentTypeError(
            "shape must be B,Hq,Hkv,Sq,Sk,D,causal, for example "
            "4,8,8,4096,4096,128,false"
        )
    try:
        config = BenchConfig(
            *(int(field) for field in fields[:6]), _parse_bool(fields[6])
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    _validate_config(config)
    return config


def _validate_config(config: BenchConfig) -> None:
    dimensions = (
        config.batch_size,
        config.num_qo_heads,
        config.num_kv_heads,
        config.qo_len,
        config.kv_len,
        config.head_dim,
    )
    if any(value <= 0 for value in dimensions):
        raise ValueError(f"all shape dimensions must be positive, got {config}")
    if config.num_qo_heads % config.num_kv_heads != 0:
        raise ValueError("num_qo_heads must be divisible by num_kv_heads")
    if config.head_dim != 128:
        raise ValueError("QK MXFP8 / PV NVFP4 attention requires head_dim=128")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the FlashInfer SM120 QK MXFP8 / PV NVFP4 attention "
            "kernel with prequantized and inclusive timing."
        )
    )
    parser.add_argument(
        "--shape",
        action="append",
        type=_parse_shape,
        help=(
            "B,Hq,Hkv,Sq,Sk,D,causal. May be repeated. The four canonical "
            "shapes are used when omitted."
        ),
    )
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--return-lse", action="store_true")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument(
        "--rounds",
        type=int,
        default=5,
        help="Number of repeated rounds; candidate/reference order alternates.",
    )
    parser.add_argument(
        "--compare-nvfp4",
        action="store_true",
        help="Also time FlashInfer's pure-NVFP4 attention as an informational baseline.",
    )
    parser.add_argument(
        "--no-cuda-graph",
        action="store_true",
        help="Use ordinary launches for attention-only timing.",
    )
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def _quantile(sorted_values: list[float], fraction: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _summarize(samples_ms: list[float]) -> dict[str, float | int]:
    ordered = sorted(samples_ms)
    median = statistics.median(ordered)
    return {
        "samples": len(ordered),
        "median_ms": median,
        "p25_ms": _quantile(ordered, 0.25),
        "p75_ms": _quantile(ordered, 0.75),
        "p95_ms": _quantile(ordered, 0.95),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "relative_iqr": (
            (_quantile(ordered, 0.75) - _quantile(ordered, 0.25)) / median
            if median
            else 0.0
        ),
    }


def _measure_gpu_ms(
    fn: Callable[[], object],
    *,
    warmup: int,
    repeat: int,
    use_cuda_graph: bool,
) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    replay: Callable[[], object] = fn
    graph = None
    if use_cuda_graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        torch.cuda.synchronize()
        replay = graph.replay

    samples = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return samples


def _attention_flops(config: BenchConfig) -> int:
    if config.causal:
        valid_pairs = sum(
            max(0, min(config.kv_len, row + config.kv_len - config.qo_len + 1))
            for row in range(config.qo_len)
        )
    else:
        valid_pairs = config.qo_len * config.kv_len
    return 4 * config.batch_size * config.num_qo_heads * valid_pairs * config.head_dim


def _add_throughput(
    stats: dict[str, float | int], config: BenchConfig
) -> dict[str, float | int]:
    stats = dict(stats)
    stats["attention_tflops"] = (
        _attention_flops(config) / float(stats["median_ms"]) / 1.0e9
    )
    return stats


def _nvidia_smi_snapshot() -> dict[str, str] | None:
    fields = (
        "name,driver_version,clocks.current.sm,clocks.max.sm,temperature.gpu,power.draw"
    )
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                f"--query-gpu={fields}",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    values = [value.strip() for value in output.splitlines()[0].split(",")]
    keys = (
        "name",
        "driver_version",
        "sm_clock_mhz",
        "max_sm_clock_mhz",
        "temperature_c",
        "power_w",
    )
    return dict(zip(keys, values, strict=True))


def _git_metadata() -> dict[str, object]:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {}
    return {"commit": sha, "dirty": dirty}


def _bench_config(
    config: BenchConfig,
    *,
    dtype: torch.dtype,
    return_lse: bool,
    warmup: int,
    repeat: int,
    rounds: int,
    use_cuda_graph: bool,
    compare_nvfp4: bool,
) -> dict[str, object]:
    import flashinfer

    torch.manual_seed(123)
    q = torch.randn(
        config.batch_size,
        config.num_qo_heads,
        config.qo_len,
        config.head_dim,
        dtype=dtype,
        device="cuda",
    )
    k = torch.randn(
        config.batch_size,
        config.num_kv_heads,
        config.kv_len,
        config.head_dim,
        dtype=dtype,
        device="cuda",
    )
    v = torch.randn_like(k)
    sm_scale = config.head_dim**-0.5

    quantized = flashinfer.qk_mxfp8_pv_nvfp4_attention_sm120_quantize_qkv(q, k, v)
    padded_qo_len = quantized[0].shape[2]
    out = torch.empty(
        config.batch_size,
        config.num_qo_heads,
        padded_qo_len,
        config.head_dim,
        dtype=dtype,
        device="cuda",
    )
    lse = (
        torch.empty(
            config.batch_size,
            config.num_qo_heads,
            padded_qo_len,
            dtype=torch.float32,
            device="cuda",
        )
        if return_lse
        else None
    )

    def qk_attention() -> object:
        return flashinfer.qk_mxfp8_pv_nvfp4_attention_sm120_fwd(
            *quantized,
            sm_scale=sm_scale,
            causal=config.causal,
            out=out,
            lse=lse,
            return_lse=return_lse,
            unpadded_q_len=config.qo_len,
            unpadded_k_len=config.kv_len,
        )

    def quantize_only() -> object:
        return flashinfer.qk_mxfp8_pv_nvfp4_attention_sm120_quantize_qkv(q, k, v)

    def end_to_end() -> object:
        current = flashinfer.qk_mxfp8_pv_nvfp4_attention_sm120_quantize_qkv(q, k, v)
        return flashinfer.qk_mxfp8_pv_nvfp4_attention_sm120_fwd(
            *current,
            sm_scale=sm_scale,
            causal=config.causal,
            return_lse=return_lse,
            unpadded_q_len=config.qo_len,
            unpadded_k_len=config.kv_len,
        )

    # Build both JIT modules before collecting any timed samples.
    qk_attention()
    nvfp4_attention = None
    if compare_nvfp4:
        nvfp4_quantized = flashinfer.nvfp4_attention_sm120_quantize_qkv(q, k, v)
        nvfp4_out = torch.empty_like(out)
        nvfp4_lse = torch.empty_like(lse) if lse is not None else None

        def _nvfp4_attention() -> object:
            return flashinfer.nvfp4_attention_sm120_fwd(
                *nvfp4_quantized,
                sm_scale=sm_scale,
                causal=config.causal,
                per_block_mean=True,
                out=nvfp4_out,
                lse=nvfp4_lse,
                return_lse=return_lse,
                unpadded_k_len=config.kv_len,
            )

        nvfp4_attention = _nvfp4_attention
        nvfp4_attention()
    torch.cuda.synchronize()

    qk_samples: list[float] = []
    nvfp4_samples: list[float] = []
    for round_index in range(rounds):
        paths = [("qk_mxfp8_pv_nvfp4", qk_attention)]
        if nvfp4_attention is not None:
            paths.append(("nvfp4", nvfp4_attention))
        if round_index % 2:
            paths.reverse()
        for name, fn in paths:
            samples = _measure_gpu_ms(
                fn,
                warmup=warmup,
                repeat=repeat,
                use_cuda_graph=use_cuda_graph,
            )
            if name == "qk_mxfp8_pv_nvfp4":
                qk_samples.extend(samples)
            else:
                nvfp4_samples.extend(samples)

    quantize_samples = _measure_gpu_ms(
        quantize_only, warmup=warmup, repeat=repeat, use_cuda_graph=False
    )
    inclusive_samples = _measure_gpu_ms(
        end_to_end, warmup=warmup, repeat=repeat, use_cuda_graph=False
    )

    result: dict[str, object] = {
        "shape": asdict(config),
        "label": config.label,
        "qk_mxfp8_pv_nvfp4_attention": _add_throughput(_summarize(qk_samples), config),
        "qkv_quantization": _summarize(quantize_samples),
        "inclusive": _add_throughput(_summarize(inclusive_samples), config),
    }
    if nvfp4_samples:
        nvfp4_stats = _add_throughput(_summarize(nvfp4_samples), config)
        result["nvfp4_attention"] = nvfp4_stats
        result["qk_mxfp8_latency_ratio_vs_nvfp4"] = float(
            result["qk_mxfp8_pv_nvfp4_attention"]["median_ms"]
        ) / float(nvfp4_stats["median_ms"])
    return result


def _print_result(result: dict[str, object]) -> None:
    candidate = result["qk_mxfp8_pv_nvfp4_attention"]
    print(
        f"{result['label']}: {candidate['median_ms']:.4f} ms, "
        f"{candidate['attention_tflops']:.1f} TFLOP/s "
        f"(p25={candidate['p25_ms']:.4f}, p75={candidate['p75_ms']:.4f}, "
        f"p95={candidate['p95_ms']:.4f})"
    )
    if "nvfp4_attention" in result:
        baseline = result["nvfp4_attention"]
        print(
            f"  pure NVFP4: {baseline['median_ms']:.4f} ms, "
            f"{baseline['attention_tflops']:.1f} TFLOP/s; "
            f"QK-MXFP8/PV-NVFP4 ratio={result['qk_mxfp8_latency_ratio_vs_nvfp4']:.3f}x"
        )
    quantization = result["qkv_quantization"]
    inclusive = result["inclusive"]
    print(
        f"  quantization={quantization['median_ms']:.4f} ms, "
        f"inclusive={inclusive['median_ms']:.4f} ms"
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        print("Skipping: CUDA is required.")
        return
    capability = torch.cuda.get_device_capability()
    if capability not in ((12, 0), (12, 1)):
        print(f"Skipping: SM120/SM121 is required, got {capability}.")
        return
    if args.warmup < 0 or args.repeat <= 0 or args.rounds <= 0:
        raise ValueError(
            "warmup must be nonnegative; repeat and rounds must be positive"
        )

    configs = tuple(args.shape) if args.shape else CANONICAL_CONFIGS
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    metadata = {
        "git": _git_metadata(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_capability": capability,
        "dtype": args.dtype,
        "return_lse": args.return_lse,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "rounds": args.rounds,
        "cuda_graph": not args.no_cuda_graph,
        "gpu_before": _nvidia_smi_snapshot(),
    }

    results = []
    for config in configs:
        result = _bench_config(
            config,
            dtype=dtype,
            return_lse=args.return_lse,
            warmup=args.warmup,
            repeat=args.repeat,
            rounds=args.rounds,
            use_cuda_graph=not args.no_cuda_graph,
            compare_nvfp4=args.compare_nvfp4,
        )
        _print_result(result)
        results.append(result)
    metadata["gpu_after"] = _nvidia_smi_snapshot()

    payload = {"metadata": metadata, "results": results}
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
