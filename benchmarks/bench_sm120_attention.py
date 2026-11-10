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

import argparse
import csv
import math
import statistics
import sys
from dataclasses import dataclass
from typing import Sequence

import torch


DEFAULT_CONFIGS = (
    (4, 8, 8, 4096, 128, False),
    (1, 8, 8, 32768, 128, False),
)
PER_BLOCK_MEAN = True
CSV_FIELDS = (
    "batch_size",
    "num_qo_heads",
    "num_kv_heads",
    "seq_len",
    "head_dim",
    "causal",
    "return_lse",
    "dtype",
    "attention_only_ms",
    "attention_only_tflops",
    "attention_only_cuda_graph",
    "end_to_end_ms",
    "end_to_end_attention_tflops",
    "fp8_attention_only_ms",
    "fp8_attention_only_tflops",
    "nvfp4_speedup_over_fp8",
    "warmup",
    "repeat",
)


def _patch_cutlass_dsl_operand_major_mode() -> None:
    try:
        import cutlass.cute as cute
        from cutlass.cute.nvgpu.tcgen05 import OperandMajorMode
    except ImportError:
        return
    if not hasattr(cute.nvgpu, "OperandMajorMode"):
        cute.nvgpu.OperandMajorMode = OperandMajorMode


@dataclass(frozen=True)
class BenchConfig:
    batch_size: int
    num_qo_heads: int
    num_kv_heads: int
    seq_len: int
    head_dim: int
    causal: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Benchmark FlashInfer SM120 NVFP4 attention and FP8 FMHAv2.")
    )
    parser.add_argument(
        "--batch-size",
        "--batch_size",
        type=int,
        nargs="+",
        default=None,
        help="Batch size(s). Scalars are broadcast across other shape lists.",
    )
    parser.add_argument(
        "--num-heads",
        "--num_heads",
        type=int,
        nargs="+",
        default=None,
        help="Legacy alias that sets equal query/output and KV head counts.",
    )
    parser.add_argument(
        "--num-qo-heads",
        "--num_qo_heads",
        type=int,
        nargs="+",
        default=None,
        help="Number of query/output attention heads.",
    )
    parser.add_argument(
        "--num-kv-heads",
        "--num_kv_heads",
        type=int,
        nargs="+",
        default=None,
        help="Number of key/value attention heads.",
    )
    parser.add_argument(
        "--seq-len",
        "--seq_len",
        type=int,
        nargs="+",
        default=None,
        help="Sequence length(s). Must be a multiple of 128.",
    )
    parser.add_argument(
        "--head-dim",
        "--head_dim",
        type=int,
        nargs="+",
        default=None,
        help="Head dimension(s). SM120 NVFP4 and FP8 attention support 64 or 128.",
    )
    causal_group = parser.add_mutually_exclusive_group()
    causal_group.add_argument(
        "--causal",
        dest="causal",
        action="store_true",
        default=None,
        help="Benchmark causal attention.",
    )
    causal_group.add_argument(
        "--no-causal",
        dest="causal",
        action="store_false",
        help="Benchmark non-causal attention.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=("float16", "bfloat16"),
        default="bfloat16",
        help="Input/output dtype before NVFP4 quantization.",
    )
    parser.add_argument(
        "--return-lse",
        action="store_true",
        help="Compute log-sum-exp in addition to the attention output.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warmup iterations for each measured path.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=20,
        help="Number of measured iterations for each measured path.",
    )
    parser.add_argument(
        "--no-attention-cuda-graph",
        action="store_true",
        help=(
            "Measure attention-only with normal CUDA events. By default the "
            "attention-only path uses CUDA Graph replay to report pure GPU "
            "kernel time without TVM FFI launch overhead."
        ),
    )
    parser.add_argument(
        "--save-results-to",
        type=str,
        default=None,
        help="Optional path to save benchmark results as CSV.",
    )
    return parser.parse_args()


def skip_unless_sm120() -> None:
    if not torch.cuda.is_available():
        print("Skipping: NVFP4 attention SM120 benchmark requires CUDA.")
        sys.exit(0)

    capability = torch.cuda.get_device_capability()
    if capability not in ((12, 0), (12, 1)):
        print(f"Current device capability: {capability}.")
        print(
            "Skipping: NVFP4 attention SM120 benchmark requires compute capability (12, 0) or (12, 1)."
        )
        sys.exit(0)


def torch_dtype(dtype: str) -> torch.dtype:
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype}")


def expand_values(name: str, values: Sequence[int] | None, default: int) -> list[int]:
    if values is None:
        return [default]
    if not values:
        raise ValueError(f"{name} must not be empty")
    return list(values)


def broadcast_shape_lists(values: dict[str, list[int]]) -> dict[str, list[int]]:
    max_len = max(len(v) for v in values.values())
    out = {}
    for name, vals in values.items():
        if len(vals) == 1:
            out[name] = vals * max_len
        elif len(vals) == max_len:
            out[name] = vals
        else:
            raise ValueError(
                f"{name} has {len(vals)} values, expected 1 or {max_len} values"
            )
    return out


def build_configs(args: argparse.Namespace) -> list[BenchConfig]:
    if args.num_heads is not None and (
        args.num_qo_heads is not None or args.num_kv_heads is not None
    ):
        raise ValueError(
            "--num-heads cannot be combined with --num-qo-heads or --num-kv-heads"
        )

    num_qo_heads = args.num_heads if args.num_heads is not None else args.num_qo_heads
    num_kv_heads = args.num_heads if args.num_heads is not None else args.num_kv_heads
    has_custom_shape = any(
        arg is not None
        for arg in (
            args.batch_size,
            num_qo_heads,
            num_kv_heads,
            args.seq_len,
            args.head_dim,
        )
    )
    if not has_custom_shape:
        return [
            BenchConfig(
                batch_size=batch_size,
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                seq_len=seq_len,
                head_dim=head_dim,
                causal=args.causal if args.causal is not None else causal,
            )
            for batch_size, num_qo_heads, num_kv_heads, seq_len, head_dim, causal in DEFAULT_CONFIGS
        ]

    values = broadcast_shape_lists(
        {
            "batch_size": expand_values("batch_size", args.batch_size, 4),
            "num_qo_heads": expand_values("num_qo_heads", num_qo_heads, 8),
            "num_kv_heads": expand_values("num_kv_heads", num_kv_heads, 8),
            "seq_len": expand_values("seq_len", args.seq_len, 4096),
            "head_dim": expand_values("head_dim", args.head_dim, 128),
        }
    )
    causal = args.causal if args.causal is not None else False
    return [
        BenchConfig(
            batch_size=values["batch_size"][idx],
            num_qo_heads=values["num_qo_heads"][idx],
            num_kv_heads=values["num_kv_heads"][idx],
            seq_len=values["seq_len"][idx],
            head_dim=values["head_dim"][idx],
            causal=causal,
        )
        for idx in range(len(values["batch_size"]))
    ]


def validate_config(config: BenchConfig) -> None:
    if config.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {config.batch_size}")
    if config.num_qo_heads <= 0 or config.num_kv_heads <= 0:
        raise ValueError(
            "num_qo_heads and num_kv_heads must be positive, "
            f"got {config.num_qo_heads} and {config.num_kv_heads}"
        )
    if (
        config.num_qo_heads < config.num_kv_heads
        or config.num_qo_heads % config.num_kv_heads != 0
    ):
        raise ValueError(
            "num_qo_heads must be greater than or equal to and divisible by "
            f"num_kv_heads, got {config.num_qo_heads} and {config.num_kv_heads}"
        )
    if config.seq_len <= 0 or config.seq_len % 128 != 0:
        raise ValueError(
            f"seq_len must be positive and divisible by 128, got {config.seq_len}"
        )
    if config.head_dim not in (64, 128):
        raise ValueError(f"head_dim must be 64 or 128, got {config.head_dim}")


def attention_flops(config: BenchConfig) -> float:
    factor = 2 if config.causal else 4
    return (
        factor
        * config.batch_size
        * config.num_qo_heads
        * config.seq_len
        * config.seq_len
        * config.head_dim
    )


def tflops_per_sec(config: BenchConfig, ms: float) -> float:
    return attention_flops(config) / ms / 1e9


def dtype_label(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def quantize_e4m3(x: torch.Tensor) -> tuple[torch.Tensor, float]:
    scale = max(
        x.abs().max().float().item() / torch.finfo(torch.float8_e4m3fn).max,
        1.0e-12,
    )
    return (x / scale).to(torch.float8_e4m3fn), scale


def median_gpu_ms(
    fn,
    warmup: int,
    repeat: int,
    use_cuda_graph: bool = False,
    cold_l2_cache: bool = True,
    num_iters_within_graph: int = 1,
) -> float:
    _patch_cutlass_dsl_operand_major_mode()
    import flashinfer.testing

    measurements = flashinfer.testing.bench_gpu_time(
        fn,
        dry_run_iters=warmup,
        repeat_iters=repeat,
        cold_l2_cache=cold_l2_cache,
        use_cuda_graph=use_cuda_graph,
        num_iters_within_graph=num_iters_within_graph,
    )
    return statistics.median(measurements)


def bench_config(
    config: BenchConfig,
    dtype: torch.dtype,
    warmup: int,
    repeat: int,
    attention_cuda_graph: bool,
    return_lse: bool,
) -> dict[str, object]:
    _patch_cutlass_dsl_operand_major_mode()
    import flashinfer

    validate_config(config)
    torch.manual_seed(123)

    q = torch.randn(
        config.batch_size,
        config.num_qo_heads,
        config.seq_len,
        config.head_dim,
        dtype=dtype,
        device="cuda",
    )
    k = torch.randn(
        config.batch_size,
        config.num_kv_heads,
        config.seq_len,
        config.head_dim,
        dtype=dtype,
        device="cuda",
    )
    v = torch.randn_like(k)

    sm_scale = 1.0 / math.sqrt(config.head_dim)
    out = torch.empty_like(q)
    lse = None
    if return_lse:
        lse = torch.empty(
            config.batch_size,
            config.num_qo_heads,
            config.seq_len,
            dtype=torch.float32,
            device="cuda",
        )

    quantized_qkv = flashinfer.nvfp4_attention_sm120_quantize_qkv(
        q, k, v, per_block_mean=PER_BLOCK_MEAN
    )
    flashinfer.nvfp4_attention_sm120_fwd(
        *quantized_qkv,
        sm_scale=sm_scale,
        causal=config.causal,
        per_block_mean=PER_BLOCK_MEAN,
        out=out,
        lse=lse,
        out_dtype=dtype,
        return_lse=return_lse,
    )
    torch.cuda.synchronize()

    def attention_only():
        return flashinfer.nvfp4_attention_sm120_fwd(
            *quantized_qkv,
            sm_scale=sm_scale,
            causal=config.causal,
            per_block_mean=PER_BLOCK_MEAN,
            out=out,
            lse=lse,
            out_dtype=dtype,
            return_lse=return_lse,
        )

    def end_to_end():
        qkv = flashinfer.nvfp4_attention_sm120_quantize_qkv(
            q, k, v, per_block_mean=PER_BLOCK_MEAN
        )
        return flashinfer.nvfp4_attention_sm120_fwd(
            *qkv,
            sm_scale=sm_scale,
            causal=config.causal,
            per_block_mean=PER_BLOCK_MEAN,
            out=out,
            lse=lse,
            out_dtype=dtype,
            return_lse=return_lse,
        )

    attention_only_ms = median_gpu_ms(
        attention_only,
        warmup,
        repeat,
        use_cuda_graph=attention_cuda_graph,
        cold_l2_cache=not attention_cuda_graph,
        num_iters_within_graph=1,
    )
    end_to_end_ms = median_gpu_ms(end_to_end, warmup, repeat)
    attention_only_tflops = tflops_per_sec(config, attention_only_ms)
    end_to_end_tflops = tflops_per_sec(config, end_to_end_ms)

    fp8_baseline_available = (
        config.num_qo_heads == config.num_kv_heads and dtype == torch.bfloat16
    )
    fp8_attention_only_ms = None
    fp8_attention_only_tflops = None
    nvfp4_speedup_over_fp8 = None
    if fp8_baseline_available:
        from flashinfer.prefill import fmha_v2_prefill_sm120

        q_bshd = q.permute(0, 2, 1, 3).contiguous()
        k_bshd = k.permute(0, 2, 1, 3).contiguous()
        v_bshd = v.permute(0, 2, 1, 3).contiguous()
        q_fp8, q_scale = quantize_e4m3(q_bshd)
        k_fp8, k_scale = quantize_e4m3(k_bshd)
        v_fp8, v_scale = quantize_e4m3(v_bshd)
        out_fp8 = torch.empty_like(q_bshd, dtype=dtype)
        lse_fp8 = None
        if return_lse:
            lse_fp8 = torch.empty(
                config.batch_size,
                config.seq_len,
                config.num_qo_heads,
                2,
                dtype=torch.float32,
                device=q.device,
            )
        scale_bmm1_d = torch.tensor(
            [q_scale * k_scale * sm_scale], dtype=torch.float32, device=q.device
        )
        scale_bmm2_d = torch.tensor([v_scale], dtype=torch.float32, device=q.device)

        def fp8_attention_only():
            return fmha_v2_prefill_sm120(
                q_fp8,
                k_fp8,
                v_fp8,
                out_fp8,
                num_heads=config.num_qo_heads,
                head_dim=config.head_dim,
                seq_len=config.seq_len,
                scale_softmax=1.0,
                scale_bmm1=q_scale * k_scale * sm_scale,
                scale_bmm2=v_scale,
                scale_bmm1_d=scale_bmm1_d,
                scale_bmm2_d=scale_bmm2_d,
                causal=config.causal,
                return_lse=return_lse,
                lse=lse_fp8,
            )

        fp8_attention_only()
        torch.cuda.synchronize()
        fp8_attention_only_ms = median_gpu_ms(
            fp8_attention_only,
            warmup,
            repeat,
            use_cuda_graph=attention_cuda_graph,
            cold_l2_cache=not attention_cuda_graph,
            num_iters_within_graph=1,
        )
        fp8_attention_only_tflops = tflops_per_sec(config, fp8_attention_only_ms)
        nvfp4_speedup_over_fp8 = fp8_attention_only_ms / attention_only_ms

    print(
        "nvfp4_attention_sm120 "
        f"B={config.batch_size} Hq={config.num_qo_heads} "
        f"Hkv={config.num_kv_heads} S={config.seq_len} "
        f"D={config.head_dim} causal={config.causal} dtype={dtype}: "
        f"return_lse={return_lse}, "
        f"attention_only={attention_only_ms:.3f} ms "
        f"({attention_only_tflops:.3f} TFLOPs/s, "
        f"cuda_graph={attention_cuda_graph}), "
        f"end_to_end={end_to_end_ms:.3f} ms "
        f"({end_to_end_tflops:.3f} attention-TFLOPs/s)"
    )
    if fp8_baseline_available:
        print(
            "fmha_v2_prefill_sm120_fp8 "
            f"B={config.batch_size} H={config.num_qo_heads} S={config.seq_len} "
            f"D={config.head_dim} causal={config.causal}: "
            f"attention_only={fp8_attention_only_ms:.3f} ms "
            f"({fp8_attention_only_tflops:.3f} TFLOPs/s, "
            f"cuda_graph={attention_cuda_graph}), "
            f"nvfp4_speedup={nvfp4_speedup_over_fp8:.3f}x"
        )
    else:
        print(
            "fmha_v2_prefill_sm120_fp8 "
            f"B={config.batch_size} Hq={config.num_qo_heads} "
            f"Hkv={config.num_kv_heads} S={config.seq_len} "
            f"D={config.head_dim} causal={config.causal}: "
            "attention_only=unavailable "
            "(requires equal Q/KV head counts and BF16 input/output)"
        )
    return {
        "batch_size": config.batch_size,
        "num_qo_heads": config.num_qo_heads,
        "num_kv_heads": config.num_kv_heads,
        "seq_len": config.seq_len,
        "head_dim": config.head_dim,
        "causal": config.causal,
        "return_lse": return_lse,
        "dtype": dtype_label(dtype),
        "attention_only_ms": attention_only_ms,
        "attention_only_tflops": attention_only_tflops,
        "attention_only_cuda_graph": attention_cuda_graph,
        "end_to_end_ms": end_to_end_ms,
        "end_to_end_attention_tflops": end_to_end_tflops,
        "fp8_attention_only_ms": fp8_attention_only_ms,
        "fp8_attention_only_tflops": fp8_attention_only_tflops,
        "nvfp4_speedup_over_fp8": nvfp4_speedup_over_fp8,
        "warmup": warmup,
        "repeat": repeat,
    }


def write_csv(path: str, rows: list[dict[str, object]]) -> None:
    with open(path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results saved to: {path}")


def main() -> None:
    args = parse_args()
    skip_unless_sm120()

    if args.warmup < 0:
        raise ValueError(f"warmup must be non-negative, got {args.warmup}")
    if args.repeat <= 0:
        raise ValueError(f"repeat must be positive, got {args.repeat}")

    dtype = torch_dtype(args.dtype)
    configs = build_configs(args)
    results = []
    for config in configs:
        results.append(
            bench_config(
                config,
                dtype,
                args.warmup,
                args.repeat,
                attention_cuda_graph=not args.no_attention_cuda_graph,
                return_lse=args.return_lse,
            )
        )

    if args.save_results_to:
        write_csv(args.save_results_to, results)


if __name__ == "__main__":
    main()
