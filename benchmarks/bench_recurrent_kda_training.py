# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Cold-L2 CUPTI benchmark for the paired recurrent-KDA training API.

The forward timing includes the exact checkpoint-producing route plus the
full-FP32-state recurrence that produces the public final state. A strict
grouped low-head route materializes both C16 and C32 contexts in that one
forward call. The backward timing consumes the saved context and therefore
excludes forward recomputation.
"""

import argparse
import json
import math
import os
import platform
import subprocess
import warnings
from dataclasses import dataclass
from importlib.metadata import version as distribution_version
from pathlib import Path

import numpy as np
import torch

from flashinfer.kda_training import (
    recurrent_kda_training_backward,
    recurrent_kda_training_forward,
)
from flashinfer.testing import bench_gpu_time
from flashinfer.utils import get_compute_capability


@dataclass(frozen=True)
class _Shape:
    seq_lens: tuple[int, ...]
    num_qk_heads: int
    num_v_heads: int
    seed: int
    layout: str = "packed"


_EXACT_PACKED_SHAPES = {
    "packed_1024x8_h96": _Shape((1024,) * 8, 96, 96, 819208),
}

_PRIMARY_SHAPES = {
    "portfolio_01_hq4_hv8_t32768": _Shape((3200,) * 9 + (3968,), 4, 8, 24001),
    "portfolio_02_hq2_hv8_t18432": _Shape((2000,) * 8 + (2432,), 2, 8, 24002),
    "portfolio_03_hq4_hv4_t32768": _Shape((2656,) * 11 + (3552,), 4, 4, 24003),
    "portfolio_04_hq2_hv4_t18432": _Shape((1648,) * 10 + (1952,), 2, 4, 24004),
    "portfolio_05_hq4_hv8_t18432": _Shape((2000,) * 8 + (2432,), 4, 8, 24005),
    "portfolio_06_hq2_hv8_t32768": _Shape((3200,) * 9 + (3968,), 2, 8, 24006),
    "portfolio_07_hq2_hv4_t18432": _Shape((2000,) * 8 + (2432,), 2, 4, 24007),
    "portfolio_08_hq4_hv4_t18432": _Shape((1648,) * 10 + (1952,), 4, 4, 24008),
    "portfolio_09_hq2_hv8_t32768": _Shape((2656,) * 11 + (3552,), 2, 8, 24009),
    "portfolio_10_hq4_hv8_t18432": _Shape((1648,) * 10 + (1952,), 4, 8, 24010),
    "portfolio_11_hq4_hv4_t32768": _Shape((3200,) * 9 + (3968,), 4, 4, 24011),
    "portfolio_12_hq2_hv4_t32768": _Shape((2656,) * 11 + (3552,), 2, 4, 24012),
    "portfolio_13_hq4_hv4_t18432": _Shape((2000,) * 8 + (2432,), 4, 4, 24013),
    "portfolio_14_hq2_hv4_t32768": _Shape((3200,) * 9 + (3968,), 2, 4, 24014),
    "portfolio_15_hq4_hv8_t32768": _Shape((2656,) * 11 + (3552,), 4, 8, 24015),
    "portfolio_16_hq2_hv8_t18432": _Shape((1648,) * 10 + (1952,), 2, 8, 24016),
    "fixed_b8_t1024_h96": _Shape((1024,) * 8, 96, 96, 36072, "fixed"),
    "fixed_b8_t2048_h96": _Shape((2048,) * 8, 96, 96, 37096, "fixed"),
    "fixed_b8_t4096_h96": _Shape((4096,) * 8, 96, 96, 39144, "fixed"),
    "fixed_b8_t8192_h96": _Shape((8192,) * 8, 96, 96, 43240, "fixed"),
    "fixed_b8_t16384_h96": _Shape((16384,) * 8, 96, 96, 51432, "fixed"),
}

_FALLBACK_SHAPES = {
    "fallback_grouped_row_mixed": _Shape((17, 33, 65), 4, 8, 24005),
    "fallback_grouped_c32_deep_tail": _Shape((4097,), 1, 8, 24105),
    "fallback_row_split_mixed": _Shape((17, 33), 1, 1, 24018),
    "fallback_high_head_row_mixed": _Shape((17, 33), 16, 16, 24017),
}

_SHAPES = _EXACT_PACKED_SHAPES | _PRIMARY_SHAPES | _FALLBACK_SHAPES
_PRIMARY_SHAPE_NAMES = tuple(_PRIMARY_SHAPES)

_FLA_BASELINE_COMMIT = "97bcb883dafd3fa5b859917184e4abfb1c4e8a71"


def _require_timing_dependencies() -> tuple[str, str]:
    try:
        from cupti import cupti  # noqa: F401
        import elftools  # noqa: F401
    except Exception as error:
        raise RuntimeError(
            "the recurrent KDA training benchmark requires a working "
            "cupti-python installation"
        ) from error
    cupti_version = distribution_version("cupti-python")
    pyelftools_version = distribution_version("pyelftools")
    if cupti_version != "13.3.1" or pyelftools_version != "0.32":
        raise RuntimeError(
            "the reportable benchmark requires cupti-python 13.3.1 and "
            f"pyelftools 0.32, got {cupti_version} and {pyelftools_version}"
        )
    return cupti_version, pyelftools_version


def _make_inputs(shape: _Shape, seed: int) -> dict[str, torch.Tensor | None]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    if shape.layout == "fixed":
        if len(set(shape.seq_lens)) != 1:
            raise ValueError("fixed layout requires uniform sequence lengths")
        batch_size = len(shape.seq_lens)
        sequence_length = shape.seq_lens[0]
        qk_shape = (batch_size, sequence_length, shape.num_qk_heads, 128)
        value_shape = (batch_size, sequence_length, shape.num_v_heads, 128)
        cu_seqlens = None
    elif shape.layout == "packed":
        total_tokens = sum(shape.seq_lens)
        qk_shape = (1, total_tokens, shape.num_qk_heads, 128)
        value_shape = (1, total_tokens, shape.num_v_heads, 128)
        cu_seqlens = torch.tensor(
            [0, *torch.tensor(shape.seq_lens).cumsum(0).tolist()],
            dtype=torch.int64,
            device="cuda",
        )
    else:
        raise ValueError(f"unsupported layout: {shape.layout}")
    state_shape = (len(shape.seq_lens), shape.num_v_heads, 128, 128)

    def bf16(shape, multiplier=1.0):
        return (torch.randn(shape, generator=generator, device="cuda") * multiplier).to(
            torch.bfloat16
        )

    return {
        "q": bf16(qk_shape),
        "k": bf16(qk_shape),
        "v": bf16(value_shape),
        "g": bf16(value_shape, 0.1),
        "beta": bf16(value_shape[:-1]),
        "A_log": torch.log(
            torch.rand((shape.num_v_heads,), generator=generator, device="cuda") + 1.0
        ),
        "dt_bias": torch.randn(
            (shape.num_v_heads, 128), generator=generator, device="cuda"
        )
        * 0.1,
        "initial_state": torch.randn(state_shape, generator=generator, device="cuda")
        * 0.02,
        "cu_seqlens": cu_seqlens,
        "do": bf16(value_shape, 0.1),
        "dfinal_state": torch.randn(state_shape, generator=generator, device="cuda")
        * 0.1,
    }


def _median_ms(fn, warmup_ms: int, bench_ms: int) -> tuple[float, list[float]]:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error",
            message=r".*Falling back to CUDA events.*",
            category=UserWarning,
        )
        measurements = bench_gpu_time(
            fn,
            enable_cupti=True,
            cold_l2_cache=True,
            use_cuda_graph=False,
            dry_run_time_ms=warmup_ms,
            repeat_time_ms=bench_ms,
        )
    samples = [float(value) for value in measurements]
    return float(np.median(samples)), samples


def _prepare_fla_full_dag(inputs):
    os.environ["FLA_FLASH_KDA"] = "0"
    import fla
    from fla.ops.kda import chunk_kda

    fla_commit = subprocess.run(
        ["git", "-C", str(Path(fla.__file__).resolve().parent), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if fla_commit != _FLA_BASELINE_COMMIT:
        raise RuntimeError(
            "the reportable benchmark requires pinned FLA commit "
            f"{_FLA_BASELINE_COMMIT}, got {fla_commit}"
        )
    for diff_args in (("diff", "--quiet"), ("diff", "--cached", "--quiet")):
        status = subprocess.run(
            ["git", "-C", str(Path(fla.__file__).resolve().parent), *diff_args],
            check=False,
        ).returncode
        if status != 0:
            raise RuntimeError("the pinned FLA checkout has tracked modifications")

    names = ("q", "k", "v", "g", "beta", "A_log", "dt_bias", "initial_state")
    leaves = {
        name: inputs[name].detach().clone().requires_grad_(True) for name in names
    }
    leaves["dt_bias"] = (
        inputs["dt_bias"].detach().reshape(-1).clone().requires_grad_(True)
    )
    cu_seqlens_cpu = (
        None if inputs["cu_seqlens"] is None else inputs["cu_seqlens"].detach().cpu()
    )

    def run_fla_full_dag():
        output, final_state = chunk_kda(
            leaves["q"],
            leaves["k"],
            leaves["v"],
            leaves["g"],
            leaves["beta"],
            scale=1.0 / math.sqrt(128),
            initial_state=leaves["initial_state"],
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            safe_gate=True,
            lower_bound=-5.0,
            state_v_first=True,
            cu_seqlens=inputs["cu_seqlens"],
            cu_seqlens_cpu=cu_seqlens_cpu,
            A_log=leaves["A_log"],
            dt_bias=leaves["dt_bias"],
            chunk_size=32,
        )
        return torch.autograd.grad(
            (output, final_state),
            tuple(leaves[name] for name in names),
            grad_outputs=(inputs["do"], inputs["dfinal_state"]),
        )

    return run_fla_full_dag, getattr(fla, "__version__", "unknown"), fla_commit


def _benchmark_shape(
    name: str,
    shape: _Shape,
    seed: int,
    *,
    warmup_ms: int,
    bench_ms: int,
    skip_fla: bool,
    cupti_version: str,
    pyelftools_version: str,
) -> dict:
    inputs = _make_inputs(shape, seed)
    output = torch.empty_like(inputs["v"])
    final_state = torch.empty_like(inputs["initial_state"])
    output, final_state, context = recurrent_kda_training_forward(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["g"],
        inputs["beta"],
        inputs["A_log"],
        inputs["dt_bias"],
        inputs["initial_state"],
        inputs["cu_seqlens"],
        out=output,
        final_state_out=final_state,
    )
    gradients = (
        torch.empty_like(inputs["q"]),
        torch.empty_like(inputs["k"]),
        torch.empty_like(inputs["v"]),
        torch.empty_like(inputs["g"]),
        torch.empty_like(inputs["beta"]),
        torch.empty_like(inputs["A_log"]),
        torch.empty_like(inputs["dt_bias"]),
        torch.empty_like(inputs["initial_state"]),
    )
    recurrent_kda_training_backward(
        context, inputs["do"], inputs["dfinal_state"], out=gradients
    )
    torch.cuda.synchronize()

    def run_forward():
        return recurrent_kda_training_forward(
            inputs["q"],
            inputs["k"],
            inputs["v"],
            inputs["g"],
            inputs["beta"],
            inputs["A_log"],
            inputs["dt_bias"],
            inputs["initial_state"],
            inputs["cu_seqlens"],
            out=output,
            final_state_out=final_state,
            context_out=context,
        )

    def run_backward():
        return recurrent_kda_training_backward(
            context, inputs["do"], inputs["dfinal_state"], out=gradients
        )

    def run_full_dag():
        recurrent_kda_training_forward(
            inputs["q"],
            inputs["k"],
            inputs["v"],
            inputs["g"],
            inputs["beta"],
            inputs["A_log"],
            inputs["dt_bias"],
            inputs["initial_state"],
            inputs["cu_seqlens"],
            out=output,
            final_state_out=final_state,
            context_out=context,
        )
        return recurrent_kda_training_backward(
            context, inputs["do"], inputs["dfinal_state"], out=gradients
        )

    forward_ms, forward_samples = _median_ms(run_forward, warmup_ms, bench_ms)
    backward_ms, backward_samples = _median_ms(run_backward, warmup_ms, bench_ms)
    full_dag_ms, full_dag_samples = _median_ms(run_full_dag, warmup_ms, bench_ms)
    result = {
        "shape": name,
        "suite": (
            "primary"
            if name in _PRIMARY_SHAPES
            else "exact_packed"
            if name in _EXACT_PACKED_SHAPES
            else "fallback"
        ),
        "seed": seed,
        "layout": shape.layout,
        "physical_batch_size": int(inputs["q"].shape[0]),
        "seq_lens": list(shape.seq_lens),
        "total_tokens": sum(shape.seq_lens),
        "num_sequences": len(shape.seq_lens),
        "num_qk_heads": shape.num_qk_heads,
        "num_v_heads": shape.num_v_heads,
        "route": context._route.tag,
        "forward_median_ms": forward_ms,
        "backward_median_ms": backward_ms,
        "full_dag_median_ms": full_dag_ms,
        "forward_samples_ms": forward_samples,
        "backward_samples_ms": backward_samples,
        "full_dag_samples_ms": full_dag_samples,
        "forward_uses_private_fp32_final_state_recurrence": True,
        "backward_recomputes_forward": False,
        "cupti_python": cupti_version,
        "pyelftools": pyelftools_version,
        "timing_backend": "CUPTI activity",
        "cold_l2_cache": True,
        "cuda_graph": False,
        "cuda_event_fallback": False,
        "warmup_ms": warmup_ms,
        "bench_ms": bench_ms,
        "forward_sample_count": len(forward_samples),
        "backward_sample_count": len(backward_samples),
        "full_dag_sample_count": len(full_dag_samples),
        "fla_baseline_skipped": skip_fla,
        "reportable": not skip_fla,
    }
    if not skip_fla:
        fla_full_dag, fla_version, fla_commit = _prepare_fla_full_dag(inputs)
        fla_full_dag_ms, fla_full_dag_samples = _median_ms(
            fla_full_dag, warmup_ms, bench_ms
        )
        delta_ms = full_dag_ms - fla_full_dag_ms
        result.update(
            {
                "fla_full_dag_median_ms": fla_full_dag_ms,
                "fla_full_dag_samples_ms": fla_full_dag_samples,
                "full_dag_delta_ms_vs_fla": delta_ms,
                "full_dag_delta_percent_vs_fla": 100.0 * delta_ms / fla_full_dag_ms,
                "full_dag_speedup_vs_fla": fla_full_dag_ms / full_dag_ms,
                "fla_chunk_size": 32,
                "fla_flash_kda": os.environ["FLA_FLASH_KDA"],
                "fla_version": fla_version,
                "fla_commit": fla_commit,
                "fla_full_dag_sample_count": len(fla_full_dag_samples),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup-ms", type=int, default=25)
    parser.add_argument("--bench-ms", type=int, default=100)
    parser.add_argument(
        "--seed", type=int, help="override the fixed seed for every selected shape"
    )
    parser.add_argument(
        "--shape",
        action="append",
        choices=tuple(_SHAPES),
        help="benchmark one named shape; repeat to select multiple shapes",
    )
    parser.add_argument(
        "--all-shapes",
        action="store_true",
        help="benchmark the 16 portfolio rows and five fixed B8/H96 rows",
    )
    parser.add_argument("--json", type=Path)
    parser.add_argument(
        "--skip-fla", action="store_true", help="skip the pinned FLA full-DAG peer"
    )
    args = parser.parse_args()
    if args.all_shapes and args.shape:
        parser.error("--all-shapes and --shape are mutually exclusive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    capability = get_compute_capability(device)
    if capability not in {(10, 0), (10, 3)}:
        raise RuntimeError("the training benchmark requires SM100a or SM103a")
    cupti_version, pyelftools_version = _require_timing_dependencies()
    names = (
        list(_PRIMARY_SHAPE_NAMES)
        if args.all_shapes
        else (args.shape or ["packed_1024x8_h96"])
    )
    results = [
        _benchmark_shape(
            name,
            _SHAPES[name],
            _SHAPES[name].seed if args.seed is None else args.seed,
            warmup_ms=args.warmup_ms,
            bench_ms=args.bench_ms,
            skip_fla=args.skip_fla,
            cupti_version=cupti_version,
            pyelftools_version=pyelftools_version,
        )
        for name in names
    ]
    report = {
        "gpu": torch.cuda.get_device_name(device),
        "compute_capability": list(capability),
        "target": "sm_100a" if capability == (10, 0) else "sm_103a",
        "host_architecture": platform.machine(),
        "timing_backend": "CUPTI activity",
        "cupti_python": cupti_version,
        "pyelftools": pyelftools_version,
        "cold_l2_cache": True,
        "cuda_graph": False,
        "cuda_event_fallback": False,
        "warmup_ms": args.warmup_ms,
        "bench_ms": args.bench_ms,
        "fla_baseline_skipped": args.skip_fla,
        "reportable": not args.skip_fla,
        "results": results,
    }
    print(json.dumps(report, indent=2))
    if args.json is not None:
        args.json.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
