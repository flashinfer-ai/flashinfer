# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Cold-L2 CUPTI benchmark for the paired recurrent-KDA training API.

The only reportable latency is one callback containing the public forward
immediately followed by backward on that forward's saved context. It is not a
sum of separately measured forward and backward medians. The output, final
state, and all eight gradients are checked against the pinned FLA peer before
either callback is timed.
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


_PORTFOLIO_SHAPES = {
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

_SELECTOR_SHAPES = {
    "fixed_b8_t4096_h4": _Shape((4096,) * 8, 4, 4, 46004, "fixed"),
    "fixed_b8_t4096_h8": _Shape((4096,) * 8, 8, 8, 46008, "fixed"),
    "packed_512_1024_1536_2048_2560_h96": _Shape(
        (512, 1024, 1536, 2048, 2560), 96, 96, 7685096
    ),
    "fixed_b2_t1024_h96": _Shape((1024,) * 2, 96, 96, 47102, "fixed"),
    "fixed_b4_t2048_h96": _Shape((2048,) * 4, 96, 96, 47204, "fixed"),
    "fixed_b1_t512_h8": _Shape((512,), 8, 8, 47512, "fixed"),
    "fixed_b4_t1024_h96": _Shape((1024,) * 4, 96, 96, 47104, "fixed"),
    "fixed_b5_t2048_h96": _Shape((2048,) * 5, 96, 96, 47205, "fixed"),
    "fixed_b1_t1024_h8": _Shape((1024,), 8, 8, 48024, "fixed"),
    "packed_512_1024_h32": _Shape((512, 1024), 32, 32, 48512),
    "packed_511_1025_h32": _Shape((511, 1025), 32, 32, 48511),
    "fixed_b1_t17_h1": _Shape((17,), 1, 1, 48017, "fixed"),
}

_ROUTE_COVERAGE_SHAPES = {
    "grouped_c32_t4097_hq1_hv8": _Shape((4097,), 1, 8, 24105),
    "grouped_row_17_33_65_hq4_hv8": _Shape((17, 33, 65), 4, 8, 24005),
}

_SHAPES = _PORTFOLIO_SHAPES | _SELECTOR_SHAPES | _ROUTE_COVERAGE_SHAPES
assert len(_SHAPES) == 35

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
        cu_seqlens_cpu = None
    elif shape.layout == "packed":
        total_tokens = sum(shape.seq_lens)
        qk_shape = (1, total_tokens, shape.num_qk_heads, 128)
        value_shape = (1, total_tokens, shape.num_v_heads, 128)
        cu_seqlens_cpu = torch.tensor(
            [0, *torch.tensor(shape.seq_lens).cumsum(0).tolist()],
            dtype=torch.int64,
        )
        cu_seqlens = cu_seqlens_cpu.to(device="cuda")
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
        "cu_seqlens_cpu": cu_seqlens_cpu,
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


def _prepare_fla_paired(inputs):
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
    cu_seqlens_cpu = inputs["cu_seqlens_cpu"]

    def run_fla_paired():
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
        gradients = torch.autograd.grad(
            (output, final_state),
            tuple(leaves[name] for name in names),
            grad_outputs=(inputs["do"], inputs["dfinal_state"]),
        )
        gradients = (
            *gradients[:-2],
            gradients[-2].reshape_as(inputs["dt_bias"]),
            gradients[-1],
        )
        return output, final_state, gradients

    return run_fla_paired, getattr(fla, "__version__", "unknown"), fla_commit


def _assert_paired_close(public_result, reference_result) -> None:
    public_output, public_final, public_gradients = public_result
    reference_output, reference_final, reference_gradients = reference_result
    torch.testing.assert_close(
        public_output, reference_output, atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        public_final, reference_final, atol=1e-2, rtol=1e-2
    )
    for public, reference in zip(
        public_gradients, reference_gradients, strict=True
    ):
        torch.testing.assert_close(public, reference, atol=1e-2, rtol=1e-2)


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
        cu_seqlens_cpu=inputs["cu_seqlens_cpu"],
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

    def run_paired():
        paired_output, paired_final, paired_context = recurrent_kda_training_forward(
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
            cu_seqlens_cpu=inputs["cu_seqlens_cpu"],
        )
        paired_gradients = recurrent_kda_training_backward(
            paired_context,
            inputs["do"],
            inputs["dfinal_state"],
            out=gradients,
        )
        return paired_output, paired_final, paired_gradients

    fla_paired = None
    fla_version = None
    fla_commit = None
    if not skip_fla:
        fla_paired, fla_version, fla_commit = _prepare_fla_paired(inputs)
        _assert_paired_close(run_paired(), fla_paired())
        torch.cuda.synchronize()

    paired_ms, paired_samples = _median_ms(run_paired, warmup_ms, bench_ms)
    result = {
        "shape": name,
        "suite": (
            "portfolio"
            if name in _PORTFOLIO_SHAPES
            else "selector"
            if name in _SELECTOR_SHAPES
            else "route_coverage"
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
        "paired_median_ms": paired_ms,
        "paired_samples_ms": paired_samples,
        "paired_api_boundary": "forward_then_saved_context_backward",
        "separate_forward_backward_medians_reported": False,
        "forward_output_dtype": str(output.dtype),
        "forward_final_state_dtype": str(final_state.dtype),
        "backward_gradient_dtypes": [str(value.dtype) for value in gradients],
        "forward_context_reused_across_samples": True,
        "backward_recomputes_forward": False,
        "cupti_python": cupti_version,
        "pyelftools": pyelftools_version,
        "timing_backend": "CUPTI activity",
        "cold_l2_cache": True,
        "cuda_graph": False,
        "cuda_event_fallback": False,
        "warmup_ms": warmup_ms,
        "bench_ms": bench_ms,
        "paired_sample_count": len(paired_samples),
        "fla_baseline_skipped": skip_fla,
        "reportable": not skip_fla,
    }
    if not skip_fla:
        assert fla_paired is not None
        fla_paired_ms, fla_paired_samples = _median_ms(
            fla_paired, warmup_ms, bench_ms
        )
        delta_ms = paired_ms - fla_paired_ms
        result.update(
            {
                "fla_paired_median_ms": fla_paired_ms,
                "fla_paired_samples_ms": fla_paired_samples,
                "paired_delta_ms_vs_fla": delta_ms,
                "paired_delta_percent_vs_fla": 100.0 * delta_ms / fla_paired_ms,
                "paired_speedup_vs_fla": fla_paired_ms / paired_ms,
                "fla_chunk_size": 32,
                "fla_flash_kda": os.environ["FLA_FLASH_KDA"],
                "fla_version": fla_version,
                "fla_commit": fla_commit,
                "fla_paired_sample_count": len(fla_paired_samples),
                "correctness_gate": {
                    "output": True,
                    "final_state": True,
                    "all_eight_gradients": True,
                    "atol": 1e-2,
                    "rtol": 1e-2,
                },
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
        help="benchmark all 35 portfolio, selector, and route-coverage shapes",
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
        list(_SHAPES)
        if args.all_shapes
        else (args.shape or ["fixed_b8_t1024_h96"])
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
        "paired_api_boundary": "forward_then_saved_context_backward",
        "separate_forward_backward_medians_reported": False,
        "shape_count": len(names),
        "results": results,
    }
    print(json.dumps(report, indent=2))
    if args.json is not None:
        args.json.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
