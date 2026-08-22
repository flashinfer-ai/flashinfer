# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Cold-L2 CUPTI benchmark for the paired recurrent-KDA training API.

The forward timing includes the exact checkpoint-producing recurrence plus the
private serving recurrence that produces the public final state. The backward
timing consumes the already-saved context and therefore excludes forward
recomputation.
"""

import argparse
import json
import math
import os
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


def _require_timing_dependencies() -> tuple[str, str]:
    try:
        from cupti import cupti  # noqa: F401
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


def _make_inputs(seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    token_shape = (1, 8192, 96, 128)
    state_shape = (8, 96, 128, 128)

    def bf16(shape, multiplier=1.0):
        return (torch.randn(shape, generator=generator, device="cuda") * multiplier).to(
            torch.bfloat16
        )

    return {
        "q": bf16(token_shape),
        "k": bf16(token_shape),
        "v": bf16(token_shape),
        "g": bf16(token_shape, 0.1),
        "beta": bf16(token_shape[:-1]),
        "A_log": torch.log(torch.rand((96,), generator=generator, device="cuda") + 1.0),
        "dt_bias": torch.randn((96, 128), generator=generator, device="cuda") * 0.1,
        "initial_state": torch.randn(state_shape, generator=generator, device="cuda")
        * 0.02,
        "cu_seqlens": torch.arange(0, 8193, 1024, dtype=torch.int64, device="cuda"),
        "do": bf16(token_shape, 0.1),
        "dfinal_state": torch.randn(state_shape, generator=generator, device="cuda")
        * 0.1,
    }


def _median_ms(fn, warmup_ms: int, bench_ms: int) -> tuple[float, list[float]]:
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

    names = ("q", "k", "v", "g", "beta", "A_log", "dt_bias", "initial_state")
    leaves = {
        name: inputs[name].detach().clone().requires_grad_(True) for name in names
    }
    leaves["dt_bias"] = (
        inputs["dt_bias"].detach().reshape(-1).clone().requires_grad_(True)
    )
    cu_seqlens_cpu = inputs["cu_seqlens"].detach().cpu()

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

    return run_fla_full_dag, getattr(fla, "__version__", "unknown")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup-ms", type=int, default=25)
    parser.add_argument("--bench-ms", type=int, default=100)
    parser.add_argument("--seed", type=int, default=819208)
    parser.add_argument("--json", type=Path)
    parser.add_argument(
        "--skip-fla", action="store_true", help="skip the pinned FLA full-DAG peer"
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if get_compute_capability(torch.device("cuda")) not in {(10, 0), (10, 3)}:
        raise RuntimeError("the training benchmark requires SM100a or SM103a")
    cupti_version, pyelftools_version = _require_timing_dependencies()

    inputs = _make_inputs(args.seed)
    output = torch.empty_like(inputs["q"])
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
        torch.empty_like(inputs["q"]),
        torch.empty_like(inputs["q"]),
        torch.empty_like(inputs["q"]),
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

    forward_ms, forward_samples = _median_ms(run_forward, args.warmup_ms, args.bench_ms)
    backward_ms, backward_samples = _median_ms(
        run_backward, args.warmup_ms, args.bench_ms
    )
    full_dag_ms, full_dag_samples = _median_ms(
        run_full_dag, args.warmup_ms, args.bench_ms
    )
    result = {
        "shape": "packed_1024x8_h96",
        "forward_median_ms": forward_ms,
        "backward_median_ms": backward_ms,
        "full_dag_median_ms": full_dag_ms,
        "forward_samples_ms": forward_samples,
        "backward_samples_ms": backward_samples,
        "full_dag_samples_ms": full_dag_samples,
        "forward_uses_private_serving_final_state_recurrence": True,
        "forward_promotes_bf16_final_state_to_fp32": True,
        "backward_recomputes_forward": False,
        "cupti_python": cupti_version,
        "pyelftools": pyelftools_version,
        "timing_backend": "CUPTI activity",
    }
    if not args.skip_fla:
        fla_full_dag, fla_version = _prepare_fla_full_dag(inputs)
        fla_full_dag_ms, fla_full_dag_samples = _median_ms(
            fla_full_dag, args.warmup_ms, args.bench_ms
        )
        result.update(
            {
                "fla_full_dag_median_ms": fla_full_dag_ms,
                "fla_full_dag_samples_ms": fla_full_dag_samples,
                "full_dag_speedup_vs_fla": fla_full_dag_ms / full_dag_ms,
                "fla_chunk_size": 32,
                "fla_flash_kda": os.environ["FLA_FLASH_KDA"],
                "fla_version": fla_version,
            }
        )
    print(json.dumps(result, indent=2))
    if args.json is not None:
        args.json.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
