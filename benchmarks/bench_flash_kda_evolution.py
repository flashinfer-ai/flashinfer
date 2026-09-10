# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Validate and CUPTI-time the generated Blackwell recurrent-KDA inventory."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from bench_recurrent_kda_prefill import (
    LEGACY_CASES,
    PRODUCTION_CASES,
    _hardware_metadata,
    _require_cupti,
)
from flashinfer.jit.flash_kda_evolution import FLASH_KDA_EVOLUTION_VARIANTS
from flashinfer.kda import recurrent_kda
from flashinfer.kda_evolution import prepare_flash_kda_evolution
from flashinfer.testing import bench_gpu_time


def _make_inputs(case):
    total_tokens = sum(case.seq_lens)
    shape = (1, total_tokens, case.num_heads, 128)
    generator = torch.Generator(device="cuda").manual_seed(case.seed)
    q = torch.randn(shape, generator=generator, device="cuda").to(torch.bfloat16)
    k = torch.randn(shape, generator=generator, device="cuda").to(torch.bfloat16)
    v = torch.randn(shape, generator=generator, device="cuda").to(torch.bfloat16)
    g = torch.randn(shape, generator=generator, device="cuda").to(torch.bfloat16)
    beta = torch.randn(
        (1, total_tokens, case.num_heads),
        generator=generator,
        device="cuda",
    ).to(torch.bfloat16)
    A_log = torch.rand(
        (case.num_heads,),
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    )
    dt_bias = torch.rand(
        (case.num_heads, 128),
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    )
    initial_state = (
        torch.randn(
            (len(case.seq_lens), case.num_heads, 128, 128),
            generator=generator,
            device="cuda",
        )
        * 0.25
    ).to(torch.bfloat16)
    offsets = [0]
    for length in case.seq_lens:
        offsets.append(offsets[-1] + length)
    cu_seqlens = (
        torch.tensor(offsets, dtype=torch.int64, device="cuda") if case.packed else None
    )
    return q, k, v, g, beta, A_log, dt_bias, initial_state, cu_seqlens


def _diagnostic(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    delta = (actual.float() - expected.float()).abs()
    close = torch.isclose(actual, expected, atol=1e-2, rtol=1e-2)
    mismatch_count = int((~close).sum())
    return {
        "correct": mismatch_count == 0,
        "mismatch_count": mismatch_count,
        "element_count": actual.numel(),
        "max_abs": float(delta.max()),
    }


def _measure(run, *, dry_run_iters: int, repeat_iters: int):
    samples = bench_gpu_time(
        run,
        enable_cupti=True,
        cold_l2_cache=True,
        use_cuda_graph=False,
        dry_run_iters=dry_run_iters,
        repeat_iters=repeat_iters,
    )
    samples_ms = [float(value) for value in samples]
    return float(np.median(samples_ms)), samples_ms


def _run_case(case, *, correctness_only: bool, dry_run_iters: int, repeat_iters: int):
    q, k, v, g, beta, A_log, dt_bias, initial_state, cu_seqlens = _make_inputs(case)
    scale = 1.0 / math.sqrt(128)
    reference_initial = initial_state.clone()
    expected_out, expected_state = recurrent_kda(
        q,
        k,
        v,
        g,
        beta,
        A_log,
        dt_bias,
        initial_state=reference_initial,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        scale=scale,
        lower_bound=-5.0,
        cu_seqlens=cu_seqlens,
        beta_is_logit=True,
        backend="cake",
    )
    assert expected_state is not None

    actual_out = torch.empty_like(q)
    actual_state = torch.empty_like(initial_state)
    prepared = prepare_flash_kda_evolution(
        q,
        k,
        v,
        g,
        beta,
        A_log,
        dt_bias,
        initial_state,
        actual_out,
        actual_state,
        scale=scale,
        lower_bound=-5.0,
        cu_seqlens=cu_seqlens,
    )
    prepared.launch()
    torch.cuda.synchronize()
    out_diagnostic = _diagnostic(actual_out, expected_out)
    state_diagnostic = _diagnostic(actual_state, expected_state)
    correct = out_diagnostic["correct"] and state_diagnostic["correct"]
    row = {
        "name": case.name,
        "num_heads": case.num_heads,
        "seq_lens": list(case.seq_lens),
        "layout": "packed" if case.packed else "fixed",
        "seed": case.seed,
        "variant": prepared.variant,
        "target": prepared.target,
        "grid_x": prepared.grid_x,
        "correct": correct,
        "out": out_diagnostic,
        "final_state": state_diagnostic,
    }
    if correct and not correctness_only:
        median_ms, samples_ms = _measure(
            prepared.launch,
            dry_run_iters=dry_run_iters,
            repeat_iters=repeat_iters,
        )
        row.update(
            {
                "median_ms": median_ms,
                "samples_ms": samples_ms,
                "timing_backend": "cupti",
                "cold_l2": True,
                "cuda_graph": False,
            }
        )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("six", "full"), default="full")
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument("--dry-run-iters", type=int, default=20)
    parser.add_argument("--repeat-iters", type=int, default=100)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    if args.dry_run_iters <= 0 or args.repeat_iters <= 0:
        parser.error("--dry-run-iters and --repeat-iters must be positive")

    if not args.correctness_only:
        _require_cupti()
    cases = LEGACY_CASES if args.suite == "six" else PRODUCTION_CASES
    rows = [
        _run_case(
            case,
            correctness_only=args.correctness_only,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.repeat_iters,
        )
        for case in cases
    ]
    report = {
        "suite": args.suite,
        "hardware": _hardware_metadata(torch.device("cuda")),
        "variant_inventory": sorted(FLASH_KDA_EVOLUTION_VARIANTS),
        "variant_count": len(FLASH_KDA_EVOLUTION_VARIANTS),
        "all_correct": all(row["correct"] for row in rows),
        "correctness_shape_count": len(rows),
        "timed_shape_count": sum("median_ms" in row for row in rows),
        "rows": rows,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "all_correct": report["all_correct"],
                "correctness_shape_count": report["correctness_shape_count"],
                "timed_shape_count": report["timed_shape_count"],
                "variant_count": report["variant_count"],
            },
            sort_keys=True,
        )
    )
    if not report["all_correct"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
