# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CUPTI benchmark for the public recurrent-KDA backward API.

The timed scope includes forward-tape reconstruction, all backward kernels,
required workspace resets, and all eight gradients. Allocation, JIT loading,
packed-offset validation, metadata construction, and TMA descriptor
preparation happen during untimed warmup. Every measurement uses cold L2.
"""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from flashinfer.kda_backward import (
    RecurrentKDABackwardWorkspace,
    recurrent_kda_backward,
)
from flashinfer.testing import bench_gpu_time
from flashinfer.utils import get_compute_capability


@dataclass(frozen=True)
class Case:
    name: str
    seq_lens: tuple[int, ...]
    num_heads: int
    packed: bool
    seed: int


PERFORMANCE_CASES = (
    Case("fixed_t4096_h32", (4096,), 32, False, 409632),
    Case("fixed_t8192_h96", (8192,), 96, False, 819296),
    Case(
        "packed_mixed_h96",
        (1300, 547, 2048, 963, 271, 3063),
        96,
        True,
        819206,
    ),
    Case("packed_1024x8_h96", (1024,) * 8, 96, True, 819208),
)


def _offsets(seq_lens: tuple[int, ...]) -> tuple[int, ...]:
    result = [0]
    for length in seq_lens:
        result.append(result[-1] + length)
    return tuple(result)


def _make_inputs(case: Case) -> dict:
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(case.seed)
    total_tokens = sum(case.seq_lens)
    token_shape = (1, total_tokens, case.num_heads, 128)
    state_shape = (len(case.seq_lens), case.num_heads, 128, 128)

    def bf16(shape, multiplier=1.0):
        return (
            torch.randn(
                shape,
                generator=generator,
                device=device,
                dtype=torch.float32,
            )
            * multiplier
        ).to(torch.bfloat16)

    return {
        "q": bf16(token_shape),
        "k": bf16(token_shape),
        "v": bf16(token_shape),
        "g": bf16(token_shape, 0.1),
        "beta": bf16(token_shape[:-1]),
        "A_log": torch.log(
            torch.rand(
                (case.num_heads,),
                generator=generator,
                device=device,
                dtype=torch.float32,
            )
            + 1.0
        ),
        "dt_bias": torch.randn(
            (case.num_heads, 128),
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        * 0.1,
        "initial_state": torch.randn(
            state_shape,
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        * 0.02,
        "do": bf16(token_shape, 0.1),
        "dfinal_state": torch.randn(
            state_shape,
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        * 0.1,
        "cu_seqlens": (
            torch.tensor(_offsets(case.seq_lens), dtype=torch.int64, device=device)
            if case.packed
            else None
        ),
    }


def _make_outputs(inputs: dict) -> tuple[torch.Tensor, ...]:
    return (
        torch.empty_like(inputs["q"]),
        torch.empty_like(inputs["q"]),
        torch.empty_like(inputs["q"]),
        torch.empty_like(inputs["q"]),
        torch.empty_like(inputs["beta"]),
        torch.empty_like(inputs["A_log"]),
        torch.empty_like(inputs["dt_bias"]),
        torch.empty_like(inputs["initial_state"]),
    )


def _measure(case: Case, warmup_ms: int, bench_ms: int) -> dict:
    inputs = _make_inputs(case)
    outputs = _make_outputs(inputs)
    workspace = RecurrentKDABackwardWorkspace("cuda")

    def run():
        return recurrent_kda_backward(
            **inputs,
            workspace=workspace,
            out=outputs,
            scale=1.0 / math.sqrt(128),
            lower_bound=-5.0,
        )

    run()
    torch.cuda.synchronize()
    measurements = bench_gpu_time(
        run,
        enable_cupti=True,
        cold_l2_cache=True,
        use_cuda_graph=False,
        dry_run_time_ms=warmup_ms,
        repeat_time_ms=bench_ms,
    )
    samples_ms = [float(value) for value in measurements]
    return {
        "case": case.name,
        "seq_lens": list(case.seq_lens),
        "num_heads": case.num_heads,
        "total_tokens": sum(case.seq_lens),
        "median_ms": float(np.median(samples_ms)),
        "min_ms": min(samples_ms),
        "samples_ms": samples_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        choices=tuple(case.name for case in PERFORMANCE_CASES),
        help="Run one case; repeat the option to select multiple cases.",
    )
    parser.add_argument("--warmup-ms", type=int, default=25)
    parser.add_argument("--bench-ms", type=int, default=100)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    if args.warmup_ms <= 0 or args.bench_ms <= 0:
        parser.error("--warmup-ms and --bench-ms must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if get_compute_capability(torch.device("cuda")) not in {(10, 0), (10, 3)}:
        raise RuntimeError("recurrent-KDA backward benchmark requires SM100a or SM103a")

    selected = set(args.case or (case.name for case in PERFORMANCE_CASES))
    results = [
        _measure(case, args.warmup_ms, args.bench_ms)
        for case in PERFORMANCE_CASES
        if case.name in selected
    ]
    for result in results:
        print(
            f"{result['case']}: {result['median_ms']:.4f} ms "
            f"(min {result['min_ms']:.4f} ms)"
        )
    if args.json is not None:
        args.json.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
