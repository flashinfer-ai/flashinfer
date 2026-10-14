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

# Benchmark the experimental SM110 GQA decode kernel with cold-L2 CUPTI.

from __future__ import annotations

import argparse
import json
import math
import statistics
from importlib.metadata import version

import torch
import torch.nn.functional as F

from flashinfer import sm110_gqa_decode
from flashinfer.testing import bench_gpu_time_with_cupti


SHAPES = (
    ("cap64", 1, 64, [1], 95601),
    ("b4_cap256", 4, 256, [64, 127, 191, 256], 95602),
    ("cap1024", 1, 1024, [1024], 95603),
    ("cap4096", 1, 4096, [3968], 95604),
)


def _require_cupti() -> None:
    try:
        from cupti import cupti as cupti_module

        major = int(version("cupti-python").split(".")[0])
    except Exception as error:
        raise RuntimeError("cupti-python 13 or newer is required") from error
    if cupti_module is None or major < 13:
        raise RuntimeError("cupti-python 13 or newer is required")


def _measure(fn) -> float:
    samples = bench_gpu_time_with_cupti(
        fn,
        dry_run_time_ms=100,
        repeat_time_ms=1000,
        cold_l2_cache=True,
    )
    return float(statistics.median(samples))


def _measure_shape(
    label: str,
    batch: int,
    capacity: int,
    lengths: list[int],
    seed: int,
) -> dict[str, object]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(
        batch,
        32,
        128,
        dtype=torch.float16,
        device="cuda",
        generator=generator,
    )
    kv = torch.randn(
        batch,
        2,
        8,
        capacity,
        128,
        dtype=torch.float16,
        device="cuda",
        generator=generator,
    )
    sequence_lengths = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    positions = torch.arange(capacity, device="cuda")
    mask = positions.view(1, 1, 1, capacity) < sequence_lengths.view(batch, 1, 1, 1)
    sdpa_q = q.unsqueeze(2)
    sdpa_k = kv[:, 0]
    sdpa_v = kv[:, 1]

    def candidate() -> torch.Tensor:
        return sm110_gqa_decode(q, kv, sequence_lengths)

    def torch_sdpa() -> torch.Tensor:
        return F.scaled_dot_product_attention(
            sdpa_q,
            sdpa_k,
            sdpa_v,
            attn_mask=mask,
            scale=1.0 / math.sqrt(128),
            enable_gqa=True,
        )

    candidate()
    torch_sdpa()
    torch.cuda.synchronize()
    candidate_ms = _measure(candidate)
    torch_ms = _measure(torch_sdpa)
    return {
        "label": label,
        "batch": batch,
        "capacity": capacity,
        "sequence_lengths": lengths,
        "sm110_gqa_decode_ms": candidate_ms,
        "torch_sdpa_ms": torch_ms,
        "speedup_vs_torch_sdpa": torch_ms / candidate_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str)
    args = parser.parse_args()
    _require_cupti()
    if torch.cuda.get_device_capability() != (11, 0):
        raise RuntimeError("this benchmark requires an exact SM110 GPU")

    rows = [_measure_shape(*shape) for shape in SHAPES]
    result = {
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "cuda": torch.version.cuda,
        "timing": "same-process cold-L2 CUPTI median GPU activity span",
        "output_allocation": "both timed paths allocate and return their output",
        "rows": rows,
    }
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.output:
        with open(args.output, "w") as output:
            output.write(payload + "\n")


if __name__ == "__main__":
    main()
