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

"""Benchmark B200 CAKE K1 parallelism against the M64/M128 oracle.

Every implementation is invoked through ``flashinfer.kda.recurrent_kda``.
The script first requires bitwise-identical output and recurrent state, then
measures all three physical routes with cold L2. Speedup is always reported
against ``min(CAKE-M64, CAKE-M128)`` for the same shape.
"""

import argparse
import importlib
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator, Optional

import numpy as np
import torch

from flashinfer.kda import recurrent_kda
from flashinfer.kda_prefill import RecurrentKDAPrefillWorkspace
from flashinfer.testing import bench_gpu_time
from flashinfer.utils import get_compute_capability

kda_prefill = importlib.import_module("flashinfer.kda_prefill")


@contextmanager
def _physical_route(
    route: Optional[tuple[str, int, int]],
) -> Iterator[None]:
    original = kda_prefill._select_flash_kda_prefill_variant
    if route is not None:
        kda_prefill._select_flash_kda_prefill_variant = (
            lambda route=route, **_kwargs: route
        )
    try:
        yield
    finally:
        kda_prefill._select_flash_kda_prefill_variant = original


def _measure(
    run: Callable[[], object],
    *,
    enable_cupti: bool,
    warmup_ms: int,
    bench_ms: int,
) -> tuple[float, list[float]]:
    samples = [
        float(value)
        for value in bench_gpu_time(
            run,
            enable_cupti=enable_cupti,
            cold_l2_cache=True,
            use_cuda_graph=False,
            dry_run_time_ms=warmup_ms,
            repeat_time_ms=bench_ms,
        )
    ]
    return float(np.median(samples)), samples


def _run_case(
    *,
    batch_size: int,
    sequence_length: int,
    num_heads: int,
    seed: int,
    enable_cupti: bool,
    warmup_ms: int,
    bench_ms: int,
    state_rotations: int,
) -> dict:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    shape = (batch_size, sequence_length, num_heads, 128)

    def randn(dims: tuple[int, ...]) -> torch.Tensor:
        return torch.randn(dims, generator=generator, device="cuda").to(
            torch.bfloat16
        )

    q, k, v, g = (randn(shape) for _ in range(4))
    beta = randn((batch_size, sequence_length, num_heads))
    A_log = torch.rand((num_heads,), generator=generator, device="cuda")
    dt_bias = torch.rand(
        (num_heads, 128), generator=generator, device="cuda"
    )
    initial = randn((batch_size, num_heads, 128, 128))
    outputs = {
        name: torch.empty_like(q) for name in ("m64", "m128", "k1_parallel")
    }
    states = {name: initial.clone() for name in outputs}
    state_pools = {
        name: initial.unsqueeze(0)
        .expand(state_rotations, *initial.shape)
        .clone()
        for name in outputs
    }
    state_cursors = {name: 0 for name in outputs}
    workspaces = {
        name: RecurrentKDAPrefillWorkspace(q.device) for name in outputs
    }

    def launch(name: str, *, timed: bool = False) -> object:
        state = states[name]
        if timed:
            state_index = state_cursors[name]
            if state_index >= state_rotations:
                raise RuntimeError(
                    f"{name} exhausted {state_rotations} preinitialized state slots"
                )
            state_cursors[name] += 1
            state = state_pools[name][state_index]
        return recurrent_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=128**-0.5,
            initial_state=state,
            output=outputs[name],
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            lower_bound=-5.0,
            beta_is_logit=True,
            prefill_workspace=workspaces[name],
        )

    routes = {
        "m64": ("m64", 0, 0),
        "m128": ("m128", 0, 0),
        "k1_parallel": None,
    }
    for name, route in routes.items():
        with _physical_route(route):
            launch(name)
    torch.cuda.synchronize()
    for name in ("m128", "k1_parallel"):
        torch.testing.assert_close(outputs[name], outputs["m64"], atol=0, rtol=0)
        torch.testing.assert_close(states[name], states["m64"], atol=0, rtol=0)

    timings = {}
    samples = {}
    for name, route in routes.items():
        state_cursors[name] = 0
        torch.cuda.synchronize()
        with _physical_route(route):
            timings[name], samples[name] = _measure(
                lambda name=name: launch(name, timed=True),
                enable_cupti=enable_cupti,
                warmup_ms=warmup_ms,
                bench_ms=bench_ms,
            )

    oracle_ms = min(timings["m64"], timings["m128"])
    return {
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "num_heads": num_heads,
        "task_count": batch_size * num_heads,
        "m64_ms": timings["m64"],
        "m128_ms": timings["m128"],
        "oracle_ms": oracle_ms,
        "k1_parallel_ms": timings["k1_parallel"],
        "speedup_vs_oracle": oracle_ms / timings["k1_parallel"],
        "samples_ms": samples,
        "correctness": "bitwise output and state",
        "timing_backend": "cupti" if enable_cupti else "cuda_event",
        "cold_l2": True,
        "cuda_graph": False,
        "same_initial_state_per_timed_call": True,
        "state_slots_used": state_cursors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--sequence-lengths", type=int, nargs="+", default=[1024, 2048, 4096, 8192]
    )
    parser.add_argument("--num-heads", type=int, nargs="+", default=[8, 16, 24, 32])
    parser.add_argument("--warmup-ms", type=int, default=20)
    parser.add_argument("--bench-ms", type=int, default=100)
    parser.add_argument("--state-rotations", type=int, default=2048)
    parser.add_argument("--cupti", action="store_true")
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if get_compute_capability(torch.device("cuda")) != (10, 0):
        raise RuntimeError("CAKE K1 owner/helper benchmarking requires CC 10.0")
    properties = torch.cuda.get_device_properties(0)
    metadata = {
        "device_name": properties.name,
        "compute_capability": [10, 0],
        "multiprocessor_count": properties.multi_processor_count,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
    }
    print(json.dumps(metadata, sort_keys=True))

    results = []
    for sequence_length in args.sequence_lengths:
        for num_heads in args.num_heads:
            result = _run_case(
                batch_size=args.batch_size,
                sequence_length=sequence_length,
                num_heads=num_heads,
                seed=args.seed + sequence_length + num_heads,
                enable_cupti=args.cupti,
                warmup_ms=args.warmup_ms,
                bench_ms=args.bench_ms,
                state_rotations=args.state_rotations,
            )
            result["hardware"] = metadata
            results.append(result)
            print(
                f"B={args.batch_size} T={sequence_length:5d} H={num_heads:2d} "
                f"M64={result['m64_ms']:.6f} ms "
                f"M128={result['m128_ms']:.6f} ms "
                f"K1={result['k1_parallel_ms']:.6f} ms "
                f"speedup={result['speedup_vs_oracle']:.3f}x"
            )

    if args.json is not None:
        args.json.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
