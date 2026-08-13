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

"""Benchmark SM100-family CAKE K1 parallelism against the M64/M128 oracle.

Every implementation is invoked through ``flashinfer.kda.recurrent_kda``.
The script first requires bitwise-identical output and recurrent state, then
measures physical routes with cold L2. Speedup is always reported against
``min(CAKE-M64, CAKE-M128)`` for the same shape. Optional forced C4/C8 and
mailbox-depth configurations preserve the evidence used to tune dispatch.
"""

import argparse
import importlib
import json
import random
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


def _parse_forced_config(value: str) -> tuple[int, int]:
    try:
        cluster_size_text, mailbox_depth_text = value.split(":", 1)
        cluster_size = int(cluster_size_text)
        mailbox_depth = int(mailbox_depth_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"expected CLUSTER_SIZE:MAILBOX_DEPTH, got {value!r}"
        ) from error
    if cluster_size not in (4, 8):
        raise argparse.ArgumentTypeError("forced cluster size must be 4 or 8")
    producer_instances = (cluster_size - 1) * 5
    if mailbox_depth <= 0 or mailbox_depth % producer_instances != 0:
        raise argparse.ArgumentTypeError(
            "forced mailbox depth must be a positive multiple of "
            f"{producer_instances} for C{cluster_size}"
        )
    return cluster_size, mailbox_depth


def _parse_varlen_profile(value: str) -> tuple[int, ...]:
    try:
        lengths = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated positive lengths, got {value!r}"
        ) from error
    if not lengths or any(length <= 0 for length in lengths):
        raise argparse.ArgumentTypeError(
            f"varlen profile lengths must be positive, got {value!r}"
        )
    return lengths


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
) -> list[float]:
    return [
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


def _run_case(
    *,
    batch_size: int,
    sequence_length: int,
    varlen_profile: Optional[tuple[int, ...]],
    num_heads: int,
    seed: int,
    enable_cupti: bool,
    warmup_ms: int,
    bench_ms: int,
    state_rotations: int,
    forced_configs: list[tuple[int, int]],
    measurement_rounds: int,
) -> dict:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    packed = varlen_profile is not None
    sequence_lengths = (
        varlen_profile
        if varlen_profile is not None
        else (sequence_length,) * batch_size
    )
    num_sequences = len(sequence_lengths)
    total_tokens = sum(sequence_lengths)
    shape = (
        (1, total_tokens, num_heads, 128)
        if packed
        else (batch_size, sequence_length, num_heads, 128)
    )

    def randn(dims: tuple[int, ...]) -> torch.Tensor:
        return torch.randn(dims, generator=generator, device="cuda").to(torch.bfloat16)

    q, k, v, g = (randn(shape) for _ in range(4))
    beta = randn(shape[:-1])
    A_log = torch.rand((num_heads,), generator=generator, device="cuda")
    dt_bias = torch.rand((num_heads, 128), generator=generator, device="cuda")
    initial = randn((num_sequences, num_heads, 128, 128))
    offsets = [0]
    for length in sequence_lengths:
        offsets.append(offsets[-1] + length)
    cu_seqlens = (
        torch.tensor(offsets, dtype=torch.int64, device="cuda") if packed else None
    )
    seq_order = (
        torch.tensor(
            sorted(
                range(num_sequences),
                key=sequence_lengths.__getitem__,
                reverse=True,
            ),
            dtype=torch.int32,
            device="cuda",
        )
        if packed
        else None
    )
    forced_routes = {
        f"c{cluster_size}_d{mailbox_depth}": (
            "m128_k1_parallel",
            cluster_size,
            mailbox_depth,
        )
        for cluster_size, mailbox_depth in forced_configs
    }
    routes = {
        "m64": ("m64", 0, 0),
        "m128": ("m128", 0, 0),
        **forced_routes,
        "k1_parallel": None,
    }
    outputs = {name: torch.empty_like(q) for name in routes}
    states = {name: initial.clone() for name in outputs}
    timed_state_pool = (
        initial.unsqueeze(0).expand(state_rotations, *initial.shape).clone()
    )
    state_cursor = 0
    workspaces = {name: RecurrentKDAPrefillWorkspace(q.device) for name in outputs}

    def launch(name: str, *, timed: bool = False) -> object:
        nonlocal state_cursor
        state = states[name]
        if timed:
            if state_cursor >= state_rotations:
                raise RuntimeError(
                    f"{name} exhausted {state_rotations} preinitialized state slots"
                )
            state = timed_state_pool[state_cursor]
            state_cursor += 1
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
            cu_seqlens=cu_seqlens,
            seq_order=seq_order,
            prefill_workspace=workspaces[name],
        )

    for name, route in routes.items():
        with _physical_route(route):
            launch(name)
    torch.cuda.synchronize()
    for name in routes:
        if name == "m64":
            continue
        torch.testing.assert_close(outputs[name], outputs["m64"], atol=0, rtol=0)
        torch.testing.assert_close(states[name], states["m64"], atol=0, rtol=0)

    samples = {name: [] for name in routes}
    state_slots_used = {name: [] for name in routes}
    route_orders = []
    order_generator = random.Random(seed)
    for _ in range(measurement_rounds):
        route_order = list(routes)
        order_generator.shuffle(route_order)
        route_orders.append(route_order)
        for name in route_order:
            route = routes[name]
            timed_state_pool.copy_(initial.unsqueeze(0))
            state_cursor = 0
            torch.cuda.synchronize()
            with _physical_route(route):
                round_samples = _measure(
                    lambda name=name: launch(name, timed=True),
                    enable_cupti=enable_cupti,
                    warmup_ms=warmup_ms,
                    bench_ms=bench_ms,
                )
            state_slots_used[name].append(state_cursor)
            samples[name].extend(round_samples)

    timings = {name: float(np.median(values)) for name, values in samples.items()}

    oracle_ms = min(timings["m64"], timings["m128"])
    forced_results = {
        name: {
            "cluster_size": route[1],
            "mailbox_depth": route[2],
            "latency_ms": timings[name],
            "speedup_vs_oracle": oracle_ms / timings[name],
        }
        for name, route in forced_routes.items()
    }
    return {
        "batch_size": 1 if packed else batch_size,
        "num_sequences": num_sequences,
        "sequence_length": sequence_length if not packed else None,
        "sequence_lengths": list(sequence_lengths),
        "total_tokens": total_tokens,
        "layout": "packed" if packed else "fixed",
        "num_heads": num_heads,
        "task_count": num_sequences * num_heads,
        "m64_ms": timings["m64"],
        "m128_ms": timings["m128"],
        "oracle_ms": oracle_ms,
        "k1_parallel_ms": timings["k1_parallel"],
        "speedup_vs_oracle": oracle_ms / timings["k1_parallel"],
        "forced_results": forced_results,
        "samples_ms": samples,
        "correctness": "bitwise output and state",
        "timing_backend": "cupti" if enable_cupti else "cuda_event",
        "cold_l2": True,
        "cuda_graph": False,
        "same_initial_state_per_timed_call": True,
        "state_slots_used_per_round": state_slots_used,
        "measurement_rounds": measurement_rounds,
        "route_orders": route_orders,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--sequence-lengths", type=int, nargs="+", default=[1024, 2048, 4096, 8192]
    )
    parser.add_argument(
        "--varlen-profiles",
        type=_parse_varlen_profile,
        nargs="*",
        default=[],
        metavar="L0,L1,...",
        help=(
            "benchmark packed varlen profiles instead of fixed shapes, for "
            "example: --varlen-profiles 8192 4096,3072,2048,1024"
        ),
    )
    parser.add_argument("--num-heads", type=int, nargs="+", default=[8, 16, 24, 32])
    parser.add_argument("--warmup-ms", type=int, default=20)
    parser.add_argument("--bench-ms", type=int, default=100)
    parser.add_argument("--state-rotations", type=int, default=2048)
    parser.add_argument(
        "--measurement-rounds",
        type=int,
        default=1,
        help="repeat timings with deterministic shuffled route order",
    )
    parser.add_argument(
        "--forced-configs",
        type=_parse_forced_config,
        nargs="*",
        default=[],
        metavar="C:D",
        help=(
            "force additional cluster-size/mailbox-depth routes, for "
            "example: --forced-configs 4:15 4:30 8:35"
        ),
    )
    parser.add_argument("--cupti", action="store_true")
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.measurement_rounds <= 0:
        parser.error("--measurement-rounds must be positive")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    compute_capability = get_compute_capability(torch.device("cuda"))
    if compute_capability not in ((10, 0), (10, 3)):
        raise RuntimeError(
            "CAKE K1 owner/helper benchmarking requires CC 10.0 or CC 10.3"
        )
    properties = torch.cuda.get_device_properties(0)
    metadata = {
        "device_name": properties.name,
        "compute_capability": list(compute_capability),
        "multiprocessor_count": properties.multi_processor_count,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
    }
    print(json.dumps(metadata, sort_keys=True))

    results = []
    cases = (
        [(sum(profile), profile) for profile in args.varlen_profiles]
        if args.varlen_profiles
        else [(sequence_length, None) for sequence_length in args.sequence_lengths]
    )
    for sequence_length, varlen_profile in cases:
        for num_heads in args.num_heads:
            result = _run_case(
                batch_size=args.batch_size,
                sequence_length=sequence_length,
                varlen_profile=varlen_profile,
                num_heads=num_heads,
                seed=args.seed + sequence_length + num_heads,
                enable_cupti=args.cupti,
                warmup_ms=args.warmup_ms,
                bench_ms=args.bench_ms,
                state_rotations=args.state_rotations,
                forced_configs=args.forced_configs,
                measurement_rounds=args.measurement_rounds,
            )
            result["hardware"] = metadata
            results.append(result)
            shape_label = (
                f"B={args.batch_size} T={sequence_length:5d}"
                if varlen_profile is None
                else "L=" + ",".join(str(length) for length in varlen_profile)
            )
            print(
                f"{shape_label} H={num_heads:2d} "
                f"M64={result['m64_ms']:.6f} ms "
                f"M128={result['m128_ms']:.6f} ms "
                f"K1={result['k1_parallel_ms']:.6f} ms "
                f"speedup={result['speedup_vs_oracle']:.3f}x",
                flush=True,
            )
            for name, forced in result["forced_results"].items():
                print(
                    f"  {name} depth={forced['mailbox_depth']:2d} "
                    f"latency={forced['latency_ms']:.6f} ms "
                    f"speedup={forced['speedup_vs_oracle']:.3f}x",
                    flush=True,
                )

    if args.json is not None:
        args.json.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
