"""Compare state-tensor RNG reservation with the current eager sampling path.

Example: python benchmarks/bench_sampling_rng.py --batch-sizes 1 32 128

Reports host enqueue time and completed wall time per call (one synchronization
per measurement round). Neither metric is GPU-only kernel time or model
throughput. Baseline and current paths alternate order after JIT warmup, using
identical generator states and sampling kernels.
"""

import argparse
from functools import partial
import json
import statistics
import time

import torch

from flashinfer import sampling
from flashinfer.utils import get_default_generators


def _state_tensor_reservation(increment, generator=None, device=None):
    # Reference implementation from #9148, including its reservation lock.
    if generator is None:
        generator = get_default_generators(device)
    with sampling._rng_state_lock:
        state = generator.get_state()
        seed, offset = map(int, state.view(torch.int64))
        next_offset = offset + (increment + 3) // 4 * 4
        generator.set_state(
            torch.tensor(
                [seed, next_offset], dtype=torch.int64, device=torch.device("cpu")
            ).view(torch.uint8)
        )
    return seed, offset


def _measure(call, iterations):
    torch.cuda.synchronize()
    start = time.perf_counter_ns()
    for _ in range(iterations):
        call()
    enqueued = time.perf_counter_ns()
    torch.cuda.synchronize()
    completed = time.perf_counter_ns()
    return [(end - start) / iterations / 1000 for end in (enqueued, completed)]


def _compare(call, generator, args):
    current = sampling.get_seed_and_offset
    implementations = {"state_tensor": _state_tensor_reservation, "current": current}
    measurements = {name: [] for name in implementations}
    try:
        # Compilation and allocator initialization are outside all timed regions.
        for reserve in implementations.values():
            sampling.get_seed_and_offset = reserve
            generator.manual_seed(1234)
            for _ in range(args.warmup):
                call()
        for round_index in range(args.rounds):
            order = list(implementations)
            if round_index % 2:
                order.reverse()
            for name in order:
                sampling.get_seed_and_offset = implementations[name]
                generator.manual_seed(1234)
                generator.set_offset(128)
                measurements[name].append(_measure(call, args.iterations))
    finally:
        sampling.get_seed_and_offset = current
    return {
        name: {
            "host_us": statistics.median(row[0] for row in rows),
            "completed_us": statistics.median(row[1] for row in rows),
            "rounds_us": rows,
        }
        for name, rows in measurements.items()
    }


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 32, 128])
    parser.add_argument("--vocab-sizes", type=int, nargs="+", default=[32000, 128256])
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=9)
    args = parser.parse_args()
    device = torch.device("cuda", torch.cuda.current_device())
    print(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(device),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "config": vars(args),
            }
        ),
        flush=True,
    )
    with torch.random.fork_rng(devices=[device.index]):
        for default in (False, True):
            generator = (
                torch.cuda.default_generators[device.index]
                if default
                else torch.Generator(device=device)
            )
            generator_arg = None if default else generator

            def reserve():
                return sampling.get_seed_and_offset(4, generator_arg, device)

            print(
                json.dumps(
                    {
                        "routine": "reservation",
                        "default_generator": default,
                        **_compare(reserve, generator, args),
                    }
                ),
                flush=True,
            )
            for batch in args.batch_sizes:
                for vocab in args.vocab_sizes:
                    inputs = torch.randn(
                        batch,
                        vocab,
                        device=device,
                        generator=torch.Generator(device=device).manual_seed(42),
                    )
                    probs = inputs.softmax(-1)
                    calls = {
                        "sampling_from_probs": partial(
                            sampling.sampling_from_probs, probs
                        ),
                        "sampling_from_logits": partial(
                            sampling.sampling_from_logits, inputs
                        ),
                        "top_k_top_p_sampling_from_probs": partial(
                            sampling.top_k_top_p_sampling_from_probs,
                            probs,
                            100,
                            0.9,
                            filter_apply_order="joint",
                        ),
                    }
                    for name, sample in calls.items():
                        call = partial(sample, generator=generator_arg)
                        print(
                            json.dumps(
                                {
                                    "routine": name,
                                    "default_generator": default,
                                    "batch": batch,
                                    "vocab": vocab,
                                    **_compare(call, generator, args),
                                }
                            ),
                            flush=True,
                        )


if __name__ == "__main__":
    main()
