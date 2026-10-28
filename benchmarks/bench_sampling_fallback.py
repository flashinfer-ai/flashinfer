"""Measure probability sampling with and without the preceding softmax.

Run on each revision using separate JIT caches. Timings cover the sampling
stage, not a model forward pass or an end-to-end serving request.
"""

import argparse
import json
import random
import statistics

import torch

import flashinfer
from flashinfer.testing import bench_gpu_time
from flashinfer.testing.utils import calculate_rotation_count


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 16, 128])
    parser.add_argument("--vocab-sizes", type=int, nargs="+", default=[151936])
    parser.add_argument("--std", type=float, default=1.0)
    parser.add_argument("--order-seed", type=int, default=42)
    parser.add_argument("--cold-l2", action="store_true")
    parser.add_argument("--non-deterministic", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(42)
    order_rng = random.Random(args.order_seed)
    common = dict(seed=42, offset=0, deterministic=not args.non_deterministic)
    samplers = {
        "plain": lambda p: flashinfer.sampling.sampling_from_probs(p, **common),
        "top_k": lambda p: flashinfer.sampling.top_k_sampling_from_probs(
            p, 50, **common
        ),
        "top_p": lambda p: flashinfer.sampling.top_p_sampling_from_probs(
            p, 0.9, **common
        ),
        "min_p": lambda p: flashinfer.sampling.min_p_sampling_from_probs(
            p, 0.1, **common
        ),
        "joint": lambda p: flashinfer.sampling.top_k_top_p_sampling_from_probs(
            p, 50, 0.9, filter_apply_order="joint", **common
        ),
    }
    print(json.dumps({"gpu": torch.cuda.get_device_name(), "torch": torch.__version__}))
    case_index = 0
    for batch in args.batch_sizes:
        for vocab in args.vocab_sizes:
            logits = torch.randn(batch, vocab, device="cuda") * args.std
            probs = torch.softmax(logits, dim=-1)
            graph_iters = 10
            if args.cold_l2:
                # Visit every rotated buffer before replaying the graph; otherwise
                # small inputs can still fit in L2 across successive replays.
                rotations = calculate_rotation_count([logits])
                graph_iters = ((graph_iters + rotations - 1) // rotations) * rotations
                if graph_iters > 4096:
                    raise ValueError(
                        "Use larger inputs for the cold-L2 graph benchmark"
                    )
            cases = [
                (name, with_softmax)
                for name in samplers
                for with_softmax in [False, True]
            ]
            order_rng.shuffle(cases)
            for name, with_softmax in cases:
                sample = samplers[name]

                def run(x):
                    return sample(torch.softmax(x, dim=-1) if with_softmax else x)

                measurements = bench_gpu_time(
                    run,
                    input_args=(logits if with_softmax else probs,),
                    enable_cupti=False,
                    use_cuda_graph=True,
                    cold_l2_cache=args.cold_l2,
                    num_iters_within_graph=graph_iters,
                    dry_run_iters=5,
                    repeat_iters=30,
                )
                print(
                    json.dumps(
                        dict(
                            case_index=case_index,
                            order_seed=args.order_seed,
                            batch=batch,
                            vocab=vocab,
                            sampler=name,
                            with_softmax=with_softmax,
                            std=args.std,
                            deterministic=common["deterministic"],
                            cold_l2=args.cold_l2,
                            median_us=statistics.median(measurements) * 1000,
                        )
                    ),
                    flush=True,
                )
                case_index += 1


if __name__ == "__main__":
    main()
