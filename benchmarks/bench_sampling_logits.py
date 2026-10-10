"""Benchmark the public logits sampler; run unchanged on each revision to compare.

Example: python benchmarks/bench_sampling_logits.py --output logits-sampling.json
Compilation, allocations and Python dispatch are excluded from CUDA-graph timing.
"""

import argparse
import json
import math
import statistics

import torch

import flashinfer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 8, 32, 128, 512])
    parser.add_argument(
        "--vocabs", nargs="+", type=int, default=[32000, 128256, 151936]
    )
    parser.add_argument("--rounds", type=int, default=11)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if min(args.batches + args.vocabs + [args.rounds]) < 1:
        parser.error("shapes and rounds must be positive")
    torch.manual_seed(12345)
    rows = []
    for batch in args.batches:
        for vocab in args.vocabs:
            for cache in ["hot", "streaming_128MiB"]:
                count = (
                    30
                    if cache == "hot"
                    else max(2, math.ceil((128 << 20) / (batch * vocab * 4)))
                )
                inputs = torch.randn(
                    1 if cache == "hot" else count, batch, vocab, device="cuda"
                )
                sequence = (
                    [inputs[0]] * count if cache == "hot" else list(inputs.unbind())
                )

                def sample(x):
                    return flashinfer.sampling_from_logits(x, seed=12345, offset=17)

                for _ in range(3):
                    sample(sequence[0])
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for x in sequence:
                        sample(x)
                graph.replay()
                torch.cuda.synchronize()
                times = []
                for _ in range(args.rounds):
                    start, end = (
                        torch.cuda.Event(enable_timing=True) for _ in range(2)
                    )
                    start.record()
                    graph.replay()
                    end.record()
                    end.synchronize()
                    times.append(start.elapsed_time(end) * 1000 / count)
                row = dict(
                    batch=batch,
                    vocab=vocab,
                    cache=cache,
                    us=times,
                    median_us=statistics.median(times),
                )
                rows.append(row)
                print(json.dumps(row), flush=True)
                del graph, sequence, inputs, x
    result = dict(
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        flashinfer=flashinfer.__version__,
        dtype="float32",
        rows=rows,
    )
    with open(args.output, "w") as output:
        json.dump(result, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
