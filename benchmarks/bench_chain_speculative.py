"""Benchmark chain speculative sampling; run unchanged on each revision to compare.

Example: python benchmarks/bench_chain_speculative.py --output chain-spec.json
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
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 8, 32, 128])
    parser.add_argument("--vocabs", nargs="+", type=int, default=[32000, 128256])
    parser.add_argument("--speculate", nargs="+", type=int, default=[1, 3, 5, 8])
    parser.add_argument("--rounds", type=int, default=11)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if min(args.batches + args.vocabs + args.speculate + [args.rounds]) < 1:
        parser.error("shapes and rounds must be positive")
    torch.manual_seed(12345)
    rows = []
    for batch in args.batches:
        for vocab in args.vocabs:
            for speculate in args.speculate:
                bytes_per_call = batch * (2 * speculate + 1) * vocab * 4
                count = max(2, math.ceil((128 << 20) / bytes_per_call))
                count = min(count, 30)
                pre_norm_draft = torch.rand(batch, speculate, vocab, device="cuda")
                draft_probs = pre_norm_draft / pre_norm_draft.sum(dim=-1, keepdim=True)
                draft_token_ids = torch.randint(
                    vocab, (batch, speculate), device="cuda", dtype=torch.int32
                )
                pre_norm_target = torch.rand(batch, speculate + 1, vocab, device="cuda")
                target_probs = pre_norm_target / pre_norm_target.sum(
                    dim=-1, keepdim=True
                )
                accepted = torch.zeros(batch, dtype=torch.int32, device="cuda")
                emitted = torch.zeros(batch, dtype=torch.int32, device="cuda")
                seed = torch.tensor([12345], device="cuda", dtype=torch.uint64)
                offset = torch.tensor([17], device="cuda", dtype=torch.uint64)

                def sample():
                    return flashinfer.chain_speculative_sampling(
                        draft_probs,
                        draft_token_ids,
                        target_probs,
                        accepted,
                        emitted,
                        seed=seed,
                        offset=offset,
                    )

                for _ in range(3):
                    sample()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(count):
                        sample()
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
                    speculate=speculate,
                    us=times,
                    median_us=statistics.median(times),
                )
                rows.append(row)
                print(json.dumps(row), flush=True)
                # The captured graph holds references to the tensors sample() closes
                # over, so only the graph itself is released here.
                del graph
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
