# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FP8 sparse MLA decode scheduling sweep, T=7*B, H64, D512, K=128+512.

python -m benchmarks.bench_sparse_mla_sm120_decode --batches 1 2 3 4 5 7 20 --output decode.json
All fixture construction and independent checks happen before graph capture.
"""

import argparse
import json
import statistics
import time
from pathlib import Path
import torch
from tests.attention.sparse_mla_prefill_utils import (
    decode_fixture,
    numerical_metrics,
    decode_call as call,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", nargs="+", type=int, default=list(range(1, 21)))
    parser.add_argument("--output", type=Path, default=Path("decode.json"))
    parser.add_argument("--layout", choices=("inline", "footer"), default="inline")
    args = parser.parse_args()
    results = []
    for batch in args.batches:
        f = decode_fixture(batch, inline=args.layout == "inline")
        functions = {
            impl: lambda impl=impl: call(f, impl)
            for impl in ("auto", "decode-splitkv", "decode")
        }
        functions["decode-splitkv-unfused"] = lambda: call(f, "decode-splitkv", False)
        functions["decode-unfused"] = lambda: call(f, "decode", False)
        graphs = {}
        metrics = {}
        for name, fn in functions.items():
            lse = fn()
            metrics[name] = numerical_metrics(f, lse)
            metric = metrics[name]
            assert (
                metric["finite_output"]
                and metric["relative_l2"] < 0.07
                and metric["lse_max_abs"] < 0.06
            ), (name, metric)
            for _ in range(5):
                fn()
            graph = torch.cuda.CUDAGraph()
            start = torch.cuda.Event(enable_timing=True, external=True)
            end = torch.cuda.Event(enable_timing=True, external=True)
            with torch.cuda.graph(graph):
                start.record()
                for _ in range(10):
                    fn()
                end.record()
            graphs[name] = (graph, start, end)
        until = time.monotonic() + 2
        while time.monotonic() < until:
            for graph, _, _ in graphs.values():
                graph.replay()
            torch.cuda.synchronize()
        samples = {name: [] for name in graphs}
        for repeat in range(28):
            order = list(graphs)
            order = order[repeat % len(order) :] + order[: repeat % len(order)]
            for name in order:
                graph, start, end = graphs[name]
                graph.replay()
                graph.replay()
                end.synchronize()
                samples[name].append(start.elapsed_time(end) * 100)
        plans = {
            key[1]: dict(p.plan.inspect())
            for key, p in f["wrapper"]._prepared_calls.items()
        }
        row = dict(
            layout=args.layout,
            batch=batch,
            tokens=batch * 7,
            metrics=metrics,
            plans=plans,
            samples_us=samples,
            median_us={
                name: statistics.median(times) for name, times in samples.items()
            },
        )
        results.append(row)
        print(batch, row["median_us"], flush=True)
        args.output.write_text(
            json.dumps(
                dict(gpu=torch.cuda.get_device_name(), results=results), indent=2
            )
        )


if __name__ == "__main__":
    main()
