"""Compare default BHSD conversion with direct BSHD output on prequantized inputs."""

import argparse
import json
import statistics

import torch

from flashinfer.cute_dsl.sparse.bsa_attn_sm100_blk64 import bsa_attn_sm100_blk64_fwd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=39756)
    parser.add_argument("--heads", type=int, default=14)
    parser.add_argument("--topk", type=int, default=160)
    args = parser.parse_args()
    torch.manual_seed(42)
    q, k, v = [
        torch.randn(
            1, args.seq_len, args.heads, 128, device="cuda", dtype=torch.bfloat16
        )
        .mul_(0.5)
        .to(torch.float8_e4m3fn)
        for _ in range(3)
    ]
    blocks = (args.seq_len + 63) // 64
    index = (
        (
            (
                torch.arange(blocks, device="cuda")[:, None]
                + torch.arange(args.topk, device="cuda")
            )
            % blocks
        )
        .to(torch.int32)[None, None]
        .expand(1, args.heads, -1, -1)
        .contiguous()
    )
    kwargs = dict(
        q_scale=torch.ones(1, args.heads, args.seq_len, device="cuda"),
        k_scale=torch.ones(1, args.heads, (args.seq_len + 15) // 16, device="cuda"),
        v_scale=torch.ones(args.heads, 128, device="cuda"),
        kv_splits=1,
        use_clc=False,
    )
    dest = torch.empty(
        (1, args.seq_len, args.heads, 128), device="cuda", dtype=torch.bfloat16
    )
    graphs, outputs = {}, {}
    for name, out in [("default_conversion", None), ("direct_bshd", dest)]:
        for _ in range(5):
            bsa_attn_sm100_blk64_fwd(q, k, v, index, args.topk, **kwargs, out=out)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs[name] = bsa_attn_sm100_blk64_fwd(
                q, k, v, index, args.topk, **kwargs, out=out
            )[0]
        graphs[name] = graph
        graph.replay()
        torch.cuda.synchronize()
    torch.testing.assert_close(
        outputs["direct_bshd"], outputs["default_conversion"], atol=0, rtol=0
    )
    samples = {name: [] for name in graphs}
    for _ in range(7):
        for name, graph in graphs.items():
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record()
            for _ in range(20):
                graph.replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) / 20)
    print(
        json.dumps(
            dict(
                gpu=torch.cuda.get_device_name(),
                shape=[1, args.seq_len, args.heads, 128],
                topk=args.topk,
                method="CUDA events around warm CUDA graph replays; 7 rounds x 20 replays",
                milliseconds={
                    name: statistics.median(values) for name, values in samples.items()
                },
                bitwise_equal=True,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
