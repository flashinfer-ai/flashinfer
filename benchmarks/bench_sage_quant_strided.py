"""Compare the public Sage API with and without input materialization.

Example: python benchmarks/bench_sage_quant_strided.py --seq-len 39756 --heads 14
Timing covers CUDA graph replay of the complete quantization API.
"""

import argparse
import json
import statistics

import torch
import flashinfer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=39756)
    parser.add_argument("--heads", type=int, default=14)
    parser.add_argument("--dim", type=int, default=128)
    args = parser.parse_args()
    torch.manual_seed(42)
    packed = torch.randn(
        args.seq_len, 3, args.heads, args.dim, device="cuda", dtype=torch.bfloat16
    )
    views = tuple(packed[:, i] for i in range(3))
    dense = tuple(t.contiguous() for t in views)
    bounds = torch.tensor([0, args.seq_len], device="cuda", dtype=torch.int32)
    kwargs = dict(
        q_block_size=1,
        k_block_size=16,
        qk_quant_dtype=torch.float8_e4m3fn,
        cum_seq_lens_q=bounds,
        cum_seq_lens_kv=bounds,
        smooth_k=False,
    )
    graphs, outputs = {}, {}
    for name, inputs, opt in [
        ("packed_default", dense, False),
        ("strided_default", views, False),
        ("strided_direct", views, True),
    ]:
        for _ in range(5):
            flashinfer.trtllm_sage_attention_quantize(
                *inputs, **kwargs, allow_strided_input=opt
            )
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs[name] = flashinfer.trtllm_sage_attention_quantize(
                *inputs, **kwargs, allow_strided_input=opt
            )
        graphs[name] = graph
        graph.replay()
        torch.cuda.synchronize()
    for output in outputs.values():
        for a, b in zip(outputs["packed_default"], output, strict=True):
            assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))
    samples = {name: [] for name in graphs}
    for _ in range(7):
        for name, graph in graphs.items():
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record()
            for _ in range(30):
                graph.replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) / 30)
    print(
        json.dumps(
            dict(
                gpu=torch.cuda.get_device_name(),
                shape=[args.seq_len, args.heads, args.dim],
                method="CUDA events around warm CUDA graph replays; 7 rounds x 30 replays",
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
