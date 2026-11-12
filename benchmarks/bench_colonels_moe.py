import argparse
import statistics

import torch

from flashinfer.autotuner import autotune
from flashinfer.fused_moe import (
    BackendOptions,
    ColonelsBf16Config,
    ExpertConfig,
    ExecutionConfig,
    MoEActivationPack,
    MoEConfig,
    MoELayer,
    MoEWeightPack,
    QuantConfig,
    RoutingConfig,
)
from flashinfer.testing import bench_gpu_time_with_cudagraph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 16, 128, 1024])
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--intermediate", type=int, default=1024)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    args = parser.parse_args()
    torch.manual_seed(0)
    e, h, i, k = args.experts, args.hidden, args.intermediate, args.top_k
    backend = ColonelsBf16Config()
    w1 = torch.randn(e, 2 * i, h, device="cuda", dtype=torch.bfloat16) * 0.01
    w2 = torch.randn(e, h, i, device="cuda", dtype=torch.bfloat16) * 0.01
    weights = MoEWeightPack()
    weights.prepare_for(
        "colonels_bf16",
        backend.prepare_weights(
            w1, w2, num_local_experts=e, hidden_size=h, intermediate_size=i
        ),
    )
    layer = MoELayer(
        MoEConfig(
            routing=RoutingConfig(num_experts=e, top_k=k),
            quant=QuantConfig(),
            experts=ExpertConfig(intermediate_size=i),
            execution=ExecutionConfig(tune_max_num_tokens=max(8192, max(args.tokens))),
            backend=BackendOptions(candidates=(backend,)),
        )
    )
    print("tokens,hidden,intermediate,experts,top_k,median_us")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for m in args.tokens:
            x = torch.randn(m, h, device="cuda", dtype=torch.bfloat16)
            logits = torch.randn(m, e, device="cuda")
            scores, ids = logits.topk(k, dim=-1)
            act = MoEActivationPack(x, None, ids.int(), scores.softmax(dim=-1))
            with autotune(True):
                layer(act, weights)
            times = bench_gpu_time_with_cudagraph(
                lambda: layer(act, weights), cold_l2_cache=False
            )
            print(f"{m},{h},{i},{e},{k},{1000 * statistics.median(times):.3f}")
    torch.cuda.current_stream().wait_stream(stream)


if __name__ == "__main__":
    main()
