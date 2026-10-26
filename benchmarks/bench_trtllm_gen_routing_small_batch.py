"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Small-batch DeepSeek-V3-style routing (256 experts, top-8, SigmoidBias + ScaledSumNormalize):
routingCustom's single-launch small-batch kernels (1-64 tokens) vs the kernels they replace
(static block <= 4 tokens, dyn-block 5-16, block-scores + post-TopK from 17), selected with
FLASHINFER_ROUTING_SMALL_BATCH_KERNEL (read per call).

Timing: CUDA-graph replay (20 calls per graph, median over replays), alternating the two paths
over several rounds; CUPTI is not required. Ops:
  routing      the routing kernel(s) alone (trtllm_gen_routing module, preallocated outputs)
  moe          trtllm_fp4_block_scale_moe (NVFP4, FromLogits, deferred finalize)
  moe+side     the same with a BF16 shared-expert MLP on a forked stream, as SGLang's
               dual-stream DeepSeek MoE runs it (contention for SMs)

Example:
  python benchmarks/bench_trtllm_gen_routing_small_batch.py --ops routing moe --inter 512
"""

import argparse
import contextlib
import os
import statistics

import torch

from flashinfer.fused_moe import trtllm_fp4_block_scale_moe
from flashinfer.fused_moe.trtllm_gen_routing import (
    _max_num_ctas_in_batch_dim,
    get_trtllm_gen_routing_module,
)
from flashinfer.tllm_enums import ActivationType, RoutingMethodType, SfLayout

NUM_EXPERTS, TOP_K, HIDDEN = 256, 8, 6144
SWITCH = "FLASHINFER_ROUTING_SMALL_BATCH_KERNEL"


@contextlib.contextmanager
def small_batch_kernel(enabled):
    saved = os.environ.get(SWITCH)
    os.environ[SWITCH] = "1" if enabled else "0"
    try:
        yield
    finally:
        if saved is None:
            os.environ.pop(SWITCH, None)
        else:
            os.environ[SWITCH] = saved


def graph_time_us(fn, side=None, calls=20, replays=30):
    """Median per-call time of fn() replayed in a CUDA graph; side() runs on a forked stream."""
    side_stream = torch.cuda.Stream()

    def body():
        if side is not None:
            side_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side_stream):
                side()
        fn()
        if side is not None:
            torch.cuda.current_stream().wait_stream(side_stream)

    warm = torch.cuda.Stream()
    warm.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warm):
        for _ in range(3):
            body()
    torch.cuda.current_stream().wait_stream(warm)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(calls):
            body()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(replays):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000.0 / calls)
    return statistics.median(times)


def glm_inputs(T, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    logits = 2.0 * torch.randn(T, NUM_EXPERTS, generator=gen, device="cuda")
    bias = 8.0 + 0.03 * torch.randn(NUM_EXPERTS, generator=gen, device="cuda")
    return logits.contiguous(), bias.contiguous()


def routing_op(T, tile):
    logits, bias = glm_inputs(T, T)
    module = get_trtllm_gen_routing_module()
    max_ctas = _max_num_ctas_in_batch_dim(T, TOP_K, NUM_EXPERTS, tile)

    def i32(*shape):
        return torch.empty(shape, dtype=torch.int32, device="cuda")

    bufs = (
        i32(T, TOP_K),
        torch.empty(T, TOP_K, dtype=torch.bfloat16, device="cuda"),
        i32(2 * NUM_EXPERTS),
        i32(1),
        i32(T, TOP_K),
        i32(max_ctas * tile),
        i32(max_ctas),
        i32(max_ctas),
        i32(1),
    )

    def run():
        module.trtllm_gen_routing(
            logits,
            bias,
            *bufs,
            TOP_K,
            0,
            1,
            1,
            0,
            NUM_EXPERTS,
            2.5,
            tile,
            int(RoutingMethodType.DeepSeekV3),
            True,
            True,
        )

    return run


class Fp4Moe:
    def __init__(self, inter):
        gen = torch.Generator(device="cuda").manual_seed(11)

        def fp4(*shape):
            return torch.randint(
                0, 256, shape, generator=gen, device="cuda", dtype=torch.uint8
            )

        def scales(*shape):
            return (0.5 + torch.rand(*shape, generator=gen, device="cuda")).to(
                torch.float8_e4m3fn
            )

        self.inter = inter
        self.weights = dict(
            gemm1_weights=fp4(NUM_EXPERTS, 2 * inter, HIDDEN // 2),
            gemm1_weights_scale=scales(NUM_EXPERTS, 2 * inter, HIDDEN // 16),
            gemm2_weights=fp4(NUM_EXPERTS, HIDDEN, inter // 2),
            gemm2_weights_scale=scales(NUM_EXPERTS, HIDDEN, inter // 16),
            output1_scale_scalar=torch.full((NUM_EXPERTS,), 1e-3, device="cuda"),
            output1_scale_gate_scalar=torch.full((NUM_EXPERTS,), 1e-2, device="cuda"),
            output2_scale_scalar=torch.full((NUM_EXPERTS,), 1e-2, device="cuda"),
        )

    def op(self, T):
        logits, bias = glm_inputs(T, T)
        gen = torch.Generator(device="cuda").manual_seed(T + 1)
        hs = torch.randint(
            0, 256, (T, HIDDEN // 2), generator=gen, device="cuda", dtype=torch.uint8
        )
        hs_scale = (0.5 + torch.rand(T, HIDDEN // 16, generator=gen, device="cuda")).to(
            torch.float8_e4m3fn
        )

        def run():
            trtllm_fp4_block_scale_moe(
                logits,
                bias,
                hs,
                hs_scale,
                hidden_states_scale_layout=SfLayout.layout_linear,
                gemm1_bias=None,
                gemm1_alpha=None,
                gemm1_beta=None,
                gemm1_clamp_limit=None,
                gemm2_bias=None,
                num_experts=NUM_EXPERTS,
                top_k=TOP_K,
                n_group=1,
                topk_group=1,
                intermediate_size=self.inter,
                local_expert_offset=0,
                local_num_experts=NUM_EXPERTS,
                routed_scaling_factor=2.5,
                routing_method_type=int(RoutingMethodType.DeepSeekV3),
                do_finalize=False,
                enable_pdl=True,
                activation_type=int(ActivationType.Swiglu),
                **self.weights,
            )

        return run

    def shared_expert(self, T):
        gen = torch.Generator(device="cuda").manual_seed(77)
        h = torch.randn(T, HIDDEN, generator=gen, device="cuda").to(torch.bfloat16)
        w_gate_up = (
            0.02 * torch.randn(2 * self.inter, HIDDEN, generator=gen, device="cuda")
        ).to(torch.bfloat16)
        w_down = (
            0.02 * torch.randn(HIDDEN, self.inter, generator=gen, device="cuda")
        ).to(torch.bfloat16)

        def run():
            gate_up = torch.mm(h, w_gate_up.t())
            act = (
                torch.nn.functional.silu(gate_up[:, : self.inter])
                * gate_up[:, self.inter :]
            )
            torch.mm(act, w_down.t())

        return run


def main():
    p = argparse.ArgumentParser(description=__doc__.split("Example")[0])
    p.add_argument("--ops", nargs="+", default=["routing", "moe", "moe+side"])
    p.add_argument("--tokens", type=int, nargs="+", default=None)
    p.add_argument("--tile", type=int, default=8, help="routing op tile size")
    p.add_argument("--inter", type=int, default=512, help="per-rank intermediate size")
    p.add_argument("--rounds", type=int, default=3)
    args = p.parse_args()

    print(
        f"device: {torch.cuda.get_device_name()}, inter {args.inter}, tile {args.tile}"
    )
    print("op,tokens,old_us,new_us,saved_us,old_spread_us,new_spread_us")
    moe = Fp4Moe(args.inter) if any(o.startswith("moe") for o in args.ops) else None
    for op in args.ops:
        tokens = args.tokens or (
            list(range(1, 65))
            if op == "routing"
            else [1, 3, 6, 8, 12, 16, 18, 19, 24, 32, 40, 48, 56, 64]
        )
        for T in tokens:
            fn = routing_op(T, args.tile) if op == "routing" else moe.op(T)
            side = moe.shared_expert(T) if op == "moe+side" else None
            samples = {False: [], True: []}
            for _ in range(args.rounds):
                for enabled in (False, True):
                    with small_batch_kernel(enabled):
                        samples[enabled].append(graph_time_us(fn, side))
            old = statistics.median(samples[False])
            new = statistics.median(samples[True])
            spread = {k: max(v) - min(v) for k, v in samples.items()}
            print(
                f"{op},{T},{old:.2f},{new:.2f},{old - new:.2f},"
                f"{spread[False]:.2f},{spread[True]:.2f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
