#!/usr/bin/env python3
"""Small-batch W6A8 MX-FP6 MoE benchmark using synthetic quantized expert weights.

Drives the ``b12x.moe.fused_moe`` plan/bind/run flow with an
``MXFP6_E8M0_K32`` source and times CUDA-graph
replays of ``fused_moe.run``.  Graph capture is safe here: ``bind`` is
documented capture-safe (views only, never allocates) and the analogous
w4a8_mx dynamic path is graph-capture gated upstream
(tests/moe/test_w4a8_mx_tp_moe.py).
"""

from __future__ import annotations

import argparse
import pathlib
import statistics
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from benchmarks.experimental.b12x.fp6_common import (
    bf16_grouped_moe,
    check_outputs,
    fmt_us,
    make_l2_flush_fn,
    resolve_l2_flush_bytes,  # noqa: F401  (re-exported CLI helper)
    unswizzled_ue8m0_grid,
)
from b12x.preparation import PreparationSession
from benchmarks.experimental.b12x.moe_preparation import prepared_call, request_for_capacity, scratch_for


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--flush-l2", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--l2-flush-bytes", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    from b12x.moe import fused_moe
    from b12x.quantization.mxfp6 import quantize_moe_weights_to_fp6

    # Fully-qualified device: the scratch binder compares device strings
    # exactly, and tensors allocated on "cuda" report "cuda:0".
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(0)
    m, k, n, e, topk = args.m, args.k, args.n, args.experts, args.topk
    if k % 128 != 0 or n % 128 != 0:
        raise SystemExit(
            f"w6a8_mx requires K % 128 == 0 and N % 128 == 0, got K={k} N={n}"
        )

    x = torch.randn(m, k, device=device, dtype=torch.bfloat16) * 0.1
    topk_ids = torch.randint(0, e, (m, topk), device=device, dtype=torch.int32)
    topk_weights = torch.softmax(
        torch.randn(m, topk, device=device), dim=-1
    ).to(torch.float32)

    w1_bf = torch.randn(e, 2 * n, k, device=device, dtype=torch.bfloat16) * 0.15
    w2_bf = torch.randn(e, k, n, device=device, dtype=torch.bfloat16) * 0.15
    w = quantize_moe_weights_to_fp6(w1_bf, w2_bf, source_format="mxfp6_e2m3")
    w1_grid = unswizzled_ue8m0_grid(w1_bf)
    w2_grid = unswizzled_ue8m0_grid(w2_bf)

    fused_moe.clear_caches()
    source = fused_moe.PackedSource(
        format=fused_moe.PackedSourceFormat.MXFP6_E8M0_K32,
        w13_layout=fused_moe.W13Layout.W13,
    )
    weight_plan = fused_moe.plan_weights(
        source=source,
        activation=fused_moe.ActivationSpec(
            mode=fused_moe.ActivationMode.A8,
            nonlinearity="silu",
            io_dtype=torch.bfloat16,
        ),
        geometry=fused_moe.MoEGeometry(
            num_experts=e,
            hidden_size=k,
            intermediate_size=n,
        ),
    )
    prepared = fused_moe.prepare_weights(
        plan=weight_plan,
        weights=fused_moe.PackedWeights(
            w13=w.w1_fp6,
            w2=w.w2_fp6,
            w13_block_scales=w1_grid,
            w2_block_scales=w2_grid,
            w13_global_scales=w.w1_alphas,
            w2_global_scales=w.w2_alphas,
            input_scale=w.a1_gscale,
            intermediate_scale=w.a2_gscale,
        ),
    )
    declaration = fused_moe.plan_execution(
        experts=prepared,
        capacity=fused_moe.ExecutionCapacity(
            max_tokens=m,
            top_k=topk,
            warmup_token_counts=(m,),
            route_num_experts=0,
        ),
    )
    out = torch.empty(m, k, device=device, dtype=torch.bfloat16)
    request = request_for_capacity(
        declaration,
        name="fp6-moe",
        calls={
            count: prepared_call(
                output=out,
                bind=lambda state, scratch: state.bind(
                    scratch=scratch,
                    a=x,
                    experts=prepared,
                    topk_weights=topk_weights,
                    topk_ids=topk_ids,
                    output=out,
                    input_scales_static=True,
                ),
            )
            for count in getattr(declaration, "token_counts", (m,))
        },
    )
    session = PreparationSession(device=device)
    result = session.prepare((request,))
    variants = getattr(declaration, "variants", None)
    plan = declaration if variants is None else variants.get(m, declaration)
    binding = fused_moe.bind(
        plan,
        scratch=scratch_for(plan),
        a=x,
        experts=prepared,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        output=out,
        input_scales_static=True,
    )

    def launch() -> None:
        fused_moe.run(binding=binding)

    # Correctness gate before any timing (coding guideline): a silently
    # broken execution path must fail here, not report timings. Also warms
    # the compiled launch. bf16_grouped_moe is the same reference/threshold
    # approach scripts/bench_fp6_moe.py uses.
    launch()
    torch.cuda.synchronize()
    ref = bf16_grouped_moe(x, w1_bf, w2_bf, topk_ids, topk_weights, n)
    check_outputs(out, ref, label="bf16 grouped MoE", cosine_threshold=0.99)
    del w1_bf, w2_bf

    graph = torch.cuda.CUDAGraph()
    for _ in range(3):
        launch()
    torch.cuda.synchronize()
    with session.capture(), torch.cuda.graph(graph):
        launch()

    def replay() -> None:
        graph.replay()

    l2_flush = make_l2_flush_fn(enabled=args.flush_l2, bytes_hint=args.l2_flush_bytes)
    replay()
    from b12x.testing.benchmark import samples_ms
    times = samples_ms(launch, warmup=args.warmup, iters=args.iters, l2_flush=l2_flush)
    med = statistics.median(times)
    print(
        f"W6A8 MX-FP6 MoE synthetic m={m} k={k} n={n} E={e} topk={topk}: "
        f"{fmt_us(times)}  (median {med:.3f} ms)"
    )
    graph.reset()
    result.close()
    session.close()


if __name__ == "__main__":
    from b12x.testing.memory import absorb_small_page_fragments

    absorb_small_page_fragments()
    main()
