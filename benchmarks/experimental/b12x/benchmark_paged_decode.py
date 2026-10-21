#!/usr/bin/env python3
"""Prepared attention.paged_decode timings on MiMo-V2 layer families.

Families (per TP4 rank): global 16/1 heads QK192/V128, sliding-window 16/2
heads with sinks, and the DFlash drafter 16/2 heads QK128/V128 non-causal.
Each case builds a hybrid DiffKV page pool, runs one prepared plan, and reports
the forward+merge kernel time.  ``--fp8`` adds E4M3 KV rows.

    python benchmarks/experimental/b12x/benchmark_paged_decode.py [--fp8] [--contexts 4096,32768,131072]
        [--batches 1,8,32] [--q 1,4,8]
"""

from __future__ import annotations

import argparse

import torch

from b12x.attention import paged_decode
from b12x.attention.paged.reference import paged_attention_reference
from b12x.preparation import PreparationSession, PreparedCall
from b12x.testing.benchmark import measure_call

PAGE = 64
FAMILIES = {
    # name: (q_heads, kv_heads, dim_qk, dim_vo, window_left, causal, sinks)
    "global": (16, 1, 192, 128, -1, True, False),
    "swa": (16, 2, 192, 128, 127, True, True),
    "draft": (16, 2, 128, 128, 1023, False, True),
}


def _inputs(family, batch, q_len, context, fp8, gen):
    hq, hkv, dqk, dvo, window, causal, sinks = FAMILIES[family]
    width = (context + PAGE - 1) // PAGE
    pool = batch * width + 1
    packed = torch.randn((pool, 1, hkv, PAGE, dqk + dvo), generator=gen, device="cuda")
    descale = None
    if fp8:
        packed = (packed / 8).to(torch.float8_e4m3fn)
        descale = torch.tensor([8.0], device="cuda")
    else:
        packed = packed.to(torch.bfloat16)
    view = packed[:, 0].transpose(1, 2)
    perm = torch.randperm(pool, generator=gen, device="cuda").to(torch.int32)
    total = batch * q_len
    q = torch.randn((total, hq, dqk), generator=gen, device="cuda").to(torch.bfloat16)
    cu = torch.arange(0, total + 1, q_len, dtype=torch.int32, device="cuda")
    inputs = dict(
        q=q,
        k_cache=view[..., :dqk],
        v_cache=view[..., dqk:],
        page_table=perm[: batch * width].view(batch, width).contiguous(),
        cache_seqlens=torch.full((batch,), context, dtype=torch.int32, device="cuda"),
        cu_seqlens_q=cu,
        attention_sink_bias=(
            torch.randn(hq, generator=gen, device="cuda") if sinks else None
        ),
        k_descale=descale,
        v_descale=descale,
    )
    caps = paged_decode.Caps(
        device="cuda",
        num_q_heads=hq,
        num_kv_heads=hkv,
        head_dim_qk=dqk,
        head_dim_vo=dvo,
        page_size=PAGE,
        max_batch=batch,
        max_q_per_req=8,
        kv_dtype=torch.float8_e4m3fn if fp8 else torch.bfloat16,
        window_left=window,
        causal=causal,
        has_sinks=sinks,
    )
    return inputs, caps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp8", action="store_true")
    parser.add_argument("--families", default="global,swa,draft")
    parser.add_argument("--contexts", default="4096,32768,131072")
    parser.add_argument("--batches", default="1,8,32")
    parser.add_argument("--q", default="1,4,8")
    args = parser.parse_args()
    gen = torch.Generator(device="cuda").manual_seed(0)
    print(f"{'family':7s} {'kv':5s} {'ctx':>7s} {'B':>3s} {'q':>2s} {'us':>9s}")
    for family in args.families.split(","):
        for context in (int(c) for c in args.contexts.split(",")):
            for batch in (int(b) for b in args.batches.split(",")):
                for q_len in (int(q) for q in args.q.split(",")):
                    inputs, caps = _inputs(family, batch, q_len, context, args.fp8, gen)
                    plan = paged_decode.plan(caps)
                    out = torch.empty(
                        (batch * q_len, caps.num_q_heads, caps.head_dim_vo),
                        dtype=torch.bfloat16,
                        device="cuda",
                    )
                    scratch = [
                        torch.empty(s.shape, dtype=s.dtype, device="cuda")
                        for s in plan.scratch_specs()
                    ]

                    def prepare_call(state):
                        binding = state.bind(
                            plan=plan, scratch=scratch, output=out, **inputs
                        )
                        return PreparedCall(run=lambda: state.run(binding), output=out)

                    with PreparationSession(
                        device=torch.device("cuda"), autotune=False
                    ) as session:
                        result = session.prepare(
                            (plan.request(name="bench", prepare_call=prepare_call),)
                        )
                        binding = paged_decode.bind(
                            plan, scratch=scratch, output=out, **inputs
                        )
                        out.fill_(float("nan"))
                        paged_decode.run(binding)
                        torch.cuda.synchronize()
                        if (
                            not bool(torch.isfinite(out).all())
                            or not float(out.abs().amax()) > 0
                        ):
                            raise RuntimeError(
                                f"paged_decode output invalid: {family} ctx={context} "
                                f"B={batch} q={q_len}"
                            )
                        reference_inputs = dict(inputs)
                        for name in ("k_descale", "v_descale"):
                            if reference_inputs[name] is not None:
                                reference_inputs[name] = reference_inputs[name].expand(
                                    batch, caps.num_kv_heads
                                )
                        reference, _ = paged_attention_reference(
                            **reference_inputs,
                            causal=caps.causal,
                            window_left=caps.window_left,
                        )
                        torch.testing.assert_close(out, reference, rtol=0.02, atol=0.02)
                        measured = measure_call(
                            lambda: paged_decode.run(binding), eviction=lambda: None
                        )
                        us = measured.latencies_us["workload"]
                        result.close()
                    print(
                        f"{family:7s} {'fp8' if args.fp8 else 'bf16':5s} {context:7d} {batch:3d} "
                        f"{q_len:2d} {us:9.2f}",
                        flush=True,
                    )
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
