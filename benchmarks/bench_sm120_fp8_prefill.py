#!/usr/bin/env python3
"""Validate and time opt-in SM120 FP8-activation MoE on synthetic inputs.

Run with the patched FlashInfer package on an idle SM120 GPU. This measures a
single GPU kernel path, not model prefill or serving throughput. Weights are
synthetic K32 E2M1 blocks. FP8 activation error is measured against both the
BF16 kernel and a PyTorch emulation of the FP8 operation order.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time

import torch

FP4 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def fp4_codes(x):
    ax = x.abs()
    code = torch.zeros_like(ax, dtype=torch.uint8)
    for b in (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0):
        code += (ax > b).to(torch.uint8)
    return code | ((x < 0) & (code != 0)).to(torch.uint8) << 3


def codes_to_f32(c):
    mag = FP4.to(c.device)[(c & 7).long()]
    return torch.where(c & 8 != 0, -mag, mag)


def quant_modelopt(w):
    wf = w.float()
    amax = wf.abs().amax(dim=(-1, -2)).clamp_min(1e-12)
    g = amax / (6.0 * 448.0)
    blk = wf.view(*wf.shape[:-1], wf.shape[-1] // 32, 32)
    s = (
        (blk.abs().amax(-1) / 6.0 / g[:, None, None])
        .clamp(max=448.0)
        .to(torch.float8_e4m3fn)
    )
    den = s.float() * g[:, None, None]
    q = blk / torch.where(den == 0, torch.ones_like(den), den)[..., None]
    codes = fp4_codes(q).view(*wf.shape)
    c2 = codes.view(*codes.shape[:-1], codes.shape[-1] // 2, 2)
    packed = (c2[..., 0] | c2[..., 1] << 4).contiguous()
    return (codes, packed, s.contiguous(), g.float().contiguous())


def b12x_scale_factor(s):
    mx = (s.float() * 2**7).max().item()
    if 0.0 < mx < 448 * 2**7:
        return float(2 ** math.floor(math.log2(448 * 2**7 / mx)))
    return 1.0


def prep_bytes(s, F):
    h = (s.float() * F).to(torch.float16) * 2**7
    h = torch.where(h < 2, torch.zeros_like(h), h)
    return (h.view(torch.int16).to(torch.int32) << 1 >> 8).to(torch.int32) & 255


def exact_prepared_weight(codes, s, g):
    F = b12x_scale_factor(s)
    b = prep_bytes(s, F)
    sc = (
        torch.where(
            b == 0,
            torch.zeros_like(b, dtype=torch.float32),
            (1.0 + (b & 7).float() / 8.0) * 2.0 ** ((b >> 3).float() - 22.0),
        )
        / F
    )
    v = codes_to_f32(codes)
    v = v.view(*v.shape[:-1], v.shape[-1] // 32, 32) * sc[..., None]
    return v.view(codes.shape) * g[:, None, None]


def fp8_rows_kernel(v):
    v = v.float()
    amax = v.abs().amax(-1, keepdim=True).clamp_min(1e-12)
    inv = torch.tensor(448.0, dtype=torch.float32, device=v.device) / amax
    q = (v * inv).to(torch.float8_e4m3fn).float()
    return (q, amax / 448.0)


def bf16(x):
    return x.to(torch.bfloat16).float()


def moe_mirror(x, ids, w, W13, W2, I):
    T, K = x.shape
    xq, xs = fp8_rows_kernel(x)
    out = torch.zeros(T, K, dtype=torch.float32, device=x.device)
    for e in range(W13.shape[0]):
        tok, slot = (ids == e).nonzero(as_tuple=True)
        if tok.numel() == 0:
            continue
        h_u = bf16(xq[tok] @ W13[e].t())
        s_tok = xs[tok]
        gate, up = (h_u[:, :I] * s_tok, h_u[:, I:] * s_tok)
        v = bf16(bf16(torch.nn.functional.silu(gate)) * bf16(up))
        aq, as_ = fp8_rows_kernel(v)
        y = bf16(aq @ W2[e].t() * (w[tok, slot][:, None] * as_))
        out.index_add_(0, tok, y)
    return bf16(out)


def build_weights(experts, hidden=4096, intermediate=1024):
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_w4a16_prepare import (
        prepare_w4a16_packed_weights,
    )

    gen = torch.Generator(device="cuda").manual_seed(7)
    unique = min(experts, 8)
    masters = [
        torch.randn(unique, 2 * intermediate, hidden, device="cuda", generator=gen)
        / math.sqrt(hidden),
        torch.randn(unique, hidden, intermediate, device="cuda", generator=gen)
        / math.sqrt(intermediate),
    ]
    packed, scales, globals_, refs = [], [], [], []
    for master in masters:
        codes, weight, scale, global_ = quant_modelopt(master.bfloat16())
        mask = torch.rand(scale.shape, device="cuda", generator=gen) < 0.015
        scale.view(torch.uint8)[mask] = 0
        refs.append(exact_prepared_weight(codes, scale, global_))
        for seq, value in ((packed, weight), (scales, scale), (globals_, global_)):
            seq.append(
                value.repeat(math.ceil(experts / unique), *([1] * (value.ndim - 1)))[
                    :experts
                ].contiguous()
            )
    # Input W13 is [up, gate]; the prepared kernel and reference consume [gate, up].
    refs[0] = torch.cat((refs[0][:, intermediate:], refs[0][:, :intermediate]), dim=1)
    prepared = prepare_w4a16_packed_weights(
        packed[0],
        scales[0],
        globals_[0],
        packed[1],
        scales[1],
        globals_[1],
        activation="silu",
        params_dtype=torch.bfloat16,
        source_format="modelopt_e4m3_k32",
        w13_layout="w13",
        reuse_input_storage=True,
    )
    return dict(
        packed=packed, globals=globals_, prepared=prepared, refs=refs, experts=experts
    )


def problem(tokens, experts, *, skew=False, masked=False):
    gen = torch.Generator(device="cuda").manual_seed(1234 + tokens)
    x = torch.randn(tokens, 4096, device="cuda", generator=gen).bfloat16()
    logits = -torch.log(
        -torch.log(torch.rand(tokens, experts, device="cuda", generator=gen))
    )
    if skew:
        logits += (-1.1 * torch.arange(1, experts + 1, device="cuda").float().log())[
            None, :
        ]
    ids = logits.topk(8, -1).indices.int().contiguous()
    if masked:
        ids[::3, 2:] = -1
        ids[::7] = -1
    weights = torch.softmax(torch.randn(tokens, 8, device="cuda", generator=gen), -1)
    return x, ids, weights, torch.empty_like(x)


def run(weights, inputs, mode):
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch import (
        launch_sm120_moe,
    )

    x, ids, topk_weights, out = inputs
    p = weights["prepared"]
    launch_sm120_moe(
        a=x,
        w1_weight=weights["packed"][0],
        w1_weight_sf=p.w13_scale,
        w1_alpha=weights["globals"][0],
        w2_weight=weights["packed"][1],
        w2_weight_sf=p.w2_scale,
        w2_alpha=weights["globals"][1],
        topk_ids=ids,
        topk_weights=topk_weights,
        num_experts=weights["experts"],
        num_local_experts=weights["experts"],
        top_k=8,
        scatter_output=out,
        activation="silu",
        quant_mode=mode,
        source_format="modelopt_e4m3_k32",
        _prepared_weights=p,
    )
    return out


def relative_error(actual, reference):
    return (
        (actual.float() - reference.float()).norm()
        / reference.float().norm().clamp_min(1e-30)
    ).item()


def validate():
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_w4a16_fp8_act import (
        quant_rows_fp8_sigma,
        quant_rows_fp8_sigma_reference,
    )

    torch.backends.cuda.matmul.allow_tf32 = False
    weights = build_weights(8)
    report = []
    for tokens, skew, masked, dtype in (
        (513, False, False, torch.int32),
        (257, True, True, torch.int32),
        (257, True, True, torch.int64),
    ):
        inputs = problem(tokens, 8, skew=skew, masked=masked)
        inputs = (inputs[0], inputs[1].to(dtype), *inputs[2:])
        inputs[0][0].zero_()
        q, scale = quant_rows_fp8_sigma(inputs[0])
        rq, rs = quant_rows_fp8_sigma_reference(inputs[0])
        # Reciprocal lowering may place an exact midpoint on either adjacent FP8
        # code. Bound the difference to one representable step and rare ties.
        code_diff = (q.view(torch.uint8).int() - rq.view(torch.uint8).int()).abs()
        assert code_diff.max() <= 1
        assert (code_diff != 0).float().mean() < 1e-4
        torch.testing.assert_close(scale, rs, rtol=1e-6, atol=0)
        bf16_out = run(weights, inputs, "w4a16").clone()
        actual = run(weights, inputs, "w4a8_fp8_exact").clone()
        repeated = run(weights, inputs, "w4a8_fp8_exact").clone()
        assert torch.isfinite(actual).all()
        # Split-K accumulation can differ at BF16 rounding boundaries.
        repeat_error = relative_error(actual, repeated)
        assert repeat_error < 1e-4, repeat_error
        reference = moe_mirror(*inputs[:3], *weights["refs"], 1024)
        mirror_error = relative_error(actual, reference)
        bf16_error = relative_error(actual, bf16_out)
        # FP32 GEMM emulation does not reproduce every BF16 epilogue rounding.
        assert mirror_error < 0.015, mirror_error
        assert bf16_error < 0.08, bf16_error
        assert torch.count_nonzero(actual[0]) == 0
        if masked:
            assert torch.count_nonzero(actual[::7]) == 0
        report.append(
            dict(
                tokens=tokens,
                skew=skew,
                masked=masked,
                route_dtype=str(dtype),
                repeat_relative_l2=repeat_error,
                fp8_reference_relative_l2=mirror_error,
                bf16_relative_l2=bf16_error,
            )
        )
    return report


def time_ms(fn, iterations):
    for _ in range(5):
        fn()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for start, end in zip(starts, ends, strict=True):
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    return [start.elapsed_time(end) for start, end in zip(starts, ends, strict=True)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", default="1024,2048,4096,8192")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("This benchmark requires SM120")
    report = dict(
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        validation=validate(),
        shape=dict(experts=288, hidden=4096, intermediate=1024, top_k=8),
        timings=[],
    )
    print(json.dumps({"validation": report["validation"]}), flush=True)
    if not args.validate_only:
        weights = build_weights(288)
        modes = ("w4a16", "w4a8_fp8_exact")
        for tokens in [int(t) for t in args.tokens.split(",")] + [1300, 3000]:
            inputs = problem(tokens, 288, skew=tokens == 3000)
            baseline = run(weights, inputs, modes[0]).clone()
            actual = run(weights, inputs, modes[1]).clone()
            error = relative_error(actual, baseline)
            assert torch.isfinite(actual).all() and error < 0.08, error
            # Thermal preconditioning follows compilation and precedes timed rounds.
            until = time.monotonic() + 5
            while time.monotonic() < until:
                for mode in modes:
                    run(weights, inputs, mode)
            samples = {mode: [] for mode in modes}
            for round_ in range(args.rounds):
                for mode in modes[:: 1 if round_ % 2 == 0 else -1]:
                    samples[mode] += time_ms(
                        lambda: run(weights, inputs, mode), args.iterations
                    )
            row = dict(
                tokens=tokens,
                skew=tokens == 3000,
                bf16_relative_l2=error,
                modes={
                    mode: dict(
                        mean_ms=statistics.mean(xs),
                        median_ms=statistics.median(xs),
                        samples=len(xs),
                    )
                    for mode, xs in samples.items()
                },
            )
            report["timings"].append(row)
            print(json.dumps(row), flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
