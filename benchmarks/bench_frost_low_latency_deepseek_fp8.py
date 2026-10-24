# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integrated block-FP8 BF16 entry versus tuned native SM120 pipeline."""

from dataclasses import replace
import argparse
import functools
import hashlib
import importlib
import importlib.metadata as metadata
import json
from pathlib import Path
from statistics import median
import subprocess
import torch
from flashinfer.fused_moe import (
    BackendOptions,
    ExecutionConfig,
    ExpertConfig,
    MoEActivationPack,
    MoEConfig,
    MoEWeightPack,
    QuantConfig,
    QuantFormat,
    RoutingConfig,
    SM12xFp8Config,
    SwiGLU,
    FrostLowLatencyConfig,
)
from flashinfer.fused_moe.runners import SM12xFp8Runner
from flashinfer.fused_moe.backends.frost_low_latency.runner import FrostLowLatencyRunner
from flashinfer.fused_moe.prepare import _deepseek_fp8_quantize_weights
from flashinfer.testing import bench_gpu_time_with_cupti


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, default=1)
    p.add_argument("--hidden", type=int, default=2048)
    p.add_argument("--intermediate", type=int, default=512)
    p.add_argument("--experts", type=int, default=512)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--cache-mode", choices=("warm", "cold"), default="warm")
    p.add_argument("--expert-reuse", choices=("shared", "disjoint"), default="disjoint")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("This native comparison requires SM120")
    t, h, i, e, k = a.tokens, a.hidden, a.intermediate, a.experts, a.top_k
    if not (
        h > 0
        and i > 0
        and h % 128 == i % 128 == 0
        and t * k <= e
        and 1 <= k <= 32
        and 1 <= t <= 8
    ):
        p.error(
            "require positive H/I divisible by 128, T in 1..8, K in 1..32 and E >= T*K"
        )
    torch.manual_seed(20261024)
    x = torch.randn(t, h, device="cuda", dtype=torch.bfloat16)

    def weights(rows, cols):
        parts = [
            _deepseek_fp8_quantize_weights(
                (
                    torch.randn(min(32, e - b), rows, cols, device="cuda") / cols**0.5
                ).bfloat16()
            )
            for b in range(0, e, 32)
        ]
        return (
            torch.cat([z[0].view(torch.uint8) for z in parts]).view(
                torch.float8_e4m3fn
            ),
            torch.cat([z[1] for z in parts]),
        )

    w1, s1 = weights(2 * i, h)
    w2, s2 = weights(h, i)
    ids = torch.arange(t * k, device="cuda", dtype=torch.int32).reshape(t, k)
    if a.expert_reuse == "shared":
        ids.remainder_(k)
    scores = torch.randn(t, k, device="cuda").softmax(-1).bfloat16().float()
    cfg = MoEConfig(
        routing=RoutingConfig(e, k),
        quant=QuantConfig(
            weight=QuantFormat.DeepSeekFp8,
            activation=QuantFormat.DeepSeekFp8,
            per_token_scale=False,
        ),
        experts=ExpertConfig(i, local_num_experts=e),
        activation=SwiGLU(),
        backend=BackendOptions((SM12xFp8Config(),)),
        execution=ExecutionConfig(enable_pdl=False),
    )
    fcfg = replace(
        cfg, backend=BackendOptions((FrostLowLatencyConfig(max_num_tokens=8),))
    )
    fpack = MoEWeightPack()
    fpack.prepare_for(
        "frost_low_latency",
        FrostLowLatencyConfig.prepare_deepseek_fp8_weights(w1, s1, w2, s2),
    )
    frost_runner = FrostLowLatencyRunner(fcfg, x.device)
    frost_runner.check_support()
    frost_runner.build()
    frost_inputs = frost_runner.pack_inputs(
        MoEActivationPack(x, None, ids, scores), fpack
    )

    def frost(tactic):
        return frost_runner.forward(frost_inputs, ("ffma", *tactic))

    ids_saved, scores_saved = ids.clone(), scores.clone()
    pack = MoEWeightPack()
    pack.prepare_for(
        "sm12x_fp8",
        SM12xFp8Config.prepare_weights(
            w1, s1.transpose(-1, -2).contiguous(), w2, s2.transpose(-1, -2).contiguous()
        ),
    )
    native = SM12xFp8Runner(cfg, x.device)
    native.check_support()
    native.build()
    inputs = native.pack_inputs(MoEActivationPack(x, None, ids, scores), pack)
    f1, f2 = native._fc1, native._fc2
    m1 = importlib.import_module(
        "flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_fp8_fc1_act_q1"
    )
    m2 = importlib.import_module(
        "flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_fp8_fc2_finalize"
    )

    def valid(mod, cls, n, kk, extra):
        found = []
        for tactic in mod.TACTICS:
            tile, epi = mod.split_tactic(tactic)
            if cls.can_implement(n=n, k=kk, tile=tile, epi=epi, **extra):
                found.append(tactic)
        return found

    sfirst = valid(
        m1,
        m1.CuteDslSm120GroupedFp8Fc1ActQ1Op,
        i,
        h,
        {"out_dtype": torch.float8_e4m3fn},
    )
    ssecond = valid(m2, m2.CuteDslSm120GroupedFp8Fc2FinalizeOp, h, i, {})

    def call_native(tactic):
        one, two, pdl = tactic
        native.config = replace(
            native.config,
            execution=replace(native.config.execution, enable_pdl=bool(pdl)),
        )
        native._fc1 = (
            f1
            if one == -1
            else functools.partial(
                f1,
                tile=m1.split_tactic(sfirst[one])[0],
                epi=m1.split_tactic(sfirst[one])[1],
            )
        )
        native._fc2 = (
            f2
            if two == -1
            else functools.partial(
                f2,
                tile=m2.split_tactic(ssecond[two])[0],
                epi=m2.split_tactic(ssecond[two])[1],
            )
        )
        return native.forward(inputs, -1)

    # FP64 block-dequant reference. Native uses FP32 expert values before weighted
    # BF16 atomic combine; Frost rounds each expert to BF16 then combines in order.
    def quant(v):
        groups = v.float().reshape(v.shape[0], -1, 128)
        sf = groups.abs().amax(-1).clamp_min(1e-4) * (1.0 / 448.0)
        q = (
            (groups / sf[..., None])
            .clamp(-448, 448)
            .to(torch.float8_e4m3fn)
            .reshape_as(v)
        )
        return q, sf

    ref_qx, ref_xs = quant(x)
    dx = ref_qx.double() * ref_xs.double().repeat_interleave(128, 1)
    expected_f = torch.zeros(t, h, device="cuda")
    expected_n = torch.zeros_like(expected_f)
    ref_mids = []
    for token in range(t):
        for slot, expert in enumerate(ids[token].tolist()):
            dw1 = w1[expert].double() * s1[expert].double().repeat_interleave(
                128, 0
            ).repeat_interleave(128, 1)
            dw2 = w2[expert].double() * s2[expert].double().repeat_interleave(
                128, 0
            ).repeat_interleave(128, 1)
            u, g = (dw1 @ dx[token]).float().chunk(2)
            q, sf = quant((u * torch.nn.functional.silu(g))[None])
            ref_mids.append(q)
            dmid = q.double() * sf.double().repeat_interleave(128, 1)
            value = (dw2 @ dmid[0]).float()
            expected_f[token] += value.bfloat16().float() * scores[token, slot]
            expected_n[token] += value * scores[token, slot]
    expected_f = expected_f.bfloat16()
    expected_n = expected_n.bfloat16()

    def err(v, r):
        delta = v.double() - r.double()
        return dict(
            relative_l2=float(delta.norm() / r.double().norm().clamp_min(1e-30)),
            normalized_max=float(
                delta.abs().max() / r.double().abs().max().clamp_min(1e-30)
            ),
        )

    result = dict(
        workload=vars(a) | {"output": str(a.output)},
        gpu=torch.cuda.get_device_name(),
        cc=torch.cuda.get_device_capability(),
        scope="BF16 input + input block quantization + routed FC1/activation/block quantization + FC2/weighted combine; precomputed routing IDs; no host enqueue, shared experts, communication or serving",
        numerical_contract="Same canonical E4M3 weights and arbitrary FP32 128x128 scales; activation FP32 128-value max(amax,1e-4)/448 scales, FP32 preactivation. Frost BF16 expert then ordered combine; native FP32 expert then BF16 atomic combine. Routing scores rounded BF16 for both.",
        reference_boundary_delta=err(expected_f, expected_n),
        timer="CUPTI single-call first-to-last GPU activity span",
        rows=[],
        validation=[],
        native_tactic_counts=[len(sfirst), len(ssecond)],
        native_tactics=[[str(z) for z in sfirst], [str(z) for z in ssecond]],
        completed=False,
        versions={
            z: metadata.version(z)
            for z in ("torch", "nvidia-cutlass-dsl", "apache-tvm-ffi", "cupti-python")
        },
        source_hashes={
            str(p.relative_to(Path(__file__).resolve().parents[1])): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in [
                *(
                    Path(__file__).resolve().parents[1]
                    / "flashinfer/fused_moe/backends/frost_low_latency"
                ).rglob("*.py"),
                *(
                    Path(__file__).resolve().parents[1]
                    / "flashinfer/fused_moe/cute_dsl/blackwell_sm12x"
                ).rglob("*.py"),
            ]
        },
        benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )

    def hardware():
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,clocks.sm,clocks.mem,power.draw,temperature.gpu",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()

    result["hardware_start"] = hardware()
    result["actual_route_ids"] = ids.tolist()
    result["actual_route_scores"] = scores.tolist()

    def save():
        a.output.write_text(json.dumps(result, indent=2) + "\n")

    def check(v, name):
        torch.testing.assert_close(ids, ids_saved, atol=0, rtol=0)
        torch.testing.assert_close(scores, scores_saved, atol=0, rtol=0)
        errors = err(v, expected_f if name == "frost" else expected_n)
        assert (
            torch.isfinite(v).all()
            and errors["relative_l2"] < 0.015
            and errors["normalized_max"] < 0.06
        ), (name, errors)
        return errors

    def measure(name, tactic, phase="selection", group="rows"):
        call = (
            (lambda: frost(tactic))
            if name == "frost"
            else (lambda: call_native(tactic))
        )
        value = call()
        check(value, name)
        samples = [
            1000 * v
            for v in bench_gpu_time_with_cupti(
                call,
                dry_run_iters=8,
                repeat_iters=20 if phase == "selection" else 50,
                use_cuda_graph=True,
                cold_l2_cache=a.cache_mode == "cold",
            )
        ]
        row = dict(
            backend=name,
            tactic=list(tactic),
            phase=phase,
            median_us=median(samples),
            samples_us=samples,
            **check(value, name),
        )
        result[group].append(row)
        save()
        print(phase, name, tactic, row["median_us"], row["relative_l2"], flush=True)
        return row

    for first in (1, 2, 4, 8):
        for second in (1, 2, 4, 8):
            measure("frost", (first, second))
    measured = {}

    def pair(tactic):
        if tactic not in measured:
            measured[tactic] = measure("sm12x_fp8", tactic)
        return measured[tactic]

    for pdl in (0, 1):
        for one in (-1, *range(len(sfirst))):
            for two in (-1, *range(len(ssecond))):
                pair((one, two, pdl))
    result["native_search"] = (
        "Exhaustive advertised stage1 x stage2 plus default per-stage choices, each with PDL disabled and enabled"
    )
    selected = {
        name: min(
            (r for r in result["rows"] if r["backend"] == name),
            key=lambda r: r["median_us"],
        )["tactic"]
        for name in ("frost", "sm12x_fp8")
    }
    result["selected"] = selected
    save()
    for rnd in range(4):
        for label, name in (
            ("A1", "sm12x_fp8"),
            ("B1", "frost"),
            ("B2", "frost"),
            ("A2", "sm12x_fp8"),
        ):
            measure(name, selected[name], f"ABBA-{rnd}-{label}", "validation")
    result["validation_median_us"] = {
        name: median(
            r["median_us"] for r in result["validation"] if r["backend"] == name
        )
        for name in selected
    }
    result["speedup"] = (
        result["validation_median_us"]["sm12x_fp8"]
        / result["validation_median_us"]["frost"]
    )
    result["aa_ratio"] = {
        name: median(
            r["median_us"]
            for j, r in enumerate(result["validation"])
            if r["backend"] == name and j % 4 in (0, 1)
        )
        / median(
            r["median_us"]
            for j, r in enumerate(result["validation"])
            if r["backend"] == name and j % 4 in (2, 3)
        )
        for name in selected
    }
    result["hardware_end"] = hardware()
    result["completed"] = True
    save()


if __name__ == "__main__":
    main()
