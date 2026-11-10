# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Full block-FP8 GPU interval: advertised Frost candidates and tuned native controls."""

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
    ReLU2,
    SiLU,
    FrostLowLatencyConfig,
)
from flashinfer.fused_moe.runners import SM12xFp8Runner
from flashinfer.fused_moe.backends.frost_low_latency.runner import FrostLowLatencyRunner
from flashinfer.fused_moe.prepare import _deepseek_fp8_quantize_weights
from flashinfer.testing import bench_gpu_time_with_cupti


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--activation", choices=("swiglu", "relu2", "silu"), default="swiglu"
    )
    p.add_argument("--route-seed", type=int, default=None)
    p.add_argument("--tokens", type=int, default=1)
    p.add_argument("--hidden", type=int, default=2048)
    p.add_argument("--intermediate", type=int, default=512)
    p.add_argument("--experts", type=int, default=512)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--cache-mode", choices=("warm", "cold"), default="warm")
    p.add_argument(
        "--expert-reuse", choices=("shared", "paired", "disjoint"), default="disjoint"
    )
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    activation = {"swiglu": SwiGLU, "relu2": ReLU2, "silu": SiLU}[a.activation]()
    from tests.moe.test_frost_low_latency_bf16 import _activation

    arch = torch.cuda.get_device_capability()
    native_available = arch == (9, 0) or a.activation == "swiglu"
    if arch not in ((9, 0), (12, 0)):
        raise RuntimeError("Requires SM90 or SM120")
    from cupti import cupti

    assert int(metadata.version("cupti-python").split(".")[0]) >= 13 and callable(
        cupti.activity_enable
    )
    kind = "wgmma" if arch == (9, 0) else "warp_mma"
    native_name = "cutlass_fp8_block" if arch == (9, 0) else "sm12x_fp8"
    source_root = Path(__file__).resolve().parents[1]
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

    w1, s1 = weights((2 if activation.is_gated else 1) * i, h)
    w2, s2 = weights(h, i)
    ids = torch.arange(t * k, device="cuda", dtype=torch.int32).reshape(t, k)
    if a.expert_reuse == "shared":
        ids.remainder_(k)
    elif a.expert_reuse == "paired":
        ids.copy_(
            torch.arange(t, device=x.device, dtype=torch.int32)[:, None] // 2 * k
            + torch.arange(k, device=x.device, dtype=torch.int32)[None, :]
        )
    if a.route_seed is not None:
        generator = torch.Generator().manual_seed(a.route_seed)
        expert_perm = torch.randperm(e, generator=generator).to(
            x.device, dtype=torch.int32
        )
        slot_perm = torch.stack(
            [torch.randperm(k, generator=generator) for _ in range(t)]
        ).to(x.device)
        ids.copy_(expert_perm[ids.long()].gather(1, slot_perm))
    scores = torch.randn(t, k, device="cuda").softmax(-1).bfloat16().float()
    cfg = MoEConfig(
        routing=RoutingConfig(e, k),
        quant=QuantConfig(
            weight=QuantFormat.DeepSeekFp8,
            activation=QuantFormat.DeepSeekFp8,
            per_token_scale=False,
        ),
        experts=ExpertConfig(i, local_num_experts=e),
        activation=activation,
        backend=BackendOptions((SM12xFp8Config(),)),
        execution=ExecutionConfig(enable_pdl=False),
    )
    fcfg = replace(
        cfg, backend=BackendOptions((FrostLowLatencyConfig(max_num_tokens=8),))
    )
    fpack = MoEWeightPack()
    fpack.prepare_for(
        "frost_low_latency",
        FrostLowLatencyConfig.prepare_deepseek_fp8_weights(
            w1, s1, w2, s2, activation=activation
        ),
    )
    frost_runner = FrostLowLatencyRunner(fcfg, x.device)
    frost_runner.check_support()
    frost_runner.build()
    frost_inputs = frost_runner.pack_inputs(
        MoEActivationPack(x, None, ids, scores), fpack
    )

    def frost(tactic):
        return frost_runner.forward(frost_inputs, ("ffma", *tactic))

    def grouped_execute(inputs, config, tactic):
        return frost_runner.forward(inputs, (kind, *tactic))

    grouped_choices = [
        v[1:]
        for v in frost_runner.get_valid_tactics(frost_inputs, None)
        if v[0] == kind
    ]
    ids_saved, scores_saved = ids.clone(), scores.clone()
    if arch == (9, 0):
        from flashinfer.fused_moe import CutlassFp8BlockConfig
        from flashinfer.fused_moe.runners import CutlassFp8BlockRunner
        from flashinfer.autotuner import autotune

        ncfg = replace(cfg, backend=BackendOptions((CutlassFp8BlockConfig(),)))
        pack = MoEWeightPack()
        pack.prepare_for(
            "cutlass_fp8_block",
            dict(
                fc1_expert_weights=w1,
                fc2_expert_weights=w2,
                fc1_block_scale=s1,
                fc2_block_scale=s2,
            ),
        )
        native = CutlassFp8BlockRunner(ncfg, x.device)
        native.check_support()
        native.build()
        inputs = native.pack_inputs(MoEActivationPack(x, None, ids, scores), pack)
        tuning = native.pack_inputs(
            MoEActivationPack(x.clone(), None, ids.clone(), scores.clone()), pack
        )
        with autotune(True):
            native_choices = native.get_valid_tactics(tuning, None)
        native_choices = [(-1, -1), *native_choices]
        sfirst = ssecond = []

        def call_native(tactic):
            return native.forward(inputs, tuple(tactic))
    elif native_available:
        pack = MoEWeightPack()
        pack.prepare_for(
            "sm12x_fp8",
            SM12xFp8Config.prepare_weights(
                w1,
                s1.transpose(-1, -2).contiguous(),
                w2,
                s2.transpose(-1, -2).contiguous(),
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

    if not native_available:
        sfirst = ssecond = []
        native_choices = []

        def call_native(tactic):
            raise AssertionError("Unsupported native activation cannot be timed")

    # Independent FP64 dequant references retain each backend's documented
    # intermediate rounding boundaries, rather than relaxing error tolerances.
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

    def native_hopper_quant(v):
        groups = v.float().reshape(v.shape[0], -1, 128)
        amax = groups.abs().amax(-1)
        multiplier = torch.where(amax != 0, 448.0 / amax, torch.ones_like(amax))
        q = (
            (groups * multiplier[..., None])
            .clamp(-448, 448)
            .to(torch.float8_e4m3fn)
            .reshape_as(v)
        )
        return q, multiplier.reciprocal()

    if arch == (9, 0):
        qnx, snx = native_hopper_quant(x)
        native_dx = qnx.double() * snx.double().repeat_interleave(128, 1)
    ref_qx, ref_xs = quant(x)
    dx = ref_qx.double() * ref_xs.double().repeat_interleave(128, 1)
    expected_f = torch.zeros(t, h, device="cuda")
    expected_n = torch.zeros_like(expected_f)
    for token in range(t):
        for slot, expert in enumerate(ids[token].tolist()):
            dw1 = w1[expert].double() * s1[expert].double().repeat_interleave(
                128, 0
            ).repeat_interleave(128, 1)
            dw2 = w2[expert].double() * s2[expert].double().repeat_interleave(
                128, 0
            ).repeat_interleave(128, 1)
            pre = (dw1 @ dx[token]).float()
            u, g = pre.chunk(2) if activation.is_gated else (pre, torch.zeros_like(pre))
            q, sf = quant(_activation(u, g, activation)[None])
            dmid = q.double() * sf.double().repeat_interleave(128, 1)
            value = (dw2 @ dmid[0]).float()
            expected_f[token] += value.bfloat16().float() * scores[token, slot]
            if arch == (9, 0):
                # Source: CUTLASS BlockScaleFC1/FC2 instantiate BF16 GEMM output;
                # doActivation<T> writes BF16 before the second dynamic quantizer.
                npre = (dw1 @ native_dx[token]).bfloat16().float()
                nu, ng = (
                    npre.chunk(2)
                    if activation.is_gated
                    else (npre, torch.zeros_like(npre))
                )
                nmid = _activation(nu, ng, activation).bfloat16()
                nq, ns = native_hopper_quant(nmid[None])
                ndmid = nq.double() * ns.double().repeat_interleave(128, 1)
                native_value = (dw2 @ ndmid[0]).bfloat16().float()
                expected_n[token] += native_value * scores[token, slot]
            else:
                # SM12x uses BF16 epilogues in both GEMMs: activation is
                # narrowed before Q1, and FC2 narrows before route weighting.
                # Each weighted contribution is then narrowed for BF16 atomics.
                nmid = _activation(u, g, activation).bfloat16()
                nq, ns = quant(nmid[None])
                ndmid = nq.double() * ns.double().repeat_interleave(128, 1)
                native_value = (dw2 @ ndmid[0]).bfloat16().float()
                contribution = (native_value * scores[token, slot]).bfloat16()
                # Use ordered FP32 accumulation for the reference; the fixed
                # tolerance covers the kernel's nondeterministic BF16 atomics.
                expected_n[token] += contribution.float()
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
        native_available=native_available,
        activation=repr(activation),
        workload=vars(a) | {"output": str(a.output)},
        gpu=torch.cuda.get_device_name(),
        cc=torch.cuda.get_device_capability(),
        scope="BF16 input + input block quantization + routed FC1/activation/block quantization + FC2/weighted combine; precomputed routing IDs; no host enqueue, shared experts, communication or serving",
        numerical_contract="Same canonical E4M3 weights and arbitrary FP32 128x128 scales; activation FP32 128-value max(amax,1e-4)/448 scales, FP32 preactivation. Frost FP32 preactivation and BF16 expert then ordered combine; architecture-native typed boundaries are described separately. Routing scores rounded BF16 for both.",
        native_numerical_contract=(
            "CUTLASS BF16 GEMM1 output, BF16 activation output before dynamic quant, BF16 GEMM2 partial then FP32 ordered combine; quant multiplier 448/amax and reciprocal scale, zero multiplier=1"
            if arch == (9, 0)
            else "SM120 FP32 preactivation, BF16 activation before Q1 quantization, BF16 FC2 output before route weighting, BF16 weighted contribution and atomic combine; reference sums contributions in FP32 before final BF16"
        ),
        reference_boundary_delta=err(expected_f, expected_n)
        if native_available
        else None,
        timer="CUPTI single-call first-to-last GPU activity span",
        rows=[],
        validation=[],
        native_tactic_counts=[len(sfirst), len(ssecond)]
        if arch == (12, 0)
        else [len(native_choices)],
        native_source_hashes={
            f: hashlib.sha256((source_root / f).read_bytes()).hexdigest()
            for f in (
                "csrc/fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh",
                "csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/include/moe_kernels.h",
                "csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_gemm_kernel.cuh",
            )
        },
        grouped_candidates=grouped_choices,
        native_tactics=[[str(z) for z in sfirst], [str(z) for z in ssecond]],
        completed=False,
        versions={
            z: metadata.version(z)
            for z in ("torch", "nvidia-cutlass-dsl", "apache-tvm-ffi", "cupti-python")
        },
        source_hashes={
            str(p.relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                *(
                    source_root / "flashinfer/fused_moe/backends/frost_low_latency"
                ).rglob("*.py"),
                *(source_root / "flashinfer/fused_moe/cute_dsl/blackwell_sm12x").rglob(
                    "*.py"
                ),
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
        errors = err(v, expected_f if name in ("frost", "grouped_tc") else expected_n)
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
            else (
                (lambda: grouped_execute(frost_inputs, fcfg, tuple(tactic)))
                if name == "grouped_tc"
                else (lambda: call_native(tactic))
            )
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
    for tactic in grouped_choices:
        measure("grouped_tc", tactic)
    if arch == (12, 0) and native_available:
        native_choices = [
            (one, two, pdl)
            for pdl in (0, 1)
            for one in (-1, *range(len(sfirst)))
            for two in (-1, *range(len(ssecond)))
        ]
    for tactic in native_choices:
        measure(native_name, tactic)
    result["native_search"] = (
        "Exhaustive advertised SM120 stage1 x stage2 plus defaults, with PDL off/on"
        if arch == (12, 0)
        else "CUTLASS per-stage top two Cartesian product plus heuristic; not exhaustive"
    )
    families = (
        ["frost"]
        + (["grouped_tc"] if grouped_choices else [])
        + ([native_name] if native_available else [])
    )
    selected = {
        name: min(
            (r for r in result["rows"] if r["backend"] == name),
            key=lambda r: r["median_us"],
        )["tactic"]
        for name in families
    }
    if not native_available:
        result["native_search"] = (
            "Unsupported activation on native SM12x FP8; FMA comparison only"
        )
    result["selected"] = selected
    save()
    result["paired"] = {}
    target = "grouped_tc" if grouped_choices else "frost"
    baselines = (["frost"] if grouped_choices else []) + (
        [native_name] if native_available else []
    )
    for baseline in baselines:
        for rnd in range(2):
            for label, name in (
                ("A1", baseline),
                ("B1", target),
                ("B2", target),
                ("A2", baseline),
            ):
                measure(
                    name, selected[name], f"{baseline}-ABBA-{rnd}-{label}", "validation"
                )
        cohort = [
            r
            for r in result["validation"]
            if r["phase"].startswith(baseline + "-ABBA-")
        ]
        times = {
            name: median(r["median_us"] for r in cohort if r["backend"] == name)
            for name in (baseline, target)
        }
        result["paired"][baseline] = {
            "median_us": times,
            "target": target,
            "speedup": times[baseline] / times[target],
        }
        save()
    result["hardware_end"] = hardware()
    result["completed"] = True
    save()


if __name__ == "__main__":
    main()
