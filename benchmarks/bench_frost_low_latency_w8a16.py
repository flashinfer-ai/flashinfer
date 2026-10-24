# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""E4M3-weight/BF16-activation routed MoE: Frost FMA versus native cuTile.

Synthetic model geometry, not serving throughput. Search native stage pairs,
freeze winners, validate with ABBA and independent warm CUDA-event chains.
"""

import argparse
import hashlib
import inspect
import importlib.metadata as metadata
import json
from pathlib import Path
from statistics import median
import subprocess
import torch
from flashinfer.fused_moe.backends.frost_low_latency.runner import FrostLowLatencyRunner
from flashinfer.fused_moe import (
    FrostLowLatencyConfig,
    ExecutionConfig,
    BackendOptions,
    CuTileFp8PerTensorBf16Config,
    ExpertConfig,
    MoEActivationPack,
    MoEConfig,
    MoELayer,
    MoEWeightPack,
    QuantConfig,
    QuantFormat,
    RoutingConfig,
)
from flashinfer.autotuner import autotune
from flashinfer.testing import bench_gpu_time_with_cupti


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokens", type=int, default=1)
    p.add_argument(
        "--native-full-blocks", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument("--hidden", type=int, default=2048)
    p.add_argument("--intermediate", type=int, default=512)
    p.add_argument("--experts", type=int, default=512)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--cache-mode", choices=("warm", "cold"), default="warm")
    p.add_argument("--expert-reuse", choices=("shared", "disjoint"), default="disjoint")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    import cupti.cupti  # noqa: F401

    if int(metadata.version("cupti-python").split(".")[0]) < 13:
        raise RuntimeError("Install cupti-python >= 13 for this benchmark")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    t, h, i, e, k = a.tokens, a.hidden, a.intermediate, a.experts, a.top_k
    assert h % 128 == i % 128 == 0 and t * k <= e and 1 <= t <= 8 and k <= 32
    torch.manual_seed(20261024)
    x = torch.randn(t, h, device="cuda", dtype=torch.bfloat16)

    def weights(rows, cols):
        parts = []
        for begin in range(0, e, 32):
            value = (
                (torch.randn(min(32, e - begin), rows, cols, device="cuda") / cols**0.5)
                .bfloat16()
                .float()
            )
            sf = value.abs().amax((1, 2)).clamp_min(1e-12) / 448.0
            parts.append(
                (
                    (value / sf[:, None, None])
                    .clamp(-448, 448)
                    .to(torch.float8_e4m3fn),
                    sf,
                )
            )
        return torch.cat([q.view(torch.uint8) for q, sf in parts]).view(
            torch.float8_e4m3fn
        ), torch.cat([sf for q, sf in parts])

    w1, s1 = weights(2 * i, h)
    w2, s2 = weights(h, i)
    ids = torch.arange(t * k, device="cuda", dtype=torch.int32).reshape(t, k)
    if a.expert_reuse == "shared":
        ids.remainder_(k)
    scores = torch.randn(t, k, device="cuda").softmax(-1).bfloat16().float()
    original_ids, original_scores = ids.clone(), scores.clone()
    frost_cfg = MoEConfig(
        RoutingConfig(e, k),
        QuantConfig(weight=QuantFormat.FP8PerTensor, activation=QuantFormat.BF16),
        ExpertConfig(i),
        backend=BackendOptions((FrostLowLatencyConfig(max_num_tokens=8),)),
    )
    frost_view = FrostLowLatencyConfig.prepare_fp8_weight_only_weights(w1, s1, w2, s2)
    frost_pack = MoEWeightPack()
    frost_pack.prepare_for("frost_low_latency", frost_view)
    frost_runner = FrostLowLatencyRunner(frost_cfg, x.device)
    frost_runner.check_support()
    frost_runner.build()
    frost_inputs = frost_runner.pack_inputs(
        MoEActivationPack(x, None, ids, scores), frost_pack
    )

    def frost(tactic):
        return frost_runner.forward(frost_inputs, ("ffma", *tactic))

    option = CuTileFp8PerTensorBf16Config()
    view = option.prepare_weights(
        w1, s1, w2, s2, num_local_experts=e, hidden_size=h, intermediate_size=i
    )
    cfg = MoEConfig(
        RoutingConfig(e, k),
        QuantConfig(weight=QuantFormat.FP8PerTensor, activation=QuantFormat.BF16),
        ExpertConfig(i),
        backend=BackendOptions((option,)),
        execution=ExecutionConfig(tune_max_num_tokens=t),
    )
    pack = MoEWeightPack()
    pack.prepare_for("cutile_fp8_per_tensor", view)
    act = MoEActivationPack(x, None, ids, scores)
    layer = MoELayer(cfg)
    with autotune(False):
        layer(act, pack)
    bucket, (runner, initial) = next(iter(layer._winners.items()))
    # Stage-ranking hooks mutate routing tensors. Give them private routing storage.
    tuning_inputs = runner.pack_inputs(
        MoEActivationPack(x, None, ids.clone(), scores.clone()), pack
    )
    with autotune(True):
        advertised = runner.get_valid_tactics(tuning_inputs, None)
    torch.testing.assert_close(ids, original_ids, atol=0, rtol=0)
    torch.testing.assert_close(scores, original_scores, atol=0, rtol=0)
    # Search all advertised stage pairs for the selected native block size and
    # modes, using the real workload. Also retain the initial default.
    actual_inputs = runner.pack_inputs(act, pack)
    block = runner._factorized_block_sizes(ids.numel())[0]
    stage_choices = [
        runner._stage_tactics(actual_inputs, stage=stage, block_size=block)
        for stage in (1, 2)
    ]
    full_pairs = [
        (mode, block, *one, *two)
        for mode in runner._fp8_sorted_modes(actual_inputs)
        for one in stage_choices[0]
        for two in stage_choices[1]
    ]
    if a.native_full_blocks:
        full_pairs = []
        for block_candidate in runner._block_sizes:
            choices = [
                runner._stage_tactics(
                    actual_inputs, stage=stage, block_size=block_candidate
                )
                for stage in (1, 2)
            ]
            for mode in (0, *runner._fp8_sorted_modes(actual_inputs)):
                full_pairs.extend(
                    (mode, block_candidate, *one, *two)
                    for one in choices[0]
                    for two in choices[1]
                )
    advertised = [*advertised, *full_pairs]
    native_tactics = []
    for tactic in [initial, *advertised]:
        if tactic not in native_tactics:
            native_tactics.append(tactic)

    def native(tactic):
        layer._winners[bucket] = (
            runner,
            tuple(tactic) if isinstance(tactic, (tuple, list)) else tactic,
        )
        return layer(act, pack)

    expected = torch.zeros(t, h, device="cuda")
    for token in range(t):
        for slot, expert in enumerate(ids[token].tolist()):
            value = (
                ((w1[expert].double() @ x[token].double()).float() * s1[expert])
                .bfloat16()
                .float()
            )
            up, gate = value.chunk(2)
            hidden = (up * torch.nn.functional.silu(gate)).bfloat16()
            value = (
                ((w2[expert].double() @ hidden.double()).float() * s2[expert])
                .bfloat16()
                .float()
            )
            expected[token] += value * scores[token, slot]
    expected = expected.bfloat16()

    def err(v, r):
        delta = v.double() - r.double()
        return dict(
            relative_l2=float(delta.norm() / r.double().norm().clamp_min(1e-30)),
            normalized_max=float(
                delta.abs().max() / r.double().abs().max().clamp_min(1e-30)
            ),
        )

    moe_root = Path(inspect.getfile(MoELayer)).parent
    source_files = {
        Path(__file__),
        *moe_root.glob("*.py"),
        *(moe_root / "cutile").rglob("*.py"),
        *(moe_root.parent / "cutile").rglob("*.py"),
        *Path(inspect.getfile(FrostLowLatencyRunner)).parent.rglob("*.py"),
    }
    result = dict(
        workload=vars(a) | {"output": str(a.output)},
        gpu=torch.cuda.get_device_name(),
        cc=torch.cuda.get_device_capability(),
        scope="BF16 input + routed FC1/activation + FC2/combine, identical per-expert E4M3 weights and FP32 scales; host/preparation/routing network/shared experts/communication/serving excluded",
        numerical_contract="BF16 inputs, E4M3 weights with per-expert FP32 scales, FP32 dot accumulation, BF16 FC1 preactivation and activation, BF16 expert outputs and FP32 weighted combine; same reference for both.",
        timer="CUPTI single-call first-to-last GPU activity span",
        rows=[],
        validation=[],
        rejected=[],
        completed=False,
        versions={
            z: metadata.version(z)
            for z in (
                "torch",
                "nvidia-cutlass-dsl",
                "apache-tvm-ffi",
                "cupti-python",
                "cuda-tile",
            )
        },
        source_hashes={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(source_files)
        },
        actual_route_ids=ids.cpu().tolist(),
        actual_route_scores=scores.cpu().tolist(),
        native_stage_counts=list(map(len, stage_choices)),
        native_block=block,
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

    def save():
        a.output.write_text(json.dumps(result, indent=2) + "\n")

    def check(v, name):
        errors = {
            key: err(v, ref)
            for key, ref in {"matched_bf16_boundaries": expected}.items()
        }
        own = errors["matched_bf16_boundaries"]
        assert (
            torch.isfinite(v).all()
            and own["relative_l2"] < 0.015
            and own["normalized_max"] < 0.06
        ), (name, errors)
        return errors

    def measure(name, tactic, phase="selection", group="rows"):
        call = (lambda: frost(tactic)) if name == "frost" else (lambda: native(tactic))
        output = call()
        check(output, name)
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
            tactic=list(tactic) if isinstance(tactic, (tuple, list)) else tactic,
            phase=phase,
            median_us=median(samples),
            samples_us=samples,
            errors=check(output, name),
        )
        result[group].append(row)
        save()
        print(phase, name, tactic, row["median_us"], row["errors"], flush=True)
        return row

    for first in (1, 2, 4, 8):
        for second in (1, 2, 4, 8):
            measure("frost", (first, second))
    for tactic in native_tactics:
        measure("cutile_w8a16", tactic)
    result["native_search"] = (
        "All advertised stage pairs across all supported native block sizes, gather and supported sorted modes, plus factorized shortlist and default"
        if a.native_full_blocks
        else "All advertised stage pairs for the heuristic native block and supported sorted modes, plus factorized shortlist and default; not an exhaustive arbitrary block-size search"
    )
    result["native_tactic_count"] = len(native_tactics)
    selected = {
        name: min(
            (r for r in result["rows"] if r["backend"] == name),
            key=lambda r: r["median_us"],
        )["tactic"]
        for name in ("frost", "cutile_w8a16")
    }
    result["selected"] = selected
    save()
    for rnd in range(4):
        for label, name in (
            ("A1", "cutile_w8a16"),
            ("B1", "frost"),
            ("B2", "frost"),
            ("A2", "cutile_w8a16"),
        ):
            measure(name, selected[name], f"ABBA-{rnd}-{label}", "validation")
    result["validation_median_us"] = {
        name: median(
            r["median_us"] for r in result["validation"] if r["backend"] == name
        )
        for name in selected
    }
    result["speedup"] = (
        result["validation_median_us"]["cutile_w8a16"]
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

    def event_chain(call):
        call()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(128):
                call()
        for _ in range(3):
            graph.replay()
        times = []
        for _ in range(20):
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end) * 1000 / 128)
        return dict(median_us=median(times), samples_us=times)

    result["warm_event_chain_control"] = {
        name: event_chain(
            (lambda: frost(selected[name]))
            if name == "frost"
            else (lambda: native(selected[name]))
        )
        for name in selected
    }
    torch.testing.assert_close(ids, original_ids, atol=0, rtol=0)
    torch.testing.assert_close(scores, original_scores, atol=0, rtol=0)
    result["hardware_end"] = hardware()
    result["completed"] = True
    save()


if __name__ == "__main__":
    main()
