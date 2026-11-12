# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-tensor FP8 Frost/CUTLASS GPU comparison with explicit tactic search.

Synthetic tensors at model geometry; kernel pipeline timing, not serving.
Freeze tactic winners before independent AB/BA validation. CUPTI measures GPU
activity spans; an explicit globaltimer option retains timing-kernel boundaries.
"""

import argparse
import json
from pathlib import Path
from statistics import median
import hashlib
import importlib.metadata as metadata
import subprocess


import torch
import flashinfer.fused_moe as moe
from flashinfer.fused_moe import (
    BackendOptions,
    ExpertConfig,
    FrostLowLatencyConfig,
    MoEActivationPack,
    MoEConfig,
    MoEWeightPack,
    QuantConfig,
    QuantFormat,
    RoutingConfig,
    SwiGLU,
    ReLU2,
    SiTU,
)
from flashinfer.fused_moe.backends.frost_low_latency.runner import FrostLowLatencyRunner
from tests.moe.test_frost_low_latency_bf16 import _activation
from flashinfer.fused_moe.core import cutlass_fused_moe, get_cutlass_fused_moe_module
from flashinfer.testing import bench_gpu_time_with_cupti


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--activation", choices=("swiglu", "relu2", "situ"), default="swiglu"
    )
    parser.add_argument("--route-seed", type=int, default=None)
    parser.add_argument("--tokens", type=int, default=1, choices=(1, 2, 4, 8))
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--intermediate", type=int, default=512)
    parser.add_argument("--experts", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--cache-mode", choices=("warm", "cold"), default="warm")
    parser.add_argument(
        "--expert-reuse", choices=("shared", "disjoint"), default="disjoint"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timer", choices=("cupti", "globaltimer"), default="cupti")
    parser.add_argument("--validation-rounds", type=int, default=4)
    parser.add_argument("--selection-repeats", type=int, default=20)
    parser.add_argument("--validation-repeats", type=int, default=50)
    args = parser.parse_args()
    # Fail closed: neither timing path may silently substitute CUDA events.
    if args.timer == "cupti":
        import cupti.cupti  # noqa: F401

        if int(metadata.version("cupti-python").split(".")[0]) < 13:
            raise RuntimeError("Install cupti-python >= 13 for this benchmark")
    else:
        from flashinfer.utils import get_globaltimer_kernel

        stamp = get_globaltimer_kernel()
        if stamp is None:
            raise RuntimeError("Failed to build the globaltimer timestamp kernel")
    source = Path(moe.__file__).parent
    t, h, i, e, k = (
        args.tokens,
        args.hidden,
        args.intermediate,
        args.experts,
        args.top_k,
    )
    assert 1 <= k <= 32 and k <= e and (args.expert_reuse == "shared" or t * k <= e)
    assert (
        args.validation_rounds >= 2
        and args.selection_repeats > 0
        and args.validation_repeats > 0
    )
    torch.manual_seed(20261024)
    activation = {"swiglu": SwiGLU, "relu2": ReLU2, "situ": SiTU}[args.activation]()
    x = torch.randn(t, h, device="cuda", dtype=torch.bfloat16)
    w1 = (
        torch.randn(e, (2 if activation.is_gated else 1) * i, h, device="cuda") / h**0.5
    ).bfloat16()
    w2 = (torch.randn(e, h, i, device="cuda") / i**0.5).bfloat16()
    ids = torch.arange(t * k, device="cuda", dtype=torch.int32).view(t, k)
    if args.expert_reuse == "shared":
        ids.remainder_(k)
    if args.route_seed is not None:
        # Vary expert addresses and slot order without changing reuse or the checkpoint.
        generator = torch.Generator().manual_seed(args.route_seed)
        experts = torch.randperm(e, generator=generator).to(ids.device, dtype=ids.dtype)
        slots = torch.stack(
            [torch.randperm(k, generator=generator) for _ in range(t)]
        ).to(ids.device)
        ids.copy_(experts[ids.long()].gather(1, slots))
    scores = torch.randn(t, k, device="cuda").softmax(-1).bfloat16().float()
    quant = QuantConfig(
        weight=QuantFormat.FP8PerTensor, activation=QuantFormat.FP8PerTensor
    )
    cfg = MoEConfig(
        RoutingConfig(e, k),
        quant,
        ExpertConfig(i),
        activation=activation,
        backend=BackendOptions((FrostLowLatencyConfig(max_num_tokens=8),)),
    )
    weights = MoEWeightPack()
    # Calibration is per expert. Chunk only preparation to avoid materializing
    # several full-model FP32 temporaries on larger geometries. Execution reads
    # the same concatenated E4M3 bytes/scales; preparation is outside timing.
    parts = [
        FrostLowLatencyConfig.prepare_fp8_per_tensor_weights(
            w1[begin : begin + 32],
            w2[begin : begin + 32],
            hidden_states_scale_global=32.0,
            intermediate_scale_global=128.0,
            num_local_experts=min(32, e - begin),
            hidden_size=h,
            intermediate_size=i,
            activation=activation,
        )
        for begin in range(0, e, 32)
    ]
    view = {
        key: torch.cat([part[key].view(torch.uint8) for part in parts]).view(
            value.dtype
        )
        if value.ndim
        else value
        for key, value in parts[0].items()
    }
    del parts
    weights.prepare_for("frost_low_latency", view)
    x, _ = FrostLowLatencyConfig.prepare_fp8_per_tensor_activations(
        x, hidden_states_scale_global=32.0
    )
    w1, w2 = view["fc1_expert_weights"], view["fc2_expert_weights"]
    scales = [
        view[key]
        for key in (
            "fc1_dequant_scale",
            "fc2_act_quant_scale",
            "fc2_dequant_scale",
            "fc1_act_dequant_scale",
        )
    ]
    frost = FrostLowLatencyRunner(cfg, x.device)
    frost.check_support()
    frost.build()
    inputs = frost.pack_inputs(MoEActivationPack(x, None, ids, scores), weights)
    native_out = torch.empty_like(x, dtype=torch.bfloat16)
    result = {
        "activation": repr(activation),
        "workload": vars(args) | {"output": str(args.output)},
        "gpu": torch.cuda.get_device_name(),
        "cc": torch.cuda.get_device_capability(),
        "scope": "Per-tensor FP8 routed-expert GPU pipeline with precomputed routing, including required adapters, activation and combine; excludes host enqueue, weight preparation, routing selection, shared experts, communication and serving",
        "timer": args.timer,
        "numerical_contract": "Static E4M3 input/intermediate multipliers 32/128; identical quantized weights; BF16 FC1 preactivation before the configured activation, E4M3 intermediate, BF16 expert/final outputs, BF16-rounded routing scores, ordered FP32 combine. TRTLLM has a different FC1 rounding boundary and is not measured by this harness.",
        "timing_scope": (
            "First-to-last CUPTI activity span from a single-call graph replay"
            if args.timer == "cupti"
            else "Individual-call graph interval between globaltimer stamp kernels; includes boundary overhead, not subtracted"
        ),
        "rows": [],
        "rejected": [],
        "validation": [],
        "completed": False,
        "versions": {
            name: metadata.version(name)
            for name in (
                "torch",
                "nvidia-cutlass-dsl",
                "apache-tvm-ffi",
            )
        },
        "cuda_runtime": torch.version.cuda,
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_sha256": {
            str(f.relative_to(source)): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in sorted(source.rglob("*.py"))
        },
    }

    if args.timer == "cupti":
        result["versions"]["cupti-python"] = metadata.version("cupti-python")

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
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    def reference():
        out = torch.zeros_like(x, dtype=torch.float32)
        for token in range(t):
            for slot, expert in enumerate(ids[token].tolist()):
                values = (
                    (
                        (w1[expert].double() @ x[token].double()).float()
                        * view["fc1_dequant_scale"][expert]
                    )
                    .bfloat16()
                    .float()
                )
                up, gate = (
                    values.chunk(2)
                    if activation.is_gated
                    else (values, torch.zeros_like(values))
                )
                mid = (
                    (_activation(up, gate, activation) * view["fc2_act_quant_scale"])
                    .clamp(-448, 448)
                    .to(torch.float8_e4m3fn)
                )
                row = (
                    (
                        (w2[expert].double() @ mid.double()).float()
                        * view["fc2_dequant_scale"][expert]
                    )
                    .bfloat16()
                    .float()
                )
                out[token].add_(row * scores[token, slot])
        return out.bfloat16()

    expected = reference()

    def check(output):
        delta = output.double() - expected.double()
        l2 = float(delta.norm() / expected.double().norm().clamp_min(1e-30))
        maximum = float(
            delta.abs().max() / expected.double().abs().max().clamp_min(1e-30)
        )
        assert torch.isfinite(output).all() and l2 < 0.015 and maximum < 0.06, (
            l2,
            maximum,
        )
        return {"relative_l2": l2, "normalized_max": maximum}

    flush = (
        torch.empty(512 * 1024**2, device="cuda", dtype=torch.uint8)
        if args.cache_mode == "cold"
        else None
    )

    def globaltimer_samples(call, repeats):
        call()
        timestamps = torch.empty((repeats, 2), device="cuda", dtype=torch.int64)
        starts = [timestamps[j, :1] for j in range(repeats)]
        ends = [timestamps[j, 1:] for j in range(repeats)]
        stamp(starts[0])
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for start, end in zip(starts, ends, strict=True):
                if flush is not None:
                    flush.zero_()
                stamp(start)
                call()
                stamp(end)
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        return ((timestamps[:, 1] - timestamps[:, 0]).cpu().double() / 1000).tolist()

    if args.timer == "globaltimer":
        result["empty_globaltimer_interval_us"] = globaltimer_samples(lambda: None, 50)
        save()

    def measure(name, tactic, call, *, phase="selection", group="rows"):
        output = call()
        # Numerical and timing failures are fatal; only the known native
        # shared-memory resource rejection below is excluded from selection.
        check(output)
        repeats = (
            args.selection_repeats if phase == "selection" else args.validation_repeats
        )
        if args.timer == "cupti":
            samples = [
                1000 * v
                for v in bench_gpu_time_with_cupti(
                    call,
                    dry_run_iters=8,
                    repeat_iters=repeats,
                    use_cuda_graph=True,
                    cold_l2_cache=args.cache_mode == "cold",
                )
            ]
        else:
            samples = globaltimer_samples(call, repeats)
        errors = check(output)
        assert samples and all(v > 0 for v in samples), samples
        row = dict(
            backend=name,
            tactic=list(tactic),
            phase=phase,
            median_us=median(samples),
            samples_us=samples,
            **errors,
        )
        result[group].append(row)
        save()
        print(phase, name, tactic, row["median_us"], flush=True)
        return row

    for tactic in frost.get_valid_tactics(inputs, None):
        measure("frost", tactic, lambda tactic=tactic: frost.forward(inputs, tactic))
    arch = "".join(map(str, torch.cuda.get_device_capability()))
    module = get_cutlass_fused_moe_module(arch)
    runner = module.MoERunner(
        x_dtype=torch.float8_e4m3fn,
        weight_dtype=torch.float8_e4m3fn,
        output_dtype=torch.bfloat16,
        top_k=k,
        tp_size=1,
        tp_rank=0,
        ep_size=1,
        ep_rank=0,
        cluster_size=1,
        cluster_rank=0,
        enable_alltoall=False,
        use_deepseek_fp8_block_scale=False,
        use_w4_group_scaling=False,
        use_mxfp8_act_scaling=False,
        min_latency_mode=False,
        enable_pdl=True,
        activation_type=activation.type,
        use_packed_weights=False,
        use_fused_finalize=True,
        use_wfp4afp8_humming=False,
    )
    stages = []
    for stage in (1, 2):
        runner.gemm_idx_for_tuning = stage
        stages.append(runner.get_valid_tactics([x, w1, None, w2, None], None))
    result["native_tactic_counts"] = list(map(len, stages))

    def native(tactic):
        cutlass_fused_moe(
            x,
            ids,
            scores,
            w1,
            w2,
            torch.bfloat16,
            scales,
            output=native_out,
            profile_ids=list(tactic),
            tune_max_num_tokens=t,
            enable_pdl=True,
            activation_type=activation.type,
        )
        return native_out

    measured = {}

    def pair(tactic):
        tactic = tuple(tactic)
        if tactic not in measured:
            try:
                measured[tactic] = measure("cutlass", tactic, lambda: native(tactic))
            except RuntimeError as exc:
                if "GPU lacks the shared memory resources to run " not in str(exc):
                    raise
                # This native tactic can be enumerated on SM120 despite needing
                # more shared memory than the architecture allows. Log it; do
                # not suppress numerical, asynchronous, or other launch errors.
                torch.cuda.synchronize()
                measured[tactic] = None
                result["rejected"].append(
                    {
                        "backend": "cutlass",
                        "tactic": list(tactic),
                        "error": str(exc).splitlines()[0],
                    }
                )
                save()
        return measured[tactic]

    pair((-1, -1))
    if len(stages[0]) * len(stages[1]) <= 1024:
        result["native_search"] = (
            "exhaustive advertised GEMM1 x GEMM2 Cartesian product plus default"
        )
        for a in stages[0]:
            for b in stages[1]:
                pair((a, b))
    else:
        result["native_search"] = (
            "coordinate search plus top-two pair combinations and default; not exhaustive"
        )
        first = sorted(
            filter(None, (pair((a, -1)) for a in stages[0])),
            key=lambda row: row["median_us"],
        )[:2]
        assert first
        best1 = first[0]["tactic"][0]
        second = sorted(
            filter(None, (pair((best1, b)) for b in stages[1])),
            key=lambda row: row["median_us"],
        )[:2]
        assert second
        for a in first:
            for b in second:
                pair((a["tactic"][0], b["tactic"][1]))
    result["selected"] = {
        name: min(
            (row for row in result["rows"] if row["backend"] == name),
            key=lambda row: row["median_us"],
        )["tactic"]
        for name in ("frost", "cutlass")
    }
    save()  # Winners are frozen before independent validation.
    selected = result["selected"]

    def replay(name):
        tactic = tuple(selected[name])
        return (
            (lambda: frost.forward(inputs, tactic))
            if name == "frost"
            else (lambda: native(tactic))
        )

    for iteration in range(args.validation_rounds):
        for label, name in (
            ("A1", "cutlass"),
            ("B1", "frost"),
            ("B2", "frost"),
            ("A2", "cutlass"),
        ):
            measure(
                name,
                selected[name],
                replay(name),
                phase=f"ABBA-{iteration}-{label}",
                group="validation",
            )
    result["validation_median_us"] = {
        name: median(
            row["median_us"] for row in result["validation"] if row["backend"] == name
        )
        for name in selected
    }
    result["speedup"] = (
        result["validation_median_us"]["cutlass"]
        / result["validation_median_us"]["frost"]
    )
    result["aa_ratio"] = {
        name: median(
            row["median_us"]
            for j, row in enumerate(result["validation"])
            if row["backend"] == name and j % 4 in (0, 1)
        )
        / median(
            row["median_us"]
            for j, row in enumerate(result["validation"])
            if row["backend"] == name and j % 4 in (2, 3)
        )
        for name in selected
    }
    # Independently cross-check warm-cache ranking with one long event interval.
    # This is amortized graph-chain time, NOT the single-call span above.
    if args.cache_mode == "warm":
        result["warm_event_chain_us"] = {}
        for name in selected:
            graph = torch.cuda.CUDAGraph()
            call = replay(name)
            with torch.cuda.graph(graph):
                for _ in range(128):
                    output = call()
            graph.replay()
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            samples = []
            for _ in range(7):
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end) * 1000 / 128)
            check(output)
            result["warm_event_chain_us"][name] = {
                "samples": samples,
                "median": median(samples),
            }
    result["hardware_end"] = hardware()
    result["completed"] = True
    save()


if __name__ == "__main__":
    main()
