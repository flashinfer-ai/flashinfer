# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Same-checkpoint full-call grouped Tensor Core comparison with independent reference checks.

Run from a FlashInfer source checkout with CuTe DSL >=4.8 and CUPTI installed:
  PYTHONPATH=. python benchmarks/bench_frost_low_latency_grouped_tc.py \
    --tokens 4 --route-seed 7191 --output grouped-t4.json

Timing excludes preparation, host enqueue, router-network evaluation, shared
experts and communication. Each baseline uses its runner's compound candidate
search plus its heuristic, not an exhaustive compound search. Repeat with
--baseline cutile_bf16 and --baseline cudnn_grouped_gemm_bf16 when available.
"""

import argparse
from dataclasses import replace
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
from statistics import median

import torch

from flashinfer.autotuner import autotune
from flashinfer.fused_moe import (
    BackendOptions,
    CudnnGroupedGemmBf16Config,
    CutlassBf16Config,
    CuTileBf16Config,
    MoEActivationPack,
    MoELayer,
    MoEWeightPack,
    SwiGLU,
)
from flashinfer.fused_moe.backends.frost_low_latency.runner import FrostLowLatencyRunner
from flashinfer.testing import bench_gpu_time_with_cupti
from tests.moe.test_frost_low_latency_bf16 import _inputs, _reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, choices=(4, 8), default=4)
    parser.add_argument("--route-seed", type=int, default=7191)
    parser.add_argument(
        "--baseline",
        choices=("cutlass_bf16", "cutile_bf16", "cudnn_grouped_gemm_bf16"),
        default="cutlass_bf16",
    )
    parser.add_argument(
        "--reuses",
        nargs="+",
        choices=("shared", "paired", "quartet", "disjoint"),
        default=("shared", "paired", "quartet", "disjoint"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if torch.cuda.get_device_capability() not in ((9, 0), (12, 0)):
        raise RuntimeError("This benchmark qualifies Hopper SM90 and SM120")
    t, h, i, e, k = args.tokens, 2048, 512, 512, 10
    print("Preparing checkpoint and Frost buffers", flush=True)
    cfg, x, w1, w2, ids, scores, weights = _inputs(
        SwiGLU(), tokens=t, hidden=h, intermediate=i, experts=e, topk=k
    )
    x.mul_(0.1)
    # The correctness fixture includes repeated experts. All baseline warmup
    # calls must respect the distinct-per-token grouped-GEMM contract too.
    ids.copy_(torch.arange(t * k, device=x.device).reshape(t, k))
    scores.copy_(scores.softmax(-1).bfloat16().float())
    frost = FrostLowLatencyRunner(cfg, x.device)
    frost.check_support()
    frost.build()
    act = MoEActivationPack(x, None, ids, scores)
    buffers = frost.pack_inputs(act, weights)
    tactics = frost.get_valid_tactics(buffers, None)
    families = {
        "fma": [v for v in tactics if v[0] == "ffma"],
        "grouped_tc": [v for v in tactics if v[0] in ("wgmma", "warp_mma")],
    }
    assert len(families["fma"]) == 16 and len(families["grouped_tc"]) in (2, 3, 4)
    native_config = {
        "cutlass_bf16": CutlassBf16Config,
        "cutile_bf16": CuTileBf16Config,
        "cudnn_grouped_gemm_bf16": CudnnGroupedGemmBf16Config,
    }[args.baseline]()
    native_weights = MoEWeightPack()
    native_weights.prepare_for(
        args.baseline,
        native_config.prepare_weights(
            w1,
            w2,
            num_local_experts=e,
            hidden_size=h,
            intermediate_size=i,
            activation=SwiGLU(),
        ),
    )
    native_layer = MoELayer(replace(cfg, backend=BackendOptions((native_config,))))
    print("Preparing", args.baseline, "(first use can compile)", flush=True)
    with autotune(False):
        native_layer(act, native_weights)
    print(args.baseline, "ready", flush=True)
    bucket, (native_runner, native_initial) = next(iter(native_layer._winners.items()))
    assert native_runner.backend_key == args.baseline
    root = Path(__file__).resolve().parents[1]
    sources = [
        Path(__file__).resolve(),
        root / "flashinfer/fused_moe/backends/frost_low_latency/runner.py",
        root / "flashinfer/fused_moe/backends/frost_low_latency/runtime.py",
        *sorted(
            (root / "flashinfer/fused_moe/backends/frost_low_latency/tc").glob("*.py")
        ),
    ]
    result = {
        "shape": {"T": t, "H": h, "I": i, "E": e, "top_k": k},
        "route_seed": args.route_seed,
        "dtype": "bf16",
        "activation": "default SwiGLU",
        "gpu": torch.cuda.get_device_name(),
        "software": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cutlass_dsl": version("nvidia-cutlass-dsl"),
            "tvm_ffi": version("apache-tvm-ffi"),
        },
        "source_hashes": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
        "timing": "CUPTI CUDA Graph complete routed-expert interval",
        "native_backend": args.baseline,
        "native_search": "Runner compound candidates plus default, not exhaustive",
        "selection": [],
        "validation": [],
        "completed": False,
    }

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    def check(output, reference):
        delta = output.double() - reference.double()
        errors = {
            "relative_l2": float(
                delta.norm() / reference.double().norm().clamp_min(1e-20)
            ),
            "normalized_max": float(
                delta.abs().max() / reference.double().abs().max().clamp_min(1e-20)
            ),
        }
        if (
            not torch.isfinite(output).all()
            or errors["relative_l2"] >= 0.01
            or errors["normalized_max"] >= 0.03
        ):
            raise AssertionError(errors)
        return errors

    for reuse in args.reuses:
        ids.copy_(
            torch.arange(t * k, device="cuda", dtype=torch.int32).reshape(t, k) % e
        )
        if reuse == "shared":
            ids.remainder_(k)
        elif reuse in ("paired", "quartet"):
            share = 2 if reuse == "paired" else 4
            ids.copy_(
                (
                    (torch.arange(t, device="cuda", dtype=torch.int32) // share)[
                        :, None
                    ]
                    * k
                    + torch.arange(k, device="cuda", dtype=torch.int32)[None, :]
                )
                % e
            )
        generator = torch.Generator().manual_seed(args.route_seed)
        expert_perm = torch.randperm(e, generator=generator).to(
            x.device, dtype=torch.int32
        )
        slot_perm = torch.stack(
            [torch.randperm(k, generator=generator) for _ in range(t)]
        ).to(x.device)
        ids.copy_(expert_perm[ids.long()].gather(1, slot_perm))
        scores.copy_(torch.randn_like(scores).softmax(-1).bfloat16().float())
        _, reference, _ = _reference(x, w1, w2, ids, scores, SwiGLU(), True)
        for cold in (False, True):

            def measure(name, tactic, call, phase):
                destination = call()
                check(destination, reference)

                def measured_call():
                    nonlocal destination
                    destination = call()
                    return destination

                samples = [
                    v * 1000
                    for v in bench_gpu_time_with_cupti(
                        measured_call,
                        dry_run_iters=3,
                        repeat_iters=12 if phase == "selection" else 40,
                        use_cuda_graph=True,
                        cold_l2_cache=cold,
                    )
                ]
                errors = check(destination, reference)
                row = {
                    "reuse": reuse,
                    "cold": cold,
                    "name": name,
                    "tactic": tactic,
                    "phase": phase,
                    "median_us": median(samples),
                    "samples_us": samples,
                    "errors": errors,
                }
                result["selection" if phase == "selection" else "validation"].append(
                    row
                )
                save()
                return row

            selected = {}
            calls = {}
            for name, choices in families.items():
                print("Selecting", reuse, cold, name, flush=True)
                rows = [
                    measure(
                        name,
                        tactic,
                        lambda tactic=tactic: frost.forward(buffers, tactic),
                        "selection",
                    )
                    for tactic in choices
                ]
                selected[name] = min(rows, key=lambda v: v["median_us"])["tactic"]
                calls[name] = lambda name=name: frost.forward(buffers, selected[name])
            with autotune(True):
                # Baseline tuning hooks can rewrite routing. Give discovery
                # independent live inputs, preserving the measured workload.
                tuning_act = MoEActivationPack(
                    x.clone(), None, ids.clone(), scores.clone()
                )
                choices = native_runner.get_valid_tactics(
                    native_runner.pack_inputs(tuning_act, native_weights), None
                )
            native_choices = []
            for tactic in [native_initial, -1, *choices]:
                if tactic not in native_choices:
                    native_choices.append(tactic)

            def native_call():
                return native_layer(act, native_weights)

            rows = []
            for tactic in native_choices:
                native_layer._winners[bucket] = (native_runner, tactic)
                rows.append(measure(args.baseline, tactic, native_call, "selection"))
            selected[args.baseline] = min(rows, key=lambda v: v["median_us"])["tactic"]
            native_layer._winners[bucket] = (native_runner, selected[args.baseline])
            calls[args.baseline] = native_call
            for baseline in ("fma", args.baseline):
                for repeat in range(2):
                    for current in (baseline, "grouped_tc", "grouped_tc", baseline):
                        measure(
                            current,
                            selected[current],
                            calls[current],
                            f"{baseline}-ABBA-{repeat}",
                        )
            print(reuse, "cold" if cold else "warm", selected, flush=True)
    result["completed"] = True
    save()


if __name__ == "__main__":
    main()
