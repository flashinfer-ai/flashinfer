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
SM80 defaults to cuDNN; its standalone vLLM-device comparison additionally needs
--baseline vllm_device_triton --vllm-source /path/to/fused_moe.py from upstream
commit 71c20f73f2f1764576fed799c152319c9fd1c820. This loads only two unmodified
Triton device functions, checks the pinned file hash, and requires no vLLM install.
It is not a stock vLLM or serving benchmark; see baseline_contract in the JSON.
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
    ReLU2,
    GeGLU,
    GeGLUTanh,
    SiTU,
    SwiGLUStep,
    SiLU,
)
from flashinfer.fused_moe.backends.frost_low_latency.runner import FrostLowLatencyRunner
from flashinfer.testing import bench_gpu_time_with_cupti
from tests.moe.test_frost_low_latency_bf16 import _inputs, _reference


def _require_cupti():
    """Fail closed instead of silently changing the declared timing/cache mode."""
    try:
        from cupti import cupti

        installed = version("cupti-python")
    except ImportError as exc:
        raise RuntimeError(
            "This benchmark requires cupti-python >= 13; CUDA-event fallback would invalidate its cold-cache labels"
        ) from exc
    if int(installed.split(".")[0]) < 13 or not callable(
        getattr(cupti, "activity_enable", None)
    ):
        raise RuntimeError("This benchmark requires working cupti-python >= 13")
    return installed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, choices=(4, 8), default=4)
    parser.add_argument("--shape", choices=("small", "large"), default="small")
    parser.add_argument("--route-seed", type=int, default=7191)
    parser.add_argument(
        "--baseline",
        choices=(
            "cutlass_bf16",
            "cutlass_fp16",
            "cutile_bf16",
            "cudnn_grouped_gemm_bf16",
            "vllm_device_triton",
        ),
        default=None,
    )
    parser.add_argument(
        "--reuses",
        nargs="+",
        choices=("shared", "paired", "quartet", "disjoint"),
        default=("shared", "paired", "quartet", "disjoint"),
    )
    parser.add_argument(
        "--vllm-source",
        type=Path,
        help="Pinned vLLM fused_moe.py for the standalone device comparator",
    )
    parser.add_argument(
        "--vllm-route-policy",
        choices=("both", "naive", "grouped"),
        default="both",
        help="Standalone vLLM device comparison: include expert grouping with timed FI sorting, or reproduce the naive upstream low-token policy",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument(
        "--activation",
        choices=(
            "swiglu",
            "relu2",
            "geglu",
            "silu",
            "geglu_tanh",
            "situ",
            "swiglu_step",
        ),
        default="swiglu",
    )
    args = parser.parse_args()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    cupti_version = _require_cupti()
    activation = {
        "swiglu": SwiGLU,
        "relu2": ReLU2,
        "geglu": GeGLU,
        "silu": SiLU,
        "geglu_tanh": GeGLUTanh,
        "situ": SiTU,
        "swiglu_step": SwiGLUStep,
    }[args.activation]()
    arch = torch.cuda.get_device_capability()
    if arch not in ((8, 0), (9, 0), (12, 0)):
        raise RuntimeError("This benchmark qualifies SM80, SM90 and SM120")
    if args.baseline is None:
        args.baseline = (
            "cudnn_grouped_gemm_bf16"
            if arch == (8, 0)
            else ("cutlass_fp16" if dtype == torch.float16 else "cutlass_bf16")
        )
    standalone = args.baseline == "vllm_device_triton"
    if standalone != (args.vllm_source is not None):
        parser.error(
            "--baseline vllm_device_triton requires --vllm-source (only for that baseline)"
        )
    if args.shape == "large" and arch != (8, 0):
        parser.error("The large grouped workload is qualified only on SM80")
    if arch == (8, 0) and (dtype == torch.float16 or args.activation != "swiglu"):
        if (
            args.shape != "small"
            or not standalone
            or args.activation not in ("swiglu", "relu2")
        ):
            parser.error(
                "Typed SM80 cases require small SwiGLU/ReLU2 and the pinned standalone baseline"
            )
    if standalone and args.activation not in ("swiglu", "relu2"):
        parser.error("The pinned standalone adapter supports SwiGLU and ReLU2")
    if dtype == torch.float16 and not standalone and args.baseline != "cutlass_fp16":
        parser.error("FP16 uses native CUTLASS or the pinned standalone baseline")
    if dtype == torch.bfloat16 and args.baseline == "cutlass_fp16":
        parser.error("--baseline cutlass_fp16 requires --dtype fp16")
    t = args.tokens
    h, i, e, k = (7168, 2048, 256, 8) if args.shape == "large" else (2048, 512, 512, 10)
    print("Preparing checkpoint and Frost buffers", flush=True)
    cfg, x, w1, w2, ids, scores, weights = _inputs(
        activation,
        tokens=t,
        hidden=h,
        intermediate=i,
        experts=e,
        topk=k,
        dtype=dtype,
        output_dtype=dtype,
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
        "grouped_tc": [
            v for v in tactics if v[0] in ("wgmma", "warp_mma", "ampere_mma")
        ],
    }
    assert len(families["fma"]) == 16
    if not families["grouped_tc"]:
        parser.error(
            "This architecture/dtype/activation/token case has no advertised grouped TC tactic"
        )
    if standalone:
        from benchmarks.frost_vllm_device_baseline import (
            VllmTritonComparator,
            VLLM_COMMIT,
            VLLM_SOURCE_SHA256,
        )

        native_runner = VllmTritonComparator(
            x,
            w1,
            w2,
            ids,
            scores,
            args.vllm_source,
            gated=args.activation == "swiglu",
            route_policy=args.vllm_route_policy,
        )
        native_initial = native_runner.tactics[0]
        native_selected = native_initial
        native_runner.forward(native_initial)
    elif dtype == torch.float16:
        from flashinfer.fused_moe.core import (
            get_cutlass_fused_moe_module,
            cutlass_fused_moe,
            cutlass_fused_moe_workspace_size,
        )
        from flashinfer.autotuner import AutoTuner, TuningConfig

        cc = torch.cuda.get_device_capability()
        module = get_cutlass_fused_moe_module(str(cc[0] * 10 + cc[1]))
        inner = module.MoERunner(
            x_dtype=dtype,
            weight_dtype=dtype,
            output_dtype=dtype,
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
            enable_pdl=False,
            activation_type=activation.type,
            use_packed_weights=False,
            use_fused_finalize=True,
            use_wfp4afp8_humming=False,
        )
        stage_choices = []
        with autotune(True):
            for stage in (1, 2):
                inner.gemm_idx_for_tuning = stage
                stage_choices.append(
                    AutoTuner.get().rank_tactics(
                        f"frost_grouped_fp16_native_stage{stage}",
                        [inner],
                        TuningConfig(),
                        [x, w1, None, w2, None],
                        k=2,
                        gemm_idx=stage,
                    )
                )
        inner.gemm_idx_for_tuning = None
        native_choices = [
            -1,
            *[(int(a), int(b)) for a in stage_choices[0] for b in stage_choices[1]],
        ]
        workspace = torch.empty(
            cutlass_fused_moe_workspace_size(
                t,
                h,
                i,
                e,
                k,
                x_dtype=dtype,
                weight_dtype=dtype,
                output_dtype=dtype,
                use_w4_group_scaling=False,
                device=x.device,
            ),
            device=x.device,
            dtype=torch.uint8,
        )
        native_output = torch.empty_like(x)

        def native(tactic):
            with autotune(False):
                value = cutlass_fused_moe(
                    x,
                    ids,
                    scores,
                    w1,
                    w2,
                    dtype,
                    [],
                    output=native_output,
                    use_w4_group_scaling=False,
                    tune_max_num_tokens=t,
                    activation_type=activation.type,
                    enable_pdl=False,
                    profile_ids=None if tactic == -1 else list(tactic),
                    workspace_buffer=workspace,
                )
            return value[0] if isinstance(value, (list, tuple)) else value

        native(-1)
        args.baseline = "cutlass_fp16"
    else:
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
                activation=activation,
            ),
        )
        native_layer = MoELayer(replace(cfg, backend=BackendOptions((native_config,))))
        print("Preparing", args.baseline, "(first use can compile)", flush=True)
        with autotune(False):
            native_layer(act, native_weights)
        print(args.baseline, "ready", flush=True)
        bucket, (native_runner, native_initial) = next(
            iter(native_layer._winners.items())
        )
        assert native_runner.backend_key == args.baseline
    root = Path(__file__).resolve().parents[1]
    sources = [
        Path(__file__).resolve(),
        root / "flashinfer/fused_moe/core.py",
        root / "flashinfer/fused_moe/runners.py",
        root / "flashinfer/fused_moe/backends/frost_low_latency/runner.py",
        root / "flashinfer/fused_moe/backends/frost_low_latency/runtime.py",
        *sorted(
            (root / "flashinfer/fused_moe/backends/frost_low_latency/tc").glob("*.py")
        ),
    ]
    if standalone:
        sources.append(Path(__file__).with_name("frost_vllm_device_baseline.py"))
    result = {
        "shape": {"T": t, "H": h, "I": i, "E": e, "top_k": k},
        "route_seed": args.route_seed,
        "dtype": args.dtype,
        "activation": repr(activation),
        "gpu": torch.cuda.get_device_name(),
        "software": {
            "torch": torch.__version__,
            "cupti": cupti_version,
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
        "native_search": (
            f"{len(native_runner.tactics)} shared FC1/FC2 configurations across the requested {args.vllm_route_policy} route policies; not exhaustive independent-stage pairs"
            if standalone
            else (
                "Per-stage top two, their Cartesian product plus heuristic; not exhaustive"
                if dtype == torch.float16
                else "Runner compound candidates plus default, not exhaustive"
            )
        ),
        "baseline_contract": (
            "Unmodified vLLM matmul device functions with standalone Triton activation/combine "
            "and the requested naive/grouped route policy. Grouped execution includes FI GPU sorting and a padding adapter in timing. Native typed preactivation and weighted-before-cast "
            "expert rounding differ from Frost; independent references. Not stock vLLM or serving."
            if standalone
            else "FlashInfer native runner; same checkpoint, backend-specific rounding"
        ),
        "selection": [],
        "validation": [],
        "completed": False,
    }

    if standalone:
        result["vllm_device_source"] = {
            "route_policy": args.vllm_route_policy,
            "commit": VLLM_COMMIT,
            "sha256": VLLM_SOURCE_SHA256,
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
        _, reference, _ = _reference(
            x, w1, w2, ids, scores, activation, True, output_dtype=dtype
        )
        native_reference = native_runner.reference() if standalone else reference
        for cold in (False, True):

            def measure(name, tactic, call, phase):
                destination = call()
                expected = native_reference if name == args.baseline else reference
                check(destination, expected)

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
                errors = check(destination, expected)
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
            if standalone:
                native_choices = native_runner.get_valid_tactics()

                def native_call():
                    return native_runner.forward(native_selected)

                rows = []
                for tactic in native_choices:
                    native_selected = tactic
                    rows.append(
                        measure(args.baseline, tactic, native_call, "selection")
                    )
                selected[args.baseline] = min(rows, key=lambda v: v["median_us"])[
                    "tactic"
                ]
                native_selected = selected[args.baseline]
                calls[args.baseline] = native_call
            elif dtype == torch.float16:
                rows = [
                    measure(
                        args.baseline,
                        tactic,
                        lambda tactic=tactic: native(tactic),
                        "selection",
                    )
                    for tactic in native_choices
                ]
                selected[args.baseline] = min(rows, key=lambda row: row["median_us"])[
                    "tactic"
                ]
                calls[args.baseline] = lambda: native(selected[args.baseline])
            else:
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
                    rows.append(
                        measure(args.baseline, tactic, native_call, "selection")
                    )
                selected[args.baseline] = min(rows, key=lambda v: v["median_us"])[
                    "tactic"
                ]
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
