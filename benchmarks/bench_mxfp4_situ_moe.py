# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compare planned CuTe MXFP4 SiTU MoE with TRT-LLM Gen using CUPTI.

Run from a FlashInfer checkout on SM100/SM103 with cupti-python >= 13:

    python benchmarks/bench_mxfp4_situ_moe.py --tokens 1,16,128 --output smoke.json
    python benchmarks/bench_mxfp4_situ_moe.py --full --accuracy --output kimi.json

The full shape is H7168, I3072, E896, K16, with 112 local experts by default.
Both backends consume the same native quantized operands and BF16 router
weights. All generation, preparation, compilation and accuracy work is excluded
from timing. CUPTI is required; no timing fallback is available.

Reports GPU activity span, summed kernel durations, host submission time, and
synchronized end-to-end runner latency, with raw samples and paired ratios.
GPU span includes correlated memory operations and gaps; kernel sum includes
only CONCURRENT_KERNEL activities, including graph-lowered memset kernels.
Host metrics use CUPTI timestamps with tracing enabled. Both
eager and CUDA Graph samples use the same cold-L2 policy; the flush and its
completion synchronization occur before each sample's start timestamp.

Fixtures and FP64 oracles are shared with tests/moe/mxfp4_situ_reference.py.
This standalone benchmark therefore requires the source checkout, not only an
installed wheel. Resume uses the exact same command/output and --run-id; use a
new run-id and output after changing the implementation or environment.
"""

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import gc
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys


DEFAULT_TOKENS = [1, 2, 4, 8, 16, 128, 256, 512, 1024, 2048, 4096]


def _atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w") as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def _load_reference(repository):
    name = "mxfp4_situ_benchmark_reference"
    path = repository / "tests" / "moe" / "mxfp4_situ_reference.py"
    if not path.is_file():
        raise RuntimeError(
            "Run this benchmark from a complete FlashInfer source checkout"
        )
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _revision(repository):
    result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _compiled_counts():
    """Count resident cache entries without causing new compilation or loading.

    The native routing module contains multiple kernels. Its module count must
    not be added to CuTe callable counts as if it were a single device kernel.
    """
    variants = {}
    for name, module, cache in (
        (
            "gather",
            "blockscaled_contiguous_gather_grouped_gemm_act_fusion",
            "_gather_kernel_cache",
        ),
        (
            "finalize",
            "blockscaled_contiguous_grouped_gemm_finalize_fusion",
            "_finalize_kernel_cache",
        ),
        ("route_preprocess", "mxfp4_routing", "_route_preprocess_kernel_cache"),
    ):
        module_name = "flashinfer.fused_moe.cute_dsl." + module
        loaded = sys.modules.get(module_name)
        entries = {} if loaded is None else getattr(loaded, cache, None)
        variants[name] = {
            "cache": module_name + "." + cache,
            "cache_entries": len(entries) if isinstance(entries, dict) else None,
            "distinct_compiled_callables": (
                len({id(value) for value in entries.values()})
                if isinstance(entries, dict)
                else None
            ),
        }

    def cached_modules(module_name, function_name):
        loaded = sys.modules.get(module_name)
        function = None if loaded is None else getattr(loaded, function_name, None)
        cache_info = getattr(function, "cache_info", None)
        return {
            "cache": module_name + "." + function_name,
            "cached_module_count": (
                cache_info().currsize if cache_info is not None else None
            ),
            "compiled_kernel_count": None,
            "reason": "module-cache occupancy does not enumerate device kernels or resident cubins",
        }

    counts = [item["cache_entries"] for item in variants.values()]
    return {
        "scope": "process-global resident caches after this row; cumulative across prior rows and warmup",
        "cute_dsl_compiled_callables": variants,
        "total_cute_dsl_cache_entries": (
            sum(counts) if all(value is not None for value in counts) else None
        ),
        "routing_and_auxiliary": {
            "shared_moe_utils_nvcc_module": cached_modules(
                "flashinfer.fused_moe.cute_dsl.moe_utils", "_get_moe_utils_module"
            ),
            "moe_sort": {
                "provider": "shared_moe_utils_nvcc_module; flashinfer_moe_sort",
                "compiled_kernel_count": None,
                "reason": "NVCC routing dispatch is inside the shared native module; no Python kernel inventory is exposed",
            },
            "output_clear": {
                "provider": "T<=16: fused CuTe route_preprocess; T>16: cudaMemsetAsync via shared_moe_utils_nvcc_module",
                "compiled_kernel_count": None,
                "reason": "Fused decode variant counted above; prefill clearing is a CUDA-runtime operation",
            },
            "routing_unpack_and_conversion": {
                "provider": "T<=16: fused CuTe route_preprocess; T>16: PyTorch bitwise_right_shift(out=) and copy_",
                "compiled_kernel_count": None,
                "reason": "PyTorch does not expose a per-runner inventory of resident native kernels",
            },
        },
        "trtllm_gen": cached_modules(
            "flashinfer.fused_moe.core", "_get_trtllm_moe_sm100_module_impl"
        ),
        "complete_runner_compiled_kernel_count": None,
        "complete_count_available": False,
    }


def _storage_bytes(tensors):
    storages = {
        value.untyped_storage().data_ptr(): value.untyped_storage().nbytes()
        for value in tensors
    }
    return sum(storages.values())


def _require_cupti():
    from mxfp4_situ_timing import _require_cupti as require

    require()


def _measure(fn, *, mode, warmup, repeats):
    from mxfp4_situ_timing import measure_moe_cupti

    return measure_moe_cupti(fn, mode=mode, warmup=warmup, repeats=repeats)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--tokens", default=",".join(map(str, DEFAULT_TOKENS)))
    parser.add_argument("--distributions", default="balanced,empty,hot")
    parser.add_argument("--modes", default="eager,graph")
    parser.add_argument("--routing", choices=("packed", "separate"), default="packed")
    parser.add_argument("--local-experts", type=int)
    parser.add_argument("--offset", type=int)
    parser.add_argument("--beta", type=float, default=4.0)
    parser.add_argument("--linear-beta", type=float, default=25.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--accuracy", action="store_true")
    parser.add_argument("--offline-tactics", type=Path)
    parser.add_argument("--run-id", default="default")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    tokens = [int(value) for value in args.tokens.split(",")]
    distributions = args.distributions.split(",")
    modes = args.modes.split(",")
    if not tokens or min(tokens) < 1 or args.repeats < 1 or args.warmup < 1:
        parser.error("token counts, repeats and warmup must be positive")
    if not set(distributions) <= {"balanced", "empty", "hot", "all_remote"}:
        parser.error("unsupported synthetic routing distribution")
    if not set(modes) <= {"eager", "graph"}:
        parser.error("modes must be eager,graph")
    if not all(math.isfinite(x) and x > 0 for x in (args.beta, args.linear_beta)):
        parser.error("paired TRT measurements require finite positive SiTU bounds")

    repository = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repository))
    _require_cupti()
    import torch

    from flashinfer.fused_moe.cute_dsl.mxfp4 import CuteDslMxfp4MoEWrapper

    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA device is required")
    architecture = torch.cuda.get_device_capability()
    if architecture not in ((10, 0), (10, 3)):
        raise RuntimeError("this native MXFP4 benchmark requires SM100 or SM103")
    reference = _load_reference(repository)
    hidden, intermediate, experts, top_k = (
        (7168, 3072, 896, 16) if args.full else (256, 128, 8, 2)
    )
    local = (
        args.local_experts
        if args.local_experts is not None
        else (112 if args.full else 4)
    )
    offset = (
        args.offset
        if args.offset is not None
        else (336 if args.full and local == 112 else 0)
    )
    if local < 1 or offset < 0 or offset + local > experts:
        parser.error(
            "local experts must form a valid contiguous global expert interval"
        )
    if "all_remote" in distributions and experts - local < top_k:
        parser.error("all_remote requires at least top_k nonlocal experts")
    tactics = None
    if args.offline_tactics:
        tactics = {
            int(key): tuple(value)
            for key, value in json.loads(args.offline_tactics.read_text()).items()
        }
    configuration = {
        "run_id": args.run_id,
        "source_revision": _revision(repository),
        "hidden": hidden,
        "intermediate": intermediate,
        "num_experts": experts,
        "local_experts": local,
        "local_offset": offset,
        "top_k": top_k,
        "tokens": tokens,
        "distributions": distributions,
        "modes": modes,
        "routing": args.routing,
        "routing_weights": "bfloat16",
        "beta": args.beta,
        "linear_beta": args.linear_beta,
        "seed": args.seed,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "accuracy": args.accuracy,
        "cold_l2_cache": True,
        "offline_tactics": tactics,
        "compiled_counts_schema": 3,
        "timing_schema": 2,
    }
    configuration = json.loads(json.dumps(configuration))
    environment = {
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": list(architecture),
        "device_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "flashinfer": _version("flashinfer-python"),
        "cutlass_dsl": _version("nvidia-cutlass-dsl"),
        "cupti": _version("cupti-python"),
    }
    checkpoint = args.checkpoint or Path(str(args.output) + ".checkpoint.json")
    started = datetime.now(timezone.utc)
    if checkpoint.exists():
        state = json.loads(checkpoint.read_text())
        if (
            state["configuration"] != configuration
            or state["environment"] != environment
        ):
            raise RuntimeError(
                "resume configuration/environment mismatch; use a new output and checkpoint"
            )
    else:
        state = {
            "configuration": configuration,
            "environment": environment,
            "created_at": started.isoformat(),
            "rows": {},
            "physical_elapsed_seconds": 0.0,
            "timing_scope": {
                "gpu_span_ms": "first-to-last correlated GPU kernel/memory activity, including gaps",
                "kernel_sum_ms": "sum of correlated CONCURRENT_KERNEL durations, including graph-lowered memset kernels; excludes MEMCPY/MEMSET activity records and gaps",
                "host_enqueue_ms": "CUPTI timestamp before runner submission through return",
                "synchronized_e2e_ms": "CUPTI timestamp before runner submission through device synchronization",
                "excluded": "preparation, compilation, capture, warmup, L2 flush, pre-sample synchronization, FP64 reference",
                "host_metrics": "include CUPTI instrumentation overhead",
            },
        }

    def save():
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        _atomic_json(checkpoint, state)
        _atomic_json(args.output, state)

    def complete(key):
        return state["rows"].get(key, {}).get("status") == "complete"

    if all(
        complete(f"{count}/{distribution}/{mode}")
        for count in tokens
        for distribution in distributions
        for mode in modes
    ):
        print(json.dumps({"status": "already_complete", "output": str(args.output)}))
        return
    state["status"] = "preparing"
    save()
    base = reference.make_case(
        tokens=max(tokens),
        hidden=hidden,
        intermediate=intermediate,
        num_experts=experts,
        local_num_experts=local,
        local_expert_offset=offset,
        top_k=top_k,
        seed=args.seed,
        beta=args.beta,
        linear_beta=args.linear_beta,
    )
    candidate_weights = reference.prepare_cute_weights(base)
    baseline_weights = reference.prepare_trt_weights(base)
    candidate_weight_bytes = _storage_bytes(candidate_weights)
    state["candidate_weight_bytes"] = candidate_weight_bytes
    state["baseline_weight_bytes"] = _storage_bytes(baseline_weights)
    state["weight_memory_note"] = (
        "Separate native banks coexist for paired measurement; the candidate retains one prepared bank."
    )
    active_key = None
    try:
        for count in tokens:
            for distribution in distributions:
                if all(complete(f"{count}/{distribution}/{mode}") for mode in modes):
                    continue
                ids, weights = reference.make_routing(
                    count,
                    experts,
                    top_k,
                    local,
                    offset,
                    distribution,
                    seed=args.seed + 1,
                )
                case = replace(
                    base,
                    x=base.x[:count],
                    x_scale=base.x_scale[:count],
                    topk_ids=ids,
                    topk_weights=weights,
                )
                wrapper = CuteDslMxfp4MoEWrapper(
                    experts,
                    top_k,
                    hidden,
                    intermediate,
                    num_local_experts=local,
                    local_expert_offset=offset,
                    enable_pdl=False,
                    offline_tactics=tactics,
                )
                workspace = torch.empty(
                    wrapper.get_workspace_size(count), device="cuda", dtype=torch.uint8
                )
                output = torch.empty(
                    (count, hidden), device="cuda", dtype=torch.bfloat16
                )
                packed = args.routing == "packed"
                candidate_ids = reference.pack_topk(ids, weights) if packed else ids
                plan = wrapper.plan(
                    case.x,
                    case.x_scale,
                    candidate_ids,
                    None if packed else weights,
                    *candidate_weights,
                    beta=case.beta,
                    linear_beta=case.linear_beta,
                    workspace=workspace,
                    output=output,
                )
                baseline, baseline_output = reference.make_trt_baseline(
                    case,
                    baseline_weights,
                    packed=packed,
                )
                plan.run()
                baseline()
                numerical = None
                if args.accuracy:
                    oracles = reference.reference_moe(case)
                    numerical = {
                        name: reference.paired_accuracy(output, baseline_output, oracle)
                        for name, oracle in oracles.items()
                    }
                    if not all(
                        row[backend]["finite"]
                        for row in numerical.values()
                        for backend in ("candidate", "baseline")
                    ):
                        raise RuntimeError(
                            "non-finite output or reference in accuracy evaluation"
                        )
                    del oracles
                histogram = reference.routing_histogram(case)
                for mode in modes:
                    active_key = f"{count}/{distribution}/{mode}"
                    if complete(active_key):
                        continue
                    state["status"] = "measuring"
                    state["active_row"] = active_key
                    save()
                    implementations = [
                        ("candidate", plan.run),
                        ("trtllm_gen", baseline),
                    ]
                    if (tokens.index(count) + distributions.index(distribution)) % 2:
                        implementations.reverse()
                    measurements = {
                        name: _measure(
                            fn, mode=mode, warmup=args.warmup, repeats=args.repeats
                        )
                        for name, fn in implementations
                    }
                    ratios = {
                        "trtllm_over_candidate_" + metric: (
                            measurements["trtllm_gen"][metric + "_ms"]
                            / measurements["candidate"][metric + "_ms"]
                        )
                        for metric in (
                            "gpu_span",
                            "kernel_sum",
                            "host_enqueue",
                            "synchronized_e2e",
                        )
                    }
                    state["rows"][active_key] = {
                        "status": "complete",
                        "tokens": count,
                        "distribution": distribution,
                        "mode": mode,
                        "routing": args.routing,
                        "measurements": measurements,
                        **ratios,
                        "routing_histogram": histogram,
                        "workspace_bytes": workspace.numel(),
                        "output_bytes": output.numel() * output.element_size(),
                        "candidate_weight_bytes": candidate_weight_bytes,
                        "compiled_counts": _compiled_counts(),
                        "numerical": numerical,
                    }
                    save()
                    print(
                        json.dumps({"row": active_key, **ratios}),
                        flush=True,
                    )
                del (
                    plan,
                    baseline,
                    baseline_output,
                    workspace,
                    output,
                    case,
                    implementations,
                )
                gc.collect()
    except BaseException as error:
        state["status"] = "failed"
        state["failure"] = {
            "row": active_key,
            "type": type(error).__name__,
            "message": str(error),
        }
        raise
    else:
        state["status"] = "complete"
        state.pop("failure", None)
    finally:
        state["physical_elapsed_seconds"] += (
            datetime.now(timezone.utc) - started
        ).total_seconds()
        save()
    print(
        json.dumps({"status": state["status"], "output": str(args.output)}), flush=True
    )


if __name__ == "__main__":
    main()
