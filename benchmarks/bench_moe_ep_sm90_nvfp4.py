#!/usr/bin/env python3
# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Public-API SM90 NVFP4/MXFP4 EP4 compute benchmark.

Run with torchrun --standalone --nproc-per-node=4. Weight preparation, input
staging, allocation, cooldown and padding are outside CUDA events. The measured
region is compute_staged(): MegaMoE plus top-k reduction. The deterministic
common-BF16 fixture follows the historical input/weight recipe. Exact saved
bytes across PyTorch/CUDA versions require a separate check.
Initialization and quantization are outside the measured interval.
"""

import argparse
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import statistics
import time


TOKENS = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]


def quantize_weight(torch, weight, mode):
    """RN-even E2M1; NVFP4 uses per-expert G and nearest E4M3 K16 scales."""
    assert weight.ndim == 2 and mode in ("nvfp4", "mxfp4")
    group = 16 if mode == "nvfp4" else 32
    rows, k = weight.shape
    blocks = weight.float().reshape(rows, k // group, group)
    amax = blocks.abs().amax(-1)
    if mode == "nvfp4":
        global_scale = (amax.amax() / (6 * 448)).clamp_min(1e-30)
        scale = (amax / (6 * global_scale)).clamp(0, 448).to(torch.float8_e4m3fn)
        decoded = scale.float() * global_scale
        scale_bytes = scale.view(torch.uint8)
    else:
        exponent = torch.ceil(torch.log2((amax / 6).clamp_min(2.0**-126)))
        scale_bytes = (exponent + 127).to(torch.uint8)
        decoded = torch.exp2(exponent)
        global_scale = torch.ones((), device=weight.device)
    normalized = blocks / decoded.clamp_min(1e-30).unsqueeze(-1)
    magnitude = normalized.abs().contiguous()
    boundaries = torch.tensor(
        (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0), device=weight.device
    )
    codes = torch.bucketize(magnitude, boundaries).to(torch.uint8)
    # bucketize picks the lower code at a midpoint; choose the even code.
    ties = magnitude == boundaries[codes.clamp_max(6).long()]
    codes += (ties & ((codes & 1) != 0)).to(torch.uint8)
    codes |= torch.signbit(normalized).to(torch.uint8) << 3
    codes = codes.reshape(rows, k)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
    return packed, scale_bytes.contiguous(), global_scale.float()


def bf16_weight(torch, expert, rows, k, leg, device):
    # Per-expert seed is independent of rank/chunking and format.
    generator = torch.Generator(device=device).manual_seed(
        20260914 + expert * 17 + leg * 100003
    )
    return (
        torch.randn((rows, k), generator=generator, device=device) / math.sqrt(k)
    ).to(torch.bfloat16)


def make_weight_packs(torch, args, rank, world, device, nv_pack, mx_pack):
    """Quantize one shared BF16 expert at a time; initialization is untimed."""
    local_experts = args.experts // world
    legs, summaries = [], []
    for leg, (rows, k) in enumerate(
        ((2 * args.intermediate, args.hidden), (args.hidden, args.intermediate))
    ):
        nv_raw = torch.empty(
            (local_experts, rows, k // 2), dtype=torch.uint8, device=device
        )
        mx_raw = torch.empty_like(nv_raw)
        nv_scale = torch.empty(
            (local_experts, rows, k // 16), dtype=torch.uint8, device=device
        )
        mx_scale = torch.empty(
            (local_experts, rows, k // 32), dtype=torch.uint8, device=device
        )
        nv_global = torch.empty(local_experts, dtype=torch.float32, device=device)
        for expert in range(local_experts):
            weight = bf16_weight(
                torch, rank * local_experts + expert, rows, k, leg, device
            )
            raw, scale, global_decode = quantize_weight(torch, weight, "nvfp4")
            nv_raw[expert].copy_(raw)
            nv_scale[expert].copy_(scale)
            nv_global[expert].copy_(global_decode)
            raw, scale, _ = quantize_weight(torch, weight, "mxfp4")
            mx_raw[expert].copy_(raw)
            mx_scale[expert].copy_(scale)
        summaries.append(
            {
                "leg": leg,
                "nv_scale_code_min": int(nv_scale.min().item()),
                "nv_scale_code_max": int(nv_scale.max().item()),
                "mx_scale_code_min": int(mx_scale.min().item()),
                "mx_scale_code_max": int(mx_scale.max().item()),
                "nv_global_decode_min": float(nv_global.min().item()),
                "nv_global_decode_max": float(nv_global.max().item()),
            }
        )
        legs.append((nv_raw, nv_scale, nv_global, mx_raw, mx_scale))
    first, second = legs
    packs = {
        "nv": nv_pack(
            w13=first[0],
            w2=second[0],
            w13_scale=first[1].view(torch.float8_e4m3fn),
            w2_scale=second[1].view(torch.float8_e4m3fn),
            w13_global_scale=first[2],
            w2_global_scale=second[2],
        ),
        "mx": mx_pack(
            w13=first[3],
            w2=second[3],
            w13_scale=first[4],
            w2_scale=second[4],
        ),
    }
    return packs, summaries


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=TOKENS)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--intermediate", type=int, default=3072)
    parser.add_argument("--experts", type=int, default=384)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Routing seed; weight/input seeds are fixed",
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--padding-us", type=float, default=300.0)
    parser.add_argument("--cooldown-seconds", type=float, default=5.0)
    parser.add_argument("--nv-kperm", type=int, choices=(0, 32, 64), default=64)
    parser.add_argument("--nv-overlap", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--nv-knobs-json", type=Path)
    parser.add_argument("--nv-reference-knobs-json", type=Path)
    parser.add_argument("--mx-knobs-json", type=Path)
    parser.add_argument("--mx-compact", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("sm90_nvfp4_benchmark.json")
    )
    args = parser.parse_args()
    if min(args.tokens + [args.rounds, args.warmup, args.iters]) <= 0:
        parser.error("token counts, rounds, warmup and iters must be positive")
    if min(args.padding_us, args.cooldown_seconds) < 0:
        parser.error("padding and cooldown must be non-negative")
    return args


def load_knobs(path, tokens):
    """Accept a complete tactic, or {per_token: {T: {tactic, config}}}."""
    if path is None:
        return None, {}
    value = json.loads(path.read_text())
    if "per_token" in value:
        value = value["per_token"][str(tokens)]
    tactic = value.get("tactic", value.get("requested_tactic", value))
    config = value.get("config", value.get("requested_config", {}))
    return tactic, config


def cpu_binding(local_rank, world_size):
    """Choose two physical cores, or preserve an already restricted rank binding."""
    available = sorted(os.sched_getaffinity(0))
    cores = {}
    for cpu in available:
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        key = tuple(
            (topology / name).read_text().strip()
            for name in ("physical_package_id", "core_id")
        )
        cores.setdefault(key, []).append(cpu)
    groups = list(cores.values())
    if len(groups) >= 2 * world_size:
        chosen = groups[2 * local_rank : 2 * local_rank + 2]
        os.sched_setaffinity(0, {cpu for group in chosen for cpu in group})
    elif len(groups) < 2:
        raise RuntimeError("provide at least two physical CPU cores per rank")
    return sorted(os.sched_getaffinity(0))


def calibrate_padding(torch, target_us):
    if target_us == 0:
        return 0, {"target_us": 0, "disabled": True}
    cycles = max(1, round(target_us * 1000))
    history = []
    for _ in range(3):
        samples = []
        for _ in range(3):
            begin, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            torch.cuda.synchronize()
            torch.cuda._sleep(2_000_000)
            begin.record()
            torch.cuda._sleep(cycles)
            end.record()
            end.synchronize()
            samples.append(begin.elapsed_time(end) * 1000)
        observed = statistics.median(samples)
        history.append({"cycles": cycles, "samples_us": samples})
        cycles = max(1, min(20_000_000, round(cycles * target_us / observed)))
    verification = []
    for _ in range(5):
        begin, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        torch.cuda.synchronize()
        torch.cuda._sleep(2_000_000)
        begin.record()
        torch.cuda._sleep(cycles)
        end.record()
        end.synchronize()
        verification.append(begin.elapsed_time(end) * 1000)
    if abs(statistics.median(verification) / target_us - 1) > 0.15:
        raise RuntimeError(f"GPU padding calibration failed: {verification}")
    return cycles, {
        "target_us": target_us,
        "cycles": cycles,
        "history": history,
        "verification_us": verification,
    }


def collective_check(torch, dist, group, condition, message):
    """Fan out a host-side failure before any rank enters the next GPU barrier."""
    flags = [None] * dist.get_world_size()
    dist.all_gather_object(flags, bool(condition), group=group)
    if not all(flags):
        raise RuntimeError(f"{message}; per-rank status: {flags}")


def measure(torch, dist, group, layer, output, args, cycles, reference, retained):
    attempts = retained["attempts"]
    for attempt in range(5):
        time.sleep(args.cooldown_seconds)
        for _ in range(args.warmup):
            layer.compute_staged(output=output)
        torch.cuda.synchronize()
        samples, pending, host_ns = [], [], []
        for _ in range(args.iters):
            dist.barrier()
            torch.cuda.synchronize()
            begin, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            if cycles:
                torch.cuda._sleep(cycles)
            begin.record()
            before = time.monotonic_ns()
            layer.compute_staged(output=output)
            after = time.monotonic_ns()
            pending.append(not begin.query())
            end.record()
            end.synchronize()
            samples.append(begin.elapsed_time(end) * 1000)
            host_ns.append(after - before)
        good = bool(torch.isfinite(output).all()) and torch.equal(
            output.view(torch.uint8), reference.view(torch.uint8)
        )
        collective_check(
            torch, dist, group, good, "non-finite or non-repeatable output"
        )
        row = {
            "samples_us": samples,
            "median_us": statistics.median(samples),
            "start_pending": pending,
            "host_submit_ns": host_ns,
            "repeat_bytes_equal": good,
        }
        rows = [None] * dist.get_world_size()
        dist.all_gather_object(rows, row, group=group)
        medians = sorted(x["median_us"] for x in rows)
        score = medians[-1]
        threshold = max(12.0, 0.01 * score) if output.shape[0] >= 2048 else 12.0
        gap = medians[2] - medians[1]
        attempts.append(
            {
                "attempt": attempt,
                "ranks": rows,
                "score_us": score,
                "middle_rank_gap_us": gap,
                "threshold_us": threshold,
                "qualified": gap <= threshold,
            }
        )
        if gap <= threshold:
            retained.update(score_us=score, selected_attempt=attempt)
            return retained
    raise RuntimeError(f"five rank-skew attempts exhausted: {attempts}")


def main():
    args = parse_args()
    rank, local_rank, world = (
        int(os.environ[name]) for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE")
    )
    if (
        world != 4
        or args.experts % world
        or args.hidden % 128
        or args.intermediate % 128
    ):
        raise ValueError(
            "requires EP4, experts divisible by four and K128-aligned widths"
        )
    affinity = cpu_binding(local_rank, world)
    import torch
    import torch.distributed as dist
    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpLayer,
        MoEEpTensors,
        Nvfp4WeightPack,
        PrequantizedMoEWeights,
        Sm90_Fp8_Nvfp4_Bf16_PullCutedsl_MegaMoeConfig as NVConfig,
        Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig as MXConfig,
        preprocess_sm90_pull_nvfp4_mega_weights as prepare_nv,
        preprocess_sm90_pull_mxfp4_mega_weights as prepare_mx,
    )
    from flashinfer.moe_ep.sm90_routing import (
        generate_sm90_block_permutation_routes_numpy,
    )

    torch.set_num_threads(1)
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    host_group = dist.new_group(backend="gloo", timeout=timedelta(minutes=30))
    device = torch.device("cuda", local_rank)
    packs, weight_summary = make_weight_packs(
        torch, args, rank, world, device, Nvfp4WeightPack, PrequantizedMoEWeights
    )
    prepared_nv = {}
    prepared_mx = prepare_mx(
        packs["mx"], intermediate_size=args.intermediate, hidden_size=args.hidden
    )
    results = {
        "fixture": "common_bf16_historical_recipe_v1",
        "fixture_recipe": {
            "weight_seed": "20260914 + global_expert * 17 + leg * 100003",
            "weights": "BF16(randn / sqrt(K)); separate NV K16 and MX K32 quantization",
            "input_and_score_seed": "42 + rank, reset for each token bucket",
            "scores": "softmax(randn)",
            "routing_seed": args.seed,
            "byte_identity": "recipe-compatible; cross-version saved-fixture bytes not asserted",
        },
        "weight_scale_summary_by_rank": None,
        "arguments": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "scope": "compute_staged MegaMoE + top-k reduce; staging/allocation/padding/cooldown excluded",
        "frequency": "externally controlled; CUDA events and lock requests do not prove continuous clock frequency",
        "buckets": {},
    }
    props = torch.cuda.get_device_properties(device)
    rank_metadata = {
        "rank": rank,
        "pid": os.getpid(),
        "cpu_affinity": affinity,
        "device": props.name,
        "uuid": str(getattr(props, "uuid", "unavailable")),
    }
    metadata = [None] * world
    dist.all_gather_object(metadata, rank_metadata, group=host_group)
    results["ranks"] = metadata
    weight_summaries = [None] * world
    dist.all_gather_object(weight_summaries, weight_summary, group=host_group)
    results["weight_scale_summary_by_rank"] = weight_summaries
    cycles, calibration = calibrate_padding(torch, args.padding_us)
    calibrations = [None] * world
    dist.all_gather_object(calibrations, calibration, group=host_group)
    results["padding_calibration"] = calibrations
    try:
        for tokens in args.tokens:
            generator = torch.Generator(device=device).manual_seed(42 + rank)
            hidden = torch.randn(
                (tokens, args.hidden),
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            )
            routes = generate_sm90_block_permutation_routes_numpy(
                world_size=world,
                tokens=tokens,
                topk=args.top_k,
                total_experts=args.experts,
                seed=args.seed,
            )
            ids = torch.as_tensor(routes[rank], dtype=torch.int64, device=device)
            scores = torch.softmax(
                torch.randn((tokens, args.top_k), device=device, generator=generator),
                dim=-1,
            ).to(torch.float32)
            tensors = MoEEpTensors(
                hidden_states=hidden, topk_ids=ids, topk_weights=scores
            )
            cases = [("nv", args.nv_knobs_json), ("mx", args.mx_knobs_json)]
            if args.nv_reference_knobs_json:
                cases.insert(0, ("nv_reference", args.nv_reference_knobs_json))
            layers, outputs, references, configurations = {}, {}, {}, {}
            try:
                for name, path in cases:
                    knobs, override = load_knobs(path, tokens)
                    common = {
                        "intermediate_size": args.intermediate,
                        "top_k": args.top_k,
                        "knobs": knobs,
                        "gate_up_clamp": 10.0,
                        "routing_profile": "block_permutation_v1",
                    }
                    if name.startswith("nv"):
                        overlap = {"auto": None, "on": True, "off": False}[
                            args.nv_overlap
                        ]
                        if "kperm" in override:
                            if "nvfp4_kperm" in override:
                                raise ValueError(
                                    "NV profile specifies two K-layout keys"
                                )
                            override = dict(override)
                            override["nvfp4_kperm"] = override.pop("kperm")
                        options = {
                            "nvfp4_kperm": args.nv_kperm,
                            "nvfp4_fc1_k256_overlap": overlap,
                            **override,
                        }
                        config = NVConfig(**common, **options)
                        if config.nvfp4_kperm not in prepared_nv:
                            prepared_nv[config.nvfp4_kperm] = prepare_nv(
                                packs["nv"],
                                intermediate_size=args.intermediate,
                                hidden_size=args.hidden,
                                nvfp4_kperm=config.nvfp4_kperm,
                            )
                        transformed, pack = prepared_nv[config.nvfp4_kperm], packs["nv"]
                    else:
                        config = MXConfig(
                            **common,
                            **{"compact_pull_buffer": args.mx_compact, **override},
                        )
                        transformed, pack = prepared_mx, packs["mx"]
                    layer = MoEEpLayer(
                        bootstrap=BootstrapConfig(
                            world_size=world, rank=rank, process_group=dist.group.WORLD
                        ),
                        fleet_params=FleetParams(
                            num_experts=args.experts,
                            max_tokens_per_rank=tokens,
                            token_hidden_size=args.hidden,
                            tuning_live_tokens=tokens,
                        ),
                        weights=pack,
                        backend=MegaConfig(
                            megakernel=config,
                            quantize_input=True,
                            preprocess_weights=False,
                            transformed_weights=transformed,
                        ),
                    )
                    layers[name] = layer
                    output = torch.empty_like(hidden)
                    outputs[name] = output
                    layer.warmup(tensors)
                    layer.stage_inputs(tensors)
                    layer.compute_staged(output=output)
                    torch.cuda.synchronize()
                    collective_check(
                        torch,
                        dist,
                        host_group,
                        bool(torch.isfinite(output).all()),
                        "initial output is not finite",
                    )
                    references[name] = output.clone()
                    frontend = layer._staged_workspace._frontend
                    kernel = frontend._mega.kernel
                    actual_layout = {
                        "num_ab_stage": int(kernel.num_ab_stage),
                        "compact_pull_buffer": bool(
                            kernel.token_comm.compact_pull_buffer
                        ),
                    }
                    if name.startswith("nv"):
                        conversion = kernel.nvfp4_conversion
                        actual_layout.update(
                            kperm=int(frontend.config.nvfp4_kperm),
                            fc1_overlap=bool(kernel.nvfp4_fc1_k256_overlap),
                            lut_replicas=int(conversion.lut_replicas),
                            lut_map=str(conversion.lut_map),
                            sf_stages=int(getattr(conversion, "sf_stages", 0)),
                        )
                    configuration = {
                        "requested_tactic": knobs,
                        "requested_config": options
                        if name.startswith("nv")
                        else {"compact_pull_buffer": config.compact_pull_buffer},
                        "effective_tactic": frontend.effective_tactic(),
                        "actual_layout": actual_layout,
                    }
                    rank_configurations = [None] * world
                    dist.all_gather_object(
                        rank_configurations, configuration, group=host_group
                    )
                    if any(item != configuration for item in rank_configurations):
                        raise RuntimeError(
                            "ranks resolved different benchmark configurations"
                        )
                    configurations[name] = configuration
                rounds = []
                bucket = {"rounds": rounds, "configuration": configurations}
                results["buckets"][str(tokens)] = bucket
                for round_index in range(args.rounds):
                    order = [name for name, _ in cases]
                    if round_index % 2:
                        order.reverse()
                    rows = {}
                    round_result = {"round": round_index, "order": order, "cases": rows}
                    rounds.append(round_result)
                    for name in order:
                        rows[name] = {"attempts": [], "wall_start_ns": time.time_ns()}
                        measure(
                            torch,
                            dist,
                            host_group,
                            layers[name],
                            outputs[name],
                            args,
                            cycles,
                            references[name],
                            rows[name],
                        )
                        rows[name]["wall_end_ns"] = time.time_ns()
                    ratios = {
                        "nv_over_mx": rows["nv"]["score_us"] / rows["mx"]["score_us"]
                    }
                    if "nv_reference" in rows:
                        ratios["nv_over_nv_reference"] = (
                            rows["nv"]["score_us"] / rows["nv_reference"]["score_us"]
                        )
                    round_result["paired_ratios"] = ratios
                bucket.update(
                    {
                        "latency_us": {
                            name: statistics.median(
                                r["cases"][name]["score_us"] for r in rounds
                            )
                            for name, _ in cases
                        },
                        "paired_ratios": {
                            key: {
                                "median": statistics.median(
                                    r["paired_ratios"][key] for r in rounds
                                ),
                                "range": [
                                    min(r["paired_ratios"][key] for r in rounds),
                                    max(r["paired_ratios"][key] for r in rounds),
                                ],
                            }
                            for key in rounds[0]["paired_ratios"]
                        },
                    }
                )
                if rank == 0:
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(json.dumps(results, indent=2) + "\n")
                    print(
                        f"T{tokens}: {results['buckets'][str(tokens)]['latency_us']}; paired={results['buckets'][str(tokens)]['paired_ratios']}",
                        flush=True,
                    )
            finally:
                for layer in reversed(list(layers.values())):
                    layer.destroy()
    finally:
        if rank == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(results, indent=2) + "\n")
        dist.destroy_process_group(host_group)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
