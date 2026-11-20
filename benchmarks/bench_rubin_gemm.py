# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark Rubin profiles with normal inputs, matched inputs, or both.

Run from the repository root with CUDA_VISIBLE_DEVICES selecting one GPU. Kernel configuration
comes from rubin_gemm_profiles.json. Normal mode uses the production input
generator; matched mode loads the archived arrays in benchmark code only.
Compare mode alternates the two workspaces on the same GPU. Each benchmark
first runs a small nonzero numerical reference check.
"""

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import socket
import statistics
import subprocess
import sys
import time

PROFILES = Path(__file__).with_name("rubin_gemm_profiles.json")
ROOT = Path(__file__).resolve().parents[1]


def physical_gpu():
    device = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not device.isdecimal():
        raise ValueError("Set CUDA_VISIBLE_DEVICES to one physical GPU index")
    return int(device)


def count_nonzero(tensor):
    # Some reductions promote uint8 inputs to int64. Bound the temporary
    # allocation when auditing multi-GiB weights with both backends resident.
    return sum(
        int(part.count_nonzero()) for part in tensor.reshape(-1).split(64 * 1024**2)
    )


def configuration(
    case,
    mode,
    *,
    use_shuffled_matrix=None,
    weight_layout=None,
    gated_weight_group_size=None,
    epilogue_regs=None,
    non_epilogue_regs=None,
    load_a_regs=None,
    load_sfa_regs=None,
):
    profiles = json.loads(PROFILES.read_text())
    shape = dict(profiles["cases"][case])
    x, y = shape.pop("cga")[mode]
    options = {
        **profiles["common_options"],
        **profiles["mode_options"][mode],
        "cluster_dim_x": x,
        "cluster_dim_y": y,
    }
    if use_shuffled_matrix is not None:
        options["use_shuffled_matrix"] = use_shuffled_matrix
    if weight_layout is not None:
        options["weight_layout"] = {"major_k": 0, "block_major_k": 2}[weight_layout]
    if gated_weight_group_size is not None:
        options["gated_weight_group_size"] = gated_weight_group_size
    if epilogue_regs is not None:
        options["epilogue_regs"] = epilogue_regs
    if non_epilogue_regs is not None:
        options.update(
            dict.fromkeys(
                (
                    "load_a_regs",
                    "load_b_regs",
                    "load_sfa_regs",
                    "load_sfb_regs",
                    "load_regs",
                    "load_sf_regs",
                    "copy_sf_regs",
                    "mma_regs",
                    "sync_regs",
                    "workid_regs",
                    "padding_regs",
                ),
                non_epilogue_regs,
            )
        )
    if load_a_regs is not None:
        options["load_a_regs"] = load_a_regs
    if load_sfa_regs is not None:
        options["load_sfa_regs"] = load_sfa_regs
    return shape, options


def runner_command(shape, options, seed):
    # Translate only the enums whose public CLI names differ from config keys.
    enums = {
        "batch_mode": {0: "n", 1: "m"},
        "route_act": {0: "none", 1: "tma", 2: "ldgsts"},
        "route_sfs_act": {0: "none", 1: "tma", 2: "ldgsts"},
        "tile_scheduler": {0: "static", 1: "persistent", 2: "static-persistent"},
        "cta_raster_order": {0: "rasterizeAlongM", 1: "rasterizeAlongN"},
        "weight_layout": {0: "major_k", 2: "block_major_k"},
        "act_kind": {0: "none", 1: "swiglu"},
    }
    for name in ("dtype_a", "dtype_b", "dtype_c"):
        enums[name] = {3: "bf16", 4: "e2m1"}
    for name in ("sf_layout_a", "sf_layout_b", "sf_layout_c"):
        enums[name] = {3: "128x4"}
    argv = [sys.executable, "-m", "flashinfer.prims_ts.batched_gemm.batched_gemm_run"]
    for name, value in {**shape, **options}.items():
        if name in enums:
            value = enums[name][value]
        flag = (
            "cta-swizzle-type" if name == "cta_raster_order" else name.replace("_", "-")
        )
        argv.extend(["--" + flag, str(value)])
    argv.extend(
        [
            "--routing",
            "balanced_round_robin",
            "--seed",
            str(seed),
            "--routing-seed",
            str(seed),
            "--dist",
            "default",
            "--scale-c",
            "auto",
            "--scale-gate",
            "1",
            "--benchmark",
            "--no-cuda-graphs",
            "--no-cold-l2",
        ]
    )
    return argv


def digest(tensor):
    import torch

    result = hashlib.sha256()
    for chunk in tensor.contiguous().view(torch.uint8).flatten().split(64 * 1024**2):
        result.update(memoryview(chunk.cpu().numpy()))
    return result.hexdigest()


def guarded_outputs(io):
    import cutlass
    from cutlass.cute.runtime import make_ptr
    import torch

    fc1 = io["cfg"].uses_fp4_output_quant
    size = io["logical_output_m"] * io["logical_output_n"]
    c_bytes = size // 2 if fc1 else size * 2
    buffers, guards = {}, []
    outputs = [
        ("C", "c0_dp", c_bytes, cutlass.Float4E2M1FN if fc1 else cutlass.BFloat16)
    ]
    if fc1:
        outputs.append(
            ("SFC", "sf_c_dp", io["sf_c_torch"].numel(), cutlass.Float8E4M3FN)
        )
    for name, pointer, count, dtype in outputs:
        storage = torch.full((count + 256,), 0xA5, dtype=torch.uint8, device="cuda")
        io[pointer] = make_ptr(
            dtype, storage.data_ptr(), cutlass.AddressSpace.gmem, assumed_align=16
        )
        io["_keepalive"].append(storage)
        buffers[name], guard = storage[:count], storage[count:]
        guards.append(guard)
        if name == "SFC":
            io["sf_c_torch"] = buffers[name]
    return buffers, guards


def check_launch(compiled, launch_args, buffers, guards, expected, *, zero):
    import torch

    compiled(*launch_args)
    torch.cuda.synchronize()
    assert all(bool(torch.all(g == 0xA5)) for g in guards), "Output guard overwritten"
    hashes = {name: digest(buf) for name, buf in buffers.items()}
    if expected is not None:
        assert hashes == expected, "Repeated launch changed output bytes"
    if not zero:
        assert int(torch.count_nonzero(buffers["C"])) > 0, "Unexpected all-zero output"
    return hashes


def measure(compiled, launch_args, stream, *, timer, warmups, iterations):
    import torch

    for _ in range(warmups):
        compiled(*launch_args)
    torch.cuda.synchronize()
    if timer == "cupti":
        from cutlass.testing import CuptiProfiler

        profiler = CuptiProfiler()
        profiler.start()
        try:
            for _ in range(iterations):
                compiled(*launch_args)
            torch.cuda.synchronize()
        finally:
            profiler.stop()
        durations = [ms * 1000 for _, ms in profiler.timings]
        assert len(durations) == iterations, (len(durations), iterations)
        return statistics.mean(durations), durations
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    start.record(stream)
    for _ in range(iterations):
        compiled(*launch_args)
    end.record(stream)
    end.synchronize()
    return start.elapsed_time(end) * 1000 / iterations, None


def build_candidate(
    kind, args, shape, options, stream, torch_stream, *, matched_inputs=None
):
    import torch
    from flashinfer.prims_ts.batched_gemm import batched_gemm_run as runner
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import compute_launch_grid

    matched = None
    if kind == "matched":
        from benchmarks.rubin_gemm_inputs import MatchedInputs

        matched = matched_inputs or MatchedInputs(
            args.fixture_dir, shape, fc1=args.mode == "fc1"
        )
        io = matched.build_io(shape, options)
    else:
        io = runner._build_launch_io(
            **shape,
            **options,
            seed=args.seed,
            routing_seed=args.seed,
            routing="balanced_round_robin",
            data_distribution="allZeros" if args.zero else "default",
        )
    valid_rows = sum(io["token_layout"].tokens_per_expert)
    assert valid_rows == io["total_padded_tokens"]
    owners = {t.data_ptr(): t for t in io["_keepalive"]}
    input_audit = {}
    for name in ("a_dp", "b_dp", "sfa_dp", "sfb_dp"):
        tensor = owners[io[name]._pointer]
        nonzero = count_nonzero(tensor)
        assert (nonzero == 0) == args.zero, name
        input_audit[name] = dict(
            bytes=tensor.numel() * tensor.element_size(), nonzero=nonzero
        )
        if matched is not None:
            input_audit[name]["sha256"] = digest(tensor)
    buffers, guards = guarded_outputs(io)
    torch_stream.wait_stream(torch.cuda.current_stream())
    compiled = runner._compile_for_launch(io, stream)
    launch_args = runner._launch_arg_tuple(io, stream)
    expected = None
    for _ in range(3):
        expected = check_launch(
            compiled, launch_args, buffers, guards, expected, zero=args.zero
        )
    matched_reference = None
    if matched is not None:
        matched_reference = matched.check(buffers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cubin = compiled.__cubin__
    if isinstance(cubin, str):
        cubin = Path(cubin).read_bytes()
    args.output.with_suffix(f".{kind}.cubin").write_bytes(cubin)
    metadata = dict(
        case=args.case,
        mode=args.mode,
        inputs=kind,
        shape=shape,
        options=options,
        resolved_config=io["cfg"].compile_signature(),
        warp_layout=asdict(io["cfg"].warp_layout),
        seed=args.seed if kind == "normal" else None,
        distribution=("allZeros" if args.zero else "default")
        if kind == "normal"
        else "saved matched fixture",
        routing="balanced_round_robin"
        if kind == "normal"
        else "saved matched route IDs",
        resolved_scale_c=io["resolved_scale_c"],
        host=socket.gethostname(),
        physical_gpu=physical_gpu(),
        gpu=str(torch.cuda.get_device_properties(0)),
        compiler=importlib.metadata.version("nvidia-cutlass-dsl"),
        compiler_options=os.environ["FLASHINFER_PRIMS_TS_COMPILE_OPTIONS"],
        input_audit=input_audit,
        small_numerical_check_passed=True,
        repeat_launches_checked=3,
        output_sha256=expected,
        cubin_sha256=hashlib.sha256(cubin).hexdigest(),
        launch_grid=compute_launch_grid(
            io["cfg"], problem_m=io["M"], problem_n=io["N"], early_exit_max_token_ctas=0
        ),
        timer=args.timer,
        warmups=args.warmups,
        iterations=args.iterations,
        timing="Median of window means; synchronized warmups; one workspace per input; no graphs; no explicit L2 flush",
        command=shlex.join(
            [sys.executable, "-m", "benchmarks.bench_rubin_gemm", *sys.argv[1:]]
        ),
        source_sha256={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__),
                PROFILES,
                Path(__file__).with_name("rubin_gemm_inputs.py"),
                *sorted((ROOT / "flashinfer/prims_ts/batched_gemm").glob("*.py")),
            ]
        },
        windows=[],
        complete=False,
    )
    if matched is not None:
        metadata.update(
            fixture=str(matched.path),
            fixture_manifest=matched.manifest,
            canonical_input_audit=matched.audit,
            matched_reference=matched_reference,
        )
    return dict(
        io=io,
        compiled=compiled,
        launch_args=launch_args,
        buffers=buffers,
        guards=guards,
        expected=expected,
        valid_rows=valid_rows,
        matched=matched,
        metadata=metadata,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("e32", "e256", "e384"), required=True)
    parser.add_argument(
        "--mode", choices=("fc1", "routed-bf16", "dense-bf16"), default="fc1"
    )
    parser.add_argument(
        "--output", type=Path, help="Fresh JSON path; existing results are preserved"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-shuffled-matrix", type=int, choices=(0, 1))
    parser.add_argument("--weight-layout", choices=("major_k", "block_major_k"))
    parser.add_argument("--gated-weight-group-size", type=int, choices=(1, 64))
    parser.add_argument("--epilogue-regs", type=int)
    parser.add_argument("--non-epilogue-regs", type=int)
    parser.add_argument("--load-a-regs", type=int)
    parser.add_argument("--load-sfa-regs", type=int)
    parser.add_argument(
        "--windows",
        type=int,
        default=None,
        help="Windows per input: default 5 single / 16 compare",
    )
    parser.add_argument(
        "--inputs", choices=("normal", "matched", "compare"), default="normal"
    )
    parser.add_argument(
        "--fixture-dir",
        type=Path,
        help="Matched manifest.json/A/B/SFA/SFB/route.npy directory; required for matched or compare inputs",
    )
    parser.add_argument(
        "--reverse-build-order",
        action="store_true",
        help="Build matched before normal in compare mode",
    )
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--timer", choices=("cupti", "events"), default="cupti")
    parser.add_argument(
        "--zero",
        action="store_true",
        help="Zero benchmark data/scales; numerical check stays nonzero",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Run only the small numerical correctness check",
    )
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument(
        "--print-runner-command",
        action="store_true",
        help="Print the equivalent production CLI (includes full-size reference check)",
    )
    args = parser.parse_args()
    shape, options = configuration(
        args.case,
        args.mode,
        use_shuffled_matrix=args.use_shuffled_matrix,
        weight_layout=args.weight_layout,
        gated_weight_group_size=args.gated_weight_group_size,
        epilogue_regs=args.epilogue_regs,
        non_epilogue_regs=args.non_epilogue_regs,
        load_a_regs=args.load_a_regs,
        load_sfa_regs=args.load_sfa_regs,
    )
    if args.windows is None:
        args.windows = 16 if args.inputs == "compare" else 5
    if args.zero and args.inputs != "normal":
        parser.error("--zero applies to normal inputs; matched fixtures stay unchanged")
    if args.print_runner_command and args.inputs != "normal":
        parser.error(
            "Matched inputs are benchmark-only: run this script with --inputs matched"
        )
    if args.print_config:
        print(json.dumps(dict(shape=shape, options=options), indent=2))
        return
    if args.print_runner_command:
        argv = runner_command(shape, options, args.seed)
        if args.zero:
            argv[argv.index("--dist") + 1] = "allZeros"
        print(shlex.join(argv))
        return
    if args.inputs != "normal" and args.fixture_dir is None:
        parser.error("--fixture-dir is required for matched or compare inputs")
    try:
        physical_gpu()
    except ValueError as error:
        parser.error(str(error))
    if min(args.windows, args.iterations) <= 0 or args.warmups < 0:
        parser.error("windows/iterations must be positive; warmups must be nonnegative")
    if not args.check_only and args.output is None:
        parser.error("--output is required for benchmarking")
    if args.output is not None and args.output.exists():
        parser.error(f"Preserving existing result: {args.output}")
    os.environ.setdefault("CUTE_DSL_ARCH", "sm_107a")
    os.environ.setdefault("FLASHINFER_PRIMS_TS_COMPILE_OPTIONS", "--opt-level 2")
    os.environ.setdefault("FLASHINFER_PRIMS_TS_DEBUG_CHECKS", "0")
    os.environ.setdefault("CUTE_DSL_KEEP", "ptx,cubin")

    import torch
    import cuda.bindings.driver as cuda
    from flashinfer.prims_ts.batched_gemm import batched_gemm_run as runner

    if torch.cuda.get_device_capability() != (10, 7):
        parser.error("These profiles require a Rubin SM107 GPU")
    # Fail before allocating large inputs if the explicitly selected timer is unavailable.
    if args.timer == "cupti" and not args.check_only:
        from cutlass.testing import CuptiProfiler

        CuptiProfiler()
    small = dict(
        num_experts=8,
        num_tokens=1031,
        top_k=8,
        problem_n=1280,
        problem_k=2304,
        routing="uniform",
        seed=args.seed,
        routing_seed=args.seed,
    )
    assert runner.reference_check(
        **small,
        repeat_launches=3,
        output_guard_elements=256,
        require_mixed_cga_runtime=True,
        **options,
    )
    if args.check_only:
        print("PASS", args.case, args.mode, "small nonzero correctness")
        return
    kinds = ["normal", "matched"] if args.inputs == "compare" else [args.inputs]
    if args.reverse_build_order:
        kinds.reverse()
    torch_stream = torch.cuda.Stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)
    candidates = {}
    for kind in kinds:
        candidates[kind] = build_candidate(
            kind, args, shape, options, stream, torch_stream
        )
        print("READY", args.case, args.mode, kind, flush=True)
    binaries = {c["metadata"]["cubin_sha256"] for c in candidates.values()}
    assert len(binaries) == 1, "Input comparison must use byte-identical kernels"
    telemetry = []
    for suffix, query in (
        (
            "clocks",
            "--query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu",
        ),
        (
            "processes",
            "--query-compute-apps=timestamp,gpu_uuid,pid,process_name,used_gpu_memory",
        ),
    ):
        log = args.output.with_suffix(f".{suffix}.csv").open("w")
        proc = subprocess.Popen(
            [
                "nvidia-smi",
                "-i",
                str(physical_gpu()),
                query,
                "--format=csv",
                "--loop-ms=100",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        telemetry.append((proc, log))
    sequence = []
    try:
        for index in range(args.windows):
            order = kinds if index % 2 == 0 else list(reversed(kinds))
            for kind in order:
                candidate = candidates[kind]
                before = time.time()
                us, durations = measure(
                    candidate["compiled"],
                    candidate["launch_args"],
                    torch_stream,
                    timer=args.timer,
                    warmups=args.warmups,
                    iterations=args.iterations,
                )
                window = dict(
                    index=index,
                    inputs=kind,
                    us=us,
                    durations_us=durations,
                    start=before,
                    end=time.time(),
                )
                candidate["metadata"]["windows"].append(window)
                sequence.append(window)
                partial = dict(
                    case=args.case,
                    mode=args.mode,
                    inputs=args.inputs,
                    candidates={k: c["metadata"] for k, c in candidates.items()},
                    sequence=sequence,
                    complete=False,
                )
                args.output.with_suffix(".partial.json").write_text(
                    json.dumps(partial, indent=2) + "\n"
                )
                print("WINDOW", index, kind, us, flush=True)
    finally:
        for proc, log in telemetry:
            proc.terminate()
            proc.wait(timeout=10)
            log.close()
    for kind, candidate in candidates.items():
        check_launch(
            candidate["compiled"],
            candidate["launch_args"],
            candidate["buffers"],
            candidate["guards"],
            candidate["expected"],
            zero=args.zero,
        )
        if candidate["matched"] is not None:
            candidate["metadata"]["post_matched_reference"] = candidate[
                "matched"
            ].check(candidate["buffers"])
        metadata = candidate["metadata"]
        us = statistics.median(w["us"] for w in metadata["windows"])
        metadata.update(
            us=us,
            tflops=2
            * candidate["valid_rows"]
            * shape["problem_n"]
            * shape["problem_k"]
            / (us * 1e6),
            complete=True,
            post_benchmark_guards_and_hashes_passed=True,
        )
        print(
            "RESULT",
            args.case,
            args.mode,
            kind,
            us,
            "us",
            metadata["tflops"],
            "TFLOPS",
            flush=True,
        )
    if args.inputs == "compare":
        normal, matched = (candidates[k]["metadata"] for k in ("normal", "matched"))
        paired_ratios = [
            n["us"] / m["us"]
            for n, m in zip(normal["windows"], matched["windows"], strict=True)
        ]
        result = dict(
            case=args.case,
            mode=args.mode,
            inputs="compare",
            host=socket.gethostname(),
            physical_gpu=physical_gpu(),
            candidates={k: c["metadata"] for k, c in candidates.items()},
            sequence=sequence,
            cubins_identical=True,
            build_order=kinds,
            normal_over_matched=normal["us"] / matched["us"],
            paired_normal_over_matched=paired_ratios,
            median_paired_normal_over_matched=statistics.median(paired_ratios),
            complete=True,
        )
    else:
        result = candidates[args.inputs]["metadata"]
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
