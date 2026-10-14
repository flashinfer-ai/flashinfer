"""Graph-replay evidence for safetensors IQ2_XS and NVFP4 A16 dense layers.

The public entry point is benchmark_dense_gemm.py --dtype checkpoint-a16.
One checkpoint tensor represents each dense role and logical geometry; routed
expert weights are excluded. Packing and independent oracle decoding occur
outside timing. The timed operation uses the production prepared dense API.
"""

from __future__ import annotations

from b12x._lib.quant.block_codec import BLOCK_CODECS, block_codec

from collections import defaultdict
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys

import torch
from safetensors import safe_open

from b12x._lib.intrinsics import swizzle_block_scale
from b12x._lib.runtime_control import kernel_resolution_guard
from b12x.gemm import blockscaled
from b12x.preparation import PreparationSession, PreparedCall, require_prepared
from b12x.testing.iq2_xs_reference import dequantize_blocks
from benchmarks.experimental.b12x.common import (
    make_l2_flush_fn,
    nvidia_smi_gpu_mode_snapshot,
)


def checkpoint_cases(model: Path):
    index = json.loads((model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    recipes = json.loads((model / "hf_quant_config.json").read_text())["quantization"][
        "quantized_layers"
    ]
    groups = defaultdict(list)
    for name in sorted(index):
        if not name.endswith(".weight") or ".experts." in name:
            continue
        recipe = recipes.get(name.removesuffix(".weight"), {})
        algorithm = recipe.get("quant_algo")
        if algorithm not in ("IQ2_XS", "IQ2_XXS", "Q8_0", "W4A16_NVFP4"):
            continue
        with safe_open(model / index[name], framework="pt", device="cpu") as handle:
            shape = tuple(handle.get_slice(name).get_shape())
        if algorithm.lower() in BLOCK_CODECS:
            spec = block_codec(algorithm.lower())
            if (
                recipe.get("packing") != "ggml"
                or recipe.get("group_size") != spec.block_weights
                or recipe.get("block_payload_bytes") != spec.block_bytes
                or len(shape) != 3
                or shape[-1] != spec.block_bytes
            ):
                raise ValueError(f"unsupported IQ2_XS payload: {name}")
            n, blocks, _ = shape
            k = blocks * spec.block_weights
        else:
            if recipe.get("group_size") != 16 or len(shape) != 2:
                raise ValueError(f"unsupported NVFP4 payload: {name}")
            n, stored_k = shape
            k = stored_k * 2
        role = re.sub(r"layers\.\d+", "layers.*", name)
        groups[role, algorithm, n, k].append(name)
    if not groups:
        raise ValueError("checkpoint contains no supported quantized dense layers")
    return index, [
        dict(
            role=role,
            recipe=algo.lower() if algo.lower() in BLOCK_CODECS else "nvfp4",
            n=n,
            k=k,
            weight=names[0],
            equivalent_weights=names,
        )
        for (role, algo, n, k), names in sorted(groups.items())
    ]


def load_weight(model, index, case):
    def read(name):
        with safe_open(model / index[name], framework="pt", device="cpu") as handle:
            return handle.get_tensor(name)

    name = case["weight"]
    raw = read(name)
    hashes = {name: hashlib.sha256(raw.view(torch.uint8).numpy().tobytes()).hexdigest()}
    if case["recipe"] in BLOCK_CODECS:
        weight = blockscaled.pack_weight(raw.cuda(), recipe=case["recipe"])
        decoded = dequantize_blocks(raw).bfloat16().cuda()
        return weight, decoded, 1.0, hashes
    scale_name = name.removesuffix(".weight") + ".weight_scale"
    global_name = name.removesuffix(".weight") + ".weight_scale_2"
    scales, multiplier = read(scale_name), read(global_name).reshape(1)
    for key, tensor in ((scale_name, scales), (global_name, multiplier)):
        hashes[key] = hashlib.sha256(
            tensor.view(torch.uint8).numpy().tobytes()
        ).hexdigest()
    lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6])
    codes = (
        torch.stack((raw & 15, raw >> 4), dim=-1).reshape(case["n"], case["k"]).long()
    )
    decoded = (
        (lut[codes] * scales.float().repeat_interleave(16, dim=1)).bfloat16().cuda()
    )
    multiplier = multiplier.float().cuda()
    weight = blockscaled.pack_weight(
        raw.cuda(),
        swizzle_block_scale(scales.cuda()),
        recipe="nvfp4",
        global_scale=multiplier,
    )
    return weight, decoded, multiplier, hashes


def oracle(source, decoded, multiplier):
    expected = torch.empty(
        (source.shape[0], decoded.shape[0]), device=source.device, dtype=torch.float32
    )
    for start in range(0, decoded.shape[0], 1024):
        expected[:, start : start + 1024] = (
            source.float() @ decoded[start : start + 1024].float().T
        )
    return expected * multiplier


def check_output(actual, expected):
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise AssertionError("checkpoint dense output must be finite")
    if not torch.count_nonzero(actual) or not torch.count_nonzero(expected):
        raise AssertionError("checkpoint dense output must be nonzero")
    actual = actual.float()
    cosine = float(
        torch.nn.functional.cosine_similarity(
            actual.flatten(), expected.flatten(), dim=0
        )
    )
    relative_l2 = float((actual - expected).norm() / expected.norm())
    if cosine < 0.99998 or relative_l2 > 0.004:
        raise AssertionError(
            f"checkpoint dense mismatch: cosine={cosine}, relative_l2={relative_l2}"
        )
    return dict(cosine=cosine, relative_l2=relative_l2)


def source_manifest():
    root = Path(__file__).resolve().parents[3]
    paths = [
        *sorted((root / "flashinfer/experimental/b12x/gemm/blockscaled").glob("*.py")),
        *sorted((root / "flashinfer/experimental/b12x/preparation").glob("*.py")),
        root / "flashinfer/experimental/b12x/_lib/dense_gemm.py",
        root / "flashinfer/experimental/b12x/_lib/intrinsics.py",
        root / "flashinfer/experimental/b12x/_lib/quant/iq2_xs.py",
        root / "flashinfer/experimental/b12x/_lib/quant/block_codec.py",
        root / "flashinfer/experimental/b12x/testing/iq2_xxs_reference.py",
        root / "flashinfer/experimental/b12x/testing/iq2_xs_reference.py",
        root / "benchmarks/experimental/b12x/benchmark_dense_gemm.py",
        Path(__file__).resolve(),
    ]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True
    )
    return dict(
        revision=revision.stdout.strip() if revision.returncode == 0 else None,
        source_sha256={
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths
        },
    )


def run(args):
    from benchmarks.experimental.b12x.benchmark_dense_gemm import bench_events

    if args.model_path is None or args.evidence is None:
        raise ValueError("checkpoint-a16 requires --model-path and --evidence")
    if not args.check or not args.flush_l2 or args.warmup < 20 or args.iters < 100:
        raise ValueError(
            "checkpoint evidence requires correctness, cold L2, 20 warmups and 100 trials"
        )
    if (
        args.n is not None
        or args.k is not None
        or args.profile != "default"
        or args.tune_a16
    ):
        raise ValueError(
            "checkpoint shapes and prepared startup selection own checkpoint-a16 dispatch"
        )
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (12, 0),
        (12, 1),
    ):
        raise RuntimeError("checkpoint-a16 requires SM120/SM121")
    counts = args.batch_sizes or [1, 2, 4, 8, 16, 512]
    if min(counts) <= 0 or len(set(counts)) != len(counts):
        raise ValueError("batch sizes must be distinct positive counts")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = args.model_path.resolve()
    index, cases = checkpoint_cases(model)
    if args.checkpoint_recipe is not None:
        cases = [case for case in cases if case["recipe"] == args.checkpoint_recipe]
        if not cases:
            raise ValueError("checkpoint has no dense weights for the requested recipe")
    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    flush = make_l2_flush_fn(enabled=True, bytes_hint=args.l2_flush_bytes)
    timings = []
    with args.evidence.open("x") as evidence:

        def record(row):
            evidence.write(json.dumps(row) + "\n")
            evidence.flush()

        record(
            dict(
                kind="manifest",
                command=sys.argv,
                worktree=str(Path.cwd()),
                model=str(model),
                **source_manifest(),
                device=nvidia_smi_gpu_mode_snapshot(),
                torch=torch.__version__,
                cutlass=importlib.metadata.version("nvidia-cutlass-dsl"),
                triton=importlib.metadata.version("triton"),
                counts=counts,
                tuning_cache_version=os.environ.get("B12X_TUNING_CACHE_VERSION", "1"),
                cases=cases,
                model_files={
                    name: hashlib.sha256((model / name).read_bytes()).hexdigest()
                    for name in (
                        "config.json",
                        "model.safetensors.index.json",
                        "hf_quant_config.json",
                    )
                },
                metric="cold-L2 CUDA graph replay microseconds; minimize",
                oracle="FP32 matmul of independently decoded BF16 weights; weight-only global scale",
            )
        )
        for case in cases:
            weight, decoded, multiplier, hashes = load_weight(model, index, case)
            values = weight.values
            scales = (
                weight.metadata if case["recipe"] in BLOCK_CODECS else weight.scale_mma
            )
            global_scale = (
                None if case["recipe"] in BLOCK_CODECS else weight.global_scale
            )
            for m in counts:
                source = torch.empty(
                    (m, case["k"]), device="cuda", dtype=torch.bfloat16
                )
                query = blockscaled.BlockscaledQuery(
                    recipe=case["recipe"],
                    num_tokens=m,
                    in_features=case["k"],
                    padded_in_features=case["k"],
                    out_features=case["n"],
                    activation_mode="a16",
                    workspace_form="provided",
                    workspace_nbytes=2_000_000_000,
                    expected_m=m,
                )
                plan = blockscaled.plan(query)

                def prepare(state):
                    scratch = (
                        torch.empty(
                            state.required_workspace, device="cuda", dtype=torch.uint8
                        )
                        if state.required_workspace
                        else None
                    )
                    return PreparedCall(
                        run=lambda: state.run(
                            source, values, scales, global_scale, workspace=scratch
                        ),
                        produce=lambda: source.fill_(0.125),
                        owners=(values, scales),
                        capture_safe=False,
                    )

                with PreparationSession(
                    device=source.device, autotune=True, compile_workers=1
                ) as session:
                    session.prepare(
                        (
                            plan.request(
                                name=case["weight"],
                                prepare_call=prepare,
                                benchmark_call=prepare,
                            ),
                        )
                    )
                    session.freeze()
                    state = require_prepared(
                        plan, "gemm.blockscaled_precision", source.device
                    )
                    scratch = (
                        torch.empty(
                            state.required_workspace, device="cuda", dtype=torch.uint8
                        )
                        if state.required_workspace
                        else None
                    )
                    torch.manual_seed(42 + m)
                    source.normal_(std=0.25)
                    expected = oracle(source, decoded, multiplier)

                    def launch(weight=weight):
                        return blockscaled.mm(
                            source, weight, plan=plan, workspace=scratch
                        )

                    with kernel_resolution_guard("checkpoint dense frozen replay"):
                        eager = check_output(launch(), expected)
                        torch.cuda.synchronize()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            output = launch()
                        addresses = (
                            source.data_ptr(),
                            values.data_ptr(),
                            scales.data_ptr(),
                            output.data_ptr(),
                            scratch.data_ptr() if scratch is not None else None,
                        )
                        program = state.programs["gemm"]
                        source.neg_()
                        output.fill_(float("nan"))
                        if scratch is not None:
                            scratch.fill_(255)
                        allocated = torch.cuda.memory_allocated()
                        graph.replay()
                        torch.cuda.synchronize()
                        if allocated != torch.cuda.memory_allocated():
                            raise AssertionError("graph replay changed live allocation")
                        replay = check_output(output, -expected)
                        for _ in range(args.warmup):
                            flush()
                            graph.replay()
                        torch.cuda.synchronize()
                        before = nvidia_smi_gpu_mode_snapshot()
                        samples = [
                            ms * 1000
                            for ms in bench_events(
                                graph.replay,
                                warmup=args.warmup,
                                iters=args.iters,
                                l2_flush=flush,
                            )
                        ]
                        after = nvidia_smi_gpu_mode_snapshot()
                        check_output(output, -expected)
                        assert state.programs["gemm"] is program
                        assert addresses == (
                            source.data_ptr(),
                            values.data_ptr(),
                            scales.data_ptr(),
                            output.data_ptr(),
                            scratch.data_ptr() if scratch is not None else None,
                        )
                        if args.profile_graphs:
                            flush()
                            torch.cuda.synchronize()
                            torch.cuda.profiler.start()
                            try:
                                graph.replay()
                                torch.cuda.synchronize()
                            finally:
                                torch.cuda.profiler.stop()
                            check_output(output, -expected)
                    latency = statistics.median(samples)
                    timings.append(latency)
                    record(
                        dict(
                            kind="case",
                            **case,
                            rows=m,
                            payload_sha256=hashes,
                            config=state.config.to_dict(),
                            selection_source=plan.selection.source,
                            packed_weight_bytes=(
                                values.numel() * values.element_size()
                                + scales.numel() * scales.element_size()
                            ),
                            query=vars(query) | {"codegen": dict(query.codegen)},
                            eager=eager,
                            changed_input_replay=replay,
                            stable_addresses=True,
                            stable_callable=True,
                            replay_allocation_delta=0,
                            samples_us=samples,
                            median_us=latency,
                            device_before=before,
                            device_after=after,
                        )
                    )
                    print(
                        f"  {case['recipe']} {case['role']} M={m} N={case['n']} K={case['k']} "
                        f"b12x {latency:.3f} us check cosine={replay['cosine']:.8f} "
                        f"relative_l2={replay['relative_l2']:.8f} config={state.config.to_dict()}",
                        flush=True,
                    )
                    del graph, output
            del weight, decoded
        geomean = math.exp(statistics.mean(math.log(t) for t in timings))
        record(
            dict(
                kind="result",
                cases=len(timings),
                geomean_us=geomean,
                correctness="passed",
            )
        )
        print(
            f"  geo mean: {geomean:.3f} us over {len(timings)} cases (minimize)",
            flush=True,
        )
