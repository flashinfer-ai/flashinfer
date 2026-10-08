# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""Benchmark and tune cuDNN Frost BF16 MoE kernels on the target GPU.

Use ``benchmark`` to compare the original eligible pool (CUTLASS/TRTLLM on
SM107, CUTLASS/cuTile on SM120) with the same pool plus cuDNN Frost, timing
complete routed MoE calls with alternating CUDA graph replay. The original
backends retain their unmodified tactic pools, and cuDNN Frost-only timing is
reported even when it loses cross-backend selection.

Use ``export``, ``sweep`` and ``select`` for offline kernel tuning. FC1 and FC2
normal/swap-AB candidates are shortlisted independently, then ranked as complete
MoE pairs. SM120 exports the warp-MMA template's normal STG kernels only. Export
and sweep need an idle target GPU; only export needs the cuDNN Frost compiler.
Selection copies validated winners to an explicit destination; ``shortlist``
instead writes each profile's two FC1 and two FC2 stage finalists.
"""

import argparse
import gc
import hashlib
import itertools
import json
import multiprocessing
import shutil
import statistics
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from functools import partial
from pathlib import Path

import torch

from flashinfer.fused_moe.backends.cudnn_frost.activations import (
    ACTIVATIONS,
)

if __package__:
    from .bench_cudnn_frost_common import (
        activation_reference,
        capture,
        check_idle_gpu,
        emit,
        error,
        measure,
    )
else:
    from bench_cudnn_frost_common import (
        activation_reference,
        capture,
        check_idle_gpu,
        emit,
        error,
        measure,
    )


def configure_source_jit():
    from flashinfer.jit import env

    root = Path(__file__).resolve().parents[1]
    env.FLASHINFER_CSRC_DIR = root / "csrc"
    env.FLASHINFER_INCLUDE_DIR = root / "include"
    cccl = root / "3rdparty/cccl"
    env.CCCL_INCLUDE_DIRS = [
        cccl / "cub",
        cccl / "libcudacxx/include",
        cccl / "thrust",
    ]


def benchmark(args):
    gpu_uuid = torch.cuda.get_device_properties(torch.cuda.current_device()).uuid
    check_idle_gpu(gpu_uuid)
    torch.manual_seed(args.seed)
    from flashinfer.autotuner import autotune
    from flashinfer.fused_moe import (
        BackendOptions,
        CuTileBf16Config,
        CutlassBf16Config,
        ExecutionConfig,
        ExpertConfig,
        MoEActivationPack,
        MoEConfig,
        MoELayer,
        MoEWeightPack,
        QuantConfig,
        RoutingConfig,
        TrtllmBf16Config,
    )
    from flashinfer.fused_moe.prepare import prepare_trtllm_bf16_weights

    torch.manual_seed(41)
    e, h, i = args.experts, args.hidden, args.intermediate
    if args.artifact_root is not None:
        from flashinfer.fused_moe.backends.cudnn_frost.bf16 import (
            fc2,
            runtime,
        )

        # Use one explicit pool, so re-testing an export after packaging it does
        # not produce duplicate artifact ids. Neither runtime module is changed
        # on disk, and normal application processes keep their packaged roots.
        roots = (args.artifact_root.resolve(),)
        runtime._artifact_roots = lambda: roots
        fc2._artifact_roots = lambda: roots
    tokens_list = [int(s) for s in args.tokens.split(",")]
    activation = ACTIVATIONS[args.activation]()
    from flashinfer.fused_moe.layer import _BACKEND_RUNNERS

    major, minor = torch.cuda.get_device_capability()
    arch = major * 10 + minor
    eligible = {
        cls: runner
        for cls, runner in _BACKEND_RUNNERS.items()
        if not runner.backend_key.startswith("cudnn_frost_")
        and cls.supported(arch)
        and runner.supports_quant(QuantConfig())
        and isinstance(activation, runner.supported_activation_classes)
    }
    # Fail visibly if a new applicable runner needs a prepared weight view;
    # silently omitting it would invalidate the all-backend comparison.
    unsupported_views = set(eligible) - {
        CutlassBf16Config,
        CuTileBf16Config,
        TrtllmBf16Config,
    }
    if unsupported_views:
        raise RuntimeError(
            "Add benchmark weight preparation for all applicable backends: "
            + ", ".join(sorted(cls.__name__ for cls in unsupported_views))
        )
    backend_configs = tuple(cls() for cls in eligible)
    expected_backends = {runner.backend_key for runner in eligible.values()}
    config = MoEConfig(
        routing=RoutingConfig(num_experts=e, top_k=args.top_k),
        quant=QuantConfig(),
        experts=ExpertConfig(intermediate_size=i),
        backend=BackendOptions(backend_configs),
        activation=activation,
        execution=ExecutionConfig(tune_max_num_tokens=max(tokens_list)),
    )
    w1 = (
        torch.randn(
            e,
            (2 if activation.is_gated else 1) * i,
            h,
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 0.02
    )
    w2 = torch.randn(e, h, i, device="cuda", dtype=torch.bfloat16) * 0.02
    views = {"cutlass_bf16": dict(fc1_expert_weights=w1, fc2_expert_weights=w2)}
    if CuTileBf16Config in eligible:
        views["cutile_bf16"] = CuTileBf16Config.prepare_weights(
            w1,
            w2,
            num_local_experts=e,
            hidden_size=h,
            intermediate_size=i,
            activation=activation,
        )
    if TrtllmBf16Config in eligible:
        views["trtllm_bf16_routed"] = prepare_trtllm_bf16_weights(
            w1,
            w2,
            num_local_experts=e,
            hidden_size=h,
            intermediate_size=i,
            activation=activation,
        )
    weights = MoEWeightPack(views)
    print(
        json.dumps(
            dict(
                gpu=torch.cuda.get_device_name(),
                activation=args.activation,
                seed=args.seed,
                experts=e,
                hidden=h,
                intermediate=i,
                top_k=args.top_k,
                routing=args.routing,
                mode="cuda_graph",
                torch=torch.__version__,
                rounds=args.rounds,
                iterations=args.iterations,
                graph_batch=args.graph_batch,
                probe_cudnn_frost=args.probe_cudnn_frost,
                artifact_root=str(args.artifact_root) if args.artifact_root else None,
            )
        ),
        flush=True,
    )
    for tokens in tokens_list:
        x = torch.randn(tokens, h, device="cuda", dtype=torch.bfloat16) * 0.02
        logits = torch.rand(tokens, e, device="cuda")
        if args.routing == "skew":
            logits[: tokens // 2, 0] += 2
        ids = logits.topk(args.top_k, dim=1).indices.int()
        scores = torch.rand(tokens, args.top_k, device="cuda").softmax(-1)
        act = MoEActivationPack(x, None, ids, scores)
        layers = {"original": MoELayer(config), "with_cudnn_frost": MoELayer(config)}
        for label, layer in layers.items():
            actual_backends = {runner.backend_key for runner in layer.runners}
            if actual_backends != expected_backends:
                raise RuntimeError(
                    f"{label} backend pool mismatch: expected {expected_backends}, "
                    f"got {actual_backends}"
                )
        print(
            json.dumps(
                dict(tokens=tokens, baseline_backends=sorted(expected_backends))
            ),
            flush=True,
        )
        # Benchmark-only dispatcher ablation, never a change to an existing runner.
        layers["original"]._additional_candidates = lambda *args: []
        if args.probe_cudnn_frost:
            from flashinfer.fused_moe.backends.cudnn_frost.bf16.moe import (
                CudnnFrostBf16MoeRunner,
            )

            probe = CudnnFrostBf16MoeRunner(config, x.device)
            probe.check_support()
            probe.build()
            layers["with_cudnn_frost"]._automatic_runners["cudnn_frost_bf16"] = probe
            layers["with_cudnn_frost"]._additional_candidates = lambda *args: [probe]
        outputs, graphs, winners = {}, {}, {}
        for label, layer in layers.items():
            with autotune():
                layer(act, weights)
            check_idle_gpu(gpu_uuid)
            outputs[label] = layer(act, weights).clone()
            winners[label] = [(r.backend_key, t) for r, t in layer._winners.values()]
            for _ in range(5):
                layer(act, weights)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(args.graph_batch):
                    layer(act, weights)
            graphs[label] = graph
            print(
                json.dumps(
                    dict(tokens=tokens, pool=label, winners=winners[label]),
                    default=list,  # TRTLLM may return a TVM-FFI Array tactic.
                ),
                flush=True,
            )

        # Measure the best independent cuDNN Frost candidate even if an original backend wins.
        layer = layers["with_cudnn_frost"]
        cudnn_frost = layer._automatic_runners.get("cudnn_frost_bf16")
        if cudnn_frost is not None and (
            args.probe_cudnn_frost or cudnn_frost.accepts(act, weights)
        ):
            packed = cudnn_frost.pack_inputs(act, weights)
            _, tactic = layer.tuner.choose_one(
                custom_op="moe_cudnn_frost_bf16",
                runners=[cudnn_frost],
                inputs=packed,
                tuning_config=cudnn_frost.tuning_config_for(packed),
                **cudnn_frost.launch_kwargs_for(packed),
            )
            outputs["cudnn_frost_only"] = cudnn_frost.forward(packed, tactic).clone()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(args.graph_batch):
                    cudnn_frost.forward(packed, tactic)
            graphs["cudnn_frost_only"] = graph
            print(
                json.dumps(
                    dict(
                        tokens=tokens,
                        pool="cudnn_frost_only",
                        tactic=tactic,
                        candidate_count=len(
                            cudnn_frost.get_valid_tactics(packed, None)
                        ),
                        workspace_bytes=packed.launch_state.workspace.numel(),
                    )
                ),
                flush=True,
            )
        errors = {
            label: (
                (out.float() - outputs["original"].float()).norm()
                / outputs["original"].float().norm()
            ).item()
            for label, out in outputs.items()
            if label != "original"
        }
        if not all(error < 0.015 for error in errors.values()):
            raise RuntimeError(f"MoE backend correctness check failed: {errors}")
        samples = {label: [] for label in graphs}
        for round_idx in range(args.rounds):
            check_idle_gpu(gpu_uuid)
            labels = list(graphs)
            offset = round_idx % len(labels)
            labels = labels[offset:] + labels[:offset]
            if round_idx % 2:
                labels.reverse()
            for label in labels:
                graphs[label].replay()
                start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                start.record()
                for _ in range(args.iterations):
                    graphs[label].replay()
                end.record()
                end.synchronize()
                samples[label].append(
                    start.elapsed_time(end) / (args.iterations * args.graph_batch)
                )
        check_idle_gpu(gpu_uuid)
        medians = {label: statistics.median(times) for label, times in samples.items()}
        # Pools run back to back within each rotated round, so per-round ratios
        # also cancel clock drift across rounds (e.g. power capping when SM
        # clocks are not locked), which the medians of whole runs retain.
        paired = statistics.median(
            original / frost
            for original, frost in zip(
                samples["original"], samples["with_cudnn_frost"], strict=True
            )
        )
        print(
            json.dumps(
                dict(
                    tokens=tokens,
                    routed_rows=tokens * args.top_k,
                    milliseconds=medians,
                    speedup=medians["original"] / medians["with_cudnn_frost"],
                    paired_speedup=paired,
                    relative_l2=errors,
                    samples_ms=samples,
                )
            ),
            flush=True,
        )
        if args.sweep_cudnn_frost and cudnn_frost is not None:
            packed = cudnn_frost.pack_inputs(act, weights)
            tactics = cudnn_frost.get_valid_tactics(packed, None)
            sweep_graphs = []
            for tactic in tactics:
                actual = cudnn_frost.forward(packed, tactic)
                error = (
                    (actual.float() - outputs["original"].float()).norm()
                    / outputs["original"].float().norm()
                ).item()
                if error >= 0.015 or not torch.isfinite(actual).all().item():
                    raise RuntimeError(
                        f"cuDNN Frost tactic failed correctness: {tactic}: {error}"
                    )
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(8):
                        cudnn_frost.forward(packed, tactic)
                actual.fill_(float("nan"))
                graph.replay()
                error = (
                    (actual.float() - outputs["original"].float()).norm()
                    / outputs["original"].float().norm()
                ).item()
                if not error < 0.015:
                    raise RuntimeError(
                        f"cuDNN Frost graph tactic failed correctness: {tactic}: {error}"
                    )
                sweep_graphs.append(graph)
            times = [[] for _ in tactics]
            for round_idx in range(5):
                check_idle_gpu(gpu_uuid)
                order = list(range(len(tactics)))
                shift = (round_idx * 3) % len(order)
                order = order[shift:] + order[:shift]
                if round_idx % 2:
                    order.reverse()
                for idx in order:
                    graph = sweep_graphs[idx]
                    graph.replay()
                    start, end = (
                        torch.cuda.Event(enable_timing=True) for _ in range(2)
                    )
                    start.record()
                    for _ in range(5):
                        graph.replay()
                    end.record()
                    end.synchronize()
                    times[idx].append(start.elapsed_time(end) / 40)
            check_idle_gpu(gpu_uuid)
            ranked = sorted(
                (
                    dict(milliseconds=statistics.median(t), tactic=tactic)
                    for t, tactic in zip(times, tactics, strict=True)
                ),
                key=lambda x: x["milliseconds"],
            )
            print(json.dumps(dict(tokens=tokens, cudnn_frost_sweep=ranked)), flush=True)


# SM120 warp-MMA tiles: CTA M/N, K bytes and the compute-warp grid M x N. The
# template has no swap-AB orientation, CTA pair, cluster or TMA-store epilogue.
_SM120_SMALL_TILES = ((32, 64, 64, 2, 4), (64, 64, 128, 4, 2), (128, 128, 32, 1, 8))
_SM120_TILES = (
    *((32, n, k, 2, 4) for n, k in ((64, 64), (64, 128), (128, 32), (128, 64))),
    *((32, n, k, 1, 8) for n, k in ((128, 64), (256, 32), (256, 64))),
    (32, 256, 32, 2, 4),
    *((64, 32, k, 4, 2) for k in (64, 128)),
    *((64, 64, 64, w, 8 // w) for w in (2, 4)),
    (64, 64, 128, 4, 2),
    *((64, 128, k, 2, 4) for k in (32, 64)),
    (64, 128, 64, 4, 2),
    *((64, 256, k, 2, 4) for k in (32, 64)),
    *((128, 32, 64, w, 8 // w) for w in (4, 8)),
    *((128, 64, k, 4, 2) for k in (32, 64)),
    (128, 64, 64, 8, 1),
    *((128, 128, k, w, 8 // w) for k in (32, 64) for w in (2, 4)),
    *((128, 256, k, 2, 4) for k in (32, 64)),
    *((256, 64, k, 4, 2) for k in (32, 64)),
    (256, 64, 64, 8, 1),
    *((256, 128, k, 4, 2) for k in (32, 64)),
)


def sm120_configs(small=False):
    for m, n, k_bytes, warps_m, warps_n in (
        _SM120_SMALL_TILES if small else _SM120_TILES
    ):
        name = (
            f"CONFIG_sm120_{m}x{n}x{k_bytes}_16x16x32_"
            f"cluster1x1_warps{warps_m}x{warps_n}"
        )
        yield name, 1, "stg"


def store_modes(config):
    """SM120 always stores through STG; SM100 exports both store variants."""
    return ("stg",) if config.pipeline == "sm120" else ("stg", "tma")


def mma_size_m(tile):
    """MMA-M multiplicity of one CTA (SM100) or one compute warp (SM120)."""
    parts = tile.split("_")
    m, mma_m = int(parts[2].split("x")[0]), int(parts[3].split("x")[0])
    warps = next((p for p in parts if p.startswith("warps")), None)
    return m // mma_m if warps is None else m // int(warps[5:].split("x")[0]) // mma_m


def configs(small=False, wide_swap=False, arch=(10, 7)):
    if arch[0] == 12:
        yield from sm120_configs(small)
        return
    # CTA M/N, MMA M, CTA group, cluster N. K is 128 bytes.
    normal = [(128, 128, 128, 1, 1)]
    swapped = [(128, 16, 128, 1, 1)]
    if not small:
        normal += [
            (64, 128, 64, 1, 1),
            (128, 64, 128, 1, 1),
            (128, 256, 128, 2, 1),
            (128, 256, 128, 2, 2),
            (256, 128, 128, 1, 1),
        ]
        swapped = [
            (m, n, m, cta, 1)
            for m in (64, 128)
            for n in (8, 16, 32, 64)
            for cta in (1, 2)
            if not (cta == 2 and (n == 8 or (m == 64 and n == 64)))
        ]
        swapped += [
            (256, n, 128, cta, 1)
            for n in (8, 16, 32)
            for cta in (1, 2)
            if not (cta == 2 and n == 8)
        ]
        if wide_swap:
            swapped += [
                (128, 128, 128, 1, 1),
                (128, 128, 128, 2, 1),
                (128, 256, 128, 1, 1),
                (128, 256, 128, 2, 1),
                (256, 64, 128, 1, 1),
                (256, 128, 128, 1, 1),
                (256, 128, 128, 2, 1),
            ]
    for swap, tiles in ((False, normal), (True, swapped)):
        for m, n, mma_m, cta, cn in tiles:
            name = (
                f"CONFIG_sm100_{m}x{n}x128_{mma_m}x{n}x32_"
                f"cluster{cta}x{cn}_{cta}ctamma" + ("_swapAB" if swap else "")
            )
            for mode in ("stg", "tma"):
                yield name, cta, mode


def catalog_configs(op, e, n, k, file):
    """Enumerate Frost's complete catalog, using only its capability gates.

    The catalog contains normal configurations. Explicitly expand both operand
    orientations before probing, including every MMA-M multiplicity and cluster.
    A rejected configuration is evidence too, so retain its exact reason.
    """
    from cudnn.gemm.frost import tile_config
    from cudnn.gemm.frost.compiler import probe_chain
    from cudnn.gemm.frost.graph_analyzer import analyze
    from flashinfer.fused_moe.backends.cudnn_frost.export import (
        _build_graph,
    )

    chain = analyze(_build_graph(1024, n, k, e, e, op))
    for original in tile_config.CATALOG:
        for swap in (False, True):
            config = replace(original, swap_ab=swap)
            record = dict(
                op=op,
                experts=e,
                n=n,
                k=k,
                config=config.name,
                pipeline=config.pipeline,
                swap_ab=swap,
                mma_size_m=config.mma_size_m,
                mma_tile_m=config.mma_tile_m,
                cta_tile_m=config.cta_tile_m,
                cta_tile_n=config.cta_tile_n,
                cta_tile_k_bytes=config.cta_tile_k_bytes,
                mma_tile_k_bytes=config.mma_tile_k_bytes,
                cta_group=getattr(config, "cta_group", 1),
                cluster=[config.cga_size_m, config.cga_size_n],
            )
            try:
                probe_chain(chain, config)
            except (ValueError, RuntimeError, NotImplementedError) as exc:
                emit(
                    file,
                    dict(kind="catalog_rejected", **record, error=str(exc)),
                    quiet=True,
                )
                continue
            for mode in store_modes(config):
                emit(
                    file,
                    dict(kind="catalog_candidate", **record, store_mode=mode),
                    quiet=True,
                )
                yield config.name, record["cta_group"], mode


def export_task(opts):
    """A CPU compiler worker; no benchmark timing is collected in this phase."""
    from flashinfer.fused_moe.backends.cudnn_frost.export import (
        export_one,
    )

    start = time.monotonic()
    try:
        kernel = export_one(opts)
        if opts.warm_source_jit:
            from flashinfer.fused_moe.backends.cudnn_frost.bf16 import (
                runtime,
            )

            # Instantiate the shared template before compiling. The rendered
            # digest includes geometry, preventing cross-configuration cache hits.
            source, digest = runtime._read_source(opts.output_dir, kernel)
            runtime._load_source(source, digest, kernel["arch"], 0)
        return dict(
            kind="exported",
            id=opts.id,
            kernel=kernel,
            seconds=time.monotonic() - start,
        )
    except Exception as exc:
        return dict(
            kind="export_failed",
            id=opts.id,
            error=f"{type(exc).__name__}: {exc}",
            seconds=time.monotonic() - start,
        )


def export(args, file):
    import cudnn

    source = args.cudnn_frost_source.resolve()
    if not Path(cudnn.__file__).resolve().is_relative_to(source):
        raise RuntimeError(
            f"cuDNN Frost import is outside requested source: {cudnn.__file__}"
        )
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip():
        raise RuntimeError("Export requires a clean cuDNN Frost source revision")
    major, minor = torch.cuda.get_device_capability()
    manifest = args.artifacts / "cudnn_frost_selected_kernels.json"
    payload = dict(schema_version=2, producer=dict(name="cudnn_frost"), kernels=[])
    if manifest.exists():
        payload = json.loads(manifest.read_text())
        if payload.get("schema_version") != 2:
            raise ValueError(
                "Re-export legacy object candidates into a new source directory"
            )
    existing = {r["id"] for r in payload["kernels"]}
    failed = set()
    if args.resume:
        for line in args.output.read_text().splitlines():
            record = json.loads(line)
            if record["kind"] == "export_failed":
                failed.add(record["id"])
    candidate_configs = list(configs(args.small, args.wide_swap, (major, minor)))
    if args.config:
        from cudnn.gemm.frost import tile_config

        candidate_configs = [
            (name, getattr(config, "cta_group", 1), mode)
            for name in args.config
            for config in (tile_config.by_name(name),)
            for mode in store_modes(config)
        ]
    tasks = []
    catalog_plan = None
    if args.catalog_plan:
        catalog_plan = [
            json.loads(line) for line in args.catalog_plan.read_text().splitlines()
        ]
        header = catalog_plan[0]
        if (
            header.get("producer_revision") != revision
            or header.get("arch") != f"sm_{major}{minor}a"
        ):
            raise ValueError(
                "Catalog plan must match the Frost revision and GPU architecture"
            )
    for e, h, i, _ in args.geometries:
        for stage, (op, n, k) in enumerate(
            (
                (f"grouped_gemm1_{args.activation}", i, h),
                ("grouped_gemm2", h, i),
            ),
            1,
        ):
            if args.stage_only and args.stage_only != stage:
                continue
            pool = (
                catalog_configs(op, e, n, k, file)
                if args.all_configs
                else candidate_configs
            )
            if catalog_plan is not None:
                pool = [
                    (r["config"], r["cta_group"], r["store_mode"])
                    for r in catalog_plan
                    if r.get("kind") == "catalog_candidate"
                    and (r["op"], r["experts"], r["n"], r["k"]) == (op, e, n, k)
                ]
                if not pool:
                    raise ValueError(f"Catalog plan has no records for {(op, e, n, k)}")
            for index, (name, cta, mode) in enumerate(pool):
                if index % args.num_shards != args.shard_index:
                    continue
                identity = f"sm{major}{minor}_cudnn_frost_{op}_e{e}_n{n}_k{k}_{name}_{mode}_{revision[:12]}"
                if identity in existing or identity in failed or args.catalog_only:
                    continue
                opts = argparse.Namespace(
                    op=op,
                    config=name,
                    cta_group=cta,
                    scheduler="clc",
                    store_mode=mode,
                    s=1024,
                    n=n,
                    k=k,
                    experts=e,
                    groups=e,
                    id=identity,
                    output_dir=args.artifacts,
                    cudnn_frost_revision=revision,
                    replace=False,
                    warm_source_jit=args.warm_source_jit,
                )
                tasks.append(opts)
    emit(
        file,
        dict(kind="export_plan", tasks=len(tasks), jobs=args.jobs, revision=revision),
    )
    args.artifacts.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = manifest.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        temporary.replace(manifest)

    def retain(result):
        if result["kind"] == "exported":
            payload["kernels"].append(result.pop("kernel"))
        emit(file, result)

    try:
        if args.jobs == 1:
            for index, task in enumerate(tasks):
                retain(export_task(task))
                if index % 32 == 31:
                    save()
        else:
            with ProcessPoolExecutor(
                max_workers=args.jobs,
                mp_context=multiprocessing.get_context("spawn"),
            ) as executor:
                futures = {
                    executor.submit(export_task, task): task.id for task in tasks
                }
                for index, future in enumerate(as_completed(futures)):
                    # A crashed worker is a campaign failure, not an unsupported
                    # configuration. Keep completed records and require resume.
                    retain(future.result())
                    if index % 32 == 31:
                        save()
    finally:
        save()
    emit(
        file,
        dict(
            kind="export_complete",
            attempted=len(tasks),
            total_sources=len(payload["kernels"]),
        ),
    )


def reference(act, w1, w2, activation="swiglu"):
    x, ids, scores = act.hidden_states_q, act.topk_ids, act.topk_weights
    expanded = torch.zeros(*ids.shape, x.shape[1], device=x.device, dtype=torch.float32)
    for expert in range(w1.shape[0]):
        row, slot = torch.where(ids == expert)
        mid = activation_reference(
            x[row].float() @ w1[expert].float().T, activation
        ).bfloat16()
        expanded[row, slot] = (
            mid.float() @ w2[expert].float().T
        ).bfloat16().float() * scores[row, slot, None]
    return expanded.sum(1).bfloat16()


def measure_stage(fn, out, ref, args):
    fn()
    l2 = error(out, ref)
    screening = args.stage_only is not None
    # Exhaustive stage screening visits even very slow catalog points. Eager
    # and graph correctness runs already warm the kernel; use independent
    # single-replay samples here, then refine finalists with the full protocol.
    graph = capture(fn, args.batch, warmup=0 if screening else 3)
    out.fill_(float("nan"))
    graph.replay()
    l2 = max(l2, error(out, ref))
    iterations = 1 if screening else args.iterations
    if args.adaptive and not screening:
        pilot = measure(graph, 1, args.batch)
        iterations = max(1, min(iterations, int(2.0 / (pilot * args.batch))))
    samples = [
        measure(graph, iterations, args.batch, warmup=not screening)
        for _ in range(args.rounds)
    ]
    return dict(
        timing_mode="single_replay_screen" if screening else "refinement",
        milliseconds=statistics.median(samples),
        samples=samples,
        iterations=iterations,
        relative_l2=l2,
    )


def sweep(args, file):
    from flashinfer.fused_moe.backends.cudnn_frost.bf16 import (
        fc2,
        moe,
        runtime,
    )
    from flashinfer.fused_moe import MoEActivationPack

    runtime._artifact_roots = lambda: (args.artifacts,)
    fc2._artifact_roots = runtime._artifact_roots
    uuid = torch.cuda.get_device_properties(0).uuid
    device = torch.device("cuda", 0)
    torch.manual_seed(41)
    torch.backends.cuda.matmul.allow_tf32 = False
    completed = set()
    measured = {}
    finished_stages = set()
    if args.resume:
        for line in args.output.read_text().splitlines():
            r = json.loads(line)
            if r["kind"] == "moe":
                completed.add((tuple(r["geometry"]), r["tokens"], r["routing"]))
            elif r["kind"] == "stage_candidate":
                key = (
                    tuple(r["geometry"]),
                    r["tokens"],
                    r["routing"],
                    r["stage"],
                    r["result"]["id"],
                )
                measured[key] = r["result"]
            elif r["kind"] == "stage":
                finished_stages.add(
                    (tuple(r["geometry"]), r["tokens"], r["routing"], r["stage"])
                )
    prior = {}
    if args.refine_from is not None:
        for line in args.refine_from.read_text().splitlines():
            r = json.loads(line)
            if r["kind"] == "stage":
                prior[(tuple(r["geometry"]), r["tokens"], r["routing"], r["stage"])] = r
    # Staged search across activations: activations of one FC1 class share the
    # mainloop, so screen only the tiles another sweep ranked fastest per case.
    screened_tiles = {}
    if args.candidates_from is not None:
        tiles = {
            k.artifact_id: k.tactic_metadata["tile"]
            for k in (
                *runtime._discover((args.artifacts,)),
                *fc2.discover((args.artifacts,)),
            )
        }
        for line in args.candidates_from.read_text().splitlines():
            r = json.loads(line)
            if r["kind"] == "stage":
                key = (tuple(r["geometry"]), r["tokens"], r["routing"], r["stage"])
                screened_tiles[key] = {
                    tiles[x["id"]] for x in r["ranked"][: args.candidates_top]
                }
    for e, h, i, topk in args.geometries:
        activation = ACTIVATIONS[args.activation]()
        w1 = (
            torch.randn(
                e,
                (2 if activation.is_gated else 1) * i,
                h,
                device=device,
                dtype=torch.bfloat16,
            )
            * 0.02
        )
        w2 = torch.randn(e, h, i, device=device, dtype=torch.bfloat16) * 0.02
        for tokens, routing in itertools.product(args.tokens, args.routing):
            check_idle_gpu(uuid)
            last_peer_check = time.monotonic()
            case = dict(
                geometry=[e, h, i, topk],
                tokens=tokens,
                routing=routing,
                activation=args.activation,
            )
            x = torch.randn(tokens, h, device=device, dtype=torch.bfloat16) * 0.02
            logits = torch.rand(tokens, e, device=device)
            if routing == "skew":
                logits[: tokens // 2, 0] += 2
            ids = logits.topk(topk, dim=1).indices.int()
            scores = torch.rand(tokens, topk, device=device).softmax(-1)
            act = MoEActivationPack(x, None, ids, scores)
            # Consume the same RNG draws even when resuming after a completed
            # case, so later workloads retain their original inputs.
            if ((e, h, i, topk), tokens, routing) in completed:
                continue
            if (
                args.stage_only
                and ((e, h, i, topk), tokens, routing, args.stage_only)
                in finished_stages
            ):
                continue
            rows = tokens * topk
            order = ids.flatten().argsort()
            grouped = x[order // topk].contiguous()
            counts = torch.bincount(ids.flatten().long(), minlength=e)
            offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0))).int()
            offsets_cpu = offsets.tolist()
            emit(file, dict(kind="inputs", **case, offsets=offsets_cpu))
            mid = torch.empty(rows, i, device=device, dtype=torch.bfloat16)
            down = torch.empty(rows, h, device=device, dtype=torch.bfloat16)
            up = w1[:, :i].contiguous()
            gate = w1[:, i:].contiguous() if activation.is_gated else up
            ref_mid = torch.empty_like(mid)
            ref_down = torch.empty_like(down)
            for expert, (lo, hi) in enumerate(
                zip(offsets_cpu[:-1], offsets_cpu[1:], strict=True)
            ):
                a = grouped[lo:hi].float()
                ref_mid[lo:hi] = activation_reference(
                    a @ w1[expert].float().T, args.activation
                ).bfloat16()
                ref_down[lo:hi] = (
                    ref_mid[lo:hi].float() @ w2[expert].float().T
                ).bfloat16()
            scale = torch.tensor(
                [1.0, 4.0, 25.0] if args.activation == "situ" else [1.0],
                dtype=torch.float32,
                device=device,
            )
            pools = moe._kernels(rows, h, i, e, device, activation)
            selected = []
            for stage, pool in enumerate(pools):
                if args.stage_only and args.stage_only != stage + 1:
                    continue
                if args.refine_from is not None:
                    previous = prior[((e, h, i, topk), tokens, routing, stage + 1)]
                    screened = {r["id"] for r in previous["ranked"]}
                    finalists = set(previous["selected"])
                    pool = tuple(
                        k
                        for k in pool
                        if k.artifact_id not in screened or k.artifact_id in finalists
                    )
                if args.candidates_from is not None:
                    allowed = screened_tiles[
                        ((e, h, i, topk), tokens, routing, stage + 1)
                    ]
                    pool = tuple(
                        k for k in pool if k.tactic_metadata["tile"] in allowed
                    )
                if not pool:
                    raise RuntimeError(f"No matching stage {stage} sources for {case}")
                workspace = torch.empty(
                    max(k.workspace_bytes for k in pool),
                    device=device,
                    dtype=torch.uint8,
                )
                ranked = []
                for idx, kernel in enumerate(pool):
                    if kernel.artifact_id in args.skip_kernel:
                        emit(
                            file,
                            dict(
                                kind="stage_skipped",
                                **case,
                                stage=stage + 1,
                                id=kernel.artifact_id,
                            ),
                        )
                        continue
                    key = (
                        (e, h, i, topk),
                        tokens,
                        routing,
                        stage + 1,
                        kernel.artifact_id,
                    )
                    if key in measured:
                        saved = measured[key]
                        if saved["source_sha256"] != kernel.source_sha256:
                            raise RuntimeError("Cannot resume a changed kernel source")
                        ranked.append(saved)
                        continue
                    if idx == 0 or time.monotonic() - last_peer_check >= 1.0:
                        check_idle_gpu(uuid)
                        last_peer_check = time.monotonic()
                    print(
                        json.dumps(
                            dict(
                                kind="checking",
                                **case,
                                stage=stage + 1,
                                id=kernel.artifact_id,
                            )
                        ),
                        flush=True,
                    )
                    if stage == 0:
                        fn = partial(
                            runtime._launch,
                            kernel,
                            grouped,
                            gate,
                            up,
                            offsets,
                            scale,
                            mid,
                            workspace,
                        )
                        out, ref = mid, ref_mid
                    else:
                        plan = fc2.PreparedFc2(
                            kernel, ref_mid, w2, offsets, down, workspace
                        )
                        fn, out, ref = plan.run, down, ref_down
                    try:
                        timing = measure_stage(fn, out, ref, args)
                    except Exception as exc:
                        emit(
                            file,
                            dict(
                                kind="stage_failed",
                                **case,
                                stage=stage + 1,
                                id=kernel.artifact_id,
                                error=f"{type(exc).__name__}: {exc}",
                            ),
                        )
                        if not args.keep_going:
                            raise
                        # A poisoned CUDA context must be restarted externally;
                        # it cannot validate any subsequent candidate reliably.
                        torch.cuda.synchronize()
                        continue
                    record = dict(
                        id=kernel.artifact_id,
                        swap_ab=kernel.swap_ab,
                        source_sha256=kernel.source_sha256,
                        mma_size_m=mma_size_m(kernel.tactic_metadata["tile"]),
                        **timing,
                    )
                    ranked.append(record)
                    emit(
                        file,
                        dict(
                            kind="stage_candidate",
                            **case,
                            stage=stage + 1,
                            result=record,
                        ),
                    )
                ranked.sort(key=lambda r: r["milliseconds"])
                check_idle_gpu(uuid)
                keep = set()
                bands = (1, 2, 4) if args.shortlist_by_mma else (None,)
                for swap, mma_size in itertools.product((False, True), bands):
                    keep.update(
                        r["id"]
                        for r in [
                            r
                            for r in ranked
                            if r["swap_ab"] == swap
                            and (mma_size is None or r["mma_size_m"] == mma_size)
                        ][: args.shortlist]
                    )
                selected.append(tuple(k for k in pool if k.artifact_id in keep))
                emit(
                    file,
                    dict(
                        kind="stage",
                        **case,
                        stage=stage + 1,
                        checked=len(ranked),
                        ranked=ranked,
                        selected=sorted(keep),
                    ),
                )
            if args.stage_only:
                gc.collect()
                continue
            # Independent stage rankings only shortlist; choose complete MoE here.
            plans = moe._Plans(tokens, h, i, e, topk, device, *selected, {})
            output = torch.empty_like(x)
            expected = reference(act, w1, w2, args.activation)
            ranked, graphs = [], []
            for (_key, launch), (a, b) in zip(
                plans.launches.items(), itertools.product(*selected), strict=True
            ):
                fn = partial(launch, output, x, ids, scores, w1, w2, plans.workspace)
                fn()
                l2 = error(output, expected)
                graph = capture(fn, args.batch)
                output.fill_(float("nan"))
                graph.replay()
                l2 = max(l2, error(output, expected))
                graphs.append(graph)
                ranked.append(
                    dict(
                        fc1=a.artifact_id,
                        fc2=b.artifact_id,
                        swap_fc1=a.swap_ab,
                        swap_fc2=b.swap_ab,
                        relative_l2=l2,
                        samples=[],
                    )
                )
            for round_idx in range(args.rounds):
                check_idle_gpu(uuid)
                order = list(range(len(graphs)))
                shift = round_idx % len(order)
                order = order[shift:] + order[:shift]
                if round_idx % 2:
                    order.reverse()
                for idx in order:
                    ranked[idx]["samples"].append(
                        measure(graphs[idx], args.iterations, args.batch)
                    )
            # Each shortlisted pair must reset state under changing routing.
            ids.fill_(e - 1)
            scores.mul_(0.5)
            changed = reference(act, w1, w2, args.activation)
            for record, graph in zip(ranked, graphs, strict=True):
                output.fill_(float("nan"))
                graph.replay()
                record["changed_routing_l2"] = error(output, changed)
                record["milliseconds"] = statistics.median(record["samples"])
            ranked.sort(key=lambda r: r["milliseconds"])
            emit(
                file,
                dict(
                    kind="moe",
                    **case,
                    checked=len(ranked),
                    ranked=ranked,
                    winner=ranked[0],
                ),
            )
            del graphs, plans, graph, fn, workspace
            gc.collect()


def select(args, file):
    """Copy the union of full-MoE winners, preserving existing architectures."""
    results = [json.loads(line) for line in args.results.read_text().splitlines()]
    options = results[0]["options"]
    expected = {
        (tuple(g), t, r)
        for g, t, r in itertools.product(
            options["geometries"], options["tokens"], options["routing"]
        )
    }
    cases = [r for r in results if r["kind"] == "moe"]
    actual = {(tuple(r["geometry"]), r["tokens"], r["routing"]) for r in cases}
    if actual != expected or len(cases) != len(expected):
        raise ValueError("Selection requires a complete sweep without duplicate cases")
    identities = {r["winner"][stage] for r in cases for stage in ("fc1", "fc2")}
    copy_selected(args.artifacts, args.selected_dir, identities, file)


def copy_selected(artifacts, selected_dir, identities, file):
    """Copy verified sources and manifest records, never replacing other bytes.

    Packaged records keep their order; new records are appended by identity.
    """
    from flashinfer.fused_moe.backends.cudnn_frost.runtime import (
        _safe_child,
    )

    manifest = json.loads((artifacts / "cudnn_frost_selected_kernels.json").read_text())
    if manifest.get("schema_version") != 2:
        raise ValueError(
            "Selection requires a source manifest; re-export legacy objects"
        )
    records = {r["id"]: r for r in manifest["kernels"]}
    target_manifest = selected_dir / "cudnn_frost_selected_kernels.json"
    payload = (
        json.loads(target_manifest.read_text())
        if target_manifest.exists()
        else dict(schema_version=2, producer=dict(name="cudnn_frost"), kernels=[])
    )
    if payload.get("schema_version") != 2:
        raise ValueError(f"refusing to update unsupported manifest {target_manifest}")
    previous = {r["id"]: r for r in payload["kernels"]}
    for identity in sorted(identities):
        record = records[identity]
        source = _safe_child(artifacts, record["source"]["path"])
        target = _safe_child(selected_dir, record["source"]["path"])
        digest = record["source"]["sha256"]
        if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Candidate source digest mismatch: {source}")
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                raise FileExistsError(f"Refusing to replace selected source {target}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        existing = previous.get(identity)
        if existing is not None and existing != record:
            raise ValueError(
                f"Selected manifest already has a different record for {identity}"
            )
        if existing is None:
            payload["kernels"].append(record)
            previous[identity] = record
        emit(file, dict(kind="selected", id=identity))
    temporary = target_manifest.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(target_manifest)


def rank_stage(runs, width):
    """Geomean over routings of latency normalized by each routing's best.

    ``runs`` maps a routing to candidate milliseconds; only candidates measured
    under every routing are ranked. Ties keep the artifact identity order.
    """
    common = set.intersection(*(set(times) for times in runs.values()))
    if not common:
        raise ValueError("No stage candidate was measured under every routing")
    best = {routing: min(times.values()) for routing, times in runs.items()}
    score = {
        identity: statistics.geometric_mean(
            times[identity] / best[routing] for routing, times in runs.items()
        )
        for identity in common
    }
    return sorted(common, key=lambda identity: (score[identity], identity))[:width]


def build_shortlist(results, width):
    """Combine stage sweeps into ``moe_shortlists.json`` entries.

    FC1 rankings are activation-specific. FC2 does not depend on the
    activation, so repeated FC2 sweeps of one case use each candidate's median.
    """
    fc1, fc2, arches = {}, {}, set()
    for record in results:
        if record["kind"] in ("environment", "resume"):
            arches.add(record["arch"])
        if record["kind"] != "stage":
            continue
        geometry = tuple(record["geometry"])
        times = {r["id"]: r["milliseconds"] for r in record["ranked"]}
        if record["stage"] == 1:
            case = fc1.setdefault(
                (record["activation"], geometry, record["tokens"]), {}
            )
            if record["routing"] in case:
                raise ValueError(f"duplicate FC1 stage sweep for {record}")
            case[record["routing"]] = times
        else:
            case = fc2.setdefault((geometry, record["tokens"]), {})
            for identity, value in times.items():
                case.setdefault(record["routing"], {}).setdefault(identity, []).append(
                    value
                )
    if len(arches) != 1:
        raise ValueError(f"Shortlist sweeps must share one architecture: {arches}")
    (arch,) = arches
    routings = {frozenset(case) for case in (*fc1.values(), *fc2.values())}
    if len(routings) != 1:
        raise ValueError("Every stage case must cover the same routings")
    entries = []
    for (activation, geometry, tokens), runs in fc1.items():
        if (geometry, tokens) not in fc2:
            raise ValueError(f"Missing FC2 stage sweep for {geometry}, {tokens}")
        second = {
            routing: {k: statistics.median(v) for k, v in times.items()}
            for routing, times in fc2[(geometry, tokens)].items()
        }
        e, h, i, topk = geometry
        entries.append(
            dict(
                arch=arch,
                activation=activation,
                experts=e,
                hidden=h,
                intermediate=i,
                top_k=topk,
                tokens=tokens,
                fc1=rank_stage(runs, width),
                fc2=rank_stage(second, width),
            )
        )
    return entries


def validate_shortlist(args, entries, file):
    """Check every shortlisted pair as a complete routed MoE and graph replay."""
    from flashinfer.fused_moe.backends.cudnn_frost.bf16 import (
        fc2,
        moe,
        runtime,
    )
    from flashinfer.fused_moe import MoEActivationPack

    runtime._artifact_roots = lambda: (args.artifacts,)
    fc2._artifact_roots = runtime._artifact_roots
    device = torch.device("cuda", 0)
    kernels = {
        k.artifact_id: k
        for k in (
            *runtime._discover((args.artifacts,)),
            *fc2.discover((args.artifacts,)),
        )
    }
    weights = {}
    for entry in entries:
        e, h, i = (entry[k] for k in ("experts", "hidden", "intermediate"))
        key = (entry["activation"], e, h, i)
        if key not in weights:
            weights.clear()  # bound memory to one model geometry at a time
            torch.manual_seed(41)
            gated = ACTIVATIONS[entry["activation"]]().is_gated
            weights[key] = (
                torch.randn(e, (2 if gated else 1) * i, h, device=device).bfloat16()
                * 0.02,
                torch.randn(e, h, i, device=device).bfloat16() * 0.02,
            )
        w1, w2 = weights[key]
        tokens, topk = entry["tokens"], entry["top_k"]
        first = [kernels[identity] for identity in entry["fc1"]]
        second = [kernels[identity] for identity in entry["fc2"]]
        plans = moe._Plans(tokens, h, i, e, topk, device, first, second, {})
        for routing in ("uniform", "skew"):
            torch.manual_seed(tokens)
            x = torch.randn(tokens, h, device=device, dtype=torch.bfloat16) * 0.02
            logits = torch.rand(tokens, e, device=device)
            if routing == "skew":
                logits[: tokens // 2, 0] += 2
            ids = logits.topk(topk, dim=1).indices.int()
            scores = torch.rand(tokens, topk, device=device).softmax(-1)
            act = MoEActivationPack(x, None, ids, scores)
            expected = reference(act, w1, w2, entry["activation"])
            output = torch.empty_like(x)
            l2 = 0.0
            for launch in plans.launches.values():
                fn = partial(launch, output, x, ids, scores, w1, w2, plans.workspace)
                fn()
                l2 = max(l2, error(output, expected))
                graph = capture(fn, 1, warmup=0)
                output.fill_(float("nan"))
                graph.replay()
                l2 = max(l2, error(output, expected))
            emit(
                file,
                dict(
                    kind="validated",
                    **{k: v for k, v in entry.items() if k not in ("fc1", "fc2")},
                    routing=routing,
                    pairs=len(plans.launches),
                    relative_l2=l2,
                ),
                quiet=True,
            )
        del plans
        gc.collect()


def shortlist(args, file):
    """Write two FC1 and two FC2 artifacts per profile and copy their kernels."""
    results = [
        json.loads(line)
        for path in args.results
        for line in path.read_text().splitlines()
    ]
    entries = build_shortlist(results, args.width)
    if args.validate:
        validate_shortlist(args, entries, file)
    path = args.selected_dir / "moe_shortlists.json"
    payload = (
        json.loads(path.read_text()) if path.exists() else dict(version=1, entries=[])
    )
    if payload.get("version") != 1:
        raise ValueError("unsupported Frost MoE shortlist version")

    def profile(entry):
        return tuple(
            entry[k]
            for k in (
                "arch",
                "activation",
                "experts",
                "hidden",
                "intermediate",
                "top_k",
                "tokens",
            )
        )

    index = {profile(entry): n for n, entry in enumerate(payload["entries"])}
    activations = {name: n for n, name in enumerate(ACTIVATIONS)}
    geometries = [tuple(g) for g in results[0]["options"]["geometries"]]

    def order(entry):
        # Match the packaged tables: activation, sweep geometry order, tokens.
        geometry = profile(entry)[2:6]
        rank = geometries.index(geometry) if geometry in geometries else len(geometries)
        return activations[entry["activation"]], rank, entry["tokens"]

    for entry in sorted(entries, key=order):
        existing = index.get(profile(entry))
        if existing is None:
            payload["entries"].append(entry)
        elif payload["entries"][existing] != entry:
            if not args.replace:
                raise ValueError(f"Shortlist already differs for {profile(entry)}")
            payload["entries"][existing] = entry
    copy_selected(
        args.artifacts,
        args.selected_dir,
        {
            identity
            for x in entries
            for stage in ("fc1", "fc2")
            for identity in x[stage]
        },
        file,
    )
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    emit(file, dict(kind="shortlisted", entries=len(entries)))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    comparison = commands.add_parser(
        "benchmark",
        help="Compare full MoE: CUTLASS/TRTLLM vs the same pool plus cuDNN Frost",
    )
    comparison.add_argument(
        "--activation", choices=tuple(ACTIVATIONS), default="swiglu"
    )
    comparison.add_argument("--seed", type=int, default=41)
    comparison.add_argument("--tokens", default="8192,12288")
    comparison.add_argument("--top-k", type=int, default=1)
    comparison.add_argument("--experts", type=int, default=12)
    comparison.add_argument("--hidden", type=int, default=7168)
    comparison.add_argument("--intermediate", type=int, default=3072)
    comparison.add_argument(
        "--probe-cudnn-frost",
        action="store_true",
        help="Research-only: compare matching sources outside automatic admission",
    )
    comparison.add_argument(
        "--artifact-root",
        type=Path,
        help="Research-only source directory, used instead of packaged sources",
    )
    comparison.add_argument(
        "--sweep-cudnn-frost",
        action="store_true",
        help="Separately rank every full cuDNN Frost tactic with rotated graph timing",
    )
    comparison.add_argument("--routing", choices=("uniform", "skew"), default="uniform")
    comparison.add_argument("--rounds", type=int, default=9)
    comparison.add_argument("--iterations", type=int, default=30)
    comparison.add_argument("--graph-batch", type=int, default=16)
    comparison.add_argument("--source-jit", action="store_true")

    offline = {}
    for mode, help_text in (
        ("export", "Export normal/swap-AB FC1 and FC2 candidates with cuDNN Frost"),
        ("sweep", "Check and time stage candidates, then complete MoE pairs"),
        ("select", "Copy the union of complete-MoE winners from a finished sweep"),
        (
            "shortlist",
            "Write per-profile FC1/FC2 shortlists from stage sweeps and copy them",
        ),
    ):
        command = commands.add_parser(mode, help=help_text)
        command.add_argument("--artifacts", required=True, type=Path)
        command.add_argument("--output", required=True, type=Path)
        command.set_defaults(resume=False)
        offline[mode] = command
        if mode in ("export", "sweep"):
            command.add_argument(
                "--activation", choices=tuple(ACTIVATIONS), default="swiglu"
            )
            command.add_argument(
                "--geometries",
                default="12:7168:3072:2,8:4096:14336:2,64:2048:1408:6",
                help="Comma-separated E:H:I:topk geometries",
            )

    exporter = offline["export"]
    exporter.add_argument("--cudnn-frost-source", required=True, type=Path)
    exporter.add_argument(
        "--config",
        action="append",
        help="Explicit cuDNN Frost tile config; repeat to sweep several (STG and TMA)",
    )
    exporter.add_argument(
        "--wide-swap",
        action="store_true",
        help="Also export wider swap tiles for prefill",
    )
    exporter.add_argument(
        "--small",
        action="store_true",
        help="Minimal normal/swap STG/TMA correctness pool",
    )
    exporter.add_argument(
        "--all-configs",
        action="store_true",
        help="Enumerate the complete Frost catalog and both normal/swap AB orientations",
    )
    exporter.add_argument(
        "--catalog-only",
        action="store_true",
        help="Record every supported/rejected catalog point without compiling",
    )
    exporter.add_argument(
        "--catalog-plan",
        type=Path,
        help="Reuse a revision/architecture-checked full-catalog enumeration",
    )
    exporter.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="CPU compiler processes; timing runs separately",
    )
    exporter.add_argument("--stage-only", type=int, choices=(1, 2))
    exporter.add_argument(
        "--warm-source-jit",
        action="store_true",
        help="Compile the standalone sources before serial GPU measurement",
    )
    exporter.add_argument("--num-shards", type=int, default=1)
    exporter.add_argument("--shard-index", type=int, default=0)
    exporter.add_argument(
        "--resume",
        action="store_true",
        help="Reuse successful exports and recorded failures",
    )

    sweeper = offline["sweep"]
    sweeper.add_argument("--source-jit", action="store_true")
    sweeper.add_argument(
        "--stage-only",
        type=int,
        choices=(1, 2),
        help="Measure a single stage shard without full-MoE pairs",
    )
    sweeper.add_argument(
        "--keep-going",
        action="store_true",
        help="Record candidate failures; abort if the CUDA context is poisoned",
    )
    sweeper.add_argument(
        "--skip-kernel",
        action="append",
        default=[],
        help="Explicitly exclude an artifact after a recorded crash",
    )
    sweeper.add_argument(
        "--shortlist-by-mma",
        action="store_true",
        help="Retain finalists independently for each orientation and mma_size_m=1/2/4",
    )
    sweeper.add_argument(
        "--refine-from",
        type=Path,
        help="Re-sweep prior stage finalists plus newly exported sources",
    )
    sweeper.add_argument(
        "--candidates-from",
        type=Path,
        help="Screen only tiles a completed sweep ranked fastest for the same "
        "stage case, e.g. SwiGLU's for another gated activation",
    )
    sweeper.add_argument(
        "--candidates-top",
        type=int,
        default=8,
        help="Tiles retained per case with --candidates-from",
    )
    sweeper.add_argument(
        "--resume",
        action="store_true",
        help="Append to an interrupted sweep, skipping complete cases",
    )
    sweeper.add_argument(
        "--adaptive",
        action="store_true",
        help="Limit slow stage candidates to about 2 ms per sample (at least one graph replay)",
    )
    sweeper.add_argument("--tokens", default="1,16,128,512,2048,4096,8192")
    sweeper.add_argument("--routing", default="uniform,skew")
    sweeper.add_argument(
        "--shortlist", type=int, default=2, help="Candidates per stage per orientation"
    )
    sweeper.add_argument("--rounds", type=int, default=5)
    sweeper.add_argument("--iterations", type=int, default=10)
    sweeper.add_argument("--batch", type=int, default=8)

    selector = offline["select"]
    selector.add_argument(
        "--results",
        required=True,
        type=Path,
        help="Completed sweep JSONL for selection",
    )
    selector.add_argument(
        "--selected-dir", required=True, type=Path, help="Copy full-MoE winners here"
    )

    shortlister = offline["shortlist"]
    shortlister.add_argument(
        "--results",
        required=True,
        type=Path,
        action="append",
        help="Stage sweep JSONL; repeat for FC1 activation and shared FC2 sweeps",
    )
    shortlister.add_argument(
        "--selected-dir",
        required=True,
        type=Path,
        help="Merge shortlist entries and copy their kernels here",
    )
    shortlister.add_argument(
        "--width", type=int, default=2, help="Candidates per stage per profile"
    )
    shortlister.add_argument(
        "--validate",
        action="store_true",
        help="Check every shortlisted pair with routed inputs and graph replay",
    )
    shortlister.add_argument(
        "--replace", action="store_true", help="Replace differing profile entries"
    )

    args = parser.parse_args(argv)
    if args.mode == "benchmark":
        if min(args.rounds, args.iterations, args.graph_batch) <= 0:
            comparison.error("rounds, iterations and graph-batch must be positive")
        if (
            min(args.experts, args.hidden, args.intermediate, args.top_k) <= 0
            or args.top_k > args.experts
        ):
            comparison.error("positive dimensions and top-k <= experts are required")
        try:
            tokens = list(map(int, args.tokens.split(",")))
        except ValueError:
            comparison.error("tokens must be comma-separated positive integers")
        if min(tokens) <= 0:
            comparison.error("tokens must be positive")
        return args

    command = offline[args.mode]
    args.artifacts = args.artifacts.resolve()
    if args.mode in ("select", "shortlist"):
        args.selected_dir = args.selected_dir.resolve()
        if args.mode == "shortlist" and not 1 <= args.width <= 2:
            command.error("the Frost shortlist runtime accepts one or two per stage")
        return args
    try:
        args.geometries = [
            tuple(map(int, v.split(":"))) for v in args.geometries.split(",")
        ]
    except ValueError:
        command.error("geometries must be positive E:H:I:topk with topk <= E")
    if any(len(g) != 4 or min(g) <= 0 or g[3] > g[0] for g in args.geometries):
        command.error("geometries must be positive E:H:I:topk with topk <= E")
    if args.mode == "export":
        if args.catalog_plan and (
            args.all_configs
            or args.config
            or args.small
            or args.wide_swap
            or args.catalog_only
        ):
            command.error(
                "--catalog-plan cannot be combined with other config selectors"
            )
        if args.all_configs and (args.config or args.small or args.wide_swap):
            command.error(
                "--all-configs cannot be combined with --config, --small or --wide-swap"
            )
        if args.catalog_only and not args.all_configs:
            command.error("--catalog-only requires --all-configs")
        if (
            args.jobs < 1
            or args.num_shards < 1
            or not 0 <= args.shard_index < args.num_shards
        ):
            command.error(
                "jobs/shards must be positive and 0 <= shard-index < num-shards"
            )
        if args.resume:
            if not args.output.is_file():
                command.error("--resume requires an existing export output")
            old = json.loads(args.output.read_text().splitlines()[0])["options"]
            for key in (
                "activation",
                "geometries",
                "artifacts",
                "all_configs",
                "config",
                "small",
                "wide_swap",
                "stage_only",
                "num_shards",
                "shard_index",
                "catalog_only",
                "catalog_plan",
                "warm_source_jit",
            ):
                if json.loads(json.dumps(getattr(args, key), default=str)) != old.get(
                    key
                ):
                    command.error(f"cannot resume with changed {key}")
        return args
    try:
        args.tokens = list(map(int, args.tokens.split(",")))
    except ValueError:
        command.error("tokens must be comma-separated positive integers")
    args.routing = args.routing.split(",")
    if min(args.rounds, args.iterations, args.batch, args.shortlist, *args.tokens) <= 0:
        command.error("measurement parameters and tokens must be positive")
    if any(v not in ("uniform", "skew") for v in args.routing):
        command.error("routing must be uniform and/or skew")
    if args.resume:
        if not args.output.is_file():
            command.error("--resume requires an existing sweep output")
        old = json.loads(args.output.read_text().splitlines()[0])["options"]
        for key in (
            "activation",
            "geometries",
            "tokens",
            "routing",
            "artifacts",
            "shortlist",
            "rounds",
            "iterations",
            "batch",
            "stage_only",
            "keep_going",
            "shortlist_by_mma",
            "candidates_from",
            "candidates_top",
        ):
            current = getattr(args, key)
            # Defaults for options added after a sweep output was started.
            default = dict(
                activation="swiglu",
                keep_going=False,
                shortlist_by_mma=False,
                candidates_top=8,
            ).get(key)
            if json.loads(json.dumps(current, default=str)) != old.get(key, default):
                command.error(f"cannot resume with changed {key}")
    return args


def main():
    args = parse_args()
    if getattr(args, "source_jit", False):
        configure_source_jit()
    if args.mode == "benchmark":
        benchmark(args)
        return
    if args.mode == "select":
        with args.output.open("x") as file:
            select(args, file)
        return
    if args.mode == "shortlist":
        with args.output.open("x") as file:
            shortlist(args, file)
        return
    check_idle_gpu(torch.cuda.get_device_properties(0).uuid)
    with args.output.open("a" if args.resume else "x") as file:
        emit(
            file,
            dict(
                kind="resume" if args.resume else "environment",
                gpu=str(torch.cuda.get_device_properties(0)),
                arch="sm_{}{}a".format(*torch.cuda.get_device_capability()),
                producer_revision=(
                    subprocess.check_output(
                        [
                            "git",
                            "-C",
                            str(args.cudnn_frost_source),
                            "rev-parse",
                            "HEAD",
                        ],
                        text=True,
                    ).strip()
                    if args.mode == "export"
                    else None
                ),
                torch=torch.__version__,
                options={
                    k: str(v) if isinstance(v, Path) else v
                    for k, v in vars(args).items()
                },
                clocks=subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=uuid,clocks.sm,power.limit,power.max_limit",
                        "--format=csv",
                    ],
                    text=True,
                ),
            ),
        )
        (export if args.mode == "export" else sweep)(args, file)


if __name__ == "__main__":
    main()
