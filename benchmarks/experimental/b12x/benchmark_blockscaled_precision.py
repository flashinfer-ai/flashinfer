"""Graph-replay evidence for shared-storage NVFP4/MXFP8 activation precision.

Invoked by ``benchmark_dense_gemm.py --dtype fp4-a16|fp8-a16 --evidence FILE``.
Each record preserves paired raw measurements. A run cannot promote dispatch:
an independent confirmation and review of clock/correctness evidence are required.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import itertools
import json
import pathlib
import random
import statistics
import subprocess
import sys
import time
from dataclasses import replace

import torch

from b12x.gemm import blockscaled
from b12x.gemm.blockscaled import _a16
from b12x.gemm.blockscaled._preparation import _workspace_bytes
from b12x.preparation import PreparationSession, PreparedCall
from b12x._lib.dense_gemm import dense_gemm, dense_gemm_fused_quant_a
from b12x._lib.intrinsics import quantize_grouped_nvfp4_torch
from b12x._lib.runtime_control import kernel_resolution_guard
from b12x.gemm._shared.wo_mxfp8 import quantize_mxfp8_rows_torch, dequantize_mxfp8_rows_torch
from benchmarks.experimental.b12x.common import make_l2_flush_fn

COUNTS = (*range(1, 17), 24, 32, 64, 128, 256, 512, 1024, 2048)
CONFIGS = tuple(itertools.product((64, 128), (64, 128), (1, 2, 4, 8)))


def _git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def _snapshot():
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    fields = "uuid,name,pstate,clocks.sm,clocks.mem,power.draw,power.limit,clocks_event_reasons.active"
    raw = subprocess.check_output(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"], text=True)
    uuid = str(props.uuid)
    if not uuid.startswith("GPU-"):
        uuid = "GPU-" + uuid
    rows = [line for line in raw.splitlines() if line.startswith(uuid)]
    if len(rows) != 1:
        raise RuntimeError(f"could not identify physical GPU {uuid} in nvidia-smi")
    return dict(fields=fields.split(","), values=[v.strip() for v in rows[0].split(",")])


def _reference_weight(recipe, n, k):
    source = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.25
    if recipe == "nvfp4":
        g = (2688.0 / source.abs().amax().float()).reshape(1)
        codes, sf = quantize_grouped_nvfp4_torch(source[None], torch.tensor([n], device="cuda"), g)
        packed = blockscaled.pack_weight(codes[:, :, 0], sf, recipe=recipe,
                                         global_scale=g, global_scale_kind="reciprocal")
        code = codes[:, :, 0]
        unpacked = torch.stack((code & 15, code >> 4), -1).reshape(n, k).long()
        lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6], device="cuda")
        physical = _a16.scale_storage(sf, n, k, 16).view(torch.float8_e4m3fn)
        compact = physical.view((n + 127) // 128, (k // 16 + 3) // 4, 32, 4, 4)
        compact = compact.permute(0, 3, 2, 1, 4).reshape(-1, k // 16)[:n].float()
        local = lut[unpacked] * compact.repeat_interleave(16, 1)
        return packed, local, 1.0 / g
    quant = quantize_mxfp8_rows_torch(source)
    packed = blockscaled.pack_weight(quant.values, quant.scale_rows[0])
    decoded = dequantize_mxfp8_rows_torch(quant.values, quant.scale_rows)
    return packed, decoded.float(), torch.ones(1, device="cuda")


def _check(result, reference, label):
    if not torch.isfinite(result).all() or not torch.isfinite(reference).all():
        raise RuntimeError(f"{label}: nonfinite output")
    if not torch.count_nonzero(result) or not torch.count_nonzero(reference):
        raise RuntimeError(f"{label}: zero output")
    relative = float(torch.linalg.vector_norm(result.float() - reference.float())
                     / torch.linalg.vector_norm(reference.float()))
    cosine = float(torch.nn.functional.cosine_similarity(result.float().flatten(), reference.float().flatten(), dim=0))
    if relative > 0.005 or cosine < 0.9999:
        raise RuntimeError(f"{label}: relative_l2={relative}, cosine={cosine}")
    return dict(relative_l2=relative, cosine=cosine)


def _capture(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with kernel_resolution_guard('precision benchmark capture'), torch.cuda.graph(graph):
        fn()
    graph._b12x_benchmark_call = fn
    return graph


def _prepared_call(session, source, weight, output, config, *, activation_global_scale=None):
    """Prepare the production public call with fixed caller-owned scratch."""
    query = blockscaled.query_from_call(
        source, weight, activation_mode=config.mode, out=output,
        activation_global_scale=activation_global_scale,
    )
    scratch = None
    if output is not None:
        query = replace(query, workspace_form="provided")
        scratch = torch.empty(_workspace_bytes(query, config), device=source.device, dtype=torch.uint8)
        query = replace(query, workspace_nbytes=scratch.numel())
    plan = blockscaled.plan(query, override=config)
    values, scales, global_scale, _ = _a16._weight_parts(weight)
    session.prepare((plan.request(
        name=f"precision_{config.mode}",
        prepare_call=lambda state: PreparedCall(
            run=lambda: state.run(source, values, scales, global_scale,
                                  activation_scale=activation_global_scale,
                                  out=output, workspace=scratch),
            owners=(source, values, scales, global_scale, output, scratch),
        ),
    ),))

    def call():
        return blockscaled.mm(source, weight, plan=plan, out=output, workspace=scratch,
                              activation_global_scale=activation_global_scale)

    return call, scratch


def _checked_graph(call, reference, label):
    _check(call(), reference, label)
    captured = []

    def invoke():
        captured[:] = [call()]

    graph = _capture(invoke)
    # A poisoned captured output proves replay writes real, finite results.
    captured[0].fill_(float("nan"))
    graph.replay()
    correctness = _check(captured[0], reference, label + " replay")
    return graph, correctness, captured[0]


def _warmup(graphs, warmup, flush):
    for _ in range(warmup):
        for graph in graphs.values():
            if flush is not None:
                flush()
            graph.replay()
    torch.cuda.synchronize()


def _paired(graphs, iters, flush):
    from b12x.preparation import PreparedCall
    from b12x.testing.benchmark import measure_calls
    calls = {name: PreparedCall(run=graph._b12x_benchmark_call, produce=lambda: None)
             for name, graph in graphs.items()}
    result = measure_calls(calls, warmup=0, samples=iters,
                           eviction=flush or (lambda: None), adaptive_repeats=False)
    raw = {name: result.raw_samples(name) for name in calls}
    return [dict(zip(raw, row, strict=True)) for row in zip(*raw.values(), strict=True)]


def _ratio_interval(pairs, candidate, baseline):
    rng = random.Random(42)
    ratios = []
    for _ in range(1000):
        selected = rng.choices(pairs, k=len(pairs))
        ratios.append(statistics.median(p[candidate] for p in selected)
                      / statistics.median(p[baseline] for p in selected))
    ratios.sort()
    return [ratios[25], ratios[974]]


def _clock_checks(before, after):
    a, b = (dict(zip(row["fields"], row["values"], strict=True)) for row in (before, after))
    if a["name"] == b["name"] == "NVIDIA GB10":
        checks = {
            "physical_identity": a["uuid"] == b["uuid"],
            "p0": a["pstate"] == b["pstate"] == "P0",
            "sm_clock_delta_le_30mhz": abs(float(a["clocks.sm"]) - float(b["clocks.sm"])) <= 30,
            "throttle_mask_zero": all(int(row["clocks_event_reasons.active"], 16) == 0 for row in (a, b)),
            "memory_clock": a["clocks.mem"] == b["clocks.mem"],
        }
        return dict(checks=checks, valid=all(checks.values()),
                    memory_clock_reported=a["clocks.mem"] != "[N/A]",
                    throttle_contract="GB10 diagnostic; zero throttle mask, P0, SM delta <=30MHz; NVML may not report memory clocks")
    checks = {
        "physical_identity": a["uuid"] == b["uuid"],
        "p1": a["pstate"] == b["pstate"] == "P1",
        "memory_clock": a["clocks.mem"] == b["clocks.mem"],
        "sm_clock_delta_le_30mhz": abs(float(a["clocks.sm"]) - float(b["clocks.sm"])) <= 30,
        "throttle_mask_0_or_4": all(int(row["clocks_event_reasons.active"], 16) in (0, 4) for row in (a, b)),
    }
    return dict(checks=checks, valid=all(checks.values()),
                throttle_contract="targeted Max-Q diagnostic; allow only 0x0/0x4, P1, stable memory clock, SM delta <=30MHz")


def _compile_state():
    from b12x._lib.compiler import compile_cache_info
    from b12x._lib.dense_gemm import _get_compiled_dense_gemm
    return dict(implementation="DenseGemmKernel", compiler=compile_cache_info(),
                dense_resolver=_get_compiled_dense_gemm.cache_info()._asdict())


def run(args, specs):
    if args.evidence is None or not args.check:
        raise ValueError("A16 evidence requires --evidence FILE and enabled correctness checks")
    if args.iters < 20 or args.warmup < 3:
        raise ValueError("A16 evidence requires at least 20 trials and 3 warmups")
    if torch.cuda.get_device_capability() not in ((12, 0), (12, 1)):
        raise ValueError("A16 evidence requires SM120/SM121")
    recipe = "nvfp4" if args.dtype == "fp4-a16" else "mxfp8"
    counts = args.batch_sizes or COUNTS
    if any(m <= 0 for m in counts):
        raise ValueError("benchmark M must be positive")
    root = pathlib.Path(__file__).resolve().parents[3]
    paths = [*sorted((root / "flashinfer/experimental/b12x/gemm/blockscaled").glob("*.py")),
             root / "benchmarks/experimental/b12x/benchmark_dense_gemm.py",
             root / "flashinfer/experimental/b12x/_lib/dense_gemm.py", root / "flashinfer/experimental/b12x/_lib/intrinsics.py", pathlib.Path(__file__).resolve()]
    manifest = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    with args.evidence.open("x") as evidence:
        def record(row):
            evidence.write(json.dumps(row) + "\n")
            evidence.flush()
        record(dict(kind="manifest", command=sys.argv, cwd=str(pathlib.Path.cwd()),
                    revision=_git("rev-parse", "HEAD"), dirty=_git("status", "--porcelain"),
                    source_sha256=manifest, timestamp=time.time(), recipe=recipe,
                    profile=args.profile, specs=specs,
                    torch=torch.__version__, triton=importlib.metadata.version("triton"),
                    cutlass=importlib.metadata.version("nvidia-cutlass-dsl"),
                    device=_snapshot(),
                    ratio="a16_us / baseline_us; lower is faster for b12x A16", status="research-only"))
        flush = make_l2_flush_fn(enabled=args.flush_l2, bytes_hint=args.l2_flush_bytes)
        for name, k, n, _ in specs:
            torch.manual_seed(42)
            weight, local_weight, multiplier = _reference_weight(recipe, n, k)
            for m in counts:
                with PreparationSession(device="cuda", autotune=False, compile_workers=2) as session:
                    _run_case(args, name, recipe, m, n, k, weight, local_weight, multiplier,
                              session, flush, record)


def _run_case(args, name, recipe, m, n, k, weight, local_weight, multiplier,
              session, flush, record):
    source = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.25
    options = (dict(activation_global_scale=(2688.0 / source.abs().amax().float()).reshape(1))
               if recipe == "nvfp4" else {})
    reference = (source.float() @ local_weight.to(torch.bfloat16).float().T) * multiplier
    qout = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    qcall, qscratch = _prepared_call(
        session, source, weight, qout, blockscaled.BlockscaledConfig(mode="quantized"), **options,
    )
    qcall()
    # The quantized path has its own activation oracle.
    if recipe == "nvfp4":
        aq, asc = quantize_grouped_nvfp4_torch(source[None], torch.tensor([m], device="cuda"), options["activation_global_scale"])
        qref = dense_gemm((aq, asc), (weight.values[:, :, None], weight.scale_mma),
                         ab_dtype="float4_e2m1fn", sf_dtype="float8_e4m3fn", c_dtype="bfloat16",
                         sf_vec_size=16, alpha=multiplier / options["activation_global_scale"], expected_m=m)[:, :, 0]
    else:
        aq = quantize_mxfp8_rows_torch(source)
        adeq = dequantize_mxfp8_rows_torch(aq.values, aq.scale_rows).float()
        qref = adeq @ local_weight.T
    graphs, correctness, outputs = {}, {}, [qscratch]

    def add(label, call, oracle):
        graph, check, output = _checked_graph(call, oracle, label)
        graphs[label] = graph
        correctness[label] = check
        outputs.append(output)

    add("quantized", qcall, qref)
    from b12x._lib.intrinsics import as_grouped_scale_view, as_grouped_scale_view_mx
    fp4 = recipe == "nvfp4"
    sf_offset, alpha_offset, partial_offset, total = _a16._layout(m, n, k, fp4)
    storage_k = k // 2 if fp4 else k
    values = qscratch[:m * storage_k].view(torch.uint8 if fp4 else torch.float8_e4m3fn).view(m, storage_k, 1)
    sf_size = ((m + 127) // 128) * 128 * (k // (16 if fp4 else 32))
    scale = (as_grouped_scale_view if fp4 else as_grouped_scale_view_mx)(
        qscratch[sf_offset:sf_offset + sf_size].view(1, -1), m, k)
    alpha = qscratch[alpha_offset:alpha_offset + 4].view(torch.float32)
    weight_values = weight.values if fp4 else weight.weight.values
    weight_scale = weight.scale_mma if fp4 else weight.weight.scale_mma

    def gemm_call():
        return dense_gemm(
            (values, scale), (weight_values.view(n, storage_k, 1), weight_scale),
            out=qout.view(m, n, 1), alpha=alpha, ab_dtype="float4_e2m1fn" if fp4 else "float8_e4m3fn",
            sf_dtype="float8_e4m3fn" if fp4 else "float8_e8m0fnu", c_dtype="bfloat16",
            sf_vec_size=16 if fp4 else 32, expected_m=m,
            _split_k_workspace=qscratch[partial_offset:total].view(torch.float32))[:, :, 0]

    add("quantized_gemm_only", gemm_call, qref)
    if recipe == "mxfp8":
        existing, _ = _prepared_call(
            session, source, weight, None, blockscaled.BlockscaledConfig(mode="quantized"),
        )
        add("existing_quantized", existing, qref)
        if m <= 8:
            fused_out = torch.empty(m, n, 1, device="cuda", dtype=torch.bfloat16)
            fused_scratch = torch.empty(2 * m * n, device="cuda", dtype=torch.float32)
            def fused_call():
                return dense_gemm_fused_quant_a(source, weight.weight.values[:, :, None],
                                    weight.weight.scale_mma, out=fused_out, expected_m=m,
                                    _split_k_workspace=fused_scratch)[:, :, 0]
            add("fused_mxfp8", fused_call, qref)
            outputs.extend((fused_out, fused_scratch))
    configurations = (CONFIGS if args.tune_a16 else
                      (tuple(args.a16_config) if args.a16_config is not None
                       else (64, 64, 1),))
    a16_names = []
    for config in configurations:
        label = "a16_" + "_".join(map(str, config))
        output = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
        call, scratch = _prepared_call(
            session, source, weight, output,
            blockscaled.BlockscaledConfig(mode="a16", tile_n=config[0], tile_k=config[1], split_k=config[2]),
        )
        add(label, call, reference)
        outputs.extend((output, scratch))
        a16_names.append(label)
    session.freeze()
    _warmup(graphs, args.warmup, flush)
    before = _snapshot()
    pairs = _paired(graphs, args.iters, flush)
    after = _snapshot()
    medians = {key: statistics.median(p[key] for p in pairs) for key in graphs}
    winner = min(a16_names, key=medians.get)
    baselines = [key for key in graphs if key not in a16_names and key != "quantized_gemm_only"]
    ratios = {key: medians[winner] / medians[key] for key in baselines}
    intervals = {key: _ratio_interval(pairs, winner, key) for key in baselines}
    record(dict(kind="measurement", name=name, m=m, n=n, k=k, recipe=recipe,
                l2="flushed" if args.flush_l2 else "warm", correctness=correctness,
                snapshot_before=before, snapshot_after=after, samples_us=pairs,
                medians_us=medians, winning_candidate=winner, ratios=ratios,
                ratio_ci95=intervals, clock_validation=_clock_checks(before, after),
                compile_state=_compile_state(), requires_independent_confirmation=True))
    print(f"{recipe} {name} M={m}: {winner} {medians[winner]:.3f} us; ratios={ratios}", flush=True)
    graphs.clear()
    outputs.clear()
