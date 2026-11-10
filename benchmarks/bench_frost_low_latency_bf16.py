# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public BF16 MoE GPU latency with an independently frozen tactic selection.

First run with --select. Reuse its JSON via --selection for normal and --reverse
processes. Native failures remain explicit exclusions; this is not serving E2E.
"""

import argparse
import ast
import hashlib
from importlib.metadata import version
import inspect
import json
from pathlib import Path
from statistics import median
import traceback

import torch

from flashinfer.autotuner import autotune
from flashinfer.fused_moe import (
    BackendOptions,
    CudnnFrostBf16Config,
    CudnnGroupedGemmBf16Config,
    CuTileBf16Config,
    CutlassBf16Config,
    ExecutionConfig,
    ExpertConfig,
    FrostLowLatencyConfig,
    GELU,
    GeGLU,
    GeGLUTanh,
    Identity,
    MoEActivationPack,
    MoEConfig,
    MoELayer,
    MoEWeightPack,
    PowLU,
    PrimsTsConfig,
    QuantConfig,
    QuantFormat,
    ReLU,
    ReLU2,
    RoutingConfig,
    SiLU,
    SiTU,
    SwiGLU,
    SwiGLUStep,
    TrtllmBf16Config,
)

ACTIVATIONS = {
    "swiglu": SwiGLU,
    "situ": SiTU,
    "geglu": GeGLU,
    "geglu_tanh": GeGLUTanh,
    "swiglu_step": SwiGLUStep,
    "powlu": PowLU,
    "gelu": GELU,
    "relu": ReLU,
    "relu2": ReLU2,
    "silu": SiLU,
    "identity": Identity,
}


def activate(up, gate, name):
    if name == "powlu":
        positive = gate > 0
        safe = torch.where(positive, gate, torch.ones_like(gate))
        exponent = PowLU().m / (safe.sqrt() + 1.0)
        return up * torch.where(
            positive,
            safe.pow(exponent) * torch.sigmoid(safe),
            torch.nn.functional.silu(gate),
        )
    if name == "swiglu":
        return up * torch.nn.functional.silu(gate)
    if name == "situ":
        return (
            4 * torch.tanh(gate / 4) * torch.sigmoid(gate) * (25 * torch.tanh(up / 25))
        )
    if name == "swiglu_step":
        limit = SwiGLUStep().limit
        return torch.nn.functional.silu(gate).clamp_max(limit) * up.clamp(-limit, limit)
    if name in ("geglu", "geglu_tanh"):
        return up * torch.nn.functional.gelu(
            gate, approximate="tanh" if name == "geglu_tanh" else "none"
        )
    if name == "gelu":
        return torch.nn.functional.gelu(up)
    if name in ("relu", "relu2"):
        return up.relu().square() if name == "relu2" else up.relu()
    if name == "silu":
        return torch.nn.functional.silu(up)
    assert name == "identity"
    return up


def check(output, reference):
    diff = output.double() - reference.double()
    l2 = float(diff.norm() / reference.double().norm())
    maximum = float(diff.abs().max() / reference.abs().max())
    assert torch.isfinite(output).all() and l2 < 0.01 and maximum < 0.03, (
        l2,
        maximum,
    )
    return {"relative_l2": l2, "normalized_max": maximum}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--select", action="store_true")
    mode.add_argument("--selection", type=Path)
    p.add_argument("--reverse", action="store_true")
    p.add_argument("--tokens", type=int, choices=(1, 2, 4, 8), default=1)
    p.add_argument("--hidden", type=int, default=2048)
    p.add_argument("--intermediate", type=int, default=512)
    p.add_argument("--experts", type=int, default=512)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--activation", choices=tuple(ACTIVATIONS), default="swiglu")
    p.add_argument(
        "--timer", choices=("cuda-event", "globaltimer"), default="cuda-event"
    )
    p.add_argument("--cache-mode", choices=("warm", "cold"), default="warm")
    p.add_argument("--expert-reuse", choices=("disjoint", "shared"), default="disjoint")
    p.add_argument(
        "--backends",
        nargs="+",
        choices=(
            "trtllm",
            "cutlass",
            "prims_ts",
            "cudnn_grouped",
            "cutile",
            "cudnn_frost",
        ),
        default=[
            "trtllm",
            "cutlass",
            "prims_ts",
            "cudnn_grouped",
            "cutile",
            "cudnn_frost",
        ],
    )
    a = p.parse_args()
    assert 1 <= a.top_k <= a.experts
    assert len(set(a.backends)) == len(a.backends)
    if a.expert_reuse == "disjoint":
        assert a.tokens * a.top_k <= a.experts
    import flashinfer.fused_moe as moe

    source = Path(moe.__file__).parent
    digest = hashlib.sha256()
    for path in sorted(source.rglob("*.py")):
        digest.update(str(path.relative_to(source)).encode())
        digest.update(path.read_bytes())
    digest.update(Path(__file__).read_bytes())
    identity = {
        key: value
        for key, value in vars(a).items()
        if key not in ("output", "select", "selection", "reverse")
    }
    cc = torch.cuda.get_device_capability()
    identity.update(
        source_sha256=digest.hexdigest(),
        compute_capability=cc,
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        cutlass_dsl=version("nvidia-cutlass-dsl"),
        cuda_bindings=version("cuda-bindings"),
        cudnn_backend=torch.backends.cudnn.version(),
    )
    # JSON normalization preserves source tactic types separately via repr.
    identity = json.loads(json.dumps(identity))
    frozen = json.loads(a.selection.read_text()) if a.selection else None
    if frozen:
        assert frozen["identity"] == identity, (
            "Selection source/environment/workload mismatch"
        )
        assert frozen["mode"] == "selection"
    torch.manual_seed(1092)
    t, h, i, e, k = a.tokens, a.hidden, a.intermediate, a.experts, a.top_k
    activation = ACTIVATIONS[a.activation]()
    x = torch.randn(t, h, device="cuda").bfloat16()
    w1 = (
        torch.randn(e, (2 if activation.is_gated else 1) * i, h, device="cuda") / h**0.5
    ).bfloat16()
    w2 = (torch.randn(e, h, i, device="cuda") / i**0.5).bfloat16()
    base_ids = torch.arange(t * k, device="cuda", dtype=torch.int32).reshape(t, k)
    if a.expert_reuse == "shared":
        base_ids = base_ids % k
    ids = base_ids.clone()
    scores = torch.softmax(torch.randn(t, k, device="cuda"), -1).bfloat16().float()
    act = MoEActivationPack(x, None, ids, scores)
    quant = QuantConfig(weight=QuantFormat.BF16, activation=QuantFormat.BF16)

    def reference():
        result = torch.zeros_like(x, dtype=torch.float32)
        for token in range(t):
            for slot, expert in enumerate(ids[token].tolist()):
                dot = (w1[expert].double() @ x[token].double()).float()
                up, gate = dot.chunk(2) if activation.is_gated else (dot, None)
                mid = activate(up, gate, a.activation).bfloat16()
                row = (w2[expert].double() @ mid.double()).bfloat16().float()
                result[token] = result[token] + row * scores[token, slot]
        return result.bfloat16()

    flush = None
    if a.cache_mode == "cold":
        from cuda.bindings import driver

        err, capacity = driver.cuDeviceGetAttribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, 0
        )
        assert err == driver.CUresult.CUDA_SUCCESS and capacity > 0
        flush = torch.zeros(
            max(512 * 1024**2, capacity * 2), device="cuda", dtype=torch.uint8
        )

    def capture(fn):
        fn()
        graph = torch.cuda.CUDAGraph()
        if a.timer == "globaltimer":
            times = torch.empty((16, 2), device="cuda", dtype=torch.int64)
            starts = [times[j, :1] for j in range(16)]
            ends = [times[j, 1:] for j in range(16)]
            stamp(starts[0])
            with torch.cuda.graph(graph):
                for start, end in zip(starts, ends, strict=True):
                    if flush is not None:
                        flush.add_(1)
                    stamp(start)
                    output = fn()
                    stamp(end)
            return graph, times, output
        events = [
            (
                torch.cuda.Event(enable_timing=True, external=True),
                torch.cuda.Event(enable_timing=True, external=True),
            )
            for _ in range(16)
        ]
        with torch.cuda.graph(graph):
            for start, end in events:
                if flush is not None:
                    flush.add_(1)
                start.record()
                output = fn()
                end.record()
        return graph, events, output

    def sample(record):
        graph, events, _ = record
        graph.replay()
        if a.timer == "globaltimer":
            torch.cuda.synchronize()
            intervals = ((events[:, 1] - events[:, 0]).double() / 1000).cpu().tolist()
            assert min(intervals) > 0, intervals
            return median(intervals)
        events[-1][1].synchronize()
        intervals = [start.elapsed_time(end) * 1000 for start, end in events]
        assert min(intervals) > 0, intervals
        return median(intervals)

    if a.timer == "globaltimer":
        from flashinfer.utils import get_globaltimer_kernel

        stamp = get_globaltimer_kernel()
        if stamp is None:
            raise RuntimeError("globaltimer is unavailable")

    configs = {
        "frost": (FrostLowLatencyConfig(max_num_tokens=8), "frost_low_latency"),
        "trtllm": (TrtllmBf16Config(), "trtllm_bf16_routed"),
        "cutlass": (CutlassBf16Config(), "cutlass_bf16"),
        "prims_ts": (PrimsTsConfig(), "prims_ts"),
        "cudnn_grouped": (CudnnGroupedGemmBf16Config(), "cudnn_grouped_gemm_bf16"),
        "cutile": (CuTileBf16Config(), "cutile_bf16"),
        "cudnn_frost": (CudnnFrostBf16Config(), "cudnn_frost_bf16"),
    }
    result = dict(
        identity=identity,
        mode="selection" if a.select else "frozen_validation",
        selection=str(a.selection) if frozen else None,
        reverse=a.reverse,
        selected_tactics={},
        cache_tuning={},
        excluded={},
        checks=[],
        scope=f"BF16 public routed-MoE {a.timer}-delimited individual-call GPU time; includes adapters/combine, excludes host enqueue, preparation, cache eviction, routing selection, shared experts, communication and serving.",
        numerical_contract="BF16 intermediate/expert output, packed BF16 routing scores, FP32 combine; cross-backend checks use 1% relative L2 / 3% normalized maximum, not bitwise equivalence.",
        cache_eviction_bytes=flush.numel() if flush is not None else 0,
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        a.output.write_text(json.dumps(result, indent=2) + "\n")

    if a.timer == "globaltimer":
        empty = capture(lambda: None)
        for _ in range(3):
            sample(empty)
        result["empty_interval_us"] = [sample(empty) for _ in range(20)]
        del empty

    calls = {}
    ref = reference()
    for name in ["frost", *a.backends]:
        if frozen and name in frozen["excluded"]:
            result["excluded"][name] = frozen["excluded"][name]
            continue
        try:
            backend, key = configs[name]
            if not backend.supported(cc[0] * 10 + cc[1]):
                raise NotImplementedError(
                    f"{type(backend).__name__} does not admit SM{cc[0]}{cc[1]}"
                )
            prepare = (
                backend.prepare_bf16_weights
                if name == "frost"
                else backend.prepare_weights
            )
            kwargs = dict(
                quant=quant,
                num_local_experts=e,
                hidden_size=h,
                intermediate_size=i,
                activation=activation,
            )
            kwargs = {
                key: value
                for key, value in kwargs.items()
                if key in inspect.signature(prepare).parameters
            }
            weights = MoEWeightPack()
            weights.prepare_for(key, prepare(w1, w2, **kwargs))
            cfg = MoEConfig(
                RoutingConfig(e, k),
                quant,
                ExpertConfig(i),
                activation=activation,
                backend=BackendOptions((backend,)),
                execution=ExecutionConfig(tune_max_num_tokens=t),
            )
            layer = MoELayer(cfg)
            with autotune(False):
                layer(act, weights)
            assert len(layer._winners) == 1
            bucket, (runner, initial) = next(iter(layer._winners.items()))
            fn = lambda layer=layer, weights=weights: layer(act, weights)
            if frozen:
                winner = ast.literal_eval(frozen["selected_tactics"][name])
            else:
                with autotune(True):
                    advertised = runner.get_valid_tactics(
                        runner.pack_inputs(act, weights), None
                    )
                candidates = []
                for tactic in [initial, -1, *advertised]:
                    if tactic not in candidates:
                        candidates.append(tactic)
                print(
                    name,
                    "ranking",
                    len(candidates),
                    "advertised/default tactics",
                    flush=True,
                )
                ranked = []
                for tactic in candidates:
                    layer._winners[bucket] = (runner, tactic)
                    record = capture(fn)
                    values = [sample(record) for _ in range(5)]
                    check(record[2], ref)
                    ranked.append(dict(tactic=repr(tactic), median_us=median(values)))
                    del record
                    if len(ranked) % 64 == 0:
                        print(name, len(ranked), "/", len(candidates), flush=True)
                winner = ast.literal_eval(
                    min(ranked, key=lambda row: row["median_us"])["tactic"]
                )
                result["cache_tuning"][name] = ranked
            layer._winners[bucket] = (runner, winner)
            check(fn(), ref)
            result["selected_tactics"][name] = repr(winner)
            calls[name] = fn
            default = MoELayer(cfg)
            default(act, weights)
            for b, (r, _) in list(default._winners.items()):
                default._winners[b] = (r, -1)
            calls[name + "_default"] = lambda layer=default, weights=weights: layer(
                act, weights
            )
            check(calls[name + "_default"](), ref)
            if name == "frost":
                if frozen:
                    previous = ast.literal_eval(frozen["frost_fixed_fc2_tactic"])
                else:
                    fixed_fc2 = []
                    for row in ranked:
                        tactic = ast.literal_eval(row["tactic"])
                        if tactic == -1 or (
                            isinstance(tactic, tuple)
                            and (len(tactic) == 2 or tactic[2] == 4)
                        ):
                            fixed_fc2.append(row)
                    previous = ast.literal_eval(
                        min(fixed_fc2, key=lambda row: row["median_us"])["tactic"]
                    )
                result["frost_fixed_fc2_tactic"] = repr(previous)
                control = MoELayer(cfg)
                control(act, weights)
                for b, (r, _) in list(control._winners.items()):
                    control._winners[b] = (r, previous)
                calls["frost_fixed_fc2"] = lambda layer=control, weights=weights: layer(
                    act, weights
                )
                check(calls["frost_fixed_fc2"](), ref)
            print(name, "selected", winner, flush=True)
        except (
            NotImplementedError,
            RuntimeError,
            ValueError,
            ImportError,
            AssertionError,
        ) as error:
            if name == "frost" or frozen:
                raise
            calls.pop(name, None)
            calls.pop(name + "_default", None)
            result["selected_tactics"].pop(name, None)
            result["excluded"][name] = dict(
                type=type(error).__name__,
                error=str(error),
                traceback=traceback.format_exc(),
            )
            print(name, "EXCLUDED", type(error).__name__, str(error), flush=True)
        save()
    assert any(name != "frost" for name in result["selected_tactics"]), (
        "No native comparison qualified"
    )
    # Separate captures of the same callable are A/A controls, not new tactics.
    for name in tuple(calls):
        calls[name + "_aa"] = calls[name]
    if a.reverse:
        calls = dict(reversed(list(calls.items())))
    graphs = {name: capture(fn) for name, fn in calls.items()}
    for step in range(3):
        x.copy_(torch.randn_like(x))
        ids.copy_((base_ids + step * 7) % e)
        scores.copy_(torch.softmax(torch.randn_like(scores), -1).bfloat16().float())
        for graph, _, _ in graphs.values():
            graph.replay()
        ref = reference()
        result["checks"].append(
            {name: check(record[2], ref) for name, record in graphs.items()}
        )
    samples = {name: [] for name in calls}
    for order in (list(calls), list(reversed(calls))):
        for name in order:
            for _ in range(4):
                graphs[name][0].replay()
            samples[name].extend(sample(graphs[name]) for _ in range(9))
    result["post_timing_checks"] = {
        name: check(record[2], ref) for name, record in graphs.items()
    }
    result["samples_us"] = samples
    result["median_us"] = {name: median(values) for name, values in samples.items()}
    result["completed"] = True
    save()
    print(json.dumps(result["median_us"]), flush=True)


if __name__ == "__main__":
    main()
