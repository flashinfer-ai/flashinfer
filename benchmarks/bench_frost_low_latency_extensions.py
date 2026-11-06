# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Quantized Frost activation/dtype comparisons using CUDA Graph GPU time.

Select with --search full --select, then run independent processes with
--selection FILE (normal and --reverse). Preparation and input quantization
are outside timing. Device adapters, projections and combine are included.
Backend-specific rounding references are explicit. Native CUTLASS/cuDNN NVFP4
references use the standalone quantizer to reproduce its approximate reciprocal;
Frost's quantizer has an independent reference. This is not a model test.
"""

import argparse
import ast
import hashlib
import inspect
import json
import os
import traceback
from pathlib import Path
from statistics import median
import torch
import flashinfer
from importlib.metadata import version
from bench_frost_low_latency_bf16 import ACTIVATIONS, activate, check
from bench_frost_low_latency import _decode_mx
from flashinfer import fp4_quantize, mxfp8_quantize
from flashinfer.autotuner import autotune
from flashinfer.fused_moe import (
    BackendOptions,
    CakeWarpDecodeConfig,
    CudnnFrostMxfp8Config,
    CudnnFrostMxfp8Mxfp4Config,
    CudnnFrostNvfp4Config,
    CudnnGroupedGemmMxfp8Config,
    CudnnGroupedGemmNvfp4Config,
    CuteDslConfig,
    CutlassMxfp8Config,
    CutlassMxfp8Mxfp4Config,
    CutlassNvfp4Config,
    ExecutionConfig,
    ExpertConfig,
    FrostLowLatencyConfig,
    MoEActivationPack,
    MoEConfig,
    MoELayer,
    MoEWeightPack,
    PrimsTsConfig,
    QuantConfig,
    QuantFormat,
    RoutingConfig,
    TrtllmFp4Config,
    TrtllmFp8BlockConfig,
)
from flashinfer.fused_moe.api import RoutingInputMode


def _fatal_cuda_error(exc):
    return any(
        token in str(exc).lower()
        for token in ("illegal memory", "device-side assert", "misaligned address")
    )


def _check_named(name, output, reference):
    try:
        return check(output, reference)
    except AssertionError as exc:
        raise AssertionError(f"{name}: {exc}") from exc


def _linear_scale(scale):
    e, rows, groups = scale.shape
    return (
        scale.view(torch.uint8)
        .reshape(e, rows // 128, groups // 4, 32, 4, 4)
        .permute(0, 1, 4, 3, 2, 5)
        .reshape(e, rows, groups)
        .view(torch.float8_e4m3fn)
        .float()
    )


def _decode(q, scale):
    lut = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device=q.device,
        dtype=torch.float64,
    )
    values = torch.stack((lut[(q & 15).long()], lut[(q >> 4).long()]), -1).flatten(-2)
    return (values.reshape(*values.shape[:-1], -1, 16) * scale[..., None]).flatten(-2)


def _round_mid(value, global_scale, *, unrounded_scale=False):
    groups = value.reshape(-1, 16)
    raw_scale = groups.abs().amax(-1) * (1.0 / 6.0) * global_scale
    scale = raw_scale.to(torch.float8_e4m3fn).float()
    # Prims-TS normalizes by FP32 amax/6 and only rounds the stored scale.
    # Frost/TRTLLM normalize by the already rounded E4M3 scale.
    denominator = raw_scale if unrounded_scale else scale
    scaled = torch.where(
        denominator[:, None] > 0, groups * global_scale / denominator[:, None], 0.0
    )
    levels = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=value.device)
    distance = (scaled.abs()[..., None] - levels).abs()
    code = distance.argmin(-1)
    even = distance[..., ::2]
    code = torch.where(even.amin(-1) == distance.amin(-1), even.argmin(-1) * 2, code)
    return (levels[code] * scaled.sign() * scale[:, None]).flatten()


def main():
    os.environ["FLASHINFER_NVFP4_4OVER6"] = "0"
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument(
        "--weight-format", choices=["mxfp4", "mxfp8", "nvfp4"], default="mxfp4"
    )
    p.add_argument("--activation", choices=list(ACTIVATIONS), default="situ")
    p.add_argument("--tokens", type=int, default=1)
    p.add_argument("--hidden", type=int, default=2048)
    p.add_argument("--intermediate", type=int, default=1408)
    p.add_argument("--experts", type=int, default=64)
    p.add_argument("--top-k", type=int, default=6)
    p.add_argument(
        "--backends",
        nargs="+",
        default=[
            "trtllm",
            "cute_dsl",
            "cutlass",
            "prims_ts",
            "cudnn_grouped",
            "cudnn_frost",
        ],
    )
    p.add_argument("--search", choices=["defaults", "full"], default="defaults")
    p.add_argument("--selection")
    p.add_argument(
        "--select",
        action="store_true",
        help="Save discovery results without qualification measurements",
    )
    p.add_argument("--reverse", action="store_true")
    p.add_argument("--cold", action="store_true")
    a = p.parse_args()
    t, h, i, e, k = a.tokens, a.hidden, a.intermediate, a.experts, a.top_k
    assert h % 128 == i % 128 == 0 and t * k <= e
    torch.set_num_threads(1)
    torch.manual_seed(4321)
    activation = ACTIVATIONS[a.activation]()
    fmt = {
        "mxfp4": QuantFormat.MXFP4,
        "mxfp8": QuantFormat.MXFP8,
        "nvfp4": QuantFormat.NVFP4,
    }[a.weight_format]
    quant = QuantConfig(
        weight=fmt,
        activation=QuantFormat.NVFP4 if fmt == QuantFormat.NVFP4 else QuantFormat.MXFP8,
    )
    qconfig = TrtllmFp8BlockConfig if fmt == QuantFormat.MXFP8 else TrtllmFp4Config
    w1 = (
        torch.randn(e, (2 if activation.is_gated else 1) * i, h, device="cuda") / h**0.5
    ).bfloat16()
    w2 = (torch.randn(e, h, i, device="cuda") / i**0.5).bfloat16()
    ones = torch.ones(1, device="cuda")

    def pack(w):
        rows, cols = w.shape[-2:]
        if fmt == QuantFormat.MXFP4:
            q, s = fp4_quantize(
                w.view(-1, cols),
                ones,
                sf_vec_size=32,
                sf_use_ue8m0=True,
                is_sf_swizzled_layout=False,
            )
        else:
            q, s = mxfp8_quantize(w.view(-1, cols), is_sf_swizzled_layout=False)
        return q.reshape(e, rows, -1), s.view(torch.uint8).reshape(e, rows, cols // 32)

    if fmt == QuantFormat.NVFP4:
        canonical = FrostLowLatencyConfig.prepare_nvfp4_weights(
            w1, w2, activation=activation
        )
    else:
        q1, s1 = pack(w1)
        q2, s2 = pack(w2)
    x = torch.randn(t, h, device="cuda").bfloat16()
    q, sf = qconfig.prepare_activations(x, quant=quant)
    ids = torch.arange(t * k, device="cuda", dtype=torch.int32).reshape(t, k)
    scores = torch.softmax(torch.randn(t, k, device="cuda"), -1).bfloat16().float()
    cake_scores = scores.bfloat16()
    act = MoEActivationPack(q, sf, ids, scores)

    def reference(name):
        name = name.removesuffix("_aa").removesuffix("_default")
        out = torch.zeros(t, h, device="cuda", dtype=torch.float32)
        dx = (
            _decode(q, sf.view(torch.float8_e4m3fn).float())
            if fmt == QuantFormat.NVFP4
            else _decode_mx(q, sf)
        )
        # CUTLASS/cuDNN use BF16 boundaries; Prims truncates MX exponents.
        rounded = name in ("cutlass", "cudnn_grouped", "cudnn_frost")
        for token in range(t):
            for slot, expert in enumerate(ids[token].tolist()):
                dw1 = (
                    _decode(
                        canonical["fc1_expert_weights"][expert],
                        _linear_scale(
                            canonical["fc1_weight_block_scale"][expert : expert + 1]
                        )[0],
                    )
                    if fmt == QuantFormat.NVFP4
                    else _decode_mx(q1[expert], s1[expert])
                )
                dot = (dx[token].double() @ dw1.double().T).float()
                if rounded:
                    dot = dot.bfloat16().float()
                up, gate = dot.chunk(2) if activation.is_gated else (dot, None)
                mid = activate(up, gate, a.activation)
                if rounded:
                    mid = mid.bfloat16().float()
                if fmt == QuantFormat.NVFP4:
                    if name in ("frost", "cutlass", "cudnn_grouped", "cudnn_frost"):
                        mid = mid.bfloat16().float()
                    if name in ("cutlass", "cudnn_grouped"):
                        # Match the native BF16-to-FP4 fast-math quantizer,
                        # including its treatment of exact E2M1 midpoints.
                        mq, ms = fp4_quantize(
                            mid.bfloat16()[None, :],
                            ones,
                            sf_vec_size=16,
                            is_sf_swizzled_layout=False,
                        )
                        dq = _decode(mq, ms.view(torch.float8_e4m3fn).float())[0]
                    else:
                        dq = _round_mid(mid, ones, unrounded_scale=name == "prims_ts")
                    dw2 = _decode(
                        canonical["fc2_expert_weights"][expert],
                        _linear_scale(
                            canonical["fc2_weight_block_scale"][expert : expert + 1]
                        )[0],
                    )
                else:
                    g = mid.reshape(-1, 32)
                    maximum = (
                        g.abs().amax(-1).clamp_min(torch.finfo(torch.float32).tiny)
                    )
                    exponent = (
                        torch.floor(torch.log2(maximum)) - 8
                        if name == "prims_ts"
                        else torch.ceil(torch.log2(maximum / 448))
                    ).clamp(-127, 127)
                    scale = torch.exp2(exponent)[:, None]
                    dq = (
                        (g / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float()
                        * scale
                    ).flatten()
                    dw2 = _decode_mx(q2[expert], s2[expert])
                row = (dq.double() @ dw2.double().T).bfloat16()
                if name == "frost" and fmt != QuantFormat.NVFP4:
                    out[token] = (
                        out[token].double()
                        + row.double() * scores[token, slot].double()
                    ).float()
                else:
                    out[token] = out[token] + row.float() * scores[token, slot]
        return out.bfloat16()

    configs = {
        "frost": (FrostLowLatencyConfig(max_num_tokens=8), "frost_low_latency"),
        "trtllm": (
            qconfig(),
            "trtllm_fp4_routed" if fmt == QuantFormat.MXFP4 else "trtllm_fp8_block",
        ),
        "cute_dsl": (CuteDslConfig(), "cute_dsl"),
        "cutlass": (
            (
                CutlassMxfp8Mxfp4Config()
                if fmt == QuantFormat.MXFP4
                else CutlassMxfp8Config()
            ),
            "cutlass_mxfp8_mxfp4" if fmt == QuantFormat.MXFP4 else "cutlass_mxfp8",
        ),
        "prims_ts": (PrimsTsConfig(), "prims_ts"),
        "cudnn_grouped": (CudnnGroupedGemmMxfp8Config(), "cudnn_grouped_gemm_mxfp8"),
        "cudnn_frost": (
            (
                CudnnFrostMxfp8Mxfp4Config()
                if fmt == QuantFormat.MXFP4
                else CudnnFrostMxfp8Config()
            ),
            "cudnn_frost_mxfp8_mxfp4"
            if fmt == QuantFormat.MXFP4
            else "cudnn_frost_mxfp8",
        ),
    }
    if fmt == QuantFormat.NVFP4:
        configs.update(
            trtllm=(TrtllmFp4Config(), "trtllm_fp4_routed"),
            cutlass=(CutlassNvfp4Config(), "cutlass_nvfp4"),
            cudnn_grouped=(CudnnGroupedGemmNvfp4Config(), "cudnn_grouped_gemm_nvfp4"),
            cudnn_frost=(CudnnFrostNvfp4Config(), "cudnn_frost_nvfp4"),
            cake=(CakeWarpDecodeConfig(), "cake"),
        )
    identity = {
        key: val
        for key, val in vars(a).items()
        if key not in ("output", "selection", "reverse", "select")
    }
    digest = hashlib.sha256(Path(__file__).read_bytes())
    for helper in (activate, _decode_mx):
        digest.update(Path(inspect.getfile(helper)).read_bytes())
    for f in sorted(
        (Path(flashinfer.__file__).resolve().parent / "fused_moe").rglob("*.py")
    ):
        digest.update(f.read_bytes())
    identity.update(
        source_sha256=digest.hexdigest(),
        gpu=torch.cuda.get_device_name(),
        gpu_uuid=str(getattr(torch.cuda.get_device_properties(0), "uuid", "unknown")),
        capability=list(torch.cuda.get_device_capability()),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        dsl=version("nvidia-cutlass-dsl"),
    )
    result = {
        "identity": identity,
        "excluded": {},
        "selected": {},
        "ranking": {},
        "candidate_count": {},
        "scope": "Complete local routed-MoE GPU time; adapters/combine included, no routing/shared experts/communication; backend-specific rounding references documented in this script.",
    }
    frozen = json.loads(Path(a.selection).read_text()) if a.selection else None
    if frozen:
        assert frozen["identity"] == identity
    outpath = Path(a.output)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    def save():
        outpath.write_text(json.dumps(result, indent=2) + "\n")

    flush = (
        torch.zeros(512 * 1024**2, device="cuda", dtype=torch.uint8) if a.cold else None
    )

    def capture(fn):
        fn()
        g = torch.cuda.CUDAGraph()
        events = [
            (
                torch.cuda.Event(enable_timing=True, external=True),
                torch.cuda.Event(enable_timing=True, external=True),
            )
            for _ in range(16)
        ]
        with torch.cuda.graph(g):
            for begin, end in events:
                if flush is not None:
                    flush.add_(1)
                begin.record()
                y = fn()
                end.record()
        return g, events, y

    def sample(rec):
        g, ev, _ = rec
        g.replay()
        ev[-1][1].synchronize()
        return median(b.elapsed_time(c) * 1000 for b, c in ev)

    calls = {}
    refs = {}
    for name in ["frost", *a.backends]:
        try:
            if frozen and name in frozen["excluded"]:
                result["excluded"][name] = frozen["excluded"][name]
                continue
            opt, key = configs[name]
            cc = torch.cuda.get_device_capability()
            if not opt.supported(cc[0] * 10 + cc[1]):
                raise NotImplementedError("architecture declined")
            cfg = MoEConfig(
                RoutingConfig(e, k),
                quant,
                ExpertConfig(i),
                activation=activation,
                backend=BackendOptions((opt,)),
                execution=ExecutionConfig(
                    enable_pdl=True if name == "cake" else None,
                    tune_max_num_tokens=t,
                ),
            )
            layer = MoELayer(cfg)
            backend_act = (
                MoEActivationPack(
                    q,
                    sf,
                    ids,
                    cake_scores,
                    routing_input_mode=RoutingInputMode.UnpackedPrecomputed,
                )
                if name == "cake"
                else act
            )
            weights = MoEWeightPack()
            if name == "frost":
                v = (
                    canonical
                    if fmt == QuantFormat.NVFP4
                    else opt.prepare_weights(q1, s1, q2, s2, activation=activation)
                )
            elif fmt == QuantFormat.NVFP4 and name == "cudnn_grouped":
                # Use the same quantized weights and unit global scales as the
                # other candidates, rather than the helper's per-expert amax.
                v = {
                    "fc1_expert_weights": canonical["fc1_expert_weights"],
                    "fc2_expert_weights": canonical["fc2_expert_weights"],
                    "fc1_weight_scale": canonical["fc1_weight_block_scale"].view(
                        torch.float8_e4m3fn
                    ),
                    "fc2_weight_scale": canonical["fc2_weight_block_scale"].view(
                        torch.float8_e4m3fn
                    ),
                    "fc1_dequant": canonical["fc1_dequant_scale"],
                    "fc2_dequant": canonical["fc2_dequant_scale"],
                }
            else:
                kw = dict(
                    quant=quant,
                    num_local_experts=e,
                    hidden_size=h,
                    intermediate_size=i,
                    activation=activation,
                )
                kw = {
                    key: val
                    for key, val in kw.items()
                    if key in inspect.signature(opt.prepare_weights).parameters
                }
                v = opt.prepare_weights(w1, w2, **kw)
            weights.prepare_for(key, v)
            with autotune(False):
                layer(backend_act, weights)
            bucket, (runner, initial) = next(iter(layer._winners.items()))
            fn = lambda layer=layer, weights=weights, backend_act=backend_act: layer(
                backend_act, weights
            )
            refs[name] = reference(name)
            if frozen:
                winner = ast.literal_eval(frozen["selected"][name])
            else:
                with autotune(True):
                    advertised = (
                        runner.get_valid_tactics(
                            runner.pack_inputs(backend_act, weights), None
                        )
                        if a.search == "full" or name == "frost"
                        else []
                    )
                candidates = list(
                    {
                        repr(tactic): tactic for tactic in [initial, -1, *advertised]
                    }.values()
                )
                print(name, "candidates", len(candidates), flush=True)
                ranked = []
                result["candidate_count"][name] = len(candidates)
                result["ranking"][name] = ranked
                for tactic in candidates:
                    layer._winners[bucket] = (runner, tactic)
                    try:
                        rec = capture(fn)
                        times = [sample(rec) for _ in range(5)]
                    except (NotImplementedError, RuntimeError, ValueError) as exc:
                        if _fatal_cuda_error(exc):
                            raise
                        result.setdefault("tactic_errors", {}).setdefault(
                            name, []
                        ).append({"tactic": repr(tactic), "error": str(exc)})
                        save()
                        continue
                    try:
                        check(rec[2], refs[name])
                    except AssertionError as exc:
                        if name == "frost":
                            raise
                        result.setdefault("tactic_correctness_failures", {}).setdefault(
                            name, []
                        ).append({"tactic": repr(tactic), "error": str(exc)})
                        save()
                        del rec
                        continue
                    ranked.append({"tactic": repr(tactic), "us": median(times)})
                    del rec
                    if len(ranked) % 64 == 0:
                        save()
                        print(name, len(ranked), "/", len(candidates), flush=True)
                result["ranking"][name] = ranked
                if not ranked:
                    raise ValueError(
                        "No candidate passed this benchmark's documented reference; "
                        "inspect tactic_errors and tactic_correctness_failures."
                    )
                winner = ast.literal_eval(min(ranked, key=lambda r: r["us"])["tactic"])
            layer._winners[bucket] = (runner, winner)
            check(fn(), refs[name])
            calls[name] = fn
            result["selected"][name] = repr(winner)
            default = MoELayer(cfg)
            with autotune(False):
                default(backend_act, weights)
            for b, (r, _) in list(default._winners.items()):
                default._winners[b] = (r, -1)
            calls[name + "_default"] = (
                lambda layer=default, weights=weights, backend_act=backend_act: layer(
                    backend_act, weights
                )
            )
            check(calls[name + "_default"](), refs[name])
            print(name, "winner", winner, flush=True)
        except (
            NotImplementedError,
            RuntimeError,
            ValueError,
            ImportError,
            AssertionError,
        ) as exc:
            if name == "frost" or frozen or _fatal_cuda_error(exc):
                raise
            calls.pop(name, None)
            calls.pop(name + "_default", None)
            result["selected"].pop(name, None)
            result["excluded"][name] = {
                "type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            print(name, "EXCLUDED", str(exc)[:350], flush=True)
        save()
    assert len(result["selected"]) > 1, "No native comparison"
    if a.select:
        result["selection_complete"] = True
        save()
        return
    for n in list(calls):
        calls[n + "_aa"] = calls[n]
    if a.reverse:
        calls = dict(reversed(list(calls.items())))
    graphs = {n: capture(fn) for n, fn in calls.items()}
    initial_inputs = [v.clone() for v in (q, sf, ids, scores)]
    result["checks"] = []
    for step in range(3):
        xn = torch.randn_like(x)
        qn, sn = qconfig.prepare_activations(xn, quant=quant)
        q.copy_(qn)
        sf.copy_(sn)
        ids.copy_(
            (
                torch.arange(t * k, device="cuda", dtype=torch.int32).view(t, k)
                + step * 7
            )
            % e
        )
        scores.copy_(torch.softmax(torch.randn_like(scores), -1).bfloat16().float())
        cake_scores.copy_(scores)
        for g, _, _ in graphs.values():
            g.replay()
        refs = {n: reference(n) for n in result["selected"]}
        result["checks"].append(
            {
                n: _check_named(
                    n, rec[2], refs[n.removesuffix("_aa").removesuffix("_default")]
                )
                for n, rec in graphs.items()
            }
        )
    # Select and qualify on the same routed inputs. Mutation checks above
    # exercise graph rebinding, but must not silently change the tuned workload.
    for value, initial_value in zip((q, sf, ids, scores), initial_inputs, strict=True):
        value.copy_(initial_value)
    cake_scores.copy_(scores)
    refs = {n: reference(n) for n in result["selected"]}
    samples = {n: [] for n in calls}
    for order in (list(calls), list(reversed(calls))):
        for n in order:
            for _ in range(4):
                graphs[n][0].replay()
            samples[n].extend(sample(graphs[n]) for _ in range(9))
    result["post_timing"] = {
        n: _check_named(n, rec[2], refs[n.removesuffix("_aa").removesuffix("_default")])
        for n, rec in graphs.items()
    }
    result["samples_us"] = samples
    result["median_us"] = {n: median(v) for n, v in samples.items()}
    result["completed"] = True
    save()
    print(json.dumps(result["median_us"]), flush=True)


if __name__ == "__main__":
    main()
