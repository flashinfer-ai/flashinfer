#!/usr/bin/env python3
"""Compare weighted MoE layer-distribution mixtures using matched NoDA/DA graphs.

DA need not win every layer: the target is lower total MoE time across a
representative mix of layer distributions. Report sum(weight * NoDA latency) /
sum(weight * DA latency), never an average of per-distribution speedups. These
synthetic single-rank measurements exclude communication and other model work.

DA and its per-exemplar baseline guard are experimental and off by default.
This benchmark explicitly compares DA off/on and honors FLASHINFER_DA_BASELINE_GUARD.
Repeat --mixture to compare layer-frequency mixtures, for example
--mixture 'ddist:1.1=20,ddist:2=30,ddist:4=50'. Weights are normalized and do not
change the --distributions tuning catalog, which must contain every component.
Use --skip-autotune with the same --cache for independent replay. --json-out
retains component timings and capture/selection evidence; --mixture-out records
weighted absolute latencies and speedups; --table-out saves dtype/token tables.

Use the routed expert's actual activation, including clamp and offset parameters:
DSV4-Pro uses --activation swiglu --swiglu-alpha 1 --swiglu-beta 0 --swiglu-limit 10;
MiniMax-M3 uses --activation swiglu --swiglu-alpha 1.702 --swiglu-beta 1 --swiglu-limit 7;
Nemotron 3 Ultra uses --activation relu2. These are synthetic routed-body comparisons,
not checkpoint or full-model benchmarks. Expert parallelism is num_experts/local_num_experts;
communication, shared experts, and latent projections are outside the measured region.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import hashlib
import json
import math
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

from flashinfer import reorder_rows_for_gated_act_gemm, shuffle_matrix_a
from flashinfer.autotuner import autotune
from flashinfer.fused_moe.da_config import DaMoeConfig
from flashinfer.fused_moe import (
    QuantConfig,
    QuantFormat,
    ReLU2,
    SwiGLU,
    TrtllmBf16Config,
    TrtllmFp4Config,
    TrtllmFp8BlockConfig,
    TrtllmFp8PerTensorConfig,
    TrtllmMxInt4Config,
    da_moe_acquire_graph_leases,
    da_moe_diagnostics,
    da_moe_release_resources,
    prims_ts_fp4_block_scale_moe,
    prims_ts_fp4_block_scale_routed_moe,
    trtllm_bf16_routed_moe,
    trtllm_fp4_block_scale_moe,
    trtllm_fp4_block_scale_routed_moe,
    trtllm_fp8_block_scale_routed_moe,
    trtllm_fp8_per_tensor_scale_routed_moe,
    trtllm_mxint4_block_scale_routed_moe,
)
from flashinfer.fused_moe.da_tuner import (
    DADistribution,
    RoutingRealizationFactory,
    RoutingRealizationKey,
)
from flashinfer.fused_moe.backends.prims_ts.bf16_op import (
    prims_ts_bf16_routed_moe,
)
from flashinfer.fused_moe.backends.prims_ts.fp8_op import (
    prims_ts_fp8_block_scale_routed_moe,
    prims_ts_fp8_per_tensor_scale_moe,
)
from flashinfer.tllm_enums import (
    ActivationType,
    DtypeTrtllmGen,
    Fp8QuantizationType,
    RoutingMethodType,
    WeightLayout,
)


PRECISIONS = (
    "nvfp4",
    "mxfp4",
    "w4a16",
    "bf16",
    "fp8_per_tensor",
    "fp8_block",
    "mxfp8",
    "mxint4",
)

# Ordinary backends participating independently in the shared DA lifecycle.
BACKENDS = ("trtllm", "prims_ts")

PRECISION_ALIASES = {
    # Name mapping for MXFP4 weights with MXFP8 activations.
    "mxfp4_mxfp8": "mxfp4",
    # Name mapping for MXFP4 weights with BF16 activations.
    "mxfp4_bf16": "w4a16",
}


@dataclass(frozen=True)
class DistributionMixture:
    """Relative layer frequencies, normalized independently of tuning profiles."""

    weights: tuple[tuple[str, float], ...]

    @property
    def name(self) -> str:
        return ", ".join(f"{100 * weight:g}% {name}" for name, weight in self.weights)


def _parse_mixture(value: str) -> DistributionMixture:
    """Parse distribution=weight pairs; accept frequencies or percentages as weights."""
    weights = {}
    try:
        for item in value.split(","):
            distribution, weight_text = item.rsplit("=", 1)
            name = DADistribution.parse(distribution).name
            weight = float(weight_text)
            if name in weights:
                raise ValueError(f"duplicate distribution {name}")
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError("mixture weights must be positive and finite")
            weights[name] = weight
        total = math.fsum(weights.values())
        if not math.isfinite(total):
            raise ValueError("mixture total must be finite")
    except (ValueError, OverflowError) as error:
        raise argparse.ArgumentTypeError(
            f"invalid mixture {value!r}: {error}; use ddist:1.1=20,ddist:2=30,ddist:4=50"
        ) from error
    return DistributionMixture(
        tuple((name, weight / total) for name, weight in weights.items())
    )


_MIXTURE_GROUP_FIELDS = (
    "backend",
    "precision",
    "num_tokens",
    "num_experts",
    "local_num_experts",
    "local_expert_offset",
    "top_k",
    "hidden_size",
    "intermediate_size",
    "activation",
    "swiglu_alpha",
    "swiglu_beta",
    "swiglu_limit",
    "execution_mode",
    "timing_protocol",
    "routing_input_mode",
    "baseline_guard_enabled",
)


def _summarize_mixtures(
    rows: list[dict[str, object]], mixtures: tuple[DistributionMixture, ...]
) -> list[dict[str, object]]:
    """Divide weighted latency sums, never average individual speedup ratios."""
    groups = {}
    for row in rows:
        key = tuple(row[field] for field in _MIXTURE_GROUP_FIELDS)
        group = groups.setdefault(key, {})
        distribution = row["distribution"]
        if distribution in group:
            raise ValueError(f"duplicate benchmark row for {distribution}")
        group[distribution] = row
    summaries = []
    for key, group in groups.items():
        for mixture in mixtures:
            missing = {name for name, _ in mixture.weights} - group.keys()
            if missing:
                raise ValueError(f"mixture is missing distributions: {sorted(missing)}")
            selected = [group[name] for name, _ in mixture.weights]
            for row in selected:
                if row["status"] != "pass" or not row["finite"]:
                    raise ValueError("cannot summarize failed numerical checks")
                if any(
                    not math.isfinite(row[f]) or row[f] <= 0
                    for f in ("noda_ms", "da_ms")
                ):
                    raise ValueError("mixture latencies must be positive and finite")
            noda_ms = math.fsum(
                weight * group[name]["noda_ms"] for name, weight in mixture.weights
            )
            da_ms = math.fsum(
                weight * group[name]["da_ms"] for name, weight in mixture.weights
            )
            summaries.append(
                {
                    **dict(zip(_MIXTURE_GROUP_FIELDS, key, strict=True)),
                    "mixture": mixture.name,
                    "distribution_weights": dict(mixture.weights),
                    "noda_ms": noda_ms,
                    "da_ms": da_ms,
                    "speedup_da_over_noda": noda_ms / da_ms,
                    "capture_policies": sorted(
                        {row["capture_policy"] for row in selected}
                    ),
                    "selected_bodies": {
                        name: group[name]["selected_body"]
                        for name, _ in mixture.weights
                    },
                    "max_abs_difference": max(
                        row["max_abs_difference"] for row in selected
                    ),
                    "status": "pass",
                }
            )
    return summaries


def _mixture_tables(rows: list[dict[str, object]]) -> str:
    """Render one dtype-by-token speedup matrix per mixture and geometry."""
    tables = {}
    geometry_fields = tuple(
        f
        for f in _MIXTURE_GROUP_FIELDS
        if f not in ("precision", "num_tokens", "routing_input_mode")
    )
    for row in rows:
        key = (
            row["mixture"],
            tuple(row["distribution_weights"].items()),
            *(row[f] for f in geometry_fields),
        )
        tables.setdefault(key, []).append(row)
    lines = [
        "Speedup = weighted NoDA time / weighted DA time; >1 favors DA.",
        "Synthetic MoE layer-frequency estimate; excludes communication and other model work.",
        "",
    ]
    for (name, *_), group in tables.items():
        first = group[0]
        tokens = list(dict.fromkeys(row["num_tokens"] for row in group))
        precisions = list(dict.fromkeys(row["precision"] for row in group))
        lookup = {(row["precision"], row["num_tokens"]): row for row in group}
        if len(lookup) != len(group):
            raise ValueError("duplicate dtype/token cells in mixture table")
        lines += [
            f"### {name}",
            "",
            f"{first['backend']}; E={first['num_experts']}, localE={first['local_num_experts']}, "
            f"K={first['top_k']}, H={first['hidden_size']}, I={first['intermediate_size']}; "
            f"activation={first['activation']}; "
            + (
                f"alpha={first['swiglu_alpha']}, beta={first['swiglu_beta']}, "
                f"limit={first['swiglu_limit']}; "
                if first["activation"] == "swiglu"
                else ""
            )
            + f"baseline guard={'on' if first['baseline_guard_enabled'] else 'off'}.",
            "",
            "| Dtype | " + " | ".join(map(str, tokens)) + " |",
            "|---|" + "---:|" * len(tokens),
        ]
        for precision in precisions:
            cells = [
                f"{lookup[precision, token]['speedup_da_over_noda']:.4f}x"
                if (precision, token) in lookup
                else "-"
                for token in tokens
            ]
            lines.append(f"| {precision} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


@dataclass(frozen=True)
class BenchmarkShape:
    """Static model geometry shared by matched NoDA and DA runs."""

    # Number of tokens in the graph-stable activation and routing tensors.
    num_tokens: int
    # Number of global routing experts.
    num_experts: int
    # Number of experts whose weights are resident on this device.
    local_num_experts: int
    # First global expert represented by the local weight shard.
    local_expert_offset: int
    # Number of distinct experts selected for each token.
    top_k: int
    # Input and finalized-output width.
    hidden_size: int
    # Per-expert FFN intermediate width.
    intermediate_size: int
    # DeepSeek routing group count retained in benchmark provenance.
    n_group: int
    # DeepSeek routing groups selected before expert top-k.
    topk_group: int
    # Largest token bucket admitted during tuning.
    tune_max_num_tokens: int
    # SwiGLU uses two FC1 projections; ReLU2 uses one non-gated projection.
    activation: str = "swiglu"
    swiglu_alpha: float = SwiGLU().alpha
    swiglu_beta: float = SwiGLU().beta
    swiglu_limit: float = SwiGLU().limit


@dataclass
class PreparedPrecision:
    """One precision's exact routed ABI and graph-stable mutable buffers."""

    # User-facing precision spelling written to the result table.
    name: str
    # Ordinary backend supplying every body in this prepared invocation.
    backend: str
    # Immutable model geometry used by this prepared invocation.
    shape: BenchmarkShape
    # Plain int32 expert IDs mutated between distribution replays.
    expert_ids: torch.Tensor
    # BF16 routing weights mutated with the expert IDs.
    routing_weights: torch.Tensor
    # Optional graph-stable logits used by FromLogits-only public backends.
    routing_logits: torch.Tensor | None
    # Packed int32 view for ABIs that consume score and ID in one tensor.
    packed_routing: torch.Tensor | None
    # Stable finalized BF16 destination captured into both graphs.
    output: torch.Tensor
    # Exact public routed-MoE closure for this precision.
    invoke: Callable[[], torch.Tensor]

    def stage(self, expert_ids: torch.Tensor, routing_weights: torch.Tensor) -> None:
        """Copy one live distribution into the graph-stable routing buffers."""
        self.expert_ids.copy_(expert_ids)
        self.routing_weights.copy_(routing_weights)
        if self.routing_logits is not None:
            self.routing_logits.fill_(-32)
            selected_logits = routing_weights.float().clamp_min_(1e-6).log().add_(32)
            self.routing_logits.scatter_(
                1, expert_ids.to(torch.int64), selected_logits.to(self.routing_logits)
            )
        if self.packed_routing is not None:
            packed = (expert_ids << 16) | (
                routing_weights.view(torch.int16).to(torch.int32) & 0xFFFF
            )
            self.packed_routing.copy_(packed)


@contextlib.contextmanager
def _temporary_environment(**updates: str | None) -> Iterator[None]:
    """Apply process configuration for one lifecycle phase and restore it."""
    previous = {name: os.environ.get(name) for name in updates}
    try:
        for name, value in updates.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _canonical_inputs(shape: BenchmarkShape) -> tuple[torch.Tensor, ...]:
    """Allocate deterministic BF16 activations, local weights, and routing."""
    # Fix the public benchmark seed so NoDA and DA preparation begin from identical tensors.
    torch.manual_seed(20260810)
    device = torch.device("cuda")
    hidden = (
        torch.randn(shape.num_tokens, shape.hidden_size, device=device) * 0.02
    ).to(torch.bfloat16)
    w1 = (
        torch.randn(
            shape.local_num_experts,
            (2 if shape.activation == "swiglu" else 1) * shape.intermediate_size,
            shape.hidden_size,
            device=device,
        )
        * 0.02
    ).to(torch.bfloat16)
    w2 = (
        torch.randn(
            shape.local_num_experts,
            shape.hidden_size,
            shape.intermediate_size,
            device=device,
        )
        * 0.02
    ).to(torch.bfloat16)
    # Routing IDs and weights are stable-address mutable inputs populated per distribution row.
    ids = torch.empty(shape.num_tokens, shape.top_k, device=device, dtype=torch.int32)
    weights = torch.empty(
        shape.num_tokens, shape.top_k, device=device, dtype=torch.bfloat16
    )
    return hidden, w1, w2, ids, weights


def _prepare_precision(
    name: str,
    shape: BenchmarkShape,
    backend: str = "trtllm",
    routing_input_mode: str = "routed",
) -> PreparedPrecision:
    """Prepare one exact public routed-MoE precision contract for a backend."""
    if backend not in BACKENDS:
        raise ValueError(f"Unsupported MoE backend {backend!r}")
    if routing_input_mode not in ("routed", "logits"):
        raise ValueError(f"Unsupported routing input mode {routing_input_mode!r}")
    if (
        backend == "trtllm"
        and name == "fp8_per_tensor"
        and routing_input_mode == "logits"
    ):
        raise ValueError("TRTLLM FP8 per-tensor benchmarking requires routed inputs")
    if routing_input_mode == "logits" and name not in ("nvfp4", "fp8_per_tensor"):
        raise ValueError(
            "FromLogits benchmarking currently targets NVFP4 and FP8 per-tensor"
        )
    # All precision families share one deterministic logical problem and stable output tensor.
    activation = (
        SwiGLU(shape.swiglu_alpha, shape.swiglu_beta, shape.swiglu_limit)
        if shape.activation == "swiglu"
        else ReLU2()
    )
    if (
        name == "fp8_per_tensor"
        and isinstance(activation, SwiGLU)
        and activation != SwiGLU()
    ):
        raise ValueError(
            "FP8 per-tensor benchmarking does not support custom SwiGLU parameters"
        )
    if name == "mxint4" and not isinstance(activation, SwiGLU):
        raise ValueError("MXINT4 benchmarking requires SwiGLU")
    hidden, w1, w2, ids, routing_weights = _canonical_inputs(shape)
    output = torch.empty(
        shape.num_tokens,
        shape.hidden_size,
        device=hidden.device,
        dtype=torch.bfloat16,
    )
    common = dict(
        num_experts=shape.num_experts,
        top_k=shape.top_k,
        n_group=None,
        topk_group=None,
        intermediate_size=shape.intermediate_size,
        local_expert_offset=shape.local_expert_offset,
        local_num_experts=shape.local_num_experts,
        routed_scaling_factor=1.0,
        routing_method_type=RoutingMethodType.Renormalize.value,
        output=output,
        tune_max_num_tokens=shape.tune_max_num_tokens,
        activation_type=(
            ActivationType.Swiglu.value
            if shape.activation == "swiglu"
            else ActivationType.Relu2.value
        ),
    )
    if isinstance(activation, SwiGLU) and activation != SwiGLU():
        # Scalar model semantics become graph-stable per-expert ABI tensors.
        # FP4 preparation uses unit output scales, so no parameter rescaling is needed.
        common.update(
            {
                name: torch.full(
                    (shape.local_num_experts,),
                    value,
                    device=hidden.device,
                    dtype=torch.float32,
                )
                for name, value in (
                    ("gemm1_alpha", activation.alpha),
                    ("gemm1_beta", activation.beta),
                    ("gemm1_clamp_limit", activation.limit),
                )
            }
        )
    # Quantize once outside timing, then bind a closure to the exact user-facing dtype ABI.
    if name in ("nvfp4", "mxfp4", "w4a16"):
        variant = {
            "nvfp4": QuantConfig(
                weight=QuantFormat.NVFP4, activation=QuantFormat.NVFP4
            ),
            "mxfp4": QuantConfig(
                weight=QuantFormat.MXFP4, activation=QuantFormat.MXFP8
            ),
            "w4a16": QuantConfig(weight=QuantFormat.MXFP4, activation=QuantFormat.BF16),
        }[name]
        hidden_q, hidden_scale = TrtllmFp4Config.prepare_activations(
            hidden, quant=variant
        )
        view = TrtllmFp4Config.prepare_weights(
            w1,
            w2,
            quant=variant,
            num_local_experts=shape.local_num_experts,
            hidden_size=shape.hidden_size,
            intermediate_size=shape.intermediate_size,
            activation=activation,
            device=hidden.device,
        )
        common.setdefault(
            "gemm1_alpha", None if backend == "prims_ts" else view.get("gemm1_alpha")
        )
        common.setdefault("gemm1_beta", None)
        common.setdefault("gemm1_clamp_limit", None)

        routing_logits = (
            torch.empty(
                shape.num_tokens,
                shape.num_experts,
                device=hidden.device,
                dtype=torch.bfloat16,
            )
            if routing_input_mode == "logits"
            else None
        )

        def invoke() -> torch.Tensor:
            """Invoke the exact FP4-family public ABI into the stable output."""
            kwargs = dict(
                routing_bias=None,
                hidden_states=hidden_q,
                hidden_states_scale=hidden_scale,
                gemm1_weights=view["gemm1_weights"],
                gemm1_weights_scale=view["gemm1_weights_scale"],
                gemm1_bias=None,
                gemm2_weights=view["gemm2_weights"],
                gemm2_weights_scale=view["gemm2_weights_scale"],
                gemm2_bias=None,
                output1_scale_scalar=view.get("output1_scale_scalar"),
                output1_scale_gate_scalar=view.get("output1_scale_gate_scalar"),
                output2_scale_scalar=view.get("output2_scale_scalar"),
                **common,
            )
            if routing_input_mode == "logits":
                fp4_op = (
                    prims_ts_fp4_block_scale_moe
                    if backend == "prims_ts"
                    else trtllm_fp4_block_scale_moe
                )
                result = fp4_op(routing_logits=routing_logits, **kwargs)
            else:
                fp4_op = (
                    prims_ts_fp4_block_scale_routed_moe
                    if backend == "prims_ts"
                    else trtllm_fp4_block_scale_routed_moe
                )
                result = fp4_op(topk_ids=(ids, routing_weights), **kwargs)
            # Normalize the historical tensor/list return spellings for matched benchmarking.
            return result[0] if isinstance(result, list) else result

        return PreparedPrecision(
            name,
            backend,
            shape,
            ids,
            routing_weights,
            routing_logits,
            None,
            output,
            invoke,
        )

    packed = torch.empty_like(ids)
    if name == "bf16":
        if backend == "prims_ts":
            # PrimsTS consumes its ordinary three-dimensional shuffled MajorK ABI; the TRTLLM
            # benchmark helper intentionally produces a different four-dimensional block ABI.
            gemm1_weights = torch.stack(
                [
                    shuffle_matrix_a(
                        (
                            reorder_rows_for_gated_act_gemm(weight)
                            if shape.activation == "swiglu"
                            else weight
                        ).view(torch.uint8),
                        128,
                    ).view(torch.bfloat16)
                    for weight in w1
                ]
            ).contiguous()
            gemm2_weights = torch.stack(
                [
                    shuffle_matrix_a(weight.view(torch.uint8), 128).view(torch.bfloat16)
                    for weight in w2
                ]
            ).contiguous()
            view = {
                "gemm1_weights": gemm1_weights,
                "gemm2_weights": gemm2_weights,
            }
        else:
            view = TrtllmBf16Config.prepare_weights(
                w1,
                w2,
                num_local_experts=shape.local_num_experts,
                hidden_size=shape.hidden_size,
                intermediate_size=shape.intermediate_size,
                activation=activation,
                device=hidden.device,
            )

        def invoke() -> torch.Tensor:
            """Invoke the exact packed BF16 routed ABI into the stable output."""
            bf16_op = (
                prims_ts_bf16_routed_moe
                if backend == "prims_ts"
                else trtllm_bf16_routed_moe
            )
            result = bf16_op(
                topk_ids=packed,
                hidden_states=hidden,
                gemm1_weights=view["gemm1_weights"],
                gemm2_weights=view["gemm2_weights"],
                **common,
            )
            return result[0] if isinstance(result, list) else result

    elif name == "fp8_per_tensor":
        input_scale = torch.tensor(1.0, device=hidden.device)
        intermediate_scale = torch.tensor(1.0, device=hidden.device)
        hidden_q, _ = TrtllmFp8PerTensorConfig.prepare_activations(
            hidden, hidden_states_scale_global=input_scale
        )
        view = TrtllmFp8PerTensorConfig.prepare_weights(
            w1,
            w2,
            hidden_states_scale_global=input_scale,
            intermediate_scale_global=intermediate_scale,
            num_local_experts=shape.local_num_experts,
            hidden_size=shape.hidden_size,
            intermediate_size=shape.intermediate_size,
            activation=activation,
            device=hidden.device,
        )

        if backend == "prims_ts":
            routing_logits = torch.empty(
                shape.num_tokens,
                shape.num_experts,
                device=hidden.device,
                dtype=torch.bfloat16,
            )

            def invoke() -> torch.Tensor:
                """Invoke the FromLogits-only PrimsTS FP8 per-tensor ABI."""
                result = prims_ts_fp8_per_tensor_scale_moe(
                    routing_logits=routing_logits,
                    routing_bias=None,
                    hidden_states=hidden_q,
                    gemm1_weights=view["gemm1_weights"],
                    output1_scales_scalar=view["output1_scales_scalar"],
                    output1_scales_gate_scalar=view["output1_scales_gate_scalar"],
                    gemm2_weights=view["gemm2_weights"],
                    output2_scales_scalar=view["output2_scales_scalar"],
                    use_routing_scales_on_input=False,
                    **common,
                )
                return result[0] if isinstance(result, list) else result

        else:
            routing_logits = None

            def invoke() -> torch.Tensor:
                """Invoke the exact packed TRTLLM per-tensor FP8 routed ABI."""
                result = trtllm_fp8_per_tensor_scale_routed_moe(
                    topk_ids=packed,
                    routing_bias=None,
                    hidden_states=hidden_q,
                    gemm1_weights=view["gemm1_weights"],
                    output1_scales_scalar=view["output1_scales_scalar"],
                    output1_scales_gate_scalar=view["output1_scales_gate_scalar"],
                    gemm2_weights=view["gemm2_weights"],
                    output2_scales_scalar=view["output2_scales_scalar"],
                    use_routing_scales_on_input=False,
                    **common,
                )
                return result[0] if isinstance(result, list) else result

    elif name in ("fp8_block", "mxfp8"):
        variant = (
            QuantConfig(
                weight=QuantFormat.DeepSeekFp8, activation=QuantFormat.DeepSeekFp8
            )
            if name == "fp8_block"
            else QuantConfig(weight=QuantFormat.MXFP8, activation=QuantFormat.MXFP8)
        )
        hidden_q, hidden_scale = TrtllmFp8BlockConfig.prepare_activations(
            hidden, quant=variant
        )
        view = TrtllmFp8BlockConfig.prepare_weights(
            w1,
            w2,
            quant=variant,
            num_local_experts=shape.local_num_experts,
            hidden_size=shape.hidden_size,
            intermediate_size=shape.intermediate_size,
            activation=activation,
            device=hidden.device,
        )
        fp8_type = (
            Fp8QuantizationType.DeepSeekFp8
            if name == "fp8_block"
            else Fp8QuantizationType.MxFp8
        )
        if backend == "prims_ts" and name == "fp8_block":
            # DeepSeek preparation is intentionally unshuffled for TRTLLM. PrimsTS' ordinary
            # MajorK ABI shuffles payload rows with a 64-row epilogue tile; its FP32 scales stay
            # in the original 128x128 block layout.
            view["gemm1_weights"] = torch.stack(
                [
                    shuffle_matrix_a(weight.view(torch.uint8), 64).view(
                        torch.float8_e4m3fn
                    )
                    for weight in view["gemm1_weights"]
                ]
            ).contiguous()
            view["gemm2_weights"] = torch.stack(
                [
                    shuffle_matrix_a(weight.view(torch.uint8), 64).view(
                        torch.float8_e4m3fn
                    )
                    for weight in view["gemm2_weights"]
                ]
            ).contiguous()

        def invoke() -> torch.Tensor:
            """Invoke the exact packed block-FP8 routed ABI."""
            fp8_op = (
                prims_ts_fp8_block_scale_routed_moe
                if backend == "prims_ts"
                else trtllm_fp8_block_scale_routed_moe
            )
            result = fp8_op(
                topk_ids=packed,
                routing_bias=None,
                hidden_states=hidden_q,
                hidden_states_scale=hidden_scale,
                gemm1_weights=view["gemm1_weights"],
                gemm1_weights_scale=view["gemm1_weights_scale"],
                gemm2_weights=view["gemm2_weights"],
                gemm2_weights_scale=view["gemm2_weights_scale"],
                use_shuffled_weight=backend == "prims_ts" or name == "mxfp8",
                weight_layout=WeightLayout.MajorK.value,
                fp8_quantization_type=fp8_type,
                **common,
            )
            return result[0] if isinstance(result, list) else result

    elif name == "mxint4":
        common.pop("activation_type")  # This ABI supports only SwiGLU.
        for parameter in ("gemm1_alpha", "gemm1_beta", "gemm1_clamp_limit"):
            common.setdefault(parameter, None)
        view = TrtllmMxInt4Config.prepare_weights(
            w1,
            w2,
            num_local_experts=shape.local_num_experts,
            hidden_size=shape.hidden_size,
            intermediate_size=shape.intermediate_size,
            activation=activation,
            device=hidden.device,
        )

        def invoke() -> torch.Tensor:
            """Invoke the exact packed MXINT4 routed ABI into the stable output."""
            result = trtllm_mxint4_block_scale_routed_moe(
                topk_ids=packed,
                hidden_states=hidden,
                gemm1_weights=view["gemm1_weights"],
                gemm1_weights_scale=view["gemm1_weights_scale"],
                gemm2_weights=view["gemm2_weights"],
                gemm2_weights_scale=view["gemm2_weights_scale"],
                **common,
            )
            return result[0]

    else:
        raise ValueError(f"Unsupported precision {name!r}")
    return PreparedPrecision(
        name,
        backend,
        shape,
        ids,
        routing_weights,
        routing_logits if name == "fp8_per_tensor" else None,
        packed,
        output,
        invoke,
    )


def _realization(
    factory: RoutingRealizationFactory,
    shape: BenchmarkShape,
    distribution: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate one deterministic global-expert routing realization."""
    # Normalize the distribution spelling before constructing persistent realization identity.
    parsed = DADistribution.parse(distribution)
    # Tuning consumes random numbers; isolate replay routes so fresh and cache-only
    # processes stage identical inputs without changing the tuner's RNG stream.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.cuda.manual_seed(20260810)
        realized = factory.get_or_create(
            RoutingRealizationKey(
                device=torch.device("cuda"),
                num_tokens=shape.num_tokens,
                distribution=parsed.name,
                sample_index=0,
                local_expert_offset=shape.local_expert_offset,
                num_experts=shape.num_experts,
                num_local_experts=shape.local_num_experts,
                top_k=shape.top_k,
                routing_rule_fingerprint="benchmark:renormalize",
                routed_scaling_factor=1.0,
            )
        )
    # Return the canonical mutable pair staged into both matched public graphs.
    return realized.expert_ids, realized.routing_weights


class _TimedGraph(torch.cuda.CUDAGraph):
    def __init__(self):
        super().__init__()
        # External events become replayed graph nodes. Host scheduling delays
        # around graph.replay() must not enter the measured GPU interval.
        self.start = torch.cuda.Event(enable_timing=True, external=True)
        self.end = torch.cuda.Event(enable_timing=True, external=True)


def _check_output(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Reject nonfinite, localized, and norm-relative errors before reporting time."""
    if not bool(torch.isfinite(actual).all() & torch.isfinite(expected).all()):
        raise AssertionError("benchmark outputs must be finite")
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)
    difference = torch.linalg.vector_norm(actual.float() - expected.float()).item()
    reference = torch.linalg.vector_norm(expected.float()).item()
    relative_l2 = (
        difference / reference if reference else (0.0 if not difference else math.inf)
    )
    # Small output magnitudes can hide missing computation under absolute tolerance.
    if relative_l2 > 3e-2:
        raise AssertionError(
            f"benchmark relative L2 error {relative_l2:.6g} exceeds 0.03"
        )
    return relative_l2


def _capture(invoke: Callable[[], torch.Tensor]) -> _TimedGraph:
    """Capture one already-warmed public invocation into an outer CUDA graph."""
    invoke()
    torch.cuda.synchronize()
    graph = _TimedGraph()
    with torch.cuda.graph(graph):
        graph.start.record()
        invoke()
        graph.end.record()
    return graph


def _cold_l2_buffers() -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate two independent cache-eviction buffers, each over twice L2."""
    l2_bytes = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).L2_cache_size
    elements = math.ceil((2 * l2_bytes + 4096) / 4)
    return (
        torch.empty(elements, device="cuda", dtype=torch.float32),
        torch.empty(elements, device="cuda", dtype=torch.float32),
    )


def _time_graphs_counterbalanced(
    no_da_graph: _TimedGraph,
    da_graph: _TimedGraph,
    flush_buffers: tuple[torch.Tensor, torch.Tensor],
    warmup: int,
    iterations: int,
) -> tuple[float, float]:
    """Measure matched graph replays with counterbalanced cold-L2 ordering."""
    if iterations <= 0 or iterations % 2 != 0:
        raise ValueError(
            "counterbalanced timing requires a positive, even iteration count"
        )

    # Warm both executable graphs in alternating order; warmups do not contribute to timing.
    for iteration in range(warmup):
        ordered_graphs = (
            (no_da_graph, da_graph) if iteration % 2 == 0 else (da_graph, no_da_graph)
        )
        for graph in ordered_graphs:
            graph.replay()
    torch.cuda.synchronize()
    no_da_elapsed = 0.0
    da_elapsed = 0.0
    # ABBA ordering gives each graph every measurement position and eviction buffer equally often.
    for iteration in range(iterations):
        ordered_graphs = (
            (("noda", no_da_graph), ("da", da_graph))
            if iteration % 2 == 0
            else (("da", da_graph), ("noda", no_da_graph))
        )
        for position, (label, graph) in enumerate(ordered_graphs):
            flush_buffers[(2 * iteration + position) % 2].zero_()
            torch.cuda.synchronize()
            graph.replay()
            graph.end.synchronize()
            elapsed = graph.start.elapsed_time(graph.end)
            if label == "noda":
                no_da_elapsed += elapsed
            else:
                da_elapsed += elapsed
    return no_da_elapsed / iterations, da_elapsed / iterations


def _matching_diagnostic(
    precision: str,
    shape: BenchmarkShape | None = None,
    distributions: tuple[str, ...] | None = None,
    backend: str = "trtllm",
) -> dict[str, object]:
    """Return the exact diagnostic for one precision, shape, and tuner catalog."""
    # Resolve the public operation and activation dtype expected for this precision family.
    trtllm_expected = {
        "nvfp4": "flashinfer::trtllm_fp4_block_scale_moe",
        "mxfp4": "flashinfer::trtllm_fp4_block_scale_moe",
        "w4a16": "flashinfer::trtllm_fp4_block_scale_moe",
        "bf16": "flashinfer::trtllm_bf16_moe",
        "fp8_per_tensor": "flashinfer::trtllm_fp8_per_tensor_scale_routed_moe",
        "fp8_block": "flashinfer::trtllm_fp8_block_scale_moe",
        "mxfp8": "flashinfer::trtllm_fp8_block_scale_moe",
        "mxint4": "flashinfer::trtllm_mxint4_block_scale_moe",
    }[precision]
    prims_expected = {
        "nvfp4": "flashinfer::prims_ts_fp4_block_scale_moe",
        "mxfp4": "flashinfer::prims_ts_fp4_block_scale_moe",
        "w4a16": "flashinfer::prims_ts_fp4_block_scale_moe",
        "bf16": "flashinfer::prims_ts_bf16_moe",
        "fp8_per_tensor": "flashinfer::prims_ts_fp8_per_tensor_scale_moe",
        "fp8_block": "flashinfer::prims_ts_fp8_block_scale_moe",
        "mxfp8": "flashinfer::prims_ts_fp8_block_scale_moe",
        # PrimsTS has no ordinary MXINT4 body; this value is never selected.
        "mxint4": "flashinfer::trtllm_mxint4_block_scale_moe",
    }[precision]
    expected = prims_expected if backend == "prims_ts" else trtllm_expected
    expected_dtype_act = {
        "nvfp4": DtypeTrtllmGen.E2m1,
        "mxfp4": DtypeTrtllmGen.MxE4m3,
        "w4a16": DtypeTrtllmGen.Bfloat16,
        "bf16": DtypeTrtllmGen.Bfloat16,
        "fp8_per_tensor": DtypeTrtllmGen.E4m3,
        "fp8_block": DtypeTrtllmGen.E4m3,
        "mxfp8": DtypeTrtllmGen.MxE4m3,
        "mxint4": DtypeTrtllmGen.Bfloat16,
    }[precision]
    # Filter the process registry by operation, runner dtype, concrete shape, and distribution
    # catalog so stale diagnostics from earlier benchmark rows cannot be selected.
    matches = []
    expected_distributions = (
        None
        if distributions is None
        else [DADistribution.parse(item).name for item in distributions]
    )
    for item in da_moe_diagnostics(backend):
        operation_key = json.loads(str(item["operation_key"]))
        runner_identity = json.loads(operation_key["runner_identity"])
        config_identity = json.loads(operation_key["config_identity"])
        if (
            operation_key["custom_op"] == expected
            and runner_identity["fields"]["dtype_act"] == expected_dtype_act.value
            and (
                shape is None
                or (
                    operation_key["num_tokens"] == shape.num_tokens
                    and operation_key["num_experts"] == shape.num_experts
                    and operation_key["local_expert_offset"]
                    == shape.local_expert_offset
                    and operation_key["num_local_experts"] == shape.local_num_experts
                    and operation_key["top_k"] == shape.top_k
                )
            )
            and (
                expected_distributions is None
                or config_identity["distributions"] == expected_distributions
            )
        ):
            matches.append(item)
    if not matches:
        raise RuntimeError(f"No DA diagnostic was published for {precision}")
    return matches[-1]


def _benchmark_precision(
    precision: str,
    shape: BenchmarkShape,
    distributions: tuple[str, ...],
    cache: str | None,
    tune: bool,
    warmup: int,
    iterations: int,
    backend: str = "trtllm",
    routing_input_mode: str = "routed",
) -> list[dict[str, object]]:
    """Run matched NoDA and DA graphs for all distributions of one precision."""
    if backend == "prims_ts" and precision == "fp8_per_tensor":
        routing_input_mode = "logits"
    # Both graphs replay the same exact balanced EP routes.
    prepared = _prepare_precision(precision, shape, backend, routing_input_mode)
    factory = RoutingRealizationFactory()
    first_ids, first_weights = _realization(factory, shape, distributions[0])
    prepared.stage(first_ids, first_weights)
    buckets = (shape.num_tokens,)
    # Tune and capture the shape-only baseline independently under the same AutoTuner cache.
    with _temporary_environment(FLASHINFER_DIST_AWARE_AUTOTUNE="0"):
        torch.cuda.synchronize()
        no_da_autotune_start = time.perf_counter()
        with (
            torch.cuda.nvtx.range(f"NODA_AUTOTUNE_{precision}"),
            autotune(tune, cache=cache, tuning_buckets=buckets),
        ):
            prepared.invoke()
        torch.cuda.synchronize()
        no_da_autotune_ms = (time.perf_counter() - no_da_autotune_start) * 1e3
        # Keep non-default token buckets active through cache lookup during capture.
        with autotune(False, tuning_buckets=buckets):
            no_da_graph = _capture(prepared.invoke)

    da_graph = None
    leases = ()
    try:
        distribution_text = ",".join(distributions)
        # Tune, prepare, and capture DA through the same public invocation contract.
        with _temporary_environment(
            FLASHINFER_DIST_AWARE_AUTOTUNE="1",
            FLASHINFER_DA_DISTRIBUTIONS=distribution_text,
        ):
            torch.cuda.synchronize()
            da_autotune_start = time.perf_counter()
            with (
                torch.cuda.nvtx.range(f"DA_AUTOTUNE_{precision}"),
                autotune(
                    tune,
                    cache=cache,
                    tuning_buckets=buckets,
                ),
            ):
                prepared.invoke()
            torch.cuda.synchronize()
            da_autotune_ms = (time.perf_counter() - da_autotune_start) * 1e3
            with autotune(False, tuning_buckets=buckets):
                prepared.invoke()
                torch.cuda.synchronize()
                da_graph = _capture(prepared.invoke)
            leases = da_moe_acquire_graph_leases(da_graph)
            baseline_guard_enabled = (
                DaMoeConfig.from_environment().baseline_guard_enabled
            )

        # Validate capture policy and graph-lease ownership before collecting performance rows.
        captured_diagnostic = _matching_diagnostic(
            precision, shape, distributions, backend
        )
        captured_policy = captured_diagnostic.get("policy")
        capture_fallback_reason = captured_diagnostic.get("capture_fallback_reason")
        if (
            captured_policy == "da_switch"
            and not leases
            and not capture_fallback_reason
        ):
            raise RuntimeError(
                f"{precision} did not acquire its DA switch graph lease or record a "
                "pristine capture fallback"
            )
        if captured_policy not in (
            "da_switch",
            "da_single_body",
            "da_fallback",
        ):
            raise RuntimeError(
                f"{precision} published unexpected benchmark policy {captured_policy!r}"
            )
        if captured_policy != "da_switch" and leases:
            raise RuntimeError(
                f"{precision} acquired a graph lease for non-switch policy "
                f"{captured_policy!r}"
            )
        capture_policy = (
            "noda_capture_fallback"
            if captured_policy == "da_switch" and not leases
            else captured_policy
        )
        flush_buffers = _cold_l2_buffers()
        rows: list[dict[str, object]] = []
        # Replay identical live routing contents through NoDA and DA, then time each on cold L2.
        for distribution in distributions:
            ids, weights = _realization(factory, shape, distribution)
            prepared.stage(ids, weights)
            with torch.cuda.nvtx.range(f"NODA_REPLAY_{precision}_{distribution}"):
                no_da_graph.replay()
                torch.cuda.synchronize()
            no_da_output = prepared.output.clone()
            no_da_graph.replay()
            _check_output(prepared.output, no_da_output)
            with torch.cuda.nvtx.range(f"DA_REPLAY_{precision}_{distribution}"):
                da_graph.replay()
                torch.cuda.synchronize()
            da_output = prepared.output.clone()
            da_graph.replay()
            _check_output(prepared.output, da_output)
            relative_l2 = _check_output(da_output, no_da_output)
            no_da_ms, da_ms = _time_graphs_counterbalanced(
                no_da_graph,
                da_graph,
                flush_buffers,
                warmup,
                iterations,
            )
            diagnostic = _matching_diagnostic(precision, shape, distributions, backend)
            routing_hash = hashlib.sha256(ids.cpu().numpy().tobytes())
            routing_hash.update(weights.view(torch.uint8).cpu().numpy().tobytes())
            topology = diagnostic.get("topology") or {}
            selected_body = diagnostic.get("selected_body")
            if diagnostic.get("policy") == "da_single_body":
                selected_body = 0
            row = {
                "backend": backend,
                "precision": precision,
                "distribution": DADistribution.parse(distribution).name,
                "num_tokens": shape.num_tokens,
                "num_experts": shape.num_experts,
                "local_num_experts": shape.local_num_experts,
                "local_expert_offset": shape.local_expert_offset,
                "top_k": shape.top_k,
                "hidden_size": shape.hidden_size,
                "intermediate_size": shape.intermediate_size,
                "activation": shape.activation,
                "swiglu_alpha": shape.swiglu_alpha
                if shape.activation == "swiglu"
                else None,
                "swiglu_beta": shape.swiglu_beta
                if shape.activation == "swiglu"
                else None,
                "swiglu_limit": shape.swiglu_limit
                if shape.activation == "swiglu"
                else None,
                "execution_mode": "graph",
                "timing_protocol": "counterbalanced_cold_l2_in_graph_events",
                "routing_input_mode": routing_input_mode,
                "baseline_guard_enabled": baseline_guard_enabled,
                "routing_sha256": routing_hash.hexdigest(),
                "noda_ms": no_da_ms,
                "da_ms": da_ms,
                "speedup_da_over_noda": no_da_ms / da_ms,
                "noda_autotune_ms": no_da_autotune_ms,
                "da_autotune_ms": da_autotune_ms,
                "finite": bool(torch.isfinite(da_output).all()),
                "max_abs_difference": float(
                    (da_output.float() - no_da_output.float()).abs().max()
                ),
                "relative_l2_difference": relative_l2,
                "policy": diagnostic.get("policy"),
                "capture_policy": capture_policy,
                "capture_fallback_reason": capture_fallback_reason,
                "selected_body": selected_body,
                "num_bodies": len(diagnostic.get("bodies") or []),
                "outer_nodes": topology.get("outer_node_count"),
                "outer_edges": topology.get("outer_edge_count"),
                "conditional_nodes": topology.get("conditional_node_count"),
                "is_selector_preamble_parallelizable": topology.get(
                    "is_selector_preamble_parallelizable"
                ),
                "status": "pass",
            }
            expected_dispatch = (
                selected_body is None
                if capture_policy in ("da_fallback", "noda_capture_fallback")
                else selected_body is not None
            )
            if not row["finite"] or not expected_dispatch:
                raise RuntimeError(json.dumps(row, sort_keys=True))
            rows.append(row)
    finally:
        torch.cuda.synchronize()
        no_da_graph.reset()
        if da_graph is not None:
            da_graph.reset()
        for lease in leases:
            lease.release()
        da_moe_release_resources()
    return rows


def _parse_csv(value: str) -> tuple[str, ...]:
    """Parse a nonempty comma-separated command-line list."""
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected a nonempty comma-separated list")
    return values


def _parse_args() -> argparse.Namespace:
    """Parse the preserved DA benchmark and cache compatibility contract."""
    # Preserve historical cache aliases while presenting tuning-cache terminology to new users.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=BACKENDS, default="trtllm")
    parser.add_argument(
        "--routing-input-mode", choices=("routed", "logits"), default="routed"
    )
    parser.add_argument(
        "--precision", default="nvfp4", help="Comma-separated precision list or all"
    )
    parser.add_argument(
        "--distributions",
        type=_parse_csv,
        default=_parse_csv("uniform,ddist:1.1,ddist:1.5,ddist:2,ddist:3,ddist:4"),
        help="Tuning and replay catalog; must contain every mixture component",
    )
    parser.add_argument(
        "--mixture",
        action="append",
        type=_parse_mixture,
        help="Repeatable distribution=weight list. Defaults: 20/30/50 and 50/30/20 over ddist:1.1/2/4. Weights are normalized.",
    )
    parser.add_argument(
        "--num-tokens",
        type=_parse_csv,
        default=_parse_csv("8,64,128,256,512,1024,4096,8192"),
    )
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--local-num-experts", type=int, default=32)
    parser.add_argument("--local-expert-offset", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=7168)
    parser.add_argument("--intermediate-size", type=int, default=2048)
    parser.add_argument("--activation", choices=("swiglu", "relu2"), default="swiglu")
    parser.add_argument("--swiglu-alpha", type=float, default=SwiGLU().alpha)
    parser.add_argument("--swiglu-beta", type=float, default=SwiGLU().beta)
    parser.add_argument("--swiglu-limit", type=float, default=SwiGLU().limit)
    parser.add_argument("--n-group", type=int, default=8)
    parser.add_argument("--topk-group", type=int, default=4)
    parser.add_argument("--tune-max-num-tokens", type=int, default=8192)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--iters",
        type=int,
        default=10,
        help="Positive even sample count per graph for counterbalanced timing",
    )
    # This benchmark measures replay; eager fallback has its own API contract tests.
    parser.add_argument("--execution-mode", choices=("graph",), default="graph")
    parser.add_argument("--cache", "--tuning-cache", "--bundle-output", dest="cache")
    parser.add_argument("--skip-autotune", "--cache-only", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        help="Raw per-distribution CSV (stdout shows mixture tables)",
    )
    parser.add_argument("--json-out", type=Path, help="Raw per-distribution JSON")
    parser.add_argument(
        "--mixture-out", type=Path, help="Weighted latency and speedup rows as JSON"
    )
    parser.add_argument(
        "--table-out", type=Path, help="Save the printed mixture tables as Markdown"
    )
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write raw per-distribution evidence separately from the mixture tables."""
    if not rows:
        return
    fieldnames = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    """Execute the requested precision/token matrix and persist its evidence."""
    # Normalize precision aliases and reject incompatible cache/shard arguments before GPU work.
    args = _parse_args()
    if args.precision == "all":
        requested_precision_names = (
            tuple(name for name in PRECISIONS if name != "mxint4")
            if args.backend == "prims_ts"
            else PRECISIONS
        )
    else:
        requested_precision_names = _parse_csv(args.precision)
    unknown = sorted(
        set(requested_precision_names) - set(PRECISIONS) - set(PRECISION_ALIASES)
    )
    if unknown:
        raise SystemExit(f"unknown precision(s): {', '.join(unknown)}")
    precision_names = tuple(
        PRECISION_ALIASES.get(name, name) for name in requested_precision_names
    )
    if len(set(precision_names)) != len(precision_names):
        raise SystemExit(
            "--precision must not contain duplicates or equivalent aliases"
        )
    mixtures = (
        tuple(args.mixture)
        if args.mixture
        else (
            _parse_mixture("ddist:1.1=20,ddist:2=30,ddist:4=50"),
            _parse_mixture("ddist:1.1=50,ddist:2=30,ddist:4=20"),
        )
    )
    try:
        distributions = tuple(DADistribution.parse(d).name for d in args.distributions)
        tokens = tuple(int(t) for t in args.num_tokens)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if len(set(distributions)) != len(distributions):
        raise SystemExit("--distributions must not contain equivalent duplicates")
    if len(set(tokens)) != len(tokens) or any(t <= 0 for t in tokens):
        raise SystemExit("--num-tokens must contain distinct positive integers")
    missing = {name for mixture in mixtures for name, _ in mixture.weights} - set(
        distributions
    )
    if missing:
        raise SystemExit(
            f"mixture components missing from --distributions: {sorted(missing)}"
        )
    if args.backend == "prims_ts" and "mxint4" in precision_names:
        raise SystemExit("PrimsTS does not yet provide an ordinary MXINT4 MoE body")
    if args.skip_autotune and not args.cache:
        raise SystemExit("--skip-autotune requires --cache/--tuning-cache")
    if args.iters <= 0 or args.iters % 2 != 0:
        raise SystemExit("--iters must be a positive, even number")
    if args.local_num_experts > args.num_experts:
        raise SystemExit("--local-num-experts cannot exceed --num-experts")
    if args.local_expert_offset + args.local_num_experts > args.num_experts:
        raise SystemExit("the local expert shard exceeds --num-experts")
    if args.cache and args.cache != "/dev/null":
        Path(args.cache).parent.mkdir(parents=True, exist_ok=True)
    # Execute token-major rows, releasing Python/CUDA allocator caches between precision families.
    rows: list[dict[str, object]] = []
    for num_tokens in tokens:
        shape = BenchmarkShape(
            num_tokens=num_tokens,
            num_experts=args.num_experts,
            local_num_experts=args.local_num_experts,
            local_expert_offset=args.local_expert_offset,
            top_k=args.top_k,
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size,
            n_group=args.n_group,
            topk_group=args.topk_group,
            tune_max_num_tokens=args.tune_max_num_tokens,
            activation=args.activation,
            swiglu_alpha=args.swiglu_alpha,
            swiglu_beta=args.swiglu_beta,
            swiglu_limit=args.swiglu_limit,
        )
        for precision in precision_names:
            rows.extend(
                _benchmark_precision(
                    precision,
                    shape,
                    distributions,
                    args.cache,
                    not args.skip_autotune,
                    args.warmup,
                    args.iters,
                    args.backend,
                    args.routing_input_mode,
                )
            )
            gc.collect()
            torch.cuda.empty_cache()
    # Persist CSV and optional JSON from the same in-memory row set to keep schemas identical.
    if args.out:
        _write_csv(args.out, rows)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    summaries = _summarize_mixtures(rows, mixtures)
    if args.mixture_out:
        args.mixture_out.parent.mkdir(parents=True, exist_ok=True)
        args.mixture_out.write_text(
            json.dumps(summaries, indent=2, sort_keys=True) + "\n"
        )
    table = _mixture_tables(summaries)
    print(table)
    if args.table_out:
        args.table_out.parent.mkdir(parents=True, exist_ok=True)
        args.table_out.write_text(table + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
