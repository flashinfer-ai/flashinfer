#!/usr/bin/env python3
"""Benchmark matched ordinary and distribution-aware TRTLLM routed MoE graphs."""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

from flashinfer.autotuner import autotune
from flashinfer.fused_moe import (
    QuantVariant,
    TrtllmBf16Config,
    TrtllmFp4Config,
    TrtllmFp8BlockConfig,
    TrtllmFp8PerTensorConfig,
    TrtllmMxInt4Config,
    trtllm_bf16_routed_moe,
    trtllm_fp4_block_scale_routed_moe,
    trtllm_fp8_block_scale_routed_moe,
    trtllm_fp8_per_tensor_scale_routed_moe,
    trtllm_moe_acquire_da_graph_leases,
    trtllm_moe_da_diagnostics,
    trtllm_mxint4_block_scale_routed_moe,
)
from flashinfer.fused_moe.da_tuner import (
    DADistribution,
    RoutingRealizationFactory,
    RoutingRealizationKey,
)
from flashinfer.tllm_enums import (
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

PRECISION_ALIASES = {
    # Name mapping for MXFP4 weights with MXFP8 activations.
    "mxfp4_mxfp8": "mxfp4",
    # Name mapping for MXFP4 weights with BF16 activations.
    "mxfp4_bf16": "w4a16",
}


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


@dataclass
class PreparedPrecision:
    """One precision's exact routed ABI and graph-stable mutable buffers."""

    # User-facing precision spelling written to the result table.
    name: str
    # Immutable model geometry used by this prepared invocation.
    shape: BenchmarkShape
    # Plain int32 expert IDs mutated between distribution replays.
    expert_ids: torch.Tensor
    # BF16 routing weights mutated with the expert IDs.
    routing_weights: torch.Tensor
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
        if self.packed_routing is not None:
            packed = (expert_ids << 16) | (
                routing_weights.view(torch.int16).to(torch.int32) & 0xFFFF
            )
            self.packed_routing.copy_(packed)


@contextlib.contextmanager
def _temporary_environment(**updates: str | None):
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
            2 * shape.intermediate_size,
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


def _prepare_precision(name: str, shape: BenchmarkShape) -> PreparedPrecision:
    """Prepare one exact public TRTLLM routed-MoE precision contract."""
    # All precision families share one deterministic logical problem and stable output tensor.
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
    )

    # Quantize once outside timing, then bind a closure to the exact user-facing dtype ABI.
    if name in ("nvfp4", "mxfp4", "w4a16"):
        variant = {
            "nvfp4": QuantVariant.NVFP4,
            "mxfp4": QuantVariant.MXFP4,
            "w4a16": QuantVariant.W4A16,
        }[name]
        hidden_q, hidden_scale = TrtllmFp4Config.prepare_activations(
            hidden, variant=variant
        )
        view = TrtllmFp4Config.prepare_weights(
            w1,
            w2,
            variant=variant,
            num_local_experts=shape.local_num_experts,
            hidden_size=shape.hidden_size,
            intermediate_size=shape.intermediate_size,
            device=hidden.device,
        )

        def invoke() -> torch.Tensor:
            """Invoke the exact FP4-family routed ABI into the stable output."""
            # Forward the variant-specific quantized tensors through the common public wrapper.
            result = trtllm_fp4_block_scale_routed_moe(
                topk_ids=(ids, routing_weights),
                routing_bias=None,
                hidden_states=hidden_q,
                hidden_states_scale=hidden_scale,
                gemm1_weights=view["gemm1_weights"],
                gemm1_weights_scale=view["gemm1_weights_scale"],
                gemm1_bias=None,
                gemm1_alpha=view.get("gemm1_alpha"),
                gemm1_beta=None,
                gemm1_clamp_limit=None,
                gemm2_weights=view["gemm2_weights"],
                gemm2_weights_scale=view["gemm2_weights_scale"],
                gemm2_bias=None,
                output1_scale_scalar=view.get("output1_scale_scalar"),
                output1_scale_gate_scalar=view.get("output1_scale_gate_scalar"),
                output2_scale_scalar=view.get("output2_scale_scalar"),
                **common,
            )
            # Normalize the historical tensor/list return spellings for matched benchmarking.
            return result[0] if isinstance(result, list) else result

        return PreparedPrecision(
            name, shape, ids, routing_weights, None, output, invoke
        )

    packed = torch.empty_like(ids)
    if name == "bf16":
        view = TrtllmBf16Config.prepare_weights(
            w1,
            w2,
            num_local_experts=shape.local_num_experts,
            hidden_size=shape.hidden_size,
            intermediate_size=shape.intermediate_size,
            device=hidden.device,
        )

        def invoke() -> torch.Tensor:
            """Invoke the exact packed BF16 routed ABI into the stable output."""
            result = trtllm_bf16_routed_moe(
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
            device=hidden.device,
        )

        def invoke() -> torch.Tensor:
            """Invoke the exact packed per-tensor FP8 routed ABI."""
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
            QuantVariant.DeepSeekFp8 if name == "fp8_block" else QuantVariant.MxFp8
        )
        hidden_q, hidden_scale = TrtllmFp8BlockConfig.prepare_activations(
            hidden, variant=variant
        )
        view = TrtllmFp8BlockConfig.prepare_weights(
            w1,
            w2,
            variant=variant,
            num_local_experts=shape.local_num_experts,
            hidden_size=shape.hidden_size,
            intermediate_size=shape.intermediate_size,
            device=hidden.device,
        )
        fp8_type = (
            Fp8QuantizationType.DeepSeekFp8
            if name == "fp8_block"
            else Fp8QuantizationType.MxFp8
        )

        def invoke() -> torch.Tensor:
            """Invoke the exact packed block-FP8 routed ABI."""
            result = trtllm_fp8_block_scale_routed_moe(
                topk_ids=packed,
                routing_bias=None,
                hidden_states=hidden_q,
                hidden_states_scale=hidden_scale,
                gemm1_weights=view["gemm1_weights"],
                gemm1_weights_scale=view["gemm1_weights_scale"],
                gemm2_weights=view["gemm2_weights"],
                gemm2_weights_scale=view["gemm2_weights_scale"],
                use_shuffled_weight=name == "mxfp8",
                weight_layout=WeightLayout.MajorK.value,
                fp8_quantization_type=fp8_type,
                **common,
            )
            return result[0] if isinstance(result, list) else result

    elif name == "mxint4":
        view = TrtllmMxInt4Config.prepare_weights(
            w1,
            w2,
            num_local_experts=shape.local_num_experts,
            hidden_size=shape.hidden_size,
            intermediate_size=shape.intermediate_size,
            device=hidden.device,
        )

        def invoke() -> torch.Tensor:
            """Invoke the exact packed MXINT4 routed ABI into the stable output."""
            result = trtllm_mxint4_block_scale_routed_moe(
                topk_ids=packed,
                hidden_states=hidden,
                gemm1_weights=view["gemm1_weights"],
                gemm1_weights_scale=view["gemm1_weights_scale"],
                gemm1_alpha=None,
                gemm1_beta=None,
                gemm1_clamp_limit=None,
                gemm2_weights=view["gemm2_weights"],
                gemm2_weights_scale=view["gemm2_weights_scale"],
                **common,
            )
            return result[0]

    else:
        raise ValueError(f"Unsupported precision {name!r}")
    return PreparedPrecision(name, shape, ids, routing_weights, packed, output, invoke)


def _realization(
    factory: RoutingRealizationFactory,
    shape: BenchmarkShape,
    distribution: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate one deterministic global-expert routing realization."""
    # Normalize the distribution spelling before constructing persistent realization identity.
    parsed = DADistribution.parse(distribution)
    realized = factory.get_or_create(
        RoutingRealizationKey(
            device=torch.device("cuda"),
            num_tokens=shape.num_tokens,
            distribution=parsed.name,
            sample_index=0,
            local_expert_offset=shape.local_expert_offset,
            num_local_experts=shape.local_num_experts,
            top_k=shape.top_k,
            routing_rule_fingerprint="benchmark:renormalize",
            routed_scaling_factor=1.0,
        )
    )
    # Return the canonical mutable pair staged into both matched public graphs.
    return realized.expert_ids, realized.routing_weights


def _capture(invoke: Callable[[], torch.Tensor]) -> torch.cuda.CUDAGraph:
    """Capture one already-warmed public invocation into an outer CUDA graph."""
    invoke()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        invoke()
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


def _time_graph(
    graph: torch.cuda.CUDAGraph,
    flush_buffers: tuple[torch.Tensor, torch.Tensor],
    warmup: int,
    iterations: int,
) -> float:
    """Measure graph replays after alternating independent cold-L2 flushes."""
    # Warm the executable graph separately; warmups do not contribute to requested iterations.
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    elapsed = 0.0
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    # Alternate two buffers that each exceed twice L2, and time exactly one replay per iteration.
    for iteration in range(iterations):
        flush_buffers[iteration % 2].zero_()
        torch.cuda.synchronize()
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        elapsed += start.elapsed_time(end)
    return elapsed / iterations


def _matching_diagnostic(
    precision: str,
    shape: BenchmarkShape | None = None,
    distributions: tuple[str, ...] | None = None,
) -> dict[str, object]:
    """Return the exact diagnostic for one precision, shape, and tuner catalog."""
    # Resolve the public operation and activation dtype expected for this precision family.
    expected = {
        "nvfp4": "flashinfer::trtllm_fp4_block_scale_moe",
        "mxfp4": "flashinfer::trtllm_fp4_block_scale_moe",
        "w4a16": "flashinfer::trtllm_fp4_block_scale_moe",
        "bf16": "flashinfer::trtllm_bf16_moe",
        "fp8_per_tensor": "flashinfer::trtllm_fp8_per_tensor_scale_routed_moe",
        "fp8_block": "flashinfer::trtllm_fp8_block_scale_moe",
        "mxfp8": "flashinfer::trtllm_fp8_block_scale_moe",
        "mxint4": "flashinfer::trtllm_mxint4_block_scale_moe",
    }[precision]
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
    for item in trtllm_moe_da_diagnostics():
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
) -> list[dict[str, object]]:
    """Run matched NoDA and DA graphs for all distributions of one precision."""
    # Prepare one public ABI and seed both lifecycle paths with the same first realization.
    prepared = _prepare_precision(precision, shape)
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
        no_da_graph = _capture(prepared.invoke)

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
            autotune(tune, cache=cache, tuning_buckets=buckets),
        ):
            prepared.invoke()
        torch.cuda.synchronize()
        da_autotune_ms = (time.perf_counter() - da_autotune_start) * 1e3
        prepared.invoke()
        torch.cuda.synchronize()
        da_graph = _capture(prepared.invoke)
        leases = trtllm_moe_acquire_da_graph_leases(da_graph)

    # Validate capture policy and graph-lease ownership before collecting performance rows.
    captured_diagnostic = _matching_diagnostic(precision, shape, distributions)
    captured_policy = captured_diagnostic.get("policy")
    capture_fallback_reason = captured_diagnostic.get("capture_fallback_reason")
    if captured_policy == "da_switch" and not leases and not capture_fallback_reason:
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
    try:
        for distribution in distributions:
            ids, weights = _realization(factory, shape, distribution)
            prepared.stage(ids, weights)
            with torch.cuda.nvtx.range(f"NODA_REPLAY_{precision}_{distribution}"):
                no_da_graph.replay()
                torch.cuda.synchronize()
            no_da_output = prepared.output.clone()
            with torch.cuda.nvtx.range(f"DA_REPLAY_{precision}_{distribution}"):
                da_graph.replay()
                torch.cuda.synchronize()
            da_output = prepared.output.clone()
            torch.testing.assert_close(da_output, no_da_output, rtol=3e-2, atol=3e-2)
            no_da_ms = _time_graph(no_da_graph, flush_buffers, warmup, iterations)
            da_ms = _time_graph(da_graph, flush_buffers, warmup, iterations)
            diagnostic = _matching_diagnostic(precision, shape, distributions)
            topology = diagnostic.get("topology") or {}
            selected_body = diagnostic.get("selected_body")
            if diagnostic.get("policy") == "da_single_body":
                selected_body = 0
            row = {
                "precision": precision,
                "distribution": DADistribution.parse(distribution).name,
                "num_tokens": shape.num_tokens,
                "num_experts": shape.num_experts,
                "local_num_experts": shape.local_num_experts,
                "top_k": shape.top_k,
                "hidden_size": shape.hidden_size,
                "intermediate_size": shape.intermediate_size,
                "execution_mode": "graph",
                "noda_ms": no_da_ms,
                "da_ms": da_ms,
                "speedup_da_over_noda": no_da_ms / da_ms,
                "noda_autotune_ms": no_da_autotune_ms,
                "da_autotune_ms": da_autotune_ms,
                "finite": bool(torch.isfinite(da_output).all()),
                "max_abs_difference": float(
                    (da_output.float() - no_da_output.float()).abs().max()
                ),
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
        da_graph.reset()
        for lease in leases:
            lease.release()
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
    parser.add_argument(
        "--precision", default="nvfp4", help="Comma-separated precision list or all"
    )
    parser.add_argument(
        "--distributions",
        type=_parse_csv,
        default=_parse_csv("uniform,ddist:1.1,ddist:1.5,ddist:2,ddist:3,ddist:4"),
    )
    parser.add_argument("--num-tokens", type=_parse_csv, default=("1024",))
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--local-num-experts", type=int, default=32)
    parser.add_argument("--local-expert-offset", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=7168)
    parser.add_argument("--intermediate-size", type=int, default=2048)
    parser.add_argument("--n-group", type=int, default=8)
    parser.add_argument("--topk-group", type=int, default=4)
    parser.add_argument("--tune-max-num-tokens", type=int, default=8192)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    # This benchmark measures replay; eager fallback has its own API contract tests.
    parser.add_argument("--execution-mode", choices=("graph",), default="graph")
    parser.add_argument("--cache", "--tuning-cache", "--bundle-output", dest="cache")
    parser.add_argument("--skip-autotune", "--cache-only", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def _write_csv(path: Path | None, rows: list[dict[str, object]]) -> None:
    """Write stable result columns to a file or standard output."""
    if not rows:
        return
    fieldnames = list(rows[0])
    if path is None:
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    """Execute the requested precision/token matrix and persist its evidence."""
    # Normalize precision aliases and reject incompatible cache/shard arguments before GPU work.
    args = _parse_args()
    requested_precision_names = (
        PRECISIONS if args.precision == "all" else _parse_csv(args.precision)
    )
    unknown = sorted(
        set(requested_precision_names) - set(PRECISIONS) - set(PRECISION_ALIASES)
    )
    if unknown:
        raise SystemExit(f"unknown precision(s): {', '.join(unknown)}")
    precision_names = tuple(
        PRECISION_ALIASES.get(name, name) for name in requested_precision_names
    )
    if args.skip_autotune and not args.cache:
        raise SystemExit("--skip-autotune requires --cache/--tuning-cache")
    if args.local_num_experts > args.num_experts:
        raise SystemExit("--local-num-experts cannot exceed --num-experts")
    if args.local_expert_offset + args.local_num_experts > args.num_experts:
        raise SystemExit("the local expert shard exceeds --num-experts")
    if args.cache and args.cache != "/dev/null":
        Path(args.cache).parent.mkdir(parents=True, exist_ok=True)
    # Execute token-major rows, releasing Python/CUDA allocator caches between precision families.
    rows: list[dict[str, object]] = []
    for token_text in args.num_tokens:
        shape = BenchmarkShape(
            num_tokens=int(token_text),
            num_experts=args.num_experts,
            local_num_experts=args.local_num_experts,
            local_expert_offset=args.local_expert_offset,
            top_k=args.top_k,
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size,
            n_group=args.n_group,
            topk_group=args.topk_group,
            tune_max_num_tokens=args.tune_max_num_tokens,
        )
        for precision in precision_names:
            rows.extend(
                _benchmark_precision(
                    precision,
                    shape,
                    args.distributions,
                    args.cache,
                    not args.skip_autotune,
                    args.warmup,
                    args.iters,
                )
            )
            gc.collect()
            torch.cuda.empty_cache()
    # Persist CSV and optional JSON from the same in-memory row set to keep schemas identical.
    _write_csv(args.out, rows)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
