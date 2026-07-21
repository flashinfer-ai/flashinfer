#!/usr/bin/env python3
"""DeepSeek-V3 MoE Performance Benchmark - CuteDSL vs CUTLASS vs TRTLLM.

Compares three NVFP4 MoE backends on DeepSeek-V3 configuration:
- CuteDSL: FlashInfer's CuteDSL-based implementation
- CUTLASS: NVIDIA CUTLASS-based implementation
- TRTLLM: TensorRT-LLM's implementation

Usage:
    # Throughput benchmark (large batches: 128-4096 tokens)
    python bench_moe_deepseek.py

    # Generation phase benchmark (small batches: 1-128 tokens)
    python bench_moe_deepseek.py --gen-phase

    # With Expert Parallelism simulation
    python bench_moe_deepseek.py --ep 1    # 256 local experts (no parallelism)
    python bench_moe_deepseek.py --ep 8    # 32 local experts (8-way EP)
    python bench_moe_deepseek.py --ep 16   # 16 local experts (16-way EP)

    # Custom token counts
    python bench_moe_deepseek.py --num-tokens 64,128,256

    # Reduce the expert count for a faster development run
    python bench_moe_deepseek.py --num-experts 16 --ep 1

    # Disable CUDA graph (useful for debugging or profiling)
    python bench_moe_deepseek.py --no-cuda-graph

    # Precompute routing IDs/weights before timing the selected backends
    python bench_moe_deepseek.py --routing-input-mode routed

    # Run only selected implementations (default: all three)
    python bench_moe_deepseek.py --backends cutedsl,trtllm

    # Run explicit routing distributions instead of random router logits
    python bench_moe_deepseek.py --distributions uniform,ddist:2

    # Disable CUPTI (use CUDA events for timing instead)
    python bench_moe_deepseek.py --no-cupti

    # CuTe DSL finalize modes
    python bench_moe_deepseek.py --functional-api  # atomic fused (default)
    python bench_moe_deepseek.py --functional-api --no-fused-finalize  # deterministic

Metrics:
    - ms: Latency in milliseconds
    - TFLOPS: Computational throughput
    - Speedup: CuteDSL latency / other backend latency (>1 = CuteDSL faster)
"""

import argparse
from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class DeepSeekConfig:
    hidden_size: int = 7168
    intermediate_size: int = 2048
    num_experts: int = 256
    n_group: int = 8
    topk_group: int = 4
    top_k: int = 8
    routed_scaling_factor: float = 2.5


CFG = DeepSeekConfig()
TOKEN_COUNTS = [128, 256, 512, 1024, 2048, 4096]

# Generation phase token counts (small batches typical in decode)
GEN_PHASE_TOKENS = [1, 2, 4, 8, 16, 32, 64, 128]

BACKENDS = ("cutedsl", "cutlass", "trtllm")


def parse_backends(value):
    """Parse and validate a comma-separated backend list."""
    backends = tuple(dict.fromkeys(item.strip().lower() for item in value.split(",")))
    invalid = sorted(set(backends) - set(BACKENDS))
    if invalid or not backends or "" in backends:
        expected = ",".join(BACKENDS)
        raise argparse.ArgumentTypeError(
            f"backends must be a comma-separated subset of {expected}"
        )
    return backends


def parse_distributions(value):
    """Parse the DA benchmark's comma-separated distribution grammar."""
    value = value.strip()
    if value == "full":
        vals = [round(1.1 + 0.1 * i, 1) for i in range(70)]
        return ("uniform", *(f"ddist:{v:.1f}" for v in vals))
    result = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if item == "uniform" or item.startswith("ddist:"):
            result.append(item)
        else:
            result.append(f"ddist:{float(item):.1f}")
    if not result:
        raise argparse.ArgumentTypeError("distributions must not be empty")
    return tuple(result)


def is_sm100_family():
    """Check for SM100 family (Blackwell: SM100, SM103).

    CuteDSL MoE NVFP4 kernels are optimized for SM10x architecture.
    """
    if not torch.cuda.is_available():
        return False
    props = torch.cuda.get_device_properties(0)
    return props.major == 10


def calc_tflops(n, ms, num_local_experts=None):
    """Calculate TFLOPS for MoE computation.

    With EP, only tokens routed to local experts are computed.
    Assumes uniform routing distribution across experts.
    """
    if num_local_experts is None:
        num_local_experts = CFG.num_experts

    # Fraction of work done locally (assuming uniform distribution)
    local_fraction = num_local_experts / CFG.num_experts

    flops = (
        n
        * CFG.top_k
        * local_fraction  # Only local expert pairs are computed
        * (
            2 * CFG.hidden_size * 2 * CFG.intermediate_size
            + 2 * CFG.intermediate_size * CFG.hidden_size
        )
    )
    return flops / (ms * 1e-3) / 1e12


def interleave(x, gs=64):
    M, K = x.shape[-2], x.shape[-1]
    return (
        x.view(*x.shape[:-2], 2, M // (gs * 2), gs, K)
        .transpose(-4, -3)
        .contiguous()
        .view(*x.shape)
    )


def create_inputs(n, dev="cuda", routing_bias_scale=0.01):
    """Create inputs for all backends (CuteDSL, CUTLASS, TRTLLM)."""
    from flashinfer.fp4_quantization import fp4_quantize

    torch.manual_seed(42)
    sv = 16
    FP8M = torch.finfo(torch.float8_e4m3fn).max
    FP4M = 6.0

    # Router logits and bias
    rl = torch.randn(n, CFG.num_experts, device=dev, dtype=torch.float32)
    rb = (
        torch.randn(CFG.num_experts, device=dev, dtype=torch.bfloat16)
        * routing_bias_scale
    )

    # Hidden states
    hb = torch.randn(n, CFG.hidden_size, device=dev, dtype=torch.bfloat16) / 10
    hg = FP8M * FP4M / hb.abs().max().float()

    # Weights (BF16)
    w1b = (
        torch.randn(
            CFG.num_experts,
            2 * CFG.intermediate_size,
            CFG.hidden_size,
            device=dev,
            dtype=torch.bfloat16,
        )
        / 10
    )
    w2b = (
        torch.randn(
            CFG.num_experts,
            CFG.hidden_size,
            CFG.intermediate_size,
            device=dev,
            dtype=torch.bfloat16,
        )
        / 10
    )

    # Compute per-expert global scales
    w1_gs_list, w2_gs_list = [], []
    for e in range(CFG.num_experts):
        w1_gs_list.append(FP8M * FP4M / w1b[e].abs().max().float())
        w2_gs_list.append(FP8M * FP4M / w2b[e].abs().max().float())
    w1_gs = torch.tensor(w1_gs_list, device=dev)
    w2_gs = torch.tensor(w2_gs_list, device=dev)

    # CUTLASS format: quantize with swizzled scale factors
    w1_fp4_list, w1_sf_list = [], []
    w2_fp4_list, w2_sf_list = [], []
    for e in range(CFG.num_experts):
        q1, s1 = fp4_quantize(w1b[e], w1_gs[e], sv, False, True)  # swizzled
        w1_fp4_list.append(q1)
        w1_sf_list.append(s1)
        q2, s2 = fp4_quantize(w2b[e], w2_gs[e], sv, False, True)  # swizzled
        w2_fp4_list.append(q2)
        w2_sf_list.append(s2)

    return {
        "router_logits": rl,
        "routing_bias": rb,
        "hidden_bf16": hb,
        "hidden_gs": hg,
        "w1_bf16": w1b,
        "w1_gs": w1_gs,
        "w2_bf16": w2b,
        "w2_gs": w2_gs,
        # CUTLASS specific
        "w1_fp4": torch.stack(w1_fp4_list),
        "w1_sf": torch.stack(w1_sf_list),
        "w2_fp4": torch.stack(w2_fp4_list),
        "w2_sf": torch.stack(w2_sf_list),
    }


def prepare_routed_input(inputs):
    """Compute caller-provided DeepSeek routing IDs and weights once."""
    from flashinfer.fused_moe import fused_topk_deepseek

    num_tokens = inputs["router_logits"].shape[0]
    device = inputs["router_logits"].device
    topk_weights = torch.empty(
        num_tokens, CFG.top_k, dtype=torch.float32, device=device
    )
    topk_ids = torch.empty(num_tokens, CFG.top_k, dtype=torch.int32, device=device)
    fused_topk_deepseek(
        scores=inputs["router_logits"],
        bias=inputs["routing_bias"].float(),
        n_group=CFG.n_group,
        topk_group=CFG.topk_group,
        topk=CFG.top_k,
        routed_scaling_factor=CFG.routed_scaling_factor,
        topk_values=topk_weights,
        topk_indices=topk_ids,
    )
    return topk_ids, topk_weights


def apply_routing_distribution(
    inputs, distribution, num_local_experts, local_expert_offset
):
    """Encode an explicit routing profile as logits for logits-mode benchmarks."""
    expert_ids = generate_routing_distribution(
        inputs, distribution, num_local_experts, local_expert_offset
    )
    logits = torch.full_like(inputs["router_logits"], -30.0)
    values = torch.linspace(
        30.0,
        29.0,
        steps=CFG.top_k,
        dtype=logits.dtype,
        device=logits.device,
    )
    logits.scatter_(
        1,
        expert_ids.to(torch.int64),
        values.reshape(1, -1).expand(expert_ids.shape[0], -1),
    )
    inputs["router_logits"] = logits


def generate_routing_distribution(
    inputs, distribution, num_local_experts, local_expert_offset
):
    """Generate the requested caller-provided expert assignments."""
    from flashinfer.fused_moe.dist_aware.da_utils import (
        generate_da_distribution_assignments,
        get_da_distribution_specs,
    )

    num_tokens = inputs["router_logits"].shape[0]
    device = inputs["router_logits"].device
    # Input construction consumes shape-dependent random numbers. Isolate the
    # routing fixture so matched local-expert configurations get the same draw.
    fork_devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(42)
        return generate_da_distribution_assignments(
            get_da_distribution_specs(distribution)[0],
            torch.zeros(
                num_tokens,
                CFG.top_k,
                dtype=torch.int32,
                device=device,
            ),
            num_local_experts,
            CFG.num_experts,
            CFG.top_k,
            local_expert_offset,
        )


def prepare_explicit_routed_input(
    inputs, distribution, num_local_experts, local_expert_offset
):
    """Build routed input without changing it through grouped-logits routing."""
    expert_ids = generate_routing_distribution(
        inputs, distribution, num_local_experts, local_expert_offset
    )
    weights = torch.full(
        expert_ids.shape,
        CFG.routed_scaling_factor / CFG.top_k,
        dtype=torch.float32,
        device=expert_ids.device,
    )
    return expert_ids, weights


# =============================================================================
# Benchmark Functions
# =============================================================================


def bench_cute_dsl(
    inputs,
    warmup=10,
    iters=100,
    num_local_experts=None,
    local_expert_offset=0,
    use_cuda_graph=True,
    use_cupti=True,
    use_wrapper=False,
    do_autotune=True,
    routing_input_mode="logits",
    routed_input=None,
    tuning_buckets=None,
    use_per_token_activation=False,
    use_fused_finalize=True,
):
    """Benchmark CuteDSL MoE.

    Args:
        use_wrapper: If True, use CuteDslMoEWrapper API (recommended for CUDA graph).
                    If False, use cute_dsl_fused_moe_nvfp4 functional API.
        do_autotune: If True, run the pre-warm pass under autotune(True) so the
                    autotuner profiles all buckets and populates its cache. The
                    measurement loop runs OUTSIDE the autotune context so that
                    choose_one cache lookups don't appear inside the CUDA-event
                    interval when bench_gpu_time falls back to events (i.e. when
                    both CUDA graphs and CUPTI are disabled).
        use_fused_finalize: Use atomic fused finalize; otherwise use the
            deterministic two-stage finalize.
    """
    import contextlib

    from flashinfer import SfLayout, nvfp4_quantize
    from flashinfer.autotuner import autotune
    from flashinfer.fused_moe import fused_topk_deepseek
    from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout
    from flashinfer.fp4_quantization import fp4_quantize
    from flashinfer.quantization.nvfp4_quantization_utils import (
        current_nvfp4_4over6_config,
        make_nvfp4_global_scale,
    )
    from flashinfer.testing.utils import bench_gpu_time

    if num_local_experts is None:
        num_local_experts = CFG.num_experts

    n, sv, dev = inputs["router_logits"].shape[0], 16, "cuda"
    gs1 = torch.tensor([1.0], device=dev)

    if routing_input_mode == "routed":
        ti, tv = routed_input or prepare_routed_input(inputs)
    else:
        tv = torch.empty(n, CFG.top_k, dtype=torch.float32, device=dev)
        ti = torch.empty(n, CFG.top_k, dtype=torch.int32, device=dev)

    if use_per_token_activation:
        hidden_global_scale = make_nvfp4_global_scale(
            inputs["hidden_bf16"],
            per_token_activation=True,
            nvfp4_4over6_config=current_nvfp4_4over6_config(),
        )
        xf, xs, hidden_per_token_scale = nvfp4_quantize(
            inputs["hidden_bf16"],
            hidden_global_scale,
            sfLayout=SfLayout.layout_linear,
            per_token_activation=True,
            backend="cute-dsl",
        )
    else:
        xf, xs = fp4_quantize(inputs["hidden_bf16"], gs1, sv, False, False)
        hidden_per_token_scale = None
    xs = xs.view(torch.float8_e4m3fn).reshape(n, CFG.hidden_size // sv).unsqueeze(-1)

    # Expert range for this EP partition
    expert_start = local_expert_offset
    expert_end = local_expert_offset + num_local_experts

    # Slice weights to LOCAL experts only
    w1_local = inputs["w1_bf16"][expert_start:expert_end]
    w2_local = inputs["w2_bf16"][expert_start:expert_end]

    w1i = interleave(w1_local, 64)
    w1f = w1i.view(num_local_experts * 2 * CFG.intermediate_size, CFG.hidden_size)
    w1q, w1s = fp4_quantize(w1f, gs1, sv, False, True)
    w1q = w1q.view(num_local_experts, 2 * CFG.intermediate_size, CFG.hidden_size // 2)
    w1s = convert_sf_to_mma_layout(
        w1s, 2 * CFG.intermediate_size, CFG.hidden_size, num_local_experts, sv
    )

    w2f = w2_local.view(num_local_experts * CFG.hidden_size, CFG.intermediate_size)
    w2q, w2s = fp4_quantize(w2f, gs1, sv, False, True)
    w2q = w2q.view(num_local_experts, CFG.hidden_size, CFG.intermediate_size // 2)
    w2s = convert_sf_to_mma_layout(
        w2s, CFG.hidden_size, CFG.intermediate_size, num_local_experts, sv
    )

    # Alpha sized for LOCAL experts only
    alpha, fc2sc = (
        torch.ones(num_local_experts, device=dev),
        torch.tensor([1.0], device=dev),
    )

    # Pre-convert routing bias to float32
    routing_bias_f32 = inputs["routing_bias"].float()

    if use_wrapper:
        # Use CuteDslMoEWrapper (recommended for CUDA graph)
        from flashinfer import CuteDslMoEWrapper

        moe = CuteDslMoEWrapper(
            num_experts=CFG.num_experts,
            top_k=CFG.top_k,
            hidden_size=CFG.hidden_size,
            intermediate_size=CFG.intermediate_size,
            use_cuda_graph=use_cuda_graph,
            max_num_tokens=n,
            num_local_experts=num_local_experts,
            local_expert_offset=local_expert_offset,
            use_fused_finalize=use_fused_finalize,
        )

        def run(
            x,
            x_sf,
            topk_values,
            topk_indices,
            router_logits=None,
            routing_bias=None,
        ):
            if routing_input_mode == "logits":
                fused_topk_deepseek(
                    scores=router_logits,
                    bias=routing_bias,
                    n_group=CFG.n_group,
                    topk_group=CFG.topk_group,
                    topk=CFG.top_k,
                    routed_scaling_factor=CFG.routed_scaling_factor,
                    topk_values=topk_values,
                    topk_indices=topk_indices,
                )
            return moe.run(
                x=x,
                x_sf=x_sf,
                token_selected_experts=topk_indices,
                token_final_scales=topk_values,
                w1_weight=w1q,
                w1_weight_sf=w1s,
                w1_alpha=alpha,
                fc2_input_scale=fc2sc,
                w2_weight=w2q,
                w2_weight_sf=w2s,
                w2_alpha=alpha,
                per_token_scale=hidden_per_token_scale,
            )
    else:
        # Use functional API
        from flashinfer import cute_dsl_fused_moe_nvfp4

        def run(
            x,
            x_sf,
            topk_values,
            topk_indices,
            router_logits=None,
            routing_bias=None,
        ):
            if routing_input_mode == "logits":
                fused_topk_deepseek(
                    scores=router_logits,
                    bias=routing_bias,
                    n_group=CFG.n_group,
                    topk_group=CFG.topk_group,
                    topk=CFG.top_k,
                    routed_scaling_factor=CFG.routed_scaling_factor,
                    topk_values=topk_values,
                    topk_indices=topk_indices,
                )
            return cute_dsl_fused_moe_nvfp4(
                x=x,
                x_sf=x_sf,
                token_selected_experts=topk_indices,
                token_final_scales=topk_values,
                w1_weight=w1q,
                w1_weight_sf=w1s,
                w1_alpha=alpha,
                fc2_input_scale=fc2sc,
                w2_weight=w2q,
                w2_weight_sf=w2s,
                w2_alpha=alpha,
                num_experts=CFG.num_experts,
                top_k=CFG.top_k,
                num_local_experts=num_local_experts,
                local_expert_offset=local_expert_offset,
                per_token_scale=hidden_per_token_scale,
                use_fused_finalize=use_fused_finalize,
            )

    # Pass input tensors via input_kwargs for cold L2 cache rotation
    input_kwargs = {
        "x": xf,
        "x_sf": xs,
        "topk_values": tv,
        "topk_indices": ti,
    }
    if routing_input_mode == "logits":
        input_kwargs.update(
            router_logits=inputs["router_logits"], routing_bias=routing_bias_f32
        )

    # Pre-warm: run once on the default stream so that autotuning (which
    # allocates tensors and profiles multiple tactics) finishes before
    # bench_gpu_time moves execution to a side stream for CUDA-graph
    # capture.  Autotuning on a non-default stream triggers illegal-
    # memory-access errors in the CuteDSL persistent-tile-scheduler
    # kernels.  Autotune is scoped to the pre-warm only — bench_gpu_time
    # runs outside the autotune context so that choose_one in the
    # measurement loop returns the cached tactic via the non-tuning fast
    # path (no host-side tactic-walking inside the CUDA-event interval).
    with (
        autotune(True, tuning_buckets=tuning_buckets)
        if do_autotune
        else contextlib.nullcontext()
    ):
        run(**input_kwargs)
        torch.cuda.synchronize()

    times = bench_gpu_time(
        run,
        dry_run_iters=warmup,
        repeat_iters=iters,
        cold_l2_cache=True,
        enable_cupti=use_cupti,
        use_cuda_graph=use_cuda_graph,
        input_kwargs=input_kwargs,
    )
    return np.median(times)


def bench_cutlass(
    inputs,
    warmup=10,
    iters=100,
    num_local_experts=None,
    local_expert_offset=0,
    use_cuda_graph=True,
    use_cupti=True,
    do_autotune=True,
    routing_input_mode="logits",
    routed_input=None,
    tuning_buckets=None,
):
    """Benchmark CUTLASS MoE.

    Args:
        do_autotune: See ``bench_cute_dsl`` for the autotune-scope rationale.
    """
    import contextlib

    from flashinfer.autotuner import autotune
    from flashinfer.fused_moe import fused_topk_deepseek, cutlass_fused_moe
    from flashinfer.fp4_quantization import fp4_quantize
    from flashinfer.testing.utils import bench_gpu_time

    if num_local_experts is None:
        num_local_experts = CFG.num_experts

    n, sv, dev = inputs["router_logits"].shape[0], 16, "cuda"

    if routing_input_mode == "routed":
        ti, tv = routed_input or prepare_routed_input(inputs)
    else:
        tv = torch.empty(n, CFG.top_k, dtype=torch.float32, device=dev)
        ti = torch.empty(n, CFG.top_k, dtype=torch.int32, device=dev)

    # Expert range for this EP partition
    expert_start = local_expert_offset
    expert_end = local_expert_offset + num_local_experts

    # Slice weights to LOCAL experts only (for fair EP comparison)
    w1_fp4_local = inputs["w1_fp4"][expert_start:expert_end]
    w1_sf_local = inputs["w1_sf"][expert_start:expert_end]
    w1_gs_local = inputs["w1_gs"][expert_start:expert_end]
    w2_fp4_local = inputs["w2_fp4"][expert_start:expert_end]
    w2_sf_local = inputs["w2_sf"][expert_start:expert_end]
    w2_gs_local = inputs["w2_gs"][expert_start:expert_end]

    # Prepare CUTLASS inputs
    a1_gs = torch.tensor(1.0, device=dev, dtype=torch.float32)
    a2_gs = torch.tensor(1.0, device=dev, dtype=torch.float32)

    quant_scales = [
        a1_gs,
        w1_sf_local.view(torch.int32),
        1.0 / (a1_gs * w1_gs_local),
        a2_gs,
        w2_sf_local.view(torch.int32),
        1.0 / (a2_gs * w2_gs_local),
    ]

    hidden_fp4, input_sf = fp4_quantize(inputs["hidden_bf16"], a1_gs, sv, False, True)
    output = torch.empty(n, CFG.hidden_size, dtype=torch.bfloat16, device=dev)

    # Pre-convert routing bias to float32
    routing_bias_f32 = inputs["routing_bias"].float()

    # Pre-compute values that need conversion
    w1_fp4_view = w1_fp4_local.contiguous().view(torch.long)
    w2_fp4_view = w2_fp4_local.contiguous().view(torch.long)

    # Compute EP size from config
    ep_size = CFG.num_experts // num_local_experts

    def run(
        hidden,
        sf,
        topk_values,
        topk_indices,
        router_logits=None,
        routing_bias=None,
    ):
        if routing_input_mode == "logits":
            fused_topk_deepseek(
                scores=router_logits,
                bias=routing_bias,
                n_group=CFG.n_group,
                topk_group=CFG.topk_group,
                topk=CFG.top_k,
                routed_scaling_factor=CFG.routed_scaling_factor,
                topk_values=topk_values,
                topk_indices=topk_indices,
            )
        cutlass_fused_moe(
            hidden,
            topk_indices.to(torch.int),
            topk_values,
            w1_fp4_view,
            w2_fp4_view,
            torch.bfloat16,
            quant_scales=quant_scales,
            input_sf=sf,
            output=output,
            ep_size=ep_size,
            ep_rank=0,  # Simulating rank 0 of EP
        )
        return output

    input_kwargs = {
        "hidden": hidden_fp4,
        "sf": input_sf,
        "topk_values": tv,
        "topk_indices": ti,
    }
    if routing_input_mode == "logits":
        input_kwargs.update(
            router_logits=inputs["router_logits"], routing_bias=routing_bias_f32
        )

    # Pre-warm under autotune; measurement runs outside (see bench_cute_dsl).
    with (
        autotune(True, tuning_buckets=tuning_buckets)
        if do_autotune
        else contextlib.nullcontext()
    ):
        run(**input_kwargs)
        torch.cuda.synchronize()

    times = bench_gpu_time(
        run,
        dry_run_iters=warmup,
        repeat_iters=iters,
        cold_l2_cache=True,
        enable_cupti=use_cupti,
        use_cuda_graph=use_cuda_graph,
        input_kwargs=input_kwargs,
    )
    return np.median(times)


def bench_trtllm(
    inputs,
    warmup=10,
    iters=100,
    num_local_experts=None,
    local_expert_offset=0,
    use_cuda_graph=True,
    use_cupti=True,
    do_autotune=True,
    routing_input_mode="logits",
    routed_input=None,
    tuning_buckets=None,
    use_per_token_activation=False,
):
    """Benchmark TRT-LLM-Gen MoE.

    Args:
        do_autotune: See ``bench_cute_dsl`` for the autotune-scope rationale.
    """
    import contextlib

    from flashinfer import SfLayout, nvfp4_quantize
    from flashinfer.autotuner import autotune
    from flashinfer.fused_moe import (
        RoutingMethodType,
        trtllm_fp4_block_scale_moe,
        trtllm_fp4_block_scale_routed_moe,
    )
    from flashinfer.fused_moe.core import (
        _maybe_get_cached_w3_w1_permute_indices,
        get_w2_permute_indices_with_cache,
    )
    from flashinfer.fp4_quantization import fp4_quantize, block_scale_interleave
    from flashinfer.quantization.nvfp4_quantization_utils import (
        current_nvfp4_4over6_config,
        make_nvfp4_global_scale,
    )
    from flashinfer.testing.utils import bench_gpu_time

    if num_local_experts is None:
        num_local_experts = CFG.num_experts

    n, dev = inputs["router_logits"].shape[0], inputs["router_logits"].device
    sv, etm, cache = 16, 128, {}

    # Expert range for this EP partition
    expert_start = local_expert_offset
    expert_end = local_expert_offset + num_local_experts

    hg = inputs["hidden_gs"]
    if use_per_token_activation:
        hidden_global_scale = make_nvfp4_global_scale(
            inputs["hidden_bf16"],
            per_token_activation=True,
            nvfp4_4over6_config=current_nvfp4_4over6_config(),
        )
        hfp, hsf, hidden_per_token_scale = nvfp4_quantize(
            inputs["hidden_bf16"],
            hidden_global_scale,
            sfLayout=SfLayout.layout_linear,
            per_token_activation=True,
            backend="cute-dsl",
        )
    else:
        hfp, hsf = fp4_quantize(inputs["hidden_bf16"], hg, sv, False, True)
        hidden_per_token_scale = None
    hfp = hfp.view(torch.uint8).reshape(n, CFG.hidden_size // 2)
    hsc = (
        hsf.view(torch.float8_e4m3fn)
        .flatten()[: n * CFG.hidden_size // sv]
        .reshape(n, CFG.hidden_size // sv)
    )

    def prep(bf16, gs, M, K):
        """Prepare weights for LOCAL experts only."""
        fl, sl = [], []
        for e in range(expert_start, expert_end):
            q, s = fp4_quantize(bf16[e], gs[e], sv, False, False)
            fl.append(q.view(torch.uint8).reshape(M, K // 2))
            sl.append(s.view(torch.float8_e4m3fn).reshape(M, K // sv))
        return torch.stack(fl), torch.stack(sl)

    w1f, w1s = prep(
        inputs["w1_bf16"], inputs["w1_gs"], 2 * CFG.intermediate_size, CFG.hidden_size
    )
    w2f, w2s = prep(
        inputs["w2_bf16"], inputs["w2_gs"], CFG.hidden_size, CFG.intermediate_size
    )

    def shuf(fp4, sf, perm_fn):
        """Shuffle weights for LOCAL experts only."""
        fsh, ssh = [], []
        for i in range(num_local_experts):
            p = perm_fn(cache, fp4[i], etm)
            fsh.append(fp4[i][p.to(dev)].contiguous())
            ps = perm_fn(cache, sf[i].view(torch.uint8), etm, sv)
            ssh.append(
                block_scale_interleave(sf[i].view(torch.uint8)[ps.to(dev)].contiguous())
            )
        return torch.stack(fsh), torch.stack(ssh)

    w1f, w1s = shuf(w1f, w1s, _maybe_get_cached_w3_w1_permute_indices)
    w2f, w2s = shuf(w2f, w2s, get_w2_permute_indices_with_cache)
    w1s = w1s.view(torch.float8_e4m3fn).reshape(
        num_local_experts, 2 * CFG.intermediate_size, CFG.hidden_size // sv
    )
    w2s = w2s.view(torch.float8_e4m3fn).reshape(
        num_local_experts, CFG.hidden_size, CFG.intermediate_size // sv
    )

    # Scale tensors sized for LOCAL experts only
    sc = torch.ones(num_local_experts, device=dev, dtype=torch.float32)

    common_kwargs = dict(
        gemm1_weights=w1f,
        gemm1_weights_scale=w1s,
        gemm1_bias=None,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        gemm2_weights=w2f,
        gemm2_weights_scale=w2s,
        gemm2_bias=None,
        output1_scale_scalar=sc,
        output1_scale_gate_scalar=sc,
        output2_scale_scalar=sc,
        num_experts=CFG.num_experts,
        top_k=CFG.top_k,
        n_group=CFG.n_group,
        topk_group=CFG.topk_group,
        intermediate_size=CFG.intermediate_size,
        local_expert_offset=local_expert_offset,
        local_num_experts=num_local_experts,
        routed_scaling_factor=CFG.routed_scaling_factor,
        routing_method_type=RoutingMethodType.DeepSeekV3,
        per_token_scale=hidden_per_token_scale,
        do_finalize=True,
    )

    if routing_input_mode == "routed":
        topk_ids, topk_weights = routed_input or prepare_routed_input(inputs)

        def run(topk_ids, topk_weights, hidden_states, hidden_states_scale):
            return trtllm_fp4_block_scale_routed_moe(
                topk_ids=(topk_ids, topk_weights),
                routing_bias=None,
                hidden_states=hidden_states,
                hidden_states_scale=hidden_states_scale,
                **common_kwargs,
            )

        input_kwargs = {
            "topk_ids": topk_ids,
            "topk_weights": topk_weights,
            "hidden_states": hfp,
            "hidden_states_scale": hsc,
        }
    else:

        def run(routing_logits, routing_bias, hidden_states, hidden_states_scale):
            return trtllm_fp4_block_scale_moe(
                routing_logits=routing_logits,
                routing_bias=routing_bias,
                hidden_states=hidden_states,
                hidden_states_scale=hidden_states_scale,
                **common_kwargs,
            )

        input_kwargs = {
            "routing_logits": inputs["router_logits"],
            "routing_bias": inputs["routing_bias"],
            "hidden_states": hfp,
            "hidden_states_scale": hsc,
        }

    with (
        autotune(True, tuning_buckets=tuning_buckets)
        if do_autotune
        else contextlib.nullcontext()
    ):
        run(**input_kwargs)
        torch.cuda.synchronize()

    times = bench_gpu_time(
        run,
        dry_run_iters=warmup,
        repeat_iters=iters,
        cold_l2_cache=True,
        enable_cupti=use_cupti,
        use_cuda_graph=use_cuda_graph,
        input_kwargs=input_kwargs,
    )
    return np.median(times)


# =============================================================================
# Autotune
# =============================================================================


# =============================================================================
# Main Benchmark
# =============================================================================


@dataclass
class BenchResult:
    """Single benchmark result for one backend at one token count."""

    backend: str
    tokens: int
    latency_ms: float
    tflops: float


def run_benchmark(
    token_counts,
    warmup=10,
    iters=100,
    ep_config=1,
    do_autotune=True,
    verbose=True,
    use_cuda_graph=True,
    use_cupti=True,
    use_wrapper=True,
    routing_bias_scale=0.01,
    routing_input_mode="logits",
    backends=BACKENDS,
    distributions=None,
    use_per_token_activation=False,
    use_fused_finalize=True,
):
    """
    Unified benchmark for DeepSeek-V3 MoE backends.

    Autotuning runs in each backend's pre-warm step (under ``autotune(True)``)
    so the autotuner profiles all tactics on the default stream before
    ``bench_gpu_time`` switches to a side stream / captures into a CUDA graph.
    This guarantees the autotuner sees the same API (wrapper vs functional),
    EP config, and weight shapes as the timed runs, avoiding cache-key
    mismatches; the measurement loop itself runs outside ``autotune(True)``
    so ``choose_one`` takes the non-tuning fast path and host-side cache
    walks don't appear inside the CUDA-event interval.

    The first benchmark row profiles every requested token bucket. Remaining
    rows consume those cache entries without re-entering autotune.

    All output is buffered and printed after the benchmark (and autotuning)
    completes, so autotuner log messages do not interleave with the results
    table.

    Args:
        token_counts: List of token counts to benchmark
        warmup: Warmup iterations
        iters: Benchmark iterations
        ep_config: Expert Parallelism config (1, 8, or 16)
        do_autotune: Whether to autotune during benchmarking
        verbose: Print results to stdout
        use_cuda_graph: Whether to use CUDA graph for benchmarking
        use_cupti: Whether to use CUPTI for accurate GPU timing
        use_wrapper: Whether to use CuteDslMoEWrapper API (recommended)
        routing_bias_scale: Scale for random routing bias generation
        routing_input_mode: Routing input for every selected backend
        backends: Backends to benchmark
        distributions: Optional explicit uniform/ddist routing profiles. The
            default preserves the benchmark's random router logits.
        use_per_token_activation: Whether supported FP4 MoE backends should use
            per-token NVFP4 activation scaling.
        use_fused_finalize: Use atomic fused finalize; otherwise use the
            deterministic two-stage finalize.

    Returns:
        List of BenchResult objects
    """
    if use_per_token_activation and not {"cutedsl", "trtllm"}.intersection(backends):
        raise ValueError(
            "Per-token activation requires the cutedsl or trtllm backend; "
            "CUTLASS does not consume the per-token activation scale."
        )

    # Simulate rank 0 of the requested expert-parallel configuration.
    num_local = CFG.num_experts // ep_config
    local_offset = 0

    from flashinfer.fused_moe.utils import get_hybrid_num_tokens_buckets

    tuning_buckets = get_hybrid_num_tokens_buckets(max(token_counts), min(token_counts))
    results = []
    rows_and_histograms = []

    # Note: autotune(True) is now scoped to each bench_*'s pre-warm rather than
    # wrapping the entire measurement loop.  See bench_cute_dsl for rationale.
    benchmark_distributions = distributions or (None,)
    autotune_pending = bool(do_autotune)
    for distribution in benchmark_distributions:
        for n in token_counts:
            row, histogram_record = _benchmark_single(
                n,
                warmup,
                iters,
                num_local,
                local_offset,
                use_cuda_graph,
                use_cupti,
                use_wrapper=use_wrapper,
                routing_bias_scale=routing_bias_scale,
                do_autotune=autotune_pending,
                routing_input_mode=routing_input_mode,
                backends=backends,
                distribution=distribution,
                tuning_buckets=tuning_buckets,
                use_per_token_activation=use_per_token_activation,
                use_fused_finalize=use_fused_finalize,
            )
            results.extend(row)
            rows_and_histograms.append((row, histogram_record, distribution))
            # The first call profiles every requested token bucket for every
            # selected backend. Subsequent rows must only consume that cache;
            # re-entering autotune adds noise and obscures accidental misses.
            autotune_pending = False

    if verbose:
        _print_header(
            ep_config,
            num_local,
            use_cuda_graph,
            use_cupti,
            routing_bias_scale,
            routing_input_mode,
            backends,
            use_per_token_activation=use_per_token_activation,
            use_fused_finalize=use_fused_finalize,
        )
        current_distribution = None
        for row, histogram_record, distribution in rows_and_histograms:
            if distribution is not None and distribution != current_distribution:
                print(f"Distribution: {distribution}")
                current_distribution = distribution
            _print_row(
                row,
                histogram_record,
                use_per_token_activation=use_per_token_activation,
            )
        _print_footer(ep_config, num_local)

    return results


def _benchmark_single(
    n,
    warmup,
    iters,
    num_local,
    local_offset,
    use_cuda_graph,
    use_cupti,
    use_wrapper=True,
    routing_bias_scale=0.01,
    do_autotune=True,
    routing_input_mode="logits",
    backends=BACKENDS,
    distribution=None,
    tuning_buckets=None,
    use_per_token_activation=False,
    use_fused_finalize=True,
):
    """Benchmark all backends for a single token count.

    Args:
        use_wrapper: If True, use CuteDslMoEWrapper API for CuteDSL.
        do_autotune: Forwarded to each bench_* function — wraps pre-warm only.
    """
    inputs = create_inputs(n, routing_bias_scale=routing_bias_scale)
    routed_input = None
    if distribution is not None and routing_input_mode == "routed":
        routed_input = prepare_explicit_routed_input(
            inputs, distribution, num_local, local_offset
        )
    elif distribution is not None:
        apply_routing_distribution(inputs, distribution, num_local, local_offset)
    elif routing_input_mode == "routed":
        routed_input = prepare_routed_input(inputs)
    histogram_record = _collect_expert_histogram(
        inputs,
        num_local,
        local_offset,
        routed_ids=None if routed_input is None else routed_input[0],
    )
    lat = {}
    if "cutedsl" in backends:
        lat["CuteDSL"] = bench_cute_dsl(
            inputs,
            warmup,
            iters,
            num_local,
            local_offset,
            use_cuda_graph,
            use_cupti,
            use_wrapper=use_wrapper,
            do_autotune=do_autotune,
            routing_input_mode=routing_input_mode,
            routed_input=routed_input,
            tuning_buckets=tuning_buckets,
            use_per_token_activation=use_per_token_activation,
            use_fused_finalize=use_fused_finalize,
        )
    if "cutlass" in backends and not use_per_token_activation:
        lat["CUTLASS"] = bench_cutlass(
            inputs,
            warmup,
            iters,
            num_local,
            local_offset,
            use_cuda_graph,
            use_cupti,
            do_autotune=do_autotune,
            routing_input_mode=routing_input_mode,
            routed_input=routed_input,
            tuning_buckets=tuning_buckets,
        )
    if "trtllm" in backends:
        lat["TRTLLM"] = bench_trtllm(
            inputs,
            warmup,
            iters,
            num_local,
            local_offset,
            use_cuda_graph,
            use_cupti,
            do_autotune=do_autotune,
            routing_input_mode=routing_input_mode,
            routed_input=routed_input,
            tuning_buckets=tuning_buckets,
            use_per_token_activation=use_per_token_activation,
        )

    # Build results
    results = []
    for backend, latency in lat.items():
        results.append(
            BenchResult(
                backend=backend,
                tokens=n,
                latency_ms=latency,
                tflops=calc_tflops(n, latency, num_local),
            )
        )
    return results, histogram_record


def _print_header(
    ep_config,
    num_local,
    use_cuda_graph,
    use_cupti,
    routing_bias_scale,
    routing_input_mode,
    backends,
    use_per_token_activation=False,
    use_fused_finalize=True,
):
    """Print benchmark header."""
    print("\n" + "=" * 120)
    if use_per_token_activation:
        print(f"DeepSeek-V3 MoE Benchmark: CuteDSL vs TRTLLM (EP={ep_config})")
    else:
        print(
            f"DeepSeek-V3 MoE Benchmark: CuteDSL vs CUTLASS vs TRTLLM (EP={ep_config})"
        )
    print("=" * 120)
    print(
        f"Model: hidden={CFG.hidden_size}, intermediate={CFG.intermediate_size}, "
        f"experts={CFG.num_experts}, top_k={CFG.top_k}"
    )
    print(
        f"EP Config: {num_local} local experts (simulating {CFG.num_experts // num_local}-way parallelism)"
    )
    print(
        f"CUDA Graph: {'enabled' if use_cuda_graph else 'disabled'}, CUPTI: {'enabled' if use_cupti else 'disabled'}"
    )
    print(
        f"Routing bias scale: {routing_bias_scale} "
        f"(larger values tend to create expert imbalance)"
    )
    print(f"Routing input: {routing_input_mode}")
    print(f"Backends: {','.join(backends)}")
    print(
        "CuteDSL finalize: "
        f"{'atomic fused' if use_fused_finalize else 'deterministic two-stage'}"
    )
    if use_per_token_activation:
        print("CUTLASS omitted: it does not consume the per-token activation scale.")
    print("-" * 120)
    if use_per_token_activation:
        print(
            f"{'Tokens':>6} | "
            f"{'CuteDSL':^15} | "
            f"{'TRTLLM':^15} | "
            f"{'Speedup':^9} | "
            f"{'Winner':^8} | "
            f"{'Active':^7} | "
            f"{'Stats':^14}"
        )
        print(
            f"{'':>6} | "
            f"{'ms':>7} {'TFLOPS':>7} | "
            f"{'ms':>7} {'TFLOPS':>7} | "
            f"{'TRTLLM':>9} | "
            f"{'':^8} | "
            f"{'experts':^7} | "
            f"{'min/max/median':^14}"
        )
    else:
        print(
            f"{'Tokens':>6} | "
            f"{'CuteDSL':^15} | "
            f"{'CUTLASS':^15} | "
            f"{'TRTLLM':^15} | "
            f"{'Speedup (CuteDSL/X)':^18} | "
            f"{'Winner':^8} | "
            f"{'Active':^7} | "
            f"{'Stats':^14}"
        )
        print(
            f"{'':>6} | "
            f"{'ms':>7} {'TFLOPS':>7} | "
            f"{'ms':>7} {'TFLOPS':>7} | "
            f"{'ms':>7} {'TFLOPS':>7} | "
            f"{'CUTLASS':>9} {'TRTLLM':>9} | "
            f"{'':^8} | "
            f"{'experts':^7} | "
            f"{'min/max/median':^14}"
        )
    print("-" * 120)


def _print_row(results, histogram_record, use_per_token_activation=False):
    """Print a single row of benchmark results."""
    # Extract values by backend
    r = {r.backend: r for r in results}
    cute = r.get("CuteDSL")
    cutlass = r.get("CUTLASS")
    trtllm = r.get("TRTLLM")

    def backend_columns(result):
        if result is None:
            return f"{'n/a':>7} {'n/a':>7}"
        return f"{result.latency_ms:>7.3f} {result.tflops:>7.1f}"

    def speedup_column(result):
        if cute is None or result is None:
            return f"{'n/a':>9}"
        return f"{result.latency_ms / cute.latency_ms:>8.2f}x"

    # Find winner
    winner = min(r.values(), key=lambda x: x.latency_ms).backend

    active_experts = f"{histogram_record['active_local_experts']:>3}"
    stats = (
        f"{histogram_record['min_count']:>3}/"
        f"{histogram_record['max_count']:>3}/"
        f"{histogram_record['median_count']:>7.2f}"
    )
    if use_per_token_activation:
        print(
            f"{results[0].tokens:>6} | "
            f"{backend_columns(cute)} | "
            f"{backend_columns(trtllm)} | "
            f"{speedup_column(trtllm)} | "
            f"{winner:^8} | "
            f"{active_experts:>7} | "
            f"{stats:>14}"
        )
    else:
        print(
            f"{results[0].tokens:>6} | "
            f"{backend_columns(cute)} | "
            f"{backend_columns(cutlass)} | "
            f"{backend_columns(trtllm)} | "
            f"{speedup_column(cutlass)} {speedup_column(trtllm)} | "
            f"{winner:^8} | "
            f"{active_experts:>7} | "
            f"{stats:>14}"
        )


def _print_footer(ep_config, num_local):
    """Print benchmark footer."""
    print("-" * 120)
    print("Speedup > 1.0 means CuteDSL is faster than that backend")


def _collect_expert_histogram(inputs, num_local, local_offset, routed_ids=None):
    if routed_ids is None:
        topk_indices, _ = prepare_routed_input(inputs)
    else:
        topk_indices = routed_ids

    expert_hist = torch.bincount(
        topk_indices.reshape(-1).to(torch.int64), minlength=CFG.num_experts
    )
    local_hist = expert_hist[local_offset : local_offset + num_local]
    local_hist_f32 = local_hist.to(torch.float32)
    active_local_experts = int((local_hist > 0).sum().item())
    if local_hist.numel() > 0:
        min_count = int(local_hist.min().item())
        max_count = int(local_hist.max().item())
        median_count = float(torch.quantile(local_hist_f32, 0.5).item())
    else:
        min_count = 0
        max_count = 0
        median_count = 0.0

    return {
        "active_local_experts": active_local_experts,
        "min_count": min_count,
        "max_count": max_count,
        "median_count": median_count,
    }


def main():
    parser = argparse.ArgumentParser(
        description="DeepSeek-V3 MoE Performance Benchmark"
    )
    parser.add_argument(
        "--num-tokens",
        type=str,
        default=None,
        help="Comma-separated token counts (default: 128-4096 for throughput, 1-128 for gen-phase)",
    )
    parser.add_argument(
        "--num-experts",
        type=int,
        default=DeepSeekConfig().num_experts,
        help="Total routed expert count (default: %(default)s)",
    )
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=100, help="Benchmark iterations")
    parser.add_argument("--no-autotune", action="store_true", help="Disable autotune")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    parser.add_argument(
        "--gen-phase",
        action="store_true",
        help="Use generation phase token counts (1-128 instead of 128-4096)",
    )
    parser.add_argument(
        "--ep",
        type=int,
        default=1,
        choices=[1, 4, 8, 16],
        help="Expert-parallel degree; local experts = --num-experts / --ep",
    )
    parser.add_argument(
        "--no-cuda-graph",
        action="store_true",
        help="Disable CUDA graph for benchmarking (enabled by default)",
    )
    parser.add_argument(
        "--no-cupti",
        action="store_true",
        help="Disable CUPTI for GPU timing (enabled by default)",
    )
    parser.add_argument(
        "--functional-api",
        action="store_true",
        help="Use functional API instead of CuteDslMoEWrapper for CuteDSL benchmark",
    )
    parser.add_argument(
        "--use-per-token-activation",
        action="store_true",
        help="Use per-token NVFP4 activation scaling for supported FP4 MoE backends.",
    )
    parser.add_argument(
        "--no-fused-finalize",
        action="store_false",
        dest="use_fused_finalize",
        help="Use deterministic two-stage CuTe DSL finalize instead of atomic fused finalize.",
    )
    parser.add_argument(
        "--routing-bias-scale",
        type=float,
        default=0.01,
        help="Scale for random routing bias. Larger values tend to create expert imbalance.",
    )
    parser.add_argument(
        "--routing-input-mode",
        choices=("logits", "routed"),
        default="logits",
        help=(
            "Compute routing inside each timed backend graph from logits, or "
            "precompute caller-provided IDs/weights (default: logits)"
        ),
    )
    parser.add_argument(
        "--backends",
        type=parse_backends,
        default=BACKENDS,
        metavar="BACKEND,...",
        help="Backends to run: cutedsl,cutlass,trtllm (default: all)",
    )
    parser.add_argument(
        "--distributions",
        type=parse_distributions,
        default=None,
        help=(
            "Comma-separated explicit routing distributions, using the same "
            "uniform/ddist grammar as bench_trtllm_moe_da.py, or 'full' for "
            "uniform + ddist:1.1..8.0. By default, use the benchmark's random "
            "router logits."
        ),
    )
    args = parser.parse_args()

    if args.num_experts < CFG.top_k:
        parser.error(f"--num-experts must be at least top-k ({CFG.top_k})")
    if args.num_experts % CFG.n_group != 0:
        parser.error(f"--num-experts must be divisible by n-group ({CFG.n_group})")
    if args.num_experts % args.ep != 0:
        parser.error("--num-experts must be divisible by --ep")
    CFG.num_experts = args.num_experts

    if not is_sm100_family():
        print("ERROR: Requires SM100 family GPU (Blackwell: SM100, SM103)")
        return 1

    # Determine token counts
    if args.num_tokens:
        tokens = [int(x) for x in args.num_tokens.split(",")]
    elif args.gen_phase:
        tokens = GEN_PHASE_TOKENS  # [1, 2, 4, 8, 16, 32, 64, 128]
    else:
        tokens = TOKEN_COUNTS  # [128, 256, 512, 1024, 2048, 4096]

    print("\nDeepSeek-V3 MoE Performance Benchmark")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CuteDSL API: {'Functional' if args.functional_api else 'Wrapper'}")
    print(f"Routing input: {args.routing_input_mode}")
    print(f"Backends: {','.join(args.backends)}")
    print(f"Per-token activation: {args.use_per_token_activation}")
    print(
        "CuteDSL finalize: "
        f"{'atomic fused' if args.use_fused_finalize else 'deterministic two-stage'}"
    )

    run_benchmark(
        token_counts=tokens,
        warmup=args.warmup,
        iters=args.iters,
        ep_config=args.ep,
        do_autotune=not args.no_autotune,
        verbose=not args.quiet,
        use_cuda_graph=not args.no_cuda_graph,
        use_cupti=not args.no_cupti,
        use_wrapper=not args.functional_api,
        routing_bias_scale=args.routing_bias_scale,
        routing_input_mode=args.routing_input_mode,
        backends=args.backends,
        distributions=args.distributions,
        use_per_token_activation=args.use_per_token_activation,
        use_fused_finalize=args.use_fused_finalize,
    )

    return 0


if __name__ == "__main__":
    exit(main())
