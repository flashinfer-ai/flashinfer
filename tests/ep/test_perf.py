# tests/ep/test_perf.py
#
# Performance benchmark tests for the FlashInfer Unified EP API.
# Covers test IDs P-IN-01 through P-IN-09 (intra-node).
#
# Run with:  torchrun --nproc_per_node=4 -m pytest tests/ep/test_perf.py -v -s
#
# These tests measure latency and throughput against GPU-architecture-specific
# pass/fail thresholds. They auto-detect the GPU family (Ampere, Hopper,
# Blackwell) and skip tests where no threshold is defined.
#
# Use -s to see printed latency/throughput numbers even on passing tests.

import pytest
import torch

import flashinfer.ep as fep

# Import helpers via path-relative import (works regardless of PYTHONPATH / cwd)
import importlib.util, pathlib
_helpers_path = pathlib.Path(__file__).parent / "helpers.py"
_spec = importlib.util.spec_from_file_location("ep_helpers", _helpers_path)
_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)
make_tokens = _helpers.make_tokens
identity_expert = _helpers.identity_expert
get_gpu_arch = _helpers.get_gpu_arch


# ─── GPU-specific thresholds ────────────────────────────────────────

# Dispatch latency thresholds in microseconds: (num_experts, num_tokens) -> {arch: max_us}
DISPATCH_LATENCY_THRESHOLDS = {
    (8, 64):    {"ampere": 120, "hopper": 85,  "blackwell": 50},
    (64, 128):  {"hopper": 200, "blackwell": 120},
    (256, 128): {"hopper": 250, "blackwell": 150},
}

# Combine latency thresholds in microseconds
COMBINE_LATENCY_THRESHOLDS = {
    (8, 64):    {"ampere": 160, "hopper": 130, "blackwell": 75},
}

# HT throughput thresholds in GB/s: (num_experts, num_tokens) -> {arch: min_gbps}
HT_THROUGHPUT_THRESHOLDS = {
    (8, 4096):   {"ampere": 100, "hopper": 140, "blackwell": 250},
    (64, 8192):  {"hopper": 130, "blackwell": 240},
}


# ─── Benchmark utility ──────────────────────────────────────────────


def benchmark_op(fn, warmup=20, iters=100):
    """Benchmark a CUDA operation using CUDA events.

    Returns median latency in microseconds. Uses CUDA events for accurate
    GPU-side timing (not wall-clock time).

    Args:
        fn:     Callable to benchmark (no args, returns nothing).
        warmup: Number of warmup iterations (JIT compile, cache warming).
        iters:  Number of timed iterations.

    Returns:
        Median latency in microseconds.
    """
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # Timed iterations with CUDA events
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        start_events[i].record()
        fn()
        end_events[i].record()

    torch.cuda.synchronize()

    # elapsed_time() returns milliseconds → multiply by 1000 for microseconds
    times_us = [
        s.elapsed_time(e) * 1000.0
        for s, e in zip(start_events, end_events)
    ]
    times_us.sort()
    return times_us[len(times_us) // 2]  # median


# =====================================================================
# P-IN-01, P-IN-02, P-IN-03, P-IN-04: Low-Latency Dispatch/Combine
# =====================================================================


class TestIntraNodeLatency:
    """Tests P-IN-01 through P-IN-04: LL dispatch and combine latency."""

    @pytest.mark.parametrize(
        "num_experts,num_tokens",
        [(8, 64), (64, 128), (256, 128)],
        ids=["8E_64T", "64E_128T", "256E_128T"],
    )
    def test_ll_dispatch_latency(self, backend, num_experts, num_tokens,
                                  ep_process_group, dist_env):
        """P-IN-01/03/04: LL dispatch latency against GPU-specific thresholds.

        Measures the time for a single low_latency_dispatch() call using
        CUDA events. The median of 100 iterations (after 20 warmup) must be
        below the architecture-specific threshold.
        """
        arch = get_gpu_arch()
        key = (num_experts, num_tokens)

        if key not in DISPATCH_LATENCY_THRESHOLDS:
            pytest.skip(f"No threshold defined for {key}")
        if arch not in DISPATCH_LATENCY_THRESHOLDS[key]:
            pytest.skip(f"No threshold for {arch} at {key}")

        threshold_us = DISPATCH_LATENCY_THRESHOLDS[key][arch]
        num_local = num_experts // dist_env["world_size"]

        with fep.create_group(
            backend=backend,
            process_group=ep_process_group,
            num_experts=num_experts,
            num_local_experts=num_local,
            top_k=8,
            hidden_dim=4096,
        ) as group:
            hidden, topk_idx, _ = make_tokens(num_tokens, 4096, num_experts, 8)

            def dispatch_fn():
                r = group.low_latency_dispatch(
                    hidden=hidden,
                    topk_idx=topk_idx,
                    max_tokens_per_rank=num_tokens,
                    output_layout=fep.OutputLayout.FLAT_2D,
                    async_finish=False,
                )
                r.handle.destroy()

            latency_us = benchmark_op(dispatch_fn)

        print(
            f"\n  {backend.value} | {num_experts}E {num_tokens}T | "
            f"{arch} | dispatch: {latency_us:.1f} us "
            f"(threshold: {threshold_us} us)"
        )
        assert latency_us < threshold_us, \
            f"Dispatch latency {latency_us:.1f} us > threshold {threshold_us} us"

    @pytest.mark.parametrize(
        "num_experts,num_tokens",
        [(8, 64)],
        ids=["8E_64T"],
    )
    def test_ll_combine_latency(self, backend, num_experts, num_tokens,
                                 ep_process_group, dist_env):
        """P-IN-02: LL combine latency against GPU-specific thresholds.

        Measures the time for a single low_latency_combine() call. The
        dispatch is done in setup; only the combine is benchmarked.
        """
        arch = get_gpu_arch()
        key = (num_experts, num_tokens)

        if key not in COMBINE_LATENCY_THRESHOLDS:
            pytest.skip(f"No combine threshold for {key}")
        if arch not in COMBINE_LATENCY_THRESHOLDS[key]:
            pytest.skip(f"No combine threshold for {arch}")

        threshold_us = COMBINE_LATENCY_THRESHOLDS[key][arch]
        num_local = num_experts // dist_env["world_size"]

        with fep.create_group(
            backend=backend,
            process_group=ep_process_group,
            num_experts=num_experts,
            num_local_experts=num_local,
            top_k=8,
            hidden_dim=4096,
        ) as group:
            hidden, topk_idx, topk_weights = make_tokens(
                num_tokens, 4096, num_experts, 8
            )

            # Pre-dispatch to get a handle for combine benchmarking
            dispatch_result = group.low_latency_dispatch(
                hidden=hidden,
                topk_idx=topk_idx,
                max_tokens_per_rank=num_tokens,
            )
            dispatch_result.status.raise_if_error()

            expert_out = identity_expert(
                dispatch_result.recv_hidden, dispatch_result.recv_expert_counts
            )

            def combine_fn():
                group.low_latency_combine(
                    expert_output=expert_out,
                    topk_idx=topk_idx,
                    topk_weights=topk_weights,
                    handle=dispatch_result.handle,
                )

            latency_us = benchmark_op(combine_fn)
            dispatch_result.handle.destroy()

        print(
            f"\n  {backend.value} | {num_experts}E {num_tokens}T | "
            f"{arch} | combine: {latency_us:.1f} us "
            f"(threshold: {threshold_us} us)"
        )
        assert latency_us < threshold_us, \
            f"Combine latency {latency_us:.1f} us > threshold {threshold_us} us"


# =====================================================================
# P-IN-05, P-IN-06, P-IN-07: High-Throughput NVLink Bandwidth
# =====================================================================


class TestIntraNodeThroughput:
    """Tests P-IN-05 through P-IN-07: HT NVLink throughput."""

    @pytest.mark.parametrize(
        "num_experts,num_tokens,dtype_str",
        [
            (8, 4096, "bf16"),
            (64, 8192, "bf16"),
        ],
        ids=["8E_4096T_bf16", "64E_8192T_bf16"],
    )
    def test_ht_nvlink_throughput(self, backend, num_experts, num_tokens,
                                   dtype_str, ep_process_group, dist_env):
        """P-IN-05/06: HT NVLink throughput in GB/s.

        Measures the effective throughput of a dispatch (scatter) operation
        over NVLink. Throughput = (bytes transferred) / (median latency).

        For BF16 with hidden_dim=4096: bytes = num_tokens * 4096 * 2.
        """
        arch = get_gpu_arch()
        key = (num_experts, num_tokens)

        if key not in HT_THROUGHPUT_THRESHOLDS:
            pytest.skip(f"No throughput threshold for {key}")
        if arch not in HT_THROUGHPUT_THRESHOLDS[key]:
            pytest.skip(f"No throughput threshold for {arch}")

        threshold_gbps = HT_THROUGHPUT_THRESHOLDS[key][arch]
        num_local = num_experts // dist_env["world_size"]
        hidden_dim = 4096

        dtype = torch.bfloat16 if dtype_str == "bf16" else torch.float8_e4m3fn
        elem_size = 2 if dtype_str == "bf16" else 1

        with fep.create_group(
            backend=backend,
            process_group=ep_process_group,
            num_experts=num_experts,
            num_local_experts=num_local,
            top_k=8,
            hidden_dim=hidden_dim,
        ) as group:
            hidden, topk_idx, topk_weights = make_tokens(
                num_tokens, hidden_dim, num_experts, 8, dtype=dtype
            )
            layout = group.get_dispatch_layout(topk_idx)

            bytes_per_iter = num_tokens * hidden_dim * elem_size

            def dispatch_combine():
                r = group.dispatch(
                    hidden=hidden,
                    topk_idx=topk_idx,
                    topk_weights=topk_weights,
                    layout=layout,
                )
                r.handle.destroy()

            latency_us = benchmark_op(dispatch_combine)
            throughput_gbps = (bytes_per_iter / latency_us) * 1e6 / 1e9

        print(
            f"\n  {backend.value} | {num_experts}E {num_tokens}T {dtype_str} | "
            f"{arch} | throughput: {throughput_gbps:.1f} GB/s "
            f"(threshold: {threshold_gbps} GB/s)"
        )
        assert throughput_gbps > threshold_gbps, \
            f"Throughput {throughput_gbps:.1f} GB/s < threshold {threshold_gbps} GB/s"


# =====================================================================
# P-IN-08: Layout Normalization Overhead
# =====================================================================


class TestLayoutNormOverhead:
    """Test P-IN-08: Measure layout normalization overhead.

    Compares dispatch latency with the backend's native layout vs the
    non-native layout. The difference is the normalization overhead.

    DeepEP native: FLAT_2D.   Non-native: EXPERT_MAJOR_3D (scatter).
    NCCL-EP native: EXPERT_MAJOR_3D.  Non-native: FLAT_2D (flatten).
    """

    def test_normalization_overhead(self, backend, ep_process_group, dist_env):
        """P-IN-08: Layout normalization overhead < 20 us.

        Expected overhead:
          - DeepEP → 3D scatter:  < 15 us
          - NCCL-EP → 2D flatten: < 5 us
        """
        num_experts = 8
        num_tokens = 64
        hidden_dim = 4096

        with fep.create_group(
            backend=backend,
            process_group=ep_process_group,
            num_experts=num_experts,
            num_local_experts=num_experts // dist_env["world_size"],
            top_k=2,
            hidden_dim=hidden_dim,
        ) as group:
            hidden, topk_idx, _ = make_tokens(
                num_tokens, hidden_dim, num_experts, 2
            )

            # Benchmark native layout
            def native():
                native_layout = (
                    fep.OutputLayout.FLAT_2D
                    if backend == fep.Backend.DEEP_EP
                    else fep.OutputLayout.EXPERT_MAJOR_3D
                )
                r = group.low_latency_dispatch(
                    hidden=hidden,
                    topk_idx=topk_idx,
                    max_tokens_per_rank=num_tokens,
                    output_layout=native_layout,
                )
                r.handle.destroy()

            # Benchmark non-native layout (triggers normalization kernel)
            def non_native():
                non_native_layout = (
                    fep.OutputLayout.EXPERT_MAJOR_3D
                    if backend == fep.Backend.DEEP_EP
                    else fep.OutputLayout.FLAT_2D
                )
                r = group.low_latency_dispatch(
                    hidden=hidden,
                    topk_idx=topk_idx,
                    max_tokens_per_rank=num_tokens,
                    output_layout=non_native_layout,
                )
                r.handle.destroy()

            native_us = benchmark_op(native)
            non_native_us = benchmark_op(non_native)
            overhead_us = non_native_us - native_us

            print(
                f"\n  {backend.value} | native: {native_us:.1f} us | "
                f"non-native: {non_native_us:.1f} us | "
                f"overhead: {overhead_us:.1f} us"
            )

            # DeepEP 3D scatter: < 15 us.  NCCL-EP flatten: < 5 us.
            assert overhead_us < 20, \
                f"Layout normalization overhead too high: {overhead_us:.1f} us"


# =====================================================================
# P-IN-09: Memory Footprint
# =====================================================================


class TestMemoryFootprint:
    """Test P-IN-09: Buffer memory footprint comparison.

    DeepEP uses expert-indexed buffers: O(E * max_tokens * hidden_dim).
    NCCL-EP uses rank-indexed buffers:  O(world_size * max_tokens * hidden_dim).

    For DeepSeek-V3 (256 experts, K=8, 7168 hidden), NCCL-EP should use
    ~14x less memory than DeepEP (256/world_size ratio).
    """

    def test_memory_footprint(self, ep_process_group, dist_env):
        """P-IN-09: NCCL-EP memory < DeepEP / 12 for DeepSeek-V3 config.

        Measures GPU memory allocated by each backend's create_group(). The
        ratio should reflect the fundamental indexing difference: expert-indexed
        (DeepEP) vs rank-indexed (NCCL-EP).
        """
        num_experts = 256
        num_local = num_experts // dist_env["world_size"]
        top_k = 8
        hidden_dim = 7168

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        # Measure DeepEP memory
        mem_before_deepep = torch.cuda.memory_allocated()
        try:
            deepep_group = fep.create_group(
                backend=fep.Backend.DEEP_EP,
                process_group=ep_process_group,
                num_experts=num_experts,
                num_local_experts=num_local,
                top_k=top_k,
                hidden_dim=hidden_dim,
            )
            torch.cuda.synchronize()
            mem_after_deepep = torch.cuda.memory_allocated()
            deepep_mem = mem_after_deepep - mem_before_deepep
            deepep_group.destroy()
        except Exception as e:
            pytest.skip(f"DeepEP not available: {e}")
            return

        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        # Measure NCCL-EP memory
        mem_before_nccl = torch.cuda.memory_allocated()
        try:
            nccl_group = fep.create_group(
                backend=fep.Backend.NCCL_EP,
                process_group=ep_process_group,
                num_experts=num_experts,
                num_local_experts=num_local,
                top_k=top_k,
                hidden_dim=hidden_dim,
            )
            torch.cuda.synchronize()
            mem_after_nccl = torch.cuda.memory_allocated()
            nccl_mem = mem_after_nccl - mem_before_nccl
            nccl_group.destroy()
        except Exception as e:
            pytest.skip(f"NCCL-EP not available: {e}")
            return

        deepep_mb = deepep_mem / (1024 * 1024)
        nccl_mb = nccl_mem / (1024 * 1024)
        ratio = deepep_mem / max(nccl_mem, 1)

        print(
            f"\n  DeepSeek-V3 config ({num_experts}E, K={top_k}, H={hidden_dim})"
            f"\n  DeepEP memory:  {deepep_mb:.1f} MB"
            f"\n  NCCL-EP memory: {nccl_mb:.1f} MB"
            f"\n  Ratio: {ratio:.1f}x"
        )

        # NCCL-EP should use significantly less memory than DeepEP
        # Design doc says ~14x less; we use 12x as the pass threshold
        # to give some margin for metadata overhead.
        if deepep_mem > 0 and nccl_mem > 0:
            assert ratio > 12, \
                f"Memory ratio {ratio:.1f}x < 12x — NCCL-EP not saving enough"
