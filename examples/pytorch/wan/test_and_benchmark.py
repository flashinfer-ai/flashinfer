#!/usr/bin/env python3
"""
Unified test and benchmark script for FlashInfer-optimized WanTransformer3DModel.

Features:
- Unit tests: basic forward, memory, numerical correctness, GEMM backends
- Performance benchmarks: FlashInfer vs diffusers, different GEMM backends
- Skip-softmax sparse attention benchmark (SM100/SM103 only)

Usage:
    # Run all tests
    python test_and_benchmark.py test

    # Run specific tests
    python test_and_benchmark.py test --basic --gemm

    # Benchmark GEMM backends only
    python test_and_benchmark.py benchmark --gemm-backend auto torch

    # Benchmark GEMM backends + sparse attention (combined)
    python test_and_benchmark.py benchmark --gemm-backend auto --sparse --threshold 1.0

    # Quick sanity check
    python test_and_benchmark.py quick
"""

import argparse
import gc
import time
import warnings
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from flashinfer.utils import get_compute_capability


# =============================================================================
# Utility Functions
# =============================================================================


def cleanup_gpu_memory():
    """Clean up GPU memory between tests."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def get_gpu_info() -> dict:
    """Get GPU information."""
    device = torch.device("cuda")
    major, minor = get_compute_capability(device)
    sm_version = major * 10 + minor
    return {
        "name": torch.cuda.get_device_name(0),
        "sm_version": sm_version,
        "skip_softmax_supported": sm_version in (100, 103),
        "memory_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3,
    }


def measure_latency(
    model: torch.nn.Module,
    hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    warmup_iters: int = 5,
    benchmark_iters: int = 20,
) -> Tuple[float, float]:
    """Measure forward pass latency."""
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(hidden_states, timestep, encoder_hidden_states, return_dict=False)
            torch.cuda.synchronize()

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(benchmark_iters):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(hidden_states, timestep, encoder_hidden_states, return_dict=False)
            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)

    mean_time = sum(times) / len(times)
    std_time = (sum((t - mean_time) ** 2 for t in times) / len(times)) ** 0.5
    return mean_time, std_time


def compute_accuracy_metrics(output_a: torch.Tensor, output_b: torch.Tensor) -> dict:
    """Compute accuracy metrics between two outputs."""
    diff = (output_a - output_b).abs()
    flat_a = output_a.flatten().float()
    flat_b = output_b.flatten().float()
    cosine_sim = F.cosine_similarity(flat_a.unsqueeze(0), flat_b.unsqueeze(0)).item()

    return {
        "max_abs_error": diff.max().item(),
        "mean_abs_error": diff.mean().item(),
        "cosine_similarity": cosine_sim,
    }


# =============================================================================
# Test Functions
# =============================================================================


def test_basic_forward(config_override: Optional[dict] = None) -> bool:
    """Test basic forward pass."""
    from transformer_wan_flashinfer import (
        FlashInferWanTransformer3DModel,
        WanTransformer3DConfig,
    )

    print("=" * 60)
    print("Test: Basic Forward Pass")
    print("=" * 60)

    config_dict = {
        "patch_size": (1, 2, 2),
        "num_attention_heads": 8,
        "attention_head_dim": 64,
        "in_channels": 4,
        "out_channels": 4,
        "text_dim": 512,
        "freq_dim": 256,
        "ffn_dim": 1024,
        "num_layers": 2,
        "cross_attn_norm": True,
        "eps": 1e-6,
        "rope_max_seq_len": 256,
        "gemm_backend": "torch",
    }
    if config_override:
        config_dict.update(config_override)

    config = WanTransformer3DConfig(**config_dict)
    model = FlashInferWanTransformer3DModel(config).cuda().half()

    batch_size, num_frames, height, width = 1, 8, 32, 32
    hidden_states = torch.randn(
        batch_size, 4, num_frames, height, width, device="cuda", dtype=torch.float16
    )
    timestep = torch.randint(0, 1000, (batch_size,), device="cuda")
    encoder_hidden_states = torch.randn(
        batch_size, 64, 512, device="cuda", dtype=torch.float16
    )

    with torch.no_grad():
        output = model(
            hidden_states, timestep, encoder_hidden_states, return_dict=False
        )

    expected_shape = hidden_states.shape
    actual_shape = output[0].shape

    print(f"Input shape:  {hidden_states.shape}")
    print(f"Output shape: {actual_shape}")

    passed = actual_shape == expected_shape
    print(f"Result: {'PASSED' if passed else 'FAILED'}")

    del model, hidden_states, timestep, encoder_hidden_states, output
    cleanup_gpu_memory()
    return passed


def test_memory_efficiency() -> bool:
    """Test memory usage."""
    from transformer_wan_flashinfer import (
        FlashInferWanTransformer3DModel,
        WanTransformer3DConfig,
    )

    print("=" * 60)
    print("Test: Memory Efficiency")
    print("=" * 60)

    torch.cuda.reset_peak_memory_stats()

    config = WanTransformer3DConfig(
        patch_size=(1, 2, 2),
        num_attention_heads=16,
        attention_head_dim=64,
        in_channels=4,
        out_channels=4,
        text_dim=512,
        freq_dim=256,
        ffn_dim=2048,
        num_layers=4,
        gemm_backend="torch",
    )

    model = FlashInferWanTransformer3DModel(config).cuda().half()
    model_memory = torch.cuda.max_memory_allocated() / 1024**2
    print(f"Model memory: {model_memory:.2f} MB")

    torch.cuda.reset_peak_memory_stats()

    batch_size, num_frames, height, width = 1, 16, 32, 32
    hidden_states = torch.randn(
        batch_size, 4, num_frames, height, width, device="cuda", dtype=torch.float16
    )
    timestep = torch.randint(0, 1000, (batch_size,), device="cuda")
    encoder_hidden_states = torch.randn(
        batch_size, 64, 512, device="cuda", dtype=torch.float16
    )

    with torch.no_grad():
        output = model(
            hidden_states, timestep, encoder_hidden_states, return_dict=False
        )

    peak_memory = torch.cuda.max_memory_allocated() / 1024**2
    print(f"Peak memory during forward: {peak_memory:.2f} MB")
    print("Result: PASSED")

    del model, hidden_states, timestep, encoder_hidden_states, output
    cleanup_gpu_memory()
    return True


def test_numerical_correctness() -> Optional[bool]:
    """Compare FlashInfer output with original diffusers implementation."""
    print("=" * 60)
    print("Test: Numerical Correctness (vs diffusers)")
    print("=" * 60)

    try:
        from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
    except ImportError:
        print("SKIPPED: diffusers not installed")
        return None

    from transformer_wan_flashinfer import FlashInferWanTransformer3DModel

    config_dict = {
        "patch_size": (1, 2, 2),
        "num_attention_heads": 8,
        "attention_head_dim": 64,
        "in_channels": 4,
        "out_channels": 4,
        "text_dim": 512,
        "freq_dim": 256,
        "ffn_dim": 1024,
        "num_layers": 1,
        "cross_attn_norm": True,
        "eps": 1e-6,
    }

    original_model = WanTransformer3DModel(**config_dict).cuda().float()
    flashinfer_model = (
        FlashInferWanTransformer3DModel(original_model.config).cuda().float()
    )

    try:
        flashinfer_model.load_state_dict(original_model.state_dict(), strict=False)
        print("Weights loaded from original model")
    except Exception as e:
        print(f"Warning: Could not load weights: {e}")

    torch.manual_seed(42)
    batch_size, num_frames, height, width = 1, 8, 32, 32
    hidden_states = torch.randn(
        batch_size, 4, num_frames, height, width, device="cuda", dtype=torch.float32
    )
    timestep = torch.tensor([500], device="cuda")
    encoder_hidden_states = torch.randn(
        batch_size, 64, 512, device="cuda", dtype=torch.float32
    )

    with torch.no_grad():
        original_output = original_model(
            hidden_states, timestep, encoder_hidden_states, return_dict=False
        )[0]
        flashinfer_output = flashinfer_model(
            hidden_states, timestep, encoder_hidden_states, return_dict=False
        )[0]

    metrics = compute_accuracy_metrics(original_output, flashinfer_output)
    print(f"Max absolute error:  {metrics['max_abs_error']:.6e}")
    print(f"Mean absolute error: {metrics['mean_abs_error']:.6e}")
    print(f"Cosine similarity:   {metrics['cosine_similarity']:.6f}")

    passed = metrics["max_abs_error"] < 0.1
    print(f"Result: {'PASSED' if passed else 'WARNING (may be due to weight loading)'}")

    del original_model, flashinfer_model, hidden_states, timestep, encoder_hidden_states
    cleanup_gpu_memory()
    return passed


def test_gemm_backends() -> bool:
    """Test different GEMM backends."""
    from transformer_wan_flashinfer import (
        FlashInferWanTransformer3DModel,
        WanTransformer3DConfig,
        FlashInferLinear,
        _get_best_available_backend,
        _check_gemm_backend_support,
        GEMMBackend,
    )

    print("=" * 60)
    print("Test: GEMM Backends")
    print("=" * 60)

    device = torch.device("cuda")

    print("Available backends:")
    for backend in GEMMBackend:
        supported = _check_gemm_backend_support(backend, device)
        print(f"  {backend.value}: {'supported' if supported else 'not supported'}")

    best_backend = _get_best_available_backend(device)
    print(f"Best available: {best_backend.value}\n")

    # Determine backends to test
    backends_to_test = ["torch", "auto"]
    for backend in GEMMBackend:
        if backend not in (GEMMBackend.TORCH,) and _check_gemm_backend_support(
            backend, device
        ):
            backends_to_test.append(backend.value)

    base_config = {
        "patch_size": (1, 2, 2),
        "num_attention_heads": 8,
        "attention_head_dim": 64,
        "in_channels": 4,
        "out_channels": 4,
        "text_dim": 512,
        "freq_dim": 256,
        "ffn_dim": 1024,
        "num_layers": 1,
    }

    batch_size, num_frames, height, width = 1, 8, 32, 32
    hidden_states = torch.randn(
        batch_size, 4, num_frames, height, width, device="cuda", dtype=torch.float16
    )
    timestep = torch.tensor([500], device="cuda")
    encoder_hidden_states = torch.randn(
        batch_size, 64, 512, device="cuda", dtype=torch.float16
    )

    all_passed = True
    for backend in backends_to_test:
        try:
            config = WanTransformer3DConfig(**base_config, gemm_backend=backend)
            model = FlashInferWanTransformer3DModel(config).cuda().half()
            fi_count = sum(
                1 for m in model.modules() if isinstance(m, FlashInferLinear)
            )

            with torch.no_grad():
                output = model(
                    hidden_states, timestep, encoder_hidden_states, return_dict=False
                )

            if output[0].shape != hidden_states.shape:
                print(f"  {backend}: FAILED (shape mismatch)")
                all_passed = False
            else:
                print(f"  {backend}: PASSED (FlashInferLinear: {fi_count})")
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  {backend}: FAILED ({e})")
            all_passed = False

    print(f"\nResult: {'PASSED' if all_passed else 'SOME FAILED'}")

    del hidden_states, timestep, encoder_hidden_states
    cleanup_gpu_memory()
    return all_passed


def test_skip_softmax_sparse() -> bool:
    """Test skip-softmax sparse attention."""
    from transformer_wan_flashinfer import (
        FlashInferWanTransformer3DModel,
        WanTransformer3DConfig,
    )

    print("=" * 60)
    print("Test: Skip-Softmax Sparse Attention")
    print("=" * 60)

    gpu_info = get_gpu_info()
    print(f"GPU: {gpu_info['name']} (SM{gpu_info['sm_version']})")
    print(f"Skip-softmax supported: {gpu_info['skip_softmax_supported']}")

    config = WanTransformer3DConfig(
        patch_size=(1, 2, 2),
        num_attention_heads=8,
        attention_head_dim=64,
        in_channels=4,
        out_channels=4,
        text_dim=512,
        freq_dim=256,
        ffn_dim=1024,
        num_layers=1,
        gemm_backend="torch",
        use_skip_softmax_sparse=True,
        skip_softmax_threshold_scale_factor=1.0,
    )

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model = FlashInferWanTransformer3DModel(config).cuda().half()

        batch_size, num_frames, height, width = 1, 8, 32, 32
        hidden_states = torch.randn(
            batch_size, 4, num_frames, height, width, device="cuda", dtype=torch.float16
        )
        timestep = torch.randint(0, 1000, (batch_size,), device="cuda")
        encoder_hidden_states = torch.randn(
            batch_size, 64, 512, device="cuda", dtype=torch.float16
        )

        with torch.no_grad():
            output = model(
                hidden_states, timestep, encoder_hidden_states, return_dict=False
            )

        # Check for fallback warning
        fallback_warned = any(
            "falling back" in str(warning.message).lower() for warning in w
        )

    passed = output[0].shape == hidden_states.shape
    if fallback_warned and not gpu_info["skip_softmax_supported"]:
        print("Note: Fell back to standard attention (expected on this GPU)")

    print(f"Output shape: {output[0].shape}")
    print(f"Result: {'PASSED' if passed else 'FAILED'}")

    del model, hidden_states, timestep, encoder_hidden_states, output
    cleanup_gpu_memory()
    return passed


# =============================================================================
# Benchmark Functions
# =============================================================================


def run_benchmark(
    gemm_backends: List[str],
    num_layers: int = 4,
    batch_size: int = 1,
    num_frames: int = 16,
    height: int = 64,
    width: int = 64,
    text_seq_len: int = 128,
    warmup_iters: int = 5,
    benchmark_iters: int = 20,
    compare_diffusers: bool = True,
    use_skip_softmax_sparse: bool = False,
    skip_softmax_threshold: float = 1.0,
    # Wan 2.2 model architecture params
    num_attention_heads: int = 16,
    attention_head_dim: int = 64,
    ffn_dim: int = 2048,
    text_dim: int = 512,
):
    """Run performance benchmark with optional sparse attention."""
    from transformer_wan_flashinfer import (
        FlashInferWanTransformer3DModel,
        WanTransformer3DConfig,
        FlashInferLinear,
        _get_best_available_backend,
    )

    device = torch.device("cuda")
    dtype = torch.bfloat16

    print("=" * 70)
    print("Performance Benchmark")
    print("=" * 70)

    gpu_info = get_gpu_info()
    print(f"GPU: {gpu_info['name']} (SM{gpu_info['sm_version']})")
    print(f"Best GEMM backend: {_get_best_available_backend(device).value}")
    print(f"Skip-softmax sparse: {use_skip_softmax_sparse}", end="")
    if use_skip_softmax_sparse:
        print(f" (threshold={skip_softmax_threshold})", end="")
        if not gpu_info["skip_softmax_supported"]:
            print(
                f" [NOT SUPPORTED on SM{gpu_info['sm_version']}, will fallback]", end=""
            )
    print()
    print()

    print("Configuration:")
    print(f"  Layers: {num_layers}, Batch: {batch_size}")
    print(f"  Video: {num_frames}x{height}x{width}, Text: {text_seq_len}")
    print(f"  GEMM backends: {gemm_backends}")
    print()

    config_dict = {
        "patch_size": (1, 2, 2),
        "num_attention_heads": num_attention_heads,
        "attention_head_dim": attention_head_dim,
        "in_channels": 16,
        "out_channels": 16,
        "text_dim": text_dim,
        "freq_dim": 256,
        "ffn_dim": ffn_dim,
        "num_layers": num_layers,
        "cross_attn_norm": True,
        "eps": 1e-6,
        "rope_max_seq_len": 1024,
    }

    in_channels = config_dict["in_channels"]
    hidden_states = torch.randn(
        batch_size, in_channels, num_frames, height, width, device=device, dtype=dtype
    )
    timestep = torch.randint(0, 1000, (batch_size,), device=device)
    encoder_hidden_states = torch.randn(
        batch_size, text_seq_len, text_dim, device=device, dtype=dtype
    )

    results: Dict[str, Optional[Tuple[float, float]]] = {}
    outputs: Dict[str, torch.Tensor] = {}

    # Benchmark original diffusers model
    if compare_diffusers:
        print("-" * 70)
        print("Benchmarking: diffusers (baseline)")
        try:
            from diffusers.models.transformers.transformer_wan import (
                WanTransformer3DModel,
            )

            original_model = WanTransformer3DModel(**config_dict).to(device).to(dtype)
            mean_time, std_time = measure_latency(
                original_model,
                hidden_states,
                timestep,
                encoder_hidden_states,
                warmup_iters,
                benchmark_iters,
            )
            results["diffusers"] = (mean_time, std_time)
            print(f"  Latency: {mean_time:.2f} +/- {std_time:.2f} ms")

            with torch.no_grad():
                outputs["diffusers"] = original_model(
                    hidden_states, timestep, encoder_hidden_states, return_dict=False
                )[0].clone()

            del original_model
            torch.cuda.empty_cache()
        except ImportError:
            print("  SKIPPED: diffusers not installed")
            results["diffusers"] = None

    # Benchmark FlashInfer with different backends and attention modes
    for backend in gemm_backends:
        # First, create standard attention model
        std_name = f"flashinfer_{backend}_standard"
        print("-" * 70)
        print(f"Benchmarking: FlashInfer ({backend}, standard)")

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fi_config_std = WanTransformer3DConfig(
                    **config_dict,
                    gemm_backend=backend,
                    use_skip_softmax_sparse=False,
                )
                model_std = (
                    FlashInferWanTransformer3DModel(fi_config_std).to(device).to(dtype)
                )

            fi_count = sum(
                1 for m in model_std.modules() if isinstance(m, FlashInferLinear)
            )
            print(f"  FlashInferLinear layers: {fi_count}")

            mean_time, std_time = measure_latency(
                model_std,
                hidden_states,
                timestep,
                encoder_hidden_states,
                warmup_iters,
                benchmark_iters,
            )
            results[std_name] = (mean_time, std_time)
            print(f"  Latency: {mean_time:.2f} +/- {std_time:.2f} ms")

            with torch.no_grad():
                outputs[std_name] = model_std(
                    hidden_states, timestep, encoder_hidden_states, return_dict=False
                )[0].clone()

            # If sparse mode enabled, create sparse model with same weights
            if use_skip_softmax_sparse:
                sparse_name = f"flashinfer_{backend}_sparse"
                print("-" * 70)
                print(f"Benchmarking: FlashInfer ({backend}, sparse)")

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fi_config_sparse = WanTransformer3DConfig(
                        **config_dict,
                        gemm_backend=backend,
                        use_skip_softmax_sparse=True,
                        skip_softmax_threshold_scale_factor=skip_softmax_threshold,
                    )
                    model_sparse = (
                        FlashInferWanTransformer3DModel(fi_config_sparse)
                        .to(device)
                        .to(dtype)
                    )
                    # Copy weights from standard model for fair comparison
                    model_sparse.load_state_dict(model_std.state_dict())

                print(f"  FlashInferLinear layers: {fi_count}")
                print("  (weights copied from standard model)")

                mean_time, std_time = measure_latency(
                    model_sparse,
                    hidden_states,
                    timestep,
                    encoder_hidden_states,
                    warmup_iters,
                    benchmark_iters,
                )
                results[sparse_name] = (mean_time, std_time)
                print(f"  Latency: {mean_time:.2f} +/- {std_time:.2f} ms")

                with torch.no_grad():
                    outputs[sparse_name] = model_sparse(
                        hidden_states,
                        timestep,
                        encoder_hidden_states,
                        return_dict=False,
                    )[0].clone()

                del model_sparse
                torch.cuda.empty_cache()

            del model_std
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  ERROR: {e}")
            results[std_name] = None

    # Summary - Performance
    print("=" * 70)
    print("Performance Summary")
    print("=" * 70)
    print(f"{'Model':<40} {'Latency (ms)':<20} {'Speedup':<10}")
    print("-" * 70)

    baseline = results.get("diffusers")
    for name, result in results.items():
        if result:
            mean_time, std_time = result
            speedup = (
                f"{baseline[0] / mean_time:.2f}x"
                if baseline and name != "diffusers"
                else ""
            )
            print(f"{name:<40} {mean_time:.2f} +/- {std_time:.2f}       {speedup}")
        else:
            print(f"{name:<40} {'N/A':<20}")

    # Summary - Accuracy (if sparse attention enabled)
    if use_skip_softmax_sparse and len(outputs) > 1:
        print()
        print("=" * 70)
        print("Accuracy Summary (vs standard attention)")
        print("=" * 70)

        # Find a standard attention output as reference
        ref_key = None
        for key in outputs:
            if "standard" in key:
                ref_key = key
                break

        if ref_key:
            print(f"Reference: {ref_key}")
            print(f"{'Model':<40} {'Max Error':<15} {'Cosine Sim':<15}")
            print("-" * 70)

            for name, output in outputs.items():
                if name != ref_key:
                    metrics = compute_accuracy_metrics(outputs[ref_key], output)
                    print(
                        f"{name:<40} {metrics['max_abs_error']:<15.2e} {metrics['cosine_similarity']:<15.6f}"
                    )

    if use_skip_softmax_sparse and not gpu_info["skip_softmax_supported"]:
        print()
        print(
            f"NOTE: SM{gpu_info['sm_version']} does not support skip-softmax sparse attention."
        )
        print(
            "      Sparse mode falls back to standard attention. Run on SM100/SM103 for real comparison."
        )

    cleanup_gpu_memory()


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Test and benchmark FlashInfer WanTransformer3DModel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Test command
    test_parser = subparsers.add_parser("test", help="Run unit tests")
    test_parser.add_argument("--all", action="store_true", help="Run all tests")
    test_parser.add_argument("--basic", action="store_true", help="Basic forward test")
    test_parser.add_argument(
        "--memory", action="store_true", help="Memory efficiency test"
    )
    test_parser.add_argument(
        "--numerical", action="store_true", help="Numerical correctness test"
    )
    test_parser.add_argument("--gemm", action="store_true", help="GEMM backends test")
    test_parser.add_argument(
        "--sparse", action="store_true", help="Skip-softmax sparse test"
    )

    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run performance benchmark")
    bench_parser.add_argument(
        "--gemm-backend",
        type=str,
        nargs="+",
        default=["auto"],
        help="GEMM backend(s) to test",
    )
    bench_parser.add_argument(
        "--sparse",
        action="store_true",
        help="Also benchmark skip-softmax sparse attention",
    )
    bench_parser.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="Skip-softmax threshold scale factor (used with --sparse)",
    )
    bench_parser.add_argument("--num-layers", type=int, default=4)
    bench_parser.add_argument("--batch-size", type=int, default=1)
    bench_parser.add_argument("--num-frames", type=int, default=16)
    bench_parser.add_argument("--height", type=int, default=64)
    bench_parser.add_argument("--width", type=int, default=64)
    bench_parser.add_argument("--text-seq-len", type=int, default=128)
    bench_parser.add_argument("--warmup", type=int, default=5)
    bench_parser.add_argument("--iters", type=int, default=20)
    bench_parser.add_argument(
        "--no-diffusers", action="store_true", help="Skip diffusers comparison"
    )
    # Wan 2.2 model architecture options
    bench_parser.add_argument("--num-attention-heads", type=int, default=16)
    bench_parser.add_argument("--attention-head-dim", type=int, default=64)
    bench_parser.add_argument("--ffn-dim", type=int, default=2048)
    bench_parser.add_argument("--text-dim", type=int, default=512)
    # Preset for actual Wan 2.2 720P 5s config
    bench_parser.add_argument(
        "--wan22-720p",
        action="store_true",
        help="Use actual Wan 2.2 config for 720P 5s video generation",
    )

    # Quick sanity check
    subparsers.add_parser("quick", help="Quick sanity check")

    args = parser.parse_args()

    # Print GPU info
    gpu_info = get_gpu_info()
    print(
        f"GPU: {gpu_info['name']} (SM{gpu_info['sm_version']}, {gpu_info['memory_gb']:.1f} GB)"
    )
    print()

    if args.command == "test":
        run_tests = []
        if args.all or not any(
            [args.basic, args.memory, args.numerical, args.gemm, args.sparse]
        ):
            run_tests = ["basic", "memory", "numerical", "gemm", "sparse"]
        else:
            if args.basic:
                run_tests.append("basic")
            if args.memory:
                run_tests.append("memory")
            if args.numerical:
                run_tests.append("numerical")
            if args.gemm:
                run_tests.append("gemm")
            if args.sparse:
                run_tests.append("sparse")

        results = {}
        for test_name in run_tests:
            if test_name == "basic":
                results["basic"] = test_basic_forward()
            elif test_name == "memory":
                results["memory"] = test_memory_efficiency()
            elif test_name == "numerical":
                results["numerical"] = test_numerical_correctness()
            elif test_name == "gemm":
                results["gemm"] = test_gemm_backends()
            elif test_name == "sparse":
                results["sparse"] = test_skip_softmax_sparse()
            cleanup_gpu_memory()
            print()

        print("=" * 60)
        print("Test Summary")
        print("=" * 60)
        for name, result in results.items():
            status = "PASSED" if result else ("SKIPPED" if result is None else "FAILED")
            print(f"  {name}: {status}")

    elif args.command == "benchmark":
        # Apply Wan 2.2 720P 5s preset if requested
        if args.wan22_720p:
            # Wan 2.2 model architecture
            args.num_layers = 40
            args.num_attention_heads = 40
            args.attention_head_dim = 128  # 40*128=5120 inner_dim
            args.ffn_dim = 13824
            args.text_dim = 4096
            # 720P 5s video after VAE compression:
            # - Temporal: 81 frames -> 21 latent frames (4x compression)
            # - Spatial: 720x1280 -> 90x160 (8x compression)
            args.num_frames = 21
            args.height = 90
            args.width = 160
            args.text_seq_len = 512
            print("Using Wan 2.2 720P 5s preset configuration")
            print(f"  Latent shape: {args.num_frames}x{args.height}x{args.width}")
            print(
                f"  Visual tokens: {args.num_frames * (args.height // 2) * (args.width // 2)}"
            )
            print()

        run_benchmark(
            gemm_backends=args.gemm_backend,
            num_layers=args.num_layers,
            batch_size=args.batch_size,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            text_seq_len=args.text_seq_len,
            warmup_iters=args.warmup,
            benchmark_iters=args.iters,
            compare_diffusers=not args.no_diffusers,
            use_skip_softmax_sparse=args.sparse,
            skip_softmax_threshold=args.threshold,
            num_attention_heads=args.num_attention_heads,
            attention_head_dim=args.attention_head_dim,
            ffn_dim=args.ffn_dim,
            text_dim=args.text_dim,
        )

    elif args.command == "quick":
        print("Running quick sanity check...")
        passed = test_basic_forward()
        print()
        print(f"Quick check: {'PASSED' if passed else 'FAILED'}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
