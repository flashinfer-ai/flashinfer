import os
import sys
from typing import Union

from torch import nn
import flashinfer
import numpy as np
import torch
import triton
from flashinfer.testing.utils import bench_gpu_time, bench_gpu_time_with_cudagraph

# Add the project root to Python path to import test helpers
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from tests.test_helpers.rope_reference import RotaryEmbedding

mode_ncu = bool(int(os.environ.get("FLASHINFER_MODE_NCU", "0")))


class FlashInferRotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype

        cache = self._compute_cos_sin_cache()
        self.cos_sin_cache: torch.Tensor
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def _compute_inv_freq(self, base: Union[int, float]) -> torch.Tensor:
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim
            )
        )
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        """Compute the cos and sin cache."""
        inv_freq = self._compute_inv_freq(self.base)
        t = torch.arange(self.max_position_embeddings, dtype=torch.float)

        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        return cache

    def _apply_rotary_emb(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        is_neox_style: bool,
    ) -> torch.Tensor:
        """
        Args:
            x: [num_tokens, num_heads, head_size]
            cos: [num_tokens, head_size // 2]
            sin: [num_tokens, head_size // 2]
            is_neox_style: Whether to use the Neox-style or GPT-J-style rotary
                positional embeddings.
        """
        cos = cos.unsqueeze(-2).to(x.dtype)
        sin = sin.unsqueeze(-2).to(x.dtype)
        if is_neox_style:
            x1, x2 = torch.chunk(x, 2, dim=-1)
        else:
            x1 = x[..., ::2]
            x2 = x[..., 1::2]
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        if is_neox_style:
            return torch.cat((o1, o2), dim=-1)
        else:
            return torch.stack((o1, o2), dim=-1).flatten(-2)


def benchmark_config(config_name, num_tokens, provider):
    """Benchmark a specific attention configuration."""
    input_dtype = torch.bfloat16
    device = "cuda"
    quant_dtype = torch.float8_e4m3fn

    # Configuration-specific parameters
    if config_name == "mla":
        # MLA: Original configuration for regression testing
        num_qo_heads, num_kv_heads = 128, 1
        rope_dim, no_rope_dim = 64, 512
    elif config_name == "gqa":
        # GQA: Realistic grouped-query attention
        num_qo_heads, num_kv_heads = 32, 8
        rope_dim, no_rope_dim = 64, 64
    elif config_name == "mha":
        # MHA: Standard multi-head attention
        num_qo_heads, num_kv_heads = 32, 32
        rope_dim, no_rope_dim = 64, 64
    else:
        raise ValueError(f"Unknown config: {config_name}")

    total_dim = rope_dim + no_rope_dim

    # Create input tensors for both implementations
    if config_name == "mla":
        # MLA: 2D K tensors (shared)
        q_in = torch.randn(
            num_tokens, num_qo_heads, total_dim, dtype=input_dtype, device=device
        )
        k_in = torch.randn(num_tokens, total_dim, dtype=input_dtype, device=device)
    else:
        # GQA/MHA: 3D K tensors (multiple heads)
        q_in = torch.randn(
            num_tokens, num_qo_heads, total_dim, dtype=input_dtype, device=device
        )
        k_in = torch.randn(
            num_tokens, num_kv_heads, total_dim, dtype=input_dtype, device=device
        )

    pos_ids = torch.arange(num_tokens, device=device)

    # Create reference implementation
    rope_ref = RotaryEmbedding(
        head_size=total_dim,
        rotary_dim=rope_dim,
        max_position_embeddings=4096,
        base=10000,
        is_neox_style=False,
        dtype=input_dtype,
        device=device,
    )

    run_idx = 0

    if provider == "flashinfer":
        # Split tensors for FlashInfer
        q_rope = q_in[..., :rope_dim]
        q_nope = q_in[..., rope_dim:]
        k_rope = k_in[..., :rope_dim]
        k_nope = k_in[..., rope_dim:]

        # Create output tensors
        q_rope_out = torch.empty_like(q_rope, dtype=quant_dtype)
        q_nope_out = torch.empty_like(q_nope, dtype=quant_dtype)
        k_rope_out = torch.empty_like(k_rope, dtype=quant_dtype)
        k_nope_out = torch.empty_like(k_nope, dtype=quant_dtype)

        def execute():
            nonlocal run_idx
            run_idx += 1

            if mode_ncu and run_idx == 20:
                torch.cuda.cudart().cudaProfilerStart()

            flashinfer.rope.rope_quantize_fp8(
                q_rope=q_rope,
                k_rope=k_rope,
                q_nope=q_nope,
                k_nope=k_nope,
                cos_sin_cache=rope_ref.cos_sin_cache,
                pos_ids=pos_ids,
                is_neox=False,
                q_rope_out=q_rope_out,
                k_rope_out=k_rope_out,
                q_nope_out=q_nope_out,
                k_nope_out=k_nope_out,
                quant_scale_q=1.0,
                quant_scale_kv=1.0,
            )

            if mode_ncu and run_idx == 20:
                torch.cuda.cudart().cudaProfilerStop()

    elif provider == "torch":
        # Create compiled version for better performance
        @torch.compile
        def torch_rope_quantize(q_in, k_in, pos_ids):
            # Apply RoPE using reference implementation
            q_out_f16, k_out_f16 = rope_ref.forward_native(pos_ids, q_in, k_in)

            # Quantize to FP8 (PyTorch native)
            q_out_f8 = q_out_f16.to(quant_dtype)
            k_out_f8 = k_out_f16.to(quant_dtype)
            return q_out_f8, k_out_f8

        # Warmup the compiled function
        _ = torch_rope_quantize(q_in, k_in, pos_ids)
        torch.cuda.synchronize()

        def execute():
            nonlocal run_idx
            run_idx += 1

            if mode_ncu and run_idx == 20:
                torch.cuda.cudart().cudaProfilerStart()

            _ = torch_rope_quantize(q_in, k_in, pos_ids)

            if mode_ncu and run_idx == 20:
                torch.cuda.cudart().cudaProfilerStop()

    else:
        raise ValueError(f"Unknown provider: {provider}")

    if mode_ncu:
        measurements = bench_gpu_time(execute)
    else:
        measurements = bench_gpu_time_with_cudagraph(execute)

    # Calculate statistics
    ms = np.median(measurements)
    min_ms = np.percentile(measurements, 20)
    max_ms = np.percentile(measurements, 80)

    return ms, min_ms, max_ms


# Create separate benchmark functions for each architecture
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_tokens"],
        x_vals=[768] if mode_ncu else [1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 768],
        line_arg="provider",
        line_vals=["flashinfer", "torch"],
        line_names=["FlashInfer", "PyTorch Compiled"],
        styles=[("blue", "-"), ("blue", "--")],
        ylabel="Latency (ms)",
        plot_name="mla-rope-benchmark",
        args={},
    )
)
def benchmark_mla(provider, num_tokens):
    return benchmark_config("mla", num_tokens, provider)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_tokens"],
        x_vals=[768] if mode_ncu else [1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 768],
        line_arg="provider",
        line_vals=["flashinfer", "torch"],
        line_names=["FlashInfer", "PyTorch Compiled"],
        styles=[("red", "-"), ("red", "--")],
        ylabel="Latency (ms)",
        plot_name="gqa-rope-benchmark",
        args={},
    )
)
def benchmark_gqa(provider, num_tokens):
    return benchmark_config("gqa", num_tokens, provider)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_tokens"],
        x_vals=[768] if mode_ncu else [1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 768],
        line_arg="provider",
        line_vals=["flashinfer", "torch"],
        line_names=["FlashInfer", "PyTorch Compiled"],
        styles=[("green", "-"), ("green", "--")],
        ylabel="Latency (ms)",
        plot_name="mha-rope-benchmark",
        args={},
    )
)
def benchmark_mha(provider, num_tokens):
    return benchmark_config("mha", num_tokens, provider)


if __name__ == "__main__":
    # Run all benchmarks and generate individual plots
    print("Running MLA benchmark...")
    benchmark_mla.run(print_data=False, show_plots=True, save_path=".")

    print("Running GQA benchmark...")
    benchmark_gqa.run(print_data=False, show_plots=True, save_path=".")

    print("Running MHA benchmark...")
    benchmark_mha.run(print_data=False, show_plots=True, save_path=".")

    # Collect results for summary table
    token_counts = (
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 768] if not mode_ncu else [768]
    )

    print("\n=== Summary Table ===")
    print(
        f"{'Tokens':<8} {'MLA-FI (ms)':<12} {'MLA-Torch (ms)':<14} {'GQA-FI (ms)':<12} {'GQA-Torch (ms)':<14} {'MHA-FI (ms)':<12} {'MHA-Torch (ms)':<14}"
    )
    print("-" * 90)
    for num_tokens in token_counts:
        mla_fi_ms, _, _ = benchmark_config("mla", num_tokens, "flashinfer")
        mla_torch_ms, _, _ = benchmark_config("mla", num_tokens, "torch")
        gqa_fi_ms, _, _ = benchmark_config("gqa", num_tokens, "flashinfer")
        gqa_torch_ms, _, _ = benchmark_config("gqa", num_tokens, "torch")
        mha_fi_ms, _, _ = benchmark_config("mha", num_tokens, "flashinfer")
        mha_torch_ms, _, _ = benchmark_config("mha", num_tokens, "torch")
        print(
            f"{num_tokens:<8} {mla_fi_ms:<12.5f} {mla_torch_ms:<14.5f} {gqa_fi_ms:<12.5f} {gqa_torch_ms:<14.5f} {mha_fi_ms:<12.5f} {mha_torch_ms:<14.5f}"
        )

    print("\nConfiguration details:")
    print("  MLA: 128 Q heads, 1 K head, 64+512 dims")
    print("  GQA: 32 Q heads, 8 K heads, 64+64 dims")
    print("  MHA: 32 Q heads, 32 K heads, 64+64 dims")

    print("\nPlot files saved to current directory:")
    print("  mla-rope-benchmark.png (FlashInfer vs PyTorch)")
    print("  gqa-rope-benchmark.png (FlashInfer vs PyTorch)")
    print("  mha-rope-benchmark.png (FlashInfer vs PyTorch)")
