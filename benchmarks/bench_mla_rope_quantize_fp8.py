import os
from typing import Union

from torch import nn
import flashinfer
import numpy as np
import torch
import triton
from flashinfer.testing.utils import bench_gpu_time, bench_gpu_time_with_cudagraph

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


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_tokens"],
        x_vals=[768] if mode_ncu else [1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 768],
        line_arg="provider",
        line_vals=["flashinfer"],
        line_names=["FlashInfer"],
        styles=[("blue", "-")],
        ylabel="Latency (ms)",
        plot_name="rope-latency",
        args={},
    )
)
def benchmark(
    provider,
    num_tokens,
):
    input_dtype = torch.bfloat16
    device = "cuda"
    quant_dtype = torch.float8_e4m3fn

    num_qo_heads = 128
    q_rope = torch.randn(
        num_tokens, num_qo_heads, 192, dtype=input_dtype, device=device
    )[:, :, :64]
    # TODO not 1:1 mimic yet
    k_rope = torch.randn(num_tokens, 64, dtype=input_dtype, device=device)
    q_nope = torch.randn(
        num_qo_heads, num_tokens, 512, dtype=input_dtype, device=device
    ).permute(1, 0, 2)
    k_nope = torch.randn(num_tokens, 512, dtype=input_dtype, device=device)
    pos_ids = torch.arange(num_tokens, device=device)

    q_out = torch.empty(num_tokens, num_qo_heads, 576, dtype=quant_dtype, device=device)
    k_rope_out = torch.empty(num_tokens, 64, dtype=quant_dtype, device=device)
    k_nope_out = torch.empty(num_tokens, 512, dtype=quant_dtype, device=device)

    rope_flashinfer = FlashInferRotaryEmbedding(
        head_size=576,
        rotary_dim=64,
        max_position_embeddings=4096,
        base=10000,
        is_neox_style=False,
        dtype=input_dtype,
    ).to(device)

    run_idx = 0

    def execute():
        nonlocal run_idx
        run_idx += 1

        if mode_ncu and run_idx == 20:
            torch.cuda.cudart().cudaProfilerStart()

        flashinfer.rope.mla_rope_quantize_fp8(
            # (bs, 128, 64), bf16, stride=(128 * 192, 192, 1)
            q_rope=q_rope,
            # (bs, 64), bf16, stride=(2112, 1)
            k_rope=k_rope,
            # shape=(bs, 128, 512), bf16, stride=(512, 512 * bs, 1)
            q_nope=q_nope,
            # (bs, 512), bf16, stride=(512, 1)
            k_nope=k_nope,
            cos_sin_cache=rope_flashinfer.cos_sin_cache,
            pos_ids=pos_ids,
            is_neox=False,
            # q_out: (bs, 128, 576), e4m3fn, stride=(128 * 576, 576, 1)
            # q_rope_out=q_out[..., self.kv_lora_rank:]
            # q_nope_out=q_out[..., :self.kv_lora_rank]
            q_rope_out=q_out[..., 512:],
            # (bs, 64), e4m3fn, stride=(64, 1)
            k_rope_out=k_rope_out,
            # see above
            q_nope_out=q_out[..., :512],
            # (bs, 512), stride=(512, 1)
            k_nope_out=k_nope_out,
            quant_scale_q=1.0,
            quant_scale_kv=1.0,
        )

        if mode_ncu and run_idx == 20:
            torch.cuda.cudart().cudaProfilerStop()

    if mode_ncu:
        measurements = bench_gpu_time(execute)
    else:
        measurements = bench_gpu_time_with_cudagraph(execute)
    # Calculate statistics to match original return values
    ms = np.median(measurements)
    min_ms = np.percentile(measurements, 20)
    max_ms = np.percentile(measurements, 80)

    return ms, min_ms, max_ms


if __name__ == "__main__":
    benchmark.run(print_data=True, show_plots=True, save_path="rope_benchmark.png")
