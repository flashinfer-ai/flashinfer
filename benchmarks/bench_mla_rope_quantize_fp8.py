import flashinfer
import numpy as np
import torch
import triton
from flashinfer.testing.utils import bench_gpu_time


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_tokens"],
        x_vals=[64, 128, 256, 384, 512, 768],
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
    quant_dtype = torch.float8_e4m3fn

    num_qo_heads = 128
    q_in = torch.randn(num_tokens, num_qo_heads, 576, dtype=input_dtype, device=device)
    k_in = torch.randn(num_tokens, 576, dtype=input_dtype, device=device)
    pos_ids = torch.arange(num_tokens, device=device)

    q_out = torch.empty_like(q_in, dtype=quant_dtype)
    k_out = torch.empty_like(k_in, dtype=quant_dtype)

    def execute():
        flashinfer.rope.mla_rope_quantize_fp8(
            q_in[..., :64],
            k_in[..., :64],
            q_in[..., 64:],
            k_in[..., 64:],
            rope_flashinfer.cos_sin_cache,
            pos_ids,
            is_neox=False,
            q_rope_out=q_out[..., :64],
            k_rope_out=k_out[..., :64],
            q_nope_out=q_out[..., 64:],
            k_nope_out=k_out[..., 64:],
            quant_scale_q=1.0,
            quant_scale_kv=1.0,
        )

    measurements = bench_gpu_time(lambda: rope_forward(pos_ids, query, key))
    # Calculate statistics to match original return values
    ms = np.median(measurements)
    min_ms = np.percentile(measurements, 20)
    max_ms = np.percentile(measurements, 80)

    return ms, min_ms, max_ms


if __name__ == "__main__":
    benchmark.run(print_data=True, show_plots=True, save_path="rope_benchmark.png")
