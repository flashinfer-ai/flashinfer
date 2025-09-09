from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import triton
from vllm.model_executor.layers.rotary_embedding import (
    RotaryEmbedding as vLLMRotaryEmbedding,
)

from flashinfer.rope import apply_rope_with_cos_sin_cache_inplace
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
        args={
        },
    )
)
def benchmark(
    provider,
    num_tokens,
):
    measurements = bench_gpu_time(lambda: rope_forward(pos_ids, query, key))
    # Calculate statistics to match original return values
    ms = np.median(measurements)
    min_ms = np.percentile(measurements, 20)
    max_ms = np.percentile(measurements, 80)

    return ms, min_ms, max_ms


if __name__ == "__main__":
    benchmark.run(print_data=True, show_plots=True, save_path="rope_benchmark.png")
