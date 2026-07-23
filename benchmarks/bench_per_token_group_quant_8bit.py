"""
Benchmark: per-token-group 8-bit quantization — cuTile backend vs SOTA.

Compares FlashInfer's migrated ``backend="cutile"`` kernel (PR #4019) against
the SGLang ``sgl_kernel`` reference (the SOTA baseline used by ocean-eval's
dashboard for this op) and a PyTorch-native reference.

Workloads mirror ocean-eval's ``Test_Per_Token_Group_Quant_8bit`` perf suite:
num_tokens in {128,256,384,512,768}, hidden in {2048,4096,7168}, group_size=128,
bf16 -> fp8_e4m3, row-major scales.

Provider availability is graceful: ``cutile`` needs ``cuda.tile`` (run in the
cuTile toolchain image); ``sgl`` needs ``sgl_kernel`` (run in an sglang image).
Whichever provider's backend is absent returns NaN so the other still plots.
Run the same script in each image and merge the rows for the full comparison.
"""

import numpy as np
import torch
import triton

import flashinfer  # noqa: F401
from flashinfer.quantization import per_token_group_quant_8bit
from flashinfer.testing.utils import bench_gpu_time_with_cudagraph

GROUP_SIZE = 128
EPS = 1e-10
DST_DTYPE = torch.float8_e4m3fn


def _sgl_available() -> bool:
    try:
        from sgl_kernel import sgl_per_token_group_quant_8bit  # noqa: F401

        return True
    except Exception:
        return False


def _cutile_available() -> bool:
    try:
        import cuda.tile  # noqa: F401

        return True
    except Exception:
        return False


def _torch_reference(x, group_size, dst_dtype, eps):
    finfo = torch.finfo(dst_dtype)
    x_ = x.reshape(x.numel() // group_size, group_size)
    amax = x_.abs().max(dim=-1, keepdim=True)[0].clamp(min=eps).to(torch.float32)
    x_s = amax / finfo.max
    x_q = (x_ / x_s).clamp(min=finfo.min, max=finfo.max).to(dst_dtype).reshape(x.shape)
    x_s = x_s.reshape(x.shape[:-1] + (x.shape[-1] // group_size,))
    return x_q, x_s


def benchmark_config(num_tokens, hidden_dim, provider):
    device = "cuda"
    torch.manual_seed(0)
    x = torch.randn(num_tokens, hidden_dim, dtype=torch.bfloat16, device=device)

    if provider == "cutile":
        if not _cutile_available():
            return float("nan"), float("nan"), float("nan")

        def execute():
            per_token_group_quant_8bit(
                x, GROUP_SIZE, eps=EPS, dst_dtype=DST_DTYPE, backend="cutile"
            )

    elif provider == "sgl":
        if not _sgl_available():
            return float("nan"), float("nan"), float("nan")
        from sgl_kernel import sgl_per_token_group_quant_8bit

        finfo = torch.finfo(DST_DTYPE)
        x_q = torch.empty_like(x, dtype=DST_DTYPE)
        x_s = torch.empty(
            x.shape[:-1] + (x.shape[-1] // GROUP_SIZE,),
            device=device,
            dtype=torch.float32,
        )

        def execute():
            sgl_per_token_group_quant_8bit(
                x, x_q, x_s, GROUP_SIZE, EPS, finfo.min, finfo.max, False, enable_v2=False
            )

    elif provider == "torch":

        def execute():
            _torch_reference(x, GROUP_SIZE, DST_DTYPE, EPS)

    else:
        raise ValueError(f"Unknown provider: {provider}")

    measurements = bench_gpu_time_with_cudagraph(execute)
    ms = np.median(measurements)
    return ms, np.percentile(measurements, 20), np.percentile(measurements, 80)


def _make_report(hidden_dim, plot_color):
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["num_tokens"],
            x_vals=[128, 256, 384, 512, 768],
            line_arg="provider",
            line_vals=["cutile", "sgl", "torch"],
            line_names=["cuTile", "SGLang (SOTA)", "PyTorch"],
            styles=[("orange", "-"), ("blue", "-"), ("green", "--")],
            ylabel="Latency (ms)",
            plot_name=f"per-token-group-quant-8bit-hidden{hidden_dim}",
            args={"hidden_dim": hidden_dim},
        )
    )
    def _bench(num_tokens, hidden_dim, provider):
        return benchmark_config(num_tokens, hidden_dim, provider)

    return _bench


benchmark_h2048 = _make_report(2048, "orange")
benchmark_h4096 = _make_report(4096, "orange")
benchmark_h7168 = _make_report(7168, "orange")


if __name__ == "__main__":
    print(
        f"per_token_group_quant_8bit  gs={GROUP_SIZE} bf16->fp8_e4m3  "
        f"(cutile={_cutile_available()} sgl={_sgl_available()})"
    )
    for bench in (benchmark_h2048, benchmark_h4096, benchmark_h7168):
        bench.run(print_data=True)
