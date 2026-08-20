import numpy as np
import torch

from flashinfer.testing.utils import bench_gpu_time_with_cudagraph
from flashinfer.dsv3_ops import mm_M1_16

# (label, num_experts, hidden_dim, output dtype) for the MoE routers this kernel
# is used by. The bf16-out DeepSeek-V3 row is there because SGLang's router can
# be configured either way, and the two dtypes have different store costs.
MODEL_SHAPES = [
    ("Mistral-Large-3", 128, 7168, torch.bfloat16),
    ("DeepSeek-V3", 256, 7168, torch.float32),
    ("DeepSeek-V3-bf16out", 256, 7168, torch.bfloat16),
    ("GLM-MoE-DSA", 256, 6144, torch.float32),
    ("Kimi-K2", 384, 7168, torch.float32),
    ("Kimi-K2-bf16out", 384, 7168, torch.bfloat16),
    ("Kimi-K3", 896, 7168, torch.float32),
]


@torch.compile
def reference_torch(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
):
    return torch.nn.functional.linear(x, weight, bias)


def get_data_torch(num_tokens, num_experts, hidden_dim):
    mat_a = torch.randn(num_tokens, hidden_dim, device="cuda", dtype=torch.bfloat16)
    mat_b = torch.randn(num_experts, hidden_dim, device="cuda", dtype=torch.bfloat16)
    return mat_a, mat_b


def get_data_flashinfer(num_tokens, num_experts, hidden_dim, output_dtype):
    mat_a = torch.randn(num_tokens, hidden_dim, device="cuda", dtype=torch.bfloat16)
    mat_b = torch.randn(
        num_experts, hidden_dim, device="cuda", dtype=torch.bfloat16
    ).t()
    out = torch.empty(num_tokens, num_experts, device="cuda", dtype=output_dtype)
    return mat_a, mat_b, out


def bench_router_gemm(gemm_fn, data, reps=1000, warmup_reps=1000):
    """Return the median execution time in milliseconds."""
    measurements = bench_gpu_time_with_cudagraph(
        lambda: gemm_fn(*data),
        dry_run_time_ms=warmup_reps,
        repeat_time_ms=reps,
    )
    return float(np.median(measurements))


def main():
    header = (
        f"{'model':<20} {'N':>5} {'K':>6} {'out':>9} {'M':>3} "
        f"{'torch us':>9} {'fi us':>9} {'fi+pdl us':>10} {'speedup':>8} {'TFLOP/s':>8}"
    )
    print(header)
    print("-" * len(header))
    for label, num_experts, hidden_dim, output_dtype in MODEL_SHAPES:
        for num_tokens in [1, 2, 4, 8, 16]:
            data_torch = get_data_torch(
                num_tokens=num_tokens, hidden_dim=hidden_dim, num_experts=num_experts
            )
            ms_torch = bench_router_gemm(reference_torch, data_torch)

            data_flashinfer = get_data_flashinfer(
                num_tokens=num_tokens,
                hidden_dim=hidden_dim,
                num_experts=num_experts,
                output_dtype=output_dtype,
            )
            ms_by_pdl = {
                launch_with_pdl: bench_router_gemm(
                    mm_M1_16, (*data_flashinfer, launch_with_pdl)
                )
                for launch_with_pdl in (False, True)
            }
            best = min(ms_by_pdl.values())
            flops = (2 * num_tokens * num_experts * hidden_dim) / best / 1e9
            print(
                f"{label:<20} {num_experts:>5} {hidden_dim:>6} "
                f"{str(output_dtype).replace('torch.', ''):>9} {num_tokens:>3} "
                f"{1000 * ms_torch:>9.3f} {1000 * ms_by_pdl[False]:>9.3f} "
                f"{1000 * ms_by_pdl[True]:>10.3f} {ms_torch / best:>7.2f}x "
                f"{flops:>8.1f}"
            )
        print()


if __name__ == "__main__":
    main()
