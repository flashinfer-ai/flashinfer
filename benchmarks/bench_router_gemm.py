import numpy as np
import torch

from flashinfer.testing.utils import bench_gpu_time_with_cudagraph
from flashinfer.dsv3_ops import mm_M1_16_K7168_N128, mm_M1_16_K7168_N256


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


def bench_router_gemm(gemm_fn, data, M, N, K, reps=1000, warmup_reps=1000):
    measurements = bench_gpu_time_with_cudagraph(
        lambda: gemm_fn(*data),
        dry_run_time_ms=warmup_reps,
        repeat_time_ms=reps,
    )
    ms = np.median(measurements)
    flops = (2 * M * N * K) / ms / 1e9
    add_desc = f" launch_with_pdl={data[3]}" if len(data) > 3 else ""
    print(
        f"Router GEMM function {gemm_fn} | num_tokens={M}, num_experts={N}{add_desc} | Median execution time: {1000 * ms:.3f} us | TFLOPs/s: {flops:.3f}"
    )


def main():
    hidden_dim = 7168
    for num_tokens in [1, 2, 4, 8, 16]:
        for num_experts, output_dtype, flashinfer_fn in [
            (128, torch.bfloat16, mm_M1_16_K7168_N128),
            (256, torch.float32, mm_M1_16_K7168_N256),
        ]:
            data_torch = get_data_torch(
                num_tokens=num_tokens, hidden_dim=hidden_dim, num_experts=num_experts
            )
            bench_router_gemm(
                reference_torch, data_torch, num_tokens, num_experts, hidden_dim
            )

            data_flashinfer = get_data_flashinfer(
                num_tokens=num_tokens,
                hidden_dim=hidden_dim,
                num_experts=num_experts,
                output_dtype=output_dtype,
            )
            for launch_with_pdl in [False, True]:
                bench_router_gemm(
                    flashinfer_fn,
                    (*data_flashinfer, launch_with_pdl),
                    num_tokens,
                    num_experts,
                    hidden_dim,
                )

            print()


if __name__ == "__main__":
    main()
