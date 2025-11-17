# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from typing import Optional
import csv
import os 

from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    should_use_deepgemm_for_fp8_linear,
    per_token_group_quant_fp8,
    fp8_gemm_nt,
    dispatch_w8a8_blockscale_func,
)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    CUTLASS_BLOCK_FP8_SUPPORTED,
)
from vllm.platforms import current_platform
from vllm.triton_utils import triton as vllm_triton
from flashinfer.cute_dsl.blockwise_gemm import BlockwiseGemmKernel, blockwise_gemm

assert current_platform.is_cuda(), (
    "Only support benchmarking w8a8 block fp8 kernel on CUDA device."
)

def apply_w8a8_block_fp8_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    block_size: list[int],
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    cutlass_block_fp8_supported: bool = CUTLASS_BLOCK_FP8_SUPPORTED,
    use_aiter_and_is_supported: bool = False,
    use_cute_dsl: bool = False,
    cute_dsl_gemm_params: Optional[dict] = None,
) -> torch.Tensor:
    assert input_scale is None
    # View input as 2D matrix for fp8 methods
    input_2d = input.view(-1, input.shape[-1])
    output_shape = [*input.shape[:-1], weight.shape[0]]
    output_dtype = input.dtype

    if should_use_deepgemm_for_fp8_linear(output_dtype, weight):

        input_2d = input.view(-1, input.shape[-1])
        output_shape = [*input.shape[:-1], weight.shape[0]]

        q_input, x_scale = per_token_group_quant_fp8(
            input_2d,
            block_size[1],
            column_major_scales=True,
        )
        output = torch.empty((q_input.shape[0], weight.shape[0]),
                             dtype=torch.bfloat16,
                             device=q_input.device)
        fp8_gemm_nt((q_input, x_scale), (weight, weight_scale), output)
        if bias is not None:
            output += bias
        return output.to(dtype=output_dtype).view(*output_shape)

    w8a8_blockscale_func = dispatch_w8a8_blockscale_func(
        cutlass_block_fp8_supported, use_aiter_and_is_supported)
    if use_cute_dsl:
        q_input, x_scale = per_token_group_quant_fp8(input_2d,
                                                     block_size[1],
                                                     column_major_scales=True)
        output = torch.empty((q_input.shape[0], weight.shape[0]),
                          dtype=input.dtype,
                          device=input.device)
        #print("cute:",q_input.shape, x_scale.shape, weight.shape, weight_scale.shape, output.shape)
        blockwise_gemm(
            q_input,
            x_scale,
            weight,
            weight_scale,
            output,
            ab_dtype="float8_e4m3fn",#q_input.dtype,
            sf_dtype="float32",#x_scale.dtype,
            c_dtype="bfloat16",#out.dtype,
            acc_dtype="float32",#torch.float32,
            sm_count=148,
            mma_tiler_mn=cute_dsl_gemm_params.get('mma_tiler_mn', (128, 128)),
            cluster_shape_mn=cute_dsl_gemm_params.get('cluster_shape_mn', (2, 2)),
            use_2cta_instrs=cute_dsl_gemm_params.get('use_2cta_instrs', True),
        )
    elif cutlass_block_fp8_supported:
        num_pad = 0
        if current_platform.is_device_capability(90):
            # pad first dimension to be divisible by 4 due to
            # cutlass blockwise gemm limitation for hopper
            num_pad = 4 - (input_2d.shape[0] % 4)
            if num_pad > 0:
                input_2d = torch.nn.functional.pad(input_2d,
                                                   (0, 0, 0, num_pad),
                                                   "constant", 0)
        q_input, x_scale = per_token_group_quant_fp8(input_2d,
                                                     block_size[1],
                                                     column_major_scales=True)
        output = w8a8_blockscale_func(q_input, weight, x_scale, weight_scale,
                                      block_size, input.dtype)
        #print("cutlass:",q_input.shape, x_scale.shape, weight.shape, weight_scale.shape, output.shape)
        if num_pad > 0:
            output = output[:-num_pad]
    else:

        q_input, x_scale = per_token_group_quant_fp8(
            input_2d, block_size[1], column_major_scales=False)

        output = w8a8_blockscale_func(q_input, weight, x_scale, weight_scale,
                                      block_size, input.dtype)

    if bias is not None:
        output = output + bias
    return output.to(dtype=input.dtype).view(*output_shape)

# DeepSeek-V3 weight shapes
DEEPSEEK_V3_SHAPES = [
    (512 + 64, 7168),
    (2112, 7168),
    ((128 + 64) * 128, 7168),
    (128 * (128 + 128), 512),
    (7168, 16384),
    (7168, 18432),
    (18432 * 2, 7168),
    (24576, 1536),
    (12288, 7168),
    (4096, 7168),
    (7168, 2048),
]

def test_correctness(N=576, K = 7168, M = 16 ):
    block_size = (128, 128)
    factor_for_scale = 1e-2

    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_max, fp8_min = fp8_info.max, fp8_info.min
    device = "cuda"
    # Create random FP8 tensors
    A_ref = (torch.rand(M, K, dtype=torch.bfloat16, device=device) - 0.5) * 2 * fp8_max

    B_ref = (torch.rand(N, K, dtype=torch.bfloat16, device=device) - 0.5) * 2 * fp8_max
    B = B_ref.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

    # Create scales
    block_n, block_k = block_size[0], block_size[1]
    n_tiles = (N + block_n - 1) // block_n
    k_tiles = (K + block_k - 1) // block_k

    Bs = (
        torch.rand(n_tiles, k_tiles, dtype=torch.float32, device=device)
        * factor_for_scale
    )
    output_cute = apply_w8a8_block_fp8_linear(
                A_ref, B, block_size, Bs, use_cute_dsl=True
            )
    output_cutlass = apply_w8a8_block_fp8_linear(
                A_ref, B, block_size, Bs, cutlass_block_fp8_supported=True
            )
    output_triton = apply_w8a8_block_fp8_linear(
                A_ref, B, block_size, Bs, cutlass_block_fp8_supported=False
            )
    print(output_cute)
    print(output_cutlass)
    is_close_cute_cutlass = torch.allclose(output_cute, output_cutlass, atol=1e-2, rtol=1e-2)
    is_close_cute_triton = torch.allclose(output_cute, output_triton, atol=1e-2, rtol=1e-2)
    is_close_cutlass_triton = torch.allclose(output_cutlass, output_triton, atol=1e-2, rtol=1e-2)
    print(f"is_close_cute_cutlass: {is_close_cute_cutlass}")
    print(f"is_close_cute_triton: {is_close_cute_triton}")
    print(f"is_close_cutlass_triton: {is_close_cutlass_triton}")

def build_w8a8_block_fp8_runner(M, N, K, block_size, device, use_cutlass, use_cute_dsl=False, cute_dsl_gemm_params: Optional[dict] = None):
    """Build runner function for w8a8 block fp8 matmul."""
    factor_for_scale = 1e-2

    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_max, fp8_min = fp8_info.max, fp8_info.min

    # Create random FP8 tensors
    A_ref = (torch.rand(M, K, dtype=torch.bfloat16, device=device) - 0.5) * 2 * fp8_max

    B_ref = (torch.rand(N, K, dtype=torch.bfloat16, device=device) - 0.5) * 2 * fp8_max
    B = B_ref.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

    # Create scales
    block_n, block_k = block_size[0], block_size[1]
    n_tiles = (N + block_n - 1) // block_n
    k_tiles = (K + block_k - 1) // block_k

    Bs = (
        torch.rand(n_tiles, k_tiles, dtype=torch.float32, device=device)
        * factor_for_scale
    )

    # SM90 CUTLASS requires row-major format for scales
    if use_cutlass and current_platform.is_device_capability(90):
        Bs = Bs.T.contiguous()

    def run():
        if use_cute_dsl:
            return apply_w8a8_block_fp8_linear(
                A_ref, B, block_size, Bs, use_cute_dsl=True, cute_dsl_gemm_params=cute_dsl_gemm_params
            )
        elif use_cutlass:
            return apply_w8a8_block_fp8_linear(
                A_ref, B, block_size, Bs, cutlass_block_fp8_supported=True
            )
        else:
            return apply_w8a8_block_fp8_linear(
                A_ref, B, block_size, Bs, cutlass_block_fp8_supported=False
            )

    return run


# Determine available providers
#, "w8a8-block-fp8-cute-dsl"
available_providers = ["torch-bf16", "w8a8-block-fp8-triton", "w8a8-block-fp8-cute-dsl"]
plot_title = "BF16 vs W8A8 Block FP8 GEMMs"

if CUTLASS_BLOCK_FP8_SUPPORTED:
    available_providers.append("w8a8-block-fp8-cutlass")


# @vllm_triton.testing.perf_report(
#     vllm_triton.testing.Benchmark(
#         x_names=["batch_size"],
#         x_vals=[16, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384],
#         #x_vals=[16],
#         x_log=False,
#         line_arg="provider",
#         line_vals=available_providers,
#         line_names=available_providers,
#         ylabel="TFLOP/s (larger is better)",
#         plot_name="BF16 vs W8A8 Block FP8 GEMMs",
#         args={},
#     )
# )
def benchmark_tflops(batch_size, provider, N, K, block_size=(128, 128), cute_dsl_gemm_params: Optional[dict] = None):
    M = batch_size
    device = "cuda"

    quantiles = [0.5, 0.2, 0.8]

    if provider == "torch-bf16":
        a = torch.randn((M, K), device=device, dtype=torch.bfloat16)
        b = torch.randn((N, K), device=device, dtype=torch.bfloat16)
        ms, min_ms, max_ms = vllm_triton.testing.do_bench_cudagraph(
            lambda: torch.nn.functional.linear(a, b), quantiles=quantiles
        )
    elif provider == "w8a8-block-fp8-triton":
          run_w8a8_triton = build_w8a8_block_fp8_runner(
              M, N, K, block_size, device, use_cutlass=False
          )
          ms, min_ms, max_ms = vllm_triton.testing.do_bench_cudagraph(
              lambda: run_w8a8_triton(), quantiles=quantiles
          )
    elif provider == "w8a8-block-fp8-cutlass":
        run_w8a8_cutlass = build_w8a8_block_fp8_runner(
            M, N, K, block_size, device, use_cutlass=True
        )
        ms, min_ms, max_ms = vllm_triton.testing.do_bench_cudagraph(
            lambda: run_w8a8_cutlass(), quantiles=quantiles
        )
    elif provider == "w8a8-block-fp8-cute-dsl":
        try:
          run_w8a8_cute_dsl = build_w8a8_block_fp8_runner(
              M, N, K, block_size, device, use_cutlass=False, use_cute_dsl=True, cute_dsl_gemm_params=cute_dsl_gemm_params
          )
          ms, min_ms, max_ms = vllm_triton.testing.do_bench_cudagraph(
              lambda: run_w8a8_cute_dsl(), quantiles=quantiles
          )
        except Exception as e:
            # Track failed shape
            # failed_shapes.append({
            #     'M': M, 'N': N, 'K': K,
            #     'block_size': block_size,
            #     'error': str(e)[:100]  # Keep first 100 chars of error
            # })
            print(f"  ⚠️  CUTE backend failed for shape M={M}, N={N}, K={K}: {e}")
            # Return "X" marker (use negative value to signal failure in numeric context)
            return 0, 0, 0
    else:
        raise ValueError(f"Unknown provider: {provider}")

    to_tflops = lambda t_ms: (2 * M * N * K) * 1e-12 / (t_ms * 1e-3)
    return to_tflops(ms), to_tflops(max_ms), to_tflops(min_ms)


if __name__ == "__main__":
    # correctness test
    # for M in [16, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]:
    #     for N, K in DEEPSEEK_V3_SHAPES:
    #         test_correctness(N=N, K=K, M=M)
    # exit()

    # Benchmarking DeepSeek-V3
    # block_size = (128, 128)

    # for N, K in DEEPSEEK_V3_SHAPES:
    #     print(f"\nBenchmarking DeepSeek-V3, N={N} K={K}")

    #     print(f"TFLOP/s comparison (block_size={block_size}):")
    #     benchmark_tflops.run(
    #         print_data=True,
    #         # show_plots=False,
    #         save_path=f"bench_w8a8_block_fp8_tflops_n{N}_k{K}",
    #         N=N,
    #         K=K,
    #         block_size=block_size,
    #     )
    #     #break

    # auto-tuning
    block_size = (128, 128)
    csv_file = 'benchmark_results.csv'
    file_exists = os.path.exists(csv_file)
    with open(csv_file, 'a', newline='') as f:
      writer = csv.writer(f)
      
      # Write header if file is new
      if not file_exists:
          writer.writerow([
              'M', 'N', 'K',
              'torch-bf16 (TFLOP/s)',
              'w8a8-block-fp8-triton (TFLOP/s)',
              'w8a8-block-fp8-cutlass (TFLOP/s)',
              'w8a8-block-fp8-cute-dsl (TFLOP/s)',
              'mma_tiler_mn',
              'cluster_shape_mn',
              'use_2cta_instrs'
          ])
      for M in [16, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]:
          for N, K in DEEPSEEK_V3_SHAPES:
              backend_results = {}
              for provider in ["torch-bf16", "w8a8-block-fp8-triton","w8a8-block-fp8-cutlass"]:
                  res = benchmark_tflops(M, provider, N, K, block_size)
                  #print(f"M={M}, N={N}, K={K}, provider={provider}, res={res}")
                  backend_results[provider] = res[0]
              #"w8a8-block-fp8-cute-dsl"
              max_flops = float('-inf')
              max_settings = None
              for mma_tiler_mn in [(64, 128), (128, 128), (256, 128)]:
                  for use_2cta_instrs in [True, False]:
                      for cluster_shape_mn in [(1, 1),(1, 2),(2,1),(2,2),(1,4),(4,1),(2,4),(4,2),(1,8),(8,1),(2,8),(8,2),(1,16),(16,1)]:
                          if not (
                              (not use_2cta_instrs and mma_tiler_mn[0] in [64, 128])
                              or (use_2cta_instrs and mma_tiler_mn[0] in [128, 256])
                          ):
                              continue
                          if mma_tiler_mn[1] not in (128,):
                              continue
                          if cluster_shape_mn[0] % (2 if use_2cta_instrs else 1) != 0:
                              continue
                          cute_dsl_gemm_params = {
                              'mma_tiler_mn': mma_tiler_mn,
                              'cluster_shape_mn': cluster_shape_mn,
                              'use_2cta_instrs': use_2cta_instrs
                          }
                          res = benchmark_tflops(M, "w8a8-block-fp8-cute-dsl", N, K, block_size, cute_dsl_gemm_params)
                          if res[0] > max_flops:
                              max_flops = res[0]
                              backend_results["w8a8-block-fp8-cute-dsl"] = res[0]
                              max_settings = cute_dsl_gemm_params
              #print(f"M={M}, N={N}, K={K}, min_settings={min_settings}, min_time={min_time}")
              print(f"M={M}, N={N}, K={K}")
              print(backend_results)
              print(max_settings)
              # Write data row
              writer.writerow([
                  M, N, K,
                  backend_results.get('torch-bf16', 0),
                  backend_results.get('w8a8-block-fp8-triton', 0),
                  backend_results.get('w8a8-block-fp8-cutlass', 0),
                  backend_results.get('w8a8-block-fp8-cute-dsl', 0),
                  max_settings.get('mma_tiler_mn', '') if max_settings else '',
                  max_settings.get('cluster_shape_mn', '') if max_settings else '',
                  max_settings.get('use_2cta_instrs', '') if max_settings else ''
              ])
              print("--------------------------------")
    print("\nBenchmark finished!")
