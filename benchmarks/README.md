# FlashInfer Perf Benchmarking Framework -- flashinfer_benchmark.py

A comprehensive testing and benchmarking framework for FlashInfer's kernels.

The aim of `flashinfer_benchmark.py` is to provide a single framework for benchmarking any FlashInfer kernel and replace standalone benchmarking scripts. Current support surface includes batched prefill & decode, and FP8 gemm.

## Overview

This framework provides tools to:
- Benchmark FlashInfer's Attention, GEMM, and MOE API performance from different kernel backends (FlashAttention2/3, cuDNN, cuBLAS, CUTLASS, TensorRT-LLM)
- Compare performance across different configurations
- Batch performance test multiple attention test cases

Currently supports testing most attention, gemm, and fused MOE APIs:
- Attention:
    - `BatchDecodeWithPagedKVCacheWrapper` - Decode attention with paged KV cache. Also supports computationally similar `cudnn_batch_decode_with_kv_cache` and `trtllm_batch_decode_with_kv_cache`.
    - `BatchPrefillWithPagedKVCacheWrapper` - Prefill attention with paged KV cache. Also supports computationally similar `cudnn_batch_prefill_with_kv_cache` and  `trtllm_batch_context_with_kv_cache`.
    - `BatchPrefillWithRaggedKVCacheWrapper` - Prefill attention with ragged KV cache.
    - `BatchMLAPagedAttentionWrapper` - MLA attention proposed in DeepSeek series of models. Also supports computationally similar `trtllm_batch_decode_with_kv_cache_mla`.
- GEMM:
    - `gemm_fp8_nt_groupwise` - GEMM with FP8 data types using groupwise scaling.
    - `group_gemm_fp8_nt_groupwise` - Group GEMM with FP8 data types using groupwise scaling.
    - `bmm_fp8` - Batched matrix multiplication with FP8 inputs.
    - `mm_fp4` - Maxtrix multiplication with NVFP4 inputs.
- MOE:
    - `trtllm_fp4_block_scale_moe` - MOE with FP4 quantized weights and block-wise scaling.
    - `trtllm_fp8_block_scale_moe` - MOE with FP8 quantized weights and block-wise scaling.
    - `trtllm_fp8_per_tensor_scale_moe` - MOE with FP8 quantized weights and per-tensor scaling.
    - `cutlass_fused_moe` - CUTLASS fused MoE (base/fp8/nvfp4 variants with optional TP/EP)

## Quick Start
*See samples in samples/sample_testlist.txt for more example commands.*
### Single Test Run
Example commands
```bash
# Test prefill attention with paged KV cache
python3 flashinfer_benchmark.py --routine BatchPrefillWithPagedKVCacheWrapper --backends fa2 cudnn trtllm-gen --page_size 16 --batch_size 16 --s_qo 4096 --s_kv 4096 --num_qo_heads 64 --num_kv_heads 8 --head_dim_qk 128 --head_dim_vo 128 --random_actual_seq_len --verbose --refcheck --causal --no_cuda_graph --generate_repro_command

# Test decode attention with paged KV cache
python3 flashinfer_benchmark.py --routine BatchDecodeWithPagedKVCacheWrapper --backends fa2 fa2_tc trtllm-gen cudnn --page_size 16 --batch_size 16 --s_qo 1 --s_kv 8192 --num_qo_heads 64 --num_kv_heads 8 --head_dim_qk 128 --head_dim_vo 128 --random_actual_seq_len --verbose --refcheck --generate_repro_command

# FP4 GEMM
python3 flashinfer_benchmark.py --routine mm_fp4 --m 8192 --n 4096 --k 16384 --out_dtype bfloat16 --backends cudnn cutlass trtllm --use_128x4_sf_layout --refcheck --verbose --generate_repro_command

# MOE FP4 Block Scale (DeepSeekV3 routing)
python3 flashinfer_benchmark.py --routine trtllm_fp4_block_scale_moe --num_tokens 1024 --hidden_size 1024 --intermediate_size 1024 --num_experts 128 --top_k 8 --n_group 8 --topk_group 4 --routed_scaling_factor 2.5 --use_routing_bias --routing_method deepseek_v3 --use_shuffled_weight --verbose --generate_repro_command

# MOE FP4 Block Scale (topk routing, GeGlu gated act)
python3 flashinfer_benchmark.py --routine trtllm_fp4_block_scale_moe --num_tokens 1024 --hidden_size 1024 --intermediate_size 1024 --num_experts 128 --top_k 8 --routing_method topk --use_shuffled_weight --gated_act geglu --verbose --generate_repro_command

# MOE FP8 Block Scale with DeepSeekV3 routing
python3 flashinfer_benchmark.py --routine trtllm_fp8_block_scale_moe --num_tokens 1024 --hidden_size 1024 --intermediate_size 1024 --num_experts 128 --top_k 8 --n_group 8 --topk_group 4 --routed_scaling_factor 2.5 --use_routing_bias --routing_method deepseek_v3 --use_shuffled_weight --verbose --generate_repro_command

# CUTLASS Fused MoE (base variant)
python3 flashinfer_benchmark.py --routine cutlass_fused_moe --num_tokens 32 --hidden_size 128 --intermediate_size 128 --num_experts 2 --top_k 2 --cutlass_variant base --input_dtype float16 --verbose --generate_repro_command

# CUTLASS Fused MoE with Tensor Parallelism (TP)
python3 flashinfer_benchmark.py --routine cutlass_fused_moe --num_tokens 32 --hidden_size 128 --intermediate_size 128 --num_experts 2 --top_k 2 --cutlass_variant base --input_dtype float16 --tp_size 2 --tp_rank 0 --verbose --generate_repro_command
```

### Batch Testing

Run multiple tests from a file and save results:
```bash
python3 flashinfer_benchmark.py --testlist samples/sample_testlist.txt --output_path samples/sample_testlist_output.csv
```

The output CSV will contain detailed metrics including:
- Median execution time
- Standard deviation
- TFLOPS/sec
- Memory throughput (TB/sec)
- Input flags
- Reproducer commands if `--generate_repro_command` is provided

## Command Line Arguments
### General Flags
| Flag                     | Description                                                                                                 |
|--------------------------|-------------------------------------------------------------------------------------------------------------|
| `--routine`              | Test routine to run: `BatchDecodeWithPagedKVCacheWrapper`, `BatchPrefillWithPagedKVCacheWrapper`, `BatchPrefillWithRaggedKVCacheWrapper`, `BatchMLAPagedAttentionWrapper`, `gemm_fp8_nt_groupwise`, `group_gemm_fp8_nt_groupwise`, `bmm_fp8`, `mm_fp4`, `trtllm_fp4_block_scale_moe`, `trtllm_fp8_block_scale_moe`, `trtllm_fp8_per_tensor_scale_moe`, `cutlass_fused_moe` |
| `--num_iters`            | Number of iterations for performance measurement                                                           |
| `--dry_run_iters`        | Number of warmup iterations                                                                                |
| `--no_cuda_graph`        | Disable CUDA graph to execute kernels outside of the graph.                                                |
| `--refcheck`             | Verify outputs match between different backends                                                            |
| `--allow_output_mismatch`| Continue testing even if outputs don't pass refcheck                                              |
| `--random_seed`          | Random seed for reproducibility                                                                            |
| `--output_path`          | Path to save CSV results                                                                                   |
| `--testlist`             | Path to a file containing a list of test cases to run in batch mode                                        |
| `--verbose`, `-v`        | Print additional information (can be used multiple times for more verbosity, e.g. `-vv`)                   |
| `--case_tag`              | Optional tag for the test case, useful for annotating or filtering results in the output CSV.              |
| `--generate_repro_command`| If set, prints a reproducer command for the test case and stores it in the output CSV.                     |
| `--backends`             | Space-separated list of backends to test, e.g. fa2, fa2_tc, fa3, cudnn, cutlass, trtllm, trtllm-gen, trtllm-gen-native, cublas|

### Attention Flags
| Flag                     | Description                                                                                                 |
|--------------------------|-------------------------------------------------------------------------------------------------------------|
| `--page_size`            | Page size for paged attention. Required for paged attention tests.                                          |
| `--batch_size`           | Number of sequences to process in parallel                                                                  |
| `--s_qo`                 | Query/output sequence length. Should be 1 for decode tests.                                                 |
| `--s_kv`                 | Key/value sequence length (context length)                                                                  |
| `--num_qo_heads`         | Number of query/output attention heads                                                                      |
| `--num_kv_heads`         | Number of key/value attention heads                                                                         |
| `--head_dim_qk`          | Head dimension for Q/K. Must be 128 or 192.                                                                |
| `--head_dim_vo`          | Head dimension for V/O. Usually equals head_dim_qk.                                                        |
| `--head_dim_ckv`         | Head dimension for C/K/V (MLA attention).                                                                  |
| `--head_dim_kpe`         | Head dimension for KPE (MLA attention).                                                                    |
| `--q_dtype`              | Data type for the query tensor. Default: bfloat16. Currently only bfloat16 is supported.                   |
| `--kv_dtype`             | Data type for the key and value tensors. Default: bfloat16. Currently only bfloat16 is supported.          |
| `--causal`               | Use causal attention masking (prefill only)                                                                |
| `--random_actual_seq_len`| Use random sequence lengths up to max length. If False, use max length.                                    |

### GEMM Flags
| Flag                     | Description                                                                                                 |
|--------------------------|-------------------------------------------------------------------------------------------------------------|
| `--m`                    | Number of rows of matrix A and output matrix (GEMM M dimension)                                            |
| `--n`                    | Number of columns of matrix B and output matrix (GEMM N dimension)                                         |
| `--k`                    | Number of columns of matrix A / rows of matrix B (GEMM K dimension)                                        |
| `--tile_size`            | Tile size for the GEMM operation (affects performance and scaling)                                         |
| `--group_size`           | Number of groups for group GEMM (batching multiple GEMMs together)                                         |
| `--scale_major_mode`     | Layout for FP8 scaling: `MN` (per output tile) or `K` (per input tile)                                     |
| `--out_dtype`            | Output data type: `bfloat16` or `float16`                                                                  |
| `--mma_sm`               | Number of SMs to use for the MMA operation (1 or 2)                                                        |
| `--input_dtype`          | Data type for input matrix (for FP8 GEMM, e.g. `fp8_e4m3`)                                                 |
| `--mat2_dtype`           | Data type for second matrix (for FP8 GEMM, e.g. `fp8_e4m3`)                                                |
| `--use_128x4_sf_layout`  | Use 128x4 scale/format layout for FP4 GEMM (for `mm_fp4` routine)                                          |

### MOE Flags
| Flag                     | Description                                                                                                 |
|--------------------------|-------------------------------------------------------------------------------------------------------------|
| `--num_tokens`           | Number of input tokens                                                                                     |
| `--hidden_size`          | Hidden dimension size                                                                                      |
| `--intermediate_size`    | Intermediate dimension size (FF layer dimension)                                                           |
| `--num_experts`          | Total number of experts                                                                                    |
| `--top_k`                | Number of experts to route to per token                                                                    |
| `--n_group`              | Number of expert groups (for DeepSeek routing). Default: 1                                                 |
| `--topk_group`           | Number of groups to consider for top-k routing. Default: 1                                                 |
| `--routed_scaling_factor`| Scaling factor for routing. Default: 2.5                                                                   |
| `--local_expert_offset`  | Offset of local experts in global expert space. Default: 0                                                 |
| `--local_num_experts`    | Number of experts handled by this device. Default: equals num_experts                                      |
| `--tile_tokens_dim`      | Tile dimension for tokens. Default: 8                                                                      |
| `--routing_method`       | Routing method: `renormalize`, `deepseek_v3`, `llama4`, `renormalize_naive`. Default: `deepseek_v3`.       |
| `--use_shuffled_weight`  | Whether to use shuffled weight layout                                                                      |
| `--weight_layout`        | Weight layout: 0=MajorK, 1=MajorMn,  2=BlockMajorK. Default: 0                                             |
| `--use_routing_bias`     | Whether to use routing bias                                                                                |
| `--use_routing_scales_on_input` | Whether to use routing scales on input (for Llama4 routing)                                         |
| `--input_dtype`          | Data type of the input hidden states. Default: bfloat16                                                    |
| `--weight_dtype`         | Data type of the weights (before quantization). Default: bfloat16                                          |
| `--cutlass_variant`      | CUTLASS MoE variant: `base` (no quant), `fp8` (per-tensor FP8), `nvfp4` (FP4 block-scale)                   |
| `--quantized_input`      | For `nvfp4` only: quantize input activations to FP4                                                         |
| `--tp_size`              | Tensor-parallel world size                                                                                  |
| `--tp_rank`              | Tensor-parallel rank                                                                                        |
| `--ep_size`              | Expert-parallel world size                                                                                  |
| `--ep_rank`              | Expert-parallel rank                                                                                        |
| `--gated_act`            | Gated activation function: `swiglu` (default) or `geglu`                                                   |
| `--autotune`             | Enable autotune for supported operation                                                                     |

### MOE Routing Method Compatibility

| Routing Method         | Requirements | Compatible MOE Types |
|------------------------|--------------|---------------------|
| **deepseek_v3**        | `top_k <= 8`, `topk_group <= 4`, requires `--n_group`, `--topk_group`, `--routed_scaling_factor`, `--use_routing_bias` | FP4, FP8 Block Scale |
| **renormalize**        | `top_k == 1` for FP8 Block Scale, `top_k <= 8` for FP4. Do NOT use `--n_group` or `--topk_group` | All MOE types |
| **llama4**             | `top_k == 1`, requires `--routed_scaling_factor`, `--use_routing_bias`, `--use_routing_scales_on_input`. Do NOT use `--n_group` or `--topk_group` | FP8 Per-Tensor |
| **renormalize_naive**  | `top_k == 1` for FP8 Block Scale, `top_k <= 8` for FP4. Do NOT use `--n_group` or `--topk_group` | FP4 primarily |

Notes:
- Group parameters (`--n_group`, `--topk_group`) are ONLY used with DeepSeekV3 routing method. Using them with other routing methods will cause the error: "Routing kernel with groups implies DeepSeekV3 routing method."
- Different MOE kernel implementations have different `top_k` constraints. FP8 MOE kernels (both Block Scale and Per-Tensor) have stricter limits than FP4 for non-DeepSeekV3 routing methods.
- FP8 MOE kernels require integer values for group parameters, while FP4 MOE kernels accept optional values.
- CUTLASS fused MoE (`cutlass_fused_moe`) ignores `--routing_method`, `--n_group`, and `--topk_group`; it computes routing via softmax+top-k internally from the provided logits.

## Tester Attention Backend Support Matrix
The following support surface applies to attention operations in `flashinfer_benchmark.py`. Support surface has been tested on NVIDIA B200 GPU unless noted otherwise.
| Backend            | Decode Paged| Prefill Paged | Prefill Ragged | MLA  | Notes                                    |
|--------------------|-------------|---------------|----------------|------|------------------------------------------|
| fa2                | ✓           | ✓             | ✓              | ✓    | [FlashAttention-2](https://github.com/Dao-AILab/flash-attention) without tensor cores. Does not support GQA ratio of 5          |
| fa2_tc             | ✓           | ✗             | ✗              | ✗    | [FlashAttention-2](https://github.com/Dao-AILab/flash-attention) with tensor cores                        |
| fa3                | ✗           | ✓             | ✓              | ✗    | [FlashAttention-3](https://github.com/Dao-AILab/flash-attention) implemented with CUTLASS. Hopper Only.  |
| cudnn              | ✓           | ✓             | ✓              | ✗    |  |
| cutlass            | ✗           | ✗             | ✓              | ✗    | FMHA implemented with CUTLASS.          |
| trtllm-gen         | ✓           | ✓             | ✗              | ✗    | trtllm-gen kernels called through unified wrapper interface, such as `Batch...Wrapper` by setting `backend='trtllm-gen'` |
| trtllm-gen-native  | ✓           | ✓             | ✗              | ✓    | trtllm-gen kernels called through a separate API such as `flashinfer.[prefill,decode].trtllm_batch_...` |

Notes:
- CUDA graph support is only stable with BatchDecodeWithPagedKVCacheWrapper. For BatchPrefillWithPagedKVCacheWrapper and BatchPrefillWithRaggedKVCacheWrapper, it is recommended that `--no_cuda_graph` is used.
- cudnn, cutlass, and trtllm backends are supported on [CUDA Compute Capability 10.0 GPUs](https://developer.nvidia.com/cuda-gpus) only.
- fa3 is supported on [CUDA Compute Capability 9.0 GPUs](https://developer.nvidia.com/cuda-gpus) only.

## Example Outputs
```bash
$ python3 flashinfer_benchmark.py --routine BatchPrefillWithPagedKVCacheWrapper --backends fa2 cudnn trtllm-gen --page_size 16 --batch_size 16 --s_qo 1024 --s_kv 1024 --num_qo_heads 8 --num_kv_heads 8 --head_dim_qk 128 --head_dim_vo 128 --random_actual_seq_len --refcheck --causal --no_cuda_graph --generate_repro_command -v
[INFO] args = Namespace(routine='BatchPrefillWithPagedKVCacheWrapper', no_cuda_graph=True, refcheck=True, allow_output_mismatch=False, random_seed=42, verbose=1, output_path=None, num_iters=30, dry_run_iters=5, case_tag=None, generate_repro_command=True, repro_command='', backends=['fa2', 'cudnn', 'trtllm-gen'], page_size=16, batch_size=16, s_qo=1024, s_kv=1024, num_qo_heads=8, num_kv_heads=8, head_dim_qk=128, head_dim_vo=128, head_dim_ckv=None, head_dim_kpe=None, q_dtype='bfloat16', kv_dtype='bfloat16', causal=True, random_actual_seq_len=True)
[INFO] Running testBatchPrefillWithPagedKVCacheWrapper
[INFO] FlashInfer version: 0.2.12
[INFO] To reproduce this test case, run the following command: python3 flashinfer_benchmark.py --routine BatchPrefillWithPagedKVCacheWrapper --backends fa2 cudnn trtllm-gen --page_size 16 --batch_size 16 --s_qo 1024 --s_kv 1024 --num_qo_heads 8 --num_kv_heads 8 --head_dim_qk 128 --head_dim_vo 128 --random_actual_seq_len --refcheck --causal --no_cuda_graph --generate_repro_command -v
[VERBOSE] Average actual seq len: 327
[PERF] fa2       :: median time 0.059 ms; std 0.001 ms; achieved tflops 91.278 TFLOPs/sec; achieved tb_per_sec 0.723 TB/sec
[PERF] cudnn     :: median time 0.047 ms; std 0.002 ms; achieved tflops 116.377 TFLOPs/sec; achieved tb_per_sec 0.921 TB/sec
[PERF] trtllm-gen:: median time 0.051 ms; std 0.002 ms; achieved tflops 105.873 TFLOPs/sec; achieved tb_per_sec 0.838 TB/sec

$ python3 flashinfer_benchmark.py --routine BatchPrefillWithRaggedKVCacheWrapper --backends fa2 cudnn cutlass --batch_size 16 --s_qo 1024 --s_kv 1024 --num_qo_heads 128 --num_kv_heads 128 --head_dim_qk 192 --head_dim_vo 128  --refcheck --causal --no_cuda_graph --generate_repro_command -v
[INFO] args = Namespace(routine='BatchPrefillWithRaggedKVCacheWrapper', no_cuda_graph=True, refcheck=True, allow_output_mismatch=False, random_seed=42, verbose=1, output_path=None, num_iters=30, dry_run_iters=5, case_tag=None, generate_repro_command=True, repro_command='', backends=['fa2', 'cudnn', 'cutlass'], page_size=0, batch_size=16, s_qo=1024, s_kv=1024, num_qo_heads=128, num_kv_heads=128, head_dim_qk=192, head_dim_vo=128, head_dim_ckv=None, head_dim_kpe=None, q_dtype='bfloat16', kv_dtype='bfloat16', causal=True, random_actual_seq_len=False)
[INFO] Running testBatchPrefillWithRaggedKVCacheWrapper
[INFO] FlashInfer version: 0.2.12
[INFO] To reproduce this test case, run the following command: python3 flashinfer_benchmark.py --routine BatchPrefillWithRaggedKVCacheWrapper --backends fa2 cudnn cutlass --batch_size 16 --s_qo 1024 --s_kv 1024 --num_qo_heads 128 --num_kv_heads 128 --head_dim_qk 192 --head_dim_vo 128 --refcheck --causal --no_cuda_graph --generate_repro_command -v
[VERBOSE] Average actual seq len: 1024
[PERF] fa2       :: median time 2.252 ms; std 0.038 ms; achieved tflops 305.092 TFLOPs/sec; achieved tb_per_sec 1.192 TB/sec
[PERF] cudnn     :: median time 1.178 ms; std 0.054 ms; achieved tflops 583.460 TFLOPs/sec; achieved tb_per_sec 2.279 TB/sec
[PERF] cutlass   :: median time 1.494 ms; std 0.034 ms; achieved tflops 459.866 TFLOPs/sec; achieved tb_per_sec 1.796 TB/sec

$ python3 flashinfer_benchmark.py --routine BatchDecodeWithPagedKVCacheWrapper --backends fa2 fa2_tc trtllm-gen cudnn --page_size 16 --batch_size 32 --s_qo 1 --s_kv 8192 --num_qo_heads 64 --num_kv_heads 8 --head_dim_qk 128 --head_dim_vo 128 --refcheck --generate_repro_command -v
[INFO] args = Namespace(routine='BatchDecodeWithPagedKVCacheWrapper', no_cuda_graph=False, refcheck=True, allow_output_mismatch=False, random_seed=42, verbose=1, output_path=None, num_iters=30, dry_run_iters=5, case_tag=None, generate_repro_command=True, repro_command='', backends=['fa2', 'fa2_tc', 'trtllm-gen', 'cudnn'], page_size=16, batch_size=32, s_qo=1, s_kv=8192, num_qo_heads=64, num_kv_heads=8, head_dim_qk=128, head_dim_vo=128, head_dim_ckv=None, head_dim_kpe=None, q_dtype='bfloat16', kv_dtype='bfloat16', causal=False, random_actual_seq_len=False)
[INFO] Running testBatchDecodeWithPagedKVCacheWrapper
[INFO] FlashInfer version: 0.2.12
[INFO] To reproduce this test case, run the following command: python3 flashinfer_benchmark.py --routine BatchDecodeWithPagedKVCacheWrapper --backends fa2 fa2_tc trtllm-gen cudnn --page_size 16 --batch_size 32 --s_qo 1 --s_kv 8192 --num_qo_heads 64 --num_kv_heads 8 --head_dim_qk 128 --head_dim_vo 128 --refcheck --generate_repro_command -v
[VERBOSE] Average actual seq len: 8192
[PERF] fa2       :: median time 0.712 ms; std 0.000 ms; achieved tflops 12.070 TFLOPs/sec; achieved tb_per_sec 1.510 TB/sec
[PERF] fa2_tc    :: median time 0.187 ms; std 0.002 ms; achieved tflops 46.022 TFLOPs/sec; achieved tb_per_sec 5.758 TB/sec
[PERF] trtllm-gen:: median time 0.157 ms; std 0.001 ms; achieved tflops 54.581 TFLOPs/sec; achieved tb_per_sec 6.829 TB/sec
[PERF] cudnn     :: median time 0.170 ms; std 0.000 ms; achieved tflops 50.535 TFLOPs/sec; achieved tb_per_sec 6.323 TB/sec
