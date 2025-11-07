# FlashInfer Perf Benchmarking Framework -- `flashinfer_benchmark.py`

The aim of `flashinfer_benchmark.py` is to provide a single framework for benchmarking any FlashInfer kernel and replace standalone benchmarking scripts.

## Overview

This framework provides tools to:
- Benchmark FlashInfer's Attention, GEMM, and MOE API performance from different kernel backends such as FlashAttention2/3, cuDNN, cuBLAS, CUTLASS, and TensorRT-LLM
- Compare performance across different configurations
- Batch performance test multiple attention test cases

Currently supports testing most attention, gemm, and fused MOE APIs:
- Attention:
    - `BatchDecodeWithPagedKVCacheWrapper` - Decode attention with paged KV cache.
        - Also supports computationally similar `cudnn_batch_decode_with_kv_cache` and `trtllm_batch_decode_with_kv_cache`.
    - `BatchPrefillWithPagedKVCacheWrapper` - Prefill attention with paged KV cache.
        - Also supports computationally similar `cudnn_batch_prefill_with_kv_cache` and  `trtllm_batch_context_with_kv_cache`.
    - `BatchPrefillWithRaggedKVCacheWrapper` - Prefill attention with ragged KV cache.
        - Also supports computationally similar `cudnn_batch_prefill_with_kv_cache` and  `trtllm_ragged_attention_deepseek`.
    - `BatchMLAPagedAttentionWrapper` - MLA attention proposed in DeepSeek series of models.
        - Also supports computationally similar `trtllm_batch_decode_with_kv_cache_mla`.
- GEMM:
    - `gemm_fp8_nt_groupwise` - GEMM with FP8 data types using groupwise scaling.
    - `group_gemm_fp8_nt_groupwise` - Group GEMM with FP8 data types using groupwise scaling.
    - `bmm_fp8` - Batched matrix multiplication with FP8 inputs.
    - `mm_fp4` - Matrix multiplication with NVFP4 inputs.
- MOE:
    - `trtllm_fp4_block_scale_moe` - MOE with FP4 quantized weights and block-wise scaling.
    - `trtllm_fp8_block_scale_moe` - MOE with FP8 quantized weights and block-wise scaling.
    - `trtllm_fp8_per_tensor_scale_moe` - MOE with FP8 quantized weights and per-tensor scaling.
    - `cutlass_fused_moe` - CUTLASS fused MoE (base/fp8/nvfp4 variants with optional TP/EP)

## Quick Start
### Single Test Run
A test case is generally invoked as `python3 flashinfer_benchmark.py --routine <routine_name> <flags>`.

*See samples in samples/sample_testlist.txt for various example test flags.*
Example commands and outputs areas follows

```bash
# bmm_fp8
$ python3 flashinfer_benchmark.py --routine bmm_fp8 --batch_size 256 --m 1 --n 1024 --k 7168 --input_dtype fp8_e4m3 --mat2_dtype fp8_e4m3 --out_dtype bfloat16 --backends cudnn cublas cutlass --refcheck -vv --generate_repro_command
[INFO] args = Namespace(routine='bmm_fp8', no_cuda_graph=False, use_cupti=False, refcheck=True, allow_output_mismatch=False, random_seed=42, verbose=2, output_path=None, num_iters=30, dry_run_iters=5, case_tag=None, generate_repro_command=True, repro_command='', batch_size=256, m=1, n=1024, k=7168, tile_size=128, group_size=1, scale_major_mode='MN', input_dtype='fp8_e4m3', mat2_dtype='fp8_e4m3', out_dtype='bfloat16', mma_sm=1, backends=['cudnn', 'cublas', 'cutlass'], use_128x4_sf_layout=False, use_nvfp4=False, autotune=False)
[INFO] Running testBmmFp8
[INFO] FlashInfer version: 0.3.1
[VVERBOSE] gpu_name = 'NVIDIA_B200'
[INFO] To reproduce this test case, run the following command: python3 flashinfer_benchmark.py --routine bmm_fp8 --batch_size 256 --m 1 --n 1024 --k 7168 --input_dtype fp8_e4m3 --mat2_dtype fp8_e4m3 --out_dtype bfloat16 --backends cudnn cublas cutlass --refcheck -vv --generate_repro_command
[VVERBOSE] input_fp8.shape = torch.Size([256, 1, 7168])
[VVERBOSE] input_fp8.dtype = torch.float8_e4m3fn
[VVERBOSE] mat2_fp8.shape = torch.Size([256, 7168, 1024])
[VVERBOSE] mat2_fp8.dtype = torch.float8_e4m3fn
[VVERBOSE] input_inv_s = tensor(0.0109, device='cuda:0')
[VVERBOSE] input_inv_s.dtype = torch.float32
[VVERBOSE] mat2_inv_s = tensor(0.0135, device='cuda:0')
[VVERBOSE] mat2_inv_s.dtype = torch.float32
[PERF] cudnn          :: median time 0.285 ms; std 0.000 ms; achieved tflops 13.180 TFLOPs/sec; achieved tb_per_sec 0.026 TB/sec
[PERF] cublas         :: median time 0.286 ms; std 0.000 ms; achieved tflops 13.159 TFLOPs/sec; achieved tb_per_sec 0.026 TB/sec
[PERF] cutlass        :: median time 0.266 ms; std 0.001 ms; achieved tflops 14.137 TFLOPs/sec; achieved tb_per_sec 0.028 TB/sec

# non-paged (ragged) prefill
$ python3 flashinfer_benchmark.py --routine BatchPrefillWithRaggedKVCacheWrapper --backends fa2 fa3 cutlass cudnn --batch_size 16 --s_qo 1024 --s_kv 1024 --num_qo_heads 128 --num_kv_heads 128 --head_dim_qk 192 --head_dim_vo 128 --random_actual_seq_len -vv --refcheck --causal --q_dtype bfloat16 --kv_dtype bfloat16 --allow_output_mismatch --generate_repro_command --case_tag "DeepSeek-R1"
[INFO] args = Namespace(routine='BatchPrefillWithRaggedKVCacheWrapper', no_cuda_graph=False, use_cupti=False, refcheck=True, allow_output_mismatch=True, random_seed=42, verbose=2, output_path=None, num_iters=30, dry_run_iters=5, case_tag='DeepSeek-R1', generate_repro_command=True, repro_command='', backends=['fa2', 'fa3', 'cutlass', 'cudnn'], page_size=0, batch_size=16, s_qo=1024, s_kv=1024, num_qo_heads=128, num_kv_heads=128, head_dim_qk=192, head_dim_vo=128, head_dim_ckv=None, head_dim_kpe=None, q_dtype='bfloat16', kv_dtype='bfloat16', causal=True, random_actual_seq_len=True)
[INFO] Running testBatchPrefillWithRaggedKVCacheWrapper
[INFO] FlashInfer version: 0.3.1
[VVERBOSE] gpu_name = 'NVIDIA_B200'
[INFO] To reproduce this test case, run the following command: python3 flashinfer_benchmark.py --routine BatchPrefillWithRaggedKVCacheWrapper --backends fa2 fa3 cutlass cudnn --batch_size 16 --s_qo 1024 --s_kv 1024 --num_qo_heads 128 --num_kv_heads 128 --head_dim_qk 192 --head_dim_vo 128 --random_actual_seq_len -vv --refcheck --causal --q_dtype bfloat16 --kv_dtype bfloat16 --allow_output_mismatch --generate_repro_command --case_tag DeepSeek-R1
[WARNING] fa3 for routine BatchPrefillWithRaggedKVCacheWrapper is not supported on compute capability 10.0. Skipping.
[VVERBOSE] s_qo == s_kv, making actual_seq_lens_kv the same as actual_seq_lens_q
[VERBOSE] Average actual qo seq len: 327
[VERBOSE] Average actual kv seq len: 327
[VVERBOSE] actual_seq_lens_q.flatten() = tensor([103, 436, 861, 271, 107,  72, 701,  21, 615, 122, 467, 215, 331, 459,
         88, 373], dtype=torch.int32)
[VVERBOSE] actual_seq_lens_kv.flatten() = tensor([103, 436, 861, 271, 107,  72, 701,  21, 615, 122, 467, 215, 331, 459,
         88, 373], dtype=torch.int32)
[VVERBOSE] q.shape = torch.Size([5242, 128, 192])
[VVERBOSE] k.shape = torch.Size([5242, 128, 192])
[VVERBOSE] v.shape = torch.Size([5242, 128, 128])
[VVERBOSE] qo_indptr.shape = torch.Size([17])
[VVERBOSE] kv_indptr.shape = torch.Size([17])
[VVERBOSE] scale = 0.07216878364870323
[PERF] fa2            :: median time 0.495 ms; std 0.006 ms; achieved tflops 219.336 TFLOPs/sec; achieved tb_per_sec 1.736 TB/sec
[PERF] cutlass        :: median time 0.530 ms; std 0.002 ms; achieved tflops 204.674 TFLOPs/sec; achieved tb_per_sec 1.620 TB/sec
[PERF] cudnn          :: median time 0.313 ms; std 0.000 ms; achieved tflops 346.715 TFLOPs/sec; achieved tb_per_sec 2.745 TB/sec
```

### Batch Testing

Run multiple tests from a file and save results:
```bash
python3 flashinfer_benchmark.py --testlist samples/sample_testlist.txt --output_path samples/sample_testlist_output.csv
```

See `samples/sample_testlist.txt` for an example stdout output from the above command; `samples/sample_testlist_output.csv` for csv output from the same run.

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
| `--use_cupti`            | Use CUPTI for timing GPU kernels when available. |
| `--refcheck`             | Verify outputs match between different backends                                                            |
| `--allow_output_mismatch`| Continue testing even if outputs don't pass refcheck                                              |
| `--random_seed`          | Random seed for reproducibility                                                                            |
| `--output_path`          | Path to save CSV results                                                                                   |
| `--testlist`             | Path to a file containing a list of test cases to run in batch mode                                        |
| `--verbose`, `-v`        | Print additional information (can be used multiple times for more verbosity, e.g. `-vv`)                   |
| `--case_tag`              | Optional tag for the test case, useful for annotating or filtering results in the output CSV.              |
| `--generate_repro_command`| If set, prints a reproducer command for the test case and stores it in the output CSV.                     |
| `--backends`             | Space-separated list of backends to test, e.g. fa2, fa2_tc, fa3, cudnn, cutlass, trtllm, trtllm-gen, trtllm-native, cublas|

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
| `--use_nvfp4`            | Whether to use nvfp4 quantization or mxfp4 quantization, defaults to False.(for `mm_fp4` routine)          |
| `--autotune`             | Enable autotune for supported operation (`trtllm` and `cutlass` backends for `mm_fp4` and `bmm_fp8` routines)|

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

## `flashinfer_benchmark.py` Routine & Backend Support Matrix
The following table summarizes the support surface of each routine & backend's on various [CUDA Compute Capabilities](https://developer.nvidia.com/cuda-gpus).

Each column represents a compute capability. Backends inside cells represent supported backends. A blank cell means no backend is supported for that routine at that compute capability.

<!--
Legend:
- fa2: FlashAttention-2
- fa2_tc: FlashAttention-2 (Tensor Core)
- fa3: FlashAttention-3
- cudnn: cuDNN
- cutlass: CUTLASS
- trtllm: TensorRT-LLM
- trtllm-gen: TensorRT-LLM (generic wrapper)
- trtllm-native: TensorRT-LLM (native API)
-->
| Routine | 7.5 | 8.0 | 8.6 | 8.9 | 9.0 | 10.0 | 10.3 | 12.0 |
|---------|-----|-----|-----|-----|-----|-------|-------|-------|
| **BatchDecodeWithPagedKVCacheWrapper** | fa2 | fa2, fa2_tc, cudnn | fa2, fa2_tc, cudnn | fa2, fa2_tc, cudnn | fa2, fa2_tc, cudnn | fa2, fa2_tc, cudnn, trtllm-gen, trtllm-native | fa2, fa2_tc, cudnn, trtllm-gen, trtllm-native | fa2, fa2_tc, cudnn |
| **BatchPrefillWithPagedKVCacheWrapper** |  | fa2, cudnn | fa2, cudnn | fa2, cudnn | fa2, fa3, cudnn | fa2, cudnn, trtllm-gen, trtllm-native | fa2, cudnn, trtllm-gen, trtllm-native | fa2, cudnn |
| **BatchPrefillWithRaggedKVCacheWrapper** |  | fa2, cudnn | fa2, cudnn | fa2, cudnn | fa2, fa3, cudnn | fa2, cudnn, cutlass, trtllm-native | fa2, cudnn, cutlass, trtllm-native | fa2, cudnn |
| **BatchMLAPagedAttentionWrapper** |  | fa2 | fa2 | fa2 | fa2, fa3 | fa2, cutlass, trtllm-native | fa2, cutlass, trtllm-native | fa2 |
| **gemm_fp8_nt_groupwise** |  |  |  |  |  | cutlass | cutlass |  |
| **group_gemm_fp8_nt_groupwise** |  |  |  |  |  | cutlass | cutlass |  |
| **bmm_fp8** |  |  |  | cudnn, cublas | cudnn, cublas | cudnn, cublas, cutlass | cudnn, cublas, cutlass | cudnn, cublas |
| **mm_fp4** |  |  |  |  |  | cudnn, trtllm, cutlass | cudnn, trtllm, cutlass | cudnn |
| **trtllm_fp4_block_scale_moe** |  |  |  |  |  | trtllm | trtllm |  |
| **trtllm_fp8_block_scale_moe** |  |  |  |  |  | trtllm | trtllm |  |
| **trtllm_fp8_per_tensor_scale_moe** |  |  |  |  |  | trtllm | trtllm |  |
| **cutlass_fused_moe** |  |  |  |  |  | cutlass | cutlass |  |

Backend Legend:
- fa2: FlashAttention2
- fa2_tc: FlashAttention2 (with Tensor Cores for `BatchDecodeWithPagedKVCacheWrapper`)
- fa3: FlashAttention-3
- cudnn: cuDNN
- cutlass: CUTLASS
- trtllm: TensorRT-LLM
- trtllm-gen: TensorRT-LLM
- trtllm-native: TensorRT-LLM (out-of-wrapper)
