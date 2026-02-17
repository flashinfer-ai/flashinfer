# FlashInfer Perf Benchmarking Framework -- `flashinfer_benchmark.py`

The aim of `flashinfer_benchmark.py` is to provide a single framework for benchmarking any FlashInfer kernel and replace standalone benchmarking scripts.

## Overview

This framework provides tools to:
- Benchmark FlashInfer's Attention, GEMM, MOE, Norm, Quantization, Sampling, RoPE, and Mamba API performance from different kernel backends such as FlashAttention2/3, cuDNN, cuBLAS, CUTLASS, CuTe-DSL, TensorRT-LLM, and Triton
- Compare performance across different configurations
- Batch performance test multiple test cases

Currently supports testing attention, gemm, fused MOE, normalization, quantization, sampling, RoPE, and Mamba APIs:
- Attention:
    - `BatchDecodeWithPagedKVCacheWrapper` - Decode attention with paged KV cache.
        - Also supports computationally similar `cudnn_batch_decode_with_kv_cache` and `trtllm_batch_decode_with_kv_cache`.
    - `BatchPrefillWithPagedKVCacheWrapper` - Prefill attention with paged KV cache.
        - Also supports computationally similar `cudnn_batch_prefill_with_kv_cache` and  `trtllm_batch_context_with_kv_cache`.
    - `BatchPrefillWithRaggedKVCacheWrapper` - Prefill attention with ragged KV cache.
        - Also supports computationally similar `cudnn_batch_prefill_with_kv_cache` (cudnn-native) and  `trtllm_ragged_attention_deepseek`.
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
- MOE Communication:
    - `moe_a2a_dispatch_combine` - MoE All-to-All dispatch + combine benchmark for multi-GPU expert-parallel inference. Requires `mpirun` for multi-GPU execution. Supports optional quantization (FP8, NVFP4, FP8 block-scale) and real MoE kernel computation.
- Norm:
    - `rmsnorm` - Root Mean Square Layer Normalization.
    - `rmsnorm_quant` - RMSNorm with FP8 quantized output.
    - `fused_add_rmsnorm_quant` - Fused residual add + RMSNorm with FP8 quantized output.
    - `rmsnorm_fp4quant` - RMSNorm with FP4 quantized output (CuTe-DSL, Blackwell SM10.0+).
    - `add_rmsnorm_fp4quant` - Fused residual add + RMSNorm with FP4 quantized output (CuTe-DSL, Blackwell SM10.0+).
- Quantization:
    - `mxfp8_quantize` - Quantize tensor to MxFP8 format (Blackwell SM10.0+).
    - `mxfp4_quantize` - Quantize tensor to MxFP4 format (Blackwell SM10.0+).
    - `nvfp4_quantize` - Quantize tensor to NVFP4 format with configurable scale factor layout (Blackwell SM10.0+).
    - `nvfp4_batched_quantize` - Batched NVFP4 quantization (Blackwell SM10.0+).
- Sampling:
    - `softmax` - Softmax with optional temperature scaling.
    - `sampling_from_probs` - Sample token indices from probability distributions.
    - `sampling_from_logits` - Sample token indices from logits (fused softmax + sampling).
    - `top_k_sampling_from_probs` - Top-K sampling from probabilities.
    - `top_p_sampling_from_probs` - Top-P (nucleus) sampling from probabilities.
    - `top_k_top_p_sampling_from_probs` - Combined Top-K and Top-P sampling from probabilities.
    - `top_k_top_p_sampling_from_logits` - Combined Top-K and Top-P sampling from logits.
    - `min_p_sampling_from_probs` - Min-P sampling from probabilities.
    - `top_k_renorm_probs` - Renormalize probabilities after Top-K filtering.
    - `top_p_renorm_probs` - Renormalize probabilities after Top-P filtering.
    - `top_k_mask_logits` - Mask logits outside Top-K values.
    - `chain_speculative_sampling` - Chain speculative sampling for speculative decoding.
    - `top_k` - Radix-based Top-K selection.
    - `top_k_page_table_transform` - Fused Top-K with page table lookup.
    - `top_k_ragged_transform` - Fused Top-K with ragged index transform.
- RoPE (Rotary Positional Embeddings):
    - `apply_rope` - Apply RoPE with indptr/offsets.
    - `apply_rope_pos_ids` - Apply RoPE with position IDs.
    - `apply_llama31_rope` - Apply Llama 3.1 style RoPE with indptr/offsets.
    - `apply_llama31_rope_pos_ids` - Apply Llama 3.1 style RoPE with position IDs.
    - `apply_rope_with_cos_sin_cache` - Apply RoPE with precomputed cos/sin cache.
    - `mla_rope_quantize_fp8` - MLA RoPE with FP8 quantization (SM8.9+).
    - `rope_quantize_fp8` - RoPE with FP8 quantization (SM8.9+).
    - `rope_quantize_fp8_append_paged_kv_cache` - RoPE with FP8 quantization and paged KV cache append (SM8.9+).
- Mamba (Selective State Space Models):
    - `selective_state_update` - Selective state update for Mamba layers (generation phase). Supports both single-token prediction (STP) and multi-token prediction (MTP) via `--cache_steps`. Backends: `flashinfer` (CUDA, architecture-specific kernels for base/SM90/SM100+) and `triton` (reference).

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

# RMSNorm with FP8 quantized output
$ python3 flashinfer_benchmark.py --routine rmsnorm_quant --batch_size 32 --hidden_size 4096 --input_dtype bfloat16 --out_dtype fp8_e4m3 --scale 1.0 --refcheck -vv --generate_repro_command --case_tag "rmsnorm_quant_fp8_e4m3"
[INFO] Running testRmsnormQuant
[INFO] FlashInfer version: 0.6.1
[VVERBOSE] gpu_name = 'NVIDIA_B300_SXM6_AC'
[INFO] To reproduce this test case, run the following command: python3 flashinfer_benchmark.py --routine rmsnorm_quant --batch_size 32 --hidden_size 4096 --input_dtype bfloat16 --out_dtype fp8_e4m3 --scale 1.0 --refcheck -vv --generate_repro_command --case_tag rmsnorm_quant_fp8_e4m3
[VVERBOSE] input_tensor.shape = torch.Size([32, 4096])
[VVERBOSE] input_tensor.dtype = torch.bfloat16
[VVERBOSE] weight.shape = torch.Size([4096])
[VVERBOSE] out_tensor.dtype = torch.float8_e4m3fn
[VVERBOSE] scale = 1.0
[PERF] cuda           :: median time 0.003 ms; std 0.000 ms; achieved tflops 0.229 TFLOPs/sec; achieved tb_per_sec 0.140 TB/sec

# MxFP8 Quantization (Blackwell SM10.0+ only)
$ python3 flashinfer_benchmark.py --routine mxfp8_quantize --m 2048 --k 8192 --input_dtype bfloat16 --refcheck -vv --generate_repro_command --case_tag "mxfp8_quantize"
[INFO] args = Namespace(routine='mxfp8_quantize', no_cuda_graph=False, use_cupti=False, use_cuda_events=False, refcheck=True, allow_output_mismatch=False, random_seed=42, verbose=2, output_path=None, num_iters=30, dry_run_iters=5, case_tag='mxfp8_quantize', generate_repro_command=True, repro_command='', m=2048, k=8192, input_dtype='bfloat16', is_sf_swizzled_layout=True, no_sf_swizzled_layout=False, alignment=32, enable_pdl=False, backends=['cuda'], batch_size=None, global_scale=1.0, sf_layout='128x4', do_shuffle=False, sf_vec_size=16)
[INFO] Running testMxfp8Quantize
[INFO] FlashInfer version: 0.6.1
[VVERBOSE] gpu_name = 'NVIDIA_B300_SXM6_AC'
[INFO] To reproduce this test case, run the following command: python3 flashinfer_benchmark.py --routine mxfp8_quantize --m 2048 --k 8192 --input_dtype bfloat16 --refcheck -vv --generate_repro_command --case_tag mxfp8_quantize
[VVERBOSE] input_tensor.shape = torch.Size([2048, 8192])
[VVERBOSE] input_tensor.dtype = torch.bfloat16
[VVERBOSE] is_sf_swizzled_layout = True
[VVERBOSE] alignment = 32
[VVERBOSE] enable_pdl = False
[VVERBOSE] Backend cuda: x_q.shape = torch.Size([2048, 8192]), x_q.dtype = torch.float8_e4m3fn, sf.shape = torch.Size([524288]), sf.dtype = torch.uint8
[VVERBOSE] Round-trip error: 0/16777216 (0.00%) elements differ
[PERF] cuda           :: median time 0.016 ms; std 0.000 ms; achieved tflops 3.118 TFLOPs/sec; achieved tb_per_sec 3.150 TB/sec
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
| `--routine`              | Test routine to run. See [Overview](#overview) for full list including attention, GEMM, MOE, norm, and quantization routines. |
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
| `--backends`             | Space-separated list of backends to test, e.g. fa2, fa2_tc, fa3, auto, cudnn, cudnn-native, cutlass, trtllm, trtllm-gen, trtllm-native, cublas. (`auto` currently supported for `BatchDecodeWithPagedKVCacheWrapper` and `BatchPrefillWithPagedKVCacheWrapper`.)|

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
| `--local_num_experts`    | Number of experts handled by this device. Default: equals num_experts                                      |                                                                    |
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

### MoE Communication Flags (moe_a2a_dispatch_combine)
The `moe_a2a_dispatch_combine` routine benchmarks MoE All-to-All communication for multi-GPU expert-parallel inference. It must be launched with `mpirun`.

| Flag                     | Description                                                                                                 |
|--------------------------|-------------------------------------------------------------------------------------------------------------|
| `--num_tokens`           | Number of tokens per rank (local batch size)                                                               |
| `--hidden_size`          | Hidden dimension size                                                                                      |
| `--num_experts`          | Total number of experts across all ranks                                                                   |
| `--top_k`                | Number of experts to route each token to                                                                   |
| `--input_dtype`          | Data type for hidden states payload: `bfloat16` (default) or `float16`                                     |
| `--quant_dtype`          | Quantization format: `fp8` (per-tensor), `nvfp4` (block-scale FP4), `fp8_block_scale` (block-scale FP8)    |
| `--real_math`            | Run actual MoE kernels instead of fake computation. Requires `--intermediate_size` and `--quant_dtype` to be `nvfp4` or `fp8_block_scale` |
| `--intermediate_size`    | Intermediate FFN size. Required if `--real_math` is set                                                    |
| `--max_num_tokens`       | Max tokens per rank for workspace allocation. Defaults to `--num_tokens`                                   |
| `--validate`             | Run correctness validation before benchmarking using deterministic fake MoE                                |
| `--per_phase_timing`     | Enable per-phase timing (dispatch/combine/moe_kernel). Adds slight overhead from CUDA events               |
| `--nvtx`                 | Enable NVTX markers for Nsight Systems profiling                                                           |

**Launch Examples:**
```bash
# Basic (no quantization)
mpirun -np 8 python benchmarks/flashinfer_benchmark.py \
    --routine moe_a2a_dispatch_combine \
    --num_tokens 1024 --hidden_size 7168 --num_experts 256 --top_k 8

# With FP8 quantization
mpirun -np 8 python benchmarks/flashinfer_benchmark.py \
    --routine moe_a2a_dispatch_combine \
    --num_tokens 1024 --hidden_size 7168 --num_experts 256 --top_k 8 \
    --quant_dtype fp8

# With NVFP4 quantization and real MoE kernel
mpirun -np 8 python benchmarks/flashinfer_benchmark.py \
    --routine moe_a2a_dispatch_combine \
    --num_tokens 1024 --hidden_size 7168 --num_experts 256 --top_k 8 \
    --quant_dtype nvfp4 --real_math --intermediate_size 18432

# With validation and per-phase timing
mpirun -np 8 python benchmarks/flashinfer_benchmark.py \
    --routine moe_a2a_dispatch_combine \
    --num_tokens 1024 --hidden_size 7168 --num_experts 256 --top_k 8 \
    --validate --per_phase_timing
```

### Norm Flags
| Flag                     | Description                                                                                                 |
|--------------------------|-------------------------------------------------------------------------------------------------------------|
| `--batch_size`           | Batch size (number of sequences)                                                                           |
| `--hidden_size`          | Hidden dimension size                                                                                      |
| `--num_heads`            | Number of heads for 3D input shape (batch, num_heads, hidden_size). Optional; if not set, uses 2D shape.   |
| `--input_dtype`          | Input data type: `bfloat16` (default) or `float16`                                                         |
| `--eps`                  | Epsilon for numerical stability. Default: 1e-6                                                             |
| `--enable_pdl`           | Enable programmatic dependent launch                                                                       |
| `--scale`                | Scale factor for FP8 quantization (used by `rmsnorm_quant`, `fused_add_rmsnorm_quant`). Default: 1.0       |
| `--out_dtype`            | Output dtype: `fp8_e4m3`, `fp8_e5m2` (for FP8 quant); `nvfp4`, `mxfp4` (for FP4 quant). Default: `fp8_e4m3`|
| `--use_global_scale`     | Use global scale factor for NVFP4 format (FP4 routines only)                                               |
| `--is_sf_swizzled_layout`| Use swizzled scale factor layout for tensor core GEMM (FP4 routines only)                                  |
| `--backends`             | Backend to test: `cuda` (default) or `cute-dsl` (for FP4 routines)                                         |

### Quantization Flags
| Flag                     | Description                                                                                                 |
|--------------------------|-------------------------------------------------------------------------------------------------------------|
| `--m`                    | Number of rows in input tensor                                                                             |
| `--k`                    | Number of columns in input tensor (must be divisible by 32)                                                |
| `--input_dtype`          | Input data type: `bfloat16` (default) or `float16`                                                         |
| `--is_sf_swizzled_layout`| Use swizzled layout for scale factors. Default: True                                                       |
| `--no_sf_swizzled_layout`| Disable swizzled layout for scale factors                                                                  |
| `--alignment`            | sfVecSize for quantization. Default: 32                                                                    |
| `--enable_pdl`           | Enable programmatic dependent launch                                                                       |
| `--batch_size`           | Batch size for batched quantization (`nvfp4_batched_quantize` only)                                        |
| `--global_scale`         | Global scale factor for NVFP4 quantization. Default: 1.0                                                   |
| `--sf_layout`            | Scale factor layout for NVFP4: `128x4` (default), `8x4`, or `linear`                                       |
| `--do_shuffle`           | Shuffle scale factors for TRTLLM backend (`nvfp4_quantize` only)                                           |
| `--sf_vec_size`          | Scale factor vector size for NVFP4 quantization. Default: 16                                               |
| `--backends`             | Backend to test. Default: `cuda`                                                                           |

### Sampling Flags
| Flag                     | Description                                                                                                 |
|--------------------------|-------------------------------------------------------------------------------------------------------------|
| `--batch_size`           | Batch size (number of sequences)                                                                           |
| `--vocab_size`           | Vocabulary size                                                                                            |
| `--input_dtype`          | Input data type for logits: `float32` (default), `float16`, or `bfloat16`                                  |
| `--top_k`                | Top-K value for top-k sampling. Default: 50                                                                |
| `--top_p`                | Top-P threshold for top-p (nucleus) sampling. Default: 0.9                                                 |
| `--min_p`                | Min-P threshold for min-p sampling. Default: 0.1                                                           |
| `--temperature`          | Temperature for softmax. Default: 1.0                                                                      |
| `--filter_apply_order`   | Order of applying top-k and top-p filters: `top_k_first` (default) or `joint`                              |
| `--num_speculate_tokens` | Number of speculative tokens for chain speculative sampling. Default: 5                                    |
| `--max_len`              | Max sequence length for `top_k_page_table_transform` and `top_k_ragged_transform`. Default: 4096           |
| `--num_rows`             | Number of rows for `top_k_page_table_transform` and `top_k_ragged_transform`. Defaults to batch_size       |
| `--backends`             | Backend to test: `cuda` (default)                                                                          |

### RoPE Flags
| Flag                     | Description                                                                                                 |
|--------------------------|-------------------------------------------------------------------------------------------------------------|
| `--batch_size`           | Batch size (number of sequences)                                                                           |
| `--seq_len`              | Sequence length (qkv_len or kv_len)                                                                        |
| `--num_qo_heads`         | Number of query/output heads                                                                               |
| `--num_kv_heads`         | Number of key/value heads                                                                                  |
| `--head_dim`             | Head dimension                                                                                             |
| `--rotary_dim`           | Rotary dimension (defaults to head_dim if not specified)                                                   |
| `--no_rope_dim`          | Number of dimensions without RoPE (for MLA). Default: 0                                                    |
| `--input_dtype`          | Input data type: `float16` (default) or `bfloat16`                                                         |
| `--quant_dtype`          | Quantized data type for FP8 routines: `fp8_e4m3` (default) or `fp8_e5m2`                                   |
| `--rope_scale`           | RoPE scaling factor. Default: 1.0                                                                          |
| `--rope_theta`           | RoPE theta base frequency. Default: 10000.0                                                                |
| `--interleave`           | Use interleaved rotary embedding (GPT-J style)                                                             |
| `--page_size`            | Page size for paged KV cache. Default: 16                                                                  |
| `--kv_layout`            | KV cache layout: `NHD` (default) or `HND`                                                                  |
| `--low_freq_factor`      | Low frequency factor for Llama 3.1 RoPE. Default: 1.0                                                      |
| `--high_freq_factor`     | High frequency factor for Llama 3.1 RoPE. Default: 4.0                                                     |
| `--old_context_len`      | Old context length for Llama 3.1 RoPE. Default: 8192                                                       |
| `--backends`             | Backend to test: `cuda` (default)                                                                          |

### Mamba Flags
| Flag                     | Description                                                                                                 |
|--------------------------|-------------------------------------------------------------------------------------------------------------|
| `--batch_size`           | Batch size (number of sequences)                                                                           |
| `--nheads`               | Number of SSM heads                                                                                        |
| `--dim`                  | Head dimension (headdim)                                                                                   |
| `--dstate`               | SSM state size                                                                                             |
| `--ngroups`              | Number of groups for B and C matrices. `nheads` must be divisible by `ngroups`, and `nheads/ngroups` must be 1, 8, or 16. Default: 8 |
| `--cache_steps`          | Number of steps/tokens for multi-token prediction (MTP). 0 = single-token prediction (STP). Default: 0    |
| `--input_dtype`          | Data type for input tensors (x, B, C, z): `bfloat16` (default). Only `bfloat16` is supported.             |
| `--state_dtype`          | Data type for the SSM state cache: `bfloat16` (default), `float16`, or `float32`                           |
| `--weight_dtype`         | Data type for weight tensors (dt, D, dt_bias): `float32` (default) or `bfloat16`                           |
| `--has_z`                | Include z tensor for gating (`z * sigmoid(z)` applied to output)                                           |
| `--dt_softplus`          | Apply softplus to dt before use                                                                            |
| `--backends`             | Backends to test: `flashinfer` (default), `triton` (reference). Refcheck compares against Triton reference |

## `flashinfer_benchmark.py` Routine & Backend Support Matrix
The following table summarizes the support surface of each routine & backend's on various [CUDA Compute Capabilities](https://developer.nvidia.com/cuda-gpus).

Each column represents a compute capability. Backends inside cells represent supported backends. A blank cell means no backend is supported for that routine at that compute capability.

<!--
Legend:
- fa2: FlashAttention-2
- fa2_tc: FlashAttention-2 (Tensor Core)
- fa3: FlashAttention-3
- cudnn: cuDNN (via wrapper API)
- cudnn-native: cuDNN (direct API call)
- cutlass: CUTLASS
- trtllm: TensorRT-LLM
- trtllm-gen: TensorRT-LLM (generic wrapper)
- trtllm-native: TensorRT-LLM (native API)
-->
| Routine | 7.5 | 8.0 | 8.6 | 8.9 | 9.0 | 10.0 | 10.3 | 12.0 |
|---------|-----|-----|-----|-----|-----|-------|-------|-------|
| **BatchDecodeWithPagedKVCacheWrapper** | fa2 | fa2, fa2_tc, cudnn | fa2, fa2_tc, cudnn | fa2, fa2_tc, cudnn | fa2, fa2_tc, cudnn | fa2, fa2_tc, cudnn, trtllm-gen, trtllm-native | fa2, fa2_tc, cudnn, trtllm-gen, trtllm-native | fa2, fa2_tc, cudnn |
| **BatchPrefillWithPagedKVCacheWrapper** |  | fa2, cudnn, cudnn-native | fa2, cudnn, cudnn-native | fa2, cudnn, cudnn-native | fa2, fa3, cudnn, cudnn-native | fa2, cudnn, cudnn-native, trtllm-gen, trtllm-native | fa2, cudnn, cudnn-native, trtllm-gen, trtllm-native | fa2, cudnn, cudnn-native |
| **BatchPrefillWithRaggedKVCacheWrapper** |  | fa2, cudnn, cudnn-native | fa2, cudnn, cudnn-native | fa2, cudnn, cudnn-native | fa2, fa3, cudnn, cudnn-native | fa2, cudnn, cudnn-native, cutlass, trtllm-native | fa2, cudnn, cudnn-native, cutlass, trtllm-native | fa2, cudnn, cudnn-native |
| **BatchMLAPagedAttentionWrapper** |  | fa2 | fa2 | fa2 | fa2, fa3 | fa2, cutlass, trtllm-native | fa2, cutlass, trtllm-native | fa2 |
| **gemm_fp8_nt_groupwise** |  |  |  |  |  | cutlass | cutlass |  |
| **group_gemm_fp8_nt_groupwise** |  |  |  |  |  | cutlass | cutlass |  |
| **bmm_fp8** |  |  |  | cudnn, cublas | cudnn, cublas | cudnn, cublas, cutlass | cudnn, cublas, cutlass | cudnn, cublas |
| **mm_fp4** |  |  |  |  |  | cudnn, trtllm, cutlass | cudnn, trtllm, cutlass | cudnn |
| **trtllm_fp4_block_scale_moe** |  |  |  |  |  | trtllm | trtllm |  |
| **trtllm_fp8_block_scale_moe** |  |  |  |  |  | trtllm | trtllm |  |
| **trtllm_fp8_per_tensor_scale_moe** |  |  |  |  |  | trtllm | trtllm |  |
| **cutlass_fused_moe** |  |  |  |  |  | cutlass | cutlass |  |
| **moe_a2a_dispatch_combine** |  |  |  |  |  | moe_a2a | moe_a2a |  |
| **rmsnorm** | cuda | cuda | cuda | cuda | cuda | cuda | cuda | cuda |
| **rmsnorm_quant** | cuda | cuda | cuda | cuda | cuda | cuda | cuda | cuda |
| **fused_add_rmsnorm_quant** | cuda | cuda | cuda | cuda | cuda | cuda | cuda | cuda |
| **rmsnorm_fp4quant** |  |  |  |  |  | cute-dsl | cute-dsl |  |
| **add_rmsnorm_fp4quant** |  |  |  |  |  | cute-dsl | cute-dsl |  |
| **mxfp8_quantize** |  |  |  |  |  | cuda | cuda |  |
| **mxfp4_quantize** |  |  |  |  |  | cuda | cuda |  |
| **nvfp4_quantize** |  |  |  |  |  | cuda | cuda |  |
| **nvfp4_batched_quantize** |  |  |  |  |  | cuda | cuda |  |
| **softmax** | cuda | cuda | cuda | cuda | cuda | cuda | cuda | cuda |
| **sampling_from_probs** | cuda | cuda | cuda | cuda | cuda | cuda | cuda | cuda |
| **sampling_from_logits** | cuda | cuda | cuda | cuda | cuda | cuda | cuda | cuda |
| **top_k_sampling_from_probs** | cuda | cuda | cuda | cuda | cuda | cuda | cuda | cuda |
| **top_p_sampling_from_probs** | cuda | cuda | cuda | cuda | cuda | cuda | cuda | cuda |
| **top_k_top_p_sampling_from_probs** | cuda | cuda | cuda | cuda | cuda | cuda | cuda | cuda |
| **top_k_top_p_sampling_from_logits** | cuda | cuda | cuda | cuda | cuda | cuda | cuda | cuda |
| **min_p_sampling_from_probs** | cuda | cuda | cuda | cuda | cuda | cuda | cuda | cuda |
| **top_k_renorm_probs** | cuda | cuda | cuda | cuda | cuda | cuda | cuda | cuda |
| **top_p_renorm_probs** | cuda | cuda | cuda | cuda | cuda | cuda | cuda | cuda |
| **top_k_mask_logits** | cuda | cuda | cuda | cuda | cuda | cuda | cuda | cuda |
| **chain_speculative_sampling** | cuda | cuda | cuda | cuda | cuda | cuda | cuda | cuda |
| **top_k** | cuda | cuda | cuda | cuda | cuda | cuda | cuda | cuda |
| **top_k_page_table_transform** | cuda | cuda | cuda | cuda | cuda | cuda | cuda | cuda |
| **top_k_ragged_transform** | cuda | cuda | cuda | cuda | cuda | cuda | cuda | cuda |
| **apply_rope** | cuda | cuda | cuda | cuda | cuda | cuda | cuda | cuda |
| **apply_rope_pos_ids** | cuda | cuda | cuda | cuda | cuda | cuda | cuda | cuda |
| **apply_llama31_rope** | cuda | cuda | cuda | cuda | cuda | cuda | cuda | cuda |
| **apply_llama31_rope_pos_ids** | cuda | cuda | cuda | cuda | cuda | cuda | cuda | cuda |
| **apply_rope_with_cos_sin_cache** | cuda | cuda | cuda | cuda | cuda | cuda | cuda | cuda |
| **mla_rope_quantize_fp8** |  |  |  | cuda | cuda | cuda | cuda | cuda |
| **rope_quantize_fp8** |  |  |  | cuda | cuda | cuda | cuda | cuda |
| **rope_quantize_fp8_append_paged_kv_cache** |  |  |  | cuda | cuda | cuda | cuda | cuda |
| **selective_state_update** | flashinfer, triton | flashinfer, triton | flashinfer, triton | flashinfer, triton | flashinfer, triton | flashinfer, triton | flashinfer, triton | flashinfer, triton |

Backend Legend:
- fa2: FlashAttention2
- fa2_tc: FlashAttention2 (with Tensor Cores for `BatchDecodeWithPagedKVCacheWrapper`)
- fa3: FlashAttention-3
- cublas: cuBLAS
- cudnn: cuDNN (via wrapper API)
- cudnn-native: cuDNN (direct API call)
- cutlass: CUTLASS
- trtllm: TensorRT-LLM
- trtllm-gen: TensorRT-LLM
- trtllm-native: TensorRT-LLM (out-of-wrapper)
- cuda: FlashInfer CUDA kernels
- cute-dsl: FlashInfer CuTe-DSL kernels (Blackwell SM10.0+)
- moe_a2a: MoE All-to-All communication (requires mpirun, Blackwell SM10.0+ with MNNVL)
- triton: Triton reference kernels (used for Mamba selective_state_update)
