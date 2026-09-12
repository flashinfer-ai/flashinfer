# FlashInfer Perf Benchmarking Framework -- `flashinfer_benchmark.py`

The aim of `flashinfer_benchmark.py` is to provide a single framework for benchmarking any FlashInfer kernel and replace standalone benchmarking scripts.

`bench_recurrent_kda_prefill.py --case-set h12` runs the six Kimi-K3 TP8 H12
public-API cases. `reduce_kda_h12.py` combines successful SM100a and SM103a
result files without producing a cross-shape aggregate. The benchmark defaults
to the natural device/shape dispatcher and records its resolved module; use
`--candidate-route nonpersistent` for a B200 direct-family route A/B.
`bench_recurrent_kda_prefill.py --case-set small_bh` runs the four fixed-layout
small-BH cases through the same cold-L2 CUPTI path.
`bench_recurrent_kda_prefill.py --case-set production --backend cake` runs the
complete 29-shape fixed/packed inference portfolio. Its JSON records the logical
route, every physical Cake module used (including BT16 prepare plus chain), and
the explicit per-shape dry/repeat iteration budget. Large state shapes reduce
the sample count to stay within the rotating-state capacity and set
`timing_iteration_budget.low_sample_count` when fewer than ten measured
iterations fit. `--dry-run-iters` and `--repeat-iters` request fixed iteration
counts; they are not duration targets.

## Overview

This framework provides tools to:
- Benchmark FlashInfer's Attention, GEMM, MOE, Norm, Quantization, Sampling, RoPE, Mamba, GDN, and KDA API performance from different kernel backends such as FlashAttention2/3, cuDNN, cuBLAS, CUTLASS, PrimTS, CuTe-DSL, TensorRT-LLM, and Triton
- Compare performance across different configurations
- Batch performance test multiple test cases

Currently supports testing attention, gemm, fused MOE, normalization, quantization, sampling, RoPE, Mamba, GDN (Gated Delta Net), and KDA APIs:
- Attention:
    - `BatchDecodeWithPagedKVCacheWrapper` - Decode attention with paged KV cache.
        - Also supports computationally similar `cudnn_batch_decode_with_kv_cache` and `trtllm_batch_decode_with_kv_cache`.
        - Speculative decode is supported by setting `--s_qo > 1` (subject to backend limitations noted below).
    - `BatchPrefillWithPagedKVCacheWrapper` - Prefill attention with paged KV cache.
        - Also supports computationally similar `cudnn_batch_prefill_with_kv_cache` and  `trtllm_batch_context_with_kv_cache`.
    - `BatchPrefillWithRaggedKVCacheWrapper` - Prefill attention with ragged KV cache.
        - Also supports computationally similar `cudnn_batch_prefill_with_kv_cache` (cudnn-native) and  `trtllm_ragged_attention_deepseek`.
    - `BatchMLAPagedAttentionWrapper` - MLA attention proposed in DeepSeek series of models.
        - Also supports computationally similar `trtllm_batch_decode_with_kv_cache_mla` (trtllm-native) and CuTe DSL MLA decode kernel (cute-dsl, SM100+).
    - `trtllm_batch_decode_sparse_mla_dsv4` - DeepSeek-V4 sparse MLA using the public TRTLLM-GEN API on SM100/SM103. Supports varlen prefill-style query lengths, causal SWA and compressed-cache sparse tables, FP8/BF16 inputs, sampled FP32 reference checking, and hot-path Q-tile selector benchmarks.
    - All four wrapper attention routines above accept `--backends prims-ts` on SM100/SM103. The standalone `trtllm_batch_decode_sparse_mla_dsv4` routine supports only `trtllm-gen`.
- GEMM:
    - `gemm_fp8_nt_groupwise` - GEMM with FP8 data types using groupwise scaling.
    - `group_gemm_fp8_nt_groupwise` - Group GEMM with FP8 data types using groupwise scaling.
    - `bmm_fp8` - Batched matrix multiplication with FP8 inputs.
    - `mm_mxfp8` - Dense MXFP8 matrix multiplication.
    - `mm_fp8` - Matrix multiplication with FP8 inputs using the trtllm-gen low-latency GEMM (Blackwell SM10.0+, small-M optimized, pre-shuffled weights).
    - `mm_fp4` - Matrix multiplication with NVFP4 inputs.
    - `mm_bf16` - Matrix multiplication with BF16 inputs (Blackwell SM10.0+).
    - `bmm_bf16` - Batched matrix multiplication with BF16 inputs (Blackwell SM10.0+).
- MOE:
    - `trtllm_fp4_block_scale_moe` - MOE with FP4 quantized weights and block-wise scaling.
    - `trtllm_fp8_block_scale_moe` - MOE with FP8 quantized weights and block-wise scaling.
    - `trtllm_fp8_per_tensor_scale_moe` - MOE with FP8 quantized weights and per-tensor scaling.
    - `cutlass_fused_moe` - CUTLASS fused MoE (base/fp8/nvfp4 variants with optional TP/EP)
    - `unified_moe` - Unified MoE API comparison between the CUTLASS and cuTile backends. It supports BF16 and NVFP4 W4A4 with gated SwiGLU, SwiGLU-Step, GeGLU, GeGLU-Tanh, and SiTU or non-gated GELU, ReLU, SiLU, ReLU2, and Identity; filters unsupported backends at runtime; and can autotune each backend independently.
- MOE Communication:
    - `moe_a2a_dispatch_combine` - MoE All-to-All dispatch + combine benchmark for multi-GPU expert-parallel inference. Requires `mpirun` for multi-GPU execution. Supports optional quantization (FP8, NVFP4, FP8 block-scale) and real MoE kernel computation.
- AllReduce Communication:
    - `allreduce_fusion` - AllReduce fusion benchmark for multi-GPU inference. Requires `mpirun` for multi-GPU execution. Supports TRTLLM and TRTLLM MNNVL backends with multiple fusion patterns (plain allreduce, allreduce + residual + RMSNorm).
- Norm:
    - `rmsnorm` - Root Mean Square Layer Normalization.
    - `fused_add_rmsnorm` - Fused residual add + RMSNorm.
    - `gemma_rmsnorm` - Gemma-style RMSNorm using `(weight + 1)`.
    - `gemma_fused_add_rmsnorm` - Gemma-style fused residual add + RMSNorm.
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
- GDN (Gated Delta Net linear attention, SM90+):
    - `gated_delta_rule_decode` - Single-token (T=1) gated delta rule decode. `--state_layout` selects between `gated_delta_rule_decode_pretranspose` ([B, HV, V, K] state, default) and `gated_delta_rule_decode` ([B, HV, K, V] state). `--state_dtype bfloat16` selects the BF16 state kernels (head_size=128, pretranspose only). Backends: `flashinfer` (CuTe-DSL) and `triton` (reference).
    - `gated_delta_rule_mtp` - Multi-token (T>=2) gated delta rule for speculative-decoding verification, with a state pool + indices. `--state_dtype float32` uses `gated_delta_rule_mtp`; `--state_dtype bfloat16` uses the BF16 MTP kernel via `gated_delta_rule_decode_pretranspose`. Backends: `flashinfer`, `triton`.
    - `chunk_gated_delta_rule` - Chunked GDN prefill over varlen sequences (uniform per-sequence length `--s_qo`). Backends: `flashinfer` (SM90 C++ / SM100 CuTe-DSL) and `fla` (flash-linear-attention Triton baseline, perf-only).
- KDA (SM120a):
    - `recurrent_kda_prefill` - Ordinary multi-token recurrent KDA prefill with fixed or packed inputs. Backends: `flashinfer` (automatic variant policy), `flashinfer-decomp`, `flashinfer-fused`, and optional external `cutekda` / `flash-kda` baselines.

## Quick Start
### Single Test Run
A test case is generally invoked as `python3 flashinfer_benchmark.py --routine <routine_name> <flags>`.

The unified MoE comparison runs both backends from the same routing, activation,
and weight inputs. This example uses the Nemotron-3.5-Lightning MoE shape:

```bash
python3 flashinfer_benchmark.py --routine unified_moe --backends cutlass cutile --quant-variant bf16 --num_tokens 128 --hidden_size 2688 --intermediate_size 1856 --num_experts 128 --top_k 6 --activation-type Relu2 --input_dtype bfloat16 --autotune
```

CUDA graph timing is enabled by default and captures one MoE invocation per
graph replay with cold-L2 benchmarking enabled; pass `--no_cuda_graph` for eager
timing. Without `--autotune`, results are named `cutlass` and `cutile`; autotuned
results use `cutlass_autotune` and `cutile_autotune`.

Representative Qwen3.6 and Nemotron cases are in `samples/sample_testlist.txt`.

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
| `--backends`             | Space-separated list of backends to test, e.g. fa2, fa2_tc, fa3, auto, cudnn, cudnn-native, cutlass, trtllm, trtllm-gen, trtllm-native, prims-ts, cute-dsl, cute-dsl-prims, cublas, trtllm_low_latency. (`prims_ts` aliases `prims-ts`; `auto` support is routine-dependent.)|

### Attention Flags
| Flag                     | Description                                                                                                 |
|--------------------------|-------------------------------------------------------------------------------------------------------------|
| `--page_size`            | Page size for paged attention. Required for paged attention tests.                                          |
| `--batch_size`           | Number of sequences to process in parallel                                                                  |
| `--s_qo`                 | Query/output sequence length. For decode, `1` is standard decode and `>1` enables speculative decode on supported backends. |
| `--s_kv`                 | Key/value sequence length (context length)                                                                  |
| `--num_qo_heads`         | Number of query/output attention heads                                                                      |
| `--num_kv_heads`         | Number of key/value attention heads                                                                         |
| `--head_dim_qk`          | Head dimension for Q/K. Backend-dependent; PrimTS supports 64/128/256 for FMHA decode and 128/256 for FMHA context. |
| `--head_dim_vo`          | Head dimension for V/O. Usually equals head_dim_qk.                                                        |
| `--head_dim_ckv`         | Head dimension for C/K/V (MLA attention).                                                                  |
| `--head_dim_kpe`         | Head dimension for KPE (MLA attention).                                                                    |
| `--q_dtype`              | Data type for the query tensor. Default: bfloat16. Supports float16, bfloat16, fp8_e4m3, and fp8_e5m2 where the selected backend permits them. |
| `--kv_dtype`             | Data type for the key and value tensors. Default: bfloat16. Supports float16, bfloat16, fp8_e4m3, and fp8_e5m2 where the selected backend permits them. |
| `--out_dtype`            | Data type for the output tensor. Default: same as q_dtype. Backend-dependent; PrimTS context accepts bfloat16, float16, or fp8_e4m3, while PrimTS FP8 decode accepts float16 or fp8_e4m3. FP8 ragged comparisons with non-PrimTS backends require bfloat16 or float16. |
| `--causal`               | Use causal attention masking for context/prefill. Multi-query FMHA and MLA decode use bottom-right causal masking automatically. |
| `--random_actual_seq_len`| Use random sequence lengths up to max length. If False, use max length.                                    |
| `--swa_topk`             | DSV4 sparse MLA only: sliding-window segment width. Must be 128.                                           |
| `--compressed_topk`      | DSV4 sparse MLA only: maximum compressed-cache rows selected per query. Default: 1920.                    |
| `--compressed_kv_len`    | DSV4 sparse MLA only: compressed-cache rows per request. Default: `ceil(s_kv / 4)`.                       |
| `--compressed_page_size` | DSV4 sparse MLA only: compressed-cache page size. Default: 64.                                             |
| `--kv_layout`            | DSV4 sparse MLA only: `HND` (default) or `NHD` for both KV-cache pools.                                    |

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
| `--autotune`             | Enable autotune for supported operation (`mm_fp4`, `bmm_fp8`, `mm_fp8`, `bmm_mxfp8`, `mm_mxfp8`, `mm_bf16`, `bmm_bf16` routines) |
| `--bias`                 | Use bias for `mm_bf16` (Enabled for TGV backend)                                                           |

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
| `--activation-type`      | Activation function: `Swiglu` (default), `Geglu`, `SwigluStep` (clipped SwiGLU, limit=7.0), `Relu2`, etc.  |
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
| `--use_lora`             | Carry a per-token int32 LoRA adapter ID through dispatch as an extra payload.                                                                                                                  |

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

# Multi-tenant LoRA: carry per-token adapter ID through dispatch
mpirun -np 8 python benchmarks/flashinfer_benchmark.py \
    --routine moe_a2a_dispatch_combine \
    --num_tokens 2048 --hidden_size 7168 --num_experts 256 --top_k 8 \
    --use_lora --validate
```

### AllReduce Communication Flags (allreduce_fusion)
The `allreduce_fusion` routine benchmarks AllReduce fusion operations for multi-GPU inference. It must be launched with `mpirun`. Both oneshot and twoshot strategies are benchmarked automatically and reported side by side.

| Flag                     | Description                                                                                                 |
|--------------------------|-------------------------------------------------------------------------------------------------------------|
| `--num_tokens`           | Number of tokens (rows) in the input tensor. Default: 64                                                   |
| `--hidden_size`          | Hidden dimension size. Default: 4096                                                                       |
| `--input_dtype`          | Data type for input tensors: `bfloat16` (default) or `float16`                                             |
| `--ar_backend`           | AllReduce backend: `auto` (default), `trtllm`, or `mnnvl`. `auto` uses heuristic                          |
| `--pattern`              | Fusion pattern: `allreduce` (default) or `ar_residual_rmsnorm` (AllReduce + Residual + RMSNorm)            |
| `--validate`             | Run correctness validation before benchmarking                                                             |

**Launch Examples:**
```bash
# Basic allreduce with auto backend
mpirun -np 8 python benchmarks/flashinfer_benchmark.py \
    --routine allreduce_fusion \
    --num_tokens 64 --hidden_size 4096

# With specific backend
mpirun -np 8 python benchmarks/flashinfer_benchmark.py \
    --routine allreduce_fusion \
    --num_tokens 64 --hidden_size 4096 \
    --ar_backend mnnvl

# AllReduce + Residual + RMSNorm fusion
mpirun -np 8 python benchmarks/flashinfer_benchmark.py \
    --routine allreduce_fusion \
    --num_tokens 64 --hidden_size 4096 \
    --pattern ar_residual_rmsnorm

# With validation
mpirun -np 8 python benchmarks/flashinfer_benchmark.py \
    --routine allreduce_fusion \
    --num_tokens 64 --hidden_size 4096 \
    --validate
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
| `--backends`             | Backend to test. Defaults to `cute-dsl` for rmsnorm/rmsnorm_quant/fused_add_rmsnorm/fused_add_rmsnorm_quant/gemma_rmsnorm/gemma_fused_add_rmsnorm/rmsnorm_fp4quant/add_rmsnorm_fp4quant (CuTe-DSL kernels) and `cuda` otherwise. Pass `--backends cuda` to force the CUDA JIT fallback (set `FLASHINFER_USE_CUDA_NORM=1` to actually run the CUDA path). |

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
| `--sf_layout`            | Scale factor layout for FP4 quantization: `128x4` (default), `8x4`, or `linear`                             |
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

### GDN Flags
Applies to `gated_delta_rule_decode`, `gated_delta_rule_mtp`, and `chunk_gated_delta_rule` (SM90+).

| Flag                          | Description                                                                                                 |
|-------------------------------|-------------------------------------------------------------------------------------------------------------|
| `--batch_size`                | Decode/MTP: number of concurrent requests. Prefill: number of sequences                                    |
| `--num_q_heads`               | Number of query heads. Default: 16                                                                         |
| `--num_k_heads`               | Number of key heads. Default: 16                                                                           |
| `--num_v_heads`               | Number of value heads (GVA when > `num_q_heads`). Default: 32                                              |
| `--head_size`                 | Head dimension (K = V = head_size). Default: 128                                                           |
| `--input_dtype`               | Data type for q/k/v/a/b tensors: `bfloat16` (default) or `float16`                                         |
| `--state_dtype`               | Recurrent state dtype: `float32` (default) or `bfloat16` (BF16 state kernels; decode/MTP, head_size=128, pretranspose) |
| `--state_layout`              | Decode only: `pretranspose` ([B, HV, V, K], default) or `nontranspose` ([B, HV, K, V])                     |
| `--pool_mode`                 | `single` (default, read == write slots) or `split` (pool of 2B; reads slots [0..B), writes [B..2B))        |
| `--seq_len`                   | MTP only: tokens per request (>= 2). Default: 2                                                            |
| `--s_qo`                      | Prefill only: per-sequence length (uniform). Default: 2048                                                 |
| `--update_state`              | MTP only: write the final state back (`disable_state_update=False`). BF16 state always updates in-place    |
| `--cache_intermediate_states` | MTP with `float32` state only: cache per-token intermediate states                                         |
| `--no_qk_l2norm`              | Decode/MTP: disable in-kernel Q/K L2 normalization                                                         |
| `--backends`                  | Decode/MTP: `flashinfer` (default), `triton`. Prefill: `flashinfer` (default), `fla` (requires `pip install flash-linear-attention`; perf-only, excluded from refcheck) |

Notes:
- Refcheck compares against the torch reference in `tests/gdn/reference_delta_rule.py`.
- Prefill pre-L2-normalizes k and calls the kernel with `use_qk_l2norm_in_kernel=False` so the kernel and reference see identical inputs.

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
- prims-ts: Experimental task-scheduled attention (SM100/SM103)
-->
| Routine | 7.5 | 8.0 | 8.6 | 8.9 | 9.0 | 10.0 | 10.3 | 12.0 |
|---------|-----|-----|-----|-----|-----|-------|-------|-------|
| **BatchDecodeWithPagedKVCacheWrapper** | fa2 | fa2, fa2_tc, cudnn | fa2, fa2_tc, cudnn | fa2, fa2_tc, cudnn | fa2, fa2_tc, cudnn | fa2, fa2_tc, cudnn, trtllm-gen, trtllm-native, prims-ts | fa2, fa2_tc, cudnn, trtllm-gen, trtllm-native, prims-ts | fa2, fa2_tc, cudnn |
| **BatchPrefillWithPagedKVCacheWrapper** |  | fa2, cudnn, cudnn-native | fa2, cudnn, cudnn-native | fa2, cudnn, cudnn-native | fa2, fa3, cudnn, cudnn-native | fa2, cudnn, cudnn-native, trtllm-gen, trtllm-native, prims-ts | fa2, cudnn, cudnn-native, trtllm-gen, trtllm-native, prims-ts | fa2, cudnn, cudnn-native, trtllm-fmha-v2, cute-dsl-prims |
| **BatchPrefillWithRaggedKVCacheWrapper** |  | fa2, cudnn, cudnn-native | fa2, cudnn, cudnn-native | fa2, cudnn, cudnn-native | fa2, fa3, cudnn, cudnn-native | fa2, cudnn, cudnn-native, cutlass, trtllm-native, prims-ts | fa2, cudnn, cudnn-native, cutlass, trtllm-native, prims-ts | fa2, cudnn, cudnn-native, trtllm-fmha-v2, cute-dsl-prims |
| **BatchMLAPagedAttentionWrapper** |  | fa2 | fa2 | fa2 | fa2, fa3 | fa2, cutlass, trtllm-native, cute-dsl, prims-ts | fa2, cutlass, trtllm-native, prims-ts | fa2 |
| **trtllm_batch_decode_sparse_mla_dsv4** |  |  |  |  |  | trtllm-gen | trtllm-gen |  |
| **gemm_fp8_nt_groupwise** |  |  |  |  |  | cutlass | cutlass |  |
| **group_gemm_fp8_nt_groupwise** |  |  |  |  |  | cutlass | cutlass |  |
| **bmm_fp8** |  |  |  | cudnn, cublas | cudnn, cublas | cudnn, cublas, cutlass | cudnn, cublas, cutlass | cudnn, cublas |
| **mm_fp8** |  |  |  |  |  | trtllm_low_latency | trtllm_low_latency |  |
| **mm_fp4** |  |  |  |  |  | cudnn, trtllm, cutlass | cudnn, trtllm, cutlass | cudnn |
| **mm_bf16** |  |  |  |  |  | cudnn, cutlass, tgv | cudnn, cutlass, tgv |  |
| **bmm_bf16** |  |  |  |  |  | cudnn, cutlass | cudnn, cutlass |  |
| **trtllm_fp4_block_scale_moe** |  |  |  |  |  | trtllm | trtllm |  |
| **trtllm_fp8_block_scale_moe** |  |  |  |  |  | trtllm | trtllm |  |
| **trtllm_fp8_per_tensor_scale_moe** |  |  |  |  |  | trtllm | trtllm |  |
| **cutlass_fused_moe** |  |  |  |  |  | cutlass | cutlass |  |
| **unified_moe** |  |  |  | cutlass, cutile (BF16) | cutlass, cutile (BF16) | cutlass | cutlass | cutlass, cutile |
| **moe_a2a_dispatch_combine** |  |  |  |  |  | moe_a2a | moe_a2a |  |
| **allreduce_fusion** |  |  |  |  |  | allreduce | allreduce |  |
| **rmsnorm** | cute-dsl | cute-dsl | cute-dsl | cute-dsl | cute-dsl | cute-dsl | cute-dsl | cute-dsl |
| **fused_add_rmsnorm** | cute-dsl | cute-dsl | cute-dsl | cute-dsl | cute-dsl | cute-dsl | cute-dsl | cute-dsl |
| **gemma_rmsnorm** | cute-dsl | cute-dsl | cute-dsl | cute-dsl | cute-dsl | cute-dsl | cute-dsl | cute-dsl |
| **gemma_fused_add_rmsnorm** | cute-dsl | cute-dsl | cute-dsl | cute-dsl | cute-dsl | cute-dsl | cute-dsl | cute-dsl |
| **rmsnorm_quant** | cute-dsl | cute-dsl | cute-dsl | cute-dsl | cute-dsl | cute-dsl | cute-dsl | cute-dsl |
| **fused_add_rmsnorm_quant** | cute-dsl | cute-dsl | cute-dsl | cute-dsl | cute-dsl | cute-dsl | cute-dsl | cute-dsl |
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
| **gated_delta_rule_decode** |  |  |  |  | flashinfer, triton | flashinfer, triton | flashinfer, triton | triton |
| **gated_delta_rule_mtp** |  |  |  |  | flashinfer, triton | flashinfer, triton | flashinfer, triton | triton |
| **chunk_gated_delta_rule** |  |  |  |  | flashinfer, fla | flashinfer, fla | flashinfer, fla |  |
| **recurrent_kda_prefill** |  |  |  |  |  |  |  | flashinfer, flashinfer-decomp, flashinfer-fused, cutekda, flash-kda |

Backend Legend:
- fa2: FlashAttention2
- fa2_tc: FlashAttention2 (with Tensor Cores for `BatchDecodeWithPagedKVCacheWrapper`)
- fa3: FlashAttention-3
- cublas: cuBLAS
- cudnn: cuDNN (via wrapper API)
- cudnn-native: cuDNN (direct API call)
- cutlass: CUTLASS
- tgv: TGV
- trtllm: TensorRT-LLM
- trtllm-gen: TensorRT-LLM
- trtllm-native: TensorRT-LLM (out-of-wrapper)
- prims-ts: Experimental task-scheduled attention kernels (Blackwell SM100/SM103)
- cuda: FlashInfer CUDA kernels
- cute-dsl: FlashInfer CuTe-DSL kernels (Blackwell SM10.0+)
- cute-dsl-prims: SM120 PRIMS FP8 batch-prefill kernels. Ragged inputs use
  packed NHD storage, while paged K/V uses HND storage. Paged attention accepts
  the standard combined cache or separate K/V pools without copying. The
  backend supports FP32 log2 LSE and requires `cutlass.experimental`.

SM120 PRIMS examples:

```bash
python benchmarks/flashinfer_benchmark.py \
  --routine BatchPrefillWithRaggedKVCacheWrapper \
  --backends cute-dsl-prims --batch_size 16 --s_qo 256 --s_kv 2048 \
  --num_qo_heads 32 --num_kv_heads 8 \
  --head_dim_qk 128 --head_dim_vo 128 \
  --q_dtype fp8_e4m3 --kv_dtype fp8_e4m3 \
  --out_dtype bfloat16 --causal --refcheck -vv

python benchmarks/flashinfer_benchmark.py \
  --routine BatchPrefillWithPagedKVCacheWrapper \
  --backends cute-dsl-prims --page_size 64 \
  --batch_size 16 --s_qo 256 --s_kv 2048 \
  --num_qo_heads 32 --num_kv_heads 8 \
  --head_dim_qk 128 --head_dim_vo 128 \
  --q_dtype fp8_e4m3 --kv_dtype fp8_e4m3 \
  --out_dtype bfloat16 --causal --refcheck -vv
```
- moe_a2a: MoE All-to-All communication (requires mpirun, Blackwell SM10.0+ with MNNVL)
- allreduce: AllReduce fusion communication (requires mpirun, Blackwell SM10.0+ with MNNVL)
- triton: Triton reference kernels (used for Mamba selective_state_update and GDN decode/MTP)
- fla: flash-linear-attention Triton kernels (GDN prefill baseline)
- flashinfer-decomp / flashinfer-fused: pinned SM120 KDA prefill variants
- cutekda / flash-kda: optional external SM120 KDA prefill baselines
