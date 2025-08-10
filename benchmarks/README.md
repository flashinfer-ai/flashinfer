# FlashInfer Perf Benchmarking Framework -- flashinfer_benchmark.py

A comprehensive testing and benchmarking framework for FlashInfer's kernels.

The aim of `flashinfer_benchmark.py` is to provide a single framework for benchmarking any FlashInfer kernel and replace standalone benchmarking scripts. Current support surface includes batched prefill & decode, and FP8 gemm.

## Overview

This framework provides tools to:
- Benchmark different attention implementations (FlashAttention2/3, cuDNN, CUTLASS, TensorRT-LLM)
- Benchmark the GEMM performance.
- Compare performance across different configurations
- Batch performance test multiple attention test cases
- Generate detailed performance reports

Currently supports testing:
- `BatchDecodeWithPagedKVCacheWrapper` - Decode attention with paged KV cache.
- `BatchPrefillWithPagedKVCacheWrapper` - Prefill attention with paged KV cache.
- `BatchPrefillWithRaggedKVCacheWrapper` - Prefill attention with ragged KV cache.
- `BatchMLAPagedAttentionWrapper` - MLA attention proposed in DeepSeek series of models.
- `gemm_fp8_nt_groupwise` - GEMM with FP8 data types using groupwise scaling.
- `group_gemm_fp8_nt_groupwise` - Group GEMM with FP8 data types using groupwise scaling.
- `bmm_fp8` - Batched matrix multiplication with FP8 inputs.
- `mm_fp4` - Maxtrix multiplication with NVFP4 inputs.

Support surface will expand to other operations such as MLA or non-attention operations in the future.
## Quick Start

### Single Test Run
Example commands
```bash
# Test prefill attention with paged KV cache
python3 flashinfer_benchmark.py \
    --routine BatchPrefillWithPagedKVCacheWrapper \
    --backends fa2 cudnn trtllm-gen \
    --page_size 16 \
    --batch_size 16 \
    --s_qo 4096 \
    --s_kv 4096 \
    --num_qo_heads 64 \
    --num_kv_heads 8 \
    --head_dim_qk 128 \
    --head_dim_vo 128 \
    --random_actual_seq_len \
    --verbose \
    --refcheck \
    --causal \
    --no_cuda_graph

# Test prefill attention with ragged KV cache
python3 flashinfer_benchmark.py \
    --routine BatchPrefillWithRaggedKVCacheWrapper \
    --backends fa2 cudnn cutlass \
    --batch_size 16 \
    --s_qo 4096 \
    --s_kv 4096 \
    --num_qo_heads 128 \
    --num_kv_heads 128 \
    --head_dim_qk 192 \
    --head_dim_vo 128 \
    --verbose \
    --refcheck \
    --causal \
    --no_cuda_graph

# Test decode attention with paged KV cache
python3 flashinfer_benchmark.py \
    --routine BatchDecodeWithPagedKVCacheWrapper \
    --backends fa2 fa2_tc trtllm-gen cudnn \
    --page_size 16 \
    --batch_size 16 \
    --s_qo 1 \
    --s_kv 8192 \
    --num_qo_heads 64 \
    --num_kv_heads 8 \
    --head_dim_qk 128 \
    --head_dim_vo 128 \
    --random_actual_seq_len \
    --verbose \
    --refcheck

# FP8 GEMM
python3 flashinfer_benchmark.py \
    --routine gemm_fp8_nt_groupwise \
    --m 8192 \
    --n 4096 \
    --k 16384 \
    --mma_sm 2 \
    --refcheck \
    -vv

# Group FP8 GEMM
python3 flashinfer_benchmark.py \
    --routine group_gemm_fp8_nt_groupwise \
    --m 8192 \
    --n 4096 \
    --k 16384 \
    --mma_sm 2 \
    --group_size 2 \
    --no_cuda_graph \
    --scale_major_mode K \
    --refcheck \
    -vv
```

### Batch Testing

Run multiple tests from a file and save results:
```bash
python3 flashinfer_benchmark.py --testlist samples/sample_testlist.txt --output_path sample_testlist_output.csv
```

The output CSV will contain detailed metrics including:
- Median execution time
- Standard deviation
- TFLOPS/sec
- Memory throughput (TB/sec)

## Command Line Arguments
### General Flags
| Flag                     | Description                                                                                                 |
|--------------------------|-------------------------------------------------------------------------------------------------------------|
| `--routine`              | Test routine to run: `BatchDecodeWithPagedKVCacheWrapper`, `BatchPrefillWithPagedKVCacheWrapper`, `BatchPrefillWithRaggedKVCacheWrapper`, `BatchMLAPagedAttentionWrapper`, `gemm_fp8_nt_groupwise`, `group_gemm_fp8_nt_groupwise`, `bmm_fp8`, `mm_fp4` |
| `--num_iters`            | Number of iterations for performance measurement                                                           |
| `--dry_run_iters`        | Number of warmup iterations                                                                                |
| `--no_cuda_graph`        | Disable CUDA graph to execute kernels outside of the graph.                                                |
| `--allow_output_mismatch`| Continue testing even if outputs don't match between backends                                              |
| `--refcheck`             | Verify outputs match between different backends                                                            |
| `--random_seed`          | Random seed for reproducibility                                                                            |
| `--output_path`          | Path to save CSV results                                                                                   |
| `--testlist`             | Path to a file containing a list of test cases to run in batch mode                                        |
| `--verbose`, `-v`        | Print additional information (can be used multiple times for more verbosity, e.g. `-vv`)                   |

### Attention Flags
| Flag                     | Description                                                                                                 |
|--------------------------|-------------------------------------------------------------------------------------------------------------|
| `--backends`             | List of backends to test: fa2, fa2_tc, fa3, cudnn, cutlass, trtllm, trtllm-gen, cublas                      |
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

## Tester Attention Backend Support Matrix
The following support surface applies to attention operations in `flashinfer_benchmark.py`
| Backend  | Decode Paged | Prefill Paged | Prefill Ragged | FP8  | Notes                                    |
|----------|-------------|---------------|----------------|------|------------------------------------------|
| fa2      | ✓           | ✓             | ✓              | ✗    | Does not support GQA ratio of 5          |
| fa2_tc   | ✓           | ✗             | ✗              | ✗    | Uses tensor cores                        |
| fa3      | ✗           | ✓             | ✓              | ✗    | Hopper Only      |
| cudnn    | ✓           | ✓*            | ✓*             | ✗    | *Requires specific head dims (192 or 128) |
| cutlass  | ✗           | ✗             | ✓              | ✗    |                                          |
| trtllm   | ✓           | ✗             | ✗              | ✗    |                                          |

Notes:
- CUDA graph support is only stable with BatchDecodeWithPagedKVCacheWrapper. For BatchPrefillWithPagedKVCacheWrapper and BatchPrefillWithRaggedKVCacheWrapper, it is recommended that `--no_cuda_graph` is used.
- cudnn, cutlass, and trtllm backends are supported on [CUDA Compute Capability 10.0 GPUs](https://developer.nvidia.com/cuda-gpus) only.
- fa3 is supported on [CUDA Compute Capability 9.0 GPUs](https://developer.nvidia.com/cuda-gpus) only.

## Example Outputs
```bash
$ python3 flashinfer_benchmark.py --routine BatchPrefillWithPagedKVCacheWrapper --backends fa2 cudnn trtllm-gen --page_size 16 --batch_size 16 --s_qo 1024 --s_kv 1024 --num_qo_heads 8 --num_kv_heads 8 --head_dim_qk 128 --head_dim_vo 128 --random_actual_seq_len --refcheck --causal --no_cuda_graph -vv
[INFO] args = Namespace(routine='BatchPrefillWithPagedKVCacheWrapper', no_cuda_graph=True, refcheck=True, allow_output_mismatch=False, random_seed=42, verbose=2, output_path=None, num_iters=30, dry_run_iters=5, backends=['fa2', 'cudnn', 'trtllm-gen'], page_size=16, batch_size=16, s_qo=1024, s_kv=1024, num_qo_heads=8, num_kv_heads=8, head_dim_qk=128, head_dim_vo=128, head_dim_ckv=None, head_dim_kpe=None, q_dtype='bfloat16', kv_dtype='bfloat16', causal=True, random_actual_seq_len=True)
[INFO] Running testBatchPrefillWithPagedKVCacheWrapper
[INFO] FlashInfer version: 0.2.8
[VVERBOSE] gpu_name = 'NVIDIA_B200'
[VERBOSE] Average actual seq len: 327
[VVERBOSE] actual_seq_lens_q.flatten() = tensor([103, 436, 861, 271, 107,  72, 701,  21, 615, 122, 467, 215, 331, 459,
         88, 373], dtype=torch.int32)
[VVERBOSE] q.shape = torch.Size([5242, 8, 128])
[VVERBOSE] num_pages_per_seq = 64
[VVERBOSE] total_num_pages = 1024
[VVERBOSE] kv_cache.shape = torch.Size([1024, 2, 8, 16, 128])
[VVERBOSE] kv_cache.stride() = (32768, 16384, 128, 1024, 1)
[VVERBOSE] block_tables.shape = torch.Size([16, 64])
[VVERBOSE] qo_indptr.shape = torch.Size([17])
[VVERBOSE] qo_indptr.dtype = torch.int32
[VVERBOSE] kv_indptr.shape = torch.Size([17])
[VVERBOSE] kv_indices.shape = torch.Size([335])
[VVERBOSE] kv_last_page_len.shape = torch.Size([16])
[VVERBOSE] scale = 0.08838834764831843ze([16])
[VVERBOSE] scale = 0.08838834764831843
[PERF] fa2       :: median time 0.059 ms; std 0.001 ms; achieved tflops 91.278 TFLOPs/sec; achieved tb_per_sec 0.723 TB/sec
[PERF] cudnn     :: median time 0.047 ms; std 0.002 ms; achieved tflops 116.377 TFLOPs/sec; achieved tb_per_sec 0.921 TB/sec
[PERF] trtllm-gen:: median time 0.051 ms; std 0.002 ms; achieved tflops 105.873 TFLOPs/sec; achieved tb_per_sec 0.838 TB/sec

$ python3 flashinfer_benchmark.py --routine BatchPrefillWithRaggedKVCacheWrapper --backends fa2 cudnn cutlass --batch_size 16 --s_qo 1024 --s_kv 1024 --num_qo_heads 128 --num_kv_heads 128 --head_dim_qk 192 --head_dim_vo 128  --refcheck --causal --no_cuda_graph -vv
INFO] args = Namespace(routine='BatchPrefillWithRaggedKVCacheWrapper', no_cuda_graph=True, refcheck=True, allow_output_mismatch=False, random_seed=42, verbose=2, output_path=None, num_iters=30, dry_run_iters=5, backends=['fa2', 'cudnn', 'cutlass'], page_size=0, batch_size=16, s_qo=1024, s_kv=1024, num_qo_heads=128, num_kv_heads=128, head_dim_qk=192, head_dim_vo=128, head_dim_ckv=None, head_dim_kpe=None, q_dtype='bfloat16', kv_dtype='bfloat16', causal=True, random_actual_seq_len=False)
[INFO] Running testBatchPrefillWithRaggedKVCacheWrapper
[INFO] FlashInfer version: 0.2.8
[VVERBOSE] gpu_name = 'NVIDIA_B200'
[VERBOSE] Average actual seq len: 1024
[VVERBOSE] actual_seq_lens_q.flatten() = tensor([1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024,
        1024, 1024, 1024, 1024], dtype=torch.int32)
[VVERBOSE] q.shape = torch.Size([16384, 128, 192])
[VVERBOSE] k.shape = torch.Size([16384, 128, 192])
[VVERBOSE] v.shape = torch.Size([16384, 128, 128])
[VVERBOSE] qo_indptr.shape = torch.Size([17])
[VVERBOSE] kv_indptr.shape = torch.Size([17])
[VVERBOSE] scale = 0.07216878364870323
[PERF] fa2       :: median time 2.252 ms; std 0.038 ms; achieved tflops 305.092 TFLOPs/sec; achieved tb_per_sec 1.192 TB/sec
[PERF] cudnn     :: median time 1.178 ms; std 0.054 ms; achieved tflops 583.460 TFLOPs/sec; achieved tb_per_sec 2.279 TB/sec
[PERF] cutlass   :: median time 1.494 ms; std 0.034 ms; achieved tflops 459.866 TFLOPs/sec; achieved tb_per_sec 1.796 TB/sec

$ python3 flashinfer_benchmark.py --routine BatchDecodeWithPagedKVCacheWrapper --backends fa2 fa2_tc trtllm-gen cudnn --page_size 16 --batch_size 32 --s_qo 1 --s_kv 8192 --num_qo_heads 64 --num_kv_heads 8 --head_dim_qk 128 --head_dim_vo 128 --refcheck -vv
INFO] args = Namespace(routine='BatchDecodeWithPagedKVCacheWrapper', no_cuda_graph=False, refcheck=True, allow_output_mismatch=False, random_seed=42, verbose=2, output_path=None, num_iters=30, dry_run_iters=5, backends=['fa2', 'fa2_tc', 'trtllm-gen', 'cudnn'], page_size=16, batch_size=32, s_qo=1, s_kv=8192, num_qo_heads=64, num_kv_heads=8, head_dim_qk=128, head_dim_vo=128, head_dim_ckv=None, head_dim_kpe=None, q_dtype='bfloat16', kv_dtype='bfloat16', causal=False, random_actual_seq_len=False)
[INFO] Running testBatchDecodeWithPagedKVCacheWrapper
[INFO] FlashInfer version: 0.2.8
[VVERBOSE] gpu_name = 'NVIDIA_B200'
[VERBOSE] Average actual seq len: 8192
[VVERBOSE] actual_seq_lens_kv.flatten() = tensor([8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192,
        8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192,
        8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192], device='cuda:0',
       dtype=torch.int32)
[VVERBOSE] q.shape = torch.Size([32, 64, 128])
[VVERBOSE] num_pages_per_seq = 512
[VVERBOSE] total_num_pages = 16384
[VVERBOSE] kv_cache.shape = torch.Size([16384, 2, 8, 16, 128])
[VVERBOSE] kv_cache.stride() = (32768, 16384, 128, 1024, 1)
[VVERBOSE] block_tables.shape = torch.Size([32, 512])
[VVERBOSE] kv_indptr.shape = torch.Size([33])
[VVERBOSE] kv_indices.shape = torch.Size([16384])
[VVERBOSE] kv_last_page_len.shape = torch.Size([32])
[VVERBOSE] scale = 0.08838834764831843
[PERF] fa2       :: median time 0.712 ms; std 0.000 ms; achieved tflops 12.070 TFLOPs/sec; achieved tb_per_sec 1.510 TB/sec
[PERF] fa2_tc    :: median time 0.187 ms; std 0.002 ms; achieved tflops 46.022 TFLOPs/sec; achieved tb_per_sec 5.758 TB/sec
[PERF] trtllm-gen:: median time 0.157 ms; std 0.001 ms; achieved tflops 54.581 TFLOPs/sec; achieved tb_per_sec 6.829 TB/sec
[PERF] cudnn     :: median time 0.170 ms; std 0.000 ms; achieved tflops 50.535 TFLOPs/sec; achieved tb_per_sec 6.323 TB/sec
