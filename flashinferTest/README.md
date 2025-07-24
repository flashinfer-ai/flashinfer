# FlashInfer Perf Benchmarking Framework -- flashinferTest.py

A comprehensive testing and benchmarking framework for FlashInfer's attention and GEMM kernels.

## Overview

This framework provides tools to:
- Benchmark different attention implementations (FlashAttention2/3, cuDNN, CUTLASS, TensorRT-LLM)
- Benchmark the GEMM performance.
- Compare performance across different configurations
- Batch performance test multiple attention test cases
- Generate detailed performance reports

Currently supports testing:
- `BatchDecodeWithPagedKVCacheWrapper` - Decode attention with paged KV cache
- `BatchPrefillWithPagedKVCacheWrapper` - Prefill attention with paged KV cache
- `BatchPrefillWithRaggedKVCacheWrapper` - Prefill attention with ragged KV cache
- `gemm_fp8_nt_groupwise` - GEMM with FP8 data types using groupwise scaling.
- `group_gemm_fp8_nt_groupwise` - Group GEMM with FP8 data types using groupwise scaling.

Support surface will expand to other operations such as MLA or non-attention operations in the future.
## Quick Start

### Single Test Run
Example commands
```bash
# Test prefill attention with paged KV cache
python3 flashinferTest.py \
    --routine BatchPrefillWithPagedKVCacheWrapper \
    --backends fa2 cudnn \
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
python3 flashinferTest.py \
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
python3 flashinferTest.py \
    --routine BatchDecodeWithPagedKVCacheWrapper \
    --backends fa2 fa2_tc trtllm cudnn \
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
python3 flashinferTest.py \
    --routine gemm_fp8_nt_groupwise \
    --m 8192 \
    --n 4096 \
    --k 16384 \
    --mma_sm 2 \
    --refcheck \
    -vv

# Group FP8 GEMM
python3 flashinferTest.py \
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
python3 flashinferTest.py --testlist samples/sample_testlist.txt --output_path sample_testlist_output.csv
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
| `--routine`              | Test routine to run: `BatchDecodeWithPagedKVCacheWrapper`, `BatchPrefillWithPagedKVCacheWrapper`, or `BatchPrefillWithRaggedKVCacheWrapper`, `gemm_fp8_nt_groupwise`, `group_gemm_fp8_nt_groupwise`|
| `--num_iters`            | Number of iterations for performance measurement                                                           |
| `--dry_run_iters`        | Number of warmup iterations                                                                                |
| `--no_cuda_graph`        | Disable CUDA graph to execute kernels outside of the graph.
| `--allow_output_mismatch`| Continue testing even if outputs don't match between backends                                              |
| `--refcheck`             | Verify outputs match between different backends                                                            |
| `--random_seed`          | Random seed for reproducibility                                                                            |
| `--output_path`          | Path to save CSV results                                                                                   |
| `--verbose`, `-v`        | Print additional information                                                                               |

### Attention Flags
| Flag                     | Description                                                                                                 |
|--------------------------|-------------------------------------------------------------------------------------------------------------|
| `--backends`             | List of backends to test: fa2, fa2_tc, fa3, cudnn, cutlass, trtllm                                         |
| `--page_size`            | Page size for paged attention. Required for paged attention tests.                                          |
| `--batch_size`           | Number of sequences to process in parallel                                                                  |
| `--s_qo`                 | Query/output sequence length. Should be 1 for decode tests.                                                 |
| `--s_kv`                 | Key/value sequence length (context length)                                                                  |
| `--num_qo_heads`         | Number of query/output attention heads                                                                      |
| `--num_kv_heads`         | Number of key/value attention heads
| `--head_dim_qk`          | Head dimension for Q/K. Must be 128 or 192.                                                                |
| `--head_dim_vo`          | Head dimension for V/O. Usually equals head_dim_qk.                                                        |
| `--q_dtype`              | Data type for the query tensor. Default: bfloat16. Currently only bfloat16 is supported.                   |
| `--kv_dtype`             | Data type for the key and value tensors. Default: bfloat16. Currently only bfloat16 is supported.         |
| `--causal`               | Use causal attention masking (prefill only)                                                                |
| `--random_actual_seq_len`| Use random sequence lengths up to max length. If False, use max length.                                    |

### GEMM Flags
| Flag                     | Description                                                                                                 |
|--------------------------|-------------------------------------------------------------------------------------------------------------|
| `--m`                  | Number of rows of matrix A and output matrix (GEMM M dimension)                |
| `--n`                  | Number of columns of matrix B and output matrix (GEMM N dimension)             |
| `--k`                  | Number of columns of matrix A / rows of matrix B (GEMM K dimension)            |
| `--tile_size`          | Tile size for the GEMM operation (affects performance and scaling)              |
| `--group_size`         | Number of groups for group GEMM (batching multiple GEMMs together)              |
| `--scale_major_mode`   | Layout for FP8 scaling: `MN` (per output tile) or `K` (per input tile)          |
| `--out_dtype`          | Output data type: `bfloat16` or `float16`                                       |
| `--mma_sm`             | Number of SMs to use for the MMA operation (1 or 2)                             |

## Tester Attention Backend Support Matrix
The following support surface applies to attention operations in `flashinferTest.py`
| Backend  | Decode Paged | Prefill Paged | Prefill Ragged | FP8  | Notes                                    |
|----------|-------------|---------------|----------------|------|------------------------------------------|
| fa2      | ✓           | ✓             | ✓              | ✗    | Does not support GQA ratio of 5          |
| fa2_tc   | ✓           | ✗             | ✗              | ✗    | Uses tensor cores                        |
| fa3      | ✗           | ✓             | ✓              | ✗    | Hopper Only      |
| cudnn    | ✓           | ✓*            | ✓*             | ✗    | *Requires specific head dims (192 or 128) |
| cutlass  | ✗           | ✗             | ✓              | ✗    |                                          |
| trtllm   | ✓           | ✗             | ✗              | ✗    |                                          |

Notes:
- Currently only support bfloat16 attention only.
- CUDA graph support is only stable with BatchDecodeWithPagedKVCacheWrapper. For BatchPrefillWithPagedKVCacheWrapper and BatchPrefillWithRaggedKVCacheWrapper, it is recommended that `--no_cuda_graph` is used.
- cudnn, cutlass, and trtllm backends are supported on [CUDA Comute Capability 10.0 GPUs](https://developer.nvidia.com/cuda-gpus) only.
- fa3 is supported on [CUDA Comute Capability 9.0 GPUs](https://developer.nvidia.com/cuda-gpus) only.

## Example Outputs
```bash
$ python3 flashinferTest.py --routine BatchPrefillWithPagedKVCacheWrapper --backends fa2 cudnn --page_size 16 --batch_size 16 --s_qo 1024 --s_kv 1024 --num_qo_heads 8 --num_kv_heads 8 --head_dim_qk 128 --head_dim_vo 128 --random_actual_seq_len --refcheck --causal --no_cuda_graph -vv
[INFO] args = Namespace(routine='BatchPrefillWithPagedKVCacheWrapper', backends=['fa2', 'cudnn'], page_size=16, batch_size=16, s_qo=1024, s_kv=1024, num_qo_heads=8, num_kv_heads=8, head_dim_qk=128, head_dim_vo=128, q_dtype='bfloat16', kv_dtype='bfloat16', causal=True, num_iters=30, dry_run_iters=5, no_cuda_graph=True, random_actual_seq_len=True, refcheck=True, allow_output_mismatch=False, random_seed=42, verbose=2, output_path=None)
[INFO] FlashInfer version: 0.2.8
[INFO] Running testBatchPrefillWithPagedKVCacheWrapper
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
[VVERBOSE] scale = 0.08838834764831843
[PERF] fa2     :: median time 0.094 ms; std 0.022 ms; achieved tflops 57.960 TFLOPs/sec; achieved tb_per_sec 0.459 TB/sec
[PERF] cudnn   :: median time 0.096 ms; std 0.021 ms; achieved tflops 56.661 TFLOPs/sec; achieved tb_per_sec 0.449 TB/sec

$ python3 flashinferTest.py --routine BatchPrefillWithRaggedKVCacheWrapper --backends fa2 cudnn cutlass --batch_size 16 --s_qo 1024 --s_kv 1024 --num_qo_heads 128 --num_kv_heads 128 --head_dim_qk 192 --head_dim_vo 128  --refcheck --causal --no_cuda_graph -vv
[INFO] args = Namespace(routine='BatchPrefillWithRaggedKVCacheWrapper', backends=['fa2', 'cudnn', 'cutlass'], page_size=0, batch_size=16, s_qo=1024, s_kv=1024, num_qo_heads=128, num_kv_heads=128, head_dim_qk=192, head_dim_vo=128, q_dtype='bfloat16', kv_dtype='bfloat16', causal=True, num_iters=30, dry_run_iters=5, no_cuda_graph=True, random_actual_seq_len=False, refcheck=True, allow_output_mismatch=False, random_seed=42, verbose=True, vverbose=True, output_path=None)
[INFO] FlashInfer version: 0.2.8
[INFO] Running testBatchPrefillWithRaggedKVCacheWrapper
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
[PERF] fa2     :: median time 2.197 ms; std 0.011 ms; achieved tflops 312.787 TFLOPs/sec; achieved tb_per_sec 1.222 TB/sec
[PERF] cudnn   :: median time 1.008 ms; std 0.014 ms; achieved tflops 681.979 TFLOPs/sec; achieved tb_per_sec 2.664 TB/sec
[PERF] cutlass :: median time 1.453 ms; std 0.021 ms; achieved tflops 473.035 TFLOPs/sec; achieved tb_per_sec 1.848 TB/sec

$ python3 flashinferTest.py --routine BatchDecodeWithPagedKVCacheWrapper --backends fa2 fa2_tc trtllm cudnn --page_size 16 --batch_size 32 --s_qo 1 --s_kv 8192 --num_qo_heads 64 --num_kv_heads 8 --head_dim_qk 128 --head_dim_vo 128 --refcheck -vv
[INFO] args = Namespace(routine='BatchDecodeWithPagedKVCacheWrapper', backends=['fa2', 'fa2_tc', 'trtllm', 'cudnn'], page_size=16, batch_size=32, s_qo=1, s_kv=8192, num_qo_heads=64, num_kv_heads=8, head_dim_qk=128, head_dim_vo=128, q_dtype='bfloat16', kv_dtype='bfloat16', causal=False, num_iters=30, dry_run_iters=5, no_cuda_graph=False, random_actual_seq_len=False, refcheck=True, allow_output_mismatch=False, random_seed=42, verbose=2, output_path=None)
[INFO] FlashInfer version: 0.2.8
[INFO] Running testBatchDecodeWithPagedKVCacheWrapper
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
[PERF] fa2     :: median time 0.712 ms; std 0.000 ms; achieved tflops 12.061 TFLOPs/sec; achieved tb_per_sec 1.509 TB/sec
[PERF] fa2_tc  :: median time 0.173 ms; std 0.001 ms; achieved tflops 49.779 TFLOPs/sec; achieved tb_per_sec 6.228 TB/sec
[PERF] trtllm  :: median time 0.155 ms; std 0.000 ms; achieved tflops 55.344 TFLOPs/sec; achieved tb_per_sec 6.925 TB/sec
[PERF] cudnn   :: median time 0.253 ms; std 0.000 ms; achieved tflops 33.964 TFLOPs/sec; achieved tb_per_sec 4.250 TB/sec
```
