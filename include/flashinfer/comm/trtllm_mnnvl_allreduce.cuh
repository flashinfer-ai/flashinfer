/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>

#include <iostream>

#include "../exception.h"
#include "../logging.h"
namespace flashinfer {
namespace trtllm_mnnvl_allreduce {

template <typename T>
struct AllReduceParams {
  int nranks;
  int rank;
  int buffer_M;
  int num_tokens;
  int token_dim;
  void** buffer_ptrs_dev;
  void* multicast_ptr;
  void* buffer_flags;
  bool wait_for_results;
  bool launch_with_pdl;

  void* input;
  void* output;
  cudaStream_t stream;
};

template <typename T>
struct RMSNormParams {
  void* residual_output;
  void* output;
  void const* input;
  void const* gamma;
  double epsilon;
  void* residual;
  uint32_t* buffer_flags;
  int batch;
  int hidden_dim;
  cudaStream_t stream;
  bool launch_with_pdl;
};

__device__ bool isNegZero(float v) { return v == 0.f && signbit(v); }

__device__ bool isNegZero(__nv_bfloat16 val) { return isNegZero(__bfloat162float(val)); }

__device__ bool isNegZero(__nv_half val) { return isNegZero(__half2float(val)); }

template <typename T>
inline __device__ float toFloat(T val) {
  return val;
}

template <>
inline __device__ float toFloat<__nv_bfloat16>(__nv_bfloat16 val) {
  return __bfloat162float(val);
}

template <>
inline __device__ float toFloat<__nv_half>(__nv_half val) {
  return __half2float(val);
}

template <typename T>
inline __device__ T fromFloat(float val) {
  return val;
}

template <>
inline __device__ __nv_bfloat16 fromFloat<__nv_bfloat16>(float val) {
  return __float2bfloat16(val);
}

template <>
inline __device__ __nv_half fromFloat<__nv_half>(float val) {
  return __float2half(val);
}

inline __device__ float2 loadfloat2(void const* ptr) {
  float2 return_value;
  asm volatile("ld.volatile.global.v2.f32 {%0, %1}, [%2];\n"
               : "=f"(return_value.x), "=f"(return_value.y)
               : "l"(ptr));
  return return_value;
}

template <typename T>
inline __device__ T divUp(T val, T divisor) {
  return (val + divisor - 1) / divisor;
}

__device__ struct __attribute__((aligned(32))) LamportFlags {
  uint32_t buffer_size;
  uint32_t input_offset;
  uint32_t clear_offset;
  uint32_t num_tokens_prev;
  uint32_t* offset_access_ptr;
  uint32_t* buffer_flags;

  __device__ explicit LamportFlags(uint32_t* buffer_flags)
      : offset_access_ptr(&buffer_flags[4]), buffer_flags(buffer_flags) {
    uint4 flag = reinterpret_cast<uint4*>(buffer_flags)[0];
    buffer_size = flag.z;
    input_offset = flag.x * (buffer_size << 1U);
    clear_offset = flag.y * (buffer_size << 1U);
    num_tokens_prev = flag.w;
  }

  __device__ void cta_arrive() {
    __syncthreads();
    if (threadIdx.x == 0) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000))
      asm volatile("red.async.release.global.gpu.add.u32 [%0], %1;" ::"l"(offset_access_ptr), "r"(1)
                   : "memory");
#elif (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
      asm volatile("red.global.gpu.add.u32 [%0], %1;" ::"l"(offset_access_ptr), "r"(1) : "memory");
#else
      atomicAdd(offset_access_ptr, 1);
#endif
    }
  }

  __device__ void wait_and_update(uint32_t num_tokens) {
    if (threadIdx.x == 0 && blockIdx.x == gridDim.x - 1 && blockIdx.y == 0) {
      while (*reinterpret_cast<uint32_t volatile*>(offset_access_ptr) < gridDim.x * gridDim.y) {
      }
      uint4 flag = reinterpret_cast<uint4*>(buffer_flags)[0];
      buffer_flags[0] = (flag.x + 1) % 3;
      buffer_flags[1] = (flag.y + 1) % 3;
      buffer_flags[3] = num_tokens;
      *(offset_access_ptr) = 0;
    }
  }
};

template <int WORLD_SIZE, typename T>
__global__ void twoshot_allreduce_kernel(T* output_ptr, T* shard_ptr, T** input_ptrs, T* mcast_ptr,
                                         int num_tokens, int buffer_M, int token_dim, int rank,
                                         uint32_t* buffer_flags, bool wait_for_results) {
  int elt = blockIdx.y * blockDim.x + threadIdx.x;

  if (elt >= token_dim) return;
  int token = blockIdx.x;

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif

  LamportFlags flags(buffer_flags);

  // Capture the number of tokens in previous iteration so that we can properly clear the buffer
  // The scatter stage will use the buffer in WORLD_SIZE granularity, thus we need to round up
  uint32_t clr_toks_cta =
      divUp<uint32_t>(flags.num_tokens_prev > num_tokens ? flags.num_tokens_prev : num_tokens,
                      WORLD_SIZE) *
      WORLD_SIZE;
  clr_toks_cta = divUp<uint32_t>(clr_toks_cta, gridDim.x);

  if (elt < token_dim) {
    // Scatter token
    int dest_rank = token % WORLD_SIZE;
    int dest_token_offset = token / WORLD_SIZE;
    T val = shard_ptr[token * token_dim + elt];
    if (isNegZero(val)) val = fromFloat<T>(0.f);
    input_ptrs[dest_rank][flags.input_offset + dest_token_offset * token_dim * WORLD_SIZE +
                          rank * token_dim + elt] = val;

    // Clear the buffer used by the previous call. Note the number of tokens to clear could be
    // larger than the
    // number of tokens in the current call.
    for (int clr_tok = 0; clr_tok < clr_toks_cta; clr_tok++) {
      uint32_t clr_token_idx = token + clr_tok * gridDim.x;
      if (clr_token_idx < buffer_M) {
        input_ptrs[rank][flags.clear_offset + clr_token_idx * token_dim + elt] = fromFloat<T>(-0.f);
      }
    }

    // Reduce and broadcast
    if ((token % WORLD_SIZE) == rank) {
      int local_token = token / WORLD_SIZE;
      float accum = 0.f;

      T values[WORLD_SIZE];

      while (1) {
        bool valid = true;
        for (int r = 0; r < WORLD_SIZE; r++) {
          T volatile* lamport_ptr =
              (T volatile*)&input_ptrs[rank]
                                      [flags.input_offset + local_token * token_dim * WORLD_SIZE +
                                       r * token_dim + elt];
          values[r] = *lamport_ptr;
          valid &= !isNegZero(values[r]);
        }
        if (valid) break;
      }
      for (int r = 0; r < WORLD_SIZE; r++) {
        accum += toFloat<T>(values[r]);
      }
      mcast_ptr[flags.input_offset + buffer_M * token_dim + token * token_dim + elt] =
          fromFloat<T>(accum);
    }
  }

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaTriggerProgrammaticLaunchCompletion();
#endif

  // Similarly clear broadcast buffer here
  for (int clr_tok = 0; clr_tok < clr_toks_cta; clr_tok++) {
    uint32_t clr_token_idx = token + clr_tok * gridDim.x;
    if (clr_token_idx < buffer_M) {
      input_ptrs[rank][flags.clear_offset + buffer_M * token_dim + clr_token_idx * token_dim +
                       elt] = fromFloat<T>(-0.f);
    }
  }

  // Optionally wait for results if the next layer isn't doing the Lamport check
  if (wait_for_results) {
    // Update the atomic counter to indicate the block has read the offsets
    flags.cta_arrive();
    // Only use a set of CTAs for lamport sync, reargange the grid
    constexpr int ELTS_PER_LOAD = sizeof(float2) / sizeof(T);
    // blockDim.x / ELTS_PER_LOAD should be at least the size of a warp (32)
    if (threadIdx.x < (blockDim.x / ELTS_PER_LOAD)) {
      uint64_t current_pos =
          blockIdx.x * token_dim + blockIdx.y * blockDim.x + threadIdx.x * ELTS_PER_LOAD;

      void* lamport_ptr =
          (void*)&input_ptrs[rank][flags.input_offset + buffer_M * token_dim + current_pos];
      // We have 2 assumptions here:
      // 1. The write is atomic in 8B granularity -> Each buffer in the buffer group should be
      // aligned to 8B
      // 2. The num_token * token_dim is divisible by ELTS_PER_LOAD (4 for BF16 and 2 for FP32)
      float2 val = loadfloat2(lamport_ptr);
      while (isNegZero(*(T*)&val)) {
        val = loadfloat2(lamport_ptr);
      }
      if (output_ptr) {
        *((float2*)&output_ptr[current_pos]) = val;
      }
    }

    // Update the buffer flags
    flags.wait_and_update(num_tokens);
  }
}

// Template-based dispatch functions following the same pattern as trtllm_allreduce.cuh
template <typename T, int WORLD_SIZE>
cudaError_t twoshot_allreduce_dispatch(AllReduceParams<T>& params) {
  int const num_threads = 128;
  int const num_blocks = (params.token_dim + num_threads - 1) / num_threads;

  dim3 grid(params.num_tokens, num_blocks);

  cudaLaunchConfig_t config;
  cudaLaunchAttribute attrs[1];
  config.dynamicSmemBytes = 0;
  config.stream = params.stream;
  config.gridDim = grid;
  config.blockDim = num_threads;
  config.attrs = attrs;
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = params.launch_with_pdl ? 1 : 0;
  config.numAttrs = 1;

  cudaLaunchKernelEx(&config, &twoshot_allreduce_kernel<WORLD_SIZE, T>,
                     reinterpret_cast<T*>(params.output), reinterpret_cast<T*>(params.input),
                     reinterpret_cast<T**>(params.buffer_ptrs_dev),
                     reinterpret_cast<T*>(params.multicast_ptr), params.num_tokens, params.buffer_M,
                     params.token_dim, params.rank,
                     reinterpret_cast<uint32_t*>(params.buffer_flags), params.wait_for_results);

  return cudaSuccess;
}

template <typename T>
cudaError_t twoshot_allreduce_dispatch_world_size(AllReduceParams<T>& params) {
  FLASHINFER_LOG_DEBUG("twoshot_allreduce_dispatch_world_size");
  switch (params.nranks) {
    case 2:
      return twoshot_allreduce_dispatch<T, 2>(params);
    case 4:
      return twoshot_allreduce_dispatch<T, 4>(params);
    case 8:
      return twoshot_allreduce_dispatch<T, 8>(params);
    case 16:
      return twoshot_allreduce_dispatch<T, 16>(params);
    case 32:
      return twoshot_allreduce_dispatch<T, 32>(params);
    case 64:
      return twoshot_allreduce_dispatch<T, 64>(params);
    default:
      FLASHINFER_ERROR("MNNVL AllReduce: unsupported world_size " + std::to_string(params.nranks) +
                       ". Supported sizes: {2, 4, 8, 16, 32, 64}");
      return cudaErrorInvalidValue;
  }
}

template <typename T_IN>
__device__ void copy_f4(T_IN* dst, T_IN const* src) {
  float4* dst4 = (float4*)dst;
  float4 const* src4 = (float4 const*)src;
  __pipeline_memcpy_async(dst4, src4, sizeof(float4));
}

template <typename T_IN>
__device__ void copy_f4_ldg(T_IN* dst, T_IN const* src) {
  float4* dst4 = (float4*)dst;
  float4 const* src4 = (float4*)src;
  *dst4 = *src4;
}

__device__ float4 loadfloat4(void const* ptr) {
  // Check alignment - ptr should be 16-byte aligned for safe float4 load
  if (reinterpret_cast<uintptr_t>(ptr) % 16 != 0) {
    // Fall back to scalar loads if not aligned
    float4 return_value;
    float const* float_ptr = reinterpret_cast<float const*>(ptr);
    return_value.x = float_ptr[0];
    return_value.y = float_ptr[1];
    return_value.z = float_ptr[2];
    return_value.w = float_ptr[3];
    return return_value;
  }

  float4 return_value;

  asm volatile("ld.volatile.global.v4.f32 {%0, %1, %2, %3}, [%4];\n"
               : "=f"(return_value.x), "=f"(return_value.y), "=f"(return_value.z),
                 "=f"(return_value.w)
               : "l"(ptr));

  return return_value;
}

// Safer version that checks bounds before loading
template <typename T>
__device__ float4 loadfloat4_safe(T const* ptr, int remaining_elements) {
  float return_value[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  if (remaining_elements <= 0) {
    return *(float4*)return_value;
  }

  // Check alignment - ptr should be 16-byte aligned for safe float4 load
  bool is_aligned = (reinterpret_cast<uintptr_t>(ptr) % 16 == 0);

  if (is_aligned && remaining_elements >= 4) {
    // Safe to do vectorized load
    asm volatile("ld.volatile.global.v4.f32 {%0, %1, %2, %3}, [%4];\n"
                 : "=f"(return_value[0]), "=f"(return_value[1]), "=f"(return_value[2]),
                   "=f"(return_value[3])
                 : "l"(ptr));
  } else {
    // Fall back to scalar loads with bounds checking
    float const* float_ptr = reinterpret_cast<float const*>(ptr);
    for (int i = 0; i < 4 && i < remaining_elements; i++) {
      return_value[i] = toFloat(float_ptr[i]);
    }
  }

  return *(float4*)return_value;
}

template <typename T>
inline __device__ T add(T a, T b) {
  return a + b;
}

#define FINAL_MASK 0xffffffff

template <typename T>
__inline__ __device__ T warpReduceSum(T val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    val = add<T>(val, __shfl_xor_sync(FINAL_MASK, val, mask,
                                      32));  //__shfl_sync bf16 return float when sm < 80
  return val;
}

inline __device__ float block_reduce_sum(float val) {
  __shared__ float smem[32];
  int lane_id = threadIdx.x % 32, warp_id = threadIdx.x / 32, warp_num = blockDim.x / 32;
  val = warpReduceSum(val);
  if (lane_id == 0) {
    smem[warp_id] = val;
  }
  __syncthreads();
  val = lane_id < warp_num ? smem[lane_id] : 0.f;
  val = warpReduceSum(val);
  return val;
}

template <int DIM, int NUM_THREADS, int NUM_INPUTS, typename T_OUT, typename T_IN>
__global__ void __launch_bounds__(128, 1)
    RMSNorm(T_IN* input_plus_residual, T_OUT* output_norm, T_IN const* buffer_input,
            T_IN const* gamma, float epsilon, T_IN const* residual, int batch_size,
            uint32_t* buffer_flags) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))

  static bool const LAMPORT = true;

  extern __shared__ uint8_t smem[];

  int sample = blockIdx.y;

  static int const CGA_THREADS = NUM_THREADS * 1;

  static int const ITERS = DIM / CGA_THREADS;
  float r_input[ITERS];
  float r_gamma[ITERS];

  T_IN* sh_input = (T_IN*)&smem[0];
  T_IN* sh_residual = (T_IN*)&smem[NUM_INPUTS * NUM_THREADS * ITERS * sizeof(T_IN)];
  T_IN* sh_gamma = (T_IN*)&smem[(NUM_INPUTS + 1) * NUM_THREADS * ITERS * sizeof(T_IN)];

  static int const ELTS_PER_THREAD = sizeof(float4) / sizeof(T_IN);

  int offsets[NUM_INPUTS][DIM / (1 * ELTS_PER_THREAD * NUM_THREADS)];

  LamportFlags flags(buffer_flags);
  T_IN const* input = &buffer_input[flags.input_offset + flags.buffer_size];

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaTriggerProgrammaticLaunchCompletion();
#endif

  for (int i = 0; i < NUM_INPUTS; i++) {
    for (int j = 0; j < DIM / (1 * ELTS_PER_THREAD * NUM_THREADS); j++) {
      int k = j * NUM_THREADS + threadIdx.x;
      offsets[i][j] =
          i * batch_size * DIM + sample * DIM + blockIdx.x * DIM / 1 + k * ELTS_PER_THREAD;
    }
  }

#pragma unroll
  for (int j = 0; j < DIM / (1 * ELTS_PER_THREAD * NUM_THREADS); j++) {
    int i = j * NUM_THREADS + threadIdx.x;
    copy_f4(&sh_residual[i * ELTS_PER_THREAD],
            &residual[sample * DIM + blockIdx.x * DIM + i * ELTS_PER_THREAD]);
  }

  __pipeline_commit();

#pragma unroll
  for (int j = 0; j < DIM / (ELTS_PER_THREAD * NUM_THREADS); j++) {
    int i = j * NUM_THREADS + threadIdx.x;
    copy_f4(&sh_gamma[i * ELTS_PER_THREAD], &gamma[blockIdx.x * DIM + i * ELTS_PER_THREAD]);
  }

  __pipeline_commit();
  flags.cta_arrive();

  // Load all inputs
  bool valid = false;

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  if (!LAMPORT) cudaGridDependencySynchronize();
#endif

  while (!valid) {
    valid = true;
#pragma unroll
    for (int i = 0; i < NUM_INPUTS; i++) {
      for (int j = 0; j < DIM / (ELTS_PER_THREAD * NUM_THREADS); j++) {
        int k = j * NUM_THREADS + threadIdx.x;

        float4* dst4 = (float4*)&sh_input[i * NUM_THREADS * ITERS + k * ELTS_PER_THREAD];

        // Calculate the absolute element offset from the start of buffer_input
        int element_offset = offsets[i][j];

        // The input pointer is already offset to: &buffer_input[buffer_offset + buffer_size]
        // So the actual pointer we're accessing is: input + element_offset
        // Which equals: &buffer_input[buffer_offset + buffer_size + element_offset]

        float4* src4 = (float4*)&input[element_offset];

        float4 value;
        // Check if we have enough elements remaining for a safe float4 load
        if (element_offset >= 0 && element_offset + ELTS_PER_THREAD <= flags.buffer_size) {
          value = loadfloat4(src4);
        } else {
          // Use safe load for boundary cases or out-of-bounds
          int remaining_elements = flags.buffer_size - element_offset;
          if (remaining_elements <= 0) {
            // Completely out of bounds, return zeros
            float4 return_value = {0.0f, 0.0f, 0.0f, 0.0f};
            value = return_value;
          } else {
            value = loadfloat4_safe(reinterpret_cast<T_IN const*>(src4), remaining_elements);
          }
        }

        if (LAMPORT) {
          // Assume that the 16B were written atomically, so we only need to check one value
          T_IN lowest_val = *(T_IN*)&value;
          valid &= !isNegZero(lowest_val);
        }
        *dst4 = value;
      }
    }
  }

  __syncthreads();

  // Perform the initial input reduction
  if (NUM_INPUTS > 0) {
    T_IN accum[ELTS_PER_THREAD];
    float4* accum4 = (float4*)&accum;

    for (int j = 0; j < DIM / (ELTS_PER_THREAD * NUM_THREADS); j++) {
      int k = j * NUM_THREADS + threadIdx.x;

      *accum4 = *(float4*)&sh_input[k * ELTS_PER_THREAD];

      for (int i = 1; i < NUM_INPUTS; i++) {
        float4 data = *(float4*)&sh_input[i * NUM_THREADS * ITERS + k * ELTS_PER_THREAD];
        T_IN* p_d = (T_IN*)&data;
        for (int x = 0; x < ELTS_PER_THREAD; x++) {
          accum[x] += p_d[x];
        }
      }

      // Write back to input 0's staging location.  No sync needed since all data localized to
      // thread.
      *(float4*)&sh_input[k * ELTS_PER_THREAD] = *accum4;
    }
  }

  // Wait for residual
  __pipeline_wait_prior(1);
  __syncthreads();

  float thread_sum = 0.f;

#pragma unroll
  for (int io = 0; io < ITERS / ELTS_PER_THREAD; io++) {
    float4 inp4 =
        *(float4*)&sh_input[io * NUM_THREADS * ELTS_PER_THREAD + threadIdx.x * ELTS_PER_THREAD];
    float4 res4 =
        *(float4*)&sh_residual[io * NUM_THREADS * ELTS_PER_THREAD + threadIdx.x * ELTS_PER_THREAD];

    T_IN* r_inp = (T_IN*)&inp4;
    T_IN* r_res = (T_IN*)&res4;

    float4 out4;

    T_IN* r_out = (T_IN*)&out4;

    for (int ii = 0; ii < ELTS_PER_THREAD; ii++) {
      int i = io * ELTS_PER_THREAD + ii;

      T_IN inp_plus_resid = r_inp[ii] + r_res[ii];
      r_out[ii] = inp_plus_resid;
      r_input[i] = toFloat(inp_plus_resid);

      // Accumulate the squares for RMSNorm
      thread_sum += toFloat(inp_plus_resid * inp_plus_resid);
    }

    *(float4*)&input_plus_residual[sample * DIM + blockIdx.x * DIM +
                                   io * NUM_THREADS * ELTS_PER_THREAD +
                                   threadIdx.x * ELTS_PER_THREAD] = out4;
  }

  // Wait for Gamma.  There will be a global synchronization as part of the reduction
  __pipeline_wait_prior(0);

  float cluster_sum = block_reduce_sum(thread_sum);

  float rcp_rms = rsqrtf(cluster_sum / DIM + epsilon);

#pragma unroll
  for (int io = 0; io < ITERS / ELTS_PER_THREAD; io++) {
    float4 gamma4 =
        *(float4*)&sh_gamma[io * NUM_THREADS * ELTS_PER_THREAD + threadIdx.x * ELTS_PER_THREAD];
    T_IN* r_g4 = (T_IN*)&gamma4;

    float4 out4;
    // FIXME: this only works if T_OUT == T_IN
    T_OUT* r_out = (T_OUT*)&out4;

    for (int ii = 0; ii < ELTS_PER_THREAD; ii++) {
      int i = io * ELTS_PER_THREAD + ii;
      r_gamma[i] = toFloat(r_g4[ii]);
      r_out[ii] = fromFloat<T_OUT>(r_gamma[i] * r_input[i] * rcp_rms);
    }

    *(float4*)&output_norm[sample * DIM + blockIdx.x * DIM + io * NUM_THREADS * ELTS_PER_THREAD +
                           threadIdx.x * ELTS_PER_THREAD] = out4;
  }
  // Update the buffer pointers
  flags.wait_and_update(batch_size);
#endif
}

template <typename T, int H_DIM>
cudaError_t twoshot_rmsnorm_dispatch(RMSNormParams<T>& params) {
  static constexpr int NUM_THREADS = 128;
  static constexpr int CGA_THREADS = NUM_THREADS;
  constexpr int iters = H_DIM / CGA_THREADS;

  dim3 grid(1, params.batch, 1);

  cudaLaunchConfig_t config;
  cudaLaunchAttribute attrs[1];
  config.stream = params.stream;
  config.gridDim = grid;
  config.blockDim = NUM_THREADS;
  config.attrs = attrs;
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = params.launch_with_pdl ? 1 : 0;
  config.numAttrs = 1;

  size_t shmem_size = 3 * NUM_THREADS * iters * sizeof(T);
  config.dynamicSmemBytes = shmem_size;

  cudaFuncSetAttribute(&RMSNorm<H_DIM, NUM_THREADS, 1, T, T>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);

  cudaLaunchKernelEx(
      &config, &RMSNorm<H_DIM, NUM_THREADS, 1, T, T>, reinterpret_cast<T*>(params.residual_output),
      reinterpret_cast<T*>(params.output), reinterpret_cast<T const*>(params.input),
      reinterpret_cast<T const*>(params.gamma), static_cast<float>(params.epsilon),
      reinterpret_cast<T const*>(params.residual), params.batch, params.buffer_flags);

  return cudaSuccess;
}

template <typename T>
cudaError_t twoshot_rmsnorm_dispatch_hidden_dim(RMSNormParams<T>& params) {
  FLASHINFER_LOG_DEBUG("twoshot_rmsnorm_dispatch_hidden_dim");
  switch (params.hidden_dim) {
    case 2048:
      return twoshot_rmsnorm_dispatch<T, 2048>(params);
    case 4096:
      return twoshot_rmsnorm_dispatch<T, 4096>(params);
    case 5120:
      return twoshot_rmsnorm_dispatch<T, 5120>(params);  // Llama-4
    case 7168:
      return twoshot_rmsnorm_dispatch<T, 7168>(params);  // DeepSeek
    case 8192:
      return twoshot_rmsnorm_dispatch<T, 8192>(params);
    default:
      FLASHINFER_ERROR("MNNVL TwoShot RMSNorm: unsupported hidden_dim " +
                       std::to_string(params.hidden_dim) +
                       ". Supported sizes: {2048, 4096, 5120, 7168, 8192}");
      return cudaErrorInvalidValue;
  }
}

}  // namespace trtllm_mnnvl_allreduce
}  // namespace flashinfer
