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

#include <tuple>
#include <type_traits>

#include "../exception.h"
#include "../logging.h"
#include "../utils.cuh"
#include "../vec_dtypes.cuh"

namespace flashinfer {

namespace trtllm_allreduce {

constexpr size_t WARP_SIZE = 32;
constexpr size_t MAX_ALL_REDUCE_BLOCKS = 24;
constexpr size_t MAX_RANKS_PER_NODE = 16;
constexpr size_t DEFAULT_BLOCK_SIZE = 512;
constexpr size_t NUM_POINTERS_PER_RANK = 7;

namespace details {

static constexpr int kBytesPerAccess = 16;
static constexpr int kWarpSize = 32;
static constexpr int kMaxCtaSize = 1024;
static constexpr int kClusterMaxSize = 8;
static constexpr int kLamportTokenNumThreshold = 16;
static constexpr int kLamportHiddenSizeThreshold = 256;

}  // namespace details

enum class AllReduceStrategyType : int8_t {
  NCCL = 0,
  MIN_LATENCY = 1,
  UB = 2,
  AUTO = 3,
  ONESHOT = 4,
  TWOSHOT = 5,
  LOWPRECISION = 6,
};

enum class AllReduceStrategyConfig : int8_t {
  USE_MEMCPY = 1 << 0,
  PUSH_MODE = 1 << 1,
};

//////////////////////

enum class AllReduceFusionOp : int8_t {
  NONE = 0,
  RESIDUAL_RMS_NORM = 1,
  LAST_PROCESS_FOR_UB = 2,
  RESIDUAL_RMS_PREPOST_NORM = 3,
  RESIDUAL_RMS_NORM_QUANT_FP8 = 4,
  RESIDUAL_RMS_NORM_QUANT_NVFP4 = 5,
  RESIDUAL_RMS_NORM_OUT_QUANT_FP8 = 6,
  RESIDUAL_RMS_NORM_OUT_QUANT_NVFP4 = 7,
  MOE_ALLREDUCE_RESIDUAL_RMS_NORM = 8,
  MOE_FINALIZE_ALLREDUCE_RESIDUAL_RMS_NORM = 9,
};

template <typename T>
bool is_lamport_supported(int token_num, int hidden_size) {
  if (!std::is_same_v<T, half> && !std::is_same_v<T, __nv_bfloat16>) {
    return false;
  }
  if (token_num > details::kLamportTokenNumThreshold) {
    return false;
  }
  if (hidden_size < details::kLamportHiddenSizeThreshold) {
    return false;
  }
  return true;
}

struct AllReduceFusionParams {
  AllReduceFusionParams()
      : bias_buffer(nullptr),
        residual_buffer(nullptr),
        weight_buffer(nullptr),
        weight_buffer_pre_residual_norm(nullptr),
        intermediate_buffer(nullptr) {}

  // gemm bias
  void const* bias_buffer;
  // residuial add
  void const* residual_buffer;
  // rms norm
  int hidden_size;                              // equal to normalized_shape
  void const* weight_buffer;                    // norm elem-wise affine gamma
  void const* weight_buffer_pre_residual_norm;  // for gemma norm before residual
  float eps;
  // new residual
  void* intermediate_buffer;
  void* lamport_peer_comm_buffer_ptrs[MAX_RANKS_PER_NODE * 3];
};

template <typename T>
struct AllReduceParams {
  size_t elts_total;
  size_t elts_per_rank;
  size_t elts_per_block;
  size_t rank_offset;
  size_t ranks_per_node;
  size_t local_rank;
  uint32_t barrier_flag;
  uint32_t* peer_barrier_ptrs_in[MAX_RANKS_PER_NODE];
  uint32_t* peer_barrier_ptrs_out[MAX_RANKS_PER_NODE];
  void* peer_comm_buffer_ptrs[MAX_RANKS_PER_NODE];
  void* local_output_buffer_ptr;
  void const* local_input_buffer_ptr;

  AllReduceFusionParams fusion_params;

  static AllReduceParams deserialize(int64_t* buffer, size_t tpSize, size_t tpRank, int token_num,
                                     int hidden_size, AllReduceFusionOp op) {
    void* const* buffer_ptrs = reinterpret_cast<void* const*>(buffer);
    int flag_offset;
    if (op == AllReduceFusionOp::RESIDUAL_RMS_NORM &&
        is_lamport_supported<T>(token_num, hidden_size)) {
      flag_offset = 0;
    } else {
      flag_offset = 1;
    }
    auto const flag_ptr = &buffer[NUM_POINTERS_PER_RANK * tpSize + flag_offset];
    // cannot use 0 since 0 represents released state for barrier
    *flag_ptr += 1;
    uint32_t flag_value = *flag_ptr;
    AllReduceParams params;
    // Even plugins use ping buffers, odd plugins use pong.
    // That way, we don't need to wait for other GPUs to be done
    // before copying input tensor to workspace.
    auto const buffer_offset = (flag_value % 2 == 0) ? 0 : tpSize;

    for (int i = 0; i < tpSize; ++i) {
      params.peer_comm_buffer_ptrs[i] = buffer_ptrs[buffer_offset + i];
    }
    for (int i = 0; i < tpSize; ++i) {
      params.peer_barrier_ptrs_in[i] = reinterpret_cast<uint32_t*>(buffer_ptrs[2 * tpSize + i]);
    }
    for (int i = 0; i < tpSize; ++i) {
      params.peer_barrier_ptrs_out[i] = reinterpret_cast<uint32_t*>(buffer_ptrs[3 * tpSize + i]);
    }
    params.barrier_flag = flag_value;
    params.ranks_per_node = tpSize;
    params.local_rank = tpRank;

    return params;
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
struct neg_zero {
  static constexpr T value = -T(0);
};

template <>
struct neg_zero<half> {
  static constexpr unsigned short neg_zero_bits = 0x8000U;
  static constexpr __half value = __half_raw{neg_zero_bits};
};

template <>
struct neg_zero<nv_bfloat16> {
  static constexpr unsigned short neg_zero_bits = 0x8000U;
  static constexpr __nv_bfloat16 value = __nv_bfloat16_raw{neg_zero_bits};
};

template <>
struct neg_zero<float> {
  static constexpr unsigned int neg_zero_bits = 0x80000000U;
  static constexpr float value = -0.0f;
};

template <typename T>
__device__ static constexpr T neg_zero_v = neg_zero<T>::value;

template <typename T>
__device__ bool is_negative_zero(T) {
  return false;
}

// float specialization
template <>
__device__ bool is_negative_zero<float>(float x) {
  return (__float_as_int(x) == 0x80000000);
}

// double specialization
template <>
__device__ bool is_negative_zero<double>(double x) {
  return (__double_as_longlong(x) == 0x8000000000000000ULL);
}

// __half specialization
template <>
__device__ bool is_negative_zero<__half>(__half x) {
  return (__half_as_ushort(x) == 0x8000);
}

// __nv_bfloat16 specialization
template <>
__device__ bool is_negative_zero<__nv_bfloat16>(__nv_bfloat16 x) {
  return (__bfloat16_as_ushort(x) == 0x8000);
}

template <typename T, uint32_t VEC_SIZE>
__device__ __forceinline__ bool has_neg_zero(const vec_t<T, VEC_SIZE>& vec) {
#pragma unroll
  for (int i = 0; i < VEC_SIZE; ++i) {
    if (is_negative_zero(vec[i])) {
      return true;
    }
  }
  return false;
}

template <typename T, uint32_t VEC_SIZE>
__device__ __forceinline__ void remove_neg_zero(vec_t<T, VEC_SIZE>& vec) {
#pragma unroll
  for (int i = 0; i < VEC_SIZE; ++i) {
    vec[i] = (is_negative_zero(vec[i])) ? static_cast<T>(0.f) : vec[i];
  }
}

template <typename T>
__device__ __forceinline__ void set_neg_zero(T* addr) {
  vec_t<T, 16 / sizeof(T)> val;
  val.fill(neg_zero_v<T>);
  val.store_global_volatile(addr);
}

static inline __device__ void st_flag_release(uint32_t const& flag, uint32_t* flag_addr) {
#if __CUDA_ARCH__ >= 700
  asm volatile("st.global.release.sys.b32 [%1], %0;" ::"r"(flag), "l"(flag_addr));
#else
  __threadfence_system();
  asm volatile("st.global.volatile.b32 [%1], %0;" ::"r"(flag), "l"(flag_addr));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint32_t ld_flag_acquire(uint32_t* flag_addr) {
  uint32_t flag;
#if __CUDA_ARCH__ >= 700
  asm volatile("ld.global.acquire.sys.b32 %0, [%1];" : "=r"(flag) : "l"(flag_addr));
#else
  asm volatile("ld.global.volatile.b32 %0, [%1];" : "=r"(flag) : "l"(flag_addr));
#endif
  return flag;
}

template <typename T, uint32_t VEC_SIZE>
__device__ __forceinline__ vec_t<T, VEC_SIZE> vec_add(const vec_t<T, VEC_SIZE>& a,
                                                      const vec_t<T, VEC_SIZE>& b) {
  vec_t<T, VEC_SIZE> ret;
#pragma unroll
  for (int i = 0; i < VEC_SIZE; ++i) {
    ret[i] = static_cast<float>(a[i]) + static_cast<float>(b[i]);
  }
  return ret;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__inline__ __device__ void multi_gpu_barrier(uint32_t** signals, uint32_t const flag,
                                             size_t const local_rank, size_t const world_size,
                                             int const tidx, int const bidx) {
  // After this function, at least one block in each GPU has reached the barrier
  if (tidx < world_size) {
    // we can think of signals having the shape [world_size, world_size]
    // Dimension 0 is the "listening" dimension, dimension 1 is "emitting" dimension

    // Block 0 broadcasts its flag (local_rank on emitting dimension) to all receivers
    size_t offset = (flag % 2) ? world_size : 0;

    if (bidx == 0) {
      st_flag_release(flag, signals[tidx] + offset + local_rank);
    }

    // All blocks check that corresponding block 0 on other GPUs have set the flag
    // No deadlock because block #0 is always the first block started
    uint32_t* peer_barrier_d = signals[local_rank] + offset + tidx;
    while (ld_flag_acquire(peer_barrier_d) != flag) {
    }
  }

  __syncthreads();
}

__inline__ __device__ void block_barrier(uint32_t** signals, uint32_t const flag,
                                         size_t const local_rank, size_t const world_size,
                                         int const tidx, int const bidx, int const grid_size) {
  // After this function, the block of id == bidx of each GPU has reached the barrier
  if (tidx < world_size) {
    // we can think of signals having the shape [world_size, 2, num_blocks, world_size]
    // (+ an offset on dim 2 to account for flags used in multi_gpu_barrier)
    // Dimension 0 is the "listening" dimension, dimension 3 is "emitting" dimension

    // Block broadcast its flag (local_rank on emitting dimension) to all receivers
    uint32_t flag_block_offset = world_size + bidx * world_size;

    if (flag % 2 == 1) {
      flag_block_offset += (grid_size + 1) * world_size;
    }

    st_flag_release(flag, signals[tidx] + flag_block_offset + local_rank);

    // Blocks check that corresponding blocks on other GPUs have also set the flag
    uint32_t* peer_barrier_d = signals[local_rank] + flag_block_offset + tidx;

    while (ld_flag_acquire(peer_barrier_d) != flag) {
    }
  }

  __syncthreads();
}

namespace reduce_fusion {

inline __device__ float warp_reduce_sum(float val) {
  val += __shfl_xor_sync(~0, val, 16);
  val += __shfl_xor_sync(~0, val, 8);
  val += __shfl_xor_sync(~0, val, 4);
  val += __shfl_xor_sync(~0, val, 2);
  val += __shfl_xor_sync(~0, val, 1);
  return val;
}

inline __device__ float block_reduce_sum(float val) {
  __shared__ float smem[details::kWarpSize];
  int lane_id = threadIdx.x % details::kWarpSize, warp_id = threadIdx.x / details::kWarpSize,
      warp_num = blockDim.x / details::kWarpSize;
  val = warp_reduce_sum(val);
  if (lane_id == 0) {
    smem[warp_id] = val;
  }
  __syncthreads();
  val = lane_id < warp_num ? smem[lane_id] : 0.f;
  val = warp_reduce_sum(val);
  return val;
}

template <typename T, uint32_t VEC_SIZE>
inline __device__ float accumulate(float acc, vec_t<T, VEC_SIZE>& vec) {
#pragma unroll
  for (int i = 0; i < VEC_SIZE; ++i) {
    float v = static_cast<float>(vec[i]);
    acc += v * v;
  }
  return acc;
}

template <typename T, bool Affine, uint32_t VEC_SIZE>
inline __device__ vec_t<T, VEC_SIZE> rms_norm(float denom, vec_t<T, VEC_SIZE>& vec,
                                              vec_t<T, VEC_SIZE>& weight) {
  vec_t<T, VEC_SIZE> ret;
#pragma unroll
  for (int i = 0; i < VEC_SIZE; ++i) {
    float v1 = static_cast<float>(vec[i]);
    if constexpr (Affine) {
      float v2 = static_cast<float>(weight[i]);
      ret[i] = static_cast<T>(v1 * denom * v2);
    } else {
      ret[i] = static_cast<T>(v1 * denom);
    }
  }
  return ret;
}

template <typename T, bool Bias = false, bool Residual = false, bool Affine = false,
          bool UseSmem = false>
__global__ void rms_norm_kernel(AllReduceParams<T> params) {
  static constexpr uint32_t VEC_SIZE = 16 / sizeof(T);

  extern __shared__ uint8_t smem_ptr[];
  T* smem = reinterpret_cast<T*>(smem_ptr);

  int bid = blockIdx.x, tid = threadIdx.x;

  T const* bias_buffer = reinterpret_cast<T const*>(params.fusion_params.bias_buffer);
  T const* residual_buffer = reinterpret_cast<T const*>(params.fusion_params.residual_buffer);
  T const* weight_buffer = reinterpret_cast<T const*>(params.fusion_params.weight_buffer);
  T* local_final_output_buffer = reinterpret_cast<T*>(params.local_output_buffer_ptr);
  T* intermediate_buffer = reinterpret_cast<T*>(params.fusion_params.intermediate_buffer);

  int block_offset = bid * params.fusion_params.hidden_size;
  int thread_offset = tid * VEC_SIZE;

  if constexpr (Residual) {
    residual_buffer += block_offset;
  }
  local_final_output_buffer += block_offset;
  intermediate_buffer += block_offset;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && (__CUDA_ARCH__ < 1200))
  cudaGridDependencySynchronize();
#endif

  vec_t<T, VEC_SIZE> inter_vec, weight_vec;
  float acc = 0.f;
  for (int offset = thread_offset; offset < params.fusion_params.hidden_size;
       offset += blockDim.x * VEC_SIZE) {
    inter_vec.load(intermediate_buffer + offset);
    if constexpr (Bias) {
      vec_t<T, VEC_SIZE> bias_vec;
      bias_vec.load(bias_buffer + offset);
      inter_vec = vec_add<T, VEC_SIZE>(inter_vec, bias_vec);
    }
    if constexpr (Residual) {
      vec_t<T, VEC_SIZE> residual_vec;
      residual_vec.load(residual_buffer + offset);
      inter_vec = vec_add<T, VEC_SIZE>(inter_vec, residual_vec);
      inter_vec.store(intermediate_buffer + offset);
    }
    acc = accumulate<T, VEC_SIZE>(acc, inter_vec);
    if constexpr (UseSmem) {
      inter_vec.store(&smem[offset]);
    }
  }
  acc = block_reduce_sum(acc);
  float denom = rsqrtf(acc / params.fusion_params.hidden_size + params.fusion_params.eps);
  for (int offset = thread_offset; offset < params.fusion_params.hidden_size;
       offset += blockDim.x * VEC_SIZE) {
    if constexpr (UseSmem) {
      inter_vec.load(&smem[offset]);
    }
    if constexpr (Affine) {
      weight_vec.load(weight_buffer + offset);
    }
    inter_vec = rms_norm<T, Affine, VEC_SIZE>(denom, inter_vec, weight_vec);
    inter_vec.store(&local_final_output_buffer[offset]);
  }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && (__CUDA_ARCH__ < 1200))
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

template <typename T, bool Bias = false, bool Residual = false, bool Affine = false>
__global__ void rms_pre_post_norm_kernel(
    AllReduceParams<T> params)  // for gemma2 pre residual + post residual norm
{
  static constexpr uint32_t VEC_SIZE = 16 / sizeof(T);

  int bid = blockIdx.x, tid = threadIdx.x;

  T const* bias_buffer = reinterpret_cast<T const*>(params.fusion_params.bias_buffer);
  T const* residual_buffer = reinterpret_cast<T const*>(params.fusion_params.residual_buffer);
  T const* weight_buffer = reinterpret_cast<T const*>(params.fusion_params.weight_buffer);
  T const* weight_buffer_pre_residual_norm =
      reinterpret_cast<T const*>(params.fusion_params.weight_buffer_pre_residual_norm);
  T* local_final_output_buffer = reinterpret_cast<T*>(params.local_output_buffer_ptr);
  T* intermediate_buffer = reinterpret_cast<T*>(params.fusion_params.intermediate_buffer);

  int block_offset = bid * params.fusion_params.hidden_size;
  int thread_offset = tid * VEC_SIZE;

  if constexpr (Residual) {
    residual_buffer += block_offset;
  }
  local_final_output_buffer += block_offset;
  intermediate_buffer += block_offset;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && (__CUDA_ARCH__ < 1200))
  cudaGridDependencySynchronize();
#endif

  vec_t<T, VEC_SIZE> inter_vec, weight_vec, weight_vec_pre_residual_norm, bias_vec;
  float acc = 0.f;
  float acc_pre_residual_norm = 0.f;
  for (int offset = thread_offset; offset < params.fusion_params.hidden_size;
       offset += blockDim.x * VEC_SIZE) {
    inter_vec.load(intermediate_buffer + offset);
    if constexpr (Bias) {
      bias_vec.load(bias_buffer + offset);
    }

    if constexpr (Bias) {
      inter_vec = vec_add<T, VEC_SIZE>(inter_vec, bias_vec);
    }

    // pre-residual norm.
    acc_pre_residual_norm = accumulate<T, VEC_SIZE>(acc_pre_residual_norm, inter_vec);
    acc_pre_residual_norm = block_reduce_sum(acc_pre_residual_norm);
    float denom_pre_residual_norm =
        rsqrtf(acc_pre_residual_norm / params.fusion_params.hidden_size + params.fusion_params.eps);

    if constexpr (Affine) {
      weight_vec_pre_residual_norm.load(weight_buffer_pre_residual_norm + thread_offset);
    }
    inter_vec = rms_norm<T, Affine, VEC_SIZE>(denom_pre_residual_norm, inter_vec,
                                              weight_vec_pre_residual_norm);

    if constexpr (Residual) {
      vec_t<T, VEC_SIZE> residual_vec;
      residual_vec.load(residual_buffer + offset);
      inter_vec = vec_add<T, VEC_SIZE>(inter_vec, residual_vec);
      inter_vec.store(intermediate_buffer + offset);
    }
    acc = accumulate<T, VEC_SIZE>(acc, inter_vec);
  }
  acc = block_reduce_sum(acc);
  float denom = rsqrtf(acc / params.fusion_params.hidden_size + params.fusion_params.eps);
  for (int offset = thread_offset; offset < params.fusion_params.hidden_size;
       offset += blockDim.x * VEC_SIZE) {
    if constexpr (Affine) {
      weight_vec.load(weight_buffer + offset);
    }
    inter_vec = rms_norm<T, Affine, VEC_SIZE>(denom, inter_vec, weight_vec);
    inter_vec.store(&local_final_output_buffer[offset]);
  }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && (__CUDA_ARCH__ < 1200))
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

template <typename T, bool Bias = false, bool Residual = false, bool Affine = false>
cudaError_t rms_norm_kernel_launcher(AllReduceParams<T>& params, AllReduceFusionOp fusionOp,
                                     bool launch_with_pdl, cudaStream_t stream) {
  static constexpr uint32_t VEC_SIZE = 16 / sizeof(T);
  FLASHINFER_CHECK(params.fusion_params.hidden_size % VEC_SIZE == 0,
                   "hidden_size must be a multiple of ", VEC_SIZE);
  if (fusionOp == AllReduceFusionOp::RESIDUAL_RMS_PREPOST_NORM) {
    FLASHINFER_CHECK(params.fusion_params.hidden_size <= 8192,
                     "hidden_size must be less than or equal to 8192");
  }
  int need_threads = params.fusion_params.hidden_size / VEC_SIZE;
  int cta_size;
  if (need_threads <= details::kMaxCtaSize) {
    cta_size = (need_threads + details::kWarpSize - 1) / details::kWarpSize * details::kWarpSize;
  } else {
    cta_size = details::kMaxCtaSize;
  }
  int cta_num = params.elts_total / params.fusion_params.hidden_size;
  int smem_size = 0;
  if (cta_size * details::kBytesPerAccess / sizeof(T) < params.fusion_params.hidden_size) {
    smem_size = params.fusion_params.hidden_size * sizeof(T);
    cudaLaunchConfig_t kernelConfig = {0};
    kernelConfig.gridDim = cta_num;
    kernelConfig.blockDim = cta_size;
    kernelConfig.dynamicSmemBytes = smem_size;
    kernelConfig.stream = stream;

    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute[0].val.programmaticStreamSerializationAllowed = launch_with_pdl;
    kernelConfig.attrs = attribute;
    kernelConfig.numAttrs = 1;

    if (fusionOp == AllReduceFusionOp::RESIDUAL_RMS_NORM) {
      FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(
          &kernelConfig, rms_norm_kernel<T, Bias, Residual, Affine, true>, params));
    } else {  // AllReduceFusionOp::RESIDUAL_RMS_PREPOST_NORM
      FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(
          &kernelConfig, rms_pre_post_norm_kernel<T, Bias, Residual, Affine>, params));
    }
  } else {
    cudaLaunchConfig_t kernelConfig = {0};
    kernelConfig.gridDim = cta_num;
    kernelConfig.blockDim = cta_size;
    kernelConfig.dynamicSmemBytes = smem_size;
    kernelConfig.stream = stream;

    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute[0].val.programmaticStreamSerializationAllowed = launch_with_pdl;
    kernelConfig.attrs = attribute;
    kernelConfig.numAttrs = 1;

    if (fusionOp == AllReduceFusionOp::RESIDUAL_RMS_NORM) {
      FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(
          &kernelConfig, rms_norm_kernel<T, Bias, Residual, Affine, false>, params));
    } else {  // AllReduceFusionOp::RESIDUAL_RMS_PREPOST_NORM
      FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(
          &kernelConfig, rms_pre_post_norm_kernel<T, Bias, Residual, Affine>, params));
    }
  }
  return cudaSuccess;
}

template <typename T, int RanksPerNode, bool PushMode>
struct Reducer;

template <typename T, int RanksPerNode>
struct Reducer<T, RanksPerNode, true> {
  constexpr static uint32_t VEC_SIZE = 16 / sizeof(T);
  static __device__ __forceinline__ vec_t<T, VEC_SIZE> allreduce(AllReduceParams<T>& params,
                                                                 int global_offset) {
    int ping = params.barrier_flag % 3;
    int pong = (params.barrier_flag + 2) % 3;
    T const* local_input_buffer = reinterpret_cast<T const*>(params.local_input_buffer_ptr);
    T* local_shared_buffer = reinterpret_cast<T*>(
        params.fusion_params
            .lamport_peer_comm_buffer_ptrs[params.local_rank + ping * MAX_RANKS_PER_NODE]);
    T* local_clean_buffer = reinterpret_cast<T*>(
        params.fusion_params
            .lamport_peer_comm_buffer_ptrs[params.local_rank + pong * MAX_RANKS_PER_NODE]);
    local_input_buffer += global_offset;
    local_shared_buffer += global_offset;
    local_clean_buffer += global_offset;
    T* buffers[RanksPerNode];
#pragma unroll
    for (int ii = 0; ii < RanksPerNode; ++ii) {
      int rank = (params.local_rank + ii) % RanksPerNode;
      buffers[ii] = reinterpret_cast<T*>(
                        params.fusion_params
                            .lamport_peer_comm_buffer_ptrs[rank + ping * MAX_RANKS_PER_NODE]) +
                    global_offset + params.local_rank * params.elts_total;
    }
    vec_t<T, VEC_SIZE> sum_vec, val;
    val.load(local_input_buffer);
#pragma unroll
    for (int ii = 1; ii < RanksPerNode; ++ii) {
      val.store_global_volatile(buffers[ii]);
    }
    sum_vec = val;
#pragma unroll
    for (int ii = 1; ii < RanksPerNode; ++ii) {
      int rank = (params.local_rank + ii) % RanksPerNode;
      set_neg_zero<T>(local_clean_buffer + rank * params.elts_total);
    }
    vec_t<T, VEC_SIZE> vals[RanksPerNode - 1];
    bool done = false;
    while (!done) {
      done = true;
#pragma unroll
      for (int ii = 1; ii < RanksPerNode; ++ii) {
        int rank = (params.local_rank + ii) % RanksPerNode;
        vals[ii - 1].load_global_volatile(local_shared_buffer + rank * params.elts_total);
      }
#pragma unroll
      for (int ii = 0; ii < RanksPerNode - 1; ii++) {
        done &= !has_neg_zero<T, VEC_SIZE>(vals[ii]);
      }
    }

#pragma unroll
    for (int ii = 1; ii < RanksPerNode; ++ii) {
      sum_vec = vec_add<T, VEC_SIZE>(sum_vec, vals[ii - 1]);
    }
    return sum_vec;
  }
};

template <typename T, int RanksPerNode>
struct Reducer<T, RanksPerNode, false> {
  static constexpr uint32_t VEC_SIZE = 16 / sizeof(T);
  static __device__ __forceinline__ vec_t<T, VEC_SIZE> allreduce(AllReduceParams<T>& params,
                                                                 int global_offset) {
    int ping = params.barrier_flag % 3;
    int pong = (params.barrier_flag + 2) % 3;
    T const* local_input_buffer = reinterpret_cast<T const*>(params.local_input_buffer_ptr);
    T* local_shared_buffer = reinterpret_cast<T*>(
        params.fusion_params
            .lamport_peer_comm_buffer_ptrs[params.local_rank + ping * MAX_RANKS_PER_NODE]);
    T* local_clean_buffer = reinterpret_cast<T*>(
        params.fusion_params
            .lamport_peer_comm_buffer_ptrs[params.local_rank + pong * MAX_RANKS_PER_NODE]);
    local_input_buffer += global_offset;
    local_shared_buffer += global_offset;
    local_clean_buffer += global_offset;
    T* buffers[RanksPerNode];
#pragma unroll
    for (int ii = 0; ii < RanksPerNode; ++ii) {
      int rank = (params.local_rank + ii) % RanksPerNode;
      buffers[ii] = reinterpret_cast<T*>(
                        params.fusion_params
                            .lamport_peer_comm_buffer_ptrs[rank + ping * MAX_RANKS_PER_NODE]) +
                    global_offset;
    }
    vec_t<T, VEC_SIZE> sum_vec, val;
    val.load(local_input_buffer);
    val.store_global_volatile(reinterpret_cast<int4*>(local_shared_buffer));
    sum_vec = val;
#pragma unroll
    for (int ii = 1; ii < RanksPerNode; ++ii) {
      do {
        val.load_global_volatile(reinterpret_cast<int4*>(buffers[ii]));
      } while (has_neg_zero<T, VEC_SIZE>(val));
      sum_vec = vec_add<T, VEC_SIZE>(sum_vec, val);
    }
    set_neg_zero<T>(local_clean_buffer);
    return sum_vec;
  }
};

template <int ClusterSize, typename T, int RanksPerNode, bool Bias = false, bool Affine = false,
          bool PushMode = true>
__global__ void lamport_style_one_shot_all_reduce_norm_kernel(AllReduceParams<T> params) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && (__CUDA_ARCH__ < 1200))
  namespace cg = cooperative_groups;
  static_assert(RanksPerNode <= MAX_RANKS_PER_NODE);
  static constexpr uint32_t VEC_SIZE = 16 / sizeof(T);

  cg::cluster_group cluster = cg::this_cluster();

  __shared__ float cluster_acc, cluster_acc_sum;

  int bid = blockIdx.x, tid = threadIdx.x;
  int cluster_id = bid / ClusterSize, cluster_block_rank = bid % ClusterSize;

  int token_id = cluster_id;
  int cluster_offset = token_id * params.fusion_params.hidden_size;
  int block_offset = cluster_block_rank * params.fusion_params.hidden_size / ClusterSize;
  int thread_offset = tid * VEC_SIZE;

  int inner_token_offset = block_offset + thread_offset;
  int global_offset = cluster_offset + inner_token_offset;

  T const* bias_buffer = reinterpret_cast<T const*>(params.fusion_params.bias_buffer);
  T const* residual_buffer = reinterpret_cast<T const*>(params.fusion_params.residual_buffer);
  T const* weight_buffer = reinterpret_cast<T const*>(params.fusion_params.weight_buffer);
  T* local_final_output_buffer = reinterpret_cast<T*>(params.local_output_buffer_ptr);
  T* intermediate_buffer = reinterpret_cast<T*>(params.fusion_params.intermediate_buffer);

  local_final_output_buffer += global_offset;
  intermediate_buffer += global_offset;
  residual_buffer += global_offset;
  bias_buffer += inner_token_offset;
  weight_buffer += inner_token_offset;

  vec_t<T, VEC_SIZE> weight_vec, bias_vec, residual_vec;
  residual_vec.load(residual_buffer);
  if constexpr (Bias) {
    bias_vec.load(bias_buffer);
  }
  if constexpr (Affine) {
    weight_vec.load(weight_buffer);
  }

  cudaGridDependencySynchronize();

  float acc = 0.f;
  vec_t<T, VEC_SIZE> sum_vec;
  sum_vec = Reducer<T, RanksPerNode, PushMode>::allreduce(params, global_offset);

  if constexpr (Bias) {
    sum_vec = vec_add<T, VEC_SIZE>(sum_vec, bias_vec);
  }
  sum_vec = vec_add<T, VEC_SIZE>(sum_vec, residual_vec);
  sum_vec.store(intermediate_buffer);
  acc = accumulate<T, VEC_SIZE>(acc, sum_vec);
  acc = block_reduce_sum(acc);
  if (ClusterSize > 1) {
    if (threadIdx.x == 0) {
      cluster_acc = acc;
    }
    cluster.sync();
    if (threadIdx.x == 0) {
      acc = 0.f;
#pragma unroll
      for (int ii = 0; ii < ClusterSize; ++ii) {
        acc += *cluster.map_shared_rank(&cluster_acc, ii);
      }
      cluster_acc_sum = acc;
    }
    __syncthreads();
    acc = cluster_acc_sum;
    cluster.sync();
  }

  float denom = rsqrtf(acc / params.fusion_params.hidden_size + params.fusion_params.eps);
  sum_vec = rms_norm<T, Affine, VEC_SIZE>(denom, sum_vec, weight_vec);
  sum_vec.store(local_final_output_buffer);

  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

int heuristic_min_warp_number(int tp_size, int hidden_size) {
  if (hidden_size >= 4096) {
    return 4;
  }
  if (tp_size == 2) {
    return 32;
  } else {
    return 16;
  }
}

template <typename T, int RanksPerNode, bool Bias, bool Affine>
cudaError_t lamport_style_one_shot_all_reduce_norm_kernel_launcher(AllReduceParams<T> params,
                                                                   bool launch_with_pdl,
                                                                   cudaStream_t stream) {
  static constexpr uint32_t VEC_SIZE = 16 / sizeof(T);
  FLASHINFER_CHECK(params.fusion_params.hidden_size % VEC_SIZE == 0,
                   "hidden_size must be a multiple of ", VEC_SIZE);
  int threads_per_token = params.fusion_params.hidden_size / VEC_SIZE;
  int warps_per_token = (threads_per_token + details::kWarpSize - 1) / details::kWarpSize;
  int token_num = params.elts_total / params.fusion_params.hidden_size;
  int warp_min_number = heuristic_min_warp_number(RanksPerNode, params.fusion_params.hidden_size);
  int cluster_size = std::min(((warps_per_token + warp_min_number - 1) / warp_min_number),
                              details::kClusterMaxSize);
  int cta_size = warps_per_token / cluster_size * details::kWarpSize;
  FLASHINFER_CHECK(cta_size <= details::kMaxCtaSize, "cta_size must be less than or equal to ",
                   details::kMaxCtaSize);
  int cta_num = token_num * cluster_size;
  cudaLaunchConfig_t kernel_config = {0};
  kernel_config.gridDim = cta_num;
  kernel_config.blockDim = cta_size;
  kernel_config.dynamicSmemBytes = 0;
  kernel_config.stream = stream;

  cudaLaunchAttribute attribute[2];
  attribute[0].id = cudaLaunchAttributeClusterDimension;
  attribute[0].val.clusterDim.x = cluster_size;
  attribute[0].val.clusterDim.y = 1;
  attribute[0].val.clusterDim.z = 1;
  kernel_config.attrs = attribute;
  kernel_config.numAttrs = 1;
  if (launch_with_pdl) {
    attribute[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute[1].val.programmaticStreamSerializationAllowed = 1;
    kernel_config.numAttrs++;
  }
#define LAUNCH_LAMPORT_KERNEL(CLUSTER_SIZE)                                                \
  if (cluster_size == CLUSTER_SIZE) {                                                      \
    FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(                                               \
        &kernel_config,                                                                    \
        lamport_style_one_shot_all_reduce_norm_kernel<CLUSTER_SIZE, T, RanksPerNode, Bias, \
                                                      Affine>,                             \
        params));                                                                          \
    return cudaSuccess;                                                                    \
  }
  LAUNCH_LAMPORT_KERNEL(1);
  LAUNCH_LAMPORT_KERNEL(2);
  LAUNCH_LAMPORT_KERNEL(3);
  LAUNCH_LAMPORT_KERNEL(4);
  LAUNCH_LAMPORT_KERNEL(5);
  LAUNCH_LAMPORT_KERNEL(6);
  LAUNCH_LAMPORT_KERNEL(7);
  LAUNCH_LAMPORT_KERNEL(8);
#undef LAUNCH_LAMPORT_KERNEL
}

template <typename T, int RanksPerNode, bool Bias = false, bool Affine = false,
          bool UseSmem = false>
__global__ void __launch_bounds__(1024, 1)
    one_shot_all_reduce_norm_kernel(AllReduceParams<T> params) {
  static constexpr uint32_t VEC_SIZE = 16 / sizeof(T);

  extern __shared__ uint8_t smem_ptr[];
  T* smem = reinterpret_cast<T*>(smem_ptr);

  int bid = blockIdx.x, tid = threadIdx.x;
  int norm_num = params.elts_total / params.fusion_params.hidden_size;
  int norm_per_block = (norm_num + gridDim.x - 1) / gridDim.x;
  int norm_this_block = std::min(norm_per_block, norm_num - bid * norm_per_block);

  T const* local_input_buffer = reinterpret_cast<T const*>(params.local_input_buffer_ptr);
  T const* bias_buffer = reinterpret_cast<T const*>(params.fusion_params.bias_buffer);
  T const* residual_buffer = reinterpret_cast<T const*>(params.fusion_params.residual_buffer);
  T const* weight_buffer = reinterpret_cast<T const*>(params.fusion_params.weight_buffer);
  T* local_shared_buffer = reinterpret_cast<T*>(params.peer_comm_buffer_ptrs[params.local_rank]);
  T* local_final_output_buffer = reinterpret_cast<T*>(params.local_output_buffer_ptr);
  T* intermediate_buffer = reinterpret_cast<T*>(params.fusion_params.intermediate_buffer);

  int block_offset = bid * norm_per_block * params.fusion_params.hidden_size;
  int thread_offset = tid * VEC_SIZE;

  local_input_buffer += block_offset;
  residual_buffer += block_offset;
  local_shared_buffer += block_offset;
  local_final_output_buffer += block_offset;
  intermediate_buffer += block_offset;

  T* buffers[RanksPerNode];
#pragma unroll
  for (int ii = 0; ii < RanksPerNode; ++ii) {
    int rank = (params.local_rank + ii) % RanksPerNode;
    buffers[ii] = reinterpret_cast<T*>(params.peer_comm_buffer_ptrs[rank]);
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && (__CUDA_ARCH__ < 1200))
  cudaGridDependencySynchronize();
#endif

  for (int offset = thread_offset; offset < norm_this_block * params.fusion_params.hidden_size;
       offset += blockDim.x * VEC_SIZE) {
    *reinterpret_cast<int4*>(&local_shared_buffer[offset]) =
        *reinterpret_cast<int4 const*>(&local_input_buffer[offset]);
  }
  block_barrier(params.peer_barrier_ptrs_in, params.barrier_flag, params.local_rank, RanksPerNode,
                tid, bid, gridDim.x);
  for (int norm_idx = 0; norm_idx < norm_this_block; ++norm_idx) {
    int norm_offset = norm_idx * params.fusion_params.hidden_size;
    float acc = 0.f;
    vec_t<T, VEC_SIZE> sum_vec, weight_vec, bias_vec, residual_vec;
    for (int offset = thread_offset; offset < params.fusion_params.hidden_size;
         offset += blockDim.x * VEC_SIZE) {
      vec_t<T, VEC_SIZE> vals[RanksPerNode];
      sum_vec.fill(T(0));
      if constexpr (Bias) {
        bias_vec.load(bias_buffer + offset);
      }
      residual_vec.load(residual_buffer + norm_offset + offset);
#pragma unroll
      for (int ii = 0; ii < RanksPerNode; ++ii) {
        vals[ii].load(buffers[ii] + block_offset + norm_offset + offset);
      }
#pragma unroll
      for (int ii = 0; ii < RanksPerNode; ++ii) {
        sum_vec = vec_add<T, VEC_SIZE>(sum_vec, vals[ii]);
      }
      if constexpr (Bias) {
        sum_vec = vec_add<T, VEC_SIZE>(sum_vec, bias_vec);
      }
      sum_vec = vec_add<T, VEC_SIZE>(sum_vec, residual_vec);
      sum_vec.store(&intermediate_buffer[norm_offset + offset]);
      acc = accumulate<T, VEC_SIZE>(acc, sum_vec);
      if constexpr (UseSmem) {
        sum_vec.store(&smem[offset]);
      }
    }
    acc = block_reduce_sum(acc);
    float denom = rsqrtf(acc / params.fusion_params.hidden_size + params.fusion_params.eps);
    for (int offset = thread_offset; offset < params.fusion_params.hidden_size;
         offset += blockDim.x * VEC_SIZE) {
      if constexpr (UseSmem) {
        sum_vec.load(&smem[offset]);
      }
      if constexpr (Affine) {
        weight_vec.load(weight_buffer + offset);
      }
      sum_vec = rms_norm<T, Affine, VEC_SIZE>(denom, sum_vec, weight_vec);
      sum_vec.store(&local_final_output_buffer[norm_offset + offset]);
    }
  }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && (__CUDA_ARCH__ < 1200))
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

template <typename T, int RanksPerNode, bool Bias = false, bool Affine = false>
__global__ void __launch_bounds__(1024, 1)
    one_shot_prenorm_all_reduce_norm_kernel(AllReduceParams<T> params) {
  static constexpr uint32_t VEC_SIZE = 16 / sizeof(T);

  int bid = blockIdx.x, tid = threadIdx.x;
  int norm_num = params.elts_total / params.fusion_params.hidden_size;
  int norm_per_block = (norm_num + gridDim.x - 1) / gridDim.x;
  int norm_this_block = std::min(norm_per_block, norm_num - bid * norm_per_block);

  T const* local_input_buffer = reinterpret_cast<T const*>(params.local_input_buffer_ptr);
  T const* bias_buffer = reinterpret_cast<T const*>(params.fusion_params.bias_buffer);
  T const* residual_buffer = reinterpret_cast<T const*>(params.fusion_params.residual_buffer);
  T const* weight_buffer = reinterpret_cast<T const*>(params.fusion_params.weight_buffer);
  T const* weight_buffer_pre_residual_norm =
      reinterpret_cast<T const*>(params.fusion_params.weight_buffer_pre_residual_norm);
  T* local_shared_buffer = reinterpret_cast<T*>(params.peer_comm_buffer_ptrs[params.local_rank]);
  T* local_final_output_buffer = reinterpret_cast<T*>(params.local_output_buffer_ptr);
  T* intermediate_buffer = reinterpret_cast<T*>(params.fusion_params.intermediate_buffer);

  int block_offset = bid * norm_per_block * params.fusion_params.hidden_size;
  int thread_offset = tid * VEC_SIZE;

  local_input_buffer += block_offset;
  residual_buffer += block_offset;
  local_shared_buffer += block_offset;
  local_final_output_buffer += block_offset;
  intermediate_buffer += block_offset;

  T* buffers[RanksPerNode];
#pragma unroll
  for (int ii = 0; ii < RanksPerNode; ++ii) {
    int rank = (params.local_rank + ii) % RanksPerNode;
    buffers[ii] = reinterpret_cast<T*>(params.peer_comm_buffer_ptrs[rank]);
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && (__CUDA_ARCH__ < 1200))
  cudaGridDependencySynchronize();
#endif

  for (int offset = thread_offset; offset < norm_this_block * params.fusion_params.hidden_size;
       offset += blockDim.x * VEC_SIZE) {
    *reinterpret_cast<int4*>(&local_shared_buffer[offset]) =
        *reinterpret_cast<int4 const*>(&local_input_buffer[offset]);
  }
  block_barrier(params.peer_barrier_ptrs_in, params.barrier_flag, params.local_rank, RanksPerNode,
                tid, bid, gridDim.x);
  for (int norm_idx = 0; norm_idx < norm_this_block; ++norm_idx) {
    int norm_offset = norm_idx * params.fusion_params.hidden_size;
    float acc = 0.f;
    float acc_pre_residual_norm = 0.f;
    vec_t<T, VEC_SIZE> sum_vec, weight_vec, bias_vec, residual_vec, weight_vec_pre_residual_norm;
    for (int offset = thread_offset; offset < params.fusion_params.hidden_size;
         offset += blockDim.x * VEC_SIZE) {
      vec_t<T, VEC_SIZE> vals[RanksPerNode];
      sum_vec.fill(T(0));
      if constexpr (Bias) {
        bias_vec.load(bias_buffer + offset);
      }
      residual_vec.load(residual_buffer + norm_offset + offset);
#pragma unroll
      for (int ii = 0; ii < RanksPerNode; ++ii) {
        vals[ii].load(buffers[ii] + block_offset + norm_offset + offset);
      }
#pragma unroll
      for (int ii = 0; ii < RanksPerNode; ++ii) {
        sum_vec = vec_add<T, VEC_SIZE>(sum_vec, vals[ii]);
      }

      if constexpr (Bias) {
        sum_vec = vec_add<T, VEC_SIZE>(sum_vec, bias_vec);
      }

      // norm1 is pre-residual norm.
      acc_pre_residual_norm = accumulate<T, VEC_SIZE>(acc_pre_residual_norm, sum_vec);

      acc_pre_residual_norm = block_reduce_sum(acc_pre_residual_norm);

      float denom_pre_residual_norm = rsqrtf(
          acc_pre_residual_norm / params.fusion_params.hidden_size + params.fusion_params.eps);
      if constexpr (Affine) {
        weight_vec_pre_residual_norm.load(weight_buffer_pre_residual_norm + thread_offset);
      }
      sum_vec = rms_norm<T, Affine, VEC_SIZE>(denom_pre_residual_norm, sum_vec,
                                              weight_vec_pre_residual_norm);

      sum_vec = vec_add<T, VEC_SIZE>(sum_vec, residual_vec);
      sum_vec.store(&intermediate_buffer[norm_offset + offset]);
      acc = accumulate<T, VEC_SIZE>(acc, sum_vec);
    }
    acc = block_reduce_sum(acc);
    float denom = rsqrtf(acc / params.fusion_params.hidden_size + params.fusion_params.eps);
    if constexpr (Affine) {
      weight_vec.load(weight_buffer + thread_offset);
    }
    sum_vec = rms_norm<T, Affine, VEC_SIZE>(denom, sum_vec, weight_vec);
    sum_vec.store(&local_final_output_buffer[norm_offset + thread_offset]);
  }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && (__CUDA_ARCH__ < 1200))
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

template <typename T, int RanksPerNode, bool Bias, bool Affine>
cudaError_t one_shot_all_reduce_norm_kernel_launcher(AllReduceParams<T>& params,
                                                     AllReduceFusionOp fusionOp,
                                                     bool launch_with_pdl, cudaStream_t stream) {
  int token_num = params.elts_total / params.fusion_params.hidden_size;

  if (fusionOp == AllReduceFusionOp::RESIDUAL_RMS_PREPOST_NORM) {
    FLASHINFER_CHECK(params.fusion_params.hidden_size <= 8192,
                     "hidden_size must be less than or equal to 8192");
  }

  if (is_lamport_supported<T>(token_num, params.fusion_params.hidden_size) &&
      (fusionOp != AllReduceFusionOp::RESIDUAL_RMS_PREPOST_NORM)) {
    lamport_style_one_shot_all_reduce_norm_kernel_launcher<T, RanksPerNode, Bias, Affine>(
        params, launch_with_pdl, stream);
  } else {
    static constexpr uint32_t VEC_SIZE = 16 / sizeof(T);
    FLASHINFER_CHECK(params.fusion_params.hidden_size % VEC_SIZE == 0,
                     "hidden_size must be a multiple of ", VEC_SIZE);
    int need_threads = params.fusion_params.hidden_size / VEC_SIZE;
    int cta_size;
    if (need_threads <= details::kMaxCtaSize) {
      cta_size = (need_threads + details::kWarpSize - 1) / details::kWarpSize * details::kWarpSize;
    } else {
      cta_size = details::kMaxCtaSize;
    }
    int norm_num = params.elts_total / params.fusion_params.hidden_size;
    int cta_num = std::min(norm_num, static_cast<int>(MAX_ALL_REDUCE_BLOCKS));
    int smem_size = 0;

    if (cta_size * VEC_SIZE < params.fusion_params.hidden_size) {
      smem_size = params.fusion_params.hidden_size * sizeof(T);
      cudaLaunchConfig_t kernelConfig = {0};
      kernelConfig.gridDim = cta_num;
      kernelConfig.blockDim = cta_size;
      kernelConfig.dynamicSmemBytes = smem_size;
      kernelConfig.stream = stream;

      cudaLaunchAttribute attribute[1];
      attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
      attribute[0].val.programmaticStreamSerializationAllowed = launch_with_pdl;
      kernelConfig.attrs = attribute;
      kernelConfig.numAttrs = 1;
      if (fusionOp == AllReduceFusionOp::RESIDUAL_RMS_NORM) {
        FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(
            &kernelConfig, one_shot_all_reduce_norm_kernel<T, RanksPerNode, Bias, Affine, true>,
            params));
      } else {  // fusionOp == AllReduceFusionOp::RESIDUAL_RMS_PREPOST_NORM
        FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(
            &kernelConfig, one_shot_prenorm_all_reduce_norm_kernel<T, RanksPerNode, Bias, Affine>,
            params));
      }
    } else {
      cudaLaunchConfig_t kernelConfig = {0};
      kernelConfig.gridDim = cta_num;
      kernelConfig.blockDim = cta_size;
      kernelConfig.dynamicSmemBytes = smem_size;
      kernelConfig.stream = stream;

      cudaLaunchAttribute attribute[1];
      attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
      attribute[0].val.programmaticStreamSerializationAllowed = launch_with_pdl;
      kernelConfig.attrs = attribute;
      kernelConfig.numAttrs = 1;

      if (fusionOp == AllReduceFusionOp::RESIDUAL_RMS_NORM) {
        FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(
            &kernelConfig, one_shot_all_reduce_norm_kernel<T, RanksPerNode, Bias, Affine, false>,
            params));
      } else {  // fusionOp == AllReduceFusionOp::RESIDUAL_RMS_PREPOST_NORM
        FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(
            &kernelConfig, one_shot_prenorm_all_reduce_norm_kernel<T, RanksPerNode, Bias, Affine>,
            params));
      }
    }
  }
  return cudaSuccess;
}

template <typename T>
__global__ void lamport_initialize_kernel(T* buffer, size_t size) {
  static constexpr uint32_t VEC_SIZE = 16 / sizeof(T);
  for (size_t offset = (blockIdx.x * blockDim.x + threadIdx.x) * VEC_SIZE; offset < size;
       offset += gridDim.x * blockDim.x * VEC_SIZE) {
    set_neg_zero<T>(&buffer[offset]);
  }
}

template <typename T>
cudaError_t lamport_initialize_kernel_launcher(void* buffer, size_t size, cudaStream_t stream) {
  static constexpr uint32_t VEC_SIZE = 16 / sizeof(T);
  int block_size = 1024;
  int grid_size = (size + 1024 * VEC_SIZE - 1) / (1024 * VEC_SIZE);
  lamport_initialize_kernel<T>
      <<<grid_size, block_size, 0, stream>>>(reinterpret_cast<T*>(buffer), size);
  auto status = cudaGetLastError();
  return status;
}
};  // namespace reduce_fusion

template <typename T, int RANKS_PER_NODE, bool COPY_INPUT = true, bool PUSH_MODE = false>
static __global__ void oneShotAllReduceKernel(AllReduceParams<T> params) {
  // Suppose that two GPUs participate in the AR exchange, and we start four blocks.
  // The message is partitioned into chunks as detailed below:
  //               message
  //       |-------------------|
  // GPU 0 | B0 | B1 | B2 | B3 |
  // GPU 1 | B0 | B1 | B2 | B3 |
  //
  // Here the step-by-step behavior of one block:
  // 1. B0 copies the chunk it  is responsible for, from local_input to shareable buffer
  // 2. B0 on GPU 0 and B0 on GPU 1 wait for each other (block_barrier)
  // 3. B0 on GPU 0 pull and sum the chunk from GPU 1, writes the result to local_output
  //
  // With COPY_INPUT == false, skip step 1. and use gpu_barrier instead of block barrier during
  // step 2. We only to know if the other GPU as arrived at the AR kernel, that would mean that data
  // is ready
  //
  // With PUSH_MODE, we consider that the shared buffer is of size:
  // params.peer_comm_buffer_ptrs: [world_size, world_size, message_size]
  //
  // Here the step-by-step behavior of one block:
  // 1. B0 push the chunk is it responsible for into all other GPUs:
  //    params.peer_comm_buffer_ptrs[:, local_gpu, B0 slice]
  // 2. block sync so the block is shared by other GPUs
  // 3. Reduce along second dimension params.peer_comm_buffer_ptrs[local_gpu, :, B0 slice]

  int const bidx = blockIdx.x;
  int const tidx = threadIdx.x;
  int const grid_size = gridDim.x;

  static constexpr uint32_t VEC_SIZE = 16 / sizeof(T);

  T const* local_input_buffer = reinterpret_cast<T const*>(params.local_input_buffer_ptr);
  T* local_shared_buffer = reinterpret_cast<T*>(params.peer_comm_buffer_ptrs[params.local_rank]);
  T* local_output_buffer = reinterpret_cast<T*>(params.local_output_buffer_ptr);

  // Start and end offsets of the thread
  size_t const chunk_start = bidx * params.elts_per_block + tidx * VEC_SIZE;
  size_t const chunk_end = std::min((bidx + 1) * params.elts_per_block, params.elts_total);

  T* buffers[RANKS_PER_NODE];
#pragma unroll
  for (int ii = 0; ii < RANKS_PER_NODE; ++ii) {
    // buffers[0] is always the local buffers. Helps load balancing reads.
    int rank = (params.local_rank + ii) % RANKS_PER_NODE;
    buffers[ii] = reinterpret_cast<T*>(params.peer_comm_buffer_ptrs[rank]);
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && (__CUDA_ARCH__ < 1200))
  cudaGridDependencySynchronize();
#endif

  if constexpr (PUSH_MODE || COPY_INPUT) {
    // Copy from local buffer to shareable buffer
    for (size_t iter_offset = chunk_start; iter_offset < chunk_end;
         iter_offset += blockDim.x * VEC_SIZE) {
      if constexpr (PUSH_MODE) {
#pragma unroll
        for (int ii = 0; ii < RANKS_PER_NODE; ++ii) {
          *reinterpret_cast<int4*>(
              &buffers[ii][params.local_rank * params.elts_total + iter_offset]) =
              *reinterpret_cast<int4 const*>(&local_input_buffer[iter_offset]);
        }
      } else {
        *reinterpret_cast<int4*>(&local_shared_buffer[iter_offset]) =
            *reinterpret_cast<int4 const*>(&local_input_buffer[iter_offset]);
      }
    }

    // wait for equivalent blocks of other GPUs to have copied data to their shareable buffer
    block_barrier(params.peer_barrier_ptrs_in, params.barrier_flag, params.local_rank,
                  RANKS_PER_NODE, tidx, bidx, grid_size);
  } else {
    // In the non-copy case, we assume that once the kernel has been started, data is ready to be
    // consumed
    multi_gpu_barrier(params.peer_barrier_ptrs_in, params.barrier_flag, params.local_rank,
                      RANKS_PER_NODE, tidx, bidx);
  }

  // Each block accumulates the values from the different GPUs on the same node.
  for (size_t iter_offset = chunk_start; iter_offset < chunk_end;
       iter_offset += blockDim.x * VEC_SIZE) {
    // Iterate over the different ranks/devices on the node to load the values.
    vec_t<T, VEC_SIZE> vals[RANKS_PER_NODE];
#pragma unroll
    for (int ii = 0; ii < RANKS_PER_NODE; ++ii) {
      if constexpr (PUSH_MODE) {
        vals[ii].load(&buffers[params.local_rank][ii * params.elts_total + iter_offset]);
      } else {
        vals[ii].load(&buffers[ii][iter_offset]);
      }
    }

    // Sum the values from the different ranks.
    vec_t<T, VEC_SIZE> sums;
    sums.fill(T(0));
#pragma unroll
    for (int rank = 0; rank < RANKS_PER_NODE; ++rank) {
      // Always reduce from rank 0 to ensure stable reduce order.
      int ii = (rank + RANKS_PER_NODE - params.local_rank) % RANKS_PER_NODE;
      sums = vec_add<T, VEC_SIZE>(sums, vals[ii]);
    }
    // Store to the destination buffer.
    sums.store(&local_output_buffer[iter_offset]);
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && (__CUDA_ARCH__ < 1200))
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

template <typename T, int RANKS_PER_NODE, bool COPY_INPUT = true, bool PUSH_MODE = false,
          bool Bias = false, bool Residual = false>
static __global__ void __launch_bounds__(512, 1) twoShotAllReduceKernel(AllReduceParams<T> params) {
  // Suppose that two GPUs participate in the AR exchange, and we start two blocks.
  // The message is partitioned into chunks as detailed below:
  //               message
  //       |-------------------|
  //       |--GPU 0--|--GPU 1--| (GPU responsibility parts)
  // GPU 0 | B0 | B1 | B0 | B1 |
  // GPU 1 | B0 | B1 | B0 | B1 |
  //
  // Here the step-by-step behavior of one block:
  // 1. B0 copies all chunks is it responsible for, from local_input to shareable buffer
  // 2. B0 on GPU 0 and B0 on GPU 1 wait for each other (block_barrier #0)
  // 3. B0 on GPU 0 gather and sum the B0 chunks from GPU 1, that are in the GPU 0 responsibility
  //    part (the first half of the message, see GPU responsibility row above)
  // 3bis. Likewise, B0 on GPU 1 copies and sum the chunks for GPU 0,
  //       where GPU 1 is responsible: the second half of the message.
  // 4. B0 on GPU 0 and B0 on GPU 1 wait for each other (block_barrier #1)
  // 5. B0 writes result to local_output. It gathers each chunk from its responsible GPU.
  //    For example, here it reads the first chunk from GPU 0 and second chunk from GPU 1.
  //
  // With COPY_INPUT == false, skip step 1. and use gpu_barrier instead of block barrier during
  // step 2. We only to know if the other GPU as arrived at the AR kernel, that would mean that data
  // is ready to be read.
  //
  // Note that compared to one-shot, one block (CTA) writes multiple input chunks and write multiple
  // output chunks. However, it's only responsible for the summation of a single chunk.
  //
  // With PUSH_MODE, we consider that the shared buffer is of size:
  // params.peer_comm_buffer_ptrs: [world_size, world_size, message_size / world_size]
  //
  // Here the step-by-step behavior of one block:
  // 1. B0 push the chunks is it responsible for into the corresponding GPUs:
  //    params.peer_comm_buffer_ptrs[target_gpu, local_gpu, current B0 slice]
  // 2. block sync so the blocks have been shared by other GPUs
  // 3. Reduce along second dimension params.peer_comm_buffer_ptrs[local_gpu, :, B0 slice]
  // 4. block barrier (corresponding blocks have finished reduction)
  // 5. pull and write on local buffer, by reading params.peer_comm_buffer_ptrs[:, 0, B0 slice]
  // (reduction result is
  //    written at index 0 of 2nd dim)

  int const bidx = blockIdx.x;
  int const tidx = threadIdx.x;
  int const grid_size = gridDim.x;

  static constexpr uint32_t VEC_SIZE = 16 / sizeof(T);

  T const* local_input_buffer = reinterpret_cast<T const*>(params.local_input_buffer_ptr);
  T* local_shared_buffer = reinterpret_cast<T*>(params.peer_comm_buffer_ptrs[params.local_rank]);
  T* local_output_buffer = reinterpret_cast<T*>(params.local_output_buffer_ptr);

  size_t const chunk_start = bidx * params.elts_per_block + tidx * VEC_SIZE;
  size_t const chunk_end = min(chunk_start + params.elts_per_block, params.elts_per_rank);

  T* buffers[RANKS_PER_NODE];
  int ranks[RANKS_PER_NODE];
#pragma unroll
  for (int ii = 0; ii < RANKS_PER_NODE; ++ii) {
    // A mapping of the ranks to scatter reads as much as possible
    int rank = (params.local_rank + ii) % RANKS_PER_NODE;
    ranks[ii] = rank;
    buffers[ii] = reinterpret_cast<T*>(params.peer_comm_buffer_ptrs[rank]);
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && (__CUDA_ARCH__ < 1200))
  cudaGridDependencySynchronize();
#endif

  if constexpr (PUSH_MODE || COPY_INPUT) {
    // Copy all blocks from local buffer to shareable buffer
    for (size_t local_offset = chunk_start; local_offset < chunk_end;
         local_offset += blockDim.x * VEC_SIZE) {
#pragma unroll
      for (int ii = 0; ii < RANKS_PER_NODE; ++ii) {
        size_t offset_rank = ranks[ii] * params.elts_per_rank + local_offset;
        if (offset_rank >= params.elts_total) {
          continue;
        }

        if constexpr (PUSH_MODE) {
          *reinterpret_cast<int4*>(
              &buffers[ii][params.local_rank * params.elts_per_rank + local_offset]) =
              *reinterpret_cast<int4 const*>(&local_input_buffer[offset_rank]);
        } else {
          *reinterpret_cast<int4*>(&local_shared_buffer[offset_rank]) =
              *reinterpret_cast<int4 const*>(&local_input_buffer[offset_rank]);
        }
      }
    }
    block_barrier(params.peer_barrier_ptrs_in, params.barrier_flag, params.local_rank,
                  RANKS_PER_NODE, tidx, bidx, grid_size);
  } else {
    // In the non-copy case, we assume that once the kernel has been started, data is ready to be
    // consumed
    multi_gpu_barrier(params.peer_barrier_ptrs_in, params.barrier_flag, params.local_rank,
                      RANKS_PER_NODE, tidx, bidx);
  }

  // Each block accumulates the values from the different GPUs on the same node.
  for (size_t local_offset = chunk_start; local_offset < chunk_end;
       local_offset += blockDim.x * VEC_SIZE) {
    size_t const responsible_block_offset = local_offset + params.rank_offset;

    // Iterate over the different ranks/devices on the node to load the values.
    vec_t<T, VEC_SIZE> vals[RANKS_PER_NODE];
#pragma unroll
    for (int ii = 0; ii < RANKS_PER_NODE; ++ii) {
      if constexpr (PUSH_MODE) {
        vals[ii].load(&local_shared_buffer[ii * params.elts_per_rank + local_offset]);
      } else {
        vals[ii].load(&buffers[ii][responsible_block_offset]);
      }
    }

    // Sum the values from the different ranks.
    vec_t<T, VEC_SIZE> sums;
    sums.fill(T(0));
#pragma unroll
    for (int rank = 0; rank < RANKS_PER_NODE; ++rank) {
      // Always reduce from rank 0 to ensure stable reduce order.
      int ii = (rank + RANKS_PER_NODE - params.local_rank) % RANKS_PER_NODE;
      sums = vec_add<T, VEC_SIZE>(sums, vals[ii]);
    }

    // Store to the local buffer.
    if constexpr (PUSH_MODE) {
      sums.store(&local_shared_buffer[local_offset]);
    } else {
      sums.store(&local_shared_buffer[responsible_block_offset]);
    }
  }

  block_barrier(params.peer_barrier_ptrs_out, params.barrier_flag, params.local_rank,
                RANKS_PER_NODE, tidx, bidx, grid_size);

  // Gather all needed elts from other intra-node ranks
  for (size_t local_offset = chunk_start; local_offset < chunk_end;
       local_offset += blockDim.x * VEC_SIZE) {
#pragma unroll
    for (int ii = 0; ii < RANKS_PER_NODE; ++ii) {
      // use round-robin gathering from other ranks
      size_t offset_rank = ranks[ii] * params.elts_per_rank + local_offset;
      if (offset_rank >= params.elts_total) {
        continue;
      }
      vec_t<T, VEC_SIZE> sums, residual_vec, bias_vec;
      if constexpr (Bias) {
        bias_vec.load(reinterpret_cast<T const*>(params.fusion_params.bias_buffer) +
                      offset_rank % params.fusion_params.hidden_size);
      }
      if constexpr (Residual) {
        residual_vec.load(reinterpret_cast<T const*>(params.fusion_params.residual_buffer) +
                          offset_rank);
      }
      if constexpr (PUSH_MODE) {
        *reinterpret_cast<int4*>(&local_output_buffer[offset_rank]) =
            *reinterpret_cast<int4*>(&buffers[ii][local_offset]);
        sums.load(&buffers[ii][local_offset]);
      } else {
        *reinterpret_cast<int4*>(&local_output_buffer[offset_rank]) =
            *reinterpret_cast<int4*>(&buffers[ii][offset_rank]);
        sums.load(&buffers[ii][offset_rank]);
      }
      if constexpr (Bias) {
        sums = vec_add<T, VEC_SIZE>(sums, bias_vec);
      }
      if constexpr (Residual) {
        sums = vec_add<T, VEC_SIZE>(sums, residual_vec);
      }
      sums.store(&local_output_buffer[offset_rank]);
    }
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && (__CUDA_ARCH__ < 1200))
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

template <typename T>
bool configurationSupported(AllReduceStrategyType algo, size_t msg_size, size_t n_ranks) {
  size_t elts_per_thread = 16 / sizeof(T);
  int const msg_align =
      (algo == AllReduceStrategyType::TWOSHOT) ? n_ranks * elts_per_thread : elts_per_thread;
  bool supported_algo =
      (algo == AllReduceStrategyType::ONESHOT || algo == AllReduceStrategyType::TWOSHOT);
  return supported_algo && (msg_size % msg_align == 0);
}

template <typename T>
std::tuple<int, int> kernelLaunchConfig(AllReduceStrategyType algo, AllReduceParams<T>& params,
                                        size_t elts_per_thread) {
  int blocks_per_grid = 1, threads_per_block = DEFAULT_BLOCK_SIZE;

  switch (algo) {
    case AllReduceStrategyType::ONESHOT: {
      FLASHINFER_CHECK(params.elts_total % elts_per_thread == 0,
                       "hidden_size must be a multiple of ", elts_per_thread);
      size_t const total_threads = round_up(params.elts_total / elts_per_thread, WARP_SIZE);
      threads_per_block = std::min(DEFAULT_BLOCK_SIZE, total_threads);
      blocks_per_grid = std::min(static_cast<size_t>(MAX_ALL_REDUCE_BLOCKS),
                                 ceil_div(total_threads, threads_per_block));
      params.elts_per_block =
          round_up(ceil_div(params.elts_total, blocks_per_grid), elts_per_thread);
      break;
    }
    case AllReduceStrategyType::TWOSHOT: {
      FLASHINFER_CHECK(params.elts_total % (elts_per_thread * params.ranks_per_node) == 0,
                       "hidden_size must be a multiple of ",
                       elts_per_thread * params.ranks_per_node);
      size_t const total_threads =
          round_up(params.elts_total / (elts_per_thread * params.ranks_per_node), WARP_SIZE);

      /*
      threads_per_block = std::min(DEFAULT_BLOCK_SIZE, total_threads);
      blocks_per_grid = std::min(static_cast<size_t>(MAX_ALL_REDUCE_BLOCKS), ceil_div(total_threads,
      threads_per_block));
      */
      while (total_threads % blocks_per_grid != 0 ||
             total_threads / blocks_per_grid > DEFAULT_BLOCK_SIZE) {
        blocks_per_grid += 1;
      }

      threads_per_block = total_threads / blocks_per_grid;

      // NOTE: need to adjust here
      if (blocks_per_grid > MAX_ALL_REDUCE_BLOCKS) {
        size_t iter_factor = 1;
        while (blocks_per_grid / iter_factor > MAX_ALL_REDUCE_BLOCKS ||
               blocks_per_grid % iter_factor) {
          iter_factor += 1;
        }
        blocks_per_grid /= iter_factor;
      }
      params.elts_per_rank = params.elts_total / params.ranks_per_node;
      params.rank_offset = params.local_rank * params.elts_per_rank;
      params.elts_per_block =
          round_up(ceil_div(params.elts_per_rank, blocks_per_grid), elts_per_thread);
      break;
    }
    default:
      FLASHINFER_ERROR("Algorithm not supported here.");
  }

  return std::make_tuple(blocks_per_grid, threads_per_block);
}

template <typename T, int RANKS_PER_NODE, bool PUSH_MODE = false, bool USE_MEMCPY = false,
          bool Bias = false, bool Affine = false>
cudaError_t AllReduceNormKernelLaunch(AllReduceStrategyType algo, AllReduceFusionOp fusionOp,
                                      AllReduceParams<T>& params, bool launch_with_pdl,
                                      cudaStream_t stream) {
  FLASHINFER_CHECK((fusionOp == AllReduceFusionOp::RESIDUAL_RMS_NORM ||
                    fusionOp == AllReduceFusionOp::RESIDUAL_RMS_PREPOST_NORM),
                   "Unsupported AllReduceFusionOp: %d", static_cast<int>(fusionOp));
  if (algo == AllReduceStrategyType::ONESHOT) {
    return reduce_fusion::one_shot_all_reduce_norm_kernel_launcher<T, RANKS_PER_NODE, Bias, Affine>(
        params, fusionOp, launch_with_pdl, stream);
  } else {
    FLASHINFER_CHECK(!(USE_MEMCPY && PUSH_MODE), "Memcpy cannot be used with PUSH_MODE.");
    size_t elts_per_thread = 16 / sizeof(T);
    auto [blocks_per_grid, threads_per_block] =
        kernelLaunchConfig<T>(algo, params, elts_per_thread);
    if (USE_MEMCPY) {
      cudaMemcpyAsync(params.peer_comm_buffer_ptrs[params.local_rank],
                      params.local_input_buffer_ptr, params.elts_total * sizeof(T),
                      cudaMemcpyDeviceToDevice, stream);
    }
    auto output_ptr = params.local_output_buffer_ptr;
    params.local_output_buffer_ptr = params.fusion_params.intermediate_buffer;

    cudaLaunchConfig_t kernelConfig = {0};
    kernelConfig.gridDim = blocks_per_grid;
    kernelConfig.blockDim = threads_per_block;
    kernelConfig.dynamicSmemBytes = 0;
    kernelConfig.stream = stream;

    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute[0].val.programmaticStreamSerializationAllowed = launch_with_pdl;
    kernelConfig.attrs = attribute;
    kernelConfig.numAttrs = 1;

    FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(
        &kernelConfig,
        twoShotAllReduceKernel<T, RANKS_PER_NODE, !USE_MEMCPY, PUSH_MODE, Bias, true>, params));

    params.local_output_buffer_ptr = output_ptr;
    return reduce_fusion::rms_norm_kernel_launcher<T, false, false, Affine>(
        params, fusionOp, launch_with_pdl, stream);
  }
  return cudaSuccess;
}

template <typename T, int RANKS_PER_NODE, bool PUSH_MODE = false, bool USE_MEMCPY = false>
cudaError_t AllReduceNormDispatch(AllReduceStrategyType algo, AllReduceFusionOp fusionOp,
                                  AllReduceParams<T>& params, bool launch_with_pdl,
                                  cudaStream_t stream) {
  if (params.fusion_params.bias_buffer && params.fusion_params.weight_buffer) {
    return AllReduceNormKernelLaunch<T, RANKS_PER_NODE, PUSH_MODE, USE_MEMCPY, true, true>(
        algo, fusionOp, params, launch_with_pdl, stream);
  } else if (params.fusion_params.bias_buffer && !params.fusion_params.weight_buffer) {
    return AllReduceNormKernelLaunch<T, RANKS_PER_NODE, PUSH_MODE, USE_MEMCPY, true, false>(
        algo, fusionOp, params, launch_with_pdl, stream);
  } else if (!params.fusion_params.bias_buffer && params.fusion_params.weight_buffer) {
    return AllReduceNormKernelLaunch<T, RANKS_PER_NODE, PUSH_MODE, USE_MEMCPY, false, true>(
        algo, fusionOp, params, launch_with_pdl, stream);
  } else {
    return AllReduceNormKernelLaunch<T, RANKS_PER_NODE, PUSH_MODE, USE_MEMCPY, false, false>(
        algo, fusionOp, params, launch_with_pdl, stream);
  }
  return cudaSuccess;
}

template <typename T, int RANKS_PER_NODE, bool PUSH_MODE = false, bool USE_MEMCPY = false>
cudaError_t AllReduceDispatch(AllReduceStrategyType algo, AllReduceFusionOp fusionOp,
                              AllReduceParams<T>& params, bool launch_with_pdl,
                              cudaStream_t stream) {
  FLASHINFER_CHECK(fusionOp == AllReduceFusionOp::NONE, "Unsupported AllReduceFusionOp: %d",
                   static_cast<int>(fusionOp));
  FLASHINFER_CHECK(!(USE_MEMCPY && PUSH_MODE), "Memcpy cannot be used with PUSH_MODE.");
  size_t elts_per_thread = 16 / sizeof(T);
  auto [blocks_per_grid, threads_per_block] = kernelLaunchConfig(algo, params, elts_per_thread);
  if (USE_MEMCPY) {
    cudaMemcpyAsync(params.peer_comm_buffer_ptrs[params.local_rank], params.local_input_buffer_ptr,
                    params.elts_total * sizeof(T), cudaMemcpyDeviceToDevice, stream);
  }
  if (algo == AllReduceStrategyType::ONESHOT) {
    auto* kernel_instance = &oneShotAllReduceKernel<T, RANKS_PER_NODE, !USE_MEMCPY, PUSH_MODE>;
    cudaLaunchConfig_t config;
    config.gridDim = blocks_per_grid;
    config.blockDim = threads_per_block;
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute[0].val.programmaticStreamSerializationAllowed = launch_with_pdl;
    config.attrs = attribute;
    config.numAttrs = 1;
    cudaLaunchKernelEx(&config, kernel_instance, params);
  } else {
    auto* kernel_instance = &twoShotAllReduceKernel<T, RANKS_PER_NODE, !USE_MEMCPY, PUSH_MODE>;
    cudaLaunchConfig_t config;
    config.gridDim = blocks_per_grid;
    config.blockDim = threads_per_block;
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute[0].val.programmaticStreamSerializationAllowed = launch_with_pdl;
    config.attrs = attribute;
    config.numAttrs = 1;
    cudaLaunchKernelEx(&config, kernel_instance, params);
  }
  return cudaSuccess;
}

template <typename T, int RANKS_PER_NODE, bool PUSH_MODE = false, bool USE_MEMCPY = false>
cudaError_t AllReduceDispatchMemcpy(AllReduceStrategyType algo, AllReduceFusionOp fusionOp,
                                    AllReduceParams<T>& params, bool launch_with_pdl,
                                    cudaStream_t stream) {
  if (fusionOp == AllReduceFusionOp::NONE) {
    FLASHINFER_LOG_DEBUG("AllReduceDispatch enabled");
    return AllReduceDispatch<T, RANKS_PER_NODE, PUSH_MODE, USE_MEMCPY>(algo, fusionOp, params,
                                                                       launch_with_pdl, stream);
  } else {
    FLASHINFER_LOG_DEBUG("AllReduceNormDispatch enabled");
    return AllReduceNormDispatch<T, RANKS_PER_NODE, PUSH_MODE, USE_MEMCPY>(algo, fusionOp, params,
                                                                           launch_with_pdl, stream);
  }
}

template <typename T, int RANKS_PER_NODE, bool PUSH_MODE = false>
cudaError_t AllReduceDispatchPushMode(AllReduceStrategyType algo, AllReduceStrategyConfig config,
                                      AllReduceFusionOp fusionOp, AllReduceParams<T>& params,
                                      bool launch_with_pdl, cudaStream_t stream) {
  if (static_cast<std::underlying_type_t<AllReduceStrategyConfig>>(config) &
      static_cast<std::underlying_type_t<AllReduceStrategyConfig>>(
          AllReduceStrategyConfig::USE_MEMCPY)) {
    return AllReduceDispatchMemcpy<T, RANKS_PER_NODE, PUSH_MODE, true>(algo, fusionOp, params,
                                                                       launch_with_pdl, stream);
  } else {
    return AllReduceDispatchMemcpy<T, RANKS_PER_NODE, PUSH_MODE, false>(algo, fusionOp, params,
                                                                        launch_with_pdl, stream);
  }
}

template <typename T, int RANKS_PER_NODE>  //, bool USE_MEMCPY = false, bool PUSH_MODE = false>
cudaError_t AllReduceDispatchRanksPerNode(AllReduceStrategyType algo,
                                          AllReduceStrategyConfig config,
                                          AllReduceFusionOp fusionOp, AllReduceParams<T>& params,
                                          bool launch_with_pdl, cudaStream_t stream) {
  if (static_cast<std::underlying_type_t<AllReduceStrategyConfig>>(config) &
      static_cast<std::underlying_type_t<AllReduceStrategyConfig>>(
          AllReduceStrategyConfig::PUSH_MODE)) {
    return AllReduceDispatchPushMode<T, RANKS_PER_NODE, true>(algo, config, fusionOp, params,
                                                              launch_with_pdl, stream);
  } else {
    return AllReduceDispatchPushMode<T, RANKS_PER_NODE, false>(algo, config, fusionOp, params,
                                                               launch_with_pdl, stream);
  }
}

template <typename T>
cudaError_t AllReduceDispatchType(AllReduceParams<T>& params, AllReduceStrategyType strat,
                                  AllReduceStrategyConfig config, AllReduceFusionOp fusionOp,
                                  bool launch_with_pdl, cudaStream_t stream) {
  switch (params.ranks_per_node) {
    case 2:
      return AllReduceDispatchRanksPerNode<T, /*RANKS_PER_NODE=*/2>(strat, config, fusionOp, params,
                                                                    launch_with_pdl, stream);
    case 4:
      return AllReduceDispatchRanksPerNode<T, 4>(strat, config, fusionOp, params, launch_with_pdl,
                                                 stream);
    case 6:
      return AllReduceDispatchRanksPerNode<T, 6>(strat, config, fusionOp, params, launch_with_pdl,
                                                 stream);
    case 8:
      return AllReduceDispatchRanksPerNode<T, 8>(strat, config, fusionOp, params, launch_with_pdl,
                                                 stream);
    case 16:
      return AllReduceDispatchRanksPerNode<T, 16>(strat, config, fusionOp, params, launch_with_pdl,
                                                  stream);
    default:
      FLASHINFER_ERROR("Custom all reduce only supported on {2, 4, 6, 8, 16} GPUs per node.");
  }
  return cudaSuccess;
}

template <typename T>
cudaError_t customAllReduce(AllReduceParams<T>& params, AllReduceStrategyType strat,
                            AllReduceStrategyConfig config, AllReduceFusionOp fusionOp,
                            bool launch_with_pdl, cudaStream_t stream) {
  FLASHINFER_CHECK(configurationSupported<T>(strat, params.elts_total, params.ranks_per_node),
                   "Custom all-reduce configuration unsupported");

  return AllReduceDispatchType<T>(params, strat, config, fusionOp, launch_with_pdl, stream);
}

template <typename T>
cudaError_t lamportInitialize(void* buffer, size_t size, cudaStream_t stream) {
  if (size == 0) {
    return cudaSuccess;
  }
  FLASHINFER_LOG_INFO("lamportInitialize start: buffer: {}, size: {}", buffer, size);
  return reduce_fusion::lamport_initialize_kernel_launcher<T>(buffer, size, stream);
}

// lamport: 3 buffers for synchronization
template <typename T>
cudaError_t lamportInitializeAll(void* buffer_0, void* buffer_1, void* buffer_2, size_t size,
                                 cudaStream_t stream) {
  auto status = lamportInitialize<T>(buffer_0, size / sizeof(T), stream);
  FLASHINFER_CHECK(status == cudaSuccess, "lamportInitialize failed with error code " +
                                              std::string(cudaGetErrorString(status)));

  status = lamportInitialize<T>(buffer_1, size / sizeof(T), stream);
  FLASHINFER_CHECK(status == cudaSuccess, "lamportInitialize failed with error code " +
                                              std::string(cudaGetErrorString(status)));

  status = lamportInitialize<T>(buffer_2, size / sizeof(T), stream);
  FLASHINFER_CHECK(status == cudaSuccess, "lamportInitialize failed with error code " +
                                              std::string(cudaGetErrorString(status)));
  cudaDeviceSynchronize();
  return cudaSuccess;
}

}  // namespace trtllm_allreduce
}  // namespace flashinfer
