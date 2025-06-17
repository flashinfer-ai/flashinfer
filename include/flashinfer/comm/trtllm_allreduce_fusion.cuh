#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cuda/std/optional>
#include <tuple>
#include <type_traits>

#include "../exception.h"
#include "../logging.h"
#include "../utils.cuh"
#include "../vec_dtypes.cuh"

namespace flashinfer {

namespace trtllm_allreduce_fusion {

namespace details {

static constexpr int kBytesPerAccess = 16;
static constexpr int kOneShotMaxToken = 128;
static constexpr int kBarrierFlagCount = 256;

}  // namespace details

namespace utils {

#define FINAL_MASK 0xffffffff

template <typename T, int NUM>
__inline__ __device__ T warpReduceSumV2(T* val) {
#pragma unroll
  for (int i = 0; i < NUM; i++) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
      val[i] += __shfl_xor_sync(FINAL_MASK, val[i], mask, 32);
  }
  return (T)(0.0f);
}

template <typename T, int NUM>
__inline__ __device__ T blockReduceSumV2(T* val) {
  static __shared__ T shared[NUM][33];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  warpReduceSumV2<T, NUM>(val);

  if (lane == 0) {
#pragma unroll
    for (int i = 0; i < NUM; i++) {
      shared[i][wid] = val[i];
    }
  }

  __syncthreads();

  bool is_mask = threadIdx.x < (blockDim.x / 32.f);
#pragma unroll
  for (int i = 0; i < NUM; i++) {
    val[i] = is_mask ? shared[i][lane] : (T)(0.0f);
  }
  warpReduceSumV2<T, NUM>(val);
  return (T)0.0f;
}

inline int getSMVersion() {
  int device{-1};
  FLASHINFER_CUDA_CALL(cudaGetDevice(&device));
  int sm_major = 0;
  int sm_minor = 0;
  FLASHINFER_CUDA_CALL(
      cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, device));
  FLASHINFER_CUDA_CALL(
      cudaDeviceGetAttribute(&sm_minor, cudaDevAttrComputeCapabilityMinor, device));
  return sm_major * 10 + sm_minor;
}

}  // namespace utils

enum class AllReduceFusionPattern : int {
  kAllReduce = 0,
  kARResidualRMSNorm = 1,
  kARResidualRMSNormFP8Quant = 2,
  kARResidualRMSNormFP4Quant = 3,
  // The difference between these two and the standard version is that the NormOut version outputs
  // the result of the norm.
  kARResidualRMSNormOutFP8Quant = 4,
  kARResidualRMSNormOutFP4Quant = 5
};

enum class FP4QuantizationSFLayout {
  // Block scale factors are stored in swizzled layout for cutlass FP4 kernel. Scale factor
  // blocks are organized in 512-byte blocks in global memory, with each block having 128x4 FP8
  // values. The SF matrix dimensions are therefore padded - rows to the nearest multiple of 128 and
  // columns to the nearest multiple of 4.
  //
  // The scale factor block rows map to data block rows in an interleaved pattern:
  // For a scale factor row 'i', it maps to data block row: (i % 4) * 32 + (i / 4)
  // Column 'j' in the scale factor block corresponds to scaling the j-th block in the data tensor.
  //
  // Please refer to https://nvbugs/4165523 for more details about the swizzled layout.
  SWIZZLED,
  // Block scale factors are stored in linear layout (row-major). This is used in some trtllm-gen
  // kernels standard.
  LINEAR
};

template <typename T>
struct AllReduceFusionParams {
  int nranks;
  int rank;
  int size;
  int hidden_dim;
  void** workspace;
  void* allreduce_in;
  void* allreduce_out;
  void* residual_in;
  void* residual_out;
  void* norm_out;
  void* quant_out;
  void* scale_out;
  void* rms_gamma;
  float rms_eps;
  float scale_factor;
  bool use_oneshot;
  FP4QuantizationSFLayout layout = FP4QuantizationSFLayout::SWIZZLED;
  cudaStream_t stream;
  AllReduceFusionPattern pattern;
  bool trigger_completion_at_end = true;
};

template <int NRanks>
struct SyncComm {
  __device__ __forceinline__ SyncComm(void** workspace) {
    counter_ptr = &reinterpret_cast<int*>(workspace[NRanks * 3])[0];
    flag_ptr = &reinterpret_cast<int*>(workspace[NRanks * 3])[1];
    flag_value = *flag_ptr;
    for (int r = 0; r < NRanks; ++r) {
      comm_bufs[r] = workspace[r];
      barrier_flags[r] = workspace[NRanks + r];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      atomicAdd(counter_ptr, 1);
    }
  }

  __device__ __forceinline__ void update(int new_flag_value) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      while (*reinterpret_cast<int volatile*>(counter_ptr) != gridDim.x) {
      }
      *flag_ptr = new_flag_value;
      *counter_ptr = 0;
    }
  }

  int* counter_ptr;
  int* flag_ptr;
  void* comm_bufs[NRanks];
  void* barrier_flags[NRanks];
  int flag_value;
};

template <int NRanks>
struct LamportComm {
  __device__ __forceinline__ LamportComm(void** workspace, int rank) {
    counter_ptr = &reinterpret_cast<int*>(workspace[NRanks * 3])[0];
    flag_ptr = &reinterpret_cast<int*>(workspace[NRanks * 3])[2];
    clear_ptr = &reinterpret_cast<int*>(workspace[NRanks * 3])[4];
    flag_value = *flag_ptr;
    int comm_size = reinterpret_cast<int*>(workspace[NRanks * 3])[3];
    clear_size = *clear_ptr;
    int data_offset = flag_value % 3;
    int clear_offset = (flag_value + 2) % 3;
    for (int r = 0; r < NRanks; ++r) {
      data_bufs[r] =
          reinterpret_cast<uint8_t*>(workspace[2 * NRanks + r]) + data_offset * comm_size;
    }
    clear_buf = reinterpret_cast<uint8_t*>(workspace[2 * NRanks + rank]) + clear_offset * comm_size;
    __syncthreads();
    if (threadIdx.x == 0) {
      atomicAdd(counter_ptr, 1);
    }
  }

  __device__ __forceinline__ void update(int new_clear_size) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      while (*reinterpret_cast<int volatile*>(counter_ptr) != gridDim.x) {
      }
      *flag_ptr = (flag_value + 1) % 3;
      *clear_ptr = new_clear_size;
      *counter_ptr = 0;
    }
  }

  int* counter_ptr;
  int* flag_ptr;
  int* clear_ptr;
  uint8_t* data_bufs[NRanks];
  uint8_t* clear_buf;
  int clear_size;
  int flag_value;
};

template <int NRanks>
class Barrier {
 public:
  __device__ __forceinline__ Barrier(int rank, SyncComm<NRanks> const& comm) {
    if (threadIdx.x < NRanks) {
      m_flag_value = comm.flag_value;
      int current_rank = rank;
      int target_rank = threadIdx.x;
      m_target_flag = reinterpret_cast<int*>(comm.barrier_flags[target_rank]) + current_rank;
      m_current_flag = reinterpret_cast<int*>(comm.barrier_flags[current_rank]) +
                       blockIdx.x * NRanks + target_rank;
    }
  }

  __device__ __forceinline__ void sync() {
    __syncthreads();
    if (threadIdx.x < NRanks) {
      m_flag_value = next_flag(m_flag_value);
      // To avoid the ABA problem, we need to synchronize the correct flag value to all
      // barrier_flags, even if the corresponding CTA has not been launched.
      for (int flag_idx = blockIdx.x; flag_idx < details::kBarrierFlagCount;
           flag_idx += gridDim.x) {
        st_flag(m_target_flag + flag_idx * NRanks, m_flag_value);
      }
      while (ld_flag(m_current_flag) == prev_flag(m_flag_value)) {
      }
    }
    __syncthreads();
  }

 protected:
  __device__ __forceinline__ void st_flag(int* addr, int flag) {
    asm volatile("st.global.release.sys.b32 [%1], %0;" ::"r"(flag), "l"(addr));
  }

  __device__ __forceinline__ int ld_flag(int* addr) {
    int flag;
    asm volatile("ld.global.acquire.sys.b32 %0, [%1];" : "=r"(flag) : "l"(addr));
    return flag;
  }

  __device__ __forceinline__ int next_flag(int flag) { return flag == 2 ? 0 : flag + 1; }

  __device__ __forceinline__ int prev_flag(int flag) { return flag == 0 ? 2 : flag - 1; }

 public:
  int m_flag_value;

 private:
  int* m_target_flag;
  int* m_current_flag;
};

template <AllReduceFusionPattern Pattern, typename T>
class FusedOp {
  static constexpr int VEC_SIZE = details::kBytesPerAccess / sizeof(T);

 public:
  __device__ __forceinline__ FusedOp(AllReduceFusionParams<T> const& params, int access_id,
                                     int access_id_in_token)
      : m_params(params), m_access_id(access_id), m_access_id_in_token(access_id_in_token) {
    if constexpr (HasRMSNorm<Pattern>) {
      m_gamma_val = reinterpret_cast<float4*>(params.rms_gamma)[m_access_id_in_token];
    }
    if constexpr (HasResidual<Pattern>) {
      m_residual_val = reinterpret_cast<float4*>(params.residual_in)[m_access_id];
    }
    if constexpr (GetQuantType<Pattern> == QuantType::kFP8) {
      m_scale_factor = 1.f / *params.scale_factor;
    } else if constexpr (GetQuantType<Pattern> == QuantType::kFP4) {
      m_scale_factor = *params.scale_factor;
    }
  }

  template <typename T>
  __device__ __forceinline__ void update(int access_id) {
    if (m_access_id != access_id) {
      m_access_id = access_id;
      if constexpr (HasResidual<Pattern>) {
        m_residual_val.load(reinterpret_cast<T*>(m_params.residual_in) + m_access_id * VEC_SIZE);
      }
    }
  }

  template <typename T, uint32_t VEC_SIZE>
  __device__ __forceinline__ void operator()(vec_t<T, VEC_SIZE> val, int token_id) {
    if constexpr (HasAllReduceOut<Pattern>) {
      val.store(reinterpret_cast<T*>(m_params.allreduce_out) + m_access_id * VEC_SIZE);
    }
    if constexpr (HasResidual<Pattern>) {
      val = vec_add<T, VEC_SIZE>(val, m_residual_val);
      if constexpr (HasResidualOut<Pattern>) {
        val.store(reinterpret_cast<T*>(m_params.residual_out) + m_access_id * VEC_SIZE);
      }
    }
    if constexpr (HasRMSNorm<Pattern>) {
      val = rms_norm(val, m_gamma_val);
      if constexpr (HasNormOut<Pattern>) {
        val.store(reinterpret_cast<T*>(m_params.norm_out) + m_access_id * VEC_SIZE);
      }
    }
    // todo(yingyi): add quant support
    //     if constexpr (GetQuantType<Pattern> == QuantType::kFP4) {
    //       constexpr int SF_VEC_SIZE = 16;
    //       using PackedVec = PackedVec<DType>;
    //       PackedVec pack_val = *reinterpret_cast<PackedVec const*>(&val);
    //       auto sf_out = cvt_quant_to_fp4_get_sf_out_offset<uint32_t, 2, SF_VEC_SIZE>(
    //           std::nullopt, token_id, m_access_id_in_token, std::nullopt, m_params.hidden_dim,
    //           reinterpret_cast<uint32_t*>(m_params.scale_out), m_params.layout);
    //       reinterpret_cast<uint32_t*>(m_params.quant_out)[m_access_id] =
    //           cvt_warp_fp16_to_fp4<DType, SF_VEC_SIZE, false>(pack_val, m_scale_factor, sf_out);
    //     } else if constexpr (GetQuantType<Pattern> == QuantType::kFP8) {
    //       using PackedQuantizedType = std::conditional_t<std::is_same_v<DType, float>, float,
    //       float2>; PackedQuantizedType ret;
    // #pragma unroll
    //       for (int i = 0; i < kMathCount; ++i) {
    //         reinterpret_cast<__nv_fp8_e4m3*>(&ret)[i] = static_cast<__nv_fp8_e4m3>(
    //             static_cast<float>(reinterpret_cast<DType*>(&val)[i]) * m_scale_factor);
    //       }
    //       reinterpret_cast<PackedQuantizedType*>(m_params.quant_out)[m_access_id] = ret;
    //     } else {
    //       static_assert(GetQuantType<Pattern> == QuantType::kNone, "Invalid quant type");
    //     }
  }

 protected:
  __device__ __forceinline__ vec_t<T, VEC_SIZE> rms_norm(vec_t<T, VEC_SIZE> const& residual,
                                                         vec_t<T, VEC_SIZE> const& gamma) {
    __shared__ float s_val;
    vec_t<T, VEC_SIZE> norm_out;
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      float v = static_cast<float>(reinterpret_cast<T const*>(&residual)[i]);
      acc += v * v;
    }
    utils::blockReduceSumV2<float, 1>(&acc);
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cg::cluster_group cluster = cg::this_cluster();
    if (cluster.num_blocks() > 1) {
      if (threadIdx.x == 0) {
        s_val = acc;
        acc = 0.f;
      }
      cluster.sync();
      if (threadIdx.x == 0) {
        for (int i = 0; i < cluster.num_blocks(); ++i) {
          acc += *cluster.map_shared_rank(&s_val, i);
        }
      }
      cluster.sync();
    }
#endif
    if (threadIdx.x == 0) {
      s_val = rsqrtf(acc / m_params.hidden_dim + m_params.rms_eps);
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < kMathCount; ++i) {
      reinterpret_cast<DType*>(&norm_out)[i] =
          static_cast<DType>(static_cast<float>(reinterpret_cast<DType const*>(&residual)[i]) *
                             s_val * static_cast<float>(reinterpret_cast<DType const*>(&gamma)[i]));
    }
    return norm_out;
  }

 private:
  AllReduceFusionParams<T> const& m_params;
  int m_access_id;
  int m_access_id_in_token;
  float m_scale_factor;
  vec_t<T, VEC_SIZE> m_residual_val;
  vec_t<T, VEC_SIZE> m_gamma_val;
};

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
  vec_t<T, details::kBytesPerAccess / sizeof(T)> val;
  val.fill(neg_zero_v<T>);
  val.store_global_volatile(addr);
}

template <typename T, uint32_t VEC_SIZE, int NRanks, bool Fp32Acc>
__device__ __forceinline__ vec_t<T, VEC_SIZE> allreduce_sum(vec_t<T, VEC_SIZE>* vals) {
  if constexpr (Fp32Acc) {
    static_assert(!std::is_same_v<T, float>);
    float acc_f32[VEC_SIZE];
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      acc_f32[i] = static_cast<float>(reinterpret_cast<T*>(&vals[0])[i]);
    }
#pragma unroll
    for (int r = 1; r < NRanks; ++r) {
#pragma unroll
      for (int i = 0; i < VEC_SIZE; ++i) {
        acc_f32[i] += static_cast<float>(reinterpret_cast<T*>(&vals[r])[i]);
      }
    }
    vec_t<T, VEC_SIZE> acc;
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      acc[i] = static_cast<T>(acc_f32[i]);
    }
    return acc;
  } else {
    vec_t<T, VEC_SIZE> acc = vals[0];
#pragma unroll
    for (int r = 1; r < NRanks; ++r) {
      acc = vec_add<T, VEC_SIZE>(acc, vals[r]);
    }
    return acc;
  }
}

template <typename T>
class IndexHelper {
 public:
  __device__ __forceinline__ IndexHelper(AllReduceFusionParams<T> const& params) {
    static constexpr int VEC_SIZE = details::kBytesPerAccess / sizeof(T);
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();
    cg::grid_group grid = cg::this_grid();
    token_id = grid.cluster_rank();
    access_id_in_token = cluster.thread_rank();
    token_stride = grid.num_clusters();
#else
    token_id = blockIdx.x;
    access_id_in_token = threadIdx.x;
    token_stride = gridDim.x;
#endif
    access_id = token_id * params.hidden_dim / VEC_SIZE + access_id_in_token;
    access_stride = token_stride * params.hidden_dim / VEC_SIZE;
    tot_access = params.size / VEC_SIZE;
  }

  int token_id;
  int access_id_in_token;
  int token_stride;
  int access_id;
  int access_stride;
  int tot_access;
};

template <AllReduceFusionPattern Pattern, typename T, int NRanks, bool Fp32Acc,
          bool TriggerCompletionAtEnd = true>
__global__ void allreduce_fusion_kernel_oneshot_lamport(AllReduceFusionParams<T> params) {
  static constexpr int VEC_SIZE = details::kBytesPerAccess / sizeof(T);
  IndexHelper<T> index_helper(params);
  int token_id = index_helper.token_id;
  int access_id_in_token = index_helper.access_id_in_token;
  int token_stride = index_helper.token_stride;
  int access_id = index_helper.access_id;
  int access_stride = index_helper.access_stride;
  int tot_access = index_helper.tot_access;
  vec_t<T, VEC_SIZE> clear_vec;
  clear_vec.fill(neg_zero_v<T>);
  FusedOp<Pattern, T> fused_op(params, access_id, access_id_in_token);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
  if constexpr (!TriggerCompletionAtEnd) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
#endif
  LamportComm<NRanks> comm(params.workspace, params.rank);
  int clear_access = comm.clear_size / VEC_SIZE;

  for (int idx = access_id; idx < tot_access; idx += access_stride) {
    vec_t<T, VEC_SIZE> val;
    val.load(reinterpret_cast<T*>(params.allreduce_in) + idx * VEC_SIZE);
    remove_neg_zero(val);
#pragma unroll
    for (int r = 0; r < NRanks; ++r) {
      // Push data to other ranks
      val.store(reinterpret_cast<T*>(comm.data_bufs[r]) +
                (params.rank * tot_access + idx) * VEC_SIZE);
    }
  }
  for (int idx = access_id; idx < clear_access; idx += access_stride) {
    // Clear comm buffer that previous kernel used
    clear_vec.store(reinterpret_cast<T*>(comm.clear_buf) + idx * VEC_SIZE);
  }

  for (int idx = access_id, tidx = token_id; idx < tot_access;
       idx += access_stride, tidx += token_stride) {
    fused_op.update(idx);
    vec_t<T, VEC_SIZE> vals[NRanks];
    bool done = false;
    while (!done) {
      done = true;
#pragma unroll
      for (int r = 0; r < NRanks; ++r) {
        // LDG.128 from local rank
        vals[r].load_global_volatile(reinterpret_cast<T*>(comm.data_bufs[params.rank]) +
                                     (r * tot_access + idx) * VEC_SIZE);
        done &= !has_neg_zero(vals[r]);
      }
    }
    vec_t<T, VEC_SIZE> sum_val = allreduce_sum<T, VEC_SIZE, NRanks, Fp32Acc>(vals);
    fused_op(sum_val, tidx);
  }

  comm.update(params.size * NRanks);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  if constexpr (TriggerCompletionAtEnd) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
#endif
}

template <AllReduceFusionPattern Pattern, typename T, int NRanks, bool Fp32Acc>
__global__ void allreduce_fusion_kernel_twoshot_sync(AllReduceFusionParams<T> params,
                                                     std::array<int, NRanks> begin_tokens,
                                                     std::array<int, NRanks> token_num_per_ranks) {
  static constexpr int VEC_SIZE = details::kBytesPerAccess / sizeof(T);
  IndexHelper<T> index_helper(params);
  int token_id = index_helper.token_id;
  int access_id_in_token = index_helper.access_id_in_token;
  int token_stride = index_helper.token_stride;
  int access_id = index_helper.access_id;
  int access_stride = index_helper.access_stride;
  int tot_access = index_helper.tot_access;
  FusedOp<Pattern, T> fused_op(params, access_id, access_id_in_token);
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif
  SyncComm<NRanks> comm(params.workspace);
#pragma unroll
  for (int r = 0; r < NRanks; ++r) {
    int comm_access_id = access_id + begin_tokens[r] * params.hidden_dim / VEC_SIZE;
    int comm_tot_access = (begin_tokens[r] + token_num_per_ranks[r]) * params.hidden_dim / VEC_SIZE;
    for (int idx = comm_access_id; idx < comm_tot_access; idx += access_stride) {
      reinterpret_cast<float4*>(comm.comm_bufs[params.rank])[idx] =
          reinterpret_cast<float4*>(params.allreduce_in)[idx];
    }
  }
  Barrier<NRanks> barrier(params.rank, comm);
  barrier.sync();
  int comm_access_id = access_id + begin_tokens[params.rank] * params.hidden_dim / VEC_SIZE;
  int comm_tot_access =
      (begin_tokens[params.rank] + token_num_per_ranks[params.rank]) * params.hidden_dim / VEC_SIZE;
  for (int idx = comm_access_id; idx < comm_tot_access; idx += access_stride) {
    vec_t<T, VEC_SIZE> vals[NRanks];
#pragma unroll
    for (int r = 0; r < NRanks; ++r) {
      vals[r].load(reinterpret_cast<T*>(comm.comm_bufs[r]) + idx * VEC_SIZE);
    }
    vec_t<T, VEC_SIZE> sum_val = allreduce_sum<T, VEC_SIZE, NRanks, Fp32Acc>(vals);
#pragma unroll
    for (int r = 0; r < NRanks; ++r) {
      sum_val.store(reinterpret_cast<T*>(comm.comm_bufs[r]) + (tot_access + idx) * VEC_SIZE);
    }
  }
  barrier.sync();
#pragma unroll
  for (int r = 0; r < NRanks; ++r) {
    int comm_access_id = access_id + begin_tokens[r] * params.hidden_dim / VEC_SIZE;
    int comm_token_id = token_id + begin_tokens[r];
    int comm_tot_access = (begin_tokens[r] + token_num_per_ranks[r]) * params.hidden_dim / VEC_SIZE;
    for (int idx = comm_access_id, tidx = comm_token_id; idx < comm_tot_access;
         idx += access_stride, tidx += token_stride) {
      fused_op.update(idx);
      vec_t<T, VEC_SIZE> sum_val;
      sum_val.load(reinterpret_cast<T*>(comm.comm_bufs[params.rank]) +
                   (tot_access + idx) * VEC_SIZE);
      fused_op(sum_val, tidx);
    }
  }
  comm.update(barrier.m_flag_value);
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

int get_sm_count() {
  static int sm_count = 0;
  if (sm_count == 0) {
    int device_id;
    FLASHINFER_CUDA_CALL(cudaGetDevice(&device_id));
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, device_id);
    sm_count = device_prop.multiProcessorCount;
  }
  return sm_count;
}

template <AllReduceFusionPattern Pattern, typename T, int NRanks, bool Fp32Acc,
          bool TriggerCompletionAtEnd = true>
cudaError_t launch_oneshot_lamport(AllReduceFusionParams<T> const& params,
                                   cudaLaunchConfig_t& cfg) {
  FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(
      &cfg,
      allreduce_fusion_kernel_oneshot_lamport<Pattern, T, NRanks, Fp32Acc, TriggerCompletionAtEnd>,
      params));
  return cudaSuccess;
}

template <AllReduceFusionPattern Pattern, typename T, int NRanks, bool Fp32Acc>
cudaError_t launch_twoshot_sync(AllReduceFusionParams<T> const& params, cudaLaunchConfig_t& cfg,
                                std::array<int, NRanks> begin_tokens,
                                std::array<int, NRanks> token_num_per_ranks) {
  FLASHINFER_CUDA_CALL(
      cudaLaunchKernelEx(&cfg, allreduce_fusion_kernel_twoshot_sync<Pattern, T, NRanks, Fp32Acc>,
                         params, begin_tokens, token_num_per_ranks));
  return cudaSuccess;
}

bool use_oneshot(int token_num) { return token_num <= details::kOneShotMaxToken; }

template <AllReduceFusionPattern Pattern, typename T, int NRanks, bool Fp32Acc>
cudaError_t allreduce_fusion_kernel_launcher(AllReduceFusionParams<T> const& params,
                                             bool launch_with_pdl) {
  static constexpr int VEC_SIZE = details::kBytesPerAccess / sizeof(T);
  FLASH_CHECK(params.size % params.hidden_dim == 0);
  FLASH_CHECK(params.hidden_dim % VEC_SIZE == 0);
  static int SM = utils::getSMVersion();
  int token_num = params.size / params.hidden_dim;
  bool oneshot = params.use_oneshot;
  int cluster_num = token_num;
  std::array<int, NRanks> begin_tokens, token_num_per_ranks;
  if (!oneshot) {
    int remaining_token = token_num % NRanks;
    int token_num_per_rank = token_num / NRanks;
    cluster_num = token_num_per_rank;
    if (remaining_token) {
      cluster_num++;
    }
    for (int r = 0; r < NRanks; ++r) {
      begin_tokens[r] = r * token_num_per_rank + (remaining_token > r ? r : remaining_token);
      token_num_per_ranks[r] = token_num_per_rank + (remaining_token > r ? 1 : 0);
    }
  }
  int threads_per_token = params.hidden_dim / VEC_SIZE;
  int cluster_size;
  if (SM >= 90) {
    cluster_size = 8;
  } else {
    cluster_size = 1;
  }
  while (threads_per_token % cluster_size != 0 && cluster_size > 1) {
    cluster_size /= 2;
  }
  int threads_per_block = threads_per_token / cluster_size;
  while (threads_per_block < 128 && cluster_size >= 2) {
    threads_per_block *= 2;
    cluster_size /= 2;
  }
  FLASH_CHECK(oneshot || threads_per_block >= params.nranks);
  int block_size = threads_per_block;
  FLASH_CHECK(block_size <= 1024 && cluster_size > 0);
  int sm_count = get_sm_count();
  int grid_size = (std::min(sm_count, cluster_num * cluster_size) / cluster_size) * cluster_size;
  cudaLaunchConfig_t cfg;
  cudaLaunchAttribute attribute[2];
  cfg.gridDim = grid_size;
  cfg.blockDim = block_size;
  cfg.dynamicSmemBytes = 0;
  cfg.stream = params.stream;
  attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute[0].val.programmaticStreamSerializationAllowed = launch_with_pdl ? 1 : 0;
  attribute[1].id = cudaLaunchAttributeClusterDimension;
  attribute[1].val.clusterDim.x = cluster_size;
  attribute[1].val.clusterDim.y = 1;
  attribute[1].val.clusterDim.z = 1;
  cfg.attrs = attribute;
  cfg.numAttrs = SM >= 90 ? 2 : 0;
  if (oneshot) {
    bool trigger_completion_at_end = params.trigger_completion_at_end;
    if (trigger_completion_at_end) {
      FLASHINFER_CUDA_CALL(launch_oneshot_lamport<Pattern, T, NRanks, Fp32Acc, true>(params, cfg));
    } else {
      FLASHINFER_CUDA_CALL(launch_oneshot_lamport<Pattern, T, NRanks, Fp32Acc, false>(params, cfg));
    }
  } else {
    FLASHINFER_CUDA_CALL(launch_twoshot_sync<Pattern, T, NRanks, Fp32Acc>(params, cfg, begin_tokens,
                                                                          token_num_per_ranks));
  }
  return cudaSuccess;
}

template <typename T>
cudaError_t allreduce_fusion_op(AllReduceFusionParams<T> const& params, bool launch_with_pdl,
                                bool fp32_acc) {
#define DISPATCH_ACC_TYPE(T, Pattern, NRanks)                                                      \
  if constexpr (std::is_same_v<T, float>) {                                                        \
    return allreduce_fusion_kernel_launcher<Pattern, T, NRanks, false>(params, launch_with_pdl);   \
  } else {                                                                                         \
    if (fp32_acc) {                                                                                \
      return allreduce_fusion_kernel_launcher<Pattern, T, NRanks, true>(params, launch_with_pdl);  \
    } else {                                                                                       \
      return allreduce_fusion_kernel_launcher<Pattern, T, NRanks, false>(params, launch_with_pdl); \
    }                                                                                              \
  }

#define DISPATCH_PATTERN(T, NRanks)                                                                \
  if (params.pattern == AllReduceFusionPattern::kAllReduce) {                                      \
    DISPATCH_ACC_TYPE(T, AllReduceFusionPattern::kAllReduce, NRanks);                              \
  } else if (params.pattern == AllReduceFusionPattern::kARResidualRMSNorm) {                       \
    DISPATCH_ACC_TYPE(T, AllReduceFusionPattern::kARResidualRMSNorm, NRanks);                      \
  } else if (params.pattern == AllReduceFusionPattern::kARResidualRMSNormFP8Quant) {               \
    DISPATCH_ACC_TYPE(T, AllReduceFusionPattern::kARResidualRMSNormFP8Quant, NRanks);              \
  } else if (params.pattern == AllReduceFusionPattern::kARResidualRMSNormFP4Quant) {               \
    if constexpr (!std::is_same_v<T, float>) {                                                     \
      DISPATCH_ACC_TYPE(T, AllReduceFusionPattern::kARResidualRMSNormFP4Quant, NRanks);            \
    } else {                                                                                       \
      FLASH_CHECK_WITH_INFO(false,                                                                 \
                            "allreduce_fusion_kernel: "                                            \
                            "AllReduceFusionPattern=kARResidualRMSNormFP4Quant can not work with " \
                            "DType=float!");                                                       \
    }                                                                                              \
  } else if (params.pattern == AllReduceFusionPattern::kARResidualRMSNormOutFP8Quant) {            \
    DISPATCH_ACC_TYPE(T, AllReduceFusionPattern::kARResidualRMSNormOutFP8Quant, NRanks);           \
  } else if (params.pattern == AllReduceFusionPattern::kARResidualRMSNormOutFP4Quant) {            \
    if constexpr (!std::is_same_v<T, float>) {                                                     \
      DISPATCH_ACC_TYPE(T, AllReduceFusionPattern::kARResidualRMSNormOutFP4Quant, NRanks);         \
    } else {                                                                                       \
      FLASH_CHECK_WITH_INFO(                                                                       \
          false,                                                                                   \
          "allreduce_fusion_kernel: AllReduceFusionPattern=kARResidualRMSNormOutFP4Quant can not " \
          "work with "                                                                             \
          "DType=float!");                                                                         \
    }                                                                                              \
  } else {                                                                                         \
    FLASH_CHECK_WITH_INFO(false, "allreduce_fusion_kernel: unsupported pattern!");                 \
  }

#define DISPATCH_RANKS(NRanks)   \
  if (params.nranks == NRanks) { \
    DISPATCH_PATTERN(T, NRanks); \
  }

  DISPATCH_RANKS(2);
  DISPATCH_RANKS(4);
  DISPATCH_RANKS(8);
  DISPATCH_RANKS(16);
  FLASH_CHECK_WITH_INFO(
      false, "allreduce_fusion_kernel: unsupported ranks number! Supported ranks: 2, 4, 8, 16. ");
}

}  // namespace trtllm_allreduce_fusion

}  // namespace flashinfer
